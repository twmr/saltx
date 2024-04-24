# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import logging
import numbers
import operator
from typing import NamedTuple

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import create_matrix_block, create_vector_block
from petsc4py import PETSc
from ufl import dx, elem_mult, inner, nabla_grad

from saltx import jacobian
from saltx.log import Timer

log = logging.getLogger(__name__)


class MatVecCollection(NamedTuple):
    mat_dF_dvw: PETSc.Mat

    # for F computation
    vec_F_petsc: PETSc.Vec
    # for J computation
    vec_dF_dk_seq: list[PETSc.Vec]
    vec_dF_ds_seq: list[PETSc.Vec]


class NonLinearProblem:
    def __init__(
        self,
        V,
        ka,
        gt,
        dielec: float | fem.Function,
        n: int,
        pump,
        invperm: fem.Function | None = None,
        ds_obc=None,  # only needed for 1D
        max_nmodes=5,
    ):
        self.V = V
        self.Ws = [V.clone() for _ in range(max_nmodes * 2)]
        self.mesh = V.mesh
        self.ka = fem.Constant(V.mesh, complex(ka, 0))
        self.gt = fem.Constant(V.mesh, complex(gt, 0))

        self.dielec = dielec
        self.invperm = invperm or 1

        # in most cases pump is D0 * pump_profile, but can also be more
        # generalized expressions (see the pump profile in the exceptional
        # point system)
        self.pump = pump

        # dielectric loss of the cold (not-pumped) cavity
        self.sigma_c = None

        self.n = n
        self.ds_obc = ds_obc

        self.zero = 0  # fem.Constant(self.V.mesh, 0j)

        self.matvec_coll_map: dict[int, MatVecCollection] = {}

        # TODO do this lazily
        self._b_vectors = [fem.Function(V) for _ in range(max_nmodes)]
        self._form_constants = [
            # k, s
            (fem.Constant(self.mesh, 0j), fem.Constant(self.mesh, 0j))
            for _ in range(max_nmodes)
        ]
        self._max_spaces = [(self.V.clone(), self.V.clone()) for _ in range(max_nmodes)]
        self._k_hbt_constants = [
            # k_hbt (k is real valued)
            fem.Constant(self.mesh, complex(1.0, 0.0))
            for _ in range(max_nmodes)
        ]
        self._cur_forms = {}
        self._cur_Q_hbt_forms = {}

        topo_dim = V.mesh.topology.dim
        self._mult = elem_mult if topo_dim > 1 else operator.mul
        self._curl = ufl.curl if topo_dim > 1 else nabla_grad

    def update_b_and_k_for_forms(self, refined_modes) -> None:
        assert len(refined_modes) <= len(self._b_vectors)

        for refined_mode, b, k in zip(
            refined_modes, self._b_vectors, self._k_hbt_constants
        ):
            b.x.array[:] = refined_mode.array
            k.value = refined_mode.k

    def get_Q_hbt_form(self, nmodes: int) -> fem.forms.Form:
        try:
            return self._cur_Q_hbt_forms[nmodes]
        except KeyError:
            Q_hbt_form = self._create_Q_hbt_form(nmodes)
            self._cur_Q_hbt_forms[Q_hbt_form] = Q_hbt_form
            return Q_hbt_form

    def _create_Q_hbt_form(self, nmodes: int) -> fem.forms.Form:
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        ka = self.ka
        gt = self.gt
        sht = 0
        for midx, (b, k) in enumerate(zip(self._b_vectors, self._k_hbt_constants)):
            if midx == nmodes:
                break
            gk = gt / (k - ka + 1j * gt)
            sht += abs(gk * b) ** 2

        # create the final Q form
        return fem.form(self.pump / (1 + sht) * inner(u, v) * dx)

    def _create_newton_forms(self, nmodes):
        spaces = self._max_spaces[:nmodes]

        modes_data = []
        for midx, (b, (k, s)) in enumerate(zip(self._b_vectors, self._form_constants)):
            if midx == nmodes:
                break
            modes_data.append((b, k, s))
        del b, k, s

        pump = self.pump
        assert pump is not None
        dielec = self.dielec
        # invperm has a real and an imaginary part
        invp = self.invperm

        gt = self.gt
        ka = self.ka

        # invperm has a real and an imaginary part
        invp = self.invperm

        # gammak = lambda k: gt / (k - ka + 1j * gt)
        # dgammak_dk = lambda k: -gt / (k - ka + 1j * gt) ** 2
        def Gk(k):
            return gt**2 / ((k - ka) ** 2 + gt**2)  # in Reals

        def dGk_dk(k):
            return -2 * (k - ka) / ((k - ka) ** 2 + gt**2) * Gk(k)  # p. 116

        sht = sum(Gk(k) * s**2 * abs(b) ** 2 for (b, k, s) in modes_data)

        curl = self._curl
        mult = self._mult

        F_components = []

        def Lre(testf, trialf):
            return inner(mult(ufl.real(invp), curl(trialf)), curl(testf)) * dx

        def Lim(testf, trialf):
            if isinstance(invp, numbers.Real):
                return 0
            return inner(mult(ufl.imag(invp), curl(trialf)), curl(testf)) * dx

        def R(testf, trialf):
            if self.ds_obc is None:
                return self.zero
            return inner(trialf, testf) * self.ds_obc

        def N(testf, trialf):
            if self.sigma_c is None:
                return self.zero
            return self.sigma_c * inner(trialf, testf) * dx

        def Mre(testf, trialf):
            return ufl.real(dielec) * inner(trialf, testf) * dx

        def Mim(testf, trialf):
            if isinstance(dielec, numbers.Real):
                return 0
            return ufl.imag(dielec) * inner(trialf, testf) * dx

        def Q(testf, trialf):
            return pump / (1 + sht) * inner(trialf, testf) * dx

        def dQ_dx(testf, trialf, dsht_dx):
            # x is either k or s
            denom = (1 + sht) ** 2
            return -dsht_dx * pump / denom * inner(trialf, testf) * dx

        def dQx_dy(testf, trialf, x_row, y_col, k_col, s_col):
            # the derivative is w.r.t the mode y
            denom = (1 + sht) ** 2
            return (
                (-2 * s_col**2 * Gk(k_col))
                / denom
                * (pump * x_row * y_col * inner(trialf, testf))
                * dx
            )

        def calc_Fre_and_Fim(b, k, re_space, im_space):
            re, im = ufl.TestFunction(re_space), ufl.TestFunction(im_space)

            v = ufl.real(b)
            w = ufl.imag(b)
            krat = (k - ka) / gt

            # Sb = -formL + 1j * k * R + k**2 * M + k**2 * gammak * Q
            F_re = (
                # mult with v
                -Lre(re, v)
                + k**2 * Mre(re, v)
                + k**2 * krat * Gk(k) * Q(re, v)
                # mult with w
                + Lim(re, w)
                - k * R(re, w)
                - k**2 * Mim(re, w)
                + k**2 * Gk(k) * Q(re, w)
            )
            F_im = (
                # mult with v
                -Lim(im, v)
                + k * R(im, v)
                + k**2 * Mim(im, v)
                - k**2 * Gk(k) * Q(im, v)
                # multi with w
                - Lre(im, w)
                + k**2 * Mre(im, w)
                + k**2 * krat * Gk(k) * Q(im, w)
            )

            # Fre + i*Fim =
            #  -Lrev + k2 Mrev + k2*krat*Gk*Qv  + Limw -kRw - k2Mimw + k2*Gk*Qw
            #  + i(-Limv -kRv + k2*Mimv - k2 Gk * Qv - Lrew + k2Mrew + k2*krat*Gk*Qw)

            # L terms:
            # -Lv + Limw -iLimv - iLrew
            # = v*(-L -iL) + w*(Lim - iLre)
            # =              iw*(-iLim -Lre)
            # = (v + iw)*(-Lre - iLim)

            # M terms:
            # k2Mrev - k2Mimw +ik2*Mimv +ik2Mrew
            # k2*[v*(Mre +iMim) + iw*(iMim +Mre)]

            if self.sigma_c is not None:
                # TODO include this in R()

                # Note that (w, re) and (v, im) is not a typo!!
                F_re += -k * self.sigma_c * inner(w, re) * dx
                F_im += k * self.sigma_c * inner(v, im) * dx

            F_components.append(F_re)
            F_components.append(F_im)

        def calc_dF_dk_and_dF_ds(
            local_dF_dk_column, local_dF_ds_column, b, k, s, re_space, im_space
        ):
            # this is for the diagonal blocks
            re, im = ufl.TestFunction(re_space), ufl.TestFunction(im_space)

            v = ufl.real(b)
            w = ufl.imag(b)
            krat = (k - ka) / gt

            # todo do this for all ks
            # -> create ufl ticket
            # dF_re_dk0 = ufl.derivative(F_re, k)
            # dF_im_dk0 = ufl.derivative(F_im, k)

            dsht_dk = dGk_dk(k) * s**2 * abs(b) ** 2
            dsht_ds = Gk(k) * 2 * s * abs(b) ** 2

            dF_re_dk = (
                # mult with v
                2 * k * Mre(re, v)
                + 2 * k * krat * Gk(k) * Q(re, v)
                + k**2 * (1 / gt) * Gk(k) * Q(re, v)
                + k**2 * krat * dGk_dk(k) * Q(re, v)
                + k**2 * krat * Gk(k) * dQ_dx(re, v, dsht_dk)
                # mult with w
                - R(re, w)
                - 2 * k * Mim(re, w)
                + 2 * k * Gk(k) * Q(re, w)
                + k**2 * dGk_dk(k) * Q(re, w)
                + k**2 * Gk(k) * dQ_dx(re, w, dsht_dk)
            )
            dF_im_dk = (
                # mult with v
                R(im, v)
                + 2 * k * Mim(im, v)
                - 2 * k * Gk(k) * Q(im, v)
                - k**2 * dGk_dk(k) * Q(im, v)
                - k**2 * Gk(k) * dQ_dx(im, v, dsht_dk)
                # multi with w
                + 2 * k * Mre(im, w)
                + 2 * k * krat * Gk(k) * Q(im, w)
                + k**2 * (1 / gt) * Gk(k) * Q(im, w)
                + k**2 * krat * dGk_dk(k) * Q(im, w)
                + k**2 * krat * Gk(k) * dQ_dx(im, w, dsht_dk)
            )

            if self.sigma_c is not None:
                # Note that (w, re) and (v, im) is not a typo!!
                dF_re_dk += -self.sigma_c * inner(w, re) * dx
                dF_im_dk += self.sigma_c * inner(v, im) * dx

            local_dF_dk_column.extend([dF_re_dk, dF_im_dk])

            dF_re_ds = (
                # mult with v
                k**2 * krat * Gk(k) * dQ_dx(re, v, dsht_ds)
                # mult with w
                + k**2 * Gk(k) * dQ_dx(re, w, dsht_ds)
            )
            dF_im_ds = (
                # mult with v
                -(k**2) * Gk(k) * dQ_dx(im, v, dsht_ds)
                # multi with w
                + k**2 * krat * Gk(k) * dQ_dx(im, w, dsht_ds)
            )
            local_dF_ds_column.extend([dF_re_ds, dF_im_ds])

        def calc_dFx_dky_and_dFx_dsy(
            local_dF_dk_column,
            local_dF_ds_column,
            bx,
            kx,
            sx,
            rex_space,
            imx_space,
            by,
            ky,
            sy,
            _rey_space,
            _imy_space,
        ):
            # this is for the off-diagonal blocks
            rex, imx = ufl.TestFunction(rex_space), ufl.TestFunction(imx_space)

            v = ufl.real(bx)
            w = ufl.imag(bx)
            kxrat = (kx - ka) / gt

            dsht_dky = dGk_dk(ky) * sy**2 * abs(by) ** 2
            dsht_dsy = Gk(ky) * 2 * sy * abs(by) ** 2

            dF_re_dk = +(kx**2) * kxrat * Gk(kx) * dQ_dx(
                rex, v, dsht_dky
            ) + kx**2 * Gk(kx) * dQ_dx(rex, w, dsht_dky)
            dF_im_dk = -(kx**2) * Gk(kx) * dQ_dx(
                imx, v, dsht_dky
            ) + kx**2 * kxrat * Gk(kx) * dQ_dx(imx, w, dsht_dky)
            local_dF_dk_column.extend([dF_re_dk, dF_im_dk])

            dF_re_ds = kx**2 * kxrat * Gk(kx) * dQ_dx(
                rex, v, dsht_dsy
            ) + kx**2 * Gk(kx) * dQ_dx(rex, w, dsht_dsy)
            dF_im_ds = -(kx**2) * Gk(kx) * dQ_dx(
                imx, v, dsht_dsy
            ) + kx**2 * kxrat * Gk(kx) * dQ_dx(imx, w, dsht_dsy)
            local_dF_ds_column.extend([dF_re_ds, dF_im_ds])

        def call_A_diag_block(mode_index, b, k, s, Wre, Wim):
            v = ufl.real(b)
            w = ufl.imag(b)
            # Note that the trial spaces are on the column spaces
            tri_re, tri_im = ufl.TrialFunction(Wre), ufl.TrialFunction(Wim)
            # Note that the test spaces are on the row spaces
            test_re, test_im = ufl.TestFunction(Wre), ufl.TestFunction(Wim)

            krat = (k - ka) / gt
            k2Gk = k**2 * Gk(k)

            # diag terms in the current (diag) block:
            dFRe_dv = (
                -Lre(test_re, tri_re)
                + k**2 * Mre(test_re, tri_re)
                + k2Gk * krat * dQx_dy(test_re, tri_re, v, v, k, s)
                + k2Gk * krat * Q(test_re, tri_re)
                + k2Gk * dQx_dy(test_re, tri_re, w, v, k, s)
            )

            dFIm_dw = (
                -Lre(test_im, tri_im)
                + k**2 * Mre(test_im, tri_im)
                + k2Gk
                * (
                    krat * (dQx_dy(test_im, tri_im, w, w, k, s) + Q(test_im, tri_im))
                    - dQx_dy(test_im, tri_im, v, w, k, s)
                )
            )

            # off-diag terms in the current (diag) block:
            dFRe_dw = (
                Lim(test_re, tri_im)
                + -k * R(test_re, tri_im)
                - k * N(test_re, tri_im)
                - k**2 * Mim(test_re, tri_im)
                + k2Gk
                * (
                    krat * dQx_dy(test_re, tri_im, v, w, k, s)
                    + dQx_dy(test_re, tri_im, w, w, k, s)
                    + Q(test_re, tri_im)
                )
            )

            dFIm_dv = (
                -Lim(test_im, tri_re)
                + k * R(test_im, tri_re)
                + k * N(test_im, tri_re)
                + k**2 * Mim(test_im, tri_re)
                + k2Gk
                * (
                    krat * dQx_dy(test_im, tri_re, w, v, k, s)
                    - (dQx_dy(test_im, tri_re, v, v, k, s) + Q(test_im, tri_re))
                )
            )

            off = 2 * mode_index
            a_form_array[off, off] = dFRe_dv
            a_form_array[off, off + 1] = dFRe_dw
            a_form_array[off + 1, off] = dFIm_dv
            a_form_array[off + 1, off + 1] = dFIm_dw

        def call_A_offdiag_block(
            mode_row_index,
            mode_col_index,
            b_row,
            k_row,
            s_row,
            Wre_row,
            Wim_row,
            b_col,
            k_col,
            s_col,
            Wre_col,
            Wim_col,
        ):
            v_row = ufl.real(b_row)
            w_row = ufl.imag(b_row)
            v_col = ufl.real(b_col)
            w_col = ufl.imag(b_col)

            tri_re, tri_im = ufl.TrialFunction(Wre_col), ufl.TrialFunction(Wim_col)
            test_re, test_im = ufl.TestFunction(Wre_row), ufl.TestFunction(Wim_row)

            krat = (k_row - ka) / gt
            k2Gk = k_row**2 * Gk(k_row)

            col_v = (v_col, k_col, s_col)
            col_w = (w_col, k_col, s_col)

            # diag terms in the current (offdiag) block:
            dFRe_dv = k2Gk * (
                krat * dQx_dy(test_re, tri_re, v_row, *col_v)
                + dQx_dy(test_re, tri_re, w_row, *col_v)
            )

            dFIm_dw = k2Gk * (
                krat * dQx_dy(test_im, tri_im, w_row, *col_w)
                - dQx_dy(test_im, tri_im, v_row, *col_w)
            )

            # off-diag terms in the current (offdiag) block:
            dFRe_dw = k2Gk * (
                krat * dQx_dy(test_re, tri_im, v_row, *col_w)
                + dQx_dy(test_re, tri_im, w_row, *col_w)
            )

            dFIm_dv = k2Gk * (
                krat * dQx_dy(test_im, tri_re, w_row, *col_v)
                - dQx_dy(test_im, tri_re, v_row, *col_v)
            )

            offy = 2 * mode_row_index
            offx = 2 * mode_col_index
            a_form_array[offy, offx] = dFRe_dv
            a_form_array[offy, offx + 1] = dFRe_dw
            a_form_array[offy + 1, offx] = dFIm_dv
            a_form_array[offy + 1, offx + 1] = dFIm_dw

        log.debug("create form array objects")
        a_form_array = np.array(
            [[None for _ in range(2 * nmodes)] for _ in range(2 * nmodes)], dtype=object
        )
        dF_dk_seq, dF_ds_seq = [], []
        # product loop for filling the a_form_array with forms
        for mode_row_index, (
            (b_row, k_row, s_row),
            (Wre_row, Wim_row),
        ) in enumerate(zip(modes_data, spaces)):
            calc_Fre_and_Fim(b_row, k_row, Wre_row, Wim_row)

            for mode_col_index, (
                (b_col, k_col, s_col),
                (Wre_col, Wim_col),
            ) in enumerate(zip(modes_data, spaces)):
                if mode_row_index == mode_col_index:
                    call_A_diag_block(
                        mode_row_index, b_col, k_col, s_col, Wre_col, Wim_col
                    )
                else:
                    call_A_offdiag_block(
                        mode_row_index,
                        mode_col_index,
                        b_row,
                        k_row,
                        s_row,
                        Wre_row,
                        Wim_row,
                        b_col,
                        k_col,
                        s_col,
                        Wre_col,
                        Wim_col,
                    )

        # column vectors (vec_F_petsc, vec_dF_ds_seq, vec_dF_dk_seq)
        for mode_col_index, (
            (b_col, k_col, s_col),
            (Wre_col, Wim_col),
        ) in enumerate(zip(modes_data, spaces)):
            local_dF_dk_column, local_dF_ds_column = [], []
            for mode_row_index, (
                (b_row, k_row, s_row),
                (Wre_row, Wim_row),
            ) in enumerate(zip(modes_data, spaces)):
                if mode_col_index == mode_row_index:
                    calc_dF_dk_and_dF_ds(
                        local_dF_dk_column,
                        local_dF_ds_column,
                        b_col,
                        k_col,
                        s_col,
                        Wre_col,
                        Wim_col,
                    )
                else:
                    calc_dFx_dky_and_dFx_dsy(
                        local_dF_dk_column,
                        local_dF_ds_column,
                        b_row,
                        k_row,
                        s_row,
                        Wre_row,
                        Wim_row,
                        b_col,
                        k_col,
                        s_col,
                        Wre_col,
                        Wim_col,
                    )

            dF_dk_seq.append(local_dF_dk_column)
            dF_ds_seq.append(local_dF_ds_column)

        with Timer(log.debug, "Calling fem.form(F)"):
            F_components = fem.form(F_components)
        with Timer(log.debug, "Calling fem.form(a)"):
            a_form_array = fem.form(a_form_array)
        with Timer(log.debug, "Creating fem.form s for dF/dk dF/ds"):
            dF_dk_seq = fem.form([x for x in dF_dk_seq])
            dF_ds_seq = fem.form([x for x in dF_ds_seq])
        log.debug("create form array objects done")

        return F_components, a_form_array, dF_dk_seq, dF_ds_seq

    def get_forms(self, nmodes):
        try:
            return self._cur_forms[nmodes]
        except KeyError:
            cur_forms = self._create_newton_forms(nmodes)
            self._cur_forms[nmodes] = cur_forms
            return cur_forms

    def assemble_F_and_J(self, L, A, minfos, bcs):
        # assemble F(minfos) into the vector L
        # assemble J(minfos) into the matrix A

        # Reset the residual vector
        with L.localForm() as L_local:
            L_local.set(0.0)

        # N = A.getSize()[0]
        # assert N == A.getSize()[1]
        assert A.getSize()[0] == L.getSize()

        n = self.n

        nmodes = len(minfos)
        modes_data = []
        for minfo, b, (k, s) in zip(minfos, self._b_vectors, self._form_constants):
            b.x.array[:] = minfo.cmplx_array
            k.value = complex(minfo.k, 0)
            s.value = complex(minfo.s, 0)
            modes_data.append((b, k, s, minfo.dof_at_maximum))
        del b, k, s

        log.info(
            f"eval F and J at k={[m.k for m in minfos]}, s={[m.s for m in minfos]}"
        )

        etbm1s = []

        bcscalar = PETSc.ScalarType(0)

        bcdofs_seq = [bc._cpp_object.dof_indices()[0] for bc in bcs]
        # we have to add the same BC for the subspaces self.Ws, because this is required
        # for the block_matrix_assembly
        # TODO rename new_bcs to just bcs
        new_bcs = [
            fem.dirichletbc(bcscalar, bcdofs, W)
            for bcdofs in bcdofs_seq
            for spacetuple in self._max_spaces[:nmodes]
            for W in spacetuple
        ]

        for b_row, _, _, dof_at_maximum in modes_data:
            etbm1 = b_row.vector.getValue(dof_at_maximum) - 1
            if abs(etbm1) > 1e-12:
                log.debug(f"{etbm1=}")
            etbm1s.extend([etbm1.real, etbm1.imag])

        # TODO create this in __init__
        F_components, a_form_array, dF_dk_seq, dF_ds_seq = self.get_forms(nmodes)

        try:
            matvec_coll = self.matvec_coll_map[nmodes]
        except KeyError:
            with Timer(log.debug, "creation of sparse dF_dvw matrix and vectors"):
                log.debug("BEFORE CMB")
                mat_dF_dvw = create_matrix_block(a_form_array)
                log.debug("AFTER CMB")

                vec_F_petsc = create_vector_block(F_components)
                log.debug("AFTER CVB")
                vec_dF_dk_seq = [create_vector_block(dF_dk) for dF_dk in dF_dk_seq]
                log.debug("AFTER CVB (dF_dk_seq)")

                vec_dF_ds_seq = [create_vector_block(dF_ds) for dF_ds in dF_ds_seq]
                log.debug("AFTER CVB (dF_ds_seq)")
                matvec_coll = MatVecCollection(
                    mat_dF_dvw=mat_dF_dvw,
                    vec_F_petsc=vec_F_petsc,
                    vec_dF_dk_seq=vec_dF_dk_seq,
                    vec_dF_ds_seq=vec_dF_ds_seq,
                )

                self.matvec_coll_map[nmodes] = matvec_coll

        with matvec_coll.vec_F_petsc.localForm() as F_local:
            F_local.set(0.0)
        fem.petsc.assemble_vector_block(
            matvec_coll.vec_F_petsc, F_components, a_form_array, bcs=new_bcs
        )

        # log.debug(f"norm F_petsc {F_petsc.norm(0)}")
        # S b = L.sub(0)
        # e^T b - 1 = L.sub(1)

        f_vals = matvec_coll.vec_F_petsc.getArray()
        L.setValues(range(2 * nmodes * n), f_vals.real)
        L.setValues(range(2 * nmodes * n, 2 * nmodes * (n + 1)), np.asarray(etbm1s))

        log.info(f"current norm of F: {L.norm(0)}")

        # 1 x n

        mat_dF_dvw = matvec_coll.mat_dF_dvw
        with Timer(log.debug, f"ass bilinear forms {mat_dF_dvw.getSize()=}"):
            mat_dF_dvw.zeroEntries()  # not sure if this is really needed
            fem.petsc.assemble_matrix_block(
                mat_dF_dvw,
                a_form_array,
                bcs=new_bcs,
            )
            mat_dF_dvw.assemble()

        with Timer(log.debug, "ass linear forms"):
            vec_dk_seq = matvec_coll.vec_dF_dk_seq
            vec_ds_seq = matvec_coll.vec_dF_ds_seq
            for i, (dF_dk, dF_ds) in enumerate(zip(dF_dk_seq, dF_ds_seq)):
                vec = vec_dk_seq[i]
                with vec.localForm() as vec_local:
                    vec_local.set(0.0)

                fem.petsc.assemble_vector_block(
                    vec, fem.form(dF_dk), a_form_array, bcs=new_bcs
                )

                vec = vec_ds_seq[i]
                with vec.localForm() as vec_local:
                    vec_local.set(0.0)

                fem.petsc.assemble_vector_block(
                    vec, fem.form(dF_ds), a_form_array, bcs=new_bcs
                )

        with Timer(log.debug, "Assembly of Jacobian into block real matrix"):
            jacobian.assemble_salt_jacobian_block_matrix(
                A,
                # for two modes:
                # [[dF1_re_dv1, dF1_re_dw1, dF1_re_dv2, dF1_re_dw2],
                #  [dF1_im_dv1, dF1_im_dw1, dF1_im_dv2, dF1_im_dw2],
                #  [dF2_re_dv1, dF2_re_dw1, dF2_re_dv2, dF2_re_dw2],
                #  [dF2_im_dv1, dF2_im_dw1, dF2_im_dv2, dF2_im_dw2]]
                matvec_coll.mat_dF_dvw,
                # for two modes:
                # [dFx_dk1, dFx_dk2]
                matvec_coll.vec_dF_dk_seq,
                matvec_coll.vec_dF_ds_seq,
                [minfo.dof_at_maximum for minfo in minfos],
                nmodes=len(dF_dk_seq),
            )

    def create_A(self, nmodes=1):
        # A contains the jacobian
        tstf = [ufl.TestFunction(W) for W in self.Ws]
        trif = [ufl.TrialFunction(W) for W in self.Ws]

        form_array = fem.form(
            [
                [inner(trif[j], tstf[i]) * ufl.dx for j in range(2 * nmodes)]
                for i in range(2 * nmodes)
            ]
        )

        with Timer(log.debug, "create_A"):
            block_mat = create_matrix_block(form_array)

            block_mat.zeroEntries()  # not sure if this is really needed

            fem.petsc.assemble_matrix_block(
                block_mat,
                form_array,
                bcs=[],
                diagonal=1.0,
            )
            block_mat.assemble()

            with Timer(log.debug, "create_salt_jacobian"):
                A = jacobian.create_salt_jacobian_block_matrix(
                    block_mat,
                    nmodes=nmodes,
                )
                log.debug(f"[Newton] for {nmodes=}: {A.getSizes()[0]=}")
        return A

    def create_L(self, nmodes=1):
        n_fem = self.n
        L = PETSc.Vec().createSeq((2 * n_fem + 2) * nmodes)
        log.debug(f"[Newton] for {nmodes=}: {L.getSize()=}")
        return L

    def create_dx(self, nmodes=1):
        n_fem = self.n
        return PETSc.Vec().createSeq((2 * n_fem + 2) * nmodes)
