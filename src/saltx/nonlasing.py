# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import logging
import operator
from typing import Any

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import create_matrix, create_vector
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, elem_mult, inner, nabla_grad

from . import jacobian
from .log import Timer

log = logging.getLogger(__name__)


def ass_linear_form_into_vec(vec, form, bcs):
    with vec.localForm() as vec_local:
        vec_local.set(0.0)
    # TODO handle ghost-values??
    fem.petsc.assemble_vector(vec, form)
    fem.set_bc(vec, bcs)
    vec.assemble()


def ass_bilinear_form(mat, form, bcs, diagonal):
    mat.zeroEntries()  # not sure if this is really needed

    fem.petsc.assemble_matrix(
        mat,
        form,
        bcs=bcs,
        diagonal=diagonal,
    )
    mat.assemble()


class NonLasingLinearProblem:
    """Newton solver for the lasing modes below threshold.

    This class can be used to determine the threshold of the first laser
    mode. For this case the pump has to be set in the optimizer that
    brings the mode to the threshold.
    """

    def __init__(
        self,
        V,
        ka,
        gt,
        # FIXME the type hints are wrong
        dielec: float | fem.Function,
        invperm: fem.Function | None,
        pump: float | fem.Function,
        bcs,
        # TODO what is the type of ufl.ds?
        ds_obc: Any | None,
    ):
        self.V = V

        # this constant will be updated in assemble_F_and_J
        self.k_constant = fem.Constant(self.V.mesh, complex(1.0, 0))
        # The initial guess b will also be updated in assemble_F_and_J
        self.b = fem.Function(self.V)

        self.form_Sb: fem.forms.FormMetaClass
        self.form_dFdk: fem.forms.FormMetaClass
        self.form_dFdu: fem.forms.FormMetaClass

        self.ka = ka
        self.gt = gt

        # size of the fem matrices
        self.n = V.dofmap.index_map.size_global

        self.dielec = dielec
        self.invperm = invperm or 1
        self.pump = pump

        # dielectric loss of the cold (not-pumped) cavity
        self.sigma_c = None

        self.bcs = bcs
        self.ds_obc = ds_obc

        topo_dim = V.mesh.topology.dim
        self._mult = elem_mult if topo_dim > 1 else operator.mul
        self._curl = ufl.curl if topo_dim > 1 else nabla_grad

        with Timer(log.debug, "Create fem forms"):
            self.create_forms()

        # initialize PETSc vectors to avoid repeated allocation in every iteration of
        # the Newton method.
        with Timer(print, "Create initial matrix/vectors for J & F"):
            self.vec_F_petsc = create_vector(self.form_Sb)
            self.mat_dF_du = create_matrix(self.form_dFdu)
            self.vec_dF_dk = create_vector(self.form_dFdk)

    def _demo_check_solutions(self, x: PETSc.Vec) -> None:
        b = fem.Function(self.V)
        b.x.array[:] = x.getValues(range(self.n))
        k = fem.Constant(self.V.mesh, x.getValue(self.n))
        print(f"eval F at k={k.value}")

        pump = self.pump
        dielec = self.dielec
        invperm = self.invperm
        ka = self.ka
        gt = self.gt

        curl = self._curl
        mult = self._mult

        gammak = gt / (k - ka + 1j * gt)

        u = ufl.TrialFunction(self.V)
        formL = inner(mult(invperm, curl(u)), curl(ufl.conj(b))) * dx
        M = dielec * inner(u, ufl.conj(b)) * dx
        Q = pump * inner(u, ufl.conj(b)) * dx

        Sb = -formL + k**2 * M + k**2 * gammak * Q
        if self.ds_obc is not None:
            R = inner(u, ufl.conj(b)) * self.ds_obc
            Sb += 1j * k * R
        if self.sigma_c is not None:
            N = self.sigma_c * inner(u, ufl.conj(b)) * dx
            Sb += 1j * k * N

        F_petsc = self.vec_F_petsc
        with Timer(print, "ass linear form F"):
            ass_linear_form_into_vec(F_petsc, fem.form(Sb), self.bcs)
        print(f"norm F_petsc {F_petsc.norm(0)}")

    def assemble_F_and_J(
        self, L: PETSc.Vec, A: PETSc.Mat, x: PETSc.Vec, dof_at_maximum: int
    ) -> None:
        # assemble F(x) into the vector L
        # and J(x) into the matrix A

        # Reset the residual vector
        with L.localForm() as L_local:
            L_local.set(0.0)

        assert self.n + 1 == L.getSize()

        b = self.b
        b.x.array[:] = x.getValues(range(self.n))
        k = self.k_constant
        k.value = x.getValue(self.n)

        print(f"eval F at k={k.value}")

        F_petsc = self.vec_F_petsc
        with Timer(print, "ass linear form F"):
            ass_linear_form_into_vec(F_petsc, self.form_Sb, self.bcs)
        print(f"norm F_petsc {F_petsc.norm(0)}")

        etbm1 = b.vector.getValue(dof_at_maximum) - 1
        if abs(etbm1) > 1e-12:
            print(f"{etbm1=}")

        # S b = L.sub(0)
        # e^T b - 1 = L.sub(1)

        L.setValues(range(self.n), F_petsc)
        L.setValue(self.n, etbm1)

        print(f"current norm of F: {L.norm(0)}")

        with Timer(print, "ass bilinear form dF/du"):
            ass_bilinear_form(
                self.mat_dF_du, self.form_dFdu, bcs=self.bcs, diagonal=1.0
            )

        with Timer(print, "ass linear form dF/dk"):
            ass_linear_form_into_vec(self.vec_dF_dk, self.form_dFdk, self.bcs)

        jacobian.assemble_complex_singlemode_jacobian_matrix(
            A, self.mat_dF_du, self.vec_dF_dk, dof_at_maximum
        )

    def create_forms(self):
        pump = self.pump
        dielec = self.dielec
        invperm = self.invperm
        ka = self.ka
        gt = self.gt

        curl = self._curl
        mult = self._mult

        b = self.b
        k = self.k_constant

        gammak = gt / (k - ka + 1j * gt)

        u = ufl.TrialFunction(self.V)
        formL = inner(mult(invperm, curl(u)), curl(ufl.conj(b))) * dx
        M = dielec * inner(u, ufl.conj(b)) * dx
        Q = pump * inner(u, ufl.conj(b)) * dx

        Sb = -formL + k**2 * M + k**2 * gammak * Q
        if self.ds_obc is not None:
            R = inner(u, ufl.conj(b)) * self.ds_obc
            Sb += 1j * k * R
        if self.sigma_c is not None:
            N = self.sigma_c * inner(u, ufl.conj(b)) * dx
            Sb += 1j * k * N
        with Timer(print, "fem.form(Sb)"):
            self.form_Sb = fem.form(Sb)

        v = ufl.TestFunction(self.V)

        L = inner(mult(invperm, curl(u)), curl(ufl.conj(b))) * dx
        M = dielec * inner(u, ufl.conj(b)) * dx
        Q = pump * inner(u, ufl.conj(b)) * dx

        # Sb = -L + 1j * k * R + k**2 * M + 1j * k * N + k**2 * gammak * Q
        dgammak_dk = -gt / (k - ka + 1j * gt) ** 2

        dFdk = 2 * k * M + 2 * k * gammak * Q + k**2 * dgammak_dk * Q
        if self.ds_obc is not None:
            R = inner(u, ufl.conj(b)) * self.ds_obc
            dFdk += 1j * R
        if self.sigma_c is not None:
            N = self.sigma_c * inner(u, ufl.conj(b)) * dx
            dFdk += 1j * N

        L = inner(mult(invperm, curl(u)), curl(v)) * dx
        M = dielec * inner(u, v) * dx
        Q = pump * inner(u, v) * dx

        dFdu = -L + k**2 * M + k**2 * gammak * Q
        if self.ds_obc is not None:
            R = inner(u, v) * self.ds_obc
            dFdu += 1j * k * R
        if self.sigma_c is not None:
            N = self.sigma_c * inner(u, v) * dx
            dFdu += 1j * k * N

        self.form_dFdk = fem.form(dFdk)
        self.form_dFdu = fem.form(dFdu)

    def create_A(self, n_fem):
        A = PETSc.Mat().create(MPI.COMM_WORLD)
        N = n_fem + 1
        A.setSizes([N, N])
        A.setUp()
        return A

    def create_L(self, n_fem):
        return PETSc.Vec().createSeq(n_fem + 1)

    def create_dx(self, n_fem):
        # n_fem (complex-valued) entries for b, 1 for k
        return PETSc.Vec().createSeq(n_fem + 1)
