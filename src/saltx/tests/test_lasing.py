# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests of the lasing.py module."""
import logging

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx import newtils, algorithms
from saltx.lasing import NonLinearProblem

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


def test_assemble_F_and_J():
    # TODO improve this unit-test
    D0 = 0.37
    use_real_jac = True

    dielec = 1.2**2
    pump_profile = 1.0
    ka = 10.0
    gt = 4.0

    radius = 3.0 * gt
    vscale = 0.5 * gt / radius
    rg_params = (ka, radius, vscale)
    Print(f"RG params: {rg_params}")
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, nx=1000)

    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    ds_obc = ufl.ds
    # Define Dirichlet boundary condition on the left
    bcs_dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0.0),
    )

    Print(f"{bcs_dofs=}")
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs, V),
    ]

    bcs_norm_constraint = fem.locate_dofs_geometrical(
        V,
        lambda x: x[0] > 0.75,
    )
    # I only want to impose the norm constraint on a single node
    # can this be done in a simpler way?
    bcs_norm_constraint = bcs_norm_constraint[:1]
    Print(f"{bcs_norm_constraint=}")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def assemble_form(form, zero_diag=False):
        if zero_diag:
            mat = fem.petsc.assemble_matrix(
                fem.form(form),
                bcs=bcs,
                diagonal=PETSc.ScalarType(0),
            )
        else:
            mat = fem.petsc.assemble_matrix(fem.form(form), bcs=bcs)
        mat.assemble()
        return mat

    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx)
    M = assemble_form(dielec * inner(u, v) * dx, zero_diag=True)
    Q = assemble_form(D0 * pump_profile * inner(u, v) * dx, zero_diag=True)
    R = assemble_form(inner(u, v) * ds_obc, zero_diag=True)

    # there is a better way to determine n (using the dofmap)
    n = L.getSize()[0]
    Print(
        f"{L.getSize()=},  DOF: {L.getInfo()['nz_used']}, MEM: {L.getInfo()['memory']}"
    )

    nevp_inputs = algorithms.NEVPInputs(
        ka=ka,
        gt=gt,
        rg_params=rg_params,
        L=L,
        M=M,
        N=None,
        Q=Q,
        R=R,
        bcs_norm_constraint=bcs_norm_constraint,
    )
    modes = algorithms.get_nevp_modes(nevp_inputs, bcs=bcs)
    evals = np.asarray([mode.k for mode in modes])

    et = PETSc.Vec().createSeq(n)
    et.setValue(bcs_norm_constraint[0], 1.0)
    nlp = NonLinearProblem(
        V,
        ka,
        gt,
        et,
        dielec=dielec,
        n=n,
        use_real_jac=use_real_jac,
        ds_obc=ds_obc,
    )
    nlp.set_pump(D0)
    nlA = nlp.create_A()
    nlL = nlp.create_L()
    initial_x = nlp.create_dx()

    modesel = np.argwhere(evals.imag > 0).flatten()[0]
    mode = modes[modesel]
    assert mode.k.imag > 0

    minfos = [
        newtils.NewtonModeInfo(
            k=mode.k.real, s=0.1, re_array=mode.array.real, im_array=mode.array.imag
        )
    ]

    newtils.fill_vector_with_modeinfos(initial_x, minfos)

    nlp.assemble_F_and_J(nlL, nlA, minfos, mode.bcs)

    # nlA.view()

    c0, v0 = nlA.getRow(0)
    c1, v1 = nlA.getRow(1)

    np.testing.assert_array_equal(
        c0, np.array([0, 1, 2, 3, 3001, 3002, 3003, 3004, 6002, 6003])
    )
    np.testing.assert_array_equal(
        v0,
        np.array(
            [
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        ),
    )

    np.testing.assert_array_equal(
        c1,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 6002, 6003]
        ),
    )
    # np.testing.assert_array_equal(
    #     v1.real,
    #     np.array(
    #         [
    #             0.00000000e00 + 0.00000000e00j,
    #             -8.66664890e03 - 2.78999149e-21j,
    #             -7.11754950e02 + 5.19887646e-22j,
    #             4.87842162e03 - 5.19882434e-22j,
    #             1.66668147e02 - 2.32494620e-22j,
    #             4.87842162e03 - 5.19889383e-22j,
    #             -7.11754950e02 + 5.19880696e-22j,
    #             0.00000000e00 + 0.00000000e00j,
    #             4.64195573e-03 + 2.05594203e-20j,
    #             -8.64977907e-04 - 3.83102645e-21j,
    #             8.64977104e-04 + 3.83101933e-21j,
    #             3.86828924e-04 + 1.71327864e-21j,
    #             8.64978175e-04 + 3.83102882e-21j,
    #             -8.64976836e-04 - 3.83101696e-21j,
    #             3.71376379e-05 + 0.00000000e00j,
    #             5.26918379e-10 + 0.00000000e00j,
    #         ]
    #     ).real,
    # )

    expected_v1 = np.array(
        [
            0.00000000e00 + 0.00000000e00j,
            -8.66664890e03 - 2.78999149e-21j,
            -7.11754950e02 + 5.19887646e-22j,
            4.87842162e03 - 5.19882434e-22j,
            1.66668147e02 - 2.32494620e-22j,
            4.87842162e03 - 5.19889383e-22j,
            -7.11754950e02 + 5.19880696e-22j,
            0.00000000e00 + 0.00000000e00j,
            4.64195573e-03 + 2.05594203e-20j,
            -8.64977907e-04 - 3.83102645e-21j,
            8.64977104e-04 + 3.83101933e-21j,
            3.86828924e-04 + 1.71327864e-21j,
            8.64978175e-04 + 3.83102882e-21j,
            -8.64976836e-04 - 3.83101696e-21j,
            3.71376379e-05 + 0.00000000e00j,
            5.26918379e-10 + 0.00000000e00j,
        ]
    )
    # FIXME why is the accuracy so small?
    np.testing.assert_almost_equal(v1, expected_v1, decimal=5)
    # assert abs(v1 - expected_v1).max() < 1e-10

    # Asize = nlA.getSize()[0]
    # Lsize = nlL.getSize()

    # vals_A = nlA.getValues(range(Asize), range(Asize)).real
    # vals_L = nlL.getValues(range(Asize)).real
    # with np.printoptions(
    #     linewidth=400, suppress=False, precision=2, threshold=sys.maxsize
    # ):
    #     print(vals_A)
    #     print(vals_L)
