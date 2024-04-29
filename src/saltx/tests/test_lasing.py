# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
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

from saltx import algorithms, newtils
from saltx.lasing import NonLinearProblem

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


def test_assemble_F_and_J():
    # TODO improve this unit-test
    D0 = 0.37

    dielec = 1.2**2
    pump_profile = 1.0
    ka = 10.0
    gt = 4.0

    radius = 3.0 * gt
    vscale = 0.5 * gt / radius
    rg_params = (ka, radius, vscale)
    Print(f"RG params: {rg_params}")
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, nx=1000)

    V = fem.functionspace(msh, ("Lagrange", 3))

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

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def assemble_form(form, diag=1.0):
        mat = fem.petsc.assemble_matrix(fem.form(form), bcs=bcs, diagonal=diag)
        mat.assemble()
        return mat

    def to_const(real_value: float) -> fem.Constant:
        return fem.Constant(V.mesh, complex(real_value, 0))

    D0 = to_const(D0)
    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx)
    M = assemble_form(dielec * inner(u, v) * dx, diag=0.0)
    Q = assemble_form(D0 * pump_profile * inner(u, v) * dx, diag=0.0)
    R = assemble_form(inner(u, v) * ds_obc, diag=0.0)

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
        bcs=bcs,
    )
    modes = algorithms.get_nevp_modes(nevp_inputs)
    evals = np.asarray([mode.k for mode in modes])

    nlp = NonLinearProblem(
        V,
        ka,
        gt,
        dielec=dielec,
        n=n,
        pump=D0,
        ds_obc=ds_obc,
    )
    nlA = nlp.create_A()
    nlL = nlp.create_L()
    initial_x = nlp.create_dx()

    modesel = np.argwhere(evals.imag > 0).flatten()[0]
    mode = modes[modesel]
    assert mode.k.imag > 0

    minfos = [
        newtils.NewtonModeInfo(
            k=mode.k.real,
            s=0.1,
            re_array=mode.array.real,
            im_array=mode.array.imag,
            dof_at_maximum=mode.dof_at_maximum,
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

    expected_v1 = np.array(
        [
            0.00000000e00 + 0.00000000e00j,
            -8.66664890e03 - 2.79001631e-21j,
            -7.11754950e02 + 5.19890275e-22j,
            4.87842162e03 - 5.19888097e-22j,
            1.66668147e02 - 2.32499407e-22j,
            4.87842162e03 - 5.19891001e-22j,
            -7.11754950e02 + 5.19887371e-22j,
            0.00000000e00 + 0.00000000e00j,
            4.64195961e-03 + 2.05594547e-20j,
            -8.64978319e-04 - 3.83103010e-21j,
            8.64977991e-04 + 3.83102719e-21j,
            3.86829674e-04 + 1.71328529e-21j,
            8.64978428e-04 + 3.83103106e-21j,
            -8.64977882e-04 - 3.83102622e-21j,
            -3.45630749e-05 + 0.00000000e00j,
            -1.39503168e-10 + 0.00000000e00j,
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
