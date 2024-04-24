# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_matrix_block, create_vector_block
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx.jacobian import (
    assemble_salt_jacobian_block_matrix,
    create_salt_jacobian_block_matrix,
)

Print = PETSc.Sys.Print


def ass_linear_form(form):
    return fem.petsc.assemble_vector(fem.form(form))


def ass_linear_form_into_vec(vec, form, a_form, bcs):
    with vec.localForm() as vec_local:
        vec_local.set(0.0)
    fem.petsc.assemble_vector_block(vec, fem.form(form), a_form, bcs=bcs)
    # vec.assemble()


@pytest.mark.parametrize("nmodes", [1, 2])
def test_create_salt_jacobian_block_matrix(nmodes):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 5)

    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    tstf = ufl.TestFunction(V)
    trif = ufl.TrialFunction(V)
    dFRe_dv = inner(trif, tstf) * dx
    dFRe_dw = inner(trif, tstf) * dx
    dFIm_dv = inner(trif, tstf) * dx
    dFIm_dw = inner(trif, tstf) * dx

    form_list = [[dFRe_dv, dFRe_dw] * nmodes, [dFIm_dv, dFIm_dw] * nmodes] * nmodes
    form_array = fem.form(form_list)

    block_mat = create_matrix_block(form_array)
    fem.petsc.assemble_matrix_block(
        block_mat,
        form_array,
        bcs=[],
        diagonal=1.0,
    )
    block_mat.assemble()

    nfem = 12
    assert block_mat.getSize() == (nfem * nmodes, nfem * nmodes)

    A = create_salt_jacobian_block_matrix(block_mat, nmodes=nmodes)
    assert A.getSize() == ((nfem + 2) * nmodes, (nfem + 2) * nmodes)

    rows_ind, cols, _ = A.getValuesCSR()

    assert len(rows_ind) == (nfem + 2) * nmodes + 1

    if nmodes > 1:
        assert (
            rows_ind
            == np.array(
                [
                    0,
                    12,
                    28,
                    44,
                    60,
                    76,
                    88,
                    100,
                    116,
                    132,
                    148,
                    164,
                    176,
                    188,
                    204,
                    220,
                    236,
                    252,
                    264,
                    276,
                    292,
                    308,
                    324,
                    340,
                    352,
                    359,
                    366,
                    373,
                    380,
                ],
                dtype=np.int32,
            )
        ).all()
        return

    assert (
        rows_ind
        == np.array(
            [0, 6, 14, 22, 30, 38, 44, 50, 58, 66, 74, 82, 88, 95, 102],
            dtype=np.int32,
        )
    ).all()

    assert (
        cols
        == np.array(
            [
                0,
                1,
                6,
                7,
                12,
                13,
                0,
                1,
                2,
                6,
                7,
                8,
                12,
                13,
                1,
                2,
                3,
                7,
                8,
                9,
                12,
                13,
                2,
                3,
                4,
                8,
                9,
                10,
                12,
                13,
                3,
                4,
                5,
                9,
                10,
                11,
                12,
                13,
                4,
                5,
                10,
                11,
                12,
                13,
                0,
                1,
                6,
                7,
                12,
                13,
                0,
                1,
                2,
                6,
                7,
                8,
                12,
                13,
                1,
                2,
                3,
                7,
                8,
                9,
                12,
                13,
                2,
                3,
                4,
                8,
                9,
                10,
                12,
                13,
                3,
                4,
                5,
                9,
                10,
                11,
                12,
                13,
                4,
                5,
                10,
                11,
                12,
                13,
                0,
                1,
                2,
                3,
                4,
                5,
                12,
                6,
                7,
                8,
                9,
                10,
                11,
                13,
            ],
            dtype=np.int32,
        )
    ).all()


@pytest.mark.parametrize("nmodes", [1, 2])
def test_create_and_assemble_salt_jacobian(nmodes):
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, nx=3000)
    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    Wre, Wim = V.clone(), V.clone()

    u0, u1 = ufl.TrialFunction(Wre), ufl.TrialFunction(Wim)
    v0, v1 = ufl.TestFunction(Wre), ufl.TestFunction(Wim)
    if nmodes == 2:
        W2re, W2im = V.clone(), V.clone()

        u2, u3 = ufl.TrialFunction(W2re), ufl.TrialFunction(W2im)
        v2, v3 = ufl.TestFunction(W2re), ufl.TestFunction(W2im)

    bcs = []

    b = fem.Function(V)
    b.x.array[:] = 1.234

    if nmodes == 1:
        bilin_form00 = -inner(nabla_grad(u0), nabla_grad(v0)) * dx
        bilin_form01 = -inner(nabla_grad(u1), nabla_grad(v0)) * dx
        bilin_form10 = -inner(nabla_grad(u0), nabla_grad(v1)) * dx
        bilin_form11 = -inner(nabla_grad(u1), nabla_grad(v1)) * dx
    if nmodes == 2:
        bilin_form00 = -inner(nabla_grad(u0), nabla_grad(v0)) * dx
        bilin_form01 = -inner(nabla_grad(u1), nabla_grad(v0)) * dx
        bilin_form02 = -inner(nabla_grad(u2), nabla_grad(v0)) * dx
        bilin_form03 = -inner(nabla_grad(u3), nabla_grad(v0)) * dx

        bilin_form10 = -inner(nabla_grad(u0), nabla_grad(v1)) * dx
        bilin_form11 = -inner(nabla_grad(u1), nabla_grad(v1)) * dx
        bilin_form12 = -inner(nabla_grad(u2), nabla_grad(v1)) * dx
        bilin_form13 = -inner(nabla_grad(u3), nabla_grad(v1)) * dx

        bilin_form20 = -inner(nabla_grad(u0), nabla_grad(v2)) * dx
        bilin_form21 = -inner(nabla_grad(u1), nabla_grad(v2)) * dx
        bilin_form22 = -inner(nabla_grad(u2), nabla_grad(v2)) * dx
        bilin_form23 = -inner(nabla_grad(u3), nabla_grad(v2)) * dx

        bilin_form30 = -inner(nabla_grad(u0), nabla_grad(v3)) * dx
        bilin_form31 = -inner(nabla_grad(u1), nabla_grad(v3)) * dx
        bilin_form32 = -inner(nabla_grad(u2), nabla_grad(v3)) * dx
        bilin_form33 = -inner(nabla_grad(u3), nabla_grad(v3)) * dx

    if nmodes == 1:
        form_array = fem.form(
            [
                [bilin_form00, bilin_form01],
                [bilin_form10, bilin_form11],
            ]
        )
    elif nmodes == 2:
        form_array = fem.form(
            [
                [bilin_form00, bilin_form01, bilin_form02, bilin_form03],
                [bilin_form10, bilin_form11, bilin_form12, bilin_form13],
                [bilin_form20, bilin_form21, bilin_form22, bilin_form23],
                [bilin_form30, bilin_form31, bilin_form32, bilin_form33],
            ]
        )

    else:
        raise ValueError(nmodes)
    mat_dF_dvw = create_matrix_block(form_array)

    fem.petsc.assemble_matrix_block(
        mat_dF_dvw,
        form_array,
        bcs=bcs,
    )
    mat_dF_dvw.assemble()

    lin_form = [inner(ufl.real(b), v0) * dx, inner(ufl.imag(b), v1) * dx]
    if nmodes == 2:
        lin_form = [
            inner(ufl.real(b), v0) * dx,
            inner(ufl.imag(b), v1) * dx,
            inner(ufl.real(b), v2) * dx,
            inner(ufl.imag(b), v3) * dx,
        ]

    vec_dF_dk = create_vector_block(fem.form(lin_form))
    vec_dF_ds = create_vector_block(fem.form(lin_form))
    ass_linear_form_into_vec(vec_dF_dk, lin_form, form_array, bcs)
    fem.set_bc(vec_dF_dk, bcs)
    ass_linear_form_into_vec(vec_dF_ds, lin_form, form_array, bcs)
    fem.set_bc(vec_dF_ds, bcs)

    Print("Create salt jacobian")
    A = create_salt_jacobian_block_matrix(mat_dF_dvw, nmodes=nmodes)
    Print("Assemble salt jacobian with values")
    # FIXME doesn't support nmodes > 1

    if nmodes == 1:
        dFdk_seq = [vec_dF_dk]
        dFds_seq = [vec_dF_ds]
    if nmodes == 2:
        # F1
        vec_dF_dk1, vec_dF_dk2 = vec_dF_dk, vec_dF_dk

        dFdk_seq = [vec_dF_dk1, vec_dF_dk2]

        vec_dF_ds1, vec_dF_ds2 = vec_dF_ds, vec_dF_ds

        dFds_seq = [vec_dF_ds1, vec_dF_ds2]

    n = V.dofmap.index_map.size_global
    assert n > 500
    dof_at_maximum_seq = [500] * nmodes
    assemble_salt_jacobian_block_matrix(
        A,
        mat_dF_dvw,
        dFdk_seq,
        dFds_seq,
        dof_at_maximum_seq,
        nmodes=nmodes,
    )
