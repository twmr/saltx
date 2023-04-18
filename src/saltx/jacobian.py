# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from .log import Timer


def create_salt_jacobian_block_matrix(dF_dvw: PETSc.Mat, nmodes: int = 1) -> PETSc.Mat:
    """creates matrix with the expected sparsity structure but doesn't fill it
    with values."""
    # CSR matrix example:
    # Input : 0  0  0  0
    #         5  8  0  0
    #         0  0  3  0
    #         0  6  0  0
    # values = [5 8 3 6]
    # IA = [0 0 2 3 4]  # rows_ind
    # JA = [0 1 2 1]    # cols

    # TODO nmodes > 2 support
    two_nmodes_n = dF_dvw.getSize()[0]  # 2*nmodes*nfem
    N = two_nmodes_n + 2 * nmodes
    n = two_nmodes_n // (2 * nmodes)

    with Timer(print, "getValuesCSR"):
        rows_ind, cols, values = dF_dvw.getValuesCSR()

    assert rows_ind.size - 1 == 2 * n * nmodes
    assert values.size == cols.size
    # rows_ind == IA
    # cols_ind == JA

    final_rows_ind = np.zeros(rows_ind.size + 2 * nmodes, dtype=rows_ind.dtype)
    # add two*nmodes dense column vectors
    final_rows_ind[: rows_ind.size] = rows_ind + 2 * nmodes * np.arange(rows_ind.size)

    # row-indices of the last three rows in the big NxN matrix

    nm3 = two_nmodes_n - 1  # last row before the first et row vector
    # nm2 = N - 2  # second last
    # nm1 = N - 1  # last

    assert final_rows_ind.size == N + 1
    # since the size of final_rows_ind is N+1, we have to add +1 to the mn123
    # indices
    assert final_rows_ind[nm3 + 1] > 0
    assert (final_rows_ind[-2 * nmodes :] == 0).all()
    # now we have to set the last two*nmodes rows

    len_et = n

    bottom_row_indices = range(nm3 + 2, N + 1)
    # TOOD pass a single fixed_phase_dof value to create_real_matrix instead of
    # two vectors, then we need two following two lines:
    diagonal_entry = 1  # we have to add zeros at the diag.
    assert rows_ind.size == bottom_row_indices[0]
    for row_index in bottom_row_indices:
        final_rows_ind[row_index] = final_rows_ind[row_index - 1] + (
            len_et + diagonal_entry
        )

    # calculate cols
    all_cols = []

    prev_row_ind = 0
    cur_col_ptr = 0
    # there is shift by one (compared to right_col_vector_indices), because
    # final_rows_ind starts with 0:
    newcols = np.r_[range(nm3 + 1, N)]
    for rid in rows_ind[1:]:
        nnz_in_current_row = rid - prev_row_ind
        if nnz_in_current_row:
            all_cols.append(
                np.r_[cols[cur_col_ptr : cur_col_ptr + nnz_in_current_row], newcols]
            )
            cur_col_ptr += nnz_in_current_row
        else:
            all_cols.append(newcols)

        prev_row_ind = rid

    # last two*nmodes rows:
    # f = Real(e^T (v+1jw) - 1) = e^T v - 1
    # g = Imag(e^T (v+1jw) - 1) = e^T w
    # -> dfx_dvx = dgx_dwx = ex^T  # x is the mode index
    # [df1_dv1, 0,       0,       0,      | 0, 0, 0, 0]
    # [0,       dg1_dw1, 0,       0,      | 0, 0, 0, 0]
    # [0,       0,       df2_dv2, 0,      | 0, 0, 0, 0]
    # [0,       0,       0,       dg2_dw2,| 0, 0, 0, 0]

    # TODO see above TODO comment
    # all_cols.append(np.r_[fixed_phase_dof, fixed_phase_dof + n])

    if nmodes == 1:
        all_cols.append(
            np.r_[
                np.arange(len_et, dtype=final_rows_ind.dtype),
                # previous last diagonal entry:
                nm3 + 1,
                np.arange(len_et, 2 * len_et, dtype=final_rows_ind.dtype),
                # last diagonal entry:
                nm3 + 2,
            ]
        )
    elif nmodes == 2:
        all_cols.append(
            np.r_[
                np.arange(len_et, dtype=final_rows_ind.dtype),
                # diagonal entry
                nm3 + 1,
                np.arange(len_et, 2 * len_et, dtype=final_rows_ind.dtype),
                # diagonal entry
                nm3 + 2,
                np.arange(2 * len_et, 3 * len_et, dtype=final_rows_ind.dtype),
                # previous last diagonal entry:
                nm3 + 3,
                np.arange(3 * len_et, 4 * len_et, dtype=final_rows_ind.dtype),
                # last diagonal entry:
                nm3 + 4,
            ]
        )
    else:
        raise ValueError(f"{nmodes=} not supported")

    final_cols = np.concatenate(all_cols, dtype=final_rows_ind.dtype)
    # TODO is there a petsc4py function that doesn't require values?
    final_values = np.ones(final_rows_ind[-1], dtype=np.complex128)

    assert final_rows_ind.dtype == np.int32
    assert final_cols.dtype == np.int32
    assert final_values.dtype == np.complex128

    with Timer(print, "createAIJ"):
        A = PETSc.Mat().createAIJWithArrays(
            (N, N),
            (final_rows_ind, final_cols, final_values),
        )
        A.zeroEntries()
    return A


def assemble_salt_jacobian_block_matrix(
    A: PETSc.Mat,
    dF_dvw: PETSc.Mat,
    dFReIm_dk_seq: list[PETSc.Vec],
    dFReIm_ds_seq: list[PETSc.Vec],
    dof_at_maximum_seq: list[int],
    nmodes: int,
) -> None:
    """
    Parameters
    ----------
    A
        Jacobian matrix (=result)
    """
    t0 = time.monotonic()

    A.zeroEntries()
    N = A.getSize()[0]

    n = dF_dvw.getSize()[0] // (2 * nmodes)

    assert 2 * n * nmodes + 2 * nmodes == N
    assert len(dFReIm_ds_seq) == nmodes
    assert len(dFReIm_dk_seq) == nmodes
    assert len(dof_at_maximum_seq) == nmodes

    if False:  # for debugging
        fnamemap = dict(
            dF_dvw=dF_dvw,
        )

        for fname, pobj in fnamemap.items():
            viewer = PETSc.Viewer().createBinary(
                fname, mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD
            )
            viewer(pobj)

        raise ValueError("END")

    def insert_values(target_matrix, block_matrix, row_offset, col_offset):
        rows_ind, cols, values = block_matrix.getValuesCSR()

        cols_to = cols + col_offset

        rows_ind_to = np.empty(
            N + 1, dtype=np.int32
        )  # +1 because rows_ind_to[0] is always 0
        _idx = 0
        rows_ind_to[_idx : _idx + row_offset] = 0
        _idx = _idx + row_offset
        rows_ind_to[_idx : _idx + len(rows_ind)] = rows_ind
        _idx = _idx + len(rows_ind)
        rows_ind_to[_idx:] = len(values)

        t_setCSR_0 = time.monotonic()
        target_matrix.setValuesCSR(
            rows_ind_to, cols_to, values, addv=PETSc.InsertMode.ADD
        )
        print(f"single setValuesCSR call took {time.monotonic()-t_setCSR_0:.2f}s")

    t_insert_0 = time.monotonic()
    # matrix, row offset, col offset
    insert_values(A, dF_dvw, 0, 0)
    print(f"insert values of J took {time.monotonic()-t_insert_0:.2e}s")

    addv = PETSc.InsertMode.ADD

    # column vectors
    # we have 2*nmodes additional col vectors
    # k1, s1, k2, s2, ...., k_nmodes, s_nmodes
    for kidx, dF_dk in enumerate(dFReIm_dk_seq):
        col_idx = 2 * n * nmodes + 2 * kidx
        A.setValues(
            range(2 * n * nmodes),
            col_idx,
            dF_dk.array.real,
            addv=addv,
        )
    for sidx, dF_ds in enumerate(dFReIm_ds_seq):
        col_idx = 2 * n * nmodes + 2 * sidx + 1
        A.setValues(
            range(2 * n * nmodes),
            col_idx,
            dF_ds.array.real,
            addv=addv,
        )

    # row vectors
    # we have 2*nmodes additional row vectors
    for midx, dof_at_maximum in enumerate(dof_at_maximum_seq):
        row_idx = 2 * n * nmodes + 2 * midx
        col_shift = 2 * n * midx
        A.setValue(row_idx, col_shift + dof_at_maximum, 1.0, addv=addv)
        A.setValue(row_idx + 1, col_shift + n + dof_at_maximum, 1.0, addv=addv)

    print(f"insert values + J.setValues of J took {time.monotonic()-t_insert_0:.2e}s")

    # scalars
    # this is needed s.t. PETSc doesn't
    # complain that the diagonal entry is missing
    # (see https://lists.mcs.anl.gov/pipermail/petsc-users/2016-October/030704.html)

    for i in range(2 * nmodes):
        A.setValue(2 * n * nmodes + i, 2 * n * nmodes + i, 0j, addv=addv)

    t1 = time.monotonic()
    A.assemble()
    print(
        f"fill+assembly of real J took {time.monotonic()-t0:.2e}s "
        f"(assemble {time.monotonic()-t1:.2e}s)"
    )


def assemble_complex_singlemode_jacobian_matrix(
    A: PETSc.Mat,
    dF_du,
    dF_dk,
    dof_at_maximum: int,
) -> None:
    """
    Parameters
    ----------
    A
        Jacobian matrix (=result)
    """
    t0 = time.monotonic()

    A.zeroEntries()
    N = A.getSize()[0]
    n = N - 1

    rows_ind, cols, values = dF_du.getValuesCSR()

    cols_to = cols
    rows_ind_to = np.empty(
        N + 1, dtype=np.int32
    )  # +1 because rows_ind_to[0] is always 0
    _idx = 0
    rows_ind_to[_idx : _idx + len(rows_ind)] = rows_ind
    _idx = _idx + len(rows_ind)
    rows_ind_to[_idx:] = len(values)

    A.setValuesCSR(rows_ind_to, cols_to, values, addv=PETSc.InsertMode.ADD)

    # column vectors
    A.setValues(
        range(n),
        n,
        dF_dk,
        addv=PETSc.InsertMode.ADD,
    )

    # row vectors
    A.setValue(n, dof_at_maximum, 1.0, addv=PETSc.InsertMode.ADD)

    # scalars

    # this is needed s.t. PETSc doesn't complain that the diagonal entry is missing (see
    # https://lists.mcs.anl.gov/pipermail/petsc-users/2016-October/030704.html)
    A.setValue(n, n, 0j, addv=PETSc.InsertMode.ADD)

    t1 = time.monotonic()
    A.assemble()
    print(
        f"fill+assembly of complex J took {time.monotonic()-t0:.2e}s "
        f"(assemble {time.monotonic()-t1:.2e}s)"
    )
