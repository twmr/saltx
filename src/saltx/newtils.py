# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dataclasses

import numpy as np
from petsc4py import PETSc


@dataclasses.dataclass(slots=True)
class NewtonModeInfo:
    k: float
    s: float
    re_array: np.ndarray
    im_array: np.ndarray

    @property
    def cmplx_array(self):
        return self.re_array + 1j * self.im_array


def check_vector_real(vector: PETSc.Vec, nmodes: int, threshold: float = 1e-17):
    # vector could be initial_x, delta_x
    N = vector.getSize()
    n = N // (2 * nmodes) - 1
    assert 2 * nmodes * (n + 1) == N

    array = vector.getArray()
    assert abs(array.imag).max() < 1e-15, (
        f"The imag.max of a real vector (size={N}) is "
        f"{abs(array.imag).max()} (at index "
        f"{abs(array.imag).argmax()}) "
    )


def check_matrix_real(matrix: PETSc.Mat, nmodes: int, threshold: float = 1e-17):
    N = matrix.getSize()[0]
    n = N // (2 * nmodes) - 1
    assert 2 * nmodes * (n + 1) == N

    _, _, array = matrix.getValuesCSR()
    assert (
        abs(array.imag).max() < 1e-15
    ), f"The imag.max of a real matrix (size={N}x{N}) is {abs(array.imag).max()}"


def extract_newton_mode_infos(
    vector: PETSc.Vec, nmodes: int, real_sanity_check=True
) -> list[NewtonModeInfo]:
    # vector could be initial_x, delta_x
    N = vector.getSize()
    n = N // (2 * nmodes) - 1
    assert 2 * nmodes * (n + 1) == N

    array = vector.getArray()
    if real_sanity_check:
        assert abs(array.imag).max() < 1e-15, (
            f"The imag.max of a real vector (size={N}) is "
            f"{abs(array.imag).max()} (at index "
            f"{abs(array.imag).argmax()}) "
        )
    array = array.real

    startidx_of_k_and_s = 2 * nmodes * n
    size_reim = 2 * n
    modeinfos = []
    for i in range(nmodes):
        # array containing both the real and the imaginary components of the mode with
        # index i (size: 2*n).
        real_imag_array = array[size_reim * i : size_reim * (i + 1)]

        k = array[startidx_of_k_and_s + i * 2]
        # TODO have a look when s changes the sign
        # TODO decrease relaxation parameter when s is negative
        s = array[startidx_of_k_and_s + i * 2 + 1]
        mi = NewtonModeInfo(
            k=k,
            s=s,
            re_array=real_imag_array[:n],
            im_array=real_imag_array[n:],
        )
        modeinfos.append(mi)
    return modeinfos


def extract_k_and_s(
    vector: PETSc.Vec, nmodes: int, real_sanity_check: bool = True
) -> tuple[list[float], list[float]]:
    minfos = extract_newton_mode_infos(
        vector, nmodes, real_sanity_check=real_sanity_check
    )
    k = [mi.k for mi in minfos]
    s = [mi.s for mi in minfos]
    return k, s


def fill_vector_with_modeinfos(
    vector: PETSc.Vec, modeinfos: list[NewtonModeInfo]
) -> None:
    N = vector.getSize()
    nmodes = len(modeinfos)
    n = N // (2 * nmodes) - 1
    assert 2 * nmodes * (n + 1) == N

    startidx_of_k_and_s = 2 * nmodes * n
    size_reim = 2 * n

    for i, modeinfo in enumerate(modeinfos):
        vector.setValues(
            range(size_reim * i, size_reim * (i + 1) - n), modeinfo.re_array
        )
        vector.setValues(
            range(size_reim * (i + 1) - n, size_reim * (i + 1)), modeinfo.im_array
        )
        vector.setValue(startidx_of_k_and_s + 2 * i, modeinfo.k)
        vector.setValue(startidx_of_k_and_s + 2 * i + 1, modeinfo.s)


@dataclasses.dataclass
class NewtonMatricesAndSolver:
    A: PETSc.Mat
    L: PETSc.Vec
    delta_x: PETSc.Vec
    initial_x: PETSc.Vec
    solver: PETSc.KSP


def create_multimode_solvers_and_matrices(
    nlp,
    max_nmodes: int,
) -> dict[int, NewtonMatricesAndSolver]:
    newton_operators = {}

    for nmodes in range(1, max_nmodes + 1):
        A = nlp.create_A(nmodes=nmodes)
        L = nlp.create_L(nmodes=nmodes)
        delta_x = nlp.create_dx(nmodes=nmodes)
        initial_x = nlp.create_dx(nmodes=nmodes)

        solver = PETSc.KSP().create(nlp.mesh.comm)
        solver.setOperators(A)
        # Preconditioner (this has a huge impact on performance!!!)
        PC = solver.getPC()
        PC.setType("lu")
        PC.setFactorSolverType("mumps")

        newton_operators[nmodes] = NewtonMatricesAndSolver(
            A=A, L=L, delta_x=delta_x, initial_x=initial_x, solver=solver
        )
    return newton_operators
