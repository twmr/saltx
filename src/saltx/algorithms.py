# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from __future__ import annotations

import dataclasses
import logging
import time
import typing

import numpy as np
import pandas as pd
from dolfinx import fem, geometry
from dolfinx.fem import DirichletBC
from petsc4py import PETSc
from slepc4py import SLEPc

from saltx import newtils
from saltx.log import Timer

if typing.TYPE_CHECKING:
    from saltx.lasing import NonLinearProblem

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


class RefinementError(Exception):
    """Raised when `refine_modes` can't successfully refine the given modes."""


@dataclasses.dataclass
class NEVPNonLasingMode:  # The solutions of a nonlinear (in k) EVP
    array: np.ndarray
    k: complex
    error: float
    # TODO introduce a pump_parameter instead of D0
    # D0: float
    bcs_name: str | None
    bcs: list
    dof_at_maximum: int


@dataclasses.dataclass
class NEVPNonLasingModeRealK:  # The newton modes (above the threshold)
    array: np.ndarray  # fixed phase
    s: float
    k: float
    dof_at_maximum: int
    # TODO introduce a pump_parameter instead of D0
    # D0: float
    newton_info_df: pd.DataFrame
    newton_deltax_norm: float
    newton_error: float
    setup_time: float
    computation_time: float
    converged: bool


@dataclasses.dataclass
class NEVPInputs:
    gt: float
    ka: float
    rg_params: tuple[float]
    L: PETSc.Mat
    M: PETSc.Mat
    # cold cavity dielectric loss term
    N: PETSc.Mat | None
    # The matrix Q depends on a pump parameter (usually this pump parameter is called
    # D0)
    Q: PETSc.Mat
    R: PETSc.Mat | None  # can only be used for 1D systems
    bcs: list[DirichletBC]


class Evaluator:
    """Evaluates a mode at certain points."""

    def __init__(self, V, msh, points):
        self.V = V
        self.msh = msh
        self.points = points

        if points.ndim == 2:
            # points is 2 x N or 3 x N
            npoints = np.zeros((3, points.shape[1]))
            npoints[: points.shape[0]] = points
        elif points.ndim == 1:
            npoints = np.zeros((3, points.size))
            npoints[0] = points
        else:
            raise ValueError("Unsupported ndim")

        bb_tree = geometry.bb_tree(msh, msh.topology.dim)

        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, npoints.T)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(
            msh, cell_candidates, npoints.T
        )
        for i, point in enumerate(npoints.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        self.points_on_proc = np.array(points_on_proc, dtype=np.float64)
        self.cells = cells

        self.femfunction = fem.Function(V)

    def __call__(self, mode) -> np.ndarray:
        self.femfunction.x.array[:] = (
            mode if isinstance(mode, np.ndarray) else mode.array
        )
        return self.femfunction.eval(self.points_on_proc, self.cells).flatten()


def get_nevp_modes(
    nevp_inputs: NEVPInputs,
    bcs_name: str = "default",
) -> list[NEVPNonLasingMode]:
    """Calculate the non-linear eigenmodes using a CISS solver.

    This function uses the CISS solver from NEP package from SLEPc. For
    details about this solver see
    https://slepc.upv.es/documentation/reports/str11.pdf
    """
    t0 = time.monotonic()
    gt = nevp_inputs.gt
    ka = nevp_inputs.ka
    L = nevp_inputs.L
    M = nevp_inputs.M
    N = nevp_inputs.N
    Q = nevp_inputs.Q
    R = nevp_inputs.R

    # Prefactors used for the split operator:
    # (for L) f1=1.0,
    # (for M) f2=k^2,
    # (for Q) f3=k^2*gamma(k)
    # (for R) f4=1j*k
    # (for N) f5=f4  # TODO think about merging R and N
    f1 = SLEPc.FN().create()
    f1.setType(SLEPc.FN.Type.RATIONAL)
    f1.setRationalNumerator([1.0])
    f2 = SLEPc.FN().create()
    f2.setType(SLEPc.FN.Type.RATIONAL)
    f2.setRationalNumerator([1.0, 0.0, 0.0])

    # k**2 * gamma(k) = gt * k**2 / (k - ka +igt)
    f3 = SLEPc.FN().create()
    f3.setType(SLEPc.FN.Type.RATIONAL)
    f3.setRationalNumerator([gt, 0.0, 0.0])
    f3.setRationalDenominator([1.0, -ka + 1j * gt])

    f4 = SLEPc.FN().create()
    f4.setType(SLEPc.FN.Type.RATIONAL)
    f4.setRationalNumerator([1j, 0])

    # Setup the solver
    nep = SLEPc.NEP().create()

    operators = [L, M, Q, R] if Q else [L, M, R]
    fractions = [f1, f2, f3, f4] if Q else [f1, f2, f4]
    if R is None and N is None:
        operators = operators[:-1]
        fractions = fractions[:-1]
    elif R is None:
        assert N is not None
        operators[-1] = N
    elif N is None:
        assert R is not None
    else:
        assert N is not None
        assert R is not None
        operators.append(N)
        fractions.append(f4)

    # TODO add proper support (document it) for solving the QB states with Q=0
    nep.setSplitOperator(
        operators,
        fractions,
        PETSc.Mat.Structure.SUBSET,
    )

    nep.setTolerances(tol=1e-7)
    nep.setDimensions(nev=24)
    nep.setType(SLEPc.NEP.Type.CISS)

    RG = nep.getRG()
    RG.setType(SLEPc.RG.Type.ELLIPSE)
    Print(f"RG params: {nevp_inputs.rg_params}")
    RG.setEllipseParameters(*nevp_inputs.rg_params)

    nep.setFromOptions()

    # Solve the problem
    nep.solve()

    its = nep.getIterationNumber()
    sol_type = nep.getType()
    nev, ncv, mpd = nep.getDimensions()
    Print(f"Number of iterations of the {sol_type} method: {its}")
    Print(f"({nev=}, {ncv=}, {mpd=})")
    tol, maxit = nep.getTolerances()
    Print(f"Stopping condition: {tol=:.4g}, {maxit=}")
    nconv = nep.getConverged()
    Print(f"Number of converged eigenpairs {nconv}")
    assert nconv

    x = L.createVecs("right")

    modes = []
    Print()
    Print("        lam              ||T(lam)x||  m[norm_idx]")
    Print("----------------- ------------------ ------------")
    for i in range(nconv):
        _lam = nep.getEigenpair(i, x)
        res = nep.computeError(i)
        mode = x.getArray().copy()

        dof_at_maximum = np.abs(mode).argmax()
        val_maximum = mode[np.abs(mode).argmax()]
        Print(
            f" {_lam.real:9f}{_lam.imag:+9f} j {res:12g}   "
            f"{val_maximum.real:2g} j {val_maximum.imag:2g}"
        )

        # fix norm and the phase
        mode /= val_maximum
        modes.append(
            NEVPNonLasingMode(
                array=mode,
                k=_lam,
                error=res,
                bcs_name=bcs_name,
                bcs=nevp_inputs.bcs or [],
                dof_at_maximum=dof_at_maximum,
            )
        )
    Print()
    log.info(f"NEVP solver took {time.monotonic()-t0:.1f}s")
    return modes


def refine_modes(
    modeinfos,
    bcs,
    solver,
    nlp,
    nlA,
    nlL,
    delta_x,
    initial_x,
    fail_early=False,
):
    if fail_early:
        refmodes = _refine_modes(
            modeinfos,
            bcs,
            solver,
            nlp,
            nlA,
            nlL,
            delta_x,
            initial_x,
            relaxation_parameter=1.0,
        )
        if all([refmode.converged for refmode in refmodes]):
            return refmodes
        raise RefinementError("Couldn't successfully refine modes")

    relparams = [1.0, 0.8, 0.6, 0.5]

    while relparams:
        relaxation_parameter = relparams.pop(0)
        log.info(f"Current relaxation parameter={relaxation_parameter}")
        try:
            refmodes = _refine_modes(
                modeinfos,
                bcs,
                solver,
                nlp,
                nlA,
                nlL,
                delta_x,
                initial_x,
                relaxation_parameter=relaxation_parameter,
            )
            if all([refmode.converged for refmode in refmodes]):
                return refmodes
            log.error("Convergence error in refine_modes -> Perform retry")
        except Exception:
            log.exception(f"Refine modes failed at {relaxation_parameter=}")

            # TODO improve this (Do I really need to create a new solver?)
            new_solver = PETSc.KSP().create()
            new_solver.setOperators(solver.getOperators()[0])
            # Preconditioner (this has a huge impact on performance!!!)
            PC = new_solver.getPC()
            PC.setType("lu")
            PC.setFactorSolverType("mumps")
            solver = new_solver

    raise RefinementError("Couldn't successfully refine mode")


def _refine_modes(
    modeinfos,
    bcs,
    solver,
    nlp,
    nlA,
    nlL,
    delta_x,
    initial_x,
    relaxation_parameter,
):
    nmodes = len(modeinfos)
    newtils.fill_vector_with_modeinfos(initial_x, modeinfos)
    initial_dof_at_maximum_seq = [minfo.dof_at_maximum for minfo in modeinfos]

    # TODO maybe we should call set bcs here in nlp:
    # nlp.bcs = mode.bcs (instead of passing the bcs to assemble_*)

    max_iterations = 30
    i = 0

    newton_steps = []
    converged = None
    minfos = modeinfos

    sanity_checks = False  # could be disabled in the future
    while i < max_iterations and converged is None:
        tstart = time.monotonic()
        with Timer(Print, "assemble J matrix and F vec"):
            nlp.assemble_F_and_J(nlL, nlA, minfos, bcs)

        nlL.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
        # Scale residual by -1
        nlL.scale(-1)
        nlL.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        if sanity_checks:
            newtils.check_vector_real(nlL, nmodes=nmodes)
            newtils.check_matrix_real(nlA, nmodes=nmodes)

        with Timer(Print, "Solve KSP"):
            solver.solve(nlL, delta_x)

        initial_x += delta_x * relaxation_parameter

        if sanity_checks:
            newtils.check_vector_real(delta_x, nmodes=nmodes, threshold=1e-15)

        minfos = newtils.extract_newton_mode_infos(
            initial_x, nmodes=nmodes, real_sanity_check=sanity_checks
        )
        for minfo, dof_at_maximum in zip(minfos, initial_dof_at_maximum_seq):
            minfo.dof_at_maximum = dof_at_maximum

        cur_k, cur_s = [mi.k for mi in minfos], [mi.s for mi in minfos]
        delta_k, delta_s = newtils.extract_k_and_s(
            delta_x, nmodes=nmodes, real_sanity_check=sanity_checks
        )
        Print(f"DELTA k: {delta_k}, s: {delta_s}")

        i += 1

        # Compute norm of update
        correction_norm = delta_x.norm(0)

        newton_steps.append(
            cur_k + cur_s + [correction_norm, time.monotonic() - tstart]
        )

        print(f"----> Iteration {i}: Correction norm {correction_norm}")
        if correction_norm < 1e-10:
            if correction_norm < 1e-16:
                # sometimes the correction norm is 0.0 (not sure why)
                converged = False
            else:
                converged = True
                break
        if correction_norm > 1e6:
            converged = False
    else:
        converged = False

    if not converged:
        log.error(f"Mode Refinement of {modeinfos} didn't converge")

    Print(
        f"Initial k: {[mi.k for mi in modeinfos]}, s: {[mi.s for mi in modeinfos]} "
        f"{relaxation_parameter=}"
    )
    newton_df = pd.DataFrame(
        newton_steps,
        columns=(
            [f"k{i}" for i in range(nmodes)]
            + [f"s{i}" for i in range(nmodes)]
            + ["corrnorm", "dt"]
        ),
    )
    Print(newton_df)

    real_modes = []
    computation_time = time.monotonic() - tstart
    for minfo in minfos:
        only_mode_values = (minfo.re_array + 1j * minfo.im_array) * minfo.s
        real_mode = NEVPNonLasingModeRealK(
            array=only_mode_values,
            k=minfo.k,
            s=minfo.s,
            # D0=nlp.D0,
            dof_at_maximum=minfo.dof_at_maximum,
            converged=converged,
            setup_time=0,  # tsetup_end - tsetup_start,
            computation_time=computation_time,
            newton_info_df=newton_df,
            newton_deltax_norm=correction_norm,
            newton_error=-1,
        )
        real_modes.append(real_mode)
    return real_modes


def constant_pump_algorithm(
    nevp_modes: list[NEVPNonLasingMode],
    nevp_inputs: NEVPInputs,
    nlp: NonLinearProblem,
    newton_operators: dict[int, newtils.NewtonMatricesAndSolver],
    s_init: float = 1.0,
    first_mode_index: int | None = None,
    real_axis_threshold=1e-10,
):
    modes = nevp_modes
    evals = np.asarray([mode.k for mode in modes])
    minfos = []

    active_modes = 0
    while True:
        active_modes += 1

        # I think this criterion is not good enough for the 1D slab system used in the
        # Generalizations paper (Fig 3) due to the complicated eigenvalue trajectories
        # (function of D0)
        mode = modes[evals.imag.argmax()]
        if mode.k.imag < -1e-10 and active_modes == 1:
            # no mode above threshold found
            return []

        if first_mode_index is not None and active_modes == 1:
            mode = modes[first_mode_index]

        minfos.append(
            newtils.NewtonModeInfo(
                k=mode.k.real,
                s=s_init,
                re_array=mode.array.real,
                im_array=mode.array.imag,
                dof_at_maximum=mode.dof_at_maximum,
            )
        )
        assert len(minfos) == active_modes

        refined_modes = refine_modes(
            minfos,
            mode.bcs,
            newton_operators[active_modes].solver,
            nlp,
            newton_operators[active_modes].A,
            newton_operators[active_modes].L,
            newton_operators[active_modes].delta_x,
            newton_operators[active_modes].initial_x,
        )
        assert all(rm.converged for rm in refined_modes)

        log.debug(
            f"Before assembly of Q with custom sht (active_modes:{len(refined_modes)})"
        )
        # this modifies the Q matrix in nevp_inputs
        nlp.update_b_and_k_for_forms(refined_modes)
        Q = nevp_inputs.Q
        Q.zeroEntries()
        fem.petsc.assemble_matrix(
            Q, nlp.get_Q_hbt_form(active_modes), bcs=mode.bcs, diagonal=0.0
        )
        Q.assemble()
        log.debug("After assembly of Q with custom sht")

        modes = get_nevp_modes(nevp_inputs)
        evals = np.asarray([mode.k for mode in modes])

        number_of_modes_close_to_real_axis = np.sum(
            np.abs(evals.imag) < real_axis_threshold
        )
        Print(
            f"Number of modes close to real axis: {number_of_modes_close_to_real_axis}"
        )

        assert number_of_modes_close_to_real_axis == active_modes

        # TODO add a couple of sanity checks (refined_modes vs modes)

        number_of_modes_above_real_axis = np.sum(evals.imag > real_axis_threshold)
        Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")
        if number_of_modes_above_real_axis == 0:
            return refined_modes

        minfos = [
            # The modes have to be normalized
            newtils.NewtonModeInfo(
                k=mode.k,
                s=s_init,
                re_array=mode.array.real,
                im_array=mode.array.imag,
                dof_at_maximum=mode.dof_at_maximum,
            )
            for mode in modes
            if abs(mode.k.imag) < real_axis_threshold
        ]
        assert len(minfos) == active_modes
    raise RuntimeError("unreachable point reached")
