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
import ufl
from dolfinx import fem, geometry
from petsc4py import PETSc
from slepc4py import SLEPc
from ufl import dx, inner

from saltx import newtils
from saltx.log import Timer

if typing.TYPE_CHECKING:
    from saltx.lasing import NonLinearProblem

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


@dataclasses.dataclass
class NEVPNonLasingMode:  # The solutions of a nonlinear (in k) EVP
    array: np.ndarray
    k: complex
    error: float
    # TODO introduce a pump_parameter instead of D0
    # D0: float
    bcs_name: str | None
    bcs: list


@dataclasses.dataclass
class NEVPNonLasingModeRealK:  # The newton modes (above the threshold)
    array: np.ndarray  # fixed phase
    s: float
    k: float
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
    # cold cavity conduction loss
    N: PETSc.Mat | None
    # The matrix Q depends on a pump parameter (usually this pump parameter is
    # called D0)
    Q: PETSc.Mat
    R: PETSc.Mat | None  # can only be used for 1D systems
    bcs_norm_constraint: np.ndarray


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

        bb_tree = geometry.BoundingBoxTree(msh, msh.topology.dim)

        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions(bb_tree, npoints.T)
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
    custom_Q=None,
    bcs_name="default",
    bcs=None,
) -> list[NEVPNonLasingMode]:
    """Calculate the non-linear eigenmodes using a CISS solver.

    This function uses the CISS solver from NEP package from SLEPc. For
    details about this solver see
    https://slepc.upv.es/documentation/reports/str11.pdf
    """
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

    if custom_Q is None:
        custom_Q = Q

    operators = [L, M, custom_Q, R] if custom_Q else [L, M, R]
    fractions = [f1, f2, f3, f4] if custom_Q else [f1, f2, f4]
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
    Print("Number of iterations of the method: %i" % its)
    sol_type = nep.getType()
    Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = nep.getDimensions()
    Print("")
    Print("Subspace dimension: %i" % ncv)
    tol, maxit = nep.getTolerances()
    Print("Stopping condition: tol=%.4g" % tol)
    Print("")

    nconv = nep.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    x = L.createVecs("right")

    modes = []
    assert nconv
    Print()
    Print("        lam              ||T(lam)x||  m[norm_idx]")
    Print("----------------- ------------------ ------------")
    for i in range(nconv):
        _lam = nep.getEigenpair(i, x)
        res = nep.computeError(i)
        mode = x.getArray().copy()
        norm_val = mode[nevp_inputs.bcs_norm_constraint[0]]
        Print(
            f" {_lam.real:9f}{_lam.imag:+9f} j {res:12g}   "
            f"{norm_val.real:2g} j {norm_val.imag:2g}"
        )

        # fix norm and the phase
        mode /= mode[nevp_inputs.bcs_norm_constraint[0]]
        modes.append(
            NEVPNonLasingMode(
                array=mode, k=_lam, error=res, bcs_name=bcs_name, bcs=bcs or []
            )
        )
    Print()
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
        raise ValueError("Couldn't successfully refine modes")

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

    raise ValueError("Couldn't successfully refine mode")


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

        initial_x += relaxation_parameter * delta_x

        if sanity_checks:
            newtils.check_vector_real(delta_x, nmodes=nmodes, threshold=1e-15)

        minfos = newtils.extract_newton_mode_infos(
            initial_x, nmodes=nmodes, real_sanity_check=sanity_checks
        )
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

    Print(f"Initial k: {[mi.k for mi in modeinfos]}, {relaxation_parameter=}")
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
    pump_expr,
    nlp: NonLinearProblem,
    newton_operators: dict[int, newtils.NewtonMatricesAndSolver],
    to_const: callable,
    assemble_form: callable,
    system,  # container for system related parameters
    s_init: float = 1.0,
    first_mode_index: int | None = None,
):
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    modes = nevp_modes
    evals = np.asarray([mode.k for mode in modes])
    minfos = []

    ka = to_const(system.ka)
    gt = to_const(system.gt)

    for i in range(len(nevp_modes)):
        # I think this criterion is not good enough for the 1D slab system used in the
        # Generalizations paper (Fig 3) due to the complicated eigenvalue trajectories
        # (function of D0)
        mode = modes[evals.imag.argmax()]
        if mode.k.imag < -1e-10 and i == 0:
            # no mode above threshold found
            return []

        if first_mode_index is not None and i == 0:
            mode = modes[first_mode_index]

        minfos.append(
            newtils.NewtonModeInfo(
                k=mode.k.real,
                s=s_init,
                re_array=mode.array.real,
                im_array=mode.array.imag,
            )
        )

        refined_modes = refine_modes(
            minfos,
            mode.bcs,
            newton_operators[i + 1].solver,
            nlp,
            newton_operators[i + 1].A,
            newton_operators[i + 1].L,
            newton_operators[i + 1].delta_x,
            newton_operators[i + 1].initial_x,
        )
        assert all(rm.converged for rm in refined_modes)

        sht = 0
        for refined_mode in refined_modes:
            k_sht = to_const(refined_mode.k)
            b_sht = fem.Function(system.V)
            b_sht.x.array[:] = refined_mode.array
            gk_sht = gt / (k_sht - ka + 1j * gt)
            sht += abs(gk_sht * b_sht) ** 2

        log.debug(f"Before assembly of Q with custom sht (nmodes:{len(refined_modes)})")
        Q_with_sht = assemble_form(
            pump_expr / (1 + sht) * inner(u, v) * dx,
            zero_diag=True,
        )
        log.debug("After assembly of Q with custom sht")
        modes = get_nevp_modes(nevp_inputs, custom_Q=Q_with_sht, bcs=mode.bcs)
        evals = np.asarray([mode.k for mode in modes])

        number_of_modes_close_to_real_axis = np.sum(np.abs(evals.imag) < 1e-10)
        Print(
            f"Number of modes close to real axis: {number_of_modes_close_to_real_axis}"
        )

        assert number_of_modes_close_to_real_axis == i + 1

        # TODO add a couple of sanity checks (refined_modes vs modes)

        number_of_modes_above_real_axis = np.sum(evals.imag > 1e-10)
        Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")
        if number_of_modes_above_real_axis == 0:
            return refined_modes

        minfos = [
            newtils.NewtonModeInfo(
                k=refined_mode.k,
                s=s_init,
                re_array=refined_mode.array.real,
                im_array=refined_mode.array.imag,
            )
            for refined_mode in refined_modes
        ]
    raise RuntimeError("unreachable point reached")
