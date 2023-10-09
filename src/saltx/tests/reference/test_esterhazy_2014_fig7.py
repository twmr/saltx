# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""
import dataclasses
import itertools
import pickle
import sys
import time
from collections import namedtuple
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from ufl import curl, dx, elem_mult, inner

from saltx import algorithms, newtils
from saltx.assemble import assemble_form
from saltx.lasing import NonLinearProblem
from saltx.plot import plot_ellipse, plot_meshfunctions
from saltx.pml import RectPML

repo_dir = Path(__file__).parent.parent.parent.parent.parent

Print = PETSc.Sys.Print

log = getLogger(__name__)

# k of approx laser modes
Ka = 5.38
Kb = 4.8
Kc = 4.25
Kd = 5.92
allK = [Ka, Kb, Kc, Kd]

# we use m.k.imag > eval_threshold to avoid refining a mode that is exactly at or very
# close to the threshold
eval_threshold = 1e-8
# TODO Try to decrease this threshold to at least 1e-9
real_axis_threshold = 5e-8


@dataclasses.dataclass()
class ModeResults:
    k: float
    intensity: float


@dataclasses.dataclass()
class PumpStepResults:
    D0: float
    modes: list[ModeResults]

    # contains len(modes) real-valued eigenvalues and the complex eigenvalues of the
    # non-active modes
    nevp_eigenvalues: list[complex]
    duration: float


@dataclasses.dataclass()
class PumpSweepResults:
    pump_step_results: list[PumpStepResults]


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


# needed for mapping quadrant modes to a circle mode
quadrant_signs_for_bcs = {
    "full_dbc": np.array([[-1, 1], [1, -1]]),
    "full_nbc": np.array([[1, 1], [1, 1]]),
    "mixed1": np.array([[1, 1], [-1, -1]]),
    "mixed2": np.array([[-1, 1], [-1, 1]]),
}


def map_function_to_circle_mesh(
    mode_on_quarter_circle: algorithms.NEVPNonLasingMode,
    V_circle,
    V_quarter_circle,
) -> fem.Function:
    """Creates a full-circle FEM func from a quarter-circle func with BCS."""
    circle_mode = fem.Function(V_circle)
    quadrant_signs = quadrant_signs_for_bcs[mode_on_quarter_circle.bcs_name]

    def interpolate(points: np.ndarray) -> np.ndarray:
        # `points` is a 3 x N matrix
        eval_points = np.abs(points)

        # eval quarter_mode at the `eval_points` and return the values
        quarter_evaluator = algorithms.Evaluator(
            V_quarter_circle, V_quarter_circle.mesh, eval_points
        )
        evaluated_mode = quarter_evaluator(mode_on_quarter_circle)  # has shape (N,)

        if quadrant_signs[0, 0] == -1:  # upper left quadrant
            mask = np.logical_and(points[0, :] < 0, points[1, :] >= 0)
            evaluated_mode[mask] *= -1
        if quadrant_signs[0, 1] == -1:  # upper right quadrant
            mask = np.logical_and(points[0, :] >= 0, points[1, :] >= 0)
            evaluated_mode[mask] *= -1
        if quadrant_signs[1, 0] == -1:  # lower left quadrant
            mask = np.logical_and(points[0, :] < 0, points[1, :] < 0)
            evaluated_mode[mask] *= -1
        if quadrant_signs[1, 1] == -1:  # lower right quadrant
            mask = np.logical_and(points[0, :] >= 0, points[1, :] < 0)
            evaluated_mode[mask] *= -1

        return evaluated_mode

    circle_mode.interpolate(interpolate)
    return circle_mode


@pytest.fixture()
def system():
    ka = 4.83
    gt = 1.00
    epsc = (2 + 0.01j) ** 2

    mshxdmf = "circle_with_pml0.xdmf"

    radius = 1.5 * gt
    vscale = 0.9 * gt / radius
    rg_params = (ka, radius, vscale)
    print(f"RG params: {rg_params}")
    del radius
    del vscale

    pxdmf = (repo_dir / "data" / "meshes" / mshxdmf).resolve()

    with XDMFFile(MPI.COMM_WORLD, pxdmf, "r") as fh:
        msh = fh.read_mesh(name="mcav")
    del fh

    # TODO try to extract the relevant information (Physical Surfice "CAV") from the
    # gmsh file(s)

    # Copied from (def test_manual_integration_domains())
    tdim = msh.topology.dim
    cell_map = msh.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    cell_indices = np.arange(0, num_cells)
    cell_values = np.zeros_like(cell_indices, dtype=np.intc)
    marked_cells = mesh.locate_entities(
        msh, tdim, lambda x: abs(x[0] + 1j * x[1]) <= 1.0 + sys.float_info.epsilon
    )

    circle_meshtag = 7  # this is an arbitrary number
    cell_values[marked_cells] = circle_meshtag
    mt_cells = mesh.meshtags(msh, tdim, cell_indices, cell_values)

    dx_circle = ufl.Measure("dx", subdomain_data=mt_cells, domain=msh)(circle_meshtag)

    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    pml_start = 1.2
    pml_end = 1.8

    rectpml = RectPML(pml_start=pml_start, pml_end=pml_end)

    invperm = fem.Function(fem.VectorFunctionSpace(msh, ("DG", 0)))
    invperm.interpolate(rectpml.invperm_eval)
    invperm = ufl.as_vector((invperm[0], invperm[1]))

    Qfs = fem.FunctionSpace(msh, ("DG", 0))
    cav_dofs = fem.locate_dofs_geometrical(Qfs, lambda x: abs(x[0] + 1j * x[1]) <= 1.0)

    pump_profile = fem.Function(Qfs)
    pump_profile.x.array[:] = 0j
    pump_profile.x.array[cav_dofs] = np.full_like(
        cav_dofs,
        1.0,
        dtype=PETSc.ScalarType,
    )

    dielec = fem.Function(Qfs)
    dielec.interpolate(rectpml.dielec_eval)
    dielec.x.array[cav_dofs] = np.full_like(
        cav_dofs,
        epsc,
        dtype=PETSc.ScalarType,
    )

    if False:
        plot_meshfunctions(msh, pump_profile, dielec, invperm)

    X, Y = np.meshgrid(
        np.linspace(-pml_end, pml_end, 8 * 32),
        np.linspace(-pml_end, pml_end, 8 * 32),
    )
    points = np.vstack([X.flatten(), Y.flatten()])
    evaluator = algorithms.Evaluator(V, msh, points)
    del points

    phi = np.linspace(0, 2 * np.pi, 256)
    points = np.vstack([np.cos(phi), np.sin(phi)])
    evaluator_circle = algorithms.Evaluator(V, msh, points)

    n = V.dofmap.index_map.size_global

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)


@pytest.fixture()
def system_quarter():
    ka = 4.83
    gt = 1.00
    epsc = (2 + 0.01j) ** 2

    mshxdmf = "quarter_circle_with_pml0.xdmf"

    radius = 1.5 * gt
    vscale = 0.9 * gt / radius
    rg_params = (ka, radius, vscale)
    print(f"RG params: {rg_params}")
    del radius
    del vscale

    pxdmf = (repo_dir / "data" / "meshes" / mshxdmf).resolve()

    with XDMFFile(MPI.COMM_WORLD, pxdmf, "r") as fh:
        msh = fh.read_mesh(name="mcav")
    del fh

    V = fem.FunctionSpace(msh, ("Lagrange", 4))

    pml_start = 1.2
    pml_end = 1.8

    rectpml = RectPML(pml_start=pml_start, pml_end=pml_end)

    invperm = fem.Function(fem.VectorFunctionSpace(msh, ("DG", 0)))
    invperm.interpolate(rectpml.invperm_eval)
    invperm = ufl.as_vector((invperm[0], invperm[1]))

    Qfs = fem.FunctionSpace(msh, ("DG", 0))
    cav_dofs = fem.locate_dofs_geometrical(Qfs, lambda x: abs(x[0] + 1j * x[1]) <= 1.0)

    pump_profile = fem.Function(Qfs)
    pump_profile.x.array[:] = 0j
    pump_profile.x.array[cav_dofs] = np.full_like(
        cav_dofs,
        1.0,
        dtype=PETSc.ScalarType,
    )

    dielec = fem.Function(Qfs)
    dielec.interpolate(rectpml.dielec_eval)
    dielec.x.array[cav_dofs] = np.full_like(
        cav_dofs,
        epsc,
        dtype=PETSc.ScalarType,
    )

    if False:
        plot_meshfunctions(msh, pump_profile, dielec, invperm)

    X, Y = np.meshgrid(
        np.linspace(0, pml_end, 8 * 32),
        np.linspace(0, pml_end, 8 * 32),
    )
    points = np.vstack([X.flatten(), Y.flatten()])
    evaluator = algorithms.Evaluator(V, msh, points)
    del points

    n = V.dofmap.index_map.size_global

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)


def solve_nevp_wrapper(
    ka,
    gt,
    rg_params,
    V,
    invperm,
    dielec,
    pump,
    bcs: dict[str, list],
):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    all_modes = []
    for bcs_name, local_bcs in bcs.items():
        assert isinstance(local_bcs, list)
        Print(f"------------> Now solving modes with bcs={bcs_name}")

        L = assemble_form(-inner(elem_mult(invperm, curl(u)), curl(v)) * dx, local_bcs)
        M = assemble_form(dielec * inner(u, v) * dx, local_bcs, diag=0.0)
        Q = assemble_form(pump * inner(u, v) * dx, local_bcs, diag=0.0)

        Print(
            f"{L.getSize()=},  DOF: {L.getInfo()['nz_used']}, MEM:"
            f" {L.getInfo()['memory']}"
        )

        nevp_inputs = algorithms.NEVPInputs(
            ka=ka,
            gt=gt,
            rg_params=rg_params,
            L=L,
            M=M,
            N=None,
            Q=Q,
            R=None,
            bcs=local_bcs,
        )
        all_modes.extend(
            algorithms.get_nevp_modes(
                nevp_inputs,
                bcs_name=bcs_name,
            )
        )
    evals = np.asarray([mode.k for mode in all_modes])

    sorted_evals = np.sort_complex(evals)
    Print("lam (including all bcs)")
    Print("        lam     ")
    Print("-----------------")
    for _lam in sorted_evals:
        Print(f" {_lam.real:9f}{_lam.imag:+9f} j")

    Print("lam (including all bcs) above real axis")
    Print("        lam     ")
    Print("-----------------")
    for _lam in sorted_evals:
        if _lam.imag < 0:
            continue
        Print(f" {_lam.real:9f}{_lam.imag:+9f} j")

    return all_modes, evals


@dataclasses.dataclass()
class ModePair:
    k: complex
    k_rel_err: float
    mode1: algorithms.NEVPNonLasingMode
    mode2: algorithms.NEVPNonLasingMode


def find_pairs(
    modes_above_threshold,
    rel_k_distance: float = 1e-7,
    check_bcs_name: bool = True,
):
    # group the ones with close k together
    for mode1, mode2 in itertools.combinations(modes_above_threshold, 2):
        k_rel_err = abs(mode1.k - mode2.k) / abs(mode1.k)
        if k_rel_err < rel_k_distance:
            if check_bcs_name:
                assert mode1.bcs_name != mode2.bcs_name
            if mode1.bcs_name.startswith("mixed"):
                if check_bcs_name:
                    assert mode2.bcs_name.startswith("mixed"), (
                        mode1.bcs_name,
                        mode2.bcs_name,
                    )
                    # switch order if necessary
                    if mode1.bcs_name == "mixed2":
                        mode1, mode2 = mode2, mode1
            else:
                if check_bcs_name:
                    assert not mode2.bcs_name.startswith("mixed"), (
                        mode1.bcs_name,
                        mode2.bcs_name,
                    )
                    # switch order if necessary
                    if mode1.bcs_name == "full_dbc":
                        mode1, mode2 = mode2, mode1

            yield ModePair(mode1.k, k_rel_err, mode1, mode2)


def create_bcs_on_quarter_mesh(system_quarter) -> dict[str, list]:
    """Creates the BCs for the different symmetries of a quarter mesh."""

    def on_outer_boundary(x):
        return np.isclose(x[0], system_quarter.pml_end) | np.isclose(
            x[1], system_quarter.pml_end
        )

    bcs_dofs_dbc = fem.locate_dofs_geometrical(
        system_quarter.V,
        lambda x: np.isclose(x[0], 0) | np.isclose(x[1], 0) | on_outer_boundary(x),
    )
    bcs_dofs_nbc = fem.locate_dofs_geometrical(
        system_quarter.V,
        # at the outer pml we impose DBC but at the symmetry axes we impose NBC.
        on_outer_boundary,
    )
    bcs_dofs_mixed1 = fem.locate_dofs_geometrical(
        system_quarter.V,
        # DBC at x-axis, NBC at y-axis
        lambda x: np.isclose(x[1], 0) | on_outer_boundary(x),
    )
    bcs_dofs_mixed2 = fem.locate_dofs_geometrical(
        system_quarter.V,
        # DBC at y-axis, NBC at x-axis
        lambda x: np.isclose(x[0], 0) | on_outer_boundary(x),
    )

    zero = PETSc.ScalarType(0)
    return {
        "full_dbc": [fem.dirichletbc(zero, bcs_dofs_dbc, system_quarter.V)],
        "full_nbc": [fem.dirichletbc(zero, bcs_dofs_nbc, system_quarter.V)],
        "mixed1": [fem.dirichletbc(zero, bcs_dofs_mixed1, system_quarter.V)],
        "mixed2": [fem.dirichletbc(zero, bcs_dofs_mixed2, system_quarter.V)],
    }


def norm_mode(mode: np.ndarray) -> int:
    dof_at_maximum = np.abs(mode).argmax()
    val_maximum = mode[dof_at_maximum]
    # fix norm and the phase
    mode /= val_maximum
    return dof_at_maximum


def build_circulating_mode(
    k: complex, mode1: np.ndarray, mode2: np.ndarray
) -> algorithms.NEVPNonLasingMode:
    _mode = mode1 - 1j * mode2
    _dof_at_maximum = norm_mode(_mode)
    return algorithms.NEVPNonLasingMode(
        array=_mode,
        k=k,
        error=0.0,
        bcs_name="default",
        bcs=[],  # on the circle grid we don't need bcs
        dof_at_maximum=_dof_at_maximum,
    )


def find_circulating_mode_above_real_axis(ctrl_modes) -> algorithms.NEVPNonLasingMode:
    pairs = list(
        find_pairs(
            [m for m in ctrl_modes if m.k.imag > eval_threshold],
            rel_k_distance=1e-5,  # don't know why I need a higher tolerance
            check_bcs_name=False,
        )
    )
    Print(f"k of pairs: {[p.k for p in pairs]}")
    Print([p.k_rel_err for p in pairs])
    Print([(p.mode1.bcs_name, p.mode2.bcs_name) for p in pairs])

    maxkimag_pairs = list(sorted(pairs, key=lambda x: -x.k.imag))
    assert maxkimag_pairs

    circulating_modes = [
        build_circulating_mode(
            maxkimag_pair.k,
            maxkimag_pair.mode1.array,
            maxkimag_pair.mode2.array,
        )
        for maxkimag_pair in maxkimag_pairs
    ]
    assert len(circulating_modes) == 1

    # TODO not sure if mode is really a circulating mode
    return circulating_modes[0]


def refine_two_circulating_modes(rmode, mode2, newton_operators, nlp, bcs):
    """Refine a mode (that was a single lasing mode at the previous pump step)
    and a 2nd mode that is above the threshold."""
    minfos = [
        newtils.NewtonModeInfo(
            k=rmode.k.real,
            s=1.0,
            re_array=rmode.array.real / rmode.s,
            im_array=rmode.array.imag / rmode.s,
            dof_at_maximum=rmode.dof_at_maximum,
        ),
        newtils.NewtonModeInfo(
            k=mode2.k.real,
            s=1.0,
            re_array=mode2.array.real,
            im_array=mode2.array.imag,
            dof_at_maximum=mode2.dof_at_maximum,
        ),
    ]

    active_modes = 2
    refined_modes = algorithms.refine_modes(
        minfos,
        bcs,
        newton_operators[active_modes].solver,
        nlp,
        newton_operators[active_modes].A,
        newton_operators[active_modes].L,
        newton_operators[active_modes].delta_x,
        newton_operators[active_modes].initial_x,
        fail_early=True,
    )
    assert all(rm.converged for rm in refined_modes)
    assert len(refined_modes) == active_modes
    return refined_modes


def determine_circulating_mode_at_D0(
    system, system_quarter, D0: float
) -> tuple[complex, algorithms.NEVPNonLasingMode]:
    bcs = create_bcs_on_quarter_mesh(system_quarter)

    modes, _ = solve_nevp_wrapper(
        system_quarter.ka,
        system_quarter.gt,
        system_quarter.rg_params,
        system_quarter.V,
        system_quarter.invperm,
        system_quarter.dielec,
        real_const(system_quarter.V, D0) * system_quarter.pump_profile,
        bcs,
    )

    defaultD0 = D0 == 0.1
    if defaultD0:
        assert len(modes) == 30
        assert modes[0].bcs_name == "full_dbc"

    def scatter_plot(vals, title):
        fig, ax = plt.subplots()
        fig.suptitle(title)

        ax.scatter(vals.real, vals.imag)
        ax.set_xlabel("k.real")
        ax.set_ylabel("k.imag")

        plot_ellipse(ax, system.rg_params)
        list(map(lambda x: ax.axvline(x=x), allK))

        ax.grid(True)

    # scatter_plot(
    #     np.asarray([mode.k for mode in modes]), f"Non-Interacting thresholds at {D0=}"
    # )
    # plt.show()

    pairs = list(
        find_pairs([m for m in modes if m.k.imag > eval_threshold], rel_k_distance=1e-6)
    )
    Print(f"k of pairs: {[p.k for p in pairs]}")
    Print([p.k_rel_err for p in pairs])
    Print([(p.mode1.bcs_name, p.mode2.bcs_name) for p in pairs])

    maxkimag_pairs = list(sorted(pairs, key=lambda x: -x.k.imag))
    assert maxkimag_pairs

    # we loop over the maxkimag pairs because it can happen that the mode with the
    # highest k-imag doesn't lase (in this case an exception is raised when this mode
    # and the other mode that is above threshold is refined)
    circulating_modes = []
    for maxkimag_pair in maxkimag_pairs:
        Print(f"{maxkimag_pair.k=}".center(80, "-"))

        mode_1a = map_function_to_circle_mesh(
            maxkimag_pair.mode1, system.V, system_quarter.V
        )
        mode_1b = map_function_to_circle_mesh(
            maxkimag_pair.mode2, system.V, system_quarter.V
        )

        circulating_mode1 = build_circulating_mode(
            maxkimag_pair.k, mode_1a.vector.getArray(), mode_1b.vector.getArray()
        )
        circulating_modes.append(circulating_mode1)

    return circulating_modes


# D0=0.076 is close to first threshold (see the text in the paper)
@pytest.mark.parametrize(
    "D0_range",
    [
        # [0.1]  # single laser mode k=5.378
        # [0.14]  # single laser mode k=5.378
        # [0.170]  # NEW: only one mode lases
        # [0.170, 0.1725, 0.175, 0.178],  # NEW: works, only one mode lases at k=4.81
        # np.linspace(0.170, 0.24, 14),  # NEW: works, only one mode lases at k=4.81
        # np.linspace(0.08, 0.14, 6),  # NEW: works, only one mode lases at k=5.378
        # [0.145, 0.147, 0.148, 0.149, 0.15, 0.151, 0.152, 0.153],
        #                        # NEW: works, shows crossing of two modes
        # [0.13],  # NEW: works, Not the mode with the highest kimag lases!!!!!!
        np.linspace(0.13, 0.15, 8),  # NEW: works, transition from 1 mode to two modes
        # np.linspace(0.13, 0.162, 12),  # NEW: works, transition from single- to
        #                              # multi-modes to single-mode
        # np.linspace(0.135, 0.162, 15),  # NEW: works, transition from single- to
        #                                 # multi-modes to single-mode
        # [0.153, 0.158], # NEW: works, the mode shut-off can be clearly seen
        # [0.162],  # NEW works, single mode lases at k=4.81
        # [0.2],  # NEW works, single mode lases at k=4.81
        # [0.3], # NEW works, single mode lases at k=4.81
        # [0.5], # NEW works, single mode lases at k=4.81
        # [0.1538], # NEW works, single mode lases at k=4.81
        # [0.150],  # NEW works, two modes laser mode k=5.378, k=4.81
        # np.linspace(0.13, 0.076, 4),  # single laser mode only
        # np.linspace(0.13, 0.15, 8),  # single laser mode only
        # [0.145],  # two lasing modes
        # [0.142, 0.1435, 0.145, 0.147, 0.15],  # from one to two lasing modes
        # [0.145, 0.147, 0.15, 0.152, 0.153, 0.1532],  # only two lasing modes
    ],
)
def test_solve_multimode_D0range(system, system_quarter, D0_range):
    """Determine the lasing modes starting at `D0=D0_range[0]`."""
    # For solving the NEVP we use the quarter circle mesh with different boundary
    # conditions. We then refine a circulating mode with Im(k) > 0 s.t. it reaches the
    # real axis. The mode-refining is done using the full-circle mesh, because a
    # circulating (sum of a mode with DBC and a mode with NBC) mode doesn't have well
    # defined BCs for the quarter mesh.

    D0_start = D0_range[0]

    # the circulating mode is a FEM function on the full (circle) system
    circulating_modes = determine_circulating_mode_at_D0(
        system, system_quarter, D0=D0_start
    )

    arbitrary_default_value = 100.200300
    D0_constant_circle = real_const(system.V, arbitrary_default_value)

    fem_mode = fem.Function(system.V)
    internal_intensity_form = fem.form(abs(fem_mode) ** 2 * system.dx_circle)

    def calculate_intensity_in_circle(mode_on_circle):
        fem_mode.x.array[:] = mode_on_circle.array
        internal_intensity = fem.assemble_scalar(internal_intensity_form)
        assert internal_intensity.imag < 1e-15
        return internal_intensity.real

    nlp = NonLinearProblem(
        system.V,
        system.ka,
        system.gt,
        dielec=system.dielec,
        invperm=system.invperm,
        n=system.n,
        pump=D0_constant_circle * system.pump_profile,
        ds_obc=None,
    )
    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=2)

    circulating_mode_results = {}
    for circulating_mode in circulating_modes:
        Print(f"{circulating_mode.k.real=}".center(80, "-"))
        # TODO break out of the loop as soon as we have found a converging mode-set
        minfos = [
            newtils.NewtonModeInfo(
                k=circulating_mode.k.real,
                s=1.0,
                re_array=circulating_mode.array.real,
                im_array=circulating_mode.array.imag,
                dof_at_maximum=circulating_mode.dof_at_maximum,
            )
        ]
        active_modes = 1

        bcs = circulating_mode.bcs
        D0_constant_circle.value = D0_range[0]
        refined_modes = algorithms.refine_modes(
            minfos,
            bcs,
            newton_operators[active_modes].solver,
            nlp,
            newton_operators[active_modes].A,
            newton_operators[active_modes].L,
            newton_operators[active_modes].delta_x,
            newton_operators[active_modes].initial_x,
        )
        assert all(rm.converged for rm in refined_modes)
        assert len(refined_modes) == active_modes
        assert all(rmode.k.imag < 1e-9 for rmode in refined_modes)
        minfos = [
            newtils.NewtonModeInfo(
                k=rmode.k.real,
                s=rmode.s,
                re_array=rmode.array.real / rmode.s,
                im_array=rmode.array.imag / rmode.s,
                dof_at_maximum=rmode.dof_at_maximum,
            )
            for rmode in refined_modes
        ]
        # check if the eigenvalue of a new mode is above the real axis
        nlp.update_b_and_k_for_forms(refined_modes)

        # TODO don't call the NEVP solver for every D0, because the NEVP solver for
        # the full circle system is computationally intensive.

        Q_form = nlp.get_Q_hbt_form(nmodes=len(refined_modes))
        _u = ufl.TrialFunction(system.V)
        _v = ufl.TestFunction(system.V)

        ctrl_modes = algorithms.get_nevp_modes(
            algorithms.NEVPInputs(
                ka=system.ka,
                gt=system.gt,
                rg_params=system.rg_params,
                L=assemble_form(
                    -inner(elem_mult(system.invperm, curl(_u)), curl(_v)) * dx, bcs
                ),
                M=assemble_form(system.dielec * inner(_u, _v) * dx, bcs, diag=0.0),
                N=None,
                Q=assemble_form(Q_form, bcs, diag=0.0),
                R=None,
                bcs=bcs,
            )
        )

        ctrl_evals = np.asarray([cm.k for cm in ctrl_modes])

        number_of_modes_close_to_real_axis = np.sum(
            np.abs(ctrl_evals.imag) < real_axis_threshold
        )
        Print(
            "Number of modes close to real axis: "
            f"{number_of_modes_close_to_real_axis}"
        )
        # 2 degenerate modes (should be CW and CCW modes) are close to the real axis
        assert number_of_modes_close_to_real_axis == 2 * active_modes

        number_of_modes_above_real_axis = np.sum(ctrl_evals.imag > real_axis_threshold)
        Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")
        circulating_mode_results[circulating_mode.k.real] = (
            number_of_modes_close_to_real_axis,
            number_of_modes_above_real_axis,
        )

        if not number_of_modes_above_real_axis:
            break

        if number_of_modes_above_real_axis == 4:
            Print(
                "We don't support more than 2 lasing modes yet -> continue with next "
                "pair"
            )
            continue

        Print(
            "Check if it is possible to bring a 2nd mode to the real axis and break "
            "out of the for loop"
        )
        assert active_modes == 1
        assert len(refined_modes) == 1

        # find the 2 ctrl_modes above threshold, and create a circulating mode
        mode2 = find_circulating_mode_above_real_axis(ctrl_modes)
        Print("Refine two circulating modes")
        try:
            refined_modes = refine_two_circulating_modes(
                refined_modes[0], mode2, newton_operators, nlp, bcs
            )
        except algorithms.RefinementError:
            continue
        minfos = [
            newtils.NewtonModeInfo(
                k=rmode.k.real,
                s=rmode.s,
                re_array=rmode.array.real / rmode.s,
                im_array=rmode.array.imag / rmode.s,
                dof_at_maximum=rmode.dof_at_maximum,
            )
            for rmode in refined_modes
        ]
        active_modes = 2
        break
    else:
        raise RuntimeError(
            f"Not possible to find set of max. two lasing modes at D0={D0_range[0]}"
        )

    Print(f"{circulating_mode_results=}")

    sweep_results = PumpSweepResults([])

    for D0 in D0_range:
        t0_current_pump_step = time.monotonic()
        current_pump_step_results = PumpStepResults(
            D0=D0, modes=[], nevp_eigenvalues=[], duration=-1
        )
        sweep_results.pump_step_results.append(current_pump_step_results)

        log.info(f"--------> D0={D0}")
        D0_constant_circle.value = D0
        while True:
            log.info(f"In while True body D0={D0}")
            try:
                refined_modes = algorithms.refine_modes(
                    minfos,
                    bcs,
                    newton_operators[active_modes].solver,
                    nlp,
                    newton_operators[active_modes].A,
                    newton_operators[active_modes].L,
                    newton_operators[active_modes].delta_x,
                    newton_operators[active_modes].initial_x,
                    fail_early=True,
                )
                break
            except algorithms.RefinementError:
                if len(minfos) > 1:
                    log.warning("Check for shut-down of mode")
                    # try to decrease minfos and check if refine_mode converges

                    # TODO use a better metric to determine the mode that shut down.
                    # minfos = minfos[:1]
                    minfos = minfos[1:]
                    log.warning(f"Check if mode at k={minfos[0].k} is a lasing mode")
                    active_modes = 1
                else:
                    raise

        log.info(f"--------> after while True loop at D0={D0}")
        assert all(rm.converged for rm in refined_modes)
        assert len(refined_modes) == active_modes

        # refined_mode[0] is the solution of a nonlinear EVP
        # it should still be a circulating mode

        current_mode_results = [
            ModeResults(
                k=rmode.k.real,
                intensity=calculate_intensity_in_circle(rmode),
            )
            for rmode in refined_modes
        ]

        assert all(rmode.k.imag < 1e-9 for rmode in refined_modes)
        minfos = [
            newtils.NewtonModeInfo(
                k=rmode.k.real,
                s=rmode.s,
                re_array=rmode.array.real / rmode.s,
                im_array=rmode.array.imag / rmode.s,
                dof_at_maximum=rmode.dof_at_maximum,
            )
            for rmode in refined_modes
        ]
        log.info(f"--------> len refined modes: {len(minfos)} D0={D0}")

        # check if the eigenvalue of a new mode is above the real axis
        nlp.update_b_and_k_for_forms(refined_modes)

        # TODO don't call the NEVP solver for every D0, because the NEVP solver for
        # the full circle system is computationally intensive.

        Q_form = nlp.get_Q_hbt_form(nmodes=len(refined_modes))
        _u = ufl.TrialFunction(system.V)
        _v = ufl.TestFunction(system.V)

        ctrl_modes = algorithms.get_nevp_modes(
            algorithms.NEVPInputs(
                ka=system.ka,
                gt=system.gt,
                rg_params=system.rg_params,
                L=assemble_form(
                    -inner(elem_mult(system.invperm, curl(_u)), curl(_v)) * dx, bcs
                ),
                M=assemble_form(system.dielec * inner(_u, _v) * dx, bcs, diag=0.0),
                N=None,
                Q=assemble_form(Q_form, bcs, diag=0.0),
                R=None,
                bcs=bcs,
            )
        )
        ctrl_evals = np.asarray([cm.k for cm in ctrl_modes])
        current_pump_step_results.nevp_eigenvalues = ctrl_evals.tolist()

        number_of_modes_close_to_real_axis = np.sum(
            np.abs(ctrl_evals.imag) < real_axis_threshold
        )
        Print(
            "Number of modes close to real axis: "
            f"{number_of_modes_close_to_real_axis}"
        )
        # 2 degenerate modes (should be CW and CCW modes) are close to the real axis
        assert number_of_modes_close_to_real_axis == 2 * active_modes

        number_of_modes_above_real_axis = np.sum(ctrl_evals.imag > real_axis_threshold)
        Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")

        # FIXME this raises when D0 is close to the 2nd threshold (around 0.165
        # according to fig 7), but why does it raise before this threshold is
        # reached?
        # assert number_of_modes_above_real_axis == 2

        if number_of_modes_above_real_axis:
            assert active_modes == 1
            assert len(refined_modes) == 1

            # find the 2 ctrl_modes above threshold, and create a circulating mode
            mode2 = find_circulating_mode_above_real_axis(ctrl_modes)

            refined_modes = refine_two_circulating_modes(
                refined_modes[0], mode2, newton_operators, nlp, bcs
            )
            minfos = [
                newtils.NewtonModeInfo(
                    k=rmode.k.real,
                    s=rmode.s,
                    re_array=rmode.array.real / rmode.s,
                    im_array=rmode.array.imag / rmode.s,
                    dof_at_maximum=rmode.dof_at_maximum,
                )
                for rmode in refined_modes
            ]
            active_modes = 2

            current_mode_results[:] = [
                ModeResults(
                    k=rmode.k.real,
                    intensity=calculate_intensity_in_circle(rmode),
                )
                for rmode in refined_modes
            ]

            # TODO run nevp-solver again, even though it is not strictly needed here
            # and use return value for current_pump_step_results.nevp_eigenvalues

        current_pump_step_results.modes.extend(current_mode_results)
        current_pump_step_results.duration = time.monotonic() - t0_current_pump_step

    if isinstance(D0_range, np.ndarray):
        pass
    elif D0_range == [0.1]:
        pass
        # TODO
        # assert intensity_map[0.1] == pytest.approx(0.742, rel=1e-3)
    elif D0_range == [0.145]:
        pass
        # TODO
        # assert intensity_map[0.145] == pytest.approx(1.8479903859036326, rel=1e-3)
        # assert intensity_map_mode2[0.145] == pytest.approx(
        # 0.239567162395648, rel=1e-3)

    with open("fig7_results.pickle", "wb") as fh:
        pickle.dump(sweep_results, fh, protocol=pickle.HIGHEST_PROTOCOL)

    fig, ax = plt.subplots()
    mode1_refdata = np.loadtxt(
        "./data/references/esterhazy_fig7/mode1.csv", delimiter=","
    )
    mode2_refdata = np.loadtxt(
        "./data/references/esterhazy_fig7/mode2.csv", delimiter=","
    )
    ax.plot(mode1_refdata[:, 0], mode1_refdata[:, 1], "x")
    ax.plot(mode2_refdata[:, 0], mode2_refdata[:, 1], "x")

    moderes = np.asarray(
        [
            (pump_step_result.D0, mode.intensity, mode.k)
            for pump_step_result in sweep_results.pump_step_results
            for mode in pump_step_result.modes
        ]
    )
    # TODO write the results to a file

    # partition the results into two buckets (mode1 and mode2)
    k_modethreshold = 5.0
    mode1 = []
    mode2 = []

    for pump_step_result in sweep_results.pump_step_results:
        for mode in pump_step_result.modes:
            cur_mode = mode1 if mode.k > k_modethreshold else mode2
            cur_mode.append((pump_step_result.D0, mode.intensity))

    if mode1:
        mode1 = np.asarray(mode1)
        ax.plot(mode1[:, 0], mode1[:, 1], "-o")

    if mode2:
        mode2 = np.asarray(mode2)
        ax.plot(mode2[:, 0], mode2[:, 1], "-o")

    ax.grid(True)

    fig, ax = plt.subplots()
    ax.plot(moderes[:, 0], moderes[:, 2], "o")
    ax.grid(True)

    plt.show()


def test_plot_modeintensities():
    # `mode1` and `mode2` are determined by saltx
    mode1 = np.array(
        [
            [0.13, 1.67668968],
            [0.13082051, 1.70274043],
            [0.13164103, 1.72881426],
            [0.13246154, 1.754911],
            [0.13328205, 1.78103047],
            [0.13410256, 1.8071725],
            [0.13492308, 1.83333693],
            [0.13574359, 1.85952359],
            [0.1365641, 1.88573231],
            [0.13738462, 1.91196293],
            [0.13820513, 1.93821529],
            [0.13902564, 1.96448924],
            [0.13984615, 1.99078461],
            [0.14066667, 2.01710126],
            [0.14148718, 2.04343904],
            [0.14230769, 2.06979779],
            [0.14312821, 2.09617736],
            [0.14394872, 2.07208521],
            [0.14476923, 1.89723158],
            [0.14558974, 1.72202368],
            [0.14641026, 1.54646082],
            [0.14723077, 1.37054537],
            [0.14805128, 1.1942798],
            [0.14887179, 1.01766655],
            [0.14969231, 0.84070799],
            [0.15051282, 0.66340636],
            [0.15133333, 0.48576359],
            [0.15215385, 0.3077804],
            [0.15297436, 0.12944835],
        ]
    )
    mode2 = np.array(
        [
            [0.14394872, 0.03921891],
            [0.14476923, 0.19554877],
            [0.14558974, 0.35216049],
            [0.14641026, 0.50905454],
            [0.14723077, 0.66622904],
            [0.14805128, 0.82368202],
            [0.14887179, 0.98141154],
            [0.14969231, 1.13941572],
            [0.15051282, 1.29769276],
            [0.15133333, 1.45624112],
            [0.15215385, 1.6150602],
            [0.15297436, 1.77415649],
            [0.15379487, 1.89537901],
            [0.15461538, 1.91630179],
            [0.1554359, 1.93723855],
            [0.15625641, 1.95818919],
            [0.15707692, 1.97915363],
            [0.15789744, 2.00013177],
            [0.15871795, 2.02112354],
            [0.15953846, 2.04212883],
            [0.16035897, 2.06314758],
            [0.16117949, 2.08417968],
            [0.162, 2.10522507],
        ]
    )

    fig, ax = plt.subplots()
    # parse the reference results from the paper
    mode1_refdata = np.loadtxt(
        "./data/references/esterhazy_fig7/mode1.csv", delimiter=","
    )
    mode2_refdata = np.loadtxt(
        "./data/references/esterhazy_fig7/mode2.csv", delimiter=","
    )
    ax.plot(mode1_refdata[:, 0], mode1_refdata[:, 1], "x")
    ax.plot(mode1[:, 0], mode1[:, 1], "-o", label="Mode1")

    ax.plot(mode2_refdata[:, 0], mode2_refdata[:, 1], "x")
    ax.plot(mode2[:, 0], mode2[:, 1], "-o", label="Mode2")

    ax.legend()
    ax.set_xlabel("D0")
    ax.set_xlabel("intensity")
    ax.grid(True)
    plt.show()


def test_evaltraj(system, system_quarter):
    """Determine the complex eigenvalues of the modes without a hole burning
    term."""

    bcs = create_bcs_on_quarter_mesh(system_quarter)
    D0_constant = real_const(system_quarter.V, 1.0)
    D0_range = np.linspace(0.05, 0.2, 4)

    vals = []
    for D0 in D0_range:
        Print(f"--> {D0=}")
        D0_constant.value = D0
        modes, _ = solve_nevp_wrapper(
            system_quarter.ka,
            system_quarter.gt,
            system_quarter.rg_params,
            system_quarter.V,
            system_quarter.invperm,
            system_quarter.dielec,
            D0_constant * system_quarter.pump_profile,
            bcs,
        )
        evals = np.asarray([m.k for m in modes])
        vals.append(np.vstack([D0 * np.ones(evals.shape), evals]).T)

    def scatter_plot(vals, title):
        fig, ax = plt.subplots()
        fig.suptitle(title)

        merged = np.vstack(vals)
        X, Y, C = (
            merged[:, 1].real,
            merged[:, 1].imag,
            merged[:, 0].real,
        )
        norm = plt.Normalize(C.min(), C.max())

        sc = ax.scatter(X, Y, c=C, norm=norm)
        ax.set_xlabel("k.real")
        ax.set_ylabel("k.imag")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("D0", loc="top")

        plot_ellipse(ax, system_quarter.rg_params)
        list(map(lambda x: ax.axvline(x=x), allK))

        ax.grid(True)

    scatter_plot(
        vals, f"Non-Interacting thresholds (Range D0={D0_range[0]} .. {D0_range[-1]})"
    )

    plt.show()
