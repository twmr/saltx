# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""
import sys
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


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


def map_function_to_circle_mesh(
    mode_on_quarter_circle: algorithms.NEVPNonLasingMode,
    quadrant_signs: np.ndarray,
    V_circle,
    V_quarter_circle,
) -> fem.Function:
    """Creates a full-circle FEM func from a quarter-circle func with BCS."""
    circle_mode = fem.Function(V_circle)

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


@pytest.fixture
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


@pytest.fixture
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
    return all_modes, evals


def test_solve_single_mode_D0range(system, system_quarter):
    """Determine the lasing mode starting at D0=0.1."""
    # For solving the NEVP we use the quarter circle mesh with different boundary
    # conditions. We then refine a circulating mode with Im(k) > 0 s.t. it reaches the
    # real axis. The mode-refining is done using the full-circle mesh, because a
    # circulating (sum of a mode with DBC and a mode with NBC) mode doesn't have well
    # defined BCs for the quarter mesh.

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
    # bcs_dofs_mixed = fem.locate_dofs_geometrical(
    #     system_quarter.V,
    #     # DBC at x-axis, NBC at y-axis
    #     lambda x: np.isclose(x[1], 0) | on_outer_boundary(x),
    # )

    bcs = {
        "full_dbc": [
            fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_dbc, system_quarter.V),
        ],
        "full_nbc": [
            fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_nbc, system_quarter.V),
        ],
        # "mixed": [
        #     fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_mixed, system_quarter.V),
        # ],
    }

    # D0range = [0.076]  # close to first threshold (see the text in the paper)
    D0range = [0.1]  # the convergence of refine_modes is better at D0=1.0
    # TODO figure out why the convergence is not so good at smaller D0
    D0_constant = real_const(system_quarter.V, D0range[0])
    D0_constant_circle = real_const(system.V, D0range[0])

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

    assert len(modes) == 14
    assert modes[0].bcs_name == "full_dbc"

    def scatter_plot(vals, title):
        fig, ax = plt.subplots()
        fig.suptitle(title)

        ax.scatter(vals.real, vals.imag)
        ax.set_xlabel("k.real")
        ax.set_ylabel("k.imag")

        plot_ellipse(ax, system.rg_params)

        ax.grid(True)

    scatter_plot(np.asarray([mode.k for mode in modes]), "Non-Interacting thresholds")
    plt.show()

    dbc_modes_above_threshold = [
        mode for mode in modes if mode.bcs_name == "full_dbc" and mode.k.imag > 0
    ]
    assert len(dbc_modes_above_threshold) > 0

    k_fm = 5.380  # approx k first mode
    dbc_quarter_mode = dbc_modes_above_threshold[
        np.argmin([abs(m.k.real - k_fm) for m in dbc_modes_above_threshold])
    ]

    nbc_modes_above_threshold = [
        mode for mode in modes if mode.bcs_name == "full_nbc" and mode.k.imag > 0
    ]
    assert len(nbc_modes_above_threshold) > 0
    nbc_quarter_mode = nbc_modes_above_threshold[
        np.argmin([abs(m.k.real - k_fm) for m in nbc_modes_above_threshold])
    ]

    # transform the modes on the quarter mesh to modes on the full mesh
    dbc_mode = map_function_to_circle_mesh(
        dbc_quarter_mode, np.array([[-1, 1], [1, -1]]), system.V, system_quarter.V
    )
    nbc_mode = map_function_to_circle_mesh(
        nbc_quarter_mode, np.array([[1, 1], [1, 1]]), system.V, system_quarter.V
    )

    dbc_internal_intensity = fem.assemble_scalar(
        fem.form(abs(dbc_mode) ** 2 * system.dx_circle)
    )
    assert dbc_internal_intensity.imag < 1e-15
    assert dbc_internal_intensity.real == pytest.approx(0.662, rel=0.1)

    nbc_internal_intensity = fem.assemble_scalar(
        fem.form(abs(nbc_mode) ** 2 * system.dx_circle)
    )
    assert nbc_internal_intensity.imag < 1e-15
    assert nbc_internal_intensity.real == pytest.approx(0.662, rel=0.1)

    debug = False
    if debug:
        # plot the modes on the circle mesh
        for title, vals in [
            ("DBC mode", system.evaluator(dbc_mode.x.array).reshape(system.X.shape)),
            ("NBC mode", system.evaluator(nbc_mode.x.array).reshape(system.X.shape)),
        ]:
            _, ax = plt.subplots()
            ax.pcolormesh(
                system.X,
                system.Y,
                # abs(vals) ** 2,
                vals.real,
                # vmin=0.0,
            )
            ax.set_title(title)
        plt.show()

    # make sure that k of the dbc and nbc modes is the same, s.t. we can build a
    # circulating mode.
    _lam_dbc, _lam_nbc = (
        dbc_modes_above_threshold[0].k,
        nbc_modes_above_threshold[0].k,
    )
    assert _lam_dbc.imag > 0
    np.testing.assert_allclose(_lam_nbc, _lam_dbc)
    _lam = _lam_dbc

    mode_dbc = dbc_mode.vector.getArray().copy()
    mode_nbc = nbc_mode.vector.getArray().copy()

    #    = "cos"    - 1j "sin"
    mode = mode_nbc - 1j * mode_dbc

    if debug:
        vals = system.evaluator(mode).reshape(system.X.shape)
        _, ax = plt.subplots()
        # a nice ring pattern should be shown here
        ax.pcolormesh(
            system.X,
            system.Y,
            abs(vals) ** 2,
            vmin=0.0,
        )

        vals = system.evaluator_circle(mode)  # .reshape(system.X.shape)
        _, ax = plt.subplots()
        # ax.pcolormesh(
        #     system.X,
        #     system.Y,
        #     vals.real,
        #     #abs(vals) ** 2,
        #     # vmin=0.0,
        # )

        # The intensity fluctuates a tiny bit
        ax.plot(system.phi - np.pi, abs(vals) ** 2)
        ax.set_ylim(0, 1.0)
        ax.set_title("mode")

        vals_dbc = system.evaluator_circle(mode_dbc)
        vals_nbc = system.evaluator_circle(mode_nbc)

        _, ax = plt.subplots()
        ax.plot(system.phi - np.pi, abs(vals_nbc.real - 1j * vals_dbc.real) ** 2)
        ax.set_ylim(0, 1.0)
        ax.set_title("nbc-1jdbc")

        _, ax = plt.subplots()
        ax.plot(system.phi - np.pi, abs(vals_nbc) ** 2, label="intens nbc")
        ax.plot(system.phi - np.pi, abs(vals_dbc) ** 2, label="intens dbc")
        ax.legend()

        _, ax = plt.subplots()
        ax.plot(system.phi - np.pi, vals_nbc.real, label="real nbc")
        ax.plot(system.phi - np.pi, vals_dbc.real, label="real dbc")
        ax.legend()

        _, ax = plt.subplots()
        ax.plot(system.phi - np.pi, np.angle(vals_nbc), label="phase nbc")
        ax.plot(system.phi - np.pi, np.angle(vals_dbc), label="phase dbc")
        ax.legend()

        plt.show()

    dof_at_maximum = np.abs(mode).argmax()
    val_maximum = mode[dof_at_maximum]
    # fix norm and the phase
    mode /= val_maximum

    def on_outer_boundary(x):
        return np.isclose(abs(x[0]), system.pml_end) | np.isclose(
            abs(x[1]), system.pml_end
        )

    bcs = []

    mode = algorithms.NEVPNonLasingMode(
        array=mode,
        k=_lam,
        error=0.0,
        bcs_name="default",
        bcs=bcs,
        dof_at_maximum=dof_at_maximum,
    )

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
    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=1)

    minfos = [
        newtils.NewtonModeInfo(
            k=mode.k.real,
            s=1.0,
            re_array=mode.array.real,
            im_array=mode.array.imag,
            dof_at_maximum=mode.dof_at_maximum,
        )
    ]
    active_modes = 1

    fem_mode = fem.Function(system.V)
    internal_intensity_form = fem.form(abs(fem_mode) ** 2 * system.dx_circle)

    intensity_map = {}
    # for D0 in np.linspace(0.1, 0.076, 8):
    for D0 in np.linspace(0.1, 0.16, 8):
        log.info(f"--------> D0={D0}")
        D0_constant_circle.value = D0
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
        assert len(refined_modes) == 1

        # determine internal intensity and compare it with the value from figure 7
        fem_mode.x.array[:] = refined_modes[0].array
        internal_intensity = fem.assemble_scalar(internal_intensity_form)
        assert internal_intensity.imag < 1e-15
        intensity_map[D0] = internal_intensity.real

        rmode = refined_modes[0]
        minfos = [
            newtils.NewtonModeInfo(
                k=rmode.k.real,
                s=rmode.s,
                re_array=rmode.array.real / rmode.s,
                im_array=rmode.array.imag / rmode.s,
                dof_at_maximum=rmode.dof_at_maximum,
            )
        ]

        if False:
            # check if the eigenvalue of a new mode is above the real axis
            nlp.update_b_and_k_for_forms(refined_modes)

            # TODO don't call the NEVP solver for every D0, because the NEVP solver for
            # the full circle system is computationally intensive.

            Q_form = nlp.get_Q_hbt_form(nmodes=1)
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

            # TODO Try to decrease this threshold to at least 1e-9
            real_axis_threshold = 2e-8
            number_of_modes_close_to_real_axis = np.sum(
                np.abs(ctrl_evals.imag) < real_axis_threshold
            )
            Print(
                "Number of modes close to real axis: "
                f"{number_of_modes_close_to_real_axis}"
            )
            # 2 degenerate modes are close to the real axis
            assert number_of_modes_close_to_real_axis == 2

            number_of_modes_above_real_axis = np.sum(
                ctrl_evals.imag > real_axis_threshold
            )
            Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")

            # this raises when D0 is close to the 2nd threshold (around 0.165 according
            # to fig 7), but why does it raise before this threshold is reached?
            assert number_of_modes_above_real_axis == 0

    assert intensity_map[0.1].real == pytest.approx(0.742, rel=1e-3)

    fig, ax = plt.subplots()
    refdata = np.loadtxt("./data/references/esterhazy_fig7/mode1.csv", delimiter=",")
    ax.plot(refdata[:, 0], refdata[:, 1], "x")
    ax.plot(list(intensity_map.keys()), list(intensity_map.values()), "-o")

    plt.show()
