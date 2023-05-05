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


def test_solve_fixed_pump(system, system_quarter):
    """Determine the lasing mode at D0=0.1."""
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
    bcs_dofs_mixed = fem.locate_dofs_geometrical(
        system_quarter.V,
        # DBC at x-axis, NBC at y-axis
        lambda x: np.isclose(x[1], 0) | on_outer_boundary(x),
    )

    bcs = {
        "full_dbc": [
            fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_dbc, system_quarter.V),
        ],
        "full_nbc": [
            fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_nbc, system_quarter.V),
        ],
        "mixed": [
            fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_mixed, system_quarter.V),
        ],
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

    assert len(modes) == 22
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

    dbc_mode = fem.Function(system.V)

    def dbc_interpolate(points: np.ndarray) -> np.ndarray:
        # points is a 3 x N matrix
        quarter_mode = dbc_quarter_mode
        eval_points = np.abs(points)
        #  now eval quarter_mode at the eval_points and return the values
        quarter_evaluator = algorithms.Evaluator(
            system_quarter.V, system_quarter.msh, eval_points
        )
        emode = quarter_evaluator(quarter_mode)  # has shape (N,)
        positive_sign_mask = np.logical_or(
            np.logical_and(points[0, :] > 0, points[1, :] > 0),
            np.logical_and(points[0, :] < 0, points[1, :] < 0),
        )
        emode[~positive_sign_mask] *= -1

        return emode

    dbc_mode.interpolate(dbc_interpolate)

    nbc_mode = fem.Function(system.V)

    def nbc_interpolate(points: np.ndarray) -> np.ndarray:
        # points is a 3 x N matrix
        quarter_mode = nbc_quarter_mode

        eval_points = np.abs(points)
        quarter_evaluator = algorithms.Evaluator(
            system_quarter.V, system_quarter.msh, eval_points
        )
        return quarter_evaluator(quarter_mode)  # has shape (N,)

    nbc_mode.interpolate(nbc_interpolate)

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
                abs(vals) ** 2,
                vmin=0.0,
            )
            ax.set_title(title)
        plt.show()

    _lam = dbc_modes_above_threshold[0].k
    assert _lam.imag > 0

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
        ax.plot(system.phi - np.pi, abs(vals) ** 2)

        vals = system.evaluator_circle(mode_dbc)  # .reshape(system.X.shape)
        _, ax = plt.subplots()
        # ax.pcolormesh(
        #     system.X,
        #     system.Y,
        #     # vals.real,
        #     abs(vals) ** 2,
        #     vmin=0.0,
        # )
        ax.plot(system.phi - np.pi, abs(vals) ** 2)

        vals = system.evaluator_circle(mode_nbc)  # .reshape(system.X.shape)
        _, ax = plt.subplots()
        # ax.pcolormesh(
        #     system.X,
        #     system.Y,
        #     # vals.real,
        #     abs(vals) ** 2,
        #     vmin=0.0,
        # )
        ax.plot(system.phi - np.pi, abs(vals) ** 2)
        plt.show()

    dof_at_maximum = np.abs(mode).argmax()
    val_maximum = mode[np.abs(mode).argmax()]
    # fix norm and the phase
    mode /= val_maximum

    def on_outer_boundary(x):
        return np.isclose(abs(x[0]), system.pml_end) | np.isclose(
            abs(x[1]), system.pml_end
        )

    bcs_dofs_circle = fem.locate_dofs_geometrical(
        system.V,
        on_outer_boundary,
    )

    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_circle, system.V),
    ]

    mode = algorithms.NEVPNonLasingMode(
        array=mode,
        k=_lam,
        error=0.0,
        bcs_name="default",
        bcs=bcs,
        dof_at_maximum=dof_at_maximum,
    )

    # TODO do this till 0.15 where still only a single mode is lasing

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
    refined_modes = algorithms.refine_modes(
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
    assert len(refined_modes) == 1

    # determine internal intensity and compare it against the value from figure 7

    fem_mode = fem.Function(system.V)
    fem_mode.x.array[:] = refined_modes[0].array
    internal_intensity = fem.assemble_scalar(
        fem.form(abs(fem_mode) ** 2 * system.dx_circle)
    )
    assert internal_intensity.imag < 1e-15
    assert internal_intensity.real == pytest.approx(0.742, rel=1e-3)


def test_plot():
    fig, ax = plt.subplots()
    refdata = np.loadtxt("./data/figures/esterhazy_fig7/mode1.csv", delimiter=",")
    ax.plot(refdata[:, 0], refdata[:, 1], "x")

    # my results
    ax.plot(
        [0.08, 0.1, 0.13339275245890964, 0.15946653060338278],
        [0.14217403550027424, 0.7422087350381327, 1.785008156714247, 2.626127947527164],
        "-o",
    )

    plt.show()
