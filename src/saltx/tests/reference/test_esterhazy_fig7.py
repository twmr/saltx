# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""

import time
from collections import namedtuple
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from ufl import curl, dx, elem_mult, inner

from saltx import algorithms
from saltx.assemble import assemble_form
from saltx.log import Timer
from saltx.nonlasing import NonLasingLinearProblem
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
    """Determine the lasing mode at D0=0.076."""
    # For solving the NEVP we use the quarter circle mesh with different boundary
    # conditions. We then refine a circulating mode with Im(k) > 0 s.t. it reaches the
    # real axis. The mode-refining is done using the full-circle mesh, because this is
    # needed for the multi mode support anyway.

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

    D0range = [0.076]  # close to first threshold (see the text in the paper)
    D0_constant = real_const(system_quarter.V, D0range[0])

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

    # TODO transform the modes on the quarter mesh to modes on the full mesh
    # TODO then create a circulating mode by summing up (make sure that the
    #      amplitude is correct first) the single full_dbc mode and the single
    #      full_nbc mode.
    # TODO refine this mode
    # TODO do this till 0.15 where still only a single mode is lasing

    return

    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    L = assemble_form(-inner(elem_mult(system.invperm, curl(u)), curl(v)) * dx, bcs)
    M = assemble_form(system.dielec * inner(u, v) * dx, bcs, diag=0.0)

    nevp_inputs = algorithms.NEVPInputs(
        ka=system.ka,
        gt=system.gt,
        rg_params=system.rg_params,
        L=L,
        M=M,
        N=None,
        Q=None,
        R=None,
        bcs=bcs,
    )

    nevp_inputs.Q = assemble_form(
        D0_constant * system.pump_profile * inner(u, v) * dx,
        bcs,
        diag=0.0,
    )
    modes = algorithms.get_nevp_modes(nevp_inputs)

    nllp = NonLasingLinearProblem(
        V=system.V,
        ka=system.ka,
        gt=system.gt,
        dielec=system.dielec,
        invperm=system.invperm,
        pump=D0_constant * system.pump_profile,
        bcs=bcs,
        ds_obc=None,
    )

    nlA = nllp.create_A(system.n)
    nlL = nllp.create_L(system.n)
    delta_x = nllp.create_dx(system.n)
    initial_x = nllp.create_dx(system.n)

    solver = PETSc.KSP().create(system.msh.comm)
    solver.setOperators(nlA)

    if False:

        def monitor(ksp, its, rnorm):
            print(f"{its}, {rnorm}")

        solver.setMonitor(monitor)

    # Preconditioner (this has a huge impact on performance!!!)
    PC = solver.getPC()
    PC.setType("lu")
    PC.setFactorSolverType("mumps")

    vals = []
    max_iterations = 20

    t0 = time.monotonic()
    for midx in [10]:  # range(7,13):
        initial_mode = modes[midx]

        initial_x.setValues(range(system.n), initial_mode.array)
        initial_x.setValue(system.n, initial_mode.k)
        assert initial_x.getSize() == system.n + 1

        for D0 in D0range:
            log.info(f" {D0=} ".center(80, "#"))
            log.error(f"Starting newton algorithm for mode @ k = {initial_mode.k}")
            D0_constant.value = D0

            i = 0

            newton_steps = []
            while i < max_iterations:
                tstart = time.monotonic()
                with Timer(log.info, "assemble F vec and J matrix"):
                    nllp.assemble_F_and_J(
                        nlL, nlA, initial_x, initial_mode.dof_at_maximum
                    )
                nlL.ghostUpdate(
                    addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
                )
                # Scale residual by -1
                nlL.scale(-1)
                nlL.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT_VALUES,
                    mode=PETSc.ScatterMode.FORWARD,
                )

                with Timer(Print, "Solve KSP"):
                    solver.solve(nlL, delta_x)

                relaxation_param = 1.0
                initial_x += relaxation_param * delta_x

                cur_k = initial_x.getValue(system.n)
                Print(f"DELTA k: {delta_x.getValue(system.n)}")

                i += 1

                # Compute norm of update
                correction_norm = delta_x.norm(0)

                newton_steps.append((cur_k, correction_norm, time.monotonic() - tstart))

                Print(f"----> Iteration {i}: Correction norm {correction_norm}")
                if correction_norm < 1e-10:
                    break

            if correction_norm > 1e-10:
                raise RuntimeError(f"mode at {initial_mode.k} didn't converge")

            Print(f"Initial k: {initial_mode.k} ...")
            df = pd.DataFrame(newton_steps, columns=["k", "corrnorm", "dt"])
            vals.append(np.array([D0, cur_k]))
            Print(df)

            # use the current mode as an initial guess for the mode at the next D0
            # -> we keep initial_x as is.
            evals = np.asarray([mode.k for mode in modes])
            vals.append(np.vstack([D0 * np.ones(evals.shape), evals]).T)

    t_total = time.monotonic() - t0
    log.info(
        f"The eval trajectory code ({D0range.size} D0 steps) took"
        f"{t_total:.1f}s (avg per iteration: {t_total/D0range.size:.3f}s)"
    )

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

        plot_ellipse(ax, system.rg_params)

        ax.grid(True)

    scatter_plot(vals, "Non-Interacting thresholds")
    plt.show()
