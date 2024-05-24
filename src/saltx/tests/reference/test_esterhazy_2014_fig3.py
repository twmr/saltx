# Copyright (C) 2024 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""
# TODO the system parameters seem to be wrong
# the threshold of the fig 3 plot is: 0.0741 (webplotdigitizer)

import logging
import time
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx import algorithms
from saltx.assemble import assemble_form
from saltx.nonlasing import NonLasingLinearProblem
from saltx.plot import plot_ellipse

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


@pytest.fixture()
def system():
    pump_profile = 1.0

    # not sure if this is correct
    dielec = 2.0**2

    L = 1.0  # 0.1 mm
    ka = 30.0  # 300 mm^-1
    gt = 0.3  # 3 mm^-1

    radius = 1.5 * gt
    vscale = 1.3 * gt / radius
    rg_params = (ka + 0.14j, radius, vscale)
    Print(f"RG params: {rg_params}")
    del radius
    del vscale

    msh = mesh.create_interval(MPI.COMM_WORLD, points=(0, L), nx=1024)

    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    evaluator = algorithms.Evaluator(
        V,
        msh,
        np.array([1]),
    )

    ds_obc = ufl.ds

    # it is not explicitly stated if the system is open only on one edge or on both
    bcs_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))

    Print(f"{bcs_dofs=}")
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs, V),
    ]

    n = V.dofmap.index_map.size_global

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)


def test_evaltraj(system, infra):
    """Determine the non-interacting eigenvalues of the system from the first
    threshold till the second threshold using a newton solver (nonlasing newton
    solver)."""
    # D0range = np.linspace(0.07, 0.22, 16)[::-1]
    # D0range = np.linspace(0.01, 0.20, 26)[::-1]
    # D0range = np.array(
    #     [
    #         0.0739,
    #         0.0741,
    #         0.0743,
    #     ]
    # )
    D0range = np.linspace(0.07, 0.15, 20)[::-1]

    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0_constant = real_const(system.V, D0range[0])

    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx, system.bcs, name="L")
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0, name="M")
    Q = assemble_form(
        D0_constant * system.pump_profile * inner(u, v) * dx,
        system.bcs,
        diag=0.0,
        name="Q",
    )
    R = assemble_form(inner(u, v) * system.ds_obc, system.bcs, diag=0.0, name="R")

    nevp_inputs = algorithms.NEVPInputs(
        ka=system.ka,
        gt=system.gt,
        rg_params=system.rg_params,
        L=L,
        M=M,
        N=None,
        Q=Q,
        R=R,
        bcs=system.bcs,
    )

    modes = algorithms.get_nevp_modes(nevp_inputs)

    nllp = NonLasingLinearProblem(
        V=system.V,
        ka=system.ka,
        gt=system.gt,
        dielec=system.dielec,
        invperm=None,
        sigma_c=None,
        pump=D0_constant * system.pump_profile,
        bcs=system.bcs,
        ds_obc=ufl.ds,
    )

    nlA = nllp.create_A()
    nlL = nllp.create_L()
    delta_x = nllp.create_dx()
    initial_x = nllp.create_dx()

    solver = PETSc.KSP().create(system.msh.comm)
    solver.setOperators(nlA)

    PC = solver.getPC()
    PC.setType("lu")
    PC.setFactorSolverType("mumps")

    if False:
        initial_mode = modes[0]
        initial_x.setValues(range(system.n), initial_mode.array)
        initial_x.setValue(system.n, initial_mode.k)
        assert initial_x.getSize() == system.n + 1

        nllp.assemble_F_and_J(nlL, nlA, initial_x, initial_mode.dof_at_maximum)
        return

    t0 = time.monotonic()
    all_parametrized_modes = defaultdict(list)
    for midx in range(len(modes)):
        initial_mode = modes[midx]
        cur_dof_at_maximum = initial_mode.dof_at_maximum

        initial_x.setValues(range(system.n), initial_mode.array)
        initial_x.setValue(system.n, initial_mode.k)
        assert initial_x.getSize() == system.n + 1

        for _Di, D0 in enumerate(D0range):
            log.info(f" {D0=} ".center(80, "#"))
            log.error(f"Starting newton algorithm for mode @ k = {initial_mode.k}")
            D0_constant.value = D0

            new_nlm = algorithms.newton(
                nllp,
                nlL,
                nlA,
                initial_x,
                delta_x,
                solver,
                cur_dof_at_maximum,
                initial_mode.bcs,
            )
            cur_dof_at_maximum = new_nlm.dof_at_maximum

            all_parametrized_modes[D0].append(new_nlm)
            # In this loop we use the current mode as an initial guess for the mode at
            # the next D0 -> we keep initial_x as is.

    t_total = time.monotonic() - t0
    log.info(
        f"The eval trajectory code ({D0range.size} D0 steps) took"
        f"{t_total:.1f}s (avg per iteration: {t_total/D0range.size:.3f}s)",
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
        ka, gt = system.ka, system.gt
        ax.plot(ka, -gt, "ro", label="singularity"),

        ax.grid(True)
        return fig

    fig = scatter_plot(
        np.asarray(
            [
                (D0, mode.k)
                for D0, modes in all_parametrized_modes.items()
                for mode in modes
            ],
        ),
        "Non-Interacting thresholds",
    )

    infra.save_plot(fig)
    # TODO plot some mode profiles
    plt.show()
