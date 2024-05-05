# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""
import logging
import time
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx import algorithms, newtils
from saltx.assemble import assemble_form
from saltx.lasing import NonLinearProblem
from saltx.log import Timer
from saltx.nonlasing import NonLasingLinearProblem
from saltx.plot import plot_ellipse

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


@pytest.fixture()
def system():
    pump_profile = 1.0
    dielec = 1.01**2
    ka = 250.0
    gt = 7.5

    radius = 3.2 * gt
    vscale = 1.35 * gt / radius
    rg_params = (ka + 4j, radius, vscale)
    Print(f"RG params: {rg_params}")
    del radius
    del vscale

    msh = mesh.create_interval(MPI.COMM_WORLD, points=(0, 0.1), nx=1024)

    V = fem.functionspace(msh, ("Lagrange", 3))

    evaluator = algorithms.Evaluator(
        V,
        msh,
        np.array([0.1]),
    )

    ds_obc = ufl.ds
    bcs_dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0.0),
    )

    Print(f"{bcs_dofs=}")
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs, V),
    ]

    n = V.dofmap.index_map.size_global

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)


def nonlasing_newton(
    nllp,
    nlL,
    nlA,
    initial_x,
    delta_x,
    solver,
    mode,
    max_iterations=25,
    correction_norm_threshold=1e-10,
):
    """Upon success, ``initial_x`` contains the mode and k."""
    # TODO return a dataclass??
    # TODO clean this up and move it algorithms.py

    n = initial_x.getSize() - 1
    i = 0
    newton_steps = []
    while i < max_iterations:
        tstart = time.monotonic()
        with Timer(log.info, "assemble F vec and J matrix"):
            nllp.assemble_F_and_J(nlL, nlA, initial_x, mode.dof_at_maximum)
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
        initial_x += delta_x * relaxation_param

        cur_k = initial_x.getValue(n)
        Print(f"DELTA k: {delta_x.getValue(n)}")

        i += 1

        # Compute norm of update
        correction_norm = delta_x.norm(0)

        newton_steps.append((cur_k, correction_norm, time.monotonic() - tstart))

        Print(f"----> Iteration {i}: Correction norm {correction_norm}")
        if correction_norm < correction_norm_threshold:
            break

    if correction_norm > correction_norm_threshold:
        raise RuntimeError(f"mode at {mode.k} didn't converge")

    Print(f"Initial k: {mode.k} ...")
    df = pd.DataFrame(newton_steps, columns=["k", "corrnorm", "dt"])
    Print(df)

    new_mode = initial_x.getArray().copy()[:-1]  # remove the k

    dof_at_maximum = np.abs(new_mode).argmax()
    val_maximum = new_mode[np.abs(new_mode).argmax()]
    Print(
        f" {cur_k.real:9f}{cur_k.imag:+9f} j {correction_norm:12g}   "
        f"{val_maximum.real:2g} j {val_maximum.imag:2g}"
    )

    # fix norm and the phase
    new_mode /= val_maximum
    return algorithms.NEVPNonLasingMode(
        array=new_mode,
        k=cur_k,
        error=correction_norm,
        bcs_name="default",
        bcs=mode.bcs,
        dof_at_maximum=dof_at_maximum,
    )


def test_evaltraj(system):
    """Determine the non-interacting eigenvalues of the system from the first
    threshold till the second threshold using a newton solver (nonlasing newton
    solver)."""
    D0range = np.linspace(0.1523, 0.49, 16)[::-1]
    # D0range = np.linspace(0.49, 0.8, 20)
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0_constant = real_const(system.V, D0range[0])

    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx, system.bcs)
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0)
    Q = assemble_form(
        D0_constant * system.pump_profile * inner(u, v) * dx, system.bcs, diag=0.0
    )
    R = assemble_form(inner(u, v) * system.ds_obc, system.bcs, diag=0.0)

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

    nlA = nllp.create_A(system.n)
    nlL = nllp.create_L(system.n)
    delta_x = nllp.create_dx(system.n)
    initial_x = nllp.create_dx(system.n)

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

        initial_x.setValues(range(system.n), initial_mode.array)
        initial_x.setValue(system.n, initial_mode.k)
        assert initial_x.getSize() == system.n + 1

        for Di, D0 in enumerate(D0range):
            log.info(f" {D0=} ".center(80, "#"))
            log.error(f"Starting newton algorithm for mode @ k = {initial_mode.k}")
            D0_constant.value = D0

            all_parametrized_modes[D0].append(
                nonlasing_newton(
                    nllp, nlL, nlA, initial_x, delta_x, solver, initial_mode
                )
            )
            # In this loop we use the current mode as an initial guess for the mode at
            # the next D0 -> we keep initial_x as is.

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
        ka, gt = system.ka, system.gt
        ax.plot(ka, -gt, "ro", label="singularity")

        ax.grid(True)

    scatter_plot(
        np.asarray(
            [
                (D0, mode.k)
                for D0, modes in all_parametrized_modes.items()
                for mode in modes
            ]
        ),
        "Non-Interacting thresholds",
    )

    # TODO plot some mode profiles
    plt.show()


def test_intensity_vs_pump_esterhazy(system):
    D0range = np.linspace(0.1523, 0.49, 32)[::-1]
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0_constant = real_const(system.V, 1.0)

    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx, system.bcs)
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0)
    Q_form = D0_constant * system.pump_profile * inner(u, v) * dx
    Q = assemble_form(Q_form, system.bcs, diag=0.0)
    R = assemble_form(inner(u, v) * system.ds_obc, system.bcs, diag=0.0)

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

    nlp = NonLinearProblem(
        system.V,
        system.ka,
        system.gt,
        dielec=system.dielec,
        n=system.n,
        pump=D0_constant * system.pump_profile,
        ds_obc=system.ds_obc,
    )

    nllp = NonLasingLinearProblem(
        V=system.V,
        ka=system.ka,
        gt=system.gt,
        dielec=system.dielec,
        invperm=None,
        sigma_c=None,
        pump=D0_constant * system.pump_profile,
        bcs=system.bcs,
        ds_obc=system.ds_obc,
    )

    nlA = nllp.create_A(system.n)
    nlL = nllp.create_L(system.n)
    delta_x = nllp.create_dx(system.n)
    initial_xs = [nllp.create_dx(system.n) for _ in range(4)]

    solver = PETSc.KSP().create(system.msh.comm)
    solver.setOperators(nlA)
    PC = solver.getPC()
    PC.setType("lu")
    PC.setFactorSolverType("mumps")

    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=2)

    aevals = []
    results = []  # list of (D0, intensity) tuples

    modes = []
    active_modes = []

    all_nonlasing_modes = {}

    for D_index, D0 in enumerate(D0range):
        Print(f" {D0=} ".center(80, "#"))
        D0_constant.value = D0
        if not D_index:
            assemble_form(Q_form, system.bcs, diag=0.0, mat=nevp_inputs.Q)
            modes = algorithms.get_nevp_modes(nevp_inputs)
            assert len(modes) == 4
            for mode, initial_x in zip(modes, initial_xs):
                initial_x.setValues(range(system.n), mode.array)
                initial_x.setValue(system.n, mode.k)
        else:
            # use the previous modes as an initial-guess for the newton solver
            modes = [
                nonlasing_newton(nllp, nlL, nlA, initial_x, delta_x, solver, mode)
                for mode, initial_x in zip(modes, initial_xs)
            ]
        evals = np.asarray([mode.k for mode in modes])
        assert evals.size
        aevals.append(np.vstack((D0 * np.ones(evals.shape), evals)).T)
        all_nonlasing_modes[D0] = modes

    log.info("Finished calculating all non-lasing modes")

    cpa_counter = 0
    for D0, modes in all_nonlasing_modes.items():
        Print(f" {D0=} ".center(80, "#"))
        D0_constant.value = D0

        s_init = 1.0
        real_axis_threshold = 3e-9
        if not active_modes:
            log.warning("Starting CPA")
            active_modes = algorithms.constant_pump_algorithm(
                modes,
                nevp_inputs,
                nlp,
                newton_operators,
                real_axis_threshold=3e-9,
                s_init=s_init,
            )
            log.warning("Finished CPA")
        else:
            minfos = [
                newtils.NewtonModeInfo(
                    k=mode.k.real,
                    s=mode.s,
                    re_array=mode.array.real / mode.s,
                    im_array=mode.array.imag / mode.s,
                    dof_at_maximum=mode.dof_at_maximum,
                )
                for mode in active_modes
            ]
            n_active_modes = len(active_modes)
            nops = newton_operators[n_active_modes]
            try:
                refined_modes = algorithms.refine_modes(
                    minfos,
                    system.bcs,
                    nops.solver,
                    nlp,
                    nops.A,
                    nops.L,
                    nops.delta_x,
                    nops.initial_x,
                    fail_early=True,
                )
            except algorithms.RefinementError:
                # the modes couldn't be refined
                log.error("Starting CPA")
                active_modes = algorithms.constant_pump_algorithm(
                    modes,
                    nevp_inputs,
                    nlp,
                    newton_operators,
                    real_axis_threshold=real_axis_threshold,
                    s_init=s_init,
                )
                cpa_counter += 1
                log.error("Finished CPA")
            else:
                assert all(rm.converged for rm in refined_modes)

                check_other_activated_modes = False
                if check_other_activated_modes:
                    log.debug(
                        "Before assembly of Q with custom hole-burning term"
                        "(active_modes:{len(refined_modes)})"
                    )
                    # this modifies the Q matrix in nevp_inputs
                    nlp.update_b_and_k_for_forms(refined_modes)
                    Q = nevp_inputs.Q
                    Q.zeroEntries()
                    fem.petsc.assemble_matrix(
                        Q,
                        nlp.get_Q_hbt_form(n_active_modes),
                        bcs=system.bcs,
                        diagonal=0.0,
                    )
                    Q.assemble()
                    log.debug("After assembly of Q with custom SHB term")

                    modes = algorithms.get_nevp_modes(nevp_inputs)
                    evals = np.asarray([mode.k for mode in modes])

                    number_of_modes_close_to_real_axis = np.sum(
                        np.abs(evals.imag) < real_axis_threshold
                    )
                    Print(
                        f"Number of modes close to real axis: "
                        f"{number_of_modes_close_to_real_axis}"
                    )

                    assert number_of_modes_close_to_real_axis == n_active_modes

                    number_of_modes_above_real_axis = np.sum(
                        evals.imag > real_axis_threshold
                    )
                    Print(
                        f"Number of modes above real axis: "
                        f"{number_of_modes_above_real_axis}"
                    )
                    if number_of_modes_above_real_axis > 0:
                        raise RuntimeError(
                            "New mode turned on. This is not "
                            "expected for this test-case"
                        )

                active_modes = refined_modes

        if False:
            active_evals = np.asarray([mode.k for mode in active_modes])
            aevals.append(np.vstack((D0 * np.ones(active_evals.shape), active_evals)).T)

        for mode in active_modes:
            mode_values = system.evaluator(mode)
            mode_intensity = abs(mode_values) ** 2
            Print(f"-> {mode_intensity=}")
            results.append((D0, mode_intensity))

    assert cpa_counter == 1  # when the number of modes has changed

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
        ax.plot(system.ka, -system.gt, "ro", label="singularity")

        ax.grid(True)

    scatter_plot(aevals, "Non-Interacting thresholds")

    # Extracted from fig 4 using webplotdigitizer
    # 1st threshold: D0 ~ 0.138
    # 2nd threshold: D0 ~ 0.189
    # 3rd threshold: D0 ~ 0.45

    # The first threshold can analytically be determined for the modeled laser:
    # -> [D0=1.523e-01  k=2.472e+02]

    # TODO The threshold of the first mode matches the one of the paper, but the 2nd
    # mode activates at a higher pump at not around D0=0.189.
    fig, ax = plt.subplots()
    ax.plot(
        [D0 for (D0, _) in results],
        [intens for (_, intens) in results],
        "x",
    )
    ax.set_xlabel("Pump D0")
    ax.set_ylabel("Modal intensity at right lead")
    ax.grid(True)
    plt.show()
