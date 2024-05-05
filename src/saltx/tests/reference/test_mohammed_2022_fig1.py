# Copyright (C) 2024 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces Fig1 of "Nonlinear exceptional-point lasing with ab initio
Maxwell-Bloch theory".

See https://doi.org/10.1063/5.0105963
"""
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc
from scipy.optimize import root
from ufl import dx, inner, nabla_grad

from saltx import algorithms
from saltx.assemble import assemble_form
from saltx.nonlasing import NonLasingLinearProblem

log = logging.getLogger(__name__)


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


def test_determine_first_threshold_contour_fig1(mohammed_system):
    system = mohammed_system
    # first set D2=0 and scan D1
    # set D1=initialD1
    # solve nevp at the D1,D2
    # -> increase D1 and use newton
    # until one eval is above real axis
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    initial_D1 = 0.6
    d1_constant = real_const(system.V, initial_D1)
    d2_constant = real_const(system.V, 0.0)

    pump_expr = d1_constant * system.pump_left + d2_constant * system.pump_right

    ds_obc = ufl.ds
    L = assemble_form(
        -inner(system.invperm * nabla_grad(u), nabla_grad(v)) * dx, system.bcs
    )
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0)
    N = assemble_form(system.sigma_c * inner(u, v) * dx, system.bcs, diag=0.0)
    R = assemble_form(inner(u, v) * ds_obc, system.bcs, diag=0.0)
    Q = assemble_form(pump_expr * inner(u, v) * dx, system.bcs, diag=0.0)

    nevp_inputs = algorithms.NEVPInputs(
        ka=system.ka,
        gt=system.gt,
        rg_params=system.rg_params,
        L=L,
        M=M,
        N=N,
        Q=Q,
        R=R,
        bcs=system.bcs,
    )

    modes = algorithms.get_nevp_modes(nevp_inputs)
    evals = np.asarray([mode.k for mode in modes])
    assert evals.imag.max() < 0

    nllp = NonLasingLinearProblem(
        V=system.V,
        ka=system.ka,
        gt=system.gt,
        dielec=system.dielec,
        invperm=system.invperm,
        sigma_c=system.sigma_c,
        pump=pump_expr,
        bcs=system.bcs,
        ds_obc=ds_obc,
    )

    nlA = nllp.create_A(system.n)
    nlL = nllp.create_L(system.n)
    delta_x = nllp.create_dx(system.n)
    initial_x1 = nllp.create_dx(system.n)  # for mode1
    initial_x2 = nllp.create_dx(system.n)  # for mode2

    solver = PETSc.KSP().create(system.msh.comm)
    solver.setOperators(nlA)

    PC = solver.getPC()
    PC.setType("lu")
    PC.setFactorSolverType("mumps")

    ##############################################
    m1 = modes[0]
    cur_k1 = m1.k
    cur_dof1 = m1.dof_at_maximum
    initial_x1.setValues(range(system.n), m1.array)
    initial_x1.setValue(system.n, m1.k)
    assert initial_x1.getSize() == system.n + 1

    m2 = modes[1]
    cur_k2 = m2.k
    cur_dof2 = m2.dof_at_maximum
    initial_x2.setValues(range(system.n), m2.array)
    initial_x2.setValue(system.n, m2.k)
    assert initial_x2.getSize() == system.n + 1

    D1_range = np.linspace(initial_D1, 0.80, 20)
    all_parametrized_modes = defaultdict(list)
    vals = []
    for _Di, D1val in enumerate(D1_range):
        log.info(f" {D1val=} ".center(80, "#"))

        if False:
            nllp._demo_check_solutions(initial_x1)
            nllp._demo_check_solutions(initial_x2)

        d1_constant.value = D1val

        # FIXME mention previous pump step
        log.error(f"Starting newton algorithm for mode1 @ k = {cur_k1}")
        new_nlm1 = algorithms.newton(
            nllp, nlL, nlA, initial_x1, delta_x, solver, cur_dof1, m1.bcs
        )
        all_parametrized_modes[D1val].append(new_nlm1)

        log.error(f"Starting newton algorithm for mode2 @ k = {cur_k2}")
        new_nlm2 = algorithms.newton(
            nllp, nlL, nlA, initial_x2, delta_x, solver, cur_dof2, m2.bcs
        )
        all_parametrized_modes[D1val].append(new_nlm2)

        cur_dof1 = new_nlm1.dof_at_maximum
        cur_dof2 = new_nlm2.dof_at_maximum
        cur_k1 = new_nlm1.k
        cur_k2 = new_nlm2.k
        vals.append(np.array([D1val, cur_k1, cur_k2]))

        if cur_k1.imag > 0:
            log.info("mode1.k above real axis")
            break
        if cur_k2.imag > 0:
            log.info("mode2.k above real axis")
            break

        # use the current mode as an initial guess for the mode at the next D0
        # -> we keep initial_x as is.

    fig, axes = plt.subplots(nrows=2, sharex=True)

    k1 = np.asarray([k1 for _, k1, _ in vals])
    k2 = np.asarray([k2 for _, _, k2 in vals])
    D1 = np.asarray([D1val for D1val, _, _ in vals]).real
    norm = plt.Normalize(D1.min(), D1.max())

    sc1 = axes[0].scatter(k1.real, k1.imag, c=D1, norm=norm)
    axes[1].scatter(k2.real, k2.imag, c=D1, norm=norm)

    axes[0].grid(True)
    axes[1].grid(True)
    fig.colorbar(sc1, ax=axes)
    plt.show()

    ########################
    # find the first threshold when D2=0.0
    # then track the threshold-mode for increasing D2

    def objfunc(D1val):
        d1_constant.value = D1val
        log.error(f"objfunc: {D1val=}")

        new_nlm1 = algorithms.newton(
            nllp, nlL, nlA, initial_x1, delta_x, solver, cur_dof1, m1.bcs
        )
        all_parametrized_modes[D1val.item()].append(new_nlm1)

        new_nlm2 = algorithms.newton(
            nllp, nlL, nlA, initial_x2, delta_x, solver, cur_dof2, m2.bcs
        )
        all_parametrized_modes[D1val.item()].append(new_nlm2)

        return max([new_nlm2.k.imag, new_nlm1.k.imag])

    results = []  # (D1, D2)
    D2range = np.linspace(0.0, 0.4, 12)
    D2range = np.linspace(0.0, 0.8, 40)
    prev_D1_result = D1val

    for D2val in D2range:
        d2_constant.value = D2val

        root_result = root(objfunc, prev_D1_result, tol=1e-8)
        prev_D1_result = root_result.x.item()
        results.append([prev_D1_result, D2val])
        if prev_D1_result < D2val:
            break

    fig, ax = plt.subplots()

    data = np.asarray(results)
    ax.plot(data[:, 0], data[:, 1], "x")

    # 45deg line
    ax.plot([0, 1], [0, 1], "-")

    # TODO plot data from paper

    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlabel("D1")
    ax.set_ylabel("D2")
    plt.show()
