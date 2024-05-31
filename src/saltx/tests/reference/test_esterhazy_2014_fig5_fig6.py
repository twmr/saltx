# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces exception point results of "Scalable numerical approach for the
steady-state ab initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""

import logging
import time
from collections import defaultdict, namedtuple
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.optimize
import ufl
from dolfinx import fem
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx import algorithms, nonlasing
from saltx.assemble import assemble_form
from saltx.mesh import create_combined_interval_mesh, create_dcells
from saltx.nonlasing import NonLasingInitialX, NonLasingLinearProblem
from saltx.plot import plot_ellipse

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print
reference_test_dir = Path(__file__).parent


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


@pytest.fixture()
def system():
    use_pml = False
    if not use_pml:
        domains = [
            (None, Fraction("1"), 1000),
            (None, Fraction("0.1"), 100),
            (None, Fraction("1"), 1000),
        ]
        xstart = Fraction("0.0")
    else:
        domains = [
            (None, Fraction("0.8"), 500),
            (None, Fraction("0.2"), 100),
            (None, Fraction("1"), 1000),
            (None, Fraction("0.1"), 100),
            (None, Fraction("1"), 1000),
            (None, Fraction("0.2"), 100),
            (None, Fraction("0.8"), 500),
        ]
        xstart = Fraction("-1.0")
    msh = create_combined_interval_mesh(xstart, domains)
    dcells = create_dcells(msh, xstart, domains)

    dielec = fem.Function(fem.FunctionSpace(msh, ("DG", 0)))
    invperm = fem.Function(fem.FunctionSpace(msh, ("DG", 0)))
    pump_left = fem.Function(fem.FunctionSpace(msh, ("DG", 0)))
    pump_right = fem.Function(fem.FunctionSpace(msh, ("DG", 0)))

    n_cav = 3 + 0.13j
    ka = 9.46
    # the width of a lorentian/cauchy function is 2*gt
    # 2*gt = 0.4

    # it seems as if gt=1.0 was used for the calculations and not gt=0.2, because the
    # saltx with gt=1.0 perfrectly match the fig6 results from the paper.
    gt = 1.0

    radius = 1.0 * gt
    vscale = 0.1 * gt / radius
    rg_params = (ka, radius, vscale)
    Print(f"RG params: {rg_params}")
    del radius
    del vscale

    def cset(func, cells, value):
        func.x.array[cells] = np.full_like(
            cells,
            value,
            dtype=PETSc.ScalarType,
        )

    cells = dcells[2 if use_pml else 0]
    cset(dielec, cells, n_cav**2)
    cset(invperm, cells, 1.0)
    cset(pump_left, cells, 1.0)
    cset(pump_right, cells, 0.0)
    cells = dcells[3 if use_pml else 1]
    cset(dielec, cells, 1.0)
    cset(invperm, cells, 1.0)
    cset(pump_left, cells, 0.0)
    cset(pump_right, cells, 0.0)
    cells = dcells[4 if use_pml else 2]
    cset(dielec, cells, n_cav**2)
    cset(invperm, cells, 1.0)
    cset(pump_left, cells, 0.0)
    cset(pump_right, cells, 1.0)
    if use_pml:
        alpha_pml = 2j
        for cells in [dcells[0], dcells[6]]:
            cset(dielec, cells, 1 + alpha_pml)
            cset(invperm, cells, 1.0 / (1 + alpha_pml))
            # PML alpha term must no be included to avoid wrong eigenvalues
            cset(pump_left, cells, 0.0)
            cset(pump_right, cells, 0.0)
        for cells in [dcells[1], dcells[5]]:
            cset(dielec, cells, 1)
            cset(invperm, cells, 1)
            cset(pump_left, cells, 0.0)
            cset(pump_right, cells, 0.0)

    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    evaluator = algorithms.Evaluator(
        V,
        msh,
        # we only care about the mode intensity at the left and right
        np.array([0.0, 2.1]),
    )

    fine_evaluator = algorithms.Evaluator(
        V,
        msh,
        np.linspace(0.0, 2.1, 4 * 128),
    )

    ds_obc = ufl.ds
    bcs = []
    bcs_norm_constraint = fem.locate_dofs_geometrical(
        V,
        lambda x: x[0] > 0.75,
    )
    bcs_norm_constraint = bcs_norm_constraint[:1]
    Print(f"{bcs_norm_constraint=}")

    n = V.dofmap.index_map.size_global
    et = PETSc.Vec().createSeq(n)
    et.setValue(bcs_norm_constraint[0], 1.0)

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)


def get_D0s(pump_parameter: float) -> tuple[float, float]:
    Dmax = 1.2
    D0left, D0right = Dmax * min(pump_parameter, 1.0), 0.0
    if pump_parameter > 1.0:
        D0right = Dmax * (pump_parameter - 1.0)

    Print(f"{D0left=} {D0right=}")
    return D0left, D0right


def calc_logdetT(k: complex) -> float:
    ik = 1j * k

    # system parameters
    n1 = n3 = 3 + 0.13j  # refactive index of the two 100um slabs
    n2 = 1.0  # air-gap

    L1 = L3 = 0.1  # 100um
    L2 = 0.01  # air gap 10um

    # aux variables
    L12 = L1 + L2
    L123 = L1 + L2 + L3

    ik1 = ik * n1
    ik2 = ik * n2
    ik3 = ik * n3

    # Here are the analytic solutions of the Helmholtz equation in the different regions
    # of the laser:
    # R0-: A*exp(-ikx)
    # R1 : B*exp(ikn1x) + C*exp(-ikn1x)
    # R2 : D*exp(ikn2x) + E*exp(-ikn2x)
    # R3 : F*exp(ikn3x) + G*exp(-ikn3x)
    # R0+: H*exp(ikx)

    # at each of the four interfaces (x=0, x=L1, x=L12, x=L123) continuity conditions
    # (f(x-) = f(x+) and f'(x-) = f'(x+)) need to be fulfilled.

    T = np.array(
        [
            # x=0
            [
                1,  # A
                -1,  # B
                -1,  # C
                0,  # D
                0,  # E
                0,  # F
                0,  # G
                0,  # H
            ],
            [
                -ik,
                -ik1,
                +ik1,
                0,
                0,
                0,
                0,
                0,
            ],
            # x=L1
            [
                0,
                np.exp(ik1 * L1),
                np.exp(-ik1 * L1),
                -np.exp(ik2 * L1),
                -np.exp(-ik2 * L1),
                0,
                0,
                0,
            ],
            [
                0,
                +ik1 * np.exp(ik1 * L1),
                -ik1 * np.exp(-ik1 * L1),
                -ik2 * np.exp(ik2 * L1),
                +ik2 * np.exp(-ik2 * L1),
                0,
                0,
                0,
            ],
            # x=L12
            [
                0,
                0,
                0,
                np.exp(ik2 * L12),
                np.exp(-ik2 * L12),
                -np.exp(ik3 * L12),
                -np.exp(-ik3 * L12),
                0,
            ],
            [
                0,
                0,
                0,
                +ik2 * np.exp(ik2 * L12),
                -ik2 * np.exp(-ik2 * L12),
                -ik3 * np.exp(ik3 * L12),
                +ik3 * np.exp(-ik3 * L12),
                0,
            ],
            # x=L123
            [
                0,
                0,
                0,
                0,
                0,
                np.exp(ik3 * L123),
                np.exp(-ik3 * L123),
                -np.exp(ik * L123),
            ],
            [
                0,
                0,
                0,
                0,
                0,
                +ik3 * np.exp(ik3 * L123),
                -ik3 * np.exp(-ik3 * L123),
                -ik * np.exp(ik * L123),
            ],
        ]
    )
    return np.log(np.abs(np.linalg.det(T)))


def test_nopump():
    """Tests if the complex eigenvalues at d=0 (no pump) match the one in the
    paper.

    This is tested semi-analytically without employing the FEM.
    """

    def plot(K, limitv=True):
        logdet = np.asarray(
            [
                calc_logdetT(
                    k=k,
                )
                for k in K.flatten()
            ]
        ).reshape(K.shape)

        fig, ax = plt.subplots()

        if limitv:
            cax = ax.pcolormesh(Kr, Ki, logdet, shading="gouraud", vmin=20, vmax=15)
        else:
            cax = ax.pcolormesh(Kr, Ki, logdet, shading="gouraud")

        ax.set_xlabel("k.real")
        ax.set_ylabel("k.imag")
        fig.colorbar(cax, ax=ax)
        plt.show()

    if False:
        Nx, Ny = 64 * 4, 64 * 4
        ks_real = np.linspace(93, 96, Nx)
        ks_imag = np.linspace(-5, -5.5, Ny)
        Kr, Ki = np.meshgrid(ks_real, ks_imag)
        K = Kr + 1j * Ki
        plot(K, limitv=False)

    if False:
        # around eigenvalue of one cold cavity mode
        kcenter = 93.45 - 5.1j
        dk = 0.5 + 0.1j

        Nx, Ny = 64 * 2, 64 * 2
        ks_real = np.linspace((kcenter - dk).real, (kcenter + dk).real, Nx)
        ks_imag = np.linspace((kcenter - dk).imag, (kcenter + dk).imag, Ny)
        Kr, Ki = np.meshgrid(ks_real, ks_imag)
        K = Kr + 1j * Ki
        plot(K, limitv=False)

    kinits = [93.5 - 5.2j, 95.7 - 5.3j]
    solutions = []
    for kinit in kinits:

        def error(k):
            return calc_logdetT(k=k[0] + 1j * k[1])

        dk = 0.5
        result = scipy.optimize.minimize(
            error,
            [kinit.real, kinit.imag],
            method="Nelder-Mead",
            # method="Powell",
            options={"maxiter": 1000},
            bounds=[
                (kinit.real - dk, kinit.real + dk),
                (kinit.imag - dk, kinit.imag + dk),
            ],
        )
        assert result.success
        solutions.append(result.x[0] + 1j * result.x[1])

        assert result.fun < -8

        print(result)

    assert abs(solutions[0] - (93.42 - 5.142j)) < 0.1
    assert abs(solutions[1] - (95.86 - 5.275j)) < 0.1

    plt.show()


def test_evaltraj(system, infra):
    """Determine the non-interacting eigenvalues of the system from the first
    threshold till the second threshold using a newton solver (nonlasing newton
    solver)."""

    # pump_range = np.linspace(0.003, 0.28, 25)
    # pump_range = np.linspace(0.85, 1.6, 40)
    pump_range = np.linspace(0.01, 0.02, 2)
    pump_range = np.linspace(0.94, 0.99, 10)
    pump_range = np.linspace(0.94, 0.002, 60)
    pump_range = np.linspace(1.8, 0.002, 100)
    # pump_range = np.linspace(1.8, 1.6, 10)

    # The first threshold is close to D0=0.16

    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0left, D0right = get_D0s(pump_range[0])
    D0left_constant = real_const(system.V, D0left)
    D0right_constant = real_const(system.V, D0right)

    L = assemble_form(
        -inner(system.invperm * nabla_grad(u), nabla_grad(v)) * dx, system.bcs, name="L"
    )
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0, name="M")
    Q = assemble_form(
        (D0left_constant * system.pump_left + D0right_constant * system.pump_right)
        * inner(u, v)
        * dx,
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
        invperm=system.invperm,
        sigma_c=None,
        pump=(
            D0left_constant * system.pump_left + D0right_constant * system.pump_right
        ),
        bcs=system.bcs,
        ds_obc=ufl.ds,
    )

    nl_newton_operators = nonlasing.create_solver_and_matrices(nllp, nmodes=len(modes))

    def update_dofmax_of_initial_mode(nlm, init_x: NonLasingInitialX) -> None:
        init_x.dof_at_maximum = nlm.dof_at_maximum

    def init_mode(mode_idx: int) -> NonLasingInitialX:
        init_x: NonLasingInitialX = nl_newton_operators.initial_x_seq[mode_idx]
        mode = modes[mode_idx]

        init_x.vec.setValues(range(system.n), mode.array)
        init_x.vec.setValue(system.n, mode.k)
        assert init_x.vec.getSize() == system.n + 1
        update_dofmax_of_initial_mode(mode, init_x)
        return init_x

    if False:
        initial_x = init_mode(0)
        nllp.assemble_F_and_J(
            nl_newton_operators.L,
            nl_newton_operators.A,
            initial_x.vec,
            initial_x.dof_at_maximum,
        )
        return

    t0 = time.monotonic()
    all_parametrized_modes = defaultdict(list)
    for midx in range(len(modes)):
        initial_x = init_mode(midx)
        cur_dof_at_maximum = initial_x.dof_at_maximum

        for pump in pump_range:
            log.info(f" {pump=} ".center(80, "#"))
            log.error(f"Starting newton algorithm for mode @ k = {initial_x.k}")

            D0left_constant.value, D0right_constant.value = get_D0s(pump)

            new_nlm = algorithms.newton(
                nllp,
                nl_newton_operators.L,
                nl_newton_operators.A,
                initial_x.vec,
                nl_newton_operators.delta_x,
                nl_newton_operators.solver,
                cur_dof_at_maximum,
                initial_x.bcs,
            )
            cur_dof_at_maximum = new_nlm.dof_at_maximum

            all_parametrized_modes[pump].append(new_nlm)
            # In this loop we use the current mode as an initial guess for the mode at
            # the next D0 -> we keep initial_x as is.

    t_total = time.monotonic() - t0
    log.info(
        f"The eval trajectory code ({pump_range.size} D0 steps) took"
        f"{t_total:.1f}s (avg per iteration: {t_total/pump_range.size:.3f}s)",
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

        sc = ax.scatter(X, Y, c=C, norm=norm, alpha=1.0, label="saltx")
        ax.set_xlabel("Re(k)")
        ax.set_ylabel("Im(k)")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("D0", loc="top")

        plot_ellipse(ax, system.rg_params)
        ka, gt = system.ka, system.gt
        ax.plot(ka, -gt, "ro", label="singularity")
        ax.plot([9.342, 9.586], [-0.5142, -0.5275], "rx", label="cold-cavity mode")

        ax.grid(True)
        return fig, ax

    fig, ax = scatter_plot(
        np.asarray(
            [
                (D0, mode.k)
                for D0, modes in all_parametrized_modes.items()
                for mode in modes
            ],
        ),
        "Non-Interacting thresholds",
    )
    scale = 0.1
    data = np.loadtxt(reference_test_dir / "esterhazy_ep_fig6_red.csv", delimiter=",")
    ax.plot(data[:, 0] * scale, data[:, 1] * scale, "ro", alpha=0.1, label="paper")
    data = np.loadtxt(reference_test_dir / "esterhazy_ep_fig6_blue.csv", delimiter=",")
    ax.plot(data[:, 0] * scale, data[:, 1] * scale, "bo", alpha=0.1, label="paper")
    ax.legend()

    ax.set_xlim([9.325, 9.6])
    ax.set_ylim([-0.6, 0.1])
    infra.save_plot(fig, name="full")

    fig, ax = scatter_plot(
        np.asarray(
            [
                (D0, mode.k)
                for D0, modes in all_parametrized_modes.items()
                for mode in modes
            ],
        ),
        "Non-Interacting thresholds",
    )

    scale = 0.1
    data = np.loadtxt(
        reference_test_dir / "esterhazy_ep_fig6_red_zoomed.csv", delimiter=","
    )
    ax.plot(data[:, 0] * scale, data[:, 1] * scale, "ro", alpha=0.1, label="paper")
    data = np.loadtxt(
        reference_test_dir / "esterhazy_ep_fig6_blue_zoomed.csv", delimiter=","
    )
    ax.plot(data[:, 0] * scale, data[:, 1] * scale, "bo", alpha=0.1, label="paper")
    ax.legend()

    ax.set_xlim([9.41, 9.56])
    ax.set_ylim([-0.04, 0.04])
    infra.save_plot(fig, name="zoom")
    # TODO plot some mode profiles
    plt.show()


def test_plot_fig6_ref_data_from_paper(infra):
    fig, ax = plt.subplots()
    ax.grid(True)

    data = np.loadtxt(
        reference_test_dir / "esterhazy_ep_fig6_red_zoomed.csv", delimiter=","
    )
    ax.plot(data[:, 0], data[:, 1], "ro", alpha=0.2, label="paper")
    data = np.loadtxt(
        reference_test_dir / "esterhazy_ep_fig6_blue_zoomed.csv", delimiter=","
    )
    ax.plot(data[:, 0], data[:, 1], "bo", alpha=0.2, label="paper")

    ax.set_xlabel("Re(k) [mm^-1]")
    ax.set_ylabel("Im(k) [mm^-1]")
    infra.save_plot(fig, name="b")

    fig, ax = plt.subplots()
    ax.grid(True)

    data = np.loadtxt(reference_test_dir / "esterhazy_ep_fig6_red.csv", delimiter=",")
    ax.plot(data[:, 0], data[:, 1], "ro", alpha=0.2, label="paper")
    data = np.loadtxt(reference_test_dir / "esterhazy_ep_fig6_blue.csv", delimiter=",")
    ax.plot(data[:, 0], data[:, 1], "bo", alpha=0.2, label="paper")

    ax.set_xlabel("Re(k) [mm^-1]")
    ax.set_ylabel("Im(k) [mm^-1]")
    infra.save_plot(fig, name="a")
    plt.show()
