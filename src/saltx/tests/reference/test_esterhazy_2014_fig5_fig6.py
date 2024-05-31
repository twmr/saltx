# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces exception point results of "Scalable numerical approach for the
steady-state ab initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


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
