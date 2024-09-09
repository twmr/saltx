# Copyright (C) 2024 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.

The cold cavity modes of the 1D slab laser (open on the right lead, closed on the left),
which is used in Fig 3, can be calculated as follows:

The length of the laser is L=1 and eps_c = 2**2 => n_c = 2

Aexp(iknx) + Bexp(-iknx) = Cexp(ikx)
C*exp(iRek x)*exp(-Imk x)

x=0, A=-B  # since at x=0 there is a hard wall

w.l.o.g C=1

A(exp(iknL) - exp(-iknL)) = exp(ikL) at x=L (1)
iknA*exp(iknL) + iknA*exp(-iknL) = ik*exp(ikL) (2) = iknA(exp(iknL) + exp(-iknL))

(2)/(1):

=> ikn(exp(iknL) + exp(-iknL))/(exp(iknL) - exp(-iknL)) = ik

/ik:

=> n(exp(iknL) + exp(-iknL))/(exp(iknL) - exp(-iknL)) = 1 => cot(knL) = 1/n

# Note that log(-x) = log(|x|) + i*pi; see
https://math.stackexchange.com/questions/2089690/log-of-a-negative-number
#  --> -i * log(-x) = -i * log(|x|) + pi

=> knL = 1/2(2*m*pi - i log((-n-1)/(n-1))), m in Z
=> knL = 1/2(2*(m+1)*pi - i log( |(-n-1)/(n-1)| )), m in Z

=> Imk = -1/(2nL) * log(3) = -np.log(3) / 4 / 1.0 = -0.27465 !!!

(see Solution for the variable x in
https://www.wolframalpha.com/input?i=%28exp%28i*x%29+%2B+exp%28-i*x%29%29+%2F+%28exp%28i*x%29+-+exp%28-i*x%29%29+%3D+1%2Fp)
Rek = 1/(2nL)*(2m+1)*pi

In [7]: 1/4.0*(2*np.arange(25)+1)*np.pi
Out[7]:
array([ 0.78539816,  2.35619449,  3.92699082,  5.49778714,  7.06858347,
        8.6393798 , 10.21017612, 11.78097245, 13.35176878, 14.9225651 ,
       16.49336143, 18.06415776, 19.63495408, 21.20575041, 22.77654674,
       24.34734307, 25.91813939, 27.48893572, 29.05973205, 30.63052837,
       32.2013247 , 33.77212103, 35.34291735, 36.91371368])
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

from saltx import algorithms, nonlasing
from saltx.assemble import assemble_form
from saltx.nonlasing import NonLasingInitialX, NonLasingLinearProblem
from saltx.plot import plot_parametrized_ciss_eigenvalues

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


@pytest.fixture
def system():
    pump_profile = 1.0

    # not sure if this is correct
    dielec = 2.0**2

    L = 1.0  # 0.1 mm
    ka = 30.0  # 300 mm^-1
    gt = 0.3  # 3 mm^-1

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

    # The NEVP solver is used for determining the eigenmodes, which are very close to
    # the cold cavity modes
    radius = 4 * system.gt
    vscale = 0.02 * system.gt / radius
    rg_params = (system.ka + -0.274j, radius, vscale)
    Print(f"RG params: {rg_params}")
    del radius
    del vscale

    D0range = np.linspace(0.003, 0.28, 25)
    # The first threshold is close to D0=0.16

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
        rg_params=rg_params,
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

        for _Di, D0 in enumerate(D0range):
            log.info(f" {D0=} ".center(80, "#"))
            log.error(f"Starting newton algorithm for mode @ k = {initial_x.k}")
            D0_constant.value = D0

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

            all_parametrized_modes[D0].append(new_nlm)
            # In this loop we use the current mode as an initial guess for the mode at
            # the next D0 -> we keep initial_x as is.

    t_total = time.monotonic() - t0
    log.info(
        f"The eval trajectory code ({D0range.size} D0 steps) took"
        f"{t_total:.1f}s (avg per iteration: {t_total/D0range.size:.3f}s)",
    )

    fig, ax = plt.subplots()
    fig.suptitle("Non-Interacting thresholds")

    cold_cavity_k_real = np.array([29.05973205, 30.63052837])
    cold_cavity_k_imag = np.array([-0.27465] * 2)
    ax.plot(cold_cavity_k_real, cold_cavity_k_imag, "rx", label="cold-cavity evals")
    plot_parametrized_ciss_eigenvalues(
        ax,
        np.asarray(
            [
                (D0, mode.k)
                for D0, modes in all_parametrized_modes.items()
                for mode in modes
            ],
        ),
        parametername="D0",
        rg_params=rg_params,
        kagt=(system.ka, system.gt),
    )

    infra.save_plot(fig)
    # TODO plot some mode profiles
    plt.show()
