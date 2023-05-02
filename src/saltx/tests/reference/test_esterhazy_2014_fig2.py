# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Reproduces results of "Scalable numerical approach for the steady-state ab
initio laser theory".

See https://link.aps.org/doi/10.1103/PhysRevA.90.023816.
"""
import enum
import logging
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_matrix
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx import algorithms, newtils
from saltx.assemble import assemble_form
from saltx.lasing import NonLinearProblem
from saltx.plot import plot_ciss_eigenvalues

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print


class BCType(enum.Enum):
    NONE = enum.auto()  # Length = 2,
    DBC = enum.auto()  # Length = 1 with DBC at x=0
    NBC = enum.auto()  # Length = 1 with NBC at x=0


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


def determine_meshtags_for_1d(msh):
    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    left_facets = locate_entities_boundary(msh, msh.topology.dim - 1, left)
    left_vals = np.full(left_facets.shape, 1, np.intc)

    right_facets = locate_entities_boundary(msh, msh.topology.dim - 1, right)
    right_vals = np.full(right_facets.shape, 2, np.intc)

    indices = np.hstack((left_facets, right_facets))
    values = np.hstack((left_vals, right_vals))

    indices, pos = np.unique(indices, return_index=True)
    return meshtags(msh, msh.topology.dim - 1, indices, values[pos])


@pytest.fixture
def system(bc_type):
    dielec = 1.2**2
    pump_profile = 1.0
    ka = 10.0
    gt = 4.0

    # Note that in the esterhazy paper the center of the ellipse is slightly shifted to
    # the right (it is centered around k=11.5)
    radius = 3.0 * gt
    vscale = 0.5 * gt / radius
    rg_params = (ka, radius, vscale)
    Print(f"RG params: {rg_params}")
    del radius
    del vscale

    double_size = bc_type == BCType.NONE
    if double_size:
        msh = mesh.create_interval(MPI.COMM_WORLD, points=(-1, 1), nx=1000)
    else:
        msh = mesh.create_unit_interval(MPI.COMM_WORLD, nx=1000)

    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    evaluator = algorithms.Evaluator(
        V,
        msh,
        # we only care about the mode intensity at the left and right
        # (double_size=True) or only at the right lead (double_size=False).
        np.array([-1.0, 1.0] if double_size else [1.0]),
    )

    ds_obc = ufl.ds
    if double_size:
        # for the double size system we only have outgoing boundary conditions (left and
        # right lead)
        bcs = []
    elif bc_type == BCType.NBC:
        # for a system with NBC we only have outgoing boundary conditions on the right
        # and NBC on the left.
        bcs = []

        marker = determine_meshtags_for_1d(msh)
        ds = ufl.Measure("ds", subdomain_data=marker, domain=msh)
        # ds(1) corresponds to the left boundary
        # ds(2) corresponds to the right boundary

        ds_obc = ds(2)  # at the right lead we impose OBC
    else:
        # Define Dirichlet boundary condition on the left
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


@pytest.mark.parametrize("bc_type", [BCType.DBC])
@pytest.mark.parametrize("first_threshold", [True, False])
def test_eval_traj(bc_type, system, first_threshold):
    """Plot the eigenvalues as a function of D0."""
    refine_first_mode = True
    if first_threshold:
        # we want to study the trajectories around the first threshold
        # the first threshold is at D0=0.267 and k=11.53.

        # we don't care about the newton solver in this test

        # TODO this study/test could also be done using the NonLasingLinearProblem.
        refine_first_mode = False
        D0range = np.linspace(0.2, 0.3, 10)
    else:
        D0range = np.linspace(0.3, 0.7, 10)

    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0_constant = real_const(system.V, 1.0)

    log.info("Before first assembly")
    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx, system.bcs)
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0)
    R = assemble_form(inner(u, v) * system.ds_obc, system.bcs, diag=0.0)
    Q_form = fem.form(D0_constant * system.pump_profile * inner(u, v) * dx)
    Q = create_matrix(Q_form)
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

    if refine_first_mode:
        nlp = NonLinearProblem(
            system.V,
            system.ka,
            system.gt,
            dielec=system.dielec,
            n=system.n,
            pump=D0_constant * system.pump_profile,
            ds_obc=system.ds_obc,
        )
        newton_operators = newtils.create_multimode_solvers_and_matrices(
            nlp, max_nmodes=1
        )

    vals = []
    vals_after_refine = []
    for D0 in D0range:
        log.info(f" {D0=} ".center(80, "#"))
        D0_constant.value = D0
        assemble_form(Q_form, system.bcs, diag=0.0, mat=nevp_inputs.Q)

        modes = algorithms.get_nevp_modes(nevp_inputs)
        evals = np.asarray([mode.k for mode in modes])

        if refine_first_mode:
            mode = modes[evals.imag.argmax()]  # k ~ 11 mode
            assert 11.0 < mode.k.real < 12.0

            minfos = [
                newtils.NewtonModeInfo(
                    k=mode.k.real,
                    s=1.0,
                    re_array=mode.array.real,
                    im_array=mode.array.imag,
                    dof_at_maximum=mode.dof_at_maximum,
                )
            ]

            refined_mode = algorithms.refine_modes(
                minfos,
                mode.bcs,
                newton_operators[1].solver,
                nlp,
                newton_operators[1].A,
                newton_operators[1].L,
                newton_operators[1].delta_x,
                newton_operators[1].initial_x,
            )[0]

            assert refined_mode.converged

            # solve again the NEVP with CISS, but with a single mode in the hole burning
            # term.
            nlp.update_b_and_k_for_forms([refined_mode])
            assemble_form(
                nlp.get_Q_hbt_form(nmodes=1), system.bcs, diag=0.0, mat=nevp_inputs.Q
            )

            sht_modes = algorithms.get_nevp_modes(nevp_inputs)

            imag_evals = np.asarray([m.k.imag for m in sht_modes])
            number_of_modes_close_to_real_axis = np.sum(np.abs(imag_evals) < 1e-10)
            Print(
                "Number of modes close to real axis: "
                f"{number_of_modes_close_to_real_axis}"
            )
            assert number_of_modes_close_to_real_axis == 1

            number_of_modes_above_real_axis = np.sum(imag_evals > 1e-10)
            Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")

            sht_evals = np.asarray([m.k for m in sht_modes])
            vals_after_refine.append(
                np.vstack([D0 * np.ones(sht_evals.shape), sht_evals]).T
            )
        vals.append(np.vstack([D0 * np.ones(evals.shape), evals]).T)

    if first_threshold:
        # see caption of Fig 1 of esterhazy paper
        assert vals[-4][0, 0] == pytest.approx(0.2666667)
        modeidx = 4
        assert vals[-4][modeidx, 1].real == pytest.approx(11.533018)
        assert abs(vals[-4][modeidx, 1].imag) < 3e-4

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

        ax.grid(True)

    scatter_plot(vals, "Non-Interacting thresholds")

    # TODO spline the trajectories and then find the root

    if refine_first_mode:
        scatter_plot(
            vals_after_refine, "Thresholds when mode around k.real~11 is refined"
        )

    plt.show()


@pytest.mark.parametrize(
    "D0, bc_type",
    [
        (0.37, BCType.DBC),
        (0.38, BCType.DBC),
        (0.38, BCType.NBC),
        (0.3, BCType.DBC),
        (0.3, BCType.NONE),
        (1.0, BCType.DBC),
        (1.0, BCType.NONE),
    ],
)
def test_solve(D0, bc_type, system):
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)
    D0_constant = real_const(system.V, D0)

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

    evals = np.asarray([mode.k for mode in modes])

    nlp = NonLinearProblem(
        system.V,
        system.ka,
        system.gt,
        dielec=system.dielec,
        n=system.n,
        pump=D0_constant * system.pump_profile,
        ds_obc=system.ds_obc,
    )
    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=1)

    modeselectors = []
    if D0 == 1.0:
        if system.double_size:
            modeselectors = range(3, 18)
        else:
            # according to the esterhazy paper there are 8 ev above the
            # threshold at D0=1.0, see figure 2
            assert np.sum(evals.imag > 0) == 8
            modeselectors = range(1, 9)
    elif D0 == 0.3:
        if system.double_size:
            modeselectors = [7, 8, 9]
        else:
            modeselectors = [3, 4]
    else:
        modeselectors = np.argwhere(evals.imag > 0).flatten()

    for modesel in modeselectors:
        mode = modes[modesel]
        assert mode.k.imag > 0

        minfos = [
            newtils.NewtonModeInfo(
                k=mode.k.real,
                s=1.0,
                re_array=mode.array.real,
                im_array=mode.array.imag,
                dof_at_maximum=mode.dof_at_maximum,
            )
        ]

        refined_modes = algorithms.refine_modes(
            minfos,
            mode.bcs,
            newton_operators[1].solver,
            nlp,
            newton_operators[1].A,
            newton_operators[1].L,
            newton_operators[1].delta_x,
            newton_operators[1].initial_x,
        )
        assert len(refined_modes) == 1
        refined_mode = refined_modes[0]

        if refined_mode.converged:
            mode_values = system.evaluator(refined_mode)
            mode_intensity = abs(mode_values) ** 2
            Print(f"-> {mode_intensity=}")

        if D0 == 1.0:
            assert refined_mode.converged
            assert len(refined_mode.newton_info_df) <= 9

            # TODO add more checks
            continue

        if D0 in (0.37, 0.38):
            # solve again the NEVP with CISS, but with a single mode in the hole burning
            # term.
            nlp.update_b_and_k_for_forms([refined_mode])

            assemble_form(
                nlp.get_Q_hbt_form(nmodes=1), system.bcs, diag=0.0, mat=nevp_inputs.Q
            )

            sht_modes = algorithms.get_nevp_modes(nevp_inputs)

            imag_evals = np.asarray([m.k.imag for m in sht_modes])
            number_of_modes_close_to_real_axis = np.sum(np.abs(imag_evals) < 1e-10)
            Print(
                "Number of modes close to real axis: "
                f"{number_of_modes_close_to_real_axis}"
            )
            assert number_of_modes_close_to_real_axis == 1

            number_of_modes_above_real_axis = np.sum(imag_evals > 1e-10)
            Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")

            if D0 == 0.37:
                # at D0=0.37 (below 2nd mode turns on to lase) we only have a single
                # mode at k ~ 11.46 (refining the other modes and inserting them
                # into the SHT leads to some eigenmodes of the NEVP that are above
                # the real axis)
                if modesel == 3:
                    # k ~ 9.45
                    assert number_of_modes_above_real_axis == 1  # k=11
                elif modesel == 4:
                    # k ~ 11.46
                    # this is the mode at D0=3.7
                    assert number_of_modes_above_real_axis == 0
                elif modesel == 5:
                    # k ~ 13.6
                    assert number_of_modes_above_real_axis == 2

            elif D0 == 0.38:
                if bc_type == BCType.NBC:
                    # TODO explanation
                    if modesel == 3:
                        # k ~ 8.443957
                        assert number_of_modes_above_real_axis == 2  # k=11, k=12
                    elif modesel == 4:
                        # k ~ 10.47
                        assert number_of_modes_above_real_axis == 0
                    elif modesel == 4:
                        # k ~ 12.58
                        assert number_of_modes_above_real_axis == 1  # 10.47
                else:
                    # at D0=0.38 there are two lasermodes, i.e., there is
                    # always at least one mode above the real axis.
                    if modesel == 3:
                        # k ~ 9.45
                        assert number_of_modes_above_real_axis == 1  # k=11
                    elif modesel == 4:
                        # k ~ 11.46
                        assert number_of_modes_above_real_axis == 1
                    elif modesel == 5:
                        # k ~ 13.6
                        assert number_of_modes_above_real_axis == 2

            # TODO check that one eigenmode has a eval-real part that is close
            # to refined_mode.k
            continue

        last_newton_step = refined_mode.newton_info_df.iloc[-1]
        if system.double_size:
            if modesel == 7:
                assert last_newton_step.k0 == pytest.approx(9.455132713237349)
                assert last_newton_step.s0 == pytest.approx(0.21664958093514616)
                # TODO improve convergence
                assert last_newton_step.corrnorm < 1e-10
            elif modesel == 8:
                assert last_newton_step.k0 == pytest.approx(10.48029282556277)
                assert last_newton_step.s0 == pytest.approx(0.5112783322774705)
                # assert last_newton_step[2] < 1e-10
            elif modesel == 9:
                assert last_newton_step.k0 == pytest.approx(11.527333)
                assert last_newton_step.s0 == pytest.approx(0.49687456154420656)
                assert last_newton_step.corrnorm < 1e-10
        elif bc_type == BCType.NBC:
            # TODO add some checks
            pass
        else:
            if modesel == 4:
                assert last_newton_step.k0 == pytest.approx(11.527333)
                assert last_newton_step.s0 == pytest.approx(0.4968826337362412)
            elif modesel == 3:
                assert last_newton_step.k0 == pytest.approx(9.45513271323724)
                assert last_newton_step.s0 == pytest.approx(0.21664872927707485)
            assert last_newton_step.corrnorm < 1e-10

        assert len(refined_mode.newton_info_df) <= 6

    # fix, ax = plt.subplots()

    # ref = initial_x.getArray()[:-2]
    # modevals = abs(ref[:n] + 1j * ref[n : 2 * n]) ** 2
    # ax.plot(modevals)
    # plt.show()


@pytest.mark.parametrize(
    "D0, bc_type",
    [
        (0.56, BCType.DBC),
    ],
)
def test_multimode_solve(D0, bc_type, system):
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0_constant = real_const(system.V, D0)

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
    modes = algorithms.get_nevp_modes(
        nevp_inputs,
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
    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=2)

    multi_modes = algorithms.constant_pump_algorithm(
        modes,
        nevp_inputs,
        nlp,
        newton_operators,
        s_init=1.0,
    )

    assert len(multi_modes) == 2

    info_df = multi_modes[0].newton_info_df

    assert info_df.iloc[-1].k0 == pytest.approx(1.14844e1)
    assert info_df.iloc[-1].k1 == pytest.approx(9.42298e0)
    assert info_df.iloc[-1].s0 == pytest.approx(1.393e0, rel=1e-3)
    assert info_df.iloc[-1].s1 == pytest.approx(7.952e-1, rel=1e-3)

    assert len(info_df) == 5  # 5 iterations of the newton method

    assert info_df.iloc[-1].corrnorm < 1e-10


@pytest.mark.parametrize(
    "bc_type, D0range",
    [
        (BCType.DBC, np.linspace(0.2668, 0.37, 5)),
        (BCType.DBC, np.linspace(0.3, 0.7, 15)),
        # (BCType.DBC, [0.56, 0.57, 0.58, 0.59, 0.6]),
        (BCType.DBC, [0.56]),
        (BCType.DBC, [1.0]),
        (BCType.DBC, [0.9, 1.0, 1.1, 1.2]),
        # TODO determine threshold for NBC modes
        (BCType.NBC, np.linspace(0.2668, 0.37, 5)),
    ],
    ids=[
        "DBCsinglemodes",
        "DBCmultimodes",
        "DBCmultimodes_singleD0",
        "paper",
        "DBCaround1",
        "NBCsinglemoes",
    ],
)
def test_intensity_vs_pump_esterhazy(bc_type, D0range, system):
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    D0_constant = real_const(system.V, 1.0)

    L = assemble_form(-inner(nabla_grad(u), nabla_grad(v)) * dx, system.bcs)
    M = assemble_form(system.dielec * inner(u, v) * dx, system.bcs, diag=0.0)
    Q = assemble_form(
        D0_constant * system.pump_profile * inner(u, v) * dx, system.bcs, diag=0.0
    )
    R = assemble_form(inner(u, v) * system.ds_obc, system.bcs, diag=0.0)

    Print(
        f"(complex-valued) NEVP: {L.getSize()=},  DOF: {L.getInfo()['nz_used']}, "
        f"MEM: {L.getInfo()['memory']}"
    )

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

    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=2)

    aevals = []
    results = []  # list of (D0, intensity) tuples

    def to_const(real_value):
        return fem.Constant(system.V.mesh, complex(real_value, 0))

    for D0 in D0range:
        Print(f" {D0=} ".center(80, "#"))
        D0_constant.value = D0
        assemble_form(
            D0_constant * system.pump_profile * inner(u, v) * dx,
            system.bcs,
            diag=0.0,
            mat=nevp_inputs.Q,
        )
        modes = algorithms.get_nevp_modes(nevp_inputs)
        evals = np.asarray([mode.k for mode in modes])
        assert evals.size

        if False:
            # we only care about the mode with the highest imag value because otherwise
            # we think that there exists a 2nd mode, which then doesn't converge
            # (refine_modes)
            mode = modes[evals.imag.argmax()]

            minfos = [
                newtils.NewtonModeInfo(
                    k=mode.k.real,
                    s=1.0,
                    re_array=mode.array.real,
                    im_array=mode.array.imag,
                    dof_at_maximum=mode.dof_at_maximum,
                )
            ]

            refined_mode = algorithms.refine_modes(
                minfos,
                mode.bcs,
                newton_operators[1].solver,
                nlp,
                newton_operators[1].A,
                newton_operators[1].L,
                newton_operators[1].delta_x,
                newton_operators[1].initial_x,
            )[0]

            assert refined_mode.converged

            # solve again the NEVP with CISS, but with a single mode in the hole burning
            # term.
            nlp.update_b_and_k_for_forms([refined_mode])

            assemble_form(
                nlp.get_Q_hbt_form(nmodes=1), system.bcs, diag=0.0, mat=nevp_inputs.Q
            )

            sht_modes = algorithms.get_nevp_modes(nevp_inputs)

            imag_evals = np.asarray([m.k.imag for m in sht_modes])
            number_of_modes_close_to_real_axis = np.sum(np.abs(imag_evals) < 1e-10)
            Print(
                "Number of modes close to real axis: "
                f"{number_of_modes_close_to_real_axis}"
            )
            assert number_of_modes_close_to_real_axis == 1

            number_of_modes_above_real_axis = np.sum(imag_evals > 1e-10)
            Print(f"Number of modes above real axis: {number_of_modes_above_real_axis}")

            if number_of_modes_above_real_axis > 0:
                second_mode = sht_modes[imag_evals.argmax()]
                assert second_mode.k.imag > 1e-10

                minfos = [
                    newtils.NewtonModeInfo(
                        k=refined_mode.k,
                        s=1.0,
                        re_array=refined_mode.array.real,
                        im_array=refined_mode.array.imag,
                        dof_at_maximum=refined_mode.dof_at_maximum,
                    ),
                    newtils.NewtonModeInfo(
                        k=second_mode.k.real,
                        s=1.0,
                        re_array=second_mode.array.real,
                        im_array=second_mode.array.imag,
                        dof_at_maximum=second_mode.dof_at_maximum,
                    ),
                ]

                refined_modes = algorithms.refine_modes(
                    minfos,
                    second_mode.bcs,
                    newton_operators[2].solver,
                    nlp,
                    newton_operators[2].A,
                    newton_operators[2].L,
                    newton_operators[2].delta_x,
                    newton_operators[2].initial_x,
                    # fail_early=True,
                )

                assert len(refined_modes) == 2
                assert [m.converged for m in refined_modes]
                multi_modes = refined_modes
            else:
                multi_modes = [refined_mode]
        else:
            multi_modes = algorithms.constant_pump_algorithm(
                modes,
                nevp_inputs,
                nlp,
                newton_operators,
            )
            multi_evals = np.asarray([mode.k for mode in multi_modes])
            number_of_modes_close_to_real_axis = np.sum(
                np.abs(multi_evals.imag) < 1e-10
            )
            # if number_of_modes_close_to_real_axis > 1:
            #     breakpoint()

        for mode in multi_modes:
            mode_values = system.evaluator(mode)
            mode_intensity = abs(mode_values) ** 2
            Print(f"-> {mode_intensity=}")
            results.append((D0, mode_intensity))
            aevals.append(evals)

    fig, ax = plt.subplots()
    ax.plot(
        [D0 for (D0, _) in results],
        [intens for (_, intens) in results],
        "x",
    )
    ax.set_xlabel("Pump D0")
    ax.set_ylabel("Modal intensity at right lead")

    # TODO the threshold for NBC modes is not correct
    ax.axvline(x=0.26674748)
    ax.grid(True)

    plot_ciss_eigenvalues(
        np.concatenate(aevals), params=system.rg_params, kagt=(system.ka, system.gt)
    )

    plt.show()
