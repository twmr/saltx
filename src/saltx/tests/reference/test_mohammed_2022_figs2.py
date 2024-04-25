import logging
from collections import namedtuple
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pytest
import ufl
from dolfinx import fem
from petsc4py import PETSc
from ufl import dx, inner, nabla_grad

from saltx import algorithms, newtils
from saltx.assemble import assemble_form
from saltx.lasing import NonLinearProblem
from saltx.mesh import create_combined_interval_mesh, create_dcells
from saltx.tests.reference import ref_data_mohammed as rdm

log = logging.getLogger(__name__)

Print = PETSc.Sys.Print

# according to Fig S2(a) the roots of the intensity are at:
D2D1_shutoff = 0.5067
D2D1_turnon = 0.67


@pytest.fixture()
def system():
    a = 1.0
    n_cav = 3.0

    _k = 0.71
    _g = 0.2
    D0ratio_ep = (_k - _g) / (_k + _g)
    del _k, _g

    ka = 2 * np.pi / a
    gt = 0.1 * 2 * np.pi / a
    # cold cavity conduction loss
    sigma_inside_cavity = 0.5 * ka

    # thickness = Length of the cavities
    t = 5 * a / n_cav  # note that a/n_cav is the lambda inside cavity
    d = 0.2174 * a  # gap between cavities
    domains = [
        (None, Fraction(t), 100),
        (None, Fraction(d), 10),
        (None, Fraction(t), 100),
    ]
    xstart = Fraction("0.0")
    msh = create_combined_interval_mesh(xstart, domains)
    dcells = create_dcells(msh, xstart, domains)

    V_DG0 = fem.FunctionSpace(msh, ("DG", 0))
    dielec = fem.Function(V_DG0)
    sigma_c = fem.Function(V_DG0)
    invperm = fem.Function(V_DG0)
    pump_left = fem.Function(V_DG0)
    pump_right = fem.Function(V_DG0)

    def cset(func, cells, value):
        func.x.array[cells] = np.full_like(
            cells,
            value,
            dtype=PETSc.ScalarType,
        )

    cells = dcells[0]
    cset(dielec, cells, n_cav**2)
    cset(sigma_c, cells, sigma_inside_cavity)
    cset(invperm, cells, 1.0)
    cset(pump_left, cells, 1.0)
    cset(pump_right, cells, 0.0)
    cells = dcells[1]
    cset(dielec, cells, 1.0)
    cset(sigma_c, cells, 0.0)
    cset(invperm, cells, 1.0)
    cset(pump_left, cells, 0.0)
    cset(pump_right, cells, 0.0)
    cells = dcells[2]
    cset(dielec, cells, n_cav**2)
    cset(sigma_c, cells, sigma_inside_cavity)
    cset(invperm, cells, 1.0)
    cset(pump_left, cells, 0.0)
    cset(pump_right, cells, 1.0)

    radius = 0.5 * gt
    vscale = 0.5 * gt / radius
    rg_params = (ka, radius, vscale)
    del radius, vscale

    V = fem.FunctionSpace(msh, ("Lagrange", 3))

    n = V.dofmap.index_map.size_global
    bcs = []

    evaluator = algorithms.Evaluator(V, msh, np.asarray([0, 2 * t + d]))
    fine_evaluator = algorithms.Evaluator(V, msh, np.linspace(0, 2 * t + d, 512))

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)


def real_const(V, real_value: float) -> fem.Constant:
    return fem.Constant(V.mesh, complex(real_value, 0))


def calculate_mode_and_intensity(
    system, D1: int, D2: int, initial_s: float
) -> tuple[float, float, float]:
    """Calculate the first laser mode.

    Parameters
    ----------
    system
    D1
        pump strength in the left wedge
    D2
        pump strength in the right wedge
    initial_s
        initial guess of the parameter s (the amplitude) of the first mode

    Returns
    -------
    fac
        f*a/c of the mode
    intens
        Intensity of the mode
    s
        `mode.s`
    """
    u = ufl.TrialFunction(system.V)
    v = ufl.TestFunction(system.V)

    d1_constant = real_const(system.V, D1)
    d2_constant = real_const(system.V, D2)

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
    assert evals.size == 2

    nlp = NonLinearProblem(
        system.V,
        system.ka,
        system.gt,
        dielec=system.dielec,
        invperm=system.invperm,
        n=system.n,
        pump=pump_expr,
        ds_obc=ds_obc,
    )
    nlp.sigma_c = system.sigma_c

    newton_operators = newtils.create_multimode_solvers_and_matrices(nlp, max_nmodes=1)

    if evals.imag.max() < 1e-9:
        return np.nan, 0.0, 1.0

    rmode = modes[0]
    assert rmode.k.imag > 1e-9
    minfos = [
        newtils.NewtonModeInfo(
            k=rmode.k.real,
            s=initial_s,
            re_array=rmode.array.real,
            im_array=rmode.array.imag,
            dof_at_maximum=rmode.dof_at_maximum,
        )
    ]

    refined_modes = algorithms.refine_modes(
        minfos,
        system.bcs,
        newton_operators[1].solver,
        nlp,
        newton_operators[1].A,
        newton_operators[1].L,
        newton_operators[1].delta_x,
        newton_operators[1].initial_x,
        fail_early=False,
    )

    # TODO make sure that the other mode is below the threshold

    first_mode = refined_modes[0]
    fac = first_mode.k.real / (2 * np.pi)  # f*a/c

    # calculate the intensity of the mode at the two outer edges of the cavity
    intens = np.sum(abs(system.evaluator(first_mode)) ** 2)
    return fac, intens, first_mode.s


def test_single_mode_pump_trajectory_D1_0p85(system):
    fixed_D1 = 0.85
    fac, intens, s = calculate_mode_and_intensity(system, fixed_D1, 0.2 * fixed_D1, 0.1)

    # intens/2. is very close to the expected Pout value from the paper (=0.0625)
    # The factor 2 is not clear
    correction_factor = 2.0
    assert intens / correction_factor == pytest.approx(0.06260821514272946, rel=1e-4)
    assert fac == pytest.approx(1.0013936748532917, rel=1e-4)
    assert s == pytest.approx(0.33704142485274613, rel=1e-4)

    figs2a, ax1 = plt.subplots()
    ax1.plot(rdm.figs2a_data[:, 0], rdm.figs2a_data[:, 1], "x", label="paper reference")

    figs2b, ax2 = plt.subplots()
    ax2.plot(rdm.figs2b_data[:, 0], rdm.figs2b_data[:, 1], "x", label="paper reference")

    all_intensitites = []
    all_facs = []
    all_s = []
    ratios = np.linspace(0.2, 0.9, 32)
    prev_s = 0.5
    for d0ratio in ratios:
        log.info(f"################ {d0ratio=} ###################")
        fac, intens, s = calculate_mode_and_intensity(
            system, fixed_D1, d0ratio * fixed_D1, prev_s
        )
        prev_s = s
        all_intensitites.append(intens)
        all_facs.append(fac)
        all_s.append(s)

    ax1.plot(
        ratios, np.array(all_intensitites) / correction_factor, "-x", label="saltx"
    )
    ax1.grid(True)
    ax1.set_xlabel("D2/D1")
    ax1.set_ylabel("Mode Intensity")
    ax1.axvline(x=D2D1_shutoff)
    ax1.axvline(x=D2D1_turnon)
    ax1.legend()

    ax2.plot(ratios, all_facs, "-", label="saltx")
    ax1.grid(True)
    ax2.set_xlabel("D2/D1")
    ax2.set_ylabel("f*a/c")
    ax2.axvline(x=D2D1_shutoff)
    ax2.axvline(x=D2D1_turnon)
    ax2.legend()

    plt.show()


def test_single_mode_pump_trajectory_D1_0p95(system):
    fixed_D1 = 0.95
    fac, intens, s = calculate_mode_and_intensity(
        system, fixed_D1, 0.2 * fixed_D1, initial_s=0.5
    )

    # The factor 2 is not clear
    correction_factor = 2.0
    assert intens / correction_factor == pytest.approx(0.16648743285132597, rel=1e-4)
    assert fac == pytest.approx(1.0013936748532917, rel=1e-4)
    assert s == pytest.approx(0.5481240251516201, rel=1e-4)

    figs2a, ax1 = plt.subplots()
    ax1.plot(
        rdm.figs1_intensity_single_mode_data[:, 0],
        rdm.figs1_intensity_single_mode_data[:, 1],
        "x",
        label="paper reference",
    )

    figs2b, ax2 = plt.subplots()
    ax2.plot(
        rdm.figs1_freq_mode1_data[:, 0],
        rdm.figs1_freq_mode1_data[:, 1],
        "x",
        label="paper reference",
    )

    all_intensitites = []
    all_facs = []
    all_s = []
    prev_s = 0.5
    ratios = np.linspace(0.2, 0.8, 32)
    for d0ratio in ratios:
        log.info(f"################ {d0ratio=} ###################")
        fac, intens, s = calculate_mode_and_intensity(
            system, fixed_D1, d0ratio * fixed_D1, prev_s
        )
        prev_s = s
        all_intensitites.append(intens)
        all_facs.append(fac)
        all_s.append(s)

    ax1.plot(
        ratios, np.array(all_intensitites) / correction_factor, "-x", label="saltx"
    )
    ax1.grid(True)
    ax1.set_xlabel("D2/D1")
    ax1.set_ylabel("Mode Intensity")
    ax1.legend()

    ax2.plot(ratios, all_facs, "-", label="saltx")
    ax1.grid(True)
    ax2.set_xlabel("D2/D1")
    ax2.set_ylabel("f*a/c")
    ax2.legend()

    plt.show()
