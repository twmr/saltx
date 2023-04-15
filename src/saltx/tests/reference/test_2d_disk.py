# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Calculate the NEVP modes (fixed pump) of a micro-disk laser and use analytic
results to check that their numerical eigenvalues are accurate.

See https://arxiv.org/abs/1505.07691 for a good overview about the analytical treatment
of uniformly pumped micro-disk lasers.

A cylindrically symmetric micro-disk laser has the following eigenmodes:

.. code-block:: python

    X, Y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    R = abs(X+1j*Y)
    PHI = np.angle(X+1j*Y)
    # this expression is only valid inside the disk R < 1.0
    # IIRC when using exp, in the expansion we need to sum from l=-inf to +inf.

    # for quantum_l = 0 we only have
    scipy.special.jv(quantum_l, nc_eff * k_eval * R)*np.cos(quantum_l*PHI),
    # for quantum_l > 0 we also have
    scipy.special.jv(quantum_l, nc_eff * k_eval * R)*np.sin(quantum_l*PHI),

    # The only difference between the two degenerate modes is their angular component.

    # Linear combinations of the two degenerate eigenmodes are also eigenmodes of the
    # laser.

In addition to the (angular) quantum number l, there is also a radial quantum number j,
whose value corresponds to the number of lobes (in radial direction) of the mode.

Note that it is possible that j > l, because their always exists an infinite number of
pairs (l, j) for a given l.
"""
import enum
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from saltx import algorithms
from saltx.plot import plot_ciss_eigenvalues, plot_meshfunctions
from saltx.pml import RectPML
from ufl import curl, dx, elem_mult, inner

repo_dir = Path(__file__).parent.parent.parent.parent.parent

Print = PETSc.Sys.Print


class BCType(enum.Enum):
    FULL_DBC = enum.auto()
    FULL_NBC = enum.auto()
    MIXED = enum.auto()


# All Eigenvalue (including the ones below the real axis)
# of system with n=1.2
all_slepc_evals_n1p2 = [
    (BCType.FULL_DBC, (4.99741165299048 - 1.0044583704511423j)),
    (BCType.FULL_NBC, (4.997557044803381 - 1.004263615716161j)),
    (BCType.FULL_NBC, (6.0093852317180065 - 0.9488384819521537j)),
    (BCType.FULL_DBC, (6.009489704240313 - 0.9487713694555954j)),
    (BCType.MIXED, (6.020710826471189 - 0.9194130702602626j)),
    (BCType.FULL_NBC, (6.267996749411585 - 0.9175441952529688j)),
    (BCType.FULL_DBC, (7.029647098342176 - 0.7720737546607993j)),
    (BCType.FULL_NBC, (7.029911340047983 - 0.7714859842888492j)),
    (BCType.MIXED, (7.180012770597796 - 0.7967691275173563j)),
    (BCType.MIXED, (7.591398455940238 - 0.7172189181270842j)),
    (BCType.MIXED, (7.971500420143529 - 0.5701212430423237j)),
    (BCType.FULL_DBC, (8.22747087857712 - 0.5922364689786475j)),
    (BCType.FULL_NBC, (8.227511312804886 - 0.592343402081493j)),
    (BCType.FULL_DBC, (8.714261161566204 - 0.49819363361713737j)),
    (BCType.FULL_NBC, (8.71557348648514 - 0.4973168923279837j)),
    (BCType.FULL_DBC, (8.825460778058192 - 0.373799294621448j)),
    (BCType.FULL_NBC, (8.826180875627271 - 0.37387294818081424j)),
    (BCType.FULL_NBC, (8.857993170692545 - 0.47165873324861673j)),
    (BCType.MIXED, (9.153851715728974 - 0.41663584682135507j)),
    (BCType.MIXED, (9.623062686788947 - 0.22390392692080885j)),
    (BCType.MIXED, (9.702574399706574 - 0.3507519113823908j)),
    (BCType.MIXED, (9.943256408772895 - 0.32891417665771117j)),
    (BCType.FULL_NBC, (10.012712752106157 - 0.30817215207349463j)),
    (BCType.FULL_DBC, (10.012753316308082 - 0.3081128065043159j)),
    (BCType.FULL_NBC, (10.396691842423195 - 0.1285736106684406j)),
    (BCType.FULL_DBC, (10.397082934670337 - 0.1279988523277493j)),
    (BCType.FULL_DBC, (10.630835000529414 - 0.29131335782863754j)),
    (BCType.FULL_NBC, (10.63270417874674 - 0.29073092773422493j)),
    (BCType.MIXED, (10.847968559075895 - 0.26673570258826634j)),
    (BCType.FULL_NBC, (10.959443379194026 - 0.2914389947600466j)),
    (BCType.FULL_DBC, (10.960120034558551 - 0.29279042846806974j)),
    (BCType.FULL_NBC, (11.063868807117984 - 0.2939664918547568j)),
    (BCType.MIXED, (11.170543575494996 - 0.08111076369278382j)),
    (BCType.MIXED, (11.550667485725803 - 0.3033348856681137j)),
    (BCType.FULL_NBC, (11.687857983387909 - 0.2796913933653498j)),
    (BCType.FULL_DBC, (11.688646408797014 - 0.27978905675377536j)),
    (BCType.FULL_NBC, (11.958890512099197 - 0.07111612722066239j)),
    (BCType.FULL_DBC, (11.959186466194042 - 0.07219232527693782j)),
    (BCType.MIXED, (11.967851821013568 - 0.33306662868868603j)),
    (BCType.MIXED, (12.165821937652717 - 0.34938150932669065j)),
    (BCType.FULL_DBC, (12.486805448008532 - 0.36249162907346916j)),
    (BCType.FULL_NBC, (12.488445283623381 - 0.3648349202415286j)),
    (BCType.MIXED, (12.550804023644273 - 0.32961652621115733j)),
    (BCType.MIXED, (12.77052178740847 - 0.08684102843847323j)),
    (BCType.FULL_NBC, (12.996641616624752 - 0.4161497024333433j)),
    (BCType.FULL_DBC, (12.999865008403264 - 0.4199518335712151j)),
    (BCType.FULL_NBC, (13.287417427291087 - 0.4460191519514948j)),
    (BCType.FULL_DBC, (13.288737368227505 - 0.45231016024783177j)),
    (BCType.FULL_NBC, (13.382415843235384 - 0.45919177284530427j)),
    (BCType.FULL_NBC, (13.441789767635315 - 0.3951601767526225j)),
    (BCType.FULL_DBC, (13.443069965396573 - 0.3969701043264942j)),
    (BCType.MIXED, (13.453726876351451 - 0.44618878075586504j)),
    (BCType.FULL_DBC, (13.605634654693747 - 0.1137798699494105j)),
    (BCType.FULL_NBC, (13.60739393679792 - 0.11409401423695503j)),
    (BCType.MIXED, (14.058155392927556 - 0.5142321207224017j)),
    (BCType.MIXED, (14.359460627942337 - 0.4627680321441885j)),
    (BCType.MIXED, (14.435987737358577 - 0.5513034094390287j)),
    (BCType.FULL_NBC, (14.444068666377715 - 0.5324813398083786j)),
    (BCType.FULL_DBC, (14.44703410040661 - 0.5246463681307222j)),
    (BCType.MIXED, (14.463901256915003 - 0.14315779684078686j)),
    (BCType.MIXED, (14.619163023946374 - 0.5659140146798929j)),
    (BCType.FULL_NBC, (15.135225739445096 - 0.592460455718324j)),
    (BCType.FULL_DBC, (15.139476438531492 - 0.6014764654176521j)),
    (BCType.FULL_DBC, (15.292628055096984 - 0.5221591127218277j)),
    (BCType.FULL_NBC, (15.294379576760743 - 0.5194444638862245j)),
    (BCType.FULL_NBC, (15.335894048368678 - 0.1694903602565708j)),
    (BCType.FULL_DBC, (15.337620321334967 - 0.16722695726762749j)),
    (BCType.MIXED, (15.454677169033275 - 0.5992538884840142j)),
    (BCType.FULL_NBC, (15.592203528028302 - 0.627031828165236j)),
    (BCType.FULL_DBC, (15.593411510404733 - 0.6329616269131402j)),
]


@pytest.fixture
def system():
    ka = 10.0
    gt = 4.0
    D0 = 0.17
    n = 1.2

    mshxdmf = "quadrant_with_pml0.xdmf"
    # mshxdmf = "circle_with_pml0.xdmf"

    radius = 1.5 * gt
    vscale = 0.5 * gt / radius
    rg_params = (ka, radius, vscale)
    print(f"RG params: {rg_params}")
    del radius
    del vscale

    is_quadrant = mshxdmf.startswith("quadrant")
    pxdmf = (repo_dir / "data" / "meshes" / mshxdmf).resolve()

    with XDMFFile(MPI.COMM_WORLD, pxdmf, "r") as fh:
        msh = fh.read_mesh(name="mcav")
    del fh

    V = fem.FunctionSpace(msh, ("Lagrange", 4))

    fixture_locals = locals()
    nt = namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)
    return nt


def build_single_T_nonlinear(quantum_l, k, ka, gt, D0, nc):
    # Used for solving the TLM (one needs to determine D0 s.t. det(T) is ~0 for a real
    # k.

    # this formula is copied from https://arxiv.org/pdf/1505.07691.pdf Eq 32
    # we assume that the single scatterer is active, i.e., it is pumped
    nc_eff = np.sqrt(nc**2 + gt * D0 / (k - ka + 1j * gt))
    n0 = 1.0
    r = 1.0
    arg1 = nc_eff * k * r
    arg2 = n0 * k * r

    T = np.array(
        [
            [
                scipy.special.jv(quantum_l, arg1),
                -scipy.special.hankel1(quantum_l, arg2),
            ],
            [
                nc_eff * scipy.special.jvp(quantum_l, arg1),
                -n0 * scipy.special.h1vp(quantum_l, arg2),
            ],
        ]
    )
    return np.log(np.abs(np.linalg.det(T)))


def test_ref_ellipse(system):
    # check that all eigenvalues lie within the elliptical contour
    plot_ciss_eigenvalues([k for _, k in all_slepc_evals_n1p2], system.rg_params)


def test_check_eigenvalues(system):
    # first test that the full NBC modes have the same eigenvalues as the full DBC modes

    full_dbc = [k for typ, k in all_slepc_evals_n1p2 if typ == BCType.FULL_DBC]
    full_nbc = [k for typ, k in all_slepc_evals_n1p2 if typ == BCType.FULL_NBC]

    # There are 4 more NBC modes (these modes must be l=0 modes) than DBC modes.
    assert len(full_dbc) + 4 == len(full_nbc)
    # TODO group them in pairs and make sure that they are equal

    # now test that they are indeed eigenvalues, by determining the the log(det()) of
    # the operator T for several quantum numbers.
    results = []
    for i, (m, k) in enumerate(all_slepc_evals_n1p2):
        logdet = np.asarray(
            [
                build_single_T_nonlinear(
                    quantum_l=quantum_l,
                    k=k,
                    ka=system.ka,
                    gt=system.gt,
                    D0=system.D0,
                    nc=system.n,
                )
                for quantum_l in range(0, 20)
            ]
        )
        quantum_l = logdet.argmin()

        # Check that the modes with an even l are NBC/DBC modes and the ones with an odd
        # l are the MIXED modes.
        if m in (BCType.FULL_DBC, BCType.FULL_NBC):
            assert quantum_l % 2 == 0
        else:
            assert m == BCType.MIXED
            assert quantum_l % 2 == 1

        # print(f"({m}, {k}, {quantum_l}),")
        results.append(logdet)

    df = pd.DataFrame(results).T
    df.columns = [f"k{i}" for i in range(1, len(all_slepc_evals_n1p2) + 1)]
    # the rows correspond to l

    print(df)
    print(
        "Min of each column (values < -8.0 are expected, "
        "because log(abs(det(T))) are returned:"
    )
    print(df.min())
    print(df.idxmin())
    assert df.min().max() < -7.5
    print("value counts:")
    value_counts = df.idxmin().value_counts()
    print(value_counts)

    # There are 4 NBC modes with l=0
    assert value_counts[0] == 4


def solve_nevp_wrapper(
    ka,
    gt,
    D0,
    rg_params,
    V,
    invperm,
    dielec,
    pump_profile,
    bcs_norm_constraint,
    bcs: dict[str, list],
):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    all_modes = []
    for bcs_name, local_bcs in bcs.items():
        assert isinstance(local_bcs, list)
        Print(f"------------> Now solving modes with bcs={bcs_name}")

        def assemble_form(form, diag=1.0):
            mat = fem.petsc.assemble_matrix(
                fem.form(form), bcs=local_bcs, diagonal=diag
            )
            mat.assemble()
            return mat

        L = assemble_form(-inner(elem_mult(invperm, curl(u)), curl(v)) * dx)
        M = assemble_form(dielec * inner(u, v) * dx, diag=0.0)
        Q = assemble_form(D0 * pump_profile * inner(u, v) * dx, diag=0.0)

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
            bcs_norm_constraint=bcs_norm_constraint,
        )
        all_modes.extend(
            algorithms.get_nevp_modes(
                nevp_inputs,
                bcs_name=bcs_name,
                bcs=local_bcs,
            )
        )
    evals = np.asarray([mode.k for mode in all_modes])
    return all_modes, evals


def test_solve(system):
    pml_start = 1.2
    pml_end = 1.8

    if system.is_quadrant:
        # x=0 and y=0 are the boundaries at which we want to impose different BC

        def on_outer_boundary(x):
            return np.isclose(x[0], pml_end) | np.isclose(x[1], pml_end)

        bcs_dofs_dbc = fem.locate_dofs_geometrical(
            system.V,
            lambda x: np.isclose(x[0], 0) | np.isclose(x[1], 0) | on_outer_boundary(x),
        )
        bcs_dofs_nbc = fem.locate_dofs_geometrical(
            system.V,
            # at the outer pml we impose DBC but at the symmetry axes we impose NBC.
            on_outer_boundary,
        )
        bcs_dofs_mixed = fem.locate_dofs_geometrical(
            system.V,
            # DBC at x-axis, NBC at y-axis
            lambda x: np.isclose(x[1], 0) | on_outer_boundary(x),
        )

        bcs = {
            "full_dbc": [
                fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_dbc, system.V),
            ],
            "full_nbc": [
                fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_nbc, system.V),
            ],
            "mixed": [
                fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs_mixed, system.V),
            ],
        }

    else:
        bcs_dofs = fem.locate_dofs_geometrical(
            system.V,
            lambda x: np.isclose(x[0], -pml_end)
            | np.isclose(x[0], pml_end)
            | np.isclose(x[1], -pml_end)
            | np.isclose(x[1], pml_end),
        )

        bcs = {
            "nosym": [
                fem.dirichletbc(PETSc.ScalarType(0), bcs_dofs, system.V),
            ],
        }

    X, Y = np.meshgrid(
        np.linspace(0 if system.is_quadrant else -pml_end, pml_end, 8 * 32),
        np.linspace(0 if system.is_quadrant else -pml_end, pml_end, 8 * 32),
    )
    points = np.vstack([X.flatten(), Y.flatten()])
    evaluator = algorithms.Evaluator(system.V, system.msh, points)

    bcs_norm_constraint = fem.locate_dofs_geometrical(
        system.V,
        lambda x: np.logical_and(abs(x[0] - 0.75) < 0.15, abs(x[1] - 0.75) < 0.15),
    )
    # I only want to impose the norm constraint on a single node
    # can this be done in a simpler way?
    bcs_norm_constraint = bcs_norm_constraint[:1]
    Print(f"{bcs_norm_constraint=}")

    rectpml = RectPML(pml_start=pml_start, pml_end=pml_end)

    invperm = fem.Function(fem.VectorFunctionSpace(system.msh, ("DG", 0)))
    invperm.interpolate(rectpml.invperm_eval)
    invperm = ufl.as_vector((invperm[0], invperm[1]))

    Qfs = fem.FunctionSpace(system.msh, ("DG", 0))
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
        system.n**2,
        dtype=PETSc.ScalarType,
    )

    if False:
        plot_meshfunctions(system.msh, pump_profile, dielec, invperm)
        return

    modes, _ = solve_nevp_wrapper(
        system.ka,
        system.gt,
        system.D0,
        system.rg_params,
        system.V,
        invperm,
        dielec,
        pump_profile,
        bcs_norm_constraint,
        bcs,
    )

    Print("All modes: (sorted by k.real)")
    modes = list(sorted(modes, key=lambda m: m.k.real))
    for mode in modes:
        # if mode.k.imag > 0:
        Print(f" Mode({mode.bcs_name:>10}) k={mode.k}")

    def add_pml_lines(ax):
        ax.axhline(y=pml_start, c="w")
        ax.axvline(x=pml_start, c="w")
        ax.axhline(y=-pml_start, c="w")
        ax.axvline(x=-pml_start, c="w")
        phis = np.linspace(0, 2 * np.pi, 128)
        ax.plot(np.cos(phis), np.sin(phis), "w-")
        if system.is_quadrant:
            ax.set_xlim(0, pml_end)
            ax.set_ylim(0, pml_end)

    plot_mode = False
    if plot_mode:
        for mode in modes:
            _, ax = plt.subplots()
            ax.pcolormesh(X, Y, abs(evaluator(mode).reshape(X.shape)) ** 2, vmin=0.0)
            add_pml_lines(ax)
            ax.set_title(f"{mode.k=}")

            plt.show()
