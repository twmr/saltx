# Copyright (C) 2024 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from collections import namedtuple
from fractions import Fraction

import numpy as np
import pytest
from dolfinx import fem
from petsc4py import PETSc

from saltx import algorithms
from saltx.mesh import create_combined_interval_mesh, create_dcells


@pytest.fixture()
def mohammed_system():
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

    V_DG0 = fem.functionspace(msh, ("DG", 0))
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

    V = fem.functionspace(msh, ("Lagrange", 3))

    n = V.dofmap.index_map.size_global
    bcs = []

    evaluator = algorithms.Evaluator(V, msh, np.asarray([0, 2 * t + d]))
    fine_evaluator = algorithms.Evaluator(V, msh, np.linspace(0, 2 * t + d, 512))

    fixture_locals = locals()
    return namedtuple("System", list(fixture_locals.keys()))(**fixture_locals)
