# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import sys
from fractions import Fraction

import numpy as np
import ufl
from dolfinx import mesh
from mpi4py import MPI


def create_combined_interval_mesh(xstart, domains):
    degree = 1
    cell = ufl.Cell("interval", geometric_dimension=1)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

    xcur = xstart

    merged_ints = []
    for i, (_, length, ncells) in enumerate(domains):
        next_xcur = xcur + length
        linspace = np.linspace(xcur, next_xcur, ncells)
        xcur = next_xcur
        if i:
            # remove first point
            linspace = linspace[1:]
        merged_ints.append(linspace)

    # convert it into a 2d column vec
    total_linspace = np.concatenate(merged_ints).reshape(-1, 1)

    vertices = total_linspace
    n_verts = vertices.shape[0]

    # see : https://fenicsproject.discourse.group/t/mesheditor-equivalent/7917
    vert_ids = np.arange(n_verts - 1).reshape(-1, 1)  # (n_verts-1) x 1 matrix
    cells = np.hstack([vert_ids, vert_ids + 1])

    # why do we need a domain for create_mesh and not for create_unit_interval?
    return mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, domain)

    # V = fem.FunctionSpace(msh, ("P", 1))
    # log.info(V.tabulate_dof_coordinates(), msh.geometry.dim)
    # log.info(msh.topology.index_map(msh.topology.dim).size_local)


def create_dcells(msh, xstart: Fraction, domains) -> list[np.ndarray]:
    """Domain cell creator for 1D systems."""
    xcur = xstart

    epsilon = sys.float_info.epsilon
    results = []

    for _, length, _ in domains:
        next_xcur = xcur + length
        # log.info(f"{xcur=}, {next_xcur=}")

        def finder(x: np.ndarray):
            return (xcur - epsilon <= x[0]) & (x[0] <= next_xcur + epsilon)

        cells = mesh.locate_entities(msh, msh.topology.dim, finder)
        xcur = next_xcur
        results.append(cells)

    return results
