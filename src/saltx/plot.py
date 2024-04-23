# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import ufl.tensors
from dolfinx.plot import create_vtk_mesh
from matplotlib.patches import Ellipse


def plot_ellipse(ax, params):
    if params is None:
        return

    center = (params[0].real, params[0].imag)
    radius = params[1]
    width = radius * 2
    height = params[2] * width

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        edgecolor="r",
        fc="None",
        lw=2,
    )
    ax.add_patch(ellipse)


def plot_ciss_eigenvalues(
    lambdas: np.ndarray,
    params: tuple | None = None,
    kagt: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots()

    ax.plot(np.real(lambdas), np.imag(lambdas), "x")

    plot_ellipse(ax, params)

    if kagt is not None:
        ka, gt = kagt
        ax.plot(ka, -gt, "ro", label="singularity")
    ax.grid(True)
    plt.show()


def plot_meshfunctions(msh, pump_profile, dielec, invperm):
    import pyvista

    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(msh, msh.topology.dim))
    # plotter = pyvista.Plotter()
    # num_local_cells = msh.topology.index_map(msh.topology.dim).size_local
    # grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
    # grid.set_active_scalars("Marker")
    # actor = plotter.add_mesh(grid, show_edges=True)
    # plotter.view_xy()
    # plotter.show()

    # plotter = pyvista.Plotter()
    # grid.cell_data[
    #     "Pump"
    # ] = pump_profile.x.array.real  # [ct.indices < num_local_cells].real
    # grid.set_active_scalars("Pump")
    # plotter.add_mesh(grid, show_edges=True)
    # plotter.view_xy()
    # plotter.show()

    grid.cell_data[
        "DielecRe"
    ] = dielec.x.array.real  # [ct.indices < num_local_cells].real
    grid.cell_data[
        "DielecIm"
    ] = dielec.x.array.imag  # [ct.indices < num_local_cells].imag

    plotter = pyvista.Plotter()
    grid.set_active_scalars("DielecRe")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()

    plotter = pyvista.Plotter()
    grid.set_active_scalars("DielecIm")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()

    if isinstance(invperm, ufl.tensors.ListTensor):
        # FIXME doesn't work
        invperm_components = [invperm[i] for i in range(invperm.ufl_shape[0])]
    else:
        invperm_components = [invperm]

    for i, component in enumerate(invperm_components):
        plotter = pyvista.Plotter()
        grid.cell_data[
            f"InvpermRe{i}"
        ] = component.x.array.real  # [ct.indices < num_local_cells].real
        grid.set_active_scalars(f"InvpermRe{i}")
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.show()

        plotter = pyvista.Plotter()
        grid.cell_data[
            f"InvpermIm{i}"
        ] = component.x.array.imag  # [ct.indices < num_local_cells].imag
        grid.set_active_scalars(f"InvpermIm{i}")
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.show()

    # cells, types, x = create_vtk_mesh(V)
    # grid = pyvista.UnstructuredGrid(cells, types, x)
    # # grid.point_data["u"] = pump.x.array.real
    # # grid.set_active_scalars("u")
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # warped = grid.warp_by_scalar()
    # plotter.add_mesh(warped)
    # plotter.show()


def plot_function(function):
    import pyvista

    topology, cell_types, geometry = create_vtk_mesh(function.function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    grid.point_data["u"] = function.x.array.real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.add_points(warped.points, color="red")
    # plotter.view_xy()


def plot_function2(msh, dcells, dfuncs):
    import pyvista

    plotter = pyvista.Plotter(window_size=[800, 800], shape=(1, 1))
    # Filter out ghosted cells
    num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
    marker = np.zeros(num_cells_local, dtype=np.int32)

    topology, cell_types, x = create_vtk_mesh(
        msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32)
    )

    marker = dfuncs.dielec.x.array.real

    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.cell_data["Marker"] = marker
    grid.set_active_scalars("Marker")
    plotter.subplot(0, 0)
    plotter.add_mesh(grid, show_edges=True)

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        plotter.screenshot("fundamentals_mesh.png")


def plot_mesh(dolfinx_mesh):
    import pyvista

    msh = dolfinx_mesh
    topology, cell_types, geometry = create_vtk_mesh(msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_points(grid.points, color="red", point_size=10)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        plotter.screenshot("fundamentals_mesh.png")
