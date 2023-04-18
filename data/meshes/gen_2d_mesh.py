#!/usr/bin/env python
import sys
from pathlib import Path

# The Python API is entirely defined in the `gmsh.py' module (which contains the full
# documentation of all the functions in the API):
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

geodir = Path(__file__).parent


def gen(geo_fname):
    model = gmsh.model()
    gmsh.model.add("mcav")

    geo = (geodir / geo_fname).resolve()
    assert geo.exists()

    gmsh.open(str(geo))

    # We can then generate a 2D mesh...
    model.mesh.generate(2)

    pmsh = geo.parent / (geo.stem + ".msh")

    # ... and save it to disk
    gmsh.write(str(pmsh))

    pxdmf = geo.parent / (geo.stem + f"{MPI.COMM_WORLD.rank}.xdmf")

    msh, cell_markers, facet_markers = gmshio.model_to_mesh(
        model, MPI.COMM_SELF, 0, gdim=2
    )
    msh.name = "mcav"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"

    with XDMFFile(msh.comm, pxdmf, "w") as xfile:
        xfile.write_mesh(msh)
        xfile.write_meshtags(cell_markers)
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        xfile.write_meshtags(facet_markers)


def main():
    gmsh.initialize()

    for geo in ("circle_with_pml.geo", "quarter_circle_with_pml.geo"):
        gen(geo)

        # Remember that by default, if physical groups are defined, Gmsh will export in
        # the output mesh file only those elements that belong to at least one physical
        # group. To force Gmsh to save all elements, you can use
        #
        # gmsh.option.setNumber("Mesh.SaveAll", 1)

        # By default, Gmsh saves meshes in the latest version of the Gmsh mesh file
        # format (the `MSH' format). You can save meshes in other mesh formats by
        # specifying a filename with a different extension. For example
        #
        #   gmsh.write("t1.unv")
        #
        # will save the mesh in the UNV format. You can also save the mesh in older
        # versions of the MSH format: simply set
        #
        #   gmsh.option.setNumber("Mesh.MshFileVersion", x)
        #
        # for any version number `x'. As an alternative, you can also not specify the
        # format explicitly, and just choose a filename with the `.msh2' or `.msh4'
        # extension.

        # To visualize the model we can run the graphical user interface with
        # `gmsh.fltk.run()'. Here we run it only if "-nopopup" is not provided in the
        # command line arguments:
        if "-nopopup" not in sys.argv:
            gmsh.fltk.run()

        # Note that starting with Gmsh 3.0, models can be built using other geometry
        # kernels than the default "built-in" kernel. To use the OpenCASCADE CAD kernel
        # instead of the built-in kernel, you should use the functions with the
        # `gmsh.model.occ' prefix.
        #
        # Different CAD kernels have different features. With OpenCASCADE, instead of
        # defining the surface by successively defining 4 points, 4 curves and 1 curve
        # loop, one can define the rectangular surface directly with
        #
        # gmsh.model.occ.addRectangle(.2, 0, 0, .1, .3)
        #
        # After synchronization with the Gmsh model with
        #
        # gmsh.model.occ.synchronize()
        #
        # the underlying curves and points could be accessed with
        # gmsh.model.getBoundary().
        #
        # See e.g. `t16.py', `t18.py', `t19.py' or `t20.py' for complete examples based
        # on OpenCASCADE, and `examples/api' for more.

    # This should be called when you are done using the Gmsh Python API:
    gmsh.finalize()


if __name__ == "__main__":
    main()
