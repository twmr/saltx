# SALT + FEniCSx = saltx

saltx is an efficient FEM based solver for the equations of the
steady-state ab initio laser theory (SALT).

In contrast to existing time-domain general Maxwell-Bloch solvers, a
SALT solver is far more efficient, since the time-dependence is
removed due a multi-periodic ansatz of the laser modes. As has been
shown in (Esterhazy, 2014) for low-symmetry laser geometries a direct
solver of the SALT equations outperforms previous methods that relied
on setting up a parametrized basis of constant flux states. Such a
direct solver is implemented in saltx.

The building blocks of this direct solver are

+ A nonlinear (in the eigenvalue k) eigenvalue solver whose
  eigenvalues lie within an elliptical contour centered around the
  real-axis.
+ A Newton-Raphson method for solving a fully nonlinear (both in k as
  well as in the eigenmode, since the eigenmodes enter the equations
  non-linearly due to the spatial hole burning term) eigenvalue
  problem.

For the former, saltx uses the NEPCISS solver from the SLEPc package
and for the latter, a PETSc KSP solver is used. Furthermore, this
python-package stands on the shoulders of FEniCSx, which is a
computing platform for the solution of PDEs. It is used for the
efficient assembly of all the FEM matrices.

It should be noted that FEniCSx has a very-helpful QA forum at
https://fenicsproject.discourse.group/.

## Modeling of lasers

In order to model a laser, the following parameters need to be known

* geometry (1D, 2D or 3D)
* dielectric function (real- or complex-valued)
* dielectric loss
* pump profile
* pump strength
* gain medium parameters ($k_a$, $\gamma_\bot$)

These inputs can be used as an input to the constant pump algorithm,
which returns all laser modes (if any) for the specified pump
strength.

saltx is tested against some reference (1D and 2D) systems from
various publications to verify that the code works as expected.

So far only one and two dimensional systems are supported, but it
shouldn't be a lot of work to add support for full 3D vectorial
systems. Support for them is planned for one of the next releases.

## License

saltx is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

saltx is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public
License along with saltx. If not, see http://www.gnu.org/licenses/.

## Attribution

If you think that this software is useful for your work, please
consider citing

```bibtex
@misc{saltx,
  title = {saltx: An efficient FEM-based laser mode solver}
  author = {Hisch, Thomas},
  year = {2023},
  note = {https://github.com/thisch/saltx},

}
@article{PhysRevA.90.023816,
  title = {Scalable numerical approach for the steady-state ab initio laser theory},
  author = {Esterhazy, S. and Liu, D. and Liertzer, M. and Cerjan, A. and Ge, L. and Makris, K. G. and Stone, A. D. and Melenk, J. M. and Johnson, S. G. and Rotter, S.},
  journal = {Phys. Rev. A},
  volume = {90},
  issue = {2},
  pages = {023816},
  numpages = {15},
  year = {2014},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.90.023816},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.90.023816}
}
```

## Installation

It is recommended to install the python package from this git
repository into a conda-environment, whose packages are listed in the
`ci/requirements/py310.yml`.

## Usage

See the unit-tests in `tests/reference/`.
