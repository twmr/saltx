# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Utilities for setting up the domain functions in the PML region."""

import sys

import numpy as np
from petsc4py import PETSc

epsilon = sys.float_info.epsilon


class RectPML:
    def __init__(
        self,
        pml_start: float,
        pml_end: float,
        alpha: complex = 2j,
    ):
        self.pml_start = pml_start
        self.pml_end = pml_end
        self.alpha = alpha

    def invperm_eval(self, x: np.ndarray):
        # x is a gdim x N matrix
        # TODO read gdim from mesh
        gdim = 2
        retval = np.ones((gdim, x.shape[1]), dtype=PETSc.ScalarType)

        cond_pml = (abs(x[0]) >= self.pml_start - epsilon) | (
            abs(x[1]) >= self.pml_start - epsilon
        )

        # retval[:, cond_x] = np.array([1, 2, 3], dtype=PETSc.ScalarType).reshape(3, 1)
        # retval[:, cond_y] = np.array([4, 5, 6], dtype=PETSc.ScalarType).reshape(3, 1)
        # retval[:, cond_xy] = np.array([7, 8, 9], dtype=PETSc.ScalarType).reshape(3, 1)
        # # p retval.shape
        # (3, 2144)
        # # p retval[0,:]
        # array([7.+0.j, 7.+0.j, 7.+0.j, ..., 7.+0.j, 7.+0.j, 7.+0.j])
        # # p retval[1,:]
        # array([8.+0.j, 8.+0.j, 8.+0.j, ..., 8.+0.j, 8.+0.j, 8.+0.j])
        # # p retval[2,:]
        # array([9.+0.j, 9.+0.j, 9.+0.j, ..., 9.+0.j, 9.+0.j, 9.+0.j])

        sz = 1
        # when cond_x we know that

        # xcoord = x[0, cond_pml]
        # sx = 1 - alpha_pml * np.maximum((abs(xcoord) - pml_start) / pml_depth, 0) ** 2
        # ycoord = x[1, cond_pml]
        # sy = 1 - alpha_pml * np.maximum((abs(ycoord) - pml_start) / pml_depth, 0) ** 2

        sx = 1 + self.alpha * np.where(
            abs(x[0, cond_pml]) >= self.pml_start - epsilon, 1, 0
        )
        sy = 1 + self.alpha * np.where(
            abs(x[1, cond_pml]) >= self.pml_start - epsilon, 1, 0
        )

        # retval[:, cond_pml] = #np.array(
        #     [sx / (sy * sz), sy / (sx * sz), sz / (sx * sy)], dtype=PETSc.ScalarType
        # ).reshape(3, 1)

        retval[0, cond_pml] = sx / (sy * sz)
        # upper_pml: sx: 1, sy: 1 + alpha, sz: 1
        #   => 1/(1+alpha) = 1/(1-alpha**2) * (1 - alpha) = 1/5 * (1-2j)
        # right_pml: sx: 1 + alpha, sy: 1, sz: 1
        #   => 1 + alpha = 1+2j
        # edge_pml : sx: 1 + alpha, sy: 1 + alpha, sz: 1
        #   => 1.

        retval[1, cond_pml] = sy / (sx * sz)
        # retval[2, cond_pml] = sz / (sx * sy)

        return retval

    def dielec_eval(self, x: np.ndarray) -> np.ndarray:
        # x is a 3 x N matrix
        retval = np.ones(x.shape[1], dtype=PETSc.ScalarType)

        cond_pml = (abs(x[0]) >= self.pml_start - epsilon) | (
            abs(x[1]) >= self.pml_start - epsilon
        )

        sx = np.where(
            abs(x[0, cond_pml]) >= self.pml_start - epsilon, 1.0 + self.alpha, 1.0
        )
        sy = np.where(
            abs(x[1, cond_pml]) >= self.pml_start - epsilon, 1 + self.alpha, 1
        )
        sz = 1

        # retval[0, cond_pml] = (sy * sz) / sx
        # retval[1, cond_pml] = (sx * sz) / sy
        # retval[2, cond_pml] = (sx * sy) / sz

        # because we only care about TM-modes, we only have a z-component of the
        # electric field.
        retval[cond_pml] = (sx * sy) / sz

        return retval
