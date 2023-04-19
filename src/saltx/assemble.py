# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import logging

from dolfinx import fem

from saltx.log import Timer

log = logging.getLogger(__name__)


def assemble_form(form, bcs, diag=1.0, mat=None):
    if isinstance(form, fem.forms.FormMetaClass):
        log.error("fem.form form already created")
        fform = form
    else:
        with Timer(log.error, "assemble_form"):
            fform = fem.form(form)

    if mat is None:
        mat = fem.petsc.assemble_matrix(fform, bcs=bcs, diagonal=diag)
    else:
        mat.zeroEntries()
        fem.petsc.assemble_matrix(mat, fform, bcs=bcs, diagonal=diag)
    mat.assemble()
    return mat
