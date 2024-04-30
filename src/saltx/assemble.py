# Copyright (C) 2023 Thomas Wimmer
#
# This file is part of saltx (https://github.com/twmr/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import logging

from dolfinx import fem

from saltx.log import Timer

log = logging.getLogger(__name__)


def assemble_form(form, bcs, diag=1.0, mat=None, name=""):
    if isinstance(form, fem.forms.Form):
        log.warning(f"fem.form {name} already created")
        fform = form
    else:
        with Timer(log.info, f"assemble_form {name}"):
            fform = fem.form(form)

    suffix = ", new matrix" if mat is None else ", existing matrix"
    with Timer(log.info, f"assemble_matrix {name}{suffix}"):
        if mat is None:
            mat = fem.petsc.assemble_matrix(fform, bcs=bcs, diagonal=diag)
        else:
            mat.zeroEntries()
            fem.petsc.assemble_matrix(mat, fform, bcs=bcs, diagonal=diag)
        mat.assemble()
    return mat
