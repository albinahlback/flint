/*
    Copyright (C) 2011 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "nmod_poly.h"
#include "nmod_poly_mat.h"

void
nmod_poly_mat_init_set(nmod_poly_mat_t A, const nmod_poly_mat_t B)
{
    nmod_poly_mat_init(A, B->r, B->c, nmod_poly_mat_modulus(B));
    nmod_poly_mat_set(A, B);
}
