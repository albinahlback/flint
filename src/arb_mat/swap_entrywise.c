/*
    Copyright (C) 2012 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "arb.h"
#include "arb_mat.h"

void arb_mat_swap_entrywise(arb_mat_t mat1, arb_mat_t mat2)
{
    slong i, j;

    for (i = 0; i < arb_mat_nrows(mat1); i++)
        for (j = 0; j < arb_mat_ncols(mat1); j++)
            arb_swap(arb_mat_entry(mat2, i, j), arb_mat_entry(mat1, i, j));
}