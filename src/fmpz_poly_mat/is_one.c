/*
    Copyright (C) 2011 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz_poly.h"
#include "fmpz_poly_mat.h"

int
fmpz_poly_mat_is_one(const fmpz_poly_mat_t A)
{
    slong i, j;

    if (A->r == 0 || A->c == 0)
        return 1;

    for (i = 0; i < A->r; i++)
    {
        for (j = 0; j < A->c; j++)
        {
            if (i == j)
            {
                if (!fmpz_poly_is_one(fmpz_poly_mat_entry(A, i, j)))
                    return 0;
            }
            else
            {
                if (!fmpz_poly_is_zero(fmpz_poly_mat_entry(A, i, j)))
                    return 0;
            }
        }
    }

    return 1;
}
