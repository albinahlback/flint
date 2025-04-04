/*
    Copyright (C) 2011 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz.h"
#include "fmpz_poly.h"
#include "fmpz_poly_mat.h"

slong
fmpz_poly_mat_nullspace(fmpz_poly_mat_t res, const fmpz_poly_mat_t mat)
{
    slong i, j, k, n, rank, nullity;
    slong * pivots;
    slong * nonpivots;
    fmpz_poly_mat_t tmp;
    fmpz_poly_t den;

    n = mat->c;

    fmpz_poly_init(den);
    fmpz_poly_mat_init_set(tmp, mat);
    rank = fmpz_poly_mat_rref(tmp, den, tmp);
    nullity = n - rank;

    fmpz_poly_mat_zero(res);

    if (rank == 0)
    {
        for (i = 0; i < nullity; i++)
            fmpz_poly_one(fmpz_poly_mat_entry(res, i, i));
    }
    else if (nullity)
    {
        pivots = flint_malloc(rank * sizeof(slong));
        nonpivots = flint_malloc(nullity * sizeof(slong));

        for (i = j = k = 0; i < rank; i++)
        {
            while (fmpz_poly_is_zero(fmpz_poly_mat_entry(tmp, i, j)))
            {
                nonpivots[k] = j;
                k++;
                j++;
            }
            pivots[i] = j;
            j++;
        }
        while (k < nullity)
        {
            nonpivots[k] = j;
            k++;
            j++;
        }

        fmpz_poly_set(den, fmpz_poly_mat_entry(tmp, 0, pivots[0]));

        for (i = 0; i < nullity; i++)
        {
            for (j = 0; j < rank; j++)
                fmpz_poly_set(fmpz_poly_mat_entry(res, pivots[j], i),
                    fmpz_poly_mat_entry(tmp, j, nonpivots[i]));
            fmpz_poly_neg(fmpz_poly_mat_entry(res, nonpivots[i], i), den);
        }

        flint_free(pivots);
        flint_free(nonpivots);
    }

    fmpz_poly_clear(den);
    fmpz_poly_mat_clear(tmp);
    return nullity;
}
