/*
    Copyright (C) 2011 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "long_extras.h"
#include "fmpz_poly.h"
#include "fmpz_poly_mat.h"

void
fmpz_poly_mat_init(fmpz_poly_mat_t A, slong rows, slong cols)
{
    slong i;

    if (rows != 0)
        A->rows = flint_malloc(rows * sizeof(fmpz_poly_struct *));
    else
        A->rows = NULL;

    A->r = rows;
    A->c = cols;

    if (rows != 0 && cols != 0)
    {
        slong num;
        int of;

        of = z_mul_checked(&num, rows, cols);

        if (of)
            flint_throw(FLINT_ERROR, "Overflow creating a %wd x %wd object\n", rows, cols);

        A->entries = flint_malloc(num * sizeof(fmpz_poly_struct));

        for (i = 0; i < rows * cols; i++)
            fmpz_poly_init(A->entries + i);

        for (i = 0; i < rows; i++)
            A->rows[i] = A->entries + i * cols;
    }
    else
    {
        A->entries = NULL;
        if (rows != 0)
        {
            for (i = 0; i < rows; i++)
                A->rows[i] = NULL;
        }
    }
}

void
fmpz_poly_mat_init_set(fmpz_poly_mat_t A, const fmpz_poly_mat_t B)
{
    fmpz_poly_mat_init(A, B->r, B->c);
    fmpz_poly_mat_set(A, B);
}
