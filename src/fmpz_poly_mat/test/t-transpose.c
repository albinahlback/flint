/*
    Copyright (C) 2025 Lars Göttgens

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "test_helpers.h"
#include "fmpz_poly_mat.h"

TEST_FUNCTION_START(fmpz_poly_mat_transpose, state)
{
    slong m, n, rep;

    /* Rectangular transpose */
    for (rep = 0; rep < 100 * flint_test_multiplier(); rep++)
    {
        fmpz_poly_mat_t A, B, C;

        m = n_randint(state, 20);
        n = n_randint(state, 20);

        fmpz_poly_mat_init(A, m, n);
        fmpz_poly_mat_init(B, n, m);
        fmpz_poly_mat_init(C, m, n);

        fmpz_poly_mat_randtest(A, state, 1 + n_randint(state, 10),
            1 + n_randint(state, 100));
        fmpz_poly_mat_randtest(B, state, 1 + n_randint(state, 10),
            1 + n_randint(state, 100));

        fmpz_poly_mat_transpose(B, A);
        fmpz_poly_mat_transpose(C, B);

        if (!fmpz_poly_mat_equal(C, A))
        {
            flint_printf("FAIL: C != A\n");
            fflush(stdout);
            flint_abort();
        }

        fmpz_poly_mat_clear(A);
        fmpz_poly_mat_clear(B);
        fmpz_poly_mat_clear(C);
    }

    /* Self-transpose */
    for (rep = 0; rep < 100 * flint_test_multiplier(); rep++)
    {
        fmpz_poly_mat_t A, B;

        m = n_randint(state, 20);

        fmpz_poly_mat_init(A, m, m);
        fmpz_poly_mat_init(B, m, m);

        fmpz_poly_mat_randtest(A, state, 1 + n_randint(state, 10),
            1 + n_randint(state, 100));
        fmpz_poly_mat_set(B, A);
        fmpz_poly_mat_transpose(B, B);
        fmpz_poly_mat_transpose(B, B);

        if (!fmpz_poly_mat_equal(B, A))
        {
            flint_printf("FAIL: B != A\n");
            fflush(stdout);
            flint_abort();
        }

        fmpz_poly_mat_clear(A);
        fmpz_poly_mat_clear(B);
    }

    TEST_FUNCTION_END(state);
}
