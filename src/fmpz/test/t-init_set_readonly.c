/*
    Copyright (C) 2010 Sebastian Pancratz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include <gmp.h>
#include "test_helpers.h"
#include "long_extras.h"
#include "fmpz.h"

TEST_FUNCTION_START(fmpz_init_set_readonly, state)
{
    int i;

    /* Create some small fmpz integers, clear the mpz_t */
    for (i = 0; i < 10000 * flint_test_multiplier(); i++)
    {
        fmpz_t f;
        mpz_t z;

        *f = z_randint(state, COEFF_MAX + 1);

        mpz_init(z);
        fmpz_get_mpz(z, f);

        {
            fmpz_t g;

            fmpz_init_set_readonly(g, z);
            fmpz_clear_readonly(g);
        }

        mpz_clear(z);
    }

    /* Create some small fmpz integers, do *not* clear the mpz_t */
    for (i = 0; i < 10000 * flint_test_multiplier(); i++)
    {
        fmpz_t f;
        mpz_t z;

        *f = z_randint(state, COEFF_MAX + 1);

        mpz_init(z);
        fmpz_get_mpz(z, f);

        {
            fmpz_t g;

            fmpz_init_set_readonly(g, z);
        }

        mpz_clear(z);
    }

    /* Create some more fmpz integers */
    for (i = 0; i < 10000 * flint_test_multiplier(); i++)
    {
        fmpz_t f;
        mpz_t z;

        fmpz_init(f);
        fmpz_randtest(f, state, 2 * FLINT_BITS);
        mpz_init(z);
        fmpz_get_mpz(z, f);

        {
            fmpz_t g, h;

            fmpz_init_set_readonly(g, z);
            fmpz_init(h);
            fmpz_set_mpz(h, z);

            if (!fmpz_equal(g, h) || !_fmpz_is_canonical(h))
                TEST_FUNCTION_FAIL(
                        "g = %{fmpz}\n"
                        "h = %{fmpz}\n"
                        "z = %{mpz}\n",
                        g, h, z);

            fmpz_clear_readonly(g);
            fmpz_clear(h);
        }

        fmpz_clear(f);
        mpz_clear(z);
    }

    TEST_FUNCTION_END(state);
}
