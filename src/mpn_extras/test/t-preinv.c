/*
    Copyright (C) 2026 Albin Ahlbäck

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "test_helpers.h"
#include "mpn_extras.h"

#define N_MIN 1
#define N_MAX 1

typedef void (* preinv_t)(mp_ptr, mp_srcptr);
preinv_t invert[] = {nn_preinv_1};

static inline void
gmp_invert(mp_ptr rp, mp_srcptr dp, mp_size_t n)
{
    mp_ptr np = alloca(sizeof(mp_limb_t) * (2 * n + 1));
    memset(np, 0, sizeof(mp_limb_t) * (2 * n + 1));
    np[2 * n] = 1;
    mp_ptr scratch = alloca(sizeof(mp_limb_t) * n);
    mpn_tdiv_qr(rp, scratch, 0, np, 2 * n + 1, dp, n);
    mpn_sub_1(rp, rp, n, 1);
}

TEST_FUNCTION_START(nn_preinv, state)
{
    slong ix;
    int result;
    mp_ptr rp, rp2, dp;

    rp = flint_malloc(sizeof(mp_limb_t) * N_MAX);
    rp2 = flint_malloc(sizeof(mp_limb_t) * (N_MAX + 1));
    dp = flint_malloc(sizeof(mp_limb_t) * N_MAX);

    for (ix = 0; ix < 10000 * flint_test_multiplier(); ix++)
    {
        mp_size_t n = n_randint(state, N_MAX - N_MIN + 1) + N_MIN;

        flint_mpn_rrandom(dp, state, n);
        dp[n - 1]|= UWORD(1) << (FLINT_BITS - 1); /* normalize */
        invert[n - 1](rp, dp);
        
        result = mpn_cmp(rp, rp2, n) == 0;
        if (!result)
            TEST_FUNCTION_FAIL(
                    "Reciprocal `floor(B^2 / d) - B - 1' is wrong\n"
                    "n = %wd\n"
                    "Expected limbs: %{ulong*}\n"
                    "Got limbs:      %{ulong*}\n",
                    n, rp, rp2);
    }

    flint_free(rp);
    flint_free(rp2);
    flint_free(dp);

    TEST_FUNCTION_END(state);
}

#undef N_MIN
#undef N_MAX
