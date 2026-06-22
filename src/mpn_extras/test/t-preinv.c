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

#define LG_MIN 0
#define LG_MAX 3
#define N_MAX (1 << LG_MAX)

typedef void (* preinv_t)(mp_ptr, mp_srcptr);
preinv_t invert[] = {nn_preinv_1, nn_preinv_2, nn_preinv_4, nn_preinv_8};

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
    slong lg;
    int result;
    mp_ptr rp, rp2, dp;
    mp_size_t n;

    rp = flint_malloc(sizeof(mp_limb_t) * N_MAX);
    rp2 = flint_malloc(sizeof(mp_limb_t) * (N_MAX + 2));
    dp = flint_malloc(sizeof(mp_limb_t) * N_MAX);

    for (ix = 0; ix < 10000000 * flint_test_multiplier(); ix++)
    {
        lg = n_randint(state, LG_MAX - LG_MIN + 1) + LG_MIN;
        n = UWORD(1) << lg;

        flint_mpn_rrandom(dp, state, n);
        dp[n - 1] |= UWORD(1) << (FLINT_BITS - 1); /* normalize */
        invert[lg](rp, dp);
        gmp_invert(rp2, dp, n);

        result = mpn_cmp(rp, rp2, n) == 0;
        if (!result)
            TEST_FUNCTION_FAIL(
                    "Reciprocal `floor(B^2 / d) - B - 1' is wrong\n"
                    "n = %wd\n"
                    "d:              %{ulong*}\n"
                    "Expected limbs: %{ulong*}\n"
                    "Got limbs:      %{ulong*}\n",
                    n, dp, n, rp, n, rp2, n);
    }

    /* test edge-cases B / 2 and B - 1 */
    for (lg = LG_MIN; lg <= LG_MAX; lg++)
    {
        n = UWORD(1) << lg;

        for (ix = 0; ix < n - 1; ix++)
            dp[ix] = UWORD(0);
        dp[n - 1] = UWORD(1) << (FLINT_BITS - 1);
        invert[lg](rp, dp);
        gmp_invert(rp2, dp, n);

        result = mpn_cmp(rp, rp2, n) == 0;
        if (!result)
            TEST_FUNCTION_FAIL(
                    "Reciprocal `floor(B^2 / (B / 2)) - B - 1' is wrong\n"
                    "n = %wd\n"
                    "d:              %{ulong*}\n"
                    "Expected limbs: %{ulong*}\n"
                    "Got limbs:      %{ulong*}\n",
                    n, dp, n, rp, n, rp2, n);

        for (ix = 0; ix < n; ix++)
            dp[ix] = ~UWORD(0);
        invert[lg](rp, dp);
        gmp_invert(rp2, dp, n);

        result = mpn_cmp(rp, rp2, n) == 0;
        if (!result)
            TEST_FUNCTION_FAIL(
                    "Reciprocal `floor(B^2 / (B - 1)) - B - 1' is wrong\n"
                    "n = %wd\n"
                    "d:              %{ulong*}\n"
                    "Expected limbs: %{ulong*}\n"
                    "Got limbs:      %{ulong*}\n",
                    n, dp, n, rp, n, rp2, n);
    }

#if 0
    /* TEMP (hand-optimization): validate nn_preinv_8b against the reference. */
    {
#define nn_preinv_tmp nn_preinv_8b
        n = 8;

        for (ix = 0; ix < 3000000 * flint_test_multiplier(); ix++)
        {
            flint_mpn_rrandom(dp, state, n);
            dp[n - 1] |= UWORD(1) << (FLINT_BITS - 1);
            nn_preinv_tmp(rp, dp);
            gmp_invert(rp2, dp, n);
            if (mpn_cmp(rp, rp2, n) != 0)
                TEST_FUNCTION_FAIL(
                        TEMPLATE_STR(nn_preinv_tmp) " reciprocal is wrong\n"
                        "d:              %{ulong*}\n"
                        "Expected limbs: %{ulong*}\n"
                        "Got limbs:      %{ulong*}\n",
                        dp, n, rp2, n, rp, n);
        }

        for (ix = 0; ix < n - 1; ix++)
            dp[ix] = UWORD(0);
        dp[n - 1] = UWORD(1) << (FLINT_BITS - 1);          /* B / 2 */
        nn_preinv_tmp(rp, dp);
        gmp_invert(rp2, dp, n);
        if (mpn_cmp(rp, rp2, n) != 0)
            TEST_FUNCTION_FAIL(
                    TEMPLATE_STR(nn_preinv_tmp) " is wrong for d = B / 2\n"
                    "Expected limbs: %{ulong*}\n"
                    "Got limbs:      %{ulong*}\n",
                    rp2, n, rp, n);

        for (ix = 0; ix < n; ix++)
            dp[ix] = ~UWORD(0);                             /* B - 1 */
        nn_preinv_tmp(rp, dp);
        gmp_invert(rp2, dp, n);
        if (mpn_cmp(rp, rp2, n) != 0)
            TEST_FUNCTION_FAIL(
                    TEMPLATE_STR(nn_preinv_tmp) " is wrong for d = B - 1\n"
                    "Expected limbs: %{ulong*}\n"
                    "Got limbs:      %{ulong*}\n",
                    rp2, n, rp, n);
#undef nn_preinv_tmp
    }
#endif

    flint_free(rp);
    flint_free(rp2);
    flint_free(dp);

    TEST_FUNCTION_END(state);
}

#undef LG_MIN
#undef LG_MAX
#undef N_MAX
