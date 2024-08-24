/*
   Copyright 2024 (C) Vincent Neiger

   This file is part of FLINT.

   FLINT is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License (LGPL) as published
   by the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.  See <https://www.gnu.org/licenses/>.
   */

#include "profiler.h"
#include "ulong_extras.h"

void sample(void * arg, ulong count)
{
    nn_ptr array = (nn_ptr) flint_malloc(1000*sizeof(ulong));
    FLINT_TEST_INIT(state);

    for (ulong i = 0; i < count; i++)
    {
        ulong bits = n_randint(state, FLINT_BITS - 1) + 1;  // 1...63
        ulong d = n_randbits(state, bits);  // 0 < d < 2**(FLINT_BITS-1) required by mulmod_shoup
        ulong a = n_randlimb(state);  // a is arbitrary

        for (ulong j = 0; j < 1000; j++)
            array[j] = n_randint(state, d);  // must be < d

        prof_start();
        for (ulong j = 0; j < 1000; j++)
        {
            ulong aj, r;
            udiv_qrnnd(aj, r, array[j], UWORD(0), d);
            array[j] = n_mulmod_shoup(array[j], a, aj, d);
        }
        prof_stop();
    }

    flint_rand_clear(state);
    flint_free(array);
}

void sample_no_precomp(void * arg, ulong count)
{
    nn_ptr array = (nn_ptr) flint_malloc(1000*sizeof(ulong));
    FLINT_TEST_INIT(state);

    for (ulong i = 0; i < count; i++)
    {
        ulong bits = n_randint(state, FLINT_BITS - 1) + 1;  // 1...63
        ulong d = n_randbits(state, bits);  // 0 < d < 2**(FLINT_BITS-1) required by mulmod_shoup
        ulong a = n_randint(state, d);  // a must be < d

        for (ulong j = 0; j < 1000; j++)
            array[j] = n_randlimb(state);  // array[j] is arbitrary

        ulong apre, r;
        udiv_qrnnd(apre, r, a, UWORD(0), d);

        prof_start();
        for (ulong j = 0; j < 1000; j++)
            array[j] = n_mulmod_shoup(a, array[j], apre, d);
        prof_stop();
    }

    flint_rand_clear(state);
    flint_free(array);
}

int main(void)
{
    double min, max;

    flint_printf("mulmod_shoup min time / max time is:\n");

    prof_repeat(&min, &max, sample, NULL);
    flint_printf("   - including precomputation: %.3f cycles / %.3f cycles\n",
            (min/(double)FLINT_CLOCK_SCALE_FACTOR)/1000, (max/(double)FLINT_CLOCK_SCALE_FACTOR)/1000);

    prof_repeat(&min, &max, sample_no_precomp, NULL);
    flint_printf("   - excluding precomputation: %.3f cycles / %.3f cycles\n",
            (min/(double)FLINT_CLOCK_SCALE_FACTOR)/1000, (max/(double)FLINT_CLOCK_SCALE_FACTOR)/1000);

    return 0;
}
