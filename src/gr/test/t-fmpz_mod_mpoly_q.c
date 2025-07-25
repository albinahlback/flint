/*
    Copyright (C) 2023 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "test_helpers.h"
#include "mpoly.h"
#include "fmpz_mod_mpoly_q.h"
#include "gr.h"

TEST_FUNCTION_START(gr_fmpz_mod_mpoly_q, state)
{
    gr_ctx_t ZZxy;
    slong iter;
    int flags = 0;

    for (iter = 0; iter < 10; iter++)
    {    
        fmpz_t m;

        fmpz_init(m);
        fmpz_randtest_unsigned(m, state, n_randint(state, 2) ? 4 : 100);
        fmpz_nextprime(m, m, 0);

        gr_ctx_init_fmpz_mod_mpoly_q(ZZxy, n_randint(state, 3), mpoly_ordering_randtest(state), m);
        ZZxy->size_limit = 100;
        gr_test_ring(ZZxy, 1000, flags);
        gr_ctx_clear(ZZxy);

        fmpz_clear(m);
    }
    TEST_FUNCTION_END(state);
}
