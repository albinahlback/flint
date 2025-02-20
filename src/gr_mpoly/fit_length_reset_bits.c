/*
    Copyright (C) 2020 Daniel Schultz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "mpoly.h"
#include "gr_mpoly.h"

void gr_mpoly_fit_length_reset_bits(
    gr_mpoly_t A,
    slong len,
    flint_bitcnt_t bits,
    gr_mpoly_ctx_t ctx)
{
    slong N = mpoly_words_per_exp(bits, GR_MPOLY_MCTX(ctx));
    _gr_mpoly_fit_length(&A->coeffs, &A->coeffs_alloc,
                               &A->exps, &A->exps_alloc, N, len, ctx);
    A->bits = bits;
}
