/*
    Copyright (C) 2019 Daniel Schultz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "mpoly.h"
#include "fq_nmod_mpoly.h"

void fq_nmod_mpoly_set_term_exp_ui(fq_nmod_mpoly_t A,
                     slong i, const ulong * exp, const fq_nmod_mpoly_ctx_t ctx)
{
    slong N;
    flint_bitcnt_t exp_bits;

    if (i >= A->length)
    {
        flint_throw(FLINT_ERROR, "fq_nmod_mpoly_set_term_exp_ui: index is out of range");
    }

    exp_bits = mpoly_exp_bits_required_ui(exp, ctx->minfo);
    exp_bits = mpoly_fix_bits(exp_bits, ctx->minfo);
    fq_nmod_mpoly_fit_length_fit_bits(A, A->length, exp_bits, ctx);

    N = mpoly_words_per_exp(A->bits, ctx->minfo);
    mpoly_set_monomial_ui(A->exps + N*i, exp, A->bits, ctx->minfo);
}
