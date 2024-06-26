/*
    Copyright (C) 2019 Daniel Schultz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fq_nmod.h"
#include "n_poly.h"
#include "mpoly.h"
#include "fq_nmod_mpoly.h"

static slong _fq_nmod_mpoly_scalar_addmul_n_fq(
    ulong * Acoeffs, ulong * Aexps,
    ulong * Bcoeffs, const ulong * Bexps, slong Blen,
    ulong * Ccoeffs, const ulong * Cexps, slong Clen,
    const ulong * f,
    slong N,
    const ulong * cmpmask,
    const fq_nmod_ctx_t fqctx)
{
    slong d = fq_nmod_ctx_degree(fqctx);
    slong i = 0, j = 0, k = 0;
    ulong * tmp;
    TMP_INIT;

    TMP_START;

    tmp = (ulong *) TMP_ALLOC(d*N_FQ_MUL_ITCH*sizeof(ulong));

    while (i < Blen && j < Clen)
    {
        int cmp = mpoly_monomial_cmp(Bexps + N*i, Cexps + N*j, N, cmpmask);

        if (cmp > 0)
        {
            mpoly_monomial_set(Aexps + N*k, Bexps + N*i, N);
            _n_fq_set(Acoeffs + d*k, Bcoeffs + d*i, d);
            i++;
            k++;
        }
        else if (cmp == 0)
        {
            mpoly_monomial_set(Aexps + N*k, Bexps + N*i, N);
            _n_fq_addmul(Acoeffs + d*k, Bcoeffs + d*i, Ccoeffs + d*j, f, fqctx, tmp);
            k += !_n_fq_is_zero(Acoeffs + d*k, d);
            i++;
            j++;
        }
        else
        {
            mpoly_monomial_set(Aexps + N*k, Cexps + N*j, N);
            _n_fq_mul(Acoeffs + d*k, Ccoeffs + d*j, f, fqctx, tmp);
            j++;
            k++;
        }
    }

    while (i < Blen)
    {
        mpoly_monomial_set(Aexps + k*N, Bexps + i*N, N);
        _n_fq_set(Acoeffs + d*k, Bcoeffs + d*i, d);
        i++;
        k++;
    }

    while (j < Clen)
    {
        mpoly_monomial_set(Aexps + k*N, Cexps + j*N, N);
        _n_fq_mul(Acoeffs + d*k, Ccoeffs + d*j, f, fqctx, tmp);
        j++;
        k++;
    }

    TMP_END;

    return k;
}

void fq_nmod_mpoly_scalar_addmul_fq_nmod(
    fq_nmod_mpoly_t A,
    const fq_nmod_mpoly_t B,
    const fq_nmod_mpoly_t C,
    const fq_nmod_t e,
    const fq_nmod_mpoly_ctx_t ctx)
{
    slong d = fq_nmod_ctx_degree(ctx->fqctx);
    ulong * Bexps = B->exps, * Cexps = C->exps;
    flint_bitcnt_t Abits = FLINT_MAX(B->bits, C->bits);
    slong N = mpoly_words_per_exp(Abits, ctx->minfo);
    ulong * cmpmask;
    int freeBexps = 0, freeCexps = 0;
    ulong * f;
    TMP_INIT;

    if (fq_nmod_mpoly_is_zero(B, ctx))
    {
        fq_nmod_mpoly_scalar_mul_fq_nmod(A, C, e, ctx);
        return;
    }
    else if (fq_nmod_mpoly_is_zero(C, ctx) || fq_nmod_is_zero(e, ctx->fqctx))
    {
        fq_nmod_mpoly_set(A, B, ctx);
        return;
    }

    TMP_START;
    cmpmask = (ulong*) TMP_ALLOC(N*sizeof(ulong));
    mpoly_get_cmpmask(cmpmask, N, Abits, ctx->minfo);
    f = (ulong *) TMP_ALLOC(d*sizeof(ulong));
    n_fq_set_fq_nmod(f, e, ctx->fqctx);

    if (Abits != B->bits)
    {
        freeBexps = 1;
        Bexps = (ulong *) flint_malloc(N*B->length*sizeof(ulong));
        mpoly_repack_monomials(Bexps, Abits, B->exps, B->bits,
                                                    B->length, ctx->minfo);
    }

    if (Abits != C->bits)
    {
        freeCexps = 1;
        Cexps = (ulong *) flint_malloc(N*C->length*sizeof(ulong));
        mpoly_repack_monomials(Cexps, Abits, C->exps, C->bits,
                                                    C->length, ctx->minfo);
    }

    if (A == B || A == C)
    {
        fq_nmod_mpoly_t T;
        fq_nmod_mpoly_init(T, ctx);
        fq_nmod_mpoly_fit_length_reset_bits(T, B->length + C->length, Abits, ctx);
        T->length = _fq_nmod_mpoly_scalar_addmul_n_fq(T->coeffs, T->exps,
                                                 B->coeffs, Bexps, B->length,
                                                 C->coeffs, Cexps, C->length,
                                                    f, N, cmpmask, ctx->fqctx);
        fq_nmod_mpoly_swap(A, T, ctx);
        fq_nmod_mpoly_clear(T, ctx);
    }
    else
    {
        fq_nmod_mpoly_fit_length_reset_bits(A, B->length + C->length, Abits, ctx);
        A->length = _fq_nmod_mpoly_scalar_addmul_n_fq(A->coeffs, A->exps,
                                                 B->coeffs, Bexps, B->length,
                                                 C->coeffs, Cexps, C->length,
                                                    f, N, cmpmask, ctx->fqctx);
    }

    if (freeBexps)
        flint_free(Bexps);

    if (freeCexps)
        flint_free(Cexps);

    TMP_END;
}
