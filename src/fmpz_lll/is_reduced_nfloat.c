/*
    Copyright (C) 2014 Abhinav Baid
    Copyright (C) 2025 Albin Ahlbäck

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz.h"
#include "fmpz_mat.h"
#include "nfloat.h"
#include "gr_mat.h"
#include "fmpz_lll.h"

// FIXME
int nfloat_max(nfloat_ptr, nfloat_srcptr, nfloat_srcptr, gr_ctx_t);
int nfloat_sub_ui(nfloat_ptr, nfloat_srcptr, ulong, gr_ctx_t);
int nfloat_ui_sub(nfloat_ptr, ulong, nfloat_srcptr, gr_ctx_t);
int _nfloat_cmp_ui(nfloat_srcptr, ulong, gr_ctx_t);
int nfloat_mul_d(nfloat_ptr, nfloat_srcptr, double, gr_ctx_t);
int nfloat_sub_d(nfloat_ptr, nfloat_srcptr, double, gr_ctx_t);

int fmpz_lll_is_reduced_arf(const fmpz_mat_t B, const fmpz_lll_t fl, flint_bitcnt_t prec);

#define AA(i, j) gr_mat_entry_ptr(A, i, j, ctx)
#define BB(i, j) fmpz_mat_entry(B, i, j)
#define QQ(i, j) gr_mat_entry_ptr(Q, i, j, ctx)
#define RR(i, j) gr_mat_entry_ptr(R, i, j, ctx)
#define VV(i, j) gr_mat_entry_ptr(V, i, j, ctx)
#define WWd(i, j) gr_mat_entry_ptr(Wd, i, j, ctx)
#define WWu(i, j) gr_mat_entry_ptr(Wu, i, j, ctx)
#define BND(i, j) gr_mat_entry_ptr(bound, i, j, ctx)
#define BND2(i, j) gr_mat_entry_ptr(bound2, i, j, ctx)
#define BND3(i, j) gr_mat_entry_ptr(bound3, i, j, ctx)
#define BNDT(i, j) gr_mat_entry_ptr(boundt, i, j, ctx)
#define MM(i, j) gr_mat_entry_ptr(mm, i, j, ctx)
#define MN(i, j) gr_mat_entry_ptr(mn, i, j, ctx)
#define RM(i, j) gr_mat_entry_ptr(rm, i, j, ctx)
#define RN(i, j) gr_mat_entry_ptr(rn, i, j, ctx)
#define ABSR(i, j) gr_mat_entry_ptr(absR, i, j, ctx)

FLINT_FORCE_INLINE int _nfloat_sgn(nfloat_ptr res, gr_ctx_t ctx)
{
    slong exp = NFLOAT_EXP(res);
    ulong sgn = NFLOAT_SGNBIT(res);
    return (exp == NFLOAT_EXP_ZERO) ? 0 : (sgn << 1) - 1;
}

FLINT_FORCE_INLINE void _nfloat_neg_ip(nfloat_ptr res, gr_ctx_t ctx)
{
    NFLOAT_SGNBIT(res) ^= 1;
}

FLINT_FORCE_INLINE void _nfloat_abs_ip(nfloat_ptr res, gr_ctx_t ctx)
{
    NFLOAT_SGNBIT(res) = 0;
}

// TODO
//
// * Reduce number of variables
// * Reduce number of init and clear
// * Fuse instructions
// * Conditional moves could be optimized

// Wu, Wd: (m, n)
// WU, WD: (n, n) {Used to be Wu, Wd}
// mm, rm: (m, n) {Used to be Zu, Zd}

int
fmpz_lll_is_reduced_nfloat(const fmpz_mat_t B, const fmpz_lll_t fl,
                           flint_bitcnt_t prec)
{
    int ret;
    gr_ctx_t ctx;
    gr_mat_t A, R, V, Wu, Wd, bound;
    gr_mat_t bound2, bound3, boundt, mm, rm, mn, rn, absR;
    nfloat_ptr du, dd; // vectors
    nfloat_ptr s, norm, ti, tj, tmp;
    slong m = B->c, n = B->r;
    slong i, j, k;

    if (n <= 1)
        return 1;

    ret = nfloat_ctx_init(ctx, prec, 0);
    if (ret != GR_SUCCESS)
        return fmpz_lll_is_reduced_arf(B, fl, prec);

    du = flint_malloc(ctx->sizeof_elem * (2 * n + 5));
    dd   = GR_ENTRY(du,     n + 0, ctx->sizeof_elem);
    s    = GR_ENTRY(du, 2 * n + 0, ctx->sizeof_elem);
    norm = GR_ENTRY(du, 2 * n + 1, ctx->sizeof_elem);
    ti   = GR_ENTRY(du, 2 * n + 2, ctx->sizeof_elem);
    tj   = GR_ENTRY(du, 2 * n + 3, ctx->sizeof_elem);
    tmp  = GR_ENTRY(du, 2 * n + 4, ctx->sizeof_elem);

    nfloat_init(s, ctx);
    nfloat_init(norm, ctx);
    nfloat_init(ti, ctx);
    nfloat_init(tj, ctx);
    nfloat_init(tmp, ctx);

    gr_mat_init(A, m, n, ctx);
    gr_mat_init(R, n, n, ctx);
    gr_mat_init(V, n, n, ctx);
    ret |= gr_mat_zero(R, ctx);
    ret |= gr_mat_zero(V, ctx);

    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            nfloat_set_fmpz(AA(j, i), BB(i, j), ctx); // RNDN

    if (fl->rt == Z_BASIS)
    {
        gr_mat_t Q;

        /* NOTE: this algorithm should *not* be changed */
        gr_mat_init(Q, m, n, ctx);

        ret |= gr_mat_set(Q, A, ctx);

        for (k = 0; k < n; k++)
        {
            for (i = 0; i < k; i++)
            {
                nfloat_zero(s, ctx);
                for (j = 0; j < m; j++)
                    nfloat_addmul(s, QQ(j, i), QQ(j, k), ctx); // RNDN
                nfloat_set(RR(i, k), s, ctx); // RNDN
                for (j = 0; j < m; j++)
                    nfloat_submul(QQ(j, k), s, QQ(j, i), ctx); // RNDN
            }
            nfloat_zero(s, ctx);
            for (j = 0; j < m; j++)
                nfloat_addmul(s, QQ(j, k), QQ(j, k), ctx); // RNDN
            nfloat_sqrt(RR(k, k), s, ctx); // RNDN
            if (!nfloat_is_zero(RR(k, k), ctx))
            {
                nfloat_inv(s, RR(k, k), ctx); // RNDN
                for (j = 0; j < m; j++)
                    nfloat_mul(QQ(j, k), QQ(j, k), s, ctx); // RNDN
            }
        }
        gr_mat_clear(Q, ctx);

        for (j = n - 1; j >= 0; j--)
        {
            nfloat_inv(VV(j, j), RR(j, j), ctx); // RNDN
            for (i = j + 1; i < n; i++)
            {
                for (k = j + 1; k < n; k++)
                    nfloat_addmul(VV(j, i), VV(j, k), VV(k, i), ctx); // RNDN
                _nfloat_neg_ip(VV(j, i), ctx);
                nfloat_mul(VV(j, i), VV(j, j), VV(j, i), ctx); // RNDN
            }
        }

        gr_mat_init(Wu, n, n, ctx);
        gr_mat_init(Wd, n, n, ctx);
        _nfloat_vec_init(du, n, ctx);
        _nfloat_vec_init(dd, n, ctx);

        nfloat_mat_mul(Wd, R, V, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(dd + i, WWd(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(Wu, R, V, ctx); // RNDU
        for (i = 0; i < n; i++)
            nfloat_sub_ui(dd + i, WWu(i, i), 1, ctx); // RNDU
        nfloat_zero(norm, ctx);
        for (i = 0; i < n; i++)
        {
            nfloat_zero(s, ctx);
            for (j = 0; j < n; j++)
            {
                // NOTE: Can use cmpabs here instead
                if (i != j)
                {
                    nfloat_abs(ti, WWd(i, j), ctx);
                    nfloat_abs(tj, WWu(i, j), ctx);
                }
                else
                {
                    nfloat_abs(ti, dd + i, ctx);
                    nfloat_abs(tj, du + i, ctx);
                }
                nfloat_max(tmp, ti, tj, ctx);
                nfloat_add(s, s, tmp, ctx); // RNDU
            }
            nfloat_max(norm, norm, s, ctx);
        }
        if (_nfloat_cmp_ui(norm, 1, ctx) >= 0)
            goto fail_clear_all;

        gr_mat_init(bound, n, n, ctx);

        for (i = 0; i < n; i++)
        {
            nfloat_sub_ui(dd + i, WWd(i, i), 2, ctx); // RNDD
            nfloat_sub_ui(du + i, WWu(i, i), 2, ctx); // RNDU
        }
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                if (j != i)
                {
                    // FIXME: Use cmpabs instead
                    nfloat_abs(ti, WWd(i, j), ctx);
                    nfloat_abs(tj, WWu(i, j), ctx);
                    nfloat_max(BND(i, j), ti, tj, ctx);
                    if (j < i)
                        continue;
                }
                else
                {
                    // FIXME: Use cmpabs instead
                    nfloat_abs(ti, dd + i, ctx);
                    nfloat_abs(tj, du + i, ctx);
                    nfloat_max(BND(i, j), ti, tj, ctx);
                }
                nfloat_sqr(ti, norm, ctx); // RNDU
                nfloat_ui_sub(tj, 1, norm, ctx); // RNDU
                nfloat_div(tmp, ti, tj, ctx); // RNDU
                nfloat_add(BND(i, j), BND(i, j), tmp, ctx); // RNDU
            }

        gr_mat_init(mm, n, n, ctx);
        gr_mat_init(mn, n, n, ctx);
        gr_mat_init(rm, n, n, ctx);
        gr_mat_init(rn, n, n, ctx);
        gr_mat_init(bound2, n, n, ctx);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                // HERE
                nfloat_add(tmp, WWu(i, j), WWd(i, j), ctx); // RNDU
                nfloat_mul_2exp_si(MM(j, i), tmp, -2, ctx);
                nfloat_sub(RM(j, i), MM(j, i), WWd(i, j), ctx); // RNDU
                nfloat_mul_2exp_si(MN(i, j), tmp, -2, ctx);
                nfloat_sub(RN(i, j), MN(i, j), WWd(i, j), ctx); // RNDU
            }
        nfloat_mat_mul(Wd, mm, mn, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(WWd(i, i), WWd(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(Wu, mm, mn, ctx); // RNDU
        for (i = 0; i < n; i++)
        {
            nfloat_sub_ui(WWu(i, i), WWu(i, i), 1, ctx); // RNDU
            for (j = 0; j < n; j++)
            {
                // FIXME: Use cmpabs instead
                nfloat_abs(ti, WWd(i, j), ctx);
                nfloat_abs(tj, WWu(i, j), ctx);
                nfloat_max(WWu(i, j), ti, tj, ctx);
                _nfloat_abs_ip(MM(i, j), ctx);
                _nfloat_abs_ip(MN(i, j), ctx);
            }
        }
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND2(i, j), MN(i, j), RN(i, j), ctx); // RNDU
        nfloat_mat_mul(bound2, rm, bound2, ctx); // RNDU
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND2(i, j), BND2(i, j), WWu(i, j), ctx); // RNDU
        nfloat_mat_mul(Wu, mm, rn, ctx); // RNDU
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND2(i, j), BND2(i, j), WWu(i, j), ctx); // RNDU

        gr_mat_clear(Wu, ctx);
        gr_mat_clear(Wd, ctx);
        gr_mat_clear(mm, ctx);
        gr_mat_clear(mn, ctx);
        gr_mat_clear(rm, ctx);
        gr_mat_clear(rn, ctx);

        gr_mat_init(Wu, m, n, ctx);
        gr_mat_init(Wd, m, n, ctx);
        gr_mat_init(mm, n, m, ctx);
        gr_mat_init(mn, m, n, ctx);
        gr_mat_init(rm, n, m, ctx);
        gr_mat_init(rn, m, n, ctx);

        nfloat_mat_mul(Wd, A, V, ctx); // RNDD
        nfloat_mat_mul(Wu, A, V, ctx); // RNDU

        gr_mat_clear(A, ctx);
        gr_mat_clear(V, ctx);

        gr_mat_init(bound3, n, n, ctx);

        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
            {
                nfloat_add(tmp, WWu(i, j), WWd(i, j), ctx); // RNDU
                nfloat_mul_2exp_si(MM(j, i), tmp, -2, ctx);
                nfloat_sub(RM(j, i), MM(j, i), WWd(i, j), ctx); // RNDU
                nfloat_mul_2exp_si(MN(i, j), tmp, -2, ctx);
                nfloat_sub(RN(i, j), MN(i, j), WWd(i, j), ctx); // RNDU
            }

        gr_mat_clear(Wd, ctx);
        gr_mat_clear(Wu, ctx);

        gr_mat_init(Wd, n, n, ctx);
        gr_mat_init(Wu, n, n, ctx);

        nfloat_mat_mul(Wd, mm, mn, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(WWd(i, i), WWd(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(Wu, mm, mn, ctx); // RNDU
        for (i = 0; i < n; i++)
        {
            nfloat_sub_ui(WWu(i, i), WWu(i, i), 1, ctx); // RNDU
            for (j = 0; j < m; j++)
            {
                if (j < n)
                {
                    nfloat_abs(ti, WWd(i, j), ctx);
                    nfloat_abs(tj, WWu(i, j), ctx);
                    nfloat_max(WWu(i, j), ti, tj, ctx);
                }
                nfloat_abs(MM(i, j), MM(i, j), ctx);
            }
        }

        gr_mat_clear(Wd, ctx);
        gr_mat_init(Wd, m, n, ctx);

        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
            {
                nfloat_abs(MN(i, j), MN(i, j), ctx);
                nfloat_add(WWd(i, j), MN(i, j), RN(i, j), ctx); // RNDU
            }
        nfloat_mat_mul(bound3, rm, Wd, ctx); // RNDU
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND3(i, j), BND3(i, j), WWu(i, j), ctx); // RNDU
        nfloat_mat_mul(Wu, mm, rn, ctx); // RNDU
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND3(i, j), BND3(i, j), WWu(i, j), ctx); // RNDU

        gr_mat_clear(Wu, ctx);
        gr_mat_clear(Wd, ctx);
        gr_mat_clear(mm, ctx);
        gr_mat_clear(mn, ctx);
        gr_mat_clear(rm, ctx);
        gr_mat_clear(rn, ctx);

        gr_mat_init(boundt, n, n, ctx);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                nfloat_set(BNDT(j, i), BND(i, j), ctx);
                nfloat_set(ti, BND2(i, j), ctx);
                nfloat_set(tj, BND3(i, j), ctx);
                nfloat_add(BND2(i, j), ti, tj, ctx); // RNDU
            }
        nfloat_mat_mul(bound, bound2, bound, ctx); // RNDU
        nfloat_mat_mul(bound, boundt, bound, ctx); // RNDU

        gr_mat_clear(bound2, ctx);
        gr_mat_clear(bound3, ctx);
        gr_mat_clear(boundt, ctx);

        nfloat_zero(norm, ctx);
        for (i = 0; i < n; i++)
        {
            nfloat_zero(s, ctx);
            for (j = 0; j < n; j++)
            {
                nfloat_abs(tmp, BND(i, j), ctx);
                nfloat_add(s, s, tmp, ctx); // RNDU
            }
            nfloat_max(norm, norm, s, ctx); // RNDU
        }
        if (_nfloat_cmp_ui(norm, 1, ctx) >= 0)
            goto fail_clear_R_bound_bla;

        gr_mat_init(absR, n, n, ctx);
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                if (j >= i)
                {
                    nfloat_sqr(ti, norm, ctx); // RNDU
                    nfloat_ui_sub(tj, 1, norm, ctx); // RNDU
                    nfloat_div(tmp, ti, tj, ctx); // RNDU
                    nfloat_add(BND(i, j), BND(i, j), tmp, ctx); // RNDU
                }
                else
                    nfloat_zero(BND(i, j), ctx);
                nfloat_abs(ABSR(i, j), RR(i, j), ctx);
            }
        nfloat_mat_mul(bound, bound, absR, ctx); // RNDU

        gr_mat_clear(absR, ctx);

        for (i = 0; i < n - 1; i++)
        {
            nfloat_sub(tmp, RR(i, i), BND(i, i), ctx); // RNDD
            nfloat_mul_d(ti, tmp, fl->eta, ctx); // RNDD
            for (j = i + 1; j < n; j++)
            {
                nfloat_abs(tmp, RR(i, j), ctx);
                nfloat_add(tj, tmp, BND(i, j), ctx); // RNDU
                if (_nfloat_cmp(tj, ti, ctx) > 0)
                    goto fail_clear_R_bound_bla;
            }
            nfloat_add(ti, RR(i, i), BND(i, i), ctx); // RNDU
            nfloat_sub(tj, RR(i + 1, i + 1), BND(i + 1, i + 1), ctx); // RNDD
            nfloat_abs(tmp, RR(i, i + 1), ctx);
            nfloat_sub(norm, tmp, BND(i, i + 1), ctx); // RNDD
            nfloat_div(tmp, norm, ti, ctx); // RNDD
            nfloat_sqr(norm, tmp, ctx); // RNDD
            nfloat_sub_d(s, norm, fl->delta, ctx); // RNDD
            nfloat_neg(s, s, ctx);
            nfloat_sqrt(tmp, s, ctx); // RNDU
            nfloat_mul(s, tmp, ti, ctx); // RNDU
            if (_nfloat_cmp(s, tj, ctx) > 0)
                goto fail_clear_R_bound_bla;
        }

        gr_mat_clear(R, ctx);
        gr_mat_clear(bound, ctx);
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            nfloat_set(RR(j, j), AA(j, j), ctx);
            for (i = 0; i <= j - 1; i++)
            {
                nfloat_set(RR(i, j), AA(j, i), ctx); // RNDN
                for (k = 0; k <= i - 1; k++)
                    nfloat_sub(RR(i, j), RR(k, i), RR(k, j), ctx); // RNDN
                if (!nfloat_is_zero(RR(i, i), ctx))
                {
                    nfloat_div(RR(i, j), RR(i, j), RR(i, i), ctx); // RNDN
                    nfloat_submul(RR(j, j), RR(i, j), RR(i, j), ctx); // RNDN
                }
            }

            /* going to take sqrt and then divide by it */
            if (_nfloat_sgn(RR(j, j), ctx) <= 0)
                goto fail_clear_A_R_V;

            nfloat_sqrt(RR(j, j), RR(j, j), ctx); // RNDN
        }

        for (j = n - 1; j >= 0; j--)
        {
            nfloat_inv(VV(j, j), RR(j, j), ctx); // RNDN
            for (i = j + 1; i < n; i++)
            {
                for (k = j + 1; k < n; k++)
                    nfloat_addmul(VV(j, i), VV(k, i), RR(j, k), ctx); // RNDN
                _nfloat_neg_ip(VV(j, i), ctx);
                nfloat_mul(VV(j, i), VV(j, i), VV(j, j), ctx); // RNDN
            }
        }

        gr_mat_init(Wu, n, n, ctx);
        gr_mat_init(Wd, n, n, ctx);
        du = flint_malloc(ctx->sizeof_elem * n);
        dd = flint_malloc(ctx->sizeof_elem * n);
        _nfloat_vec_init(du, n, ctx);
        _nfloat_vec_init(dd, n, ctx);

        nfloat_mat_mul(Wd, R, V, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(dd + i, WWd(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(Wu, R, V, ctx); // RNDU
        for (i = 0; i < n; i++)
            nfloat_sub_ui(dd + i, WWu(i, i), 1, ctx); // RNDU
        nfloat_zero(norm, ctx);

        for (i = 0; i < n; i++)
        {
            nfloat_zero(s, ctx);
            for (j = 0; j < n; j++)
            {
                // NOTE: Can use cmpabs here instead
                if (i != j)
                {
                    nfloat_abs(ti, WWd(i, j), ctx);
                    nfloat_abs(tj, WWu(i, j), ctx);
                }
                else
                {
                    nfloat_abs(ti, dd + i, ctx);
                    nfloat_abs(tj, du + i, ctx);
                }
                nfloat_max(tmp, ti, tj, ctx);
                nfloat_add(s, s, tmp, ctx); // RNDU
            }
            nfloat_max(norm, norm, s, ctx);
        }
        if (_nfloat_cmp_ui(norm, 1, ctx) >= 0)
        {
fail_clear_all:
            gr_mat_clear(Wu, ctx);
            gr_mat_clear(Wd, ctx);
            _nfloat_vec_clear(du, n, ctx);
            _nfloat_vec_clear(dd, n, ctx);
            nfloat_clear(s, ctx);
            nfloat_clear(norm, ctx);
            nfloat_clear(ti, ctx);
            nfloat_clear(tj, ctx);
            nfloat_clear(tmp, ctx);
            flint_free(du);
fail_clear_A_R_V:
            gr_mat_clear(A, ctx);
            gr_mat_clear(R, ctx);
            gr_mat_clear(V, ctx);
            return 0;
        }

        gr_mat_init(bound, n, n, ctx);

        for (i = 0; i < n; i++)
        {
            nfloat_sub_ui(dd + i, WWd(i, i), 2, ctx); // RNDD
            nfloat_sub_ui(du + i, WWu(i, i), 2, ctx); // RNDU
        }
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                if (j != i)
                {
                    // FIXME: Use cmpabs instead
                    nfloat_abs(ti, WWd(i, j), ctx);
                    nfloat_abs(tj, WWu(i, j), ctx);
                    nfloat_max(BND(i, j), ti, tj, ctx);
                    if (j < i)
                        continue;
                }
                else
                {
                    // FIXME: Use cmpabs instead
                    nfloat_abs(ti, dd + i, ctx);
                    nfloat_abs(tj, du + i, ctx);
                    nfloat_max(BND(i, j), ti, tj, ctx);
                }
                nfloat_sqr(ti, norm, ctx); // RNDU
                nfloat_ui_sub(tj, 1, norm, ctx); // RNDU
                nfloat_div(tmp, ti, tj, ctx); // RNDU
                nfloat_add(BND(i, j), BND(i, j), tmp, ctx); // RNDU
            }
        }

        gr_mat_init(mm, n, n, ctx);
        gr_mat_init(mn, n, n, ctx);
        gr_mat_init(rm, n, n, ctx);
        gr_mat_init(rn, n, n, ctx);
        gr_mat_init(bound2, n, n, ctx);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                nfloat_add(tmp, WWu(i, j), WWd(i, j), ctx); // RNDU
                nfloat_mul_2exp_si(MM(j, i), tmp, -2, ctx);
                nfloat_sub(RM(j, i), MM(j, i), WWd(i, j), ctx); // RNDU
                nfloat_mul_2exp_si(MN(i, j), tmp, -2, ctx);
                nfloat_sub(RN(i, j), MN(i, j), WWd(i, j), ctx); // RNDU
            }
        nfloat_mat_mul(Wd, mm, mn, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(WWd(i, i), WWd(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(Wu, mm, mn, ctx); // RNDU
        for (i = 0; i < n; i++)
        {
            nfloat_sub_ui(WWu(i, i), WWu(i, i), 1, ctx); // RNDU
            for (j = 0; j < n; j++)
            {
                // FIXME: Can do 
                nfloat_abs(ti, WWd(i, j), ctx);
                nfloat_abs(tj, WWu(i, j), ctx);
                nfloat_max(WWu(i, j), ti, tj, ctx);
                nfloat_abs(MM(i, j), MM(i, j), ctx);
                nfloat_abs(MN(i, j), MN(i, j), ctx);
            }
        }
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND2(i, j), MN(i, j), RN(i, j), ctx); // RNDU
        nfloat_mat_mul(bound2, rm, bound2, ctx); // RNDU
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND2(i, j), BND2(i, j), WWu(i, j), ctx); // RNDU
        nfloat_mat_mul(Wu, mm, rn, ctx); // RNDU
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                nfloat_add(BND2(i, j), BND2(i, j), WWu(i, j), ctx); // RNDU

        gr_mat_clear(Wu, ctx);
        gr_mat_clear(Wd, ctx);
        gr_mat_clear(mm, ctx);
        gr_mat_clear(mn, ctx);
        gr_mat_clear(rm, ctx);
        gr_mat_clear(rn, ctx);

        gr_mat_init(Wu, m, n, ctx);
        gr_mat_init(Wd, m, n, ctx);
        gr_mat_init(mm, n, m, ctx);
        gr_mat_init(mn, m, n, ctx);
        gr_mat_init(rm, n, m, ctx);
        gr_mat_init(rn, m, n, ctx);

        ret |= gr_mat_transpose(mm, V, ctx);
        nfloat_mat_mul(Wd, mm, A, ctx); // RNDD
        nfloat_mat_mul(Wu, mm, A, ctx); // RNDU

        gr_mat_clear(A, ctx);

        gr_mat_init(bound3, n, n, ctx);

        nfloat_mat_mul(mm, Wd, V, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(MM(i, i), MM(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(rm, Wd, V, ctx); // RNDU
        for (i = 0; i < n; i++)
            nfloat_sub_ui(RM(i, i), RM(i, i), 1, ctx); // RNDU

        nfloat_mat_mul(mn, Wu, V, ctx); // RNDD
        for (i = 0; i < n; i++)
            nfloat_sub_ui(MN(i, i), MN(i, i), 1, ctx); // RNDD
        nfloat_mat_mul(rn, Wu, V, ctx); // RNDU
        for (i = 0; i < n; i++)
            nfloat_sub_ui(RN(i, i), RN(i, i), 1, ctx); // RNDU

        gr_mat_clear(Wd, ctx);
        gr_mat_clear(Wu, ctx);
        gr_mat_clear(V, ctx);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                // FIXME: Use cmpabs instead
                nfloat_abs(ti, MM(i, j), ctx);
                nfloat_abs(tj, MN(i, j), ctx);
                nfloat_max(BND3(i, j), ti, tj, ctx);
                nfloat_abs(tmp, RM(i, j), ctx);
                nfloat_max(BND3(i, j), BND3(i, j), tmp, ctx);
                nfloat_abs(tmp, RN(i, j), ctx);
                nfloat_max(BND3(i, j), BND3(i, j), tmp, ctx);
            }

        gr_mat_clear(mm, ctx);
        gr_mat_clear(mn, ctx);
        gr_mat_clear(rm, ctx);
        gr_mat_clear(rn, ctx);

        gr_mat_init(boundt, n, n, ctx);

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                // FIXME: Can be optimized
                nfloat_set(BNDT(j, i), BND(i, j), ctx);
                nfloat_set(ti, BND2(i, j), ctx);
                nfloat_set(tj, BND3(i, j), ctx);
                nfloat_add(BND2(i, j), ti, tj, ctx); // RNDU
            }
        nfloat_mat_mul(bound, bound2, bound, ctx); // RNDU
        nfloat_mat_mul(bound, boundt, bound, ctx); // RNDU

        gr_mat_clear(bound2, ctx);
        gr_mat_clear(bound3, ctx);
        gr_mat_clear(boundt, ctx);

        nfloat_zero(norm, ctx);
        for (i = 0; i < n; i++)
        {
            nfloat_zero(s, ctx);
            for (j = 0; j < n; j++)
            {
                // FIXME: Can be optimized
                nfloat_abs(tmp, BND(i, j), ctx);
                nfloat_add(s, s, tmp, ctx); // RNDU
            }
            nfloat_max(norm, norm, s, ctx); // RNDU
        }
        if (_nfloat_cmp_ui(norm, 1, ctx) >= 0)
            goto fail_clear_R_bound_bla;

        gr_mat_init(absR, n, n, ctx);
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
            {
                if (j >= i)
                {
                    nfloat_sqr(ti, norm, ctx); // RNDU
                    nfloat_ui_sub(tj, 1, norm, ctx); // RNDU
                    nfloat_div(tmp, ti, tj, ctx); // RNDU
                    nfloat_add(BND(i, j), BND(i, j), tmp, ctx); // RNDU
                }
                else
                    nfloat_zero(BND(i, j), ctx);
                nfloat_abs(ABSR(i, j), RR(i, j), ctx);
            }
        nfloat_mat_mul(bound, bound, absR, ctx); // RNDU

        gr_mat_clear(absR, ctx);

        for (i = 0; i < n - 1; i++)
        {
            nfloat_sub(tmp, RR(i, i), BND(i, i), ctx); // RNDD
            nfloat_mul_d(ti, tmp, fl->eta, ctx); // RNDD
            for (j = i + 1; j < n; j++)
            {
                nfloat_abs(tmp, RR(i, j), ctx);
                nfloat_add(tj, tmp, BND(i, j), ctx); // RNDU
                if (_nfloat_cmp(tj, ti, ctx) > 0)
                    goto fail_clear_R_bound_bla;
            }
            nfloat_add(ti, RR(i, i), BND(i, i), ctx); // RNDU
            nfloat_sub(tj, RR(i + 1, i + 1), BND(i + 1, i + 1), ctx); // RNDD
            nfloat_abs(tmp, RR(i, i + 1), ctx);
            nfloat_sub(norm, tmp, BND(i, i + 1), ctx); // RNDD
            nfloat_div(tmp, norm, ti, ctx); // RNDD
            nfloat_sqr(norm, tmp, ctx); // RNDD
            nfloat_sub_d(s, norm, fl->delta, ctx); // RNDD
            nfloat_neg(s, s, ctx);
            nfloat_sqrt(tmp, s, ctx); // RNDU
            nfloat_mul(s, tmp, ti, ctx); // RNDU
            if (_nfloat_cmp(s, tj, ctx) > 0)
            {
fail_clear_R_bound_bla:
                gr_mat_clear(R, ctx);
                gr_mat_clear(bound, ctx);
                nfloat_clear(s, ctx);
                nfloat_clear(norm, ctx);
                nfloat_clear(ti, ctx);
                nfloat_clear(tj, ctx);
                nfloat_clear(tmp, ctx);
                return 0;
            }
        }

        gr_mat_clear(R, ctx);
        gr_mat_clear(bound, ctx);
    }

    _nfloat_vec_clear(du, n, ctx);
    _nfloat_vec_clear(dd, n, ctx);
    nfloat_clear(s, ctx);
    nfloat_clear(norm, ctx);
    nfloat_clear(ti, ctx);
    nfloat_clear(tj, ctx);
    nfloat_clear(tmp, ctx);
    flint_free(du);

    FLINT_ASSERT(fl->rt == Z_BASIS
                 ? fmpz_mat_is_reduced(B, fl->delta, fl->eta)
                 : fmpz_mat_is_reduced_gram(B, fl->delta, fl->eta));

    return 1;
}
