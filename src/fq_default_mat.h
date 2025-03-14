/*
    Copyright (C) 2021 William Hart
    Copyright (C) 2021 Daniel Schultz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#ifndef FQ_DEFAULT_MAT_H
#define FQ_DEFAULT_MAT_H

#ifdef FQ_DEFAULT_MAT_INLINES_C
#define FQ_DEFAULT_MAT_INLINE
#else
#define FQ_DEFAULT_MAT_INLINE static inline
#endif

#include "nmod_mat.h"
#include "fmpz_mod.h"
#include "fmpz_mod_mat.h"
#include "fq_mat.h"
#include "fq_nmod_mat.h"
#include "fq_zech_mat.h"
#include "fq_default.h"
#include "gr.h"
#include "gr_mat.h"

#ifdef __cplusplus
 extern "C" {
#endif

typedef union fq_default_mat_struct
{
    fq_mat_t fq;
    fq_nmod_mat_t fq_nmod;
    fq_zech_mat_t fq_zech;
    nmod_mat_t nmod;
    fmpz_mod_mat_t fmpz_mod;
} fq_default_mat_struct;

typedef fq_default_mat_struct fq_default_mat_t[1];

/* Memory management  ********************************************************/

FQ_DEFAULT_MAT_INLINE void fq_default_mat_init(fq_default_mat_t mat,
                            slong rows, slong cols, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_init(mat->fq_zech, rows, cols, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_init(mat->fq_nmod, rows, cols, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_init(mat->nmod, rows, cols, FQ_DEFAULT_CTX_NMOD(ctx).n);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_init(mat->fmpz_mod, rows, cols, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_init(mat->fq, rows, cols, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_init_set(fq_default_mat_t mat,
                        const fq_default_mat_t src, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_init_set(mat->fq_zech, src->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_init_set(mat->fq_nmod, src->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_init_set(mat->nmod, src->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_init_set(mat->fmpz_mod, src->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_init_set(mat->fq, src->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_swap(fq_default_mat_t mat1,
                               fq_default_mat_t mat2, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_swap(mat1->fq_zech, mat2->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_swap(mat1->fq_nmod, mat2->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_swap(mat1->nmod, mat2->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_swap(mat1->fmpz_mod, mat2->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_swap(mat1->fq, mat2->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_set(fq_default_mat_t mat1,
                       const fq_default_mat_t mat2, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_set(mat1->fq_zech, mat2->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_set(mat1->fq_nmod, mat2->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_set(mat1->nmod, mat2->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_set(mat1->fmpz_mod, mat2->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_set(mat1->fq, mat2->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_clear(fq_default_mat_t mat,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_clear(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_clear(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_clear(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_clear(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_clear(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE int fq_default_mat_equal(const fq_default_mat_t mat1,
                       const fq_default_mat_t mat2, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_equal(mat1->fq_zech, mat2->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_equal(mat1->fq_nmod, mat2->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_equal(mat1->nmod, mat2->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_equal(mat1->fmpz_mod, mat2->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_equal(mat1->fq, mat2->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE int fq_default_mat_is_zero(const fq_default_mat_t mat,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_is_zero(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_is_zero(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_is_zero(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_is_zero(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_is_zero(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE int fq_default_mat_is_one(const fq_default_mat_t mat,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_is_one(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_is_one(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_is_one(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_is_one(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_is_one(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE int
fq_default_mat_is_empty(const fq_default_mat_t mat, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_is_empty(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_is_empty(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_is_empty(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_is_empty(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_is_empty(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE int
fq_default_mat_is_square(const fq_default_mat_t mat,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_is_square(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_is_square(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_is_square(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_is_square(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_is_square(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_transpose(fq_default_mat_t B, const fq_default_mat_t A, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_transpose(B->nmod, A->nmod);
    }
    else
    {
        GR_MUST_SUCCEED(gr_mat_transpose((gr_mat_struct *) B, (const gr_mat_struct *) A, FQ_DEFAULT_GR_CTX(ctx)));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_entry(fq_default_t val, const fq_default_mat_t mat,
		                  slong i, slong j, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_set(val->fq_zech,
                        fq_zech_mat_entry(mat->fq_zech, i, j), FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_set(val->fq_nmod,
                        fq_nmod_mat_entry(mat->fq_nmod, i, j), FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        val->nmod = nmod_mat_entry(mat->nmod, i, j);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_set(val->fmpz_mod, fmpz_mod_mat_entry(mat->fmpz_mod, i, j));
    }
    else
    {
        fq_set(val->fq, fq_mat_entry(mat->fq, i, j), FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_entry_set(fq_default_mat_t mat, slong i, slong j,
                              const fq_default_t x, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_entry_set(mat->fq_zech, i, j, x->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_entry_set(mat->fq_nmod, i, j, x->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_entry(mat->nmod, i, j) = x->nmod;
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_set(fmpz_mod_mat_entry(mat->fmpz_mod, i, j), x->fmpz_mod);
    }
    else
    {
        fq_mat_entry_set(mat->fq, i, j, x->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_entry_set_fmpz(fq_default_mat_t mat, slong i, slong j,
                                    const fmpz_t x, const fq_default_ctx_t ctx)
{
   fq_default_t c;
   fq_default_init(c, ctx);
   fq_default_set_fmpz(c, x, ctx);
   fq_default_mat_entry_set(mat, i, j, c, ctx);
   fq_default_clear(c, ctx);
}

FQ_DEFAULT_MAT_INLINE slong
fq_default_mat_nrows(const fq_default_mat_t mat, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_nrows(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_nrows(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_nrows(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_nrows(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_nrows(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE slong
fq_default_mat_ncols(const fq_default_mat_t mat, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_ncols(mat->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_ncols(mat->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_ncols(mat->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_ncols(mat->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_ncols(mat->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_swap_rows(fq_default_mat_t mat,
                    slong * perm, slong r, slong s, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_swap_rows(mat->fq_zech, perm, r, s, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_swap_rows(mat->fq_nmod, perm, r, s, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_swap_rows(mat->nmod, perm, r, s);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_swap_rows(mat->fmpz_mod, perm, r, s, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_swap_rows(mat->fq, perm, r, s, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_invert_rows(fq_default_mat_t mat,
                                      slong * perm, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_invert_rows(mat->fq_zech, perm, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_invert_rows(mat->fq_nmod, perm, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_invert_rows(mat->nmod, perm);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_invert_rows(mat->fmpz_mod, perm, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_invert_rows(mat->fq, perm, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_swap_cols(fq_default_mat_t mat,
                    slong * perm, slong r, slong s, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_swap_cols(mat->fq_zech, perm, r, s, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_swap_cols(mat->fq_nmod, perm, r, s, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_swap_cols(mat->nmod, perm, r, s);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_swap_cols(mat->fmpz_mod, perm, r, s, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_swap_cols(mat->fq, perm, r, s, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void
fq_default_mat_invert_cols(fq_default_mat_t mat,
                                      slong * perm, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_invert_cols(mat->fq_zech, perm, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_invert_cols(mat->fq_nmod, perm, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_invert_cols(mat->nmod, perm);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_invert_cols(mat->fmpz_mod, perm, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_invert_cols(mat->fq, perm, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Assignment  ***************************************************************/

FQ_DEFAULT_MAT_INLINE void fq_default_mat_zero(fq_default_mat_t A,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_zero(A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_zero(A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_zero(A->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_zero(A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_zero(A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_one(fq_default_mat_t A,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_one(A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_one(A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_one(A->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_one(A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_one(A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Conversions ***************************************************************/

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_set_nmod_mat(fq_default_mat_t mat1,
		             const nmod_mat_t mat2, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_set_nmod_mat(mat1->fq_zech, mat2, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_set_nmod_mat(mat1->fq_nmod, mat2, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_set(mat1->nmod, mat2);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        fmpz_mod_mat_set_nmod_mat(mat1->fmpz_mod, mat2, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_set_nmod_mat(mat1->fq, mat2, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_set_fmpz_mod_mat(fq_default_mat_t mat1,
                         const fmpz_mod_mat_t mat2, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_set_fmpz_mod_mat(mat1->fq_zech, mat2, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_set_fmpz_mod_mat(mat1->fq_nmod, mat2, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        fmpz_mat_get_nmod_mat(mat1->nmod, mat2);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_set(mat1->fmpz_mod, mat2, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_set_fmpz_mod_mat(mat1->fq, mat2, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_set_fmpz_mat(fq_default_mat_t mat1,
		             const fmpz_mat_t mat2, const fq_default_ctx_t ctx)
{
    fmpz_mod_mat_t mod_mat;

    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fmpz_mod_ctx_t ctx2;
        fmpz_t prime;
        fmpz_init_set_ui(prime, fq_zech_ctx_prime(FQ_DEFAULT_CTX_FQ_ZECH(ctx)));
        fmpz_mod_ctx_init(ctx2, prime);
        fmpz_mod_mat_init(mod_mat, mat2->r, mat2->c, ctx2);
        fmpz_mod_mat_set_fmpz_mat(mod_mat, mat2, ctx2);
        fq_zech_mat_set_fmpz_mod_mat(mat1->fq_zech, mod_mat, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
        fmpz_mod_mat_clear(mod_mat, ctx2);
        fmpz_mod_ctx_clear(ctx2);
        fmpz_clear(prime);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {

        fmpz_mod_ctx_t ctx2;
        fmpz_t prime;
        fmpz_init_set_ui(prime, fq_nmod_ctx_prime(FQ_DEFAULT_CTX_FQ_NMOD(ctx)));
        fmpz_mod_ctx_init(ctx2, prime);
        fmpz_mod_mat_init(mod_mat, mat2->r, mat2->c, ctx2);
        fmpz_mod_mat_set_fmpz_mat(mod_mat, mat2, ctx2);
        fq_nmod_mat_set_fmpz_mod_mat(mat1->fq_nmod, mod_mat, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
        fmpz_mod_mat_clear(mod_mat, ctx2);
        fmpz_mod_ctx_clear(ctx2);
        fmpz_clear(prime);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        fmpz_mat_get_nmod_mat(mat1->nmod, mat2);
        return;
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_set_fmpz_mat(mat1->fmpz_mod, mat2, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
        return;
    }
    else
    {
        fmpz_mod_ctx_t ctx2;
        fmpz_mod_ctx_init(ctx2, fq_ctx_prime(FQ_DEFAULT_CTX_FQ(ctx)));
        fmpz_mod_mat_init(mod_mat, mat2->r, mat2->c, ctx2);
        fmpz_mod_mat_set_fmpz_mat(mod_mat, mat2, ctx2);
        fq_mat_set_fmpz_mod_mat(mat1->fq, mod_mat, FQ_DEFAULT_CTX_FQ(ctx));
        fmpz_mod_mat_clear(mod_mat, ctx2);
        fmpz_mod_ctx_clear(ctx2);
    }
}

/* Windows and concatenation */

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_window_init(fq_default_mat_t window,
    const fq_default_mat_t mat, slong r1, slong c1, slong r2, slong c2,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_window_init(window->fq_zech,
                               mat->fq_zech, r1, c1, r2, c2, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_window_init(window->fq_nmod,
                               mat->fq_nmod, r1, c1, r2, c2, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_window_init(window->nmod,
                               mat->nmod, r1, c1, r2, c2);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_window_init(window->fmpz_mod,
                               mat->fmpz_mod, r1, c1, r2, c2, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_window_init(window->fq,
                                         mat->fq, r1, c1, r2, c2, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_window_clear(fq_default_mat_t window,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_window_clear(window->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_window_clear(window->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_window_clear(window->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_window_clear(window->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_window_clear(window->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }

}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_concat_horizontal(fq_default_mat_t res,
          const fq_default_mat_t mat1,  const fq_default_mat_t mat2,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_concat_horizontal(res->fq_zech,
                               mat1->fq_zech, mat2->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_concat_horizontal(res->fq_nmod,
                               mat1->fq_nmod, mat2->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_concat_horizontal(res->nmod, mat1->nmod, mat2->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_concat_horizontal(res->fmpz_mod,
                                       mat1->fmpz_mod, mat2->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_concat_horizontal(res->fq, mat1->fq, mat2->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_concat_vertical(fq_default_mat_t res,
          const fq_default_mat_t mat1,  const fq_default_mat_t mat2,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_concat_vertical(res->fq_zech,
                               mat1->fq_zech, mat2->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_concat_vertical(res->fq_nmod,
                               mat1->fq_nmod, mat2->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_concat_vertical(res->nmod, mat1->nmod, mat2->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_concat_vertical(res->fmpz_mod,
                                     mat1->fmpz_mod, mat2->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_concat_vertical(res->fq, mat1->fq, mat2->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Input and output  *********************************************************/

#ifdef FLINT_HAVE_FILE
int fq_default_mat_fprint(FILE * file, const fq_default_mat_t mat, const fq_default_ctx_t ctx);
int fq_default_mat_fprint_pretty(FILE * file, const fq_default_mat_t mat, const fq_default_ctx_t ctx);
#endif

int fq_default_mat_print(const fq_default_mat_t mat, const fq_default_ctx_t ctx);
int fq_default_mat_print_pretty(const fq_default_mat_t mat, const fq_default_ctx_t ctx);

/* TODO: Read functions */

/* Random matrix generation  *************************************************/

FQ_DEFAULT_MAT_INLINE void fq_default_mat_randtest(fq_default_mat_t mat,
                                flint_rand_t state, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_randtest(mat->fq_zech, state, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_randtest(mat->fq_nmod, state, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_randtest(mat->nmod, state);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_randtest(mat->fmpz_mod, state, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_randtest(mat->fq, state, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_randrank(fq_default_mat_t mat,
                    flint_rand_t state, slong rank, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_randrank(mat->fq_zech, state, rank, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_randrank(mat->fq_nmod, state, rank, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_randrank(mat->nmod, state, rank);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_randrank(mat->fmpz_mod, state, rank, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_randrank(mat->fq, state, rank, FQ_DEFAULT_CTX_FQ(ctx));
    }
}


FQ_DEFAULT_MAT_INLINE
void fq_default_mat_randops(fq_default_mat_t mat,
                   flint_rand_t state, slong count, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_randops(mat->fq_zech, state, count, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_randops(mat->fq_nmod, state, count, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_randops(mat->nmod, state, count);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_randops(mat->fmpz_mod, state, count, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_randops(mat->fq, state, count, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_randtril(fq_default_mat_t mat,
                      flint_rand_t state, int unit, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_randtril(mat->fq_zech, state, unit, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_randtril(mat->fq_nmod, state, unit, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_randtril(mat->nmod, state, unit);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_randtril(mat->fmpz_mod, state, unit, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_randtril(mat->fq, state, unit, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_randtriu(fq_default_mat_t mat,
                      flint_rand_t state, int unit, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_randtriu(mat->fq_zech, state, unit, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_randtriu(mat->fq_nmod, state, unit, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_randtriu(mat->nmod, state, unit);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_randtriu(mat->fmpz_mod, state, unit, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_randtriu(mat->fq, state, unit, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Norms */

/* Transpose */

/* Addition and subtraction */

FQ_DEFAULT_MAT_INLINE void fq_default_mat_add(fq_default_mat_t C,
                      const fq_default_mat_t A, const fq_default_mat_t B,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_add(C->fq_zech, A->fq_zech, B->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_add(C->fq_nmod, A->fq_nmod, B->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_add(C->nmod, A->nmod, B->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_add(C->fmpz_mod, A->fmpz_mod, B->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_add(C->fq, A->fq, B->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_sub(fq_default_mat_t C,
                      const fq_default_mat_t A, const fq_default_mat_t B,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_sub(C->fq_zech, A->fq_zech, B->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_sub(C->fq_nmod, A->fq_nmod, B->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_sub(C->nmod, A->nmod, B->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_sub(C->fmpz_mod, A->fmpz_mod, B->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_sub(C->fq, A->fq, B->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_neg(fq_default_mat_t B,
                          const fq_default_mat_t A, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_neg(B->fq_zech, A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_neg(B->fq_nmod, A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_neg(B->nmod, A->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_neg(B->fmpz_mod, A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_neg(B->fq, A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_submul(fq_default_mat_t D,
                        const fq_default_mat_t C, const fq_default_mat_t A,
                          const fq_default_mat_t B, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_submul(D->fq_zech,
                         C->fq_zech, A->fq_zech, B->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_submul(D->fq_nmod,
                         C->fq_nmod, A->fq_nmod, B->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_submul(D->nmod, C->nmod, A->nmod, B->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_submul(D->fmpz_mod, C->fmpz_mod, A->fmpz_mod, B->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_submul(D->fq, C->fq, A->fq, B->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Scalar operations */

/* Multiplication */

FQ_DEFAULT_MAT_INLINE void fq_default_mat_mul(fq_default_mat_t C,
                      const fq_default_mat_t A, const fq_default_mat_t B,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_mul(C->fq_zech, A->fq_zech, B->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_mul(C->fq_nmod, A->fq_nmod, B->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_mul(C->nmod, A->nmod, B->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_mul(C->fmpz_mod, A->fmpz_mod, B->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_mul(C->fq, A->fq, B->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE slong fq_default_mat_lu(slong * P,
                  fq_default_mat_t A, int rank_check, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_lu(P, A->fq_zech, rank_check, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_lu(P, A->fq_nmod, rank_check, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_lu(P, A->nmod, rank_check);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_lu(P, A->fmpz_mod, rank_check, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_lu(P, A->fq, rank_check, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Inverse *******************************************************************/

FQ_DEFAULT_MAT_INLINE int fq_default_mat_inv(fq_default_mat_t B,
                                  fq_default_mat_t A, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_inv(B->fq_zech, A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_inv(B->fq_nmod, A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_inv(B->nmod, A->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_inv(B->fmpz_mod, A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_inv(B->fq, A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Solving *******************************************************************/

FQ_DEFAULT_MAT_INLINE slong fq_default_mat_rref(fq_default_mat_t B, const fq_default_mat_t A,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_rref(B->fq_zech, A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_rref(B->fq_nmod, A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_set(B->nmod, A->nmod);
        return nmod_mat_rref(B->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_rref(B->fmpz_mod, A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_rref(B->fq, A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE slong fq_default_mat_nullspace(fq_default_mat_t X,
                          const fq_default_mat_t A, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_nullspace(X->fq_zech, A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_nullspace(X->fq_nmod, A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_nullspace(X->nmod, A->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_nullspace(X->fmpz_mod, A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_nullspace(X->fq, A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE slong fq_default_mat_rank(const fq_default_mat_t A,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_rank(A->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_rank(A->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_rank(A->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_rank(A->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_rank(A->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_solve_tril(fq_default_mat_t X,
                         const fq_default_mat_t L, const fq_default_mat_t B,
                                          int unit, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_solve_tril(X->fq_zech, L->fq_zech, B->fq_zech, unit, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_solve_tril(X->fq_nmod, L->fq_nmod, B->fq_nmod, unit, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_solve_tril(X->nmod, L->nmod, B->nmod, unit);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_solve_tril(X->fmpz_mod, L->fmpz_mod, B->fmpz_mod, unit, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_solve_tril(X->fq, L->fq, B->fq, unit, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

FQ_DEFAULT_MAT_INLINE void fq_default_mat_solve_triu(fq_default_mat_t X,
                        const fq_default_mat_t U, const fq_default_mat_t B,
                                          int unit, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_solve_triu(X->fq_zech, U->fq_zech, B->fq_zech, unit, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_solve_triu(X->fq_nmod, U->fq_nmod, B->fq_nmod, unit, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_solve_triu(X->nmod, U->nmod, B->nmod, unit);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_solve_triu(X->fmpz_mod, U->fmpz_mod, B->fmpz_mod, unit, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_solve_triu(X->fq, U->fq, B->fq, unit, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Nonsingular solving *******************************************************/

FQ_DEFAULT_MAT_INLINE int fq_default_mat_solve(fq_default_mat_t X,
                      const fq_default_mat_t A, const fq_default_mat_t C,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_solve(X->fq_zech,
                                     A->fq_zech, C->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_solve(X->fq_nmod,
                                     A->fq_nmod, C->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_solve(X->nmod, A->nmod, C->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_solve(X->fmpz_mod, A->fmpz_mod, C->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_solve(X->fq, A->fq, C->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Solving *******************************************************************/

FQ_DEFAULT_MAT_INLINE int fq_default_mat_can_solve(fq_default_mat_t X,
                      const fq_default_mat_t A, const fq_default_mat_t B,
                                                    const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        return fq_zech_mat_can_solve(X->fq_zech,
                                     A->fq_zech, B->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        return fq_nmod_mat_can_solve(X->fq_nmod,
                                     A->fq_nmod, B->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        return nmod_mat_can_solve(X->nmod, A->nmod, B->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        return fmpz_mod_mat_can_solve(X->fmpz_mod, A->fmpz_mod, B->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        return fq_mat_can_solve(X->fq, A->fq, B->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Transforms ****************************************************************/

FQ_DEFAULT_MAT_INLINE
void fq_default_mat_similarity(fq_default_mat_t A, slong r,
                                      fq_default_t d, const fq_default_ctx_t ctx)
{
    if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_ZECH)
    {
        fq_zech_mat_similarity(A->fq_zech, r, d->fq_zech, FQ_DEFAULT_CTX_FQ_ZECH(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FQ_NMOD)
    {
        fq_nmod_mat_similarity(A->fq_nmod, r, d->fq_nmod, FQ_DEFAULT_CTX_FQ_NMOD(ctx));
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_NMOD)
    {
        nmod_mat_similarity(A->nmod, r, d->nmod);
    }
    else if (_FQ_DEFAULT_TYPE(ctx) == _FQ_DEFAULT_FMPZ_MOD)
    {
        fmpz_mod_mat_similarity(A->fmpz_mod, r, d->fmpz_mod, FQ_DEFAULT_CTX_FMPZ_MOD(ctx));
    }
    else
    {
        fq_mat_similarity(A->fq, r, d->fq, FQ_DEFAULT_CTX_FQ(ctx));
    }
}

/* Characteristic polynomial *************************************************/

/* this prototype really lives in fq_poly_templates.h
 * FQ_DEFAULT_MAT_INLINE
 * void fq_default_mat_charpoly(fq_default_poly_t) p,
 *                            fq_default_mat_t A, const fq_default_ctx_t ctx)
 * {
 *   fq_default_mat_charpoly_danilevsky(p, A, ctx)
 * }
 */

/* Minimal polynomial ********************************************************/

/* this prototype really lives in fq_poly_templates.h
 * FQ_DEFAULT_MAT_INLINE
 * void fq_default_mat_minpoly(fq_default_poly_t) p,
 *                   const fq_default_mat_t X, const fq_default_ctx_t ctx)
 */

#ifdef __cplusplus
}
#endif

#endif
