/*
    Copyright (C) 2022 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "gr.h"
#include "gr_mat.h"

int
_gr_mat_write(gr_stream_t out, const gr_mat_t mat, int linebreaks, gr_ctx_t ctx)
{
    int status;
    slong r, c;
    slong i, j;
    slong sz;

    sz = ctx->sizeof_elem;
    r = gr_mat_nrows(mat, ctx);
    c = gr_mat_ncols(mat, ctx);

    status = GR_SUCCESS;
    gr_stream_write(out, "[");

    for (i = 0; i < r; i++)
    {
        gr_stream_write(out, "[");

        for (j = 0; j < c; j++)
        {
            status |= gr_write(out, GR_MAT_ENTRY(mat, i, j, sz), ctx);
            if (j < c - 1)
                gr_stream_write(out, ", ");
        }

        if (i < r - 1)
            gr_stream_write(out, linebreaks ? "],\n" : "], ");
        else
            gr_stream_write(out, "]");
    }
    gr_stream_write(out, "]");
    return status;
}

int
gr_mat_write(gr_stream_t out, const gr_mat_t mat, gr_ctx_t ctx)
{
    return _gr_mat_write(out, mat, 1, ctx);
}
