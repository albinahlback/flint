/*
    Copyright (C) 2021 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fexpr.h"

void
fexpr_set_si(fexpr_t res, slong c)
{
    if (c >= FEXPR_COEFF_MIN && c <= FEXPR_COEFF_MAX)
    {
        res->data[0] = ((ulong) c << FEXPR_TYPE_BITS);
    }
    else
    {
        fexpr_fit_size(res, 2);
        res->data[0] = ((c > 0) ? FEXPR_TYPE_BIG_INT_POS : FEXPR_TYPE_BIG_INT_NEG) | (2 << FEXPR_TYPE_BITS);
        res->data[1] = FLINT_UABS(c);
    }
}
