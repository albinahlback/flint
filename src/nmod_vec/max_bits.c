/*
    Copyright (C) 2010 William Hart
    Copyright (C) 2021 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "nmod_vec.h"
#include "longlong.h"

flint_bitcnt_t _nmod_vec_max_bits(nn_srcptr vec, slong len)
{
    slong i;
    ulong mask = 0;

    for (i = 0; i < len; i++)
    {
        mask |= vec[i];

        if (mask >= (UWORD(1) << (FLINT_BITS - 1)))
            return FLINT_BITS;
    }

    return FLINT_BIT_COUNT(mask);
}
