/*
    Copyright (C) 2008, 2009, 2010 William Hart
    Copyright (C) 2014 Abhinav Baid

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "mpfr_vec.h"

int
_mpfr_vec_equal(mpfr_srcptr vec1, mpfr_srcptr vec2, slong len)
{
    slong i;
    if (vec1 == vec2)
        return 1;

    for (i = 0; i < len; i++)
        if (!mpfr_equal_p(vec1 + i, vec2 + i))
            return 0;

    return 1;
}
