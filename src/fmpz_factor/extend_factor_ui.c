/*
    Copyright (C) 2008, 2009 William Hart
    Copyright (C) 2010 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "ulong_extras.h"
#include "fmpz.h"
#include "fmpz_factor.h"

void
_fmpz_factor_extend_factor_ui(fmpz_factor_t factor, ulong n)
{
    slong i, len;
    n_factor_t nfac;

    if (n == 0)
    {
        _fmpz_factor_set_length(factor, 0);
        factor->sign = 0;
        return;
    }

    n_factor_init(&nfac);
    n_factor(&nfac, n, 0);

    len = factor->num;

    _fmpz_factor_fit_length(factor, len + nfac.num);
    _fmpz_factor_set_length(factor, len + nfac.num);

    for (i = 0; i < nfac.num; i++)
    {
        fmpz_set_ui(factor->p + len + i, nfac.p[i]);
        factor->exp[len + i] = nfac.exp[i];
    }
}
