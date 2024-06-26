/*
    Copyright (C) 2011 Sebastian Pancratz
    Copyright (C) 2012 Lina Kulakova
    Copyright (C) 2013 Mike Hansen

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#ifdef T

#include "templates.h"

void
TEMPLATE(T, poly_factor_realloc) (TEMPLATE(T, poly_factor_t) fac, slong alloc,
                                  const TEMPLATE(T, ctx_t) ctx)
{
    if (alloc == 0)             /* Clear up, reinitialise */
    {
        TEMPLATE(T, poly_factor_clear) (fac, ctx);
        TEMPLATE(T, poly_factor_init) (fac, ctx);
    }
    else if (fac->alloc)        /* Realloc */
    {
        if (fac->alloc > alloc)
        {
            slong i;

            for (i = alloc; i < fac->num; i++)
                TEMPLATE(T, poly_clear) (fac->poly + i, ctx);

            fac->poly =
                flint_realloc(fac->poly,
                              alloc * sizeof(TEMPLATE(T, poly_struct)));
            fac->exp = flint_realloc(fac->exp, alloc * sizeof(slong));
            fac->alloc = alloc;
        }
        else if (fac->alloc < alloc)
        {
            slong i;

            fac->poly =
                flint_realloc(fac->poly,
                              alloc * sizeof(TEMPLATE(T, poly_struct)));
            fac->exp = flint_realloc(fac->exp, alloc * sizeof(slong));

            for (i = fac->alloc; i < alloc; i++)
            {
                TEMPLATE(T, poly_init) (fac->poly + i, ctx);
                fac->exp[i] = WORD(0);
            }
            fac->alloc = alloc;
        }
    }
    else                        /* Nothing allocated already so do it now */
    {
        slong i;

        fac->poly = flint_malloc(alloc * sizeof(TEMPLATE(T, poly_struct)));
        fac->exp = flint_calloc(alloc, sizeof(slong));

        for (i = 0; i < alloc; i++)
            TEMPLATE(T, poly_init) (fac->poly + i, ctx);
        fac->num = 0;
        fac->alloc = alloc;
    }
}


#endif
