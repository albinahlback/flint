/*=============================================================================

    This file is part of FLINT.

    FLINT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    FLINT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FLINT; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

=============================================================================*/
/******************************************************************************

    Copyright (C) 2013 Mike Hansen

******************************************************************************/

#include <stdlib.h>
#include "ulong_extras.h"
#include "fq_poly.h"

int
main(void)
{
    int iter;
    flint_rand_t state;
    flint_randinit(state);

    flint_printf("factor_kaltofen_shoup....");
    fflush(stdout);

    for (iter = 0; iter < 10 * flint_test_multiplier(); iter++)
    {
        fq_ctx_t ctx;
        fq_poly_t poly1, poly, q, r, product;
        fq_poly_factor_t res;
        
        slong i, j, length, num;
        slong exp[5];

        fq_ctx_randtest(ctx, state);

        fq_poly_init(poly1, ctx);
        fq_poly_init(poly, ctx);
        fq_poly_init(q, ctx);
        fq_poly_init(r, ctx);

        fq_poly_one(poly1, ctx);

        length = n_randint(state, 7) + 2;
        do
        {
            fq_poly_randtest(poly, state, length, ctx);
            if (poly->length)
                fq_poly_make_monic(poly, poly, ctx);
        }
        while ((poly->length != length) || (!fq_poly_is_irreducible(poly, ctx)));

        exp[0] = n_randint(state, 5) + 1;
        for (i = 0; i < exp[0]; i++)
        {
            fq_poly_mul(poly1, poly1, poly, ctx);
        }

        num = n_randint(state, 5) + 1;

        for (i = 1; i < num; i++)
        {
            do
            {
                length = n_randint(state, 5) + 2;
                fq_poly_randtest(poly, state, length, ctx);
                if (poly->length)
                {
                    fq_poly_make_monic(poly, poly, ctx);
                    fq_poly_divrem(q, r, poly1, poly, ctx);
                }
            }
            while ((poly->length != length) || (!fq_poly_is_irreducible(poly, ctx))
                   || (r->length == 0));

            exp[i] = n_randint(state, 5) + 1;
            for (j = 0; j < exp[i]; j++)
            {
                fq_poly_mul(poly1, poly1, poly, ctx);
            }
        }

        fq_poly_factor_init(res, ctx);
        fq_poly_factor_kaltofen_shoup(res, poly1, ctx);

        if (res->num != num)
        {
            flint_printf("Error: number of factors incorrect: %ld != %ld\n",
                   res->num, num);
            abort();
        }

        fq_poly_init(product, ctx);
        fq_poly_one(product, ctx);
        for (i = 0; i < res->num; i++)
        {
            for (j = 0; j < res->exp[i]; j++)
            {
                fq_poly_mul(product, product, res->poly + i, ctx);
            }
        }

        fq_poly_scalar_mul_fq(product, product,
                              poly1->coeffs + (poly1->length - 1),
                              ctx);

        if (!fq_poly_equal(poly1, product, ctx))
        {
            flint_printf
                ("Error: product of factors does not equal to the original polynomial\n");
            fq_ctx_print(ctx);
            flint_printf("\n");
            flint_printf("poly:\n");
            fq_poly_print_pretty(poly1, "x", ctx);
            flint_printf("\n");
            flint_printf("product:\n");
            fq_poly_print_pretty(product, "x", ctx);
            flint_printf("\n");
            abort();
        }

        fq_ctx_clear(ctx);
        fq_poly_clear(product, ctx);
        fq_poly_clear(q, ctx);
        fq_poly_clear(r, ctx);
        fq_poly_clear(poly1, ctx);
        fq_poly_clear(poly, ctx);
        fq_poly_factor_clear(res, ctx);
    }

    flint_randclear(state);
    _fmpz_cleanup();
    flint_printf("PASS\n");
    return 0;
}
