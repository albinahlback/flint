/*
    Copyright (C) 2015 Fredrik Johansson

    This file is part of Arb.

    Arb is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.  See <http://www.gnu.org/licenses/>.
*/

#include "test_helpers.h"
#include "acb.h"
#include "acb_hypgeom.h"

/* generated with mpmath */
static const double testdata[] = {
    0.0, 0.0,
    -0.02701626593453869896, 0.06085851132589568114,
    0.092391418427377575111, 0.29443833672801130076,
    0.65131374739365937388, 0.37063665572827331764,
    0.83847032733386007862, -0.28730577566514336534,
    -0.14750357092078405673, -0.30269451746476217073,
    -0.048646822931479694311, 0.038926542608778397121,
    0.0048197328091099892156, 0.003641138129582944611,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.027574808444335508972, 0.11025410152651193448,
    0.35400136349600893569, 0.24420566191536183849,
    0.67425312681608106523, -0.11728998324910652574,
    -0.069322941709900597259, -0.18037674817459565026,
    -0.018729775699827662658, 0.013254956213452933465,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.14771800806690767222, 0.11532328561497280466,
    0.50057392828327004059, -0.029559886614345590592,
    -0.02552211332726607337, -0.080244885187214384683,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.28209479177387814347, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.28209479177387814347, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.14771800806690767222, 0.11532328561497280466,
    0.50057392828327004059, -0.029559886614345590592,
    -0.02552211332726607337, -0.080244885187214384683,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.027574808444335508972, 0.11025410152651193448,
    0.35400136349600893569, 0.24420566191536183849,
    0.67425312681608106523, -0.11728998324910652574,
    -0.069322941709900597259, -0.18037674817459565026,
    -0.018729775699827662658, 0.013254956213452933465,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    -0.02701626593453869896, 0.06085851132589568114,
    0.092391418427377575111, 0.29443833672801130076,
    0.65131374739365937388, 0.37063665572827331764,
    0.83847032733386007862, -0.28730577566514336534,
    -0.14750357092078405673, -0.30269451746476217073,
    -0.048646822931479694311, 0.038926542608778397121,
    0.0048197328091099892156, 0.003641138129582944611,
    0.0, 0.0,
    -0.033798002071615353821, 0.018033964330683617275,
    -0.071988993423855434033, 0.19195223773954952294,
    0.23381678888295549361, 0.58288246446413288682,
    1.0666048448920927459, 0.45816450098677839043,
    0.99160821344691798668, -0.56670953266315770842,
    -0.27930640165163989333, -0.44051756827083040505,
    -0.092578419347396006256, 0.086641542063743427664,
    0.015474322098595795308, 0.010316287906840124695,
    0.00051935757617903067671, -0.0014726339944978705874,
};

TEST_FUNCTION_START(acb_hypgeom_spherical_y, state)
{
    {
        slong i, n, m;
        acb_t z, w, x, y;

        acb_init(z);
        acb_init(w);
        acb_init(x);
        acb_init(y);

        i = 0;

        arb_set_str(acb_realref(x), "0.2", 64);
        arb_set_str(acb_imagref(x), "0.3", 64);
        arb_set_str(acb_realref(y), "0.3", 64);
        arb_set_str(acb_imagref(y), "0.4", 64);

        for (n = -4; n <= 4; n++)
        {
            for (m = -4; m <= 4; m++)
            {
                acb_hypgeom_spherical_y(z, n, m, x, y, 64);

                acb_set_d_d(w, testdata[2 * i], testdata[2 * i + 1]);
                mag_set_d(arb_radref(acb_realref(w)), 1e-13);
                mag_set_d(arb_radref(acb_imagref(w)), 1e-13);

                if (!acb_overlaps(z, w))
                {
                    flint_printf("FAIL: value\n\n");
                    flint_printf("n = %wd, m = %wd\n", n, m);
                    flint_printf("z = "); acb_printd(z, 20); flint_printf("\n\n");
                    flint_printf("w = "); acb_printd(w, 20); flint_printf("\n\n");
                    flint_abort();
                }

                i++;
            }
        }

        acb_clear(z);
        acb_clear(w);
        acb_clear(x);
        acb_clear(y);
    }

    TEST_FUNCTION_END(state);
}

