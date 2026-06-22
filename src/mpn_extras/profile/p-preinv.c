/*
    Copyright (C) 2026 Albin Ahlbäck

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

/*
    Latency of nn_preinv_N against GMP's inversion.  For a fair comparison we
    pit nn_preinv_N against the unfolded n-th branch of GMP's mpn_invert
    (mpn/generic/invert.c), with the size n held constant:

        nn_preinv_1  <->  invert_limb
        nn_preinv_2  <->  build (B^4 - 1) and mpn_divrem_2
        nn_preinv_n  <->  build (B^{2n} - 1), invert_pi1 and mpn_sbpi1_div_q
                          for n = 4, 8, 16, ...

    Note that mpn_invert returns floor((B^{2n} - 1) / d) - B^n while nn_preinv_N
    returns floor(B^{2n} / d) - B^n - 1; the two differ only when d | B^{2n},
    which never happens for the normalized random divisors used here.

    ---- How latency is measured (the chain) ----

    Each iteration feeds the previous call's OUTPUT back in as the divisor, a
    true dependency chain through the whole routine, with almost no surrounding
    work:

        f(rp, dp);                       rp = reciprocal(dp)
        swap(rp, dp);                    dp now points at the fresh output
        dp[n-1] |= 2^63;                 re-normalise it -> next divisor

    The swap is a free pointer exchange; the only per-iteration overhead is the
    single OR that re-normalises the divisor (and its store-to-load forward).  We
    do NOT time a separate "marshaling" loop and subtract it: that subtraction
    over-counts, because in the real loop parts of the marshaling overlap the
    routine (the divisor's low limbs are computed while the div already runs; the
    low output limbs are produced before the top one), whereas a stand-alone
    overhead loop has nothing to overlap and exposes its full latency.  Reporting
    the raw chained time, minus nothing, is both simpler and more honest.

    Re-normalising is not cosmetic: nn_preinv forms d0 = (d[n-1]>>1)+1 and does
    divq 2^125, d0; if the top bit were clear, d0 could be tiny, the quotient
    would exceed 2^64 and div would #DE.  Setting it also keeps the fed-back
    divisor a valid normalized divisor.

    ---- Why the measurement is structured the way it is ----

    On a CPU with a moving clock (boost / thermal throttling / the powersave
    governor) a naive harness is untrustworthy in two independent ways:

      1. Frequency.  Timing in TSC ticks and multiplying by a single core/TSC
         factor calibrated once at start-up is wrong the moment the clock moves:
         a routine measured later, at a lower clock, burns more TSC ticks for the
         same work and, scaled by the stale factor, looks slower than it is.

      2. Context.  The per-call latency of these long dependency chains is
         surprisingly sensitive to what ran just before (branch-predictor /
         op-cache / memory-disambiguation state).  A routine measured in a
         different position in the run can shift by >10% at a FIXED clock.

    Both effects made nn_preinv_8b look ~4 cycles slower than nn_preinv_8 in the
    old harness (8b was always measured last), when at equal clock and context it
    is in fact ~13% faster.

    The cure is to measure every routine for a given size TOGETHER, in rotated
    interleaved rounds (bench_group): each round times every routine once,
    rotating which goes first, and we keep the per-routine min over rounds.  All
    routines then share the same thermal state, the same clock, and statistically
    the same surrounding context, so their comparison is fair.  A dependent
    1-cycle/iter reference loop, timed inside the same rounds, gives the core
    clock for that window, so all figures are reported in CORE CYCLES -- a
    property of the work, independent of the clock.

    Caveat: each routine's divisors are its own fed-back outputs (re-seeded from
    the clumpy pool at the start of every burst).  For the branchless nn_preinv_N
    this distribution barely affects latency; for the branchy GMP paths it is a
    minor difference from real clumpy divisors -- treat the GMP columns as
    indicative, the nn_preinv comparison as exact.
*/

#if defined(__x86_64__) && defined(__GNUC__)

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <x86intrin.h>
#include "mpn_extras.h"

#define MAXN 8                          /* largest size, in limbs */

typedef void (*preinv_fn)(mp_ptr, mp_srcptr);

/* GMP primitives of the unfolded mpn_invert.  mpn_divrem_2 and mpn_invert_limb
   come from gmp.h / mpn_extras.h; sbpi1_div_q is internal but exported. */
mp_limb_t __gmpn_sbpi1_div_q(mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t,
                             mp_limb_t);
mp_limb_t __gmpn_invert_limb(mp_limb_t);

/* invert_pi1 from gmp-impl.h, returning inv.inv32 (the 3/2 inverse). */
static inline mp_limb_t
invert_pi1_inv32(mp_limb_t d1, mp_limb_t d0)
{
    mp_limb_t v, p, t1, t0, mask;
    __uint128_t t;

    v = __gmpn_invert_limb(d1);
    p = d1 * v;
    p += d0;
    if (p < d0)
    {
        v--;
        mask = -(mp_limb_t) (p >= d1);
        p -= d1;
        v += mask;
        p -= mask & d1;
    }
    t = (__uint128_t) d0 * v;
    t1 = (mp_limb_t) (t >> 64);
    t0 = (mp_limb_t) t;
    p += t1;
    if (p < t1)
    {
        v--;
        if (p >= d1 && (p > d1 || t0 >= d0))
            v--;
    }
    return v;
}

/* ---- GMP wrappers with the nn_preinv_N signature ---- */
static void
gmp_invert_1(mp_ptr ip, mp_srcptr dp)
{
    ip[0] = __gmpn_invert_limb(dp[0]);
}

static void
gmp_invert_2(mp_ptr ip, mp_srcptr dp)
{
    mp_limb_t xp[4];
    xp[0] = ~UWORD(0);
    xp[1] = ~UWORD(0);
    xp[2] = ~dp[0];
    xp[3] = ~dp[1];
    mpn_divrem_2(ip, 0, xp, 4, dp);
}

/* invert_pi1 + sbpi1_div_q for n = gmp_n limbs (n > 2). */
static mp_size_t gmp_n;
static void
gmp_invert_generic(mp_ptr ip, mp_srcptr dp)
{
    mp_limb_t xp[2 * MAXN], inv;
    mp_size_t n = gmp_n, i;
    for (i = 0; i < n; i++)
        xp[i] = ~UWORD(0);
    for (i = 0; i < n; i++)
        xp[n + i] = ~dp[i];
    inv = invert_pi1_inv32(dp[n - 1], dp[n - 2]);
    __gmpn_sbpi1_div_q(ip, xp, 2 * n, dp, n, inv);
}

static inline uint64_t
rdtsc_fenced(void)
{
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

/* ---- non-uniform "clumpy" 64-bit words, like real divisors ---- */
static uint64_t rng_state;
static inline uint64_t
rng(void)
{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t
nonuniform64(void)
{
    uint64_t v = 0;
    int pos = 0, bit = rng() & 1;
    while (pos < 64)
    {
        int run = 1 + (int) (rng() % 9);
        if (pos + run > 64) run = 64 - pos;
        if (bit)
            v |= (((run == 64) ? ~UWORD(0) : ((UWORD(1) << run) - 1)) << pos);
        pos += run;
        bit ^= 1;
    }
    return v;
}

#define K         4096
#define HI        (UWORD(1) << 63)
#define CAL_ITERS 100000ULL             /* 1-core-cycle/iter reference loop */

static uint64_t D[K][MAXN];

static void
gen_data(void)
{
    long i, j;
    for (i = 0; i < K; i++)
        for (j = 0; j < MAXN; j++)
            D[i][j] = nonuniform64();
}

/* One burst: N chained calls of f (output fed back as the next divisor); the
   two ping-pong buffers start from a normalized clumpy divisor.  Returns TSC. */
static inline uint64_t
swap_burst(preinv_fn f, mp_ptr A, mp_ptr B, mp_size_t n, long N, long seed_row)
{
    mp_ptr rp = A, dp = B;
    uint64_t c0;
    long i;
    for (i = 0; i < n; i++) dp[i] = D[seed_row & (K - 1)][i];
    dp[n - 1] |= HI;
    c0 = rdtsc_fenced();
    for (i = 0; i < N; i++)
    {
        f(rp, dp);
        { mp_ptr t = rp; rp = dp; dp = t; }   /* free pointer swap */
        dp[n - 1] |= HI;                       /* re-normalize -> next divisor */
    }
    return rdtsc_fenced() - c0;
}

/* Latency of nf routines over n-limb divisors, measured together in rotated
   interleaved rounds; returns min TSC/call in tsc_pc[] and, in *cpt, the core
   cycles per TSC tick for THIS window.  Rotating which routine leads each round,
   and keeping the per-routine min, makes the routines share clock, thermal state
   and surrounding context -- the only way the comparison is meaningful when the
   clock and the pipeline state drift during the run.

   The core-clock reference (a dependent 1-cycle/iter loop, known CAL_ITERS core
   cycles) is timed once per round, inside the same rounds, so it runs at exactly
   the clock the routines ran at -- even for the light sizes, where a lone
   reference loop can boost to a different clock than a div/mulx-heavy routine.
   Core cycles = TSC * cpt are then a property of the work, not the clock. */
static void
bench_group(const preinv_fn *fs, int nf, mp_size_t n, long N, long rounds,
            double *tsc_pc, double *cpt)
{
    mp_limb_t A[MAXN], B[MAXN];
    uint64_t best[8], best_cal = UINT64_MAX;
    int k;
    long r;

    for (k = 0; k < nf; k++) best[k] = UINT64_MAX;

    for (k = 0; k < nf; k++)            /* warmup: ramp boost, prime predictors */
        swap_burst(fs[k], A, B, n, 200000, k);

    for (r = 0; r < rounds; r++)
    {
        uint64_t x = 0, ci = CAL_ITERS, cc0, cdt;

        for (k = 0; k < nf; k++)
        {
            int idx = (k + (int) r) % nf;      /* rotate the leader each round */
            uint64_t dt = swap_burst(fs[idx], A, B, n, N, r * nf + k);
            if (dt < best[idx]) best[idx] = dt;
        }

        cc0 = rdtsc_fenced();           /* core-clock reference, same window */
        __asm__ volatile (
            ".p2align 5\n\t"            /* align the loop: keep it 1 cycle/iter
                                          regardless of surrounding code layout */
            "1:\n\t" "add $1, %0\n\t" "dec %1\n\t" "jnz 1b\n\t"
            : "+r"(x), "+r"(ci) :: "cc");
        cdt = rdtsc_fenced() - cc0;
        if (cdt < best_cal) best_cal = cdt;
    }

    *cpt = (double) CAL_ITERS / (double) best_cal;
    for (k = 0; k < nf; k++) tsc_pc[k] = (double) best[k] / (double) N;
}

/* ============================ what gets measured ============================
   One row per size.  `var` holds the nn_preinv variant(s) to compare (var[0] is
   the reference); `rival` is the GMP routine for that size.  All are measured
   together, so their columns share clock and context.

   To add a second nn_preinv for a size, bump nvar and fill var[1]/var_label[1],
   e.g. for size 2:
       { 2, 2, { nn_preinv_2, nn_preinv_2b }, { "nn_preinv_2", "nn_preinv_2b" },
             gmp_invert_2, "divrem_2" },
   To drop nn_preinv_8 in favour of nn_preinv_8b, make its row nvar = 1 with
   var[0] = nn_preinv_8b.
   ------------------------------------------------------------------------- */
typedef struct
{
    mp_size_t   n;
    int         nvar;                   /* number of nn_preinv variants (>= 1) */
    preinv_fn   var[2];
    const char *var_label[2];
    preinv_fn   rival;                  /* the GMP routine to compare against */
    const char *rival_label;
}
bench_case;

static const bench_case cases[] = {
    {1, 1, {nn_preinv_1}, {"nn_preinv_1" },
        gmp_invert_1,       "invert_limb"},
    {2, 1, {nn_preinv_2}, {"nn_preinv_2"},
        gmp_invert_2,       "divrem_2" },
    {4, 1, {nn_preinv_4}, {"nn_preinv_4"},
        gmp_invert_generic, "invert_pi1 + sbpi1_div_q"},
    {8, 1, {nn_preinv_8}, {"nn_preinv_8"},
        gmp_invert_generic, "invert_pi1 + sbpi1_div_q"},
};

int main(void)
{
    cpu_set_t set;
    long N = 3000, rounds = 2000;
    int ci;

    CPU_ZERO(&set); CPU_SET(0, &set);
    sched_setaffinity(0, sizeof(set), &set);

    rng_state = 0xdeadbeefcafef00dULL;
    gen_data();

    /* global warmup: ramp the boost clock before the first measured size */
    {
        double warm[1], wcpt;
        preinv_fn fw[1] = { nn_preinv_8 };
        bench_group(fw, 1, 8, N, 200, warm, &wcpt);
    }

    printf("chain         : %ld calls x %ld rounds (min)\n\n", N, rounds);
    printf("                            core cy/call   speedup\n");
    /* speedup: for an nn_preinv variant it is (reference / variant), so >1 means
       the variant is faster; for the GMP rival it is (rival / reference), so >1
       means nn_preinv is that much faster than GMP. */

    for (ci = 0; ci < (int) (sizeof(cases) / sizeof(cases[0])); ci++)
    {
        const bench_case *bc = &cases[ci];
        preinv_fn fs[3];
        double pc[3], cpt;
        int nf = bc->nvar + 1, k;

        for (k = 0; k < bc->nvar; k++) fs[k] = bc->var[k];
        fs[bc->nvar] = bc->rival;
        gmp_n = bc->n;                  /* used by gmp_invert_generic */

        bench_group(fs, nf, bc->n, N, rounds, pc, &cpt);

        printf("%-2ld %-24s:  %10.2f\n",/* GMP rival */
               (long) bc->n, bc->rival_label, pc[bc->nvar] * cpt);
        printf("%-2ld %-24s:  %10.2f    %5.2fx\n",         /* reference variant */
               (long) bc->n, bc->var_label[0], pc[0] * cpt,
               pc[bc->nvar] / pc[0]);
        for (k = 1; k < bc->nvar; k++)                    /* further variants */
            printf("%-2ld %-24s:  %10.2f    %5.2fx\n",
                   (long) bc->n, bc->var_label[k], pc[k] * cpt,
                   pc[0] / pc[k]);
    }

    return 0;
}

#else

int main(void)
{
    return 0;
}

#endif
