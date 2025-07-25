.. _fmpz-mod-poly:

**fmpz_mod_poly.h** -- polynomials over integers mod n
===============================================================================

The :type:`fmpz_mod_poly_t` data type represents elements of
`\mathbb{Z}/n\mathbb{Z}[x]` for a fixed modulus `n`. The
:type:`fmpz_mod_poly` module provides routines for memory management,
basic arithmetic and some higher level functions such as GCD, etc.

Each coefficient of an :type:`fmpz_mod_poly_t` is of type :type:`fmpz`
and represents an integer reduced modulo the fixed modulus `n` in the
range `[0,n)`.

Unless otherwise specified, all functions in this section permit
aliasing between their input arguments and between their input and
output arguments.

The :type:`fmpz_mod_poly_t` type is a typedef for an array of length 1
of :type:`fmpz_mod_poly_struct`'s. This permits passing parameters of
type :type:`fmpz_mod_poly_t` by reference.

In reality one never deals directly with the ``struct`` and simply
deals with objects of type :type:`fmpz_mod_poly_t`. For simplicity we
will think of an :type:`fmpz_mod_poly_t` as a ``struct``, though in
practice to access fields of this ``struct``, one needs to dereference
first, e.g. to access the ``length`` field of an
:type:`fmpz_mod_poly_t` called ``poly1`` one writes ``poly1->length``.

An :type:`fmpz_mod_poly_t` is said to be *normalised* if either
``length`` is zero, or if the leading coefficient of the polynomial is
non-zero. All :type:`fmpz_mod_poly` functions expect their inputs to
be normalised and all coefficients to be reduced modulo `n`, and
unless otherwise specified they produce output that is normalised with
coefficients reduced modulo `n`.

Simple example
--------------

The following example computes the square of the polynomial `5x^3 + 6`
in `\mathbb{Z}/7\mathbb Z[x]`.

.. code:: c

   #include "flint/fmpz_mod.h"
   #include "flint/fmpz_mod_poly.h"
   int main(void)
   {
       fmpz_mod_ctx_t ctx;
       fmpz_mod_poly_t x, y;

       fmpz_mod_ctx_init_ui(ctx, 7);
       fmpz_mod_poly_init(x, ctx);
       fmpz_mod_poly_init(y, ctx);
       fmpz_mod_poly_set_coeff_ui(x, 3, 5, ctx);
       fmpz_mod_poly_set_coeff_ui(x, 0, 6, ctx);
       fmpz_mod_poly_sqr(y, x, ctx);

       flint_printf("x = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n"
                    "x^2 = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n",
                    x, ctx, y, ctx);

       fmpz_mod_poly_clear(x, ctx);
       fmpz_mod_poly_clear(y, ctx);
       fmpz_mod_ctx_clear(ctx);
   }

The output is:

::

   4 7  6 0 0 5
   7 7  1 0 0 4 0 0 4

Types, macros and constants
-------------------------------------------------------------------------------

.. type:: fmpz_mod_poly_struct

    A structure holding a polynomial over the integers modulo `n`.

.. type:: fmpz_mod_poly_t

    An array of length 1 of ``fmpz_mod_poly_struct``.


Memory management
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_init(fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Initialises ``poly`` for use with context ``ctx`` and set it to zero.
    A corresponding call to :func:`fmpz_mod_poly_clear` must be made to free the memory used by the polynomial.

.. function:: void fmpz_mod_poly_init2(fmpz_mod_poly_t poly, slong alloc, const fmpz_mod_ctx_t ctx)

    Initialises ``poly`` with space for at least ``alloc`` coefficients
    and sets the length to zero.  The allocated coefficients are all set to
    zero.

.. function:: void fmpz_mod_poly_clear(fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Clears the given polynomial, releasing any memory used.  It must
    be reinitialised in order to be used again.

.. function:: void fmpz_mod_poly_realloc(fmpz_mod_poly_t poly, slong alloc, const fmpz_mod_ctx_t ctx)

    Reallocates the given polynomial to have space for ``alloc``
    coefficients.  If ``alloc`` is zero the polynomial is cleared
    and then reinitialised.  If the current length is greater than
    ``alloc`` the polynomial is first truncated to length ``alloc``.

.. function:: void fmpz_mod_poly_fit_length(fmpz_mod_poly_t poly, slong len, const fmpz_mod_ctx_t ctx)

    If ``len`` is greater than the number of coefficients currently
    allocated, then the polynomial is reallocated to have space for at
    least ``len`` coefficients.  No data is lost when calling this
    function.

    The function efficiently deals with the case where it is called
    many times in small increments by at least doubling the number of
    allocated coefficients when length is larger than the number of
    coefficients currently allocated.

.. function:: void _fmpz_mod_poly_normalise(fmpz_mod_poly_t poly)

    Sets the length of ``poly`` so that the top coefficient is non-zero.
    If all coefficients are zero, the length is set to zero.  This function
    is mainly used internally, as all functions guarantee normalisation.

.. function:: void _fmpz_mod_poly_set_length(fmpz_mod_poly_t poly, slong len)

    Demotes the coefficients of ``poly`` beyond ``len`` and sets
    the length of ``poly`` to ``len``.

.. function:: void fmpz_mod_poly_truncate(fmpz_mod_poly_t poly, slong len, const fmpz_mod_ctx_t ctx)

    If the current length of ``poly`` is greater than ``len``, it
    is truncated to have the given length.  Discarded coefficients are
    not necessarily set to zero.

.. function:: void fmpz_mod_poly_set_trunc(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, slong n, const fmpz_mod_ctx_t ctx)

    Notionally truncate ``poly`` to length `n` and set ``res`` to the
    result. The result is normalised.


Randomisation
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_randtest(fmpz_mod_poly_t f, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Sets the polynomial~`f` to a random polynomial of length up~``len``.

.. function:: void fmpz_mod_poly_randtest_irreducible(fmpz_mod_poly_t f, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Sets the polynomial~`f` to a random irreducible polynomial of length
    up~``len``, assuming ``len`` is positive.

.. function:: void fmpz_mod_poly_randtest_not_zero(fmpz_mod_poly_t f, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Sets the polynomial~`f` to a random polynomial of length up~``len``,
    assuming ``len`` is positive.

.. function:: void fmpz_mod_poly_randtest_monic(fmpz_mod_poly_t poly, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Generates a random monic polynomial with length ``len``.

.. function:: void fmpz_mod_poly_randtest_monic_irreducible(fmpz_mod_poly_t poly, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Generates a random monic irreducible polynomial with length ``len``.

.. function:: void fmpz_mod_poly_randtest_monic_primitive(fmpz_mod_poly_t poly, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Generates a random monic irreducible primitive polynomial with
    length ``len``.


.. function:: void fmpz_mod_poly_randtest_trinomial(fmpz_mod_poly_t poly, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Generates a random monic trinomial of length ``len``.

.. function:: int fmpz_mod_poly_randtest_trinomial_irreducible(fmpz_mod_poly_t poly, flint_rand_t state, slong len, slong max_attempts, const fmpz_mod_ctx_t ctx)

    Attempts to set ``poly`` to a monic irreducible trinomial of
    length ``len``.  It will generate up to ``max_attempts``
    trinomials in attempt to find an irreducible one.  If
    ``max_attempts`` is ``0``, then it will keep generating
    trinomials until an irreducible one is found.  Returns `1` if one
    is found and `0` otherwise.

.. function:: void fmpz_mod_poly_randtest_pentomial(fmpz_mod_poly_t poly, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Generates a random monic pentomial of length ``len``.

.. function:: int fmpz_mod_poly_randtest_pentomial_irreducible(fmpz_mod_poly_t poly, flint_rand_t state, slong len, slong max_attempts, const fmpz_mod_ctx_t ctx)

    Attempts to set ``poly`` to a monic irreducible pentomial of
    length ``len``.  It will generate up to ``max_attempts``
    pentomials in attempt to find an irreducible one.  If
    ``max_attempts`` is ``0``, then it will keep generating
    pentomials until an irreducible one is found.  Returns `1` if one
    is found and `0` otherwise.

.. function:: void fmpz_mod_poly_randtest_sparse_irreducible(fmpz_mod_poly_t poly, flint_rand_t state, slong len, const fmpz_mod_ctx_t ctx)

    Attempts to set ``poly`` to a sparse, monic irreducible polynomial
    with length ``len``.  It attempts to find an irreducible
    trinomial.  If that does not succeed, it attempts to find a
    irreducible pentomial.  If that fails, then ``poly`` is just
    set to a random monic irreducible polynomial.



Attributes
--------------------------------------------------------------------------------


.. function:: slong fmpz_mod_poly_degree(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Returns the degree of the polynomial.  The degree of the zero
    polynomial is defined to be `-1`.

.. function:: slong fmpz_mod_poly_length(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Returns the length of the polynomial, which is one more than
    its degree.

.. function:: fmpz * fmpz_mod_poly_lead(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Returns a pointer to the first leading coefficient of ``poly``
    if this is non-zero, otherwise returns ``NULL``.


Assignment and basic manipulation
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_set(fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Sets the polynomial ``poly1`` to the value of ``poly2``.

.. function:: void fmpz_mod_poly_swap(fmpz_mod_poly_t poly1, fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Swaps the two polynomials.  This is done efficiently by swapping
    pointers rather than individual coefficients.

.. function:: void fmpz_mod_poly_zero(fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to the zero polynomial.

.. function:: void fmpz_mod_poly_one(fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to the constant polynomial `1`.

.. function:: void fmpz_mod_poly_zero_coeffs(fmpz_mod_poly_t poly, slong i, slong j, const fmpz_mod_ctx_t ctx)

    Sets the coefficients of `X^k` for `k \in [i, j)` in the polynomial
    to zero.

.. function:: void fmpz_mod_poly_reverse(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, slong n, const fmpz_mod_ctx_t ctx)

    This function considers the polynomial ``poly`` to be of length `n`,
    notionally truncating and zero padding if required, and reverses
    the result.  Since the function normalises its result ``res`` may be
    of length less than `n`.


Conversion
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_set_ui(fmpz_mod_poly_t f, ulong c, const fmpz_mod_ctx_t ctx)

    Sets the polynomial `f` to the constant `c` reduced modulo `p`.

.. function:: void fmpz_mod_poly_set_fmpz(fmpz_mod_poly_t f, const fmpz_t c, const fmpz_mod_ctx_t ctx)

    Sets the polynomial `f` to the constant `c` reduced modulo `p`.

.. function:: void fmpz_mod_poly_set_fmpz_poly(fmpz_mod_poly_t f, const fmpz_poly_t g, const fmpz_mod_ctx_t ctx)

    Sets `f` to `g` reduced modulo `p`, where `p` is the modulus that
    is part of the data structure of `f`.

.. function:: void fmpz_mod_poly_get_fmpz_poly(fmpz_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_ctx_t ctx)

    Sets `f` to `g`.  This is done simply by lifting the coefficients
    of `g` taking representatives `[0, p) \subset \mathbf{Z}`.

.. function:: void fmpz_mod_poly_get_nmod_poly(nmod_poly_t f, const fmpz_mod_poly_t g)

   Sets `f` to `g` assuming the modulus of both polynomials is the same (no
   checking is performed).

.. function:: void fmpz_mod_poly_set_nmod_poly(fmpz_mod_poly_t f, const nmod_poly_t g)

   Sets `f` to `g` assuming the modulus of both polynomials is the same (no
   checking is performed).


Comparison
--------------------------------------------------------------------------------


.. function:: int fmpz_mod_poly_equal(const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Returns non-zero if the two polynomials are equal, otherwise returns zero.

.. function:: int fmpz_mod_poly_equal_trunc(const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, slong n, const fmpz_mod_ctx_t ctx)

    Notionally truncates the two polynomials to length `n` and returns non-zero
    if the two polynomials are equal, otherwise returns zero.

.. function:: int fmpz_mod_poly_is_zero(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Returns non-zero if the polynomial is zero.

.. function:: int fmpz_mod_poly_is_one(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Returns non-zero if the polynomial is the constant `1`.

.. function:: int fmpz_mod_poly_is_gen(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Returns non-zero if the polynomial is the degree `1` polynomial `x`.


Getting and setting coefficients
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_set_coeff_fmpz(fmpz_mod_poly_t poly, slong n, const fmpz_t x, const fmpz_mod_ctx_t ctx)

    Sets the coefficient of `X^n` in the polynomial to `x`,
    assuming `n \geq 0`.

.. function:: void fmpz_mod_poly_set_coeff_ui(fmpz_mod_poly_t poly, slong n, ulong x, const fmpz_mod_ctx_t ctx)

    Sets the coefficient of `X^n` in the polynomial to `x`,
    assuming `n \geq 0`.

.. function:: void fmpz_mod_poly_get_coeff_fmpz(fmpz_t x, const fmpz_mod_poly_t poly, slong n, const fmpz_mod_ctx_t ctx)

    Sets `x` to the coefficient of `X^n` in the polynomial,
    assuming `n \geq 0`.

.. function:: void fmpz_mod_poly_set_coeff_mpz(fmpz_mod_poly_t poly, slong n, const mpz_t x, const fmpz_mod_ctx_t ctx)

    Sets the coefficient of `X^n` in the polynomial to `x`,
    assuming `n \geq 0`.

.. function:: void fmpz_mod_poly_get_coeff_mpz(mpz_t x, const fmpz_mod_poly_t poly, slong n, const fmpz_mod_ctx_t ctx)

    Sets `x` to the coefficient of `X^n` in the polynomial,
    assuming `n \geq 0`.


Shifting
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_shift_left(fmpz * res, const fmpz * poly, slong len, slong n)

    Sets ``(res, len + n)`` to ``(poly, len)`` shifted left by
    `n` coefficients.

    Inserts zero coefficients at the lower end.  Assumes that ``len``
    and `n` are positive, and that ``res`` fits ``len + n`` elements.
    Supports aliasing between ``res`` and ``poly``.

.. function:: void fmpz_mod_poly_shift_left(fmpz_mod_poly_t f, const fmpz_mod_poly_t g, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` shifted left by `n` coeffs.  Zero
    coefficients are inserted.

.. function:: void _fmpz_mod_poly_shift_right(fmpz * res, const fmpz * poly, slong len, slong n)

    Sets ``(res, len - n)`` to ``(poly, len)`` shifted right by
    `n` coefficients.

    Assumes that ``len`` and `n` are positive, that ``len > n``,
    and that ``res`` fits ``len - n`` elements.  Supports aliasing
    between ``res`` and ``poly``, although in this case the top
    coefficients of ``poly`` are not set to zero.

.. function:: void fmpz_mod_poly_shift_right(fmpz_mod_poly_t f, const fmpz_mod_poly_t g, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` shifted right by `n` coefficients.  If `n`
    is equal to or greater than the current length of ``poly``, ``res``
    is set to the zero polynomial.


Addition and subtraction
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_add(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the sum of ``(poly1, len1)`` and
    ``(poly2, len2)``.  It is assumed that ``res`` has
    sufficient space for the longer of the two polynomials.

.. function:: void fmpz_mod_poly_add(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the sum of ``poly1`` and ``poly2``.

.. function:: void fmpz_mod_poly_add_series(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, slong n, const fmpz_mod_ctx_t ctx)

    Notionally truncate ``poly1`` and ``poly2`` to length `n` and set
    ``res`` to the sum.

.. function:: void _fmpz_mod_poly_sub(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``(poly1, len1)`` minus ``(poly2, len2)``.  It
    is assumed that ``res`` has sufficient space for the longer of the
    two polynomials.

.. function:: void fmpz_mod_poly_sub(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly1`` minus ``poly2``.

.. function:: void fmpz_mod_poly_sub_series(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, slong n, const fmpz_mod_ctx_t ctx)

    Notionally truncate ``poly1`` and ``poly2`` to length `n` and set
    ``res`` to the difference.

.. function:: void _fmpz_mod_poly_neg(fmpz * res, const fmpz * poly, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``(res, len)`` to the negative of ``(poly, len)``
    modulo `p`.

.. function:: void fmpz_mod_poly_neg(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the negative of ``poly`` modulo `p`.


Scalar multiplication and division
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_scalar_mul_fmpz(fmpz * res, const fmpz * poly, slong len, const fmpz_t x, const fmpz_mod_ctx_t ctx)
              void _fmpz_mod_poly_scalar_mul_ui(fmpz * res, const fmpz * poly, slong len, ulong x, const fmpz_mod_ctx_t ctx)

    Sets ``(res, len``) to ``(poly, len)`` multiplied by `x`,
    reduced modulo `p`.

.. function:: void fmpz_mod_poly_scalar_mul_fmpz(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_t x, const fmpz_mod_ctx_t ctx)
              void fmpz_mod_poly_scalar_mul_ui(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, ulong x, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` multiplied by `x`.

.. function:: void fmpz_mod_poly_scalar_addmul_fmpz(fmpz_mod_poly_t rop, const fmpz_mod_poly_t op, const fmpz_t x, const fmpz_mod_ctx_t ctx)

    Adds to ``rop`` the product of ``op`` by the scalar ``x``.

.. function:: void _fmpz_mod_poly_scalar_div_fmpz(fmpz * res, const fmpz * poly, slong len, const fmpz_t x, const fmpz_mod_ctx_t ctx)

    Sets ``(res, len``) to ``(poly, len)`` divided by `x` (i.e.
    multiplied by the inverse of `x \pmod{p}`). The result is reduced modulo
    `p`.

.. function:: void fmpz_mod_poly_scalar_div_fmpz(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_t x, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` divided by `x`, (i.e. multiplied by the
    inverse of `x \pmod{p}`). The result is reduced modulo `p`.


Multiplication
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_mul(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz_mod_ctx_t ctx)

    Sets ``(res, len1 + len2 - 1)`` to the product of ``(poly1, len1)``
    and ``(poly2, len2)``.  Assumes ``len1 >= len2 > 0``.  Allows
    zero-padding of the two input polynomials.

.. function:: void fmpz_mod_poly_mul(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the product of ``poly1`` and ``poly2``.

.. function:: void _fmpz_mod_poly_mullow(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``(res, n)`` to the lowest `n` coefficients of the product of
    ``(poly1, len1)`` and ``(poly2, len2)``.

    Assumes ``len1 >= len2 > 0`` and ``0 < n <= len1 + len2 - 1``.
    Allows for zero-padding in the inputs.  Does not support aliasing between
    the inputs and the output.

.. function:: void fmpz_mod_poly_mullow(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the lowest `n` coefficients of the product of
    ``poly1`` and ``poly2``.

.. function:: void _fmpz_mod_poly_sqr(fmpz * res, const fmpz * poly, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the square of ``poly``.

.. function:: void fmpz_mod_poly_sqr(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Computes ``res`` as the square of ``poly``.

.. function:: void fmpz_mod_poly_mulhigh(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, slong start, const fmpz_mod_ctx_t ctx)

    Computes the product of ``poly1`` and ``poly2`` and writes the
    coefficients from ``start`` onwards into the high coefficients of
    ``res``, the remaining coefficients being arbitrary.

.. function:: void _fmpz_mod_poly_mulmod(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz * f, slong lenf, const fmpz_mod_ctx_t ctx)

    Sets ``res, len1 + len2 - 1`` to the remainder of the product of
    ``poly1`` and ``poly2`` upon polynomial division by ``f``.

    It is required that ``len1 + len2 - lenf > 0``, which is equivalent
    to requiring that the result will actually be reduced. Otherwise, simply
    use ``_fmpz_mod_poly_mul`` instead.

    Aliasing of ``f`` and ``res`` is not permitted.

.. function:: void fmpz_mod_poly_mulmod(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_poly_t f, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the remainder of the product of ``poly1`` and
    ``poly2`` upon polynomial division by ``f``.

.. function:: void _fmpz_mod_poly_mulmod_preinv(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz * f, slong lenf, const fmpz * finv, slong lenfinv, const fmpz_mod_ctx_t ctx)

    Sets ``res, len1 + len2 - 1`` to the remainder of the product of
    ``poly1`` and ``poly2`` upon polynomial division by ``f``.

    It is required that ``finv`` is the inverse of the reverse of ``f``
    mod ``x^lenf``. It is required that ``len1 + len2 - lenf > 0``,
    which is equivalent to requiring that the result will actually be reduced.
    It is required that ``len1 < lenf`` and ``len2 < lenf``.
    Otherwise, simply use ``_fmpz_mod_poly_mul`` instead.

    Aliasing of ``f`` or ``finv`` and ``res`` is not permitted.

.. function:: void fmpz_mod_poly_mulmod_preinv(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_poly_t f, const fmpz_mod_poly_t finv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the remainder of the product of ``poly1`` and
    ``poly2`` upon polynomial division by ``f``. ``finv`` is the
    inverse of the reverse of ``f``. It is required that ``poly1`` and
    ``poly2`` are reduced modulo ``f``.


Products
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_product_roots_fmpz_vec(fmpz * poly, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``(poly, n + 1)`` to the monic polynomial which is the product
    of `(x - x_0)(x - x_1) \cdots (x - x_{n-1})`, the roots `x_i` being
    given by ``xs``. It is required that the roots are canonical.

    Aliasing of the input and output is not allowed.


.. function:: void fmpz_mod_poly_product_roots_fmpz_vec(fmpz_mod_poly_t poly, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to the monic polynomial which is the product
    of `(x - x_0)(x - x_1) \cdots (x - x_{n-1})`, the roots `x_i` being
    given by ``xs``. It is required that the roots are canonical.


.. function:: int fmpz_mod_poly_find_distinct_nonzero_roots(fmpz * roots, const fmpz_mod_poly_t A, const fmpz_mod_ctx_t ctx)

    If ``A`` has `\deg(A)` distinct nonzero roots in `\mathbb{F}_p`, write these roots out to ``roots[0]`` to ``roots[deg(A) - 1]`` and return ``1``.
    Otherwise, return ``0``. It is assumed that ``A`` is nonzero and that the modulus of ``A`` is prime.
    This function uses Rabin's probabilistic method via gcd's with `(x + \delta)^{\frac{p-1}{2}} - 1`.


Powering

--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_pow(fmpz * rop, const fmpz * op, slong len, ulong e, const fmpz_mod_ctx_t ctx)

    Sets ``rop = poly^e``, assuming that `e > 1` and ``elen > 0``,
    and that ``res`` has space for ``e*(len - 1) + 1`` coefficients.
    Does not support aliasing.

.. function:: void fmpz_mod_poly_pow(fmpz_mod_poly_t rop, const fmpz_mod_poly_t op, ulong e, const fmpz_mod_ctx_t ctx)

    Computes ``rop = poly^e``.  If `e` is zero, returns one,
    so that in particular ``0^0 = 1``.

.. function:: void _fmpz_mod_poly_pow_trunc(fmpz * res, const fmpz * poly, ulong e, slong trunc, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the low ``trunc`` coefficients of ``poly``
    (assumed to be zero padded if necessary to length ``trunc``) to
    the power ``e``. This is equivalent to doing a powering followed
    by a truncation. We require that ``res`` has enough space for
    ``trunc`` coefficients, that ``trunc > 0`` and that
    ``e > 1``. Aliasing is not permitted.

.. function:: void fmpz_mod_poly_pow_trunc(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, ulong e, slong trunc, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the low ``trunc`` coefficients of ``poly``
    to the power ``e``. This is equivalent to doing a powering
    followed by a truncation.

.. function:: void _fmpz_mod_poly_pow_trunc_binexp(fmpz * res, const fmpz * poly, ulong e, slong trunc, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the low ``trunc`` coefficients of ``poly``
    (assumed to be zero padded if necessary to length ``trunc``) to
    the power ``e``. This is equivalent to doing a powering followed
    by a truncation. We require that ``res`` has enough space for
    ``trunc`` coefficients, that ``trunc > 0`` and that
    ``e > 1``. Aliasing is not permitted. Uses the binary
    exponentiation method.

.. function:: void fmpz_mod_poly_pow_trunc_binexp(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, ulong e, slong trunc, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the low ``trunc`` coefficients of ``poly``
    to the power ``e``. This is equivalent to doing a powering
    followed by a truncation. Uses the binary exponentiation method.

.. function:: void _fmpz_mod_poly_powmod_ui_binexp(fmpz * res, const fmpz * poly, ulong e, const fmpz * f, slong lenf, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e > 0``.

    We require ``lenf > 1``. It is assumed that ``poly`` is already
    reduced modulo ``f`` and zero-padded as necessary to have length
    exactly ``lenf - 1``. The output ``res`` must have room for
    ``lenf - 1`` coefficients.

.. function:: void fmpz_mod_poly_powmod_ui_binexp(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, ulong e, const fmpz_mod_poly_t f, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e >= 0``.

.. function:: void _fmpz_mod_poly_powmod_ui_binexp_preinv(fmpz * res, const fmpz * poly, ulong e, const fmpz * f, slong lenf, const fmpz * finv, slong lenfinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e > 0``.
    We require ``finv`` to be the inverse of the reverse of ``f``.

    We require ``lenf > 1``. It is assumed that ``poly`` is already
    reduced modulo ``f`` and zero-padded as necessary to have length
    exactly ``lenf - 1``. The output ``res`` must have room for
    ``lenf - 1`` coefficients.

.. function:: void fmpz_mod_poly_powmod_ui_binexp_preinv(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, ulong e, const fmpz_mod_poly_t f, const fmpz_mod_poly_t finv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e >= 0``.
    We require ``finv`` to be the inverse of the reverse of ``f``.

.. function:: void _fmpz_mod_poly_powmod_fmpz_binexp(fmpz * res, const fmpz * poly, const fmpz_t e, const fmpz * f, slong lenf, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e > 0``.

    We require ``lenf > 1``. It is assumed that ``poly`` is already
    reduced modulo ``f`` and zero-padded as necessary to have length
    exactly ``lenf - 1``. The output ``res`` must have room for
    ``lenf - 1`` coefficients.

.. function:: void fmpz_mod_poly_powmod_fmpz_binexp(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_t e, const fmpz_mod_poly_t f, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e >= 0``.

.. function:: void _fmpz_mod_poly_powmod_fmpz_binexp_preinv(fmpz * res, const fmpz * poly, const fmpz_t e, const fmpz * f, slong lenf, const fmpz * finv, slong lenfinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e > 0``.
    We require ``finv`` to be the inverse of the reverse of ``f``.

    We require ``lenf > 1``. It is assumed that ``poly`` is already
    reduced modulo ``f`` and zero-padded as necessary to have length
    exactly ``lenf - 1``. The output ``res`` must have room for
    ``lenf - 1`` coefficients.

.. function:: void fmpz_mod_poly_powmod_fmpz_binexp_preinv(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_t e, const fmpz_mod_poly_t f, const fmpz_mod_poly_t finv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``poly`` raised to the power ``e``
    modulo ``f``, using binary exponentiation. We require ``e >= 0``.
    We require ``finv`` to be the inverse of the reverse of ``f``.

.. function:: void _fmpz_mod_poly_powmod_x_fmpz_preinv(fmpz * res, const fmpz_t e, const fmpz * f, slong lenf, const fmpz * finv, slong lenfinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``x`` raised to the power ``e`` modulo ``f``,
    using sliding window exponentiation. We require ``e > 0``.
    We require ``finv`` to be the inverse of the reverse of ``f``.

    We require ``lenf > 2``. The output ``res`` must have room for
    ``lenf - 1`` coefficients.

.. function:: void fmpz_mod_poly_powmod_x_fmpz_preinv(fmpz_mod_poly_t res, const fmpz_t e, const fmpz_mod_poly_t f, const fmpz_mod_poly_t finv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to ``x`` raised to the power ``e``
    modulo ``f``, using sliding window exponentiation. We require
    ``e >= 0``. We require ``finv`` to be the inverse of the reverse of
    ``

.. function:: void _fmpz_mod_poly_powers_mod_preinv_naive(fmpz ** res, const fmpz * f, slong flen, slong n, const fmpz * g, slong glen, const fmpz * ginv, slong ginvlen, const fmpz_mod_ctx_t ctx)

    Compute ``f^0, f^1, ..., f^(n-1) mod g``, where ``g`` has length ``glen``
    and ``f`` is reduced mod ``g`` and has length ``flen`` (possibly zero
    spaced). Assumes ``res`` is an array of ``n`` arrays each with space for
    at least ``glen - 1`` coefficients and that ``flen > 0``. We require that
    ``ginv`` of length ``ginvlen`` is set to the power series inverse of the
    reverse of ``g``.

.. function:: void fmpz_mod_poly_powers_mod_naive(fmpz_mod_poly_struct * res, const fmpz_mod_poly_t f, slong n, const fmpz_mod_poly_t g, const fmpz_mod_ctx_t ctx)

    Set the entries of the array ``res`` to ``f^0, f^1, ..., f^(n-1) mod g``.
    No aliasing is permitted between the entries of ``res`` and either of the
    inputs.

.. function:: void _fmpz_mod_poly_powers_mod_preinv_threaded_pool(fmpz ** res, const fmpz * f, slong flen, slong n, const fmpz * g, slong glen, const fmpz * ginv, slong ginvlen, const fmpz_mod_ctx_t p, thread_pool_handle * threads, slong num_threads)

    Compute ``f^0, f^1, ..., f^(n-1) mod g``, where ``g`` has length ``glen``
    and ``f`` is reduced mod ``g`` and has length ``flen`` (possibly zero
    spaced). Assumes ``res`` is an array of ``n`` arrays each with space for
    at least ``glen - 1`` coefficients and that ``flen > 0``. We require that
    ``ginv`` of length ``ginvlen`` is set to the power series inverse of the
    reverse of ``g``.

.. function:: void fmpz_mod_poly_powers_mod_bsgs(fmpz_mod_poly_struct * res, const fmpz_mod_poly_t f, slong n, const fmpz_mod_poly_t g, const fmpz_mod_ctx_t ctx)

    Set the entries of the array ``res`` to ``f^0, f^1, ..., f^(n-1) mod g``.
    No aliasing is permitted between the entries of ``res`` and either of the
    inputs.

.. function:: void fmpz_mod_poly_frobenius_powers_2exp_precomp(fmpz_mod_poly_frobenius_powers_2exp_t pow, const fmpz_mod_poly_t f, const fmpz_mod_poly_t finv, ulong m, const fmpz_mod_ctx_t ctx)

    If ``p = f->p``, compute `x^{(p^1)}`, `x^{(p^2)}`, `x^{(p^4)}`, ...,
    `x^{(p^{(2^l)})} \pmod{f}` where `2^l` is the greatest power of `2` less than
    or equal to `m`.

    Allows construction of `x^{(p^k)}` for `k = 0`, `1`, ..., `x^{(p^m)} \pmod{f}`
    using :func:`fmpz_mod_poly_frobenius_power`.

    Requires precomputed inverse of `f`, i.e. newton inverse.

.. function:: void fmpz_mod_poly_frobenius_powers_2exp_clear(fmpz_mod_poly_frobenius_powers_2exp_t pow, const fmpz_mod_ctx_t ctx)

    Clear resources used by the ``fmpz_mod_poly_frobenius_powers_2exp_t``
    struct.

.. function:: void fmpz_mod_poly_frobenius_power(fmpz_mod_poly_t res, fmpz_mod_poly_frobenius_powers_2exp_t pow, const fmpz_mod_poly_t f, ulong m, const fmpz_mod_ctx_t ctx)

    If ``p = f->p``, compute `x^{(p^m)} \pmod{f}`.

    Requires precomputed frobenius powers supplied by
    ``fmpz_mod_poly_frobenius_powers_2exp_precomp``.

    If `m == 0` and `f` has degree `0` or `1`, this performs a division.
    However an impossible inverse by the leading coefficient of `f` will have
    been caught by ``fmpz_mod_poly_frobenius_powers_2exp_precomp``.

.. function:: void fmpz_mod_poly_frobenius_powers_precomp(fmpz_mod_poly_frobenius_powers_t pow, const fmpz_mod_poly_t f, const fmpz_mod_poly_t finv, ulong m, const fmpz_mod_ctx_t ctx)

    If ``p = f->p``, compute `x^{(p^0)}`, `x^{(p^1)}`, `x^{(p^2)}`, `x^{(p^3)}`,
    ..., `x^{(p^m)} \pmod{f}`.

    Requires precomputed inverse of `f`, i.e. newton inverse.

.. function:: void fmpz_mod_poly_frobenius_powers_clear(fmpz_mod_poly_frobenius_powers_t pow, const fmpz_mod_ctx_t ctx)

    Clear resources used by the ``fmpz_mod_poly_frobenius_powers_t``
    struct.


Division
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_divrem_basecase(fmpz * Q, fmpz * R, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    Computes ``(Q, lenA - lenB + 1)``, ``(R, lenA)`` such that
    `A = B Q + R` with `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`.

    Assumes that the leading coefficient of `B` is invertible
    modulo `p`, and that ``invB`` is the inverse.

    Assumes that `\operatorname{len}(A), \operatorname{len}(B) > 0`.  Allows zero-padding in
    ``(A, lenA)``.  `R` and `A` may be aliased, but apart from this no
    aliasing of input and output operands is allowed.

.. function:: void fmpz_mod_poly_divrem_basecase(fmpz_mod_poly_t Q, fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Computes `Q`, `R` such that `A = B Q + R` with
    `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`.

    Assumes that the leading coefficient of `B` is invertible
    modulo `p`.

.. function:: void _fmpz_mod_poly_divrem_newton_n_preinv (fmpz * Q, fmpz * R, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz * Binv, slong lenBinv, const fmpz_mod_ctx_t ctx)

    Computes `Q` and `R` such that `A = BQ + R` with `\operatorname{len}(R)` less than
    ``lenB``, where `A` is of length ``lenA`` and `B` is of length
    ``lenB``. We require that `Q` have space for ``lenA - lenB + 1``
    coefficients. Furthermore, we assume that `Binv` is the inverse of the
    reverse of `B` mod `x^{\operatorname{len}(B)}`. The algorithm used is to call
    :func:`div_newton_n_preinv` and then multiply out and compute
    the remainder.

.. function:: void fmpz_mod_poly_divrem_newton_n_preinv(fmpz_mod_poly_t Q, fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_poly_t Binv, const fmpz_mod_ctx_t ctx)

    Computes `Q` and `R` such that `A = BQ + R` with `\operatorname{len}(R) < \operatorname{len}(B)`.
    We assume `Binv` is the inverse of the reverse of `B` mod `x^{\operatorname{len}(B)}`.

    It is required that the length of `A` is less than or equal to
    2*the length of `B` - 2.

    The algorithm used is to call :func:`div_newton_n` and then multiply out
    and compute the remainder.

.. function:: void _fmpz_mod_poly_div_newton_n_preinv (fmpz * Q, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz * Binv, slong lenBinv, const fmpz_mod_ctx_t ctx)

    Notionally computes polynomials `Q` and `R` such that `A = BQ + R` with
    `\operatorname{len}(R)` less than ``lenB``, where ``A`` is of length ``lenA``
    and ``B`` is of length ``lenB``, but return only `Q`.

    We require that `Q` have space for ``lenA - lenB + 1`` coefficients
    and assume that the leading coefficient of `B` is a unit. Furthermore, we
    assume that `Binv` is the inverse of the reverse of `B` mod `x^{\operatorname{len}(B)}`.

    The algorithm used is to reverse the polynomials and divide the
    resulting power series, then reverse the result.

.. function:: void fmpz_mod_poly_div_newton_n_preinv(fmpz_mod_poly_t Q, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_poly_t Binv, const fmpz_mod_ctx_t ctx)

    Notionally computes `Q` and `R` such that `A = BQ + R` with
    `\operatorname{len}(R) < \operatorname{len}(B)`, but returns only `Q`.

    We assume that the leading coefficient of `B` is a unit and that `Binv` is
    the inverse of the reverse of `B` mod `x^{\operatorname{len}(B)}`.

    It is required that the length of `A` is less than or equal to
    2*the length of `B` - 2.

    The algorithm used is to reverse the polynomials and divide the
    resulting power series, then reverse the result.

.. function:: ulong fmpz_mod_poly_remove(fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_ctx_t ctx)

    Removes the highest possible power of ``g`` from ``f`` and
    returns the exponent.

.. function:: void _fmpz_mod_poly_rem_basecase(fmpz * R, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    Notationally, computes `Q`, `R` such that `A = B Q + R` with
    `0 \leq \operatorname{len}(R) < \operatorname{len}(B)` but only sets ``(R, lenB - 1)``.

    Allows aliasing only between `A` and `R`.  Allows zero-padding
    in `A` but not in `B`.  Assumes that the leading coefficient
    of `B` is a unit modulo `p`.

.. function:: void fmpz_mod_poly_rem_basecase(fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Notationally, computes `Q`, `R` such that `A = B Q + R` with
    `0 \leq \operatorname{len}(R) < \operatorname{len}(B)` assuming that the leading term
    of `B` is a unit.

.. function:: void _fmpz_mod_poly_div(fmpz * Q, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    Notationally, computes `Q`, `R` such that `A = B Q + R` with
    `0 \leq \operatorname{len}(R) < \operatorname{len}(B)` but only sets ``(Q, lenA - lenB + 1)``.

    Assumes that the leading coefficient of `B` is a unit modulo `p`.

.. function:: void fmpz_mod_poly_div(fmpz_mod_poly_t Q, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Notationally, computes `Q`, `R` such that `A = B Q + R` with
    `0 \leq \operatorname{len}(R) < \operatorname{len}(B)` assuming that the leading term
    of `B` is a unit.

.. function:: void _fmpz_mod_poly_divrem(fmpz * Q, fmpz * R, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    Computes ``(Q, lenA - lenB + 1)``, ``(R, lenB - 1)`` such that
    `A = B Q + R` and `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`.

    Assumes that `B` is non-zero, that the leading coefficient
    of `B` is invertible modulo `p` and that ``invB`` is
    the inverse.

    Assumes `\operatorname{len}(A) \geq \operatorname{len}(B) > 0`.  Allows zero-padding in
    ``(A, lenA)``.  No aliasing of input and output operands is
    allowed.

.. function:: void fmpz_mod_poly_divrem(fmpz_mod_poly_t Q, fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Computes `Q`, `R` such that `A = B Q + R` and
    `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`.

    Assumes that `B` is non-zero and that the leading coefficient
    of `B` is invertible modulo `p`.

.. function:: void fmpz_mod_poly_divrem_f(fmpz_t f, fmpz_mod_poly_t Q, fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Either finds a non-trivial factor~`f` of the modulus~`p`, or computes
    `Q`, `R` such that `A = B Q + R` and `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`.

    If the leading coefficient of `B` is invertible in `\mathbf{Z}/(p)`,
    the division with remainder operation is carried out, `Q` and `R` are
    computed correctly, and `f` is set to `1`.  Otherwise, `f` is set to
    a non-trivial factor of `p` and `Q` and `R` are not touched.

    Assumes that `B` is non-zero.

.. function:: void _fmpz_mod_poly_rem(fmpz * R, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    Notationally, computes ``(Q, lenA - lenB + 1)``, ``(R, lenB - 1)``
    such that `A = B Q + R` and `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`, returning
    only the remainder part.

    Assumes that `B` is non-zero, that the leading coefficient
    of `B` is invertible modulo `p` and that ``invB`` is
    the inverse.

    Assumes `\operatorname{len}(A) \geq \operatorname{len}(B) > 0`.  Allows zero-padding in
    ``(A, lenA)``.  No aliasing of input and output operands is
    allowed.

.. function:: void fmpz_mod_poly_rem_f(fmpz_t f, fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    If `f` returns with the value `1` then the function operates as
    ``_fmpz_mod_poly_rem``, otherwise `f` will be set to a nontrivial
    factor of `p`.

.. function:: void fmpz_mod_poly_rem(fmpz_mod_poly_t R, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Notationally, computes `Q`, `R` such that `A = B Q + R`
    and `0 \leq \operatorname{len}(R) < \operatorname{len}(B)`, returning only the remainder
    part.

    Assumes that `B` is non-zero and that the leading coefficient
    of `B` is invertible modulo `p`.


Divisibility testing
--------------------------------------------------------------------------------


.. function:: int _fmpz_mod_poly_divides_classical(fmpz * Q, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    Returns `1` if `(B, lenB)` divides `(A, lenA)` and sets
    `(Q, lenA - lenB + 1)` to the quotient. Otherwise, returns `0` and sets
    `(Q, lenA - lenB + 1)` to zero. We require that `lenA >= lenB > 0`.

.. function:: int fmpz_mod_poly_divides_classical(fmpz_mod_poly_t Q, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Returns `1` if `B` divides `A` and sets `Q` to the quotient. Otherwise
    returns `0` and sets `Q` to zero.

.. function:: int _fmpz_mod_poly_divides(fmpz * Q, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    Returns `1` if `(B, lenB)` divides `(A, lenA)` and sets
    `(Q, lenA - lenB + 1)` to the quotient. Otherwise, returns `0` and sets
    `(Q, lenA - lenB + 1)` to zero. We require that `lenA >= lenB > 0`.

.. function:: int fmpz_mod_poly_divides(fmpz_mod_poly_t Q, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Returns `1` if `B` divides `A` and sets `Q` to the quotient. Otherwise
    returns `0` and sets `Q` to zero.

Power series inversion
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_inv_series(fmpz * Qinv, const fmpz * Q, slong Qlen, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``(Qinv, n)`` to the inverse of ``(Q, n)`` modulo `x^n`,
    where `n \geq 1`, assuming that the bottom coefficient of `Q` is
    invertible modulo `p` and that its inverse is ``cinv``.

.. function:: void fmpz_mod_poly_inv_series(fmpz_mod_poly_t Qinv, const fmpz_mod_poly_t Q, slong n, const fmpz_mod_ctx_t ctx)

    Sets ``Qinv`` to the inverse of ``Q`` modulo `x^n`,
    where `n \geq 1`, assuming that the bottom coefficient of
    `Q` is a unit.

.. function:: void fmpz_mod_poly_inv_series_f(fmpz_t f, fmpz_mod_poly_t Qinv, const fmpz_mod_poly_t Q, slong n, const fmpz_mod_ctx_t ctx)

    Either sets `f` to a nontrivial factor of `p` with the value of
    ``Qinv`` undefined, or sets ``Qinv`` to the inverse of ``Q``
    modulo `x^n`, where `n \geq 1`.


Power series division
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_div_series(fmpz * Q, const fmpz * A, slong Alen, const fmpz * B, slong Blen, slong n, const fmpz_mod_ctx_t ctx)

    Set ``(Q, n)`` to the quotient of the series ``(A, Alen``) and
    ``(B, Blen)`` assuming ``Alen, Blen <= n``. We assume the bottom
    coefficient of ``B`` is invertible modulo `p`.

.. function:: void fmpz_mod_poly_div_series(fmpz_mod_poly_t Q, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, slong n, const fmpz_mod_ctx_t ctx)

    Set `Q` to the quotient of the series `A` by `B`, thinking of the series as
    though they were of length `n`. We assume that the bottom coefficient of
    `B` is a unit.


Greatest common divisor
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_make_monic(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    If ``poly`` is non-zero, sets ``res`` to ``poly`` divided
    by its leading coefficient.  This assumes that the leading coefficient
    of ``poly`` is invertible modulo `p`.

    Otherwise, if ``poly`` is zero, sets ``res`` to zero.

.. function:: void fmpz_mod_poly_make_monic_f(fmpz_t f, fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Either set `f` to `1` and ``res`` to ``poly`` divided by its leading
    coefficient or set `f` to a nontrivial factor of `p` and leave ``res``
    undefined.

.. function:: slong _fmpz_mod_poly_gcd(fmpz * G, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    Sets `G` to the greatest common divisor of `(A, \operatorname{len}(A))`
    and `(B, \operatorname{len}(B))` and returns its length.

    Assumes that `\operatorname{len}(A) \geq \operatorname{len}(B) > 0` and that the vector `G` has
    space for sufficiently many coefficients.

    Assumes that ``invB`` is the inverse of the leading coefficients
    of `B` modulo the prime number `p`.

.. function:: void fmpz_mod_poly_gcd(fmpz_mod_poly_t G, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Sets `G` to the greatest common divisor of `A` and `B`.

    In general, the greatest common divisor is defined in the polynomial
    ring `(\mathbf{Z}/(p \mathbf{Z}))[X]` if and only if `p` is a prime
    number.  Thus, this function assumes that `p` is prime.

.. function:: slong _fmpz_mod_poly_gcd_euclidean_f(fmpz_t f, fmpz * G, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    Either sets `f = 1` and `G` to the greatest common divisor
    of `(A, \operatorname{len}(A))` and `(B, \operatorname{len}(B))` and returns its length,
    or sets `f \in (1,p)` to a non-trivial factor of `p` and
    leaves the contents of the vector `(G, lenB)` undefined.

    Assumes that `\operatorname{len}(A) \geq \operatorname{len}(B) > 0` and that the vector `G` has
    space for sufficiently many coefficients.

    Does not support aliasing of any of the input arguments
    with any of the output argument.

.. function:: void fmpz_mod_poly_gcd_euclidean_f(fmpz_t f, fmpz_mod_poly_t G, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Either sets `f = 1` and `G` to the greatest common divisor
    of `A` and `B`, or ` \in (1,p)` to a non-trivial factor of `p`.

    In general, the greatest common divisor is defined in the polynomial
    ring `(\mathbf{Z}/(p \mathbf{Z}))[X]` if and only if `p` is a prime
    number.

.. function:: slong _fmpz_mod_poly_gcd_f(fmpz_t f, fmpz * G, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    Either sets `f = 1` and `G` to the greatest common divisor
    of `(A, \operatorname{len}(A))` and `(B, \operatorname{len}(B))` and returns its length,
    or sets `f \in (1,p)` to a non-trivial factor of `p` and
    leaves the contents of the vector `(G, lenB)` undefined.

    Assumes that `\operatorname{len}(A) \geq \operatorname{len}(B) > 0` and that the vector `G` has
    space for sufficiently many coefficients.

    Does not support aliasing of any of the input arguments
    with any of the output arguments.

.. function:: void fmpz_mod_poly_gcd_f(fmpz_t f, fmpz_mod_poly_t G, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Either sets `f = 1` and `G` to the greatest common divisor
    of `A` and `B`, or `f \in (1,p)` to a non-trivial factor of `p`.

    In general, the greatest common divisor is defined in the polynomial
    ring `(\mathbf{Z}/(p \mathbf{Z}))[X]` if and only if `p` is a prime
    number.

.. function:: slong _fmpz_mod_poly_hgcd(fmpz **M, slong * lenM, fmpz * A, slong * lenA, fmpz * B, slong * lenB, const fmpz * a, slong lena, const fmpz * b, slong lenb, const fmpz_mod_ctx_t ctx)

    Computes the HGCD of `a` and `b`, that is, a matrix~`M`, a sign~`\sigma`
    and two polynomials `A` and `B` such that

    .. math::

        (A,B)^t = \sigma M^{-1} (a,b)^t.

    Assumes that `\operatorname{len}(a) > \operatorname{len}(b) > 0`.

    Assumes that `A` and `B` have space of size at least `\operatorname{len}(a)`
    and `\operatorname{len}(b)`, respectively.  On exit, ``*lenA`` and ``*lenB``
    will contain the correct lengths of `A` and `B`.

    Assumes that ``M[0]``, ``M[1]``, ``M[2]``, and ``M[3]``
    each point to a vector of size at least `\operatorname{len}(a)`.

.. function:: slong _fmpz_mod_poly_xgcd_euclidean_f(fmpz_t f, fmpz * G, fmpz * S, fmpz * T, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    If `f` returns with the value `1` then the function operates as per
    ``_fmpz_mod_poly_xgcd_euclidean``, otherwise `f` is set to a nontrivial
    factor of `p`.

.. function:: void fmpz_mod_poly_xgcd_euclidean_f(fmpz_t f, fmpz_mod_poly_t G, fmpz_mod_poly_t S, fmpz_mod_poly_t T, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    If `f` returns with the value `1` then the function operates as per
    ``fmpz_mod_poly_xgcd_euclidean``, otherwise `f` is set to a nontrivial
    factor of `p`.

.. function:: slong _fmpz_mod_poly_xgcd(fmpz * G, fmpz * S, fmpz * T, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invB, const fmpz_mod_ctx_t ctx)

    Computes the GCD of `A` and `B` together with cofactors `S` and `T`
    such that `S A + T B = G`.  Returns the length of `G`.

    Assumes that `\operatorname{len}(A) \geq \operatorname{len}(B) \geq 1` and
    `(\operatorname{len}(A),\operatorname{len}(B)) \neq (1,1)`.

    No attempt is made to make the GCD monic.

    Requires that `G` have space for `\operatorname{len}(B)` coefficients.  Writes
    `\operatorname{len}(B)-1` and `\operatorname{len}(A)-1` coefficients to `S` and `T`, respectively.
    Note that, in fact, `\operatorname{len}(S) \leq \max(\operatorname{len}(B) - \operatorname{len}(G), 1)` and
    `\operatorname{len}(T) \leq \max(\operatorname{len}(A) - \operatorname{len}(G), 1)`.

    No aliasing of input and output operands is permitted.

.. function:: void fmpz_mod_poly_xgcd(fmpz_mod_poly_t G, fmpz_mod_poly_t S, fmpz_mod_poly_t T, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Computes the GCD of `A` and `B`. The GCD of zero polynomials is
    defined to be zero, whereas the GCD of the zero polynomial and some other
    polynomial `P` is defined to be `P`. Except in the case where
    the GCD is zero, the GCD `G` is made monic.

    Polynomials ``S`` and ``T`` are computed such that
    ``S*A + T*B = G``. The length of ``S`` will be at most
    ``lenB`` and the length of ``T`` will be at most ``lenA``.

.. function:: void fmpz_mod_poly_xgcd_f(fmpz_t f, fmpz_mod_poly_t G, fmpz_mod_poly_t S, fmpz_mod_poly_t T, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    If `f` returns with the value `1` then the function operates as per
    ``fmpz_mod_poly_xgcd``, otherwise `f` is set to a nontrivial
    factor of `p`.

.. function:: slong _fmpz_mod_poly_gcdinv_euclidean(fmpz * G, fmpz * S, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invA, const fmpz_mod_ctx_t ctx)

    Computes ``(G, lenA)``, ``(S, lenB-1)`` such that
    `G \cong S A \pmod{B}`, returning the actual length of `G`.

    Assumes that `0 < \operatorname{len}(A) < \operatorname{len}(B)`.

.. function:: void fmpz_mod_poly_gcdinv_euclidean(fmpz_mod_poly_t G, fmpz_mod_poly_t S, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Computes polynomials `G` and `S`, both reduced modulo~`B`,
    such that `G \cong S A \pmod{B}`, where `B` is assumed to
    have `\operatorname{len}(B) \geq 2`.

    In the case that `A = 0 \pmod{B}`, returns `G = S = 0`.

.. function:: slong _fmpz_mod_poly_gcdinv_euclidean_f(fmpz_t f, fmpz * G, fmpz * S, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_t invA, const fmpz_mod_ctx_t ctx)

    If `f` returns with value `1` then the function operates as per
    :func:`_fmpz_mod_poly_gcdinv_euclidean`, otherwise `f` is set to a
    nontrivial factor of `p`.

.. function:: void fmpz_mod_poly_gcdinv_euclidean_f(fmpz_t f, fmpz_mod_poly_t G, fmpz_mod_poly_t S, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    If `f` returns with value `1` then the function operates as per
    :func:`fmpz_mod_poly_gcdinv_euclidean`, otherwise `f` is set to a
    nontrivial factor of the modulus of `A`.

.. function:: slong _fmpz_mod_poly_gcdinv(fmpz * G, fmpz * S, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    Computes ``(G, lenA)``, ``(S, lenB-1)`` such that
    `G \cong S A \pmod{B}`, returning the actual length of `G`.

    Assumes that `0 < \operatorname{len}(A) < \operatorname{len}(B)`.

.. function:: slong _fmpz_mod_poly_gcdinv_f(fmpz_t f, fmpz * G, fmpz * S, const fmpz * A, slong lenA, const fmpz * B, slong lenB, const fmpz_mod_ctx_t ctx)

    If `f` returns with value `1` then the function operates as per
    :func:`_fmpz_mod_poly_gcdinv`, otherwise `f` will be set to a nontrivial
    factor of `p`.

.. function:: void fmpz_mod_poly_gcdinv(fmpz_mod_poly_t G, fmpz_mod_poly_t S, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    Computes polynomials `G` and `S`, both reduced modulo~`B`,
    such that `G \cong S A \pmod{B}`, where `B` is assumed to
    have `\operatorname{len}(B) \geq 2`.

    In the case that `A = 0 \pmod{B}`, returns `G = S = 0`.

.. function:: void fmpz_mod_poly_gcdinv_f(fmpz_t f, fmpz_mod_poly_t G, fmpz_mod_poly_t S, const fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_ctx_t ctx)

    If `f` returns with value `1` then the function operates as per
    :func:`fmpz_mod_poly_gcdinv`, otherwise `f` will be set to a nontrivial
    factor of `p`.

.. function:: int _fmpz_mod_poly_invmod(fmpz * A, const fmpz * B, slong lenB, const fmpz * P, slong lenP, const fmpz_mod_ctx_t ctx)

    Attempts to set ``(A, lenP-1)`` to the inverse of ``(B, lenB)``
    modulo the polynomial ``(P, lenP)``.  Returns `1` if ``(B, lenB)``
    is invertible and `0` otherwise.

    Assumes that `0 < \operatorname{len}(B) < \operatorname{len}(P)`, and hence also `\operatorname{len}(P) \geq 2`,
    but supports zero-padding in ``(B, lenB)``.

    Does not support aliasing.

    Assumes that `p` is a prime number.

.. function:: int _fmpz_mod_poly_invmod_f(fmpz_t f, fmpz * A, const fmpz * B, slong lenB, const fmpz * P, slong lenP, const fmpz_mod_ctx_t ctx)

    If `f` returns with the value `1`, then the function operates as per
    :func:`_fmpz_mod_poly_invmod`. Otherwise `f` is set to a nontrivial
    factor of `p`.

.. function:: int fmpz_mod_poly_invmod(fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_poly_t P, const fmpz_mod_ctx_t ctx)

    Attempts to set `A` to the inverse of `B` modulo `P` in the polynomial
    ring `(\mathbf{Z}/p\mathbf{Z})[X]`, where we assume that `p` is a prime
    number.

    If `\deg(P) < 2`, raises an exception.

    If the greatest common divisor of `B` and `P` is~`1`, returns~`1` and
    sets `A` to the inverse of `B`.  Otherwise, returns~`0` and the value
    of `A` on exit is undefined.

.. function:: int fmpz_mod_poly_invmod_f(fmpz_t f, fmpz_mod_poly_t A, const fmpz_mod_poly_t B, const fmpz_mod_poly_t P, const fmpz_mod_ctx_t ctx)

    If `f` returns with the value `1`, then the function operates as per
    :func:`fmpz_mod_poly_invmod`. Otherwise `f` is set to a nontrivial
    factor of `p`.


Minpoly
--------------------------------------------------------------------------------


.. function:: slong _fmpz_mod_poly_minpoly_bm(fmpz * poly, const fmpz * seq, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to the coefficients of a minimal generating
    polynomial for sequence ``(seq, len)`` modulo `p`.

    The return value equals the length of ``poly``.

    It is assumed that `p` is prime and ``poly`` has space for at least
    `len+1` coefficients. No aliasing between inputs and outputs is
    allowed.

.. function:: void fmpz_mod_poly_minpoly_bm(fmpz_mod_poly_t poly, const fmpz * seq, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to a minimal generating polynomial for sequence
    ``seq`` of length ``len``.

    Assumes that the modulus is prime.

    This version uses the Berlekamp-Massey algorithm, whose running time
    is proportional to ``len`` times the size of the minimal generator.

.. function:: slong _fmpz_mod_poly_minpoly_hgcd(fmpz * poly, const fmpz * seq, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to the coefficients of a minimal generating
    polynomial for sequence ``(seq, len)`` modulo `p`.

    The return value equals the length of ``poly``.

    It is assumed that `p` is prime and ``poly`` has space for at least
    `len+1` coefficients. No aliasing between inputs and outputs is
    allowed.

.. function:: void fmpz_mod_poly_minpoly_hgcd(fmpz_mod_poly_t poly, const fmpz * seq, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to a minimal generating polynomial for sequence
    ``seq`` of length ``len``.

    Assumes that the modulus is prime.

    This version uses the HGCD algorithm, whose running time is
    `O(n \log^2 n)` field operations, regardless of the actual size of
    the minimal generator.

.. function:: slong _fmpz_mod_poly_minpoly(fmpz * poly, const fmpz * seq, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to the coefficients of a minimal generating
    polynomial for sequence ``(seq, len)`` modulo `p`.

    The return value equals the length of ``poly``.

    It is assumed that `p` is prime and ``poly`` has space for at least
    `len+1` coefficients. No aliasing between inputs and outputs is
    allowed.

.. function:: void fmpz_mod_poly_minpoly(fmpz_mod_poly_t poly, const fmpz * seq, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``poly`` to a minimal generating polynomial for sequence
    ``seq`` of length ``len``.

    A minimal generating polynomial is a monic polynomial
    `f = x^d + c_{d-1}x^{d-1} + \cdots + c_1 x + c_0`,
    of minimal degree `d`, that annihilates any consecutive `d+1` terms
    in ``seq``. That is, for any `i < len - d`,

    `seq_i = -\sum_{j=0}^{d-1} seq_{i+j}*f_j.`

    Assumes that the modulus is prime.

    This version automatically chooses the fastest underlying
    implementation based on ``len`` and the size of the modulus.



Resultant
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_resultant(fmpz_t res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz_mod_ctx_t ctx)

    Returns the resultant of ``(poly1, len1)`` and
    ``(poly2, len2)``.

    Assumes that ``len1 >= len2 > 0``.

    The complexity is only guaranteed to be quasilinear if the modulus is prime.

.. function:: void fmpz_mod_poly_resultant(fmpz_t res, const fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_ctx_t ctx)

    Computes the resultant of $f$ and $g$.


Discriminant
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_discriminant(fmpz_t d, const fmpz * poly, slong len, const fmpz_mod_ctx_t ctx)

    Set `d` to the discriminant of ``(poly, len)``. Assumes ``len > 1``.

.. function:: void fmpz_mod_poly_discriminant(fmpz_t d, const fmpz_mod_poly_t f, const fmpz_mod_ctx_t ctx)

    Set `d` to the discriminant of `f`.
    We normalise the discriminant so that
    `\operatorname{disc}(f) = (-1)^(n(n-1)/2) \operatorname{res}(f, f') /
    \operatorname{lc}(f)^(n - m - 2)`, where ``n = len(f)`` and
    ``m = len(f')``. Thus `\operatorname{disc}(f) =
    \operatorname{lc}(f)^(2n - 2) \prod_{i < j} (r_i - r_j)^2`, where
    `\operatorname{lc}(f)` is the leading coefficient of `f` and `r_i` are the
    roots of `f`.


Derivative
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_derivative(fmpz * res, const fmpz * poly, slong len, const fmpz_mod_ctx_t ctx)

    Sets ``(res, len - 1)`` to the derivative of ``(poly, len)``.
    Also handles the cases where ``len`` is `0` or `1` correctly.
    Supports aliasing of ``res`` and ``poly``.

.. function:: void fmpz_mod_poly_derivative(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the derivative of ``poly``.


Evaluation
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_evaluate_fmpz(fmpz_t res, const fmpz * poly, slong len, const fmpz_t a, const fmpz_mod_ctx_t ctx)

    Evaluates the polynomial ``(poly, len)`` at the integer `a` and sets
    ``res`` to the result.  Aliasing between ``res`` and `a` or any
    of the coefficients of ``poly`` is not supported.

.. function:: void fmpz_mod_poly_evaluate_fmpz(fmpz_t res, const fmpz_mod_poly_t poly, const fmpz_t a, const fmpz_mod_ctx_t ctx)

    Evaluates the polynomial ``poly`` at the integer `a` and sets
    ``res`` to the result.

    As expected, aliasing between ``res`` and `a` is supported.  However,
    ``res`` may not be aliased with a coefficient of ``poly``.


Multipoint evaluation
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_evaluate_fmpz_vec_iter(fmpz * ys, const fmpz * coeffs, slong len, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Evaluates (``coeffs``, ``len``) at the ``n`` values
    given in the vector ``xs``, writing the output values
    to ``ys``. The values in ``xs`` should be reduced
    modulo the modulus.

    Uses Horner's method iteratively.

.. function:: void fmpz_mod_poly_evaluate_fmpz_vec_iter(fmpz * ys, const fmpz_mod_poly_t poly, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Evaluates ``poly`` at the ``n`` values given in the vector
    ``xs``, writing the output values to ``ys``. The values in
    ``xs`` should be reduced modulo the modulus.

    Uses Horner's method iteratively.

.. function:: void _fmpz_mod_poly_evaluate_fmpz_vec_fast_precomp(fmpz * vs, const fmpz * poly, slong plen, fmpz_poly_struct * const * tree, slong len, const fmpz_mod_ctx_t ctx)

    Evaluates (``poly``, ``plen``) at the ``len`` values given by the precomputed subproduct tree ``tree``.

.. function:: void _fmpz_mod_poly_evaluate_fmpz_vec_fast(fmpz * ys, const fmpz * poly, slong plen, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Evaluates (``coeffs``, ``len``) at the ``n`` values
    given in the vector ``xs``, writing the output values
    to ``ys``. The values in ``xs`` should be reduced
    modulo the modulus.

    Uses fast multipoint evaluation, building a temporary subproduct tree.

.. function:: void fmpz_mod_poly_evaluate_fmpz_vec_fast(fmpz * ys, const fmpz_mod_poly_t poly, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Evaluates ``poly`` at the ``n`` values given in the vector
    ``xs``, writing the output values to ``ys``. The values in
    ``xs`` should be reduced modulo the modulus.

    Uses fast multipoint evaluation, building a temporary subproduct tree.

.. function:: void _fmpz_mod_poly_evaluate_fmpz_vec(fmpz * ys, const fmpz * coeffs, slong len, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Evaluates (``coeffs``, ``len``) at the ``n`` values
    given in the vector ``xs``, writing the output values
    to ``ys``. The values in ``xs`` should be reduced
    modulo the modulus.

.. function:: void fmpz_mod_poly_evaluate_fmpz_vec(fmpz * ys, const fmpz_mod_poly_t poly, const fmpz * xs, slong n, const fmpz_mod_ctx_t ctx)

    Evaluates ``poly`` at the ``n`` values given in the vector
    ``xs``, writing the output values to ``ys``. The values in
    ``xs`` should be reduced modulo the modulus.


Composition
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_compose(fmpz * res, const fmpz * poly1, slong len1, const fmpz * poly2, slong len2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition of ``(poly1, len1)`` and
    ``(poly2, len2)``.

    Assumes that ``res`` has space for ``(len1-1)*(len2-1) + 1``
    coefficients, although in `\mathbf{Z}_p[X]` this might not actually
    be the length of the resulting polynomial when `p` is not a prime.

    Assumes that ``poly1`` and ``poly2`` are non-zero polynomials.
    Does not support aliasing between any of the inputs and the output.

.. function:: void fmpz_mod_poly_compose(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition of ``poly1`` and ``poly2``.

    To be precise about the order of composition, denoting ``res``,
    ``poly1``, and ``poly2`` by `f`, `g`, and `h`, respectively,
    sets `f(t) = g(h(t))`.



Square roots
--------------------------------------------------------------------------------

The series expansions for `\sqrt{h}` and `1/\sqrt{h}` are defined
by means of the generalised binomial theorem
``h^r = (1+y)^r =
\sum_{k=0}^{\infty} {r \choose k} y^k.``
It is assumed that `h` has constant term `1` and that the coefficients
`2^{-k}` exist in the coefficient ring (i.e. `2` must be invertible).

.. function:: void _fmpz_mod_poly_invsqrt_series(fmpz * g, const fmpz * h, slong hlen, slong n, const fmpz_mod_ctx_t ctx)

    Set the first `n` terms of `g` to the series expansion of `1/\sqrt{h}`.
    It is assumed that `n > 0` and `h > 0`. Aliasing is not permitted.

.. function:: void fmpz_mod_poly_invsqrt_series(fmpz_mod_poly_t g, const fmpz_mod_poly_t h, slong n, const fmpz_mod_ctx_t ctx)

    Set `g` to the series expansion of `1/\sqrt{h}` to order `O(x^n)`.
    It is assumed that `h` has constant term 1.

.. function:: void _fmpz_mod_poly_sqrt_series(fmpz * g, const fmpz * h, slong hlen, slong n, const fmpz_mod_ctx_t ctx)

    Set the first `n` terms of `g` to the series expansion of `\sqrt{h}`.
    It is assumed that `n > 0` and `h > 0`. Aliasing is not permitted.

.. function:: void fmpz_mod_poly_sqrt_series(fmpz_mod_poly_t g, const fmpz_mod_poly_t h, slong n, const fmpz_mod_ctx_t ctx)

    Set `g` to the series expansion of `\sqrt{h}` to order `O(x^n)`.
    It is assumed that `h` has constant term 1.

.. function:: int _fmpz_mod_poly_sqrt(fmpz * s, const fmpz * p, slong n, const fmpz_mod_ctx_t ctx)

    If ``(p, n)`` is a perfect square, sets ``(s, n / 2 + 1)``
    to a square root of `p` and returns 1. Otherwise returns 0.

.. function:: int fmpz_mod_poly_sqrt(fmpz_mod_poly_t s, const fmpz_mod_poly_t p, const fmpz_mod_ctx_t ctx)

    If `p` is a perfect square, sets `s` to a square root of `p`
    and returns 1. Otherwise returns 0.


Modular composition
--------------------------------------------------------------------------------


.. function:: void _fmpz_mod_poly_compose_mod(fmpz * res, const fmpz * f, slong lenf, const fmpz * g, const fmpz * h, slong lenh, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero and that the length of `g` is one less than the
    length of `h` (possibly with zero padding). The output is not allowed
    to be aliased with any of the inputs.

.. function:: void fmpz_mod_poly_compose_mod(fmpz_mod_poly_t res, const fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_poly_t h, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero.

.. function:: void _fmpz_mod_poly_compose_mod_horner(fmpz * res, const fmpz * f, slong lenf, const fmpz * g, const fmpz * h, slong lenh, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero and that the length of `g` is one less than the
    length of `h` (possibly with zero padding). The output is not allowed
    to be aliased with any of the inputs.

    The algorithm used is Horner's rule.

.. function:: void fmpz_mod_poly_compose_mod_horner(fmpz_mod_poly_t res, const fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_poly_t h, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero. The algorithm used is Horner's rule.

.. function:: void _fmpz_mod_poly_compose_mod_brent_kung(fmpz * res, const fmpz * f, slong len1, const fmpz * g, const fmpz * h, slong len3, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero and that the length of `g` is one less than the
    length of `h` (possibly with zero padding). We also require that
    the length of `f` is less than the length of `h`. The output is not
    allowed to be aliased with any of the inputs.

    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void fmpz_mod_poly_compose_mod_brent_kung(fmpz_mod_poly_t res, const fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_poly_t h, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero and that `f` has smaller degree than `h`.
    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void _fmpz_mod_poly_reduce_matrix_mod_poly (fmpz_mat_t A, const fmpz_mat_t B, const fmpz_mod_poly_t f, const fmpz_mod_ctx_t ctx)

    Sets the ith row of ``A`` to the reduction of the ith row of `B` modulo
    `f` for `i=1,\ldots,\sqrt{\deg(f)}`. We require `B` to be at least
    a `\sqrt{\deg(f)}\times \deg(f)` matrix and `f` to be nonzero.

.. function:: void _fmpz_mod_poly_precompute_matrix_worker(void * arg_ptr)

    Worker function version of ``_fmpz_mod_poly_precompute_matrix``.
    Input/output is stored in ``fmpz_mod_poly_matrix_precompute_arg_t``.

.. function:: void _fmpz_mod_poly_precompute_matrix (fmpz_mat_t A, const fmpz * f, const fmpz * g, slong leng, const fmpz * ginv, slong lenginv, const fmpz_mod_ctx_t ctx)

    Sets the ith row of ``A`` to `f^i` modulo `g` for
    `i=1,\ldots,\sqrt{\deg(g)}`. We require `A` to be
    a `\sqrt{\deg(g)}\times \deg(g)` matrix. We require
    ``ginv`` to be the inverse of the reverse of ``g`` and `g` to be
    nonzero. ``f`` has to be reduced modulo ``g`` and of length one less
    than ``leng`` (possibly with zero padding).

.. function:: void fmpz_mod_poly_precompute_matrix(fmpz_mat_t A, const fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_poly_t ginv, const fmpz_mod_ctx_t ctx)

    Sets the ith row of ``A`` to `f^i` modulo `g` for
    `i=1,\ldots,\sqrt{\deg(g)}`. We require `A` to be
    a `\sqrt{\deg(g)}\times \deg(g)` matrix. We require
    ``ginv`` to be the inverse of the reverse of ``g``.

.. function:: void _fmpz_mod_poly_compose_mod_brent_kung_precomp_preinv_worker(void * arg_ptr)

    Worker function version of
    :func:`_fmpz_mod_poly_compose_mod_brent_kung_precomp_preinv`.
    Input/output is stored in
    ``fmpz_mod_poly_compose_mod_precomp_preinv_arg_t``.

.. function:: void _fmpz_mod_poly_compose_mod_brent_kung_precomp_preinv(fmpz * res, const fmpz * f, slong lenf, const fmpz_mat_t A, const fmpz * h, slong lenh, const fmpz * hinv, slong lenhinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero. We require that the ith row of `A` contains `g^i` for
    `i=1,\ldots,\sqrt{\deg(h)}`, i.e. `A` is a
    `\sqrt{\deg(h)}\times \deg(h)` matrix. We also require that
    the length of `f` is less than the length of `h`. Furthermore, we require
    ``hinv`` to be the inverse of the reverse of ``h``.
    The output is not allowed to be aliased with any of the inputs.

    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void fmpz_mod_poly_compose_mod_brent_kung_precomp_preinv(fmpz_mod_poly_t res, const fmpz_mod_poly_t f, const fmpz_mat_t A, const fmpz_mod_poly_t h, const fmpz_mod_poly_t hinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that the
    ith row of `A` contains `g^i` for `i=1,\ldots,\sqrt{\deg(h)}`, i.e. `A` is
    a `\sqrt{\deg(h)}\times \deg(h)` matrix. We require that `h` is nonzero and
    that `f` has smaller degree than `h`. Furthermore, we require ``hinv``
    to be the inverse of the reverse of ``h``. This version of Brent-Kung
    modular composition is particularly useful if one has to perform several
    modular composition of the form `f(g)` modulo `h` for fixed `g` and `h`.

.. function:: void _fmpz_mod_poly_compose_mod_brent_kung_preinv(fmpz * res, const fmpz * f, slong lenf, const fmpz * g, const fmpz * h, slong lenh, const fmpz * hinv, slong lenhinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero and that the length of `g` is one less than the
    length of `h` (possibly with zero padding). We also require that
    the length of `f` is less than the length of `h`. Furthermore, we require
    ``hinv`` to be the inverse of the reverse of ``h``.
    The output is not allowed to be aliased with any of the inputs.

    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void fmpz_mod_poly_compose_mod_brent_kung_preinv(fmpz_mod_poly_t res, const fmpz_mod_poly_t f, const fmpz_mod_poly_t g, const fmpz_mod_poly_t h, const fmpz_mod_poly_t hinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f(g)` modulo `h`. We require that
    `h` is nonzero and that `f` has smaller degree than `h`. Furthermore,
    we require ``hinv`` to be the inverse of the reverse of ``h``.
    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void _fmpz_mod_poly_compose_mod_brent_kung_vec_preinv(fmpz_mod_poly_struct * res, const fmpz_mod_poly_struct * polys, slong len1, slong l, const fmpz * g, slong glen, const fmpz * h, slong lenh, const fmpz * hinv, slong lenhinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f_i(g)` modulo `h` for `1\leq i \leq l`,
    where `f_i` are the ``l`` elements of ``polys``. We require that `h` is
    nonzero and that the length of `g` is less than the length of `h`. We
    also require that the length of `f_i` is less than the length of `h`. We
    require ``res`` to have enough memory allocated to hold ``l``
    ``fmpz_mod_poly_struct``'s. The entries of ``res`` need to be initialised
    and ``l`` needs to be less than ``len1`` Furthermore, we require ``hinv``
    to be the inverse of the reverse of ``h``. The output is not allowed to be
    aliased with any of the inputs.

    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void fmpz_mod_poly_compose_mod_brent_kung_vec_preinv(fmpz_mod_poly_struct * res, const fmpz_mod_poly_struct * polys, slong len1, slong n, const fmpz_mod_poly_t g, const fmpz_mod_poly_t h, const fmpz_mod_poly_t hinv, const fmpz_mod_ctx_t ctx)

    Sets ``res`` to the composition `f_i(g)` modulo `h` for `1\leq i \leq n`
    where `f_i` are the ``n`` elements of ``polys``. We require ``res`` to
    have enough memory allocated to hold ``n`` ``fmpz_mod_poly_struct``'s.
    The entries of ``res`` need to be initialised and ``n`` needs to be less
    than ``len1``. We require that `h` is nonzero and that `f_i` and `g` have
    smaller degree than `h`. Furthermore, we require ``hinv`` to be the
    inverse of the reverse of ``h``. No aliasing of ``res`` and
    ``polys`` is allowed.
    The algorithm used is the Brent-Kung matrix algorithm.

.. function:: void _fmpz_mod_poly_compose_mod_brent_kung_vec_preinv_threaded_pool(fmpz_mod_poly_struct * res, const fmpz_mod_poly_struct * polys, slong lenpolys, slong l, const fmpz * g, slong glen, const fmpz * poly, slong len, const fmpz * polyinv, slong leninv, const fmpz_mod_ctx_t ctx, thread_pool_handle * threads, slong num_threads)

    Multithreaded version of
    :func:`_fmpz_mod_poly_compose_mod_brent_kung_vec_preinv`. Distributing the
    Horner evaluations across :func:`flint_get_num_threads` threads.

.. function:: void fmpz_mod_poly_compose_mod_brent_kung_vec_preinv_threaded_pool(fmpz_mod_poly_struct * res, const fmpz_mod_poly_struct * polys, slong len1, slong n, const fmpz_mod_poly_t g, const fmpz_mod_poly_t poly, const fmpz_mod_poly_t polyinv, const fmpz_mod_ctx_t ctx, thread_pool_handle * threads, slong num_threads)

    Multithreaded version of
    :func:`fmpz_mod_poly_compose_mod_brent_kung_vec_preinv`. Distributing the
    Horner evaluations across :func:`flint_get_num_threads` threads.

.. function:: void fmpz_mod_poly_compose_mod_brent_kung_vec_preinv_threaded(fmpz_mod_poly_struct * res, const fmpz_mod_poly_struct * polys, slong len1, slong n, const fmpz_mod_poly_t g, const fmpz_mod_poly_t poly, const fmpz_mod_poly_t polyinv, const fmpz_mod_ctx_t ctx)

    Multithreaded version of
    :func:`fmpz_mod_poly_compose_mod_brent_kung_vec_preinv`. Distributing the
    Horner evaluations across :func:`flint_get_num_threads` threads.


Subproduct trees
--------------------------------------------------------------------------------


.. function:: fmpz_poly_struct ** _fmpz_mod_poly_tree_alloc(slong len)

    Allocates space for a subproduct tree of the given length, having
    linear factors at the lowest level.

.. function:: void _fmpz_mod_poly_tree_free(fmpz_poly_struct ** tree, slong len)

    Free the allocated space for the subproduct.

.. function:: void _fmpz_mod_poly_tree_build(fmpz_poly_struct ** tree, const fmpz * roots, slong len, const fmpz_mod_ctx_t ctx)

    Builds a subproduct tree in the preallocated space from
    the ``len`` monic linear factors `(x-r_i)` where `r_i` are given by
    ``roots``. The top level product is not computed.


Radix conversion
--------------------------------------------------------------------------------

The following functions provide the functionality to solve the
radix conversion problems for polynomials, which is to express
a polynomial `f(X)` with respect to a given radix `r(X)` as

    .. math::

        f(X) = \sum_{i = 0}^{N} b_i(X) r(X)^i


where `N = \lfloor\deg(f) / \deg(r)\rfloor`.
The algorithm implemented here is a recursive one, which performs
Euclidean divisions by powers of `r` of the form `r^{2^i}`, and it
has time complexity `\Theta(\deg(f) \log \deg(f))`.
It facilitates the repeated use of precomputed data, namely the
powers of `r` and their power series inverses.  This data is stored
in objects of type ``fmpz_mod_poly_radix_t`` and it is computed
using the function :func:`fmpz_mod_poly_radix_init`, which only
depends on~`r` and an upper bound on the degree of~`f`.

.. function:: void _fmpz_mod_poly_radix_init(fmpz **Rpow, fmpz **Rinv, const fmpz * R, slong lenR, slong k, const fmpz_t invL, const fmpz_mod_ctx_t ctx)

    Computes powers of `R` of the form `R^{2^i}` and their Newton inverses
    modulo `x^{2^{i} \deg(R)}` for `i = 0, \dotsc, k-1`.

    Assumes that the vectors ``Rpow[i]`` and ``Rinv[i]`` have space
    for `2^i \deg(R) + 1` and `2^i \deg(R)` coefficients, respectively.

    Assumes that the polynomial `R` is non-constant, i.e. `\deg(R) \geq 1`.

    Assumes that the leading coefficient of `R` is a unit and that the
    argument ``invL`` is the inverse of the coefficient modulo~`p`.

    The argument~`p` is the modulus, which in `p`-adic applications is
    typically a prime power, although this is not necessary.  Here, we
    only assume that `p \geq 2`.

    Note that this precomputed data can be used for any `F` such that
    `\operatorname{len}(F) \leq 2^k \deg(R)`.

.. function:: void fmpz_mod_poly_radix_init(fmpz_mod_poly_radix_t D, const fmpz_mod_poly_t R, slong degF, const fmpz_mod_ctx_t ctx)

    Carries out the precomputation necessary to perform radix conversion
    to radix~`R` for polynomials~`F` of degree at most ``degF``.

    Assumes that `R` is non-constant, i.e. `\deg(R) \geq 1`,
    and that the leading coefficient is a unit.

.. function:: void _fmpz_mod_poly_radix(fmpz **B, const fmpz * F, fmpz **Rpow, fmpz **Rinv, slong degR, slong k, slong i, fmpz * W, const fmpz_mod_ctx_t ctx)

    This is the main recursive function used by the
    function :func:`fmpz_mod_poly_radix`.

    Assumes that, for all `i = 0, \dotsc, N`, the vector
    ``B[i]`` has space for `\deg(R)` coefficients.

    The variable `k` denotes the factors of `r` that have
    previously been counted for the polynomial `F`, which
    is assumed to have length `2^{i+1} \deg(R)`, possibly
    including zero-padding.

    Assumes that `W` is a vector providing temporary space
    of length `\operatorname{len}(F) = 2^{i+1} \deg(R)`.

    The entire computation takes place over `\mathbf{Z} / p \mathbf{Z}`,
    where `p \geq 2` is a natural number.

    Thus, the top level call will have `F` as in the original
    problem, and `k = 0`.

.. function:: void fmpz_mod_poly_radix(fmpz_mod_poly_struct **B, const fmpz_mod_poly_t F, const fmpz_mod_poly_radix_t D, const fmpz_mod_ctx_t ctx)

    Given a polynomial `F` and the precomputed data `D` for the radix `R`,
    computes polynomials `B_0, \dotsc, B_N` of degree less than `\deg(R)`
    such that

    .. math::

        F = B_0 + B_1 R + \dotsb + B_N R^N,


    where necessarily `N = \lfloor\deg(F) / \deg(R)\rfloor`.

    Assumes that `R` is non-constant, i.e.\ `\deg(R) \geq 1`,
    and that the leading coefficient is a unit.


Input and output
--------------------------------------------------------------------------------

The printing options supported by this module are very similar to
what can be found in the two related modules ``fmpz_poly`` and
``nmod_poly``.
Consider, for example, the polynomial `f(x) = 5x^3 + 2x + 1` in
`(\mathbf{Z}/6\mathbf{Z})[x]`.  Its simple string representation
is ``"4 6  1 2 0 5"``, where the first two numbers denote the
length of the polynomial and the modulus.  The pretty string
representation is ``"5*x^3+2*x+1"``.

.. function:: int _fmpz_mod_poly_fprint(FILE * file, const fmpz * poly, slong len, const fmpz_t p)

    Prints the polynomial ``(poly, len)`` to the stream ``file``.

    In case of success, returns a positive value.  In case of failure,
    returns a non-positive value.

.. function:: int fmpz_mod_poly_fprint(FILE * file, const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Prints the polynomial to the stream ``file``.

    In case of success, returns a positive value.  In case of failure,
    returns a non-positive value.

.. function:: int fmpz_mod_poly_fprint_pretty(FILE * file, const fmpz_mod_poly_t poly, const char * x, const fmpz_mod_ctx_t ctx)

    Prints the pretty representation of ``(poly, len)`` to the stream
    ``file``, using the string ``x`` to represent the indeterminate.

    In case of success, returns a positive value.  In case of failure,
    returns a non-positive value.

.. function:: int fmpz_mod_poly_print(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)

    Prints the polynomial to ``stdout``.

    In case of success, returns a positive value.  In case of failure,
    returns a non-positive value.

.. function:: int fmpz_mod_poly_print_pretty(const fmpz_mod_poly_t poly, const char * x, const fmpz_mod_ctx_t ctx)

    Prints the pretty representation of ``poly`` to ``stdout``,
    using the string ``x`` to represent the indeterminate.

    In case of success, returns a positive value.  In case of failure,
    returns a non-positive value.

Inflation and deflation
--------------------------------------------------------------------------------


.. function:: void fmpz_mod_poly_inflate(fmpz_mod_poly_t result, const fmpz_mod_poly_t input, ulong inflation, const fmpz_mod_ctx_t ctx)

    Sets ``result`` to the inflated polynomial `p(x^n)` where
    `p` is given by ``input`` and `n` is given by ``inflation``.

.. function:: void fmpz_mod_poly_deflate(fmpz_mod_poly_t result, const fmpz_mod_poly_t input, ulong deflation, const fmpz_mod_ctx_t ctx)

    Sets ``result`` to the deflated polynomial `p(x^{1/n})` where
    `p` is given by ``input`` and `n` is given by ``deflation``.
    Requires `n > 0`.

.. function:: ulong fmpz_mod_poly_deflation(const fmpz_mod_poly_t input, const fmpz_mod_ctx_t ctx)

    Returns the largest integer by which ``input`` can be deflated.
    As special cases, returns 0 if ``input`` is the zero polynomial
    and 1 of ``input`` is a constant polynomial.

Berlekamp-Massey Algorithm
--------------------------------------------------------------------------------

    The fmpz_mod_berlekamp_massey_t manages an unlimited stream of points `a_1, a_2, \dots .`
    At any point in time, after, say, `n` points have been added, a call to :func:`fmpz_mod_berlekamp_massey_reduce` will
    calculate the polynomials `U`, `V` and `R` in the extended euclidean remainder sequence with

    .. math::

        U*x^n + V*(a_1*x^{n-1} + \cdots + a_{n-1}*x + a_n) = R, \quad \deg(U) < \deg(V) \le n/2, \quad \deg(R) < n/2.

    The polynomials `V` and `R` may be obtained with :func:`fmpz_mod_berlekamp_massey_V_poly` and :func:`fmpz_mod_berlekamp_massey_R_poly`.
    This class differs from :func:`fmpz_mod_poly_minpoly` in the following respect. Let `v_i` denote the coefficient of `x^i` in `V`.
    :func:`fmpz_mod_poly_minpoly` will return a polynomial `V` of lowest degree that annihilates the whole sequence `a_1, \dots, a_n` as

    .. math::

        \sum_{i} v_i a_{j + i} = 0, \quad 1 \le j \le n - \deg(V).

    The cost is that a polynomial of degree `n-1` might be returned and the return is not generally uniquely determined by the input sequence.
    For the fmpz_mod_berlekamp_massey_t we have

    .. math::

        \sum_{i,j} v_i a_{j+i} x^{-j} = -U + \frac{R}{x^n}\text{,}

    and it can be seen that `\sum_{i} v_i a_{j + i}` is zero for `1 \le j < n - \deg(R)`. Thus whether or not `V` has annihilated the whole sequence may be checked by comparing the degrees of `V` and `R`.

.. function:: void fmpz_mod_berlekamp_massey_init(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)

    Initialize ``B`` with an empty stream.

.. function:: void fmpz_mod_berlekamp_massey_clear(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)

    Free any space used by ``B``.

.. function:: void fmpz_mod_berlekamp_massey_start_over(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)

    Empty the stream of points in ``B``.

.. function:: void fmpz_mod_berlekamp_massey_add_points(fmpz_mod_berlekamp_massey_t B, const fmpz * a, slong count, const fmpz_mod_ctx_t ctx)
              void fmpz_mod_berlekamp_massey_add_zeros(fmpz_mod_berlekamp_massey_t B, slong count, const fmpz_mod_ctx_t ctx)
              void fmpz_mod_berlekamp_massey_add_point(fmpz_mod_berlekamp_massey_t B, const fmpz_t a, const fmpz_mod_ctx_t ctx)

    Add point(s) to the stream processed by ``B``. The addition of any number of points will not update the `V` and `R` polynomial.

.. function:: int fmpz_mod_berlekamp_massey_reduce(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)

    Ensure that the polynomials `V` and `R` are up to date. The return value is ``1`` if this function changed `V` and ``0`` otherwise.
    For example, if this function is called twice in a row without adding any points in between, the return of the second call should be ``0``.
    As another example, suppose the object is emptied, the points `1, 1, 2, 3` are added, then reduce is called. This reduce should return ``1`` with `\deg(R) < \deg(V) = 2` because the Fibonacci sequence has been recognized. The further addition of the two points `5, 8` and a reduce will result in a return value of ``0``.

.. function:: slong fmpz_mod_berlekamp_massey_point_count(const fmpz_mod_berlekamp_massey_t B)

    Return the number of points stored in ``B``.

.. function:: const fmpz * fmpz_mod_berlekamp_massey_points(const fmpz_mod_berlekamp_massey_t B)

    Return a pointer the array of points stored in ``B``. This may be ``NULL`` if :func:`fmpz_mod_berlekamp_massey_point_count` returns ``0``.

.. function:: const fmpz_mod_poly_struct * fmpz_mod_berlekamp_massey_V_poly(const fmpz_mod_berlekamp_massey_t B)

    Return the polynomial ``V`` in ``B``.

.. function:: const fmpz_mod_poly_struct * fmpz_mod_berlekamp_massey_R_poly(const fmpz_mod_berlekamp_massey_t B)

    Return the polynomial ``R`` in ``B``.
