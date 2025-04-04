.. _d-mat:

**d_mat.h** -- double precision matrices
===============================================================================



Memory management
--------------------------------------------------------------------------------


.. function:: void d_mat_init(d_mat_t mat, slong rows, slong cols)

    Initialises a matrix with the given number of rows and columns for use. 

.. function:: void d_mat_clear(d_mat_t mat)

    Clears the given matrix.


Basic assignment and manipulation
--------------------------------------------------------------------------------


.. function:: void d_mat_set(d_mat_t mat1, const d_mat_t mat2)

    Sets ``mat1`` to a copy of ``mat2``. The dimensions of 
    ``mat1`` and ``mat2`` must be the same.

.. function:: void d_mat_swap_entrywise(d_mat_t mat1, d_mat_t mat2)

    Swaps two matrices by swapping the individual entries rather than swapping
    the contents of the structs.

.. function:: double d_mat_entry(d_mat_t mat, slong i, slong j)

    Returns the entry of ``mat`` at row `i` and column `j`.
    Both `i` and `j` must not exceed the dimensions of the matrix.
    This function is implemented as a macro.

.. function:: double d_mat_get_entry(const d_mat_t mat, slong i, slong j)

    Returns the entry of ``mat`` at row `i` and column `j`.
    Both `i` and `j` must not exceed the dimensions of the matrix.
    
.. function:: double * d_mat_entry_ptr(const d_mat_t mat, slong i, slong j)

    Returns a pointer to the entry of ``mat`` at row `i` and column
    `j`. Both `i` and `j` must not exceed the dimensions of the matrix.
    
.. function:: void d_mat_zero(d_mat_t mat)

    Sets all entries of ``mat`` to 0.


Random matrix generation
--------------------------------------------------------------------------------


.. function:: void d_mat_randtest(d_mat_t mat, flint_rand_t state, slong minexp, slong maxexp)

    Sets the entries of ``mat`` to random signed numbers with exponents
    between ``minexp`` and ``maxexp`` or zero.


Input and output
--------------------------------------------------------------------------------


.. function:: void d_mat_print(const d_mat_t mat)

    Prints the given matrix to the stream ``stdout``.


Comparison
--------------------------------------------------------------------------------


.. function:: int d_mat_equal(const d_mat_t mat1, const d_mat_t mat2)

    Returns a non-zero value if ``mat1`` and ``mat2`` have 
    the same dimensions and entries, and zero otherwise.
    
.. function:: int d_mat_approx_equal(const d_mat_t mat1, const d_mat_t mat2, double eps)

    Returns a non-zero value if ``mat1`` and ``mat2`` have 
    the same dimensions and entries within ``eps`` of each other,
    and zero otherwise.

.. function:: int d_mat_is_square(const d_mat_t mat)

    Returns a non-zero value if the number of rows is equal to the
    number of columns in ``mat``, and otherwise returns zero.


Transpose
--------------------------------------------------------------------------------


.. function:: void d_mat_transpose(d_mat_t B, const d_mat_t A)

    Sets `B` to `A^T`, the transpose of `A`. Dimensions must be compatible.
    Aliasing is allowed for square matrices.


Matrix multiplication
--------------------------------------------------------------------------------


.. function:: void d_mat_mul_classical(d_mat_t C, const d_mat_t A, const d_mat_t B)

    Sets ``C`` to the matrix product `C = A B`. The matrices must have
    compatible dimensions for matrix multiplication (an exception is raised
    otherwise). Aliasing is allowed.
