dnl
dnl Copyright (C) 2026 Albin Ahlbäck
dnl
dnl This file is part of FLINT.
dnl
dnl FLINT is free software: you can redistribute it and/or modify it under
dnl the terms of the GNU Lesser General Public License (LGPL) as published
dnl by the Free Software Foundation; either version 3 of the License, or
dnl (at your option) any later version.  See <https://www.gnu.org/licenses/>.
dnl

include(`config.m4')

define(`rp', `%rdi')
define(`dp', `%rsi')

define(`r0', `%rax')
define(`r1', `%rdx')
define(`r2', `%rcx')
define(`r3', `%rsi') C aliased with dp
define(`r4', `%r8')
define(`r5', `%r9')
define(`r6', `%r10')
define(`r7', `%r11')
define(`r8', `%rbx') C Push from here
define(`r9', `%rbp')
define(`ra', `%r12')
define(`rb', `%r13')
define(`rc', `%r14')
define(`rd', `%r15')

	TEXT

dnl TODO: register allocate

	ALIGN(16)
PROLOGUE(nn_preinv_1)
	mov	0*8(dp), r3		C r3 = d
	mov	r3, r4
	xor	R32(r2), R32(r2)
	shr	$1, r4
	setc	R8(r2)			C r2 = (d & 1) == 1
	adc	$0, r4			C r4 = d0
	movabs	$0x2000000000000000, r1
	xor	r0, r0			C (r1, r0) = 2^{125}
	div	r4			C r0 = x0, r1 = e0
	shl	$63, r2
	sar	$63, r2
	mov	$63, R32(r5)
	mov	$1, R32(r6)
	and	r0, r2			C r2 = x0 * (2 d0 - d)
	shlx	r5, r1, r7
	shrx	r6, r1, r1		C (r7, r1) = 2 * e0
	add	r2, r8
	adc	$0, R32(r7)		C (r7, r8) = 2 * e0 + x0 * (2 d0 - d)
					C	   = 2^{126} - x0 d
	mov	r0, r1			C %rdx = x0
	mulx	r8, r10, r10		C r10 = mulhi(x0, (2^{126} - x0 d))
	shr	$60, r10		C r10 = \lfloor x0 * (2^{126} - x0 d) / 2^{124} \rfloor
	shl	$2, r0
	add	r10, r0
	sub	$1, r0			C r0 = x1 = 2^{2} x0 + \lfloor x0 * (2^{126} - x0 d) / 2^{124} \rfloor - 2^{64} - 1
	mov	r0, r2
	xor	R32(r4), R32(r4)
	add	$1, r2
	adc	$1, r4			C (r4, r2) = x1 + 2^{64} + 1
	mov	r3, r1			C %rdx = d
	mulx	r2, r11, r12
	mulx	r4, r13, r14
	add	r12, r13
	adc	$0, R32(r14)		C (r14, r13, r11) = (x1 + 2^{64} + 1) d
	sub	$1, r1			C r1 = d - 1
	add	r1, r11
	adc	$0, r13
	adc	$0, R32(r14)		C r14 = \lfloor ((x1 + 2^{64} + 1) d + (d - 1)) / 2^{128} \rfloor
	sub	$1, r14
	sub	r14, r0			C r0 = x2
	mov	r0, 0*8(rp)
	ret
EPILOGUE()
