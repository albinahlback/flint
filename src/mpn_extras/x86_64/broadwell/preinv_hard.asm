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

define(`s0', `%rax')
define(`s1', `%rdx')
define(`s2', `%rcx')
define(`s3', `%r8')
define(`s4', `%r9')
define(`s5', `%r10')
define(`s6', `%r11')
define(`s7', `%rbx') C Push from here
define(`s8', `%rbp')
define(`s9', `%r12')
define(`sA', `%r13')
define(`sB', `%r14')
define(`sC', `%r15')
define(`sD', `%rsi') C aliased with dp
define(`sE', `%rdi') C aliased with rp

	TEXT

	ALIGN(16)
PROLOGUE(nn_preinv_1)
	mov	0x00(dp), s3	C s3 = d
	mov	s3, dp
	shr	dp
	inc	dp		C dp = d0 = floor(d / 2) + 1
	movabs	$0x2000000000000000, s1
	xor	R32(s0),R32(s0)	C (s1, s0) = 2^{125}
	div	dp		C s0 = x0, s1 = e0
	mov	s3, s2
	and	$1, R32(s2)
	neg	s2		C s2 = -(d & 1)
	and	s0, s2		C s2 = x0 (d & 1)
	lea	(s0, s0), s5	C s5 = 2 x0
	sub	s2, s5		C s5 = x0 (2 d0 - d)
	lea	(s1, s1), s4	C s4 = 2 e0
	add	s5, s4		C (CF, s4) = 2^{126} - x0 d
	sbb	s5, s5
	and	s0, s5		C s5 = 2^126 - x0 d >= 2^{64} ? x0 : 0
	mov	s0, s1
	mulx	s4, s6, s6
	shl	$2, s0		C s0 = 4 x0 mod 2^64
	add	s5, s6
	shr	$60, s6		C s6 = floor(x0 (2^126 - x0 d) / 2^124)
	add	s6, s0		C s0 = x1 + 1
	mov	s0, s1
	mulx	s3, s4, s5	C (s5, s4) = (x1 + 1) d
	mov	s3, s6
	dec	s6
	add	s6, s4		C s4 = (x1 + 2) d - 1 (mod 2^64)
	adc	s3, s5		C (CF, s5, s4) = (x1 + 2^{64} + 2) d - 1
	sbb	$0, s0		C s0 = x2
	mov	s0, 0x00(rp)

	ret
EPILOGUE()

	ALIGN(16)
PROLOGUE(nn_preinv_2)
	push	s7
	push	s8
	push	s9

	C s0 = x0, s1 = e0
	mov	0x08(dp), s2
	shr	s2
	inc	s2		C s2 = d0
	movabs	$0x2000000000000000, s1
	xor	R32(s0),R32(s0)	C (s1, s0) = 2^125
	div	s2		C x0 = s0, e0 = s1

	C (s5, s3) = x1
	mov	0x00(dp), s2
	mov	$1, R32(s7)
	mov	0x08(dp), s8
	shrd	$1, s8, s2	C s2 = floor(d / 2) mod 2^{64}
	add	s7, s2		C (CF, s2) = d1 - 2^64 (d0 - 1)
	sbb	s7, s7
	and	s0, s7		C s7 = d1 - 2^64 (d0 - 1) >= 2^{64} ? x0 : 0
	lea	(s1, s0), s4
	sub	s7, s4		C s4 = e0 + x0 - s7
	mov	s0, s1
	mulx	s2, s7, s8	C (s8, s7) = x0 (d1 mod 2^{64})
	neg	s7
	sbb	s8, s4		C s4 = floor((2^{189} - x0 d1) / 2^{64})
	mulx	s4, s8, s8	C s8 = mulhi_2(x0, 2^{189} - x0 d1) / 2^{128}
	lea	(, s8, 8), s3
	shr	$61, s8
	lea	(s0, s8), s5	C (s5, s3) = x1

	C (s7, s2, s0) = r = 2^{254} - x1 d
	C s4 = c = floor(mulhi_3(x1, r) / 2^252)
	C (s5, s3) = x2 + 1
	mov	s3, s1
	mulx	0x00(dp), s0, s6
	mulx	0x08(dp), s2, s4
	add	s2, s6
	adc	$0, s4
	mov	s5, s1
	mulx	0x00(dp), s2, s7
	xor	R32(s8), R32(s8)
	adcx	s2, s6
	adox	s7, s4
	mulx	0x08(dp), s2, s7
	adcx	s2, s4		C (s4, s6, s0) = -r0 = (x1 d) mod 2^{192}
	neg	s0
	mov	s8, s2
	sbb	s6, s2
	mov	s8, s7
	sbb	s4, s7		C (s7, s2, s0) = r = 2^{254} - x1 d
	mov	s0, s9		C s9 = r0[0]
	mov	s5, s1
	mulx	s7, s0, s4
	mulx	s2, s6, s6
	add	s6, s0
	adc	s8, s4
	mov	s3, s1
	mulx	s7, s6, s6
	add	s6, s0
	adc	s8, s4		C (s4, s0) = mulhi_3(x1, R2) / 2^{192}
	shld	$4, s0, s4	C s4 = c = floor(mulhi_3(x1, R2) / 2^{252})
	shld	$2, s3, s5
	shl	$2, s3		C (s5, s3) = 2^{2} x1
	add	s4, s3
	adc	$0, s5		C (s5, s3) = x2 + 1

	C x3 = x2 + 1 - [4 r < (c + 1) d]
	lea	1(s4), s1	C %rdx = c + 1
	mulx	0x00(dp), s0, s6
	shld	$2, s2, s7
	mulx	0x08(dp), s4, s8
	shld	$2, s9, s2
	shl	$2, s9		C s9 = (s7, s2, s9) = 4 r
	add	s4, s6
	adc	$0, s8		C (s8, s6, s0) = (c + 1) d
	sub	s0, s9
	sbb	s6, s2
	sbb	s8, s7		C CF = borrow(4 r - (c + 1) d)
	sbb	$0, s3
	sbb	$0, s5		C (s5, s3) = x3
	mov	s3, 0x00(rp)
	mov	s5, 0x08(rp)

	pop	s9
	pop	s8
	pop	s7
	ret
EPILOGUE()

	ALIGN(16)
PROLOGUE(nn_preinv_4)
	push	s7
	push	s8
	push	s9
	push	sA
	push	sB
	push	sC
	sub	$0x48, %rsp

	C (s5..s2) = ~(d >> 1)
	C (s0, s1) = (x0, e0)
	mov	0x18(dp), s9
	mov	$1, R32(s0)
	shrx	s0, s9, sA
	movabs	$0x2000000000000000, s1
	xor	R32(s0),R32(s0)	C (s1, s0) = 2^125
	inc	sA		C sA = d0
	div	sA		C (s0, s1) = (x0, e0)
	mov	0x00(dp), s2
	mov	0x08(dp), s3
	mov	0x10(dp), s4
	mov	0x18(dp), s5
	shrd	$1, s3, s2
	not	s2
	shrd	$1, s4, s3
	not	s3
	shrd	$1, s5, s4
	not	s4
	shr	s5
	not	s5

	C (s8, s0) = x1 = 2^58 x0 + floor(mulhi_2(x0, 2^189 - x0 d1) / 2^131)
	C where 2^189 - x0 d1 = 2^64 e0 + x0 (2^64 d0 - d1)
	mov	s1, s6
	mov	s0, s1
	mulx	s4, s7, s7	C s7 = hi(x0 (2^64 d0 - d1))
	add	s6, s7		C s7 = hi(2^189 - x0 d1)
	mulx	s7, s7, s7	C s7 = mulhi_2(x0, 2^189 - x0 d1) / 2^128
	shr	$3, s7		C s7 = floor(mulhi_2(...) / 2^131)
	mov	s0, s8
	shl	$58, s0
	shr	$6, s8
	add	s7, s0
	adc	$0, s8

	C (sC, sA) = r1 = mulhi_2(x1, -d2) mod 2^256
	mov	s0, s1
	mulx	s3, sA, sA
	mulx	s4, sB, sC
	imul	s5, s1
	add	sB, sA
	adc	s1, sC
	mov	s8, s1
	mulx	s2, s6, s6
	mulx	s3, s9, s3
	add	s6, sA
	adc	s3, sC
	imul	s4, s1
	add	s9, sA
	adc	s1, sC

	C (s8, s6, s5, s2) = x2 = 2^134 x1 + mulhi_4(x1, r1) / 2^241
	mov	s0, s1
	mulx	sC, s2, s2
	mov	s8, s1
	mulx	sA, s3, s3
	shld	$6, s0, s8
	add	s3, s2
	mulx	sC, s4, s5
	adc	$0, s5
	shl	$6, s0		C (s8, s0, *, *) = 2^134 x1
	add	s4, s2
	adc	$0, s5		C (s5, s2) = mulhi_4(x1, 2^375 - x1 d2) / 2^256
	mov	s5, s6
	shr	$49, s6
	shld	$15, s2, s5
	shl	$15, s2		C (s6, s5, s2) = mulhi_4(...) / 2^241
	add	s0, s6
	adc	$0, s8
	mov	s2, 0x00(%rsp)
	mov	s5, 0x08(%rsp)
	mov	s6, 0x10(%rsp)
	mov	s8, 0x18(%rsp)

	C (s6, s5, *, *) = %rsp[47:20] = r = -x2 d mod 2^320
	mov	0x00(%rsp), s1
	mulx	0x00(dp), s3, s4
	mulx	0x08(dp), s0, s5
	add	s0, s4
	mulx	0x10(dp), s0, s6
	adc	s0, s5
	mulx	0x18(dp), s0, s9
	adc	s0, s6
	adc	$0, s9
	mov	0x08(%rsp), s1
	xor	sA, sA
	mulx	0x00(dp), s0, s2
	adcx	s0, s4
	adox	s2, s5
	mulx	0x08(dp), s0, s2
	adcx	s0, s5
	adox	s2, s6
	mulx	0x10(dp), s0, s2
	adcx	s0, s6
	adox	s2, s9
	mulx	0x18(dp), s0, s2
	adcx	s0, s9
	mov	0x10(%rsp), s1
	xor	sA, sA
	mulx	0x00(dp), s0, s2
	adcx	s0, s5
	adox	s2, s6
	mulx	0x08(dp), s0, s2
	adcx	s0, s6
	adox	s2, s9
	mulx	0x10(dp), s0, s2
	adcx	s0, s9
	mov	0x18(%rsp), s1
	xor	sA, sA
	mulx	0x00(dp), s0, s2
	adcx	s0, s6
	adox	s2, s9
	mulx	0x08(dp), s0, s2
	adcx	s0, s9		C (s9, s6, s5, s4, s3) = x2 d mod 2^320
	neg	s3
	mov	s3, 0x20(%rsp)
	mov	sA, s3
	sbb	s4, s3
	mov	s3, 0x28(%rsp)
	mov	sA, s4
	sbb	s5, s4
	mov	s4, 0x30(%rsp)
	mov	sA, s5
	sbb	s6, s5
	mov	s5, 0x38(%rsp)
	mov	sA, s6
	sbb	s9, s6
	mov	s6, 0x40(%rsp)

	C s3 = c = floor(mulhi_7(...) / 2^508)
	mov	0x18(%rsp), s1
	mulx	s6, s3, s4
	mulx	s5, s0, s0
	add	s0, s3
	adc	$0, s4
	mov	0x10(%rsp), s1
	mulx	s6, s0, s0
	add	s0, s3
	adc	$0, s4		C (s4, s3) = mulhi_7(x2, 2^510 - x2 d) / 2^448
	shrd	$60, s4, s3	C s3 = c = floor(mulhi_7(...) / 2^508)

	C %rsp[1f:00] = x3 + 1 = (2^2 x2 + c) mod 2^256
	mov	0x18(%rsp), s6
	mov	0x10(%rsp), s5
	shld	$2, s5, s6
	mov	0x08(%rsp), s4
	shld	$2, s4, s5
	mov	0x00(%rsp), s2
	shld	$2, s2, s4
	shl	$2, s2
	add	s3, s2
	adc	$0, s4
	adc	$0, s5
	adc	$0, s6
	mov	s2, 0x00(%rsp)
	mov	s4, 0x08(%rsp)
	mov	s5, 0x10(%rsp)
	mov	s6, 0x18(%rsp)

	C x4 = (x3 + 1) - [4 r < (c + 1) d]
	lea	1(s3), s1
	mulx	0x00(dp), s3, s4
	mov	0x40(%rsp), s8
	mulx	0x08(dp), s0, s5
	mov	0x38(%rsp), s9
	add	s0, s4
	mulx	0x10(dp), s0, s6
	mov	0x30(%rsp), sA
	adc	s0, s5
	mulx	0x18(dp), s0, s7
	mov	0x28(%rsp), sB
	adc	s0, s6
	mov	0x20(%rsp), sC
	adc	$0, s7		C (s7, s6, s5, s4, s3) = (c + 1) d
	shld	$2, s9, s8
	shld	$2, sA, s9
	shld	$2, sB, sA
	shld	$2, sC, sB
	shl	$2, sC		C (sC, sB, sA, s9, s8) = 4 r
	sub	s3, sC
	sbb	s4, sB
	sbb	s5, sA
	sbb	s6, s9
	sbb	s7, s8		C CF = borrow(4 r - (c + 1) d)
	mov	0x00(%rsp), s3
	mov	0x08(%rsp), s4
	mov	0x10(%rsp), s5
	mov	0x18(%rsp), s6
	sbb	$0, s3
	sbb	$0, s4
	sbb	$0, s5
	sbb	$0, s6
	mov	s3, 0x00(rp)
	mov	s4, 0x08(rp)
	mov	s5, 0x10(rp)
	mov	s6, 0x18(rp)

	add	$0x48, %rsp
	pop	sC
	pop	sB
	pop	sA
	pop	s9
	pop	s8
	pop	s7
	ret
EPILOGUE()

	ALIGN(16)
PROLOGUE(nn_preinv_8)
	C Push (s7..sE)
	push	sA
	push	s8
	push	s9
	push	s7
	push	sB
	push	sC
	push	sE

	C (s9..s2) = ~(d >> 1) -> %rsp[3f:00]
	C (s0, s1) = (x0, e0)
	mov	0x38(dp), s9
	mov	$1, R32(s0)
	shrx	s0, s9, sA
	movabs	$0x2000000000000000, s1
	xor	R32(s0),R32(s0)	C (s1, s0) = 2^125
	inc	sA		C sA = d0
	div	sA		C (s0, s1) = (x0, e0)
	mov	0x00(dp), s2
	mov	0x08(dp), s3
	sub	$0x80, %rsp
	mov	0x10(dp), s4
	mov	0x18(dp), s5
	shrd	$1, s3, s2
	not	s2
	mov	0x20(dp), s6
	mov	s2, 0x00(%rsp)
	shrd	$1, s4, s3
	not	s3
	mov	0x28(dp), s7
	mov	s3, 0x08(%rsp)
	shrd	$1, s5, s4
	not	s4
	mov	0x30(dp), s8
	mov	s4, 0x10(%rsp)
	shrd	$1, s6, s5
	not	s5
	mov	s5, 0x18(%rsp)
	shrd	$1, s7, s6
	not	s6
	mov	s6, 0x20(%rsp)
	shrd	$1, s8, s7
	not	s7
	mov	s7, 0x28(%rsp)
	shrd	$1, s9, s8
	not	s8
	mov	s8, 0x30(%rsp)
	shr	$1, s9
	not	s9
	mov	s9, 0x38(%rsp)

	C (s0, s1) = x1
	mov	s1, s2
	mov	s0, s1
	mulx	s8, s4, s4	C s4 = hi(x0 * lo(-d1))
	add	s2, s4		C s4 = hi(2^189 - x0 d1)
	mulx	s4, s4, s4
	shr	$3, s4		C s4 = floor(mulhi_2(x0, 2^189 - x0 d1)/2^131)
	shl	$58, s1
	shr	$6, s0		C (s0, s1) = x0 << 58
	add	s4, s1
	adc	$0, s0		C (s0, s1) = x1

	C (s5, s4) = r1 = mulhi_2(x1, -d2) mod 2^256
	mulx	0x28(%rsp), s2, s2
	mulx	0x30(%rsp), s4, s5
	mov	s1, s6
	imul	0x38(%rsp), s1
	add	s2, s4
	adc	s1, s5
	mov	s0, s1		C (s1, s6) = x1
	xor	R32(s0), R32(s0)
	mulx	0x20(%rsp), s7, s7
	mulx	0x28(%rsp), s8, s9
	mov	s1, sA
	add	s7, s4
	adc	s9, s5
	imul	0x30(%rsp), sA
	add	s8, s4
	adc	sA, s5

	C (s4, s3) = floor(mulhi_4(x1, r1) / 2^260)
	mulx	s4, s2, s2
	mulx	s5, s3, s4
	mov	s1, s7
	mov	s6, s1		C (s7, s1) = x1
	add	s2, s3
	adc	s0, s4
	mulx	s5, s5, s5
	add	s5, s3
	adc	s0, s4
	shrd	$4, s4, s3
	shr	$4, s4

	C %rsp[0x5f:0x40] = x2 = (x1 << 115) + floor(mulhi_4(...) / 2^260)
	mov	s7, s8
	shr	$13, s8
	shld	$51, s1, s7
	shl	$51, s1		C (s8, s7, s1, *) = x1 << 115
	add	s4, s1
	adc	s0, s7
	adc	s0, s8
	mov	s3, 0x40(%rsp)
	mov	s1, 0x48(%rsp)
	mov	s7, 0x50(%rsp)
	mov	s8, 0x58(%rsp)

	C (s5, sC, sA, s8) = r2 = mulhi_4(x2, -d3) mod 2^512
	C
	C   0 1 2 3 4 5 6 7
	C 0       h x x x l
	C 1     h x x x l
	C 2   h x x x l
	C 3 h x x x l
	mov	s3, s1
	mulx	0x18(%rsp), s8, s8
	mulx	0x20(%rsp), s0, sA
	mulx	0x28(%rsp), s2, sC
	mulx	0x30(%rsp), s3, s5
	imul	0x38(%rsp), s1
	add	s0, s8
	adc	s2, sA
	adc	s3, sC
	adc	s1, s5
	mov	0x48(%rsp), s1
	test	R32(s0), R32(s0)
	mulx	0x10(%rsp), s4, s4
	adcx	s4, s8
	mulx	0x18(%rsp), s6, s7
	adox	s6, s8
	adcx	s7, sA
	mulx	0x20(%rsp), s9, sB
	adox	s9, sA
	adcx	sB, sC
	mulx	0x28(%rsp), s0, s2
	adox	s0, sC
	adcx	s2, s5
	mulx	0x30(%rsp), s3, s1	C s1 unused
	adox	s3, s5
	mov	0x50(%rsp), s1
	test	R32(s0), R32(s0)
	mulx	0x08(%rsp), s7, s7
	adcx	s7, s8
	mulx	0x10(%rsp), s9, sB
	adox	s9, s8
	adcx	sB, sA
	mulx	0x18(%rsp), s3, s7
	adox	s3, sA
	adcx	s7, sC
	mulx	0x20(%rsp), s9, sB
	adox	s9, sC
	adcx	sB, s5
	mulx	0x28(%rsp), s3, s7	C s7 unused
	adox	s3, s5
	mov	0x58(%rsp), s1
	test	R32(s0), R32(s0)
	mulx	0x00(%rsp), sB, sB
	adcx	sB, s8
	mulx	0x08(%rsp), s3, s7
	adox	s3, s8
	adcx	s7, sA
	mulx	0x10(%rsp), s9, sB
	adox	s9, sA
	adcx	sB, sC
	mulx	0x18(%rsp), s3, s7
	adox	s3, sC
	adcx	s7, s5
	mulx	0x20(%rsp), s9, sB	C sB unused
	adox	s9, s5

	C (sB, sA, s8, s2) = mulhi_8(x2, r2) / 2^512
	mov	s8, s1
	mulx	0x58(%rsp), s2, s2
	mov	sA, s1
	xor	R32(s0), R32(s0)
	mulx	0x50(%rsp), s3, s3
	adcx	s3, s2
	mulx	0x58(%rsp), s4, s8
	adox	s4, s2
	adcx	s0, s8
	mov	sC, s1
	mulx	0x48(%rsp), s6, s6
	adox	s0, s8
	adcx	s6, s2
	mulx	0x50(%rsp), s7, s9
	adox	s7, s2
	adcx	s9, s8
	mulx	0x58(%rsp), sB, sA
	adox	sB, s8
	adcx	s0, sA
	mov	s5, s1
	mulx	0x40(%rsp), s3, s3
	adox	s0, sA
	mulx	0x48(%rsp), s4, sC
	adcx	s3, s2
	adox	s4, s2
	mulx	0x50(%rsp), s6, s7
	adcx	sC, s8
	adox	s6, s8
	mulx	0x58(%rsp), s9, sB
	adcx	s7, sA
	adox	s9, sA
	adcx	s0, sB
	adox	s0, sB

	C %rsp[0x7f:0x40] = x3 = (x2 << 275) + mulhi_8(...) / 2^471
	mov	sB, s0
	shr	$23, s0
	shld	$41, sA, sB
	shld	$41, s8, sA
	shld	$41, s2, s8
	shl	$41, s2		C (s0, sB, sA, s8, s2) = mulhi_8(...) / 2^471
	mov	0x58(%rsp), s6
	mov	0x50(%rsp), s4
	mov	0x48(%rsp), s3
	mov	0x40(%rsp), s1
	shld	$19, s4, s6
	shld	$19, s3, s4
	shld	$19, s1, s3
	shl	$19, s1		C (s6, s4, s3, s1) = x2 << 19
	add	s1, s0
	adc	$0, s3
	adc	$0, s4
	adc	$0, s6		C (s6, s4, s3, s0, sB, sA, s8, s2) = x3
	mov	s2, 0x40(%rsp)
	mov	s8, 0x48(%rsp)
	mov	sA, 0x50(%rsp)
	mov	sB, 0x58(%rsp)
	mov	s0, 0x60(%rsp)
	mov	s3, 0x68(%rsp)
	mov	s4, 0x70(%rsp)
	mov	s6, 0x78(%rsp)

	C (s9, s5, sC, sA, s8, s6, s4, s2, s0) = 4 x3 d - 1 (mod 2^576)
	C                                      = ~(4 r) (mod 2^576)
	mov	s2, s1
	mulx	0x00(dp), s0, s2
	mulx	0x08(dp), s3, s4
	add	s3, s2
	mulx	0x10(dp), s5, s6
	adc	s5, s4
	mulx	0x18(dp), s7, s8
	adc	s7, s6
	mulx	0x20(dp), s9, sA
	adc	s9, s8
	mulx	0x28(dp), sB, sC
	adc	sB, sA
	mulx	0x30(dp), s3, s5
	adc	s3, sC
	mulx	0x38(dp), s7, s9
	mov	0x48(%rsp), s1
	adc	s7, s5
	adc	$0, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), s3, s7
	adcx	s3, s2
	adox	s7, s4
	mulx	0x08(dp), sB, sE
	adcx	sB, s4
	adox	sE, s6
	mulx	0x10(dp), s3, s7
	adcx	s3, s6
	adox	s7, s8
	mulx	0x18(dp), sB, sE
	adcx	sB, s8
	adox	sE, sA
	mulx	0x20(dp), s3, s7
	adcx	s3, sA
	adox	s7, sC
	mulx	0x28(dp), sB, sE
	adcx	sB, sC
	adox	sE, s5
	mulx	0x30(dp), s3, s7
	adcx	s3, s5
	adox	s7, s9
	mulx	0x38(dp), sB, sE
	mov	0x50(%rsp), s1
	adcx	sB, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), s3, s7
	adcx	s3, s4
	adox	s7, s6
	mulx	0x08(dp), sB, sE
	adcx	sB, s6
	adox	sE, s8
	mulx	0x10(dp), s3, s7
	adcx	s3, s8
	adox	s7, sA
	mulx	0x18(dp), sB, sE
	adcx	sB, sA
	adox	sE, sC
	mulx	0x20(dp), s3, s7
	adcx	s3, sC
	adox	s7, s5
	mulx	0x28(dp), sB, sE
	adcx	sB, s5
	adox	sE, s9
	mulx	0x30(dp), s3, s7
	mov	0x58(%rsp), s1
	adcx	s3, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), sB, sE
	adcx	sB, s6
	adox	sE, s8
	mulx	0x08(dp), s3, s7
	adcx	s3, s8
	adox	s7, sA
	mulx	0x10(dp), sB, sE
	adcx	sB, sA
	adox	sE, sC
	mulx	0x18(dp), s3, s7
	adcx	s3, sC
	adox	s7, s5
	mulx	0x20(dp), sB, sE
	adcx	sB, s5
	adox	sE, s9
	mulx	0x28(dp), s3, s7
	mov	0x60(%rsp), s1
	adcx	s3, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), sB, sE
	adcx	sB, s8
	adox	sE, sA
	mulx	0x08(dp), s3, s7
	adcx	s3, sA
	adox	s7, sC
	mulx	0x10(dp), sB, sE
	adcx	sB, sC
	adox	sE, s5
	mulx	0x18(dp), s3, s7
	adcx	s3, s5
	adox	s7, s9
	mulx	0x20(dp), sB, sE
	mov	0x68(%rsp), s1
	adcx	sB, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), s3, s7
	adcx	s3, sA
	adox	s7, sC
	mulx	0x08(dp), sB, sE
	adcx	sB, sC
	adox	sE, s5
	mulx	0x10(dp), s3, s7
	adcx	s3, s5
	adox	s7, s9
	mulx	0x18(dp), sB, sE
	mov	0x70(%rsp), s1
	adcx	sB, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), s3, s7
	adcx	s3, sC
	adox	s7, s5
	mulx	0x08(dp), sB, sE
	adcx	sB, s5
	adox	sE, s9
	mulx	0x10(dp), s3, s7
	mov	0x78(%rsp), s1
	adcx	s3, s9
	test	R32(s0), R32(s0)
	mulx	0x00(dp), sB, sE
	adcx	sB, s5
	adox	sE, s9
	mulx	0x08(dp), s3, s7
	adcx	s3, s9		C (s9..s0) = x3 d (mod 2^576)

	C (s9, s5, sC, sA, s8, s6, s4, s2, s0) = r = -(x3 d) mod 2^576
	not	s2
	not	s4
	not	s6
	not	s8
	not	sA
	not	sC
	not	s5
	not	s9
	not	s0
	add	$1, s0
	adc	$0, s2
	adc	$0, s4
	adc	$0, s6
	adc	$0, s8
	adc	$0, sA
	adc	$0, sC
	adc	$0, s5
	adc	$0, s9

	C s3 = c = floor(mulhi_15(x3, r) / 2^1020)
	mov	0x78(%rsp), s1
	mulx	s9, s3, s7
	mulx	s5, sB, sB
	add	sB, s3
	adc	$0, s7
	mov	0x70(%rsp), s1
	mulx	s9, sB, sB
	add	sB, s3
	adc	$0, s7
	shrd	$60, s7, s3

	C (s9, s5, sC, sA, s8, s6, s4, s2, s0) = ~(4 r)
	shld	$2, s5, s9
	shld	$2, sC, s5
	shld	$2, sA, sC
	shld	$2, s8, sA
	shld	$2, s6, s8
	shld	$2, s4, s6
	shld	$2, s2, s4
	shld	$2, s0, s2
	shl	$2, s0
	not	s0
	not	s2
	not	s4
	not	s6
	not	s8
	not	sA
	not	sC
	not	s5
	not	s9

	C s8 = q = -[4 r < (c + 1) d]
	C        = -[~(4 r) + (c + 1) d overflows]
	lea	1(s3), s1
	test	R32(s0), R32(s0)
	mulx	0x00(dp), s7, sB
	adcx	s7, s0
	adox	sB, s2
	mulx	0x08(dp), sE, s0
	adcx	sE, s2
	adox	s0, s4
	mulx	0x10(dp), s7, sB
	adcx	s7, s4
	adox	sB, s6
	mulx	0x18(dp), s2, s4
	adcx	s2, s6
	adox	s4, s8
	mulx	0x20(dp), sE, s0
	adcx	sE, s8
	adox	s0, sA
	mov	$0, R32(s8)
	mulx	0x28(dp), s7, sB
	adcx	s7, sA
	adox	sB, sC
	mulx	0x30(dp), s2, s4
	adcx	s2, sC
	adox	s4, s5
	mulx	0x38(dp), sE, s0
	adcx	sE, s5
	adox	s0, s9
	adcx	s8, s9
	adox	s8, s8
	adcx	s8, s8		C s8 = 2 OF + CF, nonzero iff it overflowed
	neg	s8		C CF = [overflowed]
	sbb	s8, s8		C s8 = -[~(4 r) + (c + 1) d overflows] = q

	C rp[3f:00] = x5 = 4 x3 + c + q (mod 2^512)
	mov	0x80(%rsp), sE	C Load rp
	mov	0x78(%rsp), sA
	mov	0x70(%rsp), s9
	mov	0x68(%rsp), s7
	shld	$2, s9, sA
	mov	0x60(%rsp), s6
	shld	$2, s7, s9
	mov	0x58(%rsp), s5
	shld	$2, s6, s7
	mov	0x50(%rsp), s4
	shld	$2, s5, s6
	mov	0x48(%rsp), s2
	shld	$2, s4, s5
	mov	0x40(%rsp), s1
	shld	$2, s2, s4
	add	$0x88, %rsp	C 0x88 because sE is already restored
	shld	$2, s1, s2
	shl	$2, s1
	xor	R32(s0), R32(s0)
	adcx	s3, s1
	adox	s8, s1
	pop	sC
	mov	s1, 0x00(rp)
	adcx	s0, s2
	adox	s8, s2
	pop	sB
	mov	s2, 0x08(rp)
	adcx	s0, s4
	adox	s8, s4
	mov	s4, 0x10(rp)
	adcx	s0, s5
	adox	s8, s5
	mov	s5, 0x18(rp)
	adcx	s0, s6
	adox	s8, s6
	mov	s6, 0x20(rp)
	adcx	s0, s7
	adox	s8, s7
	mov	s7, 0x28(rp)
	adcx	s0, s9
	adox	s8, s9
	pop	s7
	mov	s9, 0x30(rp)
	adcx	s0, sA
	adox	s8, sA
	pop	s9
	pop	s8
	mov	sA, 0x38(rp)
	pop	sA

	ret
EPILOGUE()
