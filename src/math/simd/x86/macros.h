#pragma once
#include <x86intrin.h>

#define __SIMD_FN(a,b,c) _mm##a##_##b##_##c
#define SIMD_FN(w,o,t) __SIMD_FN(w,o,t)

#define __SIMD_FN_MASK(a,b,c,d) _mm##a##_##b##_##c##_##d
#define SIMD_FN_MASK(w,o,t,x) __SIMD_FN_MASK(w,o,t,x)

#define __SIMD_MASK_FN(a,b,c) _mm##_mask##a##_##b##_##c
#define SIMD_MASK_FN(w,o,t) __SIMD_MASK_FN(w,o,t)

#define __SIMD_MASKZ_FN(a,b,c) _mm##_maskz##a##_##b##_##c
#define SIMD_MASKZ_FN(w,o,t) __SIMD_MASKZ_FN(w,o,t)

#define SIMD_ZERO SIMD_FN(PREFIX,setzero,TYPE)
#define SIMD_SET SIMD_FN(PREFIX,set,TYPE)
#define SIMD_SET1 SIMD_FN(PREFIX,set1,TYPE)

#define SIMD_ADD SIMD_FN(PREFIX,add,TYPE)
#define SIMD_SUB SIMD_FN(PREFIX,sub,TYPE)
#define SIMD_MUL SIMD_FN(PREFIX,mul,TYPE)
#define SIMD_DIV SIMD_FN(PREFIX,div,TYPE)

#define SIMD_MIN SIMD_FN(PREFIX,min,TYPE)
#define SIMD_MAX SIMD_FN(PREFIX,max,TYPE)

#define SIMD_FMADD SIMD_FN(PREFIX,fmadd,TYPE)
#define SIMD_FMSUB SIMD_FN(PREFIX,fmsub,TYPE)

#define SIMD_SHIFTL SIMD_FN(PREFIX,slli,ITYPE)
#define SIMD_SHIFTR SIMD_FN(PREFIX,srli,ITYPE)

#define SIMD_BLEND SIMD_FN(PREFIX,blend,TYPE)

#define SIMD_AMD SIMD_FN(PREFIX,and,TYPE)
#define SIMD_OR SIMD_FN(PREFIX,or,TYPE)
#define SIMD_XOR SIMD_FN(PREFIX,xor,TYPE)

#define SIMD_CMP_MASK SIMD_FN_MASK(PREFIX,cmp,TYPE,mask)

#define SIMD_MASK_BLEND SIMD_MASK_FN(PREFIX,blend,TYPE)

#define SIMD_MASK_ADD SIMD_MASK_FN(PREFIX,add,TYPE)
#define SIMD_MASK_SUB SIMD_MASK_FN(PREFIX,sub,TYPE)
#define SIMD_MASK_MUL SIMD_MASK_FN(PREFIX,mul,TYPE)
#define SIMD_MASK_DIV SIMD_MASK_FN(PREFIX,div,TYPE)

#define SIMD_MASK_AND SIMD_MASK_FN(PREFIX,and,ITYPE)
#define SIMD_MASK_OR  SIMD_MASK_FN(PREFIX,or, ITYPE)
#define SIMD_MASK_XOR SIMD_MASK_FN(PREFIX,xor,ITYPE)

#define SIMD_MASKZ_MOV SIMD_MASKZ_FN(PREFIX,mov,ITYPE)

#define SIMD_MASKZ_SET1 SIMD_MASKZ_FN(PREFIX,set1,ITYPE)

#define IL inline
#define ARGA const T &a
#define ARGB const T &b
#define ARGC const T &c
#define ARGS const T &src
#define ARGM const U mask
#define OP operator
#define RT return *this
#define RET return

namespace simd
{
	const uint32_t TP32_SIGN = UINT32_MAX >> 1;
	const __m128 M128_TP32_SIGN = _mm_set1_epi32(TP32_SIGN);
	const __m256 M256_TP32_SIGN = _mm256_set1_epi32(TP32_SIGN);
	const __m512 M512_TP32_SIGN = _mm512_set1_epi32(TP32_SIGN);
	const __m128 M128_ZERO = _mm_set1_ps(0.0f);
	const __m256 M256_ZERO = _mm256_set1_ps(0.0f);
	const __m512 M512_ZERO = _mm512_set1_ps(0.0f);

	#define T double2
	#define C 2
	#define B double
	#define M __m128
	#define U uint8_t
	#define PREFIX
	#define TYPE pd
	#define ITYPE epi64
	struct alignas(sizeof(B)*C) T
	{
		M m;

		T() {};
		T(B x) : m(SIMD_SET1(x)) {};
		T(B d00, B d01) : m(SIMD_SET(d01,d00)) {};
		T(M x) : m(x) {};

		IL T OP+(ARGA) const {RET SIMD_ADD(m,a.m);};
		IL T OP-(ARGA) const {RET SIMD_SUB(m,a.m);};
		IL T OP*(ARGA) const {RET SIMD_MUL(m,a.m);};
		IL T OP/(ARGA) const {RET SIMD_DIV(m,a.m);};
		IL T& OP+=(ARGA) {m = SIMD_ADD(m,a.m); RT;};
		IL T& OP-=(ARGA) {m = SIMD_SUB(m,a.m); RT;};
		IL T& OP*=(ARGA) {m = SIMD_MUL(m,a.m); RT;};
		IL T& OP/=(ARGA) {m = SIMD_DIV(m,a.m); RT;};

		IL U OP< (ARGA) const {RET SIMD_CMP_MASK(m,a.m,0x11);};
		IL U OP> (ARGA) const {RET SIMD_CMP_MASK(m,a.m,0x1e);};
		IL U OP==(ARGA) const {RET SIMD_CMP_MASK(m,a.m,0x00);};
		IL U OP<=(ARGA) const {RET SIMD_CMP_MASK(m,a.m,0x12);};
		IL U OP>=(ARGA) const {RET SIMD_CMP_MASK(m,a.m,0x1d);};
		IL U OP!=(ARGA) const {RET SIMD_CMP_MASK(m,a.m,0x1c);};

		IL T OP<<(U s) const {RET SIMD_SHIFTL(m,s);}
		IL T OP>>(U s) const {RET SIMD_SHIFTR(m,s);}
		IL T& OP<<=(U s) {m = SIMD_SHIFTL(m,s); RT;};
		IL T& OP>>=(U s) {m = SIMD_SHIFTR(m,s); RT;};

		IL B OP[](U i) const {RET ((B*)&m)[i];};
		IL B& OP[](U i) {RET ((B*)&m)[i];};

		static IL T zero() {RET SIMD_ZERO();};
	};

	IL T min(ARGA,ARGB) {RET SIMD_MIN(a.m,b.m);};
	IL T max(ARGA,ARGB) {RET SIMD_MAX(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {RET min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {RET SIMD_MASK_BLEND(mask,a.m,b.m);};
	IL T abs(ARGA) {RET SIMD_AMD(a.m,M128_TP32_SIGN);};
	IL U lt(ARGA,ARGB) {RET SIMD_CMP_MASK(a.m,b.m,0x11);};
	IL U gt(ARGA,ARGB) {RET SIMD_CMP_MASK(a.m,b.m,0x1e);};
	IL U eq(ARGA,ARGB) {RET SIMD_CMP_MASK(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {RET SIMD_CMP_MASK(a.m,b.m,0x12);};
	IL U ge(ARGA,ARGB) {RET SIMD_CMP_MASK(a.m,b.m,0x1d);};
	IL U ne(ARGA,ARGB) {RET SIMD_CMP_MASK(a.m,b.m,0x1c);};
	IL T fmadd(ARGA,ARGB,ARGC) {RET SIMD_FMADD(a.m,b.m,c.m);};
	IL T fmsub(ARGA,ARGB,ARGC) {RET SIMD_FMSUB(a.m,b.m,c.m);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_ADD(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_SUB(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_MUL(src.m,mask,a.m,b.m);};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_DIV(src.m,mask,a.m,b.m);};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_AND(src.m,mask,a.m,b.m);};
	IL T mask_or (ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_OR (src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {RET SIMD_MASK_XOR(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {RET SIMD_MASKZ_MOV(mask,a.m);};
	//IL T exp(ARGA) {return exp_ps(a.m);};
	//IL T log(ARGA) {return log_ps(a.m);};
	//IL T pow(ARGA,ARGB) {return pow_ps(a.m,b.m);};
	#undef T
	#undef C
	#undef B
	#undef M
	#undef U
	#undef PREFIX
	#undef TYPE
	#undef ITYPE


	#define T __simd_type__
	template <int C, typename B, typename M, typename U>
	struct alignas(sizeof(B)*C) __simd_type__
	{
		M m;

		T() {};

		template<typename... TVals> requires (sizeof...(TVals)==C) && (std::is_same_v<B,TVals> &&...)
		T(const TVals ... a);
		
		T(const B &x);

		T operator+(ARGA) const;
		T operator-(ARGA) const;
		T operator*(ARGA) const;
		T operator/(ARGA) const;
		T& operator+=(ARGA);
		T& operator-=(ARGA);
		T& operator*=(ARGA);
		T& operator/=(ARGA);

		U operator< (ARGA) const;
		U operator> (ARGA) const;
		U operator==(ARGA) const;
		U operator<=(ARGA) const;
		U operator>=(ARGA) const;
		U operator!=(ARGA) const;

		T operator<<(U s) const;
		T operator>>(U s) const;
		T& operator<<=(U s);
		T& operator>>=(U s);

		IL B operator[](U i) const {RET ((B*)&m)[i];};
		IL B& operator[](U i) {RET ((B*)&m)[i];};

		static IL T zero() {RET SIMD_ZERO();};
	};
	#undef T
};

#undef __SIMD_FN
#undef SIMD_FN

#undef __SIMD_FN_MASK
#undef SIMD_FN_MASK

#undef __SIMD_MASK_FN
#undef SIMD_MASK_FN

#undef __SIMD_MASKZ_FN
#undef SIMD_MASKZ_FN

#undef SIMD_ZERO
#undef SIMD_SET
#undef SIMD_SET1

#undef SIMD_ADD
#undef SIMD_SUB
#undef SIMD_MUL
#undef SIMD_DIV

#undef SIMD_MIN
#undef SIMD_MAX

#undef SIMD_FMADD
#undef SIMD_FMSUB

#undef SIMD_SHIFTL
#undef SIMD_SHIFTR

#undef SIMD_BLEND

#undef SIMD_AND
#undef SIMD_OR
#undef SIMD_XOR

#undef SIMD_CMP_MASK

#undef SIMD_MASK_BLEND

#undef SIMD_MASK_ADD
#undef SIMD_MASK_SUB
#undef SIMD_MASK_MUL
#undef SIMD_MASK_DIV

#undef SIMD_MASK_AND
#undef SIMD_MASK_OR
#undef SIMD_MASK_XOR

#undef SIMD_MASKZ_MOV

#undef IL
#undef ARGA
#undef ARGB
#undef ARGC
#undef ARGS
#undef ARGM
#undef OP
#undef RT
#undef RET