#pragma once
#include <cstdint>
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
#include "utils/avx512_mathfun.h"
#include "utils/avx_mathfun.h"
#include "utils/sse_mathfun.h"

#define LOOP for(uint32_t i = 0; i < I; i++)
#define OPC(o) c[i] = this->operator[](i) o a[i];
#define OPE(o) this->operator[](i) o##= a[i];
#define IL inline
#define RC return c
#define CMP(o) c |= this->operator[](i) o a[i] ? 1u << i : 0
#define MASK_OP(o) c[i] = mask >> i & 0x1 ? a[i] o b[i] : src[i]

#ifdef TEST_SIMD
namespace __hw_simd__
#else
namespace simd
#endif
{
	#define ARGA const T &a
	#define ARGB const T &b
	#define ARGC const T &c
	#define ARGS const T &src
	#define ARGM const U mask

	const uint32_t TP32_SIGN = UINT32_MAX >> 1;
	const __m128 M128_TP32_SIGN = _mm_set1_epi32(TP32_SIGN);
	const __m256 M256_TP32_SIGN = _mm256_set1_epi32(TP32_SIGN);
	const __m512 M512_TP32_SIGN = _mm512_set1_epi32(TP32_SIGN);
	const uint64_t TP64_SIGN = UINT64_MAX >> 1;
	const __m128 M128_TP64_SIGN = _mm_maskz_set1_epi64(UINT8_MAX,TP64_SIGN);
	const __m256 M256_TP64_SIGN = _mm256_maskz_set1_epi64(UINT8_MAX,TP64_SIGN);
	const __m512 M512_TP64_SIGN = _mm512_maskz_set1_epi64(UINT8_MAX,TP64_SIGN);
	const __m128 M128_ZERO = _mm_set1_ps(0.0f);
	const __m256 M256_ZERO = _mm256_set1_ps(0.0f);
	const __m512 M512_ZERO = _mm512_set1_ps(0.0f);

	#define T float4
	#define I 4
	#define B float
	#define U uint8_t
	struct alignas(sizeof(B)*I) float4
	{
		__m128 m;

		T() {};
		T(B x) : m(_mm_set1_ps(x)) {};
		T(B f00, B f01, B f02, B f03) : m(_mm_set_ps(f03,f02,f01,f00)) {};
		T(__m128 x) : m(x) {};

		IL T operator+(const T &a) const {return _mm_add_ps(m,a.m);};
		IL T operator-(const T &a) const {return _mm_sub_ps(m,a.m);};
		IL T operator*(const T &a) const {return _mm_mul_ps(m,a.m);};
		IL T operator/(const T &a) const {return _mm_div_ps(m,a.m);};
		IL T& operator+=(const T &a) {m = _mm_add_ps(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm_sub_ps(m,a.m); return *this;};
		IL T& operator*=(const T &a) {m = _mm_mul_ps(m,a.m); return *this;};
		IL T& operator/=(const T &a) {m = _mm_div_ps(m,a.m); return *this;};

		IL U operator<(ARGA) const {return _mm_cmp_ps_mask(m,a.m,0x11);};
		IL U operator>(ARGA) const {return _mm_cmp_ps_mask(m,a.m,0x1e);};
		IL U operator==(ARGA) const {return _mm_cmp_ps_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm_cmp_ps_mask(m,a.m,0x12);};
		IL U operator>=(ARGA) const {return _mm_cmp_ps_mask(m,a.m,0x1d);};
		IL U operator!=(ARGA) const {return _mm_cmp_ps_mask(m,a.m,0x1c);};

		IL T operator<<(uint8_t s) const {return _mm_slli_epi32(m,s);}
		IL T operator>>(uint8_t s) const {return _mm_srli_epi32(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm_slli_epi32(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm_srli_epi32(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm_min_ps(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm_max_ps(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm_mask_blend_ps(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm_and_ps(a.m,M128_TP32_SIGN);};
	IL U lt(ARGA,ARGB) {return _mm_cmp_ps_mask(a.m,b.m,0x11);};
	IL U gt(ARGA,ARGB) {return _mm_cmp_ps_mask(a.m,b.m,0x1e);};
	IL U eq(ARGA,ARGB) {return _mm_cmp_ps_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm_cmp_ps_mask(a.m,b.m,0x12);};
	IL U ge(ARGA,ARGB) {return _mm_cmp_ps_mask(a.m,b.m,0x1d);};
	IL U ne(ARGA,ARGB) {return _mm_cmp_ps_mask(a.m,b.m,0x1c);};
	IL T fma(ARGA,ARGB,ARGC) {return _mm_fmadd_ps(a.m,b.m,c.m);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_add_ps(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_sub_ps(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_mul_ps(src.m,mask,a.m,b.m);};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_div_ps(src.m,mask,a.m,b.m);};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_and_epi32(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_or_epi32(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_xor_epi32(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm_maskz_mov_epi32(mask,a.m);};
	IL T exp(ARGA) {return exp_ps(a.m);};
	IL T log(ARGA) {return log_ps(a.m);};
	//IL T pow(ARGA,ARGB) {return pow_ps(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T float8
	#define I 8
	#define B float
	#define U uint8_t
	struct alignas(sizeof(B)*I) float8
	{
		__m256 m;

		T() {};
		T(B x) : m(_mm256_set1_ps(x)) {};
		T(
			B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07
		) : m(_mm256_set_ps(f07,f06,f05,f04,f03,f02,f01,f00)) {};
		T(__m256 x) : m(x) {};

		IL T operator+(const T &a) const {return _mm256_add_ps(m,a.m);};
		IL T operator-(const T &a) const {return _mm256_sub_ps(m,a.m);};
		IL T operator*(const T &a) const {return _mm256_mul_ps(m,a.m);};
		IL T operator/(const T &a) const {return _mm256_div_ps(m,a.m);};
		IL T& operator+=(const T &a) {m = _mm256_add_ps(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm256_sub_ps(m,a.m); return *this;};
		IL T& operator*=(const T &a) {m = _mm256_mul_ps(m,a.m); return *this;};
		IL T& operator/=(const T &a) {m = _mm256_div_ps(m,a.m); return *this;};

		IL U operator<(ARGA) const {return _mm256_cmp_ps_mask(m,a.m,0x11);};
		IL U operator>(ARGA) const {return _mm256_cmp_ps_mask(m,a.m,0x1e);};
		IL U operator==(ARGA) const {return _mm256_cmp_ps_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm256_cmp_ps_mask(m,a.m,0x12);};
		IL U operator>=(ARGA) const {return _mm256_cmp_ps_mask(m,a.m,0x1d);};
		IL U operator!=(ARGA) const {return _mm256_cmp_ps_mask(m,a.m,0x1c);};

		IL T operator<<(uint8_t s) const {return _mm256_slli_epi32(m,s);}
		IL T operator>>(uint8_t s) const {return _mm256_srli_epi32(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm256_slli_epi32(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm256_srli_epi32(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		static IL T zero() {T r; r.m = _mm256_setzero_ps(); return r;};
		static IL T loadu(B* const f) {T x; x.m = _mm256_loadu_ps(f); return x;};
		static IL T load(B* const f) {T x; x.m = _mm256_load_ps(f); return x;};
		static IL T maskz_loadu(B* const f, U m) {T x; _mm256_mask_loadu_ps(x.m,m,f); return x;};
		static IL void storeu(B* const f, ARGA) {_mm256_storeu_ps(f,a.m);};
		static IL void store(B* const f, ARGA) {_mm256_store_ps(f,a.m);};
		static IL void mask_storeu(B* const f, ARGA, U m) {_mm256_mask_storeu_ps(f,m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm256_min_ps(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm256_max_ps(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm256_mask_blend_ps(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm256_and_ps(a.m,M256_TP32_SIGN);};
	IL U lt(ARGA,ARGB) {return _mm256_cmp_ps_mask(a.m,b.m,0x11);};
	IL U gt(ARGA,ARGB) {return _mm256_cmp_ps_mask(a.m,b.m,0x1e);};
	IL U eq(ARGA,ARGB) {return _mm256_cmp_ps_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm256_cmp_ps_mask(a.m,b.m,0x12);};
	IL U ge(ARGA,ARGB) {return _mm256_cmp_ps_mask(a.m,b.m,0x1d);};
	IL U ne(ARGA,ARGB) {return _mm256_cmp_ps_mask(a.m,b.m,0x1c);};
	IL T fma(ARGA,ARGB,ARGC) {return _mm256_fmadd_ps(a.m,b.m,c.m);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_add_ps(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_sub_ps(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_mul_ps(src.m,mask,a.m,b.m);};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_div_ps(src.m,mask,a.m,b.m);};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_and_epi32(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_or_epi32(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_xor_epi32(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm256_maskz_mov_epi32(mask,a.m);};
	IL U bit_and(ARGA,ARGB) {return _mm256_test_epi32_mask(a.m,b.m);};
	IL T fast_exp(ARGA)
	{
		constexpr float aa = (1 << 23) / 0.69314718f;
		constexpr float bb = (1 << 23) * (127 - 0.043677448f);
		float8 x = float8(aa) * a + float8(bb);

		constexpr float cc = (1 << 23);
		constexpr float dd = (1 << 23) * 255;

		uint8_t mask = x < float8(cc) | x > float8(dd);
		x = blend(x,0.0f,mask&x<float8(cc));
		return x;
	};
	IL T exp(ARGA) {return exp256_ps(a.m);};
	IL T log(ARGA) {return log256_ps(a.m);};
	//IL T pow(ARGA,ARGB) {return _mm256_pow_ps(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T float16
	#define I 16
	#define B float
	#define U uint16_t
	struct alignas(sizeof(B)*I) float16
	{
		__m512 m;

		T() {};
		T(B x) : m(_mm512_set1_ps(x)) {};
		T(
			B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07,
			B f08, B f09, B f10, B f11, B f12, B f13, B f14, B f15
		) : m(_mm512_set_ps(f15,f14,f13,f12,f11,f10,f09,f08,f07,f06,f05,f04,f03,f02,f01,f00)) {};
		T(__m512 x) : m(x) {};

		IL T operator+(const T &a) const {return _mm512_add_ps(m,a.m);};
		IL T operator-(const T &a) const {return _mm512_sub_ps(m,a.m);};
		IL T operator*(const T &a) const {return _mm512_mul_ps(m,a.m);};
		IL T operator/(const T &a) const {return _mm512_div_ps(m,a.m);};
		IL T& operator+=(const T &a) {m = _mm512_add_ps(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm512_sub_ps(m,a.m); return *this;};
		IL T& operator*=(const T &a) {m = _mm512_mul_ps(m,a.m); return *this;};
		IL T& operator/=(const T &a) {m = _mm512_div_ps(m,a.m); return *this;};

		IL U operator<(ARGA) const {return _mm512_cmp_ps_mask(m,a.m,0x11);};
		IL U operator>(ARGA) const {return _mm512_cmp_ps_mask(m,a.m,0x1e);};
		IL U operator==(ARGA) const {return _mm512_cmp_ps_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm512_cmp_ps_mask(m,a.m,0x12);};
		IL U operator>=(ARGA) const {return _mm512_cmp_ps_mask(m,a.m,0x1d);};
		IL U operator!=(ARGA) const {return _mm512_cmp_ps_mask(m,a.m,0x1c);};

		IL T operator<<(uint8_t s) const {return _mm512_slli_epi32(m,s);}
		IL T operator>>(uint8_t s) const {return _mm512_srli_epi32(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm512_slli_epi32(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm512_srli_epi32(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		static IL T zero() {T r; r.m = _mm512_setzero_ps(); return r;};
		static IL T loadu(B* const f) {T x; x.m = _mm512_loadu_ps(f); return x;};
		static IL T load(B* const f) {T x; x.m = _mm512_load_ps(f); return x;};
		static IL T maskz_loadu(B* const f, U m) {T x; _mm512_mask_loadu_ps(x.m,m,f); return x;};
		static IL void storeu(B* const f, ARGA) {_mm512_storeu_ps(f,a.m);};
		static IL void store(B* const f, ARGA) {_mm512_store_ps(f,a.m);};
		static IL void mask_storeu(B* const f, ARGA, U m) {_mm512_mask_storeu_ps(f,m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm512_min_ps(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm512_max_ps(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm512_mask_blend_ps(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm512_and_ps(a.m,M512_TP32_SIGN);};
	IL U lt(ARGA,ARGB) {return _mm512_cmp_ps_mask(a.m,b.m,0x11);};
	IL U gt(ARGA,ARGB) {return _mm512_cmp_ps_mask(a.m,b.m,0x1e);};
	IL U eq(ARGA,ARGB) {return _mm512_cmp_ps_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm512_cmp_ps_mask(a.m,b.m,0x12);};
	IL U ge(ARGA,ARGB) {return _mm512_cmp_ps_mask(a.m,b.m,0x1d);};
	IL U ne(ARGA,ARGB) {return _mm512_cmp_ps_mask(a.m,b.m,0x1c);};
	IL T fma(ARGA,ARGB,ARGC) {return _mm512_fmadd_ps(a.m,b.m,c.m);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_add_ps(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_sub_ps(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_mul_ps(src.m,mask,a.m,b.m);};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_div_ps(src.m,mask,a.m,b.m);};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_and_epi32(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_or_epi32(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_xor_epi32(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm512_maskz_mov_epi32(mask,a.m);};
	IL T exp(ARGA) {return exp512_ps(a.m);};
	IL T log(ARGA) {return log512_ps(a.m);};
	//IL T pow(ARGA,ARGB) {return _mm512_pow_ps(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T double2
	#define I 2
	#define B double
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m128 m;

		T() {};
		T(B x) : m(_mm_set1_pd(x)) {};
		T(B d00, B d01) : m(_mm_set_pd(d01,d00)) {};
		T(__m128 x) : m(x) {};

		IL T operator+(const T &a) const {return _mm_add_pd(m,a.m);};
		IL T operator-(const T &a) const {return _mm_sub_pd(m,a.m);};
		IL T operator*(const T &a) const {return _mm_mul_pd(m,a.m);};
		IL T operator/(const T &a) const {return _mm_div_pd(m,a.m);};
		IL T& operator+=(const T &a) {m = _mm_add_pd(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm_sub_pd(m,a.m); return *this;};
		IL T& operator*=(const T &a) {m = _mm_mul_pd(m,a.m); return *this;};
		IL T& operator/=(const T &a) {m = _mm_div_pd(m,a.m); return *this;};

		IL U operator<(ARGA) const {return _mm_cmp_pd_mask(m,a.m,0x11);};
		IL U operator>(ARGA) const {return _mm_cmp_pd_mask(m,a.m,0x1e);};
		IL U operator==(ARGA) const {return _mm_cmp_pd_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm_cmp_pd_mask(m,a.m,0x12);};
		IL U operator>=(ARGA) const {return _mm_cmp_pd_mask(m,a.m,0x1d);};
		IL U operator!=(ARGA) const {return _mm_cmp_pd_mask(m,a.m,0x1c);};

		IL T operator<<(uint8_t s) const {return _mm_slli_epi64(m,s);}
		IL T operator>>(uint8_t s) const {return _mm_srli_epi64(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm_slli_epi64(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm_srli_epi64(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm_min_pd(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm_max_pd(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm_mask_blend_pd(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm_and_pd(a.m,M128_TP32_SIGN);};
	IL U lt(ARGA,ARGB) {return _mm_cmp_pd_mask(a.m,b.m,0x11);};
	IL U gt(ARGA,ARGB) {return _mm_cmp_pd_mask(a.m,b.m,0x1e);};
	IL U eq(ARGA,ARGB) {return _mm_cmp_pd_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm_cmp_pd_mask(a.m,b.m,0x12);};
	IL U ge(ARGA,ARGB) {return _mm_cmp_pd_mask(a.m,b.m,0x1d);};
	IL U ne(ARGA,ARGB) {return _mm_cmp_pd_mask(a.m,b.m,0x1c);};
	IL T fma(ARGA,ARGB,ARGC) {return _mm_fmadd_pd(a.m,b.m,c.m);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_add_pd(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_sub_pd(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_mul_pd(src.m,mask,a.m,b.m);};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_div_pd(src.m,mask,a.m,b.m);};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_and_epi64(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_or_epi64(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_xor_epi64(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm_maskz_mov_epi64(mask,a.m);};
	//IL T exp(ARGA) {return exp_pd(a.m);};
	//IL T log(ARGA) {return log_pd(a.m);};
	//IL T pow(ARGA,ARGB) {return pow_ps(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T double4
	#define I 4
	#define B double
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m256 m;

		T() {};
		T(B x) : m(_mm256_set1_pd(x)) {};
		T(B d00, B d01, B d02, B d03) : m(_mm256_set_pd(d03,d02,d01,d00)) {};
		T(__m256 x) : m(x) {};

		IL T operator+(const T &a) const {return _mm256_add_pd(m,a.m);};
		IL T operator-(const T &a) const {return _mm256_sub_pd(m,a.m);};
		IL T operator*(const T &a) const {return _mm256_mul_pd(m,a.m);};
		IL T operator/(const T &a) const {return _mm256_div_pd(m,a.m);};
		IL T& operator+=(const T &a) {m = _mm256_add_pd(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm256_sub_pd(m,a.m); return *this;};
		IL T& operator*=(const T &a) {m = _mm256_mul_pd(m,a.m); return *this;};
		IL T& operator/=(const T &a) {m = _mm256_div_pd(m,a.m); return *this;};

		IL U operator<(ARGA) const {return _mm256_cmp_pd_mask(m,a.m,0x11);};
		IL U operator>(ARGA) const {return _mm256_cmp_pd_mask(m,a.m,0x1e);};
		IL U operator==(ARGA) const {return _mm256_cmp_pd_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm256_cmp_pd_mask(m,a.m,0x12);};
		IL U operator>=(ARGA) const {return _mm256_cmp_pd_mask(m,a.m,0x1d);};
		IL U operator!=(ARGA) const {return _mm256_cmp_pd_mask(m,a.m,0x1c);};

		IL T operator<<(uint8_t s) const {return _mm256_slli_epi64(m,s);}
		IL T operator>>(uint8_t s) const {return _mm256_srli_epi64(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm256_slli_epi64(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm256_srli_epi64(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm256_min_pd(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm256_max_pd(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm256_mask_blend_pd(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm256_and_pd(a.m,M256_TP64_SIGN);};
	IL U lt(ARGA,ARGB) {return _mm256_cmp_pd_mask(a.m,b.m,0x11);};
	IL U gt(ARGA,ARGB) {return _mm256_cmp_pd_mask(a.m,b.m,0x1e);};
	IL U eq(ARGA,ARGB) {return _mm256_cmp_pd_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm256_cmp_pd_mask(a.m,b.m,0x12);};
	IL U ge(ARGA,ARGB) {return _mm256_cmp_pd_mask(a.m,b.m,0x1d);};
	IL U ne(ARGA,ARGB) {return _mm256_cmp_pd_mask(a.m,b.m,0x1c);};
	IL T fma(ARGA,ARGB,ARGC) {return _mm256_fmadd_pd(a.m,b.m,c.m);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_add_pd(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_sub_pd(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_mul_pd(src.m,mask,a.m,b.m);};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_div_pd(src.m,mask,a.m,b.m);};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_and_epi64(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_or_epi64(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_xor_epi64(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm256_maskz_mov_epi64(mask,a.m);};
	//IL T exp(ARGA) {return exp_pd(a.m);};
	//IL T log(ARGA) {return log_pd(a.m);};
	//IL T pow(ARGA,ARGB) {return pow_ps(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int8_16
	#define I 16
	#define B int8_t
	#define U uint16_t
	struct alignas(sizeof(B)*I) T
	{
		__m128i m;

		T() {};
		T(__m128i x) : m(x) {};
		T(B x) : m(_mm_maskz_set1_epi8(UINT16_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm_add_epi8(m,a.m);};
		IL T operator-(const T &a) const {return _mm_sub_epi8(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm_add_epi8(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm_sub_epi8(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm_cmp_epi8_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm_cmp_epi8_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm_cmp_epi8_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm_cmp_epi8_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm_cmp_epi8_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm_cmp_epi8_mask(m,a.m,0x04);};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm_min_epi8(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm_max_epi8(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm_mask_blend_epi8(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm_abs_epi8(a.m);};
	IL U lt(ARGA,ARGB) {return _mm_cmp_epi8_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm_cmp_epi8_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm_cmp_epi8_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm_cmp_epi8_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm_cmp_epi8_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm_cmp_epi8_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_add_epi8(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_sub_epi8(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {return _mm_maskz_mov_epi8(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int8_32
	#define I 32
	#define B int8_t
	#define U uint32_t
	struct alignas(sizeof(B)*I) T
	{
		__m256i m;

		T() {};
		T(__m256i x) : m(x) {};
		T(B x) : m(_mm256_maskz_set1_epi8(UINT32_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm256_add_epi8(m,a.m);};
		IL T operator-(const T &a) const {return _mm256_sub_epi8(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm256_add_epi8(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm256_sub_epi8(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm256_cmp_epi8_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm256_cmp_epi8_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm256_cmp_epi8_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm256_cmp_epi8_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm256_cmp_epi8_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm256_cmp_epi8_mask(m,a.m,0x04);};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm256_min_epi8(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm256_max_epi8(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm256_mask_blend_epi8(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm256_abs_epi8(a.m);};
	IL U lt(ARGA,ARGB) {return _mm256_cmp_epi8_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm256_cmp_epi8_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm256_cmp_epi8_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm256_cmp_epi8_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm256_cmp_epi8_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm256_cmp_epi8_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_add_epi8(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_sub_epi8(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {return _mm256_maskz_mov_epi8(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int8_64
	#define I 64
	#define B int8_t
	#define U uint64_t
	struct alignas(sizeof(B)*I) T
	{
		__m512i m;

		T() {};
		T(__m512i x) : m(x) {};
		T(B x) : m(_mm512_maskz_set1_epi8(UINT64_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm512_add_epi8(m,a.m);};
		IL T operator-(const T &a) const {return _mm512_sub_epi8(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm512_add_epi8(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm512_sub_epi8(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm512_cmp_epi8_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm512_cmp_epi8_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm512_cmp_epi8_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm512_cmp_epi8_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm512_cmp_epi8_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm512_cmp_epi8_mask(m,a.m,0x04);};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm512_min_epi8(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm512_max_epi8(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm512_mask_blend_epi8(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm512_abs_epi8(a.m);};
	IL U lt(ARGA,ARGB) {return _mm512_cmp_epi8_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm512_cmp_epi8_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm512_cmp_epi8_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm512_cmp_epi8_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm512_cmp_epi8_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm512_cmp_epi8_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_add_epi8(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_sub_epi8(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {return _mm512_maskz_mov_epi8(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int16_8
	#define I 8
	#define B int16_t
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m128i m;

		T() {};
		T(__m128i x) : m(x) {};
		T(B x) : m(_mm_maskz_set1_epi16(UINT8_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm_add_epi16(m,a.m);};
		IL T operator-(const T &a) const {return _mm_sub_epi16(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm_add_epi16(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm_sub_epi16(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm_cmp_epi16_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm_cmp_epi16_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm_cmp_epi16_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm_cmp_epi16_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm_cmp_epi16_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm_cmp_epi16_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm_slli_epi16(m,s);}
		IL T operator>>(uint8_t s) const {return _mm_srli_epi16(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm_slli_epi16(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm_srli_epi16(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		U operator&(ARGA) const {return _mm_test_epi16_mask(m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm_min_epi16(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm_max_epi16(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm_mask_blend_epi16(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm_abs_epi16(a.m);};
	IL U lt(ARGA,ARGB) {return _mm_cmp_epi16_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm_cmp_epi16_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm_cmp_epi16_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm_cmp_epi16_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm_cmp_epi16_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm_cmp_epi16_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_add_epi16(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_sub_epi16(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {return _mm_maskz_mov_epi16(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int16_16
	#define I 16
	#define B int16_t
	#define U uint16_t
	struct alignas(sizeof(B)*I) T
	{
		__m256i m;

		T() {};
		T(__m256i x) : m(x) {};
		T(B x) : m(_mm256_maskz_set1_epi16(UINT8_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm256_add_epi16(m,a.m);};
		IL T operator-(const T &a) const {return _mm256_sub_epi16(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm256_add_epi16(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm256_sub_epi16(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm256_cmp_epi16_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm256_cmp_epi16_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm256_cmp_epi16_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm256_cmp_epi16_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm256_cmp_epi16_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm256_cmp_epi16_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm256_slli_epi16(m,s);}
		IL T operator>>(uint8_t s) const {return _mm256_srli_epi16(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm256_slli_epi16(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm256_srli_epi16(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		U operator&(ARGA) const {return _mm256_test_epi16_mask(m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm256_min_epi16(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm256_max_epi16(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm256_mask_blend_epi16(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm256_abs_epi16(a.m);};
	IL U lt(ARGA,ARGB) {return _mm256_cmp_epi16_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm256_cmp_epi16_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm256_cmp_epi16_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm256_cmp_epi16_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm256_cmp_epi16_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm256_cmp_epi16_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_add_epi16(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_sub_epi16(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {return _mm256_maskz_mov_epi16(mask,a.m);};
	IL uint8_t bit_and(ARGA,ARGB) {return _mm256_test_epi16_mask(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int16_32
	#define I 32
	#define B int16_t
	#define U uint32_t
	struct alignas(sizeof(B)*I) T
	{
		__m512i m;

		T() {};
		T(__m512i x) : m(x) {};
		T(B x) : m(_mm512_maskz_set1_epi16(UINT8_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm512_add_epi16(m,a.m);};
		IL T operator-(const T &a) const {return _mm512_sub_epi16(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm512_add_epi16(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm512_sub_epi16(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm512_cmp_epi16_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm512_cmp_epi16_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm512_cmp_epi16_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm512_cmp_epi16_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm512_cmp_epi16_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm512_cmp_epi16_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm512_slli_epi16(m,s);}
		IL T operator>>(uint8_t s) const {return _mm512_srli_epi16(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm512_slli_epi16(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm512_srli_epi16(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		U operator&(ARGA) const {return _mm512_test_epi16_mask(m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm512_min_epi16(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm512_max_epi16(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm512_mask_blend_epi16(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm512_abs_epi16(a.m);};
	IL U lt(ARGA,ARGB) {return _mm512_cmp_epi16_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm512_cmp_epi16_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm512_cmp_epi16_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm512_cmp_epi16_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm512_cmp_epi16_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm512_cmp_epi16_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_add_epi16(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_sub_epi16(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {return _mm512_maskz_mov_epi16(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int32_8
	#define I 8
	#define B int32_t
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m256i m;

		T() {};
		T(__m256i x) : m(x) {};
		T(B x) : m(_mm256_maskz_set1_epi32(UINT8_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm256_add_epi32(m,a.m);};
		IL T operator-(const T &a) const {return _mm256_sub_epi32(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm256_add_epi32(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm256_sub_epi32(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm256_cmp_epi32_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm256_cmp_epi32_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm256_cmp_epi32_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm256_cmp_epi32_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm256_cmp_epi32_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm256_cmp_epi32_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm256_slli_epi32(m,s);}
		IL T operator>>(uint8_t s) const {return _mm256_srli_epi32(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm256_slli_epi32(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm256_srli_epi32(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		//TODO: this is not quite normal, but it is in use...
		U operator&(ARGA) const {return _mm256_test_epi32_mask(m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm256_min_epi32(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm256_max_epi32(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm256_mask_blend_epi32(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm256_abs_epi32(a.m);};
	IL U lt(ARGA,ARGB) {return _mm256_cmp_epi32_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm256_cmp_epi32_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm256_cmp_epi32_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm256_cmp_epi32_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm256_cmp_epi32_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm256_cmp_epi32_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_add_epi32(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_sub_epi32(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_and_epi32(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_or_epi32(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_xor_epi32(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm256_maskz_mov_epi32(mask,a.m);};
	IL U bit_and(ARGA,ARGB) {return _mm256_test_epi32_mask(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int32_16
	#define I 16
	#define B int32_t
	#define U uint16_t
	struct alignas(sizeof(B)*I) T
	{
		__m512i m;

		T() {};
		T(__m512i x) : m(x) {};
		T(B x) : m(_mm512_maskz_set1_epi32(UINT8_MAX,x)) {};

		IL T operator+(const T &a) const {return _mm512_add_epi32(m,a.m);};
		IL T operator-(const T &a) const {return _mm512_sub_epi32(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm512_add_epi32(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm512_sub_epi32(m,a.m); return *this;};
		IL T& operator*=(const T &a) {m = _mm512_mul_epi32(m,a.m); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm512_cmp_epi32_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm512_cmp_epi32_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm512_cmp_epi32_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm512_cmp_epi32_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm512_cmp_epi32_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm512_cmp_epi32_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm512_slli_epi32(m,s);}
		IL T operator>>(uint8_t s) const {return _mm512_srli_epi32(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm512_slli_epi32(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm512_srli_epi32(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};

		U operator&(ARGA) const {return _mm512_test_epi32_mask(m,a.m);};
	};

	IL T min(ARGA,ARGB) {return _mm512_min_epi32(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm512_max_epi32(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm512_mask_blend_epi32(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm512_abs_epi32(a.m);};
	IL U lt(ARGA,ARGB) {return _mm512_cmp_epi32_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm512_cmp_epi32_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm512_cmp_epi32_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm512_cmp_epi32_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm512_cmp_epi32_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm512_cmp_epi32_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_add_epi32(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_sub_epi32(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_and_epi32(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_or_epi32(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_xor_epi32(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm512_maskz_mov_epi32(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int64_2
	#define I 2
	#define B int64_t
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m128i m;
		
		T() {};
		T(__m128i x) : m(x) {};
		T(B x) : m(_mm_maskz_set1_epi64(x,UINT8_MAX)) {};

		IL T operator+(const T &a) const {return _mm_add_epi64(m,a.m);};
		IL T operator-(const T &a) const {return _mm_sub_epi64(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm_add_epi64(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm_sub_epi64(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm_cmp_epi64_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm_cmp_epi64_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm_cmp_epi64_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm_cmp_epi64_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm_cmp_epi64_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm_cmp_epi64_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm_slli_epi64(m,s);}
		IL T operator>>(uint8_t s) const {return _mm_srli_epi64(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm_slli_epi64(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm_srli_epi64(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm_min_epi64(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm_max_epi64(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm_mask_blend_epi64(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm_abs_epi64(a.m);};
	IL U lt(ARGA,ARGB) {return _mm_cmp_epi64_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm_cmp_epi64_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm_cmp_epi64_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm_cmp_epi64_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm_cmp_epi64_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm_cmp_epi64_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_add_epi64(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_sub_epi64(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_and_epi64(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_or_epi64(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm_mask_xor_epi64(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm_maskz_mov_epi64(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int64_4
	#define I 4
	#define B int64_t
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m256i m;

		T() {};
		T(__m256i x) : m(x) {};
		T(B x) : m(_mm256_maskz_set1_epi64(x,UINT8_MAX)) {};

		IL T operator+(const T &a) const {return _mm256_add_epi64(m,a.m);};
		IL T operator-(const T &a) const {return _mm256_sub_epi64(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm256_add_epi64(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm256_sub_epi64(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm256_cmp_epi64_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm256_cmp_epi64_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm256_cmp_epi64_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm256_cmp_epi64_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm256_cmp_epi64_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm256_cmp_epi64_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm256_slli_epi64(m,s);}
		IL T operator>>(uint8_t s) const {return _mm256_srli_epi64(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm256_slli_epi64(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm256_srli_epi64(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm256_min_epi64(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm256_max_epi64(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm256_mask_blend_epi64(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm256_abs_epi64(a.m);};
	IL U lt(ARGA,ARGB) {return _mm256_cmp_epi64_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm256_cmp_epi64_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm256_cmp_epi64_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm256_cmp_epi64_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm256_cmp_epi64_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm256_cmp_epi64_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_add_epi64(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_sub_epi64(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_and_epi64(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_or_epi64(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm256_mask_xor_epi64(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm256_maskz_mov_epi64(mask,a.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T int64_8
	#define I 8
	#define B int64_t
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		__m512i m;

		T() {};
		T(__m512i x) : m(x) {};
		T(
			B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07
		) : m(_mm512_set_epi64(f07,f06,f05,f04,f03,f02,f01,f00)) {};
		T(B x) : m(_mm512_set1_epi64(x)) {};

		IL T operator+(const T &a) const {return _mm512_add_epi64(m,a.m);};
		IL T operator-(const T &a) const {return _mm512_sub_epi64(m,a.m);};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {m = _mm512_add_epi64(m,a.m); return *this;};
		IL T& operator-=(const T &a) {m = _mm512_sub_epi64(m,a.m); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {return _mm512_cmp_epi64_mask(m,a.m,0x01);};
		IL U operator>(ARGA) const {return _mm512_cmp_epi64_mask(m,a.m,0x06);};
		IL U operator==(ARGA) const {return _mm512_cmp_epi64_mask(m,a.m,0x00);};
		IL U operator<=(ARGA) const {return _mm512_cmp_epi64_mask(m,a.m,0x02);};
		IL U operator>=(ARGA) const {return _mm512_cmp_epi64_mask(m,a.m,0x05);};
		IL U operator!=(ARGA) const {return _mm512_cmp_epi64_mask(m,a.m,0x04);};

		IL T operator<<(uint8_t s) const {return _mm512_slli_epi64(m,s);}
		IL T operator>>(uint8_t s) const {return _mm512_srli_epi64(m,s);}
		IL T& operator<<=(uint8_t s) {m = _mm512_slli_epi64(m,s); return *this;};
		IL T& operator>>=(uint8_t s) {m = _mm512_srli_epi64(m,s); return *this;};

		IL B operator[](int i) const {return ((B*)&m)[i];};
		IL B& operator[](int i) {return ((B*)&m)[i];};
	};

	IL T min(ARGA,ARGB) {return _mm512_min_epi64(a.m,b.m);};
	IL T max(ARGA,ARGB) {return _mm512_max_epi64(a.m,b.m);};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {return _mm512_mask_blend_epi64(mask,a.m,b.m);};
	IL T abs(ARGA) {return _mm512_abs_epi64(a.m);};
	IL U lt(ARGA,ARGB) {return _mm512_cmp_epi64_mask(a.m,b.m,0x01);};
	IL U gt(ARGA,ARGB) {return _mm512_cmp_epi64_mask(a.m,b.m,0x06);};
	IL U eq(ARGA,ARGB) {return _mm512_cmp_epi64_mask(a.m,b.m,0x00);};
	IL U le(ARGA,ARGB) {return _mm512_cmp_epi64_mask(a.m,b.m,0x02);};
	IL U ge(ARGA,ARGB) {return _mm512_cmp_epi64_mask(a.m,b.m,0x05);};
	IL U ne(ARGA,ARGB) {return _mm512_cmp_epi64_mask(a.m,b.m,0x04);};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_add_epi64(src.m,mask,a.m,b.m);};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_sub_epi64(src.m,mask,a.m,b.m);};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_and_epi64(src.m,mask,a.m,b.m);};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_or_epi64(src.m,mask,a.m,b.m);};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {return _mm512_mask_xor_epi64(src.m,mask,a.m,b.m);};
	IL T maskz_mov(ARGA,ARGM) {return _mm512_maskz_mov_epi64(mask,a.m);};
	IL U bit_and(ARGA,ARGB) {return _mm512_test_epi64_mask(a.m,b.m);};
	#undef T
	#undef I
	#undef B
	#undef U

	#undef ARGA
	#undef ARGB
	#undef ARGC
	#undef ARGS
	#undef ARGM

	inline bool if_any(const uint16_t &a) {return a != 0;};
};

#undef LOOP
#undef OPC
#undef OPE
#undef IL
#undef RC
#undef CMP
#undef MASK_OP