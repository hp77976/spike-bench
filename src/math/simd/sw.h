#pragma once
#include <cstdint>
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
//#include "avx.h"

#define LOOP for(uint32_t i = 0; i < I; i++)
#define OPC(o) c[i] = this->operator[](i) o a[i];
#define OPS(o) c[i] = this->operator[](i) o s;
#define OPSF(o) c[i] = this->u[i] o s;
#define OPE(o) this->operator[](i) o##= a[i];
#define OPH(o) this->operator[](i) o##= s;
#define OPHF(o) this->u[i] o##= s;
#define IL inline
#define RC return c
#define CMP(o) c |= this->operator[](i) o a[i] ? 1lu << i : 0
#define MASK_OP(o) c[i] = (mask >> i) & 0x1 ? a[i] o b[i] : src[i]
#define MASKU_OP(o) c.u[i] = (mask >> i) & 0x1 ? a.u[i] o b.u[i] : src.u[i]

#ifdef TEST_SIMD
namespace __sw_simd__
#else
namespace simd
#endif
{
	#define ARGA const T &a
	#define ARGB const T &b
	#define ARGC const T &c
	#define ARGS const T &src
	#define ARGM const U mask

	#define T float4
	#define I 4
	#define B float
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		union {B f[I]; uint32_t u[I];};

		T() {};
		T(B x) {LOOP f[i] = x;};
		T(B f00, B f01, B f02, B f03) {f[0] = f00; f[1] = f01; f[2] = f02; f[3] = f03;};
		T(uint32_t x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPSF(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPSF(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPHF(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPHF(>>); return *this;};

		IL B operator[](int i) const {return f[i];};
		IL B& operator[](int i) {return f[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T fma(ARGA,ARGB,ARGC) {T d; LOOP d[i] = a[i] * b[i] + c[i]; return d;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.f[i] = (mask >> i) & 0x1u ? a.f[i] : 0.0f; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
	IL T exp(ARGA) {T c; LOOP c[i] = std::exp(a[i]); RC;};
	IL T log(ARGA) {T c; LOOP c[i] = std::log(a[i]); RC;};
	IL T pow(ARGA,ARGB) {T c; LOOP c[i] = std::pow(a[i],b[i]); RC;};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T float8
	#define I 8
	#define B float
	#define U uint8_t
	struct alignas(sizeof(B)*I) T
	{
		union {B f[I]; uint32_t u[I];};

		T() {};
		T(B x) {LOOP f[i] = x;};
		T(B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07)
		{
			f[0] = f00; f[1] = f01; f[2] = f02; f[3] = f03;
			f[4] = f04; f[5] = f05; f[6] = f06; f[7] = f07;
		};
		T(uint32_t x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPSF(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPSF(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPHF(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPHF(>>); return *this;};

		IL B operator[](int i) const {return f[i];};
		IL B& operator[](int i) {return f[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T fma(ARGA,ARGB,ARGC) {T d; LOOP d[i] = a[i] * b[i] + c[i]; return d;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.f[i] = (mask >> i) & 0x1u ? a.f[i] : 0.0f; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
	IL T exp(ARGA) {T c; LOOP c[i] = std::exp(a[i]); RC;};
	IL T log(ARGA) {T c; LOOP c[i] = std::log(a[i]); RC;};
	IL T pow(ARGA,ARGB) {T c; LOOP c[i] = std::pow(a[i],b[i]); RC;};
	#undef T
	#undef I
	#undef B
	#undef U

	#define T float16
	#define I 16
	#define B float
	#define U uint16_t
	struct alignas(sizeof(B)*I) T
	{
		union {B f[I]; uint32_t u[I];};

		T() {};
		T(B x) {LOOP f[i] = x;};
		T(
			B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07,
			B f08, B f09, B f10, B f11, B f12, B f13, B f14, B f15
		)
		{
			f[ 0] = f00; f[ 1] = f01; f[ 2] = f02; f[ 3] = f03;
			f[ 4] = f04; f[ 5] = f05; f[ 6] = f06; f[ 7] = f07;
			f[ 8] = f08; f[ 9] = f09; f[10] = f10; f[11] = f11;
			f[12] = f12; f[13] = f13; f[14] = f14; f[15] = f15;
		};
		T(uint32_t x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPSF(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPSF(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPHF(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPHF(>>); return *this;};

		IL B operator[](int i) const {return f[i];};
		IL B& operator[](int i) {return f[i];};

		static IL T zero() {T r; LOOP r[i] = 0.0f; return r;};
		static IL T loadu(float* p) {T x; LOOP x[i] = p[i]; return x;};
		static IL void storeu(float* const p, const T &x) {LOOP p[i] = x[i];};
		static IL void store(float* const p, const T &x) {LOOP p[i] = x[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T fma(ARGA,ARGB,ARGC) {T d; LOOP d[i] = a[i] * b[i] + c[i]; return d;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.f[i] = (mask >> i) & 0x1u ? a.f[i] : 0.0f; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
	IL T exp(ARGA) {T c; LOOP c[i] = std::exp(a[i]); RC;};
	IL T log(ARGA) {T c; LOOP c[i] = std::log(a[i]); RC;};
	IL T pow(ARGA,ARGB) {T c; LOOP c[i] = std::pow(a[i],b[i]); RC;};
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
		union {B d[I]; uint64_t u[I];};

		T() {};
		T(B x) {LOOP d[i] = x;};
		T(B d00, B d01) {d[0] = d00; d[1] = d01;};
		T(uint32_t x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPSF(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPSF(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPHF(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPHF(>>); return *this;};

		IL B operator[](int i) const {return d[i];};
		IL B& operator[](int i) {return d[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T fma(ARGA,ARGB,ARGC) {T d; LOOP d[i] = a[i] * b[i] + c[i]; return d;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1u ? a[i] : 0.0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
	IL T exp(ARGA) {T c; LOOP c[i] = std::exp(a[i]); RC;};
	IL T log(ARGA) {T c; LOOP c[i] = std::log(a[i]); RC;};
	IL T pow(ARGA,ARGB) {T c; LOOP c[i] = std::pow(a[i],b[i]); RC;};
	#undef T
	#undef I
	#undef B
	#undef U

	const uint32_t TP32_SIGN = UINT32_MAX >> 1;
	const float4 M128_TP32_SIGN = TP32_SIGN;
	const float8 M256_TP32_SIGN = TP32_SIGN;
	const float16 M512_TP32_SIGN = TP32_SIGN;
	const float4 M128_ZERO = 0.0f;
	const float8 M256_ZERO = 0.0f;
	const float16 M512_ZERO = 0.0f;

	#define T int8_16
	#define I 16
	#define B int8_t
	#define U uint16_t
	struct alignas(sizeof(B)*I) T
	{
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};

		U operator&(ARGA) const {U c = 0; LOOP c |= u[i] & a[i] ? 1lu << i : 0; RC;};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};

		U operator&(ARGA) const {U c = 0; LOOP c |= u[i] & a[i] ? 1lu << i : 0; RC;};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};

		U operator&(ARGA) const {U c = 0; LOOP c |= u[i] & a[i] ? 1lu << i : 0; RC;};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07)
		{
			u[0] = f00; u[1] = f01; u[2] = f02; u[3] = f03;
			u[4] = f04; u[5] = f05; u[6] = f06; u[7] = f07;
		};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(
			B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07,
			B f08, B f09, B f10, B f11, B f12, B f13, B f14, B f15
		)
		{
			u[ 0] = f00; u[ 1] = f01; u[ 2] = f02; u[ 3] = f03;
			u[ 4] = f04; u[ 5] = f05; u[ 6] = f06; u[ 7] = f07;
			u[ 8] = f08; u[ 9] = f09; u[10] = f10; u[11] = f11;
			u[12] = f12; u[13] = f13; u[14] = f14; u[15] = f15;
		};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
		B u[I];

		T() {};
		T(B f00, B f01, B f02, B f03, B f04, B f05, B f06, B f07)
		{
			u[0] = f00; u[1] = f01; u[2] = f02; u[3] = f03;
			u[4] = f04; u[5] = f05; u[6] = f06; u[7] = f07;
		};
		T(B x) {LOOP u[i] = x;};

		IL T operator+(const T &a) const {T c; LOOP OPC(+); RC;};
		IL T operator-(const T &a) const {T c; LOOP OPC(-); RC;};
		IL T operator*(const T &a) const {T c; LOOP OPC(*); RC;};
		IL T operator/(const T &a) const {T c; LOOP OPC(/); RC;};
		IL T& operator+=(const T &a) {LOOP OPE(+); return *this;};
		IL T& operator-=(const T &a) {LOOP OPE(-); return *this;};
		IL T& operator*=(const T &a) {LOOP OPE(*); return *this;};
		IL T& operator/=(const T &a) {LOOP OPE(/); return *this;};

		IL U operator<(ARGA) const {U c = 0; LOOP CMP(<); RC;};
		IL U operator>(ARGA) const {U c = 0; LOOP CMP(>); RC;};
		IL U operator==(ARGA) const {U c = 0; LOOP CMP(==); RC;};
		IL U operator<=(ARGA) const {U c = 0; LOOP CMP(<=); RC;};
		IL U operator>=(ARGA) const {U c = 0; LOOP CMP(>=); RC;};
		IL U operator!=(ARGA) const {U c = 0; LOOP CMP(!=); RC;};

		IL T operator<<(uint8_t s) const {T c; LOOP OPS(<<); RC;};
		IL T operator>>(uint8_t s) const {T c; LOOP OPS(>>); RC;};
		IL T& operator<<=(uint8_t s) {LOOP OPH(<<); return *this;};
		IL T& operator>>=(uint8_t s) {LOOP OPH(>>); return *this;};

		IL B operator[](int i) const {return u[i];};
		IL B& operator[](int i) {return u[i];};
	};

	IL T min(ARGA,ARGB) {T c; LOOP c[i] = std::min(a[i],b[i]); RC;};
	IL T max(ARGA,ARGB) {T c; LOOP c[i] = std::max(a[i],b[i]); RC;};
	IL T clamp(ARGA,ARGB,ARGC) {return min(c,max(b,a));};
	IL T blend(ARGA,ARGB,ARGM) {T c; LOOP c[i] = (mask >> i) & 0x1 ? b[i] : a[i]; RC;};
	IL T abs(ARGA) {T c; LOOP c[i] = std::abs(a[i]); RC;};
	IL U lt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] < b[i] ? 1lu << i : 0; RC;};
	IL U gt(ARGA,ARGB) {U c = 0; LOOP c |= a[i] > b[i] ? 1lu << i : 0; RC;};
	IL U eq(ARGA,ARGB) {U c = 0; LOOP c |= a[i] == b[i] ? 1lu << i : 0; RC;};
	IL U le(ARGA,ARGB) {U c = 0; LOOP c |= a[i] <= b[i] ? 1lu << i : 0; RC;};
	IL U ge(ARGA,ARGB) {U c = 0; LOOP c |= a[i] >= b[i] ? 1lu << i : 0; RC;};
	IL U ne(ARGA,ARGB) {U c = 0; LOOP c |= a[i] != b[i] ? 1lu << i : 0; RC;};
	IL T mask_add(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(+); RC;};
	IL T mask_sub(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(-); RC;};
	IL T mask_mul(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(*); RC;};
	IL T mask_div(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASK_OP(/); RC;};
	IL T mask_and(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(&); RC;};
	IL T mask_or(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(|); RC;};
	IL T mask_xor(ARGS,ARGA,ARGB,ARGM) {T c; LOOP MASKU_OP(^); RC;};
	IL T maskz_mov(ARGA,ARGM) {T c; LOOP c.u[i] = (mask >> i) & 0x1u ? a.u[i] : 0; RC;};
	IL U bit_and(ARGA,ARGB) {U c = 0; LOOP c |= ((a.u[i]&b.u[i])?0x1:0x0) << i; RC;};
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
#undef OPS
#undef OPE
#undef OPH
#undef IL
#undef RC
#undef CMP
#undef MASK_OP
#undef MASKU_OP