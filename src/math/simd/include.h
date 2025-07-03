#pragma once
#ifdef ENABLE_SIMD
#include "hw.h"
#else
#include "sw.h"
#endif

namespace simd
{
	using f32_4 = float4;
	using f32_8 = float8;
	using f32_16 = float16;

	using i32_8 = int32_8;
	using i32_16 = int32_16;

	using i64_2 = int64_2;
	using i64_4 = int64_4;
	using i64_8 = int64_8;
};