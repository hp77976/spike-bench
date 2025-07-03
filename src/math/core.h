#pragma once
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string.h>
#include "../common.h"

namespace math
{
	//y: predicted answer
	//yt: real answer
	//returns loss
	CPU_GPU inline float mse(float y, float yt)
	{
		return 0.5f * (y - yt) * (y - yt);
	};

	/*
		output (prediction)
		target (truth)
	*/
	CPU_GPU inline float mae(float o, float t)
	{
		return std::abs(t - o);
	};

	CPU_GPU inline float boxcar(float v, float vmin, float vmax)
	{
		if(vmin < v && v < vmax)
			return 1.0f;
		return 0.0f;
	};

	CPU_GPU inline float heaviside(float x)
	{
		return (x > 0.0f) ? 1.0f : 0.0f;
	};

	inline float length(const std::vector<float> &v)
	{
		float x = 0.0f;
		for(uint32_t i = 0; i < v.size(); i++)
			x += v[i] * v[i];
		return std::sqrt(x);
	};

	inline std::vector<float> normalize(const std::vector<float> &v)
	{
		float l = length(v);
		std::vector<float> r = v;
		for(uint32_t i = 0; i < r.size(); i++)
			r[i] /= l;
		return r;
	};

	inline float fast_exp(float x)
	{
#define USE_FAST_EXP
#ifdef USE_FAST_EXP
		constexpr float a = (1 << 23) / 0.69314718f;
		constexpr float b = (1 << 23) * (127 - 0.043677448f);
		x = a * x + b;

		// Remove these lines if bounds checking is not needed
		constexpr float c = (1 << 23);
		constexpr float d = (1 << 23) * 255;
		if (x < c || x > d)
			x = (x < c) ? 0.0f : d;

		// With C++20 one can use std::bit_cast instead
		uint32_t n = static_cast<uint32_t>(x);
		memcpy(&x, &n, 4);
		return x;
#else
		return std::exp(x);
#endif
	};

	CPU_GPU inline uint32_t u64_bits(uint64_t u)
	{
		union
		{
			uint64_t u2;
			int64_t i2;
		} a;
		a.u2 = u;
		int64_t i = a.i2;

		i = i - ((i >> 1) & 0x5555555555555555);
		i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
		return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
	};

	CPU_GPU inline uint32_t calc_blocks(uint32_t total_size, uint32_t block_size)
	{
		return ((total_size + block_size - 1) / block_size);
	};

	template <typename T>
	CPU_GPU inline T ema(T alpha, T value, T x)
	{
		//return value - (alpha * (value - x));
		//return alpha * x + (1.0f - alpha) * value;
		return alpha * value + (T(1.0f) - alpha) * x;
	};

	struct range_t
	{
		float v_min = 0.0f;
		float v_max = 1.0f;
	};
};