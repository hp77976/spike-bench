#pragma once
#include "array.h"
#include "tensor.h"

namespace math
{
	struct conv_kernel_t
	{
		dim2 size = {4,4};
		i32 stride = 1;
		i32 padding = 0;
	};

	inline void conv_2d(
		const conv_kernel_t &kernel,
		const tensor<float> &a,
		const array<float> &b,
		array<float> &c
	)
	{
		i32 ix = a.x();
		i32 iy = a.y();

		i32 kx = std::min<i32>(kernel.size.x,ix);
		i32 ky = std::min<i32>(kernel.size.y,iy);

		i32 px = a.x();
		i32 py = a.y();

		i32 nx = c.x();
		i32 ny = c.y();

		i32 sx = kernel.stride;
		i32 sy = kernel.stride;

		for(i32 ci = 0; ci < nx; ci++)
		{
			for(i32 cj = 0; cj < ny; cj++)
			{
				float sum = 0.0f;
				for(i32 ki = 0; ki < kx; ki++)
					for(i32 kj = 0; kj < ky; kj++)
						sum += a.at(ci,cj,ki,kj) * b.at(ci+ki,cj+kj);
				c.at(ci,cj) += sum;
			}
		}
	};

	inline void conv_2d(
		const conv_kernel_t &kernel,
		const tensor<float> &a,
		const tensor<float> &b,
		tensor<float> &c
	)
	{
		i32 ix = a.x();
		i32 iy = a.y();

		i32 kx = std::min<i32>(kernel.size.x,ix);
		i32 ky = std::min<i32>(kernel.size.y,iy);

		i32 px = a.x();
		i32 py = a.y();

		i32 nx = c.x();
		i32 ny = c.y();

		i32 sx = kernel.stride;
		i32 sy = kernel.stride;

		for(i32 ci = 0; ci < nx; ci++)
		{
			for(i32 cj = 0; cj < ny; cj++)
			{
				float sum = 0.0f;
				for(i32 ki = 0; ki < kx; ki++)
					for(i32 kj = 0; kj < ky; kj++)
						sum += a.at(ci,cj,ki,kj) * b.at(ci+ki,cj+kj);
				c.at(ci,cj) += sum;
			}
		}
	};

	//template <typename E>
	inline void conv_2d_back(
		const conv_kernel_t &kernel,
		tensor<float> &w,
		const array<float> &g,
		const tensor<float> &et,
		const array<float> &ps,
		//const templates::exp_t<E> &s,
		float lr
	)
	{
		i32 ix = w.x();
		i32 iy = w.y();

		i32 kx = std::min<i32>(kernel.size.x,ix);
		i32 ky = std::min<i32>(kernel.size.y,iy);

		i32 nx = w.x();
		i32 ny = w.y();

		i32 sx = kernel.stride;
		i32 sy = kernel.stride;

		for(i32 ci = 0; ci < nx; ci++)
			for(i32 cj = 0; cj < ny; cj++)
				for(i32 ki = 0; ki < kx; ki++)
					for(i32 kj = 0; kj < ky; kj++)
						w.at(ci,cj,ki,kj) -= g.at(ci,cj) * et.at(ci+ki,cj+kj) * ps.at(ci+ki,cj+kj) * lr;
						//w.at(ci,cj,ki,kj) -= g.at(ci,cj) * s[(ci+ki)*ny+cj+kj] * lr;
	};
};