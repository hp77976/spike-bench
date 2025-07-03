#pragma once
#include "rng.h"
#include "exp.h"
//#include <cblas.h>
#include <cstring>
#include <stdexcept>

namespace math
{
	template <typename T>
	struct alignas(64) tensor : public expr::exp_t<tensor<T>>
	{
		T* m_ptr = nullptr;
		dim4 m_dims = {0,0,0,0};
		i64 m_size = 0;
		i64 m_bytes = 0;

		void __check_bounds(i64 i) const
		{
#ifdef DEBUG_TENSORS
			if(i >= m_size)
				throw std::runtime_error("Tensor out of bounds!\n");
#endif
		};

		void __check_bounds(i64 i, i64 j, i64 k, i64 l) const
		{
#ifdef DEBUG_TENSORS
			if(i >= m_dims.x || j >= m_dims.y || k >= m_dims.z || l >= m_dims.w)
				throw std::runtime_error("Tensor out of bounds!\n");
#endif
		};

		void __check_size(const tensor &f) const
		{
#ifdef DEBUG_TENSORS
			if(f.size() != m_size)
				throw std::runtime_error("Tensor size mismatch!\n");
#endif
		};

		void __alloc(dim4 dims)
		{
#ifdef DEBUG_TENSORS
			if(dims.size() == 0)
				throw std::runtime_error("Null tensor allocation!\n");
#endif
			i64 new_size = dims.size();
			bool need_realloc = new_size > m_size;
			m_size = new_size;
			
			if(need_realloc)
			{
				m_dims = dims;
				i64 min_bytes = sizeof(T) * m_size;
				i64 blocks_needed = (min_bytes + 64 - 1) / 64;
				m_bytes = (blocks_needed * 64);

				if(m_ptr != nullptr)
					std::free(m_ptr);
				m_ptr = (T*)std::aligned_alloc(64,std::max<uint64_t>(1,m_bytes));
				
				memset(m_ptr,0,m_bytes);
			}
		};

		template <typename E>
		inline tensor& operator+=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] += expr[i];
			return *this;
		};

		template <typename E>
		inline tensor& operator-=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] -= expr[i];
			return *this;
		};

		template <typename E>
		inline tensor& operator*=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] *= expr[i];
			return *this;
		};

		template <typename E>
		inline tensor& operator/=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] /= expr[i];
			return *this;
		};

		template <typename E>
		inline tensor& operator=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = expr[i];
			return *this;
		};

		tensor& resize(i64 size)
		{
			__alloc({size,1,1,1});
			return *this;
		};

		//resizes (if necessary) and zeros all data
		tensor& resize(dim2 dims)
		{
			__alloc({dims.x,dims.y,1,1});
			return *this;
		};

		//resizes (if necessary) and zeros all data
		tensor& resize(dim4 dims)
		{
			__alloc(dims);
			return *this;
		};

		tensor& free()
		{
			if(m_ptr != nullptr)
			{
				std::free(m_ptr);
				m_ptr = nullptr;
			}
			m_dims = {0,0,0,0};
			m_size = 0;
			m_bytes = 0;
			return *this;
		};

		tensor& copy_from(const tensor &t)
		{
			if(m_size < t.size())
				resize(t.shape());
			memcpy(m_ptr,t.m_ptr,m_size*sizeof(T));
			return *this;
		};

		tensor& fill(T x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = x;
			return *this;
		};

		tensor& zero()
		{
			memset(m_ptr,0,m_bytes);
			return *this;
		};
	
		T at(i64 i) const
		{
			__check_bounds(i);
			return m_ptr[i];
		};

		T at(i64 i, i64 j) const
		{
			__check_bounds(i,j,0,0);
			return m_ptr[m_dims.at(i,j)];
		};

		T at(i64 i, i64 j, i64 k) const
		{
			__check_bounds(i,j,k,0);
			return m_ptr[m_dims.at(i,j,k)];
		};

		T at(i64 i, i64 j, i64 k, i64 l) const
		{
			__check_bounds(i,j,k,l);
			return m_ptr[m_dims.at(i,j,k,l)];
		};

		T& at(i64 i)
		{
			__check_bounds(i);
			return m_ptr[i];
		};

		T& at(i64 i, i64 j)
		{
			__check_bounds(i,j,0,0);
			return m_ptr[m_dims.at(i,j)];
		};

		T& at(i64 i, i64 j, i64 k)
		{
			__check_bounds(i,j,k,0);
			return m_ptr[m_dims.at(i,j,k)];
		};

		T& at(i64 i, i64 j, i64 k, i64 l)
		{
			__check_bounds(i,j,k,l);
			return m_ptr[m_dims.at(i,j,k,l)];
		};

		CPU_GPU inline operator T*() const {return m_ptr;};

		T operator[](i64 i) const
		{
			return m_ptr[i];
		};

		T& operator[](i64 i)
		{
			return m_ptr[i];
		};

		tensor& operator+=(T x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] += x;
			return *this;
		};

		tensor& operator-=(T x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] -= x;
			return *this;
		};

		tensor& operator*=(T x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] *= x;
			return *this;
		};

		tensor& operator/=(T x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] /= x;
			return *this;
		};

		tensor& operator+=(const tensor &t)
		{
			__check_size(t);
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] += t[i];
			return *this;
		};

		tensor& operator-=(const tensor &t)
		{
			__check_size(t);
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] -= t[i];
			return *this;
		};

		tensor& operator*=(const tensor &t)
		{
			__check_size(t);
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] *= t[i];
			return *this;
		};

		tensor& operator/=(const tensor &t)
		{
			__check_size(t);
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] /= t[i];
			return *this;
		};

		T* data() const {return m_ptr;};

		i64 size() const {return m_size;};

		i64 bytes() const {return m_bytes;};

		i64 x() const {return m_dims.x;};
		
		i64 y() const {return m_dims.y;};
		
		i64 z() const {return m_dims.z;};
		
		i64 w() const {return m_dims.w;};

		template <typename T2>
		inline T2* as() const {return (T2*)m_ptr;};

		tensor& randomize(rng_t &rng, T min = 0.0f, T max = 1.0f)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = (max - min) * rng.u() + min;
			return *this;
		};

		tensor& clamp(float min, float max)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = std::clamp<float>(m_ptr[i],min,max);
			return *this;
		};

		dim4 shape() const {return m_dims;};

		i64 highest_index() const
		{
			T x = -9999;
			i64 index = -1;
			for(i64 i = 0; i < m_size; i++)
			{
				if(m_ptr[i] > x)
				{
					x = m_ptr[i];
					index = i;
				}
			}
			return index;
		};
	};

	inline void dot(const tensor<float> &a, const tensor<float> &b, tensor<float> &c)
	{
		if(a.x() > 1 && a.y() > 1 && a.z() == 1 && a.w() == 1)
		{
			if(b.x() > 1 && b.y() == 1 && b.z() == 1 && b.w() == 1)
			{
				if((a.x() == c.x()) && (a.y() == b.x()))
				{
					/*cblas_sgemv(
						CblasColMajor,
						CblasNoTrans,
						a.x(),a.y(),1.0f,
						a.data(),a.x(),
						b.data(),1,0.0f,
						c.data(),1
					);*/
				}
				else
				{
					throw std::runtime_error("Unhandled tensor float dot case: 0!\n");
				}
			}
			else
			{
				throw std::runtime_error("Unhandled tensor float dot case: 1!\n");
			}
		}
		else
		{
			throw std::runtime_error("Unhandled tensor float dot case: 2!\n");
		}
	};

	inline void dot(const tensor<float> &a, const tensor<bool> &b, tensor<float> &c)
	{
		if(a.x() > 1 && a.y() > 1 && a.z() == 1 && a.w() == 1)
		{
			if(b.x() > 1 && b.y() == 1 && b.z() == 1 && b.w() == 1)
			{
				if((a.x() == c.x()) && (a.y() == b.x()))
				{
					for(i64 i = 0; i < a.x(); i++)
						if(b.at(i))
							for(i64 j = 0; j < a.y(); j++)
								c.at(j) += a.at(i,j);
				}
				else
				{
					throw std::runtime_error("Unhandled tensor bool dot case: 0!\n");
				}
			}
			else
			{
				throw std::runtime_error("Unhandled tensor bool dot case: 1!\n");
			}
		}
		else
		{
			throw std::runtime_error("Unhandled tensor bool dot case: 2!\n");
		}
	};

	/*inline void conv_3d(
		const conv_kernel_t &kernel,
		const tensor<float> &a,
		const tensor<float> &b,
		tensor<float> &c
	)
	{
		i64 ix = a.x();
		i64 iy = a.y();
		i64 iz = a.z();
		//i64 iw = a.w();

		i64 fx = b.x();
		i64 fy = b.y();
		i64 fz = b.z();
		//i64 fw = b.w();

		i64 ox = (ix - fx + 2 * kernel.padding) / kernel.stride + 1;
		i64 oy = (iy - fy * 2 * kernel.padding) / kernel.stride + 1;

		for(i64 p = 0; p < iz; p++)
		for(i64 j = 0; j < fz; j++)
		//for(i64 i = 0; i < iw; i++)
		for(i64 l = 0; l < ox; l++)
		for(i64 n = 0; n < fx; n++)
		for(i64 k = 0; k < oy; k++)
		for(i64 m = 0; m < fy; m++)
		{
			i64 hi = l * kernel.stride + n - kernel.padding;
			i64 wi = k * kernel.stride + m - kernel.padding;
			bool pad = ((hi < 0)||(hi >= ix)) || ((wi < 0)||(wi >= iy));
			float v = pad ? 0.0f : a.at(p,hi,wi);
			c.at(p,j,l,k) += v * b.at(j,n,m);
		}
	};*/

	/*inline void conv_4d(
		const conv_kernel_t &kernel,
		const tensor<float> &a,
		const tensor<float> &b, 
		tensor<float> &c
	)
	{
		i64 ix = a.x();
		i64 iy = a.y();
		i64 iz = a.z();
		i64 iw = a.w();

		i64 fx = b.x();
		i64 fy = b.y();
		i64 fz = b.z();
		i64 fw = b.w();

		i64 ox = (ix - fx + 2 * kernel.padding) / kernel.stride + 1;
		i64 oy = (iy - fy * 2 * kernel.padding) / kernel.stride + 1;

		for(i64 p = 0; p < iz; p++)
		for(i64 j = 0; j < fz; j++)
		for(i64 i = 0; i < iw; i++)
		for(i64 l = 0; l < ox; l++)
		for(i64 n = 0; n < fx; n++)
		for(i64 k = 0; k < oy; k++)
		for(i64 m = 0; m < fy; m++)
		{
			i64 hi = l * kernel.stride + n - kernel.padding;
			i64 wi = k * kernel.stride + m - kernel.padding;
			bool pad = ((hi < 0)||(hi >= ix)) || ((wi < 0)||(wi >= iy));
			float v = pad ? 0.0f : a.at(p,i,hi,wi);
			c.at(p,j,l,k) += v * b.at(j,i,n,m);
		}
	};*/

	/*inline void convolution(const tensor<float> &src, const conv_kernel_t &kernel, tensor<float> &dst)
	{
		i64 ix = src.x();
		i64 iy = src.y();
		i64 iz = src.z();
		i64 iw = src.w();
		
		i64 kx = std::min<i64>(kernel.size.x,ix);
		i64 ky = std::min<i64>(kernel.size,iy);
		i64 kz = std::min<i64>(kernel.size,iz);
		i64 kw = std::min<i64>(kernel.size,iw);

		auto full_conv = [&](
			i64 ix_, i64 iy_, i64 iz_, i64 iw_,
			i64 kx_, i64 ky_, i64 kz_, i64 kw_
		)
		{
			float inv_modifier = 1.0f / (kx_ * ky_ * kz_ * kw_);
			for(i64 qx = 0; qx + kx <= ix; qx++)
			{
				for(i64 qy = 0; qy + ky <= iy; qy++)
				{
					for(i64 qz = 0; qz + kz <= iz; qz++)
					{
						for(i64 qw = 0; qw + kw <= iw; qw++)
						{
							float sum = 0.0f;
							for(i64 ki = 0; ki < kx_; ki++)
								for(i64 kj = 0; kj < ky_; kj++)
									for(i64 kk = 0; kk < kz_; kk++)
										for(i64 kl = 0; kl < kw_; kl++)
											sum += src.at(qx+ki,qy+kj,qz+kk,qw+kl);
							dst.at(qx,qy,qz,qw) = sum * inv_modifier;
						}
					}
				}
			}
		};

		if(ix > 1 && iy > 1 && iz == 1 && iw == 1)
			full_conv(ix,iy,1,1,kx,ky,1,1);
		else if(ix > 1 && iy > 1 && iz > 1 && iw == 1)
			full_conv(ix,iy,iz,1,kx,ky,kz,1);
		else if(ix > 1 && iy > 1 && iz > 1 && iw > 1)
			full_conv(ix,iy,iz,iw,kx,ky,kz,kw);
	};*/

	/*inline void convolution(const tensor<float> &a, const tensor<float> &b, tensor<float> &dst)
	{
		i64 ix = a.x();
		i64 iy = a.y();
		i64 iz = a.z();
		i64 iw = a.w();
		
		i64 kx = std::min<i64>(1,ix);
		i64 ky = std::min<i64>(1,iy);
		i64 kz = std::min<i64>(1,iz);
		i64 kw = std::min<i64>(1,iw);

		auto full_conv = [&](
			i64 ix_, i64 iy_, i64 iz_, i64 iw_,
			i64 kx_, i64 ky_, i64 kz_, i64 kw_
		)
		{
			float inv_modifier = 1.0f / (kx_ * ky_ * kz_ * kw_);
			for(i64 qx = 0; qx + kx <= ix; qx++)
			{
				for(i64 qy = 0; qy + ky <= iy; qy++)
				{
					for(i64 qz = 0; qz + kz <= iz; qz++)
					{
						for(i64 qw = 0; qw + kw <= iw; qw++)
						{
							float sum = 0.0f;
							for(i64 ki = 0; ki < kx_; ki++)
								for(i64 kj = 0; kj < ky_; kj++)
									for(i64 kk = 0; kk < kz_; kk++)
										for(i64 kl = 0; kl < kw_; kl++)
											sum += a.at(qx+ki,qy+kj,qz+kk,qw+kl);
							dst.at(qx,qy,qz,qw) = sum * inv_modifier;
						}
					}
				}
			}
		};

		if(ix > 1 && iy > 1 && iz == 1 && iw == 1)
			full_conv(ix,iy,1,1,kx,ky,1,1);
		else if(ix > 1 && iy > 1 && iz > 1 && iw == 1)
			full_conv(ix,iy,iz,1,kx,ky,kz,1);
		else if(ix > 1 && iy > 1 && iz > 1 && iw > 1)
			full_conv(ix,iy,iz,iw,kx,ky,kz,kw);
	};*/
};

template <typename T>
using tensor = math::tensor<T>;