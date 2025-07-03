#pragma once
#include "exp.h"
#include <cmath>
#include <stdint.h>
#include <stdexcept>
#include <string.h>
#include <memory>
//#include "../util/cuut.h"
#include <algorithm>
#include "rng.h"
#include "simd/include.h"

#ifdef USE_CUDA
void __cuda_alloc_gpu(void** ptr, uint32_t bytes);

void __cuda_free_gpu(void** ptr);

void __cuda_alloc_uni(void** ptr, uint32_t bytes);

void __cuda_free_uni(void** ptr);
#endif

namespace math
{
	enum array_location_e
	{
		ARRAY_NONE = -1,
		ARRAY_CPU = 0b00,
		ARRAY_GPU = 0b01,
		ARRAY_UNI = 0b10,
		ON_GPU = 0b11
	};

	template <typename T>
	struct alignas(64) array : public expr::exp_t<array<T>>
	{
		T* m_ptr = nullptr;
		private:
		dim2 m_shape = {0,0};
		public:
		i64 m_size = 0;
		i64 m_bytes = 0;
		bool m_shared = false;

		CPU_GPU inline i64 __get_padded_bytes(i64 count) const
		{
			i64 min_bytes = sizeof(T) * count;
			i64 blocks_needed = (min_bytes + 64 - 1) / 64;
			i64 total_bytes = blocks_needed * 64;
			return std::max<i64>(total_bytes,64);
		};

		CPU_GPU inline void __in_bounds(i64 i, i64 j) const
		{
#ifdef DEBUG_ARRAYS
			if((j < 1 && i >= m_shape.x) || (j > 0 && (i >= m_shape.x || j >= m_shape.y)))
			{
				printf("[array]\n");
				printf("\tsize: %li %li\n",m_shape.x,m_shape.y);
				printf("\ti: %li j: %li\n",i,j);
				throw std::runtime_error("Out of bounds!\n");
			}
#endif
		};

		CPU_GPU inline void __in_bounds(i64 i) const
		{
#ifdef DEBUG_ARRAYS
			if(i >= size())
			{
				printf("[array]\n");
				printf("\tsize: %li %li\n",m_shape.x,m_shape.y);
				printf("\ti: %li j: 1\n",i);
				throw std::runtime_error("Out of bounds!\n");
			}
#endif
		};
	
		CPU_GPU inline void __alloc(dim2 shape, bool shared)
		{
#ifdef DEBUG_ARRAYS
			if(shape.x == 0 || shape.y == 0)
				printf("Warning: Attempted to allocate zero sized array!\n");
			if(shape.x < 0 || shape.y < 0)
				throw std::runtime_error("Negative array allocation!\n");
#endif
			/*bool need_free = false;

			if((m_shared != shared) || (m_size < shape.size() && m_size > 0))
				need_free = true;

			if(need_free)
				__free();

			if(m_size == 0 || m_ptr == nullptr)
			{
				i64 c = std::max<i64>(1,__get_padded_bytes(shape.size()) / sizeof(T));
				cuut::ualloc(&m_ptr,c,shared);
			}

			//this is needed to make the masked dot product work
			memset(m_ptr,0,__get_padded_bytes(shape.size()));
			
			m_shared = shared;
			m_shape = shape;*/

			i64 new_size = shape.size();
			bool need_realloc = new_size > m_size;
			m_size = new_size;

			if(need_realloc)
			{
				m_shape = shape;
				i64 min_bytes = sizeof(T) * new_size;
				i64 blocks_needed = (min_bytes + 64 - 1) / 64;
				m_bytes = (blocks_needed * 64);

				if(m_ptr != nullptr)
					std::free(m_ptr);
				m_ptr = (T*)std::aligned_alloc(64,std::max<i64>(1,m_bytes));
				
				memset(m_ptr,0,m_bytes);
			}
		};

		CPU_GPU inline void __free()
		{
			if(m_ptr == nullptr || m_size == 0)
				return;

			if(m_ptr != nullptr)
				std::free(m_ptr);
			//cuut::ufree(m_ptr,m_shared);

			m_ptr = nullptr;
			m_shape = {0,0};
			m_size = 0;
		};

		CPU_GPU array(void) {};

		CPU_GPU inline operator T*() const {return m_ptr;};
		
		CPU_GPU inline T operator[](i64 i) const {return m_ptr[i];};

		CPU_GPU inline T& operator[](i64 i) {return m_ptr[i];};

		CPU_GPU inline array& resize(i64 i, i64 j = 1)
		{
			__alloc({i,j},false);
			return *this;
		};

		CPU_GPU inline array& resize(dim2 shape)
		{
			__alloc(shape,false);
			return *this;
		};

		CPU_GPU inline array& free()
		{
			if(m_ptr != nullptr)
				__free();
			return *this;
		};

		CPU_GPU inline array& copy_from(const array &arr)
		{
			if(arr.size() > m_size)
				__alloc(arr.shape(),m_shared);
			memcpy(m_ptr,arr.m_ptr,m_bytes);
			return *this;
		};

		CPU_GPU inline array& zero()
		{
			if(m_ptr != nullptr)
				for(i64 i = 0; i < m_size; i++)
					m_ptr[i] = 0.0f;
			return *this;
		};

		CPU_GPU inline array& fill(const T &x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = x;
			return *this;
		};

		template <typename E>
		inline array& operator+=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] += expr[i];
			return *this;
		};

		template <typename E>
		inline array& operator-=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] -= expr[i];
			return *this;
		};

		template <typename E>
		inline array& operator*=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] *= expr[i];
			return *this;
		};

		template <typename E>
		inline array& operator/=(const expr::exp_t<E> &expr)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] /= expr[i];
			return *this;
		};

		template <typename E>
		inline array& operator=(const expr::exp_t<E> &src_)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = src_[i];
			return *this;
		};

		CPU_GPU inline T at(i64 i) const
		{
			__in_bounds(i); return m_ptr[i];
		};
		
		CPU_GPU inline T& at(i64 i)
		{
			__in_bounds(i); return m_ptr[i];
		};

		CPU_GPU inline T at(i64 i, i64 j) const
		{
			__in_bounds(i,j); return m_ptr[m_shape.at(i,j)];
		};
		
		CPU_GPU inline T& at(i64 i, i64 j)
		{
			__in_bounds(i,j); return m_ptr[m_shape.at(i,j)];
		};

		CPU_GPU inline i64 size() const {return m_size;};

		CPU_GPU inline dim2 shape() const {return m_shape;};

		CPU_GPU inline i64 x() const {return m_shape.x;};
		
		CPU_GPU inline i64 y() const {return m_shape.y;};

		CPU_GPU inline T* data() const {return m_ptr;};

		inline i64 bytes() const {return m_bytes;};

		template <typename T2>
		inline T2* as() const {return (T2*)m_ptr;};

		CPU_GPU inline array& randomize(rng_t &rng, T min = 0.0f, T max = 1.0f)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = (max - min) * rng.u() + min;
			return *this;
		};

		CPU_GPU inline array& clamp(T min, T max)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] = std::clamp<T>(m_ptr[i],min,max);
			return *this;
		};

		inline array& operator+=(const array &arr)
		{
#ifdef DEBUG_ARRAYS
			if(arr.size() != m_size && arr.size() != 1 && m_size != 1)
				throw std::runtime_error("Array += size mismatch!\n");
#endif
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] += arr.m_ptr[i];
			return *this;
		};

		inline array& operator-=(const array &arr)
		{
#ifdef DEBUG_ARRAYS
			if(arr.size() != m_size && arr.size() != 1 && m_size != 1)
				throw std::runtime_error("Array -= size mismatch!\n");
#endif
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] -= arr.m_ptr[i];
			return *this;
		};

		inline array& operator*=(const array &arr)
		{
#ifdef DEBUG_ARRAYS
			if(arr.size() != m_size && arr.size() != 1 && m_size != 1)
				throw std::runtime_error("Array *= size mismatch!\n");
#endif
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] *= arr.m_ptr[i];
			return *this;
		};

		inline array& operator/=(const array &arr)
		{
#ifdef DEBUG_ARRAYS
			if(arr.size() != m_size && arr.size() != 1 && m_size != 1)
				throw std::runtime_error("Array /= size mismatch!\n");
#endif
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] /= arr.m_ptr[i];
			return *this;
		};

		inline array& operator+=(const T &x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] += x;
			return *this;
		};

		inline array& operator-=(const T &x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] -= x;
			return *this;
		};

		inline array& operator*=(const T &x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] *= x;
			return *this;
		};

		inline array& operator/=(const T &x)
		{
			for(i64 i = 0; i < m_size; i++)
				m_ptr[i] /= x;
			return *this;
		};

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

	inline void dot_mask8(
		const array<float> &src,
		const array<float> &dst,
		const array<uint8_t> &m,
		uint32_t m8,
		uint32_t offset = 0,
		uint32_t limit = 0
	)
	{
		uint32_t o = 0; //offset
		uint32_t c = dst.size();
		uint32_t cx8 = c * 8;
		uint8_t* mm = m.data();
		float* s = src.data();
		float* d = dst.data();

		//if(limit == 0)
			limit = m8;

		for(uint32_t mi = offset; mi < limit; mi++)
		{
			uint8_t mask = mm[mi];
			if(mask)
			{
				for(uint32_t i = 0; i < 8; i++, o += c)
				{
					float* so = &s[o];
					if(mask >> i & 0x1)
						for(uint32_t n = 0; n < c; n++)
							d[n] += so[n];
				}
			}
			else
			{
				o += cx8;
			}
		}
	};

	inline void bit_mask_trace(float* const __restrict v, u8* const __restrict m, float decay, i64 c)
	{
		simd::f32_8* v8 = (simd::f32_8*)v;
		simd::f32_8 d8 = decay;
		for(i64 i = 0; i < c; i++)
		{
			v8[i] *= d8;
			v8[i] = simd::mask_add(v8[i],v8[i],1.0f,m[i]);
		}
	};

	inline void bit_mask_dot(
		i64 ps, i64 ns, i64 ms,
		float* const wt, float* const cu, u8* const m
	)
	{
		i64 ws = ps * ns;

		i64 o = 0; //offset
		i64 cs8 = ns * 8;

		/*
			m: [0,1,2,3,-,-,-,-]
			w: [00,01,02,03],[10,11,12,13],[20,21,22,23],[30,31,32,33]
			c: [00,01,02,03]

			mi = 0
			o += 4
		*/
		i64 p = 0;

		for(i64 mi = 0; mi < ms; mi++)
		{
			u8 mask = m[mi];
			if(mask)
			{
				for(i64 i = 0; i < 8; i++, p++)
				{
					//float* wto = &wt[o];
					if((mask >> i) & 0x1)
						for(i64 n = 0; n < ns; n++)
							cu[n] += wt[p*ns+n];
							//cu[n] += wto[n];
				}
			}
			else
			{
				p += 8;
				//o += cs8;
			}
		}
	};
};