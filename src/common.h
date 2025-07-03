#pragma once
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <thread>
#include <mutex>

#ifdef __CUDACC__
#	define CPU_GPU __host__ __device__
#else
#	define CPU_GPU
#endif

template <typename T>
using sp = std::shared_ptr<T>;

template <typename K, typename V>
using umap = std::unordered_map<K,V>;

template <typename T>
using uset = std::unordered_set<T>;

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

template <typename T>
CPU_GPU inline u32 get_padded_bytes(u32 count)
{
	i64 min_bytes = sizeof(T) * count;
	i64 blocks_needed = (min_bytes + 64 - 1) / 64;
	i64 total_bytes = blocks_needed * 64;
	return total_bytes;
};

void __cuda_alloc_uni(void** ptr, u32 bytes);

void __cuda_free_uni(void** ptr);

template <typename T>
inline T* uni_alloc(u32 c)
{
	T* ptr = nullptr;
	__cuda_alloc_uni((void**)&ptr,get_padded_bytes<T>(c));
	return ptr;
};

template <typename T>
inline T* uni_alloc(u32 c, T x)
{
	T* ptr = nullptr;
	__cuda_alloc_uni((void**)&ptr,get_padded_bytes<T>(c));
	for(u32 i = 0; i < c; i++)
		ptr[i] = x;
	return ptr;
};

template <typename T>
inline void uni_free(T** ptr)
{
	if(*ptr != nullptr)
		__cuda_free_uni((void**)ptr);
	*ptr = nullptr;
};

template <typename T>
inline std::string str(T x)
{
	return std::to_string(x);
};