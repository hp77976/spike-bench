#pragma once
#include <cstdio>
#include <stdint.h>
#include <stdexcept>

template <typename S, typename T>
struct exp_t
{
	inline const S& self(void) const
	{
		return *static_cast<const S*>(this);
	};
};

template <typename LHS, typename RHS, typename T>
struct bin_exp_t : public exp_t<bin_exp_t<LHS,RHS,T>,T>
{
	const LHS &lhs;
	const RHS &rhs;

	bin_exp_t(const LHS &l, const RHS &r) : lhs(l), rhs(r) {};
};

template <typename T>
struct op_add_t
{
	inline static void run(T* c, T a, T b) {*c = a + b;};
	inline static T run(T a, T b) {return a + b;};
};

template <typename T>
struct op_sub_t
{
	inline static void run(T* c, T a, T b) {*c = a - b;};
	inline static T run(T a, T b) {return a - b;};
};

template <typename T>
struct op_mul_t
{
	inline static void run(T* c, T a, T b) {*c = a * b;};
	inline static T run(T a, T b) {return a * b;};
};

template <typename T>
struct op_div_t
{
	inline static void run(T* c, T a, T b) {*c = a / b;};
	inline static T run(T a, T b) {return a / b;};
};

template <typename T>
struct op_and_t
{
	inline static void run(T* c, T a, T b) {*c = a & b;};
	inline static T run(T a, T b) {return a & b;};
};

template <typename T>
struct op_or_t
{
	inline static void run(T* c, T a, T b) {*c = a | b;};
	inline static T run(T a, T b) {return a | b;};
};

template <typename T>
struct op_xor_t
{
	inline static void run(T* c, T a, T b) {*c = a ^ b;};
	inline static T run(T a, T b) {return a ^ b;};
};

template <typename T>
struct scalar_t : public exp_t<scalar_t<T>,T>
{
	T value;
	scalar_t(T s) : value(s) {};

	inline T eval(uint32_t i) const {return value;};
};

template <typename OP, typename L, typename R, typename T>
struct arr_exp_t : public exp_t<arr_exp_t<OP,L,R,T>,T>
{
	const L &lhs;
	const R &rhs;

	arr_exp_t(const L &l, const R &r) : lhs(l), rhs(r) {};

	inline T eval(uint32_t i) const
	{
		return OP::run(lhs.eval(i),rhs.eval(i));
	};

	template <typename A>
	inline A* resolve(A* const dst) const
	{
		for(uint32_t i = 0; i < dst->size(); i++)
			OP::run(&dst->at(i),lhs.eval(i),rhs.eval(i));
		return dst;
	};
};

template <typename OP, typename L, typename R, typename T>
struct asl_exp_t : public exp_t<asl_exp_t<OP,L,R,T>,T>
{
	const L lhs;
	const R &rhs;

	asl_exp_t(const L &l, const R &r) : lhs(l), rhs(r) {};

	inline T eval(uint32_t i) const
	{
		return OP::run(lhs,rhs.at(i));
	};

	template <typename A>
	inline A* resolve(A* const dst) const
	{
		for(uint32_t i = 0; i < dst->size(); i++)
			OP::run(&dst->at(i),lhs.eval(i),rhs.eval(i));
		return dst;
	};
};

template <typename OP, typename L, typename R, typename T>
struct asr_exp_t : public exp_t<asr_exp_t<OP,L,R,T>,T>
{
	const L &lhs;
	const R rhs;
	
	asr_exp_t(const L &l, const R &r) : lhs(l), rhs(r) {};

	inline T eval(uint32_t i) const
	{
		return OP::run(lhs.eval(i),rhs);
	};

	template <typename A>
	inline A* resolve(A* const dst) const
	{
		for(uint32_t i = 0; i < dst->size(); i++)
			OP::run(&dst->at(i),lhs.eval(i),rhs.eval(i));
		return dst;
	};
};

template <typename OP, typename L, typename R, typename T>
inline arr_exp_t<OP,L,R,T> make_exp(const exp_t<L,T> &lhs, const exp_t<R,T> &rhs)
{
	return arr_exp_t<OP,L,R,T>(lhs.self(),rhs.self());
};

template <typename L, typename R, typename T>
inline arr_exp_t<op_add_t<T>,L,R,T> operator+(const exp_t<L,T> &lhs, const exp_t<R,T> &rhs)
{
	return make_exp<op_add_t<T>,L,R,T>(lhs,rhs);
};

template <typename L, typename R, typename T>
inline arr_exp_t<op_sub_t<T>,L,R,T> operator-(const exp_t<L,T> &lhs, const exp_t<R,T> &rhs)
{
	return make_exp<op_sub_t<T>,L,R,T>(lhs,rhs);
};

template <typename L, typename R, typename T>
inline arr_exp_t<op_mul_t<T>,L,R,T> operator*(const exp_t<L,T> &lhs, const exp_t<R,T> &rhs)
{
	return make_exp<op_mul_t<T>,L,R,T>(lhs,rhs);
};

template <typename L, typename R, typename T>
inline arr_exp_t<op_div_t<T>,L,R,T> operator/(const exp_t<L,T> &lhs, const exp_t<R,T> &rhs)
{
	return make_exp<op_div_t<T>,L,R,T>(lhs,rhs);
};

template <typename OP, typename L, typename R, typename T>
inline asl_exp_t<OP,L,R,T> make_exp_l(const T &lhs, const exp_t<R,T> &rhs)
{
	return asl_exp_t<OP,L,R,T>(lhs.self(),rhs);
};

template <typename T, typename R>
inline asl_exp_t<op_add_t<T>,scalar_t<T>,R,T> operator+(const T &lhs, const exp_t<R,T> &rhs)
{
	return make_exp_l<op_add_t<T>,scalar_t<T>,R,T>(scalar_t<T>(lhs),rhs);
};

template <typename T, typename R>
inline asl_exp_t<op_sub_t<T>,scalar_t<T>,R,T> operator-(const T &lhs, const exp_t<R,T> &rhs)
{
	return make_exp_l<op_sub_t<T>,scalar_t<T>,R,T>(scalar_t<T>(lhs),rhs);
};

template <typename T, typename R>
inline asl_exp_t<op_mul_t<T>,scalar_t<T>,R,T> operator*(const T &lhs, const exp_t<R,T> &rhs)
{
	return make_exp_l<op_mul_t<T>,scalar_t<T>,R,T>(scalar_t<T>(lhs),rhs);
};

template <typename T, typename R>
inline asl_exp_t<op_div_t<T>,scalar_t<T>,R,T> operator/(const T &lhs, const exp_t<R,T> &rhs)
{
	return make_exp_l<op_div_t<T>,scalar_t<T>,R,T>(scalar_t<T>(lhs),rhs);
};

template <typename OP, typename L, typename R, typename T>
inline asr_exp_t<OP,L,R,T> make_exp_r(const exp_t<L,T> &lhs, const R &rhs)
{
	return asr_exp_t<OP,L,R,T>(lhs.self(),rhs);
};

template <typename L, typename T>
inline asr_exp_t<op_add_t<T>,L,scalar_t<T>,T> operator+(const exp_t<L,T> &lhs, const T &rhs)
{
	return make_exp_r<op_add_t<T>,L,scalar_t<T>,T>(lhs,scalar_t<T>(rhs));
};

template <typename L, typename T>
inline asr_exp_t<op_sub_t<T>,L,scalar_t<T>,T> operator-(const exp_t<L,T> &lhs, const T &rhs)
{
	return make_exp_r<op_sub_t<T>,L,scalar_t<T>,T>(lhs,scalar_t<T>(rhs));
};

template <typename L, typename T>
inline asr_exp_t<op_mul_t<T>,L,scalar_t<T>,T> operator*(const exp_t<L,T> &lhs, const T &rhs)
{
	return make_exp_r<op_mul_t<T>,L,scalar_t<T>,T>(lhs,scalar_t<T>(rhs));
};

template <typename L, typename T>
inline asr_exp_t<op_div_t<T>,L,scalar_t<T>,T> operator/(const exp_t<L,T> &lhs, const T &rhs)
{
	return make_exp_r<op_div_t<T>,L,scalar_t<T>,T>(lhs,scalar_t<T>(rhs));
};

namespace math
{
	template <typename T>
	struct alignas(128) array : public exp_t<array<T>,T>
	{
		T* m_ptr = nullptr;
		uint32_t m_size = 0;
		uint32_t m_x = 0;
		uint32_t m_y = 0;

		void in_bounds(uint32_t i, uint32_t j) const
		{
#ifdef DEBUG_ARRAYS
			if((j < 1 && i >= m_x) || (j > 0 && (i >= m_x || j >= m_y)))
			{
				printf("[array]\n");
				printf("\tsize: %u %u\n",m_x,m_y);
				printf("\ti: %u j: %u\n",i,j);
				throw std::runtime_error("Out of bounds!\n");
			}
#endif
		};

		void in_bounds(uint32_t i) const
		{
#ifdef DEBUG_ARRAYS
			if(i >= size())
			{
				printf("[array]\n");
				printf("\tsize: %u %u\n",m_x,m_y);
				printf("\ti: %u j: 1\n",i);
				throw std::runtime_error("Out of bounds!\n");
			}
#endif
		};

		void resize(uint32_t i)
		{
			if(m_ptr == nullptr)
				m_ptr = new T[i];
			m_size = i;
		};

		template <typename E>
		inline array& operator=(const exp_t<E,T> &src_)
		{
			const E &src = src_.self();
			src.resolve(this);
			return *this;
		};

		inline T at(uint32_t i) const {return m_ptr[i];};
		
		inline T& at(uint32_t i) {return m_ptr[i];};
		
		inline T eval(uint32_t i) const {return m_ptr[i];};
		
		inline T& eval(uint32_t i) {return m_ptr[i];};

		inline uint32_t size() const {return m_size;};
	};
};

void simple_syntax()
{
	math::array<float> a; a.resize(5);
	math::array<float> b; b.resize(5);
	math::array<float> c; c.resize(5);
	for(uint32_t i = 0; i < 5; i++)
	{
		a.at(i) = 1.0f;
		b.at(i) = 2.0f;
		c.at(i) = 0.0f;
	}

	float f = 2.0f;

	c = (a + b * b) * f;

	printf("c: %f\n",c.at(0));
};