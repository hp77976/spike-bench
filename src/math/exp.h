#pragma once
#include "../common.h"
#include <cmath>

namespace expr
{
	template <typename E>
	struct exp_t
	{
		const E& self() const {return static_cast<const E&>(*this);};

		float operator[](i64 i) const {return self()[i];};

		i64 size() const {return self().size();};
	};

	template <typename L, typename R>
	struct vv_add_exp_t : public exp_t<vv_add_exp_t<L,R>>
	{
		const L& l;
		const R& r;
		vv_add_exp_t(const L &l_, const R &r_) : l(l_), r(r_) {};
		float operator[](i64 i) const {return l[i] + r[i];};
		i64 size() const {return l.size();};
	};

	template <typename L, typename R>
	struct vv_sub_exp_t : public exp_t<vv_sub_exp_t<L,R>>
	{
		const L& l;
		const R& r;
		vv_sub_exp_t(const L &l_, const R &r_) : l(l_), r(r_) {};
		float operator[](i64 i) const {return l[i] - r[i];};
		i64 size() const {return l.size();};
	};

	template <typename L, typename R>
	struct vv_mul_exp_t : public exp_t<vv_mul_exp_t<L,R>>
	{
		const L& l;
		const R& r;
		vv_mul_exp_t(const L &l_, const R &r_) : l(l_), r(r_) {};
		float operator[](i64 i) const {return l[i] * r[i];};
		i64 size() const {return l.size();};
	};

	template <typename L, typename R>
	struct vv_div_exp_t : public exp_t<vv_div_exp_t<L,R>>
	{
		const L& l;
		const R& r;
		vv_div_exp_t(const L &l_, const R &r_) : l(l_), r(r_) {};
		float operator[](i64 i) const {return l[i] / r[i];};
		i64 size() const {return l.size();};
	};

	template <typename E>
	struct vs_add_exp_t : public exp_t<vs_add_exp_t<E>>
	{
		const E& e;
		float f;
		vs_add_exp_t(const E &e_, float f_) : e(e_), f(f_) {};
		float operator[](i64 i) const {return e[i] + f;};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct sv_sub_exp_t : public exp_t<sv_sub_exp_t<E>>
	{
		float f;
		const E& e;
		sv_sub_exp_t(float f_, const E &e_) : f(f_), e(e_) {};
		float operator[](i64 i) const {return f - e[i];};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct vs_sub_exp_t : public exp_t<vs_sub_exp_t<E>>
	{
		const E& e;
		float f;
		vs_sub_exp_t(const E &e_, float f_) : e(e_), f(f_) {};
		float operator[](i64 i) const {return e[i] - f;};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct vs_mul_exp_t : public exp_t<vs_mul_exp_t<E>>
	{
		const E& e;
		float f;
		vs_mul_exp_t(const E &e_, float f_) : e(e_), f(f_) {};
		float operator[](i64 i) const {return e[i] * f;};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct sv_div_exp_t : public exp_t<sv_div_exp_t<E>>
	{
		float f;
		const E& e;
		sv_div_exp_t(float f_, const E &e_) : f(f_), e(e_) {};
		float operator[](i64 i) const {return f / e[i];};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct vs_div_exp_t : public exp_t<vs_div_exp_t<E>>
	{
		const E& e;
		float f;
		vs_div_exp_t(const E &e_, float f_) : e(e_), f(f_) {};
		float operator[](i64 i) const {return e[i] / f;};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct v_exp_exp_t : public exp_t<v_exp_exp_t<E>>
	{
		const E& e;
		v_exp_exp_t(const E &e_) : e(e_) {};
		float operator[](i64 i) const {return std::exp(e[i]);};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct v_abs_exp_t : public exp_t<v_abs_exp_t<E>>
	{
		const E& e;
		v_abs_exp_t(const E &e_) : e(e_) {};
		float operator[](i64 i) const {return std::abs(e[i]);};
		i64 size() const {return e.size();};
	};

	template <typename L, typename R>
	struct vv_pow_exp_t : public exp_t<vv_pow_exp_t<L,R>>
	{
		const L& l;
		const R& r;
		vv_pow_exp_t(const L &l_, const R &r_) : l(l_), r(r_) {};
		float operator[](i64 i) const {return std::pow(l[i],r[i]);};
		i64 size() const {return l.size();};
	};

	template <typename E>
	struct vs_pow_exp_t : public exp_t<vs_pow_exp_t<E>>
	{
		const E& e;
		float f;
		vs_pow_exp_t(const E &e_, float f_) : e(e_), f(f_) {};
		float operator[](i64 i) const {return std::pow(e[i],f);};
		i64 size() const {return e.size();};
	};

	template <typename E>
	struct sv_pow_exp_t : public exp_t<sv_pow_exp_t<E>>
	{
		float f;
		const E& e;
		sv_pow_exp_t(float f_, const E &e_) : f(f_), e(e_) {};
		float operator[](i64 i) const {return std::exp(f,e[i]);};
		i64 size() const {return e.size();};
	};

	template <typename L, typename R>
	inline vv_add_exp_t<L,R> operator+(const exp_t<L> &lhs, const exp_t<R> &rhs)
	{
		return vv_add_exp_t<L,R>(lhs.self(),rhs.self());
	};

	template <typename E>
	inline vs_add_exp_t<E> operator+(const exp_t<E> &v, float f)
	{
		return vs_add_exp_t<E>(v.self(),f);
	};

	template <typename E>
	inline vs_add_exp_t<E> operator+(float f, const exp_t<E> &v)
	{
		return vs_add_exp_t<E>(v.self(),f);
	};

	template <typename L, typename R>
	inline vv_sub_exp_t<L,R> operator-(const exp_t<L> &lhs, const exp_t<R> &rhs)
	{
		return vv_sub_exp_t<L,R>(lhs.self(),rhs.self());
	};

	template <typename E>
	inline sv_sub_exp_t<E> operator-(float f, const exp_t<E> &v)
	{
		return sv_sub_exp_t<E>(f,v.self());
	};

	template <typename E>
	inline vs_sub_exp_t<E> operator-(const exp_t<E> &v, float f)
	{
		return vs_sub_exp_t<E>(v.self(),f);
	};

	template <typename L, typename R>
	inline vv_mul_exp_t<L,R> operator*(const exp_t<L> &lhs, const exp_t<R> &rhs)
	{
		return vv_mul_exp_t<L,R>(lhs.self(),rhs.self());
	};

	template <typename E>
	inline vs_mul_exp_t<E> operator*(const exp_t<E> &v, float f)
	{
		return vs_mul_exp_t<E>(v.self(),f);
	};

	template <typename E>
	inline vs_mul_exp_t<E> operator*(float f, const exp_t<E> &v)
	{
		return vs_mul_exp_t<E>(v.self(),f);
	};

	template <typename L, typename R>
	inline vv_div_exp_t<L,R> operator/(const exp_t<L> &lhs, const exp_t<R> &rhs)
	{
		return vv_div_exp_t<L,R>(lhs.self(),rhs.self());
	};

	template <typename E>
	inline sv_div_exp_t<E> operator/(float f, const exp_t<E> &v)
	{
		return sv_div_exp_t<E>(f,v.self());
	};

	template <typename E>
	inline vs_div_exp_t<E> operator/(const exp_t<E> &v, float f)
	{
		return vs_div_exp_t<E>(v.self(),f);
	};

	template <typename E>
	inline v_exp_exp_t<E> exp(const exp_t<E> &v)
	{
		return v_exp_exp_t<E>(v.self());
	};

	template <typename E>
	inline v_abs_exp_t<E> abs(const exp_t<E> &v)
	{
		return v_abs_exp_t<E>(v.self());
	};

	template <typename L, typename R>
	inline vv_pow_exp_t<L,R> pow(const exp_t<L> &lhs, const exp_t<R> &rhs)
	{
		return vv_pow_exp_t<L,R>(lhs.self(),rhs.self());
	};

	template <typename E>
	inline sv_pow_exp_t<E> pow(float f, const exp_t<E> &v)
	{
		return sv_pow_exp_t<E>(f,v.self());
	};

	template <typename E>
	inline vs_pow_exp_t<E> pow(const exp_t<E> &v, float f)
	{
		return vs_pow_exp_t<E>(v.self(),f);
	};
};

namespace math
{
	struct dim2
	{
		i64 x = 1;
		i64 y = 1;

		dim2(i64 x_, i64 y_) : x(x_), y(y_) {};

		dim2& operator=(i64 i)
		{
			x = i;
			y = 1;
			return *this;
		};

		i64 operator[](i64 i) const
		{
			switch(i)
			{
				default:
				case(0): return x;
				case(1): return y;
			}
		};

		i64& operator[](i64 i)
		{
			switch(i)
			{
				default:
				case(0): return x;
				case(1): return y;
			}
		};

		i64 size() const
		{
			return x * y;
		};

		i64 at(i64 i) const
		{
			return i;
		};

		i64 at(i64 i, i64 j) const
		{
			return (i * y) + j;
		};

		i64 dims() const
		{
			i64 d = 0;
			d += x > 1 ? 1 : 0;
			d += y > 1 ? 1 : 0;
			return d;
		};
	};

	struct dim4
	{
		i64 x = 1;
		i64 y = 1;
		i64 z = 1;
		i64 w = 1;

		dim4() : x(1), y(1), z(1), w(1) {};

		dim4(i64 x_, i64 y_, i64 z_, i64 w_) : x(x_), y(y_), z(z_), w(w_) {};

		i64 operator[](i64 i) const
		{
			switch(i)
			{
				default:
				case(0): return x;
				case(1): return y;
				case(2): return z;
				case(3): return w;
			}
		};

		i64& operator[](i64 i)
		{
			switch(i)
			{
				default:
				case(0): return x;
				case(1): return y;
				case(2): return z;
				case(3): return w;
			}
		};

		i64 size() const
		{
			return x * y * z * w;
		};

		i64 at(i64 i) const
		{
			return i;
		};

		i64 at(i64 i, i64 j) const
		{
			return (i * y) + j;
		};

		i64 at(i64 i, i64 j, i64 k) const
		{
			return (i * y + j) * z + k;
		};

		i64 at(i64 i, i64 j, i64 k, i64 l) const
		{
			//i * x * y * z + j * y * z + k * z + l
			return ((i * y + j) * z + k) * w + l;
		};

		i64 dims() const
		{
			i64 d = 0;
			d += x > 1 ? 1 : 0;
			d += y > 1 ? 1 : 0;
			d += z > 1 ? 1 : 0;
			d += w > 1 ? 1 : 0;
			return d;
		};
	};
};

using dim2 = math::dim2;
using dim4 = math::dim4;

/*



*/