#pragma once
#include <math.h>
#include <string>
#include "core.h"
#include "array.h"

namespace grad
{
/*	enum types
	{
		GRAD_SIGMOID = 0,
		GRAD_FAST_SIGMOID,
		GRAD_ELU,
		GRAD_RELU,
		GRAD_LRELU,
		GRAD_PRELU,
		GRAD_SOFT_RELU,
		GRAD_ARC_TAN,
		GRAD_TANH,
		GRAD_SOFTMAX,
		GRAD_SOFTPLUS,
		GRAD_SUPER_SPIKE,
		GRAD_EXPONENTIAL,
	};

	static const char* GRAD_TYPE_STRING = "\
		Sigmoid\0\
		Fast Sigmoid\0\
		ELU\0\
		RELU\0\
		Leaky RELU\0\
		Parametric RELU\0\
		Soft RELU\0\
		Arc Tan\0\
		Tanh\0\
		Softmax\0\
		Softplus\0\
		Super Spike\0\
		Exponential\0\
	";
*/

	namespace sigmoid
	{
		const int32_t id = 0;

		CPU_GPU inline float gx(float x)
		{
			return 1.0f / (1.0f + std::exp(-x));
		};

		CPU_GPU inline float dx(float x)
		{
			return x * (1.0f - x);
		};

		CPU_GPU inline float sg(float v, float v_th)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th)
		{
			return 0.0f; //TODO: this
		};
	};

/*
	namespace fast_sigmoid
	{
		const int32_t id = 1;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float dx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 1.0f / (1.0f + std::exp(-beta*(v-v_th)));
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return beta * sg(v,v_th,beta) * (1.0f - sg(v,v_th,beta));
		};
	};

	namespace elu
	{
		const int32_t id = 2;
		const float default_alpha = 0.05f;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x, float alpha = 0.05f)
		{
			if(x > 0.0f)
				return x;
			return alpha * (std::exp(x) - 1.0f);
		};

		CPU_GPU inline float dx(float x, float alpha = 0.05f)
		{
			if(x >= 0.0f)
				return 1.0f;
			return alpha * std::exp(x);
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace relu
	{
		const int32_t id = 3;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return std::max(0.0f,x);
		};

		CPU_GPU inline float dx(float x)
		{
			return x >= 0.0f;
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace lrelu  //leaky relu
	{
		const int32_t id = 4;
		const float default_alpha = 0.05f;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x, float alpha = 0.05f)
		{
			if(x >= 0.0f)
				return x;
			return x * alpha;
		};

		CPU_GPU inline float dx(float x, float alpha = 0.05f)
		{
			if(x >= 0.0f)
				return 1.0f;
			return alpha;
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace prelu //parametric relu
	{
		const int32_t id = 5;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float dx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace soft_relu
	{
		const int32_t id = 6;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float dx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float r, float tr, float beta)
		{
			return 1.0f / (1.0f + std::exp(-beta*(r-tr)));
		};
	};

	namespace arc_tan
	{
		const int32_t id = 7;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float dx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sg(float x)
		{
			return 1.0f / (1.0f + x * x);
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace tanh
	{
		const int32_t id = 8;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
		};

		CPU_GPU inline float dx(float x)
		{
			return 1.0f - (gx(x) * gx(x));
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace softmax
	{
		const int32_t id = 9;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float dx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sg(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace softplus
	{
		const int32_t id = 10;
		const float default_beta = 1.0f;

		CPU_GPU inline float gx(float x, float beta = default_beta)
		{
			return std::log(1.0f+std::exp(beta * x));
		};

		CPU_GPU inline float dx(float x, float beta = default_beta)
		{
			return 1.0f / (1.0f + std::exp(-beta*x));
		};

		CPU_GPU inline float sg(float v, float v_th = 30.0f, float beta = default_beta)
		{
			return 1.0f / (1.0f + std::exp(-beta * (v - v_th)));
		};

		CPU_GPU inline float sd(float v, float v_th, float beta = default_beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace super_spike
	{
		const int32_t id = 11;
		const float default_alpha = 1.0f;
		const float default_beta = 1.0f;

		//input will need to be saved for fx
		CPU_GPU inline float gx(float x, float alpha = default_alpha)
		{
			return 0.0f; //TODO: heaviside function
		};

		//requires gradient and input value
		CPU_GPU inline float dx(float x, float alpha = default_alpha)
		{
			return gx(x,alpha) / std::pow((alpha * std::abs(x) + 1.0f),2.0f);
		};

		CPU_GPU inline float sg(float v, float v_th, float beta = default_beta)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sd(float v, float v_th, float beta = default_beta)
		{
			return 0.0f; //TODO: this
		};
	};

	namespace exponential
	{
		const int32_t id = 12;
		const float default_alpha = 1.0f;

		CPU_GPU inline float gx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float dx(float x)
		{
			return 0.0f; //TODO: this
		};

		CPU_GPU inline float sg(float v, float v_th = 30.0f, float alpha = default_alpha)
		{
			return alpha * std::exp(-alpha*(v-v_th));
		};

		CPU_GPU inline float sd(float v, float v_th = 30.0f, float alpha = default_alpha)
		{
			return 0.0f; //TODO: this
		};
	};*/
};

namespace sgrad
{
	template <typename E>
	struct grad_exp_t : public expr::exp_t<grad_exp_t<E>>
	{
		const E& self() const {return static_cast<const E&>(*this);};
	};

	namespace triangle
	{
		const float default_alpha = 1.0f;

		CPU_GPU inline float fx(float v, float v_th, float alpha = default_alpha)
		{
			float ax = alpha * (v - v_th);
			float d = 1.0f - std::abs(ax);
			return (d > 0.0f) ? d : 0.0f;
		};
	};

	namespace fast_sigmoid
	{
		const float default_alpha = 1.0f;
		const float default_beta = 1.5f;

		CPU_GPU inline float fx(float v, float v_th, float alpha = default_alpha)
		{
			float ax = alpha * (v - v_th);
			float d = 1.0f + std::abs(ax);
			return alpha / (d * d);
		};

		CPU_GPU inline float base(float v, float v_th, float beta = default_beta)
		{
			//1 / (1 + e^(-1.5 * (60 - 30))) = 
			return 1.0f / (1.0f + math::fast_exp(-beta*(v-v_th)));
		};

		//no idea what this is, but it works on the m4 version
		CPU_GPU inline float fx2(float v, float v_th, float beta = default_beta)
		{
			return beta * base(v,v_th,beta) * (1.0f - base(v,v_th,beta));
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float a;
			vv_fx_exp_t(const E &l_, const E &r_, float a_) : v(l_), t(r_), a(a_) {};
			float operator[](i64 i) const
			{
				float d = 1.0f / std::abs(a*(v[i]-t[i]));
				return a / (d*d);
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float alpha = default_alpha)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),alpha);
		};

		template <typename E>
		struct vv_fx_exp2_t : public grad_exp_t<vv_fx_exp2_t<E>>
		{
			const E& v;
			const E& t;
			float b;
			vv_fx_exp2_t(const E &l_, const E &r_, float b_) : v(l_), t(r_), b(b_) {};
			float operator[](i64 i) const
			{
				float base = 1.0f / (1.0f + std::exp(-b*(v[i]-t[i])));
				return b * base * (1.0f - base);
			};
		};

		template <typename E>
		inline vv_fx_exp2_t<E> fxx2(const exp_t<E> &v, const exp_t<E> &t, float beta = default_alpha)
		{
			return vv_fx_exp2_t<E>(v.self(),t.self(),beta);
		};
	};

	namespace exponential
	{
		const float default_alpha = 1.0f;

		CPU_GPU inline float fx(float v, float v_th, float alpha = default_alpha)
		{
			float ax = alpha * std::abs(v - v_th);
			return alpha * math::fast_exp(-ax);
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float a;
			vv_fx_exp_t(const E &l_, const E &r_, float a_) : v(l_), t(r_), a(a_) {};
			float operator[](i64 i) const
			{
				float ax = a * std::abs(v[i] - t[i]);
				return a * std::exp(-ax);
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float alpha = default_alpha)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),alpha);
		};
	};

	namespace logistic_sigmoid
	{
		const float default_alpha = 1.0f;

		CPU_GPU inline float fx(float v, float v_th, float alpha = default_alpha)
		{
			float ax = alpha * (v - v_th);
			float s = 1.0f / (1.0f + std::exp(-ax));
			return alpha * s * (1.0f - s);
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float a;
			vv_fx_exp_t(const E &l_, const E &r_, float a_) : v(l_), t(r_), a(a_) {};
			float operator[](i64 i) const
			{
				float ax = a * (v[i] - t[i]);
				float s = 1.0f / (1.0f + std::exp(-ax));
				return a * s * (1.0f - s);
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float alpha = default_alpha)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),alpha);
		};
	};

	namespace super_spike
	{
		const float default_beta = 1.0f;

		CPU_GPU inline float fx(float v, float v_th, float beta = default_beta)
		{
			//from the superspike github https://github.com/fzenke/pub2018superspike
			//this seems to be identical to fsa
			float h = (v + 50e-3) * beta;
			return beta / std::pow((1.0f + std::abs(h)),2.0f);
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float b;
			vv_fx_exp_t(const E &l_, const E &r_, float b_) : v(l_), t(r_), b(b_) {};
			float operator[](i64 i) const
			{
				float h = (v[i] + 50e-3) * b;
				return b / std::pow((1.0f + std::abs(h)),2.0f);
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float beta = default_beta)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),beta);
		};
	};

	namespace soft_relu
	{
		
		const float default_beta = 0.01f;

		CPU_GPU inline float fx(float r, float tr, float beta = default_beta)
		{
			return 1.0f / (1.0f + exp(-beta*(r-tr)));
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float b;
			vv_fx_exp_t(const E &l_, const E &r_, float b_) : v(l_), t(r_), b(b_) {};
			float operator[](i64 i) const
			{
				return 1.0f / (1.0f + exp(-b*(v[i]-t[i])));
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float beta = default_beta)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),beta);
		};
	};

	namespace unknown
	{
		const float default_sigma = 1.0f;

		CPU_GPU inline float fx(float v, float v_th = 30.0f, float sigma = default_sigma)
		{
			return math::fast_exp(-(v - v_th) * (v - v_th) / (2.0f * sigma * sigma));
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float s;
			vv_fx_exp_t(const E &l_, const E &r_, float s_) : v(l_), t(r_), s(s_) {};
			float operator[](i64 i) const
			{
				return math::fast_exp(-(v[i] - t[i]) * (v[i] - t[i]) / (2.0f * s * s));
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float sigma = default_sigma)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),sigma);
		};
	};

	namespace dsrm
	{
		const float default_tau_s = 1.0f;

		CPU_GPU inline float fx(float t, float tau_s = default_tau_s)
		{
			return (1.0f / tau_s) * std::exp(-(t/tau_s)) * math::heaviside(t);
		};
	};

	namespace fsig
	{
		const float default_k = 15.0f;

		CPU_GPU inline float fx(float v, float v_th, float k = default_k)
		{
			return 1.0f / std::pow(1.0f+k*std::abs(v-v_th),2.0f);
		};

		using namespace expr;

		template <typename E>
		struct vv_fx_exp_t : public grad_exp_t<vv_fx_exp_t<E>>
		{
			const E& v;
			const E& t;
			float k;
			vv_fx_exp_t(const E &l_, const E &r_, float k_) : v(l_), t(r_), k(k_) {};
			float operator[](i64 i) const
			{
				return 1.0f / std::pow(1.0f+k*std::abs(v[i]-t[i]),2.0f);
			};
			i64 size() const {return v.size();};
		};
		
		template <typename E>
		inline vv_fx_exp_t<E> fxx(const exp_t<E> &v, const exp_t<E> &t, float k = default_k)
		{
			return vv_fx_exp_t<E>(v.self(),t.self(),k);
		};
	};

	namespace atan
	{
		CPU_GPU inline float fx(float v, float v_th)
		{
			return 1.0f / std::pow(1.0f + (M_PI * (v-v_th)),2.0f);
		};
	};

	struct util_t
	{
		float triangle_alpha = triangle::default_alpha;
		float fast_sigmoid_alpha = fast_sigmoid::default_alpha;
		float fast_sigmoid_beta = fast_sigmoid::default_beta;
		float exponential_alpha = exponential::default_alpha;
		float logistic_sigmoid_alpha = logistic_sigmoid::default_alpha;
		float super_spike_beta = super_spike::default_beta;
		float soft_relu_beta = soft_relu::default_beta;
		float unknown_sigma = unknown::default_sigma;
		float dsrm_tau_s = dsrm::default_tau_s;
		float fsig_k = dsrm::default_tau_s;
		float atan_x = 0.0f;

		const char* long_names =
"Triangle\0Fast Sigmoid A\0Fast Sigmoid B\0Exponential\0Logistic Sigmoid\0\
Super Spike\0Soft ReLU\0Unknown\0DSRM\0FSig\0ATan\0";

		const char* short_names = "Tri\0FSA\0FSB\0Exp\0LogS\0SS\0SR\0Unk\0DSRM\0FSig\0ATan\0";

		float fn(int32_t id, float x, float y) const
		{
			switch(id)
			{
				case( 0): return triangle::fx(x,y,triangle_alpha);
				case( 1): return fast_sigmoid::fx(x,y,fast_sigmoid_alpha);
				case( 2): return fast_sigmoid::fx2(x,y,fast_sigmoid_beta);
				case( 3): return exponential::fx(x,y,exponential_alpha);
				case( 4): return logistic_sigmoid::fx(x,y,logistic_sigmoid_alpha);
				case( 5): return super_spike::fx(x,y,super_spike_beta);
				case( 6): return soft_relu::fx(x,y,soft_relu_beta);
				case( 7): return unknown::fx(x,y,unknown_sigma);
				case( 8): return dsrm::fx(x,dsrm_tau_s);
				case( 9): return fsig::fx(x,y,fsig_k);
				case(10): return atan::fx(x,y);
			}
			return 0.0f;
		};

		/*template <typename E>
		const grad_exp_t<E> fn_exp(i32 id, const expr::exp_t<E> &v, const expr::exp_t<E> &t) const
		{
			switch(id)
			{
				default:
				//case(0): return triangle::fx(v,t,triangle_alpha);
				case(1): return fast_sigmoid::fxx(v,t,fast_sigmoid_alpha);
				case(2): return fast_sigmoid::fxx2(v,t,fast_sigmoid_beta);
				case(3): return exponential::fxx(v,t,exponential_alpha).self();
				case(4): return logistic_sigmoid::fxx(v,t,logistic_sigmoid_alpha).self();
				case(5): return super_spike::fxx(v,t,super_spike_beta).self();
				case(6): return soft_relu::fxx(v,t,soft_relu_beta).self();
				case(7): return unknown::fxx(v,t,unknown_sigma).self();
				//case(8): return dsrm::fx(v,dsrm_tau_s);
				case(9): return fsig::fxx(v,t,fsig_k).self();
			}
		};*/

		/*template <typename E>
		expr::exp_t<E> fn_exp(i32 id, const expr::exp_t<E> &v, float t) const
		{
			switch(id)
			{
				default:
				//case(0): return triangle::fx(v,t,triangle_alpha);
				case(1): return fast_sigmoid::fx(v,t,fast_sigmoid_alpha);
				case(2): return fast_sigmoid::fx2(v,t,fast_sigmoid_beta);
				case(3): return exponential::fx(v,t,exponential_alpha);
				case(4): return logistic_sigmoid::fx(v,t,logistic_sigmoid_alpha);
				case(5): return super_spike::fx(v,t,super_spike_beta);
				case(6): return soft_relu::fx(v,t,soft_relu_beta);
				case(7): return unknown::fx(v,t,unknown_sigma);
				//case(8): return dsrm::fx(v,dsrm_tau_s);
				case(9): return fsig::fx(v,t,fsig_k);
			}
		};*/

		float& get_value(int32_t id)
		{
			switch(id)
			{
				case( 0): return triangle_alpha;
				case( 1): return fast_sigmoid_alpha;
				case( 2): return fast_sigmoid_beta;
				case( 3): return exponential_alpha;
				case( 4): return logistic_sigmoid_alpha;
				case( 5): return super_spike_beta;
				case( 6): return soft_relu_beta;
				case( 7): return unknown_sigma;
				case( 8): return dsrm_tau_s;
				case( 9): return fsig_k;
				case(10): return atan_x;
				default: return triangle_alpha;
			}
		};

		float get_default(int32_t id)
		{
			switch(id)
			{
				case( 0): return triangle::default_alpha;
				case( 1): return fast_sigmoid::default_alpha;
				case( 2): return fast_sigmoid::default_beta;
				case( 3): return exponential::default_alpha;
				case( 4): return logistic_sigmoid::default_alpha;
				case( 5): return super_spike::default_beta;
				case( 6): return soft_relu::default_beta;
				case( 7): return unknown::default_sigma;
				case( 8): return dsrm::default_tau_s;
				case( 9): return fsig::default_k;
				case(10): return 0.0f;
				default: return 0.0f;
			}
		};

		void reset(int32_t id)
		{
			switch(id)
			{
				case(0): triangle_alpha = triangle::default_alpha;
				case(1): fast_sigmoid_alpha = fast_sigmoid::default_alpha;
				case(2): fast_sigmoid_beta = fast_sigmoid::default_beta;
				case(3): exponential_alpha = exponential::default_alpha;
				case(4): logistic_sigmoid_alpha = logistic_sigmoid::default_alpha;
				case(5): super_spike_beta = super_spike::default_beta;
				case(6): soft_relu_beta = soft_relu::default_beta;
				case(7): unknown_sigma = unknown::default_sigma;
				case(8): dsrm_tau_s = dsrm::default_tau_s;
				case(9): fsig_k = fsig::default_k;
			}
		};

		std::string get_long_name(int32_t id) const
		{
			switch(id)
			{
				case( 0): return "Triangle";
				case( 1): return "Fast Sigmoid A";
				case( 2): return "Fast Sigmoid B";
				case( 3): return "Exponential";
				case( 4): return "Logistic Sigmoid";
				case( 5): return "Super Spike";
				case( 6): return "Soft ReLU";
				case( 7): return "Unknown";
				case( 8): return "DSRM";
				case( 9): return "FSig";
				case(10): return "ATan";
				default: return "Invalid!";
			}
		};

		std::string get_short_name(int32_t id) const
		{
			switch(id)
			{
				case( 0): return "Tri";
				case( 1): return "FSA";
				case( 2): return "FSB";
				case( 3): return "Exp";
				case( 4): return "LogS";
				case( 5): return "SS";
				case( 6): return "SR";
				case( 7): return "Unk";
				case( 8): return "DSRM";
				case( 9): return "FSig";
				case(10): return "ATan";
				default: return "Invalid!";
			}
		};

		std::string get_param_name(int32_t id) const
		{
			switch(id)
			{
				case( 0): return "Alpha";
				case( 1): return "Alpha";
				case( 2): return "Beta";
				case( 3): return "Alpha";
				case( 4): return "Alpha";
				case( 5): return "Beta";
				case( 6): return "Beta";
				case( 7): return "Sigma";
				case( 8): return "Tau S";
				case( 9): return "K";
				case(10): return "N/A";
				default: return "Invalid!";
			}
		};
	};
};

struct grad_util_t
{
	struct triangle_t
	{
		static constexpr std::string long_name = "Triangle";
		static constexpr std::string short_name = "Tri";
		static constexpr float default_alpha = 1.0f;
		float alpha = default_alpha;

		static inline float fx(float v, float v_th, float alpha = default_alpha)
		{
			float ax = alpha * (v - v_th);
			float d = 1.0f - std::abs(ax);
			return (d > 0.0f) ? d : 0.0f;
		};
	} triangle;
};