#include "snn.h"
#include "neurons.h"
#include "synapses.h"
#include "../math/conv.h"
#include "../math/kernels.h"
#include "../math/food.h"
#include "../math/simd/include.h"
#include "neurons.h"
//#include <cblas.h>
#include <stdexcept>

namespace snn5
{
	void s_block_t::__reset()
	{
		last_step = -500;

		if(config.weight.use_global)
			config.weight = net->config.weight;
		
		if(config.feedback.use_global)
			config.feedback = net->config.feedback;

		const bool bpa = config.backprop.active;
		if(config.backprop.use_global)
			config.backprop = net->config.backprop;
		config.backprop.active = bpa;

		const bool mpa = config.metaplasticity.active;
		if(config.metaplasticity.use_global)
			config.metaplasticity = net->config.metaplasticity;
		config.metaplasticity.active = mpa;

		const auto &wc = config.weight;
		if(wc.use_xe_init)
		{
			float limit = std::sqrt(6.0f/(prev->size()+next->size()));
			w.randomize(net->rng,-limit,limit);
		}
		else
		{
			w.randomize(net->rng,wc.i_min,wc.i_max);
		}
		if(config.weight.enable_trace)
			et.zero();
		
		if(config.metaplasticity.active)
		{
			wr.copy_from(w);
			mp.fill(config.metaplasticity.init_m);
		}
		if(net->config.rate_backprop.active)
			ef.zero();
		
		const auto fc = config.feedback;
		fb.randomize(net->rng,fc.i_min,fc.i_max);
		if(fc.enable_trace)
			ft.zero();

		if(config.stp.active)
		{
			stp.a = config.stp.u > 0.0f ? 1.0f / config.stp.u : 1.0f;
			stp.u.zero();
			stp.x.fill(1.0f);
			if(net->config.use_gpu)
				stp.dw.fill(1.0f);
		}

		const auto &stdp_c = config.stdp;
		if(stdp_c.active)
		{
			stdp.lut.resize(512);
			for(i64 i = 0; i < 512; i++)
			{
				stdp.lut[i] = kernel::stdp_strict(
					stdp_c.alpha_ltp,stdp_c.alpha_ltd,
					stdp_c.tau_ltp,stdp_c.tau_ltd,
					0,i+stdp_c.lut_offset
				) * stdp_c.scale;
			}
		}

		inv_tau_e = std::exp(-1.0f/config.weight.trace_decay);
		inv_tau_f = std::exp(-1.0f/config.feedback.trace_decay);
	};

	void s_block_t::__free()
	{
		w.free();
		et.free();
		ef.free();
		wr.free();
		mp.free();
		fb.free();
		ft.free();
		stp.u.free();
		stp.x.free();
		stp.dw.free();
		stdp.lut.free();
	};

	s_block_t::s_block_t(
		network_t* net_, i32 id_,
		n_block_t* prev_, n_block_t* next_,
		s_block_config_t config_
	) : block_t(id_), net(net_), prev(prev_), next(next_), config(config_)
	{
		__reset();
	};

	s_block_t::~s_block_t()
	{
		__free();
	};

	void s_block_t::forward(bool no_input)
	{
		const bool is_training = net->state.is_training;

		const i64 ps = prev->size();
		const i64 ns = next->size();
		const bool* pf = prev->spike.fired;
		const bool* nf = next->spike.fired;
		
		if(config.stp.active)
		{
			const auto stp_c = config.stp;

			for(i64 p = 0; p < ps; p++)
			{
				bool spiked = pf[p];
				for(i64 n = 0; n < ns; n++)
				{
					i64 i = p * ns + n;
					float up = stp.u[i];
					float xp = stp.x[i];

					float du = kernel::stp_du(stp_c.u,stp_c.tau_u,up,spiked);
					float dx = kernel::stp_dx(xp,up+du,stp_c.tau_x,spiked);
					float dw = kernel::stp_dw(stp.a,up+du,xp);

					stp.u[i] = std::clamp<float>(up+du,0.01f,0.99f);
					stp.x[i] = std::clamp<float>(xp+dx,0.01f,0.99f);

					if(spiked)
						next->get_c()[n] += dw * w[i];
				}
			}
		}
		else
		{
			const auto psr = prev->rec;
			if(!no_input)
				for(i64 p = 0; p < psr.size; p++)
					for(i64 n = 0; n < ns; n++)
						next->c[n] += w[psr.indices[p]*ns+n];
			/*math::bit_mask_dot(
				prev->size(),next->size(),prev->spike.mask.size(),
				w,next->get_c(),prev->spike.mask
			);*/
		}

		if(net->config.metaplasticity.active && is_training)
		{
			const auto mpc = net->config.metaplasticity;

			auto mp_fn = [&](i64 i)
			{
				float alpha = mpc.base_alpha;

				if(mpc.use_dynamic_alpha)
					alpha = nachos::get_decay_rate_4_6(alpha,mp[i],10.0f,10.0f);

				w[i] += nachos::decay_regular_weight_4_5(alpha,w[i],wr[i],mpc.s_i);
			};

			if(mpc.use_post_spike_for_decay)
			{
				for(i64 n = 0; n < ns; n++)
					if(nf[n])
						for(i64 p = 0; p < ps; p++)	
							mp_fn(p*ns+n);
			}
			else
			{
				for(i64 p = 0; p < ps; p++)
					if(pf[p])
						for(i64 n = 0; n < ns; n++)
							mp_fn(p*ns+n);
			}
		}

		//if(!net->config.rate_backprop.active && config.weight.enable_trace)
		//	math::bit_mask_trace(et,prev->spike.mask,inv_tau_e,prev->spike.mask.size());
		for(i64 i = 0; i < ps; i++)
		{
			et[i] *= inv_tau_e;
			if(prev->spike.fired[i])
				et[i] += 1.0f;
		}

		last_step = net->state.time;
	};

	void s_block_t::backward()
	{
		//CBLAS_TRANSPOSE trans = net->config.use_np_layout ? CblasTrans : CblasNoTrans;
		
		float* x = nullptr;
		switch(config.backprop.value_selection)
		{
			default:
			case(0): x = w;  break;
			case(1): x = fb; break;
		}

		//using sgemv reduces score by ~1% for a 2% speedup. not worth it.
		/*cblas_sgemv(
			CblasColMajor,trans,
			prev->size(),next->size(),
			1.0f,x,prev->size(),
			next->bp.err,1,0.0f,
			prev->bp.err,1
		);*/
		for(i32 i = 0; i < prev->size(); i++)
			for(i32 j = 0; j < next->size(); j++)
				prev->bp.err.at(i) += next->bp.err.at(j) * fb.at(i,j);

		prev->config.error.adjust.apply(prev->bp.err);

		prev->surrogate();

		if(net->config.rate_backprop.active)
			prev->update_et_ef();

		if(!net->config.accumulate_grad)
			prev->bp.grad.zero();
		prev->bp.grad += prev->bp.err * prev->bp.sd;
		prev->config.gradient.adjust.apply(prev->bp.grad);
	};

	void s_block_t::update_weights()
	{
		const bool do_backprop = net->state.do_backprop;
		const bool need_weight_updates = config.backprop.active && do_backprop;

		const i64 ps = prev->size();
		const i64 ns = next->size();
		const i32* pl = prev->spike.last;
		const i32* nl = next->spike.last;
		const bool* pf = prev->spike.fired;
		const bool* nf = next->spike.fired;
		const float* pt = prev->spike.trace;
		const float* nt = next->spike.trace;

		const auto &psr = prev->rec;
		const auto &nsr = next->rec;
		const i32* psri = psr.indices;
		//const i32* psrt = psr.times;
		const i32* nsri = nsr.indices;
		//const i32* nsrt = nsr.times;

		const float w_min = config.weight.r_min;
		const float w_max = config.weight.r_max;

		const auto mpc = config.metaplasticity;
		if(mpc.active && (!mpc.require_weight_update || do_backprop))
		{
			float inv_m_max = 1.0f / mpc.max_m;
			for(i64 p = 0; p < ps; p++)
			{
				for(i64 n = 0; n < ns; n++)
				{
					float dm = nachos::get_delta_m_4_3_hack(
						mpc.base_update,mp[p*ns+n],inv_m_max
					);

					mp[p*ns+n] = nachos::get_new_metaplastic_state_4_2(
						mp[p*ns+n],dm,nt[n],pt[p],
						mpc.pre_th1,mpc.post_th1,
						mpc.pre_th2,mpc.post_th2
					);
				}
			}

			for(i64 i = 0; i < size(); i++)
			{
				wr[i] = nachos::synaptic_consolidation_4_4(w[i],wr[i],1.0f,mpc.t_ref);
				wr[i] = std::clamp<float>(wr[i],w_min,w_max);
			}
		}

		if(config.backprop.active && do_backprop)
		{
			const auto psr = prev->rec;
			float lr = config.weight.learn_rate;

			for(i64 p = 0; p < psr.size; p++)
			{
				i64 pi = psr.indices[p];
				for(i64 n = 0; n < ns; n++)
				{
					float dw = 0.0f;

					if(net->config.rate_backprop.active)
						dw = next->bp.grad[n] * lr * next->bp.et[n];
					else
						dw = next->bp.grad[n] * lr * et[pi];
					if(mpc.active)
						dw *= nachos::calc_plasticity_4_1(mp[pi*ns+n],w[pi*ns+n]);

					w[pi*ns+n] = std::clamp<float>(w[pi*ns+n]-dw,w_min,w_max);
				}
			}
		}

		if(config.stdp.active)
		{
			if(config.stdp.use_on_weights)
			{
				i32 lo = config.stdp.lut_offset;

				if(net->config.use_event_stdp)
				{
					auto stdp_fn_ep = [&](i32 p, i32 n)
					{
						i32 pi = psri[p];
						//i32 dt = nl[n] - psrt[p] - lo;
						i32 dt = nl[n] - pl[pi] - lo;
						dt = std::clamp<i32>(dt,0,511);
						float dw = stdp.lut[dt];
						if(mpc.active)
							dw *= nachos::calc_plasticity_4_1(mp[pi*ns+n],w[pi*ns+n]);
						w[pi*ns+n] -= dw;
					};

					auto stdp_fn_en = [&](i32 p, i32 n)
					{
						i32 ni = nsri[n];
						//i32 dt = nsrt[n] - pl[p] - lo;
						i32 dt = nl[ni] - pl[p] - lo;
						dt = std::clamp<i32>(dt,0,511);
						float dw = stdp.lut[dt];
						if(mpc.active)
							dw *= nachos::calc_plasticity_4_1(mp[p*ns+ni],w[p*ns+ni]);
						w[p*ns+ni] -= dw;
					};

					for(i64 p = 0; p < psr.size; p++)
						for(i64 n = 0; n < ns; n++)
							stdp_fn_ep(p,n);

					for(i64 n = 0; n < nsr.size; n++)
						for(i64 p = 0; p < ps; p++)
							stdp_fn_en(p,n);
				}
				else
				{
					auto stdp_fn = [&](i64 p, i64 n)
					{
						i32 dt = nl[n] - pl[p] - lo;
						dt = std::clamp<i32>(dt,0,511);
						float dw = stdp.lut[dt];
						if(mpc.active)
							dw *= nachos::calc_plasticity_4_1(mp[p*ns+n],w[p*ns+n]);
						w[p*ns+n] -= dw;
					};

					for(i64 p = 0; p < ps; p++)
						if(pf[p])
							for(i64 n = 0; n < ns; n++)
								stdp_fn(p,n);

					for(i64 n = 0; n < ns; n++)
						if(nf[n])
							for(i64 p = 0; p < ps; p++)
								stdp_fn(p,n);
				}
			}
		}

		//if(need_weight_updates)
		//	w.clamp(w_min,w_max);
	};

	void s_block_t::update_feedback()
	{

	};

	fc_s_block_t::fc_s_block_t(
		network_t* net_, i32 id_,
		n_block_t* prev_, n_block_t* next_,
		s_block_config_t config_
	) : s_block_t(net_,id_,prev_,next_,config_)
	{
		i64 p = prev->size();
		i64 n = next->size();
		w.resize({p,n});
		wr.resize({p,n});
		fb.resize({p,n});
		mp.resize({p,n});
		et.resize(p);
		ef.resize(p);
		ft.resize({p,n});
		stp.u.resize({p,n});
		stp.x.resize({p,n});
		stp.dw.resize({p,n});

		printf("w.size(): %li\n",w.size());
		printf("w.x(): %li\n",w.x());
		printf("w.y(): %li\n",w.y());
		printf("w.z(): %li\n",w.z());
		printf("w.w(): %li\n",w.w());

		reset();
	};

	fc_s_block_t::~fc_s_block_t()
	{
		__free();
	};

	void fc_s_block_t::reset()
	{
		__reset();
	};

	conv_s_block_t::conv_s_block_t(
		network_t* net_, i32 id_,
		n_block_t* prev_, n_block_t* next_,
		s_block_config_t config_
	) : s_block_t(net_,id_,prev_,next_,config_)
	{
		dim2 pd = prev->shape();
		dim2 nd = next->shape();
		dim4 d = {nd.x,nd.y,config.kernel.size.x,config.kernel.size.y};
		w.resize(d);
		wr.resize(d);
		fb.resize(d);
		mp.resize(d);
		et.resize({pd.x,pd.y});
		ef.resize(nd);
		ft.resize(nd);
		stp.u.resize(d);
		stp.x.resize(d);
		stp.dw.resize(d);

		reset();
	};

	conv_s_block_t::~conv_s_block_t()
	{
		__free();
	};

	void conv_s_block_t::reset()
	{
		__reset();
	};

	void conv_s_block_t::forward(bool no_input)
	{
		const bool is_training = net->state.is_training;

		const i64 ps = prev->size();
		const i64 ns = next->size();
		const bool* pf = prev->spike.fired;
		const bool* nf = next->spike.fired;
		
		if(config.stp.active)
		{
			const auto stp_c = config.stp;

			throw std::runtime_error("STP not supported on convolutional layers!\n");
		}
		else
		{
			if(!no_input)
				math::conv_2d(config.kernel,w,prev->spike.value,next->c);
		}

		if(net->config.metaplasticity.active && is_training)
		{
			const auto mpc = net->config.metaplasticity;

			throw std::runtime_error("Metaplasticity not supported on convolutional layers!\n");
		}

		//if(!net->config.rate_backprop.active && config.weight.enable_trace)
		//	math::bit_mask_trace(et,prev->spike.mask,inv_tau_e,prev->spike.mask.size());
		for(i64 i = 0; i < ps; i++)
		{
			et[i] *= inv_tau_e;
			if(prev->spike.fired[i])
				et[i] += 1.0f;
		}

		last_step = net->state.time;
	};

	void conv_s_block_t::backward()
	{
		//CBLAS_TRANSPOSE trans = net->config.use_np_layout ? CblasTrans : CblasNoTrans;
		
		tensor<float> x;
		switch(config.backprop.value_selection)
		{
			default:
			case(0): x = w;  break;
			case(1): x = fb; break;
		}

		math::conv_2d(config.kernel,x,next->bp.err,prev->bp.err);

		prev->config.error.adjust.apply(prev->bp.err);

		prev->surrogate();

		if(net->config.rate_backprop.active)
			prev->update_et_ef();

		if(!net->config.accumulate_grad)
			prev->bp.grad.zero();
		prev->bp.grad = prev->bp.err * prev->bp.sd;
		prev->config.gradient.adjust.apply(prev->bp.grad);
	};

	void conv_s_block_t::update_weights()
	{
		const bool do_backprop = net->state.do_backprop;
		const bool need_weight_updates = config.backprop.active && do_backprop;

		const i32 ps = prev->size();
		const i32 ns = next->size();
		const i32* pl = prev->spike.last;
		const i32* nl = next->spike.last;
		const bool* pf = prev->spike.fired;
		const bool* nf = next->spike.fired;
		const float* pt = prev->spike.trace;
		const float* nt = next->spike.trace;

		const auto &psr = prev->rec;
		const auto &nsr = next->rec;
		const i32* psri = psr.indices;
		//const i32* psrt = psr.times;
		const i32* nsri = nsr.indices;
		//const i32* nsrt = nsr.times;

		const float w_min = config.weight.r_min;
		const float w_max = config.weight.r_max;

		const auto mpc = config.metaplasticity;
		if(mpc.active && (!mpc.require_weight_update || do_backprop))
		{
			throw std::runtime_error("Metaplasticity not supported in convolutional layer!\n");
		}

		if(config.backprop.active && do_backprop)
		{
			float lr = config.weight.learn_rate;
			math::conv_2d_back(config.kernel,w,next->bp.grad,et,prev->spike.value,lr);
		}

		if(config.stdp.active)
		{
			throw std::runtime_error("STDP not supported in convolutional layer!\n");
		}

		if(need_weight_updates)
			w.clamp(w_min,w_max);
	};

	void conv_s_block_t::update_feedback()
	{

	};

	/*
		fc fwd
		for(i = 0; i < ps; i++)
			for(j = 0; j < ns; j++)
				next->c.at(j) += w.at(i,j) * prev->fired.at(i);
		
		fc bwd
		for(i = 0; i < ps; i++)
			for(j = 0; j < ns; j++)
				prev->err.at(i) += fb.at(i,j) * next->err.at(j);

		conv fwd
		for(i = 0; i + ks <= px; i++)
			for(j = 0; j + ks <= py; j++)
				float sum = 0.0f;
				for(k = 0; k < ks; k++)
					for(l = 0; l < ks; l++)
						sum += w.at(i+k,j+l) * fired.at(i,j);
				next->c.at(i,j) += sum;
		
		conv bwd
		for(i = 0; i + ks <= px; i++)
			for(j = 0; j + ks <= py; j++)
				float sum = 0.0f;
				for(k = 0; k < ks; k++)
					for(l = 0; l < ks; l++)
						sum += fb.at(i+k,j+l) * next->err.at(i,j);
				prev->err.at(i,j) += sum;

	*/
};