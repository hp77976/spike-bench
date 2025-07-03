#include "neurons.h"
#include "../math/simd/include.h"
#include "../math/kernels.h"
#include "../math/gradients.h"
#include "snn.h"

namespace snn5
{
	spike_data_t::spike_data_t(dim2 s, spike_data_config_t cfg) : config(cfg)
	{
		//ema_rate.resize(s);
		//bit_rate.resize(s);
		last.resize(s);
		//mask.resize(calc_blocks(s,8).size());
		//hist.resize(s);
		trace.resize(s);
		fired.resize(s);
		value.resize(s);
		//back.resize(s);
		count.resize(s);
		reset();
	};
	
	spike_data_t::~spike_data_t()
	{
		//ema_rate.free();
		//bit_rate.free();
		last.free();
		//mask.free();
		//hist.free();
		trace.free();
		value.free();
		fired.free();
		//back.free();
		count.free();
	};

	void spike_data_t::reset()
	{
		//ema_rate.zero();
		//bit_rate.zero();
		last.fill(-500);
		//mask.zero();
		//hist.zero();
		trace.zero();
		fired.zero();
		value.zero();
		//back.zero();
		count.zero();
	};

	void spike_data_t::update_rates(i64 i)
	{
		//bit_rate[i] = get_set_bits(hist[i]);
	};

	void spike_data_t::update_mask(i32 time)
	{
		i64 ls = size();

		for(i64 i = 0; i < ls; i++)
		{
			trace[i] *= config.trace_decay;
			if(fired[i])
			{
				value[i] = 1.0f;
				last[i] = time;
				trace[i] += 1.0f;
				count[i]++;
			}
			else
			{
				value[i] = 0.0f;
			}

			//ema_rate[i] = math::ema(config.ema_alpha,ema_rate[i],value[i]*config.ema_mul);
		}

		/*simd::f32_8* er = ema_rate.as<simd::f32_8>();
		simd::f32_8* v8 = value.as<simd::f32_8>();
		simd::i32_8* lt = last.as<simd::i32_8>();
		simd::f32_8* tr = trace.as<simd::f32_8>();
		simd::i32_8 t8 = time;
		simd::f32_8 a8 = config.ema_alpha;
		simd::f32_8 m8 = config.ema_mul;
		simd::f32_8 td8 = config.trace_decay;
		simd::f32_8 one = 1.0f;
		simd::f32_8 zero = 0.0f;
		i64 ls_8 = mask.size();
		simd::i64_8 last_bit = 0b01;
		for(i64 i = 0; i < ls_8; i++)
		{
			u8 temp_mask = simd::bit_and(hist.as<simd::i64_8>()[i],last_bit);

			tr[i] *= td8;
			
			if(temp_mask != 0)
			{
				mask[i] = temp_mask;
				v8[i] = simd::blend(zero,one,temp_mask);
				lt[i] = simd::blend(lt[i],t8,temp_mask);
				tr[i] = simd::mask_add(tr[i],tr[i],one,temp_mask);
			}

			er[i] = math::ema(a8,er[i],v8[i]*m8);
		}*/
	};

	i64 spike_data_t::bytes() const
	{
		i64 s = 0;
		s += sizeof(spike_data_t);
		//s += ema_rate.bytes();
		//s += bit_rate.bytes();
		s += last.bytes();
		//s += mask.bytes();
		//s += hist.bytes();
		s += trace.bytes();
		s += value.bytes();
		s += fired.bytes();
		s += count.bytes();
		return s;
	};

	i64 spike_data_t::size() const
	{
		return fired.size();
	};

	n_block_t::n_block_t(
		network_t* n, i32 id_, dim2 s, n_block_config_t cfg
	) : block_t(id_), spike(s,cfg.spike)
	{
		net = n;
		config = cfg.n_config;

		v.resize(s);
		c.resize(s);
		tr.resize(s);
		tr2.resize(s);

		bp.grad.resize(s);
		bp.err.resize(s);
		bp.sd.resize(s);
		bp.et.resize(s);
		bp.ef.resize(s);

		target.resize(s);

		rec = spike_record_t(s);

		printf("v.size(): %li\n",v.size());

		__reset();
	};

	n_block_t::~n_block_t()
	{
		__free();
	};

	void n_block_t::__reset()
	{
		v.zero();
		c.zero();
		tr.zero();
		tr2.zero();
		spike.reset();
		rec.reset();

		bp.grad.zero();
		bp.err.zero();
		bp.sd.zero();
		bp.et.zero();
		bp.ef.fill(1.0f);

		target.fill(0.0f);

		inv_tau_t = std::exp(-1.0f/spike.config.trace_decay);
		inv_tau_c = std::exp(-1.0f/config.current_decay);

		if(net != nullptr)
		{
			if(config.error.use_global)
				config.error = net->config.error;
			if(config.surrogate.use_global)
				config.surrogate = net->config.surrogate;
			if(config.input.use_global)
				config.input = net->config.input;
			if(config.target.use_global)
				config.target = net->config.target;
			if(config.output.use_global)
				config.output = net->config.output;
			spike.config = net->config.soma.spike;
		}

		last_step = -1;
	};
	
	void n_block_t::__free()
	{
		v.free();
		c.free();
		tr.free();
		tr2.free();
		target.free();
		bp.grad.free();
		bp.err.free();
		bp.sd.free();
		bp.et.free();
		bp.ef.free();
	};

	void n_block_t::surrogate()
	{
		const auto &s_cfg = config.surrogate;
		const i32 fn = s_cfg.function;
		//bp.sd = s_cfg.util.fn_exp(fn,v,tr);
		for(i64 i = 0; i < size(); i++)
			bp.sd[i] = s_cfg.util.fn(fn,v[i],tr[i]);
		s_cfg.adjust.apply(bp.sd);
	};

	void n_block_t::calculate_error()
	{
		math::array<float> rate;
		switch(net->config.error.rate_selection)
		{
			default:
			//case(0): rate = spike.ema_rate; break;
			//case(1): rate = spike.bit_rate; break;
			case(2): rate = spike.trace; break;
			case(3): rate = spike.count; break;
		}

		bp.err = (target - rate) * -1.0f;
		config.error.adjust.apply(bp.err);

		surrogate();

		if(net->config.rate_backprop.active)
			update_et_ef();

		if(!net->config.accumulate_grad)
			bp.grad.zero();
		bp.grad = bp.grad + bp.err * bp.sd;
		config.gradient.adjust.apply(bp.grad);
	};

	void n_block_t::update_et_ef()
	{	
		i32 time = net->state.time;
		bp.et = (1.0f / (time + 1.0f)) * (time * bp.et + bp.ef + bp.sd);

		float tau = net->config.rate_backprop.tau;
		if(net->config.rate_backprop.tau_use_sigmoid)
			tau = 1.0f / (1.0f - grad::sigmoid::gx(tau));
		float lam = 1.0f / (1.0f - tau);
		if(net->config.rate_backprop.manual_tau)
			lam = net->config.rate_backprop.tau;

		if(net->config.rate_backprop.hard_reset)
		{
			bp.ef = 1.0f + bp.ef * (lam * (1.0f - spike.value) - lam * v * bp.sd);
		}
		else
		{
			if(net->config.rate_backprop.detach)
				bp.ef = 1.0f + bp.ef * lam;
			else
				bp.ef = 1.0f + bp.ef * (lam * lam * bp.sd);
		}
	};

	lif_block_t::lif_block_t(
		network_t* n, i32 id_, dim2 s, n_block_config_t cfg
	) : n_block_t(n,id_,s,cfg), m_config(cfg.lif)
	{
		reset();
	};

	lif_block_t::~lif_block_t()
	{
		__free();
	};

	void lif_block_t::reset()
	{
		__reset();
		v.fill(m_config.v_rest);
		lif_tau_c = std::exp(-1.0f/config.current_decay);
	};

	void lif_block_t::step()
	{
		i64 ls = size();
		i32 time = last_step + 1;
		if(net != nullptr)
			time = net->state.time;

		spike.fired.zero();
		spike.value.zero();
		//spike.mask.zero();
				
		rec.reset();

		const auto mcfg = m_config;

		for(i64 i = 0; i < ls; i++)
		{
			if(mcfg.keep_high_v)
				if(spike.last[i] == last_step)
					v[i] = mcfg.v_reset;

			//spike.hist[i] <<= 1u;

			if(time - spike.last[i] >= mcfg.refractory_period)
			{
				float dv = mcfg.v_reset + c[i] * mcfg.resistance;
				v[i] = dv + (v[i] - dv) * lif_tau_c;
				v[i] = std::max(v[i],mcfg.v_reset);

				if(v[i] >= tr[i])
				{
					if(!mcfg.keep_high_v)
						v[i] = mcfg.v_reset;

					//spike.hist[i] |= 1u;
					spike.fired[i] = true;

					rec.indices[rec.size] = i;
					//rec.times[rec.size] = time;
					rec.size++;
				}
			}

			spike.update_rates(i);

			if(config.use_current_decay)
				c[i] *= inv_tau_c;
		}

		spike.update_mask(time);
		last_step = time;
	};

	i64 lif_block_t::bytes() const
	{
		i64 s = 0;
		s += v.bytes() + c.bytes() + tr.bytes();
		s += spike.bytes() + sizeof(lif_block_t);
		return s;
	};

	rng_block_t::rng_block_t(
		network_t* n, i32 id_, dim2 s, n_block_config_t cfg
	) : n_block_t(n,id_,s,cfg), m_config(cfg.rng)
	{
		reset();
	};

	rng_block_t::~rng_block_t()
	{
		__free();
	};

	void rng_block_t::reset()
	{
		__reset();
	};

	void rng_block_t::step()
	{
		i64 ls = size();
		i32 time = last_step + 1;
		if(net != nullptr)
			time = net->state.time;

		spike.fired.zero();
		spike.value.zero();
		//spike.mask.zero();
		//spike.back.zero();

		rec.reset();

		auto rng = [&]()
		{
			return (m_config.r_max - m_config.r_min) * net->rng.u() + m_config.r_min;
		};

		for(i64 i = 0; i < ls; i++)
		{
			//spike.hist[i] <<= 1u;

			bool fired = false;

			if(m_config.use_sum)
			{
				v[i] += c[i];

				if(v[i] >= 1.0f)
				{
					v[i] -= 1.0f;
					fired = true;
				}
			}
			else if(c[i] >= rng())
			{
				fired = true;
			}

			if(fired)
			{
				spike.fired[i] = true;
				//spike.hist[i] |= 0x1u;

				rec.indices[rec.size] = i;
				//rec.times[rec.size] = time;
				rec.size++;
			}

			spike.update_rates(i);
		}

		spike.update_mask(time);
		last_step = time;
	};

	i64 rng_block_t::bytes() const
	{
		i64 s = 0;
		s += v.bytes() + c.bytes() + tr.bytes();
		s += spike.bytes() + sizeof(rng_block_t);
		return s;
	};

	izh_block_t::izh_block_t(
		network_t* n, i32 id_, dim2 s, n_block_config_t cfg
	) : n_block_t(n,id_,s,cfg), m_config(cfg.izh)
	{
		u.resize(s);
		reset();
	};

	izh_block_t::~izh_block_t()
	{
		__free();
		u.free();
	};

	void izh_block_t::reset()
	{
		__reset();
		if(net != nullptr)
			m_config = net->config.soma.izh;
		v.fill(m_config.c);
		u.fill(m_config.c*m_config.b);
		tr.fill(m_config.tr);
		tr2.fill(m_config.tr2);
	};

	void izh_block_t::step()
	{
		i64 ls = size();
		i32 time = last_step + 1;
		if(net != nullptr)
			time = net->state.time;

		spike.fired.zero();
		spike.value.zero();
		//spike.mask.zero();
		//spike.back.zero();

		rec.reset();

		float min_v = config.min_v;
		float max_v = config.max_v;
		if(!config.use_max_v)
			max_v = 9999.0f;

		const auto mc = m_config;

		for(i64 i = 0; i < ls; i++)
		{
			if(mc.keep_high_v)
				if(spike.last[i] == last_step)
					v[i] = mc.c;

			float nv = v[i] + kernel::izh4_dvdt(v[i],u[i],c[i]);
			//spike.hist[i] <<= 1u;

			if(nv >= tr[i])
			{
				if(!mc.keep_high_v)
					v[i] = mc.c;
				u[i] += mc.d;
				
				spike.fired[i] = true;
				//spike.hist[i] |= 0x1u;

				rec.indices[rec.size] = i;
				//rec.times[rec.size] = time;
				rec.size++;

				if(mc.handle_v_reset)
					u[i] += kernel::izh4_dudt(mc.c,u[i],mc.a,mc.b);
			}
			else
			{
				v[i] = std::clamp<float>(nv,min_v,max_v);
				if(mc.handle_v_reset)
					u[i] += kernel::izh4_dudt(v[i],u[i],mc.a,mc.b);
			}

			//TODO: v may need to have been reset here!
			if(!mc.handle_v_reset)
				u[i] += kernel::izh4_dudt(v[i],u[i],mc.a,mc.b);
			
			spike.update_rates(i);
		}

		if(config.use_current_decay)
			c *= inv_tau_c;

		spike.update_mask(time);
		last_step = time;
	};

	i64 izh_block_t::bytes() const
	{
		i64 s = 0;
		s += v.bytes() + c.bytes() + tr.bytes() + u.bytes();
		s += spike.bytes() + sizeof(izh_block_t);
		return s;
	};

	izhd_block_t::izhd_block_t(
		network_t* n, i32 id_, dim2 s, n_block_config_t cfg
	) : n_block_t(n,id_,s,cfg), m_config(cfg.izhd)
	{
		c.resize(s);
		c2.resize(s);
		v2.resize(s);
		u2.resize(s);
		u.resize(s);
		reset();
		printf("D\n");
	};

	izhd_block_t::~izhd_block_t()
	{
		c.free();
		c2.free();
		v2.free();
		u2.free();
		u.free();
	};

	void izhd_block_t::reset()
	{
		__reset();
		c.zero();
		c2.zero();
		v2.fill(-75.0f);
		v.fill(-75.0f);
		u.fill(-13.0f);
		u2.fill(-18.0f);
	};

	void izhd_block_t::step()
	{
		i64 ls = size();
		i32 time = last_step + 1;
		if(net != nullptr)
			time = net->state.time;

		spike.fired.zero();
		spike.value.zero();
		//spike.mask.zero();

		rec.reset();

		for(i64 i = 0; i < ls; i++)
		{
			//spike.hist[i] <<= 1u;

			float g1 = c[i];
			float g2 = c2[i];

			float vold = v[i];
			float v2old = v2[i];

			v[i] = v[i] + (
				0.04f * v[i] * v[i] + 5.0f * v[i] + 140.0f - u[i] - 
				(g1*v[i]) + (v2old - vold) * m_config.couple / m_config.rho
			);

			v2[i] = v2[i] + (
				0.04f * v2[i] * v2[i] + 5.0f * v2[i] + 140.0f - u2[i] -
				(g2*v2[i]) - (v2old - vold) * m_config.couple
			);

			u2[i] = u2[i] + m_config.dendrites.a * (m_config.dendrites.b * v2[i] - u2[i]);

			if(v[i] >= m_config.soma.tr)
			{
				v[i] = m_config.soma.c;
				u[i] += m_config.dendrites.d;

				spike.fired[i] = true;
				//spike.hist[i] |= 0x1u;

				rec.indices[rec.size] = i;
				//rec.times[rec.size] = time;
				rec.size++;
			}
			else
			{
				v[i] = std::clamp<float>(v[i],
					m_config.soma.min_v,
					m_config.soma.max_v
				);
			}

			if(v2[i] >= m_config.dendrites.tr)
			{
				v2[i] = m_config.dendrites.c;
				u2[i] += m_config.dendrites.d;
			}
			else
			{
				v2[i] = std::clamp<float>(v[i],
					m_config.dendrites.min_v,
					m_config.dendrites.max_v
				);
			}

			spike.update_rates(i);

			if(config.use_current_decay)
				c[i] *= inv_tau_c;
		}

		spike.update_mask(time);
		last_step = time;
	};

	i64 izhd_block_t::bytes() const
	{
		i64 s = 0;
		s += v.bytes() + c.bytes() + tr.bytes() + u.bytes();
		s += spike.bytes() + sizeof(izh_block_t);
		return s;
	};

	raf_block_t::raf_block_t(
		network_t* n, i32 id_, dim2 s, n_block_config_t cfg
	) : n_block_t(n,id_,s,cfg), m_config(cfg.raf)
	{
		v_reset.resize(s).fill(m_config.v_reset);
		f.resize(s).fill(m_config.frequency);
		I.resize(s).zero();
		h.resize(s).zero();
		beta.resize(s).fill(m_config.beta);
		tr.fill(m_config.threshold);
		reset();
	};

	raf_block_t::~raf_block_t()
	{
		__free();
		f.free();
		I.free();
		h.free();
		beta.free();
		v_reset.free();
	};

	void raf_block_t::reset()
	{
		__reset();
		f.fill(m_config.frequency);
		I.zero();
		h.zero();
		beta.fill(m_config.beta);
		tr.fill(m_config.threshold);
	};

	void raf_block_t::step()
	{
		i64 ls = size();
		i32 time = last_step + 1;
		if(net != nullptr)
			time = net->state.time;

		spike.fired.zero();
		spike.value.zero();
		//spike.mask.zero();

		float vr = -m_config.threshold/2.0f;
		if(m_config.use_specific_v_reset)
			vr = m_config.v_reset;

		float beta_ = m_config.beta;
		float freq = m_config.frequency;

		for(i64 i = 0; i < ls; i++)
		{
			/*if(time == 0)
			{
				v[i] = c[i];
				I[i] = c[i];
			}*/

			spike.trace[i] *= inv_tau_t;

			float oi = I[i];
			float ov = v[i];
			
			I[i] = beta_ * oi - (1.0f - beta_) * freq * ov + c[i];
			v[i] = (1.0f - beta_) /** freq*/ * oi + beta_ * ov;

			if(v[i] >= tr[i])
			{
				if(m_config.keep_high_v)
					h[i] = v[i];
				else
					h[i] = vr;
				v[i] = vr;
				I[i] = 0.0f;
				spike.last[i] = time;
				spike.trace[i] += 1.0f;
				spike.fired[i] = true;
				//spike.hist[i] |= 0x1u;
				spike.value[i] = 1.0f;
			}
			else
			{
				h[i] = v[i];
			}

			//spike.update_rates(i);
			
			if(config.use_current_decay)
				c[i] *= inv_tau_c;
		}

		spike.update_mask(time);
		last_step = time;
	};

	void raf_block_t::surrogate()
	{
		const auto &s_cfg = config.surrogate;

		const i32 fn = s_cfg.function;
		for(i64 i = 0; i < size(); i++)
			bp.sd[i] = s_cfg.util.fn(fn,h[i],tr[i]); //uses h instead of v

		s_cfg.adjust.apply(bp.sd);
	};

	i64 raf_block_t::bytes() const
	{
		i64 s = 0;
		s += v.bytes() + c.bytes() + tr.bytes();
		s += v_reset.bytes() + beta.bytes() + f.bytes() + I.bytes();
		s += spike.bytes() + sizeof(raf_block_t);
		return s;
	};
};