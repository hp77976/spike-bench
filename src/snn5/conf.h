#pragma once
#include "../math/array.h"
#include "../math/tensor.h"
#include "../math/conv.h"
#include "../math/gradients.h"

namespace snn5
{
	enum neuron_model_e
	{
		MODEL_LIF  = 0, //leaky integrate and fire
		MODEL_IZH  = 1, //izhikevich
		MODEL_IZHD = 2, //izhikevich
		MODEL_RNG  = 3, //random
		MODEL_RAF  = 4, //resonate and fire
		MODEL_BRF  = 5, //balanced resonate and fire
	};

	enum network_type_e
	{
		NET_NONE = 0,
		NET_FEEDFORWARD,
		NET_CUSTOM,
	};
	
	struct learnable_param_config_t
	{
		bool use_global = true;
		float i_min =  -1.0f;
		float i_max =   1.0f;
		float r_min = -15.0f;
		float r_max =  15.0f;
		float learn_rate = 0.001f;
		bool enable_trace = true;
		float trace_decay = 50.0f;
		bool use_xe_init = false;

		learnable_param_config_t() {};

		learnable_param_config_t(bool enable_trace_) {enable_trace = enable_trace_;};

		void draw(bool show_global);
	};

	struct spike_data_config_t
	{
		float ema_alpha = 0.995f;
		float ema_mul = 64.0f;
		float trace_decay = 50.0f;

		void draw();
	};

	struct adjust_t
	{
		bool enable_clamp = false;
		float v_min = -1.0f;
		float v_max =  1.0f;
		float mul = 1.0f;

		void draw();

		void apply(math::array<float> data) const
		{
			if(mul == 0.0f)
			{
				data.zero();
				return;
			}

			if(mul != 1.0f)
				data *= mul;

			if(enable_clamp)
				data.clamp(v_min,v_max);
		};

		void apply(tensor<float> data) const
		{
			if(mul == 0.0f)
			{
				data.zero();
				return;
			}

			if(mul != 1.0f)
				data *= mul;

			if(enable_clamp)
				data.clamp(v_min,v_max);
		};
	};

	struct backprop_config_t
	{
		bool use_global = true;
		bool active = true;
		const char* value_items = "Weight\0Feedback\0";
		i32 value_selection = 1;

		void draw(bool show_global);
	};

	struct gradient_config_t
	{
		bool use_global = true;
		adjust_t adjust;

		void draw(bool show_global);
	};

	struct error_config_t
	{
		bool use_global = true;
		adjust_t adjust;
		const char* rate_items = "EMA\0Bit\0Trace\0Count\0";
		i32 rate_selection = 3;

		void draw(bool show_global);
	};

	struct surrogate_config_t
	{
		bool use_global = true;
		adjust_t adjust;
		const char* input_items = "V\0Rest\0Zero\0";
		i32 input_selection = 0;
		sgrad::util_t util;
		i32 function = 6;

		void draw(bool show_global);
	};

	struct rate_config_t
	{
		bool use_global = true;
		float r_min = 0.0f;
		float r_max = 1.0f;
		bool use_sum = false;

		rate_config_t() {};

		rate_config_t(float min_, float max_)
		{
			r_min = min_;
			r_max = max_;
		};

		void draw(bool show_global);
	};

	struct output_config_t
	{
		bool use_global = true;
		const char* rate_items = "EMA\0Bit\0Trace\0Count\0";
		i32 rate_selection = 3;

		void draw(bool show_global);
	};

	struct metaplasticity_config_t
	{
		bool use_global = true;
		bool active = false;
		bool require_weight_update = true;
		bool use_dynamic_alpha = false;
		bool use_post_spike_for_decay = false;
		float init_m = 0.0f;
		float max_m = 25.0f;
		float pre_th1  = 2.0f;
		float post_th1 = 1.5f;
		float pre_th2  = 1.0f;
		float post_th2 = 1.0f;
		float base_update = 0.05f;
		float base_alpha = 0.005f;
		float decay_alpha = 0.05f;
		float t_ref = 2000.0f;
		float s_i = 1.0f;
		bool use_dm_hack = false;

		void draw(bool show_global);
	};

	struct stp_config_t
	{
		bool use_global = true;
		bool active = false;
		float u = 0.45f;
		float tau_u =  50.0f;
		float tau_x = 750.0f;

		void draw(bool show_global);
	};

	struct stdp_config_t
	{
		bool use_global = true;
		bool active = false;
		bool use_on_weights = true;
		bool use_on_feedback = false;
		float alpha_ltp = 0.0010f;
		float alpha_ltd = 0.0015f;
		float tau_ltp = 20.0f;
		float tau_ltd = 20.0f;
		float scale = 0.001f;
		bool use_lut = true; //3x perf boost
		i32 lut_offset = -255;

		void draw(bool show_global);
	};

	struct soma_config_t
	{
		bool use_current_decay = false;
		float current_decay = 10.0f;

		bool use_max_v = true;
		float min_v = -90.0f;
		float max_v =  35.0f;

		bool use_v_jitter = false;

		bool tau_t_use_exp = true;
		bool tau_c_use_exp = true;

		rate_config_t input = rate_config_t(0.0f,1.0f);
		rate_config_t target = rate_config_t(1.0f,5.0f);
		output_config_t output;
		gradient_config_t gradient;
		surrogate_config_t surrogate;
		error_config_t error;

		void draw(bool show_global);
	};

	struct lif_config_t
	{
		/*
			Cm cell membrane capacity????
			Vm voltage across cell membrane
			Rm membrane resistance
			Vth potential threshold
			I current
			V voltage
			El resist membrance potential / reversal potential of leak

			dV = (-V + Rm * I) / (Rm * Cm)

			updated equilibrium potential = El + Rm * stimulus
			V = uep + (V - uep) * std::exp(-1/tau)
			if(V > Vth)
				V = El
		*/

		float v_rest = -70.0f;
		float v_reset = -65.0f;
		float tr = -50.0f;
		float resistance = 10.0f;
		i32 refractory_period = 2;
		i32 current_decay = 10; //time constant for current decay
		bool keep_high_v = true;

		void draw(bool show_global);
	};

	struct rng_config_t
	{
		float r_min = 0.0f;
		float r_max = 1.0f;
		bool use_sum = false;

		float multiplier = 1.0f;

		void draw(bool show_global);
	};

	struct izh_config_t
	{
		float a =   0.02; //recovery variable
		float b =   0.20; //recovery sensitivity
		float c = -65.00; //resting potential
		float d =   8.00; //reset potential
		float tr = 30.0f; //threshold
		float tr2 = 45.0f; //threshold for reverse spikes

		bool keep_high_v = true;
		bool handle_v_reset = true;

		void draw(bool show_global);
	};

	struct izhd_config_t
	{
		struct soma_params_t
		{
			float c = -65.00; //resting potential
			float tr = 30.0f; //threshold

			float min_v = -90.0f;
			float max_v =  35.0f;
		} soma;

		struct dendrite_params_t
		{
			float a = 0.0025f;
			float b = 0.01f;
			float c = -55.0f;
			float d = 1.0f;
			float tr = 0.0f;

			float min_v = -60.0f;
			float max_v =  10.0f;
		} dendrites;

		float rho = 1.0f; //asymmetry factor
		float couple = 0.325f;

		bool keep_high_v = true;
		bool handle_v_reset = true;

		void draw(bool show_global);
	};

	struct raf_config_t
	{
		bool use_specific_v_reset = true;
		float v_reset = 0.0f;
		float threshold = 30.0f;
		float beta = 0.99f;
		float frequency = 30.0f;
		bool keep_high_v = true;

		void draw(bool show_global);
	};

	struct braf_config_t
	{
		float refractory_period = 1.0f;
		float beta = 1.0f;
		float theta = 1.0f;
	};

	struct s_block_config_t
	{
		bool draw_window = true;

		learnable_param_config_t weight = learnable_param_config_t(true);
		learnable_param_config_t feedback = learnable_param_config_t(false);
		
		backprop_config_t backprop;
		stp_config_t stp;
		stdp_config_t stdp;
		metaplasticity_config_t metaplasticity;

		const char* type_items = "FC\0Conv\0";
		i32 type = 0;

		math::conv_kernel_t kernel;

		void draw(i32 id, bool in_window = true);
	};

	struct n_block_config_t
	{
		bool draw_window = true;
		bool use_global = true;
		soma_config_t n_config;
		lif_config_t lif;
		rng_config_t rng;
		izh_config_t izh;
		izhd_config_t izhd;
		raf_config_t raf;
		
		const char* model_items = "LIF\0Izh\0Izhd\0RNG\0RAF\0BRF\0";
		i32 model = 1;

		spike_data_config_t spike;

		n_block_config_t() {};

		n_block_config_t(neuron_model_e model_) : model(model_) {};

		void draw(bool show_global);
	};

	struct network_config_t
	{
		struct soma_config_t : public n_block_config_t
		{
			
		} soma;

		struct apical_config_t : public n_block_config_t
		{
			bool active = false;
		} apical;

		learnable_param_config_t weight;
		learnable_param_config_t feedback;

		error_config_t error;
		surrogate_config_t surrogate;
		gradient_config_t gradient;
		backprop_config_t backprop;

		struct dendrite_config_t
		{
			bool active = false;
			bool use_apical_e = false;
			bool use_apical_sd = false;
			bool update_y = false;
		} dendrites;

		metaplasticity_config_t metaplasticity;

		struct dumb_metaplasticity_config_t
		{
			bool active = false;
			bool use_target_synapse_selection = false;
		} dmp;

		struct debug_config_t
		{
			struct cpu_config_t
			{
				bool enable_blas_current = true;
				bool enable_dot_mask = true;
				bool use_safe_izh = false;
			} cpu;

			struct gpu_config_t
			{
				bool use_cpu_step_drivers = false;
				bool use_cpu_prop_spikes = false;
				bool use_cpu_calc_out_err = false;
				bool use_cpu_prop_err = false;
				bool use_cpu_update_weights = false;
				bool use_cpu_set_rate = false;
				bool use_cpu_elibility = false;

				i32 block_size = 256;
				i32 grid_x = 16;
				i32 grid_y = 16;
			} gpu;

			struct q_log_config_t
			{
				bool active = false;
				i32 size = 64;
			} ql;
		} debug;

		struct rate_backprop_config_t
		{
			bool active = false;
			float tau = 1.0f;
			bool tau_use_sigmoid = false;
			bool manual_tau = false;
			bool hard_reset = true;
			bool detach = false;
		} rate_backprop;

		bool output_no_spike = false;

		bool use_gpu = false;
		bool soma_on_gpu = false;
		bool use_np_layout = false;
		bool use_ema_selection = false;

		bool enable_reverse_spikes = false;
		bool use_event_stdp = false;
		bool accumulate_grad = false;

		rate_config_t input;
		rate_config_t target = {1.0f,5.0f};
		output_config_t output;

		void draw();

		void set(std::string param, float value);
	};
};