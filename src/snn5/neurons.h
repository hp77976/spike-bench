#pragma once
#include "base.h"
#include "conf.h"
#include "../math/array.h"
#include "../math/rng.h"

namespace snn5
{
	struct s_block_t;
	struct network_t;

	struct spike_data_t
	{
		//math::array<float> ema_rate; //this works best for backprop
		//math::array<float> bit_rate; //this works best for selecting the answer
		math::array<float> trace;
		math::array<float> value;
		math::array<float> count;
		math::array<bool>  fired;
		//math::array<bool>  back;
		math::array<i32> last;
		//math::array<u8>  mask;
		//math::array<u64> hist;

		spike_data_config_t config;

		spike_data_t(dim2 shape, spike_data_config_t cfg);

		~spike_data_t();

		void reset();

		void update_rates(i64 i);

		void update_mask(i32 time);

		i64 bytes() const;

		i64 size() const;
	};

	struct spike_record_t
	{
		math::array<i32> indices;
		//math::array<i32> times;
		i32 size = 0;

		spike_record_t() {};

		spike_record_t(dim2 shape)
		{
			indices.resize(shape);
			//times.resize(shape);
			reset();
		};

		void reset()
		{
			//indices.zero();
			//times.zero();
			size = 0;
		};
	};

	struct n_block_t : public block_t
	{
		network_t* net = nullptr;
		std::vector<s_block_t*> prev = {};
		std::vector<s_block_t*> next = {};
		i32 last_step = -1;

		math::array<float> v; //potential
		math::array<float> c; //current
		math::array<float> tr; //treshold
		math::array<float> tr2; //treshold for reverse spikes
		math::array<float> target;

		struct backprop_data_t
		{
			math::array<float> et;
			math::array<float> ef;
			math::array<float> sd;
			math::array<float> err;
			math::array<float> grad;
		} bp;

		spike_data_t spike;
		spike_record_t rec;

		float inv_tau_t = 0.0f;
		float inv_tau_c = 0.0f;

		soma_config_t config;

		protected:
		n_block_t(network_t* net, i32 id, dim2 shape, n_block_config_t cfg);

		public:
		virtual ~n_block_t();
		
		protected:
		void __reset();

		void __free();
		
		public:
		virtual void reset() override = 0;

		virtual void step() = 0;

		virtual void surrogate();

		void calculate_error();

		void update_et_ef();
		
		i64 size() const override {return v.size();};

		dim2 shape() const {return v.shape();};

		i32 block_type() const override {return BLOCK_N;};

		virtual i32 model() const = 0;

		virtual math::array<float> get_c() const {return c;};
	};

	struct lif_block_t : public n_block_t
	{
		float lif_tau_c = 0.0f;

		lif_config_t m_config;

		lif_block_t(network_t* net, i32 id, dim2 shape, n_block_config_t cfg);

		~lif_block_t() override;

		void reset() override;

		void step() override;

		i32 model() const override {return MODEL_LIF;};

		i64 bytes() const override;
	};

	struct rng_block_t : public n_block_t
	{
		rng_config_t m_config;

		rng_block_t(network_t* net, i32 id, dim2 shape, n_block_config_t cfg);

		~rng_block_t() override;
		
		void reset() override;

		void step() override;

		i32 model() const override {return MODEL_RNG;};

		i64 bytes() const override;
	};

	struct izh_block_t : public n_block_t
	{
		tensor<float> u;

		izh_config_t m_config;

		izh_block_t(network_t* net, i32 id, dim2 shape, n_block_config_t cfg);

		~izh_block_t() override;

		void reset() override;

		void step() override;

		i32 model() const override {return MODEL_IZH;};

		i64 bytes() const override;
	};

	struct izhd_block_t : public n_block_t
	{
		tensor<float> c2; //conductance, dendrites
		tensor<float> u;
		tensor<float> u2;
		tensor<float> v2;

		izhd_config_t m_config;

		izhd_block_t(network_t* net, i32 id, dim2 shape, n_block_config_t cfg);

		~izhd_block_t() override;

		void reset() override;

		void step() override;

		i32 model() const override {return MODEL_IZHD;};

		i64 bytes() const override;
	};

	struct raf_block_t : public n_block_t
	{
		tensor<float> v_reset;
		tensor<float> beta; //membrane potential decay rate
		tensor<float> f; //frequency
		tensor<float> I; //state of something? repo uses I
		tensor<float> h; //high value membrane potential for spike

		raf_config_t m_config;

		raf_block_t(network_t* net, i32 id, dim2 shape, n_block_config_t cfg);

		~raf_block_t() override;

		void reset() override;

		void step() override;

		void surrogate() override;

		i32 model() const override {return MODEL_RAF;};

		i64 bytes() const override;
	};
};