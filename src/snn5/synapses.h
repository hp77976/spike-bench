#pragma once
#include "base.h"
#include "conf.h"
#include "../math/conv.h"

namespace snn5
{
	struct n_block_t;
	struct network_t;

	enum s_block_type_e
	{
		S_BLOCK_FULLY_CONNECTED,
		S_BLOCK_CONVOLUTIONAL,
		S_BLOCK_POOLED
	};

	struct s_block_t : public block_t
	{
		network_t* net = nullptr;
		n_block_t* prev = nullptr;
		n_block_t* next = nullptr;

		tensor<float> w;
		tensor<float> wr;
		tensor<float> fb;
		tensor<float> mp;
		tensor<float> et;
		tensor<float> ef;
		tensor<float> ft;

		struct stp_data_t
		{
			tensor<float> u;
			tensor<float> x;
			tensor<float> dw; //gpu only cache (needed because of x_prev)
			float a = 0.0f;
		} stp;

		struct stdp_data_t
		{
			tensor<float> lut;
			i32 offset = 0;
		} stdp;

		float inv_tau_e = 0.0f;
		float inv_tau_f = 0.0f;

		s_block_config_t config;

		i32 last_step = -500;

		void __reset();

		void __free();

		protected:
		s_block_t(
			network_t* net, i32 id,
			n_block_t* prev, n_block_t* next,
			s_block_config_t config
		);

		public:
		virtual ~s_block_t();

		virtual void forward(bool no_input = false);

		virtual void backward();

		virtual void update_weights();

		virtual void update_feedback();

		i64 size() const override {return w.size();};
		
		dim4 shape() const {return w.shape();};

		i32 block_type() const override {return BLOCK_S;};

		virtual i32 s_type() const = 0;

		i64 bytes() const override {return 1;};
	};

	struct fc_s_block_t : public s_block_t
	{
		fc_s_block_t(
			network_t* net, i32 id,
			n_block_t* prev, n_block_t* next,
			s_block_config_t config
		);

		~fc_s_block_t() override;
		
		void reset() override;

		i32 s_type() const override {return S_BLOCK_FULLY_CONNECTED;};
	};

	struct conv_s_block_t : public s_block_t
	{
		//math::conv_kernel_t kernel;

		conv_s_block_t(
			network_t* net, i32 id,
			n_block_t* prev, n_block_t* next,
			s_block_config_t config
		);

		~conv_s_block_t() override;

		void reset() override;

		void forward(bool no_input = false) override;

		void backward() override;

		void update_weights() override;
		
		void update_feedback() override;

		i32 s_type() const override {return S_BLOCK_CONVOLUTIONAL;};
	};
};