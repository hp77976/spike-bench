#pragma once
#include "conf.h"
#include "snn.h"
#include "layout.h"
#include "../math/tensor.h"

namespace snn5
{
	struct network_wrapper_t
	{
		util::dataset_t data;
		network_t* net = nullptr;

		virtual ~network_wrapper_t();

		math::array<float> temp_output;

		bool is_set_up = false;

		void reset();

		virtual void step(bool is_training, bool do_backprop) = 0;

		virtual void set_inputs(bool is_training) = 0;

		virtual void set_targets(bool is_training) = 0;

		void zero_inputs();

		void zero_targets();

		void zero_grad();

		void zero_counts();

		void next_data(bool is_training)
		{
			if(is_training)
				data.train_index++;
			else
				data.test_index++;

			if(data.train_index >= data.train_samples.size())
				data.train_index = 0;

			if(data.test_index >= data.test_samples.size())
				data.test_index = 0;
		};

		void load_data(util::dataset_t data_)
		{
			data = data_;
			data.train_index = 0;
			data.test_index = 0;
		};

		virtual math::array<float> get_output() const = 0;
		
		virtual math::array<float> get_target() const = 0;
	};

	struct custom_network_t : public network_wrapper_t
	{
		std::function<void()> run_fn = {};
		std::function<void()> load_input_fn = {};
		std::function<void()> load_target_fn = {};

		~custom_network_t();

		void step(bool is_training, bool do_backprop) override;

		math::array<float> get_output() const override;

		math::array<float> get_target() const override;

		void set_inputs(bool is_training) override {load_input_fn();};

		void set_targets(bool is_training) override {load_target_fn();};
	};

	/*
		simple feedforward wrapper to simplify management of the base network system
	*/
	struct feedforward_network_t : public network_wrapper_t
	{
		std::vector<dim2> sizes = {};
		i32 input_id = -1;
		i32 target_id = -1;

		feedforward_network_t(net_layout_t layout);

		~feedforward_network_t();

		void step(bool is_training, bool do_backprop) override;

		math::array<float> get_output() const override;

		math::array<float> get_target() const override;

		void set_inputs(bool is_training) override;

		void set_targets(bool is_training) override;
	};

	struct teacher_network_t : public network_wrapper_t
	{
		std::vector<dim2> sizes = {};
		i32 input_id = -1;
		i32 target_id = -1;
		i32 output_id = -1;

		teacher_network_t(
			net_layout_t ffw_layout,
			n_entry_t broadcast_n,
			s_entry_t broadcast_s,
			bool broadcast_to_output
		);

		~teacher_network_t();

		void step(bool is_training, bool do_backprop) override;

		math::array<float> get_output() const override;

		math::array<float> get_target() const override;

		void set_inputs(bool is_training) override;

		void set_targets(bool is_training) override;
	};

	struct mux_network_t : public network_wrapper_t
	{
		std::vector<dim2> sizes = {};
		std::vector<i32> in_n_ids = {};
		std::vector<i32> mid_n_ids = {};
		std::vector<i32> out_n_ids = {};
		std::vector<s_block_t*> in_s = {};
		std::vector<s_block_t*> mid_s = {};
		std::vector<s_block_t*> out_s = {};
		i32 output_mode = -1;

		mux_network_t(
			net_layout_t layout,
			i32 input_mul,
			i32 output_mul,
			i32 output_mode
		);

		~mux_network_t();

		void step(bool is_training, bool do_backprop) override;

		math::array<float> get_output() const override;

		math::array<float> get_target() const override;

		void set_inputs(bool is_training) override;

		void set_targets(bool is_training) override;
	};
};