#pragma once
#include "conf.h"
#include "layout.h"

namespace snn5
{
	struct network_wrapper_t;

	struct network_builder_t
	{
		net_layout_t layout;
		std::string label;

		network_builder_t();

		virtual ~network_builder_t();

		virtual void draw() = 0;

		virtual network_wrapper_t* create() const = 0;

		virtual void load_config(network_wrapper_t* nw) = 0;
	};

	struct feedforward_builder_t : public network_builder_t
	{
		feedforward_builder_t();

		void draw() override;

		network_wrapper_t* create() const override;

		void load_config(network_wrapper_t* nw) override;
	};

	struct teacher_builder_t : public network_builder_t
	{
		bool broadcast_to_output = true;
		n_entry_t broadcast_n;
		s_entry_t broadcast_s;

		teacher_builder_t();

		void draw() override;

		network_wrapper_t* create() const override;

		void load_config(network_wrapper_t* nw) override;
	};

	struct mux_builder_t : public network_builder_t
	{
		i32 input_mul = 2;
		i32 output_mul = 2;
		i32 output_mode = 0;

		mux_builder_t();

		void draw() override;

		network_wrapper_t* create() const override;

		void load_config(network_wrapper_t* nw) override;
	};

	/*struct builder_manager_t
	{
		std::vector<network_builder_t*> builders = {};
		std::vector<std::string> builder_names = {};

		builder_manager_t();

		~builder_manager_t();

		void draw();
	};*/
};

/*

	n_in:  0,1
	n_mid: 2
	n_out: 3,4

	s_in:  0,1
	s_mid: ---
	s_out: 2,3

*/