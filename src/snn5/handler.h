#pragma once
#include "snn.h"
#include "conf.h"
#include "../util/timer.h"
#include "wrappers.h"

namespace snn5
{
	struct run_config_t
	{
		i32 epochs  =    1;
		i32 samples = 1000;
		i32 p_steps =   10;
		i32 v_steps =   69;
		i32 t_steps =    0;
		i32 b_steps =    1;
		i32 mini_test_size = 100;

		bool use_gpu = false;
		bool sequential = false;
		bool use_target_input = false;
		bool auto_test = false;
		bool use_mini_test = true;
		bool keep_logs = false;

		void draw();
	};

	struct score_t
	{
		math::array<float> adjusted_score;
		math::array<float> score;
		float live_score = 0.0f;
		util::timer_t timer;
		bool completed = false;
		float seconds = 0.0f;
	};

	struct run_control_t
	{
		bool stop = false;
		bool del = false;
		bool test = false;
		bool load_config = false;
	};

	struct run_t
	{
		struct signals_t
		{
			struct commands_t
			{
				std::atomic<i32> stop = 0;	
			} cmd;

			struct status_t
			{
				std::atomic<i32> is_done = 0;
				std::atomic<i32> is_working = 0;
			} status;
		} signals;

		network_wrapper_t* wrapper = nullptr;
		score_t train_score;
		score_t mini_test_score;
		score_t full_test_score;
		run_config_t run_config;
		std::jthread jt;
		std::string label;

		void draw(run_control_t &r_ctrl);

		void fit_scores(i32 size) {};

		run_t(snn5::network_wrapper_t* w, run_config_t rc)
		{
			wrapper = w;
			run_config = rc;
		};

		~run_t()
		{
			delete wrapper;
		};
	};

	struct run_panel_t
	{
		std::mutex trash_mutex;
		std::vector<run_t*> trash_runs = {};
		std::vector<run_t*> runs = {};

		std::function<run_t*()> create_run_fn = {};

		const char* result_items = "Train\0Mini\0Test\0";
		i32 result_selection = 0;

		i32 graph_height = 200;
		i32 graph_width  = 500;

		std::string temp_label;

		run_panel_t() {};

		~run_panel_t();

		void draw(i32 id = 0);

		void add_run(run_t* run);

		void delete_run(run_t* run);
	};

	/*struct net_layout_t
	{
		network_t* net = nullptr;
		std::function<void()> step_fn;
		std::function<math::array<float>()> get_output_fn;
	};

	struct net_wrapper_t
	{
		network_t* net = nullptr;
		std::function<void()> step_fn;
		std::function<math::array<float>()> get_output_fn;

		void step(bool is_training, bool do_backprop);

		void zero_drivers();

		void set_inputs(bool is_training);

		void set_targets(bool is_training);

		void zero_inputs();

		void zero_targets();

		math::array<float> get_output() const;

		void step_data();
	};

	struct net_builder_t
	{
		network_config_t net_config;
		std::vector<s_block_config_t> s_block_configs = {};
		std::vector<n_block_config_t> n_block_configs = {};

		virtual void step(network_t* net) = 0;

		virtual void draw();
	};*/
};