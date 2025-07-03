#pragma once
#include "conf.h"
#include "snn.h"
#include "log.h"
#include "wrappers.h"
#include "../util/timer.h"
#include "synapses.h"
#include <functional>
#include <thread>
#include <queue>
#include <mutex>

namespace snn5
{
	struct run_config_t
	{
		i32 epochs  =    1;
		i32 samples = 1000;
		i32 p_steps =   10;
		i32 v_steps =   69;
		//i32 t_steps =    0;
		i32 b_steps =    1;
		i32 mini_test_size = 100;

		//bool use_gpu = false;
		bool sequential = false;
		bool use_target_input = false;
		bool auto_test = false;
		bool use_mini_test = true;
		//bool backprop_gets_input = false;
		//bool backprop_max = false;
		bool keep_logs = false;

		void draw();
	};

	struct network_manager_t
	{
		run_config_t run_config;
		network_config_t net_config;
		std::vector<s_block_config_t> layer_configs = {{},{},{}};
		std::vector<n_block_config_t> n_block_configs = {{},{},{}};
		std::vector<i32> layer_sizes = {500,100,10};
		int32_t layer_count = 3;
		int32_t input_size = 784;

		void draw();
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
		};

		struct result_t
		{
			std::vector<float> scores = {};
			u32 score_index = 0;
			float seconds = 0.0f;
			float live_score = 0.0f;
			bool completed = false;
			timer run_timer;

			void init()
			{
				scores.resize(100);
				for(u32 i = 0; i < 100; i++)
					scores.at(i) = 0.0f;
				score_index = 0;
				seconds = 0.0f;
				live_score = -1.0f;
				completed = false;
			};
		};

		std::vector<n_log_t*> n_logs = {};
		std::vector<s_log_t*> s_logs = {};

		network_wrapper_t* net = nullptr;
		signals_t signals;
		run_config_t run_config;

		result_t train_results;
		result_t test_results;

		std::string label;

		std::jthread jt;

		bool show_logs = false;

		bool __step(bool training);

		run_t(snn5::network_wrapper_t* n, run_config_t c)
		{
			net = n;
			run_config = c;
			label.resize(32);

			if(run_config.keep_logs)
			{
				for(i32 i = 0; i < n->net->n_blocks.size(); i++)
					n_logs.push_back(new n_log_t(n->net->n_blocks[i]));
				for(i32 i = 0; i < n->net->s_blocks.size(); i++)
					s_logs.push_back(new s_log_t(n->net->s_blocks[i]));
			}
		};

		void reattach_logs()
		{
			if(run_config.keep_logs)
			{
				for(i32 i = 0; i < net->net->n_blocks.size(); i++)
					n_logs[i]->nb = net->net->n_blocks[i];
				for(i32 i = 0; i < net->net->s_blocks.size(); i++)
					s_logs[i]->sb = net->net->s_blocks[i];
			}
		};

		/*~run_t()
		{
			net->free();
			delete net;
		};*/

		void free()
		{
			//net->free();
			delete net;
			for(i32 i = 0; i < n_logs.size(); i++)
				delete n_logs[i];
			for(i32 i = 0; i < s_logs.size(); i++)
				delete s_logs[i];
		};
	};

	struct run_manager_t
	{
		uset<run_t*> runs = {};
		std::vector<run_t*> trash_runs = {};
		std::mutex trash_mutex;

		~run_manager_t()
		{
			/*while(!is_empty())
				cleanup_pass();*/
		};

		void add_run(run_t* run)
		{
			runs.insert(run);
		};

		void delete_run(run_t* run)
		{
			run->signals.cmd.stop.store(1);
			{
				std::unique_lock lock(trash_mutex);
				runs.erase(run);
				trash_runs.push_back(run);
			}
		};

		void cleanup_pass()
		{
			{
				std::unique_lock lock(trash_mutex);
				for(auto i = trash_runs.begin(); i != trash_runs.end(); i++)
				{
					if((*i)->signals.status.is_done.load() == 1)
					{
						run_t* r = (*i);
						r->free();
						delete r;
						trash_runs.erase(i);
						break;
					}
				}
			}
		};

		bool is_empty()
		{
			bool empty = false;
			{
				std::unique_lock lock(trash_mutex);
				empty = trash_runs.empty();
			}
			return empty;
		};
	};

	struct run_group_t
	{
		std::vector<run_t*> runs = {};
		std::vector<double> params = {};
		std::vector<double> scores = {};
		std::string label;

		run_group_t(std::string l)
		{
			label = l;
		};

		void add_run(run_t* r, float p)
		{
			runs.push_back(r);
			params.push_back(p);
			scores.push_back(0.0f);
		};

		void update_scores()
		{
			scores.resize(runs.size());
			for(u32 i = 0; i < runs.size(); i++)
				scores.at(i) = runs.at(i)->test_results.live_score;
		};

		bool all_done() const
		{
			for(u32 i = 0; i < runs.size(); i++)
				if(runs.at(i)->signals.status.is_done.load() == 0)
					return false;
			return true;
		};

		void stop()
		{
			for(u32 i = 0; i < runs.size(); i++)
				runs.at(i)->signals.cmd.stop.store(1);
		};
	};

	struct run_group_manager_t
	{
		run_manager_t* run_manager = nullptr;
		std::vector<run_group_t*> groups = {};
		std::string temp_label;
		std::function<void()> fn = {};
		std::string title;

		run_group_manager_t()
		{
			run_manager = new run_manager_t();
			temp_label.resize(32);
		};

		~run_group_manager_t()
		{
			while(groups.size() > 0)
				delete_group(groups.back());
			delete run_manager;
		};

		void add_group(run_group_t* group)
		{
			groups.push_back(group);
			for(u32 i = 0; i < group->runs.size(); i++)
				run_manager->add_run(group->runs.at(i));
		};

		void delete_group(run_group_t* group)
		{
			for(u32 i = 0; i < group->runs.size(); i++)
				run_manager->delete_run(group->runs.at(i));
			for(auto i = groups.begin(); i != groups.end(); i++)
			{
				if((*i) == group)
				{
					groups.erase(i);
					break;
				}
			}
			delete group;
		};
	};

	struct run_graph_t
	{
		std::vector<run_t*> trash_runs = {};
		std::vector<run_t*> runs = {};

		const char* result_items = "Train\0Mini Test\0Full Test\0";
		i32 result_selection = 0;

		i32 graph_height = 200;
		//i32 graph_width = 100;

		run_graph_t();
		
		~run_graph_t();

		void draw();
	};
};