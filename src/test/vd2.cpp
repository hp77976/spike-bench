#include "../snn5/snn.h"
#include "../snn5/builder.h"
#include <mutex>
//#include <cblas.h>
#include <math.h>
#include "../rlimgui/rlImGui.h"
#include "../implot/implot.h"
#include "../snn5/log.h"
#include "../snn5/ui.h"
#include "../snn5/misc.h"
#include "../math/kernels.h"
#include "../util/misc.h"
#include "../util/mnist3.h"
#include "../util/timer.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <queue>

void __run(snn5::run_t* r, snn5::run_t::result_t &result, bool training)
{
	r->signals.status.is_working.store(1);
	result.score_index = 0;
	result.scores.resize(100);
	for(uint32_t i = 0; i < result.scores.size(); i++)
		result.scores[i] = 0.0f;

	if(r->run_config.use_mini_test)
		r->test_results.scores.resize(10);
	for(uint32_t i = 0; i < r->test_results.scores.size(); i++)
		r->test_results.scores.at(i) = 0.0f;

	uint32_t group_size = r->run_config.samples / 10;

	uint32_t total_items = r->run_config.samples;
	
	uint32_t correct = 0;
	uint32_t item_count = 0;
	uint32_t score_index = 0;

	auto update_score = [&]()
	{
		if(r->__step(training))
			correct++;

		item_count++;
		result.live_score = (float)correct / (float)item_count;

		float pos = ((float)score_index)/((float)total_items)*100.0f;
		uint32_t ii = std::clamp<uint32_t>(pos,0,99);
		result.scores[ii] = result.live_score;
		result.score_index = ii;
		score_index++;
	};

	uint32_t mt_correct = 0;
	uint32_t mt_item_count = 0;
	uint32_t mt_score_index = 0;

	auto update_mini_score = [&](uint32_t ic, uint32_t pos)
	{
		if(r->__step(false))
			mt_correct++;

		mt_item_count++;
		r->test_results.live_score = (float)mt_correct / (float)mt_item_count;

		r->test_results.scores[pos] = r->test_results.live_score;
		r->test_results.score_index = pos;
		mt_score_index++;
	};

	if(!training)
	{
		int32_t stop_signal = 0;
		for(uint32_t mi = 0; mi < 10000; mi++)
		{
			stop_signal = r->signals.cmd.stop.load();
			if(stop_signal == 1)
				break;

			update_score();
		}

		if(stop_signal != 1)
			result.completed = true;
	}
	else
	{
		result.completed = false;
		total_items *= r->run_config.epochs;
		result.seconds = -1.0f;
		result.run_timer.start();

		u32 total_samples = r->run_config.epochs * r->run_config.samples;
		u32 ts10 = total_samples / 10;

		uint32_t offset = 0;
		uint32_t mto = 0; //mini test offset
		u32 tsi = 0; //total sample / 10 index
		u32 mti = 0; //mini test index

		for(uint32_t ei = 0; ei < r->run_config.epochs; ei++)
		{
			if(r->run_config.sequential)
			{
				for(uint32_t ni = 0; ni < 10; ni++) //number index
				{
					for(uint32_t gi = 0; gi < group_size; gi++)
					{
						if(r->signals.cmd.stop.load() == 1)
							break;

						update_score();
					}
				}
			}
			else
			{	
				for(uint32_t mi = 0; mi < r->run_config.samples; mi++)
				{
					if(r->signals.cmd.stop.load() == 1)
						break;

					update_score();
					offset++;
					if(offset >= 60000)
						offset = 0;

					tsi++;

					if(tsi >= ts10 && r->run_config.use_mini_test)
					{
						mt_correct = 0;
						mt_item_count = 0;
						training = false;

						uint32_t mts = r->run_config.mini_test_size;
						if(mti < 9)
							mts = 500;
						if(mti == 9)
							mts = 10000;
						for(uint32_t mi = 0; mi < mts; mi++)
						{
							if(r->signals.cmd.stop.load() == 1)
								break;

							update_mini_score(mts,mti);
							mto++;
							if(mto >= 10000)
								mto = 0;
						}
						training = true;
						mti++;
						tsi = 0;
					}
				}
			}
		}

		result.run_timer.stop();
		result.seconds = result.run_timer.get_seconds();
	}

	math::tensor w = r->net->net->s_blocks[0]->w;
	printf(
		"w: %li %li %li %li %li\n",
		w.size(),w.x(),w.y(),w.z(),w.w()
	);
	math::array v = r->net->net->n_blocks[0]->v;
	printf(
		"v: %li %li %li\n",
		v.size(),v.x(),v.y()
	);

	r->signals.status.is_done.store(1);
	r->signals.status.is_working.store(0);
};

void __dispatch(snn5::run_t* r, bool training)
{
	printf("dispatched: %s\n",r->label.c_str());
	if(training)
		r->jt = std::jthread([r](){__run(r,r->train_results,true);});
	else
		r->jt = std::jthread([r](){__run(r,r->test_results,false);});
};

struct controller_t
{
	std::vector<snn5::network_builder_t*> builders = {};
	snn5::network_builder_t* net_builder = nullptr;
	i32 builder_index = 0;

	snn5::run_manager_t* run_manager = nullptr;
	std::vector<snn5::run_t*> background_runs = {};
	snn5::network_manager_t* net_manager = nullptr;

	std::jthread cleanup_thread;
	std::atomic<i32> should_exit = 0;

	umap<std::string,snn5::run_group_manager_t*> group_managers = {};
	
	bool show_primary_run = false;
	snn5::run_t* primary_run = nullptr;
	std::string temp_label;

	bool show_mts = false;

	std::jthread load_thread;

	controller_t()
	{
		if(net_manager == nullptr)
			net_manager = new snn5::network_manager_t();
		run_manager = new snn5::run_manager_t();

		builders.push_back(new snn5::feedforward_builder_t());
		builders.push_back(new snn5::teacher_builder_t());
		net_builder = builders[builder_index];

		temp_label.resize(32);

		cleanup_thread = std::jthread([this](){__cleanup();});
	};

	~controller_t()
	{
		shutdown();
	};

	void shutdown()
	{
		delete_all_jobs();
		while(!run_manager->is_empty())
		{

		}
		if(primary_run != nullptr)
		{
			primary_run->signals.cmd.stop.store(1);
			while(primary_run->signals.status.is_done.load() != 1)
			{

			}
		}
		for(auto i = group_managers.begin(); i != group_managers.end(); i++)
		{
			while(!(*i).second->run_manager->is_empty())
			{

			}
		}
		//delete run_manager; //TODO: this breaks things but NEEDS to be freed!
		//delete net_manager; //TODO: this breaks things but NEEDS to be freed!
	};

	void __cleanup()
	{
		while(true)
		{
			bool do_exit = should_exit.load() == 1;

			if(do_exit)
				delete_all_jobs();

			if(!run_manager->is_empty())
			{
				run_manager->cleanup_pass();
				continue;
			}
			
			for(auto i = group_managers.begin(); i != group_managers.end(); i++)
			{
				if(!(*i).second->run_manager->is_empty())
				{
					(*i).second->run_manager->cleanup_pass();
					continue;
				}
			}

			if(do_exit)
			{
				break;
			}
		}
	};

	void stop_job(snn5::run_t* r)
	{
		r->signals.cmd.stop.store(1);
	};

	void stop_all_jobs()
	{
		for(i32 i = 0; i < background_runs.size(); i++)
			stop_job(background_runs.at(i));
		for(auto i = group_managers.begin(); i != group_managers.end(); i++)
			for(u32 j = 0; j < (*i).second->groups.size(); j++)
				for(u32 k = 0; k < (*i).second->groups[j]->runs.size(); k++)
					stop_job((*i).second->groups[j]->runs.at(k));
	};

	void delete_job(snn5::run_t* r)
	{
		for(auto i = background_runs.begin(); i != background_runs.end(); i++)
		{
			if((*i) == r)
			{
				background_runs.erase(i);
				run_manager->delete_run(r);
				break;
			}
		}
	};

	void delete_all_jobs()
	{
		stop_all_jobs();

		for(u32 i = 0; i < background_runs.size(); i++)
			run_manager->delete_run(background_runs.at(i));
		background_runs.clear();

		for(auto i = group_managers.begin(); i != group_managers.end(); i++)
			while((*i).second->groups.size() > 0)
				(*i).second->delete_group((*i).second->groups.back());
	};

	snn5::run_t* dispatch_job(std::string label, bool add_to_background_runs = true)
	{
		snn5::feedforward_network_t* n = (snn5::feedforward_network_t*)net_builder->create();
		n->load_data(mnist::get_dataset());
		snn5::run_t* r = new snn5::run_t(n,net_manager->run_config);
		r->label = label;
		if(add_to_background_runs)
			background_runs.push_back(r);
		__dispatch(r,true);
		return r;
	};
};

int main(int argv, const char** argc)
{
	std::string base_path = "../rsc/mnist/";
	mnist::load_data(base_path);

	controller_t* ctrl = new controller_t;
	ctrl->net_manager->run_config.samples = 5;
	ctrl->net_manager->run_config.p_steps = 1;
	ctrl->net_manager->run_config.v_steps = 2;
	ctrl->net_manager->run_config.b_steps = 1;
	
	//snn5::feedforward_builder_t ffwd_builder;
	ctrl->dispatch_job("wew");

	delete ctrl; ctrl = nullptr;
	mnist::free_data();

	return 0;
};