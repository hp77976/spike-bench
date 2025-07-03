#include "../snn5/snn.h"
#include "../snn5/builder.h"
#include <functional>
#include <mutex>
#include <thread>
#include <math.h>
#include "../snn5/log.h"
#include "../snn5/ui.h"
#include "../snn5/misc.h"
#include "../math/kernels.h"
#include "../util/misc.h"
#include "../util/mnist3.h"
#include "../util/timer.h"

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

	result.run_timer.stop();
	result.seconds = result.run_timer.get_seconds();

	r->signals.status.is_done.store(1);
	r->signals.status.is_working.store(0);
};

int main()
{
	mnist::load_data("../rsc/mnist/");
	timer ti; ti.start();

	snn5::feedforward_builder_t* ffw_builder = new snn5::feedforward_builder_t();
	snn5::run_config_t run_cfg;
	run_cfg.samples = 60000;

	snn5::feedforward_network_t* net = (snn5::feedforward_network_t*)ffw_builder->create();
	net->load_data(mnist::get_dataset());
	snn5::run_t* r = new snn5::run_t(net,run_cfg);
	r->label = "run";

	__run(r,r->train_results,true);

	ti.stop();
	printf("total time: %f\n",ti.get_seconds());
	mnist::free_data();
};