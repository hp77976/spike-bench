#include "../snn5/snn.h"
#include "../snn5/ui.h"
#include <stdexcept>

u32 samples = 2000;
u32 p_steps = 2;
u32 v_steps = 7;
u32 b_steps =  1;
u32 input_size = 784;

void execute(snn5::network_manager_t* nm)
{
	nm->net_config.use_gpu = true;
	snn5::network_t* net1 = nm->create_network(); net1->init(math::ARRAY_UNI);

	rng_t rng; rng.seed(123456,123456);

	math::array<float> input; input.resize(input_size).randomize(rng);
	math::array<float> target; target.resize(net1->layers.back()->size).randomize(rng);
	
	net1->set_input(input);

	auto run_nets = [&](bool do_bp)
	{		
		net1->step(do_bp,true);
	};

	for(u32 x = 0; x < samples; x++)
	{
		net1->input.rate.zero();
		net1->target.rate.zero();

		for(u32 i = 0; i < p_steps; i++)
			run_nets(false);

		net1->set_input(input);

		for(u32 i = 0; i < v_steps; i++)
			run_nets(false);

		net1->set_target(target);

		for(u32 i = 0; i < b_steps; i++)
			run_nets(true);

		input.randomize(rng);
		target.randomize(rng);
	}

	net1->free();
};

int main()
{
	snn5::network_manager_t* nm = new snn5::network_manager_t();
	nm->net_config.debug.ql.size = samples * p_steps * v_steps * b_steps;
	nm->layer_sizes = {500,10};
	nm->input_size = input_size;
	
	nm->layer_configs.at(0).stdp.active = true;
	nm->net_config.metaplasticity.active = true;
	
	execute(nm);
	
	return 0;
};