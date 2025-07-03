#include "snn.h"
#include "../math/kernels.h"
#include "../math/simd/include.h"
#include "../math/food.h"
#include "../util/misc.h"
#include "log.h"
#include "neurons.h"
#include "synapses.h"
#include <chrono>
#include <cstdint>
#include <future>
//#include <cblas.h>
#include <stack>
#include <stdexcept>
#include <thread>

namespace snn5
{
	network_t::~network_t()
	{
		for(i32 i = 0; i < n_blocks.size(); i++)
			delete n_blocks[i];
		for(i32 i = 0; i < s_blocks.size(); i++)
			delete s_blocks[i];
	};

	void network_t::reset()
	{
		for(i32 i = 0; i < n_blocks.size(); i++)
			n_blocks[i]->reset();
		for(i32 i = 0; i < s_blocks.size(); i++)
			s_blocks[i]->reset();

		state.is_training = false;
		state.do_backprop = false;
		state.time = 0;
		rng.seed(123456,7654321);
	};

	void network_t::set_input(math::array<float> rates, i32 id)
	{
		if(id >= n_blocks.size() || id < 0)
			throw std::runtime_error("No n_block with ID: "+std::to_string(id)+"\n");

		n_block_t* nb = n_blocks.at(id);
		if(rates.size() != nb->size())
		{
			printf("Neuron size: %li\n",nb->size());
			printf("Target size: %li\n",rates.size());
			throw std::runtime_error("Target size mismatch!\n");
		}

		const auto &ic = nb->config.input;
		nb->c = (ic.r_max - ic.r_min) * rates + ic.r_min;
	};

	void network_t::set_target(math::array<float> rates, i32 id)
	{
		if(id >= n_blocks.size() || id < 0)
			throw std::runtime_error("No n_block with ID: "+std::to_string(id)+"\n");

		n_block_t* nb = n_blocks.at(id);
		if(rates.size() != nb->size())
		{
			printf("Neuron size: %li\n",nb->size());
			printf("Input size: %li\n",rates.size());
			throw std::runtime_error("Input size mismatch!\n");
		}

		const auto &tc = nb->config.target;
		nb->target = (tc.r_max - tc.r_min) * rates + tc.r_min;
	};

	math::array<float> network_t::get_output(i32 id) const
	{
		if(id >= n_blocks.size() || id < 0)
			throw std::runtime_error("No n_block with ID: "+std::to_string(id)+"\n");

		n_block_t* nb = n_blocks.at(id);

		switch(config.output.rate_selection)
		{
			default:
			//case(0): return nb->spike.ema_rate;
			//case(1): return nb->spike.bit_rate;
			case(2): return nb->spike.trace;
			case(3): return nb->spike.count;
		}
	};

	void network_t::zero_input(i32 id)
	{
		if(id >= n_blocks.size() || id < 0)
			return;

		n_blocks.at(id)->c.zero();
	};

	void network_t::zero_target(i32 id)
	{
		if(id >= n_blocks.size() || id < 0)
			return;

		n_blocks.at(id)->target.zero();
	};

	void network_t::zero_grad(i32 id)
	{
		if(id >= n_blocks.size() || id < 0)
			return;

		n_blocks.at(id)->bp.grad.zero();
	};

	void network_t::zero_count(i32 id)
	{
		if(id >= n_blocks.size() || id < 0)
			return;

		n_blocks.at(id)->spike.count.zero();
	};

	i32 network_t::create_neuron_block(dim2 shape, n_block_config_t cfg)
	{
		n_block_t* nb = nullptr;
		i32 model = cfg.model; //cfg.use_global ? config.soma.model : cfg.model;
		i32 id = n_blocks.size();

		switch(model)
		{
			default:
			case(MODEL_LIF):  nb = new  lif_block_t(this,id,shape,cfg); break;
			case(MODEL_IZH):  nb = new  izh_block_t(this,id,shape,cfg); break;
			case(MODEL_IZHD): nb = new izhd_block_t(this,id,shape,cfg); break;
			case(MODEL_RNG):  nb = new  rng_block_t(this,id,shape,cfg); break;
			case(MODEL_RAF):  nb = new  raf_block_t(this,id,shape,cfg); break;
		}

		nb->id = n_blocks.size();
		n_blocks.push_back(nb);
		return nb->id;
	};

	i32 network_t::create_synapse_block(i32 prev_id, i32 next_id, s_block_config_t cfg)
	{
		if(prev_id >= n_blocks.size() || prev_id < 0)
			throw std::runtime_error("No n_block with ID: "+std::to_string(prev_id)+"\n");
		if(next_id >= n_blocks.size() || next_id < 0)
			throw std::runtime_error("No n_block with ID: "+std::to_string(next_id)+"\n");

		n_block_t* prev = n_blocks.at(prev_id);
		n_block_t* next = n_blocks.at(next_id);
		i32 id = s_blocks.size();

		s_block_t* sb = nullptr;
		switch(cfg.type)
		{
			case(0): sb = new fc_s_block_t(this,id,prev,next,cfg); break;
			case(1): sb = new conv_s_block_t(this,id,prev,next,cfg); break;
		}

		sb->id = s_blocks.size();
		s_blocks.push_back(sb);
		prev->next.push_back(sb);
		next->prev.push_back(sb);
		return sb->id;
	};

	n_block_t* network_t::get_neuron_block(i32 id) const
	{
		if(id >= n_blocks.size() || id < 0)
			return nullptr;
		return n_blocks.at(id);
	};

	s_block_t* network_t::get_synapse_block(i32 id) const
	{
		if(id >= s_blocks.size() || id < 0)
			return nullptr;
		return s_blocks.at(id);
	};
};