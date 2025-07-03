#pragma once
#include "../math/array.h"
#include "../math/gradients.h"
#include "../math/rng.h"
#include "../util/mnist3.h"
#include "conf.h"
#include <functional>
#include <thread>
#include <unordered_set>

namespace snn5
{
	CPU_GPU inline u32 calc_blocks(u32 total_size, u32 block_size)
	{
		return ((total_size + block_size - 1) / block_size);
	};

	CPU_GPU inline dim2 calc_blocks(dim2 shape, u32 block_size)
	{
		dim2 s = {1,1};
		s.x = ((shape.x + block_size - 1) / block_size);
		s.y = ((shape.y + block_size - 1) / block_size);
		return s;
	};

	CPU_GPU inline dim4 calc_blocks(dim4 shape, u32 block_size)
	{
		dim4 s;
		for(i32 i = 0; i < 4; i++)
			s[i] = ((shape[i] + block_size - 1) / block_size);
		return s;
	};

	CPU_GPU inline static u32 get_set_bits(u64 i)
	{
		i = i - ((i >> 1) & 0x5555555555555555UL);
		i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
		return (int)((((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56);
	};

	struct n_block_t;
	struct s_block_t;

	struct network_t
	{
		std::vector<n_block_t*> n_blocks = {};
		std::vector<s_block_t*> s_blocks = {};

		network_config_t config;

		struct state_info_t
		{
			bool is_training = false;
			bool do_backprop = false;
			i32 time = 0;
		} state;

		struct allocation_config_t
		{
			i8 proc = math::ARRAY_CPU;
		} alloc;

		rng_t rng;

		network_t(i8 proc = math::ARRAY_CPU) {alloc.proc = proc;};

		~network_t();

		void reset();

		void set_input(math::array<float> rates, i32 id);

		void set_target(math::array<float> rates, i32 id);

		math::array<float> get_output(i32 id) const;

		void zero_input(i32 id);

		void zero_target(i32 id);

		void zero_grad(i32 id);

		void zero_count(i32 id);

		n_block_t* get_neuron_block(i32 id) const;

		s_block_t* get_synapse_block(i32 id) const;

		i32 create_neuron_block(dim2 shape, n_block_config_t config = {});

		i32 create_synapse_block(i32 prev_id, i32 next_id, s_block_config_t config = {});

		void set_training(bool training) {state.is_training = training;};

		void set_backprop(bool backprop) {state.do_backprop = backprop;};

		void enable_training() {state.is_training = true;};

		void disable_training() {state.is_training = false;};

		void enable_backprop() {state.do_backprop = true;};

		void disable_backprop() {state.do_backprop = false;};

		i32 get_time() const {return state.time;};
	};
};