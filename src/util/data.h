#pragma once
#include "../math/array.h"
#include "../math/tensor.h"

namespace util
{
	struct sample_t
	{
		math::array<float> input;
		math::array<float> target;
	};

	struct dataset_t
	{
		std::vector<sample_t> train_samples = {};
		std::vector<sample_t> test_samples = {};
		i32 train_index = 0;
		i32 test_index = 0;
	};
};