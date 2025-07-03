#pragma once
#include "../math/array.h"
#include "../common.h"
#include "data.h"
#include "misc.h"
#include <stdexcept>

#ifdef BUILD_UI
#include <raylib.h>
#include <imgui.h>
#endif

namespace mnist
{
	struct mchar8_t
	{
		math::array<uint8_t> data;
		int32_t number = -1;

		mchar8_t() {data.resize({28,28}).zero();};

		~mchar8_t() {};

		void free() {data.free();};

		void draw(int32_t x, int32_t y, float mul = 1.0f);
	};

	struct mchar32_t
	{
		math::array<float> data;
		math::array<float> target;
		int32_t number = -1;

		mchar32_t() {data.resize({28,28}).zero();};

		mchar32_t(const mchar8_t &mc)
		{
			data.resize({28,28});
			for(uint32_t i = 0; i < data.size(); i++)
			{
				data.at(i) = (float)mc.data.at(i) / 255.0f;
				
				//this is specifically here to try to catch a valgrind error early
				if(data.at(i) > 2.0f)
					throw std::runtime_error("Pixel too bright!\n");
			}
			
			number = mc.number;

			target.resize(10).zero();
			target[number] = 1.0f;
		};

		~mchar32_t() {};

		void free() {data.free(); target.free();};

		void draw(int32_t x, int32_t y);
	};

	struct mchar8_set_t
	{
		std::vector<mchar8_t> chars = {};

		mchar8_set_t(const std::string &label_path, const std::string &image_path);

		~mchar8_set_t()
		{
			for(uint32_t i = 0; i < chars.size(); i++)
				chars.at(i).free();
		};
	};

	struct mchar32_set_t
	{
		std::vector<mchar32_t> chars = {};
		umap<int32_t,std::vector<mchar32_t>> chars_by_number = {};
		std::vector<mchar32_t> even_chars = {};
		std::vector<mchar32_t> odd_chars = {};
		std::vector<mchar32_t> chars_0_4 = {};
		std::vector<mchar32_t> chars_5_9 = {};

		mchar32_set_t(const std::string &label_path, const std::string &image_path);

		~mchar32_set_t()
		{
			for(uint32_t i = 0; i < chars.size(); i++)
				chars.at(i).free();
		};
	};

	struct char_viewer_t
	{
		int32_t char_index = 0;
		int32_t pos_x = 700;
		int32_t pos_y = 800;
		float mul = 1.0f;
		bool use_set8 = false;

		void draw(mnist::mchar8_set_t* train_set8, mnist::mchar32_set_t* train_set);

		void draw();
	};

	void load_data(std::string path);

	void free_data();

	enum data_type_e
	{
		MNIST_ALL = 0,
		MNIST_EVEN,
		MNIST_ODD,
		MNIST_0_4,
		MNIST_5_9,
		MNIST_BY_NUMBER
	};

	util::dataset_t get_dataset(i32 data_type = 0, i32 number = 0);
};