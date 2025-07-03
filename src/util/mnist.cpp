#include "data.h"
#include "mnist3.h"

namespace mnist
{
	mchar32_set_t* mnist_train = nullptr;
	mchar32_set_t* mnist_test = nullptr;

	void mchar8_t::draw(int32_t x, int32_t y, float mul)
	{
#ifdef BUILD_UI
		DrawRectangle(x-1,y-1,30,30,BLUE);
		for(int32_t i = 0; i < data.x(); i++)
			for(int32_t j = 0; j < data.y(); j++)
				DrawPixel(x+i,y+j,Color(data.at(i,j)*mul,0,0,254));
		std::string num = std::to_string(number);
		DrawText(num.c_str(),x,y+30,14,WHITE);
#endif
	};

	void mchar32_t::draw(int32_t x, int32_t y)
	{
#ifdef BUILD_UI
		DrawRectangle(x-1,y-1,30,30,RED);
		for(int32_t i = 0; i < data.x(); i++)
			for(int32_t j = 0; j < data.y(); j++)
				DrawPixel(x+i,y+j,Color(data.at(i,j)*254.0f,0,0,255));
		std::string num = std::to_string(number);
		DrawText(num.c_str(),x,y+30,14,WHITE);
#endif
	};

	mchar8_set_t::mchar8_set_t(const std::string &label_path, const std::string &image_path)
	{
		{
			FILE* f = fopen(label_path.c_str(),"rb");
			if(f == nullptr)
				throw std::runtime_error("Failed to open labels at: "+label_path+"\n");

			int32_t magic = -1;
			int32_t count = -1;

			fread(&magic,sizeof(int32_t),1,f);
			fread(&count,sizeof(int32_t),1,f);

			magic = __tiny_byte_swap(magic);
			count = __tiny_byte_swap(count);

			if(magic != 2049)
			{
				std::string magic_str = std::to_string(magic);
				throw std::runtime_error(
					"Bad magic at: "+label_path+"\nExpected 2049 got: "+label_path+"\n"
				);
			}

			chars.resize(count);
			for(uint32_t i = 0; i < count; i++)
			{
				int8_t label = -1;
				fread(&label,sizeof(int8_t),1,f);
				chars.at(i).number = label;
				if(label > 9 || label < 0)
					throw std::runtime_error("Invalid label in :"+label_path+"!\n");
			}

			fclose(f);
		}

		{
			FILE* f = fopen(image_path.c_str(),"rb");
			if(f == nullptr)
				throw std::runtime_error("Failed to open images at: "+image_path+"\n");

			int32_t magic = -1;
			int32_t count = -1;

			fread(&magic,sizeof(int32_t),1,f);
			fread(&count,sizeof(int32_t),1,f);

			magic = __tiny_byte_swap(magic);
			count = __tiny_byte_swap(count);

			if(magic != 2051)
			{
				std::string magic_str = std::to_string(magic);
				throw std::runtime_error(
					"Bad magic at: "+image_path+"\nExpected 2051 got: "+magic_str+"\n"
				);
			}

			if(count != chars.size())
				throw std::runtime_error("Mismatched sizes!\n");

			int32_t rows = -1;
			int32_t cols = -1;
			fread(&rows,sizeof(int32_t),1,f);
			fread(&cols,sizeof(int32_t),1,f);
			rows = __tiny_byte_swap(rows);
			cols = __tiny_byte_swap(cols);

			for(int32_t i = 0; i < count; i++)
			{
				for(uint32_t j = 0; j < rows; j++)
				{
					for(uint32_t k = 0; k < cols; k++)
					{
						uint8_t p;
						fread(&p,sizeof(uint8_t),1,f);
						chars.at(i).data.at(k,j) = p;
					}
				}
			}

			fclose(f);
		}
	};

	mchar32_set_t::mchar32_set_t(const std::string &label_path, const std::string &image_path)
	{
		mchar8_set_t* m8s = new mchar8_set_t(label_path,image_path);
		chars.resize(m8s->chars.size());
		for(int32_t i = 0; i < 10; i++)
			chars_by_number.insert({i,{}});
		for(uint32_t i = 0; i < m8s->chars.size(); i++)
		{
			mchar8_t c = m8s->chars.at(i);
			chars.at(i) = mchar32_t(c);
			chars_by_number.at(c.number).push_back(c);
			if(c.number % 2 == 0)
				even_chars.push_back(c);
			else
				odd_chars.push_back(c);
			if(c.number < 5)
				chars_0_4.push_back(c);
			else
				chars_5_9.push_back(c);
		}
		delete m8s;
	};

	void char_viewer_t::draw(mnist::mchar8_set_t* train_set8, mnist::mchar32_set_t* train_set)
	{
#ifdef BUILD_UI
		ImGui::Begin("Char Viewer");
		ImGui::InputInt("Index",&char_index);
		char_index = std::clamp(char_index,0,60000);
		ImGui::InputInt("X",&pos_x);
		ImGui::InputInt("Y",&pos_y);
		ImGui::Checkbox("Use 8",&use_set8);
		ImGui::InputFloat("Mul",&mul);
		if(!use_set8)
			train_set->chars.at(char_index).draw(pos_x,pos_y);
		else
			train_set8->chars.at(char_index).draw(pos_x,pos_y,mul);
		ImGui::End();
#endif
	};

	void char_viewer_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Char Viewer");
		ImGui::InputInt("Index",&char_index);
		char_index = std::clamp(char_index,0,60000);
		ImGui::InputInt("X",&pos_x);
		ImGui::InputInt("Y",&pos_y);
		ImGui::Checkbox("Use 8",&use_set8);
		ImGui::InputFloat("Mul",&mul);
		mnist_train->chars.at(char_index).draw(pos_x,pos_y);
		ImGui::End();
#endif
	};

	void load_data(std::string path)
	{
		if(mnist_train == nullptr)
		{
			mnist_train = new mnist::mchar32_set_t(
				path+"train-labels.idx1-ubyte",
				path+"train-images.idx3-ubyte"	
			);
		}

		if(mnist_test == nullptr)
		{
			mnist_test = new mnist::mchar32_set_t(
				path+"t10k-labels.idx1-ubyte",
				path+"t10k-images.idx3-ubyte"
			);
		}
	};

	void free_data()
	{
		if(mnist_train != nullptr)
			delete mnist_train;

		if(mnist_test != nullptr)
			delete mnist_test;
	};

	util::dataset_t get_dataset(i32 data_type, i32 number)
	{
		util::dataset_t ds;

		switch(data_type)
		{
			default:
			case(MNIST_ALL):
				for(i32 i = 0; i < mnist_train->chars.size(); i++)
					ds.train_samples.push_back(
						{mnist_train->chars[i].data,mnist_train->chars[i].target}
					);
				for(i32 i = 0; i < mnist_test->chars.size(); i++)
					ds.test_samples.push_back(
						{mnist_test->chars[i].data,mnist_test->chars[i].target}
					);
				break;
			case(MNIST_EVEN):
				for(i32 i = 0; i < mnist_train->even_chars.size(); i++)
					ds.train_samples.push_back(
						{mnist_train->even_chars[i].data,mnist_train->even_chars[i].target}
					);
				for(i32 i = 0; i < mnist_test->even_chars.size(); i++)
					ds.test_samples.push_back(
						{mnist_test->even_chars[i].data,mnist_test->even_chars[i].target}
					);
				break;
			case(MNIST_ODD):
				for(i32 i = 0; i < mnist_train->odd_chars.size(); i++)
					ds.train_samples.push_back(
						{mnist_train->odd_chars[i].data,mnist_train->odd_chars[i].target}
					);
				for(i32 i = 0; i < mnist_test->odd_chars.size(); i++)
					ds.test_samples.push_back(
						{mnist_test->odd_chars[i].data,mnist_test->odd_chars[i].target}
					);
				break;
			case(MNIST_0_4):
				for(i32 i = 0; i < mnist_train->chars_0_4.size(); i++)
					ds.train_samples.push_back(
						{mnist_train->chars_0_4[i].data,mnist_train->chars_0_4[i].target}
					);
				for(i32 i = 0; i < mnist_test->chars_0_4.size(); i++)
					ds.test_samples.push_back(
						{mnist_test->chars_0_4[i].data,mnist_test->chars_0_4[i].target}
					);
				break;
			case(MNIST_5_9):
				for(i32 i = 0; i < mnist_train->chars_5_9.size(); i++)
					ds.train_samples.push_back(
						{mnist_train->chars_5_9[i].data,mnist_train->chars_5_9[i].target}
					);
				for(i32 i = 0; i < mnist_test->chars_5_9.size(); i++)
					ds.test_samples.push_back(
						{mnist_test->chars_5_9[i].data,mnist_test->chars_5_9[i].target}
					);
				break;
			case(MNIST_BY_NUMBER):
				for(i32 i = 0; i < mnist_train->chars_by_number.at(number).size(); i++)
					ds.train_samples.push_back({
						mnist_train->chars_by_number.at(number)[i].data,
						mnist_train->chars_by_number.at(number)[i].target
					});
				for(i32 i = 0; i < mnist_test->chars_by_number.at(number).size(); i++)
					ds.test_samples.push_back({
						mnist_test->chars_by_number.at(number)[i].data,
						mnist_test->chars_by_number.at(number)[i].target
					});
				break;
		}
		
		return ds;	
	};
};