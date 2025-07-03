#include "../math/array.h"
#include "../common.h"

#ifdef BUILD_UI
#include <imgui.h>
#include "../implot/implot.h"
#endif

/*
	00000000
	I---
	10001000
	 I---
	12001200
	  I---
	12301230
	   I---
*/

/*
	log with a rolling view window to avoid making large copies
	or data movements during runtime. requires allocating 2x
	the window size and one branch instruction during runtime.
*/

template <typename T>
struct log_t
{
	math::array<T> array;
	uint32_t index = 0;

	log_t()
	{
		array.resize(2000);
	};

	//~log_t() {array.free();};

	void push(T x)
	{
		array.at(index) = x;
		index++;
		if(index >= 1000)
			index = 0;
		array.at(index+1000) = x;
	};

	T* data() const {return &array.data()[index];};

	void clear() {array.zero(); index = 0;};
};

template <typename T>
struct log2d_t
{
	math::array<T> array;
	math::array<uint32_t> index;

	log2d_t() {};

	void resize(int32_t i)
	{
		array.resize(i,2000);
		index.resize(i);
		index.zero();
	};

	void push(T x, int32_t i)
	{
		array.at(i,index.at(i)) = x;
		index.at(i)++;
		if(index.at(i) >= 1000)
			index.at(i) = 0;
		array.at(i,index.at(i)+1000) = x;
	};

	T* data(int32_t i) const {return &array.at(i,index.at(i));};

	void clear() {array.zero(); index.zero();};
};

struct log1k_t
{
	sp<float[]> array;
	uint32_t index = 0;

	log1k_t() {array = std::make_shared<float[]>(1000);};

	void push(float x)
	{
		array.get()[index] = x;
		index++;
	};

	float* data() const {return array.get();};
};

struct rlog_t
{
	sp<float[]> data = 0;
	sp<float[]> view = 0;
	uint64_t data_size = 0;
	uint64_t view_ratio = 1;
	uint64_t index = 0;
	bool rolling = false; //means data_size is actually 2x

	private:

	void push_data(float v)
	{
		data[index] = v;
		index++;
		if(index >= data_size)
			index = 0;
		if(rolling)
			data[index+data_size] = v;
	};

	void push_view(float v)
	{
		/*uint64_t start = index - 1;
		if(index == 0)
			start = data_size;

		uint64_t offset = start - (start % view_ratio);

		view[start/view_ratio] = 0.0f;
		for(uint32_t i = 0; i < view_ratio; i++)
			view[start/view_ratio] += data[offset+i];
		//if(start % view_ratio == 0)
		view[start/view_ratio] /= view_ratio;
		view[start/view_ratio+data_size/view_ratio] /= view[start/view_ratio];*/
	};

	public:

	void resize(uint64_t new_size)
	{
		if(data_size != new_size)
		{
			data_size = new_size;
			uint64_t s = rolling ? data_size * 2 : data_size;
			data = std::make_shared<float[]>(data_size);
			view = std::make_shared<float[]>(data_size);
			for(uint32_t i = 0; i < data_size; i++)
			{
				data[i] = 0.0f;
				view[i] = 0.0f;
			}
		}
	};

	void reshape(uint64_t new_ratio)
	{
		if(view_ratio != new_ratio && new_ratio > 1)
		{
			view_ratio = new_ratio;
			//uint64_t s = rolling ? data_size * 2 : data_size;
			//view = std::make_shared<float[]>(data_size);

			for(uint32_t i = 0; i < data_size; i++)
			{
				view[i] = 0.0f;
			}

			for(uint32_t i = 0; i < data_size / view_ratio; i++)
			{
				view[i] = 0.0f;
				for(uint32_t j = 0; j < view_ratio; j++)
					view[i] += data[i*view_ratio+j];
				view[i] /= view_ratio;
			}
		}
	};

	void push(float v)
	{
		view_ratio = 1;
		push_data(v);

		/*
			pushed 0, now idx = 1
			pushed 1, now idx = 2
			pushed 7, now idx = 8

			index = 7
			offset = index - (index % 3) //6
			for(i = 0; i < ratio (3) && i < index + ratio; i++)
				[offset + i] (6,7,8)

			0,1,2,3,4,5,6,7,8,9
			0,    1,    2,    3
		*/

		//push_view(v);

		/*if(view_ratio > 1)
		{
			if((start + view_ratio) > (data_size * 2))
				start -= data_size;

			view[start/view_ratio] = 0.0f;
			uint64_t offset = start - (start % view_ratio);
			for(uint32_t i = 0; (i < view_ratio) && (i + offset < index); i++)
				view[start/view_ratio] += data[i+offset];
			view[start/view_ratio] /= view_ratio;
			view[start/view_ratio+data_size/view_ratio] = view[start/view_ratio];
		}*/
	};

	float* get_ptr() const
	{
		if(rolling)
		{
			if(view_ratio > 1)
				return &view[index/view_ratio];
			else
				return &data[index];
		}
		return data.get();
	};

	uint64_t size() const
	{
		if(view_ratio > 1)
			return data_size / view_ratio;
		return data_size;
	};
};

template <typename T>
struct mini_log_t
{
	math::array<float> values;
	std::string label;
	bool show = false;

	mini_log_t() {};

	mini_log_t(std::string l, i32 s) {label = l; resize(s);};

	//~mini_log_t() {values.free();};

	void resize(i32 size)
	{
		values.resize(size);
	};

	void draw()
	{
#ifdef BUILD_UI
		ImGui::Checkbox(label.c_str(),&show);
#endif
	};

	void plot() const
	{
#ifdef BUILD_UI
		if(show)
			ImPlot::PlotLine(label.c_str(),values.data(),values.size());
#endif
	};

	T& operator[](i32 i) {return values[i];};
	
	T operator[](i32 i) const {return values[i];};
};