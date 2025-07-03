#pragma once

#include "snn.h"
#include "neurons.h"
#include "synapses.h"
#include "../util/log.h"
#ifdef BUILD_UI
#include "../rlimgui/rlImGui.h"
#include "../implot/implot.h"
#endif
#include <algorithm>
#include <thread>
#include <math.h>

namespace snn5
{
	//advanced log
	struct adv_log_t
	{
		struct base_log_t
		{
			log_t<float> plain;
			log_t<float> smooth;
			float temp = 0.0f;

			base_log_t()
			{
				plain.clear();
				smooth.clear();
			};

			void plot(std::string label, bool show_smooth)
			{
				float* arr = plain.data();
				if(show_smooth)
					arr = smooth.data();
#ifdef BUILD_UI
				ImPlot::PlotLine((label).c_str(),arr,1000);
#endif
			};

			void clear()
			{
				plain.index = 0;
				smooth.index = 0;
				temp = 0.0f;
			};
		};

		struct vec_log_t
		{
			std::vector<base_log_t> logs = {};
			std::vector<uint32_t> indices = {};
			std::jthread jt;
			std::atomic<int32_t> jt_flag = 1;

			void resize(uint32_t new_size)
			{
				logs.resize(new_size);
			};

			void push(const math::array<float> &array, std::vector<uint32_t> indices = {})
			{
				if(indices.size() == 0)
					for(uint32_t i = 0; i < array.size(); i++)
						logs[i].temp += array.at(i);

				for(uint32_t i = 0; i < indices.size(); i++)
					logs[indices.at(i)].temp += array.at(indices.at(i));
			};

			void eval(uint32_t count)
			{
				float inv_c = 1.0f / count;
				for(uint32_t i = 0; i < logs.size(); i++)
				{
					logs[i].plain.push(logs[i].temp*inv_c);
					logs[i].temp = 0.0f;
				}
			};

			void plot(std::string label, bool show_smooth)
			{
				for(uint32_t i = 0; i < indices.size(); i++)
					logs[indices.at(i)].plot(label+std::to_string(indices.at(i)),show_smooth);
			};

			void clear()
			{
				for(uint32_t i = 0; i < logs.size(); i++)
					logs[i].clear();
			};
		};

		struct math_log_t
		{
			base_log_t avg_log;
			base_log_t abs_log;
			base_log_t min_log;
			base_log_t max_log;

			void push(const math::array<float> &array)
			{
				float avg = 0.0f;
				float abs = 0.0f;
				for(uint32_t i = 0; i < array.size(); i++)
				{
					float f = array.at(i);
					avg += f;
					abs += std::abs(f);
					min_log.temp = std::min(min_log.temp,f);
					max_log.temp = std::max(max_log.temp,f);
				}
				avg_log.temp += (avg / array.size());
				abs_log.temp += (abs / array.size());
			};

			void eval(uint32_t count)
			{
				avg_log.plain.push(avg_log.temp/count);
				abs_log.plain.push(abs_log.temp/count);
				min_log.plain.push(min_log.temp);
				max_log.plain.push(max_log.temp);
				avg_log.temp = 0.0f;
				abs_log.temp = 0.0f;
				min_log.temp =  1000.0f;
				max_log.temp = -1000.0f;
			};

			void plot(std::string label, bool show_smooth)
			{
#ifdef BUILD_UI
				ImPlot::SetNextLineStyle(ImVec4(0.4f,0.9f,0.4f,1.0f));
				avg_log.plot(label+" Avg",show_smooth);
				ImPlot::SetNextLineStyle(ImVec4(0.6f,0.6f,0.6f,1.0f));
				abs_log.plot(label+" Abs",show_smooth);
				ImPlot::SetNextLineStyle(ImVec4(0.4f,0.4f,0.9f,1.0f));
				min_log.plot(label+" Min",show_smooth);
				ImPlot::SetNextLineStyle(ImVec4(0.9f,0.4f,0.4f,1.0f));
				max_log.plot(label+" Max",show_smooth);
#endif
			};

			void clear()
			{
				avg_log.clear();
				abs_log.clear();
				min_log.clear();
				max_log.clear();
				avg_log.temp = 0.0f;
				abs_log.temp = 0.0f;
				min_log.temp =  1000.0f;
				max_log.temp = -1000.0f;
			};
		};

		vec_log_t v_log;
		math_log_t m_log;

		log_t<float> scratch0;
		log_t<float> scratch1;
	
		uint32_t count = 0;
		int32_t draw_type = 0;
		bool enabled = false;
		bool show_smooth = false;

		std::string label;

		adv_log_t(std::string label_) : label(label_)
		{

		};

		void resize(uint32_t new_size)
		{
			v_log.resize(new_size);
		};

		void push(math::array<float> array, std::vector<uint32_t> indices = {})
		{
			if(enabled)
			{
				v_log.push(array,indices);
				m_log.push(array);
				count++;
			}
		};

		void eval()
		{
			if(enabled)
			{
				v_log.eval(count);
				m_log.eval(count);
				count = 0;
			}
		};

		void plot(std::string label, std::vector<uint32_t> indices)
		{
			if(enabled)
			{
				switch(draw_type)
				{
					case(0):
						v_log.plot(label,show_smooth);
						break;
					case(1):
						m_log.plot(label,show_smooth);
						break;
				}
			}
		};

		void clear()
		{
			v_log.clear();
			m_log.clear();
		};
	};

	template <typename T>
	struct log2_t
	{
		math::array<T> values;
		i32 index = 0;

		log2_t()
		{
			index = 0;
		};

		log2_t(i64 src_size)
		{
			values.resize(2000,src_size).zero();
			index = 0;
		};

		~log2_t()
		{
			//values.free();
		};

		/*void collect(const T &x)
		{
			values.at(index) = x;
		};*/

		void collect(const math::array<T> &x)
		{
			for(i64 i = 0; i < x.size(); i++)
				values.at(index,i) = x[i];
			index++;
			if(index >= 1000)
				index = 0;
			for(i64 i = 0; i < x.size(); i++)
				values.at(index,i) = x[i];
		};

		/*template <typename E>
		void collect(const templates::exp_t<E> &expr)
		{
			i64 i0 = index;
			i64 i1 = index + 1;
			if(i1 > values.x() / 2)
				i1 = 0;

			for(i64 i = 0; i < expr.size(); i++)
			{
				T x = expr[i];
				values.at(i0,i) = x;
				values.at(i1,i) = x;
			}

			index = i1;
		};*/

		T* data() const {return &values.data()[index];}

		T* data(i64 i) const {return &values.data()[i];};

		i64 stride() const {return values.y();};

		void reset()
		{
			values.zero();
			index = 0;
		};
	};

	struct base_log_t
	{
		log2_t<float> log;
		math::array<float> avg;
		std::string label;
		i32 counter = 0;
		i32 offset = 0;
		i32 max_visible = 50;
		bool active = false;

		base_log_t() {};
	
		base_log_t(std::string label_, i64 size)
		{
			label = label_;
			log = log2_t<float>(size);
			avg.resize(size).zero();
			counter = 0;
		};

		~base_log_t()
		{
			//avg.free();
		}

		void reset()
		{
			log.reset();
			avg.zero();
			counter = 0;
		};

		void resize(i64 new_size)
		{
			i64 x = log.values.x();
			i64 y = log.values.y();
			if(log.values.y() != new_size)
				log.values.resize(y);
		};

		void collect(const math::array<float> &x)
		{
			resize(x.size());
			if(active)
			{
				avg += x;
				counter++;
			}
		};

		void collect(const tensor<float> &x)
		{
			resize(x.size());
			if(active)
			{
				avg += x;
				counter++;
			}
		};

		void push()
		{
			if(active)
			{
				avg *= (1.0f / counter);
				log.collect(avg);
				avg.zero();
				counter = 0;
			}
		};

		void plot(i32 height, float y_min, float y_max);
	};

	struct n_log_t
	{
		n_block_t* nb = nullptr;

		base_log_t v;
		base_log_t c;
		//base_log_t ema;
		//base_log_t bit;
		base_log_t trace;
		base_log_t grad;
		base_log_t err;
		base_log_t sd;

		i32 height = 200;
		i32 merge = 50;

		n_log_t(n_block_t* nb_)
		{
			nb = nb_;
			v = base_log_t("v",nb->size());
			c = base_log_t("c",nb->size());
			//ema = base_log_t("ema",nb->size());
			//bit = base_log_t("bit",nb->size());
			trace = base_log_t("trace",nb->size());
			grad = base_log_t("grab",nb->size());
			err = base_log_t("err",nb->size());
			sd = base_log_t("sd",nb->size());
		};

		~n_log_t()
		{
			/*delete v;
			delete c;
			delete ema;
			delete bit;
			delete trace;
			delete grad;
			delete err;
			delete sd;*/
		};

		void collect()
		{
			v.collect(nb->v);
			c.collect(nb->c);
			//ema.collect(nb->spike.ema_rate);
			//bit.collect(nb->spike.bit_rate);
			trace.collect(nb->spike.trace);
			grad.collect(nb->bp.grad);
			err.collect(nb->bp.err);
			sd.collect(nb->bp.sd);

			if(v.counter >= merge)
				v.push();
			if(c.counter >= merge)
				c.push();
			//if(ema.counter >= merge)
			//	ema.push();
			//if(bit.counter >= merge)
			//	bit.push();
			if(trace.counter >= merge)
				trace.push();
			if(grad.counter >= merge)
				grad.push();
			if(err.counter >= merge)
				err.push();
			if(sd.counter >= merge)
				sd.push();
		};

		void push()
		{
			v.push();
			c.push();
			//ema.push();
			//bit.push();
			trace.push();
			grad.push();
			err.push();
			sd.push();
		};

		void draw();
	};

	struct s_log_t
	{
		s_block_t* sb = nullptr;

		base_log_t* w = nullptr;
		base_log_t* wr = nullptr;
		base_log_t* et = nullptr;

		i32 height = 200;
		i32 merge = 50;

		s_log_t(s_block_t* sb_)
		{
			sb = sb_;
			w = new base_log_t("w",sb->size());
			wr = new base_log_t("wr",sb->size());
			et = new base_log_t("et",sb->prev->size());
		};

		~s_log_t()
		{
			delete w;
			delete wr;
			delete et;
		};

		void collect()
		{
			/*w->collect(sb->w);
			wr->collect(sb->wr);
			et->collect(sb->et);*/

			if(w->counter >= merge)
				w->push();
			if(wr->counter >= merge)
				wr->push();
			if(et->counter >= merge)
				et->push();
		};

		void push()
		{
			w->push();
			wr->push();
			et->push();
		};

		void draw();
	};
};