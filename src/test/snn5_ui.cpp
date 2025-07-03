#include "../snn5/snn.h"
#include "../snn5/builder.h"
#include <chrono>
#include <functional>
#include <mutex>
//#include <cblas.h>
#include <thread>
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
#include "../util/debug.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <raylib.h>
#include <string>
#include <thread>
#include <queue>
#include <unordered_set>

void push_graph_vars(float lx, float ly, float sx = 200.0f, float sy = 200.0f)
{
	ImPlot::PushStyleVar(ImPlotStyleVar_PlotDefaultSize,{sx,sy});
	ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding,{2.0f,2.0f});
	ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding,{2.0f,2.0f});
	ImPlot::PushStyleVar(ImPlotStyleVar_LegendSpacing,{1.0f,1.0f});
	ImPlot::PushStyleVar(ImPlotStyleVar_LegendInnerPadding,{1.0f,1.0f});
	ImPlot::SetNextAxisLimits(ImAxis_X1,0.0f,lx);
	ImPlot::SetNextAxisLimits(ImAxis_Y1,0.0f,ly);
};

void pop_graph_vars()
{
	for(u32 i = 0; i < 5; i++)
		ImPlot::PopStyleVar();
};

inline bool izh(float &v, float &u, float c, float tr, float min_v, float max_v, bool early_u_update)
{
	bool fired = false;
	float nv = v + kernel::izh4_dvdt(v,u,c);
	if(early_u_update)
		u += kernel::izh4_dudt(v,u,0.02f,0.2f);
	
	if(nv >= tr)
	{
		v = -65.0f;
		u += 8.0f;
		fired = true;
	}
	else
	{
		v = std::clamp(nv,min_v,max_v);
	}

	if(!early_u_update)
		u += kernel::izh4_dudt(v,u,0.02f,0.2f);
	return fired;
};

#define FBX_MACRO(typename) \
/*\
	this is a comment\
*/ struct typename {};

FBX_MACRO(vec2);

struct conv_test_t
{
	math::array<float> input_matrix;
	math::array<float> kernel_matrix;
	math::array<float> output_matrix;

	//input size
	i32 ix = 28;
	i32 iy = 28;

	//kernel size
	i32 kx = 4;
	i32 ky = 4;

	//stride
	i32 sx = 1;
	i32 sy = 1;

	//output size (automatically calculated)
	i32 ox = 8;
	i32 oy = 8;

	i32 draw_pos_x = 300;
	i32 draw_pos_y = 300;
	i32 draw_scale = 4;

	i32 fn_po = 0;
	i32 fn_ko = 0;
	i32 fn_so = 0;

	dim4 dims;

	i32 qx = 0;
	i32 qy = 0;

	rng_t rng;

	i32 mnist_index = 0;
	bool use_mnist = false;

	void draw_matrix(
		i32 px, i32 py, math::array<float> mat, i32 scale = 1,
		float r = 254.0f, float g = 0.0f, float b = 0.0f,
		bool outline = false
	)
	{
		if(outline)
			DrawRectangle(px-scale,py-scale,(mat.x()+2)*scale,(mat.y()+2)*scale,RED);
		for(i32 i = 0; i < mat.x(); i++)
		{
			for(i32 j = 0; j < mat.y(); j++)
			{
				for(i32 k = 0; k < scale; k++)
				{
					for(i32 l = 0; l < scale; l++)
					{
						float v = mat.at(i,j);
						Color c = Color(v*r,v*g,v*b,255);
						DrawPixel(px+i*scale+k,py+j*scale+l,c);
					}
				}
			}
		}
	};

	void step_q()
	{
		if(qx + kx < ix)
		{
			qx += sx;
		}
		else
		{
			qx = 0;
			if(qy + ky < iy)
			{
				qy += sy;
			}
			else
			{
				qy = 0;
			}
		}
	};

	void step_data()
	{
		//if(ix == 28 && iy == 28 && use_mnist)
		//	input_matrix.copy_from(train_set->chars[mnist_index].data);

		//copy input to kernel
		for(i32 i = 0; i < kx; i++)
			for(i32 j = 0; j < ky; j++)
				kernel_matrix.at(i,j) = input_matrix.at(i+qx,j+qy);

		float sum = 0.0f;
		for(i32 i = 0; i < kernel_matrix.x(); i++)
			for(i32 j = 0; j < kernel_matrix.y(); j++)
				sum += kernel_matrix.at(i,j);
		output_matrix.at(qx/sx,qy/sy) = sum / kernel_matrix.size();
	};

	void reset()
	{
		qx = 0;
		qy = 0;
		kernel_matrix.zero();
		output_matrix.zero();
	};

	void draw()
	{
		ImGui::Begin("Conv Test");

		ImGui::PushItemWidth(96.0f);

		if(ImGui::Button("Randomize Input"))
		{
			input_matrix.resize({ix,iy}).zero().randomize(rng);
		}

		ImGui::Checkbox("Use MNIST",&use_mnist);
		ImGui::SameLine();
		ImGui::InputInt("Index",&mnist_index);
		mnist_index = std::clamp<i32>(mnist_index,0,1000);

		if(ImGui::Button("Step"))
			step_q();
		ImGui::SameLine();
		if(ImGui::Button("Run"))
		{
			reset();
			for(i32 i = 0; i < ix * iy; i++)
			{
				step_q();
				step_data();
			}
		}
		ImGui::SameLine();
		if(ImGui::Button("Run FN"))
		{
			reset();
			math::conv_kernel_t k;
			k.padding = fn_po;
			k.size.x = kx + fn_ko;
			k.size.y = ky + fn_ko;
			k.stride = sx + fn_so;
			tensor<float> w; w.resize({input_matrix.x(),input_matrix.y(),k.size.x,k.size.y});
			w.fill(1.0f/(k.size.x*k.size.y));
			//w.fill(1.0f);
			math::conv_2d(k,w,input_matrix,output_matrix);
			w.free();
			output_matrix.clamp(0.0f,1.0f);
		}
		ImGui::SameLine();
		if(ImGui::Button("Reset"))
			reset();

		i32 dx = dims.x;
		i32 dy = dims.y;
		i32 dz = dims.z;
		i32 dw = dims.w;

		ImGui::InputInt("dx",&dx);
		ImGui::InputInt("dy",&dy);
		ImGui::InputInt("dz",&dz);
		ImGui::InputInt("dw",&dw);

		dims.x = dx;
		dims.y = dy;
		dims.z = dz;
		dims.w = dw;
		
		ImGui::InputInt("KO",&fn_ko);
		ImGui::InputInt("SO",&fn_so);
		ImGui::InputInt("PO",&fn_po);

		ImGui::SeparatorText("Draw");
		ImGui::PushID("draw");
		ImGui::InputInt("X",&draw_pos_x);
		ImGui::SameLine();
		ImGui::InputInt("Y",&draw_pos_y);
		ImGui::InputInt("Scale",&draw_scale);
		ImGui::PopID();

		draw_pos_x = std::clamp<i32>(draw_pos_x,0,9999);
		draw_pos_y = std::clamp<i32>(draw_pos_y,0,9999);
		draw_scale = std::clamp<i32>(draw_scale,1,16);

		ImGui::SeparatorText("Input Matrix");
		ImGui::PushID("input");
		ImGui::InputInt("X",&ix);
		ImGui::SameLine();
		ImGui::InputInt("Y",&iy);
		ImGui::PopID();

		ix = std::clamp<i32>(ix,3,256);
		iy = std::clamp<i32>(iy,3,256);

		ImGui::SeparatorText("Kernel");
		ImGui::PushID("kernel");
		ImGui::InputInt("X",&kx);
		ImGui::SameLine();
		ImGui::InputInt("Y",&ky);
		ImGui::PopID();

		ImGui::SeparatorText("Stride");
		ImGui::PushID("stride");
		ImGui::InputInt("X",&sx);
		ImGui::SameLine();
		ImGui::InputInt("Y",&sy);
		ImGui::PopID();

		kx = std::clamp<i32>(kx,2,ix-1);
		ky = std::clamp<i32>(ky,2,iy-1);

		ox = 0;
		for(i32 i = 0; i + kx <= ix; i += sx)
			ox++;

		oy = 0;
		for(i32 i = 0; i + ky <= iy; i += sy)
			oy++;

		ImGui::SeparatorText("Output");
		ImGui::PushID("output");
		ImGui::BeginDisabled();
		ImGui::InputInt("X",&ox);
		ImGui::SameLine();
		ImGui::InputInt("Y",&oy);
		ImGui::EndDisabled();
		ImGui::PopID();

		ImGui::SeparatorText("Q");
		ImGui::PushID("q");
		ImGui::BeginDisabled();
		ImGui::InputInt("X",&qx);
		ImGui::SameLine();
		ImGui::InputInt("Y",&qy);
		ImGui::EndDisabled();
		ImGui::PopID();

		ImGui::PopItemWidth();

		ImGui::End();

		if(input_matrix.x() != ix || input_matrix.y() != iy)
		{
			input_matrix.resize({ix,iy});
		}
		if(kernel_matrix.x() != kx || kernel_matrix.y() != ky)
		{
			kernel_matrix.resize({kx,ky});
		}
		if(output_matrix.x() != ox || output_matrix.y() != oy)
		{
			output_matrix.resize({ox,oy});
			qx = std::clamp<i32>(qx,0,ox);
			qy = std::clamp<i32>(qy,0,oy);
		}

		step_data();

		draw_matrix(
			draw_pos_x,draw_pos_y,
			input_matrix,draw_scale,
			254.0f,0.0f,0.0f,true
		);
		i32 osx = (ix+2) * draw_scale;
		draw_matrix(
			draw_pos_x+osx,draw_pos_y,
			kernel_matrix,draw_scale,
			0.0f,0.0f,254.0f,true
		);
		osx += (kx+2) * draw_scale;
		draw_matrix(
			draw_pos_x+osx,draw_pos_y,
			output_matrix,draw_scale,
			0.0f,254.0f,0.0f,true
		);

		draw_matrix(
			draw_pos_x+qx*draw_scale,
			draw_pos_y+qy*draw_scale,
			kernel_matrix,draw_scale,
			254.0f,254.0f,0.0f
		);
	};
};

struct jitter_t
{
	bool active = false;
	float relative_scale = 0.1f;
	float absolute_sclae = 1.0f;

	float run(float &x, rng_t &rng)
	{
		if(!active)
			return x;
		float r = (x * relative_scale) * (1.0f-(rng.u()*2.0f));
		float a = (1.0f-(rng.u()*2.0f));
		x += r + a;
		return x;
	};

	void draw(std::string label)
	{
		ImGui::Text("%s",label.c_str());
		ImGui::PushID(label.c_str());
		ImGui::Checkbox("Active",&active);
		ImGui::SliderFloat("R Scale",&relative_scale,0.0f,1.0f);
		ImGui::SliderFloat("A Scale",&absolute_sclae,0.0f,10.0f);
		ImGui::PopID();
	};
};

struct elig_tester_t
{
	tensor<float> elig;
	
	elig_tester_t()
	{
		elig.resize(1000).zero();
	};

	void draw()
	{
		ImGui::Begin("Eligbility Test");
		for(u32 i = 0; i < 1000; i++)
			elig[i] = 1.0f / (i + 1.0f) * (i * elig[i] + 1.0f * 1.0f);

		push_graph_vars(1000.0f,1.0f);
		if(ImPlot::BeginPlot("Eligibilty over time"))
		{
			ImPlot::PlotLine("E",elig.data(),1000);
			ImPlot::EndPlot();
		}
		for(uint32_t i = 0; i < 5; i++)
			ImPlot::PopStyleVar();
		ImGui::End();
	};
};

struct rate_tester_t
{
	tensor<float> rates;
	tensor<u64> hist = {};
	tensor<float> v;
	tensor<float> u;
	float tr = 30.0f;
	float c_min =   0.0f;
	float c_max = 500.0f;
	float min_v = -90.0f;
	float max_v =  35.0f;
	jitter_t tr_j;
	jitter_t c_j;
	jitter_t v_reset_j;
	jitter_t u_reset_j;
	i32 steps   = 250;
	bool early_u_update = false;
	rng_t rng;

	rate_tester_t()
	{
		rates.resize(1000).zero();
		hist.resize(1000).zero();
		v.resize(1000).fill(-65.0f);
		u.resize(1000).fill(-65.0f*0.2f);
		rng.seed(1023948,97873);
	};

	~rate_tester_t()
	{
		rates.free();
		hist.free();
		v.free();
		u.free();
	};

	void exec()
	{
		for(u32 i = 0; i < 1000; i++)
		{
			float c = (c_max - c_min) * ((float)i / (float)1000) + c_min;
			float r = 0.0f;

			c = c_j.run(c,rng);

			for(i32 j = 0; j < steps; j++)
			{
				hist[i] <<= 1u;
				float tr_ = tr_j.run(tr,rng);
				if(izh(v[i],u[i],c,tr_,min_v,max_v,early_u_update))
				{
					hist[i] |= 0x1u;
					v[i] = v_reset_j.run(v[i],rng);
					u[i] = u_reset_j.run(u[i],rng);
				}
				r += snn5::get_set_bits(hist[i]);
			}

			rates[i] = r / steps * 15.625f;
		}
	};

	void draw()
	{
		ImGui::Begin("Current Test");
		
		ImGui::InputFloat("C Min",&c_min);
		ImGui::InputFloat("C Max",&c_max);
		ImGui::InputFloat("V Min",&min_v);
		ImGui::InputFloat("V Max",&max_v);
		ImGui::InputInt("Steps",&steps);
		ImGui::Checkbox("Early U Update",&early_u_update);
		ImGui::SliderFloat("Tr",&tr,0.0f,60.0f);
		tr_j.draw("Tr Jitter");
		c_j.draw("C Jitter");
		v_reset_j.draw("V Jitter");
		u_reset_j.draw("U Jitter");

		push_graph_vars(1000.0f,1000.0f);
		if(ImPlot::BeginPlot("Rates"))
		{
			ImPlot::PlotLine("Hz",rates.data(),1000);
			ImPlot::EndPlot();
		}
		for(uint32_t i = 0; i < 5; i++)
			ImPlot::PopStyleVar();

		exec();

		ImGui::End();
	};
};

struct threshold_tester_t
{
	tensor<float> rates = {};
	tensor<u64> hist = {};
	tensor<float> v;
	tensor<float> u;
	tensor<float> c;
	float tr_min  =  0.0f;
	float tr_max  = 60.0f;
	float current = 50.0f;
	float min_v  = -90.0f;
	float max_v   = 35.0f;
	i32 samples = 1000;
	i32 steps   =  250;
	bool early_u_update = false;

	threshold_tester_t()
	{
		rates.resize(samples).zero();
		hist.resize(samples).zero();
		v.resize(samples).fill(-65.0f);
		u.resize(samples).fill(-65.0f*0.2f);
		c.resize(samples).zero();
	};

	~threshold_tester_t()
	{
		rates.free();
		hist.free();
		v.free();
		u.free();
		c.free();
	};

	void exec()
	{
		c.fill(current);
		for(i32 i = 0; i < samples; i++)
		{
			float tr = (tr_max - tr_min) * ((float)i / (float)samples) + tr_min;
			float r = 0.0f;

			for(i32 j = 0; j < steps; j++)
			{
				hist[i] <<= 1u;
				if(izh(v[i],u[i],c[i],tr,min_v,max_v,early_u_update))
					hist[i] |= 0x1u;
				r += snn5::get_set_bits(hist[i]);
			}

			rates[i] = r / steps * 15.625f;
		}
	};

	void draw()
	{
		ImGui::Begin("Threshold Test");
		
		ImGui::InputFloat("Tr Min",&tr_min);
		ImGui::InputFloat("Tr Max",&tr_max);
		ImGui::InputFloat("V Min",&min_v);
		ImGui::InputFloat("V Max",&max_v);
		ImGui::InputInt("Steps",&steps);
		ImGui::Checkbox("Early U Update",&early_u_update);
		ImGui::SliderFloat("Current",&current,0.0f,500.0f);

		push_graph_vars(1000.0f,1000.0f);
		if(ImPlot::BeginPlot("Rates"))
		{
			ImPlot::PlotLine("Hz",rates.data(),samples);
			ImPlot::EndPlot();
		}
		for(uint32_t i = 0; i < 5; i++)
			ImPlot::PopStyleVar();

		exec();

		ImGui::End();
	};
};

struct izhd_tester_t
{
	snn5::izhd_block_t* ib = nullptr;

	float input_c = 0.0f;
	float input_c2 = 0.0f;

	i32 step_count = 2000;

	bool auto_run = false;

	mini_log_t<float> ema_rate = mini_log_t<float>("EMA",2000);
	mini_log_t<float> bit_rate = mini_log_t<float>("Bit",2000);
	mini_log_t<float> trace  = mini_log_t<float>("T",2000);
	mini_log_t<float> v = mini_log_t<float>("V",2000);
	mini_log_t<float> u = mini_log_t<float>("U",2000);
	mini_log_t<float> v2 = mini_log_t<float>("V2",2000);
	mini_log_t<float> u2 = mini_log_t<float>("U2",2000);

	izhd_tester_t()
	{
		ib = new snn5::izhd_block_t(nullptr,0,{1,1},{});
	};
	
	~izhd_tester_t()
	{
		delete ib;
	};

	void draw()
	{
		ImGui::Begin("Izhd Test");
		ema_rate.draw();
		ImGui::SameLine();
		bit_rate.draw();
		ImGui::SameLine();
		trace.draw();
		v.draw();
		ImGui::SameLine();
		u.draw();
		v2.draw();
		ImGui::SameLine();
		u2.draw();

		push_graph_vars(100.0f,1.0f);
		if(ImPlot::BeginPlot("Results"))
		{
			ema_rate.plot();
			bit_rate.plot();
			trace.plot();
			v.plot();
			u.plot();
			v2.plot();
			u2.plot();

			ImPlot::EndPlot();
		}
		pop_graph_vars();

		ImGui::SliderFloat("C",&input_c,0.0f,10.0f);
		ImGui::SliderFloat("C2",&input_c2,0.0f,10.0f);

		ImGui::SliderFloat("sMinV",&ib->m_config.soma.min_v,-100.0f,100.0f);
		ImGui::SliderFloat("sMaxV",&ib->m_config.soma.max_v,-100.0f,100.0f);
		ImGui::SliderFloat("dMinV",&ib->m_config.dendrites.min_v,-100.0f,100.0f);
		ImGui::SliderFloat("dMaxV",&ib->m_config.dendrites.max_v,-100.0f,100.0f);

		ImGui::Checkbox("Auto Run",&auto_run);
		//if(ImGui::Button("Run"))
		if(auto_run)
			run();
		ImGui::End();
	};

	void run()
	{
		ib->reset();
		for(i32 i = 0; i < step_count; i++)
		{
			ib->c.at(0) = input_c;
			ib->c2.at(0) = input_c2;
			ib->step();
			//ema_rate.values.at(i) = ib->spike.ema_rate.at(0);
			//bit_rate.values.at(i) = ib->spike.bit_rate.at(0);
			trace.values.at(i) = ib->spike.trace.at(0);
			v.values.at(i) = ib->v.at(0);
			u.values.at(i) = ib->u.at(0);
			v2.values.at(i) = ib->v2.at(0);
			u2.values.at(i) = ib->u2.at(0);
		}
	};
};

struct raf_tester_t
{
	snn5::raf_block_t* raf = nullptr;

	struct logs_t
	{
		mini_log_t<float> v_reset;
		mini_log_t<float> beta;
		mini_log_t<float> v;
		mini_log_t<float> f;
		mini_log_t<float> I;
		mini_log_t<float> h;
	} log;

	i32 step_count = 1000;
	i32 c_end = 100;
	float input_c = 0.0f;
	bool auto_run = false;

	raf_tester_t()
	{
		raf = new snn5::raf_block_t(nullptr,0,{1,1},{});
		log.v_reset = mini_log_t<float>("vr",step_count);
		log.beta = mini_log_t<float>("b",step_count);
		log.v = mini_log_t<float>("v",step_count);
		log.f = mini_log_t<float>("f",step_count);
		log.I = mini_log_t<float>("i",step_count);
		log.h = mini_log_t<float>("h",step_count);
	};

	void draw()
	{
		ImGui::Begin("RAF Tester");
		ImGui::SliderFloat("C",&input_c,0.0f,10.0f);

		ImGui::InputInt("C End",&c_end);
		log.v_reset.draw();
		ImGui::SameLine();
		log.beta.draw();
		ImGui::SameLine();
		log.v.draw();
		log.f.draw();
		ImGui::SameLine();
		log.I.draw();
		log.h.draw();
		//ImGui::SameLine();
		//u2.draw();

		raf->m_config.draw(false);
		ImGui::Checkbox("Current Decay",&raf->config.use_current_decay);
		ImGui::InputFloat("Cd",&raf->config.current_decay);
		//raf->config.draw(false);

		//ImGui::InputFloat("V reset",&raf->m_config.v_reset);
		//ImGui::Checkbox("Manual Vr",&raf->m_config.use_specific_v_reset);

		push_graph_vars(100.0f,1.0f);
		if(ImPlot::BeginPlot("Results"))
		{
			log.v_reset.plot();
			log.beta.plot();
			log.v.plot();
			log.f.plot();
			log.I.plot();
			log.h.plot();

			ImPlot::EndPlot();
		}
		pop_graph_vars();

		ImGui::Checkbox("Auto Run",&auto_run);
		if(auto_run)
			run();
		ImGui::End();
	};

	void run()
	{
		raf->reset();
		for(i32 i = 0; i < step_count; i++)
		{
			if(!raf->config.use_current_decay)
				raf->c.zero();
			if(i < c_end)
				raf->c += input_c;				
			raf->step();
			log.v_reset[i] = raf->v_reset[0];
			log.beta[i] = raf->beta[0];
			log.v[i] = raf->v[0];
			log.f[i] = raf->f[0];
			log.I[i] = raf->I[0];
			log.h[i] = raf->h[0];
			log.v_reset.values.clamp(-999.0f,999.9f);
			log.beta.values.clamp(-999.0f,999.9f);
			log.v.values.clamp(-999.0f,999.9f);
			log.f.values.clamp(-999.0f,999.9f);
			log.I.values.clamp(-999.0f,999.9f);
			log.h.values.clamp(-999.0f,999.9f);
		}
	};
};

struct debug_printer_t
{
	i32 max_logs = -1;
	std::string ffw_step_str = "ffw_step";
	std::string ffw_step_data_str = "ffw_step_data";
	std::string mux_step_str = "mux_step";
	std::string mux_step_data_str = "mux_step_data";

	debug_printer_t()
	{
		max_logs = debug::get_max_log_size();
	};

	void draw_section(std::string label, std::string step_str, std::string data_str)
	{
		ImGui::SeparatorText((label).c_str());
		ImGui::PushID((step_str+data_str).c_str());
		
		if(ImGui::Button("Print step logs"))
			debug::print_logs(step_str);
		ImGui::SameLine();
		if(ImGui::Button("Print data logs"))
			debug::print_logs(data_str);
		
		if(ImGui::Button("Clear step logs"))
			debug::clear_logs(step_str);
		ImGui::SameLine();
		if(ImGui::Button("Clear data logs"))
			debug::clear_logs(data_str);

		if(ImGui::Button("Write step logs"))
			debug::write_to_file(step_str,"../log/"+step_str+"_log.txt",true);
		ImGui::SameLine();
		if(ImGui::Button("Write data logs"))
			debug::write_to_file(data_str,"../log/"+data_str+"_log.txt",true);
		
		ImGui::PopID();
	};

	void draw()
	{
		ImGui::Begin("Debug Printer");
		ImGui::SetNextItemWidth(75.0f);
		ImGui::InputInt("Max Logs",&max_logs);
		if(ImGui::Button("Get max logs"))
			max_logs = debug::get_max_log_size();
		if(ImGui::Button("Set max logs"))
			debug::set_max_log_size(max_logs);
		draw_section("FFW",ffw_step_str,ffw_step_data_str);
		draw_section("Mux",mux_step_str,mux_step_data_str);
		ImGui::End();
	};
};

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

	/*mnist::mchar32_set_t* set = train_set;
	if(!training)
		set = test_set;*/

	uint32_t group_size = r->run_config.samples / 10;

	uint32_t total_items = r->run_config.samples;
	/*if(training && r->run_config.sequential)
	{
		total_items = 0;
		for(uint32_t ni = 0; ni < 10; ni++) //number index
			total_items += std::min<uint32_t>(set->chars_by_number.at(ni).size(),group_size);
	}*/

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

						//if(set->chars_by_number.at(ni).size() <= gi)
						//	continue;

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

	/*if(r->run_config.auto_test && r->test_results.live_score == -1.0f && r->signals.cmd.stop.load() != 1)
		__run(r,r->test_results,false);*/

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

enum status_e
{
	STATUS_IDLE = 0,
	STATUS_BUSY = 1
};

struct signals_t
{
	std::atomic<int32_t> stop = 0;
	std::atomic<int32_t> pause = 0;
	std::atomic<int32_t> done = 1;

	std::atomic<int32_t> status = 0;
};

struct validator_t
{
	struct delta_m_t
	{
		tensor<float> data = {};
		tensor<float> misc = {};
		float base_update = 0.05f;
		float m_max = 25.0f;
		float m_init = 0.1f;
		int32_t size = 100;
		bool range = false;

		float mul = 1.00f;
		float offset = 0.0f;
		float f0 = 0.05f;
		float f1 = 0.03f;
		float f2 = 0.01f;

		void draw()
		{
			data.resize(size).zero();
			data.at(0) = m_init;
			if(!range)
			{
				for(uint32_t i = 1; i < size; i++)
					data.at(i) += prob_mp::get_delta_m_4_3(base_update,data.at(i-1),m_max);
			}
			else
			{
				for(uint32_t i = 0; i < size; i++)
					data.at(i) += prob_mp::get_delta_m_4_3(base_update,(float)i*0.01f,m_max);
			}

			misc.resize(size).zero();
			misc.at(0) = m_init;
			for(uint32_t i = 1; i < size; i++)
			{
				float x = -(data.at(i-1)/m_max);
				float x2 = x*x;
				float x3 = x2*x;
				float x4 = x3*x;
				float z = 1.0f + (x2*0.5f) + (x3*0.16f) + (x4*0.0416);
				misc.at(i) += x * mul + offset + base_update;
			}

			ImGui::PushID("delta_m");
			ImGui::PushItemWidth(75);
			ImGui::InputFloat("Base Update",&base_update,0.0f,0.0f,"%.3f");
			ImGui::InputFloat("M Max",&m_max,0.0f,0.0f,"%.3f");
			ImGui::InputFloat("M Init",&m_init,0.0f,0.0f,"%.3f");
			ImGui::InputInt("Size",&size);
			size = std::clamp(size,1,1000);
			ImGui::Checkbox("Range",&range);
			ImGui::InputFloat("Mul",&mul);
			ImGui::InputFloat("Offset",&offset);
			ImGui::InputFloat("F0",&f0);
			ImGui::InputFloat("F1",&f1);
			ImGui::InputFloat("F2",&f2);
			ImGui::PopItemWidth();

			push_graph_vars(100,1);
			if(ImPlot::BeginPlot("Scores"))
			{
				ImPlot::PlotLine("D",data.data(),data.size());
				ImPlot::PlotLine("M",misc.data(),data.size());
				ImPlot::EndPlot();
			}
			for(i32 i = 0; i < 5; i++)
				ImPlot::PopStyleVar();
			ImGui::PopID();
		};
	} delta_m;

	void draw()
	{
		ImGui::Begin("Validator");
		delta_m.draw();
		ImGui::End();
	};
};

struct net_80_20_manager_t
{
	std::vector<snn5::n_block_config_t> neuron_configs = {};
	std::vector<snn5::s_block_config_t> s_block_configs = {};

	void step(snn5::custom_network_t* cn, snn5::network_t* const net)
	{
		net->n_blocks[0]->step();

		for(i32 i = 0; i < net->n_blocks.size(); i++)
		{
			net->n_blocks[i]->bp.grad.zero();
			net->n_blocks[i]->bp.err.zero();
			net->n_blocks[i]->bp.sd.zero();
		}

		for(i32 i = 1; i < net->n_blocks.size(); i++)
			net->n_blocks[i]->c.zero();
		
		net->s_blocks[0]->forward();
		net->s_blocks[1]->forward();
		
		net->n_blocks[1]->step();
		net->n_blocks[2]->step();
		
		net->s_blocks[2]->forward();
		net->s_blocks[3]->forward();

		net->n_blocks[3]->step();

		cn->temp_output.copy_from(net->get_output(3));

		if(net->state.is_training)
		{
			if(net->state.do_backprop && net->config.backprop.active)
				net->n_blocks[3]->calculate_error();

			net->s_blocks[2]->backward();
			net->s_blocks[3]->backward();
			net->s_blocks[0]->backward();
			net->s_blocks[1]->backward();

			net->s_blocks[2]->update_weights();
			net->s_blocks[3]->update_weights();
			net->s_blocks[0]->update_weights();
			net->s_blocks[1]->update_weights();
		}
	};

	snn5::custom_network_t* create_network()
	{
		snn5::custom_network_t* cn = new snn5::custom_network_t();
		snn5::network_t* net = new snn5::network_t();
		i32 ni  = net->create_neuron_block({784,1},snn5::MODEL_RNG);
		i32 n0p = net->create_neuron_block({400,1},snn5::MODEL_IZH);
		i32 n0n = net->create_neuron_block({100,1},snn5::MODEL_IZH);
		i32 no  = net->create_neuron_block({ 10,1},snn5::MODEL_IZH);
		i32 s0p = net->create_synapse_block(ni,n0p);
		i32 s0n = net->create_synapse_block(ni,n0n);
		i32 s1  = net->create_synapse_block(n0n,no);
		i32 s2  = net->create_synapse_block(n0p,no);

		cn->net->n_blocks = net->n_blocks;
		cn->net->s_blocks = net->s_blocks;

		net->get_synapse_block(n0p)->config.weight.use_global = false;
		net->get_synapse_block(n0p)->config.weight.i_min = 0.0f;
		net->get_synapse_block(n0p)->config.weight.r_min = 0.0f;

		net->get_synapse_block(n0p)->config.weight.use_global = false;
		net->get_synapse_block(n0n)->config.weight.i_max = 0.0f;
		net->get_synapse_block(n0n)->config.weight.r_max = 0.0f;

		//cn->n_blocks[i]->set_config()

		/*cn->load_test_data({
			{{0,test_input_data}},
			{{3,test_target_data}},0
		});
		cn->load_train_data({
			{{0,train_input_data}},
			{{3,train_target_data}},0
		});*/

		cn->net = net;
		cn->run_fn = std::bind(&net_80_20_manager_t::step,this,cn,net);
		net->reset();
		return cn;
	};
};

void step_80_20_net(snn5::custom_network_t* cn, snn5::network_t* const net)
{
	net->n_blocks[0]->step();

	for(i32 i = 0; i < net->n_blocks.size(); i++)
	{
		net->n_blocks[i]->bp.grad.zero();
		net->n_blocks[i]->bp.err.zero();
		net->n_blocks[i]->bp.sd.zero();
	}

	for(i32 i = 1; i < net->n_blocks.size(); i++)
		net->n_blocks[i]->c.zero();
	
	net->s_blocks[0]->forward();
	net->s_blocks[1]->forward();
	
	net->n_blocks[1]->step();
	net->n_blocks[2]->step();
	
	net->s_blocks[2]->forward();
	net->s_blocks[3]->forward();

	net->n_blocks[3]->step();

	cn->temp_output.copy_from(net->get_output(3));

	if(net->state.is_training)
	{
		if(net->state.do_backprop && net->config.backprop.active)
			net->n_blocks[3]->calculate_error();

		net->s_blocks[2]->backward();
		net->s_blocks[3]->backward();
		net->s_blocks[0]->backward();
		net->s_blocks[1]->backward();

		net->s_blocks[2]->update_weights();
		net->s_blocks[3]->update_weights();
		net->s_blocks[0]->update_weights();
		net->s_blocks[1]->update_weights();
	}
};

snn5::custom_network_t* create_80_20_net()
{
	snn5::custom_network_t* cn = new snn5::custom_network_t();
	snn5::network_t* net = new snn5::network_t();
	/*i32 ni  = net->create_neuron_block(784,snn5::MODEL_RNG);
	i32 n0p = net->create_neuron_block(400,snn5::MODEL_IZH);
	i32 n0n = net->create_neuron_block(100,snn5::MODEL_IZH);
	i32 no  = net->create_neuron_block( 10,snn5::MODEL_IZH);
	i32 s0p = net->create_synapse_block(ni,n0p);
	i32 s0n = net->create_synapse_block(ni,n0n);
	i32 s1  = net->create_synapse_block(n0n,no);
	i32 s2  = net->create_synapse_block(n0p,no);*/

	cn->net->n_blocks = net->n_blocks;
	cn->net->s_blocks = net->s_blocks;
	/*cn->load_test_data({
		{{0,test_input_data}},
		{{3,test_target_data}},0
	});
	cn->load_train_data({
		{{0,train_input_data}},
		{{3,train_target_data}},0
	});*/

	cn->run_fn = std::bind(&step_80_20_net,cn,net);
	return cn;
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
	bool mega_run = false;
	i32 mega_index = 0;
	float highest_mega_score = 0.0f;
	i32 highest_mega_index = 0;
	i32 max_threads = 48;
	std::atomic<i32> logs_loaded = 0;

	bool show_mts = false;

	snn5::network_info_t* net_info = nullptr;

	validator_t validator;

	std::jthread load_thread;

	controller_t()
	{
		if(net_manager == nullptr)
			net_manager = new snn5::network_manager_t();
		run_manager = new snn5::run_manager_t();

		builders.push_back(new snn5::feedforward_builder_t());
		builders.push_back(new snn5::teacher_builder_t());
		builders.push_back(new snn5::mux_builder_t());
		net_builder = builders[builder_index];

		temp_label.resize(32);

		cleanup_thread = std::jthread([this](){__cleanup();});

		auto add_gm = [&](std::string label, std::function<void()> fn)
		{
			group_managers.insert({label,new snn5::run_group_manager_t()});
			group_managers.at(label)->fn = fn;
			group_managers.at(label)->title = label;
		};

		add_gm("layer_sizes",std::bind(&controller_t::__run_layer_size_group,this));
		add_gm("learn_rates",std::bind(&controller_t::__run_learn_rate_group,this));
		add_gm("tgt_max_rates",std::bind(&controller_t::__run_tgt_max_group,this));
		add_gm("in_max_rates",std::bind(&controller_t::__run_in_max_group,this));
		
		add_gm("tri",std::bind(&controller_t::__run_tri_group,this));
		add_gm("fsiga",std::bind(&controller_t::__run_fsiga_group,this));
		add_gm("fsigb",std::bind(&controller_t::__run_fsigb_group,this));
		add_gm("exp",std::bind(&controller_t::__run_exp_group,this));
		add_gm("lsig",std::bind(&controller_t::__run_lsig_group,this));
		add_gm("sspike",std::bind(&controller_t::__run_sspike_group,this));
		add_gm("soft_relu",std::bind(&controller_t::__run_soft_relu_group,this));
		add_gm("unk",std::bind(&controller_t::__run_unk_group,this));

		add_gm("r_weights",std::bind(&controller_t::__run_r_weight_group,this));
		add_gm("tau_e",std::bind(&controller_t::__run_tau_e_group,this));
		add_gm("ema_alpha",std::bind(&controller_t::__run_ema_alpha_group,this));
		add_gm("ema_mul",std::bind(&controller_t::__run_ema_mul_group,this));
		add_gm("rbp_tau",std::bind(&controller_t::__run_rbp_tau_group,this));
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

	void __run_group(std::string label, float* ptr, std::vector<float> values)
	{
		snn5::run_group_manager_t* rgm = group_managers.at(label);
		snn5::run_group_t* rg = new snn5::run_group_t(rgm->temp_label);

		float old_value = *ptr;
		for(u32 i = 0; i < values.size(); i++)
		{
			*ptr = values[i];
			rg->add_run(dispatch_job(std::to_string(values[i]),false),values[i]);
		}
		*ptr = old_value;

		rgm->add_group(rg);
	};

	void __run_surrogate_group(std::string label, float* ptr, std::vector<float> values)
	{
		//u32 old_fn = net_manager->net_config.surrogate.function;
		__run_group(label,ptr,values);
		//net_manager->net_config.surrogate.function = old_fn;
	};

	void __run_layer_size_group()
	{
		snn5::run_group_manager_t* rgm = group_managers.at("layer_sizes");
		snn5::run_group_t* rg = new snn5::run_group_t(rgm->temp_label);

		std::vector<u32> sizes = {
			50,100,200,300,400,
			500,600,700,800,900,1000,
			1500,2000,2500,3000,4000,
			5000,6000,7000,8000,9000
		};

		u32 old_size = net_manager->layer_sizes[0];
		for(u32 i = 0; i < sizes.size(); i++)
		{
			net_manager->layer_sizes[0] = sizes[i];
			rg->add_run(dispatch_job(std::to_string(sizes[i]),false),sizes[i]);
		}
		net_manager->layer_sizes[0] = old_size;

		rgm->add_group(rg);
	};

	void __run_learn_rate_group()
	{
		std::vector<float> rates = {
			0.0100f,0.0075f,0.0050f,0.0040f,0.0030f,
			0.0020f,0.0015f,0.0012f,0.0011f,0.0010f,
			0.0009f,0.0008f,0.0007f,0.0006f,0.0005f
		};
		__run_group("learn_rates",&net_builder->layout.net_config.weight.learn_rate,rates);
	};

	void __run_tgt_max_group()
	{
		std::vector<float> rates = {2,3,4,5,8,10,15,20,25,30,40,50,60,75,100,125};
		__run_group("tgt_max_rates",&net_builder->layout.net_config.target.r_max,rates);
	};

	void __run_in_max_group()
	{
		std::vector<float> rates = {0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,1.00};
		__run_group("in_max_rates",&net_builder->layout.net_config.input.r_max,rates);
	};

	void __run_tri_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,
			0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 0;
		__run_group("tri",&net_builder->layout.net_config.surrogate.util.triangle_alpha,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_fsiga_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,
			0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 1;
		__run_group("fsiga",&net_builder->layout.net_config.surrogate.util.fast_sigmoid_alpha,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_fsigb_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,
			0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 2;
		__run_group("fsigb",&net_builder->layout.net_config.surrogate.util.fast_sigmoid_beta,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_exp_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,
			0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 3;
		__run_group("exp",&net_builder->layout.net_config.surrogate.util.exponential_alpha,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_lsig_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,
			0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 4;
		__run_group("lsig",&net_builder->layout.net_config.surrogate.util.logistic_sigmoid_alpha,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_sspike_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,
			0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0,2.5,3.0
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 5;
		__run_group("sspike",&net_builder->layout.net_config.surrogate.util.super_spike_beta,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_soft_relu_group()
	{
		std::vector<float> rates = {
			0.005,0.007,0.01,0.012,0.015,0.02,0.05,0.07,0.1,0.25,0.5,0.75,1.0,1.1,1.2
		};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 6;
		__run_group("soft_relu",&net_builder->layout.net_config.surrogate.util.soft_relu_beta,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_unk_group()
	{
		std::vector<float> rates = {1,5,10,15,20,25,30,40,50,60,70,80,90,100};
		u32 old_fn = net_builder->layout.net_config.surrogate.function;
		net_builder->layout.net_config.surrogate.function = 7;
		__run_group("unk",&net_builder->layout.net_config.surrogate.util.unknown_sigma,rates);
		net_builder->layout.net_config.surrogate.function = old_fn;
	};

	void __run_r_weight_group()
	{
		snn5::run_group_manager_t* rgm = group_managers.at("r_weights");
		snn5::run_group_t* rg = new snn5::run_group_t(rgm->temp_label);

		std::vector<float> weights = {1.0f,5.0f,10.0f,15.0f,20.0f,25.0f,30.0f,35.0f};

		float old_min = net_builder->layout.net_config.weight.r_min;
		float old_max = net_builder->layout.net_config.weight.r_max;
		for(u32 i = 0; i < weights.size(); i++)
		{
			net_builder->layout.net_config.weight.r_min = 0.0f - weights[i];
			net_builder->layout.net_config.weight.r_max = 0.0f + weights[i];
			rg->add_run(dispatch_job(std::to_string(weights[i]),false),weights[i]);
		}
		net_builder->layout.net_config.weight.r_min = old_min;
		net_builder->layout.net_config.weight.r_max = old_max;

		rgm->add_group(rg);
	};

	void __run_tau_e_group()
	{
		std::vector<float> taus = {10.0f,15.0f,20.0f,25.0f,30.0f,35.0f,40.0f,45.0f,50.0f,55.0f,60.0f,65.0f};
		__run_group("tau_e",&net_builder->layout.net_config.weight.trace_decay,taus);
	};

	void __run_ema_alpha_group()
	{
		std::vector<float> alphas = {0.960f,0.965f,0.970f,0.975f,0.980f,0.985f,0.990f,0.995f};
		__run_group("ema_alpha",&net_builder->layout.net_config.soma.spike.ema_alpha,alphas);
	};

	void __run_ema_mul_group()
	{
		std::vector<float> muls = {12.0f,16.0f,20.0f,24.0f,32.0f,40.0f,48.0f,56.0f,64.0f,72.0f,80.0f,96.0f};
		__run_group("ema_mul",&net_builder->layout.net_config.soma.spike.ema_mul,muls);
	};

	void __run_rbp_tau_group()
	{
		snn5::run_group_manager_t* rgm = group_managers.at("rbp_tau");
		snn5::run_group_t* rg = new snn5::run_group_t(rgm->temp_label);

		std::vector<float> muls = {
			0.00001f,0.00005f,
			0.0001,0.0005,
			0.001f,0.005f,
			0.01f,0.05f,
			0.1f,0.5f,
			1.0f,5.0f,
			10.0f,50.0f,
			100.0f,500.0f,
			1000.0f,5000.0f
		};

		float old_tau = net_builder->layout.net_config.rate_backprop.tau;
		for(u32 i = 0; i < muls.size(); i++)
		{
			net_builder->layout.net_config.rate_backprop.tau = muls[i];
			rg->add_run(dispatch_job(std::to_string(muls[i]),false),muls[i]);
		}
		net_builder->layout.net_config.rate_backprop.tau = old_tau;

		rgm->add_group(rg);
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

	void draw_run_group_panel(snn5::run_group_manager_t* rgm)
	{
		ImGui::PushID(rgm->title.c_str());
		push_graph_vars(100.0f,1.0f);
		ImPlot::SetNextAxisToFit(ImAxis_X1);
		if(ImPlot::BeginPlot("Scores##scsc"))
		{
			if(rgm->groups.size() > 0)
			{
				char** labels = new char*[rgm->groups[0]->runs.size()];
				std::vector<std::string> stupid = {};

				std::vector<double> vals = {};
				for(u32 i = 0; i < rgm->groups[0]->runs.size(); i++)
					vals.push_back(i);
				for(u32 i = 0; i < rgm->groups[0]->runs.size(); i++)
					stupid.push_back(std::to_string(rgm->groups[0]->params.at(i)));
				for(u32 i = 0; i < stupid.size(); i++)
				{
					bool has_dot = false;
					for(u32 j = 0; j < stupid.at(i).size(); j++)
					{
						if(stupid.at(i).at(j) == '.')
						{
							has_dot = true;
							break;
						}
					}

					if(has_dot)
						while(stupid.at(i).back() == '0')
							stupid.at(i).pop_back();
				}
				for(u32 i = 0; i < stupid.size(); i++)
					if(stupid.at(i).back() == '.')
						stupid.at(i).pop_back();
				for(u32 i = 0; i < rgm->groups[0]->runs.size(); i++)
					labels[i] = stupid.at(i).data();
				
				ImPlot::SetupAxisTicks(
					ImAxis_X1,
					vals.data(),
					rgm->groups[0]->runs.size(),
					labels
				);

				delete[] labels;
			}

			for(u32 i = 0; i < rgm->groups.size(); i++)
			{
				snn5::run_group_t* r = rgm->groups.at(i);
				r->update_scores();
				ImPlot::PlotLine(r->label.c_str(),r->scores.data(),r->runs.size());
			}

			ImPlot::EndPlot(); //program hangs here with asan enabled
		}
		pop_graph_vars();

		ImGui::SetNextItemWidth(100.0f);
		ImGui::InputText("##temp_label",rgm->temp_label.data(),32);
		ImGui::SameLine();
		if(ImGui::Button("Run"))
			rgm->fn();
		ImGui::SameLine();

		if(ImGui::Button("Delete All"))
		{
			for(u32 i = 0; i < rgm->groups.size(); i++)
				rgm->delete_group(rgm->groups.at(i));
		}

		snn5::run_group_t* group_to_remove = nullptr;
		for(u32 i = 0; i < rgm->groups.size(); i++)
		{
			ImGui::PushID(i);
			snn5::run_group_t* g = rgm->groups[i];
			ImGui::SetNextItemWidth(100.0f);
			ImGui::InputText("##label",g->label.data(),32);
			ImGui::SameLine();
			bool done = g->all_done();

			if(done) ImGui::BeginDisabled();
			ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0.6,0.1,0.1,1.0));
			if(ImGui::Button("[]")) g->stop();
			ImGui::PopStyleColor();
			if(done) ImGui::EndDisabled();
			ImGui::SameLine();
			if(ImGui::Button("X"))
				group_to_remove = g;
			ImGui::PopID();
		}

		if(group_to_remove != nullptr)
			rgm->delete_group(group_to_remove);
		ImGui::PopID();
	};

	void draw_run_panel()
	{
		ImGui::Begin("Results##14");

		ImGui::Checkbox("Show Mini Test Scores",&show_mts);

		ImGui::InputInt("Builder",&builder_index);
		builder_index = std::clamp<i32>(builder_index,0,builders.size()-1);
		net_builder = builders[builder_index];

		push_graph_vars(100.0f,1.0f);
		if(ImPlot::BeginPlot("Scores"))
		{
			//ImPlot::PlotLine("Train",run->train_scores.data(),run->train_index);
			for(i32 i = 0; i < background_runs.size(); i++)
			{
				snn5::run_t* r = background_runs.at(i);
				if(show_mts)
				{
					ImPlot::PlotLine(
						r->label.c_str(),
						r->test_results.scores.data(),
						//r->test_results.score_index
						10
					);
				}
				else
				{
					ImPlot::PlotLine(
						r->label.c_str(),
						r->train_results.scores.data(),
						r->train_results.score_index
					);
				}
			}

			if(show_primary_run && primary_run != nullptr)
			{
				snn5::run_t* r = primary_run;
				if(show_mts)
				{
					ImPlot::PlotLine(
						r->label.c_str(),
						r->test_results.scores.data(),
						10
					);
				}
				else
				{
					ImPlot::PlotLine(
						r->label.c_str(),
						r->train_results.scores.data(),
						r->train_results.score_index
					);
				}
			}

			ImPlot::EndPlot();
		}
		for(uint32_t i = 0; i < 5; i++)
			ImPlot::PopStyleVar();

		ImGui::InputText("##temp_label",temp_label.data(),32);
		ImGui::SameLine();
		if(ImGui::Button("Run##dispatcher"))
			dispatch_job(temp_label);
		if(ImGui::Button("Stop All"))
			stop_all_jobs();
		ImGui::SameLine();
		if(ImGui::Button("Delete All"))
			delete_all_jobs();

		std::vector<uint32_t> indices_to_remove = {};
		for(uint32_t i = 0; i < background_runs.size(); i++)
		{
			snn5::run_t* r = background_runs.at(i);
			bool busy = r->signals.status.is_working.load();
			bool r_done = r->signals.status.is_done.load() == 1;
			bool has_tested = r->test_results.completed;

			ImGui::PushID((std::to_string(i)+r->label).c_str());
			ImGui::SetNextItemWidth(100);
			ImGui::InputText("##title",r->label.data(),32);
			ImGui::SameLine();
			
			ImGui::BeginDisabled();
			ImGui::SetNextItemWidth(45);
			ImGui::InputFloat("##train_score",&r->train_results.live_score);
			ImGui::SameLine();
			ImGui::SetNextItemWidth(45);
			ImGui::InputFloat("##time",&r->train_results.seconds,0.0f,0.0f,"%.2f");
			ImGui::EndDisabled();
			ImGui::SameLine();

			if(!busy) ImGui::BeginDisabled();
			ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0.6,0.1,0.1,1.0));
			if(ImGui::Button("[]")) stop_job(r);
			ImGui::PopStyleColor();
			if(!busy) ImGui::EndDisabled();
			ImGui::SameLine();

			if(busy || has_tested) ImGui::BeginDisabled();
			if(ImGui::Button("Test")) __dispatch(r,false); //r->dispatch_test();
			if(busy || has_tested) ImGui::EndDisabled();
			ImGui::SameLine();

			ImGui::BeginDisabled();
			ImGui::SetNextItemWidth(45);
			ImGui::InputFloat("##test_score",&r->test_results.live_score);
			ImGui::EndDisabled();
			ImGui::SameLine();

			if(ImGui::Button("CFG"))
			{
				net_manager->run_config = r->run_config;
				net_manager->net_config = r->net->net->config;
				for(uint32_t j = 0; j < r->net->net->s_blocks.size(); j++)
					net_manager->layer_configs.at(j) = r->net->net->s_blocks.at(j)->config;
			}
			ImGui::SameLine();
			
			if(ImGui::Button("X"))
			{
				stop_job(r);
				indices_to_remove.push_back(i);
			}

			//ImGui::SameLine();
			//ImGui::Checkbox("More",&r->show_more);

			/*if(r->show_more)
			{
				u32 total_samples = r->config.samples * r->config.epochs;
				u32 steps = r->config.b_steps + r->config.v_steps + r->config.p_steps;

				ImGui::SameLine();
				ImGui::BeginDisabled();
				float samples_per_second = total_samples / r->total_seconds;
				ImGui::SetNextItemWidth(100);
				ImGui::InputFloat("Samples/s",&samples_per_second,0.0f,0.0f,"%12.2f");
				ImGui::SameLine();
				float total_steps = (steps * total_samples) / r->total_seconds;
				ImGui::SetNextItemWidth(100);
				ImGui::InputFloat("Steps/s",&total_steps,0.0f,0.0f,"%12.2f");
				ImGui::EndDisabled();
			}*/

			ImGui::PopID();
		}
		if(indices_to_remove.size() > 0)
		{
			auto ri = background_runs.begin()+indices_to_remove.at(0);
			snn5::run_t* r = (*ri);
			delete_job(r);
		}
		ImGui::End();
	};

	void draw()
	{
		//net_manager->draw();
		net_manager->run_config.draw();
		net_builder->draw();

		ImGui::Begin("NM");
		ImGui::Checkbox("Show Primary",&show_primary_run);
		bool primary_exists = primary_run != nullptr;
		bool primary_working = false;

		if(primary_exists)
			primary_working = primary_run->signals.status.is_working.load();

		if(primary_working)
			ImGui::BeginDisabled();
		if(ImGui::Button("Train"))
		{
			if(primary_exists)
			{
				/*if(primary_run->net->s_blocks.size() == net_manager->layer_configs.size())
					for(i32 i = 0; i < primary_run->net->s_blocks.size(); i++)
						primary_run->net->s_blocks[i]->config = net_manager->layer_configs[i];*/

				primary_run->reattach_logs();
				delete primary_run->net;
				primary_run->net = net_builder->create();
				primary_run->net->load_data(mnist::get_dataset());
				primary_run->signals.cmd.stop.store(0);
				primary_run->run_config = net_manager->run_config;
			}
			else
			{
				primary_run = new snn5::run_t(net_builder->create(),net_manager->run_config);
				primary_run->net->load_data(mnist::get_dataset());
				primary_run->label = "Main";
				//snn5::feedforward_network_t* n = net_manager->create_network();
				//snn5::run_t* r = new snn5::run_t(n,net_manager->run_config);
				//r->label = "Main";
				//primary_run = r;
			}

			__dispatch(primary_run,true);
		}
		if(primary_working)
			ImGui::EndDisabled();

		if(!primary_exists || !primary_working)
			ImGui::BeginDisabled();
		if(ImGui::Button("Stop"))
		{
			primary_run->signals.cmd.stop.store(1);
		}
		if(!primary_exists || !primary_working)
			ImGui::EndDisabled();

		/*if(ImGui::Button("Add Layer"))
		{
			net_manager->layer_sizes.push_back(100);
			net_manager->layer_configs.push_back({});
		}
		if(ImGui::Button("Remove Layer"))
		{
			net_manager->layer_sizes.pop_back();
			net_manager->layer_configs.pop_back();
		}
		for(u32 i = 0; i < net_manager->layer_sizes.size(); i++)
		{
			ImGui::PushID(i);
			i32 s = net_manager->layer_sizes.at(i);
			ImGui::InputInt("Size",&s,100);
			net_manager->layer_sizes.at(i) = s;
			ImGui::PopID();
		}*/
		ImGui::End();

		ImGui::Begin("Special Graphs");
		ImGui::PushID("special_graphs");
		if(ImGui::CollapsingHeader("Layer Sizes"))
			draw_run_group_panel(group_managers.at("layer_sizes"));
		if(ImGui::CollapsingHeader("Learn Rates"))
			draw_run_group_panel(group_managers.at("learn_rates"));
		if(ImGui::CollapsingHeader("V Steps"))
		{
			//tgt_max_rates
		}
		if(ImGui::CollapsingHeader("Target Max"))
			draw_run_group_panel(group_managers.at("tgt_max_rates"));
		if(ImGui::CollapsingHeader("Input Max"))
			draw_run_group_panel(group_managers.at("in_max_rates"));
		if(ImGui::CollapsingHeader("Surrogates"))
		{
			if(ImGui::CollapsingHeader("Tri"))
				draw_run_group_panel(group_managers.at("tri"));
			if(ImGui::CollapsingHeader("FSigA"))
				draw_run_group_panel(group_managers.at("fsiga"));
			if(ImGui::CollapsingHeader("FSigB"))
				draw_run_group_panel(group_managers.at("fsigb"));
			if(ImGui::CollapsingHeader("Exp"))
				draw_run_group_panel(group_managers.at("exp"));
			if(ImGui::CollapsingHeader("LSig"))
				draw_run_group_panel(group_managers.at("lsig"));
			if(ImGui::CollapsingHeader("Super Spike"))
				draw_run_group_panel(group_managers.at("sspike"));
			if(ImGui::CollapsingHeader("Soft Relu"))
				draw_run_group_panel(group_managers.at("soft_relu"));
			if(ImGui::CollapsingHeader("Unknown"))
				draw_run_group_panel(group_managers.at("unk"));
		}
		if(ImGui::CollapsingHeader("Runtime Weights"))
			draw_run_group_panel(group_managers.at("r_weights"));
		if(ImGui::CollapsingHeader("Tau E"))
			draw_run_group_panel(group_managers.at("tau_e"));
		if(ImGui::CollapsingHeader("EMA Alpha"))
			draw_run_group_panel(group_managers.at("ema_alpha"));
		if(ImGui::CollapsingHeader("EMA Mul"))
			draw_run_group_panel(group_managers.at("ema_mul"));
		if(ImGui::CollapsingHeader("RBP Tau"))
			draw_run_group_panel(group_managers.at("rbp_tau"));
		ImGui::PopID();
		ImGui::End();

		draw_run_panel();

		validator.draw();
		if(net_info != nullptr)
			net_info->draw();

		if(primary_run != nullptr)
		{
			primary_run->reattach_logs();
			if(primary_run->run_config.keep_logs)
			{
				for(i32 i = 0; i < primary_run->net->net->n_blocks.size(); i++)
					primary_run->n_logs[i]->draw();
				for(i32 i = 0; i < primary_run->net->net->s_blocks.size(); i++)
					primary_run->s_logs[i]->draw();
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

	bool jobs_running() const
	{
		for(i32 i = 0; i < background_runs.size(); i++)
			if(background_runs.at(i)->signals.status.is_done.load() != 1)
				return true;
		return false;
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

	i32 total_jobs_running()
	{
		i32 c = 0;
		for(i32 i = 0; i < background_runs.size(); i++)
			if(background_runs.at(i)->signals.status.is_done.load() != 1)
				c++;
		return c;
	};
};

int main(int argv, const char** argc)
{
	//openblas_set_num_threads(32);
	
	std::string base_path = "../rsc/mnist/";
	
	timer ti = timer();
	ti.start();

	bool no_ui = false;

	if(argv > 1)
	{
		no_ui = true;
	}

#ifdef DISABLE_LOGS
	no_ui = true;
#endif

	//init_mnist_sets(base_path);
	mnist::load_data(base_path);

	debug::start_logger();

	SetTraceLogLevel(LOG_ERROR);
	InitWindow(2160,1350,"machine");
	rlImGuiSetup(true);

	ImPlot::SetImGuiContext(rlImGuiGetContext());
	ImPlot::SetCurrentContext(ImPlot::CreateContext());
	SetTargetFPS(60);

	controller_t* ctrl = new controller_t;
	mnist::char_viewer_t cv;
	threshold_tester_t tr_test;
	rate_tester_t r_test;
	elig_tester_t e_test;
	izhd_tester_t id_test;
	conv_test_t conv_test;
	raf_tester_t raf_test;
	debug_printer_t debug_printer;
	//snn5::feedforward_builder_t ffwd_builder;

	ImGui::GetStyle().AntiAliasedLines = true;
	ImGui::GetStyle().AntiAliasedFill = true;
	ImGui::GetStyle().AntiAliasedLinesUseTex = true;
	ImGui::GetForegroundDrawList()->Flags |= ImDrawListFlags_AntiAliasedLines;
	ImGui::GetForegroundDrawList()->Flags |= ImDrawListFlags_AntiAliasedFill;

	bool should_exit = false;

	ti.stop();
	printf("Startup: %f\n",ti.get_seconds());

	while(true)
	{
		if(IsKeyDown(KEY_ESCAPE))
		{
			ctrl->shutdown();
			ctrl->should_exit = true;
			should_exit = true;
		}

		BeginDrawing();
		ClearBackground(BLACK);

		rlImGuiBegin();

		if(!should_exit)
		{
			ctrl->draw();
			//cv.draw(train_set8,train_set);
			tr_test.draw();
			r_test.draw();
			e_test.draw();
			id_test.draw();
			raf_test.draw();
			//ffwd_builder.draw();
			conv_test.draw();
			debug_printer.draw();
		}

		rlImGuiEnd();
		EndDrawing();

		if(should_exit)
			break;
	}

	printf("quit\n");

	delete ctrl; ctrl = nullptr;

	mnist::free_data();

	/*if(train_set8 != nullptr)
		delete train_set8;
	if(test_set8 != nullptr)
		delete test_set8;

	if(train_set != nullptr)
		delete train_set;
	if(test_set != nullptr)
		delete test_set;*/

	debug::stop_logger();

	return 0;
};