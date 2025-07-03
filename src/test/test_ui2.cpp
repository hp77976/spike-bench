#include "../snn5/snn.h"
#include "../snn5/ui.h"
#include "../util/mnist3.h"
#include "../rlimgui/rlImGui.h"
#include "../implot/implot.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <raylib.h>

struct izhd_log_t
{
	math::array<float> c;
	math::array<float> v;
	math::array<float> u;
	math::array<float> c2;
	math::array<float> v2;
	math::array<float> u2;

	bool show_c = true;
	bool show_v = true;
	bool show_u = true;
	bool show_c2 = true;
	bool show_v2 = true;
	bool show_u2 = true;

	snn5::izhd_block_t* target = nullptr;

	izhd_log_t(snn5::izhd_block_t* ib, i32 log_size)
	{
		target = ib;
		c.resize(ib->size(),log_size).zero();
		v.resize(ib->size(),log_size).zero();
		u.resize(ib->size(),log_size).zero();
		c2.resize(ib->size(),log_size).zero();
		v2.resize(ib->size(),log_size).zero();
		u2.resize(ib->size(),log_size).zero();
	};

	void collect(i32 time)
	{
		for(i32 i = 0; i < target->size(); i++)
		{
			c.at(i,time) = target->c[i];
			v.at(i,time) = target->v[i];
			u.at(i,time) = target->u[i];
			c2.at(i,time) = target->c2[i];
			v2.at(i,time) = target->v2[i];
			u2.at(i,time) = target->u2[i];
		}
	};

	void draw()
	{
		if(ImPlot::BeginPlot("Izhd Logs"))
		{

		}
	};
};

struct controller_t
{
	//snn5::feedforward_builder_t* ffwd_builder = nullptr;
	snn5::run_graph_t* run_graph = nullptr;
	snn5::run_config_t run_config;

	controller_t()
	{
		//ffwd_builder = new snn5::feedforward_builder_t();
		run_graph = new snn5::run_graph_t();
	};

	~controller_t()
	{
		//delete ffwd_builder; ffwd_builder = nullptr;
		delete run_graph; run_graph = nullptr;
	};

	void draw()
	{
		run_config.draw();
		//ffwd_builder->draw();
		run_graph->draw();

		ImGui::Begin("Temp Runner");
		if(ImGui::Button("Run"))
		{
			//snn5::run_t* r = new snn5::run_t(
				//ffwd_builder->create(),
			//	run_config
			//);

			//run_graph->runs.push_back(r);
		}//
		ImGui::End();
	};
};

int main(int argv, const char** argc)
{
	std::string base_path = "../rsc/mnist/";
	timer ti = timer();
	ti.start();

	mnist::load_data(base_path);

	SetTraceLogLevel(LOG_ERROR);
	InitWindow(2160,1350,"machine");
	rlImGuiSetup(true);

	ImPlot::SetImGuiContext(rlImGuiGetContext());
	ImPlot::SetCurrentContext(ImPlot::CreateContext());
	SetTargetFPS(60);

	ImGui::GetStyle().AntiAliasedLines = true;
	ImGui::GetStyle().AntiAliasedFill = true;
	ImGui::GetStyle().AntiAliasedLinesUseTex = true;
	ImGui::GetForegroundDrawList()->Flags |= ImDrawListFlags_AntiAliasedLines;
	ImGui::GetForegroundDrawList()->Flags |= ImDrawListFlags_AntiAliasedFill;

	bool should_exit = false;
	controller_t ctrl;

	ti.stop();
	printf("Startup: %f\n",ti.get_seconds());

	while(true)
	{
		if(IsKeyDown(KEY_ESCAPE))
			should_exit = true;

		BeginDrawing();
		ClearBackground(BLACK);

		rlImGuiBegin();

		if(!should_exit)
			ctrl.draw();

		rlImGuiEnd();
		EndDrawing();

		if(should_exit)
			break;
	}

	mnist::free_data();

	return 0;
};