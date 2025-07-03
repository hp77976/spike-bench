#include "../math/array.h"
#include "../rlimgui/rlImGui.h"
#include "../implot/implot.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <raylib.h>

struct log_t
{
	math::array<float> old_values;
	i32 index = 0;

	log_t()
	{

	};

	log_t(i32 source_size, i32 log_size)
	{
		old_values.resize(log_size,source_size).zero();
		index = 0;
	};

	void collect(const math::array<float> &src)
	{
		for(i32 i = 0; i < old_values.y(); i++)
			old_values.at(index,i) = src.at(i);
		index++;
	};

	void reset()
	{
		old_values.zero();
		index = 0;
	};
};

struct graph_test_t
{
	log_t log;
	math::array<float> values;
	i32 value_count = 5;
	i32 hist_length = 4;
	i32 stride = 4;
	i32 offset = 4;
	i32 ptr_os = 1;

	graph_test_t()
	{
		values.resize(value_count).zero();
		log.old_values.resize(hist_length,value_count);
	};

	void run()
	{
		log.old_values.resize(hist_length,value_count);
		log.reset();
		values.resize(value_count).zero();
		for(i32 i = 0; i < hist_length; i++)
		{
			for(i32 j = 0; j < value_count; j++)
				values[j] = (0.1f * (i+1)) + j;
			log.collect(values);
		}
	};

	void draw()
	{
		//stride goes to value count (y)
		//offset goes to hist length (x)

		ImGui::Begin("Graph Test");
		if(ImGui::Button("Run"))
			run();
		ImGui::InputInt("VC",&value_count);
		ImGui::InputInt("HL",&hist_length);
		ImGui::InputInt("Stride",&stride);
		ImGui::InputInt("Offset",&offset);
		ImGui::InputInt("ptr_os",&ptr_os);
		if(ImPlot::BeginPlot("Logs"))
		{
			for(i32 i = 0; i < value_count; i++)
			{
				std::string label = std::to_string(i);
				ImPlot::PlotLine(
					label.c_str(),&log.old_values.m_ptr[i*ptr_os],
					hist_length,1,0,0,offset*i,sizeof(float)*stride
				);
			}
			ImPlot::EndPlot();
		}
		for(i32 i = 0; i < hist_length; i++)
		{
			std::string s;
			for(i32 j = 0; j < value_count; j++)
				s += std::to_string(log.old_values.at(i,j)) + " ";
			ImGui::PushID(i+99);
			ImGui::Text("%s",s.c_str());
			ImGui::PopID();
		}
		ImGui::End();
	};
};

int main()
{
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

	graph_test_t gt;

	bool should_exit = false;

	while(true)
	{
		if(IsKeyDown(KEY_ESCAPE))
		{
			should_exit = true;
		}

		BeginDrawing();
		ClearBackground(BLACK);

		rlImGuiBegin();

		if(!should_exit)
		{
			gt.draw();
		}

		rlImGuiEnd();
		EndDrawing();

		if(should_exit)
			break;
	}

	return 0;
};