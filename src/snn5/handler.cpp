#include "handler.h"
#ifdef BUILD_UI
#include "imgui.h"
#include "../rlimgui/rlImGui.h"
#include "../implot/implot.h"
#endif

namespace snn5
{
	void run_t::draw(run_control_t &r_ctrl)
	{
#ifdef BUILD_UI
		bool busy = signals.status.is_working.load();
		bool r_done = signals.status.is_done.load() == 1;
		bool has_tested = full_test_score.completed;

		ImGui::PushID(label.c_str());
		ImGui::SetNextItemWidth(100);
		ImGui::InputText("##label",label.data(),32);
		ImGui::SameLine();

		ImGui::BeginDisabled();
		ImGui::SetNextItemWidth(45);
		ImGui::InputFloat("##train_score",&train_score.live_score);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(45);
		ImGui::InputFloat("##train_time",&train_score.seconds,0.0f,0.0f,"%.2f");
		ImGui::EndDisabled();
		ImGui::SameLine();

		if(!busy) ImGui::BeginDisabled();
		ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0.6,0.1,0.1,1.0));
		if(ImGui::Button("[]")) signals.cmd.stop.store(1);
		ImGui::PopStyleColor();
		if(!busy) ImGui::EndDisabled();
		ImGui::SameLine();

		if(busy || has_tested) ImGui::BeginDisabled();
		//if(ImGui::Button("Test")) run_full_test();
		if(busy || has_tested) ImGui::EndDisabled();
		//ImGui::SameLine();

		ImGui::BeginDisabled();
		ImGui::SetNextItemWidth(45);
		ImGui::InputFloat("##test_score",&full_test_score.live_score);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(45);
		ImGui::InputFloat("##test_time",&full_test_score.seconds,0.0f,0.0f,"%.2f");
		ImGui::EndDisabled();
		ImGui::SameLine();

		if(ImGui::Button("CFG")) r_ctrl.load_config = true;
		ImGui::SameLine();

		if(ImGui::Button("X"))
		{
			signals.cmd.stop.store(1);
			r_ctrl.del = true;
		}

		ImGui::PopID();
#endif
	};

	void run_panel_t::draw(i32 id)
	{
#ifdef BUILD_UI
		ImGui::Begin(("Run Panel "+std::to_string(id)).c_str());
		ImGui::SetNextItemWidth(65);
		ImGui::Combo("Result Type",&result_selection,result_items);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(45);
		ImGui::InputInt("Graph Height",&graph_height);
		ImGui::SetNextItemWidth(45);
		ImGui::InputInt("Graph Width",&graph_width);

		if(ImPlot::BeginPlot("Scores"))
		{
			for(i32 i = 0; i < runs.size(); i++)
			{
				run_t* r = runs.at(i);
				r->fit_scores(graph_width);
				float* score = nullptr;
				switch(result_selection)
				{
					default:
					case(0): score = r->train_score.adjusted_score.data(); break;
					case(1): score = r->mini_test_score.adjusted_score.data(); break;
					case(2): score = r->full_test_score.adjusted_score.data(); break;
				}

				ImPlot::PlotLine(r->label.c_str(),score,graph_width);
			}
			ImPlot::EndPlot();
		}

		ImGui::InputText("##temp_label",temp_label.data(),32);
		ImGui::SameLine();
		if(ImGui::Button("Run"))
		{
			run_t* r = create_run_fn();

		}
		if(ImGui::Button("Stop All"))
		{

		}
		if(ImGui::Button("Delete All"))
		{

		}

		i32 index_to_remove = -1;
		for(i32 i = 0; i < runs.size(); i++)
		{
			run_control_t r_ctrl;
			run_t* r = runs[i];
			ImGui::PushID((std::to_string(i)+r->label).c_str());
			r->draw(r_ctrl);
			ImGui::PopID();

			if(r_ctrl.del)
				index_to_remove = i;
		}
		if(index_to_remove > 0)
		{
			
		}

		ImGui::End();
#endif
	};
};