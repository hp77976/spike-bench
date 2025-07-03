#include "base.h"
#ifdef BUILD_UI
#include <imgui.h>
#endif

namespace snn5
{
	/*void learnable_param_config_t::draw(bool show_global)
	{
		ImGui::PushItemWidth(75);
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		ImGui::InputFloat("iMin",&i_min,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::InputFloat("iMax",&i_max,0.0f,0.0f,"%3.2f");
		ImGui::InputFloat("rMin",&r_min,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::InputFloat("rMax",&r_max,0.0f,0.0f,"%3.2f");
		ImGui::InputFloat("Learn Rate",&learn_rate,0.0f,0.0f,"%3.5f");
		ImGui::Checkbox("Enable Trace",&enable_trace);
		ImGui::InputFloat("Trace Tau",&trace_decay,0.0f,0.0f,"%3.2f");
		ImGui::PopItemWidth();
	};*/
};