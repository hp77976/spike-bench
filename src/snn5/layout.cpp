#include "layout.h"
#ifdef BUILD_UI
#include <imgui.h>
#endif

namespace snn5
{
	void n_entry_t::draw(i32 id)
	{
#ifdef BUILD_UI
		ImGui::PushID((std::to_string(id)+"n_entry").c_str());
		ImGui::PushItemWidth(20);
		ImGui::BeginDisabled();
		ImGui::InputInt("##id",&id,0,0);
		ImGui::EndDisabled();
		ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0.5,0.2,0.2,1.0));
		if(ImGui::Button("N Config"))
			show_config = !show_config;
		ImGui::PopStyleColor();
		ImGui::SameLine();
		ImGui::SetNextItemWidth(60);
		//ImGui::InputInt("Size",&shape.x,0,0);
		i32 s[2];
		s[0] = shape.x;
		s[1] = shape.y;
		ImGui::InputInt2("Shape",s);
		shape.x = s[0];
		shape.y = s[1];
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};
	
	void s_entry_t::draw(i32 id)
	{
#ifdef BUILD_UI
		ImGui::PushID((std::to_string(id)+"s_entry").c_str());
		ImGui::PushItemWidth(20);
		ImGui::BeginDisabled();
		ImGui::InputInt("##id",&id,0,0);
		ImGui::EndDisabled();
		ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0.1,0.5,0.2,1.0));
		if(ImGui::Button("S Config"))
			show_config = !show_config;
		ImGui::PopStyleColor();
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};
};