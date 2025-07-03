#include "log.h"
#ifdef BUILD_UI
#include "imgui.h"
#endif

namespace snn5
{
	void base_log_t::plot(i32 height, float y_min, float y_max)
	{
#ifdef BUILD_UI
		ImGui::PushID(label.c_str());
		ImGui::SetNextItemWidth(65);
		ImGui::InputInt("Offset",&offset);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(65);
		ImGui::InputInt("Max Visible",&max_visible);

		i32 x = log.values.x();
		i32 y = log.values.y();

		max_visible = std::clamp<i32>(max_visible,0,y-1);
		offset = std::clamp<i32>(offset,0,y-max_visible-1);

		ImPlot::PushStyleVar(ImPlotStyleVar_PlotDefaultSize,{200.0f,(float)height});
		ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding,{2.0f,2.0f});
		ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding,{2.0f,2.0f});
		ImPlot::PushStyleVar(ImPlotStyleVar_LegendSpacing,{1.0f,1.0f});
		ImPlot::PushStyleVar(ImPlotStyleVar_LegendInnerPadding,{1.0f,1.0f});
		ImPlot::SetNextAxisLimits(ImAxis_X1,0.0f,1000.0f);
		ImPlot::SetNextAxisLimits(ImAxis_Y1,y_min,y_max);

		if(ImPlot::BeginPlot((label+"##"+std::to_string(0)).c_str()))
		{
			ImPlot::SetupAxisLimits(ImAxis_X1,0.0f,1000.0f);
			ImPlot::SetupAxis(ImAxis_X1,0,
				ImPlotAxisFlags_AutoFit|
				ImPlotAxisFlags_NoTickLabels
			);
			for(i32 i = offset; i < std::min<i32>(y,offset+max_visible); i++)
				ImPlot::PlotLine(
					std::to_string(i).c_str(),
					log.data(i),1000,1,0,0,i*x,
					sizeof(float)*y
				);
			ImPlot::EndPlot();
		}
			
		ImPlot::PopStyleVar();
		ImPlot::PopStyleVar();
		ImPlot::PopStyleVar();
		ImPlot::PopStyleVar();
		ImPlot::PopStyleVar();
		ImGui::PopID();
#endif		
	};

	void n_log_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin(("n_log: "+std::to_string(nb->id)).c_str());
		ImGui::SetNextItemWidth(64);
		ImGui::InputInt("Height",&height);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(64);
		ImGui::InputInt("Merge",&merge);
		merge = std::clamp<i32>(merge,0,10000);
		//ImGui::SameLine();
		ImGui::Checkbox("v",&v.active); ImGui::SameLine();
		ImGui::Checkbox("c",&c.active); ImGui::SameLine();
		//ImGui::Checkbox("ema",&ema.active); ImGui::SameLine();
		//ImGui::Checkbox("bit",&bit.active); ImGui::SameLine();
		ImGui::Checkbox("trace",&trace.active); ImGui::SameLine();
		ImGui::Checkbox("grad",&grad.active); ImGui::SameLine();
		ImGui::Checkbox("err",&err.active); ImGui::SameLine();
		ImGui::Checkbox("sd",&sd.active);

		if(v.active)
			v.plot(height,-90,35);
		if(c.active)
			c.plot(height,0,100);
		//if(ema.active)
		//	ema.plot(height,0,1000);
		//if(bit.active)
		//	bit.plot(height,0,60);
		if(trace.active)
			trace.plot(height,0,100);
		if(grad.active)
			grad.plot(height,-10,10);
		if(err.active)
			err.plot(height,-10,10);
		if(sd.active)
			sd.plot(height,-10,10);

		ImGui::End();
#endif
	};

	void s_log_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin(("s_log: "+std::to_string(sb->prev->id)+"_"+std::to_string(sb->next->id)).c_str());
		ImGui::InputInt("Height",&height);
		ImGui::SameLine();
		ImGui::InputInt("Merge",&merge);
		ImGui::SameLine();
		ImGui::Checkbox("w",&w->active); ImGui::SameLine();
		ImGui::Checkbox("wr",&wr->active); ImGui::SameLine();
		ImGui::Checkbox("et",&et->active);

		if(w->active)
			w->plot(height,-20,20);
		if(wr->active)
			wr->plot(height,-20,20);
		if(et->active)
			et->plot(height,0,50);

		ImGui::End();
#endif
	};
};

/*
	perhaps incremental learning uses scratch space
	where the target selects where in a large network
	to place the learned features, and during rest the
	features are coalesced into a more efficient clump

	using the target to define the spatial
	location of learning might be a clever idea.
	unique targets would naturally chose unique
	locations in the network for learning.

	using apical "backprop" might further help
	select unique locations in the network.

	the more a target is active the more the selected
	neurons and synapses should become plastic

	backprop normally can't do incremental learning
	because it touches everything, including
	neurons that are related to other skills
*/