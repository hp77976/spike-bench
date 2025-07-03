#include "ui.h"
#ifdef BUILD_UI
#include "imgui.h"
#endif
#include "snn.h"
#include "synapses.h"
#include "log.h"

namespace snn5
{
	void run_config_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Run Config");
		ImGui::InputInt("Epochs",&epochs);
		ImGui::InputInt("Samples",&samples);
		ImGui::InputInt("P Steps",&p_steps);
		ImGui::InputInt("V Steps",&v_steps);
		//ImGui::InputInt("T Steps",&t_steps);
		ImGui::InputInt("B Steps",&b_steps);
		epochs = std::clamp<i32>(epochs,1,50);
		samples = std::clamp<i32>(samples,1,60000);
		p_steps = std::clamp<i32>(p_steps,0,1000);
		v_steps = std::clamp<i32>(v_steps,0,1000);
		//t_steps = std::clamp<i32>(t_steps,0,1000);
		b_steps = std::clamp<i32>(b_steps,0,1000);
		//ImGui::Checkbox("Use GPU",&use_gpu);
		ImGui::Checkbox("Sequential",&sequential);
		ImGui::Checkbox("Use Target Input",&use_target_input);
		ImGui::Checkbox("Auto Test",&auto_test);
		ImGui::Checkbox("Use Mini Test",&use_mini_test);
		//ImGui::Checkbox("Backprop gets Input",&backprop_gets_input);
		//ImGui::Checkbox("Backprop Max",&backprop_max);
		ImGui::Checkbox("Keep Logs",&keep_logs);
		ImGui::End();
#endif
	};

	void network_manager_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Run Config##14");
		run_config.draw();
		ImGui::End();
		ImGui::Begin("Net Config##14");
		net_config.draw();
		ImGui::End();
		for(i32 i = 0; i < layer_configs.size(); i++)
			layer_configs.at(i).draw(i);
#endif
	};

	bool run_t::__step(bool training)
	{
		u32 total_steps = 0;

		net->zero_inputs();
		net->zero_targets();
		net->zero_grad();
		net->zero_counts();
		
		for(i32 i = 0; i < run_config.p_steps; i++, total_steps++)
		{
			net->step(training,false);
			for(i32 j = 0; j < std::min(net->net->n_blocks.size(),n_logs.size()); j++)
				n_logs[j]->collect();
			for(i32 j = 0; j < std::min(net->net->s_blocks.size(),s_logs.size()); j++)
				s_logs[j]->collect();
		}

		net->set_inputs(training);
		net->set_targets(training);
		/*if(run_config.use_target_input && training)
			net->set_input(target,input.size());*/

		for(i32 i = 0; i < run_config.v_steps; i++, total_steps++)
		{
			net->step(training,false);
			for(i32 j = 0; j < std::min(net->net->n_blocks.size(),n_logs.size()); j++)
				n_logs[j]->collect();
			for(i32 j = 0; j < std::min(net->net->s_blocks.size(),s_logs.size()); j++)
				s_logs[j]->collect();
		}

		/*if(run_config.t_steps > 0)
		{
			net->zero_input();
			for(u32 i = 0; i < run_config.t_steps; i++, total_steps++)
			{
				net->step(training,false);
				for(i32 j = 0; j < std::min(net->n_blocks.size(),n_logs.size()); j++)
					n_logs[j]->collect();
				for(i32 j = 0; j < std::min(net->s_blocks.size(),s_logs.size()); j++)
					s_logs[j]->collect();
			}
		}*/

		//if(run_config.backprop_gets_input)
		{
			//net->set_input(input);
			/*if(run_config.use_target_input && training)
				net->set_input(target,input.size());*/
		}

		//if(run_config.backprop_max)
		//	net->input.rate.fill(1.0f);

		bool correct = net->get_output().highest_index() == net->get_target().highest_index();

		for(i32 i = 0; i < run_config.b_steps; i++, total_steps++)
		{
			net->step(training,true);
			for(i32 j = 0; j < std::min(net->net->n_blocks.size(),n_logs.size()); j++)
				n_logs[j]->collect();
			for(i32 j = 0; j < std::min(net->net->s_blocks.size(),s_logs.size()); j++)
				s_logs[j]->collect();
		}

		net->next_data(training);

		return correct;
	};

	run_graph_t::run_graph_t()
	{

	};

	run_graph_t::~run_graph_t()
	{
		while(trash_runs.size() > 0)
		{
			if(trash_runs.back()->signals.status.is_done.load() == 1)
			{
				delete trash_runs.back();
				trash_runs.pop_back();
			}
		}

		while(runs.size() > 0)
		{
			if(runs.back()->signals.status.is_done.load() == 1)
			{
				delete runs.back();
				runs.pop_back();
			}
		}
	};

	void run_graph_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Run Results##57");

		ImGui::SetNextItemWidth(65);
		ImGui::Combo("##results",&result_selection,result_items);
		ImGui::SameLine();
		ImGui::SetNextItemWidth(95);
		ImGui::InputInt("Height",&graph_height);
		/*ImGui::SetNextItemWidth(65);
		ImGui::InputInt("Width",&graph_width);*/

		i32 index_to_remove = -1;
		for(i32 i = 0; i < runs.size(); i++)
		{
			run_t* r = runs.at(i);
			bool is_busy = r->signals.status.is_working || !r->signals.status.is_done;

			ImGui::PushID(("run id: "+std::to_string(i)).c_str());
			ImGui::InputText("##label",r->label.data(),32);

			if(!is_busy) ImGui::BeginDisabled();
			if(ImGui::Button("[]")) r->signals.cmd.stop.store(1);
			if(!is_busy) ImGui::EndDisabled();

			if(ImGui::Button("X##delete"))
				index_to_remove = i;
			ImGui::PopID();
		}

		ImGui::End();

		if(index_to_remove != -1)
		{
			auto ri = runs.begin()+index_to_remove;
			(*ri)->signals.cmd.stop.store(1);
			trash_runs.push_back(*ri);
			runs.erase(ri);
		}

		if(trash_runs.size() > 0)
		{
			if(trash_runs.back()->signals.status.is_done)
			{
				run_t* r = trash_runs.back();
				delete r->net;
				delete r;
				trash_runs.pop_back();
			}
		}
#endif
	};
};