#include "conf.h"
#ifdef BUILD_UI
#include "imgui.h"
#endif
#include "snn.h"
#include "../util/misc.h"
#include "synapses.h"

namespace snn5
{
	bool str_eq(std::string a, std::string b)
	{
		if(a.length() != b.length())
			return false;

		for(uint32_t i = 0; i < a.length(); i++)
			if(a[i] != b[i])
				return false;

		return true;
	};

	std::vector<std::string> str_break(std::string str)
	{
		std::vector<std::string> strings = {};
		bool has_separator = false;
		uint32_t separator_pos = 0;
		for(uint32_t i = 0; i < str.length(); i++)
		{
			if(str[i] == '.')
			{
				has_separator = true;
				separator_pos = i;
				break;
			}
		}

		if(has_separator)
		{
			std::string pre;
			for(uint32_t i = 0; i < separator_pos; i++)
				pre.push_back(str[i]);

			std::string post;
			for(uint32_t i = separator_pos + 1; i < str.length(); i++)
				post.push_back(str[i]);

			return {pre,post};
		}
		else
		{
			return {str};
		}
	};

	void learnable_param_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("learnable_param_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::PushItemWidth(50);
		ImGui::InputFloat("iMin",&i_min,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::InputFloat("iMax",&i_max,0.0f,0.0f,"%3.2f");
		ImGui::InputFloat("rMin",&r_min,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::InputFloat("rMax",&r_max,0.0f,0.0f,"%3.2f");
		ImGui::SetNextItemWidth(75);
		ImGui::InputFloat("Learn Rate",&learn_rate,0.0f,0.0f,"%6.6f");
		ImGui::Checkbox("##enable_trace",&enable_trace);
		ImGui::SameLine();
		ImGui::InputFloat("Trace Decay",&trace_decay,0.0f,0.0f,"%3.2f");
		ImGui::Checkbox("Xe Init",&use_xe_init);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void spike_data_config_t::draw()
	{
#ifdef BUILD_UI
		ImGui::PushID("spike_data_conf");
		ImGui::PushItemWidth(45);
		ImGui::InputFloat("EMA Alpha",&ema_alpha,0.0f,0.0f,"%6.6f");
		ImGui::InputFloat("EMA Mul",&ema_mul,0.0f,0.0f,"%2.2f");
		ImGui::InputFloat("Trace Decay",&trace_decay,0.0f,0.0f,"%2.2f");
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};

	void adjust_t::draw()
	{
#ifdef BUILD_UI
		ImGui::PushID("adjust_config");
		ImGui::PushItemWidth(45);
		ImGui::Checkbox("Clamp",&enable_clamp);
		ImGui::SameLine();
		if(!enable_clamp)
			ImGui::BeginDisabled();
		ImGui::InputFloat("Min",&v_min,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::InputFloat("Max",&v_max,0.0f,0.0f,"%3.2f");
		if(!enable_clamp)
			ImGui::EndDisabled();
		ImGui::InputFloat("Mul",&mul,0.0f,0.0f,"%3.2f");
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};

	void backprop_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("backprop_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		ImGui::Checkbox("Active",&active);
		ImGui::SameLine();
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::PushItemWidth(85);
		ImGui::Combo("##value",&value_selection,value_items);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void gradient_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("gradient_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		adjust.draw();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void error_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("output_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		adjust.draw();
		ImGui::PushItemWidth(65);
		ImGui::Combo("##rate",&rate_selection,rate_items);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void surrogate_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("surrogate_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		adjust.draw();
		ImGui::PushItemWidth(65);
		ImGui::Combo("##input",&input_selection,input_items);
		ImGui::SetNextItemWidth(150);
		ImGui::Combo("Fn",&function,util.long_names);
		i32 &fn_id = function;
		auto &u = util;
		ImGui::InputFloat(
			(u.get_param_name(fn_id)+"##cfg_bp_"+u.get_short_name(fn_id)).c_str(),
			&u.get_value(fn_id),0.0f,0.0f,"%4.4f"
		);
		ImGui::Text("Default: %4.4f",u.get_default(fn_id));
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void rate_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("rate_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::PushItemWidth(50);
		ImGui::InputFloat("Min",&r_min,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::InputFloat("Max",&r_max,0.0f,0.0f,"%3.2f");
		ImGui::SameLine();
		ImGui::Checkbox("Sum",&use_sum);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void output_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("output_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::PushItemWidth(85);
		ImGui::Combo("##rate",&rate_selection,rate_items);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void metaplasticity_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("metaplasticity_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		ImGui::Checkbox("Active",&active);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::Checkbox("Require Weight Update",&require_weight_update);
		ImGui::Checkbox("Use Dynamic Alpha",&use_dynamic_alpha);
		ImGui::Checkbox("Use Post Spike Decay",&use_post_spike_for_decay);
		ImGui::PushItemWidth(45);
		ImGui::InputFloat("Init M",&init_m,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Max M",&max_m,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Pre Th1",&pre_th1,0.0f,0.0f,"%.3f");
		ImGui::SameLine();
		ImGui::InputFloat("Post Th1",&post_th1,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Pre Th2",&pre_th2,0.0f,0.0f,"%.3f");
		ImGui::SameLine();
		ImGui::InputFloat("Post Th2",&post_th2,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Base Update",&base_update,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Base Alpha",&base_alpha,0.0f,0.0f,"%.4f");
		ImGui::InputFloat("Decay Alpha",&decay_alpha,0.0f,0.0f,"%.4f");
		ImGui::InputFloat("T Ref",&t_ref,0.0f,0.0f,"%.1f");
		ImGui::InputFloat("s_i",&s_i,0.0f,0.0f,"%.2f");
		ImGui::Checkbox("D_M Hack",&use_dm_hack);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void stp_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("stp_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		ImGui::Checkbox("Active",&active);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::PushItemWidth(75);
		ImGui::InputFloat("U",&u,0.0f,0.0f,"%5.3f");
		ImGui::InputFloat("Tau U",&tau_u,0.0f,0.0f,"%5.3f");
		ImGui::InputFloat("Tau X",&tau_x,0.0f,0.0f,"%5.3f");
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void stdp_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("stp_config");
		if(show_global)
			ImGui::Checkbox("Use Global",&use_global);
		ImGui::Checkbox("Active",&active);
		if(use_global && show_global)
			ImGui::BeginDisabled();
		ImGui::Checkbox("On Weights",&use_on_weights);
		ImGui::Checkbox("On Feedback",&use_on_feedback);
		ImGui::PushItemWidth(75);
		ImGui::InputFloat("Alpha LTP",&alpha_ltp,0.0f,0.0f,"%5.3f");
		ImGui::InputFloat("Alpha LTD",&alpha_ltd,0.0f,0.0f,"%5.3f");
		ImGui::InputFloat("Tau LTP",&tau_ltp,0.0f,0.0f,"%5.3f");
		ImGui::InputFloat("Tau LTD",&tau_ltd,0.0f,0.0f,"%5.3f");
		ImGui::InputFloat("Scale",&scale,0.0f,0.0f,"%5.3f");
		ImGui::Checkbox("Use LUT",&use_lut);
		ImGui::PopItemWidth();
		if(use_global && show_global)
			ImGui::EndDisabled();
		ImGui::PopID();
#endif
	};

	void soma_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("neuron_config");
		ImGui::PushItemWidth(75);
		//ImGui::Checkbox("Use Current Decay",&use_current_decay);
		ImGui::Checkbox("##use_current_decay",&use_current_decay);
		ImGui::SameLine();
		ImGui::InputFloat("Current Decay",&current_decay,0.0f,0.0f,"%3.1f");
		
		
		//ImGui::Checkbox("Adjust EMA Rate",&adjust_ema_rate);
		//ImGui::InputFloat("EMA Mul",&ema_mul,0.0f,0.0f,"%6.6f");
		ImGui::Checkbox("Use Max V",&use_max_v);
		ImGui::SetNextItemWidth(45);
		ImGui::InputFloat("Min V",&min_v,0.0f,0.0f,"%3.1f");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(45);
		ImGui::InputFloat("Max V",&max_v,0.0f,0.0f,"%3.1f");
		/*ImGui::InputFloat("Rate Alpha",&spike.ema_alpha,0.0f,0.0,"%.6f");
		ImGui::InputFloat("Trace Decay",&spike.trace_decay,0.0f,0.0,"%3.1f");*/
		ImGui::Checkbox("V Jitter",&use_v_jitter);
		ImGui::Checkbox("Tau T Use Exp",&tau_t_use_exp);
		ImGui::Checkbox("Tau C Use Exp",&tau_c_use_exp);
		ImGui::PopItemWidth();
		ImGui::PushID("input");
		if(ImGui::CollapsingHeader("Input"))
			input.draw(show_global);
		ImGui::PopID();
		ImGui::PushID("target");
		if(ImGui::CollapsingHeader("Target"))
			target.draw(show_global);
		ImGui::PopID();
		if(ImGui::CollapsingHeader("Output"))
			output.draw(show_global);
		if(ImGui::CollapsingHeader("Gradient"))
			gradient.draw(show_global);
		if(ImGui::CollapsingHeader("Surrogate"))
			surrogate.draw(show_global);
		if(ImGui::CollapsingHeader("Error"))
			error.draw(show_global);
		ImGui::PopID();
#endif
	};

	void lif_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("lif_config");
		ImGui::PushItemWidth(75);
		ImGui::InputFloat("V Rest",&v_rest,0.0f,0.0f,"%3.5ff");
		ImGui::InputFloat("V Reset",&v_reset,0.0f,0.0f,"%3.5ff");
		ImGui::InputFloat("Tr",&tr,0.0f,0.0f,"%3.5ff");
		ImGui::InputFloat("Resist",&resistance,0.0f,0.0f,"%3.5ff");
		ImGui::InputInt("Refractory",&refractory_period);
		ImGui::InputInt("Current Decay",&current_decay);
		ImGui::Checkbox("Keep High V",&keep_high_v);
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};

	void izh_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("izh_config");
		ImGui::Indent();
		if(ImGui::CollapsingHeader("Params"))
		{
			ImGui::PushItemWidth(75);
			ImGui::InputFloat("A",&a,0.0f,0.0f,"%.3f");
			ImGui::InputFloat("B",&b,0.0f,0.0f,"%.3f");
			ImGui::InputFloat("C",&c,0.0f,0.0f,"%.1f");
			ImGui::InputFloat("D",&d,0.0f,0.0f,"%.1f");
			ImGui::InputFloat("Tr",&tr,0.0f,0.0f,"%.1f");
			ImGui::InputFloat("Tr2",&tr2,0.0f,0.0f,"%.1f");
			ImGui::Checkbox("Keep High V",&keep_high_v);
			ImGui::Checkbox("Handle V Reset",&handle_v_reset);
			ImGui::PopItemWidth();
		}
		ImGui::Unindent();
		ImGui::PopID();
#endif
	};

	void rng_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("rng_config");
		ImGui::PushItemWidth(75);
		ImGui::InputFloat("Mul",&multiplier,0.0f,0.0f,"%5.3f");
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};

	void raf_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("raf_config");
		ImGui::PushItemWidth(75);
		ImGui::InputFloat("Frequency",&frequency,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Beta",&beta,0.0f,0.0f,"%.3f");
		ImGui::InputFloat("Threshold",&threshold,0.0f,0.0f,"%.3f");
		ImGui::Checkbox("Set V Reset",&use_specific_v_reset);
		ImGui::InputFloat("V Reset",&v_reset,0.0f,0.0f,"%.3f");
		ImGui::Checkbox("Keep High V",&keep_high_v);
		ImGui::PopItemWidth();
		ImGui::PopID();
#endif
	};

	void s_block_config_t::draw(i32 id, bool in_window)
	{
#ifdef BUILD_UI
		if(in_window)
			ImGui::Begin(("S Block " + std::to_string(id) + " Config").c_str());
		ImGui::PushID("s_block_config");
		ImGui::PushID("weight");
		if(ImGui::CollapsingHeader("Weight"))
			weight.draw(true);
		ImGui::PopID();
		ImGui::PushID("feedback");
		if(ImGui::CollapsingHeader("Feedback"))
			feedback.draw(true);
		ImGui::PopID();
		if(ImGui::CollapsingHeader("Backprop"))
			backprop.draw(true);
		if(ImGui::CollapsingHeader("STDP"))
			stdp.draw(true);
		if(ImGui::CollapsingHeader("STP"))
			stp.draw(true);
		if(ImGui::CollapsingHeader("Metaplasticity"))
			metaplasticity.draw(true);
		ImGui::Combo("Type",&type,type_items);
		
		if(type == 1)
		{
			i32 kx = kernel.size.x;
			i32 ky = kernel.size.y;
			i32 px = kernel.padding;
			i32 sx = kernel.stride;
			ImGui::InputInt("kX",&kx);
			ImGui::InputInt("kY",&ky);
			ImGui::InputInt("Pad",&px);
			ImGui::InputInt("Stride",&sx);
			kernel.size.x = std::clamp(kx,0,100);
			kernel.size.y = std::clamp(ky,0,100);
			kernel.padding = px;
			kernel.stride = sx;
		}
		ImGui::PopID();
		if(in_window)
			ImGui::End();
#endif
	};

	void n_block_config_t::draw(bool show_global)
	{
#ifdef BUILD_UI
		ImGui::PushID("compartment_config");
		n_config.draw(show_global);
		ImGui::Combo("Model",&model,model_items);
		switch(model)
		{
			case(MODEL_LIF): lif.draw(show_global); break;
			case(MODEL_IZH): izh.draw(show_global); break;
			case(MODEL_RNG): rng.draw(show_global); break;
			case(MODEL_RAF): raf.draw(show_global); break;
			//case(MODEL_BRF): brf.draw(); break;
		}
		spike.draw();
		ImGui::PopID();
#endif
	};

	void network_config_t::draw()
	{
#ifdef BUILD_UI
		ImGui::PushID("network_config");
		ImGui::SeparatorText("Input");
		ImGui::PushID("input_config");
		input.draw(false);
		ImGui::PopID();
		ImGui::SeparatorText("Target");
		ImGui::PushID("target_config");
		target.draw(false);
		ImGui::SeparatorText("Output");
		ImGui::PushID("output_config");
		output.draw(false);
		ImGui::PopID();
		ImGui::PopID();
		if(ImGui::CollapsingHeader("Apical"))
		{
			ImGui::PushID("apical_config");
			ImGui::Checkbox("Active",&apical.active);
			bool apical_active = apical.active;
			if(!apical_active)
				ImGui::BeginDisabled();
			apical.draw(false);
			if(!apical_active)
				ImGui::EndDisabled();
			ImGui::PopID();
		}
		if(ImGui::CollapsingHeader("Soma"))
		{
			ImGui::PushID("soma_config");
			soma.draw(false);
			ImGui::PopID();
		}
		if(ImGui::CollapsingHeader("Weight"))
		{
			ImGui::PushID("weight_config");
			weight.draw(false);
			ImGui::PopID();
		}
		if(ImGui::CollapsingHeader("Feedback"))
		{
			ImGui::PushID("feedback_config");
			feedback.draw(false);
			ImGui::PopID();
		}
		if(ImGui::CollapsingHeader("Error"))
		{
			error.draw(false);
		}
		if(ImGui::CollapsingHeader("Surrogate"))
		{
			surrogate.draw(false);
		}
		if(ImGui::CollapsingHeader("Gradient"))
		{
			gradient.draw(false);
		}
		if(ImGui::CollapsingHeader("Backprop"))
		{
			backprop.draw(false);
		}
		if(ImGui::CollapsingHeader("Dendrites"))
		{
			ImGui::PushID("dendrites_config");
			ImGui::Checkbox("Active",&dendrites.active);
			ImGui::Checkbox("Apical Error",&dendrites.use_apical_e);
			ImGui::Checkbox("Apical SD",&dendrites.use_apical_sd);
			ImGui::Checkbox("Update Y",&dendrites.update_y);
			ImGui::PopID();
		}
		if(ImGui::CollapsingHeader("Metaplasticity"))
		{
			metaplasticity.draw(false);
		}
		if(ImGui::CollapsingHeader("Testing"))
		{
			ImGui::PushID("testing_config");
			ImGui::Checkbox("Use DMP",&dmp.active);
			ImGui::Checkbox("Target Based Eligibility",&dmp.use_target_synapse_selection);
			ImGui::Checkbox("Enable Apical Spikes",&apical.active);
			ImGui::PopID();
		}

		ImGui::Checkbox("Enable GPU",&use_gpu);
		ImGui::Checkbox("Use NP Layout",&use_np_layout);
		ImGui::Checkbox("Output No Spike",&output_no_spike);
		ImGui::Checkbox("EMA Selection",&use_ema_selection);
		ImGui::Checkbox("RBP Active",&rate_backprop.active);
		ImGui::Checkbox("RBP Tau Use Sigmoid",&rate_backprop.tau_use_sigmoid);
		ImGui::Checkbox("RBP Manual Tau",&rate_backprop.manual_tau);
		ImGui::Checkbox("RBP Hard Reset",&rate_backprop.hard_reset);
		ImGui::Checkbox("RBP Detach",&rate_backprop.detach);
		ImGui::SetNextItemWidth(75.0f);
		ImGui::InputFloat("RBP Tau",&rate_backprop.tau,0.0f,0.0f,"%.5f");
		ImGui::Checkbox("Reverse Spikes",&enable_reverse_spikes);
		ImGui::Checkbox("Event STDP",&use_event_stdp);
		ImGui::Checkbox("Accumulate Grad",&accumulate_grad);
		ImGui::PopID();
#endif
	};

	void network_config_t::set(std::string param, float value)
	{
		std::vector<std::string> strings = str_break(param);

		if(str_eq(strings[0],""))
		{

		}
		else if(str_eq(param,"surrogate.soft_relu.beta"))
		{
			surrogate.util.soft_relu_beta = value;
		}
		else
		{
			printf("[network_config_t] unknown param: %s\n",param.c_str());
			if(strings.size() > 0)
				printf("string[0]: %s\n",strings[0].c_str());
			if(strings.size() > 1)
				printf("string[1]: %s\n",strings[1].c_str());
		}
	};
};