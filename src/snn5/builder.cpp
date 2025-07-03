#include "builder.h"
#ifdef BUILD_UI
#include "imgui.h"
#endif
#include "wrappers.h"
#include "snn.h"

namespace snn5
{
	network_builder_t::network_builder_t()
	{

	};

	network_builder_t::~network_builder_t()
	{

	};

	feedforward_builder_t::feedforward_builder_t() : network_builder_t()
	{
		label.resize(32);
		layout.n_entries.resize(3);
		layout.s_entries.resize(2);
		layout.n_entries[0].shape = {784,1};
		layout.n_entries[0].config.model = MODEL_RNG;
		layout.n_entries[0].config.use_global = false;
		layout.n_entries[1].shape = {500,1};
		layout.n_entries[2].shape = { 10,1};
	};

	void feedforward_builder_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Feedforward Layout");
		
		if(ImGui::Button("Net Config"))
			layout.show_net_config = !layout.show_net_config;
		ImGui::SameLine();
		if(ImGui::Button("+##adder"))
		{
			layout.n_entries.push_back({});
			layout.s_entries.push_back({});
		}
		ImGui::SameLine();
		i32 n_c = layout.n_entries.size();
		if(n_c < 3)
			ImGui::BeginDisabled();
		if(ImGui::Button("-##remover"))
		{
			layout.n_entries.pop_back();
			layout.s_entries.pop_back();
		}
		if(n_c < 3)
			ImGui::EndDisabled();
		for(i32 i = 0; i < layout.n_entries.size(); i++)
		{
			layout.n_entries[i].draw(i);
			if(i < layout.s_entries.size())
				layout.s_entries[i].draw(i);
		}
		ImGui::End();

		if(layout.show_net_config)
		{
			ImGui::Begin("FFWD Net Config",&layout.show_net_config);
			layout.net_config.draw();
			ImGui::End();
		}

		for(i32 i = 0; i < layout.n_entries.size(); i++)
		{
			if(layout.n_entries[i].show_config)
			{
				ImGui::Begin(
					("F N Block Config: "+std::to_string(i)).c_str(),
					&layout.n_entries[i].show_config
				);
				layout.n_entries[i].config.draw(true);
				ImGui::End();
			}
		}

		for(i32 i = 0; i < layout.s_entries.size(); i++)
		{
			if(layout.s_entries[i].show_config)
			{
				ImGui::Begin(
					("F S Block Config: "+std::to_string(i)).c_str(),
					&layout.s_entries[i].show_config
				);
				layout.s_entries[i].config.draw(true,false);
				ImGui::End();
			}
		}
#endif
	};

	void feedforward_builder_t::load_config(network_wrapper_t* nw)
	{

	};

	network_wrapper_t* feedforward_builder_t::create() const
	{
		return new feedforward_network_t(layout);
	};

	teacher_builder_t::teacher_builder_t()
	{
		label.resize(32);
		layout.n_entries.resize(3);
		layout.s_entries.resize(2);
		layout.n_entries[0].shape = {784,1};
		layout.n_entries[0].config.model = MODEL_RNG;
		layout.n_entries[1].shape = {500,1};
		layout.n_entries[2].shape = {10,1};
		
		layout.s_entries[0].config.stdp.active = true;
		
		broadcast_n.shape = {10,1};
		broadcast_n.config.model = MODEL_RNG;
		broadcast_s.config.backprop.active = false;
	};

	void teacher_builder_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Teacher Layout");
		
		if(ImGui::Button("Net Config"))
			layout.show_net_config = !layout.show_net_config;
				
		for(i32 i = 0; i < layout.n_entries.size(); i++)
		{
			layout.n_entries[i].draw(i);
			if(i < layout.s_entries.size())
				layout.s_entries[i].draw(i);
		}

		ImGui::Checkbox("Broadcast to Output",&broadcast_to_output);
		broadcast_n.draw(-1);
		broadcast_s.draw(-1);

		ImGui::End();

		if(layout.show_net_config)
		{
			ImGui::Begin("Teacher Net Config",&layout.show_net_config);
			layout.net_config.draw();
			ImGui::End();
		}

		for(i32 i = 0; i < layout.n_entries.size(); i++)
		{
			if(layout.n_entries[i].show_config)
			{
				ImGui::Begin(
					("T N Block Config: "+std::to_string(i)).c_str(),
					&layout.n_entries[i].show_config
				);
				layout.n_entries[i].config.draw(true);
				ImGui::End();
			}
		}

		for(i32 i = 0; i < layout.s_entries.size(); i++)
		{
			if(layout.s_entries[i].show_config)
			{
				ImGui::Begin(
					("T S Block Config: "+std::to_string(i)).c_str(),
					&layout.s_entries[i].show_config
				);
				layout.s_entries[i].config.draw(true,false);
				ImGui::End();
			}
		}

		if(broadcast_n.show_config)
		{
			ImGui::Begin("Broadcast N Config",&broadcast_n.show_config);
			broadcast_n.config.draw(true);
		}

		if(broadcast_s.show_config)
		{
			ImGui::Begin("Broadcast S Config",&broadcast_s.show_config);
			broadcast_s.config.draw(true,false);
		}
#endif
	};

	network_wrapper_t* teacher_builder_t::create() const
	{
		return new teacher_network_t(layout,broadcast_n,broadcast_s,broadcast_to_output);
	};

	void teacher_builder_t::load_config(network_wrapper_t* nw)
	{

	};

	mux_builder_t::mux_builder_t() : network_builder_t()
	{
		label.resize(32);
		layout.n_entries.resize(3);
		layout.s_entries.resize(2);
		layout.n_entries[0].shape = {784,1};
		layout.n_entries[0].config.model = MODEL_RNG;
		layout.n_entries[0].config.use_global = false;
		layout.n_entries[1].shape = {500,1};
		layout.n_entries[2].shape = { 10,1};
	};

	void mux_builder_t::draw()
	{
#ifdef BUILD_UI
		ImGui::Begin("Multiplex Layout");

		ImGui::PushItemWidth(80);
		ImGui::InputInt("In Mul",&input_mul);
		ImGui::InputInt("Out Mul",&output_mul);
		ImGui::InputInt("Out Mode",&output_mode);
		input_mul = std::clamp<i32>(input_mul,1,10);
		output_mul = std::clamp<i32>(output_mul,1,10);
		output_mode = std::clamp<i32>(output_mode,0,1);
		ImGui::PopItemWidth();
		
		if(ImGui::Button("Net Config"))
			layout.show_net_config = !layout.show_net_config;
		ImGui::SameLine();
		if(ImGui::Button("+##adder"))
		{
			layout.n_entries.push_back({});
			layout.s_entries.push_back({});
		}
		ImGui::SameLine();
		i32 n_c = layout.n_entries.size();
		if(n_c < 3)
			ImGui::BeginDisabled();
		if(ImGui::Button("-##remover"))
		{
			layout.n_entries.pop_back();
			layout.s_entries.pop_back();
		}
		if(n_c < 3)
			ImGui::EndDisabled();
		for(i32 i = 0; i < layout.n_entries.size(); i++)
		{
			layout.n_entries[i].draw(i);
			if(i < layout.s_entries.size())
				layout.s_entries[i].draw(i);
		}
		ImGui::End();

		if(layout.show_net_config)
		{
			ImGui::Begin("FFWD Net Config",&layout.show_net_config);
			layout.net_config.draw();
			ImGui::End();
		}

		for(i32 i = 0; i < layout.n_entries.size(); i++)
		{
			if(layout.n_entries[i].show_config)
			{
				ImGui::Begin(
					("F N Block Config: "+std::to_string(i)).c_str(),
					&layout.n_entries[i].show_config
				);
				layout.n_entries[i].config.draw(true);
				ImGui::End();
			}
		}

		for(i32 i = 0; i < layout.s_entries.size(); i++)
		{
			if(layout.s_entries[i].show_config)
			{
				ImGui::Begin(
					("F S Block Config: "+std::to_string(i)).c_str(),
					&layout.s_entries[i].show_config
				);
				layout.s_entries[i].config.draw(true,false);
				ImGui::End();
			}
		}
#endif
	};

	void mux_builder_t::load_config(network_wrapper_t* nw)
	{

	};

	network_wrapper_t* mux_builder_t::create() const
	{
		return new mux_network_t(layout,input_mul,output_mul,output_mode);
	};
};