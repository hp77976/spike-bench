#pragma once
#ifdef BUILD_UI
#include "imgui.h"
#endif
#include "snn.h"
#include "../util/misc.h"

namespace snn5
{
	struct network_info_t
	{
		network_t* net = nullptr;

		network_info_t(network_t* const n) : net(n)
		{

		};

		void draw()
		{
#ifdef BUILD_UI
			ImGui::Begin("Network Info");
			
			//ImGui::Text("Layer Count: %lu\n",net->layers.size());

			std::string layout = "";
			/*for(uint32_t i = 0; i < net->sizes.size(); i++)
				layout += "/" + std::to_string(net->sizes.at(i));*/
			//layout.erase(layout.front());

			ImGui::Text("Layout: %s",layout.c_str());

			int64_t neuron_count = 0;
			int64_t neuron_bytes = 0;
			int64_t synapse_count = 0;
			int64_t synapse_bytes = 0;

			/*for(uint32_t i = 0; i < net->layers.size(); i++)
			{
				neuron_count += net->layers.at(i)->neuron_count();
				neuron_bytes += net->layers.at(i)->neuron_bytes();
				synapse_count += net->layers.at(i)->synapse_count();
				synapse_bytes += net->layers.at(i)->synapse_bytes();
			}*/

			int64_t total_neuron_memory = neuron_bytes * neuron_count;
			int64_t total_synapse_memory = synapse_bytes * synapse_count;
			int64_t total_memory = total_neuron_memory + total_synapse_memory;

			ImGui::Separator();

			ImGui::Text("Neuron Count: %li",neuron_count);
			ImGui::Text("Neuron Bytes: %li",neuron_bytes);
			ImGui::Text("Total Neuron Memory: %li bytes",total_neuron_memory);

			ImGui::Separator();

			ImGui::Text("Synapse Count: %li",synapse_count);
			ImGui::Text("Synapse Bytes: %li",synapse_bytes);
			ImGui::Text("Total Synapse Memory: %li bytes",total_synapse_memory);

			ImGui::Separator();

			ImGui::Text("Total Network Memory: %li bytes",total_memory);
			ImGui::Text("%li KB",total_memory/1024);
			ImGui::Text("%li MB",total_memory/1024/1024);

			ImGui::Text("Logs and other overhead memory not accounted for.");

			ImGui::End();
#endif
		};
	};
};