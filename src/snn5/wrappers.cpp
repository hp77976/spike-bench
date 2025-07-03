#include "wrappers.h"
#include "neurons.h"
#include "synapses.h"
#include "../util/debug.h"

namespace snn5
{
	network_wrapper_t::~network_wrapper_t()
	{

	};

	void network_wrapper_t::reset()
	{
		data.train_index = 0;
		data.test_index = 0;
		if(net != nullptr)
			net->reset();
	};

	void network_wrapper_t::zero_inputs()
	{
		for(i32 i = 0; i < net->n_blocks.size(); i++)
			net->n_blocks[i]->c.zero();
	};

	void network_wrapper_t::zero_targets()
	{
		for(i32 i = 0; i < net->n_blocks.size(); i++)
			net->n_blocks[i]->target.zero();
	};

	void network_wrapper_t::zero_grad()
	{
		for(i32 i = 0; i < net->n_blocks.size(); i++)
			net->n_blocks[i]->bp.grad.zero();
	};

	void network_wrapper_t::zero_counts()
	{
		for(i32 i = 0; i < net->n_blocks.size(); i++)
			net->n_blocks[i]->spike.count.zero();
	};

	custom_network_t::~custom_network_t()
	{
		
	};

	void custom_network_t::step(bool is_training, bool do_backprop)
	{
		net->state.is_training = is_training;
		net->state.do_backprop = do_backprop;
		run_fn();
		net->state.time++;
	};

	math::array<float> custom_network_t::get_output() const
	{
		return net->get_output(net->n_blocks.back()->id);
	};

	math::array<float> custom_network_t::get_target() const
	{
		return net->get_neuron_block(net->n_blocks.back()->id)->target;
	};

	feedforward_network_t::feedforward_network_t(net_layout_t layout)
	{
		net = new network_t();
		net->config = layout.net_config;
		sizes = {};
		for(i32 i = 0; i < layout.n_entries.size(); i++)
			sizes.push_back(layout.n_entries[i].shape);

		for(i32 i = 0; i < layout.n_entries.size(); i++)
			net->create_neuron_block(sizes[i],layout.n_entries[i].config);
		for(i32 i = 0; i < layout.s_entries.size(); i++)
			net->create_synapse_block(i,i+1,layout.s_entries[i].config);

		input_id = net->n_blocks.front()->id;
		target_id = net->n_blocks.back()->id;

		is_set_up = true;
	};

	feedforward_network_t::~feedforward_network_t()
	{
		if(net != nullptr)
			delete net;
	};

	void feedforward_network_t::step(bool is_training, bool do_backprop)
	{
		net->state.is_training = is_training;
		net->state.do_backprop = do_backprop;

		net->n_blocks[0]->step();
		debug::log("ffw_step","step n["+str(net->n_blocks[0]->id)+"]");
		debug::log("ffw_step_data","n["+str(net->n_blocks[0]->id)+"].c",net->n_blocks[0]->c);

		for(i64 i = 1; i < net->n_blocks.size(); i++)
		{
			n_block_t* nb = net->n_blocks[i];
			nb->bp.err.zero();
			nb->bp.sd.zero();
			debug::log("ffw_step","zero n["+str(nb->id)+"] err,sd");
		}

		for(i64 i = 0; i < net->s_blocks.size(); i++)
		{
			s_block_t* sb = net->s_blocks[i];
			n_block_t* nb = sb->next;
			nb->c.zero();
			debug::log("ffw_step","zero s["+str(sb->id)+"]->next["+str(nb->id)+"] c");
		}

		for(i64 i = 0; i < net->s_blocks.size(); i++)
		{
			s_block_t* sb = net->s_blocks[i];
			n_block_t* nb = sb->next;
			sb->forward();
			nb->step();
			debug::log("ffw_step","fwd s["+str(sb->id)+"]");
			debug::log("ffw_step","step n["+str(nb->id)+"]");
			debug::log("ffw_step_data","n["+str(nb->id)+"].c",nb->c);
		}

		temp_output.copy_from(net->get_output(target_id));
		debug::log("ffw_step","copy temp output from n["+str(target_id)+"]");
		debug::log("ffw_step_data","temp_output",temp_output);

		if(net->state.is_training)
		{
			if(net->config.backprop.active && (net->config.accumulate_grad || do_backprop))
			{
				n_block_t* nb = net->n_blocks.back();
				nb->calculate_error();
				debug::log("ffw_step","calc_err n["+str(nb->id)+"]");
				debug::log("ffw_step_data","n["+str(nb->id)+"].err",nb->bp.err);
			}

			//do not run backward on first s_block, it does nothing but waste time!
			if(net->config.backprop.active && (net->config.accumulate_grad || do_backprop))
			{
				for(auto i = net->s_blocks.rbegin(); i != net->s_blocks.rend() - 1; i++)
				{
					(*i)->backward();
					debug::log("ffw_step","back s["+str((*i)->id)+"]");
					debug::log("ffw_step_data","n["+str((*i)->prev->id)+"].err",(*i)->prev->bp.err);
				}
			}

			if(net->config.backprop.active && do_backprop)
			{
				for(auto i = net->s_blocks.begin(); i != net->s_blocks.end(); i++)
				{
					(*i)->update_weights();
					debug::log("ffw_step","upw s["+str((*i)->id)+"]");
				}
			}
		}

		net->state.time++;
	};

	math::array<float> feedforward_network_t::get_output() const
	{
		return net->get_output(target_id);
	};

	math::array<float> feedforward_network_t::get_target() const
	{
		return net->get_neuron_block(target_id)->target;
	};

	void feedforward_network_t::set_inputs(bool is_training)
	{
		if(is_training)
			net->set_input(data.train_samples[data.train_index].input,0);
		else
			net->set_input(data.test_samples[data.test_index].input,0);
	};

	void feedforward_network_t::set_targets(bool is_training)
	{
		if(is_training)
			net->set_target(data.train_samples[data.train_index].target,net->n_blocks.back()->id);
		else
			net->set_target(data.test_samples[data.test_index].target,net->n_blocks.back()->id);
	};

	teacher_network_t::teacher_network_t(
		net_layout_t layout,
		n_entry_t broadcast_n,
		s_entry_t broadcast_s,
		bool broadcast_to_output
	)
	{
		net = new network_t();
		net->config = layout.net_config;
		sizes = {};
		for(i64 i = 0; i < layout.n_entries.size(); i++)
			sizes.push_back(layout.n_entries[i].shape);

		for(i64 i = 0; i < layout.n_entries.size(); i++)
			net->create_neuron_block(sizes[i],layout.n_entries[i].config);
		for(i64 i = 0; i < layout.s_entries.size(); i++)
			net->create_synapse_block(i,i+1,layout.s_entries[i].config);

		input_id = net->n_blocks.front()->id;
		output_id = net->n_blocks.back()->id;

		target_id = net->create_neuron_block(broadcast_n.shape,broadcast_n.config);
		net->create_synapse_block(target_id,1,broadcast_s.config);

		is_set_up = true;
	};

	teacher_network_t::~teacher_network_t()
	{
		if(net != nullptr)
			delete net;
	};

	void teacher_network_t::step(bool is_training, bool do_backprop)
	{
		net->state.is_training = is_training;
		net->state.do_backprop = do_backprop;

		net->n_blocks[0]->step();
		net->n_blocks[3]->step();

		for(i64 i = 1; i < net->n_blocks.size(); i++)
		{
			//net->n_blocks[i]->bp.grad.zero();
			net->n_blocks[i]->bp.err.zero();
			net->n_blocks[i]->bp.sd.zero();
		}

		net->n_blocks[1]->get_c().zero();
		net->n_blocks[2]->get_c().zero();

		net->s_blocks[0]->forward();
		net->s_blocks.back()->forward();

		net->n_blocks[1]->step();

		net->s_blocks[1]->forward();
		net->s_blocks[1]->next->step();

		temp_output.copy_from(net->get_output(output_id));

		if(net->state.is_training)
		{
			if(net->state.do_backprop && net->config.backprop.active)
				net->n_blocks[2]->calculate_error();

			net->s_blocks[1]->backward();

			net->s_blocks[0]->update_weights();
			net->s_blocks[1]->update_weights();
		}

		net->state.time++;
	};

	math::array<float> teacher_network_t::get_output() const
	{
		return net->get_output(output_id);
	};

	math::array<float> teacher_network_t::get_target() const
	{
		return net->get_neuron_block(output_id)->target;
	};

	void teacher_network_t::set_inputs(bool is_training)
	{
		if(is_training)
			net->set_input(data.train_samples[data.train_index].input,0);
		else
			net->set_input(data.test_samples[data.test_index].input,0);

		if(is_training)
			net->set_input(data.train_samples[data.train_index].target,target_id);
		else
			net->get_neuron_block(target_id)->get_c().zero();
	};

	void teacher_network_t::set_targets(bool is_training)
	{
		if(is_training)
			net->set_target(data.train_samples[data.train_index].target,output_id);
		else
			net->set_target(data.test_samples[data.test_index].target,output_id);
	};

	mux_network_t::mux_network_t(
		net_layout_t layout,
		i32 input_mul,
		i32 output_mul,
		i32 output_mode_
	)
	{
		output_mode = output_mode_;
		net = new network_t();
		net->config = layout.net_config;
		sizes = {};
		for(i32 i = 0; i < layout.n_entries.size(); i++)
			sizes.push_back(layout.n_entries[i].shape);

		for(i32 i = 0; i < input_mul; i++)
			in_n_ids.push_back(net->create_neuron_block(sizes[0],layout.n_entries[0].config));
		for(i32 i = 1; i < layout.n_entries.size() - 1; i++)
			mid_n_ids.push_back(net->create_neuron_block(sizes[i],layout.n_entries[i].config));
		for(i32 i = 0; i < output_mul; i++)
			out_n_ids.push_back(net->create_neuron_block(sizes.back(),layout.n_entries.back().config));
		
		for(i32 i = 0; i < input_mul; i++)
		{
			i32 s_id = net->create_synapse_block(in_n_ids[i],input_mul,layout.s_entries.front().config);
			in_s.push_back(net->get_synapse_block(s_id));
		}
		
		for(i32 i = 0; i < mid_n_ids.size() - 1; i++)
		{
			i32 s_id = net->create_synapse_block(mid_n_ids[i],mid_n_ids[i+1],layout.s_entries[i+1].config);
			mid_s.push_back(net->get_synapse_block(s_id));
		}

		for(i32 i = 0; i < output_mul; i++)
		{
			i32 s_id = net->create_synapse_block(
				mid_n_ids.back(),out_n_ids[i],layout.s_entries.back().config
			);
			out_s.push_back(net->get_synapse_block(s_id));
		}

		for(i32 i = 0; i < net->n_blocks.size(); i++)
			printf("n: %i %lu %i\n",
				net->n_blocks[i]->id,net->n_blocks[i]->size(),net->n_blocks[i]->model()
			);
		for(i32 i = 0; i < net->s_blocks.size(); i++)
			printf("s: %i %i %i\n",
				net->s_blocks[i]->id,net->s_blocks[i]->prev->id,net->s_blocks[i]->next->id
			);

		printf("in_n_ids: ");
		for(i32 i = 0; i < in_n_ids.size(); i++)
			printf("%i ",in_n_ids[i]);
		printf("\n");
		printf("mid_n_ids: ");
		for(i32 i = 0; i < mid_n_ids.size(); i++)
			printf("%i ",mid_n_ids[i]);
		printf("\n");
		printf("out_n_ids: ");
		for(i32 i = 0; i < out_n_ids.size(); i++)
			printf("%i ",out_n_ids[i]);
		printf("\n");

		printf("in_s_ids: ");
		for(i32 i = 0; i < in_s.size(); i++)
			printf("%i ",in_s[i]->id);
		printf("\n");
		printf("mid_s_ids: ");
		for(i32 i = 0; i < mid_s.size(); i++)
			printf("%i ",mid_s[i]->id);
		printf("\n");
		printf("out_s_ids: ");
		for(i32 i = 0; i < out_s.size(); i++)
			printf("%i ",out_s[i]->id);
		printf("\n");

		is_set_up = true;
	};

	mux_network_t::~mux_network_t()
	{
		if(net != nullptr)
			delete net;
	};

	void mux_network_t::step(bool is_training, bool do_backprop)
	{
		net->state.is_training = is_training;
		net->state.do_backprop = do_backprop;

		for(i64 i = 0; i < in_n_ids.size(); i++)
		{
			n_block_t* nb = net->n_blocks[in_n_ids[i]];
			nb->step();
			debug::log("mux_step","step n["+str(nb->id)+"]");
			debug::log("mux_step_data","n["+str(nb->id)+"].c",nb->c);
		}

		for(i64 i = 0; i < mid_n_ids.size(); i++)
		{
			n_block_t* nb = net->n_blocks[mid_n_ids[i]];
			nb->bp.err.zero();
			nb->bp.sd.zero();
			debug::log("mux_step","zero n["+str(nb->id)+"] err,sd");
		}

		for(i64 i = 0; i < out_n_ids.size(); i++)
		{
			net->n_blocks[out_n_ids[i]]->bp.err.zero();
			net->n_blocks[out_n_ids[i]]->bp.sd.zero();
			debug::log("mux_step","zero n["+str(net->n_blocks[out_n_ids[i]]->id)+"] err,sd");
		}

		for(i64 i = 0; i < net->s_blocks.size(); i++)
		{
			net->s_blocks[i]->next->c.zero();
			debug::log("mux_step","zero s["+str(i)+"]->next["+str(net->s_blocks[i]->next->id)+"] c");
		}

		for(i64 i = 0; i < in_s.size(); i++)
		{
			in_s[i]->forward();
			debug::log("mux_step","fwd s["+str(in_s[i]->id)+"]");
		}

		in_s[0]->next->step();
		debug::log("mux_step","step n["+str(in_s[0]->next->id)+"]");
		debug::log("mux_step_data","n["+str(in_s[0]->next->id)+"].c",in_s[0]->next->c);

		for(i64 i = 0; i < mid_s.size(); i++)
		{
			s_block_t* sb = mid_s[i];
			n_block_t* nb = sb->next;
			sb->forward();
			nb->step();
			debug::log("mux_step","fwd s["+str(sb->id)+"]");
			debug::log("mux_step","step n["+str(nb->id)+"]");
			debug::log("mux_step_data","n["+str(nb->id)+"].c",nb->c);
		}

		for(i64 i = 0; i < out_s.size(); i++)
		{
			s_block_t* sb = out_s[i];
			n_block_t* nb = sb->next;
			sb->forward();
			nb->step();
			debug::log("mux_step","fwd s["+str(sb->id)+"]");
			debug::log("mux_step","step n["+str(nb->id)+"]");
			debug::log("mux_step_data","n["+str(nb->id)+"].c",nb->c);
		}

		temp_output.resize(net->get_output(out_n_ids[0]).shape()).zero();
		for(i64 i = 0; i < out_n_ids.size(); i++)
		{
			temp_output += net->get_output(out_n_ids[i]);
			debug::log("mux_step","copy temp output from n["+str(out_n_ids[i])+"]");
			debug::log("mux_step_data","temp_output",temp_output);
		}
		if(output_mode == 1)
			temp_output *= (1.0f / out_n_ids.size());

		if(net->state.is_training)
		{
			if(net->config.backprop.active && (net->config.accumulate_grad || do_backprop))
			{
				for(i64 i = 0; i < out_n_ids.size(); i++)
				{
					n_block_t* nb = net->n_blocks[out_n_ids[i]];
					nb->calculate_error();
					debug::log("mux_step","calc_err n["+str(nb->id)+"]");
					debug::log("mux_step_data","n["+str(nb->id)+"].err",nb->bp.err);
				}
			}

			bool skip_error = !do_backprop;

			//do not run backward on first s_block, it does nothing but waste time!
			if(net->config.backprop.active && (net->config.accumulate_grad || do_backprop))
			{
				for(auto i = out_s.rbegin(); i != out_s.rend(); i++)
				{
					(*i)->backward();
					debug::log("mux_step","back s["+str((*i)->id)+"]");
					debug::log("mux_step_data","n["+str((*i)->prev->id)+"].err",(*i)->prev->bp.err);
				}

				for(auto i = mid_s.rbegin(); i != mid_s.rend(); i++)
				{
					(*i)->backward();
					debug::log("mux_step","back s["+str((*i)->id)+"]");
					debug::log("mux_step_data","n["+str((*i)->prev->id)+"].err",(*i)->prev->bp.err);
				}
			}

			if(net->config.backprop.active && do_backprop)
			{
				for(auto i = net->s_blocks.begin(); i != net->s_blocks.end(); i++)
				{
					(*i)->update_weights();
					debug::log("mux_step","upw s["+str((*i)->id)+"]");
				}
			}
		}

		net->state.time++;
	};

	math::array<float> mux_network_t::get_output() const
	{
		return temp_output;
	};

	math::array<float> mux_network_t::get_target() const
	{
		return net->get_neuron_block(out_n_ids[0])->target;
	};

	void mux_network_t::set_inputs(bool is_training)
	{
		if(is_training)
		{
			for(i64 i = 0; i < in_n_ids.size(); i++)
				net->set_input(data.train_samples[data.train_index].input,in_n_ids[i]);
		}
		else
		{
			for(i64 i = 0; i < in_n_ids.size(); i++)
				net->set_input(data.test_samples[data.test_index].input,in_n_ids[i]);
		}
	};

	void mux_network_t::set_targets(bool is_training)
	{
		if(is_training)
		{
			for(i64 i = 0; i < out_n_ids.size(); i++)
				net->set_target(data.train_samples[data.train_index].target,out_n_ids[i]);
		}
		else
		{
			for(i64 i = 0; i < out_n_ids.size(); i++)
				net->set_target(data.test_samples[data.test_index].target,out_n_ids[i]);
		}
	};
};