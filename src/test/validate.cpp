#include "../snn5/snn.h"
#include "../snn5/ui.h"
#include <stdexcept>

template <typename T>
using q_log_t = snn5::q_log_t<T>;

struct error_msg_t
{
	std::string label;
	std::string message;

	error_msg_t()
	{
		label = "NONE";
		message = "";
	};

	error_msg_t(std::string l, std::string m) : label(l), message(m) {};

	void print() const
	{
		printf("[%s]",label.c_str());
		printf(" %s\n",message.c_str());
	};
};

struct error_log_t
{
	umap<std::string,std::vector<error_msg_t>> error_map = {};
	u32 max_errors = 5;

	void push(error_msg_t msg)
	{
		if(error_map.find(msg.label) == error_map.end())
			error_map.insert({msg.label,{}});
		if(error_map.at(msg.label).size() < max_errors)
			error_map.at(msg.label).push_back(msg);
	};

	void print() const
	{
		for(auto i = error_map.begin(); i != error_map.end(); i++)
		{
			u32 s = std::min<u32>(max_errors,(*i).second.size());
			for(u32 j = 0; j < s; j++)
				(*i).second.at(j).print();
		}
	};
} err_log;

bool in_range(float a, float b, float ratio = 0.99f)
{
	if(a == 0.0f && b == 0.0f)
		return true;

	if(a > b)
		std::swap(a,b);

	float r = a / b;
	return r >= ratio && (b / a < 1.01f);
};

void compare_logs(u32 lid, const q_log_t<float> &a, const q_log_t<float> &b)
{
	if(a.size() != b.size())
	{
		printf("Mismatched log szies!\n");
		printf("%s.size(): %u\n",a.name.c_str(),a.size());
		printf("%s.size(): %u\n",b.name.c_str(),b.size());
	}

	for(u32 x = 0; x < a.values.x(); x++)
	{
		for(u32 y = 0; y < a.values.y(); y++)
		{
			if(!in_range(a.values.at(x,y),b.values.at(x,y)))
			{
				std::string sx = std::to_string(x);
				std::string sy = std::to_string(y);
				std::string err;
				//err += "Value mismatch at position: "+sx+" "+sy+"!\n";
				err += "Layer: "+std::to_string(lid)+" "+a.name+".at("+sx+","+sy+") ";
				err += "a: "+std::to_string(a.values.at(x,y)) + " ";
				err += "b: "+std::to_string(b.values.at(x,y));
				//err_log.push(err);

				error_msg_t err_msg;
				err_msg.label = a.name;
				err_msg.message = err;
				err_log.push(err_msg);
			}
		}
	}
};

void compare_logs(u32 lid, const q_log_t<bool> &a, const q_log_t<bool> &b)
{
	if(a.size() != b.size())
	{
		printf("Mismatched log szies!\n");
		printf("%s.size(): %u\n",a.name.c_str(),a.size());
		printf("%s.size(): %u\n",b.name.c_str(),b.size());
	}

	for(u32 x = 0; x < a.values.x(); x++)
	{
		for(u32 y = 0; y < a.values.y(); y++)
		{
			if(a.values.at(x,y) != b.values.at(x,y))
			{
				std::string sx = std::to_string(x);
				std::string sy = std::to_string(y);
				std::string err;
				err += "Layer: "+std::to_string(lid)+" "+a.name+".at("+sx+","+sy+") ";
				err += "a: "+std::to_string(a.values.at(x,y)) + " ";
				err += "b: "+std::to_string(b.values.at(x,y));

				error_msg_t err_msg;
				err_msg.label = a.name;
				err_msg.message = err;
				err_log.push(err_msg);
			}
		}
	}
};

void set_names(snn5::layer_t* l)
{
	l->q_data->soma.v.name = "v";
	l->q_data->soma.u.name = "u";
	l->q_data->soma.c.name = "c";
	l->q_data->soma.tr.name = "tr";
	l->q_data->soma.ema_rate.name = "ema_rate";
	l->q_data->soma.bit_rate.name = "bit_rate";
	l->q_data->soma.trace.name = "trace";
	l->q_data->soma.value.name = "value";
	l->q_data->soma.fired.name = "fired";
	l->q_data->nd.err.name = "err";
	l->q_data->nd.sd.name = "sd";
	l->q_data->nd.g.name = "g";
	l->q_data->sd.w.name = "w";
	l->q_data->sd.wr.name = "wr";
	l->q_data->sd.fb.name = "fb";
	l->q_data->sd.e_trace.name = "e_trace";
	l->q_data->sd.f_trace.name = "f_trace";
	l->q_data->sd.mp.name = "mp";
	l->q_data->sd.p.name = "p";
	l->q_data->sd.stp.u.name = "stp_u";
	l->q_data->sd.stp.x.name = "stp_x";
};

u32 samples = 5;
u32 p_steps = 10;
u32 v_steps = 79;
u32 b_steps =  1;
u32 input_size = 5;

void execute(snn5::network_manager_t* nm)
{
	snn5::network_t* net0 = nm->create_network(); net0->init(math::ARRAY_UNI);
	nm->net_config.use_gpu = true;
	snn5::network_t* net1 = nm->create_network(); net1->init(math::ARRAY_UNI);
	nm->net_config.use_gpu = false;

	for(u32 j = 0; j < net0->layers.size(); j++)
	{
		set_names(net0->layers.at(j));
		set_names(net1->layers.at(j));
	}

	printf("net 0 layer 0 size: %u\n",net0->layers.at(0)->size);
	printf("net 1 layer 0 size: %u\n",net1->layers.at(0)->size);

	rng_t rng; rng.seed(123456,123456);

	math::array<float> input; input.resize(input_size).randomize(rng);
	math::array<float> target; target.resize(net0->layers.back()->size).randomize(rng);
	
	net0->set_input(input);
	net1->set_input(input);

	auto layer_check = [&](snn5::network_t* net0, snn5::network_t* net1)
	{
		for(u32 i = 0; i < net0->layers.size(); i++)
		{
			if(net0->layers[i]->size == 0)
				throw std::runtime_error("Net: 0, Null layer size!\n");
			if(net1->layers[i]->size == 0)
				throw std::runtime_error("Net: 0, Null layer size!\n");
		}
	};

	auto run_nets = [&](snn5::network_t* net0, snn5::network_t* net1, bool do_bp)
	{
		net0->step(do_bp,true);
		layer_check(net0,net1);
		for(u32 j = 0; j < net0->layers.size(); j++)
			net0->layers.at(j)->collect_logs();
		
		net1->step(do_bp,true);
		layer_check(net0,net1);
		for(u32 j = 0; j < net1->layers.size(); j++)
			net1->layers.at(j)->collect_logs();
	};

	auto run_logs = [&]()
	{
		for(u32 j = 0; j < net0->layers.size(); j++)
		{
			snn5::layer_t* n0l = net0->layers.at(j);
			snn5::layer_t* n1l = net1->layers.at(j);

			compare_logs(j,n0l->q_data->soma.v,n1l->q_data->soma.v);
			compare_logs(j,n0l->q_data->soma.u,n1l->q_data->soma.u);
			compare_logs(j,n0l->q_data->soma.c,n1l->q_data->soma.c);
			compare_logs(j,n0l->q_data->soma.tr,n1l->q_data->soma.tr);
			compare_logs(j,n0l->q_data->soma.ema_rate,n1l->q_data->soma.ema_rate);
			compare_logs(j,n0l->q_data->soma.bit_rate,n1l->q_data->soma.bit_rate);
			compare_logs(j,n0l->q_data->soma.trace,n1l->q_data->soma.trace);
			compare_logs(j,n0l->q_data->soma.value,n1l->q_data->soma.value);
			compare_logs(j,n0l->q_data->soma.fired,n1l->q_data->soma.fired);
			compare_logs(j,n0l->q_data->nd.err,n1l->q_data->nd.err);
			compare_logs(j,n0l->q_data->nd.sd,n1l->q_data->nd.sd);
			compare_logs(j,n0l->q_data->nd.g,n1l->q_data->nd.g);
			compare_logs(j,n0l->q_data->sd.w,n1l->q_data->sd.w);
			compare_logs(j,n0l->q_data->sd.wr,n1l->q_data->sd.wr);
			compare_logs(j,n0l->q_data->sd.mp,n1l->q_data->sd.mp);
		}
	};

	for(u32 x = 0; x < samples; x++)
	{
		net0->input.rate.zero();
		net1->input.rate.zero();
		net0->target.rate.zero();
		net1->target.rate.zero();

		for(u32 i = 0; i < p_steps; i++)
		{
			run_nets(net0,net1,false);
			run_logs();
		}

		net0->set_input(input);
		net1->set_input(input);

		for(u32 i = 0; i < v_steps; i++)
		{
			run_nets(net0,net1,false);
			run_logs();
		}

		net0->set_target(target);
		net1->set_target(target);

		for(u32 i = 0; i < b_steps; i++)
		{
			run_nets(net0,net1,true);
			run_logs();
		}

		input.randomize(rng);
		target.randomize(rng);
	}

	err_log.print();
	err_log.error_map.clear();

	net0->free();
	net1->free();

	delete net0;
	delete net1;
};

std::vector<std::string> str_vec = {};
std::vector<u32> int_vec = {};

struct loop_t
{
	loop_t* parent = nullptr;

	u32 size = 2;
	u32 index = 0;
	bool enabled = true;
	std::string label = "";
	std::string a_str = "OFF ";
	std::string b_str = "ON  ";
	bool* ptr = nullptr;

	std::vector<loop_t*> children = {};

	loop_t(std::string l, bool* p) : label(l), ptr(p) {};
};

std::vector<loop_t*> loop_vec = {};

void run_loop(
	const std::vector<loop_t*> &loops, i32 index,
	loop_t* final_loop, snn5::network_manager_t* nm
)
{
	if(index >= loops.size())
		return;

	loop_t* l = loops.at(index);

	auto recurse = [&]()
	{
		run_loop(loops,index+1,final_loop,nm);
		if(l == final_loop)
		{
			for(u32 j = 0; j < str_vec.size(); j++)
			{
				printf("%s:",str_vec.at(j).c_str());
				printf("%u ",int_vec.at(j));
			}
			printf("\n");
			execute(nm);
		}
	};

	if(l->parent != nullptr)
	{
		if(int_vec.at(l->parent->index) == 0)
		{
			*l->ptr = false;
			int_vec.at(l->index) = 0;
			recurse();
		}
		else
		{
			for(u32 i = 0; i < 2; i++)
			{
				if(i == 0)
				{
					*l->ptr = false;
					int_vec.at(l->index) = 0;
				}
				else if(i == 1)
				{
					*l->ptr = true;
					int_vec.at(l->index) = 1;
				}

				recurse();
			}
		}
	}
	else
	{
		for(u32 i = 0; i < 2; i++)
		{
			if(i == 0)
			{
				*l->ptr = false;
				int_vec.at(l->index) = 0;
			}
			else if(i == 1)
			{
				*l->ptr = true;
				int_vec.at(l->index) = 1;
			}

			recurse();
		}
	}	
};

void build_str_vec(std::vector<loop_t*> loops, i32 index, loop_t* final_loop)
{
	if(index >= loops.size())
		return;

	loop_t* l = loops.at(index);

	l->index = str_vec.size();
	printf("index: %u\n",l->index);
	str_vec.push_back(l->label);
	loop_vec.push_back(l);

	if(l->children.size() > 0)
		build_str_vec(l->children,0,final_loop);

	if(index < loops.size() - 1)
		build_str_vec(loops,index+1,final_loop);

	int_vec.resize(str_vec.size());
	for(u32 i = 0; i < str_vec.size(); i++)
		int_vec.at(i) = 0;
};

int main()
{
	snn5::network_manager_t* nm = new snn5::network_manager_t();
	nm->net_config.debug.ql.active = true;
	nm->net_config.debug.ql.size = samples * p_steps * v_steps * b_steps;
	nm->layer_sizes = {4,3};
	nm->input_size = input_size;
	
	nm->layer_configs.at(0).stdp.active = false;
	nm->net_config.metaplasticity.active = false;
	//nm->net_config.debug.cpu.use_safe_izh = true;
	//nm->net_config.debug.gpu.use_cpu_step_drivers = true; //PASSED
	//nm->net_config.debug.gpu.use_cpu_prop_spikes = true; //PASSED
	//nm->net_config.debug.gpu.use_cpu_calc_out_err = true;
	//nm->net_config.debug.gpu.use_cpu_prop_err = true; //PASSED
	//nm->net_config.debug.gpu.use_cpu_update_weights = true; //PASSED
	//nm->net_config.debug.gpu.use_cpu_set_rate = true; //PASSED

	std::vector<loop_t*> loops = {};
	loops.push_back(new loop_t("stp",&nm->layer_configs.at(0).stp.active));
	loops.push_back(new loop_t("stdp",&nm->layer_configs.at(0).stdp.active));
	loop_t* mp_loop = new loop_t("mp",&nm->net_config.metaplasticity.active);
	loop_t* mpps_loop = new loop_t("sp",&nm->net_config.metaplasticity.use_post_spike_for_decay);
	mpps_loop->a_str = "PRE  ";
	mpps_loop->b_str = "POST ";
	mpps_loop->parent = mp_loop;
	mp_loop->children.push_back(mpps_loop);
	loops.push_back(mp_loop);
	loops.push_back(new loop_t("np",&nm->net_config.use_np_layout));

	build_str_vec(loops,0,loops.back());
	run_loop(loop_vec,0,loop_vec.back(),nm);
	
	return 0;
};