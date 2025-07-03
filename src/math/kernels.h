#pragma once
#include <math.h>
#include "array.h"
#include "core.h"
#include "gradients.h"

/*
	this file should be used only for critical functions that are unlikely
	to need to be modified but will also be needed frequently and need to
	be correct.
*/

namespace kernel
{
	//typical STDP implementation
	//hebbian learning, rewards pre->post firing pattern
	//returns the change in weight
	CPU_GPU inline float stdp(
		float alpha_ltp, float alpha_ltd,
		float tau_ltp, float tau_ltd,
		int64_t t_pre, int64_t t_post
	)
	{
		float dt = t_post - t_pre; //delta time of spikes

		if(dt > 0.0f)
			return -alpha_ltp * exp(-dt / tau_ltp);
		else if(dt < 0.0f)
			return alpha_ltd * exp(dt / tau_ltd);
		return 0.0f;
	};

	//about 10% faster than non strict variant
	//using cached reciprocals seems to slightly harm performance???
	//might be cache related...
	CPU_GPU inline float stdp_strict(
		float alpha_ltp, float alpha_ltd,
		float tau_ltp, float tau_ltd,
		//float inv_tp, float inv_td,
		int64_t t_pre, int64_t t_post
	)
	{
		float dt = t_post - t_pre; //delta time of spikes
		//if(std::abs(dt) > 255.0f) //cutoff for LUT
		//	return 0.0f;

		if(dt > 0.0f && (dt * (1.0f / tau_ltp) < 25.0f))
			return -alpha_ltp * exp(-dt / tau_ltp);
		else if(dt < 0.0f && (dt * (1.0f / tau_ltd) > -25.0f))
			return alpha_ltd * exp(dt / tau_ltd);
		return 0.0f;

		/*if(dt > 0.0f && (dt * (inv_tp) < 25.0f))
			return -alpha_ltp * math::fast_exp(-dt * inv_tp);
		else if(dt < 0.0f && (dt * (inv_td) > -25.0f))
			return alpha_ltd * math::fast_exp(dt * inv_td);
		return 0.0f;*/
	};

	//dopamine modulated symmetric stdp
	/*CPU_GPU inline float da_stdp(
		float alpha_ltp, float alpha_ltd,
		float tau_ltp, float tau_ltd,
		int64_t t_pre, int64_t t_post,
		float reward
	)
	{
		float dt = t_post - t_pre; //delta time of spikes
		float d = 0.0f;

		if(dt > 0.0f)
			d = alpha_ltp * math::fast_exp(-dt / tau_ltp);
		else
			d = alpha_ltd * math::fast_exp(dt / tau_ltd);
		return d * reward;
	};*/

	//TODO: what is this for?
	//gets delta theta for neuron v threshold
	CPU_GPU inline float v_th_d_theta(
		float theta_init, float theta,
		float a_max, bool fired
	)
	{
		float d = (theta_init / std::abs((2.0f * theta) - theta_init));
		return (-theta + d * a_max * fired);
	};
	
	//gets new v threshold for neuron using updated theta
	CPU_GPU inline float v_th_update(float v_th_const, float theta)
	{
		return v_th_const + theta;
	};

	/*
		stp is reported to be most impactful on the input neurons
		in a network. -short term plasticity nuerons learning to learn and forget

		this was found using non-spiking networks though
	*/

	/*
		u: facilitation (decays to 0)
		x: depression (decays to 1)
		tau_u: decay rate for u
		tau_x: decay rate for x
		u_out: return value for u
		x_out: return value for x
		u_set: initial value for u
		spiked: if a spike occurred
		time: current time
		time_sp: time of pre-synaptic spike
		delta: time step in ms
	*/
	CPU_GPU inline void stp(
		float u, float tau_u,
		float x, float tau_x,
		float &u_out, float &x_out,
		float u_set, bool spiked
	)
	{
		u_out = u * (1.0f - (1.0f / tau_u)); //stp update and decay cond		
		x_out = x + (1.0f - x) * (1.0f / tau_x); //stp update and decay cond

		if(spiked)
			u_out += u_set * (1.0f - u); //firing update stp
		
		if(spiked)
			x_out -= u_out * x; //firing update stp
	};

	/*
		dx / dt = ((1-x)/t_D) - uxS(t-t_sp)

		du / dt = ((U - u) / t_F) + U (1 - u)S(t-t_sp)

		t_sp : time of pre-synaptic spike
		S : dirac delta function
		u : facilitation
		x : depression
		t_F : tau_u
		t_D : tau_x

		------------------------- c6:

		(1) du / dt = (-u / stp_tau_u) + stp_u * (1 - u_-) * S * (t - t_sp)

		(2) dx / dt = ((1 - x) / stp_tau_x) - u_+ * x_- * S * (t - t_sp)

		(3) dI / dt = (-I / TS) + A * u_+ * x - S * (t - t_sp)

		u : fraction of available resources (release prob)
		x : unknown
		u_- : value of u just prior to the spike update
		x_+ : value of x just after the spike update
		A : value of the synaptic weight
		stp_u : base value for u?
		stp_tau_u : recovery/decay value for u
		stp_tau_x : recovery/decay value for x

		u: should decay to 0
		x: should decay to 1
	*/

	//gets current to apply to neurons
	//change is weight of the synapse
	CPU_GPU inline float stp_change(
		float change, float u_set,
		float u_post, float x_pre
	)
	{
		float a = 1.0f / u_set;
		change *= (a * u_post * x_pre);
		return change;
	};

	CPU_GPU inline float stp_du(float u_set, float tau_u, float u_pre, bool spiked)
	{
		float output = u_pre - (u_pre / tau_u); 
		if(spiked)
			output += u_set * (1.0f - u_pre);
		return output;
	};

	CPU_GPU inline float stp_dx(float x_pre, float u_post, float tau_x, bool spiked)
	{
		float output = (1.0f - x_pre) / tau_x;
		if(spiked)
			output -= u_post * x_pre;
		return output;
	};

	CPU_GPU inline float stp_di(float i, float a, float u_post, float x_pre, float tau_s, bool spiked)
	{
		float output = -(i/tau_s);
		if(spiked)
			output += a * u_post * x_pre;
		return output;
	};

	CPU_GPU inline float stp_dw(float a, float u_post, float x_pre)
	{
		return a * u_post * x_pre;
	};

	//izhikevich 4 parameter neuron model to calculate for v (voltage)
	//returns a value that should be with v of a neuron
	//voltage, recovery, total current
	template <typename T>
	CPU_GPU inline T izh4_dvdt(const T &v, const T &u, const T &c, T delta = T(1.0f))
	{
		//return (((0.04f * v + 5.0f) * v + 140.f - u + c) * delta);
		return ((T(0.04f) * v * v + T(5.0f) * v + T(140.f) - u + c) * delta);
	};

	//izhikevich 4 parameter neuron model to calculate for u (recovery)
	//returns a value that should be added to u of a neuron
	//voltage, recovery
	template <typename T>
	CPU_GPU inline float izh4_dudt(
		const T &v, const T &r, const T &izh_a,
		const T &izh_b, T delta = 1.0f)
	{
		return ((izh_a * (izh_b * v - r)) * delta);
	};

	//sets the initial value for v in an izh 4 parameter model
	CPU_GPU inline float init_izh_v(float izh_c)
	{
		return izh_c;
	};

	//sets the initial value for u in an izh 4 parameter model
	CPU_GPU inline float init_izh_u(float izh_b, float izh_c)
	{
		return izh_b * izh_c;
	};

	CPU_GPU inline float lif_decay(float sum, float tau_m)
	{
		return (1.0f - (1.0f / tau_m)) * sum + (1.0f / tau_m);
	};

	/*CPU_GPU inline float raf_vt(float vt, float ut)
	{

	};*/

	/*CPU_GPU inline float raf_ut(float ut_pre, float i_ext, float e, float w, float vt_pre)
	{

	};*/
};

namespace dep
{
	
	inline float kappa(float x, float tau_l, float tau_s) //:^)
	{
		return (math::fast_exp(-x/tau_l) - math::fast_exp(-x/tau_s)) / (tau_l - tau_s);
	};

	/*inline float soma_v()
	{

	};*/

	/*inline float basal_v()
	{

	};*/

	/*inline float apical_v()
	{

	};*/
};

//fnins-15-601109

namespace plat
{
	inline float delta_t_i(float t_i, float t_i_prev, float delta_t_s)
	{
		return t_i - (t_i_prev + delta_t_s);
	};

	//pg3 eq 1
	//gets the delta membrate potential of a neuron for the next time step
	inline float soma_membrane_potential(
		float v_0_i, //hidden layer potential
		float v_0b_i, //basal dendrite potential
		float v_0a_i, //apical dendrite potential
		float g_l, float g_b, float g_a, //leak, basal, apical, conductances
		float cm, //time constant for membrane capacitance
		float delta_t //integration step (should probably be 1)
	)
	{
		float x = -v_0_i + ((g_b / g_l) * (v_0b_i-v_0_i)) + ((g_a / g_l) * (v_0a_i - v_0_i));
		return (x * delta_t + v_0_i) / cm;
	};

	//pg3 eq4
	inline float response_kernel(
		float t_l, //long time constant
		float t_s, //short time constant
		float t //input value (t - t_input_jk)
	)
	{
		return (std::exp(-t/t_l) - std::exp(-t/t_s)) * math::heaviside(t) / (t_l - t_s);
	};

	//pg3 eq3
	//run this on all spike times for input neurons to get s_input_j
	inline float filtered_input(
		math::array<int64_t> spikes, //kth spiking time of input neuron j
		int64_t t, //time
		float t_l, //long time constant
		float t_s, //short time constant
		uint32_t n
	)
	{
		float sum = 0.0f;
		for(uint32_t k = 0; k < spikes.y(); k++)
			sum += response_kernel(t_l,t_s,t-spikes.at(n,k));
		return sum;
	};

	//pg3 eq2
	//gets the potential for the basal dendrite
	inline float basal_membrane_potential(
		math::array<float> w, //synaptic weights from layer
		math::array<float> s, //filtered spiking activity from prev layer (or ema rate)
		float bias,
		uint32_t n
	)
	{
		float sum = bias;
		for(uint32_t p = 0; p < w.y(); p++)
			sum += w.at(n,p) * s.at(p);
		return sum;
	};


	//pg3 eq2
	//gets the potential for the apical dendrite
	inline float apical_membrane_potential(
		math::array<float> y, //synaptic feedback weights
		math::array<float> s, //filtered spiking activity from output layer (or ema rate)
		uint32_t n
	)
	{
		float sum = 0.0f;
		for(uint32_t p = 0; p < y.y(); p++)
			sum += y.at(n,p) * s.at(p);
		return sum;
	};

	//pg3 eq5
	//input neuron firing function
	inline float poisson_firing(
		float i_max, //max firing rate
		float v_0_i //value of input
	)
	{
		return i_max * (1.0f / (1.0f + std::exp(-v_0_i)));
	};

	//pg4 eq6
	//calculate the basal something or other for the next equation
	inline float plat_basal(
		math::array<float> w, //incoming synapse weights
		math::array<float> s, //filtered spike train (or ema rate)
		float bias, //bias 
		int32_t n
	)
	{
		float sum = bias;
		for(int32_t p = 0; p < w.y(); p++)
			sum += w.at(n,p) * s.at(p);
		return sum;
	};

	//pg4 eq6
	//calculate plateau potential for apical dendrites in hidden layer neurons
	inline float plat_apical(
		float v_1_i, float i_i,
		float g_d, float g_l,
		float v_1b_i, //use above equation to get this value
		float cm, //time constant for membrane capacitance
		float delta_t //time step
	)
	{
		float x = -v_1_i + i_i + (g_d/g_l) * (v_1b_i - v_1_i);
		return (x * delta_t - v_1_i) / cm;
	};

	//pg4 eq8
	inline float fwd_phase_plateau(
		math::array<float> v_a, //apical potential
		int64_t t1, int64_t t0, //end time of fwd phase, begin time of fwd phase?
		int64_t delta_t_s, //time delay of network dynamics
		float delta_t
	)
	{
		float sum = 0.0f;
		for(uint32_t i = 0; i < v_a.size(); i++)
			sum += v_a.at(i);

		float delta_t1 = t1 - (t0 + delta_t_s);
		return grad::sigmoid::dx((1.0f / delta_t1) * sum);
	};

	//pg4 eq8
	inline float tgt_phase_plateau(
		math::array<float> v_a, //apical potential
		int64_t t2, int64_t t1, //end time of tgt phase, end time of fwd phase?
		int64_t delta_t_s, //time delay of network dynamics
		float delta_t
	)
	{
		float sum = 0.0f;
		for(uint32_t i = 0; i < v_a.size(); i++)
			sum += v_a.at(i);

		float delta_t2 = t2 - (t1 + delta_t_s);
		return grad::sigmoid::dx((1.0f / delta_t2) * sum);
	};

	//pg4 eq9
	inline float synaptic_weight_update_loss_fn(
		math::array<float> target,
		math::array<float> output,
		float phi_max
	)
	{
		float loss = 0.0f;
		for(uint32_t i = 0; i < target.size(); i++)
		{
			float phi_f = phi_max * grad::sigmoid::dx(output.at(i));
			float diff = target.at(i) - phi_f;
			loss += diff * diff;
		}
		return loss;
	};

	//pg5 eq10
	inline float target_firing_rate(
		float max_rate,
		float plat_fwd,
		float plat_tgt
	)
	{
		return max_rate + plat_tgt - plat_fwd;
	};

	//pg5 eq14
	inline float calc_k_b(float g_b, float g_l, float g_a)
	{
		return g_b / (g_l + g_b + g_a);
	};

	//TODO: fix this
	inline float al_aw(
		float k_b, //get from previous eq (its a constant)
		float a_t, //plat potential for target phase
		float a_f, //plat potential for forward phase
		float o_max_sig_1,
		float v_0_f,
		float s_input_f //filtered spike activity (or ema rate)
	)
	{
		return -k_b * (a_t - a_f) * o_max_sig_1 * (v_0_f) * s_input_f;
	};

	//TODO: fix this
	inline float al_ab(
		float k_b, //get from previous eq (its a constant)
		float a_t, //plat potential for target phase
		float a_f, //plat potential for forward phase
		float o_max_sig_1,
		float v_0_f
	)
	{
		return -k_b * (a_t - a_f) * o_max_sig_1 * (v_0_f);
	};

	//pg5 eq15
	inline float weight_update(
		float w, //current weight
		float n_p,
		float a_l_a_w
	)
	{
		return w - (n_p * (a_l_a_w));
	};

	//pg5 eq15
	inline float bias_update(
		float b, //current bias
		float n_p,
		float a_l_a_b
	)
	{
		return b - (n_p * (a_l_a_b));
	};
};

//probablistic metaplasticity (intended for memresistors)
namespace prob_mp
{
	/*
		chance for a weight to be updated
		m: metaplasticity state of the synapse
		w: weight of the synapse
	*/
	inline float weight_update_chance(float m, float w)
	{
		return std::exp(-std::abs(m * w));
	};

	/*
		calculate delta metaplastic state

		m_ij: metaplastic state

		m_pre_thX: presynaptic activites measured by synaptic trace X^tr
		m_post_thX: postsynaptic activites measured by synaptic trace X^tr

		x_tr_j is the activity trace for the neuron j (pre-synaptic)
		x_tr_i is the activity trace for the neuron i (post-synaptic)

		threshold of neural trace for metaplasticity update (pdf page 164 (labeled 142)):
		m_pre_th:  ~1.5?
		m_post_th: ~1.5?

		pre and post thresholds should be for each layer?
	*/
	inline float get_new_metaplastic_state_4_2(
		float m_ij, float delta_m,
		float x_tr_i, float x_tr_j,
		float m_pre_th, float m_post_th
	)
	{
		if(x_tr_j >= m_pre_th && x_tr_i >= m_post_th)
			return m_ij + delta_m;
		return m_ij;
	};

	/*
		gets the delta metaplastic state via the current metaplastic state
		y: base metaplastic update (0.05)
		m: metaplastic state
		m_max: maximum metaplastic state (25)

		check with figure 4.3 : VALIDATED
	*/
	inline float get_delta_m_4_3(float y, float m, float m_max)
	{
		return y * std::exp(-(m/m_max));
	};
};