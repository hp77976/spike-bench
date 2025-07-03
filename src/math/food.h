#pragma once
#include "core.h"
#include <math.h>
#include <stdint.h>

/*
	function names are temporary (maybe)
	the end of the function is the number of the equation
	in the original nachos framework paper
*/

namespace nachos
{
	//metaplasticity

	/*
		calculate the plasticity of a synapse
		m: metaplastic state of synapse
		w: weight of the synapse

		check with figure 4.2 : VALIDATED
	*/
	template <typename T>
	CPU_GPU inline T calc_plasticity_4_1(T m, T w)
	{
		return exp(T(0.0f)-abs(m*w));
	};

	/*
		calculate delta metaplastic state

		m_pre_thX: presynaptic activites measured by synaptic trace X^tr
		m_post_thX: postsynaptic activites measured by synaptic trace X^tr

		x_tr_j is the activity trace for the neuron j (pre-synaptic)
		x_tr_i is the activity trace for the neuron i (post-synaptic)

		threshold of neural trace for metaplasticity update (pdf page 164 (labelled 142)):
		m_pre_th1:  2.5 or 2.0
		m_post_th1: 2.0 or 1.0
		m_pre_th2:  0.0 or 2.0 (original doc has a type and labels this as th1)
		m_post_th2  1.5 or 0.5

		pre and post thresholds should be for each layer?
	*/
	CPU_GPU inline float get_new_metaplastic_state_4_2(
		float m_ij, float delta_m,
		float x_tr_i, float x_tr_j,
		float m_pre_th1, float m_post_th1,
		float m_pre_th2, float m_post_th2
	)
	{
		if(x_tr_j >= m_pre_th1 && x_tr_i >= m_post_th1)
			return m_ij + delta_m;
		/*if(x_tr_j <= m_pre_th2 && x_tr_i <= m_post_th2)
			return m_ij - delta_m;*/
		return m_ij;
	};

	/*
		gets the delta metaplastic state via the current metaplastic state
		y: base metaplastic update (0.05)
		m: metaplastic state
		m_max: maximum metaplastic state (25)

		check with figure 4.3 : VALIDATED
	*/
	CPU_GPU inline float get_delta_m_4_3(float y, float m, float m_max)
	{
		return y * exp(-(m/m_max));
	};

	CPU_GPU inline float get_delta_m_4_3_quick(float y, float m, float inv_m_max)
	{
		return y * exp(-(m*inv_m_max));
	};

	CPU_GPU inline float get_delta_m_4_3_hack(float y, float m, float inv_m_max)
	{
		float x = -(m*inv_m_max);
		float x2 = x*x;
		float x3 = x*x*x;
		float x4 = x*x*x*x;
		float z = 1.0f + (x2*0.5f) + (x3*0.16f) + (x4*0.0416);


		//float x = y * (((m*0.05f)/m_max)*((m*0.03f)/m_max)*((m*0.0f)/m_max));
		//return y * z;
		return y * exp(-(m*inv_m_max));
	};

	//synpatic consolidtion

	/*
		calculate new w_ref for synaptic consolidation
		t_ref: time constant for evolution of w_ref (2000?)
		delta_t: time step (1ms)
	*/
	template <typename T>
	CPU_GPU inline T synaptic_consolidation_4_4(T w, T w_ref, T delta_t, T t_ref)
	{
		return w_ref + (delta_t / t_ref) * (w - w_ref);
	};

	/*
		decay regular weight back towards reference weight
		w_ij: weight from neuron j to i,
		w_ij_ref: reference weight from neuron j to i
		a: decay rate
		s_i: spike for neuron i? (post_synaptic)
	*/
	template <typename T>
	CPU_GPU inline T decay_regular_weight_4_5(T a, T w_ij, T w_ij_ref, T s_i)
	{
		return -a * (w_ij - w_ij_ref) * s_i;
	};

	/*
		get decay rate for weights (a)
		b: base decay rate (5e-3)
		m: ????? metaplasticity???
		e: decay parameter (10)
		n: decay parameter ( 5)

		check with figure 4.5 : VALIDATED
	*/
	template <typename T>
	CPU_GPU inline T get_decay_rate_4_6(T b, T m, T e, T n)
	{
		return pow(b * log(m+e),n);
	};

	//regularization

	/*
		get metaplastic state of a neuron
		m: metaplastic state of incoming synapses
		n: number of incoming synapses
		m_max: maximum metaplastic state
	*/
	CPU_GPU inline float get_neuron_metaplastic_state_4_7(float* const m, uint32_t n, float m_max)
	{
		float x = 0.0f;
		for(uint32_t i = 0; i < n; i++)
			x += m[i] / m_max;
		return x / n;
	};

	/*
		output neuron regularization function, returns weight for pre synapse
		e: ???
		y: ???
		w: synapse weight from neurons j to i
		m: metaplasticity state for neuron j
	*/
	CPU_GPU inline float neuron_regularization_4_8(float e, float y, float w, float m)
	{
		//TODO: RBP random backpropogation?
		return e - y * w * m;
	};

	/*
		synaptic scaling function, returns weight for synapse j to i
		a:
		w: synapse weight from neurons j to i
		n:
		k:
		w_ik: synapse weights from k to i where k is n size
	*/
	CPU_GPU inline float synapse_scaler_4_9(float a, float w, uint32_t n, float* const w_ik)
	{
		//for()
		return 0.0f; //TODO: this
	};

	/*
		a: total sum of pre-synaptic inputs
	*/
	CPU_GPU inline float synapse_scaler_4_9(float** const w, uint32_t i, uint32_t j, uint32_t n, float a)
	{
		float x = 0.0f;
		for(uint32_t k = 0; k < n; k++)
			x += std::abs(w[i][k]);
		return a * (w[i][j] / x);
	};

	//4.2 dynamic architectures

	//4.2.1 gating via generative models
	//TODO: this

	//4.2.2 lateral connections
	//TODO: this

	/*
		synapstic consolidation and heterosynaptic decay parameters
		y: regularization rate (1e-03, 0.001)
		a: synaptic scaling magnitude (15)
	*/

	/*
	
		
	*/
	/*inline float regularization_function(float y, float a)
	{

	};*/
};

namespace tacos
{
	//gets plasticity given metaplasticity and weight of a synapse
	CPU_GPU inline float get_plasticity(float m, float w)
	{
		return exp(-std::abs(m*w));
	};

	//returns new w_ref for synapse. takes weight, w_ref, and time scale
	CPU_GPU inline float update_w_ref(float w, float w_ref, float t_ref)
	{
		return w_ref + (1.0f / t_ref) * (w - w_ref);
	};

	//returns delta weight after postsynaptic neuron spikes
	CPU_GPU inline float update_w(float w, float w_ref, float alpha)
	{
		return -alpha * (w - w_ref);
	};
};