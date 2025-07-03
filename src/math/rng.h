#pragma once
#include "core.h"
#include <random>
#include <stdint.h>
#include <string.h>

enum rng_type_e
{
	RNG_TYPE_UNIFORM = 0,
	RNG_TYPE_NORMAL,
	RNG_TYPE_CAUCHY,
	RNG_TYPE_CHI_SQR
};

const std::vector<std::string> RNG_TYPE_STRINGS =
{
	"Uniform", "Normal", "Cauchy", "Chi Squared"
};

static const char* RNG_TYPE_STRING = "Uniform\0Normal\0Cauchy\0Chi Squared\0";

struct mt_rng_t
{
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<float> u_dis;
	std::normal_distribution<float> n_dis;
	std::cauchy_distribution<float> cau_dis;
	std::chi_squared_distribution<float> chi_dis;
	std::poisson_distribution<int> poisson_dis;
	uint64_t calls = 0;

	mt_rng_t()
	{
		//gen.seed(rd());
		gen.seed(0);
		gen.discard(1000);
		u_dis = std::uniform_real_distribution<float>(0.0,1.0);
		//n_dis = std::normal_distribution<float>(0.50,1);
		//cau_dis = std::cauchy_distribution<float>(0.5,0.1);
		calls = 0;
	};

	//true random
	float u() {calls++; return u_dis(gen);}; //-1.0 thru 1.0f

	//-1 to 1 uniform random
	float uf() {calls++; return (u_dis(gen) - 0.5f) * 2.0f;};

	//triple center weighted random
	float w()
	{
		float f = 0.0f;
		for(int32_t i = 0; i < 5; i++)
			f += u() * (1.0f / 5.0f);
		calls+=5; 
		return f;
	};

	//normal (gaussian)
	float n()
	{
		calls++;
		return n_dis(gen);
	};

	float cauchy()
	{
		calls++;
		return cau_dis(gen);
	};

	float chi()
	{
		calls++;
		return chi_dis(gen);
	};

	float get_type(int32_t type)
	{
		switch(type)
		{
			case(RNG_TYPE_UNIFORM):	return u();
			case(RNG_TYPE_NORMAL):	return n();
			case(RNG_TYPE_CAUCHY):	return cauchy();
			case(RNG_TYPE_CHI_SQR):	return chi();
		}
		return 0.0f;
	};
};


#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

CPU_GPU inline uint64_t mix_bits(uint64_t v)
{
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44d;
	v ^= (v >> 33);
	return v;
};

struct alignas(16) pcg32_t
{
	uint64_t state;
	uint64_t inc;

	CPU_GPU inline pcg32_t() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {};

	CPU_GPU inline pcg32_t(uint64_t seq_index, uint64_t offset) {set_sequence(seq_index,offset);};

	CPU_GPU inline pcg32_t(uint64_t seq_index) {set_sequence(seq_index);};

	CPU_GPU inline void set_sequence(uint64_t seq_index) {set_sequence(seq_index,mix_bits(seq_index));};

	CPU_GPU inline void set_sequence(uint64_t seq_index, uint64_t seed)
	{
		state = 0u;
		inc = (seq_index << 1u) | 1u;
		uniform_u32_1();
		state += seed;
		uniform_u32_1();
	};

	CPU_GPU inline void advance(int64_t i_delta)
	{
		uint64_t cur_mult = PCG32_MULT;
		uint64_t cur_plus = inc;
		uint64_t acc_mult = 1u;
		uint64_t acc_plus = 0u;
		uint64_t delta = (uint64_t)i_delta;

		while(delta > 0)
		{
			if(delta & 1)
			{
				acc_mult *= cur_mult;
				acc_plus = acc_plus * cur_mult + cur_plus;
			}

			cur_plus = (cur_mult + 1) * cur_plus;
			cur_mult *= cur_mult;
			delta /= 2;
		}

		state = acc_mult * state + acc_plus;
	};

	CPU_GPU inline float u() {return uniform_u32_1() * 0x1p-32f;};

	CPU_GPU inline uint32_t uniform_u32_1()
	{
		uint64_t old_state = state;
		state = old_state * PCG32_MULT + inc;
		uint32_t xorshifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
		uint32_t rot = (uint32_t)(old_state >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
	};
};

struct alignas(16) xos128p_t
{
	inline uint32_t rotl(const uint32_t x, int k)
	{
		return (x << k) | (x >> (32 - k));
	};

	mutable uint32_t s[4];

	inline uint32_t next(void)
	{
		uint32_t result = s[0] + s[3];
		const uint32_t t = s[1] << 9;

		s[2] ^= s[0]; s[3] ^= s[1];
		s[1] ^= s[2]; s[0] ^= s[3];

		s[2] ^= t;

		s[3] = rotl(s[3],11);

		return result;
	};

	inline void seed(uint64_t s0, uint64_t s1)
	{
		uint64_t u0 = s0 + 2039503497;
		uint64_t u1 = s1 + 9034785908;
		memcpy(&s[0],&u0,sizeof(uint64_t));
		memcpy(&s[2],&u0,sizeof(uint64_t));
	};

	inline void jump(void)
	{
		static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

		uint32_t s0 = 0;
		uint32_t s1 = 0;
		uint32_t s2 = 0;
		uint32_t s3 = 0;
		for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
			for(int b = 0; b < 32; b++) {
				if (JUMP[i] & UINT32_C(1) << b) {
					s0 ^= s[0];
					s1 ^= s[1];
					s2 ^= s[2];
					s3 ^= s[3];
				}
				next();	
			}
			
		s[0] = s0;
		s[1] = s1;
		s[2] = s2;
		s[3] = s3;
	};

	inline void long_jump(void)
	{
		static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };

		uint32_t s0 = 0;
		uint32_t s1 = 0;
		uint32_t s2 = 0;
		uint32_t s3 = 0;
		for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
			for(int b = 0; b < 32; b++) {
				if (LONG_JUMP[i] & UINT32_C(1) << b) {
					s0 ^= s[0];
					s1 ^= s[1];
					s2 ^= s[2];
					s3 ^= s[3];
				}
				next();	
			}
			
		s[0] = s0;
		s[1] = s1;
		s[2] = s2;
		s[3] = s3;
	};

	inline float f1()
	{
		uint32_t val = next();
		uint32_t bits = 32;
		uint32_t shift = 7;

		float num = val >> shift;
		float den = (1<<(bits-shift))-1;
		float fval = num/den;
		return fval;
	};
};

/*struct alignas(16) xos256p_t
{
	mutable uint64_t s[4];

	inline uint64_t rotl(const uint64_t x, int k)
	{
		return (x << k) | (x >> (64 - k));
	};

	inline uint64_t next(void)
	{
		const uint64_t result = s[0] + s[3];

		const uint64_t t = s[1] << 17;

		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];

		s[2] ^= t;

		s[3] = rotl(s[3], 45);

		return result;
	};

	inline void jump(void)
	{
		static const uint64_t JUMP[] = {
			0x180ec6d33cfd0aba,0xd5a61266f0c9392c,
			0xa9582618e03fc9aa,0x39abdc4529b1661c
		};

		uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		{
			for(int b = 0; b < 64; b++)
			{
				if(JUMP[i] & UINT64_C(1) << b)
				{
					s0 ^= s[0];
					s1 ^= s[1];
					s2 ^= s[2];
					s3 ^= s[3];
				}
				next();	
			}
		}
			
		s[0] = s0;
		s[1] = s1;
		s[2] = s2;
		s[3] = s3;
	};

	void long_jump(void)
	{
		static const uint64_t LONG_JUMP[] = {
			0x76e15d3efefdcbbf,0xc5004e441c522fb3,
			0x77710069854ee241,0x39109bb02acbe635
		};

		uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
		{
			for(int b = 0; b < 64; b++)
			{
				if(LONG_JUMP[i] & UINT64_C(1) << b)
				{
					s0 ^= s[0];
					s1 ^= s[1];
					s2 ^= s[2];
					s3 ^= s[3];
				}
				next();	
			}
		}
			
		s[0] = s0;
		s[1] = s1;
		s[2] = s2;
		s[3] = s3;
	};

	inline float f1()
	{
		uint64_t val = next();
		uint64_t bits = 64;
		uint32_t shift = 7;

		float num = val >> shift;
		float den = (1<<(bits-shift))-1;
		float fval = num/den;
		return fval;
	};
};*/

class rng_t
{
	//mt_rng_t m_mt;
	//xos128p_t m_xos;
	//Xoshiro256PlusAVX2 m_xos256 = Xoshiro256PlusAVX2(1);
	pcg32_t m_pcg;
	//std::normal_distribution<float> n_dis;
	//std::poisson_distribution<int> poisson_dis;
	//uint8_t m_backend = 0;

	public:
	CPU_GPU rng_t()
	{
		//m_xos256 = Xoshiro256PlusAVX2(1);
		//Xoshiro256PlusAVX2::FourDoubleValues(a)
	};

	CPU_GPU void seed(uint64_t s0, uint64_t s1)
	{
		/*switch(m_backend)
		{
			case(0):	m_mt.gen.seed(s0); break;
			case(1):	m_xos.seed(s0,s1); break;
			case(2):	m_pcg.set_sequence(0,s0); break;
		}*/
		//m_mt.gen.seed(s0);
		//m_xos.seed(s0,s1);
		m_pcg.set_sequence(0,s0);
		//m_pcg.advance(1000);
		//m_xos256 = Xoshiro256PlusAVX2(s0);
		//__m256d x = m_xos256.dnext4();
	};

	CPU_GPU inline float u()
	{
		/*switch(m_backend)
		{
			case(0):	return m_mt.u();
			case(1):	return m_xos.f1();
			case(2):	return m_pcg.u();
		}
		return 0.0f;*/
		//return m_mt.u();
		//return m_xos.f1();
		return m_pcg.u();
		//return m_xos256.dnext();
	};

	/*inline __m128 u4()
	{
		return _mm256_cvtpd_ps(m_xos256.dnext4());
	};*/

	CPU_GPU float w(int32_t r = 5)
	{
		float f = 0.0f;
		for(int32_t i = 0; i < r; i++)
			f += u();
		f /= r;
		//calls+=5; 
		return f;
	};

	CPU_GPU void randomize(float* ptr, float lo, float hi, uint32_t c)
	{
		for(uint32_t i = 0; i < c; i++)
			ptr[i] = (hi - lo) * u() + lo;
	};
};

/*inline void randomize(tensor<float> &t, rng_t &rng, float min, float max)
{
	for(i32 i = 0; i < t.size(); i++)
		t[i] = (max - min) * rng.u() + min;
};*/