#include <cstdlib>
#define TEST_SIMD
#include "../math/simd/sw.h"
#include "../math/simd/hw.h"
#include "../math/rng.h"
#include "../math/array.h"
#include <stdexcept>
#include <string>

namespace sw = __sw_simd__;
namespace hw = __hw_simd__;

template <typename T>
bool close_enough(T a, T b)
{
	return a == b;
};

template <>
bool close_enough(float a, float b)
{
	float ratio = a/b;
	if(ratio < 1.0f)
		ratio = 2.0f - ratio;
	if(ratio > 1.000001f)
		return false;
	return true;
};

template <typename SW, typename HW, int I>
struct data_pack_t
{
	public:
	SW sa, sb, sc;
	HW ha, hb, hc;
	uint64_t su, hu;
	std::string stage;

	data_pack_t(std::string stage_) : stage(stage_) {};

	virtual void randomize(rng_t &rng) = 0;

	void compare(std::string msg) const
	{
		for(uint32_t i = 0; i < I; i++)
		{
			if(!close_enough(this->sc[i],this->hc[i]))
			{
				printf("stage: %s\n",this->stage.c_str());
				printf("msg: %s\n",msg.c_str());
				printf("sa[%u]: %s\n",i,std::to_string(sa[i]).c_str());
				printf("ha[%u]: %s\n",i,std::to_string(ha[i]).c_str());
				printf("sb[%u]: %s\n",i,std::to_string(sb[i]).c_str());
				printf("hb[%u]: %s\n",i,std::to_string(hb[i]).c_str());
				printf("sc[%u]: %s\n",i,std::to_string(sc[i]).c_str());
				printf("hc[%u]: %s\n",i,std::to_string(hc[i]).c_str());
				throw std::runtime_error("FAILED\n");
			}
		}

		if(this->su != this->hu)
		{
			printf("stage: %s\n",this->stage.c_str());
			printf("msg: %s\n",msg.c_str());

			auto print_v = [&](std::string s, const auto &swv)
			{
				printf("%s: ",s.c_str());
				for(uint32_t i = 0; i < I; i++)
					printf("%s ",std::to_string(swv[i]).c_str());
				printf("\n");
			};

			print_v("sa",sa);
			print_v("sb",sb);
			print_v("ha",ha);
			print_v("hb",hb);

			printf("su: %lu\n",su);
			printf("hu: %lu\n",hu);
			
			printf("su: ");
			for(uint64_t i = 0; i < I; i++)
				printf("%u",(su<<i)&(0x1lu<<(I-1))?1:0);
			printf("\n");

			printf("hu: ");
			for(uint64_t i = 0; i < I; i++)
				printf("%u",(hu<<i)&(0x1lu<<(I-1))?1:0);
			printf("\n");

			throw std::runtime_error("FAILED\n");
		}
	};

	void test(rng_t &rng)
	{
#define ARI_OP(o,s) randomize(rng); sc = sa o sb; hc = ha o hb; compare(s);
		ARI_OP(+,"add");
		ARI_OP(-,"sub");
		ARI_OP(*,"mul");
		ARI_OP(/,"div");
#undef  ARI_OP

#define CMP_OP(o,s) randomize(rng); su = sa o sb; hu = ha o hb; compare(s);
		CMP_OP(<,"lt");
		CMP_OP(>,"gt");
		CMP_OP(==,"eq");
		CMP_OP(<=,"le");
		CMP_OP(>=,"ge");
		CMP_OP(!=,"ne");
#undef  CMP_OP

#define FN_OP(o,s) randomize(rng); sc = o(sa,sb); hc = o(ha,hb); compare(s);
		FN_OP(min,"min");
		FN_OP(max,"max");
#undef  FN_OP

		uint16_t mask = 0b1010101010101010;

#define MISC_OP(o,s) randomize(rng); o ; compare(s);
		MISC_OP(sc = abs(sa); hc = abs(ha),"abs");
		MISC_OP(su = lt(sa,sb); hu = lt(ha,hb),"lt_ex");
		MISC_OP(su = gt(sa,sb); hu = gt(ha,hb),"gt_ex");
		MISC_OP(su = eq(sa,sb); hu = eq(ha,hb),"eq_ex");
		MISC_OP(su = le(sa,sb); hu = le(ha,hb),"le_ex");
		MISC_OP(su = ge(sa,sb); hu = ge(ha,hb),"ge_ex");
		MISC_OP(sc = mask_add(sc,sa,sb,mask); hc = mask_add(hc,ha,hb,mask),"mask_add");
		MISC_OP(sc = mask_sub(sc,sa,sb,mask); hc = mask_sub(hc,ha,hb,mask),"mask_sub");
		MISC_OP(sc = mask_mul(sc,sa,sb,mask); hc = mask_mul(hc,ha,hb,mask),"mask_mul");
		MISC_OP(sc = mask_div(sc,sa,sb,mask); hc = mask_div(hc,ha,hb,mask),"mask_div");
		//MISC_OP(sc = mask_and(sc,sa,sb,mask); hc = mask_div(hc,ha,hb,mask),"mask_and");
		//MISC_OP(sc = mask_or(sc,sa,sb,mask); hc = mask_div(hc,ha,hb,mask),"mask_or");
		MISC_OP(sc = maskz_mov(sa,mask); hc = maskz_mov(ha,mask),"mask_z_mov");
		MISC_OP(sc = blend(sa,sb,mask); hc = blend(ha,hb,mask),"blend");
#undef  MISC_OP
	};
};

template <typename SW, typename HW, int I, typename INT>
struct int_pack_t : public data_pack_t<SW,HW,I>
{
	int_pack_t(std::string stage_) : data_pack_t<SW,HW,I>(stage_) {};

	void randomize(rng_t &rng)
	{
		for(uint32_t i = 0; i < I; i++)
		{
			INT r0 = rng.u() * 250.0f - 125.0f;
			if(r0 == 0)
				r0++;
			this->sa[i] = r0;
			this->ha[i] = r0;

			INT r1 = rng.u() * 250.0f - 125.0f;
			if(r1 == 0)
				r1++;
			this->sb[i] = r1;
			this->hb[i] = r1;

			this->sc[i] = 0;
			this->hc[i] = 0;
		}

		this->su = 0;
		this->hu = 0;
	};
};

template <typename SW, typename HW, int I>
struct fp_pack_t : public data_pack_t<SW,HW,I>
{
	fp_pack_t(std::string stage_) : data_pack_t<SW,HW,I>(stage_) {};

	void randomize(rng_t &rng)
	{
		for(uint32_t i = 0; i < I; i++)
		{
			float r0 = rng.u() * 2.0f - 1.0f;
			this->sa[i] = r0;
			this->ha[i] = r0;
			float r1 = rng.u() * 2.0f - 1.0f;
			this->sb[i] = r1;
			this->hb[i] = r1;

			this->sc[i] = 0.0f;
			this->hc[i] = 0.0f;
		}

		this->su = 0;
		this->hu = 0;
	};

	void test_special(rng_t &rng)
	{
#define MISC_OP(o,s) randomize(rng); o ; this->compare(s);
#define SA this->sa
#define SB this->sb
#define SC this->sc
#define SU this->su
#define HA this->ha
#define HB this->hb
#define HC this->hc
#define HU this->hu
		MISC_OP(SC = exp(SA); HC = exp(HA),"exp");
		MISC_OP(SC = log(SA); HC = log(HA),"log");
		MISC_OP(SC = pow(SA,SB); HC = pow(HA,HB),"pow");
#undef HA
#undef HB
#undef HC
#undef HU
#undef SA
#undef SB
#undef SC
#undef SU
#undef  MISC_OP
	};
};

void dot_mask8(
	float* a, uint8_t* m, float* c,
	uint32_t a8, uint32_t mc
)
{
	uint32_t ii = 0;
	uint32_t m8 = mc * 8;
	for(uint32_t i = 0; i < a8; i++)
	{
		uint8_t mask = m[i];

		if(mask != 0)
		{
			for(uint32_t j = 0; j < 8; j++, ii += mc)
				if(mask >> j & 0x1)
					for(uint32_t k = 0; k < mc; k++)
						c[k] += a[ii+k];
		}
		else
		{
			ii += m8;
		}
	}
};

int main()
{
	rng_t rng; rng.seed(123,456);

	fp_pack_t<sw::float4, hw::float4,  4> f4 ("float4" );
	fp_pack_t<sw::float8, hw::float8,  8> f8 ("float8" );
	fp_pack_t<sw::float16,hw::float16,16> f16("float16");

	int_pack_t<sw::int8_16,hw::int8_16,16,int8_t> i8_16 ("int8_16");
	int_pack_t<sw::int8_32,hw::int8_32,32,int8_t> i8_32 ("int8_32");
	int_pack_t<sw::int8_64,hw::int8_64,64,int8_t> i8_64 ("int8_64");

	int_pack_t<sw::int16_8, hw::int16_8,  8,int16_t> i16_8 ("int16_8");
	int_pack_t<sw::int16_16,hw::int16_16,16,int16_t> i16_16("int16_16");
	int_pack_t<sw::int16_32,hw::int16_32,32,int16_t> i16_32("int16_32");

	//int_pack_t<sw::int32_4,hw::int32_4,4,int32_t> i32_4("int32_4");
	int_pack_t<sw::int32_8, hw::int32_8,  8,int32_t> i32_8 ("int32_8");
	int_pack_t<sw::int32_16,hw::int32_16,16,int32_t> i32_16("int32_16");
	
	int_pack_t<sw::int64_2,hw::int64_2,2,int64_t> i64_2("int64_2");
	int_pack_t<sw::int64_4,hw::int64_4,4,int64_t> i64_4("int64_4");
	int_pack_t<sw::int64_8,hw::int64_8,8,int64_t> i64_8("int64_8");

	for(uint32_t i = 0; i < 4096 * 64; i++)
	{
		f4.test(rng);
		f8.test(rng);
		f16.test(rng);

		f4.test_special(rng);
		f8.test_special(rng);
		f16.test_special(rng);

		i8_16.test(rng);
		i8_32.test(rng);
		i8_64.test(rng);

		i16_8.test(rng);
		i16_16.test(rng);
		i16_32.test(rng);

		//i32_4.test(rng);
		i32_8.test(rng);
		i32_16.test(rng);

		i64_2.test(rng);
		i64_4.test(rng);
		i64_8.test(rng);
	}

	return 0;
};