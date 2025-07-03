#include "../math/array.h"
#include "../snn5/snn.h"

int main()
{
	rng_t rng;

	math::array<float> a, b;
	a.resize(64).randomize(rng);
	b.resize(64).fill(1.0f);

	snn5::network_t* net = new snn5::network_t({1,1,1});
	net->step(true,true);
	net->free();
	delete net;

	simd::float8* a8 = a.as<simd::float8>();
	simd::float8* b8 = b.as<simd::float8>();

	std::vector<uint8_t> masks = {};
	for(uint32_t i = 0; i < 8; i++)
		masks.push_back(simd::gt(a8[i],b8[i]));

	uint32_t sum = 0;
	for(uint32_t i = 0; i < masks.size(); i++)
		if(masks.at(i))
			sum++;

	printf("sum: %u\n",sum);

	a.free();
	b.free();
	return 0;
};