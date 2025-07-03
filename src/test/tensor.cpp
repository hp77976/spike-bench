#include "../math/tensor.h"

int main()
{
	/*math::tensor<float> a; a.resize(4).fill(2.0f);
	math::tensor<float> b; b.resize(4).fill(1.0f);
	math::tensor<float> c; c.resize(4).fill(0.0f);
	math::tensor<float> d; d.resize(4).fill(3.0f);
	math::tensor<float> e; e.resize(4).fill(5.0f);
	math::tensor<float> f; f.resize(4).fill(4.0f);

	c = a + b + 1.0f + d / e / 0.5f;

	printf("c.size(): %i\n",c.size());
	for(i32 i = 0; i < c.size(); i++)
		printf("%f ",c[i]);
	printf("\n");*/

	const i32 d0 = 2;
	const i32 d1 = 3;
	const i32 d2 = 2;
	const i32 d3 = 3;
	i32 index = 0;

	std::vector<std::vector<std::vector<std::vector<i32>>>> t = {};
	t.resize(d0);
	for(i32 i = 0; i < t.size(); i++)
	{
		t[i].resize(d1);
		for(i32 j = 0; j < t[i].size(); j++)
		{
			t[i][j].resize(d2);
			for(i32 k = 0; k < t[i][j].size(); k++)
			{
				t[i][j][k].resize(d3);
				for(i32 l = 0; l < t[i][j][k].size(); l++)
				{
					i32 guess = 0;
					guess += l;
					guess += k * d3;
					guess += j * d3 * d2;
					guess += i * d3 * d2 * d1;
					printf("t[%i][%i][%i][%i] :: [%i] ~~ [%i]\n",i,j,k,l,index,guess);
					t[i][j][k][l] = index;
					index++;
				}
			}
		}
	}

	return 0;
};