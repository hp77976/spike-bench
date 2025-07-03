#include <stdint.h>
#include <stdio.h>
#include <vector>

struct array_t;

struct array_result_t
{
	array_t* arg_a = nullptr;
	array_t* arg_b = nullptr;
	void* fn = nullptr;
};

struct array_t
{
	array_result_t operator+(array_t &arr) const
	{
		return array_result_t()
	};

	array_t& operator=(const array_result_t &result)
	{

	};
};