#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <stdexcept>
#include <math.h>
#include "../math/array.h"
#include "../math/tensor.h"

inline bool str_eq(std::string a, std::string b)
{
	if(a.length() != b.length())
		return false;

	for(uint32_t i = 0; i < a.length(); i++)
		if(a[i] != b[i])
			return false;
	
	return true;
};

inline std::vector<float> set_hot_one(uint32_t i, uint32_t s)
{
	std::vector<float> v = {};
	v.resize(s);
	for(uint32_t j = 0; j < s; j++)
		v[j] = 0.0f;
	v[i] = 1.0f;
	return v;
};

inline void set_hot_one(uint32_t i, std::vector<float> &out)
{
	for(uint32_t j = 0; j < out.size(); j++)
		out[j] = 0.0f;
	out[i] = 1.0f;
};

inline void val_chk(float v, std::string label = "unk")
{
	if(std::isnan(v))
		throw std::runtime_error("NaN value detected: " + label + "\n");
	if(std::isinf(v))
		throw std::runtime_error("INF value detected: " + label + "\n");
};

inline std::string bits_to_str(uint32_t u)
{
	std::string s = "";
	for(uint32_t i = 0; i < 32; i++)
		s.push_back((u << i) & (0x1u << 31u) ? '1' : '0');
	return s;
};

inline char* str_vec_to_combo_char_ptr(const std::vector<std::string> &str_vec)
{
	uint32_t length = 0;
	for(uint32_t i = 0; i < str_vec.size(); i++)
	{
		std::string str = str_vec.at(i);
		for(uint32_t j = 0; j < str.size(); j++)
		{
			if(str.at(j) == '\0')
				break;
			length++;
		}
		length++; //to account for null char
	}

	uint32_t offset = 0;
	char* items = new char[length];
	for(uint32_t i = 0; i < str_vec.size(); i++)
	{
		std::string str = str_vec.at(i);
		for(uint32_t j = 0; j < str.size(); j++)
		{
			if(str.at(j) == '\0')
				break;
			items[offset] = str.at(j);
			offset++;
		}
		items[offset] = '\0';
		offset++; //to account for null char
	}

	if(offset != length)
	{
		printf("offset: %u\n",offset);
		printf("length: %u\n",length);
	}

	return items;
};

inline int32_t __tiny_byte_swap(int32_t x)
{
	//uint8_t b[4];

	union {
		int32_t i;
		uint8_t b[4];
	} u;
	u.i = x;

	std::swap(u.b[0],u.b[3]);
	std::swap(u.b[1],u.b[2]);
	
	return u.i;
};