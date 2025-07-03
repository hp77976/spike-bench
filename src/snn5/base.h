#pragma once
#include "../math/array.h"
#include "../math/tensor.h"

namespace snn5
{
	enum block_type_e
	{
		BLOCK_N = 0,
		BLOCK_S = 1
	};

	struct block_t
	{
		i32 id = -1;

		protected:
		block_t(i32 id_) : id(id_) {};

		public:
		virtual void reset() = 0;

		virtual i64 size() const = 0;

		virtual i64 bytes() const = 0;

		virtual i32 block_type() const = 0;
	};
};