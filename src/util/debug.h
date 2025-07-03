#pragma once
#include "../common.h"
#include "../math/array.h"

namespace debug
{
	bool start_logger();

	bool stop_logger();

	void log(std::string category, std::string message);

	void log(std::string category, std::string label, math::array<float> arr);

	void print_logs(std::string category);

	void clear_logs(std::string category);

	i32 get_max_log_size();

	void set_max_log_size(i32 size);

	void write_to_file(std::string category, std::string path, bool overwrite = true);
};