#pragma once
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdint.h>

//simple utility class for testing and clocking things
struct timer
{
	std::chrono::_V2::system_clock::time_point start_;
	std::chrono::_V2::system_clock::time_point stop_;
	std::chrono::microseconds duration;

	public:
	inline timer() {};

	inline void start() {this->start_ = std::chrono::high_resolution_clock::now();};

	inline void stop()
	{
		this->stop_ = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
	};

	inline double get_live_seconds() const
	{
		std::chrono::_V2::system_clock::time_point tp = std::chrono::high_resolution_clock::now();
		auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(tp - start_);
		return double((d2 / 1000000.00f).count());
	};

	inline double get_seconds() {return double((duration / 1000000.00f).count());};
};

namespace util
{
	//simple utility class for testing and clocking things
	struct timer_t
	{
		std::chrono::_V2::system_clock::time_point start_;
		std::chrono::_V2::system_clock::time_point stop_;
		std::chrono::microseconds duration;

		public:
		inline timer_t() {};

		inline void start() {this->start_ = std::chrono::high_resolution_clock::now();};

		inline void stop()
		{
			this->stop_ = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
		};

		inline double get_live_seconds() const
		{
			std::chrono::_V2::system_clock::time_point tp = std::chrono::high_resolution_clock::now();
			auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(tp - start_);
			return double((d2 / 1000000.00f).count());
		};

		inline double get_seconds() {return double((duration / 1000000.00f).count());};
	};
};