#pragma once
#include "conf.h"
#include "../common.h"

namespace snn5
{
	struct n_entry_t;
	struct s_entry_t;

	enum entry_type_e
	{
		ENTRY_N,
		ENTRY_S
	};

	enum n_entry_type_e
	{
		N_ENTRY_INPUT,
		N_ENTRY_OUTPUT,
		N_ENTRY_HIDDEN,
	};

	struct entry_t
	{
		bool show_config = false;

		virtual ~entry_t() {};

		virtual void draw(i32 id) = 0;

		virtual i32 type() const = 0;
	};

	struct n_entry_t : public entry_t
	{
		n_block_config_t config;
		dim2 shape = {100,1};
		i32 n_type = N_ENTRY_HIDDEN;
		
		void draw(i32 id) override;

		i32 type() const override {return ENTRY_N;};
	};

	struct s_entry_t : public entry_t
	{
		s_block_config_t config;

		void draw(i32 id) override;

		i32 type() const override {return ENTRY_S;};
	};

	struct net_layout_t
	{
		std::vector<n_entry_t> n_entries = {};
		std::vector<s_entry_t> s_entries = {};
		network_config_t net_config;
		bool show_net_config;
	};
};