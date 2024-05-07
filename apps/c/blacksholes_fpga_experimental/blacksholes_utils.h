
#pragma once

#define INV_SQRT_2 sqrt(0.5)
#define EPSILON 0.0001

struct GridParameter
{
	unsigned int logical_size_x;
	unsigned int logical_size_y;

	unsigned int act_size_x;
	unsigned int act_size_y;

	unsigned int grid_size_x;
	unsigned int grid_size_y;

	unsigned int batch;
	unsigned int num_iter;
};

struct BlacksholesParameter
{
	float spot_price;
	float strike_price;
	float time_to_maturity; //in years
	float volatility;
	float risk_free_rate;
	float delta_t;
	float delta_S;
	unsigned int N;
	unsigned int K;
	float SMaxFactor;
	bool stable;
};