
#pragma once

#include <cmath>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include "blacksholes_utils.h"
//#include "blacksholes_ops.h"



float standard_normal_CDF(float val);

/*
 * @brief 	European call option calculation with exact solution.
 *
 * 			C = N(d1)*S - N(d2)*K*exp(-r*t)
 *
 * 				N(x)	- Standard Normal CDF
 * 				S		- Spot price
 * 				K		- Strike price
 * 				r		- Risk free rate
 * 				t		- Time to maturity
 * 				sigma	- Volatility
 *
 * 			d1 = (log(S/K) + (r + (sigmaˆ2) / 2)t) / (sigma *sqrt(t))
 *
 * 			d2 = d1 - sigma * sqrt(t)
 *
 * @param	spot_price			The price of the underlying instrument at t=0
 * 			strike_price		The exercising price of the option contract
 * 			time_to_maturity	The expire time from the contract date in years.
 * 			risk_free_rate		Risk free interest rate, which is an estimated value
 * 								in case the rate is changing
 * 			volatility			Volatility of the underlying instrument which is an
 * 								estimated value from the past stock price of the
 * 								underlying instrument.
 *
 * @return 	estimated value of the option.
 */
float blacksholes_call_option(float spot_price, float strike_price,
		float time_to_maturity, float risk_free_rate, float volatility);

//exact solution. To check the correctness
float test_blacksholes_call_option(BlacksholesParameter calcParam, double * time_to_run=nullptr);

void bs_explicit1(float* current, float *next, GridParameter& gridData, BlacksholesParameter& computeParam);

float get_call_option(float* data, BlacksholesParameter& computeParam);
// copy of instvan's implementation explicit1 in BS_1D_CPU
int bs_explicit2(float* current, float *next, GridParameter gridData, std::vector<BlacksholesParameter> & computeParam);

void intialize_grid(float* grid, GridParameter gridProp, BlacksholesParameter& computeParam);

bool stencil_stability(BlacksholesParameter& computeParam, bool verbose=false);

double square_error(float* current, float* next, GridParameter gridData);

int copy_grid(float* curent, float* next, GridParameter gridData);

bool verify(float * grid_data1, float *  grid_data2, int size[1], int d_m[1], int d_p[1], int range[2]);
