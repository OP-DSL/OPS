/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief Test application for 1D fpga blacksholes application based on OPS-DSL
  * @author Beniel Thileepan, Istvan Reguly (some components)
  * @details
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing multi-block Structured mesh applications.
  *  Coded in C API.
  */

#include "blacksholes_cpu.h"

float standard_normal_CDF(float val)
{
	return 0.5 * erfc(-val * INV_SQRT_2);
}

float blacksholes_call_option(float spot_price, float strike_price,
		float time_to_maturity, float risk_free_interest_rate, float volatility)
{
	float d1 = (log(spot_price / strike_price) + (risk_free_interest_rate + pow(volatility,2) / 2) * time_to_maturity)
							/ (volatility * sqrt(time_to_maturity));

	float d2 = d1 - volatility * sqrt(time_to_maturity);
	float return_on_portfolio = standard_normal_CDF(d1) * spot_price;
	float return_on_deposit = standard_normal_CDF(d2) * strike_price * exp(-risk_free_interest_rate * time_to_maturity);

	return return_on_portfolio - return_on_deposit;
}

float test_blacksholes_call_option(BlacksholesParameter calcParam, double * time_to_run)
{
	//testing blacksholes_call_option

//	float spot_price = 62;
//	float strike_price = 60;
//	float t = 40.0/365.0; //40 days
//	float volatility = 0.32; //32%
//	float risk_free_rate = 0.04; //4%
	float spot_price = calcParam.spot_price;
	float strike_price = calcParam.strike_price;
	float t = calcParam.time_to_maturity;
	float volatility = calcParam.volatility;
	float risk_free_rate = calcParam.risk_free_rate;

	std::cout << "*********************************************"  << std::endl;
	std::cout << "** testing blacksholes direct calculations **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	std::cout << "spot_price		: " << spot_price << std::endl;
	std::cout << "strike_price		: " << strike_price << std::endl;
	std::cout << "time_to_maturity	: " << t << std::endl;
	std::cout << "volatility		: " << volatility << std::endl;
	std::cout << "risk_free_rate	: " << risk_free_rate << std::endl;
	std::cout << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	float option_price = blacksholes_call_option(spot_price, strike_price, t, risk_free_rate, volatility);
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "option_price 		: " << option_price << std::endl;
	std::chrono::duration<double, std::micro> time_us = stop - start;
	std::cout << "runtime			: " << time_us.count() <<"us" << std::endl; 
	std::cout << "============================================="  << std::endl << std::endl;

	if (time_to_run != nullptr)
		*time_to_run = time_us.count();
		
	return  option_price;
}


// golden non optimised stencil computation on host PC. 1D stencil calculation
void bs_explicit1(float* current, float *next, GridParameter& gridData, BlacksholesParameter& computeParam)
{
	float alpha = computeParam.volatility *  computeParam.volatility *  computeParam.delta_t;
	float beta =  computeParam.risk_free_rate *  computeParam.delta_t;

	float ak[gridData.grid_size_x];
	float bk[gridData.grid_size_x];
	float ck[gridData.grid_size_x];

	//Initializing coefficients
	for (unsigned int j = 1; j < gridData.act_size_x - 1; j++)
	{
		unsigned int index = j;
		ak[j] = 0.5 * (alpha * index * index - beta * index);
		bk[j] = 1 - alpha * index * index - beta;
		ck[j] = 0.5 * (alpha * index * index + beta * index);
	}

	for (unsigned int i = 0; i <  computeParam.N; i+=2)
	{
		for (unsigned int j = 1; j < gridData.act_size_x - 1; j++) //excluding ghost
		{
			next[j] = ak[j] * current[j - 1]
								+ bk[j] * current[j]
								+ ck[j] * current[j + 1];
		}

		for (unsigned int j = 1; j < gridData.act_size_x - 1; j++) //excluding ghost
		{
			current[j]	= ak[j] * next[j - 1]
								+ bk[j] * next[j]
								+ ck[j] * next[j + 1];

//				std::cout << "grid_id: " << j << " ak: " << ak[j] << " bk: " << bk[j] << " ck: " << ck[j] << " current(j-1): " << current[offset + (j - 1)]
//						<< " current(j): " << 	current[offset + j] << " current(j+1): "	<< current[offset + (j + 1)] << std::endl;
		}
	}
}

//get the exact call option pricing for given spot price and strike price
float get_call_option(float* data, BlacksholesParameter& computeParam)
{
	float index 	= (float)computeParam.spot_price / ((float) computeParam.strike_price * computeParam.SMaxFactor) * computeParam.K;
	unsigned int indexLower 	= (int)std::floor(index);
	unsigned int indexUpper 	= indexLower + 1;

	float option_price = 0.0;

	if (indexUpper < computeParam.K)
		option_price = (data[indexLower] * (indexUpper - index) + data[indexUpper] * (index - indexLower));
	else
		option_price = data[computeParam.K];

	return option_price;
}


// copy of instvan's implementation explicit1 in BS_1D_CPU
int bs_explicit2(float* current, float *next, GridParameter gridData, std::vector<BlacksholesParameter> & computeParam)
{
	assert(computeParam.size() == gridData.batch);

	for (unsigned int bat = 0; bat < gridData.batch; bat++)
	{
		unsigned int offset = bat * gridData.grid_size_x;

		float delta_S = computeParam[bat].SMaxFactor * computeParam[bat].strike_price / (computeParam[bat].K - 1); //This is how it is defined in istvan' implementation.
		float c1 = 0.5 * computeParam[bat].delta_t * computeParam[bat].volatility * computeParam[bat].volatility / (delta_S * delta_S);
		float c2 = 0.5 * computeParam[bat].delta_t * computeParam[bat].risk_free_rate / delta_S;
		float c3 = computeParam[bat].risk_free_rate * computeParam[bat].delta_t;
		float S, lambda, gamma;
		float a[gridData.grid_size_x], b[gridData.grid_size_x], c[gridData.grid_size_x];
		//intialize data
		current[offset + 0] = 0.0f;
		next[offset + 0] = 0.0f;

//		std::cout << "Init curent[" << 0 << "]: " << current[offset + 0]  << std::endl;

		for (unsigned int i = 0; i < gridData.act_size_x - 2; i++)
		{
			current[offset + i + 1] = (i * delta_S) > computeParam[bat].strike_price ? (i * delta_S - computeParam[bat].strike_price) : 0.0f;
//			std::cout << "Init curent[" << i + 1 << "]: " << current[offset + i + 1]  << std::endl;
		}

		current[offset + gridData.act_size_x - 1] = 0.0f;
		next[offset + gridData.act_size_x - 1] = 0.0f;
//		std::cout << "Init curent[" << gridData.act_size_x - 1 << "]: " << current[offset + gridData.act_size_x - 1]  << std::endl;

		for (unsigned int i = 0; i < computeParam[bat].N; i+=2)
		{
			for (unsigned int j = 1; j < gridData.act_size_x - 1; j++) //excluding ghost
			{
				if (i == 0) //calculating coefficients
				{
					S = (j - 1) * delta_S;
					lambda = c1 * S * S;
					gamma = c2 * S;

					if (j == gridData.act_size_x - 1)
					{
						a[j] = - 2.0f * gamma;
						b[j] = + 2.0f * gamma - c3;
						c[j] = 0.0f;
					}
					else
					{
						a[j] = lambda - gamma;
						b[j] = - 2.0f * lambda - c3;
						c[j] = lambda + gamma;
					}
				}

				next[offset + j] = current[offset + j]
								 + a[j]*current[offset + j - 1]
								 + b[j]*current[offset + j]
								 + c[j]*current[offset + j + 1];
			}

			for (unsigned int j = 1; j < gridData.act_size_x - 1; j++)
			{
				current[offset + j] = next[j]
								    + a[j]*next[offset + j - 1]
									+ b[j]*next[offset + j]
									+ c[j]*next[offset + j + 1];
			}
		}
	}

	return 0;
}

bool stencil_stability(BlacksholesParameter& computeParam, bool verbose)
{


	if (computeParam.delta_t < (1/(pow(computeParam.volatility, 2)*(computeParam.N - 1) + 0.5*computeParam.risk_free_rate)))
	{
		if(verbose)
		{
			std::cout << "*********************************************"  << std::endl;
			std::cout << "**   Blacksholes stability check - PASSED  **"  << std::endl;
			std::cout << "*********************************************"  << std::endl;

			std::cout << "1/(sigmaˆ2*(K-1) + 0.5*r): " << (1/(pow(computeParam.volatility, 2)*(computeParam.K - 1) + 0.5*computeParam.risk_free_rate)) << std::endl;
			std::cout << "delta: " << computeParam.delta_t << std::endl;
			std::cout << "============================================="  << std::endl << std::endl;
		}
		return true;
	}
	else
	{
		std::cout << "*********************************************"  << std::endl;
		std::cout << "**   Blacksholes stability check - FAILED  **"  << std::endl;
		std::cout << "*********************************************"  << std::endl;

		std::cout << "1/(sigmaˆ2*(K-1) + 0.5*r): " << (1/(pow(computeParam.volatility, 2)*(computeParam.K - 1) + 0.5*computeParam.risk_free_rate)) << std::endl;
		std::cout << "delta: " << computeParam.delta_t << std::endl;
		std::cout << "============================================="  << std::endl << std::endl;

		return false;
	}
}
// function to compare difference of two grids
double square_error(float* current, float* next, struct GridParameter gridData)
{
	double sum = 0;

	for(unsigned int bat = 0; bat < gridData.batch; bat++)
	{
		int offset = bat * gridData.grid_size_x* gridData.grid_size_y;

		for(unsigned int i = 0; i < gridData.act_size_y; i++)
		{
			for(unsigned int j = 0; j < gridData.act_size_x; j++)
			{
				int index = i*gridData.grid_size_x + j+offset;
				float v1 = (next[index]);
				float v2 = (current[index]);

				if(fabs(v1-v2)/(fabs(v1) + fabs(v2)) >= 0.000001 && (fabs(v1) + fabs(v2)) > 0.000001 )
				{
					printf("i:%d j:%d v1:%f v2:%f\n", i, j, v1, v2);
				}

				sum += next[index]*next[index] - current[index]*current[index];
			}
		}
	}
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, GridParameter gridData)
{
    for(unsigned int bat = 0; bat < gridData.batch; bat++)
    {
    	int offset = bat * gridData.grid_size_x * gridData.grid_size_y;

		for(unsigned int i = 0; i < gridData.act_size_y; i++)
		{
			for(unsigned int j = 0; j < gridData.act_size_x; j++)
			{
				grid_d[i*gridData.grid_size_x + j+offset] = grid_s[i*gridData.grid_size_x + j+offset];
			}
		}
    }
    return 0;
}

void intialize_grid(float* grid, GridParameter gridProp, BlacksholesParameter& computeParam)
{
	float sMax = computeParam.strike_price * computeParam.SMaxFactor;

	for (unsigned int i = 0; i < gridProp.act_size_y; i++)
	{
		for (unsigned int j = 0; j < gridProp.act_size_x; j++)
		{
			if (j == 0)
			{
				grid[i * gridProp.grid_size_x + j] = 0;
			}
			else if (j == gridProp.act_size_x -1)
			{
				grid[i * gridProp.grid_size_x + j] = sMax;
			}
			else
			{
				grid[i * gridProp.grid_size_x + j] = std::max(j*computeParam.delta_S - computeParam.strike_price, (float)0);
			}
//				std::cout << "grid_id: " << offset + i * gridData.grid_size_x + j << " val: " << grid[offset + i * gridData.grid_size_x + j] << std::endl;
		}
	}

}

bool verify(float * grid_data1, float *  grid_data2, int size[1], int d_m[1], int d_p[1], int range[2])
{
    bool passed = true;
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

	for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
	{
		int index = i;

		if (abs(grid_data1[index] - grid_data2[index]) > EPSILON)
		{
			std::cerr << "[ERROR] value Mismatch index: (" << i << "), grid_data1: "
					<< grid_data1[index] << ", and grid_data2: " << grid_data2[index] << std::endl;
			passed = false;
		}
	}

    return passed;
}