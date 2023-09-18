#pragma once

/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

/** @file
  * @brief Host side utils
  * @author Beniel Thileepan (maintainer)
  * @details This is to manage support functions in the host side.
  */

#include <iomanip>
#include <iostream>

ops::hls::GridPropertyCore createGridPropery(const unsigned short dim,
		const ops::hls::SizeType& size,
		const ops::hls::SizeType& d_m,
		const ops::hls::SizeType& d_p,
		const unsigned short vector_factor=8)
{
	ops::hls::GridPropertyCore gridProp;
	gridProp.dim = dim;

	for (int i = 0; i < ops_max_dim; i++)
	{
		gridProp.size[i] = size[i];
		gridProp.d_m[i] = d_m[i];
		gridProp.d_p[i] = d_p[i];
		gridProp.actual_size[i] = gridProp.size[i] + gridProp.d_p[i] + gridProp.d_m[i];
		gridProp.grid_size[i] = gridProp.actual_size[i];
	}

	gridProp.xblocks = (gridProp.actual_size[0] + vector_factor - 1) / vector_factor;
	gridProp.grid_size[0] = gridProp.xblocks * vector_factor;

	//this will be changed according to the stencils
	gridProp.outer_loop_limit = gridProp.actual_size[gridProp.dim - 1] + gridProp.d_p[gridProp.dim - 1];

	gridProp.total_itr = gridProp.xblocks;

	for (int i = 1; i < gridProp.dim; i++)
	{
		gridProp.total_itr *= gridProp.actual_size[i];
	}

	return gridProp;
}

void printGridProp(ops::hls::GridPropertyCore& gridProp, std::string prompt = "")
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "  grid properties: " << prompt << std::endl;
	std::cout << "-------------------------------" << std::endl;
	std::cout << std::setw(15) << std::right << "dim: "  << gridProp.dim << std::endl;
	std::cout << std::setw(15) << std::right << "d_m: " << "(" << gridProp.d_m[0]
				<< ", " << gridProp.d_m[1] << ", " << gridProp.d_m[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "d_p: " << "(" << gridProp.d_p[0]
				<< ", " << gridProp.d_p[1] << ", " << gridProp.d_p[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "logical size: " << "(" << gridProp.size[0]
				<< ", " << gridProp.size[1] << ", " << gridProp.size[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "actual size: " << "(" << gridProp.actual_size[0]
				<< ", " << gridProp.actual_size[1] << ", " << gridProp.actual_size[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "grid size: " << "(" << gridProp.grid_size[0]
				<< ", " << gridProp.grid_size[1] << ", " << gridProp.grid_size[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "xblocks: " << gridProp.xblocks << std::endl;
	std::cout << std::setw(15) << std::right << "total iterations: " << gridProp.total_itr << std::endl;
	std::cout << std::setw(15) << std::right << "outer limit: " << gridProp.outer_loop_limit << std::endl;
	std::cout << "-------------------------------" << std::endl;
}
