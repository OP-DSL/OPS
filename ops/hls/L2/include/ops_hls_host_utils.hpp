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

ops::hls::Block ops_hls_decl_block(int dims, std::string name)
{
	ops::hls::Block block;
	block.dims = dims;
	block.name = name;

	return block;
}

//ops_decl_dat(ops_block block, int data_size, int *block_size, int *base,
//                     int *d_m, int *d_p, int *stride, T *data, char const *type,
//                     char const *name)
template <typename T>
ops::hls::Grid<T> ops_hls_decl_dat(ops::hls::Block& block, int elem_size, int* size,
		int * base, int* d_m, int* d_p, T* data_ptr, std::string type, std::string name,
		unsigned short vector_factor=8)
{
	ops::hls::SizeType size_, d_m_, d_p_;

	for (unsigned int i = 0; i < ops_max_dim; i++)
	{
		if (i < block.dims)
		{
			size_[i] = static_cast<unsigned short>(size[i]);
			d_m_[i] = static_cast<unsigned short>(-d_m[i]);
			d_p_[i] = static_cast<unsigned short>(d_p[i]);
		}
		else
		{
			size_[i] = 1;
			d_m_[i] = 0;
			d_p_[i] = 0;
		}
	}

	ops::hls::Grid<T> grid;
	grid.originalProperty = createGridPropery(block.dims, size_, d_m_, d_p_, vector_factor);

	unsigned int data_size = 1;
	for (int i = 0; i < block.dims; i++)
		data_size *= grid.originalProperty.grid_size[i];
	
	// unsigned int data_size_bytes = data_size * sizeof(T);
	grid.hostBuffer.resize(data_size);

	if (data_ptr != nullptr)
	{
		memcpy(grid.hostBuffer.data(), data_ptr, data_size);
	}

	// TODO: Else need to handle user defined hostBuffer

	grid.isHostBufDirty = false;
	grid.isDevBufDirty = false;
	return grid;
}

void getRangeAdjustedGridProp(ops::hls::GridPropertyCore& original,
		ops::hls::AccessRange& range, ops::hls::GridPropertyCore& adjusted, const unsigned short vector_factor=8)
{
	assert(range.dim == original.dim);

	for (int i = 0; i < range.dim; i++)
	{
		assert(range.end[i] <= original.actual_size[i]);
		assert(range.start[i] >= 0);
	}
	adjusted.dim = original.dim;

	for (int i = 0; i < range.dim; i++)
	{
		adjusted.size[i] = range.end[i] - range.start[i];
		adjusted.d_m[i] = 0;
		adjusted.d_p[i] = 0;
		adjusted.actual_size[i] = adjusted.size[i] + adjusted.d_p[i] + adjusted.d_m[i];

		if (i == 0)
		{
			adjusted.xblocks = (adjusted.actual_size[0] + vector_factor - 1) / vector_factor;
			adjusted.grid_size[0] = adjusted.xblocks * vector_factor;
			adjusted.total_itr = adjusted.xblocks;
		}
		else
		{
			adjusted.grid_size[i] = adjusted.actual_size[i];
			adjusted.total_itr *= adjusted.actual_size[i];
		}
	}

	adjusted.outer_loop_limit = adjusted.actual_size[adjusted.dim - 1] + (adjusted.d_m[adjusted.dim-1] + adjusted.d_p[adjusted.dim -1])/2;
}

template<typename T>
unsigned int getTotalBytes(ops::hls::GridPropertyCore& gridProp)
{
	unsigned int total_bytes = sizeof(T);

	for (unsigned short i = 0; i < gridProp.dim; i++)
	{
		total_bytes *= gridProp.grid_size[i];
	}

	return total_bytes;
}

template<typename T>
void printGrid2D(ops::hls::Grid<T> p_grid, std::string prompt="")
{
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << " [DEBUG] grid values: " << prompt << std::endl;
	std::cout << "----------------------------------------------" << std::endl;

	for (int j = 0; j < p_grid.originalProperty.grid_size[1]; j++)
	{
		for (int i = 0; i < p_grid.originalProperty.grid_size[0]; i++)
		{
			int index = i + j * p_grid.originalProperty.grid_size[0];
			std::cout << std::setw(12) << p_grid.hostBuffer[index];
		}
		std::cout << std::endl;
	}
}

template<typename T>
void printGrid2D(T* p_grid, ops::hls::GridPropertyCore& gridProperty, std::string prompt="")
{
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << " [DEBUG] grid values: " << prompt << std::endl;
	std::cout << "----------------------------------------------" << std::endl;

	for (int j = 0; j < gridProperty.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProperty.grid_size[0]; i++)
		{
			int index = i + j * gridProperty.grid_size[0];
			std::cout << std::setw(12) << p_grid[index];
		}
		std::cout << std::endl;
	}
}

void opsRange2hlsRange(int dim, int* ops_range, ops::hls::AccessRange& range, ops::hls::GridPropertyCore& p_grid)
{
	assert(static_cast<unsigned int>(dim) <= ops_max_dim);
	range.dim = static_cast<unsigned short>(dim);

	for (int i = 0; i < dim; i++)
	{
		range.start[i] = ops_range[i*2] + p_grid.d_m[i];
		range.end[i] = ops_range[i*2 + 1] + p_grid.d_m[i];
	}

//	std::cout << "[DEBUG]|" <<__func__ <<"| " << "ops_grid: " << ""
//#ifdef DEBUG_LOG
//#endif
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
