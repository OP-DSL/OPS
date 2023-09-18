
#include <iostream>
#include <stdlib.h>
#include <string>
#include <ops_hls_fpga.hpp>
#include <ops_hls_kernel.hpp>
#include <ops_hls_defs.hpp>
#include <ops_hls_host_utils.hpp>
#include "kernel_simple_copy_wrapper.hpp"
#include <xcl2.hpp>
#include <cassert>
#include <iomanip>
#include <random>
#include <utility>
//#define DEBUG_LOG

void printGrid2D(float* data, ops::hls::GridPropertyCore& gridProp, std::string prompt="")
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << " [DEBUG] grid values: " << prompt << std::endl;
	std::cout << "-------------------------------" << std::endl;

	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			std::cout << std::setw(12) << data[index];
		}
		std::cout << std::endl;
	}
}

void writeGrid(float* data, const float& val, ops::hls::GridPropertyCore& gridProp)
{
	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			data[index] = val;
		}
	}
}

void copyGrid(float* dst, float* src, ops::hls::GridPropertyCore& gridProp)
{
	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			dst[index] = src[index];
		}
	}
}

void writeGrid(ops::hls::Grid<float>& grid, const float& val)
{
	writeGrid(grid.hostBuffer.data(), val, grid.originalProperty);
}



void printAccessRange(ops::hls::AccessRange& range)
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "          Access Range         " << std::endl;
	std::cout << "-------------------------------" << std::endl;
	std::cout << std::setw(15) << std::right << "dim: "  << range.dim << std::endl;
	std::cout << std::setw(15) << std::right << "start: " << "(" << range.start[0]
				<< ", " << range.start[1] << ", " << range.start[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "end: " << "(" << range.end[0]
				<< ", " << range.end[1] << ", " << range.end[2] <<")"<< std::endl;
	std::cout << "-------------------------------" << std::endl;
}

bool verification(ops::hls::Grid<float>& grid, const float& fill, const float& val, ops::hls::AccessRange& range)
{
	ops::hls::GridPropertyCore & gridProp = grid.originalProperty;

	bool verified = true;
	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			bool cond_x = (i >= range.start[0]) and (i < range.end[0]);
			bool cond_y = (j >= range.start[1]) and (j < range.end[1]);

			if (cond_x and cond_y)
			{
				if (grid.hostBuffer[index] != val)
				{
					verified = false;
					std::cout << "[ERROR]: val missed at, i:" << i <<", j:" << j <<", with value:" << grid.hostBuffer[index] <<". expected: " << val << std::endl;
				}
			}
			else
			{
				if (grid.hostBuffer[index] != fill)
				{
					verified = false;
					std::cout << "[ERROR]: fill missed at, i:" << i <<", j:" << j <<", with value:" << grid.hostBuffer[index] <<". expected: " << fill << std::endl;
				}
			}
		}
	}
	return verified;
}

int main(int argc, char **argv)
{
	std::cout << std::endl;
	std::cout << "***************************************************" << std::endl;
	std::cout << " TEST: simple copy kernel" << std::endl;
	std::cout << "***************************************************" << std::endl << std::endl;

	std::string xclbinFile = argv[1];
	cl_int err;

	unsigned int deviceId = 0;
	ops::hls::FPGA fpga(deviceId);

	if(!fpga.xclbin(xclbinFile))
	{
		std::cerr << "[ERROR] Couldn't program fpga. exit" << std::endl;
		return (-1);
	}

	SimpleCopyWrapper kernel_simple_copy(&fpga);

	unsigned int vector_factor = 1;

	const unsigned int num_tests = 10;
	std::cout << "TOTAL NUMBER OF TESTS: " << num_tests << std::endl;

	//random generators
	std::random_device rd;
	unsigned int seed = 7;
	std::mt19937 mtSeeded(seed);
	std::mt19937 mtRandom(rd());
	std::uniform_int_distribution<unsigned short> disInt(5,40);
	std::normal_distribution<float> disFloat(100, 10);

	//summery data holders
	std::vector<ops::hls::Grid<float>> grids(num_tests);
	std::vector<ops::hls::AccessRange> ranges(num_tests);

	const float fill = 2.5;
	const float value = 12.34;

	//creating grid property and fill
	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
		ops::hls::Grid<float>& grid = grids[test_itr];
		unsigned short x_size = disInt(mtSeeded);
		grid.originalProperty = createGridPropery(2, {x_size, x_size,1},
				{1,1,0},
				{1,1,0});

		unsigned int data_size = grid.originalProperty.grid_size[0] * grid.originalProperty.grid_size[1];
		unsigned int data_size_bytes = data_size * sizeof(float);

		grid.hostBuffer.resize(data_size);
		writeGrid(grid, fill);
		std::uniform_int_distribution<unsigned short> disLow(0, x_size/3);
		std::uniform_int_distribution<unsigned short> disHigh(x_size * 2 / 3, x_size);

		ops::hls::AccessRange range;
		range.dim = 2;
		range.start[0] = disLow(mtRandom);
		range.start[1] = disLow(mtRandom);
		range.end[0] = disHigh(mtRandom);
		range.end[1] = disHigh(mtRandom);

		ranges[test_itr] = range;
	}


	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
		ops::hls::Grid<float>& grid = grids[test_itr];
		ops::hls::AccessRange& range = ranges[test_itr];

		#ifdef DEBUG_LOG
		        std::cout << std::endl;
		        std::cout << "**********************************" << std::endl;
		        std::cout << " TEST " << test_itr << std::endl;
		        std::cout << "**********************************" << std::endl;
		        std::cout << std::endl;

				printGridProp(grid.originalProperty, "original");
				printAccessRange(range);
		#endif

		kernel_simple_copy.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, grid);
		kernel_simple_copy.sendGrid(grid);
		kernel_simple_copy.run(range, grid, value);
		kernel_simple_copy.getGrid(grid);
	}

	fpga.finish();

    std::cout << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << " TEST SUMMARY " << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << std::endl;

	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
        std::cout << "TEST " << test_itr <<": ";

        if (verification(grids[test_itr], fill, value, ranges[test_itr]))
            std::cout << "PASSED";
        else
        {
            std::cout << "FAILED";

            printGrid2D(grids[test_itr].hostBuffer.data(), grids[test_itr].originalProperty, "After sync");
        }

//        std::cout << "     Event list    " << std::endl;
//        std::cout << "-------------------" << std::endl;
//        for (auto itr = grids[test_itr].allEvents.begin(); itr != grids[test_itr].allEvents.end(); ++itr)
//        {
//        	std::cout << itr->second << std::endl;
//        }
        std::cout << std::endl;
	}

	return 0;
}
