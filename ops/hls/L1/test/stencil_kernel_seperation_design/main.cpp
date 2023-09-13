#include <iostream>
#include <random>
#include "top.hpp"

#define EPSILON 0.00001
//#define DEBUG_LOG

static void initilizeGrid(stencil_type * grid_data, ops::hls::GridPropertyCore& gridProp);
static void copyGrid(stencil_type * grid_dst, stencil_type * grid_src, ops::hls::GridPropertyCore& gridProp);
static bool verify(stencil_type * grid_data1, stencil_type * grid_data2, ops::hls::GridPropertyCore & gridProp);
static void cpuGoldenKernel(float * rd_buffer, float * wr_buffer, ops::hls::GridPropertyCore& gridProp, float * coef);

int main()
{
    std::cout << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "TESTING: ops::hls::stencil2dCore impl" << std::endl;
    std::cout << "********************************************" << std::endl << std::endl;

    /* Grid Property Generation */
    ops::hls::GridPropertyCore gridProp;

    gridProp.size[0] = 10;
    gridProp.size[1] = 10;
    gridProp.dim = 2;
    gridProp.d_p[0] = 1;
    gridProp.d_p[1] = 1;
    gridProp.d_m[0] = 1;
    gridProp.d_m[1] = 1;
    gridProp.offset[0] = 0;
    gridProp.offset[1] = 0;
    gridProp.actual_size[0] = gridProp.size[0] + gridProp.d_p[0] + gridProp.d_m[0];
    gridProp.actual_size[1] = gridProp.size[1] + gridProp.d_p[1] + gridProp.d_m[1];
    unsigned short shift_bits = LOG2(vector_factor);
    gridProp.xblocks = (gridProp.actual_size[0] + vector_factor - 1) >> shift_bits;
    gridProp.grid_size[0] =  gridProp.xblocks << shift_bits;
    gridProp.grid_size[1] = gridProp.actual_size[1];
    gridProp.total_itr = gridProp.actual_size[1] * gridProp.xblocks;  //(gridProp.actual_size[1] + p/2) * xblocks.
    gridProp.outer_loop_limit = (gridProp.actual_size[1] + 1);


    stencil_type coef[] = {0.125 , 0.125 , 0.5, 0.125, 0.125};

    /* Data Generation */
    const int num_tests  = 1;
    std::cout << "TOTAL NUMER OF TESTS: " << num_tests << std::endl;
    std::vector<bool> test_summary(num_tests);

    unsigned int data_size_bytes = sizeof(stencil_type);

    for (int i = 0; i < gridProp.dim; i++)
        data_size_bytes *= gridProp.grid_size[i];

    if (data_size_bytes >= max_data_size)
        std::cerr <<  "Maximum buffer size is exceeded" << std::endl;

    float * grid_u1_cpu = (float*) aligned_alloc(4096, data_size_bytes);
    float * grid_u2_cpu = (float*) aligned_alloc(4096, data_size_bytes);
    float * grid_u1_d = (float*) aligned_alloc(4096, data_size_bytes);
    float * grid_u2_d = (float*) aligned_alloc(4096, data_size_bytes);


    for (int test_itr = 0; test_itr < num_tests; test_itr++)
    {
        std::cout << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << " TEST " << test_itr << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << std::endl;

        initilizeGrid(grid_u1_cpu, gridProp);
        copyGrid(grid_u1_d, grid_u1_cpu, gridProp);

    	cpuGoldenKernel(grid_u1_cpu, grid_u2_cpu, gridProp, coef);

        dut(gridProp.size[0], gridProp.size[1],
        		gridProp.actual_size[0], gridProp.actual_size[1],
        		gridProp.grid_size[0], gridProp.grid_size[1],
				gridProp.dim, gridProp.xblocks, gridProp.total_itr,
				gridProp.outer_loop_limit, grid_u1_d, grid_u2_d);

        test_summary[test_itr] = verify(grid_u2_cpu, grid_u2_d, gridProp);

        if (test_summary[test_itr])
        {
            std::cout << "TEST PASSED." << std::endl;
            test_summary[test_itr] = true;
        }
        else
        {
            std::cout << "TEST FAILED." << std::endl;
            test_summary[test_itr] = false;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << " TEST SUMMARY " << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << std::endl;

    for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
    {
        std::cout << "TEST " << test_itr <<": ";

        if (test_summary[test_itr])
            std::cout << "PASSED";
        else
            std::cout << "FAILED";

        std::cout << std::endl;
    }

    std::cout << std::endl;

    free (grid_u1_cpu);
    free (grid_u2_cpu);
    free (grid_u1_d);
    free (grid_u2_d); 

    return 0;
}


static void initilizeGrid(stencil_type * grid_data, ops::hls::GridPropertyCore& gridProp)
{
    std::random_device rd;
    unsigned int seed = 7;
    std::mt19937 mtSeeded(seed);
    std::mt19937 mt;
    std::uniform_real_distribution<float> dist(0,100);

    for (unsigned short j = 0; j < gridProp.actual_size[1]; j++)
    {
        for (unsigned short i = 0; i < gridProp.actual_size[0]; i++)
        {
            unsigned short index = j * gridProp.grid_size[0] + i;

            if (i == 0 || j == 0 || i == gridProp.actual_size[0] - 1 || j == gridProp.actual_size[1] - 1)
            {
                grid_data[index] = 0;
            }
            else
            {
                grid_data[index] = dist(mt);
            }
#ifdef DEBUG_LOG
                std::cout << "[DEBUG] index: " << index << ", value: " << grid_data[index] << std::endl;
#endif
        }
    }
}

static void copyGrid(stencil_type * grid_dst, stencil_type * grid_src, ops::hls::GridPropertyCore& gridProp)
{
    for (unsigned short j = 0; j < gridProp.actual_size[1]; j++)
    {
        for (unsigned short i = 0; i < gridProp.actual_size[0]; i++)
        {
            unsigned short index = j * gridProp.grid_size[0] + i;
            grid_dst[index] = grid_src[index];
        }
    }
}

static bool verify(stencil_type * grid_data1, stencil_type *  grid_data2, ops::hls::GridPropertyCore & gridProp)
{
    bool passed = true;

    for (unsigned short j = 0; j < gridProp.actual_size[1]; j++)
    {
        for (unsigned short i = 0; i < gridProp.actual_size[0]; i++)
        {
            unsigned short index = j * gridProp.grid_size[0] + i;

            if (abs(grid_data1[index] - grid_data2[index]) > EPSILON)
            {
                std::cerr << "[ERROR] value Mismatch index: (" << i << ", " << j << "), grid_data1: "
						<< grid_data1[index] << ", and grid_data2: " << grid_data2[index] << std::endl;
                passed = false;
            }
        }
    }

    return passed;
}

static void cpuGoldenKernel(float * rd_buffer, float * wr_buffer, ops::hls::GridPropertyCore& gridProp, float * coef)
{
	for (unsigned short j = 0; j < gridProp.actual_size[1]; j++)
	{
		for (unsigned short i = 0; i < gridProp.actual_size[0]; i++)
		{
			unsigned int index = j * gridProp.grid_size[0] + i;

			if (i == 0 || j == 0 || (i == gridProp.actual_size[0] - 1) || (j == gridProp.actual_size[1] - 1))
			{
				wr_buffer[index] = rd_buffer[index];
			}
			else
			{
				wr_buffer[index] = coef[0] * rd_buffer[(j-1)*gridProp.grid_size[0] + i]
									+ coef[1] * rd_buffer[j*gridProp.grid_size[0] + i - 1]
									+ coef[2] * rd_buffer[index]
									+ coef[3] * rd_buffer[(j+1)*gridProp.grid_size[0] + i]
									+ coef[4] * rd_buffer[j*gridProp.grid_size[0] + i + 1];
			}
		}
	}
}
