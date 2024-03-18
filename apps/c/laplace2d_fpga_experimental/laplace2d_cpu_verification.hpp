#pragma once

#define EPSILON 0.0001
typedef float stencil_type;

bool verify(stencil_type * grid_data1, stencil_type *  grid_data2, int size[2], int d_m[2], int d_p[2])
{
    bool passed = true;
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = 0; j < grid_size_y; j++)
    {
        for (int i = 0; i < grid_size_x; i++)
        {
            int index = j * grid_size_x + i;

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

void copyGrid(stencil_type * grid_dst, stencil_type * grid_src, int size[2], int d_m[2], int d_p[2])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    for (int j = 0; j < grid_size_y; j++)
    {
        for (int i = 0; i < grid_size_x; i++)
        {
            int index = j * grid_size_x + i;
            grid_dst[index] = grid_src[index];
        }
    }
}

void testInitGrid(stencil_type* grid_data, int size[2], int d_m[2], int d_p[2])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    for (int j = 0; j < grid_size_y; j++)
    {
        for (int i = 0; i < grid_size_x; i++)
        {
        	int index = j * grid_size_x + i;
        	grid_data[index] = index;
        }
	}
}

void initilizeGrid(stencil_type * grid_data, int size[2], int d_m[2], int d_p[2], const float& pi, const int& jmax)
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int j = 0; j < grid_size_y; j++)
    {
        for (int i = 0; i < grid_size_x; i++)
        {
            int index = j * grid_size_x + i;

            if (i == 0)
            {
                grid_data[index] = sin(pi * (j) / (jmax + 1));
            }
            else if (i == (actual_size_x - 1))
            {
                grid_data[index] = sin(pi * (j) / (jmax + 1)) * exp(-pi);
            }
            else
            {
                grid_data[index] = 0;
            }

#ifdef DEBUG_LOG
                std::cout << "[DEBUG] index: " << index << ", value: " << grid_data[index] << std::endl;
#endif
        }
    }
}

void calcGrid(stencil_type* grid1, stencil_type* grid2, int size[2], int d_m[2], int d_p[2])
{
	printf("calcGrid\n");
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    auto upper_j =
	for (unsigned int j = d_m[1]; j < (d_p[1] + size[1]); j++)
	{
		for (unsigned int i = d_m[0]; i < (d_p[0] + size[0]); i++)
		{
			int index = j * grid_size_x + i;
			float point0 = grid1[index - grid_size_x];
			float point1 = grid1[index - 1];
			float point3 = grid1[index + 1];
			float point4 = grid1[index + grid_size_x];
			printf("check\n");
//			printf("i:%d, j:%d, index: %d, p0: %f, p1: %f, p3: %f, p4: %f\n", i, j, index, point0, point1, point3, point4);
			grid2[index] = 0.25f * (point0 + point1 + point3 + point4);
		}
	}
}
