#pragma once

#define EPSILON 0.0001
typedef float stencil_type;

bool verify(stencil_type * grid_data1, stencil_type *  grid_data2, ops::hls::GridPropertyCore & gridProp)
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

void copyGrid(stencil_type * grid_dst, stencil_type * grid_src, ops::hls::GridPropertyCore& gridProp)
{
    for (unsigned short j = 0; j < gridProp.grid_size[1]; j++)
    {
        for (unsigned short i = 0; i < gridProp.grid_size[0]; i++)
        {
            unsigned short index = j * gridProp.grid_size[0] + i;
            grid_dst[index] = grid_src[index];
        }
    }
}

void testInitGrid(stencil_type* grid_data, ops::hls::GridPropertyCore& gridProp)
{
    for (unsigned short j = 0; j < gridProp.grid_size[1]; j++)
    {
        for (unsigned short i = 0; i < gridProp.grid_size[0]; i++)
        {
        	unsigned short index = j * gridProp.grid_size[0] + i;
        	grid_data[index] = index;
        }
	}
}

void initilizeGrid(stencil_type * grid_data, ops::hls::GridPropertyCore& gridProp, const float& pi, const int& jmax)
{
    for (unsigned short j = 0; j < gridProp.grid_size[1]; j++)
    {
        for (unsigned short i = 0; i < gridProp.grid_size[0]; i++)
        {
            unsigned short index = j * gridProp.grid_size[0] + i;

            if (i == 0)
            {
                grid_data[index] = sin(pi * (j) / (jmax + 1));
            }
            else if (i == (gridProp.actual_size[0] - 1))
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

void calcGrid(stencil_type* grid1, stencil_type* grid2, ops::hls::GridPropertyCore& gridProp)
{
	for (unsigned int j = gridProp.d_m[1]; j < (gridProp.d_p[1] + gridProp.size[1]); j++)
	{
		for (unsigned int i = gridProp.d_m[0]; i < (gridProp.d_p[0] + gridProp.size[0]); i++)
		{
			unsigned short index = j * gridProp.grid_size[0] + i;
			grid2[index] = 0.25f * (grid1[index - 1] + grid1[index + 1] +
					grid1[index - (gridProp.grid_size[0])] + grid1[index + gridProp.grid_size[0]]);
		}
	}
}
