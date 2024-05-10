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

/** @Test application for fpga batched temporal blocked laplace2d
  * @author Beniel Thileepan
  */

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
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int j = -d_m[1]; j < (grid_size_y - d_p[1]); j++)
    {
        for (int i = -d_m[0]; i < (actual_size_x - d_p[0]); i++)
        {
            int index = j * grid_size_x + i;

            grid2[index] = 0.25 * (grid1[index - 1] + grid1[index + 1] + grid1[index + grid_size_x] + grid1[index - grid_size_x]);

#ifdef DEBUG_LOG
                std::cout << "[DEBUG] index: " << index << ", value: " << grid2[index] << std::endl;
#endif
        }
    }
}
