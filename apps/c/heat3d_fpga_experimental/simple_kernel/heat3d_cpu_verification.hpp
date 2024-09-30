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

/** @Test application for fpga batched temporal blocked heat3d
  * @author Beniel Thileepan
  */


#pragma once

#define EPSILON 0.0001
typedef float stencil_type;

void printGridData(GridParameter& gridData, std::string prompt="")
{
	printf("--------------------------------\n");
	printf("*    Grid Data %s\n", prompt.c_str());
	printf("--------------------------------\n");

	printf("bathes: %d\n", gridData.batch);
	printf("num iter: %d\n", gridData.num_iter);
	printf("logical size: (%d, %d, %d)\n", gridData.logical_size_x, gridData.logical_size_y, gridData.logical_size_z);
	printf("actual size: (%d, %d, %d)\n", gridData.act_size_x, gridData.act_size_y, gridData.act_size_z);
	printf("grid size: (%d, %d, %d)\n", gridData.grid_size_x, gridData.grid_size_y, gridData.grid_size_z);
}

bool verify(stencil_type * grid_data1, stencil_type *  grid_data2, int size[3], int d_m[3], int d_p[3], int range[6])
{
    bool passed = true;
    int grid_size_z = size[2] - d_m[2] + d_p[2];
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int k = range[4] - d_m[2]; k < range[5] -d_m[2]; k++)
    {
        for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
        {
            for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
            {
                int index = k * grid_size_x * grid_size_y + j * grid_size_x + i;

                if (abs(grid_data1[index] - grid_data2[index]) > EPSILON)
                {
                    std::cerr << "[ERROR] value Mismatch index: (" << i << ", " << j << "," << k << "), grid_data1: "
                            << grid_data1[index] << ", and grid_data2: " << grid_data2[index] << std::endl;
                    passed = false;
                }
            }
        }
    }
    return passed;
}

void copyGrid(stencil_type * grid_dst, stencil_type * grid_src, int size[3], int d_m[3], int d_p[3], int range[6])
{
    int grid_size_z = size[2] - d_m[2] + d_p[2];
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int k = range[4] - d_m[2]; k < range[5] -d_m[2]; k++)
    {
        for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
        {
            for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
            {
                int index = k * grid_size_x * grid_size_y + j * grid_size_x + i;
                grid_dst[index] = grid_src[index];
            }
        }
    }
}


int heat3D_explicit(float * current, float *next, GridParameter gridData, std::vector<heat3DParameter> & calcParam)
{
    assert(calcParam.size() == gridData.batch);

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        //constant coefficients
        float coeff[] = {1 - 6 * calcParam[bat].K, calcParam[bat].K, calcParam[bat].K, 
                calcParam[bat].K, calcParam[bat].K, calcParam[bat].K, calcParam[bat].K};
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_z; 

        for (unsigned int iter = 0; iter < gridData.num_iter; iter+=2)
        {
            for (unsigned int k = 1; k < gridData.act_size_z - 1; k++)
            {
                for (unsigned int j = 1; j < gridData.act_size_y - 1; j++)
                {
                    for (unsigned int i = 1; i < gridData.act_size_x - 1; i++)
                    {
                        int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                                + j * gridData.grid_size_x + i;
                        int index_i_min_1 = index - 1;
                        int index_i_pls_1 = index + 1;
                        int index_j_min_1 = index - gridData.grid_size_x;
                        int index_j_pls_1 = index + gridData.grid_size_x;
                        int index_k_min_1 = index - gridData.grid_size_x * gridData.grid_size_y;
                        int index_k_pls_1 = index + gridData.grid_size_x * gridData.grid_size_y;

                        next[index] = coeff[0] * current[index] 
                                + coeff[1] * current[index_i_min_1] + coeff[2] * current[index_i_pls_1]
                                + coeff[1] * current[index_j_min_1] + coeff[2] * current[index_j_pls_1]
                                + coeff[1] * current[index_k_min_1] + coeff[2] * current[index_k_pls_1];
#ifdef DEBUG_LOG
                        printf("[VERIFICATION_INTERNAL]|%s| write_val - reg_0_0: %f \n", __func__, next[index]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_0: %f \n", __func__, current[index_k_min_1]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_1: %f \n", __func__, current[index_j_min_1]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_2: %f \n", __func__, current[index_i_min_1]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_3: %f \n", __func__, current[index]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_4: %f \n", __func__, current[index_i_pls_1]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_5: %f \n", __func__, current[index_j_pls_1]);
                        printf("[VERIFICATION_INTERNAL]|%s| read_val - reg_1_6: %f \n", __func__, current[index_k_pls_1]);
                        printf("[VERIFICATION_INTERNAL]|%s| index_val: (%d, %d, %d) \n", __func__, i, j, k);
                        printf("[VERIFICATION_INTERNAL]|%s| num_iter: %d \n", __func__, iter);
#endif
                    }

                    for (unsigned int i = 1; i < gridData.act_size_x - 1; i++)
                    {
                        int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                                + j * gridData.grid_size_x + i;
                        int index_i_min_1 = index - 1;
                        int index_i_pls_1 = index + 1;
                        int index_j_min_1 = index - gridData.grid_size_x;
                        int index_j_pls_1 = index + gridData.grid_size_x;
                        int index_k_min_1 = index - gridData.grid_size_x * gridData.grid_size_y;
                        int index_k_pls_1 = index + gridData.grid_size_x * gridData.grid_size_y;

                        current[index] = coeff[0] * next[index] 
                                + coeff[1] * next[index_i_min_1] + coeff[2] * next[index_i_pls_1]
                                + coeff[1] * next[index_j_min_1] + coeff[2] * next[index_j_pls_1]
                                + coeff[1] * next[index_k_min_1] + coeff[2] * next[index_k_pls_1];
                        
                    }
                }
            }
        }
    }

    return 0;
}

void initialize_grid(float* grid, int size[2], int d_m[3], int d_p[3], int range[6], float& angle_res_x, float angle_res_y, float angle_res_z)
{
    int grid_size_z = size[2] - d_m[2] + d_p[1];
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int k = range[4] - d_m[2]; k < range[5] - d_m[2]; k++)
    {
		for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
		{
			for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
			{
				int index = k * grid_size_x * grid_size_y
						+ j * grid_size_x + i;
				int idx[] = {i + d_m[0], j + d_m[1], k + d_m[2]};

                if (i == 0 or j == 0 or k == 0 or i == actual_size_x - 1
                        or j == grid_size_y - 1 or k == grid_size_z - 1)
                {
                    grid[index] = 0;
                }
                else
                {
					grid[index] = sin(angle_res_x * (i-1))
							* sin(angle_res_y * (j-1)) * sin(angle_res_z * (k-1));
                    // grid[index] = idx[2] * size[0] * size[1]
                    //         + idx[1] * size[0] + idx[0];
                }

			}
		}
	}

}

void copy_grid(float* grid_s, float* grid_d, GridParameter gridData)
{
	for (unsigned int k = 0; k < gridData.act_size_z; k++)
	{
		for (unsigned int j = 0; j < gridData.act_size_y; j++)
		{
			 for (unsigned int  i = 0; i < gridData.act_size_x; i++)
			 {
				int index = k * gridData.grid_size_x * gridData.grid_size_y
						+ j * gridData.grid_size_x + i;

				grid_d[index] = grid_s[index];
			 }
		}
	}

}
