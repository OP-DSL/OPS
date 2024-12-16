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

/** @Test application for fpga dataflow design. The sole purpose is to test func 
  * @author Beniel Thileepan
  */

#pragma once

#define EPSILON 0.0001

void init_a_b_cpu(float *a, float*b, float& const_a, float& const_b, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            a[index] = const_a;
            b[index] = const_b;
        }
    }
}

template <typename T>
void init_zero_cpu(T* u, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            u[index] = 0;
        }
    }
}

template <typename T>
void init_const_cpu(const T& cnst, T* u, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            u[index] = cnst;
        }
    }
}

template <typename T>
void init_index_cpu(T* u, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
    int act_size_x = size[0] - d_m[0] + d_p[0];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            int int_idx = (j + d_m[1]) * act_size_x + (i + d_m[0]);
            u[index] = int_idx;
        }
    }
}

void kernel_1_25pt_cpu(const float* a, const float* d0, const float* d1, float* u1, float* u2, 
        int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    
    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            float d0_temp_x = 0;
            float d0_temp_y = 0;
            float d0_temp_z = 0;
            float d1_temp_x = 0;
            float d1_temp_y = 0;
            float d1_temp_z = 0; 
            
            for (int k = d_m[0]; k < d_p[0]; k++)
            {
                if (k == 0)
                {
                    d0_temp_x += d0[index];
                    d1_temp_x += d1[index];
                }
                else
                {
                    d0_temp_x += d0[index + k];
                    d0_temp_y += d0[index + grid_size_x * k];
                    d0_temp_z += d0[index + grid_size_x * grid_size_y * k];
                    d1_temp_x += d1[index + k];
                    d1_temp_y += d1[index + grid_size_x * k];
                    d1_temp_z += d1[index + grid_size_x * grid_size_y * k];
                }
            }
            u1[index] = a[index] * 0.041666666666667f * (d0_temp_x + d0_temp_y + d0_temp_z);
            u2[index] = a[index] + 0.041666666666667f * (d1_temp_x + d1_temp_y + d1_temp_z);
            // u1[index] = a[index] * d0[index];
            // u2[index] = a[index] + d1[index];
        }
    }
}

void kernel_2_25pt_cpu(const float* b, const float* d0, const float* u1, const float* u2,
        float* u3, float* u4, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    
    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            u3[index] = b[index] * u1[index] + d0[index];
            u4[index] = b[index] * u2[index];
        }
    }    
}
template <typename T>
void copy_cpu(const T* in, T* out, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    
    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            out[index] = in[index];
        }
    }
}

template <typename T>
bool verify(T * grid_data1, T *  grid_data2, int size[2], int d_m[2], int d_p[2], int range[4])
{
    bool passed = true;
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
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
