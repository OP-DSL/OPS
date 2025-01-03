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

void init_a_b_cpu(float *a, float*b, float& const_a, float& const_b, int size[2], int d_m[2], int d_p[2], int range[4], int multidim = 1)
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
void init_zero_cpu(T* u, int size[2], int d_m[2], int d_p[2], int range[4], int multidim = 1)
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

            if (multidim == 1)
            {
                u[index] = 0;
            }
            else
            { 
                for (int m_dim = 0; m_dim < multidim; m_dim++)
                {
                    u[index * multidim + m_dim] = 0;
                }
            }
        }
    }
}

template <typename T>
void init_const_cpu(const T& cnst, T* u, int size[2], int d_m[2], int d_p[2], int range[4], int multidim = 1)
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

            if (multidim == 1)
            {
                u[index] = cnst;
            }
            else
            { 
                for (int m_dim = 0; m_dim < multidim; m_dim++)
                {
                    u[index * multidim + m_dim] = cnst;
                }
            }
        }
    }
}

template <typename T>
void init_index_cpu(T* u, int size[2], int d_m[2], int d_p[2], int range[4], int multidim = 1)
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
            
            if (multidim == 1)
            {
                u[index] = int_idx;
            }
            else
            { 
                for (int m_dim = 0; m_dim < multidim; m_dim++)
                {
                    u[index * multidim + m_dim] = int_idx;
                }
            }
        }
    }
}

void kernel_1_cpu(const float* a, const float* d0, float* d1,
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
            d1[index * 3] = a[index] * d0[index * 3];
            d1[index * 3 + 1] = a[index] * d0[index * 3 + 1];
            d1[index * 3 + 2] = a[index] * d0[index * 3 + 2];
        }
    }
}

void kernel_2_cpu(const float* b, const float* d0, const float* d1, float* d2,
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
            d2[index * 3] = b[index] * d1[index * 3] + d0[index * 3];
            d2[index * 3 + 1] = b[index] * d1[index * 3 + 1] + d0[index * 3 + 1];
            d2[index * 3 + 2] = b[index] * d1[index * 3 + 2] + d0[index * 3 + 2];
        }
    }    
}
template <typename T>
void copy_cpu(const T* in, T* out, int size[2], int d_m[2], int d_p[2], int range[4], int multidim = 1)
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
            
            if (multidim == 1)
            {
                out[index] = in[index];
            }
            else 
            {
                for (int m_dim = 0; m_dim < multidim; m_dim++)
                {
                    out[index * multidim + m_dim] = in[index * multidim + m_dim];
                }
            }
        }
    }
}

template <typename T>
bool verify(T * grid_data1, T *  grid_data2, int size[2], int d_m[2], int d_p[2], int range[4], int multidim = 1)
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

            if (multidim == 1)
            {
                if (abs(grid_data1[index] - grid_data2[index]) > EPSILON)
                {
                    std::cerr << "[ERROR] value Mismatch index: (" << i << ", " << j << "), grid_data1: "
                            << grid_data1[index] << ", and grid_data2: " << grid_data2[index] << std::endl;
                    passed = false;
                }
            }
            else
            {
                for (int m_dim = 0; m_dim < multidim; m_dim++)
                {
                    if (abs(grid_data1[index * multidim + m_dim] - grid_data2[index * multidim + m_dim]) > EPSILON)
                    {
                        std::cerr << "[ERROR] value Mismatch index: (" << i << ", " << j << ")[" <<  m_dim << "], grid_data1: "
                                << grid_data1[index] << ", and grid_data2: " << grid_data2[index] << std::endl;
                        passed = false;
                    }
                }
            }
        }
    }

    return passed;
}
