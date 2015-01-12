#ifndef __OPS_OPENCL_REDUCTION_H
#define __OPS_OPENCL_REDUCTION_H
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

/** @brief Core header file for the ops cuda reductions - adapted from OP2
  * @author Gihan Mudalige, Istvan Reguly
  * @details This file provides an optimised implementation for reduction of
  * OPS global variables. It is separated from the op_cuda_rt_support.h file
  * because the reduction code is based on C++ templates, while the other file
  * only includes C routines.
  */

#define OPS_READ 0
#define OPS_WRITE 1
#define OPS_RW 2
#define OPS_INC 3
#define OPS_MIN 4
#define OPS_MAX 5

void reduce_float(float value,
            __local float* scratch,
            __global float* result, int type) {

  // Perform parallel reduction
  int local_index = get_local_id(0) + get_local_id(1)*get_local_size(0)+
  get_local_id(2)*get_local_size(0)*get_local_size(1);

  int tot_size = get_local_size(0)*get_local_size(1)*get_local_size(2);

  int group_index = get_group_id(0) + get_group_id(1)*get_num_groups(0)+
  get_group_id(2)*get_num_groups(0)*get_num_groups(1);

  scratch[local_index] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = tot_size / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      if(type == OPS_MIN)
        scratch[local_index] = (mine < other) ? mine : other;
      else if(type == OPS_MAX)
        scratch[local_index] = (mine > other) ? mine : other;
      else if(type == OPS_INC)
        scratch[local_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[group_index] = scratch[0];
  }
}

void reduce_double(double value,
            __local double* scratch,
            __global double* result, int type) {

  // Perform parallel reduction
  int local_index = get_local_id(0) + get_local_id(1)*get_local_size(0)+
  get_local_id(2)*get_local_size(0)*get_local_size(1);

  int tot_size = get_local_size(0)*get_local_size(1)*get_local_size(2);

  int group_index = get_group_id(0) + get_group_id(1)*get_num_groups(0)+
  get_group_id(2)*get_num_groups(0)*get_num_groups(1);

  scratch[local_index] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = tot_size / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      double other = scratch[local_index + offset];
      double mine = scratch[local_index];
      if(type == OPS_MIN)
        scratch[local_index] = (mine < other) ? mine : other;
      else if(type == OPS_MAX)
        scratch[local_index] = (mine > other) ? mine : other;
      else if(type == OPS_INC)
        scratch[local_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[group_index] = scratch[0];
  }
}

void reduce_int(int value,
            __local int* scratch,
            __global int* result, int type) {

  // Perform parallel reduction
  int local_index = get_local_id(0) + get_local_id(1)*get_local_size(0)+
  get_local_id(2)*get_local_size(0)*get_local_size(1);

  int tot_size = get_local_size(0)*get_local_size(1)*get_local_size(2);

  int group_index = get_group_id(0) + get_group_id(1)*get_num_groups(0)+
  get_group_id(2)*get_num_groups(0)*get_num_groups(1);

  scratch[local_index] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = tot_size / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      int other = scratch[local_index + offset];
      int mine = scratch[local_index];
      if(type == OPS_MIN)
        scratch[local_index] = (mine < other) ? mine : other;
      else if(type == OPS_MAX)
        scratch[local_index] = (mine > other) ? mine : other;
      else if(type == OPS_INC)
        scratch[local_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[group_index] = scratch[0];
  }
}

#endif /* __OPS_OPENCL_REDUCTION_H */
