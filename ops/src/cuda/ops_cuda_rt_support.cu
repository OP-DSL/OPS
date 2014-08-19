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

/** @brief ops cuda specific runtime support functions
  * @author Gihan Mudalige
  * @details Implements cuda backend runtime support functions
  */


//
// header files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <ops_lib_core.h>
#include <ops_cuda_rt_support.h>


/*__global__ void copy_kernel(char *dest, char *src, int size ) {
  int tid = blockIdx.x;
  memcpy(&dest[tid],&src[tid],size);
}*/


__global__ void copy_kernel_tobuf(char * dest, char * src, 
                 int rx_s, int rx_e,
                 int ry_s, int ry_e,
                 int rz_s, int rz_e,
                 int x_step, int y_step, int z_step,
                 int size_x,int size_y,int size_z,
                 int buf_strides_x, int buf_strides_y, int buf_strides_z,
                 int elem_size){

  int idx_z = rz_s + z_step*(blockDim.z * blockIdx.z + threadIdx.z);
  int idx_y = ry_s + y_step*(blockDim.y * blockIdx.y + threadIdx.y);
  int idx_x = rx_s + x_step*(blockDim.x * blockIdx.x + threadIdx.x);
  
  if ((x_step ==1 ? idx_x < rx_e : idx_x > rx_e) &&
      (y_step ==1 ? idx_y < ry_e : idx_y > ry_e) &&
      (z_step ==1 ? idx_z < rz_e : idx_z > rz_e)) {
  
      src += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * elem_size;  
      dest  += ((idx_z-rz_s)*z_step*buf_strides_z+ (idx_y-ry_s)*y_step*buf_strides_y + (idx_x-rx_s)*x_step*buf_strides_x)*elem_size;
      memcpy(dest,src,elem_size);
  }
}

__global__ void copy_kernel_frombuf(char * dest, char * src, 
                 int rx_s, int rx_e,
                 int ry_s, int ry_e,
                 int rz_s, int rz_e,
                 int x_step, int y_step, int z_step,
                 int size_x,int size_y,int size_z,
                 int buf_strides_x, int buf_strides_y, int buf_strides_z,
                 int elem_size){

  int idx_z = rz_s + z_step*(blockDim.z * blockIdx.z + threadIdx.z);
  int idx_y = ry_s + y_step*(blockDim.y * blockIdx.y + threadIdx.y);
  int idx_x = rx_s + x_step*(blockDim.x * blockIdx.x + threadIdx.x);
  
  if ((x_step ==1 ? idx_x < rx_e : idx_x > rx_e) &&
      (y_step ==1 ? idx_y < ry_e : idx_y > ry_e) &&
      (z_step ==1 ? idx_z < rz_e : idx_z > rz_e)) {
  
      dest += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * elem_size;  
      src  += ((idx_z-rz_s)*z_step*buf_strides_z+ (idx_y-ry_s)*y_step*buf_strides_y + (idx_x-rx_s)*x_step*buf_strides_x)*elem_size;
      memcpy(dest,src,elem_size);
  }
}
  
  
void ops_cuda_halo_copy_tobuf(char * dest, char * src,
                        int rx_s, int rx_e,
                        int ry_s, int ry_e,
                        int rz_s, int rz_e,
                        int x_step, int y_step, int z_step,
                        int size_x,int size_y,int size_z,
                        int buf_strides_x, int buf_strides_y, int buf_strides_z,
                        int size) {

  int thr_x = abs(rx_s-rx_e);
  int blk_x = 1;
  if (abs(rx_s-rx_e)>8) {
    blk_x = (thr_x-1)/8+1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s-ry_e);
  int blk_y = 1;
  if (abs(ry_s-ry_e)>8) {
    blk_y = (thr_y-1)/8+1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s-rz_e);
  int blk_z = 1;
  if (abs(rz_s-rz_e)>8) {
    blk_z = (thr_z-1)/8+1;
    thr_z = 8;
  }
    
  dim3 grid(blk_x,blk_y,blk_z);
  dim3 tblock(thr_x,thr_y,thr_z);
  copy_kernel_tobuf<<<grid, tblock>>>(dest, src, 
                                  rx_s, rx_e, ry_s, ry_e, rz_s, rz_e,
                                  x_step, y_step, z_step,
                                  size_x, size_y, size_z,
                                  buf_strides_x, buf_strides_y, buf_strides_z,
                                  size);  
}

  
void ops_cuda_halo_copy_frombuf(char * dest, char * src,
                        int rx_s, int rx_e,
                        int ry_s, int ry_e,
                        int rz_s, int rz_e,
                        int x_step, int y_step, int z_step,
                        int size_x,int size_y,int size_z,
                        int buf_strides_x, int buf_strides_y, int buf_strides_z,
                        int size) {

  int thr_x = abs(rx_s-rx_e);
  int blk_x = 1;
  if (abs(rx_s-rx_e)>8) {
    blk_x = (thr_x-1)/8+1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s-ry_e);
  int blk_y = 1;
  if (abs(ry_s-ry_e)>8) {
    blk_y = (thr_y-1)/8+1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s-rz_e);
  int blk_z = 1;
  if (abs(rz_s-rz_e)>8) {
    blk_z = (thr_z-1)/8+1;
    thr_z = 8;
  }
    
  dim3 grid(blk_x,blk_y,blk_z);
  dim3 tblock(thr_x,thr_y,thr_z);
  copy_kernel_frombuf<<<grid, tblock>>>(dest, src, 
                                  rx_s, rx_e, ry_s, ry_e, rz_s, rz_e,
                                  x_step, y_step, z_step,
                                  size_x, size_y, size_z,
                                  buf_strides_x, buf_strides_y, buf_strides_z,
                                  size);  
}
  
  
  
  

