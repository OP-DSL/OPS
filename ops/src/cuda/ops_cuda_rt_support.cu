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


__global__ void copy_kernel(char *dest, char *src, int size ) {
  int tid = blockIdx.x;
  //if (tid < 32) 
    //for(int i=0; i<size; i++)
    //  dest[tid+i] = src[tid+i];
    memcpy(&dest[tid],&src[tid],size);
}

/*__global__ void kernel(int **in, int **out, int len, int N)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    for(; idx<N; idx+=gridDim.x*blockDim.x)
        memcpy(out[idx], in[idx], sizeof(int)*len);

}*/

void ops_halo_copy_dh(const char * dest, char * src, int size){
  printf("In CUDA ops_halo_copy_dh\n");
  
  cutilSafeCall(cudaMemcpy((void *)dest,src,size,cudaMemcpyDeviceToHost));
  cutilSafeCall ( cudaDeviceSynchronize ( ) );
}

void ops_halo_copy_hd(const char * dest, char * src, int size){
  printf("In CUDA ops_halo_copy_hd\n");
  
  cutilSafeCall(cudaMemcpy((void *)dest,src,size,cudaMemcpyHostToDevice));
  cutilSafeCall ( cudaDeviceSynchronize ( ) );
}

//void ops_halo_copy(const char * dest, char * src, int size, cudaStream_t stream){
void ops_halo_copy(char * dest, char * src, int size){
  //printf("In CUDA ops_halo_copy\n");
  
  //cutilSafeCall(cudaMemcpy((void *)dest,src,size,cudaMemcpyDeviceToDevice));
  //cutilSafeCall ( cudaDeviceSynchronize ( ) );
  
  //cutilSafeCall(cudaMemcpyAsync((void *)dest,src,size,cudaMemcpyDeviceToDevice,stream));
  //cutilSafeCall ( cudaDeviceSynchronize ( ) );
  
  copy_kernel<<<1,1>>>(dest,src,size);  
  
}
