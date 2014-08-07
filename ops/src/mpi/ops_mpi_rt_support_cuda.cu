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

/** @brief ops mpi run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi backend
  */


#include <ops_mpi_core.h>
#include <ops_cuda_rt_support.h>

#ifdef __cplusplus
extern "C" {
#endif

int halo_buffer_size = 0;
char *halo_buffer_d = NULL;

__global__ void ops_cuda_packer_4(const int * __restrict src, int *__restrict dest, int count, int len, int stride){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int block = idx/len;
  if (idx < count*len) {
    dest[idx] = src[stride*block + idx%len];
  }
}

__global__ void ops_cuda_packer_1(const char * __restrict src, char *__restrict dest, int count, int len, int stride){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int block = idx/len;
  if (idx < count*len) {
    dest[idx] = src[stride*block + idx%len];
  }
}

__global__ void ops_cuda_unpacker_4(const int * __restrict src, int *__restrict dest, int count, int len, int stride){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int block = idx/len;
  if (idx < count*len) {
    dest[stride*block + idx%len] = src[idx];
  }
}

__global__ void ops_cuda_unpacker_1(const char * __restrict src, char *__restrict dest, int count, int len, int stride){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int block = idx/len;
  if (idx < count*len) {
    dest[stride*block + idx%len] = src[idx];
  }
}


void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest, const ops_halo *__restrict halo) {
  const char * __restrict src = dat->data_d+src_offset*dat->elem_size;
  if (halo_buffer_size<halo->count*halo->blocklength && !OPS_gpu_direct) {
    if (halo_buffer_d!=NULL) cutilSafeCall(cudaFree(halo_buffer_d));
    cutilSafeCall(cudaMalloc((void**)&halo_buffer_d,halo->count*halo->blocklength*4));
    halo_buffer_size = halo->count*halo->blocklength*4;
  }
  char *device_buf=NULL;
  if (OPS_gpu_direct) device_buf = dest;
  else device_buf = halo_buffer_d;

  if (halo->blocklength%4 == 0) {
    int num_threads = 128;
    int num_blocks = (((halo->blocklength/4) * halo->count)-1)/num_threads + 1;
    ops_cuda_packer_4<<<num_blocks,num_threads>>>((const int *)src,(int *)device_buf,halo->count, halo->blocklength/4, halo->stride/4);
  } else {
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count)-1)/num_threads + 1;
    ops_cuda_packer_1<<<num_blocks,num_threads>>>(src,device_buf,halo->count, halo->blocklength, halo->stride);
  }
  if (!OPS_gpu_direct)
    cutilSafeCall(cudaMemcpy(dest,halo_buffer_d,halo->count*halo->blocklength,cudaMemcpyDeviceToHost));
  else
    cutilSafeCall(cudaDeviceSynchronize());
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src, const ops_halo *__restrict halo) {
  char * __restrict dest = dat->data_d+dest_offset*dat->elem_size;
  if (halo_buffer_size<halo->count*halo->blocklength && !OPS_gpu_direct) {
    if (halo_buffer_d!=NULL) cutilSafeCall(cudaFree(halo_buffer_d));
    cutilSafeCall(cudaMalloc((void**)&halo_buffer_d,halo->count*halo->blocklength*4));
    halo_buffer_size = halo->count*halo->blocklength*4;
  }

  const char *device_buf=NULL;
  if (OPS_gpu_direct) device_buf = src;
  else device_buf = halo_buffer_d;

  if (!OPS_gpu_direct)
    cutilSafeCall(cudaMemcpy(halo_buffer_d,src,halo->count*halo->blocklength,cudaMemcpyHostToDevice));
  if (halo->blocklength%4 == 0) {
    int num_threads = 128;
    int num_blocks = (((halo->blocklength/4) * halo->count)-1)/num_threads + 1;
    ops_cuda_unpacker_4<<<num_blocks,num_threads>>>((const int*)device_buf,(int *)dest,halo->count, halo->blocklength/4, halo->stride/4);
  } else {
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count)-1)/num_threads + 1;
    ops_cuda_unpacker_1<<<num_blocks,num_threads>>>(device_buf,dest,halo->count, halo->blocklength, halo->stride);
  }
 //cutilSafeCall(cudaDeviceSynchronize());
}

void ops_comm_realloc(char **ptr, int size, int prev) {
  if (OPS_gpu_direct) {
    if (*ptr == NULL) {
      cutilSafeCall(cudaMalloc((void**)ptr, size));
    } else {
      printf("Warning: cuda cache realloc\n");
      char *ptr2;
      cutilSafeCall(cudaMalloc((void**)&ptr2, size));
      cutilSafeCall(cudaMemcpy(ptr2, *ptr, prev, cudaMemcpyDeviceToDevice));
      cutilSafeCall(cudaFree(*ptr));
      *ptr = ptr2;
    }
  } else {
    if (*ptr == NULL) {
      *ptr = (char *)malloc(size);
    } else {
      *ptr = (char*)realloc(*ptr, size);
    }
  }
}

#ifdef __cplusplus
}
#endif
