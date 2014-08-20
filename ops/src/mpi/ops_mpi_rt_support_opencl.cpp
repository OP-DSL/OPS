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

/** @brief ops mpi+opencl run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi+opencl backend
  */

#include <ops_mpi_core.h>
#include <ops_opencl_rt_support.h>


int halo_buffer_size = 0;
cl_mem halo_buffer_d = NULL;
extern ops_opencl_core OPS_opencl_core;


cl_kernel packer1_kernel;
cl_kernel packer4_kernel;
cl_kernel unpacker1_kernel;
cl_kernel unpacker4_kernel;

const char packer1_kernel_src[] = "__kernel void ops_opencl_packer1() { }";
const char packer4_kernel_src[] = "__kernel void ops_opencl_packer4() { }";
const char unpacker1_kernel_src[] = "__kernel void ops_opencl_unpacker1() { }";
const char unpacker4_kernel_src[] = "__kernel void ops_opencl_unpacker4() { }";

const char buildOpts[] = "-cl-mad-enable";

static bool isbuilt_packer1_kernel = false;
static bool isbuilt_packer4_kernel = false;
static bool isbuilt_unpacker1_kernel = false;
static bool isbuilt_unpacker4_kernel = false;

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest, const ops_int_halo *__restrict halo) {
  
  cl_int ret = 0;
  if(!isbuilt_packer1_kernel){
    //attempt to attach sources to program (not compile)
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &packer1_kernel_src, 0, &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    // Create the OpenCL kernel
    packer1_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_packer1", &ret);
    clSafeCall( ret );
    isbuilt_packer1_kernel = true;
  }  
  
  if(!isbuilt_packer4_kernel){
    //attempt to attach sources to program (not compile)
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &packer4_kernel_src, 0, &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    // Create the OpenCL kernel
    packer4_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_packer4", &ret);
    clSafeCall( ret );
    isbuilt_packer4_kernel = true;
  } 
  
  const char * __restrict src = dat->data_d+src_offset*dat->elem_size;
  
  if (halo_buffer_size<halo->count*halo->blocklength) {    
    if (halo_buffer_d!=NULL) clSafeCall( clReleaseMemObject(halo_buffer_d));
    halo_buffer_d = clCreateBuffer(OPS_opencl_core.context, CL_MEM_READ_WRITE, halo->count*halo->blocklength*4,NULL, &ret);
    halo_buffer_size = halo->count*halo->blocklength*4;
  }
  
  cl_mem device_buf = halo_buffer_d;

  if (halo->blocklength%4 == 0) {
    int num_threads = 128;
    int num_blocks = (((halo->blocklength/4) * halo->count)-1)/num_threads + 1;
    
    size_t globalWorkSize[3] = {num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};

    //ops_cuda_packer_4<<<num_blocks,num_threads>>>((const int *)src,(int *)device_buf,halo->count, halo->blocklength/4, halo->stride/4);
    clSafeCall( clSetKernelArg(packer4_kernel, 0, sizeof(cl_mem), (void*) src ));
    clSafeCall( clSetKernelArg(packer4_kernel, 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(packer4_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength/4;
    int stride = halo->stride/4;
    clSafeCall( clSetKernelArg(packer4_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(packer4_kernel, 4, sizeof(cl_int), (void*) &stride));    
    clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, packer4_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );

    
  } else {
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count)-1)/num_threads + 1;
    
    size_t globalWorkSize[3] = {num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};
    
    //ops_cuda_packer_1<<<num_blocks,num_threads>>>(src,device_buf,halo->count, halo->blocklength, halo->stride);
    clSafeCall( clSetKernelArg(packer1_kernel, 0, sizeof(cl_mem), (void*) src ));
    clSafeCall( clSetKernelArg(packer1_kernel, 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(packer1_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength;
    int stride = halo->stride;
    clSafeCall( clSetKernelArg(packer1_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(packer1_kernel, 4, sizeof(cl_int), (void*) &stride));    
    clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, packer1_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
  }
  clSafeCall( clEnqueueReadBuffer(OPS_opencl_core.command_queue, (cl_mem) device_buf, CL_TRUE, 0, halo->count*halo->blocklength, dest, 0, NULL, NULL) );
  clSafeCall( clFinish(OPS_opencl_core.command_queue) );  
  
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src, const ops_int_halo *__restrict halo) {
  
  cl_int ret = 0;
  if(!isbuilt_unpacker1_kernel){
    //attempt to attach sources to program (not compile)
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &unpacker1_kernel_src, 0, &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    // Create the OpenCL kernel
    unpacker1_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_unpacker1", &ret);
    clSafeCall( ret );
    isbuilt_unpacker1_kernel = true;
  }  
  
  if(!isbuilt_unpacker4_kernel){
    //attempt to attach sources to program (not compile)
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &unpacker4_kernel_src, 0, &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    // Create the OpenCL kernel
    unpacker4_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_unpacker4", &ret);
    clSafeCall( ret );
    isbuilt_unpacker4_kernel = true;
  }
  
  char * __restrict dest = dat->data_d+dest_offset*dat->elem_size;
  
  if (halo_buffer_size<halo->count*halo->blocklength) {
    if (halo_buffer_d!=NULL) clSafeCall( clReleaseMemObject(halo_buffer_d));
    halo_buffer_d = clCreateBuffer(OPS_opencl_core.context, CL_MEM_READ_WRITE, halo->count*halo->blocklength*4,NULL, &ret);
    halo_buffer_size = halo->count*halo->blocklength*4;
  }

  cl_mem device_buf=halo_buffer_d;
  
  clSafeCall( clEnqueueWriteBuffer(OPS_opencl_core.command_queue, (cl_mem) device_buf, CL_TRUE, 0, halo->count*halo->blocklength, src, 0, NULL, NULL) );
  clSafeCall( clFinish(OPS_opencl_core.command_queue) ); 
  if (halo->blocklength%4 == 0) {
    int num_threads = 128;
    int num_blocks = (((halo->blocklength/4) * halo->count)-1)/num_threads + 1;
    
    size_t globalWorkSize[3] = {num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};
    
    //ops_cuda_unpacker_4<<<num_blocks,num_threads>>>((const int*)device_buf,(int *)dest,halo->count, halo->blocklength/4, halo->stride/4);
    clSafeCall( clSetKernelArg(unpacker4_kernel, 0, sizeof(cl_mem), (void*) dest ));
    clSafeCall( clSetKernelArg(unpacker4_kernel, 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(unpacker4_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength/4;
    int stride = halo->stride/4;
    clSafeCall( clSetKernelArg(unpacker4_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(unpacker4_kernel, 4, sizeof(cl_int), (void*) &stride));    
    clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, unpacker4_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
    
  } else {
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count)-1)/num_threads + 1;
    
    size_t globalWorkSize[3] = {num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};
    
    //ops_cuda_unpacker_1<<<num_blocks,num_threads>>>(device_buf,dest,halo->count, halo->blocklength, halo->stride);
    clSafeCall( clSetKernelArg(unpacker1_kernel, 0, sizeof(cl_mem), (void*) dest ));
    clSafeCall( clSetKernelArg(unpacker1_kernel, 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(unpacker1_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength;
    int stride = halo->stride;
    clSafeCall( clSetKernelArg(unpacker1_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(unpacker1_kernel, 4, sizeof(cl_int), (void*) &stride));    
    clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, unpacker1_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
  }
}

void ops_comm_realloc(char **ptr, int size, int prev) {
  if (*ptr == NULL) {
    *ptr = (char *)malloc(size);
  } else {
    *ptr = (char*)realloc(*ptr, size);
  }
}
