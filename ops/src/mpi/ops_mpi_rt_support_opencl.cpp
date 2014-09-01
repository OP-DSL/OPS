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


cl_kernel* packer1_kernel = NULL;
cl_kernel* packer4_kernel = NULL;
cl_kernel* unpacker1_kernel = NULL;
cl_kernel* unpacker4_kernel = NULL;

const char packer1_kernel_src[] = "__kernel void ops_opencl_packer1("
                                  "__global const char* restrict src,  __global char* restrict dest,"
                                  "const int count,"
                                  "const int blk_len,"
                                  "const int stride,"
                                  "const int src_offset,"
                                  "const int elem_size"
                                  ") {"
                                  "  int idx = get_global_id(0);"
                                  "  int block = idx/blk_len;"
                                  "  if (idx < count*blk_len) {"
                                  //"  printf(\"executing packer1_kernel_src %d %d %d %d %d, %d\\n\", idx, count, blk_len, stride, src_offset, elem_size);"
                                  "    dest[idx] = src[src_offset*elem_size + stride*block + idx%blk_len];"
                                  "  }"
                                  "}\n";

const char unpacker1_kernel_src[] = "__kernel void ops_opencl_unpacker1("
                                  "__global const char* restrict src,  __global char* restrict dest,"
                                  "const int count,"
                                  "const int blk_len,"
                                  "const int stride,"
                                  "const int dest_offset,"
                                  "const int elem_size"
                                  ") {"
                                  //"  printf(\"executing unpacker1_kernel_src\\n\");"
                                  "  int idx = get_global_id(0);"
                                  "  int block = idx/blk_len;"
                                  "  if (idx < count*blk_len) {"
                                  "    dest[dest_offset*elem_size +stride*block + idx%blk_len] = src[idx];"
                                  "  }"
                                  "}\n";

const char packer4_kernel_src[] = "__kernel void ops_opencl_packer4("
                                  "__global const int* restrict src,  __global int* restrict dest,"
                                  "const int count,"
                                  "const int len,"
                                  "const int stride) {"
                                  //"  printf(\"executing packer4_kernel_src\\n\");"
                                  /*"  int idx = get_global_id(0);"
                                  //"  int block = get_group_id(0);"
                                  "  int block = idx/len;"
                                  "  if (idx < count*len) {"
                                  "    dest[idx] = src[stride*block + idx%len];"
                                  "  }"*/
                                  "}\n";





const char unpacker4_kernel_src[] = "__kernel void ops_opencl_unpacker4("
                                  "__global const int* restrict src,  __global int* restrict dest,"
                                  "const int count,"
                                  "const int len,"
                                  "const int stride) {"
                                  //"  printf(\"executing unpacker4_kernel_src\\n\");"
                                  "  int idx = get_global_id(0);"
                                  "  int block = idx/len;"
                                  "  if (idx < count*len) {"
                                  "    dest[stride*block + idx%len] = src[idx];"
                                  "  }"
                                  "}\n";


static bool isbuilt_packer1_kernel = false;
static bool isbuilt_packer4_kernel = false;
static bool isbuilt_unpacker1_kernel = false;
static bool isbuilt_unpacker4_kernel = false;

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest, const ops_int_halo *__restrict halo) {

  cl_int ret = 0;
  if(!isbuilt_packer1_kernel){

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(packer1_kernel_src)+1;
    source_str[0] = (char*)malloc(source_size[0]);
    strcpy (source_str[0], packer1_kernel_src);

    if(packer1_kernel == NULL)
      packer1_kernel = (cl_kernel*) malloc(1*sizeof(cl_kernel));

    //attempt to attach sources to program (not compile)
    //OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &packer1_kernel_src, &source_size, &ret);
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &source_str, (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = "-g ";
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    if(ret != CL_SUCCESS) {
      char* build_log;
      size_t log_size;
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
      build_log = (char*) malloc(log_size+1);
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
      build_log[log_size] = '\0';
      fprintf(stderr, "=============== OpenCL Program Build Info ================\n\n%s", build_log);
      fprintf(stderr, "\n========================================================= \n");
      free(build_log);
      exit(EXIT_FAILURE);
    }

    // Create the OpenCL kernel
    *packer1_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_packer1", &ret);
    clSafeCall( ret );
    free(source_str[0]);
    isbuilt_packer1_kernel = true;
    printf("in packer1 build\n");
  }

  /*if(!isbuilt_packer4_kernel){

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(packer4_kernel_src)+1;
    source_str[0] = (char*)malloc(source_size[0]);
    strcpy (source_str[0], packer4_kernel_src);

    //attempt to attach sources to program (not compile)
    //OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &packer4_kernel_src, 0, &ret);
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &source_str, (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = "-g ";
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    if(ret != CL_SUCCESS) {
      char* build_log;
      size_t log_size;
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
      build_log = (char*) malloc(log_size+1);
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
      build_log[log_size] = '\0';
      fprintf(stderr, "=============== OpenCL Program Build Info ================\n\n%s", build_log);
      fprintf(stderr, "\n========================================================= \n");
      free(build_log);
      exit(EXIT_FAILURE);
    }
    // Create the OpenCL kernel
    packer4_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_packer4", &ret);
    clSafeCall( ret );
    free(source_str[0]);
    isbuilt_packer4_kernel = true;
    printf("in packer4 build\n");
  }*/

  const char * __restrict src = dat->data_d+src_offset*dat->elem_size;

  if (halo_buffer_size < halo->count*halo->blocklength) {
    if (halo_buffer_d != NULL) clSafeCall( clReleaseMemObject(halo_buffer_d));
    halo_buffer_d = clCreateBuffer(OPS_opencl_core.context, CL_MEM_READ_WRITE, halo->count*halo->blocklength*4,NULL, &ret);
    clSafeCall( ret );
    halo_buffer_size = halo->count*halo->blocklength*4;
  }

  cl_mem device_buf = halo_buffer_d;

  /*if (halo->blocklength%4 == 0) {
    printf("in packer4\n");
    int num_threads = 128;
    int num_blocks = (((halo->blocklength/4) * halo->count)-1)/num_threads + 1;

    size_t globalWorkSize[3] = {num_threads*num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};

    //ops_cuda_packer_4<<<num_blocks,num_threads>>>((const int *)src,(int *)device_buf,halo->count, halo->blocklength/4, halo->stride/4);
    //clSafeCall( clSetKernelArg(packer4_kernel, 0, sizeof(cl_mem), (void*) src ));
    clSafeCall( clSetKernelArg(packer4_kernel, 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(packer4_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength/4;
    int stride = halo->stride/4;
    clSafeCall( clSetKernelArg(packer4_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(packer4_kernel, 4, sizeof(cl_int), (void*) &stride));
    //clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, packer4_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );

  } else {*/
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count)-1)/num_threads + 1;

    size_t globalWorkSize[3] = {num_threads*num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};

    //ops_cuda_packer_1<<<num_blocks,num_threads>>>(src,device_buf,halo->count, halo->blocklength, halo->stride);
    //clSafeCall( clSetKernelArg(*packer1_kernel, 0, sizeof(cl_mem), (void*) src ));
    clSafeCall( clSetKernelArg(packer1_kernel[0], 0, sizeof(cl_mem), (void*) &dat->data_d ));
    clSafeCall( clSetKernelArg(packer1_kernel[0], 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(packer1_kernel[0], 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength;
    int stride = halo->stride;
    clSafeCall( clSetKernelArg(packer1_kernel[0], 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(packer1_kernel[0], 4, sizeof(cl_int), (void*) &stride));
    clSafeCall( clSetKernelArg(packer1_kernel[0], 5, sizeof(cl_int), (void*) &src_offset));
    clSafeCall( clSetKernelArg(packer1_kernel[0], 6, sizeof(cl_int), (void*) &dat->elem_size ));
    clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, *packer1_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
  //}
  clSafeCall( clFinish(OPS_opencl_core.command_queue) );
  clSafeCall( clEnqueueReadBuffer(OPS_opencl_core.command_queue, (cl_mem) device_buf, CL_TRUE, 0, halo->count*halo->blocklength, dest, 0, NULL, NULL) );
  clSafeCall( clFinish(OPS_opencl_core.command_queue) );

}

void ops_unpack3(ops_dat dat, const int dest_offset, const char *__restrict src, const ops_int_halo *__restrict halo) {

  cl_int ret = 0;
  if(!isbuilt_unpacker1_kernel){

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(unpacker1_kernel_src)+1;
    source_str[0] = (char*)malloc(source_size[0]);
    strcpy (source_str[0], unpacker1_kernel_src);

    if(unpacker1_kernel == NULL)
      unpacker1_kernel = (cl_kernel*) malloc(1*sizeof(cl_kernel));

    //attempt to attach sources to program (not compile)
    //OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &unpacker1_kernel_src, 0, &ret);
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &source_str, (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = "-g ";
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    if(ret != CL_SUCCESS) {
      char* build_log;
      size_t log_size;
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
      build_log = (char*) malloc(log_size+1);
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
      build_log[log_size] = '\0';
      fprintf(stderr, "=============== OpenCL Program Build Info ================\n\n%s", build_log);
      fprintf(stderr, "\n========================================================= \n");
      free(build_log);
      exit(EXIT_FAILURE);
    }

    // Create the OpenCL kernel
    *unpacker1_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_unpacker1", &ret);
    clSafeCall( ret );
    isbuilt_unpacker1_kernel = true;
    free(source_str[0]);
  }

  /*if(!isbuilt_unpacker4_kernel){

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(unpacker4_kernel_src)+1;
    source_str[0] = (char*)malloc(source_size[0]);
    strcpy (source_str[0], unpacker4_kernel_src);

    //attempt to attach sources to program (not compile)
    //OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &unpacker4_kernel_src, 0, &ret);
    OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, 1, (const char **) &source_str, (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
     char buildOpts[] = "-g ";
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
    if(ret != CL_SUCCESS) {
      char* build_log;
      size_t log_size;
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
      build_log = (char*) malloc(log_size+1);
      clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
      build_log[log_size] = '\0';
      fprintf(stderr, "=============== OpenCL Program Build Info ================\n\n%s", build_log);
      fprintf(stderr, "\n========================================================= \n");
      free(build_log);
      exit(EXIT_FAILURE);
    }

    // Create the OpenCL kernel
    unpacker4_kernel = clCreateKernel(OPS_opencl_core.program, "ops_opencl_unpacker4", &ret);
    clSafeCall( ret );
    free(source_str[0]);
    isbuilt_unpacker4_kernel = true;
  }*/

  char * __restrict dest = dat->data_d+dest_offset*dat->elem_size;

  if (halo_buffer_size < halo->count*halo->blocklength) {
    if (halo_buffer_d!=NULL) clSafeCall( clReleaseMemObject(halo_buffer_d));
    halo_buffer_d = clCreateBuffer(OPS_opencl_core.context, CL_MEM_READ_WRITE, halo->count*halo->blocklength*4,NULL, &ret);
    halo_buffer_size = halo->count*halo->blocklength*4;
  }

  cl_mem device_buf=halo_buffer_d;

  clSafeCall( clEnqueueWriteBuffer(OPS_opencl_core.command_queue, (cl_mem) device_buf, CL_TRUE, 0, halo->count*halo->blocklength, src, 0, NULL, NULL) );
  clSafeCall( clFinish(OPS_opencl_core.command_queue) );
  /*if (halo->blocklength%4 == 0) {
    printf("in unpacker4\n");

    int num_threads = 128;
    int num_blocks = (((halo->blocklength/4) * halo->count)-1)/num_threads + 1;

    size_t globalWorkSize[3] = {num_threads*num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};

    //ops_cuda_unpacker_4<<<num_blocks,num_threads>>>((const int*)device_buf,(int *)dest,halo->count, halo->blocklength/4, halo->stride/4);
    //clSafeCall( clSetKernelArg(unpacker4_kernel, 0, sizeof(cl_mem), (void*) dest ));
    clSafeCall( clSetKernelArg(unpacker4_kernel, 1, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(unpacker4_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength/4;
    int stride = halo->stride/4;
    clSafeCall( clSetKernelArg(unpacker4_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(unpacker4_kernel, 4, sizeof(cl_int), (void*) &stride));
    //clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, unpacker4_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );

  } else {*/
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count)-1)/num_threads + 1;

    size_t globalWorkSize[3] = {num_threads*num_blocks, 1, 1};
    size_t localWorkSize[3] =  {num_threads, 1, 1};

    //ops_cuda_unpacker_1<<<num_blocks,num_threads>>>(device_buf,dest,halo->count, halo->blocklength, halo->stride);
    //clSafeCall( clSetKernelArg(unpacker1_kernel, 0, sizeof(cl_mem), (void*) dest ));
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 0, sizeof(cl_mem), (void*) &device_buf ));
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 1, sizeof(cl_mem), (void*) &dat->data_d ));
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 2, sizeof(cl_int), (void*) &halo->count ));
    int blk_length = halo->blocklength;
    int stride = halo->stride;
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 3, sizeof(cl_int), (void*) &blk_length ));
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 4, sizeof(cl_int), (void*) &stride));
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 5, sizeof(cl_int), (void*) &dest_offset));
    clSafeCall( clSetKernelArg(*unpacker1_kernel, 6, sizeof(cl_int), (void*) &dat->elem_size ));
    clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, *unpacker1_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
    clSafeCall( clFinish(OPS_opencl_core.command_queue) );
  //}
}


void ops_pack3(ops_dat dat, const int src_offset, char *__restrict dest, const ops_int_halo *__restrict halo) {
  const char * __restrict src = dat->data+src_offset*dat->elem_size;

  if(dat->dirty_hd == 2){
    ops_download_dat(dat);
    dat->dirty_hd = 0;
  }
  for (unsigned int i = 0; i < halo->count; i ++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->stride;
    dest += halo->blocklength;
  }
  //printf("copy done 1\n");
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src, const ops_int_halo *__restrict halo) {
  char * __restrict dest = dat->data+dest_offset*dat->elem_size;

  if(dat->dirty_hd == 2) {
    ops_download_dat(dat);
    dat->dirty_hd = 0;
  }
  for (unsigned int i = 0; i < halo->count; i ++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->blocklength;
    dest += halo->stride;
  }
  ops_upload_dat(dat);
  dat->dirty_hd = 1;
  //printf("copy done 2\n");
}

void ops_comm_realloc(char **ptr, int size, int prev) {
  if (*ptr == NULL) {
    *ptr = (char *)xmalloc(size);
  } else {
    *ptr = (char*)xrealloc(*ptr, size);
  }
}
