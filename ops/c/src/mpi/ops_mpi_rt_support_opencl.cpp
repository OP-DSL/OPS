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

/** @file
  * @brief OPS mpi+opencl run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi+opencl
 * backend
  */

#include <ops_mpi_core.h>
#include <ops_device_rt_support.h>
#include <ops_exceptions.h>


void ops_exit_device(OPS_instance *instance) {
    delete instance->opencl_instance;
}

int halo_buffer_size = 0;
size_t halo_buffer_size2 = 0;
cl_mem halo_buffer_d = NULL;
cl_mem halo_buffer_d2 = NULL;

cl_kernel *packer1_kernel = NULL;
cl_kernel *packer1_soa_kernel = NULL;
cl_kernel *packer4_kernel = NULL;
cl_kernel *unpacker1_kernel = NULL;
cl_kernel *unpacker1_soa_kernel = NULL;
cl_kernel *unpacker4_kernel = NULL;

const char packer1_kernel_src[] =
    "__kernel void ops_opencl_packer1("
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
    "    dest[idx] = src[src_offset*elem_size + stride*block + idx%blk_len];"
    "  }"
    "}\n";

const char packer1_soa_kernel_src[] =
    "__kernel void ops_opencl_packer1_soa("
    "__global const char* restrict src,  __global char* restrict dest,"
    "const int count,"
    "const int blk_len,"
    "const int stride,"
    "const int src_offset,"
    "const int type_size,"
    "const int dim,"
    "const int size"
    ") {"
    "  int idx = get_global_id(0);"
    "  int block = idx/blk_len;"
    "  if (idx < count*blk_len) {"
    "    for (int d = 0; d < dim; d++)"
    "      dest[idx*dim + d] = src[src_offset*type_size + stride*block + idx%blk_len + d * size];"
    "  }"
    "}\n";

const char unpacker1_kernel_src[] =
    "__kernel void ops_opencl_unpacker1("
    "__global const char* restrict src,  __global char* restrict dest,"
    "const int count,"
    "const int blk_len,"
    "const int stride,"
    "const int dest_offset,"
    "const int elem_size"
    ") {"
    "  int idx = get_global_id(0);"
    "  int block = idx/blk_len;"
    "  if (idx < count*blk_len) {"
    "    dest[dest_offset*elem_size +stride*block + idx%blk_len] = src[idx];"
    "  }"
    "}\n";

const char unpacker1_soa_kernel_src[] =
    "__kernel void ops_opencl_unpacker1_soa("
    "__global const char* restrict src,  __global char* restrict dest,"
    "const int count,"
    "const int blk_len,"
    "const int stride,"
    "const int dest_offset,"
    "const int type_size,"
    "const int dim,"
    "const int size"
    ") {"
    "  int idx = get_global_id(0);"
    "  int block = idx/blk_len;"
    "  if (idx < count*blk_len) {"
    "    for (int d = 0; d < dim; d++)"
    "      dest[dest_offset*type_size +stride*block + idx%blk_len + d * size] = src[idx*dim + d];"
    "  }"
    "}\n";


const char packer4_kernel_src[] =
    "__kernel void ops_opencl_packer4("
    "__global const int* restrict src,  __global int* restrict dest,"
    "const int count,"
    "const int blk_len,"
    "const int stride,"
    "const int src_offset,"
    "const int elem_size"
    ") {"
    "  int idx = get_global_id(0);"
    "  int block = idx/blk_len;"
    "  if (idx < count*blk_len) {"
    "    dest[idx] = src[src_offset*elem_size/sizeof(int) + stride*block + "
    "idx%blk_len];"
    "  }"
    "}\n";

const char unpacker4_kernel_src[] =
    "__kernel void ops_opencl_unpacker4("
    "__global const int* restrict src,  __global int* restrict dest,"
    "const int count,"
    "const int blk_len,"
    "const int stride,"
    "const int dest_offset,"
    "const int elem_size"
    ") {"
    "  int idx = get_global_id(0);"
    "  int block = idx/blk_len;"
    "  if (idx < count*blk_len) {"
    "    dest[dest_offset*elem_size/sizeof(int) +stride*block + idx%blk_len] = "
    "src[idx];"
    "  }"
    "}\n";

static bool isbuilt_packer1_kernel = false;
static bool isbuilt_packer1_soa_kernel = false;
static bool isbuilt_packer4_kernel = false;
static bool isbuilt_unpacker1_kernel = false;
static bool isbuilt_unpacker1_soa_kernel = false;
static bool isbuilt_unpacker4_kernel = false;

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest,
              const ops_int_halo *__restrict halo) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
    dat->dirty_hd = 0;
  }

  cl_int ret = 0;
  if (!isbuilt_packer1_kernel && !OPS_instance::getOPSInstance()->OPS_soa) {

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(packer1_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], packer1_kernel_src);

    if (packer1_kernel == NULL)
      packer1_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *packer1_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_packer1", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    isbuilt_packer1_kernel = true;
    if (OPS_instance::getOPSInstance()->OPS_diags>5) ops_printf("in packer1 build\n");
  }

  if (!isbuilt_packer1_soa_kernel && OPS_instance::getOPSInstance()->OPS_soa) {

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(packer1_soa_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], packer1_soa_kernel_src);

    if (packer1_soa_kernel == NULL)
      packer1_soa_kernel = (cl_kernel *)ops_malloc(1 * sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *packer1_soa_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_packer1_soa", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    isbuilt_packer1_soa_kernel = true;
    if (OPS_instance::getOPSInstance()->OPS_diags>5) ops_printf("in packer1 soa build\n");
  }


  if (!isbuilt_packer4_kernel && !OPS_instance::getOPSInstance()->OPS_soa) {

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(packer4_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], packer4_kernel_src);

    if (packer4_kernel == NULL)
      packer4_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }
    // Create the OpenCL kernel
    *packer4_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_packer4", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    isbuilt_packer4_kernel = true;
    if (OPS_instance::getOPSInstance()->OPS_diags>5) ops_printf("in packer4 build\n");
  }

  // const char * __restrict src = dat->data_d+src_offset*dat->elem_size;

  if (halo_buffer_size < halo->count * halo->blocklength * dat->dim) {
    if (halo_buffer_d != NULL)
      clSafeCall(clReleaseMemObject(halo_buffer_d));
    halo_buffer_d =
        clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_WRITE,
                       halo->count * halo->blocklength * 4 * dat->dim, NULL, &ret);
    clSafeCall(ret);
    halo_buffer_size = halo->count * halo->blocklength * 4 * dat->dim;
  }

  cl_mem device_buf = halo_buffer_d;

  if (OPS_instance::getOPSInstance()->OPS_soa) {
    size_t num_threads = 128;
    size_t num_blocks = ((halo->blocklength * halo->count) - 1) / num_threads + 1;

    size_t globalWorkSize[3] = {num_threads * num_blocks, 1, 1};
    size_t localWorkSize[3] = {num_threads, 1, 1};

    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 0, sizeof(cl_mem),
                              (void *)&dat->data_d));
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 1, sizeof(cl_mem),
                              (void *)&device_buf));
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 2, sizeof(cl_int),
                              (void *)&halo->count));
    int blk_length = halo->blocklength;
    int stride = halo->stride;
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 3, sizeof(cl_int),
                              (void *)&blk_length));
    clSafeCall(
        clSetKernelArg(packer1_soa_kernel[0], 4, sizeof(cl_int), (void *)&stride));
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 5, sizeof(cl_int),
                              (void *)&src_offset));
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 6, sizeof(cl_int),
                              (void *)&dat->type_size));
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 7, sizeof(cl_int),
                              (void *)&dat->dim));
    int full_size = dat->type_size * dat->size[0] * dat->size[1] * dat->size[2];
    clSafeCall(clSetKernelArg(packer1_soa_kernel[0], 8, sizeof(cl_int),
                              (void *)&full_size));
    clSafeCall(clEnqueueNDRangeKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue,
                                      *packer1_soa_kernel, 3, NULL, globalWorkSize,
                                      localWorkSize, 0, NULL, NULL));

  } else if (halo->blocklength % 4 == 0) {
    size_t num_threads = 128;
    size_t num_blocks =
        (((halo->blocklength * dat->dim / 4) * halo->count) - 1) / num_threads + 1;

    size_t globalWorkSize[3] = {num_threads * num_blocks, 1, 1};
    size_t localWorkSize[3] = {num_threads, 1, 1};

    clSafeCall(clSetKernelArg(packer4_kernel[0], 0, sizeof(cl_mem),
                              (void *)&dat->data_d));
    clSafeCall(clSetKernelArg(packer4_kernel[0], 1, sizeof(cl_mem),
                              (void *)&device_buf));
    clSafeCall(clSetKernelArg(packer4_kernel[0], 2, sizeof(cl_int),
                              (void *)&halo->count));
    int blk_length = halo->blocklength * dat->dim / 4;
    int stride = halo->stride * dat->dim / 4;
    clSafeCall(clSetKernelArg(packer4_kernel[0], 3, sizeof(cl_int),
                              (void *)&blk_length));
    clSafeCall(
        clSetKernelArg(packer4_kernel[0], 4, sizeof(cl_int), (void *)&stride));
    clSafeCall(clSetKernelArg(packer4_kernel[0], 5, sizeof(cl_int),
                              (void *)&src_offset));
    clSafeCall(clSetKernelArg(packer4_kernel[0], 6, sizeof(cl_int),
                              (void *)&dat->elem_size));
    clSafeCall(clEnqueueNDRangeKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue,
                                      *packer4_kernel, 3, NULL, globalWorkSize,
                                      localWorkSize, 0, NULL, NULL));

  } else {
    size_t num_threads = 128;
    size_t num_blocks = ((halo->blocklength * dat->dim * halo->count) - 1) / num_threads + 1;

    size_t globalWorkSize[3] = {num_threads * num_blocks, 1, 1};
    size_t localWorkSize[3] = {num_threads, 1, 1};

    clSafeCall(clSetKernelArg(packer1_kernel[0], 0, sizeof(cl_mem),
                              (void *)&dat->data_d));
    clSafeCall(clSetKernelArg(packer1_kernel[0], 1, sizeof(cl_mem),
                              (void *)&device_buf));
    clSafeCall(clSetKernelArg(packer1_kernel[0], 2, sizeof(cl_int),
                              (void *)&halo->count));
    int blk_length = halo->blocklength * dat->dim;
    int stride = halo->stride * dat->dim;
    clSafeCall(clSetKernelArg(packer1_kernel[0], 3, sizeof(cl_int),
                              (void *)&blk_length));
    clSafeCall(
        clSetKernelArg(packer1_kernel[0], 4, sizeof(cl_int), (void *)&stride));
    clSafeCall(clSetKernelArg(packer1_kernel[0], 5, sizeof(cl_int),
                              (void *)&src_offset));
    clSafeCall(clSetKernelArg(packer1_kernel[0], 6, sizeof(cl_int),
                              (void *)&dat->elem_size));
    clSafeCall(clEnqueueNDRangeKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue,
                                      *packer1_kernel, 3, NULL, globalWorkSize,
                                      localWorkSize, 0, NULL, NULL));
  }
  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
  clSafeCall(clEnqueueReadBuffer(
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, (cl_mem)device_buf, CL_TRUE, 0,
      halo->count * halo->blocklength * dat->dim, dest, 0, NULL, NULL));
  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src,
                const ops_int_halo *__restrict halo) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
    dat->dirty_hd = 0;
  }

  cl_int ret = 0;
  if (!isbuilt_unpacker1_soa_kernel && OPS_instance::getOPSInstance()->OPS_soa) {

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(unpacker1_soa_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], unpacker1_soa_kernel_src);

    if (unpacker1_soa_kernel == NULL)
      unpacker1_soa_kernel = (cl_kernel *)ops_malloc(1 * sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *unpacker1_soa_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_unpacker1_soa", &ret);
    clSafeCall(ret);
    isbuilt_unpacker1_soa_kernel = true;
    ops_free(source_str[0]);
  }

  if (!isbuilt_unpacker1_kernel && !OPS_instance::getOPSInstance()->OPS_soa) {

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(unpacker1_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], unpacker1_kernel_src);

    if (unpacker1_kernel == NULL)
      unpacker1_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *unpacker1_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_unpacker1", &ret);
    clSafeCall(ret);
    isbuilt_unpacker1_kernel = true;
    ops_free(source_str[0]);
  }

  if (!isbuilt_unpacker4_kernel && !OPS_instance::getOPSInstance()->OPS_soa) {

    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(unpacker4_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], unpacker4_kernel_src);

    if (unpacker4_kernel == NULL)
      unpacker4_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *unpacker4_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_unpacker4", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    isbuilt_unpacker4_kernel = true;
  }

  // char * __restrict dest = dat->data_d+dest_offset*dat->elem_size;

  if (halo_buffer_size < halo->count * halo->blocklength) {
    if (halo_buffer_d != NULL)
      clSafeCall(clReleaseMemObject(halo_buffer_d));
    halo_buffer_d =
        clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_WRITE,
                       halo->count * halo->blocklength * 4, NULL, &ret);
    halo_buffer_size = halo->count * halo->blocklength * 4;
  }

  // char * __restrict dest = dat->data_d+dest_offset*dat->elem_size;

  if (halo_buffer_size < halo->count * halo->blocklength * dat->dim) {
    if (halo_buffer_d != NULL)
      clSafeCall(clReleaseMemObject(halo_buffer_d));
    halo_buffer_d =
        clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_WRITE,
                       halo->count * halo->blocklength * dat->dim * 4, NULL, &ret);
    halo_buffer_size = halo->count * halo->blocklength * dat->dim * 4;
  }

  cl_mem device_buf = halo_buffer_d;

  clSafeCall(clEnqueueWriteBuffer(
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, (cl_mem)device_buf, CL_TRUE, 0,
      halo->count * halo->blocklength * dat->dim, src, 0, NULL, NULL));
  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));

  if (OPS_instance::getOPSInstance()->OPS_soa) {
    size_t num_threads = 128;
    size_t num_blocks = ((halo->blocklength * halo->count) - 1) / num_threads + 1;

    size_t globalWorkSize[3] = {num_threads * num_blocks, 1, 1};
    size_t localWorkSize[3] = {num_threads, 1, 1};

    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 0, sizeof(cl_mem),
                              (void *)&device_buf));
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 1, sizeof(cl_mem),
                              (void *)&dat->data_d));
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 2, sizeof(cl_int),
                              (void *)&halo->count));
    int blk_length = halo->blocklength ;
    int stride = halo->stride;
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 3, sizeof(cl_int),
                              (void *)&blk_length));
    clSafeCall(
        clSetKernelArg(*unpacker1_soa_kernel, 4, sizeof(cl_int), (void *)&stride));
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 5, sizeof(cl_int),
                              (void *)&dest_offset));
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 6, sizeof(cl_int),
                              (void *)&dat->type_size));
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 7, sizeof(cl_int),
                              (void *)&dat->dim));
    int full_size = dat->type_size * dat->size[0] * dat->size[1] * dat->size[2];
    clSafeCall(clSetKernelArg(*unpacker1_soa_kernel, 8, sizeof(cl_int),
                              (void *)&full_size));
    clSafeCall(clEnqueueNDRangeKernel(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, *unpacker1_soa_kernel, 3, NULL,
        globalWorkSize, localWorkSize, 0, NULL, NULL));
    clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
  } else  if (halo->blocklength % 4 == 0) {
    size_t num_threads = 128;
    size_t num_blocks =
        (((halo->blocklength * dat->dim / 4) * halo->count) - 1) / num_threads + 1;

    size_t globalWorkSize[3] = {num_threads * num_blocks, 1, 1};
    size_t localWorkSize[3] = {num_threads, 1, 1};

    clSafeCall(clSetKernelArg(*unpacker4_kernel, 0, sizeof(cl_mem),
                              (void *)&device_buf));
    clSafeCall(clSetKernelArg(*unpacker4_kernel, 1, sizeof(cl_mem),
                              (void *)&dat->data_d));
    clSafeCall(clSetKernelArg(*unpacker4_kernel, 2, sizeof(cl_int),
                              (void *)&halo->count));
    int blk_length = halo->blocklength * dat->dim / 4;
    int stride = halo->stride * dat->dim / 4;
    clSafeCall(clSetKernelArg(*unpacker4_kernel, 3, sizeof(cl_int),
                              (void *)&blk_length));
    clSafeCall(
        clSetKernelArg(*unpacker4_kernel, 4, sizeof(cl_int), (void *)&stride));
    clSafeCall(clSetKernelArg(*unpacker4_kernel, 5, sizeof(cl_int),
                              (void *)&dest_offset));
    clSafeCall(clSetKernelArg(*unpacker4_kernel, 6, sizeof(cl_int),
                              (void *)&dat->elem_size));
    clSafeCall(clEnqueueNDRangeKernel(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, *unpacker4_kernel, 3, NULL,
        globalWorkSize, localWorkSize, 0, NULL, NULL));

  } else {
    size_t num_threads = 128;
    size_t num_blocks = ((halo->blocklength * dat->dim * halo->count) - 1) / num_threads + 1;

    size_t globalWorkSize[3] = {num_threads * num_blocks, 1, 1};
    size_t localWorkSize[3] = {num_threads, 1, 1};

    clSafeCall(clSetKernelArg(*unpacker1_kernel, 0, sizeof(cl_mem),
                              (void *)&device_buf));
    clSafeCall(clSetKernelArg(*unpacker1_kernel, 1, sizeof(cl_mem),
                              (void *)&dat->data_d));
    clSafeCall(clSetKernelArg(*unpacker1_kernel, 2, sizeof(cl_int),
                              (void *)&halo->count));
    int blk_length = halo->blocklength * dat->dim;
    int stride = halo->stride * dat->dim;
    clSafeCall(clSetKernelArg(*unpacker1_kernel, 3, sizeof(cl_int),
                              (void *)&blk_length));
    clSafeCall(
        clSetKernelArg(*unpacker1_kernel, 4, sizeof(cl_int), (void *)&stride));
    clSafeCall(clSetKernelArg(*unpacker1_kernel, 5, sizeof(cl_int),
                              (void *)&dest_offset));
    clSafeCall(clSetKernelArg(*unpacker1_kernel, 6, sizeof(cl_int),
                              (void *)&dat->elem_size));
    clSafeCall(clEnqueueNDRangeKernel(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, *unpacker1_kernel, 3, NULL,
        globalWorkSize, localWorkSize, 0, NULL, NULL));
    clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
  }

  dat->dirty_hd = 2;
}
//TODO: what are these?
void ops_pack3(ops_dat dat, const int src_offset, char *__restrict dest,
               const ops_int_halo *__restrict halo) {
  const char *__restrict src = dat->data + src_offset * dat->elem_size;

  if (dat->dirty_hd == 2) {
    ops_download_dat(dat);
    dat->dirty_hd = 0;
  }
  for (int i = 0; i < halo->count; i++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->stride;
    dest += halo->blocklength;
  }
}

void ops_unpack3(ops_dat dat, const int dest_offset, const char *__restrict src,
                 const ops_int_halo *__restrict halo) {
  char *__restrict dest = dat->data + dest_offset * dat->elem_size;

  if (dat->dirty_hd == 2) {
    ops_download_dat(dat);
    dat->dirty_hd = 0;
  }
  for (int i = 0; i < halo->count; i++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->blocklength;
    dest += halo->stride;
  }
  // ops_upload_dat(dat);
  dat->dirty_hd = 1;
}

char *OPS_realloc_fast(char *ptr, size_t olds, size_t news) {
  return (char*)ops_realloc(ptr, news);
}

const char copy_tobuf_kernel_src[] =
    "__kernel void ops_opencl_copy_tobuf("
    "__global char * restrict dest, __global char * restrict src,"
    "const int rx_s, const int rx_e,"
    "const int ry_s, const int ry_e,"
    "const int rz_s, const int rz_e,"
    "const int x_step, const int y_step, const int z_step,"
    "const int size_x, const int size_y, const int size_z,"
    "const int buf_strides_x, const int buf_strides_y, const int buf_strides_z,"
    "const int type_size, const int dim, const int OPS_soa"
    ") {"

    "  int idx_z = rz_s + z_step*get_global_id(2);"
    "  int idx_y = ry_s + y_step*get_global_id(1);"
    "  int idx_x = rx_s + x_step*get_global_id(0);"

    "  if((x_step ==1 ? idx_x < rx_e : idx_x > rx_e) &&"
    "     (y_step ==1 ? idx_y < ry_e : idx_y > ry_e) &&"
    "     (z_step ==1 ? idx_z < rz_e : idx_z > rz_e)) {"
    "    if (OPS_soa) src   += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size;"
    "    else         src   += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size * dim;"
    "    dest  += ((idx_z-rz_s)*z_step*buf_strides_z+ "
    "              (idx_y-ry_s)*y_step*buf_strides_y + "
    "              (idx_x-rx_s)*x_step*buf_strides_x)*type_size * dim;"
    "    for(int d=0;d<dim;d++) {"
    "      for(int i=0;i<type_size;i++)"
    "        dest[d*type_size+i] = src[i];"
    "      if (OPS_soa) src += size_x * size_y * size_z * type_size;"
    "      else src += type_size;"
    "    }"
    "  }"
    "}\n";

const char copy_frombuf_kernel_src[] =
    "__kernel void ops_opencl_copy_frombuf("
    "__global char * restrict dest, __global char * restrict src,"
    "const int rx_s, const int rx_e,"
    "const int ry_s, const int ry_e,"
    "const int rz_s, const int rz_e,"
    "const int x_step, const int y_step, const int z_step,"
    "const int size_x, const int size_y, const int size_z,"
    "const int buf_strides_x, const int buf_strides_y, const int buf_strides_z,"
    "const int type_size, const int dim, const int OPS_soa"
    "){"
    "  int idx_z = rz_s + z_step*get_global_id(2);"
    "  int idx_y = ry_s + y_step*get_global_id(1);"
    "  int idx_x = rx_s + x_step*get_global_id(0);"

    "  if((x_step ==1 ? idx_x < rx_e : idx_x > rx_e) &&"
    "     (y_step ==1 ? idx_y < ry_e : idx_y > ry_e) &&"
    "     (z_step ==1 ? idx_z < rz_e : idx_z > rz_e)) {"
    "    if (OPS_soa) dest += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size;"
    "    else         dest += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size * dim;"
    "    src  += ((idx_z-rz_s)*z_step*buf_strides_z+ "
    "             (idx_y-ry_s)*y_step*buf_strides_y + "
    "             (idx_x-rx_s)*x_step*buf_strides_x)*type_size * dim;"
    "    for(int d=0;d<dim;d++) {"
    "      for(int i=0;i<type_size;i++)"
    "        dest[i] = src[d*type_size+i];"
    "      if (OPS_soa) dest += size_x * size_y * size_z * type_size;"
    "      else dest += type_size;"
    "    }"
    "  }"
    "}\n";

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {

  cl_int ret = 0;
  if (!OPS_instance::getOPSInstance()->opencl_instance->isbuilt_copy_tobuf_kernel) {
    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(copy_tobuf_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], copy_tobuf_kernel_src);

    if (OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel == NULL)
      OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel =
        clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_opencl_copy_tobuf", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    OPS_instance::getOPSInstance()->opencl_instance->isbuilt_copy_tobuf_kernel = true;
    if (OPS_instance::getOPSInstance()->OPS_diags>5) ops_printf("in mpi OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel build\n");
  }

  dest += dest_offset;
  size_t thr_x = abs(rx_s - rx_e);
  size_t blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  size_t thr_y = abs(ry_s - ry_e);
  size_t blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  size_t thr_z = abs(rz_s - rz_e);
  size_t blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  size_t size =
      abs(src->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));

  if (halo_buffer_size2 < size) {
    if (halo_buffer_d2 != NULL)
      clSafeCall(clReleaseMemObject(halo_buffer_d2));
    halo_buffer_d2 = clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_WRITE,
                                    size * 4, NULL, &ret);
    clSafeCall(ret);
    halo_buffer_size2 = size;
  }
  cl_mem gpu_ptr = halo_buffer_d2;

  if (src->dirty_hd == 1) {
    ops_upload_dat(src);
    src->dirty_hd = 0;
  }

  size_t globalWorkSize[3] = {blk_x * thr_x, blk_y * thr_y, blk_z * thr_z};
  size_t localWorkSize[3] = {thr_x, thr_y, thr_z};

  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 0, sizeof(cl_mem),
                            (void *)&gpu_ptr));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 1, sizeof(cl_mem),
                            (void *)&src->data_d));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 2, sizeof(cl_int), (void *)&rx_s));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 3, sizeof(cl_int), (void *)&rx_e));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 4, sizeof(cl_int), (void *)&ry_s));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 5, sizeof(cl_int), (void *)&ry_e));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 6, sizeof(cl_int), (void *)&rz_s));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 7, sizeof(cl_int), (void *)&rz_e));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 8, sizeof(cl_int), (void *)&x_step));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 9, sizeof(cl_int), (void *)&y_step));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 10, sizeof(cl_int),
                            (void *)&z_step));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 11, sizeof(cl_int),
                            (void *)&src->size[0]));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 12, sizeof(cl_int),
                            (void *)&src->size[1]));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 13, sizeof(cl_int),
                            (void *)&src->size[2]));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 14, sizeof(cl_int),
                            (void *)&buf_strides_x));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 15, sizeof(cl_int),
                            (void *)&buf_strides_y));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 16, sizeof(cl_int),
                            (void *)&buf_strides_z));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 17, sizeof(cl_int),
                            (void *)&src->type_size));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 18, sizeof(cl_int),
                            (void *)&src->dim));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel[0], 19, sizeof(cl_int),
                            (void *)&OPS_instance::getOPSInstance()->OPS_soa));

  clSafeCall(clEnqueueNDRangeKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue,
                                    *OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel, 3, NULL, globalWorkSize,
                                    localWorkSize, 0, NULL, NULL));

  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
  clSafeCall(clEnqueueReadBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue,
                                 (cl_mem)halo_buffer_d2, CL_TRUE, 0,
                                 size * sizeof(char), dest, 0, NULL, NULL));
  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {
  cl_int ret = 0;
  if (!OPS_instance::getOPSInstance()->opencl_instance->isbuilt_copy_frombuf_kernel) {
    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(copy_frombuf_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], copy_frombuf_kernel_src);

    if (OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel == NULL)
      OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      OPSException ex(OPS_OPENCL_BUILD_ERROR);
      ex << build_log;
      ops_free(build_log);
      throw ex;
    }

    // Create the OpenCL kernel
    *OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel = clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program,
                                          "ops_opencl_copy_frombuf", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    OPS_instance::getOPSInstance()->opencl_instance->isbuilt_copy_frombuf_kernel = true;
    if (OPS_instance::getOPSInstance()->OPS_diags>5) ops_printf("in mpi OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel build\n");
  }

  src += src_offset;
  size_t thr_x = abs(rx_s - rx_e);
  size_t blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  size_t thr_y = abs(ry_s - ry_e);
  size_t blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  size_t thr_z = abs(rz_s - rz_e);
  size_t blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  size_t size =
      abs(dest->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));
  cl_mem gpu_ptr;
  if (halo_buffer_size2 < size) {
    if (halo_buffer_d2 != NULL)
      clSafeCall(clReleaseMemObject(halo_buffer_d2));
    halo_buffer_d2 = clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_WRITE,
                                    size, NULL, &ret);
    clSafeCall(ret);
    halo_buffer_size2 = size;
  }
  gpu_ptr = halo_buffer_d2;

  if (dest->dirty_hd == 1) {
    ops_upload_dat(dest);
    dest->dirty_hd = 0;
  }

  size_t globalWorkSize[3] = {blk_x * thr_x, blk_y * thr_y, blk_z * thr_z};
  size_t localWorkSize[3] = {thr_x, thr_y, thr_z};

  clSafeCall(clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue,
                                  (cl_mem)halo_buffer_d2, CL_TRUE, 0,
                                  size * sizeof(char), src, 0, NULL, NULL));
  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));

  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 0, sizeof(cl_mem),
                            (void *)&dest->data_d));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 1, sizeof(cl_mem),
                            (void *)&gpu_ptr));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 2, sizeof(cl_int), (void *)&rx_s));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 3, sizeof(cl_int), (void *)&rx_e));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 4, sizeof(cl_int), (void *)&ry_s));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 5, sizeof(cl_int), (void *)&ry_e));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 6, sizeof(cl_int), (void *)&rz_s));
  clSafeCall(
      clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 7, sizeof(cl_int), (void *)&rz_e));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 8, sizeof(cl_int),
                            (void *)&x_step));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 9, sizeof(cl_int),
                            (void *)&y_step));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 10, sizeof(cl_int),
                            (void *)&z_step));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 11, sizeof(cl_int),
                            (void *)&dest->size[0]));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 12, sizeof(cl_int),
                            (void *)&dest->size[1]));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 13, sizeof(cl_int),
                            (void *)&dest->size[2]));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 14, sizeof(cl_int),
                            (void *)&buf_strides_x));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 15, sizeof(cl_int),
                            (void *)&buf_strides_y));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 16, sizeof(cl_int),
                            (void *)&buf_strides_z));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 17, sizeof(cl_int),
                            (void *)&dest->type_size));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 18, sizeof(cl_int),
                            (void *)&dest->dim));
  clSafeCall(clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel[0], 19, sizeof(cl_int),
                            (void *)&OPS_instance::getOPSInstance()->OPS_soa));
  clSafeCall(clEnqueueNDRangeKernel(
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, *OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel, 3, NULL,
      globalWorkSize, localWorkSize, 0, NULL, NULL));

  dest->dirty_hd = 2;
}


const char copy_opencl_kernel_src[] =
"__kernel void ops_copy_opencl_kernel(\n"
"__global char * restrict dat0_p, __global char * restrict dat1_p,\n"
"        int base0, int base1,\n"
"        int dim, int type_size, int OPS_soa,\n"
"         int s0, int start0, int end0\n"
"#if OPS_MAX_DIM>1\n"
"        , int s1, int start1, int end1\n"
"#if OPS_MAX_DIM>2\n"
"        , int s2, int start2, int end2\n"
"#if OPS_MAX_DIM>3\n"
"        , int s3, int start3, int end3\n"
"#if OPS_MAX_DIM>4\n"
"        , int s4, int start4, int end4\n"
"#endif\n"
"#endif\n"
"#endif\n"
"#endif\n"
"        ) {\n"
"  int i = start0 + get_global_id(0);\n"
"  int j = start1 + get_global_id(1);\n"
"  int rest = get_global_id(2);\n"
"  int mult = OPS_soa ? type_size : dim*type_size;\n"
"\n"
"    long fullsize = s0;\n"
"    long idx = i*mult;\n"
"#if OPS_MAX_DIM>1\n"
"    fullsize *= s1;\n"
"    idx += j * s0 * mult;\n"
"#endif\n"
"#if OPS_MAX_DIM>2\n"
"    fullsize *= s2;\n"
"    int k = start2+rest%s2;\n"
"    idx += k * s0 * s1 * mult;\n"
"#endif\n"
"#if OPS_MAX_DIM>3\n"
"    fullsize *= s3;\n"
"    int l = start3+rest/s2;\n"
"    idx += l * s0 * s1 * s2 * mult;\n"
"#endif\n"
"#if OPS_MAX_DIM>3\n"
"    fullsize *= s4;\n"
"    int m = start4+rest/(s2*s3);\n"
"    idx += m * s0 * s1 * s2 * s3 * mult;\n"
"#endif\n"
"    if (i<end0\n"
"#if OPS_MAX_DIM>1\n"
"        && j < end1\n"
"#if OPS_MAX_DIM>2\n"
"        && k < end2\n"
"#if OPS_MAX_DIM>3\n"
"        && l < end3\n"
"#if OPS_MAX_DIM>4\n"
"        && m < end4\n"
"#endif\n"
"#endif\n"
"#endif\n"
"#endif\n"
"       )\n"
"    if (OPS_soa) { \n"
"      for (int d = 0; d < dim; d++)\n"
"        for (int c = 0; c < type_size; d++)\n"
"          dat1_p[base1+idx+d*fullsize*type_size+c] = dat0_p[base0+idx+d*fullsize*type_size+c];\n"
"    } else\n"
"      for (int d = 0; d < dim*type_size; d++)\n"
"        dat1_p[base1+idx+d] = dat0_p[base0+idx+d];\n"
"\n"
"}\n";

void ops_internal_copy_opencl(ops_kernel_descriptor *desc) {
  int range[2*OPS_MAX_DIM]={0};
  for (int d = 0; d < desc->dim; d++) {
    range[2*d] = desc->range[2*d];
    range[2*d+1] = desc->range[2*d+1];
  }
  for (int d = desc->dim; d < OPS_MAX_DIM; d++) {
    range[2*d] = 0;
    range[2*d+1] = 1;
  }
  ops_dat dat0 = desc->args[0].dat;
  double __t1 = 0.,__t2 = 0.,__c1 = 0.,__c2 = 0.;
  if (dat0->block->instance->OPS_diags>1) {
    dat0->block->instance->OPS_kernels[-1].count++;
    ops_timers_core(&__c1,&__t1);
  }
  char *dat0_p = desc->args[0].data_d;
  char *dat1_p = desc->args[1].data_d;
  int s0 = dat0->size[0];
#if OPS_MAX_DIM>1
  int s1 = dat0->size[1];
#if OPS_MAX_DIM>2
  int s2 = dat0->size[2];
#if OPS_MAX_DIM>3
  int s3 = dat0->size[3];
#if OPS_MAX_DIM>4
  int s4 = dat0->size[4];
#endif
#endif
#endif
#endif
  ops_block block = dat0->block;

  cl_int ret = 0;
  if (!block->instance->opencl_instance->isbuilt_copy_opencl_kernel) {
    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(copy_opencl_kernel_src) + 1;
    source_str[0] = (char *)ops_malloc(source_size[0]);
    strcpy(source_str[0], copy_opencl_kernel_src);

    if (block->instance->opencl_instance->copy_opencl_kernel == NULL)
      block->instance->opencl_instance->copy_opencl_kernel = (cl_kernel *)ops_calloc(1 , sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    block->instance->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(
        block->instance->opencl_instance->OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      clSafeCall(ret);
      return;
    }
    char buildOpts[16];
    sprintf(buildOpts,"-DOPS_MAX_DIM=%d",OPS_MAX_DIM);
    ret = clBuildProgram(block->instance->opencl_instance->OPS_opencl_core.program, 1, &block->instance->opencl_instance->OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          block->instance->opencl_instance->OPS_opencl_core.program, block->instance->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)ops_malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          block->instance->opencl_instance->OPS_opencl_core.program, block->instance->opencl_instance->OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      block->instance->ostream() <<
          "=============== OpenCL Program Build Info ================\n\n" <<
          build_log;
      block->instance->ostream() <<
              "\n========================================================= \n";
      throw OPSException(OPS_OPENCL_BUILD_ERROR, build_log);
      ops_free(build_log);
    }

    // Create the OpenCL kernel
    *block->instance->opencl_instance->copy_opencl_kernel = clCreateKernel(block->instance->opencl_instance->OPS_opencl_core.program,
                                          "ops_copy_opencl_kernel", &ret);
    clSafeCall(ret);
    ops_free(source_str[0]);
    block->instance->opencl_instance->isbuilt_copy_opencl_kernel = true;
    if (block->instance->OPS_diags>5 && block->instance->is_root()) block->instance->ostream() << "in copy_opencl_kernel build\n";
  }



  size_t thr_x = dat0->block->instance->OPS_block_size_x;
  size_t thr_y = dat0->block->instance->OPS_block_size_y;
  size_t thr_z = dat0->block->instance->OPS_block_size_z;
  size_t blk_x = (range[2*0+1]-range[2*0] - 1) / dat0->block->instance->OPS_block_size_x + 1;
  size_t blk_y = (range[2*1+1]-range[2*1] - 1) / dat0->block->instance->OPS_block_size_y + 1;
  size_t blk_z = ((range[2*2+1]-range[2*2] - 1) / dat0->block->instance->OPS_block_size_z + 1) *
            (range[2*3+1]-range[2*3]) *
            (range[2*4+1]-range[2*4]);
  size_t globalWorkSize[3] = {blk_x * thr_x, blk_y * thr_y, blk_z * thr_z};
  size_t localWorkSize[3] = {thr_x, thr_y, thr_z};

  if (blk_x>0 && blk_y>0 && blk_z>0) {

    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 0, sizeof(cl_mem), (void *)&dat0_p));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 1, sizeof(cl_mem), (void *)&dat1_p));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 2, sizeof(cl_int), (void*) &desc->args[0].dat->base_offset ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 3, sizeof(cl_int), (void*) &desc->args[1].dat->base_offset ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 4, sizeof(cl_int), (void*) &dat0->dim ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 5, sizeof(cl_int), (void*) &dat0->type_size ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 6, sizeof(cl_int), (void*) &dat0->block->instance->OPS_soa ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 7, sizeof(cl_int), (void*) &s0 ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 8, sizeof(cl_int), (void*) &range[2*0] ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 9, sizeof(cl_int), (void*) &range[2*0+1] ));
#if OPS_MAX_DIM>1
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 10, sizeof(cl_int), (void*) &s1 ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 11, sizeof(cl_int), (void*) &range[2*1] ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 12, sizeof(cl_int), (void*) &range[2*1+1] ));
#if OPS_MAX_DIM>2
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 13, sizeof(cl_int), (void*) &s2 ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 14, sizeof(cl_int), (void*) &range[2*2] ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 15, sizeof(cl_int), (void*) &range[2*2+1] ));
#if OPS_MAX_DIM>3
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 16, sizeof(cl_int), (void*) &s3 ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 17, sizeof(cl_int), (void*) &range[2*3] ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 18, sizeof(cl_int), (void*) &range[2*3+1] ));
#if OPS_MAX_DIM>4
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 19, sizeof(cl_int), (void*) &s4 ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 20, sizeof(cl_int), (void*) &range[2*4] ));
    clSafeCall(clSetKernelArg(block->instance->opencl_instance->copy_opencl_kernel[0], 21, sizeof(cl_int), (void*) &range[2*4+1] ));
#endif
#endif
#endif
#endif
    clSafeCall( clEnqueueNDRangeKernel(block->instance->opencl_instance->OPS_opencl_core.command_queue, block->instance->opencl_instance->copy_opencl_kernel[0], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
    clSafeCall( clFinish(block->instance->opencl_instance->OPS_opencl_core.command_queue) );
  }
  if (dat0->block->instance->OPS_diags>1) {
    ops_timers_core(&__c2,&__t2);
    int start[OPS_MAX_DIM];
    int end[OPS_MAX_DIM];
    for ( int n=0; n<desc->dim; n++ ){
      start[n] = range[2*n];end[n] = range[2*n+1];
    }
    dat0->block->instance->OPS_kernels[-1].time += __t2-__t1;
    dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[0]);
    dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[1]);
  }
}


void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_fetch_data_slab_host(dat, part, data, range);
  else {
    ops_execute(dat->block->instance);
    int range2[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range2[2*i] = range[2*i];
      range2[2*i+1] = range[2*i+1];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range2[2*i] = 0;
      range2[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_upload_dat(dat);
      dat->dirty_hd = 0;
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    size_t prod = 1;
    for (int d = 0; d < OPS_MAX_DIM; d++) {
      target->size[d] = range2[2*d+1]-range2[2*d];
      target->base_offset -= target->elem_size*prod*range2[2*d];
      prod *= target->size[d];
    }
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    desc->name = "ops_internal_copy_opencl";
    desc->device = 1;
    desc->function = ops_internal_copy_opencl;
    ops_internal_copy_opencl(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
  }
}

void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_set_data_slab_host(dat, part, data, range);
  else {
    ops_execute(dat->block->instance);
    int range2[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range2[2*i] = range[2*i];
      range2[2*i+1] = range[2*i+1];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range2[2*i] = 0;
      range2[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_upload_dat(dat);
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    size_t prod = 1;
    for (int d = 0; d < OPS_MAX_DIM; d++) {
      target->size[d] = range2[2*d+1]-range2[2*d];
      target->base_offset -= target->elem_size*prod*range2[2*d];
      prod *= target->size[d];
    }
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    desc->name = "ops_internal_copy_opencl_reverse";
    desc->device = 1;
    desc->function = ops_internal_copy_opencl;
    ops_internal_copy_opencl(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
    dat->dirty_hd = 2;
  }
}


void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_fetch_data_host(dat, part, data);
  else {
    ops_execute(dat->block->instance);
    int disp[OPS_MAX_DIM], size[OPS_MAX_DIM];
    ops_dat_get_extents(dat, 0, disp, size);
    int range[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range[2*i] = dat->base[i];
      range[2*i+1] = range[2*i] + size[i];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range[2*i] = 0;
      range[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_upload_dat(dat);
      dat->dirty_hd = 0;
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    for (int d = 0; d < OPS_MAX_DIM; d++) target->size[d] = size[d];
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    desc->name = "ops_internal_copy_opencl";
    desc->device = 1;
    desc->function = ops_internal_copy_opencl;
    ops_internal_copy_opencl(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
  }
}

void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_set_data_host(dat, part, data);
  else {
    ops_execute(dat->block->instance);
    int disp[OPS_MAX_DIM], size[OPS_MAX_DIM];
    ops_dat_get_extents(dat, 0, disp, size);
    int range[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range[2*i] = dat->base[i];
      range[2*i+1] = range[2*i] + size[i];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range[2*i] = 0;
      range[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_upload_dat(dat);
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    for (int d = 0; d < OPS_MAX_DIM; d++) target->size[d] = size[d];
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    desc->name = "ops_internal_copy_opencl_reverse";
    desc->device = 1;
    desc->function = ops_internal_copy_opencl;
    ops_internal_copy_opencl(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
    dat->dirty_hd = 2;
  }
}

