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

/** @brief ops opencl specific runtime support functions
  * @author Gihan Mudalige and Istvan Reguly (adapting OP2 OpenCL backend by
 * Endre Lazslo)
  * @details Implements opencl backend runtime support functions
  */

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

//#include <math_constants.h>

#include <ops_lib_core.h>
#include <ops_opencl_rt_support.h>

extern ops_opencl_core OPS_opencl_core;

cl_kernel *copy_tobuf_kernel = NULL;
cl_kernel *copy_frombuf_kernel = NULL;

static bool isbuilt_copy_tobuf_kernel = false;
static bool isbuilt_copy_frombuf_kernel = false;

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
    "    (y_step ==1 ? idx_y < ry_e : idx_y > ry_e) &&"
    "    (z_step ==1 ? idx_z < rz_e : idx_z > rz_e)) {"
    "    if (OPS_soa) src   += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size;"
    "    else         src   += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size * dim;"
    "    dest  += ((idx_z-rz_s)*z_step*buf_strides_z+ "
    "(idx_y-ry_s)*y_step*buf_strides_y + "
    "(idx_x-rx_s)*x_step*buf_strides_x)*type_size * dim;"
    "    for(int d=0;d<dim;d++) {"
    "      if (OPS_soa) src += size_x * size_y * size_z * type_size;"
    "      else src += type_size;"
    "      for(int i=0;i<type_size;i++)"
    "        dest[d*dim+i] = src[i];"
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
    "    (y_step ==1 ? idx_y < ry_e : idx_y > ry_e) &&"
    "    (z_step ==1 ? idx_z < rz_e : idx_z > rz_e)) {"
    "    if (OPS_soa) dest += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size;"
    "    else         dest += (idx_z*size_x*size_y + idx_y*size_x + idx_x) * type_size * dim;"
    "    src  += ((idx_z-rz_s)*z_step*buf_strides_z+ "
    "(idx_y-ry_s)*y_step*buf_strides_y + "
    "(idx_x-rx_s)*x_step*buf_strides_x)*type_size * dim;"
    "    for(int d=0;d<dim;d++) {"
    "      if (OPS_soa) dest += size_x * size_y * size_z * type_size;"
    "      else dest += type_size;"
    "      for(int i=0;i<type_size;i++)"
    "        dest[i] = src[d*dim+i];"
    "    }"
    "  }"
    "}\n";

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {

  cl_int ret = 0;
  if (!isbuilt_copy_tobuf_kernel) {
    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(copy_tobuf_kernel_src) + 1;
    source_str[0] = (char *)malloc(source_size[0]);
    strcpy(source_str[0], copy_tobuf_kernel_src);

    if (copy_tobuf_kernel == NULL)
      copy_tobuf_kernel = (cl_kernel *)malloc(1 * sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_opencl_core.program, OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_opencl_core.program, OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      free(build_log);
      exit(EXIT_FAILURE);
    }

    // Create the OpenCL kernel
    *copy_tobuf_kernel =
        clCreateKernel(OPS_opencl_core.program, "ops_opencl_copy_tobuf", &ret);
    clSafeCall(ret);
    free(source_str[0]);
    isbuilt_copy_tobuf_kernel = true;
    printf("in copy_tobuf_kernel build\n");
  }

  dest += dest_offset;
  int thr_x = abs(rx_s - rx_e);
  int blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s - ry_e);
  int blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s - rz_e);
  int blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  size_t globalWorkSize[3] = {blk_x * thr_x, blk_y * thr_y, blk_z * thr_z};
  size_t localWorkSize[3] = {thr_x, thr_y, thr_z};

  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 0, sizeof(cl_mem), (void *)&dest));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 1, sizeof(cl_mem),
                            (void *)&src->data_d));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 2, sizeof(cl_int), (void *)&rx_s));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 3, sizeof(cl_int), (void *)&rx_e));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 4, sizeof(cl_int), (void *)&ry_s));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 5, sizeof(cl_int), (void *)&ry_e));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 6, sizeof(cl_int), (void *)&rz_s));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 7, sizeof(cl_int), (void *)&rz_e));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 8, sizeof(cl_int), (void *)&x_step));
  clSafeCall(
      clSetKernelArg(copy_tobuf_kernel[0], 9, sizeof(cl_int), (void *)&y_step));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 10, sizeof(cl_int),
                            (void *)&z_step));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 11, sizeof(cl_int),
                            (void *)&src->size[0]));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 12, sizeof(cl_int),
                            (void *)&src->size[1]));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 13, sizeof(cl_int),
                            (void *)&src->size[2]));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 14, sizeof(cl_int),
                            (void *)&buf_strides_x));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 15, sizeof(cl_int),
                            (void *)&buf_strides_y));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 16, sizeof(cl_int),
                            (void *)&buf_strides_z));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 17, sizeof(cl_int),
                            (void *)&src->type_size));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 18, sizeof(cl_int),
                            (void *)&src->dim));
  clSafeCall(clSetKernelArg(copy_tobuf_kernel[0], 19, sizeof(cl_int),
                            (void *)&OPS_soa));
  clSafeCall(clEnqueueNDRangeKernel(OPS_opencl_core.command_queue,
                                    *copy_tobuf_kernel, 3, NULL, globalWorkSize,
                                    localWorkSize, 0, NULL, NULL));
  clSafeCall(clFinish(OPS_opencl_core.command_queue));
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {
  cl_int ret = 0;
  if (!isbuilt_copy_frombuf_kernel) {
    char *source_str[1];
    size_t source_size[1];
    source_size[0] = strlen(copy_frombuf_kernel_src) + 1;
    source_str[0] = (char *)malloc(source_size[0]);
    strcpy(source_str[0], copy_frombuf_kernel_src);

    if (copy_frombuf_kernel == NULL)
      copy_frombuf_kernel = (cl_kernel *)malloc(1 * sizeof(cl_kernel));

    // attempt to attach sources to program (not compile)
    OPS_opencl_core.program = clCreateProgramWithSource(
        OPS_opencl_core.context, 1, (const char **)&source_str,
        (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
      fprintf(stderr, "Error: Unable to create program from source.\n");
      clSafeCall(ret);
      return;
    }
    char buildOpts[] = " ";
    ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id,
                         buildOpts, NULL, NULL);
    if (ret != CL_SUCCESS) {
      char *build_log;
      size_t log_size;
      clSafeCall(clGetProgramBuildInfo(
          OPS_opencl_core.program, OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
      build_log = (char *)malloc(log_size + 1);
      clSafeCall(clGetProgramBuildInfo(
          OPS_opencl_core.program, OPS_opencl_core.device_id,
          CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL));
      build_log[log_size] = '\0';
      fprintf(
          stderr,
          "=============== OpenCL Program Build Info ================\n\n%s",
          build_log);
      fprintf(stderr,
              "\n========================================================= \n");
      free(build_log);
      exit(EXIT_FAILURE);
    }

    // Create the OpenCL kernel
    *copy_frombuf_kernel = clCreateKernel(OPS_opencl_core.program,
                                          "ops_opencl_copy_frombuf", &ret);
    clSafeCall(ret);
    free(source_str[0]);
    isbuilt_copy_frombuf_kernel = true;
    printf("in copy_frombuf_kernel build\n");
  }

  src += src_offset;
  int thr_x = abs(rx_s - rx_e);
  int blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s - ry_e);
  int blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s - rz_e);
  int blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  size_t globalWorkSize[3] = {blk_x * thr_x, blk_y * thr_y, blk_z * thr_z};
  size_t localWorkSize[3] = {thr_x, thr_y, thr_z};

  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 0, sizeof(cl_mem),
                            (void *)&dest->data_d));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 1, sizeof(cl_mem), (void *)&src));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 2, sizeof(cl_int), (void *)&rx_s));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 3, sizeof(cl_int), (void *)&rx_e));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 4, sizeof(cl_int), (void *)&ry_s));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 5, sizeof(cl_int), (void *)&ry_e));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 6, sizeof(cl_int), (void *)&rz_s));
  clSafeCall(
      clSetKernelArg(copy_frombuf_kernel[0], 7, sizeof(cl_int), (void *)&rz_e));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 8, sizeof(cl_int),
                            (void *)&x_step));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 9, sizeof(cl_int),
                            (void *)&y_step));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 10, sizeof(cl_int),
                            (void *)&z_step));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 11, sizeof(cl_int),
                            (void *)&dest->size[0]));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 12, sizeof(cl_int),
                            (void *)&dest->size[1]));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 13, sizeof(cl_int),
                            (void *)&dest->size[2]));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 14, sizeof(cl_int),
                            (void *)&buf_strides_x));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 15, sizeof(cl_int),
                            (void *)&buf_strides_y));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 16, sizeof(cl_int),
                            (void *)&buf_strides_z));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 17, sizeof(cl_int),
                            (void *)&dest->type_size));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 18, sizeof(cl_int),
                            (void *)&dest->dim));
  clSafeCall(clSetKernelArg(copy_frombuf_kernel[0], 19, sizeof(cl_int),
                            (void *)&OPS_soa));
  clSafeCall(clEnqueueNDRangeKernel(
      OPS_opencl_core.command_queue, *copy_frombuf_kernel, 3, NULL,
      globalWorkSize, localWorkSize, 0, NULL, NULL));

  dest->dirty_hd = 2;
}
