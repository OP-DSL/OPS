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
  * @brief OPS opencl backend implementation
  * @author Gihan Mudalige
  * @details Implements the OPS API calls for the opencl backend
  */

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <ops_lib_core.h>
#include <ops_opencl_rt_support.h>
#include <ops_exceptions.h>

void ops_init(const int argc, const char **argv, const int diags) {
  ops_init_core(argc, argv, diags);

  if ((OPS_instance::getOPSInstance()->OPS_block_size_x * OPS_instance::getOPSInstance()->OPS_block_size_y * OPS_instance::getOPSInstance()->OPS_block_size_z) > 1024) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_*");
  }
  for (int n = 1; n < argc; n++) {
    if (strncmp(argv[n], "OPS_CL_DEVICE=", 14) == 0) {
      OPS_instance::getOPSInstance()->OPS_cl_device = atoi(argv[n] + 14);
      printf("\n OPS_cl_device = %d \n", OPS_instance::getOPSInstance()->OPS_cl_device);
    }
  }

  OPS_instance::getOPSInstance()->opencl_instance = new OPS_instance_opencl();
  OPS_instance::getOPSInstance()->opencl_instance->copy_tobuf_kernel = NULL;
  OPS_instance::getOPSInstance()->opencl_instance->copy_frombuf_kernel = NULL;
  OPS_instance::getOPSInstance()->opencl_instance->isbuilt_copy_tobuf_kernel = false;
  OPS_instance::getOPSInstance()->opencl_instance->isbuilt_copy_frombuf_kernel = false;

  openclDeviceInit(argc, argv);
}

void ops_exit() {
  ops_opencl_exit(); // frees dat_d memory
  delete OPS_instance::getOPSInstance()->opencl_instance;
  ops_exit_core();   // frees lib core variables
}

ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size, int *base, int* d_m,
                           int* d_p, int* stride, char* data,
                           int type_size, char const * type, char const * name )
{

  /** ----             allocate an empty dat             ---- **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p, stride,
    data, type_size, type, name );
  int bytes = size*type_size;
  for (int i=0; i<block->dims; i++) bytes = bytes*dat->size[i];

  if(data != NULL) {
     //printf("Data read in from HDF5 file or is allocated by the user\n");
     dat->user_managed = 1; // will be reset to 0 if called from ops_decl_dat_hdf5()
     dat->is_hdf5 = 0;
     dat->hdf5_file = "none"; // will be set to an hdf5 file if called from ops_decl_dat_hdf5()
  }
  else {
    //Allocate memory immediately
    dat->data = (char *)ops_calloc(bytes, 1); // initialize data bits to 0
    dat->user_managed = 0;
    dat->mem = bytes;
  }

  // Compute offset in bytes to the base index
  dat->base_offset = 0;
  long cumsize = 1;
  for (int i = 0; i < block->dims; i++) {
    dat->base_offset +=
        (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size)
        * cumsize * (-dat->base[i] - dat->d_m[i]);
    cumsize *= dat->size[i];
  }

  ops_cpHostToDevice((void **)&(dat->data_d), (void **)&(dat->data), bytes);

  return dat;
}

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(from, to, iter_size, from_base, to_base, from_dir,
                            to_dir);
}

ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc) {
  return ops_arg_dat_core(dat, stencil, acc);
}

ops_arg ops_arg_dat_opt(ops_dat dat, int dim, ops_stencil stencil,
                        char const *type, ops_access acc, int flag) {
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->opt = flag;
  return temp;
}

ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc) {
  return ops_arg_gbl_core(data, dim, size, acc);
}

void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr) {
  ops_execute();
  ops_checkpointing_reduction(handle);
  memcpy(ptr, handle->data, handle->size);
  handle->initialized = 0;
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name) {
  // need to get data from GPU
  ops_opencl_get_data(dat);
  ops_print_dat_to_txtfile_core(dat, file_name);
}

// routine to fetch data from device
void ops_get_data(ops_dat dat) { ops_opencl_get_data(dat); }
void ops_put_data(ops_dat dat) { ops_opencl_put_data(dat); }

void ops_partition(const char *routine) { (void)routine; }

void ops_halo_transfer(ops_halo_group group) {
  // printf("In OpenCL block halo transfer\n");
  cl_int ret = 0;

  for (int h = 0; h < group->nhalos; h++) {
    ops_halo halo = group->halos[h];
    int size = halo->from->elem_size * halo->iter_size[0];
    for (int i = 1; i < halo->from->block->dims; i++)
      size *= halo->iter_size[i];
    if (size > OPS_instance::getOPSInstance()->ops_halo_buffer_size) {
      if (OPS_instance::getOPSInstance()->ops_halo_buffer_d != NULL)
        clSafeCall(clReleaseMemObject((cl_mem)OPS_instance::getOPSInstance()->ops_halo_buffer_d));
      OPS_instance::getOPSInstance()->ops_halo_buffer_d = (char*)clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context,
                                         CL_MEM_READ_WRITE, size, NULL, &ret);
      clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
      clSafeCall(ret);
      OPS_instance::getOPSInstance()->ops_halo_buffer_size = size;
    }

    // copy to linear buffer from source
    int ranges[OPS_MAX_DIM * 2];
    int step[OPS_MAX_DIM];
    int buf_strides[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      if (halo->from_dir[i] > 0) {
        ranges[2 * i] =
            halo->from_base[i] - halo->from->d_m[i] - halo->from->base[i];
        ranges[2 * i + 1] =
            ranges[2 * i] + halo->iter_size[abs(halo->from_dir[i]) - 1];
        step[i] = 1;
      } else {
        ranges[2 * i + 1] =
            halo->from_base[i] - 1 - halo->from->d_m[i] - halo->from->base[i];
        ranges[2 * i] =
            ranges[2 * i + 1] + halo->iter_size[abs(halo->from_dir[i]) - 1];
        step[i] = -1;
      }
      buf_strides[i] = 1;
      for (int j = 0; j != abs(halo->from_dir[i]) - 1; j++)
        buf_strides[i] *= halo->iter_size[j];
    }

    /*for (int k = ranges[4]; (step[2]==1 ? k < ranges[5] : k > ranges[5]); k +=
    step[2]) {
      for (int j = ranges[2]; (step[1]==1 ? j < ranges[3] : j > ranges[3]); j +=
    step[1]) {
        for (int i = ranges[0]; (step[0]==1 ? i < ranges[1] : i > ranges[1]); i
    += step[0]) {
          ops_cuda_halo_copy(OPS_instance::getOPSInstance()->ops_halo_buffer_d +
    ((k-ranges[4])*step[2]*buf_strides[2]+ (j-ranges[2])*step[1]*buf_strides[1]
    + (i-ranges[0])*step[0]*buf_strides[0])*halo->from->elem_size,
                 halo->from->data_d +
    (k*halo->from->size[0]*halo->from->size[1]+j*halo->from->size[0]+i)*halo->from->elem_size,
    halo->from->elem_size);
        }
      }
    }*/

    ops_halo_copy_tobuf((char *)OPS_instance::getOPSInstance()->ops_halo_buffer_d, 0, halo->from, ranges[0],
                        ranges[1], ranges[2], ranges[3], ranges[4], ranges[5],
                        step[0], step[1], step[2], buf_strides[0],
                        buf_strides[1], buf_strides[2]);

    // cutilSafeCall ( cudaDeviceSynchronize ( ) );
    clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));

    // copy from linear buffer to target
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      if (halo->to_dir[i] > 0) {
        ranges[2 * i] = halo->to_base[i] - halo->to->d_m[i] - halo->to->base[i];
        ranges[2 * i + 1] =
            ranges[2 * i] + halo->iter_size[abs(halo->to_dir[i]) - 1];
        step[i] = 1;
      } else {
        ranges[2 * i + 1] =
            halo->to_base[i] - 1 - halo->to->d_m[i] - halo->to->base[i];
        ranges[2 * i] =
            ranges[2 * i + 1] + halo->iter_size[abs(halo->to_dir[i]) - 1];
        step[i] = -1;
      }
      buf_strides[i] = 1;
      for (int j = 0; j != abs(halo->to_dir[i]) - 1; j++)
        buf_strides[i] *= halo->iter_size[j];
    }

    /*for (int k = ranges[4]; (step[2]==1 ? k < ranges[5] : k > ranges[5]); k +=
    step[2]) {
      for (int j = ranges[2]; (step[1]==1 ? j < ranges[3] : j > ranges[3]); j +=
    step[1]) {
        for (int i = ranges[0]; (step[0]==1 ? i < ranges[1] : i > ranges[1]); i
    += step[0]) {
          ops_cuda_halo_copy(halo->to->data_d +
    (k*halo->to->size[0]*halo->to->size[1]+j*halo->to->size[0]+i)*halo->to->elem_size,
               OPS_instance::getOPSInstance()->ops_halo_buffer_d + ((k-ranges[4])*step[2]*buf_strides[2]+
    (j-ranges[2])*step[1]*buf_strides[1] +
    (i-ranges[0])*step[0]*buf_strides[0])*halo->to->elem_size,
    halo->to->elem_size);
        }
      }
    }*/

    ops_halo_copy_frombuf(halo->to, (char *)OPS_instance::getOPSInstance()->ops_halo_buffer_d, 0, ranges[0],
                          ranges[1], ranges[2], ranges[3], ranges[4], ranges[5],
                          step[0], step[1], step[2], buf_strides[0],
                          buf_strides[1], buf_strides[2]);

    // cutilSafeCall ( cudaDeviceSynchronize ( ) );
    clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
    halo->to->dirty_hd = 2;
  }
}

void ops_timers(double *cpu, double *et) {
  clSafeCall(clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue));
  ops_timers_core(cpu, et);
}
