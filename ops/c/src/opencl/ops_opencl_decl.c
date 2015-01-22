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

/** @brief ops opencl backend implementation
  * @author Gihan Mudalige
  * @details Implements the OPS API calls for the opencl backend
  */

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#include <ops_lib_cpp.h>
#include <ops_opencl_rt_support.h>

char *ops_halo_buffer = NULL;
cl_mem ops_halo_buffer_d = NULL;
int ops_halo_buffer_size = 0;

extern ops_opencl_core OPS_opencl_core;

#ifndef __XDIMS__ //perhaps put this into a separate headder file
#define __XDIMS__
int xdim0;
int xdim1;
int xdim2;
int xdim3;
int xdim4;
int xdim5;
int xdim6;
int xdim7;
int xdim8;
int xdim9;
int xdim10;
int xdim11;
int xdim12;
int xdim13;
int xdim14;
int xdim15;
int xdim16;
int xdim17;
int xdim18;
int xdim19;
int ydim0;
int ydim1;
int ydim2;
int ydim3;
int ydim4;
int ydim5;
int ydim6;
int ydim7;
int ydim8;
int ydim9;
int ydim10;
int ydim11;
int ydim12;
int ydim13;
int ydim14;
int ydim15;
int ydim16;
int ydim17;
int ydim18;
int ydim19;
#endif /* __XDIMS__ */

void ops_init ( int argc, char ** argv, int diags )
{
  ops_init_core ( argc, argv, diags );

  if ((OPS_block_size_x*OPS_block_size_y) > 1024) {
    printf ( " OPS_block_size_x*OPS_block_size_y should be less than 1024 -- error OPS_block_size_*\n" );
    exit ( -1 );
  }
  for ( int n = 1; n < argc; n++ )
  {
    if ( strncmp ( argv[n], "OPS_CL_DEVICE=", 14 ) == 0 )
    {
      OPS_cl_device = atoi ( argv[n] + 14 );
      printf ( "\n OPS_cl_device = %d \n", OPS_cl_device );
    }
  }

  openclDeviceInit ( argc, argv );
}

void ops_exit()
{
  ops_opencl_exit(); // frees dat_d memory
  ops_exit_core(); // frees lib core variables
}

ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size, int *base, int* d_m,
                           int* d_p, char* data,
                           int type_size, char const * type, char const * name )
{

  /** ----             allocate an empty dat             ---- **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p, data, type_size, type, name );

  int bytes = size*type_size;
  for (int i=0; i<block->dims; i++) bytes = bytes*dat->size[i];
  dat->data = (char*) calloc(bytes, 1); //initialize data bits to 0
  dat->user_managed = 0;

  ops_cpHostToDevice ( ( void ** ) &( dat->data_d ),
    ( void ** ) &( dat->data ), bytes );

  return dat;
}

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int* from_base, int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(from, to, iter_size, from_base, to_base, from_dir, to_dir);
}

ops_arg ops_arg_dat( ops_dat dat, int dim, ops_stencil stencil, char const * type, ops_access acc )
{
  return ops_arg_dat_core( dat, stencil, acc );
}

ops_arg ops_arg_dat_opt( ops_dat dat, int dim, ops_stencil stencil, char const * type, ops_access acc, int flag )
{
  ops_arg temp = ops_arg_dat_core( dat, stencil, acc );
  (&temp)->opt = flag;
  return temp;
}

ops_arg ops_arg_gbl_char( char * data, int dim, int size, ops_access acc )
{
  return ops_arg_gbl_core( data, dim, size, acc );
}

void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr){
  ops_execute();
  ops_checkpointing_reduction(handle);
  memcpy(ptr, handle->data, handle->size);
  handle->initialized = 0;
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name)
{
  //need to get data from GPU
  ops_opencl_get_data(dat);
  ops_print_dat_to_txtfile_core(dat, file_name);
}

void ops_partition(char* routine)
{
  (void)routine;
}

void ops_halo_transfer(ops_halo_group group)
{
  //printf("In OpenCL block halo transfer\n");
  cl_int ret = 0;

  for (int h = 0; h < group->nhalos; h++) {
    ops_halo halo = group->halos[h];
    int size = halo->from->elem_size * halo->iter_size[0];
    for (int i = 1; i < halo->from->block->dims; i++) size *= halo->iter_size[i];
    if (size > ops_halo_buffer_size) {
      if (ops_halo_buffer_d!=NULL) clSafeCall( clReleaseMemObject(ops_halo_buffer_d));
      ops_halo_buffer_d = clCreateBuffer(OPS_opencl_core.context, CL_MEM_READ_WRITE, size, NULL, &ret);
      clSafeCall( clFinish(OPS_opencl_core.command_queue) );
      clSafeCall( ret );
      ops_halo_buffer_size = size;
    }

    //copy to linear buffer from source
    int ranges[OPS_MAX_DIM*2];
    int step[OPS_MAX_DIM];
    int buf_strides[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      if (halo->from_dir[i] > 0) {
        ranges[2*i] = halo->from_base[i] - halo->from->d_m[i] - halo->from->base[i];
        ranges[2*i+1] = ranges[2*i] + halo->iter_size[abs(halo->from_dir[i])-1];
        step[i] = 1;
      } else {
        ranges[2*i+1] = halo->from_base[i] - 1  - halo->from->d_m[i] - halo->from->base[i];
        ranges[2*i] = ranges[2*i+1] + halo->iter_size[abs(halo->from_dir[i])-1];
        step[i] = -1;
      }
      buf_strides[i] = 1;
      for (int j = 0; j != abs(halo->from_dir[i])-1; j++) buf_strides[i] *= halo->iter_size[j];
    }

    /*for (int k = ranges[4]; (step[2]==1 ? k < ranges[5] : k > ranges[5]); k += step[2]) {
      for (int j = ranges[2]; (step[1]==1 ? j < ranges[3] : j > ranges[3]); j += step[1]) {
        for (int i = ranges[0]; (step[0]==1 ? i < ranges[1] : i > ranges[1]); i += step[0]) {
          ops_cuda_halo_copy(ops_halo_buffer_d + ((k-ranges[4])*step[2]*buf_strides[2]+ (j-ranges[2])*step[1]*buf_strides[1] + (i-ranges[0])*step[0]*buf_strides[0])*halo->from->elem_size,
                 halo->from->data_d + (k*halo->from->size[0]*halo->from->size[1]+j*halo->from->size[0]+i)*halo->from->elem_size, halo->from->elem_size);
        }
      }
    }*/

    ops_halo_copy_tobuf((char *)ops_halo_buffer_d, 0, halo->from,
                       ranges[0], ranges[1],
                       ranges[2], ranges[3],
                       ranges[4], ranges[5],
                       step[0], step[1], step[2],
                       buf_strides[0], buf_strides[1], buf_strides[2]);

    //cutilSafeCall ( cudaDeviceSynchronize ( ) );
    clSafeCall( clFinish(OPS_opencl_core.command_queue) );

    //copy from linear buffer to target
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      if (halo->to_dir[i] > 0) {
        ranges[2*i] = halo->to_base[i] - halo->to->d_m[i] - halo->to->base[i];
        ranges[2*i+1] = ranges[2*i] + halo->iter_size[abs(halo->to_dir[i])-1];
        step[i] = 1;
      } else {
        ranges[2*i+1] = halo->to_base[i] - 1 - halo->to->d_m[i] - halo->to->base[i];
        ranges[2*i] = ranges[2*i+1] + halo->iter_size[abs(halo->to_dir[i])-1];
        step[i] = -1;
      }
      buf_strides[i] = 1;
      for (int j = 0; j != abs(halo->to_dir[i])-1; j++) buf_strides[i] *= halo->iter_size[j];
    }

    /*for (int k = ranges[4]; (step[2]==1 ? k < ranges[5] : k > ranges[5]); k += step[2]) {
      for (int j = ranges[2]; (step[1]==1 ? j < ranges[3] : j > ranges[3]); j += step[1]) {
        for (int i = ranges[0]; (step[0]==1 ? i < ranges[1] : i > ranges[1]); i += step[0]) {
          ops_cuda_halo_copy(halo->to->data_d + (k*halo->to->size[0]*halo->to->size[1]+j*halo->to->size[0]+i)*halo->to->elem_size,
               ops_halo_buffer_d + ((k-ranges[4])*step[2]*buf_strides[2]+ (j-ranges[2])*step[1]*buf_strides[1] + (i-ranges[0])*step[0]*buf_strides[0])*halo->to->elem_size, halo->to->elem_size);
        }
      }
    }*/

    ops_halo_copy_frombuf(halo->to, (char *)ops_halo_buffer_d, 0,
                   ranges[0], ranges[1],
                   ranges[2], ranges[3],
                   ranges[4], ranges[5],
                   step[0], step[1], step[2],
                   buf_strides[0], buf_strides[1], buf_strides[2]);

    //cutilSafeCall ( cudaDeviceSynchronize ( ) );
    clSafeCall( clFinish(OPS_opencl_core.command_queue) );
    halo->to->dirty_hd = 2;
  }
}
