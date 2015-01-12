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

/** @brief ops mpi+opencl declaration
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the OPS API calls for the mpi+cuda backend
  */

#include <mpi.h>
#include <ops_mpi_core.h>

//#include <math_constants.h>
#include <ops_opencl_rt_support.h>

extern char *halo_buffer_d;
extern char *ops_buffer_send_1;
extern char *ops_buffer_recv_1;
extern char *ops_buffer_send_2;
extern char *ops_buffer_recv_2;


void ops_init_opencl ( int argc, char ** argv, int diags )
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

void
ops_init ( int argc, char ** argv, int diags )
{
  int flag = 0;
  MPI_Initialized(&flag);
  if(!flag) {
    MPI_Init(&argc, &argv);
  }

  MPI_Comm_dup(MPI_COMM_WORLD, &OPS_MPI_GLOBAL);
  MPI_Comm_rank(OPS_MPI_GLOBAL, &ops_my_global_rank);
  MPI_Comm_size(OPS_MPI_GLOBAL, &ops_comm_global_size);

  ops_init_opencl ( argc, argv, diags );
}

void ops_exit()
{
  ops_mpi_exit();
  if (halo_buffer_d!=NULL) clReleaseMemObject((cl_mem)(halo_buffer_d));
  free(ops_buffer_send_1);
  free(ops_buffer_recv_1);
  free(ops_buffer_send_2);
  free(ops_buffer_recv_2);

  int flag = 0;
  MPI_Finalized(&flag);
  if(!flag) MPI_Finalize();
  ops_opencl_exit();
  ops_exit_core();
}

ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size,
                           int *base, int* d_m, int* d_p, char* data,
                           int type_size, char const * type, char const * name )
{

/** ---- allocate an empty dat based on the local array sizes computed
         above on each MPI process                                      ---- **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p, data, type_size, type, name );

  dat->user_managed = 0;

  //note that currently we assume replicated dats are read only or initialized just once
  //what to do if not ?? How will the halos be handled

  //TODO: proper allocation and TAILQ
  //create list to hold sub-grid decomposition geometries for each mpi process
  OPS_sub_dat_list = (sub_dat_list *)xrealloc(OPS_sub_dat_list, OPS_dat_index*sizeof(sub_dat_list));

  //store away product array prod[] and MPI_Types for this ops_dat
  sub_dat_list sd= (sub_dat_list)xmalloc(sizeof(sub_dat));
  sd->dat = dat;
  sd->dirtybit = 1;
  sd->dirty_dir_send =( int *)xmalloc(sizeof(int)*2*block->dims*MAX_DEPTH);
  for(int i = 0; i<2*block->dims*MAX_DEPTH;i++) sd->dirty_dir_send[i] = 1;
  sd->dirty_dir_recv =( int *)xmalloc(sizeof(int)*2*block->dims*MAX_DEPTH);
  for(int i = 0; i<2*block->dims*MAX_DEPTH;i++) sd->dirty_dir_recv[i] = 1;
  for(int i = 0; i<OPS_MAX_DIM; i++) {sd->d_ip[i] = 0; sd->d_im[i] = 0;}

  OPS_sub_dat_list[dat->index] = sd;

  return dat;
}

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int* from_base, int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(from, to, iter_size, from_base, to_base, from_dir, to_dir);
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name)
{
  if(OPS_sub_block_list[dat->block->index]->owned == 1) {
    ops_opencl_get_data(dat);
    ops_print_dat_to_txtfile_core(dat, file_name);
  }
}
