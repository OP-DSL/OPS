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

/** @brief ops mpi declaration
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the OPS API calls for the mpi backend
  */

#include <mpi.h>
#include <ops_mpi_core.h>

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

  ops_init_core ( argc, argv, diags );
}

void ops_exit()
{
  ops_mpi_exit();
  //ops_rt_exit();
  ops_exit_core();

  int flag = 0;
  MPI_Finalized(&flag);
  if(!flag) MPI_Finalize();
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
  if(OPS_sub_block_list[dat->block->index]->owned == 1)
    ops_print_dat_to_txtfile_core(dat, file_name);
}

void ops_decl_const_char( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

void ops_H_D_exchanges_host(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr){
  ops_execute();
  ops_checkpointing_reduction(handle);
  memcpy(ptr, handle->data, handle->size);
  handle->initialized = 0;
}


/************* Functions only use in the Fortran Backend ************/

int getRange(ops_block block, int* start, int* end, int* range){

  int owned = -1;
  int block_dim = block->dims;
  sub_block_list sb = OPS_sub_block_list[block->index];

  if (sb->owned) {

    owned = 1;

    /*convert to C indexing*/
    for ( int n=0; n<block_dim; n++ ){
      range[2*n] -= 1;
      //range[2*n+1] -= 1; -- c indexing end is exclusive so do not reduce
    }

    for ( int n=0; n<block_dim; n++ ){
      start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
      if (start[n] >= range[2*n]) {
        start[n] = 0;
      }
      else {
        start[n] = range[2*n] - start[n];
      }
      if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
      if (end[n] >= range[2*n+1]) {
        end[n] = range[2*n+1] - sb->decomp_disp[n];
      }
      else {
        end[n] = sb->decomp_size[n];
      }
      if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
        end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
    }

    /*revert to Fortran indexing*/
    for ( int n=0; n<block_dim; n++ ){
      range[2*n] += 1;
      start[n] += 1;
      //end[n] += 1; -- no need as fortran indexing is inclusive
    }
  }
  //for ( int n=0; n<block_dim; n++ ){
  //  printf("start[%d] = %d, end[%d] = %d\n", n,start[n],n,end[n]);
  //}
  return owned;
}

void getIdx(ops_block block, int* start, int* idx) {
  int block_dim = block->dims;
  sub_block_list sb = OPS_sub_block_list[block->index];
  for ( int n=0; n<block_dim; n++ ) {
    idx[n] = sb->decomp_disp[n]+start[n];
  }
  //printf("start[0] = %d, idx[0] = %d\n",start[0], idx[0]);
}


int* getDatSizeFromOpsArg (ops_arg * arg){
  return arg->dat->size;
}

int getDatDimFromOpsArg (ops_arg * arg){
  return arg->dat->dim;
}

//need differet routines for 1D, 2D 3D etc.
int getDatBaseFromOpsArg1D (ops_arg * arg, int* start, int dim){

  /*convert to C indexing*/
  start[0] -= 1;

  int dat = arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  //printf("start[0] = %d, base = %d, dim = %d, d_m[0] = %d dat = %d\n",
   //      start[0],arg->dat->base[0],dim, arg->dat->d_m[0], dat);

  //set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++) d_m[d] = arg->dat->d_m[d] + OPS_sub_dat_list[arg->dat->index]->d_im[d];
  int base = dat * 1 *
   (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  return base/(dat/dim)+1;
}

int getDatBaseFromOpsArg2D (ops_arg * arg, int* start, int dim){
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;

  int dat = arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  //set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++) d_m[d] = arg->dat->d_m[d] + OPS_sub_dat_list[arg->dat->index]->d_im[d];
  int base = dat * 1 *
   (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]) ;
  base = base + dat *
    arg->dat->size[0] *
    (start[1] * arg->stencil->stride[1] - arg->dat->base[1] - d_m[1]);

  //printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  return base/(dat/dim)+1;
}

int getDatBaseFromOpsArg3D (ops_arg * arg, int* start, int dim){
  return 1;
}

char* getReductionPtrFromOpsArg(ops_arg* arg, ops_block block) {
  //return (char *)((ops_reduction)arg->data)->data;
  //printf("block->index %d ((ops_reduction)arg->data)->size = %d\n",block->index, ((ops_reduction)arg->data)->size);
  return (char *)((ops_reduction)arg->data)->data + ((ops_reduction)arg->data)->size * block->index;
}

char* getGblPtrFromOpsArg(ops_arg* arg) {
  return (char *)(arg->data);
}