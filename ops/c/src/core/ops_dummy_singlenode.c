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

/** @brief dummy function for CPU backend
  * @author Gihan Mudalige
  * @details Implements dummy functions from the MPI backend for the sequential
  * cpu backend (OpenMP and Sequential)
  */


#include "ops_lib_core.h"

void ops_set_dirtybit_host(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++) {
    if((args[n].argtype == OPS_ARG_DAT) &&
       (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE || args[n].acc == OPS_RW) ) {
      args[n].dat->dirty_hd = 1;
    }
  }
}

ops_arg ops_arg_reduce ( ops_reduction handle, int dim, const char *type, ops_access acc) {
  return ops_arg_reduce_core(handle, dim, type, acc);
}

ops_reduction ops_decl_reduction_handle(int size, const char *type, const char *name) {
  return ops_decl_reduction_handle_core(size, type, name);
}

void ops_execute_reduction(ops_reduction handle) {
  (void)handle;
}

int ops_is_root()
{
  return 1;
}

void ops_set_halo_dirtybit(ops_arg *arg)
{
  (void)arg;
}

void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range)
{
  (void)arg;
  (void)iter_range;
}

void ops_exchange_halo(ops_arg* arg, int d)
{
  (void)arg;
}

void ops_exchange_halo2(ops_arg* arg, int* d_pos, int* d_neg /*depth*/)
{
  (void)arg;
  (void)d_pos;
  (void)d_neg;
}

void ops_exchange_halo3(ops_arg* arg, int* d_pos, int* d_neg /*depth*/, int *iter_range)
{
  (void)arg;
  (void)d_pos;
  (void)d_neg;
  (void)iter_range;
}

void ops_halo_exchanges(ops_arg* args, int nargs, int *range) {
  (void)args;
  (void)range;
}

void ops_mpi_reduce_float(ops_arg* args, float* data)
{
  (void)args;
  (void)data;
}

void ops_mpi_reduce_double(ops_arg* args, double* data)
{
  (void)args;
  (void)data;
}

void ops_mpi_reduce_int(ops_arg* args, int* data)
{
  (void)args;
  (void)data;
}

void ops_compute_moment(double t, double *first, double *second) {
  *first = t;
  *second = t*t;
}

void ops_printf(const char* format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void ops_fprintf(FILE *stream, const char *format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stream, format, argptr);
  va_end(argptr);
}

void ops_checkpointing_filename(const char *file_name, char *filename_out) {
  strcpy(filename_out, file_name);
}

void ops_checkpointing_calc_range(ops_dat dat, const int *range, int *discarded_range) {
  for (int d = 0; d < dat->block->dims; d++) {
    discarded_range[2*d] = range[2*d]-dat->base[d]-dat->d_m[d];
    discarded_range[2*d+1] = discarded_range[2*d] + range[2*d+1] - range[2*d];
  }
}

void ops_timers(double * cpu, double * et){
    ops_timers_core(cpu,et);
}


/************* Functions only use in the s Backend ************/

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
  for (int d = 0; d < block_dim; d++) d_m[d] = arg->dat->d_m[d];
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
  //printf("start[0] = %d, start[1] = %d, base(1) = %d, base(2) = %d, dim = %d, dat = %d\n",
     //    start[0],start[1],arg->dat->base[0], arg->dat->base[1], dim, dat);

  //printf("arg->dat->size[0] = %d, arg->dat->size[1] = %d\n",arg->dat->size[0],arg->dat->size[1]);


  //set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++) d_m[d] = arg->dat->d_m[d];
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

char* getReductionPtrFromOpsArg(ops_arg* arg) {
  return (char *)((ops_reduction)arg->data)->data;
}

char* getGblPtrFromOpsArg(ops_arg* arg) {
  return (char *)(arg->data);
}