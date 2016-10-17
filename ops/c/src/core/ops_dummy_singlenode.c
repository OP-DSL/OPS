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

#ifndef __XDIMS__ // perhaps put this into a separate headder file
#define __XDIMS__
int xdim0, xdim1, xdim2, xdim3, xdim4, xdim5, xdim6, xdim7, xdim8, xdim9,
    xdim10, xdim11, xdim12, xdim13, xdim14, xdim15, xdim16, xdim17, xdim18,
    xdim19, xdim20, xdim21, xdim22, xdim23, xdim24, xdim25, xdim26, xdim27,
    xdim28, xdim29, xdim30, xdim31, xdim32, xdim33, xdim34, xdim35, xdim36,
    xdim37, xdim38, xdim39, xdim40, xdim41, xdim42, xdim43, xdim44, xdim45,
    xdim46, xdim47, xdim48, xdim49, xdim50, xdim51, xdim52, xdim53, xdim54,
    xdim55, xdim56, xdim57, xdim58, xdim59, xdim60, xdim61, xdim62, xdim63,
    xdim64, xdim65, xdim66, xdim67, xdim68, xdim69, xdim70, xdim71, xdim72,
    xdim73, xdim74, xdim75, xdim76, xdim77, xdim78, xdim79, xdim80, xdim81,
    xdim82, xdim83, xdim84, xdim85, xdim86, xdim87, xdim88, xdim89, xdim90,
    xdim91, xdim92, xdim93, xdim94, xdim95, xdim96, xdim97, xdim98, xdim99;
#endif /* __XDIMS__ */

#ifndef __YDIMS__
#define __YDIMS__
int ydim0, ydim1, ydim2, ydim3, ydim4, ydim5, ydim6, ydim7, ydim8, ydim9,
    ydim10, ydim11, ydim12, ydim13, ydim14, ydim15, ydim16, ydim17, ydim18,
    ydim19, ydim20, ydim21, ydim22, ydim23, ydim24, ydim25, ydim26, ydim27,
    ydim28, ydim29, ydim30, ydim31, ydim32, ydim33, ydim34, ydim35, ydim36,
    ydim37, ydim38, ydim39, ydim40, ydim41, ydim42, ydim43, ydim44, ydim45,
    ydim46, ydim47, ydim48, ydim49, ydim50, ydim51, ydim52, ydim53, ydim54,
    ydim55, ydim56, ydim57, ydim58, ydim59, ydim60, ydim61, ydim62, ydim63,
    ydim64, ydim65, ydim66, ydim67, ydim68, ydim69, ydim70, ydim71, ydim72,
    ydim73, ydim74, ydim75, ydim76, ydim77, ydim78, ydim79, ydim80, ydim81,
    ydim82, ydim83, ydim84, ydim85, ydim86, ydim87, ydim88, ydim89, ydim90,
    ydim91, ydim92, ydim93, ydim94, ydim95, ydim96, ydim97, ydim98, ydim99;
#endif /* __YDIMS__ */

#ifndef __MULTIDIMS__
#define __MULTIDIMS__
int multi_d0, multi_d1, multi_d2, multi_d3, multi_d4, multi_d5, multi_d6,
    multi_d7, multi_d8, multi_d9, multi_d10, multi_d11, multi_d12, multi_d13,
    multi_d14, multi_d15, multi_d16, multi_d17, multi_d18, multi_d19, multi_d20,
    multi_d21, multi_d22, multi_d23, multi_d24, multi_d25, multi_d26, multi_d27,
    multi_d28, multi_d29, multi_d30, multi_d31, multi_d32, multi_d33, multi_d34,
    multi_d35, multi_d36, multi_d37, multi_d38, multi_d39, multi_d40, multi_d41,
    multi_d42, multi_d43, multi_d44, multi_d45, multi_d46, multi_d47, multi_d48,
    multi_d49, multi_d50, multi_d51, multi_d52, multi_d53, multi_d54, multi_d55,
    multi_d56, multi_d57, multi_d58, multi_d59, multi_d60, multi_d61, multi_d62,
    multi_d63, multi_d64, multi_d65, multi_d66, multi_d67, multi_d68, multi_d69,
    multi_d70, multi_d71, multi_d72, multi_d73, multi_d74, multi_d75, multi_d76,
    multi_d77, multi_d78, multi_d79, multi_d80, multi_d81, multi_d82, multi_d83,
    multi_d84, multi_d85, multi_d86, multi_d87, multi_d88, multi_d89, multi_d90,
    multi_d91, multi_d92, multi_d93, multi_d94, multi_d95, multi_d96, multi_d97,
    multi_d98, multi_d99;
#endif /*__MULTIDIMS__*/

void ops_set_dirtybit_host(ops_arg *args, int nargs) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].argtype == OPS_ARG_DAT) &&
        (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE ||
         args[n].acc == OPS_RW)) {
      args[n].dat->dirty_hd = 1;
    }
  }
}

ops_arg ops_arg_reduce(ops_reduction handle, int dim, const char *type,
                       ops_access acc) {
  return ops_arg_reduce_core(handle, dim, type, acc);
}

ops_reduction ops_decl_reduction_handle(int size, const char *type,
                                        const char *name) {
  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 ||
      strcmp(type, "double precision") == 0)
    type = "double";
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0)
    type = "float";
  else if (strcmp(type, "int") == 0 || strcmp(type, "integer") == 0 ||
           strcmp(type, "integer(4)") == 0 || strcmp(type, "int(4)") == 0)
    type = "int";

  return ops_decl_reduction_handle_core(size, type, name);
}

void ops_execute_reduction(ops_reduction handle) { (void)handle; }

int ops_is_root() { return 1; }

void ops_set_halo_dirtybit(ops_arg *arg) { (void)arg; }

void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range) {
  (void)arg;
  (void)iter_range;
}

void ops_halo_exchanges_datlist(ops_dat *dats, int ndats, int *depths) {
  (void)dats;
  (void)depths;
}

void ops_halo_exchanges(ops_arg *args, int nargs, int *range) {
  (void)args;
  (void)range;
}

void ops_mpi_reduce_float(ops_arg *args, float *data) {
  (void)args;
  (void)data;
}

void ops_mpi_reduce_double(ops_arg *args, double *data) {
  (void)args;
  (void)data;
}

void ops_mpi_reduce_int(ops_arg *args, int *data) {
  (void)args;
  (void)data;
}

void ops_compute_moment(double t, double *first, double *second) {
  *first = t;
  *second = t * t;
}

void ops_printf(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void ops_fprintf(FILE *stream, const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stream, format, argptr);
  va_end(argptr);
}

bool ops_checkpointing_filename(const char *file_name, char *filename_out,
                                char *filename_out2) {
  strcpy(filename_out, file_name);
  filename_out2 = "";
  return false;
}

void ops_checkpointing_duplicate_data(ops_dat dat, int my_type, int my_nelems,
                                      char *my_data, int *my_range,
                                      int *rm_type, int *rm_elems,
                                      char **rm_data, int **rm_range) {
  *rm_type = 0;
  *rm_elems = 0;
}

void ops_get_dat_full_range(ops_dat dat, int **full_range) {
  *full_range = dat->size;
}

void ops_checkpointing_calc_range(ops_dat dat, const int *range,
                                  int *discarded_range) {
  for (int d = 0; d < dat->block->dims; d++) {
    discarded_range[2 * d] = range[2 * d] - dat->base[d] - dat->d_m[d];
    discarded_range[2 * d + 1] =
        discarded_range[2 * d] + range[2 * d + 1] - range[2 * d];
  }
}

bool ops_get_abs_owned_range(ops_block block, int *range, int *start, int *end, int *disp) {
  for (int n = 0; n < block->dims; n++) {
    start[n] = range[2 * n];
    end[n] = range[2 * n + 1];
    disp[n] = 0;
  }
  return true;
}

int ops_get_proc() {
  return 0;
}

/************* Functions only use in the Fortran Backend ************/

int *getDatSizeFromOpsArg(ops_arg *arg) { return arg->dat->size; }

int getDatDimFromOpsArg(ops_arg *arg) { return arg->dat->dim; }

// need differet routines for 1D, 2D 3D etc.
int getDatBaseFromOpsArg1D(ops_arg *arg, int *start, int dim) {

  /*convert to C indexing*/
  start[0] -= 1;

  int dat = arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // printf("start[0] = %d, base = %d, dim = %d, d_m[0] = %d dat = %d\n",
  //      start[0],arg->dat->base[0],dim, arg->dat->d_m[0], dat);

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d];
  int base = dat * 1 *
             (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  return base / (dat / dim) + 1;
}

int getDatBaseFromOpsArg2D(ops_arg *arg, int *start, int dim) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;

  int dat = arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d];
  int base = dat * 1 *
             (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  base = base +
         dat * arg->dat->size[0] *
             (start[1] * arg->stencil->stride[1] - arg->dat->base[1] - d_m[1]);

  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  return base / (dat / dim) + 1;
}

int getDatBaseFromOpsArg3D(ops_arg *arg, int *start, int dim) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;
  start[2] -= 1;

  int dat = arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d];
  int base = dat * 1 *
             (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  base = base +
         dat * arg->dat->size[0] *
             (start[1] * arg->stencil->stride[1] - arg->dat->base[1] - d_m[1]);
  base = base +
         dat * arg->dat->size[0] * arg->dat->size[1] *
             (start[2] * arg->stencil->stride[2] - arg->dat->base[2] - d_m[2]);

  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  start[2] += 1;
  return base / (dat / dim) + 1;
}

char *getReductionPtrFromOpsArg(ops_arg *arg, ops_block block) {
  return (char *)((ops_reduction)arg->data)->data;
}

char *getGblPtrFromOpsArg(ops_arg *arg) { return (char *)(arg->data); }

int getRange(ops_block block, int *start, int *end, int *range) { return 1; }

void getIdx(ops_block block, int *start, int *idx) {
  int block_dim = block->dims;
  for (int n = 0; n < block_dim; n++) {
    idx[n] = start[n];
  }
}