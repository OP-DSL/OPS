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

/** @brief ops mpi core routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the core mpi decl routines for the OPS mpi backend
  */

#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

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

#ifndef __ZDIMS__
#define __ZDIMS__
int zdim0, zdim1, zdim2, zdim3, zdim4, zdim5, zdim6, zdim7, zdim8, zdim9,
    zdim10, zdim11, zdim12, zdim13, zdim14, zdim15, zdim16, zdim17, zdim18,
    zdim19, zdim20, zdim21, zdim22, zdim23, zdim24, zdim25, zdim26, zdim27,
    zdim28, zdim29, zdim30, zdim31, zdim32, zdim33, zdim34, zdim35, zdim36,
    zdim37, zdim38, zdim39, zdim40, zdim41, zdim42, zdim43, zdim44, zdim45,
    zdim46, zdim47, zdim48, zdim49, zdim50, zdim51, zdim52, zdim53, zdim54,
    zdim55, zdim56, zdim57, zdim58, zdim59, zdim60, zdim61, zdim62, zdim63,
    zdim64, zdim65, zdim66, zdim67, zdim68, zdim69, zdim70, zdim71, zdim72,
    zdim73, zdim74, zdim75, zdim76, zdim77, zdim78, zdim79, zdim80, zdim81,
    zdim82, zdim83, zdim84, zdim85, zdim86, zdim87, zdim88, zdim89, zdim90,
    zdim91, zdim92, zdim93, zdim94, zdim95, zdim96, zdim97, zdim98, zdim99;
#endif /* __ZDIMS__ */

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

void ops_timers(double *cpu, double *et) {
  MPI_Barrier(MPI_COMM_WORLD);
  ops_timers_core(cpu, et);
}

void ops_printf(const char *format, ...) {
  if (ops_my_global_rank == MPI_ROOT) {
    va_list argptr;
    va_start(argptr, format);
    vprintf(format, argptr);
    va_end(argptr);
  }
}

void ops_fprintf(FILE *stream, const char *format, ...) {
  if (ops_my_global_rank == MPI_ROOT) {
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stream, format, argptr);
    va_end(argptr);
  }
}

void ops_compute_moment(double t, double *first, double *second) {
  double times[2] = {0.0};
  double times_reduced[2] = {0.0};
  int comm_size;
  times[0] = t;
  times[1] = t * t;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Reduce(times, times_reduced, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  *first = times_reduced[0] / (double)comm_size;
  *second = times_reduced[1] / (double)comm_size;
}

int ops_is_root() {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  return (my_rank == MPI_ROOT);
}

int ops_get_proc() {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  return my_rank;
}

int ops_num_procs() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

void ops_set_dirtybit_host(ops_arg *args, int nargs) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].argtype == OPS_ARG_DAT) &&
        (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE ||
         args[n].acc == OPS_RW)) {
      args[n].dat->dirty_hd = 1;
    }
  }
}

ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc) {
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  return temp;
}

ops_arg ops_arg_dat_opt(ops_dat dat, int dim, ops_stencil stencil,
                        char const *type, ops_access acc, int flag) {
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  (&temp)->opt = flag;
  return temp;
}

ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc) {
  return ops_arg_gbl_core(data, dim, size, acc);
}

ops_arg ops_arg_reduce(ops_reduction handle, int dim, const char *type,
                       ops_access acc) {
  int was_initialized = handle->initialized;
  ops_arg temp = ops_arg_reduce_core(handle, dim, type, acc);
  if (!was_initialized) {
    for (int i = 1; i < OPS_block_index; i++) {
      memcpy(handle->data + i * handle->size, handle->data, handle->size);
    }
  }
  return temp;
}

ops_reduction ops_decl_reduction_handle(int size, const char *type,
                                        const char *name) {

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 ||
      strcmp(type, "double precision") == 0)
    type = "double\0";
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0)
    type = "float";
  else if (strcmp(type, "int") == 0 || strcmp(type, "integer") == 0 ||
           strcmp(type, "integer(4)") == 0 || strcmp(type, "int(4)") == 0)
    type = "int";

  ops_reduction red = ops_decl_reduction_handle_core(size, type, name);
  if (OPS_block_index < 1) {
    printf("Error: ops_decl_reduction_handle() should only be called after \
      declaring at least one ops_block\n -- Aborting\n");
    MPI_Abort(OPS_MPI_GLOBAL, 2);
  }
  red->data =
      (char *)realloc(red->data, red->size * (OPS_block_index) * sizeof(char));
  return red;
}

bool ops_checkpointing_filename(const char *file_name, char *filename_out,
                                char *filename_out2) {
  sprintf(filename_out, "%s.%d", file_name, ops_my_global_rank);
  sprintf(filename_out2, "%s.%d.dup", file_name,
          (ops_my_global_rank + OPS_ranks_per_node) % ops_comm_global_size);
  return (OPS_enable_checkpointing > 1);
}

void ops_checkpointing_calc_range(ops_dat dat, const int *range,
                                  int *discarded_range) {
  for (int d = 0; d < OPS_MAX_DIM; d++) {
    discarded_range[2 * d] = 0;
    discarded_range[2 * d + 1] = 0;
  }
  sub_block *sb = OPS_sub_block_list[dat->block->index];
  sub_dat *sd = OPS_sub_dat_list[dat->index];
  if (!sb->owned)
    return;

  for (int d = 0; d < dat->block->dims; d++) {
    if (sd->decomp_size[d] - sd->d_im[d] + sd->d_ip[d] != dat->size[d])
      printf("Problem 1\n");

    discarded_range[2 * d] =
        MAX(0, range[2 * d] - (sd->decomp_disp[d] - sd->d_im[d]));
    discarded_range[2 * d + 1] =
        MAX(0, MIN(dat->size[d],
                   range[2 * d + 1] - (sd->decomp_disp[d] - sd->d_im[d])));
    // if (range[2*d+1] >= (sd->decomp_disp[d]+sd->decomp_size[d]))
    //   discarded_range[2*d+1] = dat->size[d]; //Never save intra-block halo
    // else if (range[2*d+1] <= sd->decomp_disp[d])
    //   discarded_range[2*d+1] = 0;
    // else
    //   discarded_range[2*d+1] = (range[2*d+1] - (sd->decomp_disp[d] -
    //   sd->d_im[d]));
  }
}

char *OPS_checkpoiting_dup_buffer = NULL;
int OPS_checkpoiting_dup_buffer_size = 0;
int recv_stats[2 + 2 * OPS_MAX_DIM];

void ops_checkpointing_duplicate_data(ops_dat dat, int my_type, int my_nelems,
                                      char *my_data, int *my_range,
                                      int *rm_type, int *rm_elems,
                                      char **rm_data, int **rm_range) {

  MPI_Status statuses[2];
  MPI_Request requests[2];
  int send_stats[2 + 2 * OPS_MAX_DIM];
  send_stats[0] = my_type;
  send_stats[1] = my_nelems;
  memcpy(&send_stats[2], my_range, 2 * OPS_MAX_DIM * sizeof(int));
  MPI_Isend(send_stats, 2 + 2 * OPS_MAX_DIM, MPI_INT,
            (ops_my_global_rank + OPS_ranks_per_node) % ops_comm_global_size,
            1000 + OPS_dat_index + dat->index, OPS_MPI_GLOBAL, &requests[0]);
  int bytesize = dat->elem_size / dat->dim;
  MPI_Isend(my_data, my_nelems * bytesize, MPI_CHAR,
            (ops_my_global_rank + OPS_ranks_per_node) % ops_comm_global_size,
            1000 + dat->index, OPS_MPI_GLOBAL, &requests[1]);

  MPI_Recv(recv_stats, 2 + 2 * OPS_MAX_DIM, MPI_INT,
           (ops_comm_global_size + ops_my_global_rank - OPS_ranks_per_node) %
               ops_comm_global_size,
           1000 + OPS_dat_index + dat->index, OPS_MPI_GLOBAL, &statuses[0]);
  if (recv_stats[1] * bytesize > OPS_checkpoiting_dup_buffer_size) {
    OPS_checkpoiting_dup_buffer =
        (char *)realloc(OPS_checkpoiting_dup_buffer,
                        recv_stats[1] * bytesize * 2 * sizeof(char));
    OPS_checkpoiting_dup_buffer_size = recv_stats[1] * bytesize * 2;
  }
  *rm_data = OPS_checkpoiting_dup_buffer;
  MPI_Recv(*rm_data, recv_stats[1] * bytesize, MPI_CHAR,
           (ops_comm_global_size + ops_my_global_rank - OPS_ranks_per_node) %
               ops_comm_global_size,
           1000 + dat->index, OPS_MPI_GLOBAL, &statuses[1]);
  *rm_type = recv_stats[0];
  *rm_elems = recv_stats[1];
  *rm_range = &recv_stats[2];
  MPI_Waitall(2, requests, statuses);
}

void ops_get_dat_full_range(ops_dat dat, int **full_range) {
  *full_range = OPS_sub_dat_list[dat->index]->gbl_size;
}

bool ops_get_abs_owned_range(ops_block block, int *range, int *start, int *end, int *disp) {
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) {
    for (int n = 0; n < block->dims; n++) {
      start[n] = 0;
      end[n] = 0;
      disp[n] = 0;
    }
    return false;
  }

  for (int n = 0; n < block->dims; n++) {
    start[n] = MAX(sb->decomp_disp[n], range[2*n]);
    end[n] = MIN(sb->decomp_disp[n] + sb->decomp_size[n], range[2*n+1]);

    if (sb->id_m[n] == MPI_PROC_NULL && range[2 * n] < 0)
      start[n] = range[2 * n];

    if (sb->id_p[n] == MPI_PROC_NULL &&
        (range[2 * n + 1] > sb->decomp_disp[n] + sb->decomp_size[n]))
      end[n] = range[2 * n + 1];

    disp[n] = sb->decomp_disp[n];
  
  }
  return true;
}

/************* Functions only use in the Fortran Backend ************/

int getRange(ops_block block, int *start, int *end, int *range) {

  int owned = -1;
  int block_dim = block->dims;
  /*convert to C indexing*/
  sub_block_list sb = OPS_sub_block_list[block->index];

  if (sb->owned) {
    owned = 1;
    for (int n = 0; n < block_dim; n++) {
      range[2 * n] -= 1;
      // range[2*n+1] -= 1; -- c indexing end is exclusive so do not reduce
    }
    for (int n = 0; n < block_dim; n++) {
      start[n] = sb->decomp_disp[n];
      end[n] = sb->decomp_disp[n] + sb->decomp_size[n];
      if (start[n] >= range[2 * n]) {
        start[n] = 0;
      } else {
        start[n] = range[2 * n] - start[n];
      }
      if (sb->id_m[n] == MPI_PROC_NULL && range[2 * n] < 0)
        start[n] = range[2 * n];
      if (end[n] >= range[2 * n + 1]) {
        end[n] = range[2 * n + 1] - sb->decomp_disp[n];
      } else {
        end[n] = sb->decomp_size[n];
      }
      if (sb->id_p[n] == MPI_PROC_NULL &&
          (range[2 * n + 1] > sb->decomp_disp[n] + sb->decomp_size[n]))
        end[n] += (range[2 * n + 1] - sb->decomp_disp[n] - sb->decomp_size[n]);
    }

    /*revert to Fortran indexing*/
    for (int n = 0; n < block_dim; n++) {
      range[2 * n] += 1;
      start[n] += 1;
      // end[n] += 1; -- no need as fortran indexing is inclusive
    }

    // for ( int n=0; n<block_dim; n++ ){
    //  printf("start[%d] = %d, end[%d] = %d\n", n,start[n],n,end[n]);
    //}
  }
  return owned;
}

void getIdx(ops_block block, int *start, int *idx) {
  int block_dim = block->dims;
  sub_block_list sb = OPS_sub_block_list[block->index];
  for (int n = 0; n < block_dim; n++) {
    idx[n] = sb->decomp_disp[n] + start[n];
  }
  // printf("start[0] = %d, idx[0] = %d\n",start[0], idx[0]);
}

int *getDatSizeFromOpsArg(ops_arg *arg) { return arg->dat->size; }

int getDatDimFromOpsArg(ops_arg *arg) { return arg->dat->dim; }

// need differet routines for 1D, 2D 3D etc.
int getDatBaseFromOpsArg1D(ops_arg *arg, int *start, int dim) {

  /*convert to C indexing*/
  start[0] -= 1;

  int dat = OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // printf("start[0] = %d, base = %d, dim = %d, d_m[0] = %d dat = %d\n",
  //      start[0],arg->dat->base[0],dim, arg->dat->d_m[0], dat);

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d] + OPS_sub_dat_list[arg->dat->index]->d_im[d];
  int base = dat * 1 *
             (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  return base / (arg->dat->type_size) + 1;
}

int getDatBaseFromOpsArg2D(ops_arg *arg, int *start, int dim) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;

  int dat = OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d] + OPS_sub_dat_list[arg->dat->index]->d_im[d];
  int base = dat * 1 *
             (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  base = base +
         dat * arg->dat->size[0] *
             (start[1] * arg->stencil->stride[1] - arg->dat->base[1] - d_m[1]);

  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  return base / (arg->dat->type_size) + 1;
}

int getDatBaseFromOpsArg3D(ops_arg *arg, int *start, int dim) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;
  start[2] -= 1;

  int dat = OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d] + OPS_sub_dat_list[arg->dat->index]->d_im[d];
  int base = dat * 1 *
             (start[0] * arg->stencil->stride[0] - arg->dat->base[0] - d_m[0]);
  base = base +
         dat * arg->dat->size[0] *
             (start[1] * arg->stencil->stride[1] - arg->dat->base[1] - d_m[1]);
  base = base +
         dat * arg->dat->size[0] * arg->dat->size[1] *
             (start[2] * arg->stencil->stride[2] - arg->dat->base[2] - d_m[2]);

  if (arg->dat->amr == 1) {
    printf("Internal error: getDatBaseFromOpsArg3D called for an AMR dataset!\n");
    exit(-1);
  }
  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  start[2] += 1;
  return base / (arg->dat->type_size) + 1;
}

int getDatBaseFromOpsArg3DAMR(ops_arg *arg, int *start, int dim, int amrblock) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;
  start[2] -= 1;

  int dat = OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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

  if (arg->dat->amr == 1) {
    int bsize = arg->dat->elem_size;
    for (int d = 0; d < block_dim; d++) {
      bsize *= arg->dat->size[d];
    }
    base += (amrblock - 1)*bsize;
  }


  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  start[2] += 1;
  return base / (arg->dat->type_size) + 1;
}
char *getReductionPtrFromOpsArg(ops_arg *arg, ops_block block) {
  // return (char *)((ops_reduction)arg->data)->data;
  // printf("block->index %d ((ops_reduction)arg->data)->size =
  // %d\n",block->index, ((ops_reduction)arg->data)->size);
  return (char *)((ops_reduction)arg->data)->data +
         ((ops_reduction)arg->data)->size * block->index;
}

char *getGblPtrFromOpsArg(ops_arg *arg) { return (char *)(arg->data); }
