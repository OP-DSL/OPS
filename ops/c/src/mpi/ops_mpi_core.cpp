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
  * @brief OPS mpi core routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the core mpi decl routines for the OPS mpi backend
  */

#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_exceptions.h>
#include <string>

#ifndef __XDIMS__ // perhaps put this into a separate header file
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

#ifndef __UDIMS__
#define __UDIMS__
int udim0, udim1, udim2, udim3, udim4, udim5, udim6, udim7, udim8, udim9,
    udim10, udim11, udim12, udim13, udim14, udim15, udim16, udim17, udim18,
    udim19, udim20, udim21, udim22, udim23, udim24, udim25, udim26, udim27,
    udim28, udim29, udim30, udim31, udim32, udim33, udim34, udim35, udim36,
    udim37, udim38, udim39, udim40, udim41, udim42, udim43, udim44, udim45,
    udim46, udim47, udim48, udim49, udim50, udim51, udim52, udim53, udim54,
    udim55, udim56, udim57, udim58, udim59, udim60, udim61, udim62, udim63,
    udim64, udim65, udim66, udim67, udim68, udim69, udim70, udim71, udim72,
    udim73, udim74, udim75, udim76, udim77, udim78, udim79, udim80, udim81,
    udim82, udim83, udim84, udim85, udim86, udim87, udim88, udim89, udim90,
    udim91, udim92, udim93, udim94, udim95, udim96, udim97, udim98, udim99;
#endif /* __UDIMS__ */

#ifndef __VDIMS__
#define __VDIMS__
int vdim0, vdim1, vdim2, vdim3, vdim4, vdim5, vdim6, vdim7, vdim8, vdim9,
    vdim10, vdim11, vdim12, vdim13, vdim14, vdim15, vdim16, vdim17, vdim18,
    vdim19, vdim20, vdim21, vdim22, vdim23, vdim24, vdim25, vdim26, vdim27,
    vdim28, vdim29, vdim30, vdim31, vdim32, vdim33, vdim34, vdim35, vdim36,
    vdim37, vdim38, vdim39, vdim40, vdim41, vdim42, vdim43, vdim44, vdim45,
    vdim46, vdim47, vdim48, vdim49, vdim50, vdim51, vdim52, vdim53, vdim54,
    vdim55, vdim56, vdim57, vdim58, vdim59, vdim60, vdim61, vdim62, vdim63,
    vdim64, vdim65, vdim66, vdim67, vdim68, vdim69, vdim70, vdim71, vdim72,
    vdim73, vdim74, vdim75, vdim76, vdim77, vdim78, vdim79, vdim80, vdim81,
    vdim82, vdim83, vdim84, vdim85, vdim86, vdim87, vdim88, vdim89, vdim90,
    vdim91, vdim92, vdim93, vdim94, vdim95, vdim96, vdim97, vdim98, vdim99;
#endif /* __VDIMS__ */

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
  MPI_Barrier(OPS_MPI_GLOBAL);
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

void printf2(OPS_instance *instance, const char *format, ...) {
  if (ops_my_global_rank == MPI_ROOT) {
    char buf[1000];
    va_list argptr;
    va_start(argptr, format);
    vsprintf(buf,format, argptr);
    va_end(argptr);
    instance->ostream() << buf;
  }
}

void ops_printf2(OPS_instance *instance, const char *format, ...) {
  if (ops_my_global_rank == MPI_ROOT) {
    char buf[1000];
    va_list argptr;
    va_start(argptr, format);
    vsprintf(buf,format, argptr);
    va_end(argptr);
    instance->ostream() << buf;
  }
}


void fprintf2(std::ostream &stream, const char *format, ...) {
  if (ops_my_global_rank == MPI_ROOT) {
    char buf[1000];
    va_list argptr;
    va_start(argptr, format);
    vsprintf(buf, format, argptr);
    va_end(argptr);
    stream << buf;
  }
}

void ops_fprintf2(std::ostream &stream, const char *format, ...) {
  if (ops_my_global_rank == MPI_ROOT) {
    char buf[1000];
    va_list argptr;
    va_start(argptr, format);
    vsprintf(buf, format, argptr);
    va_end(argptr);
    stream << buf;
  }
}


void ops_compute_moment(double t, double *first, double *second) {
  double times[2] = {0.0};
  double times_reduced[2] = {0.0};
  int comm_size;
  times[0] = t;
  times[1] = t * t;
  MPI_Comm_size(OPS_MPI_GLOBAL, &comm_size);
  MPI_Reduce(times, times_reduced, 2, MPI_DOUBLE, MPI_SUM, 0, OPS_MPI_GLOBAL);

  *first = times_reduced[0] / (double)comm_size;
  *second = times_reduced[1] / (double)comm_size;
}

int _ops_is_root(OPS_instance *instance) {
  int my_rank;
  MPI_Comm_rank(OPS_MPI_GLOBAL, &my_rank);
  return (my_rank == MPI_ROOT);
}


int ops_is_root() {
  return _ops_is_root(OPS_instance::getOPSInstance());
}


int ops_get_proc() {
  int my_rank;
  MPI_Comm_rank(OPS_MPI_GLOBAL, &my_rank);
  return my_rank;
}

int ops_num_procs() {
  int size;
  MPI_Comm_size(OPS_MPI_GLOBAL, &size);
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

ops_arg ops_arg_reduce(ops_reduction handle, int dim, const char *type,
                       ops_access acc) {
  int was_initialized = handle->initialized;
  ops_arg temp = ops_arg_reduce_core(handle, dim, type, acc);
  if (!was_initialized) {
    for (int i = 1; i < OPS_instance::getOPSInstance()->OPS_block_index; i++) {
      memcpy(handle->data + i * handle->size, handle->data, handle->size);
    }
  }
  return temp;
}

ops_reduction _ops_decl_reduction_handle(OPS_instance *instance, int size, const char *type,
                                        const char *name) {

  if (strcmp(type, "double") == 0 ||
      strcmp(type, "real(8)") == 0 ||
      strcmp(type, "real(kind=8)") == 0 ||
      strcmp(type, "double precision") == 0)
  {
    type = "double";
  }
  else if (strcmp(type, "float") == 0 ||
           strcmp(type, "real") == 0 ||
           strcmp(type, "real(4)") == 0 ||
           strcmp(type, "real(kind=4)") == 0)
  {
    type = "float";
  }
  else if (strcmp(type, "int") == 0 ||
           strcmp(type, "int(4)") == 0 ||
           strcmp(type, "integer") == 0 ||
           strcmp(type, "integer(4)") == 0 ||
           strcmp(type, "integer(kind=4)") == 0)
  {
    type = "int";
  }
  else
  {
    OPSException ex(OPS_NOT_IMPLEMENTED);
    ex << "Error: Unknown data type for ops_decl_reduction_handle";
    throw ex;
  }

  ops_reduction red = ops_decl_reduction_handle_core(instance, size, type, name);
  if (instance->OPS_block_index < 1)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_decl_reduction_handle() should only be called after \
                                           declaring at least one ops_block");
  
  red->data = (char *)ops_realloc(red->data,
                                  red->size * (instance->OPS_block_index) * sizeof(char));
  return red;
}

ops_reduction ops_decl_reduction_handle(int size, const char *type,
                                        const char *name) {
  return _ops_decl_reduction_handle(OPS_instance::getOPSInstance(), size, type, name);
}


bool ops_checkpointing_filename(const char *file_name, std::string &filename_out,
                                std::string &filename_out2)
{
  filename_out = file_name;
  filename_out += ".";
  filename_out += std::to_string(ops_my_global_rank);

  filename_out2 = file_name;
  filename_out2 += ".";
  filename_out2 += std::to_string(
       (ops_my_global_rank + OPS_instance::getOPSInstance()->OPS_ranks_per_node) % ops_comm_global_size);
  filename_out2 += ".dup";

  return (OPS_instance::getOPSInstance()->OPS_enable_checkpointing > 1);
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

char *OPS_checkpointing_dup_buffer = NULL;
int OPS_checkpointing_dup_buffer_size = 0;
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
            (ops_my_global_rank + OPS_instance::getOPSInstance()->OPS_ranks_per_node) % ops_comm_global_size,
            1000 + OPS_instance::getOPSInstance()->OPS_dat_index + dat->index, OPS_MPI_GLOBAL, &requests[0]);
  int bytesize = dat->elem_size / dat->dim;
  MPI_Isend(my_data, my_nelems * bytesize, MPI_CHAR,
            (ops_my_global_rank + OPS_instance::getOPSInstance()->OPS_ranks_per_node) % ops_comm_global_size,
            1000 + dat->index, OPS_MPI_GLOBAL, &requests[1]);

  MPI_Recv(recv_stats, 2 + 2 * OPS_MAX_DIM, MPI_INT,
           (ops_comm_global_size + ops_my_global_rank - OPS_instance::getOPSInstance()->OPS_ranks_per_node) %
               ops_comm_global_size,
           1000 + OPS_instance::getOPSInstance()->OPS_dat_index + dat->index, OPS_MPI_GLOBAL, &statuses[0]);
  if (recv_stats[1] * bytesize > OPS_checkpointing_dup_buffer_size) {
      OPS_checkpointing_dup_buffer =
        (char *)ops_realloc(OPS_checkpointing_dup_buffer,
                            recv_stats[1] * bytesize * 2 * sizeof(char));
      OPS_checkpointing_dup_buffer_size = recv_stats[1] * bytesize * 2;
  }
  *rm_data = OPS_checkpointing_dup_buffer;
  MPI_Recv(*rm_data, recv_stats[1] * bytesize, MPI_CHAR,
           (ops_comm_global_size + ops_my_global_rank - OPS_instance::getOPSInstance()->OPS_ranks_per_node) %
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

extern "C" int getRange(ops_block block, int *start, int *end, int *range) {

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

extern "C" void getIdx(ops_block block, int *start, int *idx) {
  int block_dim = block->dims;
  sub_block_list sb = OPS_sub_block_list[block->index];
  for (int n = 0; n < block_dim; n++) {
    idx[n] = sb->decomp_disp[n] + start[n];
  }
  // printf("start[0] = %d, idx[0] = %d\n",start[0], idx[0]);
}

extern "C" int *getDatSizeFromOpsArg(ops_arg *arg) { return arg->dat->size; }

extern "C" int getDatDimFromOpsArg(ops_arg *arg) { return arg->dat->dim; }

// need different routines for 1D, 2D 3D etc.
extern "C" int getDatBaseFromOpsArg1D(ops_arg *arg, int *start, int dim) {

  /*convert to C indexing*/
  start[0] -= 1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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

extern "C" int getDatBaseFromOpsArg2D(ops_arg *arg, int *start, int dim) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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

extern "C" int getDatBaseFromOpsArg3D(ops_arg *arg, int *start, int dim) {
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;
  start[2] -= 1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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

  // printf("base = %d\n",base/(dat/dim));
  /*revert to Fortran indexing*/
  start[0] += 1;
  start[1] += 1;
  start[2] += 1;
  return base / (arg->dat->type_size) + 1;
}

extern "C" char *getReductionPtrFromOpsArg(ops_arg *arg, ops_block block) {
  // return (char *)((ops_reduction)arg->data)->data;
  // printf("block->index %d ((ops_reduction)arg->data)->size =
  // %d\n",block->index, ((ops_reduction)arg->data)->size);
  return (char *)((ops_reduction)arg->data)->data +
         ((ops_reduction)arg->data)->size * block->index;
}

extern "C" char *getGblPtrFromOpsArg(ops_arg *arg) { return (char *)(arg->data); }

ops_dat ops_dat_copy_mpi_core(ops_dat orig_dat) {
   // So MPI I'm not going to try change ...
  ops_dat dat = ops_dat_alloc_core(orig_dat->block);
  OPS_sub_dat_list = (sub_dat_list *)ops_realloc(
      OPS_sub_dat_list, OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(sub_dat_list));

  sub_dat_list sd = (sub_dat_list)ops_calloc(1, sizeof(sub_dat));
  sd->dirtybit = 1;
  sd->dirty_dir_send =
      (int *)ops_malloc(sizeof(int) * 2 * dat->block->dims * MAX_DEPTH);
  sd->dirty_dir_recv =
      (int *)ops_malloc(sizeof(int) * 2 * dat->block->dims * MAX_DEPTH);
  OPS_sub_dat_list[dat->index] = sd;
  int *dirt1 = sd->dirty_dir_send;
  int *dirt2 = sd->dirty_dir_recv;
  memcpy(OPS_sub_dat_list[dat->index], OPS_sub_dat_list[orig_dat->index], sizeof(sub_dat));
  sd->dat = dat;
  sd->dirty_dir_send = dirt1;
  sd->dirty_dir_recv = dirt2;
  size_t *prod_t = (size_t *)ops_malloc((orig_dat->block->dims + 1) * sizeof(size_t));
  memcpy(prod_t, &OPS_sub_dat_list[orig_dat->index]->prod[-1], (orig_dat->block->dims + 1) * sizeof(size_t));
  sd->prod = &prod_t[1];
  memcpy(dirt1, OPS_sub_dat_list[orig_dat->index]->dirty_dir_send, sizeof(int) * 2 * dat->block->dims * MAX_DEPTH);
  memcpy(dirt2, OPS_sub_dat_list[orig_dat->index]->dirty_dir_recv, sizeof(int) * 2 * dat->block->dims * MAX_DEPTH);
  sd->halos = (ops_int_halo *)ops_calloc(MAX_DEPTH * orig_dat->block->dims , sizeof(ops_int_halo));
  memcpy(sd->halos, OPS_sub_dat_list[orig_dat->index]->halos, MAX_DEPTH * orig_dat->block->dims* sizeof(ops_int_halo));

  return dat;
}

ops_kernel_descriptor * ops_dat_deep_copy_mpi_core(ops_dat target, ops_dat source) {
  int range[2*OPS_MAX_DIM];
  sub_dat_list sdo = OPS_sub_dat_list[source->index];
  for (int i = 0; i < source->block->dims; i++) {
    range[2*i] = sdo->gbl_base[i] + sdo->gbl_d_m[i];
    range[2*i+1] = range[2*i] + sdo->gbl_size[i];
  }
  for (int i = source->block->dims; i < OPS_MAX_DIM; i++) {
    range[2*i] = 0;
    range[2*i+1] = 1;
  }
  ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, source, range);

  return desc;
}
