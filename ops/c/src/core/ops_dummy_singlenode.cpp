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
  * @brief dummy function for CPU backend
  * @author Gihan Mudalige
  * @details Implements dummy functions from the MPI backend for the sequential
  * cpu backend (OpenMP and Sequential)
  */

#include "ops_lib_core.h"
#include <ops_exceptions.h>

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
  if (!was_initialized && handle->multithreaded) {
    char *old_data = handle->data;
    int count, stride;
    ops_amr_reduction_size(&count, &stride, handle->size);
    handle->data = (char*)malloc(count*stride);
    for (int i = 0; i < count; i++) {
      memcpy(handle->data + i * stride, old_data, handle->size);
    }
    free(old_data);
  } else if (handle->multithreaded) {
    ops_printf("Error, reduction handle %s first used outside of a block-parallel loop, then inside one. You must get its value in-between\n",handle->name);
    exit(-1);
  }
  return temp;
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

ops_reduction ops_decl_reduction_handle_batch(int size, const char *type,
                                        const char *name, int batchsize) {
  ops_reduction r =  ops_decl_reduction_handle(size, type, name);
  r->batchsize = batchsize;
  r->data = (char*)ops_realloc(r->data,size * batchsize * sizeof(char));
  return r;
}
void ops_execute_reduction(ops_reduction handle) { (void)handle; }

int ops_is_root() { return 1; }

int ops_num_procs() { return 1; }

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

int compute_ranges(ops_arg *args, int nargs, ops_block block, int *range, int * start, int * end, int *arg_idx) {
  for (int n = 0; n < block->dims; n++) {
    start[n] = range[2 * n];
    end[n] = range[2 * n + 1];
    arg_idx[n] = range[2 * n];
  }
  return true;
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

extern "C" int ops_amr_lazy_offset(ops_dat dat);

int getDatBaseFromOpsArg(ops_arg *arg, int *start2, int datdim, int dim, int amr, int amrblock) {
  /*convert to C indexing*/
  int start[OPS_MAX_DIM];
  for (int d = 0; d < dim; d++)
    start[d] = start2[d]-1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
  int block_dim = arg->dat->block->dims;

  // set up initial pointers
  int d_m[OPS_MAX_DIM];
  for (int d = 0; d < block_dim; d++)
    d_m[d] = arg->dat->d_m[d];
  int base = 0;
  int cumsize = 1;
  for (int d = 0; d < dim; d++) {
    int startl = start[d];
    if (arg->argtype == OPS_ARG_PROLONG) startl=start[d]/arg->stencil->mgrid_stride[d];
    else if (arg->argtype == OPS_ARG_RESTRICT) startl=start[d]*arg->stencil->mgrid_stride[d];
    base = base + dat * cumsize *  (startl * arg->stencil->stride[d] - arg->dat->base[d] - d_m[d]);
    cumsize *= arg->dat->size[d];
  }

  if (arg->dat->amr == 1 || amr) {
    int bsize = arg->dat->elem_size;
    for (int d = 0; d < block_dim; d++) {
      bsize *= arg->dat->size[d];
    }
    if (!amr) {
      printf("Internal error: getDatBaseFromOpsArg called for an AMR dataset!\n");
      exit(-1);
    } else if (arg->dat->amr) {
      base += (amrblock - 1)*bsize;
    } else {
      base += bsize * ops_amr_lazy_offset(arg->dat);
    }
  }

  return base / (arg->dat->type_size) + 1;
}

int getDatBaseFromOpsArg3DAMR(ops_arg *arg, int *start, int dim, int amrblock) { 
  return getDatBaseFromOpsArg(arg, start, dim, 3, 1, amrblock);
}
int getDatBaseFromOpsArg1D(ops_arg *arg, int *start, int dim) { 
  return getDatBaseFromOpsArg(arg, start, dim, 1, 0, 0);
}
int getDatBaseFromOpsArg2D(ops_arg *arg, int *start, int dim) { 
  return getDatBaseFromOpsArg(arg, start, dim, 2, 0, 0);
}
int getDatBaseFromOpsArg3D(ops_arg *arg, int *start, int dim) { 
  return getDatBaseFromOpsArg(arg, start, dim, 3, 0, 0);
}

char *getReductionPtrFromOpsArg(ops_arg *arg, ops_block block) {
  if (OPS_instance::getOPSInstance()->ops_loop_over_blocks) {
    int count, stride;
    ops_amr_reduction_size(&count, &stride, ((ops_reduction)arg->data)->size);
    return (char *)((ops_reduction)arg->data)->data + stride * ops_amr_lazy_offset_idx();  
  } else 
    return (char *)((ops_reduction)arg->data)->data;
}

char *getGblPtrFromOpsArg(ops_arg *arg) { return (char *)(arg->data); }

char *getMgridStrideFromArg(ops_arg *arg) { return (char *)(arg->stencil->mgrid_stride); }

int getRange(ops_block block, int *start, int *end, int *range) { return 1; }
int getRange2(ops_arg *args, int nargs, ops_block block, int *start, int *end, int *range, int *arg_idx) { return 1; }

void getIdx(ops_block block, int *start, int *idx) {
  int block_dim = block->dims;
  for (int n = 0; n < block_dim; n++) {
    idx[n] = start[n];
  }
}

int ops_dat_get_local_npartitions(ops_dat dat) {
  return 1;
}

void ops_dat_get_raw_metadata(ops_dat dat, int part, int *disp, int *size, int *stride, int *d_m, int *d_p) {
  ops_dat_get_extents(dat, part, disp, size);
  if (stride != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      stride[d] = dat->size[d];
  if (d_m != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      d_m[d] = dat->d_m[d];
  if (d_p != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      d_p[d] = dat->d_p[d];

}

char* ops_dat_get_raw_pointer(ops_dat dat, int part, ops_stencil stencil, ops_memspace *memspace) {
  if (*memspace == OPS_HOST) {
    if (dat->data_d != NULL && dat->dirty_hd == 2) ops_get_data(dat);
  } else if (*memspace == OPS_DEVICE) {
    if (dat->data_d != NULL && dat->dirty_hd == 1) ops_put_data(dat);
  } else if (dat->dirty_hd == 2 && dat->data_d != NULL) *memspace = OPS_DEVICE;
  else if (dat->dirty_hd == 1) *memspace = OPS_HOST;
  else if (dat->data_d != NULL) *memspace = OPS_DEVICE;
  else *memspace = OPS_HOST;
  return (*memspace == OPS_HOST ? dat->data : dat->data_d) + dat->base_offset;
}

void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc) {
  ops_memspace memspace = 0;
  ops_dat_get_raw_pointer(dat, part, NULL, &memspace);
  if (acc != OPS_READ)
    dat->dirty_hd = (memspace == OPS_HOST ? 1 : 2);
}

void ops_dat_fetch_data(ops_dat dat, int part, char *data) {
  ops_get_data(dat);
  int lsize[OPS_MAX_DIM] = {1};
  int ldisp[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(dat, part, ldisp, lsize);
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    ldisp[d] = 0;
  }
  lsize[0] *= dat->elem_size/dat->dim; //now in bytes
  if (dat->block->dims>3) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for dims>3");
  if (OPS_instance::getOPSInstance()->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for SoA");

  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      memcpy(&data[k*lsize[0]*lsize[1]+j*lsize[0]],
             &dat->data[((j-dat->d_m[1] + (k-dat->d_m[2])*dat->size[1])*dat->size[0] - dat->d_m[0])* dat->elem_size],
             lsize[0]);
}
void ops_dat_set_data(ops_dat dat, int part, char *data) {
  int lsize[OPS_MAX_DIM] = {1};
  int ldisp[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(dat, part, ldisp, lsize);
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    ldisp[d] = 0;
  }
  lsize[0] *= dat->elem_size/dat->dim; //now in bytes
  if (dat->block->dims>3) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_set_data not implemented for dims>3");
  if (OPS_instance::getOPSInstance()->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_set_data not implemented for SoA");

  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      memcpy(&dat->data[((j-dat->d_m[1] + (k-dat->d_m[2])*dat->size[1])*dat->size[0] - dat->d_m[0])* dat->elem_size],
             &data[k*lsize[0]*lsize[1]+j*lsize[0]],
             lsize[0]);

  dat->dirty_hd = 1;
}

int ops_dat_get_global_npartitions(ops_dat dat) {
  return 1;
}

void ops_dat_get_extents(ops_dat dat, int part, int *disp, int *size) {
  if (disp != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      disp[d] = 0;
  if (size != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      size[d] = dat->size[d] + dat->d_m[d] - dat->d_p[d];
}

