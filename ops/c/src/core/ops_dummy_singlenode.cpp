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
#include <string>
#include <assert.h>

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

ops_reduction _ops_decl_reduction_handle(OPS_instance *instance, int size, const char *type,
                                        const char *name) {
  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 ||
      strcmp(type, "double precision") == 0)
    type = "double";
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0)
    type = "float";
  else if (strcmp(type, "int") == 0 || strcmp(type, "integer") == 0 ||
           strcmp(type, "integer(4)") == 0 || strcmp(type, "int(4)") == 0)
    type = "int";

  return ops_decl_reduction_handle_core(instance, size, type, name);
}

ops_reduction ops_decl_reduction_handle(int size, const char *type,
                                        const char *name) {
  return _ops_decl_reduction_handle(OPS_instance::getOPSInstance(), size, type, name);
}

void ops_execute_reduction(ops_reduction handle) { (void)handle; }

int _ops_is_root(OPS_instance* instance) { (void)instance; return 1; }

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
  (void)ndats;
}

void ops_halo_exchanges(ops_arg *args, int nargs, int *range) {
  (void)args;
  (void)range;
  (void)nargs;
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

void printf2(OPS_instance *instance, const char *format, ...) {
  char buf[1000];
  va_list argptr;
  va_start(argptr, format);
  vsnprintf(buf,1000, format, argptr);
  va_end(argptr);
  instance->ostream() << buf;
}

void ops_printf2(OPS_instance *instance, const char *format, ...) {
  char buf[1000];
  va_list argptr;
  va_start(argptr, format);
  vsnprintf(buf,1000,format, argptr);
  va_end(argptr);
  instance->ostream() << buf;
}


void fprintf2(std::ostream &stream, const char *format, ...) {
  char buf[1000];
  va_list argptr;
  va_start(argptr, format);
  vsnprintf(buf, 1000, format, argptr);
  va_end(argptr);
  stream << buf;
}

void ops_fprintf2(std::ostream &stream, const char *format, ...) {
  char buf[1000];
  va_list argptr;
  va_start(argptr, format);
  vsnprintf(buf, 1000,format, argptr);
  va_end(argptr);
  stream << buf;
}

void ops_fprintf(FILE *stream, const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stream, format, argptr);
  va_end(argptr);
}

bool ops_checkpointing_filename(const char *file_name, std::string &filename_out,
                                std::string &filename_out2) {
  filename_out = file_name;
  (void)filename_out2;
  //if (filename_out2!=NULL) *filename_out2='\0';
  return false;
}

void ops_checkpointing_duplicate_data(ops_dat dat, int my_type, int my_nelems,
                                      char *my_data, int *my_range,
                                      int *rm_type, int *rm_elems,
                                      char **rm_data, int **rm_range) {
  *rm_type = 0;
  *rm_elems = 0;
  (void)dat;
  (void)my_type;
  (void)my_nelems;
  (void)my_data;
  (void)my_range;
  (void)rm_data;
  (void)rm_range;
  
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
  (void)args; (void)nargs;
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
extern "C" {

int *getDatSizeFromOpsArg(ops_arg *arg) { return arg->dat->size; }

int getDatDimFromOpsArg(ops_arg *arg) { return arg->dat->dim; }

// need differet routines for 1D, 2D 3D etc.
int getDatBaseFromOpsArg1D(ops_arg *arg, int *start, int dim) {
    (void)dim;
  /*convert to C indexing*/
  start[0] -= 1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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
  return base / (arg->dat->type_size) + 1;
}

int getDatBaseFromOpsArg2D(ops_arg *arg, int *start, int dim) {
    (void)dim;
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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
  return base / (arg->dat->type_size) + 1;
}

int getDatBaseFromOpsArg3D(ops_arg *arg, int *start, int dim) {
    (void)dim;
  /*convert to C indexing*/
  start[0] -= 1;
  start[1] -= 1;
  start[2] -= 1;

  int dat = OPS_instance::getOPSInstance()->OPS_soa ? arg->dat->type_size : arg->dat->elem_size;
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
  return base / (arg->dat->type_size) + 1;
}

char *getReductionPtrFromOpsArg(ops_arg *arg, ops_block block) {
    (void)block;
  return (char *)((ops_reduction)arg->data)->data;
}

char *getGblPtrFromOpsArg(ops_arg *arg) { return (char *)(arg->data); }

int getRange(ops_block block, int* start, int* end, int* range) { (void)block;(void)start;(void)end;(void)range; return 1; }

void getIdx(ops_block block, int *start, int *idx) {
  int block_dim = block->dims;
  for (int n = 0; n < block_dim; n++) {
    idx[n] = start[n];
  }
}

}

int ops_dat_get_local_npartitions(ops_dat dat) {
    (void)dat;
  return 1;
}

size_t ops_dat_get_slab_extents(ops_dat dat, int part, int *disp, int *size, int *slab) {
  int sizel[OPS_MAX_DIM], displ[OPS_MAX_DIM];
  ops_dat_get_extents(dat, part, displ, sizel);
  size_t bytes = dat->elem_size;
  for (int d = 0; d < dat->block->dims; d++) {
    if (slab[2*d]<0 || slab[2*d+1]>sizel[d]) {
      OPSException ex(OPS_RUNTIME_ERROR);
      ex << "Error: ops_dat_get_slab_extents() called on " << dat->name << " with slab ranges in dimension "<<d<<": "<<slab[2*d]<<"-"<<slab[2*d+1]<<" beyond data size 0-" <<sizel[d];
    throw ex;
    }
    disp[d] = slab[2*d];
    size[d] = slab[2*d+1]-slab[2*d];
    bytes *= size[d];
  }
  return bytes;
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
    (void)stencil; (void)part;
    if (dat->dirty_hd == OPS_DEVICE || *memspace == OPS_DEVICE) {
        if(dat->data_d == NULL) {
            OPSException ex(OPS_RUNTIME_ERROR);
            ex << "Error: ops_dat_get_raw_pointer() sees a NULL device buffer when the buffer should not "
                "be NULL.   This should never happen.  ops_dat name: " << dat->name;
            throw ex;
        }
    }

    if (*memspace == OPS_HOST) {
        if (dat->dirty_hd == OPS_DEVICE) {
            // User wants raw pointer on host, and data is dirty on the device
            // Fetch the data from device to host
            ops_get_data(dat);
        }
        else {
            // Data is diry on host - nothing to do
        }
    } else if (*memspace == OPS_DEVICE) {
        if (dat->dirty_hd == OPS_HOST) {
            // User wants raw pointer on device, and data is dirty on the host
            // Upload the data from host to device
            ops_put_data(dat);
        }
        else {
            // Data is dirty on device - nothing to do
        }
    } 
    else {
        // User has not specified where they want the pointer
        // We need a default behaviour:
        if (dat->dirty_hd == OPS_DEVICE)        *memspace = OPS_DEVICE;
        else if (dat->dirty_hd == OPS_HOST)     *memspace = OPS_HOST;
        else if (dat->data_d != NULL)           *memspace = OPS_DEVICE;
        else                                    *memspace = OPS_HOST;
    }

    assert(*memspace==OPS_HOST || *memspace==OPS_DEVICE);
    // Lock the ops_dat with the current memspace
    dat->locked_hd = *memspace;
    return (*memspace == OPS_HOST ? dat->data : dat->data_d) + dat->base_offset;
}

void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc) {
    (void)part;
    if (dat->locked_hd==0) {
        // Dat is unlocked
        OPSException ex(OPS_RUNTIME_ERROR);
        ex << "Error: ops_dat_release_raw_data() called, but with no matching ops_dat_get_raw_pointer() beforehand: " << dat->name;
        throw ex;
    }
    if (acc != OPS_READ)
        dat->dirty_hd = dat->locked_hd; // dirty on host or device depending on where the pointer was obtained

    // Unlock the ops_dat
    dat->locked_hd = 0;
}

void ops_dat_release_raw_data_memspace(ops_dat dat, int part, ops_access acc, ops_memspace *memspace) {
    (void)part;
    if (dat->locked_hd==0) {
        OPSException ex(OPS_RUNTIME_ERROR);
        ex << "Error: ops_dat_release_raw_data_memspace() called, but with no matching ops_dat_get_raw_pointer() beforehand: " << dat->name;
        throw ex;
    }
    if (acc != OPS_READ)
        dat->dirty_hd = *memspace; // dirty on host or device depending on argument

    // Unlock the ops_dat
    dat->locked_hd = 0;
}

template <int dir>
void copy_loop(char *data, char *ddata, int *lsize, int *dsize, int *d_m, int elem_size) {
  //TODO: add OpenMP here if needed
#if OPS_MAX_DIM>4
  for (int m = 0; m < lsize[4]; m++) {
    size_t moff = m * lsize[0] * lsize[1] * lsize[2] * lsize[3];
    size_t moff2 = (m-d_m[4])*dsize[3]*dsize[2]*dsize[1]*dsize[0];
#else
  size_t moff = 0;
  size_t moff2 = 0;
#endif
#if OPS_MAX_DIM>3
  for (int l = 0; l < lsize[3]; l++) {
    size_t loff = l * lsize[0] * lsize[1] * lsize[2];
    size_t loff2 = (l-d_m[3])*dsize[2]*dsize[1]*dsize[0];
#else
  size_t loff = 0;
  size_t loff2 = 0;
#endif
  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      if (dir == 0)
        memcpy(&data[moff + loff + k*lsize[0]*lsize[1]+j*lsize[0]],
            &ddata[(moff2+loff2+(k-d_m[2])*dsize[1]*dsize[0] + (j-d_m[1])*dsize[0] - d_m[0])* elem_size],
             lsize[0]);
      else
        memcpy(&ddata[(moff2+loff2+(k-d_m[2])*dsize[1]*dsize[0] + (j-d_m[1])*dsize[0] - d_m[0])* elem_size],
             &data[moff + loff + k*lsize[0]*lsize[1]+j*lsize[0]],
             lsize[0]);
#if OPS_MAX_DIM>3
  }
#endif
#if OPS_MAX_DIM>4
  }
#endif
}

template <int dir>
void copy_loop_slab(char *data, char *ddata, int *lsize, int *dsize, int *d_m, int elem_size, int *range2) {
  //TODO: add OpenMP here if needed
#if OPS_MAX_DIM>4
  for (int m = 0; m < lsize[4]; m++) {
    size_t moff = m * lsize[0] * lsize[1] * lsize[2] * lsize[3];
    size_t moff2 = (range2[2*4]+m-d_m[4])*dsize[3]*dsize[2]*dsize[1]*dsize[0];
#else
  size_t moff = 0;
  size_t moff2 = 0;
#endif
#if OPS_MAX_DIM>3
  for (int l = 0; l < lsize[3]; l++) {
    size_t loff = l * lsize[0] * lsize[1] * lsize[2];
    size_t loff2 = (range2[2*3]+l-d_m[3])*dsize[2]*dsize[1]*dsize[0];
#else
  size_t loff = 0;
  size_t loff2 = 0;
#endif
  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      if (dir == 0)
      memcpy(&data[moff + loff + k*lsize[0]*lsize[1]+j*lsize[0]],
             &ddata[(moff2+loff2+(range2[2*2]+k-d_m[2])*dsize[1]*dsize[0] + (range2[2*1]+j-d_m[1])*dsize[0] + range2[2*0] - d_m[0])* elem_size],
             lsize[0]);
      else
      memcpy(&ddata[(moff2+loff2+(range2[2*2]+k-d_m[2])*dsize[1]*dsize[0] + (range2[2*1]+j-d_m[1])*dsize[0] + range2[2*0] - d_m[0])* elem_size],
          &data[moff + loff + k*lsize[0]*lsize[1]+j*lsize[0]],
             lsize[0]);
#if OPS_MAX_DIM>3
  }
#endif
#if OPS_MAX_DIM>4
  }
#endif
}


void ops_dat_fetch_data(ops_dat dat, int part, char *data) {
  ops_dat_fetch_data_memspace(dat, part, data, OPS_HOST);
}

void ops_dat_fetch_data_host(ops_dat dat, int part, char *data) {
  ops_execute(dat->block->instance);
  ops_get_data(dat);
  int lsize[OPS_MAX_DIM] = {1};
  int ldisp[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(dat, part, ldisp, lsize);
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    ldisp[d] = 0;
  }
  lsize[0] *= dat->elem_size; //now in bytes
  if (dat->block->dims>5) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for dims>5");
  if (dat->block->instance->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for SoA");
  copy_loop<0>(data, dat->data, lsize, dat->size, dat->d_m, dat->elem_size);
}


void ops_dat_fetch_data_slab_host(ops_dat dat, int part, char *data, int *range) {
    (void)part;
  ops_execute(dat->block->instance);
  ops_get_data(dat);
  int lsize[OPS_MAX_DIM] = {1};
  int range2[2*OPS_MAX_DIM] = {0};
  for (int d = 0; d < dat->block->dims; d++) {
    lsize[d] = range[2*d+1]-range[2*d+0];
    range2[2*d] = range[2*d];
    range2[2*d+1] = range[2*d+1];
  }
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    range2[2*d] = 0;
    range2[2*d+1] = 1;
  }
  lsize[0] *= dat->elem_size; //now in bytes
  if (dat->block->dims>5) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for dims>5");
  if (dat->block->instance->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for SoA");
  copy_loop_slab<0>(data, dat->data, lsize, dat->size, dat->d_m, dat->elem_size, range2);
}


void ops_dat_set_data(ops_dat dat, int part, char *data) {
  ops_dat_set_data_memspace(dat, part, data, OPS_HOST);
}


void ops_dat_set_data_host(ops_dat dat, int part, char *data) {
  ops_execute(dat->block->instance);
  int lsize[OPS_MAX_DIM] = {1};
  int ldisp[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(dat, part, ldisp, lsize);
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    ldisp[d] = 0;
  }
  lsize[0] *= dat->elem_size; //now in bytes
  if (dat->block->dims>5) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_set_data not implemented for dims>5");
  if (dat->block->instance->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_set_data not implemented for SoA");
  copy_loop<1>(data, dat->data, lsize, dat->size, dat->d_m, dat->elem_size);
  dat->dirty_hd = 1;
}

void ops_dat_set_data_slab_host(ops_dat dat, int part, char *data, int *range) {
    (void)part;
  ops_execute(dat->block->instance);
  ops_get_data(dat);
  int lsize[OPS_MAX_DIM] = {1};
  int range2[2*OPS_MAX_DIM] = {0};
  for (int d = 0; d < dat->block->dims; d++) {
    lsize[d] = range[2*d+1]-range[2*d+0];
    range2[2*d] = range[2*d];
    range2[2*d+1] = range[2*d+1];
  }
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    range2[2*d] = 0;
    range2[2*d+1] = 1;
  }
  lsize[0] *= dat->elem_size; //now in bytes
  if (dat->block->dims>5) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for dims>5");
  if (dat->block->instance->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for SoA");
  copy_loop_slab<1>(data, dat->data, lsize, dat->size, dat->d_m, dat->elem_size, range2);
  dat->dirty_hd = 1;
}


int ops_dat_get_global_npartitions(ops_dat dat) {
    (void)dat;
  return 1;
}

void ops_dat_get_extents(ops_dat dat, int part, int *disp, int *size) {
    (void)part;
  if (disp != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      disp[d] = 0;
  if (size != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      size[d] = dat->size[d] + dat->d_m[d] - dat->d_p[d];
}




ops_dat ops_dat_copy(ops_dat orig_dat) 
{
   // Allocate an empty dat on a block
   // The block has no internal data buffers
  ops_dat dat = ops_dat_alloc_core(orig_dat->block);
  // Do a deep copy from orig_dat into the new dat
  ops_dat_deep_copy(dat, orig_dat);
  return dat;
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name) {
  // printf("file %s, name %s type = %s\n",file_name, dat->name, dat->type);
  // need to get data from GPU
  ops_get_data(dat);
  ops_print_dat_to_txtfile_core(dat, file_name);
}

void ops_NaNcheck(ops_dat dat) {
  char buffer[1]={'\0'};
  // need to get data from GPU
  ops_get_data(dat);
  ops_NaNcheck_core(dat, buffer);
}


void _ops_partition(OPS_instance *instance, const char *routine) {
  (void)instance;
  (void)routine;
}

void _ops_partition(OPS_instance *instance, const char *routine, std::map<std::string, void*>& opts) {
  (void)instance;
  (void)routine;
  (void)opts;
}

void ops_partition(const char *routine) {
  (void)routine;
}

void ops_partition_opts(const char *routine, std::map<std::string, void*>& opts) {
  (void)routine;
  (void)opts;
}

void ops_timers(double *cpu, double *et) {
  ops_timers_core(cpu, et);
}

void _ops_exit(OPS_instance *instance) {
  if (instance->is_initialised == 0) return;
  if (instance->ops_halo_buffer!=NULL) ops_free(instance->ops_halo_buffer);
  if (instance->OPS_consts_bytes > 0) {
    ops_free(instance->OPS_consts_h);
    if (instance->OPS_gbl_prev!=NULL) ops_device_freehost(instance, (void**)&instance->OPS_gbl_prev);
    if (instance->OPS_consts_d!=NULL) ops_device_free(instance, (void**)&instance->OPS_consts_d);
  }
  if (instance->OPS_reduct_bytes > 0) {
    ops_free(instance->OPS_reduct_h);
    if (instance->OPS_reduct_d!=NULL) ops_device_free(instance, (void**)&instance->OPS_reduct_d);
  }

  ops_exit_core(instance);
  ops_exit_device(instance);
}

void ops_dat_deep_copy(ops_dat target, ops_dat source) 
{
  /* The constraint is that OPS makes it very easy for users to alias ops_dats.  A deep copy
    * should work even if dats have been aliased.  Suppose a user has written something like
    *
    *    ops_dat x = ops_decl_dat( ... );
    *    ops_dat y = ops_decl_dat( ... );
    *    ops_dat z = x;
    *    ops_dat_deep_copy(x, y);
    *
    * In this case we cannot call ops_free_dat(x) since that would leave 'z' pointing at invalid memory.
    * OPS has no knowledge of 'z' - there is no entry in any internal tables corresponding to 'z'.
    * Hence the only way this function can work is if we leave (*x) intact (i.e the ops_dat_core pointed at
    * by x) and change the entries inside the ops_dat_core.  Then 'z' will continue to point at valid data.
    *
    * If the blocks in source and target are different, then the deep copy could entail MPI re-distribution of 
    * data. For the moment, perhaps we ignore this ... ?
    */
  // Copy the metadata.  This will reallocate target->data if necessary
  int realloc = ops_dat_copy_metadata_core(target, source);
  if(realloc && source->block->instance->OPS_hybrid_gpu) {
    if(target->data_d != nullptr) {
      ops_device_free(source->block->instance, (void**)&(target->data_d));
      target->data_d = nullptr;
    }
    ops_device_malloc(source->block->instance, (void**)&(target->data_d), target->mem);
  }
   // Metadata and buffers are set up
   // Enqueue a lazy copy of data from source to target
  int range[2*OPS_MAX_DIM];
  for (int i = 0; i < source->block->dims; i++) {
    range[2*i] = source->base[i] + source->d_m[i];
    range[2*i+1] = range[2*i] + source->size[i];
  }
  for (int i = source->block->dims; i < OPS_MAX_DIM; i++) {
    range[2*i] = 0;
    range[2*i+1] = 1;
  }
  ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, source, range);
  if (source->block->instance->OPS_hybrid_gpu) {
    desc->name = "ops_internal_copy_device";
    desc->device = 1;
    desc->function = ops_internal_copy_device;
  } else {
    desc->name = "ops_internal_copy_seq";
    desc->device = 0;
    desc->function = ops_internal_copy_seq;
  }
  ops_enqueue_kernel(desc);
}

void _ops_init(OPS_instance *instance, const int argc, const char * const argv[], const int diags) {
  ops_init_core(instance, argc, argv, diags);
  ops_init_device(instance, argc, argv, diags);
}

void ops_init(const int argc, const char *const argv[], const int diags) {
  _ops_init(OPS_instance::getOPSInstance(), argc, argv, diags);
}

void ops_exit() { _ops_exit(OPS_instance::getOPSInstance()); }