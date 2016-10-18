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
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
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

/** @brief ops core library function declarations
  * @author Gihan Mudalige
  * @details function declarations headder file for the core library functions
  * utilized by all OPS backends
  */

#ifndef __OPS_LIB_CORE_H
#define __OPS_LIB_CORE_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/queue.h> //contains double linked list implementation

#include "ops_macros.h"
#include "ops_util.h"

/*
* enum list for ops_par_loop
*/

#define OPS_READ 0
#define OPS_WRITE 1
#define OPS_RW 2
#define OPS_INC 3
#define OPS_MIN 4
#define OPS_MAX 5

#define OPS_ARG_GBL 0
#define OPS_ARG_DAT 1
#define OPS_ARG_IDX 2

#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 80

#define OPS_MAX_DIM 3

/*
 * * zero constants
 * */

#define ZERO_double 0.0;
#define INFINITY_double DBL_MAX;

#define ZERO_float 0.0f;
#define INFINITY_float FLT_MAX;

#define ZERO_int 0;
#define INFINITY_int INT_MAX;

#define ZERO_long 0;
#define INFINITY_long LONG_MAX;

#define ZERO_ll 0;
#define INFINITY_ll LLONG_MAX;

#define ZERO_uint 0;
#define INFINITY_uint UINT_MAX;

#define ZERO_ulong 0;
#define INFINITY_ulong ULONG_MAX;

#define ZERO_ull 0;
#define INFINITY_ull ULLONG_MAX;

#define ZERO_bool 0;

#ifdef __cplusplus
extern "C" {
#endif

/*
* essential typedefs
*/
#ifndef __PGI
typedef unsigned int uint;
#endif
typedef long long ll;
typedef unsigned long long ull;

typedef int ops_access;   // holds OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN,
                          // OP_MAX
typedef int ops_arg_type; // holds OP_ARG_GBL, OP_ARG_DAT

/*
* structures
*/

typedef struct {
  int index;        /* index */
  int dims;         /* dimension of block, 2D,3D .. etc*/
  char const *name; /* name of block */
} ops_block_core;

typedef ops_block_core *ops_block;

typedef struct {
  int index;             /* index */
  ops_block block;       /* block on which data is defined */
  int dim;               /* number of elements per grid point*/
  int elem_size;         /* number of bytes per grid point*/
  int size[OPS_MAX_DIM]; /* size of the array in each block dimension --
                            including halo*/
  int base[OPS_MAX_DIM]; /* base offset to 0,0,... from the start of each
                            dimension*/
  int d_m[OPS_MAX_DIM];  /* halo depth in each dimension, negative direction (at
                            0
                            end)*/
  int d_p[OPS_MAX_DIM];  /* halo depth in each dimension, positive direction (at
                            size end)*/
  char *data;            /* data on host */
  char *data_d;          /* data on device */
  char const *name;      /* name of dataset */
  char const *type;      /* datatype */
  int dirty_hd;          /* flag to indicate dirty status on host and device */
  int user_managed;      /* indicates whether the user is managing memory */
  int is_hdf5; /* indicates whether the data is to read from an hdf5 file*/
  char const *hdf5_file; /* name of hdf5 file from which this dataset was read*/
  int e_dat;             /* flag to indicate if this is an edge dat*/
  long mem; /*memory in bytes allocated to this dat (under MPI, this will be
               memory held on a single MPI proc)*/
  long base_offset; /* computed quantity, giving offset in bytes to the base
                       index */
} ops_dat_core;

typedef ops_dat_core *ops_dat;

// struct definition for a double linked list entry to hold an ops_dat
struct ops_dat_entry_core {
  ops_dat dat;
  TAILQ_ENTRY(ops_dat_entry_core)
  entries; /*holds pointers to next and
             previous entries in the list*/
};

typedef struct ops_dat_entry_core ops_dat_entry;

typedef TAILQ_HEAD(, ops_dat_entry_core) Double_linked_list;

typedef struct {
  ops_block_core *block;
  Double_linked_list datasets;
  int num_datasets;

} ops_block_descriptor;

typedef struct {
  int index;        /* index */
  int dims;         /* dimensionality of the stencil */
  int points;       /* number of stencil elements */
  char const *name; /* name of pointer */
  int *stencil;     /* elements in the stencil */
  int *stride;      /* stride of the stencil */
} ops_stencil_core;

typedef ops_stencil_core *ops_stencil;

typedef struct {
  ops_dat dat;          /* dataset */
  ops_stencil stencil;  /* the stencil */
  int dim;              /* dimension of data */
  char *data;           /* data on host */
  char *data_d;         /* data on device (for CUDA)*/
  ops_access acc;       /* access type */
  ops_arg_type argtype; /* arg type */
  int opt; /*falg to indicate whether this is an optional arg, 0 - optional, 1 -
              not optional*/
} ops_arg;

typedef struct {
  char *data;       /* The data */
  int size;         /* size of data in bytes */
  int initialized;  /* flag indicating whether data has been initialized*/
  int index;        /* unique identifier */
  ops_access acc;   /* Type of reduction it was used for last time */
  const char *type; /* Type */
  const char *name; /* Name */
} ops_reduction_core;
typedef ops_reduction_core *ops_reduction;

typedef struct {
  char *name;      /* name of kernel function */
  int count;       /* number of times called */
  double time;     /* total execution time */
  float transfer;  /* bytes of data transfer (used) */
  double mpi_time; /* time spent in MPI calls */
  //  double       mpi_gather;
  //  double       mpi_scatter;
  //  double       mpi_sendrecv;
  //  double       mpi_reduct;
} ops_kernel;

typedef struct {
  ops_dat from;
  ops_dat to;
  int iter_size[OPS_MAX_DIM];
  int from_base[OPS_MAX_DIM];
  int to_base[OPS_MAX_DIM];
  int from_dir[OPS_MAX_DIM];
  int to_dir[OPS_MAX_DIM];
  int index;
} ops_halo_core;

typedef ops_halo_core *ops_halo;

typedef struct {
  int nhalos;
  ops_halo *halos;
  int index;
} ops_halo_group_core;

typedef ops_halo_group_core *ops_halo_group;

typedef struct ops_kernel_descriptor {
  const char *name;           /* name of kernel */
  ops_arg *args;              /* list of arguments to pass in */
  int nargs;                  /* number of arguments */
  int index;                  /* index of the loop */
  int dim;                    /* number of dimensions */
  int range[2 * OPS_MAX_DIM]; /* process local execution range */
  int orig_range[2 * OPS_MAX_DIM]; /* original execution range */
  ops_block block;            /* block to execute on */
  void (*function)(struct ops_kernel_descriptor
                       *desc); /* Function pointer to a wrapper to be called */
} ops_kernel_descriptor;

/*
* min / max definitions
*/

#ifndef MIN
#define MIN(a, b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? (a) : (b))
#endif
#ifndef SIGN
#define SIGN(a, b) ((b < 0.0) ? (a * (-1)) : (a))
#endif

/*
 * alignment macro based on example on page 50 of CUDA Programming Guide version
 * 3.0
 * rounds up to nearest multiple of 16 bytes
 */

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ROUND_UP_64(bytes) (((bytes) + 63) & ~63)

/*******************************************************************************
* Global constants
*******************************************************************************/

extern int OPS_hybrid_gpu, OPS_gpu_direct;
extern int OPS_kern_max, OPS_kern_curr;
extern ops_kernel *OPS_kernels;

extern int ops_current_kernel;

extern int OPS_diags;

extern int OPS_block_index, OPS_block_max, OPS_dat_index, OPS_dat_max,
    OPS_halo_group_index, OPS_halo_group_max, OPS_halo_index, OPS_halo_max,
    OPS_reduction_index, OPS_reduction_max, OPS_stencil_index;
extern ops_reduction *OPS_reduction_list;

extern ops_block_descriptor *OPS_block_list;
extern ops_stencil *OPS_stencil_list;
extern ops_halo *OPS_halo_list;
extern ops_halo_group *OPS_halo_group_list;
extern Double_linked_list OPS_dat_list; // Head of the double linked list
extern ops_arg *OPS_curr_args;
extern int OPS_enable_checkpointing;
extern double OPS_checkpointing_time;

/*******************************************************************************
* Core lib function prototypes
*******************************************************************************/

void ops_init(int argc, char **argv, int diags_level);
void ops_exit();

ops_dat ops_decl_dat_char(ops_block, int, int *, int *, int *, int *, char *,
                          int, char const *, char const *);
ops_dat ops_decl_dat_mpi_char(ops_block block, int size, int *dat_size,
                              int *base, int *d_m, int *d_p, char *data,
                              int type_size, char const *type,
                              char const *name);

ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc);
ops_arg ops_arg_dat_opt(ops_dat dat, int dim, ops_stencil stencil,
                        char const *type, ops_access acc, int flag);
ops_arg ops_arg_idx();
ops_arg ops_arg_reduce(ops_reduction handle, int dim, const char *type,
                       ops_access acc);
ops_arg ops_arg_reduce_core(ops_reduction handle, int dim, const char *type,
                            ops_access acc);

ops_reduction ops_decl_reduction_handle(int size, const char *type,
                                        const char *name);
ops_reduction ops_decl_reduction_handle_core(int size, const char *type,
                                             const char *name);
void ops_execute_reduction(ops_reduction handle);

ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc);
void ops_decl_const_char(int, char const *, int, char *, char const *);
void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr);

void ops_init_core(int argc, char **argv, int diags_level);

void ops_exit_core(void);

ops_block ops_decl_block(int dims, const char *name);

ops_dat ops_decl_dat_core(ops_block block, int data_size, int *block_size,
                          int *base, int *d_m, int *d_p, char *data,
                          int type_size, char const *type, char const *name);

ops_dat ops_decl_dat_temp_core(ops_block block, int data_size, int *block_size,
                               int *base, int *d_m, int *d_p, char *data,
                               int type_size, char const *type,
                               char const *name);

void ops_decl_const_core(int dim, char const *type, int typeSize, char *data,
                         char const *name);

ops_stencil ops_decl_stencil(int dims, int points, int *stencil,
                             char const *name);
ops_stencil ops_decl_strided_stencil(int dims, int points, int *sten,
                                     int *stride, char const *name);

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir);
ops_halo ops_decl_halo_core(ops_dat from, ops_dat to, int *iter_size,
                            int *from_base, int *to_base, int *from_dir,
                            int *to_dir);
ops_halo_group ops_decl_halo_group(int nhalos, ops_halo *halos);
void ops_halo_transfer(ops_halo_group group);

ops_arg ops_arg_dat_core(ops_dat dat, ops_stencil stencil, ops_access acc);
ops_arg ops_arg_gbl_core(char *data, int dim, int size, ops_access acc);

void ops_printf(const char *format, ...);
void ops_fprintf(FILE *stream, const char *format, ...);

void ops_diagnostic_output();
void ops_timing_output(FILE *stream);
void ops_timing_output_stdout();

void ops_timers(double *cpu, double *et);
void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name);
void ops_print_dat_to_txtfile_core(ops_dat dat, const char *file_name);

void ops_get_data(ops_dat dat);

void ops_timing_realloc(int, const char *);
void ops_timers_core(double *cpu, double *et);
float ops_compute_transfer(int dims, int *start, int *end, ops_arg *arg);

void ops_register_args(ops_arg *args, const char *name);
int ops_stencil_check_1d(int arg_idx, int idx0, int dim0);
int ops_stencil_check_2d(int arg_idx, int idx0, int idx1, int dim0, int dim1);
int ops_stencil_check_3d(int arg_idx, int idx0, int idx1, int idx2, int dim0,
                         int dim1);

int ops_stencil_check_1d_md(int arg_idx, int idx0, int mult_d, int d);
int ops_stencil_check_2d_md(int arg_idx, int idx0, int idx1, int dim0, int dim1,
                            int mult_d, int d);
int ops_stencil_check_3d_md(int arg_idx, int idx0, int idx1, int idx2, int dim0,
                            int dim1, int mult_d, int d);

/* check if these should be placed here */
void ops_set_dirtybit_host(
    ops_arg *args, int nargs); // data updated on host .. i.e. dirty on host
void ops_set_halo_dirtybit(ops_arg *arg);
void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range);
void ops_halo_exchanges(ops_arg *args, int nargs, int *range);
void ops_halo_exchanges_datlist(ops_dat *dats, int ndats, int *depths);

void ops_set_dirtybit_device(ops_arg *args, int nargs);
void ops_H_D_exchanges_host(ops_arg *args, int nargs);
void ops_H_D_exchanges_device(ops_arg *args, int nargs);
void ops_cpHostToDevice(void **data_d, void **data_h, int size);

int ops_is_root();

void ops_partition(const char *routine);

void ops_mpi_reduce_float(ops_arg *args, float *data);
void ops_mpi_reduce_double(ops_arg *args, double *data);
void ops_mpi_reduce_int(ops_arg *args, int *data);

void ops_compute_moment(double t, double *first, double *second);

void ops_dump3(ops_dat dat, const char *name);

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z);
void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z);

/* lazy execution */
void ops_enqueue_kernel(ops_kernel_descriptor *desc);
void ops_execute();
bool ops_get_abs_owned_range(ops_block block, int *range, int *start, int *end, int *disp);
int ops_get_proc();

#ifdef __cplusplus
}
#endif

#include "ops_checkpointing.h"
#include "ops_hdf5.h"

#endif /* __OP_LIB_CORE_H */
