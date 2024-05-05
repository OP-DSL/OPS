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

/** @file
 * @brief OPS internal types and function declarations
 * @author Istvan Reguly
 * @details this header contains type and function declarations
 * not needed by ops_lib_core.h, but needed by backends and generated code
 */

#ifndef __OPS_INTERNAL2
#define __OPS_INTERNAL2

#ifndef DOXYGEN_SHOULD_SKIP_THIS


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

#define ZERO_char 0;
#define INFINITY_char CHAR_MAX;

#define ZERO_short 0;
#define INFINITY_short SHRT_MAX;

#define ZERO_bool 0;

/*
 * alignment macro based on example on page 50 of CUDA Programming Guide version
 * 3.0
 * rounds up to nearest multiple of 16 bytes
 */

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ROUND_UP_64(bytes) (((bytes) + 63) & ~63)

// struct definition for a double linked list entry to hold an ops_dat
struct ops_dat_entry_core {
  ops_dat dat;
  TAILQ_ENTRY(ops_dat_entry_core)
  entries; /**< holds pointers to next and previous entries in the list*/
};

typedef struct ops_dat_entry_core ops_dat_entry;

typedef TAILQ_HEAD(, ops_dat_entry_core) Double_linked_list;

/** Storage for OPS block - OPS dataset associations */
struct ops_block_descriptor {
  ops_block_core *block;        /**< pointer to the block */
  Double_linked_list datasets;  /**< list of datasets associated with this block */
  int num_datasets;             /**< number of datasets */

};

/** Storage for OPS parallel loop statistics */
struct ops_kernel {
  char *name;      /**< name of kernel function */
  int count;       /**< number of times called */
  double time;     /**< total execution time */
  float transfer;  /**< bytes of data transfer (used) */
  double mpi_time; /**< time spent in MPI calls */
  //  double       mpi_gather;
  //  double       mpi_scatter;
  //  double       mpi_sendrecv;
  //  double       mpi_reduct;
};

/** Storage for OPS parallel handles */
struct ops_kernel_descriptor {
  char *name;           /**< name of kernel */
  int name_len;         /**< kernel name length */
  size_t hash;                /**< hash of loop */
  ops_arg *args;              /**< list of arguments to pass in */
  int nargs;                  /**< number of arguments */
  int index;                  /**< index of the loop */
  int dim;                    /**< number of dimensions */
  int isdevice;                 /**< flag to indicate if loop runs on device */
  int *range;                 /**< process local execution range */
  int *orig_range;            /**< original execution range */
  ops_block block;            /**< block to execute on */
  void (*func)(struct ops_kernel_descriptor *desc); /**< Function pointer to a wrapper to be called */
  void (*startup_func)(struct ops_kernel_descriptor *desc); /**< Function pointer to a wrapper to be called */
  void (*cleanup_func)(struct ops_kernel_descriptor *desc); /**< Function pointer to a wrapper to be called */

};

///
/// Struct duplicating information in MPI_Datatypes for (strided) halo access
///

typedef struct {
  int count;       ///< number of blocks
  int blocklength; ///< size of blocks
  int stride;      ///< stride between blocks
} ops_int_halo;


ops_reduction ops_decl_reduction_handle_core(OPS_instance *instance, int size, const char *type,
                                             const char *name);
void ops_execute_reduction(ops_reduction handle);

ops_arg ops_arg_reduce_core(ops_reduction handle, int dim, const char *type,
                            ops_access acc);

void ops_init_core(OPS_instance *instance, const int argc, const char *const argv[], const int diags_level);

void ops_exit_lazy(OPS_instance *instance);
void ops_exit_core(OPS_instance *instance);




ops_dat ops_decl_dat_core(ops_block block, int data_size, int *block_size,
                          int *base, int *d_m, int *d_p, int *stride, char *data,
                          int type_size, char const *type, char const *name);

ops_dat ops_decl_dat_temp_core(ops_block block, int data_size, int *block_size,
                               int *base, int *d_m, int *d_p, int *stride, char *data,
                               int type_size, char const *type,
                               char const *name);

ops_dat ops_decl_dat_mpi_char(ops_block block, int size, int *dat_size,
                              int *base, int *d_m, int *d_p, int *stride, char *data,
                              int type_size, char const *type,
                              char const *name);

void ops_decl_const_core(int dim, char const *type, int typeSize, char *data,
                         char const *name);

ops_halo ops_decl_halo_core(OPS_instance *instance, ops_dat from, ops_dat to, int *iter_size,
                            int *from_base, int *to_base, int *from_dir,
                            int *to_dir);



ops_arg ops_arg_dat_core(ops_dat dat, ops_stencil stencil, ops_access acc);
ops_arg ops_arg_gbl_core(char *data, int dim, int size, ops_access acc);

OPS_FTN_INTEROP
void ops_print_dat_to_txtfile_core(ops_dat dat, const char *file_name);

void ops_NaNcheck(ops_dat dat);
void ops_NaNcheck_core(ops_dat dat, char *buffer);

void ops_timing_realloc(OPS_instance *instance, int, const char *);
float ops_compute_transfer(int dims, int *start, int *end, ops_arg *arg);

void ops_register_args(OPS_instance *instance, ops_arg *args, const char *name);
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
OPS_FTN_INTEROP
void ops_set_dirtybit_host(
    ops_arg *args, int nargs); // data updated on host .. i.e. dirty on host
void ops_set_halo_dirtybit(ops_arg *arg);
OPS_FTN_INTEROP
void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range);
OPS_FTN_INTEROP
void ops_halo_exchanges(ops_arg *args, int nargs, int *range);
void ops_halo_exchanges_datlist(ops_dat *dats, int ndats, int *depths);

OPS_FTN_INTEROP
void ops_set_dirtybit_device(ops_arg *args, int nargs);
OPS_FTN_INTEROP
void ops_H_D_exchanges_host(ops_arg *args, int nargs);
OPS_FTN_INTEROP
void ops_H_D_exchanges_device(ops_arg *args, int nargs);
void ops_cpHostToDevice(OPS_instance *instance, void **data_d, void **data_h, size_t size);

void ops_init_arg_idx(int *arg_idx, int ndims, ops_arg *args, int nargs);

void ops_mpi_reduce_float(ops_arg *args, float *data);
void ops_mpi_reduce_double(ops_arg *args, double *data);
void ops_mpi_reduce_int(ops_arg *args, int *data);

void ops_dat_fetch_data_host(ops_dat dat, int part, char *data);
void ops_dat_fetch_data_slab_host(ops_dat dat, int part, char *data, int *range);

void ops_dat_set_data_host(ops_dat dat, int part, char *data);
void ops_dat_set_data_slab_host(ops_dat dat, int part, char *local_buf,
                                int *local_range);

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
OPS_FTN_INTEROP
void ops_execute(OPS_instance *instance=NULL);
bool ops_get_abs_owned_range(ops_block block, int *range, int *start, int *end, int *disp);
int compute_ranges(ops_arg* args, int nargs, ops_block block, int* range, int* start, int* end, int* arg_idx);
int ops_get_proc();
int ops_num_procs();
void ops_put_data(ops_dat dat);
OPS_FTN_INTEROP
void create_kerneldesc_and_enque(char const* kernel_name, ops_arg *args, int nargs, int index, int dim, int isdevice, int *range, ops_block block, void (*func)(struct ops_kernel_descriptor *desc));


/*******************************************************************************
* Random number generations
*******************************************************************************/
void ops_randomgen_init_host(unsigned int seed, int options, std::mt19937 &ops_rand_gen);
void ops_fill_random_uniform_host(ops_dat dat, std::mt19937 &ops_rand_gen);
void ops_fill_random_normal_host(ops_dat dat, std::mt19937 &ops_rand_gen);

/*******************************************************************************
* Memory allocation functions
*******************************************************************************/
void* ops_malloc (size_t size);
void* ops_realloc (void *ptr, size_t size);
void  ops_free (void *ptr);
void* ops_calloc (size_t num, size_t size);
void ops_init_zero(char *data, size_t bytes);
void ops_convert_layout(char *in, char *out, ops_block block, int size, int *dat_size, int *dat_size_orig, int type_size, int hybrid_layout);

//Includes for common device backends
void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags);
void ops_device_free(OPS_instance *instance, void** ptr);
void ops_device_freehost(OPS_instance *instance, void** ptr);
void ops_device_exit(OPS_instance *instance);
void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes);
void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes);
void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size);
void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size);
void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size);
void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size);
void ops_device_sync(OPS_instance *instance);
void ops_exit_device(OPS_instance *instance);
//
void reallocConstArrays(OPS_instance *instance, int consts_bytes);
void mvConstArraysToDevice(OPS_instance *instance, int consts_bytes);

void reallocConstArrays(OPS_instance *instance, int consts_bytes);
void mvConstArraysToDevice(OPS_instance *instance, int consts_bytes);

void _ops_init(OPS_instance *instance, const int argc, const char * const argv[], const int diags_level);
ops_block _ops_decl_block(OPS_instance *instance, int dims, const char * name);
ops_stencil _ops_decl_stencil(OPS_instance *instance, int dims, int points, int *stencil,
    char const *name);
ops_stencil _ops_decl_strided_stencil(OPS_instance *instance, int dims, int points, int *sten,
    int *stride, char const *name);
ops_stencil _ops_decl_restrict_stencil(OPS_instance *instance, int dims, int points, int *sten,
    int *stride, char const * name);
ops_stencil _ops_decl_prolong_stencil(OPS_instance *instance, int dims, int points, int *sten,
    int *stride, char const * name);
ops_halo _ops_decl_halo(OPS_instance *instance, ops_dat from, ops_dat to, int *iter_size, int *from_base,
    int *to_base, int *from_dir, int *to_dir);
ops_halo_group _ops_decl_halo_group(OPS_instance *instance, int nhalos, ops_halo *halos);
ops_reduction _ops_decl_reduction_handle(OPS_instance *instance, int size, const char *type,
                                        const char *name);
void ops_free_dat_core(ops_dat dat);
void _ops_free_dat(ops_dat dat);
void _ops_reset_power_counters(OPS_instance *instance);
void _ops_diagnostic_output(OPS_instance *instance);
void _ops_timing_output(OPS_instance *instance,std::ostream &stream);

void _ops_timing_output_stdout(OPS_instance *instance);
int _ops_is_root(OPS_instance *instance);
void _ops_partition(OPS_instance *instance, const char *routine);
void _ops_partition(OPS_instance *instance, const char *routine, std::map<std::string, void*>& opts);
void _ops_exit(OPS_instance *instance);

void ops_printf2(OPS_instance *instance, const char *format, ...);
void printf2(OPS_instance *instance, const char *format, ...);

void ops_fprintf2(std::ostream &, const char *format, ...);
void fprintf2(std::ostream &, const char *format, ...);

ops_dat ops_dat_alloc_core(ops_block block);
int ops_dat_copy_metadata_core(ops_dat target, ops_dat orig_dat);
ops_kernel_descriptor * ops_dat_deep_copy_core(ops_dat target, ops_dat orig_dat, int *range);
void ops_internal_copy_seq(ops_kernel_descriptor *desc);
void ops_internal_copy_device(ops_kernel_descriptor *desc);

OPS_FTN_INTEROP
void ops_upload_gbls(ops_arg* args, int nargs);

//
// wrapper functions to handle MPI global reductions
//

inline void ops_mpi_reduce(ops_arg *args, float *data) {
  ops_mpi_reduce_float(args, data);
}

inline void ops_mpi_reduce(ops_arg *args, double *data) {
  ops_mpi_reduce_double(args, data);
}

inline void ops_mpi_reduce(ops_arg *args, int *data) {
  ops_mpi_reduce_int(args, data);
}

class OPS_instance;

#include "ops_checkpointing.h"
#include "ops_hdf5.h"
#include "ops_tridiag.h"
#include "ops_instance.h"

#ifdef OPS_CPP_API
template <class T> void ops_reduction_core::get_result(T *ptr) {
  if (type_error(ptr, this->type)) {
    OPSException ex(OPS_INVALID_ARGUMENT);
    ex << "Error: incorrect type specified for constant " << this->name << " in ops_reduction_result";
    throw ex;
  }
  ops_reduction_result_char(this, sizeof(T), (char *)ptr);
}
template <class T>
ops_dat ops_block_core::decl_dat(int data_size, int *block_size, int *base,
    int *d_m, int *d_p, int *stride, T *data, char const *type,
    char const *name) {

  if (type_error(data, type)) {
    OPSException ex(OPS_INVALID_ARGUMENT);
    ex << "Error: incorrect type specified for dataset " << name;
    throw ex;
  }

  return ops_decl_dat_char(this, data_size, block_size, base, d_m, d_p,
      stride, (char *)data, sizeof(T), type, name);
}
template <class T>
ops_dat ops_block_core::decl_dat(int data_size, int *block_size, int *base,
    int *d_m, int *d_p, T *data, char const *type,
    char const *name) {

  if (type_error(data, type)) {
    OPSException ex(OPS_INVALID_ARGUMENT);
    ex << "Error: incorrect type specified for dataset " << name;
    throw ex;
  }

  int stride[OPS_MAX_DIM];
  for (int i = 0; i < OPS_MAX_DIM; i++) stride[i] = 1;
  return ops_decl_dat_char(this, data_size, block_size, base, d_m, d_p,
      stride, (char *)data, sizeof(T), type, name);
}
#endif

#if !defined(OPS_CPP_API) || defined(OPS_INTERNAL_API)
template <typename T>
void ops_decl_const2(char const *name, int dim, char const *type, T *data) {
  ops_decl_const_char(OPS_instance::getOPSInstance(), dim, type, sizeof(T),
                      (char *)data, name);
}
#endif
#endif

#endif
