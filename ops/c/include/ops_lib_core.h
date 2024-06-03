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
  * @brief OPS core library function declarations
  * @author Gihan Mudalige
  * @details function declarations header file for the core library functions
  * utilized by all OPS backends
  */

#ifndef __OPS_LIB_CORE_H
#define __OPS_LIB_CORE_H

// Needed for size_t
#include <string>
#include <map>
#include <cstring>
#ifdef __unix__
#include <string.h>
#include "queue.h" //contains double linked list implementation
#include <strings.h>
#endif
#include <stdint.h>
#include <complex>
#include <random>

/** default byte alignment for allocations made by OPS */
#ifndef OPS_ALIGNMENT
#define OPS_ALIGNMENT 64
#endif

/** Maximum internal halo depth over MPI -
 * extend if you get a runtime error about this being too small
 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 15

/**
 * maximum number of spatial dimensions supported.
 * Can reduce to save on size of metadata
 */
#define OPS_MAX_DIM 5

#ifndef __unix__
#define __restrict__ __restrict
#endif

/*
* essential typedefs
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

typedef std::complex<double> complexd;
typedef std::complex<float> complexf;

#if (defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) && !(defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
#include <hip/hip_fp16.h>
#elif !(defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) && (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
#include <cuda_fp16.h>
#elif defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_fp16.h>
typedef __half half;
//#elif defined(__SYCL_DEVICE_ONLY__)
#elif defined(__INTEL_SYCL__)
#include <CL/sycl.hpp>
#elif defined(__STDCPP_FLOAT16_T__) || defined(FLT16_MIN)
typedef _Float16 half;
#else
typedef uint16_t half;
#endif

/*
 * * zero constants
 * */

#ifndef ZERO_INF
#define ZERO_INF
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

#define ZERO_complexd complexd(0,0)
#define INFINITY_complexd complexd(DBL_MAX,DBL_MAX)

#define ZERO_complexf complexf(0,0)
#define INFINITY_complexf complexf(FLT_MAX,FLT_MAX)
#endif

/**
 * type for memory space flags - 1 for host, 2 for device
 */
typedef int ops_memspace;
#define OPS_HOST 1
#define OPS_DEVICE 2

typedef int ops_access;   // holds OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN,
                          // OP_MAX
typedef int ops_arg_type; // holds OP_ARG_GBL, OP_ARG_DAT


/*
* Forward declarations of structures
*/
class OPS_instance;
class ops_block_core;
class ops_dat_core;
struct ops_reduction_core;
struct ops_arg;


/** Storage for OPS blocks */
class ops_block_core {
public:
  int index;        /**< index */
  int dims;         /**< dimension of block, 2D,3D .. etc*/
  char const *name; /**< name of block */
  OPS_instance *instance; /**<< pointer the the OPS_instance*/

#ifdef OPS_CPP_API
/**
 * This routine defines a dataset.
 *
 * The @p size allows to declare different sized data arrays on a given block.
 * @p d_m and @p d_p are depth of the "block halos" that are used to indicate
 * the offset from the edge of a block (in both the negative and positive
 * directions of each dimension).
 *
 * @tparam T
 * @param data_size   dimension of dataset (number of items per grid element)
 * @param block_size  size in each dimension of the block
 * @param base        base indices in each dimension of the block
 * @param d_m         padding from the face in the negative direction for each
 *                    dimension (used for block halo)
 * @param d_p         padding from the face in the positive direction for each
 *                    dimension (used for block halo)
 * @param stride
 * @param data        input data of type @p T
 * @param type        the name of type used for output diagnostics
 *                    (e.g. "double", "float")
 * @param name        a name used for output diagnostics
 * @return
 */
  template <class T>
  ops_dat_core* decl_dat(int data_size, int *block_size, int *base,
        int *d_m, int *d_p, int *stride, T *data, char const *type,
        char const *name);
/**
 * This routine defines a dataset.
 *
 * The @p size allows to declare different sized data arrays on a given block.
 * @p d_m and @p d_p are depth of the "block halos" that are used to indicate
 * the offset from the edge of a block (in both the negative and positive
 * directions of each dimension).
 *
 * @tparam T
 * @param data_size   dimension of dataset (number of items per grid element)
 * @param block_size  size in each dimension of the block
 * @param base        base indices in each dimension of the block
 * @param d_m         padding from the face in the negative direction for each
 *                    dimension (used for block halo)
 * @param d_p         padding from the face in the positive direction for each
 *                    dimension (used for block halo)
 * @param data        input data of type @p T
 * @param type        the name of type used for output diagnostics
 *                    (e.g. "double", "float")
 * @param name        a name used for output diagnostics
 * @return
 */
  template <class T>
  ops_dat_core* decl_dat(int data_size, int *block_size, int *base,
        int *d_m, int *d_p, T *data, char const *type,
        char const *name);
#endif
};

typedef ops_block_core *ops_block;

/** Storage for OPS reduction handles */
struct ops_reduction_core {
  char *data;       /**< The data */
  int size;         /**< size of data in bytes */
  int initialized;  /**< flag indicating whether data has been initialized */
  int index;        /**< unique identifier */
  ops_access acc;   /**< Type of reduction it was used for last time */
  char *type; /**< Type */
  char *name; /**< Name */
  OPS_instance *instance;
#ifdef OPS_CPP_API

/**
 * This routine returns the reduced value held by a reduction handle. During lazy execution,
 * this will trigger the execution of all preceding queued operations
 *
 * @tparam T
 * @param handle  the ::ops_reduction handle
 * @param ptr     a pointer to write the results to, memory size has to match
 *                the declared
 */
  template <class T> void get_result(T *ptr);
#endif
};

typedef ops_reduction_core *ops_reduction;


class ops_stencil_core;

/** Storage for OPS datasets */
class ops_dat_core {
  public:
  int index;             /**< index */
  ops_block block;       /**< block on which data is defined */
  int dim;               /**< number of elements per grid point */
  int type_size;         /**< bytes per primitive = elem_size/dim */
  int elem_size;         /**< number of bytes per grid point */
  int size[OPS_MAX_DIM]; /**< size of the array in each block dimension --
                              including halo */
  int base[OPS_MAX_DIM]; /**< base offset to 0,0,... from the start of each
                              dimension */
  int d_m[OPS_MAX_DIM];  /**< halo depth in each dimension, negative direction
                          *   (at 0 end) */
  int d_p[OPS_MAX_DIM];  /**< halo depth in each dimension, positive direction
                          *   (at size end) */
  int x_pad;             /**< padding in x-dimension for allocating aligned memory */
  char *data;            /**< data on host */
  char *data_d;          /**< data on device */
  char const *name;      /**< name of dataset */
  char const *type;      /**< datatype */
  int dirty_hd;          /**< flag to indicate dirty status on host and device*/
  int locked_hd;         /**< flag to indicate that the user has obtained a raw data pointer,
                          *   and whether the raw pointer is held on the host or device */
  int user_managed;      /**< indicates whether the user is managing memory */
  int is_hdf5;           /**< indicates whether the data is to read from an
                          *   hdf5 file */
  char const *hdf5_file; /**< name of hdf5 file from which this dataset was
                          *   read */
  int e_dat;             /**< flag to indicate if this is an edge dat */
  size_t mem;              /**< memory in bytes allocated to this dat (under MPI,
                          *   this will be memory held on a single MPI proc) */
  size_t base_offset;      /**< computed quantity, giving offset in bytes to the
                          *   base index */
  int stride[OPS_MAX_DIM];/**< stride[*] > 1 if this dat is a coarse dat under
                           *   multi-grid*/


  // Default constructor zeros out all data in the struct
  ops_dat_core() { memset(this, 0, sizeof(ops_dat_core)); }
  ~ops_dat_core();



#ifdef OPS_CPP_API
/**
 * Write the details of an ::ops_block to a named text file.
 *
 * When used under an MPI parallelization each MPI process will write its own
 * data set separately to the text file. As such it does not use MPI I/O.
 * The data can be viewed using a simple text editor
 * @param file_name  text file to write to
 */
  void print_to_txtfile(const char *file_name);
/**
 * Makes sure OPS has downloaded data from the device
 */
  void get_data();
/**
 * This routine returns the number of chunks of the given dataset held by the
 * current process.
 *
 * @return
 */
  int  get_local_npartitions();
/**
 * This routine returns the MPI displacement and size of a given chunk of the
 * given dataset on the current process.
 *
 * @param part  the chunk index (has to be 0)
 * @param disp  an array populated with the displacement of the chunk within the
 *              "global" distributed array
 * @param size  an array populated with the spatial extents
 */
  void get_extents(int part, int *disp, int *size);
/**
 * This routine returns the MPI displacement and size of the intersection of a
 * hyper-slab with the given dataset on the current process.
 *
 * @param part  the chunk index (has to be 0)
 * @param disp  an array populated with the displacement of the chunk within the
 *              "global" distributed array
 * @param size  an array populated with the spatial extents
 * @param slab  index ranges of the hyperslab. Ordering: {begin_0, end_0, begin_1, end_1,...}
 * @return the size in bytes of the intersection between the dataset and the slab on this process
 */
  size_t get_slab_extents(int part, int *disp, int *size, int *slab);
/**
 * This routine returns array shape metadata corresponding to the ops_dat.
 * Any of the arguments may be NULL.
 *
 * @param part     the chunk index (has to be 0)
 * @param disp     an array populated with the displacement of the chunk within the
 *                 "global" distributed array
 * @param size     an array populated with the spatial extents
 * @param stride   an array populated strides in spatial dimensions needed for
 *                 column-major indexing
 * @param d_m      an array populated with padding on the left in each dimension
 *                 note that these are negative values
 * @param d_p      an array populated with padding on the right in each dimension
 */
  void get_raw_metadata(int part, int *disp, int *size, int *stride, int *d_m, int *d_p);
/**
 * This routine returns a pointer to the internally stored data, with MPI halo
 * regions automatically updated as required by the supplied stencil.
 * The strides required to index into the dataset are also given.
 * You may have to call ops_execute before calling this to make sure all
 * computations have finished.
 *
 * @param part     the chunk index (has to be 0)
 * @param stencil  a stencil used to determine required MPI halo exchange depths
 * @param memspace when set to OPS_HOST or OPS_DEVICE, returns a pointer to data in
 *                 that memory space, otherwise must be set to 0, and returns
 *                 whether data is in the host or on the device
 *
 * @return
 */
  char* get_raw_pointer(int part, ops_stencil_core *stencil, ops_memspace *memspace);
/**
 * Indicates to OPS that a dataset previously accessed with
 * get_raw_pointer() is released by the user, and also tells OPS how it
 * was accessed.
 * A single call to release_raw_data() releases all pointers obtained by previous calls to
 * get_raw_pointer() calls on the same dat. Calls to release_raw_data() must be separated by calls to
 * get_raw_pointer(), i.e. it is illegal to release raw data access multiple times without first
 * starting raw data access. The data that the user wishes keep is in the memory
 * space (buffer) indicated by the LAST call to get_raw_pointer().  Data in any other memory spaces
 * is discarded.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0
 * @param acc   the kind of access that was used by the user
 *              (::OPS_READ if it was read only,
 *              ::OPS_WRITE if it was overwritten,
 *              ::OPS_RW if it was read and written)
 */
  void release_raw_data(int part, ops_access acc);
/**
 * Indicates to OPS that a dataset previously accessed with
 * ops_dat_get_raw_pointer() is released by the user, and also tells OPS how it
 * was accessed, and where it was accessed.
 * A single call to release_raw_data() releases all pointers obtained by previous calls to
 * get_raw_pointer() calls on the same dat. Calls to release_raw_data() must be separated by calls to
 * get_raw_pointer(), i.e. it is illegal to release raw data access multiple times without first
 * starting raw data access.
 * The *memspace argument tells OPS in which memory space the data is that the user wants to keep.
 * Data in all other memory spaces will be discarded.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0
 * @param acc   the kind of access that was used by the user
 *              (::OPS_READ if it was read only,
 *              ::OPS_WRITE if it was overwritten,
 *              ::OPS_RW if it was read and written)
 * @param memspace has to be set to either OPS_HOST or OPS_DEVICE to indicate where the data was modified
 */
  void release_raw_data(int part, ops_access acc, ops_memspace *memspace);
/**
 * This routine copies the data held by OPS to the user-specified
 * memory location, which needs to be at least as large as indicated
 * by the `sizes` parameter of ops_dat_get_extents().

 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be filled by OPS
 * @param memspace the memory space where the data pointer is
 */
  void fetch_data(int part, char *data, ops_memspace memspace = OPS_HOST);
/**
 * This routine copies a hyperslab of the data held by OPS to the user-specified
 * memory location, which needs to be at least as large as the product of ranges
 * defined by the `range` parameter

 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be filled by OPS
 * @param range index ranges of the hyperslab. Ordering: {begin_0, end_0, begin_1, end_1,...}
 * @param memspace the memory space where the data pointer is
 */
  void fetch_data_slab(int part, char *data, int *range, ops_memspace memspace = OPS_HOST);
/**
 * This routine copies the data given by the user to the internal data structure
 * used by OPS.
 *
 * User data needs to be laid out in column-major order and strided as indicated
 * by the `sizes` parameter of ops_dat_get_extents().
 *
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to CPU memory which should be copied to OPS
 * @param memspace the memory space where the data pointer is
 */
  void set_data(int part, char *data, ops_memspace memspace = OPS_HOST);
/**
 * This routine copies to a hyperslab of the data held by OPS from the user-specified
 * memory location, which needs to be at least as large as the product of ranges
 * defined by the `range` parameter

 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be read by OPS
 * @param range index ranges of the hyperslab. Ordering: {begin_0, end_0, begin_1, end_1,...}
 * @param memspace the memory space where the data pointer is
 */
  void set_data_slab(int part, char *data, int *range, ops_memspace memspace = OPS_HOST);
/**
 * This routine returns the number of chunks of the given dataset held by all
 * processes.
 *
 * @return
 */
  int get_global_npartitions();

#endif
};

typedef ops_dat_core *ops_dat;

/** Storage for OPS stencils */
class ops_stencil_core {
public:
  int index;        /**< index */
  int dims;         /**< dimensionality of the stencil */
  int points;       /**< number of stencil elements */
  char const *name; /**< name of pointer */
  int *stencil;     /**< elements in the stencil */
  int *stride;      /**< stride of the stencil */
  int *mgrid_stride;/**< stride of the stencil under multi_grid */
  int type;         /**< 0 for regular, 1 for prolongate, 2 for restrict */
};

typedef ops_stencil_core *ops_stencil;

/** Storage for OPS parallel loop arguments */
struct ops_arg {
  ops_dat dat;          /**< dataset */
  ops_stencil stencil;  /**< the stencil */
  int dim;              /**< dimension of data */
  int elem_size;        /**< #of bytes per primitive element */
  char *data;           /**< data on host */
  char *data_d;         /**< data on device (for CUDA)*/
  ops_access acc;       /**< access type */
  ops_arg_type argtype; /**< arg type */
  int opt;              /**< flag to indicate whether this is an optional arg,
                         *   0 - optional, 1 - not optional */
};

/** Storage for OPS halos */
struct ops_halo_core {
  ops_dat from;                   /**< dataset from which the halo is read */
  ops_dat to;                     /**< dataset to which the halo is written */
  int iter_size[OPS_MAX_DIM];     /**< size of halo region */
  int from_base[OPS_MAX_DIM];     /**< start position to copy from */
  int to_base[OPS_MAX_DIM];       /**< start position to copy to */
  int from_dir[OPS_MAX_DIM];      /**< copy from orientation */
  int to_dir[OPS_MAX_DIM];        /**< size to orientation */
  int index;                      /**< index of halo */
};

typedef ops_halo_core *ops_halo;

/** Storage for OPS halo groups */
typedef struct {
  int nhalos;                     /**< number of halos */
  ops_halo *halos;                /**< list of halos */
  int index;                      /**< index of halo group */
  OPS_instance *instance;
#ifdef OPS_CPP_API
/**
 * This routine exchanges all halos in a halo group and will block execution
 * of subsequent computations that depend on the exchanged data.
 *
 */
  void halo_transfer();
#endif
} ops_halo_group_core;

typedef ops_halo_group_core *ops_halo_group;

/*
* min / max definitions
*/

#ifndef MIN
#define MIN(a, b) ((a < b) ? (a) : (b))
#endif
#ifndef MIN3
#define MIN3(a, b, c) MIN(a,MIN(b,c))
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? (a) : (b))
#endif
#ifndef MAX3
#define MAX3(a, b, c) MAX(a,MAX(b,c))
#endif
#ifndef SIGN
#define SIGN(a, b) ((b < 0.0) ? (a * (-1)) : (a))
#endif


#include <ops_internal1.h>

/*******************************************************************************
* C API
*******************************************************************************/

#if !defined(OPS_CPP_API) || defined(OPS_INTERNAL_API)
/**
 * This routine must be called before all other OPS routines.
 *
 * @param argc         the usual command line argument
 * @param argv         the usual command line argument
 * @param diags_level  an integer which defines the level of debugging
 *                     diagnostics and reporting to be performed.
 *                     Can be overridden with the -OPS_DIAGS= runtime argument
 *
 * Currently, higher `diags_level`s perform the following checks:
 *
 * | diags level | checks                                                      |
 * | ----------- | ----------------------------------------------------------- |
 * |  = 1        | no diagnostics, default to achieve best runtime performance.|
 * |  > 1        | print block decomposition and ops par loop timing breakdown.|
 * |  > 4        | print intra-block halo buffer allocation feedback (for OPS internal development only)                 |
 * |  > 5        | check if intra-block halo MPI sends depth match MPI receives depth (for OPS internal development only)|
 *
 */
OPS_FTN_INTEROP
void ops_init(const int argc, const char *const argv[], const int diags_level);

/**
 * This routine must be called last to cleanly terminate the OPS computation.
 */
OPS_FTN_INTEROP
void ops_exit();

OPS_FTN_INTEROP
void ops_set_soa(const int soa_val);

/**
 * This routine defines a structured grid block.
 *
 * @param dims  dimension of the block
 * @param name  a name used for output diagnostics
 * @return
 */
OPS_FTN_INTEROP
ops_block ops_decl_block(int dims, const char *name);

/**
 * This routine defines a dataset.
 *
 * The @p size allows to declare different sized data arrays on a given block.
 * @p d_m and @p d_p are depth of the "block halos" that are used to indicate
 * the offset from the edge of a block (in both the negative and positive
 * directions of each dimension).
 *
 * @tparam T
 * @param block       structured block
 * @param data_size   dimension of dataset (number of items per grid element)
 * @param block_size  size in each dimension of the block
 * @param base        base indices in each dimension of the block
 * @param d_m         padding from the face in the negative direction for each
 *                    dimension (used for block halo)
 * @param d_p         padding from the face in the positive direction for each
 *                    dimension (used for block halo)
 * @param stride
 * @param data        input data of type @p T
 * @param type        the name of type used for output diagnostics
 *                    (e.g. "double", "float")
 * @param name        a name used for output diagnostics
 * @return
 */
template <class T>
ops_dat ops_decl_dat(ops_block block, int data_size, int *block_size, int *base,
                     int *d_m, int *d_p, int *stride, T *data, char const *type,
                     char const *name) {

  if (type_error(data, type)) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: incorrect type specified for dataset " << name;
    throw ex;
  }

  return ops_decl_dat_char(block, data_size, block_size, base, d_m, d_p,
                           stride, (char *)data, sizeof(T), type, name);
}

/**
 * This routine defines a dataset.
 *
 * The @p size allows to declare different sized data arrays on a given block.
 * @p d_m and @p d_p are depth of the "block halos" that are used to indicate
 * the offset from the edge of a block (in both the negative and positive
 * directions of each dimension).
 *
 * @tparam T
 * @param block       structured block
 * @param data_size   dimension of dataset (number of items per grid element)
 * @param block_size  size in each dimension of the block
 * @param base        base indices in each dimension of the block
 * @param d_m         padding from the face in the negative direction for each
 *                    dimension (used for block halo)
 * @param d_p         padding from the face in the positive direction for each
 *                    dimension (used for block halo)
 * @param data        input data of type @p T
 * @param type        the name of type used for output diagnostics
 *                    (e.g. "double", "float")
 * @param name        a name used for output diagnostics
 * @return
 */
template <class T>
ops_dat ops_decl_dat(ops_block block, int data_size, int *block_size, int *base,
                     int *d_m, int *d_p, T *data, char const *type,
                     char const *name) {

  int stride[OPS_MAX_DIM];
  for (int i = 0; i < OPS_MAX_DIM; i++) stride[i] = 1;
  return ops_decl_dat_char(block, data_size, block_size, base, d_m, d_p,
                           stride, (char *)data, sizeof(T), type, name);
}



/**
 * Deallocates an OPS dataset
 * @param dat     dataset to deallocate
 */
OPS_FTN_INTEROP
void ops_free_dat(ops_dat dat);

/**
 * Makes a copy of a dataset
 * @param orig_dat the dataset to be copied
 * @return the copy
 */
ops_dat ops_dat_copy(ops_dat orig_dat);

/**
 * Makes a deep copy of the data held in source
 * @param source the dataset to be copied
 * @param target the target of the copy
 */
void ops_dat_deep_copy(ops_dat target, ops_dat orig_dat);
#endif

/**
 * Passes an accessor to the value(s) at the current grid point to the user kernel.
 *
 * The ACC<type>& reference and its () operator has to be used for accessing data
 *
 * @param dat      dataset
 * @param dim
 * @param stencil  stencil for accessing data
 * @param type     string representing the type of data held in dataset
 * @param acc      access type
 * @return
 */
OPS_FTN_INTEROP
ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc);

/**
 * Passes an accessor to the value(s) at the current grid point to the user kernel if flag is true
 *
 * The ACC<type>& reference and its () operator has to be used for accessing data
 *
 * @param dat      dataset
 * @param dim
 * @param stencil  stencil for accessing data
 * @param type     string representing the type of data held in dataset
 * @param acc      access type
 * @param flag     indicates whether the optional argument is enabled (non-0) or not (0)
 * @return
 */
OPS_FTN_INTEROP
ops_arg ops_arg_dat_opt(ops_dat dat, int dim, ops_stencil stencil,
                        char const *type, ops_access acc, int flag);
/**
 * Returns an array of integers (in the user kernel) that have the index of
 * the current grid point, i.e. `idx[0]` is the index in x, `idx[1]` is
 * the index in y, etc.
 * This is a globally consistent index, so even if the block is distributed
 * across different MPI partitions, it gives you the same indexes.
 * Generally used to generate initial geometry.
 *
 * @return
 */
OPS_FTN_INTEROP
ops_arg ops_arg_idx();

/**
 * Passes a pointer to a variable that needs to be incremented
 * (or swapped for min/max reduction) by the user kernel.
 *
 * @param handle  an ::ops_reduction handle
 * @param dim     array dimension (according to @p type)
 * @param type    string representing the type of data held in data
 * @param acc     access type
 * @return
 */
OPS_FTN_INTEROP
ops_arg ops_arg_reduce(ops_reduction handle, int dim, const char *type,
                       ops_access acc);

/**
 * Passes a scalar or small array that is invariant of the iteration space.
 *
 * (not to be confused with ::ops_decl_const, which facilitates
 * global scope variables).
 *
 * @tparam T
 * @param data  data array
 * @param dim   array dimension
 * @param type  string representing the type of data held in data
 * @param acc   access type
 * @return
 */
template <class T>
ops_arg ops_arg_gbl(T *data, int dim, char const *type, ops_access acc) {
    (void)type;
  return ops_arg_gbl_char((char *)data, dim, sizeof(T), acc);
}



#if !defined(OPS_CPP_API) || defined(OPS_INTERNAL_API)
/**
 * This routine defines a reduction handle to be used in a parallel loop.
 *
 * @param size  size of data in bytes
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param name  name of the dat used for output diagnostics
 * @return
 */
OPS_FTN_INTEROP
ops_reduction ops_decl_reduction_handle(int size, const char *type,
                                        const char *name);


/**
 * This routine defines a stencil.
 *
 * @param dims     dimension of loop iteration
 * @param points   number of points in the stencil
 * @param stencil  stencil for accessing data
 * @param name     string representing the name of the stencil
 * @return
 */
OPS_FTN_INTEROP
ops_stencil ops_decl_stencil(int dims, int points, int *stencil,
                             char const *name);

/**
 * This routine defines a strided stencil.
 *
 * The semantics for the index of data to be accessed, for stencil point `p`,
 * in dimension `m` are defined as:
 * ```
 *    stride[m]*loop index[m] + stencil[p*dims+m]
 * ```
 * where `loop_index[m]` is the iteration index (within the user-defined
 * iteration space) in the different dimensions.
 * If, for one or more dimensions, both `stride[m]` and `stencil[p*dims+m]`
 * are zero, then one of the following must be true;
 *   - the dataset being referenced has size 1 for these dimensions
 *   - these dimensions are to be omitted and so the dataset has dimension
 *     equal to the number of remaining dimensions.
 *
 * @param dims     dimension of loop iteration
 * @param points   number of points in the stencil
 * @param stencil  stencil for accessing data
 * @param stride   stride for accessing data
 * @param name     string representing the name of the stencil
 * @return
 */
OPS_FTN_INTEROP
ops_stencil ops_decl_strided_stencil(int dims, int points, int *sten,
                                     int *stride, char const *name);
OPS_FTN_INTEROP
ops_stencil ops_decl_restrict_stencil( int dims, int points, int *sten,
                                       int *stride, char const * name);
OPS_FTN_INTEROP
ops_stencil ops_decl_prolong_stencil( int dims, int points, int *sten,
                                      int *stride, char const * name);

/**
 * This routine defines a halo relationship between two datasets defined on two
 * different blocks.
 *
 * A @p from_dir [1,2] and a @p to_dir [2,1] means that x in the first block
 * goes to y in the second block, and y in first block goes to x in second
 * block.
 * A negative sign indicates that the axis is flipped.
 *
 * (Simple example: a transfer from (1:2,0:99,0:99) to (-1:0,0:99,0:99) would
 * use @p iter_size = [2,100,100], @p from_base = [1,0,0],
 * @p to_base = [-1,0,0], @p from_dir = [0,1,2], @p to_dir = [0,1,2].
 * In more complex cases this allows for transfers between blocks with
 * different orientations.)
 *
 * @param from       origin dataset
 * @param to         destination dataset
 * @param iter_size  defines an iteration size
 *                   (number of indices to iterate over in each direction)
 * @param from_base  indices of starting point in @p from dataset
 * @param to_base    indices of starting point in @p to dataset
 * @param from_dir   direction of incrementing for @p from for each dimension
 *                   of @p iter_size
 * @param to_dir     direction of incrementing for @p to for each dimension
 *                   of @p iter_size
 * @return
 */
ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir);


/**
 * This routine defines a collection of halos.
 * Semantically, when an exchange is triggered for all halos
 * in a group, there is no order defined in which they are carried out.
 *
 * @param nhalos  number of halos in @p halos
 * @param halos   array of halos
 * @return
 */
OPS_FTN_INTEROP
ops_halo_group ops_decl_halo_group(int nhalos, ops_halo *halos);

/**
 * This routine exchanges all halos in a halo group and will block execution
 * of subsequent computations that depend on the exchanged data.
 *
 * @param group  the halo group
 */
OPS_FTN_INTEROP
void ops_halo_transfer(ops_halo_group group);

/**
 * This routine returns the reduced value held by a reduction handle. During lazy execution,
 * this will trigger the execution of all preceding queued operations
 *
 * @tparam T
 * @param handle  the ::ops_reduction handle
 * @param ptr     a pointer to write the results to, memory size has to match
 *                the declared
 */
template <class T> void ops_reduction_result(ops_reduction handle, T *ptr) {
  if (type_error(ptr, handle->type)) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: incorrect type specified for constant " << handle->name << " in ops_reduction_result";
    throw ex;
  }
  if (!handle->initialized) {
    OPSException ex(OPS_INVALID_ARGUMENT);
    ex << "Error: ops_reduction_result called for " << handle->name << " but the handle was not previously used in a reduction since the last ops_reduction_result call.";
    throw ex;
  }
  ops_reduction_result_char(handle, sizeof(T), (char *)ptr);
}

/**
 * This routine updates/changes the value of a constant.
 *
 * @tparam T
 * @param name  a name used to identify the constant
 * @param dim   dimension of dataset (number of items per element)
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param data  pointer to new values for constant of type @p T
 */
template <class T>
void ops_update_const(char const *name, int dim, char const *type, T *data) {
  if (type_error(data, type)) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: incorrect type specified for constant " << name << " in ops_update_const";
    throw ex;
  }
  ops_decl_const2(name, dim, type, data);
}

/**
 * This routine defines a global constant: a variable in global scope.
 *
 * Global constants need to be declared upfront so that they can be correctly
 * handled for different parallelizations. For e.g. CUDA on GPUs.
 * Once defined they remain unchanged throughout the program, unless changed
 * by a call to ops_update_const().
 * @tparam T
 * @param name  a name used to identify the constant
 * @param dim   dimension of dataset (number of items per element)
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param data  pointer to input data of type @p T
 */
template <class T>
void ops_decl_const(char const *name, int dim, char const *type, T *data) {
  if (type_error(data, type)) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: incorrect type specified for constant " << name << " in ops_decl_const";
    throw ex;
  }
  ops_decl_const2(name, dim, type, data);
}

#endif

/**
 * This routine simply prints a variable number of arguments on the root process;
 * it is created in place of the standard C printf() function which would
 * print the same on each MPI process.
 *
 * @param format
 * @param ...
 */
OPS_FTN_INTEROP
void ops_printf(const char *format, ...);

/**
 * This routine simply prints a variable number of arguments on the root process;
 * it is created is in place of the standard C printf function which would
 * print the same on each MPI process.
 *
 * @param stream
 * @param format
 * @param ...
 */
OPS_FTN_INTEROP
void ops_fprintf(FILE *stream, const char *format, ...);

#if !defined(OPS_CPP_API) || defined(OPS_INTERNAL_API)
/**
 * This routine prints out various useful bits of diagnostic info about sets,
 * mappings and datasets.
 *
 * Usually used right after an @p ops partition() call to print out the details
 * of the decomposition.
 */
OPS_FTN_INTEROP
void ops_diagnostic_output();

/**
 * Print OPS performance performance details to output stream.
 *
 * @param stream  output stream, use std::cout to print to standard out
 */
void ops_timing_output(std::ostream &stream);
OPS_FTN_INTEROP
void ops_timing_output_stdout();
#endif

/**
 * gettimeofday() based timer to start/end timing blocks of code.
 *
 *
 * @param cpu  variable to hold the CPU time at the time of invocation
 * @param et   variable to hold the elapsed time at the time of invocation
 */
OPS_FTN_INTEROP
void ops_timers_core(double *cpu, double *et);
OPS_FTN_INTEROP
void ops_timers(double *cpu, double *et);

#if !defined(OPS_CPP_API) || defined(OPS_INTERNAL_API)
/**
 * Write the details of an ::ops_block to a named text file.
 *
 * When used under an MPI parallelization each MPI process will write its own
 * data set separately to the text file. As such it does not use MPI I/O.
 * The data can be viewed using a simple text editor

 * @param dat        ::ops_dat to to be written
 * @param file_name  text file to write to
 */
OPS_FTN_INTEROP
void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name);

/**
 * Makes sure OPS has downloaded data from the device
 */
void ops_get_data(ops_dat dat);

/**
 * Returns one one the root MPI process
 */
OPS_FTN_INTEROP
int ops_is_root();

/**
 * Triggers a multi-block partitioning across a distributed memory set of
 * processes.
 * (links to a dummy function for single node parallelizations).
 * This routine should only be called after all the ops_decl_block() and
 * ops_decl_dat() statements have been declared. Must be called before any
 * calling any parallel loops
 *
 * @param routine  string describing the partitioning method.
 *                 Currently this string is not used internally, but is simply
 *                 a place-holder to indicate different partitioning methods
 *                 in the future.
 */
OPS_FTN_INTEROP
void ops_partition(const char *routine);

void ops_partition_opts(const char *routine, std::map<std::string, void*>& opts);
/*******************************************************************************
* External access support
*******************************************************************************/

/**
 * This routine returns the number of chunks of the given dataset held by the
 * current process.
 *
 * @param dat  the dataset
 * @return
 */
OPS_FTN_INTEROP
int ops_dat_get_local_npartitions(ops_dat dat);

/**
 * This routine returns the MPI displacement and size of a given chunk of the
 * given dataset on the current process.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param disp  an array populated with the displacement of the chunk within the
 *              "global" distributed array
 * @param size  an array populated with the spatial extents
 */
OPS_FTN_INTEROP
void ops_dat_get_extents(ops_dat dat, int part, int *disp, int *size);

/**
 * This routine returns the MPI displacement and size of the intersection of a
 * hyper-slab with the given dataset on the current process.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param disp  an array populated with the displacement of the chunk within the
 *              "global" distributed array
 * @param size  an array populated with the spatial extents
 * @param slab  index ranges of the hyperslab. Ordering: {begin_0, end_0, begin_1, end_1,...}
 * @return the size in bytes of the intersection between the dataset and the slab on this process
 */
OPS_FTN_INTEROP
size_t ops_dat_get_slab_extents(ops_dat dat, int part, int *disp, int *size, int *slab);

/**
 * This routine returns array shape metadata corresponding to the ops_dat.
 * Any of the arguments may be NULL.
 *
 * @param data     the datasets
 * @param part     the chunk index (has to be 0)
 * @param disp     an array populated with the displacement of the chunk within the
 *                 "global" distributed array
 * @param size     an array populated with the spatial extents
 * @param stride   an array populated strides in spatial dimensions needed for
 *                 column-major indexing
 * @param d_m      an array populated with padding on the left in each dimension
 *                 note that these are negative values
 * @param d_p      an array populated with padding on the right in each dimension
 */
OPS_FTN_INTEROP
void ops_dat_get_raw_metadata(ops_dat dat, int part, int *disp, int *size, int *stride, int *d_m, int *d_p);

/**
 * This routine returns a pointer to the internally stored data, with MPI halo
 * regions automatically updated as required by the supplied stencil.
 * The strides required to index into the dataset are also given.
 *
 * You may have to call ops_execute before calling this to make sure all
 * computations have finished.
 *
 * @param dat      the dataset
 * @param part     the chunk index (has to be 0)
 * @param stencil  a stencil used to determine required MPI halo exchange depths
 * @param memspace when set to OPS_HOST or OPS_DEVICE, returns a pointer to data in
 *                 that memory space, otherwise must be set to 0, and returns
 *                 whether data is in the host or on the device
 *
 * @return
 */
OPS_FTN_INTEROP
char* ops_dat_get_raw_pointer(ops_dat dat, int part, ops_stencil stencil, ops_memspace *memspace);

/**
 * Indicates to OPS that a dataset previously accessed with
 * ops_dat_get_raw_pointer() is released by the user, and also tells OPS how it
 * was accessed.
 *
 * A single call to ops_dat_release_raw_data() releases all pointers obtained by previous calls to
 * ops_dat_get_raw_pointer() calls on the same dat. Calls to ops_dat_release_raw_data() must be separated by calls to
 * ops_dat_get_raw_pointer(), i.e. it is illegal to release raw data access multiple times without first
 * starting raw data access. The data that the user wishes keep is in the memory
 * space (buffer) indicated by the LAST call to ops_dat_get_raw_pointer().  Data in any other memory spaces
 * is discarded.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0
 * @param acc   the kind of access that was used by the user
 *              (::OPS_READ if it was read only,
 *              ::OPS_WRITE if it was overwritten,
 *              ::OPS_RW if it was read and written)
 */
OPS_FTN_INTEROP
void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc);

/**
 * Indicates to OPS that a dataset previously accessed with
 * ops_dat_get_raw_pointer() is released by the user, and also tells OPS how it
 * was accessed.
 *
 * A single call to ops_dat_release_raw_data_memspace() releases all pointers obtained by previous calls to
 * ops_dat_get_raw_pointer() calls on the same dat. Calls to ops_dat_release_raw_data_memspace() must be separated by calls to
 * ops_dat_get_raw_pointer(), i.e. it is illegal to release raw data access multiple times without first
 * starting raw data access.
 * The *memspace argument tells OPS in which memory space the data is that the user wants to keep.
 * Data in all other memory spaces will be discarded.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0
 * @param acc   the kind of access that was used by the user
 *              (::OPS_READ if it was read only,
 *              ::OPS_WRITE if it was overwritten,
 *              ::OPS_RW if it was read and written)
 * @param memspace has to be set to either OPS_HOST or OPS_DEVICE to indicate where the data was modified
 */
OPS_FTN_INTEROP
void ops_dat_release_raw_data_memspace(ops_dat dat, int part, ops_access acc, ops_memspace *memspace);


/**
 * This routine copies the data held by OPS to the user-specified
 * memory location, which needs to be at least as large as indicated
 * by the `sizes` parameter of ops_dat_get_extents().

 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to CPU memory which should be filled by OPS
 */
OPS_FTN_INTEROP
void ops_dat_fetch_data(ops_dat dat, int part, char *data);

/**
 * This routine copies the data held by OPS to the user-specified
 * memory location, which needs to be at least as large as indicated
 * by the `sizes` parameter of ops_dat_get_extents().

 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be filled by OPS
 * @param memspace the memory space where the data pointer is
 */
OPS_FTN_INTEROP
void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace);

/**
 * This routine copies a hyperslab of the data held by OPS to the user-specified
 * memory location, which needs to be at least as large as the product of ranges
 * defined by the `range` parameter

 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be filled by OPS
 * @param range index ranges of the hyperslab. Ordering: {begin_0, end_0, begin_1, end_1,...}
 * @param memspace the memory space where the data pointer is
 */
OPS_FTN_INTEROP
void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace);

/**
 * This routine copies the data given by the user to the internal data structure
 * used by OPS.
 *
 * User data needs to be laid out in column-major order and strided as indicated
 * by the `sizes` parameter of ops_dat_get_extents().
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to CPU memory which should be copied to OPS
 */
OPS_FTN_INTEROP
void ops_dat_set_data(ops_dat dat, int part, char *data);

/**
 * This routine copies the data from the user-specified
 * memory location, which needs to be at least as large as indicated
 * by the `sizes` parameter of ops_dat_get_extents(), to the OPS-held
 * memory space

 * @param dat   the dataset to be filled
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be read by OPS
 * @param memspace the memory space where the data pointer is
 */
OPS_FTN_INTEROP
void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace);

/**
 * This routine copies to a hyperslab of the data held by OPS from the user-specified
 * memory location, which needs to be at least as large as the product of ranges
 * defined by the `range` parameter

 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be read by OPS
 * @param range index ranges of the hyperslab. Ordering: {begin_0, end_0, begin_1, end_1,...}
 * @param memspace the memory space where the data pointer is
 */
OPS_FTN_INTEROP
void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace);


/**
 * This routine returns the number of chunks of the given dataset held by all
 * processes.
 *
 * @param dat  the dataset
 * @return
 */
OPS_FTN_INTEROP
int ops_dat_get_global_npartitions(ops_dat dat);
#endif

/*******************************************************************************
* Random number generations
*******************************************************************************/
void ops_randomgen_init(unsigned int seed, int options);
void ops_fill_random_uniform(ops_dat dat);
void ops_fill_random_normal(ops_dat dat);
void ops_randomgen_exit();


/**
 * This class is an accessor to data stored in ops_dats. It is
 * only to be used in user kernels and functions called from within
 * user kernels. The user should never explicitly construct such an
 * object, these are constucted by OPS and passed by reference to
 * the user kernel.
 *
 * Data can be accessed using the overloaded () operator - with as many
 * arguments as many dimensional the dataset is (i.e. 2 in 2D). An extra
 * argument is used for datasets that have multiple values at each gridpoint.
 * Arguments are always relative offsets w.r.t. the current grid point.
 *
 */

template<typename T>
class ACC {
public:
  //////////////////////////////////////////////////
  // 1D
  /////////////////////////////////////////////////
#if defined(OPS_1D)
  __host__ __device__
  ACC(T *_ptr) : ptr(_ptr) {}
  __host__ __device__
  ACC(int _mdim, int _sizex, T *_ptr) :
#ifdef OPS_SOA
    sizex(_sizex),
#else
    mdim(_mdim),
#endif
    ptr(_ptr)
  {}
  __host__ __device__
  const T& operator()(int xoff) const {return *(ptr + xoff);}
  __host__ __device__
  T& operator()(int xoff) {return *(ptr + xoff);}
  __host__ __device__
  const T& operator()(int d, int xoff) const {
#ifdef OPS_SOA
    return *(ptr + xoff + d * sizex);
#else
    return *(ptr + d + xoff*mdim );
#endif
  }
  __host__ __device__
  T& operator()(int d, int xoff) {
#ifdef OPS_SOA
    return *(ptr + xoff + d * sizex);
#else
    return *(ptr + d + xoff*mdim );
#endif
  }

  __host__ __device__
  void combine_max(int xoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMax(&operator()(xoff), val);
    #else
    #pragma omp critical
    operator()(xoff) = std::max(operator()(xoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_min(int xoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMin(&operator()(xoff), val);
    #else
    #pragma omp critical
    operator()(xoff) = std::min(operator()(xoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_inc(int xoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicAdd(&operator()(xoff), val);
    #else
    #pragma omp critical
    operator()(xoff) += val;
    #endif

    return ;
  }
#endif

  //////////////////////////////////////////////////
  // 2D
  /////////////////////////////////////////////////
#if defined(OPS_2D)
  __host__ __device__
  ACC(int _sizex, T *_ptr) : sizex(_sizex), ptr(_ptr) {}
  __host__ __device__
  ACC(int _mdim, int _sizex, int _sizey, T *_ptr) : sizex(_sizex),
#ifdef OPS_SOA
    sizey(_sizey),
#else
    mdim(_mdim),
#endif
    ptr(_ptr)
  {}
  __host__ __device__
  const T& operator()(int xoff, int yoff) const {return *(ptr + xoff + yoff*sizex);}
  __host__ __device__
  T& operator()(int xoff, int yoff) {return *(ptr + xoff + yoff*sizex);}
  __host__ __device__
  const T& operator()(int d, int xoff, int yoff) const {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + d * sizex*sizey);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim );
#endif
  }
  __host__ __device__
  T& operator()(int d, int xoff, int yoff) {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + d * sizex*sizey);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim );
#endif
  }
  __host__ __device__
  void combine_max(int xoff, int yoff,const T val){

    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMax(&operator()(xoff, yoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff) = std::max(operator()(xoff, yoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_min(int xoff, int yoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMin(&operator()(xoff, yoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff) = std::min(operator()(xoff, yoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_inc(int xoff, int yoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicAdd(&operator()(xoff, yoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff) += val;
    #endif

    return ;
  }

#endif
  //////////////////////////////////////////////////
  // 3D
  /////////////////////////////////////////////////
#if defined(OPS_3D)
  __host__ __device__
  ACC(int _sizex, int _sizey, T *_ptr) : sizex(_sizex), sizey(_sizey), ptr(_ptr) {}
  __host__ __device__
  ACC(int _mdim, int _sizex, int _sizey, int _sizez, T *_ptr) : sizex(_sizex), sizey(_sizey),
#ifdef OPS_SOA
    sizez(_sizez),
#else
    mdim(_mdim),
#endif
    ptr(_ptr)
  {}
  __host__ __device__
  const T& operator()(int xoff, int yoff, int zoff) const {return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey);}
  __host__ __device__
  T& operator()(int xoff, int yoff, int zoff) {return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey);}
  __host__ __device__
  const T& operator()(int d, int xoff, int yoff, int zoff) const {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + d * sizex*sizey*sizez);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim + zoff*sizex*sizey*mdim);
#endif
  }
  __host__ __device__
  T& operator()(int d, int xoff, int yoff, int zoff) {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + d * sizex*sizey*sizez);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim + zoff*sizex*sizey*mdim);
#endif
  }

  __host__ __device__
  void combine_max(int xoff, int yoff, int zoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMax(&operator()(xoff, yoff, zoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff, zoff) = std::max(operator()(xoff, yoff, zoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_min(int xoff, int yoff, int zoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMin(&operator()(xoff, yoff, zoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff, zoff) = std::min(operator()(xoff, yoff, zoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_inc(int xoff, int yoff, int zoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicAdd(&operator()(xoff, yoff, zoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff, zoff) += val;
    #endif

    return ;
  }
#endif

  //////////////////////////////////////////////////
  // 4D
  /////////////////////////////////////////////////
#if defined(OPS_4D)
  __host__ __device__
  ACC(int _sizex, int _sizey, int _sizez, T *_ptr) : sizex(_sizex), sizey(_sizey), sizez(_sizez), ptr(_ptr) {}
  __host__ __device__
  ACC(int _mdim, int _sizex, int _sizey, int _sizez, int _sizeu, T *_ptr) : sizex(_sizex), sizey(_sizey), sizez(_sizez),
#ifdef OPS_SOA
    sizeu(_sizeu),
#else
    mdim(_mdim),
#endif
    ptr(_ptr)
  {}
  __host__ __device__
  const T& operator()(int xoff, int yoff, int zoff, int uoff) const {return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + uoff*sizex*sizey*sizez);}
  __host__ __device__
  T& operator()(int xoff, int yoff, int zoff, int uoff) {return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + uoff*sizex*sizey*sizez);}
  __host__ __device__
  const T& operator()(int d, int xoff, int yoff, int zoff, int uoff) const {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + uoff*sizex*sizey*sizez + d * sizex*sizey*sizez*sizeu);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim + zoff*sizex*sizey*mdim + uoff*sizex*sizey*sizez*mdim);
#endif
  }
  __host__ __device__
  T& operator()(int d, int xoff, int yoff, int zoff, int uoff) {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + uoff*sizex*sizey*sizez + d * sizex*sizey*sizez*sizeu);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim + zoff*sizex*sizey*mdim + uoff*sizex*sizey*sizez*mdim);
#endif
  }

  __host__ __device__
  void combine_max(int xoff, int yoff, int zoff, int uoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMax(&operator()(xoff, yoff, zoff, uoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff, zoff, uoff) = std::max(operator()(xoff, yoff, zoff, uoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_min(int xoff, int yoff, int zoff, int uoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicMin(&operator()(xoff, yoff, zoff, uoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff, zoff, uoff) = std::min(operator()(xoff, yoff, zoff, uoff), val);
    #endif

    return ;
  }

  __host__ __device__
  void combine_inc(int xoff, int yoff, int zoff, int uoff,const T val){
    
    #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    atomicAdd(&operator()(xoff, yoff, zoff, uoff), val);
    #else
    #pragma omp critical
    operator()(xoff, yoff, zoff, uoff) += val;
    #endif

    return ;
  }
#endif



  __host__ __device__
  void next(int offset) {
    ptr += offset;
  }


private:
#if defined(OPS_2D) || defined(OPS_3D) || defined(OPS_4D) || defined (OPS_5D) || (defined(OPS_1D) && defined(OPS_SOA))
  int sizex;
#endif
#if defined(OPS_3D) || defined(OPS_4D) || defined (OPS_5D) || (defined(OPS_2D) && defined(OPS_SOA))
  int sizey;
#endif
#if defined(OPS_4D) || defined (OPS_5D) || (defined(OPS_3D) && defined(OPS_SOA))
  int sizez;
#endif
#if defined (OPS_5D) || (defined(OPS_4D) && defined(OPS_SOA))
  int sizeu;
#endif
#if defined(OPS_5D) && defined(OPS_SOA)
  int sizev;
#endif
#ifndef OPS_SOA
  int mdim;
#endif
  T *__restrict__ ptr;
};

#include <ops_internal2.h>

#endif /* __OP_LIB_CORE_H */
