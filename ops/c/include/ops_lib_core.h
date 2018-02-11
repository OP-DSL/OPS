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

/**
 * maximum number of simulataneous OPS instances 
 * in a shared memory environment
 */
#ifndef OPS_MAX_INSTANCES
#define OPS_MAX_INSTANCES 64
#endif 


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

#define ZERO_bool 0;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

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

/** Storage for OPS blocks */
typedef struct {
  int index;        /**< index */
  int dims;         /**< dimension of block, 2D,3D .. etc*/
  char const *name; /**< name of block */
} ops_block_core;

typedef ops_block_core *ops_block;

/** Storage for OPS datasets */
typedef struct {
<<<<<<< HEAD
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
  char *data;            /**< data on host */
  char *data_d;          /**< data on device */
  char const *name;      /**< name of dataset */
  char const *type;      /**< datatype */
  int dirty_hd;          /**< flag to indicate dirty status on host and device*/
  int user_managed;      /**< indicates whether the user is managing memory */
  int is_hdf5;           /**< indicates whether the data is to read from an
                          *   hdf5 file */
  char const *hdf5_file; /**< name of hdf5 file from which this dataset was
                          *   read */
  int e_dat;             /**< flag to indicate if this is an edge dat */
  long mem;              /**< memory in bytes allocated to this dat (under MPI,
                          *   this will be memory held on a single MPI proc) */
  long base_offset;      /**< computed quantity, giving offset in bytes to the
                          *   base index */
  int stride[OPS_MAX_DIM];/**< stride[*] > 1 if this dat is a coarse dat under
                           *   multi-grid*/
=======
  int index;             /* index */
  ops_block block;       /* block on which data is defined */
  int dim;               /* number of elements per grid point*/
  int type_size;         /* bytes per primitive = elem_size/dim */
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
  int amr; /* flag indicating wheter AMR dataset */
>>>>>>> 2cb93bb... Latest patches for AMR loops
} ops_dat_core;

typedef ops_dat_core *ops_dat;

// struct definition for a double linked list entry to hold an ops_dat
struct ops_dat_entry_core {
  ops_dat dat;
  TAILQ_ENTRY(ops_dat_entry_core)
  entries; /**< holds pointers to next and previous entries in the list*/
};

typedef struct ops_dat_entry_core ops_dat_entry;

typedef TAILQ_HEAD(, ops_dat_entry_core) Double_linked_list;

/** Storage for OPS block - OPS dataset associations */
typedef struct {
  ops_block_core *block;        /**< pointer to the block */
  Double_linked_list datasets;  /**< list of datasets associated with this block */
  int num_datasets;             /**< number of datasets */

} ops_block_descriptor;

/** Storage for OPS stencils */
typedef struct {
  int index;        /**< index */
  int dims;         /**< dimensionality of the stencil */
  int points;       /**< number of stencil elements */
  char const *name; /**< name of pointer */
  int *stencil;     /**< elements in the stencil */
  int *stride;      /**< stride of the stencil */
  int *mgrid_stride;/**< stride of the stencil under multi_grid */
  int type;         /**< 0 for regular, 1 for prolongate, 2 for restrict */
} ops_stencil_core;

typedef ops_stencil_core *ops_stencil;

/** Storage for OPS parallel loop arguments */
typedef struct {
  ops_dat dat;          /**< dataset */
  ops_stencil stencil;  /**< the stencil */
  int field;            /* field of multi-dimensional data accessed XXX*/
  int dim;              /**< dimension of data */
  char *data;           /**< data on host */
  char *data_d;         /**< data on device (for CUDA)*/
  ops_access acc;       /**< access type */
  ops_arg_type argtype; /**< arg type */
  int opt;              /**< falg to indicate whether this is an optional arg,
                         *   0 - optional, 1 - not optional */
} ops_arg;

/** Storage for OPS reduction handles */
typedef struct {
  char *data;       /**< The data */
  int size;         /**< size of data in bytes */
  int initialized;  /**< flag indicating whether data has been initialized */
  int index;        /**< unique identifier */
  ops_access acc;   /**< Type of reduction it was used for last time */
  const char *type; /**< Type */
  const char *name; /**< Name */
} ops_reduction_core;
typedef ops_reduction_core *ops_reduction;

/** Storage for OPS parallel loop statistics */
typedef struct {
  char *name;      /**< name of kernel function */
  int count;       /**< number of times called */
  double time;     /**< total execution time */
  float transfer;  /**< bytes of data transfer (used) */
  double mpi_time; /**< time spent in MPI calls */
  //  double       mpi_gather;
  //  double       mpi_scatter;
  //  double       mpi_sendrecv;
  //  double       mpi_reduct;
} ops_kernel;

/** Storage for OPS halos */
typedef struct {
  ops_dat from;                   /**< dataset from which the halo is read */
  ops_dat to;                     /**< dataset to which the halo is written */
  int iter_size[OPS_MAX_DIM];     /**< size of halo region */
  int from_base[OPS_MAX_DIM];     /**< start position to copy from */
  int to_base[OPS_MAX_DIM];       /**< start position to copy to */
  int from_dir[OPS_MAX_DIM];      /**< copy from orientation */
  int to_dir[OPS_MAX_DIM];        /**< size to orientation */
  int index;                      /**< index of halo */
} ops_halo_core;

typedef ops_halo_core *ops_halo;

/** Storage for OPS halo groups */
typedef struct {
  int nhalos;                     /**< number of halos */
  ops_halo *halos;                /**< list of halos */
  int index;                      /**< index of halo group */
} ops_halo_group_core;

typedef ops_halo_group_core *ops_halo_group;

/** Storage for OPS parallel handles */
typedef struct ops_kernel_descriptor {
  const char *name;           /**< name of kernel */
  unsigned long hash;         /**< hash of loop */
  ops_arg *args;              /**< list of arguments to pass in */
  int nargs;                  /**< number of arguments */
  int index;                  /**< index of the loop */
  int dim;                    /**< number of dimensions */
  int device;                 /**< flag to indicate if loop runs on device */
  int range[2 * OPS_MAX_DIM]; /**< process local execution range */
  int orig_range[2 * OPS_MAX_DIM]; /**< original execution range */
  ops_block block;            /**< block to execute on */
  void (*function)(struct ops_kernel_descriptor
                      *desc); /**< Function pointer to a wrapper to be called */
} ops_kernel_descriptor;

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

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/*
 * alignment macro based on example on page 50 of CUDA Programming Guide version
 * 3.0
 * rounds up to nearest multiple of 16 bytes
 */

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ROUND_UP_64(bytes) (((bytes) + 63) & ~63)


#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/*******************************************************************************
* Core lib function prototypes
*******************************************************************************/

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
void ops_init(const int argc, const char **argv, const int diags_level);

/**
 * This routine must be called last to cleanly terminate the OPS computation.
 */
void ops_exit();

ops_dat ops_decl_amrdat_char(ops_block, int, int *, int *, int *, int *, char *,
                          int, char const *, char const *);
ops_dat ops_decl_amrdat_mpi_char(ops_block block, int size, int *dat_size,
                              int *base, int *d_m, int *d_p, char *data,
                              int type_size, char const *type,
                              char const *name);

/**
 * This routine defines a structured grid block.
 *
 * @param dims  dimension of the block
 * @param name  a name used for output diagnostics
 * @return
 */
ops_block ops_decl_block(int dims, const char *name);


/**
 * Deallocates an OPS dataset
 * @param dat     dataset to deallocate
 */
void ops_free_dat(ops_dat dat); 


/**
 * Passes a pointer to the value(s) at the current grid point to the user kernel.
 *
 * The ::OPS_ACC macro has to be used for dereferencing the pointer.
 *
 * @param dat      dataset
 * @param dim
 * @param stencil  stencil for accessing data
 * @param type     string representing the type of data held in dataset
 * @param acc      access type
 * @return
 */
ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc);

ops_arg ops_arg_dptr(ops_dat dat, char* data, int field, int dim, ops_stencil stencil, char const *type,
                    ops_access acc); //XXX
/**
 * Passes a pointer to the value(s) at the current grid point to the user kernel if flag is true
 *
 * The ::OPS_ACC macro has to be used for dereferencing the pointer.
 *
 * @param dat      dataset
 * @param dim
 * @param stencil  stencil for accessing data
 * @param type     string representing the type of data held in dataset
 * @param acc      access type
 * @param flag     indicates whether the optional argument is enabled (non-0) or not (0)
 * @return
 */
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
ops_arg ops_arg_reduce(ops_reduction handle, int dim, const char *type,
                       ops_access acc);


/**
 * This routine defines a reduction handle to be used in a parallel loop.
 *
 * @param size  size of data in bytes
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param name  name of the dat used for output diagnostics
 * @return
 */
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
ops_stencil ops_decl_strided_stencil(int dims, int points, int *sten,
                                     int *stride, char const *name);
ops_stencil ops_decl_restrict_stencil( int dims, int points, int *sten,
                                       int *stride, char const * name);
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
ops_halo_group ops_decl_halo_group(int nhalos, ops_halo *halos);

/**
 * This routine exchanges all halos in a halo group and will block execution
 * of subsequent computations that depend on the exchanged data.
 *
 * @param group  the halo group
 */
void ops_halo_transfer(ops_halo_group group);

/**
 * This routine simply prints a variable number of arguments on the root process;
 * it is created in place of the standard C printf() function which would
 * print the same on each MPI process.
 *
 * @param format
 * @param ...
 */
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
void ops_fprintf(FILE *stream, const char *format, ...);

/**
 * This routine prints out various useful bits of diagnostic info about sets,
 * mappings and datasets.
 *
 * Usually used right after an @p ops partition() call to print out the details
 * of the decomposition.
 */
void ops_diagnostic_output();

/**
 * Print OPS performance performance details to output stream.
 *
 * @param stream  output stream, use stdout to print to standard out
 */
void ops_timing_output(FILE *stream);
void ops_timing_output_stdout();

/**
 * gettimeofday() based timer to start/end timing blocks of code.
 *
 *
 * @param cpu  variable to hold the CPU time at the time of invocation
 * @param et   variable to hold the elapsed time at the time of invocation
 */
void ops_timers(double *cpu, double *et);

/**
 * Write the details of an ::ops_block to a named text file.
 *
 * When used under an MPI parallelization each MPI process will write its own
 * data set separately to the text file. As such it does not use MPI I/O.
 * The data can be viewed using a simple text editor

 * @param dat        ::ops_dat to to be written
 * @param file_name  text file to write to
 */
void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name);

/**
 * Makes sure OPS has downloaded data from the device
 */
void ops_get_data(ops_dat dat);

/**
 * Returns one one the root MPI process
 */
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
void ops_partition(const char *routine);

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
void ops_dat_get_extents(ops_dat dat, int part, int *disp, int *size);

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
void ops_dat_get_raw_metadata(ops_dat dat, int part, int *disp, int *size, int *stride, int *d_m, int *d_p);

/**
 * type for memory space flags - 1 for host, 2 for device
 */
typedef int ops_memspace;
#define OPS_HOST 1
#define OPS_DEVICE 2
/**
 * This routine returns a pointer to the internally stored data, with MPI halo
 * regions automatically updated as required by the supplied stencil.
 * The strides required to index into the dataset are also given.
 *
 * @param dat      the dataset
 * @param part     the chunk index (has to be 0)
 * @param stencil  a stencil used to determine required MPI halo exchange depths
 * @param memspace when set to OPS_HOST or OPS_DEVICE, returns a pointer to data in
 *                 that memory space, otherwise must be set to 0, and returns
 *                 wheter data is in the host or on the device
 *
 * @return
 */
char* ops_dat_get_raw_pointer(ops_dat dat, int part, ops_stencil stencil, ops_memspace *memspace);

/**
 * Indicates to OPS that a dataset previously accessed with
 * ops_dat_get_raw_pointer() is released by the user, and also tells OPS how it
 * was accessed.
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0
 * @param acc   the kind of access that was used by the user
 *              (::OPS_READ if it was read only,
 *              ::OPS_WRITE if it was overwritten,
 *              ::OPS_RW if it was read and written)
 */
void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc);

/**
 * This routine copies the data held by OPS to the user-specified
 * memory location, which needs to be at least as large as indicated
 * by the `sizes` parameter of ops_dat_get_extents().

 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be filled by OPS
 */
void ops_dat_fetch_data(ops_dat dat, int part, char *data);

/**
 * This routine copies the data given by the user to the internal data structure
 * used by OPS.
 *
 * User data needs to be laid out in column-major order and strided as indicated
 * by the `sizes` parameter of ops_dat_get_extents().
 *
 * @param dat   the dataset
 * @param part  the chunk index (has to be 0)
 * @param data  pointer to memory which should be copied to OPS
 */
void ops_dat_set_data(ops_dat dat, int part, char *data);

/**
 * This routine returns the number of chunks of the given dataset held by all
 * processes.
 *
 * @param dat  the dataset
 * @return
 */
int ops_dat_get_global_npartitions(ops_dat dat);


#ifndef DOXYGEN_SHOULD_SKIP_THIS

ops_reduction ops_decl_reduction_handle_core(int size, const char *type,
                                             const char *name);
void ops_execute_reduction(ops_reduction handle);

ops_arg ops_arg_reduce_core(ops_reduction handle, int dim, const char *type,
                            ops_access acc);
ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc);
void ops_decl_const_char(int, char const *, int, char *, char const *);
void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr);

void ops_init_core(const int argc, const char **argv, const int diags_level);

void ops_exit_core(void);



ops_dat ops_decl_dat_char(ops_block, int, int *, int *, int *, int *, int *, char *,
                          int, char const *, char const *);

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

ops_halo ops_decl_halo_core(ops_dat from, ops_dat to, int *iter_size,
                            int *from_base, int *to_base, int *from_dir,
                            int *to_dir);



ops_arg ops_arg_dat_core(ops_dat dat, ops_stencil stencil, ops_access acc);
ops_arg ops_arg_gbl_core(char *data, int dim, int size, ops_access acc);


void ops_print_dat_to_txtfile_core(ops_dat dat, const char *file_name);

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

void ops_init_arg_idx(int *arg_idx, int ndims, ops_arg *args, int nargs);

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
int compute_ranges(ops_arg* args, int nargs, ops_block block, int* range, int* start, int* end, int* arg_idx);
int ops_get_proc();
int ops_num_procs();
void ops_put_data(ops_dat dat);

/*******************************************************************************
* Memory allocation functions
*******************************************************************************/
void* ops_malloc (size_t size);
void* ops_realloc (void *ptr, size_t size);
void  ops_free (void *ptr);
void* ops_calloc (size_t num, size_t size);


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
class OPS_instance;
extern OPS_instance *ops_instances[];
#endif

#include "ops_checkpointing.h"
#include "ops_hdf5.h"
#include "ops_tridiag.h"
#ifdef __cplusplus
#include "ops_instance.h"
#endif

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* __OP_LIB_CORE_H */
