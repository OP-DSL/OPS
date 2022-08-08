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
  * @details function declarations header file for the core library functions
  * utilized by all OPS backends
  */

#ifndef __OPS_INSTANCE_H
#define __OPS_INSTANCE_H

#if defined(_OPENMP)
  #include <omp.h>
#endif

class OPS_instance_tiling;
class OPS_instance_checkpointing;
class OPS_instance_opencl;

/**
 * This class encapsulates "global" scope data required for OPS instances.
 * To support multiple instances of OPS in a shared memory environment,
 * you need to add support for your own threading library in the implementation
 * of getOPSInstance() in src/core/ops_instance.cpp
 *
 * Currently supported threading libraries: OpenMP
 */

class OPS_instance {
  public:
    std::ostream& ostream() {return *_ostream;}
    void set_ostream(std::ostream &s);
#if !defined(OPS_CPP_API) || defined(OPS_INTERNAL_API)
  	static OPS_instance* getOPSInstance();
#endif

   static int numInstances();

  	OPS_instance();

// #ifdef OPS_CPP_API
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
  	OPS_instance(const int argc, const char *const argv[], const int diags_level, std::ostream &ostream = std::cout);
  	
    void exit();
    ~OPS_instance() {
      this->exit();
    }

/**
 * This routine defines a structured grid block.
 *
 * @param dims  dimension of the block
 * @param name  a name used for output diagnostics
 * @return
 */
    ops_block decl_block(int dims, const char * name);
/**
 * This routine defines a reduction handle to be used in a parallel loop.
 *
 * @param size  size of data in bytes
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param name  name of the dat used for output diagnostics
 * @return
 */
    ops_reduction decl_reduction_handle(int size, const char *type,
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
    ops_stencil decl_stencil(int dims, int points, int *stencil,
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
    ops_stencil decl_strided_stencil(int dims, int points, int *sten,
                                     int *stride, char const *name);
    ops_stencil decl_restrict_stencil( int dims, int points, int *sten,
                                       int *stride, char const * name);
    ops_stencil decl_prolong_stencil( int dims, int points, int *sten,
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
    ops_halo decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
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
    ops_halo_group decl_halo_group(int nhalos, ops_halo *halos);

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
    void decl_const(char const *name, int dim, char const *type, T *data) {
      if (type_error(data, type)) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: incorrect type specified for constant " << name
           << " in OPS_instance::decl_const";
        throw ex;
      }
      ops_decl_const_char(this, dim, type, sizeof(T), (char *)data, name);
    }

/**
 * This routine updates/changes the value of a constant.
 *
 * @tparam T
 * @param name  a name used to identify the constant
 * @param dim   dimension of dataset (number of items per element)
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param data  pointer to input data of type @p T
 */
    template <class T>
    void update_const(char const *name, int dim, char const *type, T *data) {
      decl_const(name, dim, type, data);
    }

/**
 * This routine prints out various useful bits of diagnostic info about sets,
 * mappings and datasets.
 *
 * Usually used right after an @p ops partition() call to print out the details
 * of the decomposition.
 */
    void diagnostic_output();
/**
 * Print OPS performance performance details to output stream.
 *
 * @param stream  output stream, use std::cout to print to standard out
 */
    void timing_output(std::ostream &stream);
    void timing_output_stdout();

/**
 * Returns one one the root MPI process
 */
    int is_root();

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
    void partition(const char *routine);
	void partition(const char *routine, std::map<std::string, void*>& opts);
// #endif

	/*******************************************************************************
	* Global constants
	*******************************************************************************/


	//Blocks, Dats, Stencils Halos, Reductions
	int OPS_block_index, OPS_block_max, OPS_dat_index, OPS_dat_max,
	OPS_halo_group_index, OPS_halo_group_max, OPS_halo_index, OPS_halo_max,
	OPS_reduction_index, OPS_reduction_max, OPS_stencil_index, OPS_stencil_max;
	ops_block_descriptor *OPS_block_list;
	ops_stencil *OPS_stencil_list;
	ops_halo *OPS_halo_list;
	ops_halo_group *OPS_halo_group_list;
	Double_linked_list OPS_dat_list;
	ops_reduction *OPS_reduction_list;
	

	// Checkpointing
 	int OPS_enable_checkpointing;
	double OPS_checkpointing_time;
	int ops_thread_offload;
	int ops_checkpoint_inmemory;
	int ops_lock_file;
	ops_backup_state backup_state;
	char *OPS_dat_ever_written;
	ops_checkpoint_types *OPS_dat_status;
	int OPS_ranks_per_node;

	//SEQ execution
	int arg_idx[OPS_MAX_DIM];


	// Debugging
	ops_arg *OPS_curr_args;
	const char *OPS_curr_name;

	//Diagnostics
	int OPS_kern_max, OPS_kern_curr;
	ops_kernel *OPS_kernels;
	
	//Tiling
	int ops_enable_tiling;
	int ops_cache_size;
	int ops_tiling_mpidepth;
	double ops_tiled_halo_exchange_time;
	OPS_instance_tiling *tiling_instance;
	OPS_instance_checkpointing *checkpointing_instance;
  int tilesize_x, tilesize_y, tilesize_z;

	//Other runtime configuration args
	int ops_force_decomp[OPS_MAX_DIM];
	int OPS_realloc;
	int OPS_soa;
	int OPS_diags;

	// CUDA & OpenCL
	int OPS_hybrid_gpu, OPS_gpu_direct;
	int OPS_block_size_x;
	int OPS_block_size_y;
	int OPS_block_size_z;
	char *OPS_consts_h, *OPS_consts_d, *OPS_reduct_h, *OPS_reduct_d;
	int OPS_consts_bytes, OPS_reduct_bytes;
	int OPS_cl_device;
	char *ops_halo_buffer;
	char *ops_halo_buffer_d;
	int ops_halo_buffer_size;
	int OPS_gbl_changed;
	char *OPS_gbl_prev;
	OPS_instance_opencl *opencl_instance;

  int is_initialised;

  ops_stencil OPS_internal_0[OPS_MAX_DIM];
private:
	void init_globals();
  std::ostream *_ostream;
};

#endif //__OPS_INSTANCE_H
