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

#ifndef __OPS_INSTANCE_H
#define __OPS_INSTANCE_H

#include <ops_lib_core.h>
#include <ops_checkpointing.h>

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
  private:
	OPS_instance();
  public:
	static OPS_instance* getOPSInstance();
	static void destroyOPSInstance();

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
  int ops_loop_over_blocks;
  int *ops_loop_over_blocks_predicate;
  int ops_loop_over_blocks_condition;

  int ops_batch_size;
  int OPS_hybrid_layout;

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

};

#endif //__OPS_INSTANCE_H
