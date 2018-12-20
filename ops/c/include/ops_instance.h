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

class OPS_instace_mpi;
class OPS_instance_tiling;
class OPS_instance_checkpointing;
class OPS_instance_opencl;

class OPS_instance {
  private:
  	OPS_instance() {}
  public:
  	static OPS_instance* getOPSInstance() {
  		OPS_instance **ptr;
		#if defined(_OPENMP)
  		ptr = &ops_instances[omp_get_thread_num()];
		#else
  		ptr = &ops_instances[0];
		#endif

  		if (*ptr == 0) *ptr = new OPS_instance();
  		return *ptr;
  	}
	/*******************************************************************************
	* Global constants
	*******************************************************************************/


	//Blocks, Dats, Stencils Halos, Reductions
	int OPS_block_index=0, OPS_block_max=0, OPS_dat_index=0, OPS_dat_max=0,
	OPS_halo_group_index=0, OPS_halo_group_max=0, OPS_halo_index=0, OPS_halo_max=0,
	OPS_reduction_index=0, OPS_reduction_max=0, OPS_stencil_index=0, OPS_stencil_max=0;
	ops_block_descriptor *OPS_block_list=NULL;
	ops_stencil *OPS_stencil_list=NULL;
	ops_halo *OPS_halo_list=NULL;
	ops_halo_group *OPS_halo_group_list=NULL;
	Double_linked_list OPS_dat_list;
	ops_reduction *OPS_reduction_list = NULL;
	

	// Checkpointing
 	int OPS_enable_checkpointing=0;
	double OPS_checkpointing_time=0.0;
	int ops_thread_offload = 0;
	int ops_checkpoint_inmemory = 0;
	int ops_lock_file = 0;
	ops_backup_state backup_state=OPS_NONE;
	char *OPS_dat_ever_written;
	ops_checkpoint_types *OPS_dat_status=NULL;
	int OPS_ranks_per_node;


	// Debugging
	ops_arg *OPS_curr_args = NULL;
	const char *OPS_curr_name = NULL;

	//Diagnostics
	int OPS_kern_max=0, OPS_kern_curr=0;
	ops_kernel *OPS_kernels=NULL;
	
	//Tiling
	int ops_enable_tiling = 0;
	int ops_cache_size = 0;
	int ops_tiling_mpidepth = -1;
	double ops_tiled_halo_exchange_time=0.0;
	OPS_instance_tiling *tiling_instance;
	OPS_instance_checkpointing *checkpointing_instance;

	//Other runtime configuration args
	int ops_force_decomp[OPS_MAX_DIM] = {0};
	int OPS_realloc = 0;
	int OPS_soa=0;
	int OPS_diags=0;

	// CUDA & OpenCL
	int OPS_hybrid_gpu=0, OPS_gpu_direct=0;
	int OPS_block_size_x = 32;
	int OPS_block_size_y = 4;
	int OPS_block_size_z = 1;
	char *OPS_consts=NULL, *OPS_consts_d=NULL, *OPS_reduct_h=NULL, *OPS_reduct_d=NULL;
	int OPS_consts_bytes = 0, OPS_reduct_bytes = 0;
	int OPS_cl_device=0;
	char *ops_halo_buffer = NULL;
	char *ops_halo_buffer_d = NULL;
	int ops_halo_buffer_size = 0;
	int OPS_gbl_changed = 1;
	char *OPS_gbl_prev = NULL;
	OPS_instance_opencl *opencl_instance;


	//MPI
	OPS_instace_mpi *mpi_instance;

};

#ifndef MPI_VERSION
class OPS_instace_mpi {
};
#endif

#endif //__OPS_INSTANCE_H