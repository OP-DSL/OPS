#include <ops_instance.h>

OPS_instance::OPS_instance() {

  		//Blocks, Dats, Stencils Halos, Reductions
	OPS_block_index=0; OPS_block_max=0; OPS_dat_index=0; OPS_dat_max=0;
	OPS_halo_group_index=0; OPS_halo_group_max=0; OPS_halo_index=0; OPS_halo_max=0;
	OPS_reduction_index=0; OPS_reduction_max=0; OPS_stencil_index=0; OPS_stencil_max=0;
	OPS_block_list=NULL;
	OPS_stencil_list=NULL;
	OPS_halo_list=NULL;
	OPS_halo_group_list=NULL;
	OPS_reduction_list = NULL;
	

	// Checkpointing
	OPS_enable_checkpointing=0;
	OPS_checkpointing_time=0.0;
	ops_thread_offload = 0;
	ops_checkpoint_inmemory = 0;
	ops_lock_file = 0;
	backup_state=OPS_NONE;
	OPS_dat_ever_written = 0;
	OPS_dat_status=NULL;
	OPS_ranks_per_node=0;


	// Debugging
	OPS_curr_args = NULL;
	OPS_curr_name = NULL;

	//Diagnostics
	OPS_kern_max=0; OPS_kern_curr=0;
	OPS_kernels=NULL;
	
	//Tiling
	ops_enable_tiling = 0;
	ops_cache_size = 0;
	ops_tiling_mpidepth = -1;
	ops_tiled_halo_exchange_time=0.0;
	tiling_instance=NULL;
	checkpointing_instance=NULL;
  ops_loop_over_blocks = 0;
  ops_loop_over_blocks_predicate = NULL;
  ops_loop_over_blocks_condition = 0;

  //Batching
  ops_batch_size = 1;
  OPS_hybrid_layout = 0;

	//Other runtime configuration args
	for (int i = 0; i < OPS_MAX_DIM; i++) ops_force_decomp[i] = 0;
	OPS_realloc = 0;
	OPS_soa=0;
	OPS_diags=0;

	// CUDA & OpenCL
	OPS_hybrid_gpu=0; OPS_gpu_direct=0;
	OPS_block_size_x = 32;
	OPS_block_size_y = 4;
	OPS_block_size_z = 1;
	OPS_consts_h=NULL; OPS_consts_d=NULL; OPS_reduct_h=NULL; OPS_reduct_d=NULL;
	OPS_consts_bytes = 0; OPS_reduct_bytes = 0;
	OPS_cl_device=0;
	ops_halo_buffer = NULL;
	ops_halo_buffer_d = NULL;
	ops_halo_buffer_size = 0;
	OPS_gbl_changed = 1;
	OPS_gbl_prev = NULL;
	opencl_instance = NULL;

}

OPS_instance* OPS_instance::getOPSInstance() {
#pragma omp critical
	{
		OPS_instance **ptr;
#if defined(_OPENMP)
		ptr = &ops_instances[omp_get_thread_num()];
#else
		ptr = &ops_instances[0];
#endif

		if (*ptr == 0) *ptr = new OPS_instance();
        return *ptr;
    }
}

void OPS_instance::destroyOPSInstance() {
#pragma omp critical
	{
		OPS_instance **ptr;
#if defined(_OPENMP)
		ptr = &ops_instances[omp_get_thread_num()];
#else
		ptr = &ops_instances[0];
#endif

		if (*ptr != 0) {
			delete *ptr;
			*ptr = 0;
		}
	}
}
