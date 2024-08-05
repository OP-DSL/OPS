#define OPS_CPP_API
#define OPS_INTERNAL_API
#include <ops_lib_core.h>
#include <unistd.h>

#include <atomic>

// The one and only static, non-thread safe OPS_instance which is 
// read and set by getOPSInstance only.  I would very much want to 
// make this file scope only, however MPI backend currently needs a 
// hack/back-door to set this var.  So until that is resolved, it 
// stays here
OPS_instance *global_ops_instance = NULL;

namespace {


   // The number of OPS_instances in flight, set to zero at process load
   std::atomic< int32_t> numInstances(0);


   void incrementInstanceCount()
   {
     numInstances.fetch_add(1);
   }

   int getNumInstances() 
   {
      return numInstances.fetch_add(0);
   }

}


// Static OPS_instance methods

OPS_instance* OPS_instance::getOPSInstance() {
       if (global_ops_instance == NULL) global_ops_instance = new OPS_instance();
       return global_ops_instance;
}

int OPS_instance::numInstances()
{
   return getNumInstances();
}

void OPS_instance::set_ostream(std::ostream &s) {
  _ostream = &s;
}

void OPS_instance::init_globals() {

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
	ops_user_halo_exchanges_time = 0.0;

	char default_paths[1024];
	strcpy(&default_paths[0],"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj;/sys/devices/virtual/powercap/intel-rapl/intel-rapl:1/energy_uj");
	char default_dram_paths[1024];
	strcpy(&default_dram_paths[0],"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj;/sys/devices/virtual/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj");
	char* env_paths = getenv("RAPL_PATH");
	if (!env_paths) {
		env_paths = default_paths;
	}
	char* env_dram_paths = getenv("RAPL_DRAM_PATH");
	if (!env_dram_paths) {
		env_dram_paths = default_dram_paths;
	}

	//concatenate the paths
	char* paths = (char*)ops_malloc(sizeof(char)*(strlen(env_paths)+strlen(env_dram_paths)+2));
	char* paths2 = (char*)ops_malloc(sizeof(char)*(strlen(env_paths)+strlen(env_dram_paths)+2));
	strcpy(paths, env_paths);
	strcat(paths, ";");
	strcat(paths, env_dram_paths);
	strcpy(paths2, paths);

	ops_energy_paths_count = 0;
    if (paths) {
		// Count the number of paths
		int num_paths = 0;
		char* path = strtok(paths, ";");
		while (path != NULL) {
			num_paths++;
			path = strtok(NULL, ";");
		}

		// Read the paths and initial energies
		ops_energy_paths_count = num_paths;
		ops_energy_paths = (char**)ops_malloc(sizeof(char*)*num_paths);
		ops_energy_counters = (long long*)ops_malloc(sizeof(long long)*num_paths);

		num_paths = 0;
		path = strtok(paths2, ";");
		while (path != NULL) {
			ops_energy_paths[num_paths] = strdup(path); // Duplicate the string for safekeeping
			//check if path exists
			if (access(path, R_OK) == -1) {
				if (env_paths != default_paths)
					ops_printf("Error: RAPL path %s does not exist or does not have the right permissions. Skipping.\n", path);
				ops_energy_paths[num_paths] = NULL;
			} else {
				// Read initial energy
				FILE* file = fopen(path, "r");
				if (file == NULL) {
					if (env_paths != default_paths)
						ops_printf("Error: Could not open RAPL path %s. Skipping.\n", path);
				} else {
					fscanf(file, "%lld", &ops_energy_counters[num_paths]);
					fclose(file);
				}
			}
			num_paths++;
			path = strtok(NULL, ";");
		}
    }



  ops_message_count = 0;
  ops_message_size = 0;
  ops_reduction_time = 0.0;
	
	//Tiling
	ops_enable_tiling = 0;
	ops_cache_size = 0;
	ops_tiling_mpidepth = -1;
	ops_tiled_halo_exchange_time=0.0;
	tiling_instance=NULL;
	checkpointing_instance=NULL;
	tilesize_x=-1;
	tilesize_y=-1;
	tilesize_z=-1;

	//Other runtime configuration args
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

	is_initialised = 1;
	char buf[20];
	int points[OPS_MAX_DIM] = {0};
	for (int i = 0; i < OPS_MAX_DIM; i++) {
		snprintf(buf, 20, "OPS_internal_0_%d\n", i+1);
		OPS_internal_0[i] = this->decl_stencil(i+1, 1, points, buf);
	}
}

OPS_instance::OPS_instance(const int argc, const char * const argv[], const int diags_level, std::ostream &s) {
  incrementInstanceCount();
  _ostream = &s;
  this->init_globals();
  _ops_init(this, argc, argv, diags_level);
}

OPS_instance::OPS_instance() {
   incrementInstanceCount();
  _ostream = &std::cout;
	this->init_globals();
}

ops_block OPS_instance::decl_block(int dims, const char * name) {
  return _ops_decl_block(this, dims, name);
}

ops_reduction OPS_instance::decl_reduction_handle(int size, const char *type,
    const char *name) {
  return _ops_decl_reduction_handle(this, size, type, name);
}

ops_stencil OPS_instance::decl_stencil(int dims, int points, int *stencil,
    char const *name) {
  return _ops_decl_stencil(this, dims, points, stencil, name);
}

ops_stencil OPS_instance::decl_strided_stencil(int dims, int points, int *sten,
    int *stride, char const *name) {
  return _ops_decl_strided_stencil(this, dims, points, sten, stride, name);
}
ops_stencil OPS_instance::decl_restrict_stencil( int dims, int points, int *sten,
    int *stride, char const * name) {
  return _ops_decl_restrict_stencil(this, dims, points, sten, stride, name);
}
ops_stencil OPS_instance::decl_prolong_stencil( int dims, int points, int *sten,
    int *stride, char const * name) {
  return _ops_decl_prolong_stencil(this, dims, points, sten, stride, name);
}

ops_halo OPS_instance::decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
    int *to_base, int *from_dir, int *to_dir) {
  return _ops_decl_halo(this, from, to, iter_size, from_base, to_base, from_dir, to_dir);
}
ops_halo_group OPS_instance::decl_halo_group(int nhalos, ops_halo *halos) {
  return _ops_decl_halo_group(this, nhalos, halos);
}

void OPS_instance::reset_power_counters() {
  _ops_reset_power_counters(this);
}

void OPS_instance::diagnostic_output() {
  _ops_diagnostic_output(this);
}
void OPS_instance::timing_output(std::ostream &stream) {
  _ops_timing_output(this, stream);
}
void OPS_instance::timing_output_stdout() {
  _ops_timing_output(this, std::cout);
}

int OPS_instance::is_root() {
  return _ops_is_root(this);
}

void OPS_instance::partition(const char *routine) {
  _ops_partition(this, routine);
}

void OPS_instance::partition(const char *routine, std::map<std::string, void*>& opts) {
  _ops_partition(this, routine, opts);
}

void OPS_instance::exit() {
  _ops_exit(this);
}






//Forwarding calls for ops_dat
ops_dat_core::~ops_dat_core() {_ops_free_dat(this);}
void ops_dat_core::print_to_txtfile(const char *file_name) {ops_print_dat_to_txtfile(this, file_name);}
void ops_dat_core::get_data() {ops_get_data(this);}
int  ops_dat_core::get_local_npartitions() { return ops_dat_get_local_npartitions(this); }
void ops_dat_core::get_extents(int part, int *disp, int *size2) { ops_dat_get_extents(this, part, disp, size2); }
void ops_dat_core::get_raw_metadata(int part, int *disp, int *size2, int *stride2, int *d_m2, int *d_p2) { ops_dat_get_raw_metadata(this, part, disp, size2, stride2, d_m2, d_p2); }
char* ops_dat_core::get_raw_pointer(int part, ops_stencil stencil, ops_memspace *memspace) { return ops_dat_get_raw_pointer(this, part, stencil, memspace); }
void ops_dat_core::release_raw_data(int part, ops_access acc) { ops_dat_release_raw_data(this, part, acc); }
void ops_dat_core::release_raw_data(int part, ops_access acc, ops_memspace *memspace) { ops_dat_release_raw_data_memspace(this, part, acc, memspace); }
void ops_dat_core::fetch_data(int part, char *data2, ops_memspace memspace) {ops_dat_fetch_data_memspace(this, part, data2, memspace);}
void ops_dat_core::fetch_data_slab(int part, char *data2, int *range, ops_memspace memspace) {ops_dat_fetch_data_slab_memspace(this, part, data2, range, memspace);}
void ops_dat_core::set_data(int part, char *data2, ops_memspace memspace) { ops_dat_set_data_memspace(this, part, data2, memspace); }
void ops_dat_core::set_data_slab(int part, char *data2, int *range, ops_memspace memspace) { ops_dat_set_data_slab_memspace(this, part, data2, range, memspace); }
size_t ops_dat_core::get_slab_extents(int part, int *disp, int *size2, int *slab) {return ops_dat_get_slab_extents(this, part, disp, size2, slab);}
int ops_dat_core::get_global_npartitions() { return ops_dat_get_global_npartitions(this); }

void ops_halo_group_core::halo_transfer() {ops_halo_transfer(this);}
