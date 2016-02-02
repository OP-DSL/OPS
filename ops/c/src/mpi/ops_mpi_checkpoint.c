
#include <ops_lib_core.h>
#include <ops_mpi_core.h>

#ifdef CHECKPOINTING
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

#include <hdf5.h>
#include <hdf5_hl.h>

#include <mpi.h>
extern hid_t file_id_in;
extern MPI_Comm OPS_MPI_HDF5_WORLD;
extern sub_block_list *OPS_sub_block_list;// pointer to list holding sub-block
                                 // geometries
extern sub_dat_list *OPS_sub_dat_list;// pointer to list holding sub-dat
                               // details
extern ops_checkpointing_options;
void ops_fetch_dat_hdf5_file_internal(ops_dat dat, char const *file_name, int created);

hid_t plist_id;     //property list identifier

void ops_checkpoint_mpi_open(const char *file_name) {
  if (OPS_block_index > 1) {
    ops_printf("error, MPI I/O checkpoitning does not support multi-block yet\n");
    exit(-1);
  }

  sub_block *sb = OPS_sub_block_list[0];
  //create new communicator
  int my_rank, comm_size;
  //use the communicator for MPI procs holding this block
  MPI_Comm_dup(sb->comm1, &OPS_MPI_HDF5_WORLD);

  //MPI variables
  MPI_Info info  = MPI_INFO_NULL;

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    ops_printf("File %s does not exist .... creating file\n", file_name);
    MPI_Barrier(OPS_MPI_HDF5_WORLD);
    if (ops_is_root()) {
      FILE *fp; fp = fopen(file_name, "w");
      fclose(fp);
    }
    //Create a new file collectively and release property list identifier.
    file_id_in = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id_in);
  }

  file_id_in = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

}

void ops_checkpoint_mpi_close() {
  H5Pclose(plist_id);
  H5Fclose(file_id_in);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
}

typedef struct {
  int ops_backup_point;
  int ops_best_backup_point;
  char *reduction_state;
  int reduction_state_size;
  char *OPS_checkpointing_payload;
  int OPS_checkpointing_payload_nbytes;
  int OPS_chk_red_offset;
  char *OPS_chk_red_storage;
} ops_checkpoint_inmemory_control;

extern ops_checkpoint_inmemory_control ops_inm_ctrl;

void ops_checkpoint_mpi_ctrldump(hid_t file_out) {

  hid_t group_id;      //group identifier
  if(H5Lexists(file_id_in, "ctrlvars", H5P_DEFAULT) == 0) {
      if (OPS_diags>5) ops_printf("ctrlvars do not exist in file... creating ops_block\n");
      //create group - ops_block
      group_id = H5Gcreate(file_id_in, "ctrlvars", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Gclose(group_id);
  }

  //open existing group -- an ops_block is a group
  group_id = H5Gopen2(file_id_in, "ctrlvars", H5P_DEFAULT);

  //attach attributes to block
  H5LTset_attribute_int(file_id_in, "ctrlvars", "ops_backup_point", &ops_inm_ctrl.ops_backup_point, 1);
  H5LTset_attribute_int(file_id_in, "ctrlvars", "ops_best_backup_point", &ops_inm_ctrl.ops_best_backup_point, 1);
  H5LTset_attribute_char(file_id_in, "ctrlvars", "reduction_state", ops_inm_ctrl.reduction_state, ops_inm_ctrl.reduction_state_size);
  free(ops_inm_ctrl.reduction_state); ops_inm_ctrl.reduction_state=NULL;
  if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    H5LTset_attribute_char(file_id_in, "ctrlvars", "OPS_checkpointing_payload", ops_inm_ctrl.OPS_checkpointing_payload, ops_inm_ctrl.OPS_checkpointing_payload_nbytes);
  }
  H5LTset_attribute_int(file_id_in, "ctrlvars", "OPS_chk_red_offset", &ops_inm_ctrl.OPS_chk_red_offset, 1);
  H5LTset_attribute_char(file_id_in, "ctrlvars", "OPS_chk_red_storage", &ops_inm_ctrl.OPS_chk_red_storage, ops_inm_ctrl.OPS_chk_red_offset);

  H5Gclose(group_id);
}

void ops_checkpoint_mpi_save_partial(ops_dat dat, hid_t outfile, int size, int *saved_range, char *data) {
  printf("partial saving with MPI I/O not supported currently\n");
  exit(-1);
}

void ops_checkpoint_mpi_save_full(ops_dat dat, hid_t outfile, int size, char *data) {
  ops_dat_core dat_cpy_core = *dat;
  ops_dat dat_cpy = &dat_cpy_core;
  dat_cpy->data = data;
  ops_fetch_dat_hdf5_file_internal(dat_cpy, "", 1);
}

#endif