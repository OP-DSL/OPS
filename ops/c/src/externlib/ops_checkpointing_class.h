#pragma once 

#include <string>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class OPS_instance_checkpointing {
public:
  OPS_instance_checkpointing();
// checkpoint and execution progress information
  int ops_call_counter ;
  int ops_backup_point_g ;
  int ops_best_backup_point_g ;
  size_t ops_best_backup_point_size ;
  double ops_checkpoint_interval ;
  double ops_last_checkpoint ;
  bool ops_pre_backup_phase ;
  bool ops_duplicate_backup ;
  int ops_loop_max ;
  int *ops_loops_hashmap ;
  int OPS_chk_red_size ;
  int OPS_chk_red_offset_g ;
  char *OPS_chk_red_storage_g ;

  hsize_t OPS_partial_buffer_size ;
  char *OPS_partial_buffer ;

  int OPS_checkpointing_payload_nbytes_g ;
  char *OPS_checkpointing_payload_g ;

  int ops_reduction_counter ;
  int ops_sync_frequency ;
  double ops_reduction_avg_time ;

  int ops_checkpointing_options ;

// Timing
  double ops_chk_write ;
  double ops_chk_dup ;
  double ops_chk_save ;

// file managment
  std::string filename;
  std::string filename_dup;
  hid_t file;
  hid_t file_dup;
  herr_t status;

  FILE *diagf;
  int diagnostics ;

  char *params;

  ops_strat_data *ops_strat;

};
#endif
