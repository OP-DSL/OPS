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
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
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

/** @brief ops checkpointing library function declarations
  * @author Istvan Reguly
  * @details function implementations for checkpointing
  */

#ifdef CHECKPOINTING
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include <ops_lib_core.h>
#include <stdio.h>

ops_backup_state backup_state = OPS_NONE;
char *OPS_dat_ever_written = NULL;
ops_checkpoint_types *OPS_dat_status = NULL;
double OPS_checkpointing_time = 0.0;

#ifdef CHECKPOINTING

#ifndef defaultTimeout
#define defaultTimeout 1.0
#endif

#ifdef __cplusplus
extern "C" {
#endif

//Internal function definitions
void ops_download_dat(ops_dat dat);
void ops_upload_dat(ops_dat dat);
void ops_checkpointing_filename(const char *file_name, char *filename_out);
void ops_checkpointing_calc_range(ops_dat dat, const int *range, int *saved_range);

//checkpoint and execution progress information
int ops_call_counter = 0;
int ops_backup_point = -1;
double checkpoint_interval = -1;
double last_checkpoint = -1;
bool pre_backup = false;
int ops_loop_max = 0;
int *hashmap = NULL;
int OPS_chk_red_size = 0;
int OPS_chk_red_offset = 0;
char *OPS_chk_red_storage = NULL;

int OPS_partial_buffer_size = 0;
char *OPS_partial_buffer = NULL;

int OPS_checkpointing_payload_nbytes = 0;
char *OPS_checkpointing_payload = NULL;

int ops_reduction_counter = 0;
int ops_sync_frequency = 50;
double ops_reduction_avg_time = 0.0;

int ops_checkpointing_options = 0;

//file managment
char filename[100];
hid_t file;
herr_t status;

FILE *diagf;
int diagnostics = 0;

#define check_hdf5_error(err)           __check_hdf5_error      (err, __FILE__, __LINE__)
void __check_hdf5_error(herr_t err, const char *file, const int line) {
  if (err < 0) {
    printf("%s(%i) : OP2_HDF5_error() Runtime API error %d.\n", file, line, (int)err);
      exit(-1);
  }
}

bool file_exists(const char * file_name) {
    if (FILE * file = fopen(file_name, "r")) {
        fclose(file);
        return true;
    }
    return false;
}

void save_dat(ops_dat dat) {
  OPS_dat_status[dat->index] = OPS_SAVED;
  hsize_t dims[1];
  if (dat->dirty_hd == 2) ops_download_dat(dat);
  dims[0] = dat->dim;
  for (int d = 0; d < dat->block->dims; d++) dims[0] *= dat->size[d];

  if (strcmp(dat->type,"int")==0) {
    check_hdf5_error(H5LTmake_dataset(file, dat->name, 1, dims, H5T_NATIVE_INT, dat->data));
  } else if (strcmp(dat->type,"float")==0) {
    check_hdf5_error(H5LTmake_dataset(file, dat->name, 1, dims, H5T_NATIVE_FLOAT, dat->data));
  } else if (strcmp(dat->type,"double")==0) {
    check_hdf5_error(H5LTmake_dataset(file, dat->name, 1, dims, H5T_NATIVE_DOUBLE, dat->data));
  } else {
    printf("Unsupported data type in ops_arg_dat() %s\n", dat->name);
    exit(-1);
  }

  if (OPS_diags>4) printf("Backed up %s\n", dat->name);
}

void ops_pack_chk(const char *__restrict src, char *__restrict dest, const int count, const int blocklength, const int stride) {
  for (unsigned int i = 0; i < count; i ++) {
    memcpy(dest, src, blocklength);
    src += stride;
    dest += blocklength;
  }
}

void ops_unpack_chk(char *__restrict dest, const char *__restrict src, const int count, const int blocklength, const int stride) {
  for (unsigned int i = 0; i < count; i ++) {
    memcpy(dest, src, blocklength);
    src += blocklength;
    dest += stride;
  }
}

void save_dat_partial(ops_dat dat, int *range) {
  OPS_dat_status[dat->index] = OPS_SAVED;
  if (dat->dirty_hd == 2) ops_download_dat(dat);
  int saved_range[2*OPS_MAX_DIM] = {0};
  int prod[OPS_MAX_DIM+1];
  prod[0] = 1;

  //Calculate saved range (based on full size and the range where it is modified)
  ops_checkpointing_calc_range(dat, range, saved_range);

  for (int d = 0; d < dat->block->dims; d++) {
    prod[d+1] = prod[d]*dat->size[d];
  }
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {saved_range[2*d] = 0; saved_range[2*d+1] = 1;}

  //Calculate access pattern to sides (depth 1)
  int count[OPS_MAX_DIM], block_length[OPS_MAX_DIM], stride[OPS_MAX_DIM];
  for (int d = 0; d < dat->block->dims; d++) {
    count[d] = prod[dat->block->dims]/prod[d+1];
    block_length[d] = 1*prod[d]*dat->elem_size;
    stride[d] = prod[d+1]*dat->elem_size;
  }

  //Calculate total size stored (redundant at edges)
  hsize_t dims[1] = {0};
  for (int d = 0; d < dat->block->dims; d++) {
    int depth_before = saved_range[2*d];
    int depth_after = dat->size[d]-saved_range[2*d+1];
    dims[0] += (depth_before+depth_after)*block_length[d]*count[d];
  }

  //if too much redundancy, just do the usual save
  if (dims[0] >= prod[dat->block->dims]*dat->elem_size) {
    save_dat(dat);
    return;
  }
  if (dims[0] == 0) return;

  if (OPS_partial_buffer_size < dims[0]) {
    OPS_partial_buffer_size = 2*dims[0];
    OPS_partial_buffer = (char*)realloc(OPS_partial_buffer, OPS_partial_buffer_size*sizeof(char));
  }

  //Pack
  int offset = 0;
  for (int d = 0; d < dat->block->dims ; d++) {
    int depth_before = saved_range[2*d];
    if (depth_before)
      ops_pack_chk(dat->data, OPS_partial_buffer+offset, count[d], depth_before*block_length[d], stride[d]);
    offset += depth_before*block_length[d]*count[d];
    int depth_after = dat->size[d]-saved_range[2*d+1];
    int i4 = (prod[d+1]/prod[d] - (depth_after)) * prod[d] * dat->elem_size;
    if (depth_after)
      ops_pack_chk(dat->data + i4, OPS_partial_buffer+offset, count[d], depth_after*block_length[d], stride[d]);
    offset += depth_after*block_length[d]*count[d];
  }

  dims[0] = dims[0] / (dat->elem_size/dat->dim);
  if (strcmp(dat->type,"int")==0) {
    check_hdf5_error(H5LTmake_dataset(file, dat->name, 1, dims, H5T_NATIVE_INT, OPS_partial_buffer));
  } else if (strcmp(dat->type,"float")==0) {
    check_hdf5_error(H5LTmake_dataset(file, dat->name, 1, dims, H5T_NATIVE_FLOAT, OPS_partial_buffer));
  } else if (strcmp(dat->type,"double")==0) {
    check_hdf5_error(H5LTmake_dataset(file, dat->name, 1, dims, H5T_NATIVE_DOUBLE, OPS_partial_buffer));
  } else {
    printf("Unsupported data type in ops_arg_dat() %s\n", dat->name);
    exit(-1);
  }

  char buf[50];
  sprintf(buf, "%s_saved_range", dat->name);
  dims[0] = 2*OPS_MAX_DIM;
  check_hdf5_error(H5LTmake_dataset(file, buf, 1, dims, H5T_NATIVE_INT, saved_range));


  if (OPS_diags>4) printf("Backed up %s (partial)\n", dat->name);
}

void ops_restore_dataset(ops_dat dat) {
  if (!H5LTfind_dataset(file, dat->name)) return;
  char buf[50];
  sprintf(buf, "%s_saved_range", dat->name);
  hsize_t dims[1] = {0};
  dims[0] = 2*OPS_MAX_DIM;
  //if no range specified, just read back the entire dataset
  if (!H5LTfind_dataset(file, buf)) {
    dims[0] = dat->elem_size;
    for (int d = 0; d < dat->block->dims; d++) dims[0] *= dat->size[d];
    if (strcmp(dat->type,"int")==0) {
        check_hdf5_error(H5LTread_dataset (file,  dat->name, H5T_NATIVE_INT, dat->data));
    } else if (strcmp(dat->type,"float")==0) {
      check_hdf5_error(H5LTread_dataset (file,  dat->name, H5T_NATIVE_FLOAT, dat->data));
    } else if (strcmp(dat->type,"double")==0) {
      check_hdf5_error(H5LTread_dataset (file,  dat->name, H5T_NATIVE_DOUBLE, dat->data));
    } else {
      printf("Unsupported data type in ops_arg_dat() %s\n",  dat->name);
      exit(-1);
    }
    dat->dirty_hd = 1;
    if (OPS_diags>4) printf("Restored %s\n", dat->name);
  } else { //if a range is specified,
    int saved_range[OPS_MAX_DIM*2];
    check_hdf5_error(H5LTread_dataset (file,  buf, H5T_NATIVE_INT, saved_range));
    if (dat->dirty_hd == 2) ops_download_dat(dat);
    int prod[OPS_MAX_DIM+1];
    prod[0] = 1;
    for (int d = 0; d < dat->block->dims; d++) prod[d+1] = prod[d]*dat->size[d];
    //Calculate access pattern to sides (depth 1)
    int count[OPS_MAX_DIM], block_length[OPS_MAX_DIM], stride[OPS_MAX_DIM];
    for (int d = 0; d < dat->block->dims; d++) {
      count[d] = prod[dat->block->dims]/prod[d+1];
      block_length[d] = 1*prod[d]*dat->elem_size;
      stride[d] = prod[d+1]*dat->elem_size;
    }

    //Calculate total size stored (redundant at edges)
    hsize_t dims[1] = {0};
    for (int d = 0; d < dat->block->dims; d++) {
      int depth_before = saved_range[2*d];
      int depth_after = dat->size[d]-saved_range[2*d+1];
      dims[0] += (depth_before+depth_after)*block_length[d]*count[d];
    }
    if (dims[0] == 0) return;

    if (OPS_partial_buffer_size < dims[0]) {
      OPS_partial_buffer_size = 2*dims[0];
      OPS_partial_buffer = (char*)realloc(OPS_partial_buffer, OPS_partial_buffer_size*sizeof(char));
    }

    dims[0] = dims[0] / (dat->elem_size/dat->dim);
    if (strcmp(dat->type,"int")==0) {
        check_hdf5_error(H5LTread_dataset (file,  dat->name, H5T_NATIVE_INT, OPS_partial_buffer));
    } else if (strcmp(dat->type,"float")==0) {
      check_hdf5_error(H5LTread_dataset (file,  dat->name, H5T_NATIVE_FLOAT, OPS_partial_buffer));
    } else if (strcmp(dat->type,"double")==0) {
      check_hdf5_error(H5LTread_dataset (file,  dat->name, H5T_NATIVE_DOUBLE, OPS_partial_buffer));
    } else {
      printf("Unsupported data type in ops_arg_dat() %s\n",  dat->name);
      exit(-1);
    }

    //Unpack
    int offset = 0;
    for (int d = 0; d < dat->block->dims ; d++) {
      int depth_before = saved_range[2*d];
      if (depth_before)
        ops_unpack_chk(dat->data, OPS_partial_buffer+offset, count[d], depth_before*block_length[d], stride[d]);
      offset += depth_before*block_length[d]*count[d];
      int depth_after = dat->size[d]-saved_range[2*d+1];
      int i4 = (prod[d+1]/prod[d] - (depth_after)) * prod[d] * dat->elem_size;
      if (depth_after)
        ops_unpack_chk(dat->data + i4, OPS_partial_buffer+offset, count[d], depth_after*block_length[d], stride[d]);
      offset += depth_after*block_length[d]*count[d];
    }

    dat->dirty_hd = 1;
    if (OPS_diags>4) printf("Restored %s (partial)\n", dat->name);
  }
}


bool ops_checkpointing_initstate() {
    if (!file_exists(filename)) {
    backup_state = OPS_BACKUP_GATHER;
    ops_printf("//\n// OPS Checkpointing -- Backup mode\n//\n");

    for (int i = 0; i < OPS_dat_index; i++) {
      OPS_dat_status[i] = OPS_UNDECIDED;
      OPS_dat_ever_written[i] = 0;
    }
    return false;
  } else {
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    backup_state = OPS_BACKUP_LEADIN;
    ops_printf("//\n// OPS Checkpointing -- Restore mode\n//\n");

    if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      //we are not restoring anything during lead-in and will be resuming execution the first time we encounter the API call
    } else {
      double cpu,t1,t2;
      ops_timers_core(&cpu, &t1);
      //read backup point
      check_hdf5_error(H5LTread_dataset (file,  "ops_backup_point", H5T_NATIVE_INT, &ops_backup_point));

      //loading of datasets is postponed until we reach the restore point

      //restore reduction storage
      check_hdf5_error(H5LTread_dataset (file,  "OPS_chk_red_offset", H5T_NATIVE_INT, &OPS_chk_red_offset));
      OPS_chk_red_storage = (char*)malloc(OPS_chk_red_offset*sizeof(char));
      OPS_chk_red_size = OPS_chk_red_offset;
      OPS_chk_red_offset = 0;
      check_hdf5_error(H5LTread_dataset (file,  "OPS_chk_red_storage", H5T_NATIVE_CHAR, OPS_chk_red_storage));
      ops_timers_core(&cpu, &t2);
      OPS_checkpointing_time += t2-t1;
    }
    return true;
  }
}

/**
* Initialises checkpointing using the given filename
*/
bool ops_checkpointing_init(const char *file_name, double interval, int options) {
  if (!OPS_enable_checkpointing) return false;
  checkpoint_interval = interval;
  if (interval < defaultTimeout*2.0) interval = defaultTimeout*2.0; //WHAT SHOULD THIS BE? - the time it takes to back up everything
  double cpu;
  ops_timers_core(&cpu, &last_checkpoint);
  ops_checkpointing_filename(file_name, filename);


  OPS_dat_ever_written = (char*)malloc(OPS_dat_index * sizeof(char));
  OPS_dat_status = (ops_checkpoint_types*)malloc(OPS_dat_index * sizeof(ops_checkpoint_types));

  if (diagnostics) {
    diagf = fopen("checkp_diags.txt","w");
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      fprintf(diagf, "%s;%d;%d;%s\n",dat->name,dat->size[0], dat->size[1], dat->type);
    }
    fprintf(diagf, "\n");
  }

  ops_checkpointing_options = options;
  if (options == 0) {
    return ops_checkpointing_initstate();
  } else if (options & OPS_CHECKPOINT_INITPHASE) {
    if (file_exists(filename))
      return true;
    else
      return false;
  }
}

void ops_checkpointing_initphase_done() {
  if (!OPS_enable_checkpointing) return;
  if (backup_state == OPS_NONE) {
    ops_checkpointing_initstate();
  }
}

void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist) {
  if (backup_state == OPS_BACKUP_GATHER) {
    if (pre_backup) {
      double cpu,t1,t2;
      ops_timers_core(&cpu, &t1);
      //increment call counter (as if we began checkpointing at the next loop) and save the list of datasets + aux info
      ops_call_counter++;

      ops_backup_point = ops_call_counter;
      if (file_exists(filename)) remove(filename);
      //where we start backing up stuff
      file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      //write datasets
      for (int i = 0; i < ndats; i++)
          save_dat(datlist[i]);


      //write control variables and all initialized reduction handles
      hsize_t dims[1];
      dims[0] = 1;
      check_hdf5_error(H5LTmake_dataset(file, "ops_backup_point", 1, dims, H5T_NATIVE_INT, &ops_backup_point));
      int total_size = 0;
      for (int i = 0; i < OPS_reduction_index; i++)
        if (OPS_reduction_list[i]->initialized == 1)
          total_size+=OPS_reduction_list[i]->size;
      char *reduction_state = (char*)malloc(total_size*sizeof(char));
      total_size = 0;
      for (int i = 0; i < OPS_reduction_index; i++) {
        if (OPS_reduction_list[i]->initialized == 1) {
          memcpy(&reduction_state[total_size], OPS_reduction_list[i]->data, OPS_reduction_list[i]->size);
          total_size+=OPS_reduction_list[i]->size;
        }
      }
      dims[0] = total_size;
      check_hdf5_error(H5LTmake_dataset(file, "reduction_state", 1, dims, H5T_NATIVE_CHAR, reduction_state));
      free(reduction_state);
      if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
        dims[0] = OPS_checkpointing_payload_nbytes;
        check_hdf5_error(H5LTmake_dataset(file, "OPS_checkpointing_payload", 1, dims, H5T_NATIVE_CHAR, OPS_checkpointing_payload));
      }

      //write reduction history
      dims[0] = 1;
      check_hdf5_error(H5LTmake_dataset(file, "OPS_chk_red_offset", 1, dims, H5T_NATIVE_INT, &OPS_chk_red_offset));
      dims[0] = OPS_chk_red_offset;
      check_hdf5_error(H5LTmake_dataset(file, "OPS_chk_red_storage", 1, dims, H5T_NATIVE_CHAR, OPS_chk_red_storage));

      check_hdf5_error(H5Fclose(file));
      //finished backing up, reset everything, prepare to be backed up at a later point
      backup_state = OPS_BACKUP_GATHER;
      pre_backup = false;
      ops_call_counter--;
      ops_timers_core(&cpu, &t2);
      OPS_checkpointing_time += t2-t1;
      if (OPS_diags>1) ops_printf("Checkpoint created (manual datlist) in %g seconds\n", t2-t1);
    }
  } else if (backup_state == OPS_BACKUP_LEADIN) {
    // do nothing, will get triggered anyway
  }
}

bool ops_checkpointing_fastfw(int nbytes, char *payload) {
  if (backup_state == OPS_BACKUP_GATHER) {
    if (pre_backup) {
      backup_state = OPS_BACKUP_BEGIN;
      pre_backup = false;
      OPS_checkpointing_payload_nbytes = nbytes;
      OPS_checkpointing_payload = payload;
    }
  } else if (backup_state == OPS_BACKUP_LEADIN) {
    backup_state = OPS_BACKUP_GATHER;
    double cpu, now,t2;
    ops_timers_core(&cpu, &now);
    last_checkpoint = now;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_restore_dataset(item->dat);
      OPS_dat_status[item->dat->index] = OPS_UNDECIDED;
    }
    check_hdf5_error(H5LTread_dataset (file,  "OPS_checkpointing_payload", H5T_NATIVE_CHAR, payload));
    check_hdf5_error(H5Fclose(file));
    ops_timers_core(&cpu, &t2);
    OPS_checkpointing_time += t2-now;
    if (OPS_diags>1) ops_printf("\nRestored at fast-forward point (in %g seconds), continuing normal execution...\n", t2-now);
    return true;
  }
  return false;
}

bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist, int nbytes, char *payload) {
  if (backup_state == OPS_BACKUP_GATHER) {
    if (pre_backup) {
      OPS_checkpointing_payload_nbytes = nbytes;
      OPS_checkpointing_payload = payload;
      ops_checkpointing_manual_datlist(ndats, datlist);
    }
  } else if (backup_state == OPS_BACKUP_LEADIN) {
    ops_checkpointing_fastfw(nbytes,payload);
    return true;
  }
  return false;
}

void ops_checkpointing_reduction(ops_reduction red) {
  double t1,t2,cpu;
  ops_timers_core(&cpu, &t1);
  if (diagnostics && OPS_enable_checkpointing) {
    fprintf(diagf, "reduction;red->name\n");
  }
  if (OPS_chk_red_offset + red->size > OPS_chk_red_size && !(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    OPS_chk_red_size *=2;
    OPS_chk_red_size = MAX(OPS_chk_red_size, 100*red->size);
    OPS_chk_red_storage = (char *)realloc(OPS_chk_red_storage, OPS_chk_red_size * sizeof(char));
  }
  if (backup_state == OPS_BACKUP_LEADIN) {
    if (!(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      memcpy(red->data, &OPS_chk_red_storage[OPS_chk_red_offset], red->size);
      OPS_chk_red_offset += red->size;
    }
    ops_timers_core(&cpu, &t2);
    OPS_checkpointing_time += t2-t1;
    return;
  }
  ops_timers_core(&cpu, &t2);
  OPS_checkpointing_time += t2-t1;

  ops_execute_reduction(red);

  ops_timers_core(&cpu, &t1);
  if (backup_state == OPS_BACKUP_GATHER || backup_state == OPS_BACKUP_IN_PROCESS) {
    if (!(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      memcpy(&OPS_chk_red_storage[OPS_chk_red_offset], red->data, red->size);
      OPS_chk_red_offset += red->size;
    }
    ops_reduction_counter++;

    if (ops_reduction_counter % ops_sync_frequency == 0) {
      double cpu, now;
      ops_timers_core(&cpu, &now);
      double timing[2] = {now-last_checkpoint, (double)(now-last_checkpoint > checkpoint_interval)};
      ops_arg temp;
      temp.argtype = OPS_ARG_GBL;
      temp.acc = OPS_MAX;
      temp.data = (char*)timing;
      temp.dim = 2;//*sizeof(double);
      ops_mpi_reduce_double(&temp, timing);
      ops_reduction_avg_time = timing[0];
      if (ops_reduction_avg_time < 0.1 * checkpoint_interval) ops_sync_frequency=ops_reduction_counter;
      //if (ops_reduction_avg_time > 0.3 * checkpoint_interval) ops_sync_frequency/=1.5;
      //ops_printf("ops_sync_frequency %d\n", ops_sync_frequency);
      if (timing[1] == 1.0) {
        ops_reduction_counter = 0;
        if (OPS_diags>4) ops_printf("\nIt's time to checkpoint...\n");
        last_checkpoint = now;
        if (pre_backup == true && !(ops_checkpointing_options & (OPS_CHECKPOINT_FASTFW | OPS_CHECKPOINT_MANUAL_DATLIST))) {
          if (OPS_diags>1) ops_printf("Double timeout for checkpointing forcing immediate begin\n");
          backup_state = OPS_BACKUP_BEGIN;
          pre_backup = false;
        } else
          pre_backup = true;
      }
    }
  }
  ops_timers_core(&cpu, &t2);
  OPS_checkpointing_time += t2-t1;
}

#define HASHSIZE 5000
unsigned op2_hash(const char *s)
{
  unsigned hashval;
  for (hashval = 0; *s != '\0'; s++)
    hashval = *s + 31 * hashval;
  return hashval % HASHSIZE;
}

bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range, const char *s) {
  if (backup_state == OPS_NONE) return true;
  int loop_id = 0;
  int hash = (int)op2_hash(s);
  for (; loop_id < ops_loop_max; loop_id++) {
    if (hashmap[loop_id] == hash || hashmap[loop_id] == -1) break;
  }
  //if (hashmap != NULL && hashmap[loop_id] != -1 && hashmap[loop_id] != hash) loop_id++;

  if (hashmap == NULL || loop_id >= ops_loop_max) { //if we ran out of space for storing per loop data, allocate some more
    //printf("Allocing more storage for loops: ops_loop_max = %d\n",ops_loop_max);
    ops_loop_max += 100;
    hashmap = (int *)realloc(hashmap, ops_loop_max*sizeof(int));
    for (int i = ops_loop_max-100; i < ops_loop_max; i++) {
      hashmap[i] = -1;
    }
  }
  hashmap[loop_id] = hash;
  //printf("Loop %s id %d\n", s, loop_id);
  return ops_checkpointing_before(args, nargs, range, loop_id);
}

void gather_statistics(ops_arg *args, int nargs, int loop_id, int *range) {}
bool should_backup(ops_arg *args, int nargs, int loop_id, int *range) {
  //Initial strategy: if this range is over a very small part (one fifth) of the dataset, then don't do a checkpoint here
  int i = 0;
  for (; i < nargs; i ++) if (args[i].argtype == OPS_ARG_DAT && args[i].dat->e_dat == 0) break;
  if (i == nargs) return false;
  int larger = 1;
  for (int d = 0; d < args[i].dat->block->dims; d++) {
    if ((double)(range[2*d+1]-range[2*d]) <= 1) larger = 0;
  }
  return (larger == 1);
}

/**
* Checkpointing utility function called right before the execution of the parallel loop itself.
*/
bool ops_checkpointing_before(ops_arg *args, int nargs, int *range, int loop_id) {
  if (backup_state == OPS_NONE) return true;
  if (diagnostics) {
    fprintf(diagf, "loop %d;%d;%d;%d;%d;%d\n",loop_id,nargs,range[0],range[1],range[2],range[3]);
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT) {
        fprintf(diagf, "dat;%s;%s;%s;%d;%d\n", args[i].dat->name, args[i].stencil->name, args[i].dat->type, args[i].acc,args[i].opt);
      } else if (args[i].argtype == OPS_ARG_GBL) {
        fprintf(diagf, "gbl;%d;%d\n", args[i].dim, args[i].acc);
      } else if (args[i].argtype == OPS_ARG_IDX) {
        fprintf(diagf, "idx\n");
      }
    }
  }

  double cpu,t1,t2;
  ops_timers_core(&cpu, &t1);

  ops_call_counter++;
  for (int i = 0; i < nargs; i++) { //flag variables that are touched (we do this every time it is called, may be a little redundant), should make a loop_id filter
    if (args[i].argtype == OPS_ARG_DAT && args[i].argtype != OPS_READ && args[i].opt == 1 ) OPS_dat_ever_written[args[i].dat->index] = true;
  }

  if (ops_call_counter == ops_backup_point
      && backup_state == OPS_BACKUP_LEADIN
      && !(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) backup_state = OPS_BACKUP_RESTORE;

  if (backup_state == OPS_BACKUP_GATHER) {
    gather_statistics(args, nargs, loop_id, range);
    if (pre_backup && should_backup(args, nargs, loop_id, range)
      && !(ops_checkpointing_options & (OPS_CHECKPOINT_FASTFW | OPS_CHECKPOINT_MANUAL_DATLIST))) {
      backup_state = OPS_BACKUP_BEGIN;
      pre_backup = false;
    }
  } else  if (backup_state == OPS_BACKUP_LEADIN) {
    return false;
  } else if (backup_state == OPS_BACKUP_RESTORE) {
    //this is the point where we do the switch from restore mode to computation mode
    backup_state = OPS_BACKUP_GATHER;
    double cpu, now;
    ops_timers_core(&cpu, &now);
    last_checkpoint = now;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_restore_dataset(item->dat);
      OPS_dat_status[item->dat->index] = OPS_UNDECIDED;
    }

    int total_size = 0;
    for (int i = 0; i < OPS_reduction_index; i++)
      if (OPS_reduction_list[i]->initialized == 1)
        total_size+=OPS_reduction_list[i]->size;
    char *reduction_state = (char*)malloc(total_size*sizeof(char));
    check_hdf5_error(H5LTread_dataset (file,  "reduction_state", H5T_NATIVE_CHAR, reduction_state));
    total_size = 0;
    for (int i = 0; i < OPS_reduction_index; i++) {
      if (OPS_reduction_list[i]->initialized == 1) {
        memcpy(OPS_reduction_list[i]->data, &reduction_state[total_size], OPS_reduction_list[i]->size);
        total_size+=OPS_reduction_list[i]->size;
      }
    }
    free(reduction_state);
    check_hdf5_error(H5Fclose(file));
    ops_timers_core(&cpu, &t2);
    if (OPS_diags>1) ops_printf("\nRestored in %g seconds, continuing normal execution...\n", t2-t1);
  }

  if (backup_state == OPS_BACKUP_BEGIN) {
    ops_backup_point = ops_call_counter;
    if (file_exists(filename)) remove(filename);
    //where we start backing up stuff
    //printf("Creating hdf5 file %s\n", filename);
    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    //write datasets
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT &&
        OPS_dat_ever_written[args[i].dat->index] &&
        OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
        args[i].acc != OPS_WRITE && args[i].opt == 1) {
        //write it to disk
        save_dat(args[i].dat);
      } else if (args[i].argtype == OPS_ARG_DAT &&
             OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
             args[i].acc == OPS_WRITE && args[i].opt == 1) {
          save_dat_partial(args[i].dat, range);
      }
    }

    //write control variables and all initialized reduction handles
    hsize_t dims[1];
    dims[0] = 1;
    check_hdf5_error(H5LTmake_dataset(file, "ops_backup_point", 1, dims, H5T_NATIVE_INT, &ops_backup_point));
    int total_size = 0;
    for (int i = 0; i < OPS_reduction_index; i++)
      if (OPS_reduction_list[i]->initialized == 1)
        total_size+=OPS_reduction_list[i]->size;
    char *reduction_state = (char*)malloc(total_size*sizeof(char));
    total_size = 0;
    for (int i = 0; i < OPS_reduction_index; i++) {
      if (OPS_reduction_list[i]->initialized == 1) {
        memcpy(&reduction_state[total_size], OPS_reduction_list[i]->data, OPS_reduction_list[i]->size);
        total_size+=OPS_reduction_list[i]->size;
      }
    }
    dims[0] = total_size;
    check_hdf5_error(H5LTmake_dataset(file, "reduction_state", 1, dims, H5T_NATIVE_CHAR, reduction_state));
    free(reduction_state);
    if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      dims[0] = OPS_checkpointing_payload_nbytes;
      check_hdf5_error(H5LTmake_dataset(file, "OPS_checkpointing_payload", 1, dims, H5T_NATIVE_CHAR, OPS_checkpointing_payload));
    }

    //write reduction history
    dims[0] = 1;
    check_hdf5_error(H5LTmake_dataset(file, "OPS_chk_red_offset", 1, dims, H5T_NATIVE_INT, &OPS_chk_red_offset));
    dims[0] = OPS_chk_red_offset;
    check_hdf5_error(H5LTmake_dataset(file, "OPS_chk_red_storage", 1, dims, H5T_NATIVE_CHAR, OPS_chk_red_storage));

    //Check if we are done
    backup_state = OPS_BACKUP_IN_PROCESS;
    bool done = true;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      if (OPS_dat_status[item->dat->index] == OPS_UNDECIDED && OPS_dat_ever_written[item->dat->index]) {
        done = false;
      }
    }
    if (done) backup_state = OPS_BACKUP_END;
  } else  if (backup_state == OPS_BACKUP_IN_PROCESS) {
    //when we have already begun backing up, but there are a few datasets that are undecided (whether or not they should be backed up)
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT &&
        OPS_dat_ever_written[args[i].dat->index] &&
        OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
        args[i].acc != OPS_WRITE && args[i].opt == 1) {
          save_dat(args[i].dat);
      } else if (args[i].argtype == OPS_ARG_DAT &&
             OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
               args[i].acc == OPS_WRITE && args[i].opt == 1) {
            save_dat_partial(args[i].dat, range);
      }
    }
    bool done = true;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      if (OPS_dat_status[item->dat->index] == OPS_UNDECIDED && OPS_dat_ever_written[item->dat->index]) {
        done = false;
      }
    }
    double cpu, now;
    ops_timers_core(&cpu, &now);
    //if there are no undecided datasets left, or we hit the timeout trying to decide upon some, lets finish up
    if (done || (now-last_checkpoint > defaultTimeout)) backup_state = OPS_BACKUP_END;
  }

  if (backup_state == OPS_BACKUP_END) {
    //either timed out or ended, if it's the former, back up everything left
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      if (OPS_dat_status[item->dat->index] == OPS_UNDECIDED && OPS_dat_ever_written[item->dat->index]) {
        save_dat(item->dat);
        if (OPS_diags>4) printf("Timeout, force saving %s\n", item->dat->name);
      }
    }

    check_hdf5_error(H5Fclose(file));
    if (OPS_diags>1) ops_printf("\nCheckpoint created\n");
    //finished backing up, reset everything, prepare to be backed up at a later point
    backup_state = OPS_BACKUP_GATHER;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      OPS_dat_status[item->dat->index] = OPS_UNDECIDED;
    }
  }
  ops_timers_core(&cpu, &t2);
  OPS_checkpointing_time += t2-t1;
  return true;
}

void ops_checkpointing_exit() {
  if (backup_state != OPS_NONE) {
    if (backup_state == OPS_BACKUP_IN_PROCESS) {
      check_hdf5_error(H5Fclose(file));
      remove(filename);
    }
    if (diagnostics) {
      fprintf(diagf, "FINISHED\n");
      fclose(diagf);
    }
    ops_call_counter = 0;
    ops_backup_point = -1;
    checkpoint_interval = -1;
    last_checkpoint = -1;
    pre_backup = false;
    backup_state = OPS_NONE;
    ops_loop_max = 0;
    free(hashmap); hashmap = NULL;
    free(OPS_dat_ever_written); OPS_dat_ever_written = NULL;
    free(OPS_dat_status); OPS_dat_status = NULL;
    OPS_partial_buffer_size = 0;
    free(OPS_partial_buffer); OPS_partial_buffer = NULL;
    //filename =  "";
  }
}

#ifdef __cplusplus
}
#endif

#else

#ifdef __cplusplus
extern "C" {
#endif

bool ops_checkpointing_init(const char *filename, double interval, int options) {
  return false;
}
bool ops_checkpointing_before(ops_arg *args, int nargs, int *range, int loop_id) {
  return true;
}
bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range, const char *s) {
  return true;
}
void ops_checkpointing_exit() {}

void ops_checkpointing_reduction(ops_reduction red) {
  ops_execute_reduction(red);
}

void ops_checkpointing_initphase_done() {}

void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist) {}
bool ops_checkpointing_fastfw(int nbytes, char *payload) {
  return false;
}
bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist, int nbytes, char *payload) {
  return false;
}

#ifdef __cplusplus
}
#endif

#endif
