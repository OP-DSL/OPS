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

/** @file
  * @brief OPS checkpointing library function declarations
  * @author Istvan Reguly
  * @details function implementations for checkpointing
  */

#ifdef CHECKPOINTING
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include <ops_lib_core.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <ops_exceptions.h>

#ifdef CHECKPOINTING

#ifndef defaultTimeout
#define defaultTimeout 2.0
#endif
#define ROUND64L(x) ((((x)-1l) / 64l + 1l) * 64l)
#ifdef __cplusplus
extern "C" {
#endif

// TODO: True temp dat support

// Internal function definitions
void ops_download_dat(ops_dat dat);
void ops_upload_dat(ops_dat dat);
bool ops_checkpointing_filename(const char *file_name, char *filename_out,
                                char *filename_out2);
void ops_checkpointing_calc_range(ops_dat dat, const int *range,
                                  int *saved_range);
void ops_checkpointing_duplicate_data(ops_dat dat, int my_type, int my_nelems,
                                      char *my_data, int *my_range,
                                      int *rm_type, int *rm_elems,
                                      char **rm_data, int **rm_range);

class OPS_instance_checkpointing {
  OPS_instance_checkpointing() {
    ops_call_counter = 0;
    ops_backup_point = -1;
    ops_best_backup_point = -1;
    ops_best_backup_point_size = 0;
    ops_checkpoint_interval = -1;
    ops_last_checkpoint = -1;
    ops_pre_backup_phase = false;
    ops_duplicate_backup = false;
    ops_loop_max = 0;
    ops_loops_hashmap = NULL;
    OPS_chk_red_size = 0;
    OPS_chk_red_offset = 0;
    OPS_chk_red_storage = NULL;

    OPS_partial_buffer_size = 0;
    OPS_partial_buffer = NULL;

    OPS_checkpointing_payload_nbytes = 0;
    OPS_checkpointing_payload = NULL;

    ops_reduction_counter = 0;
    ops_sync_frequency = 50;
    ops_reduction_avg_time = 0.0;

    ops_checkpointing_options = 0;
    ops_chk_write = 0.0;
    ops_chk_dup = 0.0;
    ops_chk_save = 0.0;

    diagnostics = 0;

//Strategy
    ops_strat_max_loop_counter = 0;
    ops_strat_min_saved        = NULL;
    ops_strat_max_saved        = NULL;
    ops_strat_avg_saved        = NULL;
    ops_strat_saved_counter    = NULL;
    ops_strat_timescalled      = NULL;
    ops_strat_maxcalled        = NULL;
    ops_strat_lastcalled       = NULL;
    ops_strat_dat_status      = NULL;
    ops_strat_in_progress      = NULL;

  }
public:
// checkpoint and execution progress information
  int ops_call_counter ;
  int ops_backup_point ;
  int ops_best_backup_point ;
  int ops_best_backup_point_size ;
  double ops_checkpoint_interval ;
  double ops_last_checkpoint ;
  bool ops_pre_backup_phase ;
  bool ops_duplicate_backup ;
  int ops_loop_max ;
  int *ops_loops_hashmap ;
  int OPS_chk_red_size ;
  int OPS_chk_red_offset ;
  char *OPS_chk_red_storage ;

  int OPS_partial_buffer_size ;
  char *OPS_partial_buffer ;

  int OPS_checkpointing_payload_nbytes ;
  char *OPS_checkpointing_payload ;

  int ops_reduction_counter ;
  int ops_sync_frequency ;
  double ops_reduction_avg_time ;

  int ops_checkpointing_options ;

// Timing
  double ops_chk_write ;
  double ops_chk_dup ;
  double ops_chk_save ;

// file managment
  char filename[100];
  char filename_dup[100];
  hid_t file;
  hid_t file_dup;
  herr_t status;

  FILE *diagf;
  int diagnostics ;

  char *params;

//Strategy
  int   ops_strat_max_loop_counter;
  long *ops_strat_min_saved;
  long *ops_strat_max_saved;
  long *ops_strat_avg_saved;
  long *ops_strat_saved_counter;
  int * ops_strat_timescalled;
  int * ops_strat_maxcalled;
  int * ops_strat_lastcalled;
  int * *ops_strat_dat_status;
  int * ops_strat_in_progress;


};

#define ops_call_counter OPS_instance::getOPSInstance()->checkpointing_instance->ops_call_counter
#define ops_backup_point OPS_instance::getOPSInstance()->checkpointing_instance->ops_backup_point
#define ops_best_backup_point OPS_instance::getOPSInstance()->checkpointing_instance->ops_best_backup_point
#define ops_best_backup_point_size OPS_instance::getOPSInstance()->checkpointing_instance->ops_best_backup_point_size
#define ops_checkpoint_interval OPS_instance::getOPSInstance()->checkpointing_instance->ops_checkpoint_interval
#define ops_last_checkpoint OPS_instance::getOPSInstance()->checkpointing_instance->ops_last_checkpoint
#define ops_pre_backup_phase OPS_instance::getOPSInstance()->checkpointing_instance->ops_pre_backup_phase
#define ops_duplicate_backup OPS_instance::getOPSInstance()->checkpointing_instance->ops_duplicate_backup
#define ops_loop_max OPS_instance::getOPSInstance()->checkpointing_instance->ops_loop_max
#define ops_loops_hashmap OPS_instance::getOPSInstance()->checkpointing_instance->ops_loops_hashmap
#define OPS_chk_red_size OPS_instance::getOPSInstance()->checkpointing_instance->OPS_chk_red_size
#define OPS_chk_red_offset OPS_instance::getOPSInstance()->checkpointing_instance->OPS_chk_red_offset
#define OPS_chk_red_storage OPS_instance::getOPSInstance()->checkpointing_instance->OPS_chk_red_storage
#define OPS_partial_buffer_size OPS_instance::getOPSInstance()->checkpointing_instance->OPS_partial_buffer_size
#define OPS_partial_buffer OPS_instance::getOPSInstance()->checkpointing_instance->OPS_partial_buffer
#define OPS_checkpointing_payload_nbytes OPS_instance::getOPSInstance()->checkpointing_instance->OPS_checkpointing_payload_nbytes
#define OPS_checkpointing_payload OPS_instance::getOPSInstance()->checkpointing_instance->OPS_checkpointing_payload
#define ops_reduction_counter OPS_instance::getOPSInstance()->checkpointing_instance->ops_reduction_counter
#define ops_sync_frequency OPS_instance::getOPSInstance()->checkpointing_instance->ops_sync_frequency
#define ops_reduction_avg_time OPS_instance::getOPSInstance()->checkpointing_instance->ops_reduction_avg_time
#define ops_checkpointing_options OPS_instance::getOPSInstance()->checkpointing_instance->ops_checkpointing_options
#define ops_chk_write OPS_instance::getOPSInstance()->checkpointing_instance->ops_chk_write
#define ops_chk_dup OPS_instance::getOPSInstance()->checkpointing_instance->ops_chk_dup
#define ops_chk_save OPS_instance::getOPSInstance()->checkpointing_instance->ops_chk_save
#define managment OPS_instance::getOPSInstance()->checkpointing_instance->managment
#define filename OPS_instance::getOPSInstance()->checkpointing_instance->filename
#define filename_dup OPS_instance::getOPSInstance()->checkpointing_instance->filename_dup
#define file OPS_instance::getOPSInstance()->checkpointing_instance->file
#define file_dup OPS_instance::getOPSInstance()->checkpointing_instance->file_dup
#define status OPS_instance::getOPSInstance()->checkpointing_instance->status
#define diagf OPS_instance::getOPSInstance()->checkpointing_instance->diagf
#define diagnostics OPS_instance::getOPSInstance()->checkpointing_instance->diagnostics
#define params OPS_instance::getOPSInstance()->checkpointing_instance->params


#define check_hdf5_error(err) __check_hdf5_error(err, __FILE__, __LINE__)
void __check_hdf5_error(herr_t err, const char *file, const int line) {
  if (err < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: " << file << ":" << line << " OPS_HDF5_error() Runtime API error " << err << " " << params;
    throw ex;
  }
}

bool file_exists(const char *file_name) {
  if (FILE *file = fopen(file_name, "r")) {
    fclose(file);
    return true;
  }
  return false;
}

void ops_create_lock(char *fname) {
  char buffer[strlen(fname) + 5];
  sprintf(buffer, "%s.lock", fname);
  if (FILE *lock = fopen(buffer, "w+")) {
    fclose(lock);
  } else {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf("Warning: lock file already exists\n");
  }
}

void ops_create_lock_done(char *fname) {
  char buffer[strlen(fname) + 5];
  sprintf(buffer, "%s.done", fname);
  if (FILE *lock = fopen(buffer, "w+")) {
    fclose(lock);
  } else {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf("Warning: lock done file already exists\n");
  }
}

void ops_pack_chk(const char *__restrict src, char *__restrict dest,
                  const int count, const int blocklength, const int stride) {
  for (unsigned int i = 0; i < count; i++) {
    memcpy(dest, src, blocklength);
    src += stride;
    dest += blocklength;
  }
}

void ops_unpack_chk(char *__restrict dest, const char *__restrict src,
                    const int count, const int blocklength, const int stride) {
  for (unsigned int i = 0; i < count; i++) {
    memcpy(dest, src, blocklength);
    src += blocklength;
    dest += stride;
  }
}

typedef struct {
  ops_dat dat;
  hid_t outfile;
  int size;
  int saved_range[2 * OPS_MAX_DIM];
  int partial;
  char *data;
  int dup;
} ops_ramdisk_item;

ops_ramdisk_item *ops_ramdisk_item_queue = NULL;
volatile int ops_ramdisk_item_queue_head = 0;
volatile int ops_ramdisk_item_queue_tail = 0;
int ops_ramdisk_item_queue_size = 0;

#define OPS_CHK_THREAD
#ifdef OPS_CHK_THREAD
#include <pthread.h>
#include <unistd.h>
char *ops_ramdisk_buffer = NULL;
volatile long ops_ramdisk_size = 0;
volatile long ops_ramdisk_head = 0;
volatile long ops_ramdisk_tail = 0;

// Timing
double ops_chk_thr_queue = 0.0;

volatile int ops_ramdisk_ctrl_exit = 0;
volatile int ops_ramdisk_ctrl_finish = 0;
int ops_ramdisk_initialised = 0;

pthread_t thread;
void save_to_hdf5_partial(ops_dat dat, hid_t outfile, int size,
                          int *saved_range, char *data);
void save_to_hdf5_full(ops_dat dat, hid_t outfile, int size, char *data);

void *ops_saver_thread(void *payload) {
  while (!ops_ramdisk_ctrl_exit) {
    if (ops_ramdisk_item_queue_head != ops_ramdisk_item_queue_tail) {
      int tail = ops_ramdisk_item_queue_tail;
      if (ops_ramdisk_item_queue[tail].partial) {
        if (OPS_instance::getOPSInstance()->OPS_diags > 5)
          printf("Thread saving partial %s\n",
                 ops_ramdisk_item_queue[tail].dat->name);
        save_to_hdf5_partial(ops_ramdisk_item_queue[tail].dat,
                             ops_ramdisk_item_queue[tail].outfile,
                             ops_ramdisk_item_queue[tail].size /
                                 (ops_ramdisk_item_queue[tail].dat->elem_size /
                                  ops_ramdisk_item_queue[tail].dat->dim),
                             ops_ramdisk_item_queue[tail].saved_range,
                             ops_ramdisk_item_queue[tail].data);
      } else {
        if (OPS_instance::getOPSInstance()->OPS_diags > 5)
          printf("Thread saving full %s\n",
                 ops_ramdisk_item_queue[tail].dat->name);
        save_to_hdf5_full(ops_ramdisk_item_queue[tail].dat,
                          ops_ramdisk_item_queue[tail].outfile,
                          ops_ramdisk_item_queue[tail].size /
                              (ops_ramdisk_item_queue[tail].dat->elem_size /
                               ops_ramdisk_item_queue[tail].dat->dim),
                          ops_ramdisk_item_queue[tail].data);
      }
      if (ops_ramdisk_tail + ROUND64L(ops_ramdisk_item_queue[tail].size) <
          ops_ramdisk_size) {
        if (ops_ramdisk_tail < ops_ramdisk_head &&
            ops_ramdisk_tail + ROUND64L(ops_ramdisk_item_queue[tail].size) >
                ops_ramdisk_head) {
          throw OPSException(OPS_INTERNAL_ERROR, "Internal error: tail error");
        }
        ops_ramdisk_tail += ROUND64L(ops_ramdisk_item_queue[tail].size);
      } else {
        if (ops_ramdisk_head < ROUND64L(ops_ramdisk_item_queue[tail].size)) {
          throw OPSException(OPS_INTERNAL_ERROR, "Internal error: tail error 2");
        }
        ops_ramdisk_tail = ROUND64L(ops_ramdisk_item_queue[tail].size);
      }

      ops_ramdisk_item_queue_tail =
          (ops_ramdisk_item_queue_tail + 1) % ops_ramdisk_item_queue_size;
    }

    if (ops_ramdisk_item_queue_head == ops_ramdisk_item_queue_tail) {
      if (ops_ramdisk_ctrl_finish) {
        check_hdf5_error(H5Fclose(file));
        if (OPS_instance::getOPSInstance()->ops_lock_file)
          ops_create_lock(filename);
        if (ops_duplicate_backup)
          check_hdf5_error(H5Fclose(file_dup));
        if (ops_duplicate_backup && OPS_instance::getOPSInstance()->ops_lock_file)
          ops_create_lock(filename_dup);
        ops_ramdisk_ctrl_finish = 0;
        usleep((long)(ops_checkpoint_interval * 300000.0));
      }
      usleep(100000);
    }
  }
  pthread_exit(NULL); // Perhaps return whether everything was properly saved
}

void OPS_instance::getOPSInstance()->OPS_reallocate_ramdisk(long size) {
  // Wait for the queue to drain
  if (OPS_instance::getOPSInstance()->OPS_diags > 2 &&
      (ops_ramdisk_item_queue_head != ops_ramdisk_item_queue_tail))
    printf("Main thread waiting for ramdisk reallocation head %d tail %d\n",
           ops_ramdisk_item_queue_head, ops_ramdisk_item_queue_tail);
  while (ops_ramdisk_item_queue_head != ops_ramdisk_item_queue_tail)
    usleep(20000);
  ops_ramdisk_size = ROUND64L(size);
  ops_ramdisk_buffer =
      (char *)ops_realloc(ops_ramdisk_buffer, ops_ramdisk_size * sizeof(char));
  ops_ramdisk_tail = 0;
  ops_ramdisk_head = 0;
}

void ops_ramdisk_init(long size) {
  if (ops_ramdisk_initialised)
    return;
  ops_ramdisk_size = ROUND64L(size);
  ops_ramdisk_buffer = (char *)ops_malloc(ops_ramdisk_size * sizeof(char));
  ops_ramdisk_tail = 0;
  ops_ramdisk_head = 0;
  ops_ramdisk_item_queue = (ops_ramdisk_item *)ops_malloc(
      3 * OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(ops_ramdisk_item));
  ops_ramdisk_item_queue_head = 0;
  ops_ramdisk_item_queue_tail = 0;
  ops_ramdisk_item_queue_size = 3 * OPS_instance::getOPSInstance()->OPS_dat_index;
  ops_ramdisk_initialised = 1;
  int rc = pthread_create(&thread, NULL, ops_saver_thread, NULL);
  if (rc) {
    throw OPSException(OPS_INTERNAL_ERROR, "Internal error: failed pthread_create in checkpointing");
  }
}

void ops_ramdisk_queue(ops_dat dat, hid_t outfile, int size, int *saved_range,
                       char *data, int partial) {
  double c1, t1, t2;
  ops_timers_core(&c1, &t1);
  size = size * dat->elem_size / dat->dim;
  long sizeround64 = ROUND64L(size);
  if (ops_ramdisk_size < sizeround64)
    OPS_instance::getOPSInstance()->OPS_reallocate_ramdisk(
        ROUND64L(3l * (long)sizeround64 + (long)sizeround64 / 5l + 64));
  // Copy data to ramdisk
  long tail = ops_ramdisk_tail;
  long head = ops_ramdisk_head;
  while ((head < tail && head + sizeround64 >= tail) ||
         (head + sizeround64 >= ops_ramdisk_size && sizeround64 >= tail)) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf("Main thread waiting for ramdisk room %ld < %ld && %ld + %d >= "
             "%ld or %ld + %d  >= %ld && %d >= %ld\n",
             head, tail, head, sizeround64, tail, head, sizeround64,
             ops_ramdisk_size, sizeround64, tail);
    usleep(10000);
    tail = ops_ramdisk_tail;
  }

  if (head + sizeround64 >= ops_ramdisk_size)
    head = 0;

  memcpy(ops_ramdisk_buffer + head, data, size * sizeof(char));
  ops_ramdisk_head = head + sizeround64;

  // next item index
  int item_idx = ops_ramdisk_item_queue_head;
  int item_idx_next =
      (ops_ramdisk_item_queue_head + 1) % ops_ramdisk_item_queue_size;

  // wait if we are about to bite our tails
  if ((OPS_instance::getOPSInstance()->OPS_diags > 2) && (item_idx_next == ops_ramdisk_item_queue_tail))
    printf("Main thread waiting for ramdisk item queue room %d == %d\n",
           item_idx_next, ops_ramdisk_item_queue_tail);
  while (item_idx_next == ops_ramdisk_item_queue_tail)
    usleep(10000);

  // enqueue item
  ops_ramdisk_item_queue[item_idx].dat = dat;
  ops_ramdisk_item_queue[item_idx].outfile = outfile;
  ops_ramdisk_item_queue[item_idx].size = size;
  if (partial)
    memcpy(ops_ramdisk_item_queue[item_idx].saved_range, saved_range,
           2 * OPS_MAX_DIM * sizeof(int));
  ops_ramdisk_item_queue[item_idx].data = ops_ramdisk_buffer + head;
  ops_ramdisk_item_queue[item_idx].partial = partial;
  ops_ramdisk_item_queue_head = item_idx_next;
  ops_timers_core(&c1, &t2);
  ops_chk_thr_queue += t2 - t1;
}

#endif

void ops_inmemory_save(ops_dat dat, hid_t outfile, int size, int *saved_range,
                       char *data, int partial, int dup) {
  // Increment head
  int head = ops_ramdisk_item_queue_head++;

  // Save all fields and make copy of data in memory
  ops_ramdisk_item_queue[head].dat = dat;
  ops_ramdisk_item_queue[head].outfile = outfile;
  ops_ramdisk_item_queue[head].size = size;
  if (partial)
    memcpy(ops_ramdisk_item_queue[head].saved_range, saved_range,
           2 * OPS_MAX_DIM * sizeof(int));
  ops_ramdisk_item_queue[head].data =
      (char *)ops_malloc((dat->elem_size / dat->dim) * size * sizeof(char));
  memcpy(ops_ramdisk_item_queue[head].data, data,
         (dat->elem_size / dat->dim) * size * sizeof(char));
  ops_ramdisk_item_queue[head].partial = partial;
  ops_ramdisk_item_queue[head].dup = dup;
}

// Handler for saving a dataset, full or partial, with different mechanisms
void save_data_handler(ops_dat dat, hid_t outfile, int size, int *saved_range,
                       char *data, int partial, int dup) {
#ifdef OPS_CHK_THREAD
  if (OPS_instance::getOPSInstance()->ops_thread_offload)
    ops_ramdisk_queue(dat, outfile, size, saved_range, data, partial);
  else {
#endif
    if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory)
      ops_inmemory_save(dat, outfile, size, saved_range, data, partial, dup);
    else {
      if (partial)
        save_to_hdf5_partial(dat, outfile, size, saved_range, data);
      else
        save_to_hdf5_full(dat, outfile, size, data);
    }
#ifdef OPS_CHK_THREAD
  }
#endif
}

void save_to_hdf5_full(ops_dat dat, hid_t outfile, int size, char *data) {
  if (size == 0)
    return;
  double t1, t2, c1;
  ops_timers_core(&c1, &t1);
  hsize_t dims[1];
  dims[0] = size;
  if (strcmp(dat->type, "int") == 0) {
    check_hdf5_error(
        H5LTmake_dataset(outfile, dat->name, 1, dims, H5T_NATIVE_INT, data));
  } else if (strcmp(dat->type, "float") == 0) {
    check_hdf5_error(
        H5LTmake_dataset(outfile, dat->name, 1, dims, H5T_NATIVE_FLOAT, data));
  } else if (strcmp(dat->type, "double") == 0) {
    check_hdf5_error(
        H5LTmake_dataset(outfile, dat->name, 1, dims, H5T_NATIVE_DOUBLE, data));
  } else {
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Unsupported data type during checkpointing, please add in ops_checkpointing.cpp");
  }
  ops_timers_core(&c1, &t2);
  ops_chk_write += t2 - t1;
}

void save_to_hdf5_partial(ops_dat dat, hid_t outfile, int size,
                          int *saved_range, char *data) {
  if (size == 0)
    return;
  double t1, t2, c1;
  ops_timers_core(&c1, &t1);
  hsize_t dims[1];
  dims[0] = size;
  if (strcmp(dat->type, "int") == 0) {
    check_hdf5_error(
        H5LTmake_dataset(outfile, dat->name, 1, dims, H5T_NATIVE_INT, data));
  } else if (strcmp(dat->type, "float") == 0) {
    check_hdf5_error(
        H5LTmake_dataset(outfile, dat->name, 1, dims, H5T_NATIVE_FLOAT, data));
  } else if (strcmp(dat->type, "double") == 0) {
    check_hdf5_error(
        H5LTmake_dataset(outfile, dat->name, 1, dims, H5T_NATIVE_DOUBLE, data));
  } else {
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Unsupported data type during checkpointing, please add in ops_checkpointing.cpp");
  }

  char buf[50];
  sprintf(buf, "%s_saved_range", dat->name);
  dims[0] = 2 * OPS_MAX_DIM;
  check_hdf5_error(
      H5LTmake_dataset(outfile, buf, 1, dims, H5T_NATIVE_INT, saved_range));
  ops_timers_core(&c1, &t2);
  ops_chk_write += t2 - t1;
}

//
// Save a dataset to disk. If in a distributed system, send/receive from
// neighboring node and save duplicate
//
void save_dat(ops_dat dat) {
  double c1, t1, t2;
  ops_timers_core(&c1, &t1);
  OPS_instance::getOPSInstance()->OPS_dat_status[dat->index] = OPS_SAVED;
  hsize_t dims[1];
  if (dat->dirty_hd == 2)
    ops_download_dat(dat);
  dims[0] = dat->dim;
  for (int d = 0; d < dat->block->dims; d++)
    dims[0] *= dat->size[d];

  save_data_handler(dat, file, (int)dims[0], NULL, dat->data, 0, 0);

  if (ops_duplicate_backup) {
    char *rm_data;
    int my_type = 0, rm_type = 0;
    int my_nelems = (int)dims[0], rm_nelems;
    int *my_range = dat->size, *rm_range;

    double c1, t1, t2;
    ops_timers_core(&c1, &t1);
    ops_checkpointing_duplicate_data(dat, my_type, my_nelems, dat->data,
                                     my_range, &rm_type, &rm_nelems, &rm_data,
                                     &rm_range);
    ops_timers_core(&c1, &t2);
    ops_chk_dup += t2 - t1;
    if (rm_type == 0)
      save_data_handler(dat, file_dup, rm_nelems, NULL, rm_data, 0, 1);
    else
      save_data_handler(dat, file_dup, rm_nelems, rm_range, rm_data, 1, 1);
  }
  if (OPS_instance::getOPSInstance()->OPS_diags > 4)
    printf("Backed up %s\n", dat->name);
  ops_timers_core(&c1, &t2);
  ops_chk_save += t2 - t1;
}

void save_dat_partial(ops_dat dat, int *range) {
  double c1, t1, t2;
  ops_timers_core(&c1, &t1);
  OPS_instance::getOPSInstance()->OPS_dat_status[dat->index] = OPS_SAVED;
  if (dat->dirty_hd == 2)
    ops_download_dat(dat);
  int saved_range[2 * OPS_MAX_DIM] = {0};
  int prod[OPS_MAX_DIM + 1];
  prod[0] = 1;

  // Calculate saved range (based on full size and the range where it is
  // modified)
  ops_checkpointing_calc_range(dat, range, saved_range);

  for (int d = 0; d < dat->block->dims; d++) {
    prod[d + 1] = prod[d] * dat->size[d];
  }
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    saved_range[2 * d] = 0;
    saved_range[2 * d + 1] = 1;
  }

  // Calculate access pattern to sides (depth 1)
  int count[OPS_MAX_DIM], block_length[OPS_MAX_DIM], stride[OPS_MAX_DIM];
  for (int d = 0; d < dat->block->dims; d++) {
    count[d] = prod[dat->block->dims] / prod[d + 1];
    block_length[d] = 1 * prod[d] * dat->elem_size;
    stride[d] = prod[d + 1] * dat->elem_size;
  }

  // Calculate total size stored (redundant at edges)
  hsize_t dims[1] = {0};
  for (int d = 0; d < dat->block->dims; d++) {
    int depth_before = saved_range[2 * d];
    int depth_after = dat->size[d] - saved_range[2 * d + 1];
    dims[0] += (depth_before + depth_after) * block_length[d] * count[d];
  }

  // if too much redundancy, just do the usual save
  if (dims[0] >= prod[dat->block->dims] * dat->elem_size) {
    save_dat(dat);
    return;
  }

  if (OPS_partial_buffer_size < dims[0]) {
    OPS_partial_buffer_size = 2 * dims[0];
    OPS_partial_buffer = (char *)ops_realloc(
        OPS_partial_buffer, OPS_partial_buffer_size * sizeof(char));
  }

  // Pack
  int offset = 0;
  for (int d = 0; d < dat->block->dims; d++) {
    int depth_before = saved_range[2 * d];
    if (depth_before)
      ops_pack_chk(dat->data, OPS_partial_buffer + offset, count[d],
                   depth_before * block_length[d], stride[d]);
    offset += depth_before * block_length[d] * count[d];
    int depth_after = dat->size[d] - saved_range[2 * d + 1];
    int i4 = (prod[d + 1] / prod[d] - (depth_after)) * prod[d] * dat->elem_size;
    if (depth_after)
      ops_pack_chk(dat->data + i4, OPS_partial_buffer + offset, count[d],
                   depth_after * block_length[d], stride[d]);
    offset += depth_after * block_length[d] * count[d];
  }

  dims[0] = dims[0] / (dat->elem_size / dat->dim);
  save_data_handler(dat, file, (int)dims[0], saved_range, OPS_partial_buffer, 1,
                    0);

  if (ops_duplicate_backup) {
    char *rm_data;
    int my_type = 1, rm_type = 0;
    int my_nelems = (int)dims[0], rm_nelems;
    int *my_range = saved_range, *rm_range;

    double c1, t1, t2;
    ops_timers_core(&c1, &t1);
    ops_checkpointing_duplicate_data(dat, my_type, my_nelems,
                                     OPS_partial_buffer, my_range, &rm_type,
                                     &rm_nelems, &rm_data, &rm_range);
    ops_timers_core(&c1, &t2);
    ops_chk_dup += t2 - t1;
    if (rm_type == 0)
      save_data_handler(dat, file_dup, rm_nelems, NULL, rm_data, 0, 1);
    else
      save_data_handler(dat, file_dup, rm_nelems, rm_range, rm_data, 1, 1);
  }

  if (OPS_instance::getOPSInstance()->OPS_diags > 4)
    printf("Backed up %s (partial)\n", dat->name);
  ops_timers_core(&c1, &t2);
  ops_chk_save += t2 - t1;
}

void ops_restore_dataset(ops_dat dat) {
  if (!H5LTfind_dataset(file, dat->name))
    return;
  char buf[50];
  sprintf(buf, "%s_saved_range", dat->name);
  hsize_t dims[1] = {0};
  dims[0] = 2 * OPS_MAX_DIM;
  // if no range specified, just read back the entire dataset
  if (!H5LTfind_dataset(file, buf)) {
    dims[0] = dat->elem_size;
    for (int d = 0; d < dat->block->dims; d++)
      dims[0] *= dat->size[d];
    if (strcmp(dat->type, "int") == 0) {
      check_hdf5_error(
          H5LTread_dataset(file, dat->name, H5T_NATIVE_INT, dat->data));
    } else if (strcmp(dat->type, "float") == 0) {
      check_hdf5_error(
          H5LTread_dataset(file, dat->name, H5T_NATIVE_FLOAT, dat->data));
    } else if (strcmp(dat->type, "double") == 0) {
      check_hdf5_error(
          H5LTread_dataset(file, dat->name, H5T_NATIVE_DOUBLE, dat->data));
    } else {
      throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Unsupported data type during checkpointing, please add in ops_checkpointing.cpp");
    }
    dat->dirty_hd = 1;
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Restored %s\n", dat->name);
  } else { // if a range is specified,
    int saved_range[OPS_MAX_DIM * 2];
    check_hdf5_error(H5LTread_dataset(file, buf, H5T_NATIVE_INT, saved_range));
    if (dat->dirty_hd == 2)
      ops_download_dat(dat);
    int prod[OPS_MAX_DIM + 1];
    prod[0] = 1;
    for (int d = 0; d < dat->block->dims; d++)
      prod[d + 1] = prod[d] * dat->size[d];
    // Calculate access pattern to sides (depth 1)
    int count[OPS_MAX_DIM], block_length[OPS_MAX_DIM], stride[OPS_MAX_DIM];
    for (int d = 0; d < dat->block->dims; d++) {
      count[d] = prod[dat->block->dims] / prod[d + 1];
      block_length[d] = 1 * prod[d] * dat->elem_size;
      stride[d] = prod[d + 1] * dat->elem_size;
    }

    // Calculate total size stored (redundant at edges)
    hsize_t dims[1] = {0};
    for (int d = 0; d < dat->block->dims; d++) {
      int depth_before = saved_range[2 * d];
      int depth_after = dat->size[d] - saved_range[2 * d + 1];
      dims[0] += (depth_before + depth_after) * block_length[d] * count[d];
    }
    if (dims[0] == 0)
      return;

    if (OPS_partial_buffer_size < dims[0]) {
      OPS_partial_buffer_size = 2 * dims[0];
      OPS_partial_buffer = (char *)ops_realloc(
          OPS_partial_buffer, OPS_partial_buffer_size * sizeof(char));
    }

    dims[0] = dims[0] / (dat->elem_size / dat->dim);
    if (strcmp(dat->type, "int") == 0) {
      check_hdf5_error(H5LTread_dataset(file, dat->name, H5T_NATIVE_INT,
                                        OPS_partial_buffer));
    } else if (strcmp(dat->type, "float") == 0) {
      check_hdf5_error(H5LTread_dataset(file, dat->name, H5T_NATIVE_FLOAT,
                                        OPS_partial_buffer));
    } else if (strcmp(dat->type, "double") == 0) {
      check_hdf5_error(H5LTread_dataset(file, dat->name, H5T_NATIVE_DOUBLE,
                                        OPS_partial_buffer));
    } else {
      throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Unsupported data type during checkpointing, please add in ops_checkpointing.cpp");
    }

    // Unpack
    int offset = 0;
    for (int d = 0; d < dat->block->dims; d++) {
      int depth_before = saved_range[2 * d];
      if (depth_before)
        ops_unpack_chk(dat->data, OPS_partial_buffer + offset, count[d],
                       depth_before * block_length[d], stride[d]);
      offset += depth_before * block_length[d] * count[d];
      int depth_after = dat->size[d] - saved_range[2 * d + 1];
      int i4 =
          (prod[d + 1] / prod[d] - (depth_after)) * prod[d] * dat->elem_size;
      if (depth_after)
        ops_unpack_chk(dat->data + i4, OPS_partial_buffer + offset, count[d],
                       depth_after * block_length[d], stride[d]);
      offset += depth_after * block_length[d] * count[d];
    }

    dat->dirty_hd = 1;
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Restored %s (partial)\n", dat->name);
  }
}

void ops_checkpoint_prepare_files() {
  if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 5)
      ops_printf("New inmemory checkpoint, purging previous\n");
    for (int i = ops_ramdisk_item_queue_tail; i < ops_ramdisk_item_queue_head;
         i++) {
      free(ops_ramdisk_item_queue[i].data);
      ops_ramdisk_item_queue[i].data = NULL;
      ops_ramdisk_item_queue[i].dat = NULL;
    }
    ops_ramdisk_item_queue_tail = 0;
    ops_ramdisk_item_queue_head = 0;
  } else {
    double cpu, t1, t2, t3, t4;
    ops_timers_core(&cpu, &t3);
    if (file_exists(filename))
      remove(filename);
    if (ops_duplicate_backup && file_exists(filename_dup))
      remove(filename_dup);
    ops_timers_core(&cpu, &t4);
    if (OPS_instance::getOPSInstance()->OPS_diags > 5)
      printf("Removed previous file %g\n", t4 - t3);

    // where we start backing up stuff
    ops_timers_core(&cpu, &t3);
    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (ops_duplicate_backup)
      file_dup =
          H5Fcreate(filename_dup, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ops_timers_core(&cpu, &t4);
    if (OPS_instance::getOPSInstance()->OPS_diags > 5)
      printf("Opened new file %g\n", t4 - t3);
  }
}

void ops_checkpoint_complete() {
  if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory)
    return;
#ifdef OPS_CHK_THREAD
  if (OPS_instance::getOPSInstance()->ops_thread_offload)
    ops_ramdisk_ctrl_finish = 1;
  else {
#endif
    check_hdf5_error(H5Fclose(file));
    if (OPS_instance::getOPSInstance()->ops_lock_file)
      ops_create_lock(filename);
    if (ops_duplicate_backup)
      check_hdf5_error(H5Fclose(file_dup));
    if (ops_duplicate_backup && OPS_instance::getOPSInstance()->ops_lock_file)
      ops_create_lock(filename_dup);
#ifdef OPS_CHK_THREAD
  }
#endif
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
} OPS_instance::getOPSInstance()->ops_checkpoint_inmemory_control;

OPS_instance::getOPSInstance()->ops_checkpoint_inmemory_control ops_inm_ctrl;

void ops_ctrldump(hid_t file_out) {
  if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory)
    return;
  hsize_t dims[1];
  dims[0] = 1;
  check_hdf5_error(H5LTmake_dataset(file_out, "ops_backup_point", 1, dims,
                                    H5T_NATIVE_INT,
                                    &ops_inm_ctrl.ops_backup_point));
  check_hdf5_error(H5LTmake_dataset(file_out, "ops_best_backup_point", 1, dims,
                                    H5T_NATIVE_INT,
                                    &ops_inm_ctrl.ops_best_backup_point));
  dims[0] = ops_inm_ctrl.reduction_state_size;
  check_hdf5_error(H5LTmake_dataset(file_out, "reduction_state", 1, dims,
                                    H5T_NATIVE_CHAR,
                                    ops_inm_ctrl.reduction_state));
  free(ops_inm_ctrl.reduction_state);
  ops_inm_ctrl.reduction_state = NULL;
  if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    dims[0] = ops_inm_ctrl.OPS_checkpointing_payload_nbytes;
    check_hdf5_error(H5LTmake_dataset(file_out, "OPS_checkpointing_payload", 1,
                                      dims, H5T_NATIVE_CHAR,
                                      ops_inm_ctrl.OPS_checkpointing_payload));
  }
  dims[0] = 1;
  check_hdf5_error(H5LTmake_dataset(file_out, "OPS_chk_red_offset", 1, dims,
                                    H5T_NATIVE_INT,
                                    &ops_inm_ctrl.OPS_chk_red_offset));
  dims[0] = ops_inm_ctrl.OPS_chk_red_offset;
  check_hdf5_error(H5LTmake_dataset(file_out, "OPS_chk_red_storage", 1, dims,
                                    H5T_NATIVE_CHAR,
                                    ops_inm_ctrl.OPS_chk_red_storage));
}

void ops_chkp_sig_handler(int signo) {
  if (signo != SIGINT) {
    printf("Error: OPS received unknown signal %d\n", signo);
    return;
  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS) {
    printf("Error: failed in checkpointing region, aborting checkpoint\n");
    return;
  }
  if (OPS_instance::getOPSInstance()->OPS_diags > 1)
    printf("Process received SIGINT, dumping checkpoint from memory...\n");

  // No checkpoint yet
  if (ops_inm_ctrl.ops_backup_point == -1) {
    return;
  }

  // Things now all do go to disk
  OPS_instance::getOPSInstance()->ops_checkpoint_inmemory = 0;

  // Open files
  ops_checkpoint_prepare_files();

  // Save control
  ops_ctrldump(file);
  if (ops_duplicate_backup) {
    ops_ctrldump(file_dup);
  }

  // Save datasets
  for (int i = ops_ramdisk_item_queue_tail; i < ops_ramdisk_item_queue_head;
       i++) {
    save_data_handler(
        ops_ramdisk_item_queue[i].dat,
        ops_ramdisk_item_queue[i].dup ? file_dup : file,
        ops_ramdisk_item_queue[i].size, ops_ramdisk_item_queue[i].saved_range,
        ops_ramdisk_item_queue[i].data, ops_ramdisk_item_queue[i].partial,
        ops_ramdisk_item_queue[i].dup);
    free(ops_ramdisk_item_queue[i].data);
  }
  free(ops_ramdisk_item_queue);

  // Close files
  ops_checkpoint_complete();

  //(we might as well leak the rest of the memory, the process is crashing)
  exit(-1);
}

bool ops_checkpointing_initstate() {
  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_dat_index; i++) {
    OPS_instance::getOPSInstance()->OPS_dat_status[i] = OPS_UNDECIDED;
    OPS_instance::getOPSInstance()->OPS_dat_ever_written[i] = 0;
  }
  if (!file_exists(filename)) {
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_GATHER;
    ops_printf("//\n// OPS Checkpointing -- Backup mode\n//\n");
    return false;
  } else {
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_LEADIN;
    ops_printf("//\n// OPS Checkpointing -- Restore mode\n//\n");

    if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      // we are not restoring anything during lead-in and will be resuming
      // execution the first time we encounter the API call
    } else {
      double cpu, t1, t2;
      ops_timers_core(&cpu, &t1);
      // read backup point
      check_hdf5_error(H5LTread_dataset(file, "ops_backup_point",
                                        H5T_NATIVE_INT, &ops_backup_point));
      check_hdf5_error(H5LTread_dataset(file, "ops_best_backup_point",
                                        H5T_NATIVE_INT,
                                        &ops_best_backup_point));

      // loading of datasets is postponed until we reach the restore point

      // restore reduction storage
      check_hdf5_error(H5LTread_dataset(file, "OPS_chk_red_offset",
                                        H5T_NATIVE_INT, &OPS_chk_red_offset));
      OPS_chk_red_storage =
          (char *)ops_malloc(OPS_chk_red_offset * sizeof(char));
      OPS_chk_red_size = OPS_chk_red_offset;
      OPS_chk_red_offset = 0;
      check_hdf5_error(H5LTread_dataset(file, "OPS_chk_red_storage",
                                        H5T_NATIVE_CHAR, OPS_chk_red_storage));
      ops_timers_core(&cpu, &t2);
      OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;
    }
    return true;
  }
}

/**
* Initialises checkpointing using the given filename
*/
bool ops_checkpointing_init(const char *file_name, double interval,
                            int options) {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return false;
  if ((options & OPS_CHECKPOINT_MANUAL) &&
      !(options & (OPS_CHECKPOINT_MANUAL_DATLIST | OPS_CHECKPOINT_FASTFW))) {
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: cannot have manual checkpoint triggering without manual datlist and fast-forward!");
  }

  // Control structures initialized
  ops_inm_ctrl.ops_backup_point = -1;
  ops_inm_ctrl.ops_best_backup_point = -1;
  ops_inm_ctrl.reduction_state = NULL;
  ops_inm_ctrl.reduction_state_size = -1;
  ops_inm_ctrl.OPS_checkpointing_payload = NULL;
  ops_inm_ctrl.OPS_checkpointing_payload_nbytes = -1;
  ops_inm_ctrl.OPS_chk_red_offset = -1;
  ops_inm_ctrl.OPS_chk_red_storage = NULL;

  if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory) {
#ifdef OPS_CHK_THREAD
    if (OPS_instance::getOPSInstance()->ops_thread_offload) {
      ops_printf("Warning: in-memory checkpointing and thread-offload "
                 "checkpointing are mutually exclusive. Thread-offload will be "
                 "switched off\n");
      OPS_instance::getOPSInstance()->ops_thread_offload = 0;
    }
#endif
    if (signal(SIGINT, ops_chkp_sig_handler) == SIG_ERR) {
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS can't catch SIGINT - disable in-memory checkpointing");
    }
    ops_ramdisk_item_queue = (ops_ramdisk_item *)ops_malloc(
        3 * OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(ops_ramdisk_item));
    ops_ramdisk_item_queue_head = 0;
    ops_ramdisk_item_queue_tail = 0;
    ops_ramdisk_item_queue_size = 3 * OPS_instance::getOPSInstance()->OPS_dat_index;
  }

  ops_checkpoint_interval = interval;
  if (interval < defaultTimeout * 2.0)
    ops_checkpoint_interval =
        defaultTimeout *
        2.0; // WHAT SHOULD THIS BE? - the time it takes to back up everything
  double cpu;
  ops_timers_core(&cpu, &ops_last_checkpoint);
  ops_duplicate_backup =
      ops_checkpointing_filename(file_name, filename, filename_dup);

  OPS_instance::getOPSInstance()->OPS_dat_ever_written = (char *)ops_malloc(OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(char));
  OPS_instance::getOPSInstance()->OPS_dat_status = (ops_checkpoint_types *)ops_malloc(
      OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(ops_checkpoint_types));

  if (diagnostics) {
    diagf = fopen("checkp_diags.txt", "w");
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      fprintf(diagf, "%s;%d;%d;%s\n", dat->name, dat->size[0], dat->size[1],
              dat->type);
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
  return false;
}

void ops_checkpointing_initphase_done() {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return;
  if (OPS_instance::getOPSInstance()->backup_state == OPS_NONE) {
    ops_checkpointing_initstate();
  }
}

void ops_checkpointing_save_control(hid_t file_out) {
  // write control variables and all initialized reduction handles

  ops_inm_ctrl.ops_backup_point = ops_backup_point;
  ops_inm_ctrl.ops_best_backup_point = ops_best_backup_point;

  // Save the state of all ongoing reductions
  int total_size = 0;
  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_reduction_index; i++)
    if (OPS_instance::getOPSInstance()->OPS_reduction_list[i]->initialized == 1)
      total_size += OPS_instance::getOPSInstance()->OPS_reduction_list[i]->size;
  char *reduction_state = (char *)ops_malloc(total_size * sizeof(char));
  total_size = 0;
  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_reduction_index; i++) {
    if (OPS_instance::getOPSInstance()->OPS_reduction_list[i]->initialized == 1) {
      memcpy(&reduction_state[total_size], OPS_instance::getOPSInstance()->OPS_reduction_list[i]->data,
             OPS_instance::getOPSInstance()->OPS_reduction_list[i]->size);
      total_size += OPS_instance::getOPSInstance()->OPS_reduction_list[i]->size;
    }
  }

  if (ops_inm_ctrl.reduction_state != NULL)
    free(ops_inm_ctrl.reduction_state);
  ops_inm_ctrl.reduction_state = reduction_state;
  ops_inm_ctrl.reduction_state_size = total_size;

  // Save payload if specified by user
  if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    ops_inm_ctrl.OPS_checkpointing_payload =
        (char *)ops_realloc(ops_inm_ctrl.OPS_checkpointing_payload,
                            OPS_checkpointing_payload_nbytes * sizeof(char));
    memcpy(ops_inm_ctrl.OPS_checkpointing_payload, OPS_checkpointing_payload,
           OPS_checkpointing_payload_nbytes * sizeof(char));
    ops_inm_ctrl.OPS_checkpointing_payload_nbytes =
        OPS_checkpointing_payload_nbytes;
  }

  // save reduction history
  ops_inm_ctrl.OPS_chk_red_offset = OPS_chk_red_offset;
  ops_inm_ctrl.OPS_chk_red_storage = (char *)ops_realloc(
      ops_inm_ctrl.OPS_chk_red_storage, OPS_chk_red_offset * sizeof(char));
  memcpy(ops_inm_ctrl.OPS_chk_red_storage, OPS_chk_red_storage,
         OPS_chk_red_offset * sizeof(char));

  // write to file
  ops_ctrldump(file_out);
}

void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist) {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return;
  //  if (!((ops_checkpointing_options & OPS_CHECKPOINT_MANUAL_DATLIST) &&
  //  !(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW ||
  //  ops_checkpointing_options & OPS_CHECKPOINT_MANUAL))) {
  //    if (OPS_instance::getOPSInstance()->OPS_diags>1) ops_printf("Warning: ops_checkpointing_manual_datlist
  //    called, but checkpointing options do not match\n");
  //    return;
  //  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
    if (ops_pre_backup_phase) {
      double cpu, t1, t2, t3, t4;
      ops_timers_core(&cpu, &t1);
      // increment call counter (as if we began checkpointing at the next loop)
      // and save the list of datasets + aux info
      ops_call_counter++;

      ops_backup_point = ops_call_counter;
#ifdef OPS_CHK_THREAD
      // if spin-off thread, and it is still working, we need to wait for it to
      // finish
      if (OPS_instance::getOPSInstance()->ops_thread_offload) {
        if (OPS_instance::getOPSInstance()->OPS_diags > 2 && ops_ramdisk_ctrl_finish)
          printf("Main thread waiting for previous checkpoint completion\n");
        while (ops_ramdisk_ctrl_finish)
          usleep(10000);
      }
#endif
      // Remove previous files, open new ones
      ops_checkpoint_prepare_files();

      // Save all the control variables
      ops_timers_core(&cpu, &t3);
      ops_checkpointing_save_control(file);
      if (ops_duplicate_backup) {
        ops_checkpointing_save_control(file_dup);
      }

#ifdef OPS_CHK_THREAD
      if (OPS_instance::getOPSInstance()->ops_thread_offload && !ops_ramdisk_initialised) {
        long cum_size = 0;
        for (int i = 0; i < ndats; i++) {
          long size = datlist[i]->elem_size;
          for (int d = 0; d < datlist[i]->block->dims; d++)
            size *= datlist[i]->size[d];
          cum_size += size;
        }
        if (ops_duplicate_backup)
          ops_ramdisk_init(2l * cum_size + (2l * cum_size) / 5l);
        else
          ops_ramdisk_init(cum_size + cum_size / 5l);
      }
#endif

      // write datasets
      for (int i = 0; i < ndats; i++)
        save_dat(datlist[i]);
      ops_timers_core(&cpu, &t4);
      if (OPS_instance::getOPSInstance()->OPS_diags > 5)
        printf("Written new file %g\n", t4 - t3);

      // Close files
      ops_checkpoint_complete();

      // finished backing up, reset everything, prepare to be backed up at a
      // later point
      OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_GATHER;
      ops_pre_backup_phase = false;
      ops_call_counter--;
      ops_timers_core(&cpu, &t2);
      OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;
      if (OPS_instance::getOPSInstance()->OPS_diags > 1)
        ops_printf("\nCheckpoint created (manual datlist) in %g seconds\n",
                   t2 - t1);
    }
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    // do nothing, will get triggered anyway
  }
}

bool ops_checkpointing_fastfw(int nbytes, char *payload) {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return false;
  //  if (!((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW) &&
  //  !(ops_checkpointing_options & OPS_CHECKPOINT_MANUAL_DATLIST ||
  //  ops_checkpointing_options & OPS_CHECKPOINT_MANUAL))) {
  //    if (OPS_instance::getOPSInstance()->OPS_diags>1) ops_printf("Warning: ops_checkpointing_fastfw called,
  //    but checkpointing options do not match\n");
  //    return false;
  //  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
    if (ops_pre_backup_phase) {
      OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_BEGIN;
      ops_pre_backup_phase = false;
      OPS_checkpointing_payload_nbytes = nbytes;
      OPS_checkpointing_payload = payload;
    }
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_GATHER;
    double cpu, now, t2;
    ops_timers_core(&cpu, &now);
    ops_last_checkpoint = now;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_restore_dataset(item->dat);
      OPS_instance::getOPSInstance()->OPS_dat_status[item->dat->index] = OPS_UNDECIDED;
    }
    check_hdf5_error(H5LTread_dataset(file, "OPS_checkpointing_payload",
                                      H5T_NATIVE_CHAR, payload));
    check_hdf5_error(H5Fclose(file));
    ops_timers_core(&cpu, &t2);
    OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - now;
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_printf("\nRestored at fast-forward point (in %g seconds), continuing "
                 "normal execution...\n",
                 t2 - now);
    return true;
  }
  return false;
}

bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist,
                                             int nbytes, char *payload) {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return false;
  //  if (!((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW &&
  //  ops_checkpointing_options & OPS_CHECKPOINT_MANUAL_DATLIST) &&
  //  !(ops_checkpointing_options & OPS_CHECKPOINT_MANUAL))) {
  //    if (OPS_instance::getOPSInstance()->OPS_diags>1) ops_printf("Warning:
  //    ops_checkpointing_manual_datlist_fastfw called, but checkpointing
  //    options do not match\n");
  //    return false;
  //  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
    if (ops_pre_backup_phase) {
      OPS_checkpointing_payload_nbytes = nbytes;
      OPS_checkpointing_payload = payload;
      ops_checkpointing_manual_datlist(ndats, datlist);
    }
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    ops_checkpointing_fastfw(nbytes, payload);
    return true;
  }
  return false;
}

bool ops_checkpointing_manual_datlist_fastfw_trigger(int ndats,
                                                     ops_dat *datlist,
                                                     int nbytes,
                                                     char *payload) {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return false;
  if (!(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW &&
        ops_checkpointing_options & OPS_CHECKPOINT_MANUAL_DATLIST &&
        ops_checkpointing_options & OPS_CHECKPOINT_MANUAL)) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_printf("Warning: ops_checkpointing_manual_datlist_fastfw_trigger "
                 "called, but checkpointing options do not match\n");
    return false;
  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
    ops_pre_backup_phase = true;
    OPS_checkpointing_payload_nbytes = nbytes;
    OPS_checkpointing_payload = payload;
    ops_checkpointing_manual_datlist(ndats, datlist);
    ops_pre_backup_phase = false;
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    ops_checkpointing_fastfw(nbytes, payload);
    return true;
  }
  return false;
}

void ops_checkpointing_reduction(ops_reduction red) {
  double t1, t2, cpu;
  ops_timers_core(&cpu, &t1);
  if (diagnostics && OPS_instance::getOPSInstance()->OPS_enable_checkpointing) {
    fprintf(diagf, "reduction;red->name\n");
  }
  if (OPS_chk_red_offset + red->size > OPS_chk_red_size &&
      !(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    OPS_chk_red_size *= 2;
    OPS_chk_red_size = MAX(OPS_chk_red_size, 100 * red->size);
    OPS_chk_red_storage = (char *)ops_realloc(OPS_chk_red_storage,
                                              OPS_chk_red_size * sizeof(char));
  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    if (!(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      memcpy(red->data, &OPS_chk_red_storage[OPS_chk_red_offset], red->size);
      OPS_chk_red_offset += red->size;
    }
    ops_timers_core(&cpu, &t2);
    OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;
    return;
  }
  ops_timers_core(&cpu, &t2);
  OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;

  ops_execute_reduction(red);

  ops_timers_core(&cpu, &t1);
  if ((OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER ||
       OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS ||
       OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_BEGIN) &&
      !(ops_checkpointing_options & OPS_CHECKPOINT_MANUAL)) {
    if (!(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      memcpy(&OPS_chk_red_storage[OPS_chk_red_offset], red->data, red->size);
      OPS_chk_red_offset += red->size;
    }
    ops_reduction_counter++;

    // If we are in the checkpointing region, check on every reduction, whether
    // timeout occurred
    if (ops_reduction_counter % ops_sync_frequency == 0 ||
        OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS) {
      double cpu, now;
      ops_timers_core(&cpu, &now);
      double timing[2] = {now - ops_last_checkpoint,
                          (double)(now - ops_last_checkpoint >
                                   (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS
                                        ? defaultTimeout
                                        : ops_checkpoint_interval))};
      ops_arg temp;
      temp.argtype = OPS_ARG_GBL;
      temp.acc = OPS_MAX;
      temp.data = (char *)timing;
      temp.dim = 2; //*sizeof(double);
      ops_mpi_reduce_double(&temp, timing);
      ops_reduction_avg_time = timing[0];
      if (ops_reduction_avg_time < 0.1 * ops_checkpoint_interval)
        ops_sync_frequency *= 2;
      if (ops_reduction_avg_time > 0.3 * ops_checkpoint_interval)
        ops_sync_frequency = ops_sync_frequency - ops_sync_frequency / 2;
      ops_sync_frequency = MAX(ops_sync_frequency, 1);
      if (OPS_instance::getOPSInstance()->OPS_diags > 4)
        ops_printf("ops_sync_frequency %d\n", ops_sync_frequency);
      if (timing[1] == 1.0) {
        ops_reduction_counter = 0;
        if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
          if (OPS_instance::getOPSInstance()->OPS_diags > 4)
            ops_printf("\nIt's time to checkpoint... %s\n", red->name);
          ops_last_checkpoint = now;
          if (ops_pre_backup_phase == true &&
              !(ops_checkpointing_options &
                (OPS_CHECKPOINT_FASTFW | OPS_CHECKPOINT_MANUAL_DATLIST))) {
            if (OPS_instance::getOPSInstance()->OPS_diags > 1)
              ops_printf(
                  "Double timeout for checkpointing forcing immediate begin\n");
            OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_BEGIN;
            ops_pre_backup_phase = false;
          } else
            ops_pre_backup_phase = true;
        } else {
          if (OPS_instance::getOPSInstance()->OPS_diags > 4)
            ops_printf("\nTimeout for checkpoint region...\n");
          OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_END;
        }
      }
    }
  }
  ops_timers_core(&cpu, &t2);
  OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;
}

#define HASHSIZE 5000
unsigned op2_hash(const char *s) {
  unsigned hashval;
  for (hashval = 0; *s != '\0'; s++)
    hashval = *s + 31 * hashval;
  return hashval % HASHSIZE;
}

bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range,
                                   const char *s) {
  if (OPS_instance::getOPSInstance()->backup_state == OPS_NONE)
    return true;
  int loop_id = 0;
  int hash = (int)op2_hash(s);
  for (; loop_id < ops_loop_max; loop_id++) {
    if (ops_loops_hashmap[loop_id] == hash || ops_loops_hashmap[loop_id] == -1)
      break;
  }
  // if (ops_loops_hashmap != NULL && ops_loops_hashmap[loop_id] != -1 &&
  // ops_loops_hashmap[loop_id] != hash) loop_id++;

  if (ops_loops_hashmap == NULL ||
      loop_id >= ops_loop_max) { // if we ran out of space for storing per loop
                                 // data, allocate some more
    // printf("Allocing more storage for loops: ops_loop_max =
    // %d\n",ops_loop_max);
    ops_loop_max += 100;
    ops_loops_hashmap =
        (int *)ops_realloc(ops_loops_hashmap, ops_loop_max * sizeof(int));
    for (int i = ops_loop_max - 100; i < ops_loop_max; i++) {
      ops_loops_hashmap[i] = -1;
    }
  }
  ops_loops_hashmap[loop_id] = hash;
  // printf("Loop %s id %d\n", s, loop_id);
  return ops_checkpointing_before(args, nargs, range, loop_id);
}

void ops_strat_gather_statistics(ops_arg *args, int nargs, int loop_id,
                                 int *range);
bool ops_strat_should_backup(ops_arg *args, int nargs, int loop_id, int *range);
void ops_statistics_exit();

/*
void ops_strat_gather_statistics(ops_arg *args, int nargs, int loop_id, int
*range) {}
bool ops_strat_should_backup(ops_arg *args, int nargs, int loop_id, int *range)
{
  //Initial strategy: if this range is over a very small part (one fifth) of the
dataset, then don't do a checkpoint here
  int i = 0;
  for (; i < nargs; i ++) if (args[i].argtype == OPS_ARG_DAT &&
args[i].dat->e_dat == 0) break;
  if (i == nargs) return false;
  int larger = 1;
  for (int d = 0; d < args[i].dat->block->dims; d++) {
    if ((double)(range[2*d+1]-range[2*d]) <= 1) larger = 0;
  }
  return (larger == 1);
}
*/

/**
* Checkpointing utility function called right before the execution of the
* parallel loop itself.
*/
bool ops_checkpointing_before(ops_arg *args, int nargs, int *range,
                              int loop_id) {
  if (OPS_instance::getOPSInstance()->backup_state == OPS_NONE)
    return true;
  if (diagnostics) {
    fprintf(diagf, "loop %d;%d;%d;%d;%d;%d\n", loop_id, nargs, range[0],
            range[1], range[2], range[3]);
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT) {
        fprintf(diagf, "dat;%s;%s;%s;%d;%d\n", args[i].dat->name,
                args[i].stencil->name, args[i].dat->type, args[i].acc,
                args[i].opt);
      } else if (args[i].argtype == OPS_ARG_GBL) {
        fprintf(diagf, "gbl;%d;%d\n", args[i].dim, args[i].acc);
      } else if (args[i].argtype == OPS_ARG_IDX) {
        fprintf(diagf, "idx\n");
      }
    }
  }

  double cpu, t1, t2;
  ops_timers_core(&cpu, &t1);

  ops_call_counter++;
  for (int i = 0; i < nargs;
       i++) { // flag variables that are touched (we do this every time it is
              // called, may be a little redundant), should make a loop_id
              // filter
    if (args[i].argtype == OPS_ARG_DAT && args[i].acc != OPS_READ &&
        args[i].opt == 1) {
      OPS_instance::getOPSInstance()->OPS_dat_ever_written[args[i].dat->index] = true;
    }
  }

  if (ops_call_counter == ops_backup_point &&
      OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN &&
      !(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW))
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_RESTORE;

  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
    if (!(ops_checkpointing_options &
          (OPS_CHECKPOINT_FASTFW | OPS_CHECKPOINT_MANUAL_DATLIST)))
      ops_strat_gather_statistics(args, nargs, loop_id, range);

    if (ops_pre_backup_phase &&
        !(ops_checkpointing_options &
          (OPS_CHECKPOINT_FASTFW | OPS_CHECKPOINT_MANUAL_DATLIST)) &&
        ops_strat_should_backup(args, nargs, loop_id, range)) {
      OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_BEGIN;
      ops_pre_backup_phase = false;
    }
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    return false;
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_RESTORE) {
    // this is the point where we do the switch from restore mode to computation
    // mode
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_GATHER;
    double cpu, now;
    ops_timers_core(&cpu, &now);
    ops_last_checkpoint = now;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_restore_dataset(item->dat);
      OPS_instance::getOPSInstance()->OPS_dat_status[item->dat->index] = OPS_UNDECIDED;
    }

    int total_size = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_reduction_index; i++)
      if (OPS_instance::getOPSInstance()->OPS_reduction_list[i]->initialized == 1)
        total_size += OPS_instance::getOPSInstance()->OPS_reduction_list[i]->size;
    char *reduction_state = (char *)ops_malloc(total_size * sizeof(char));
    check_hdf5_error(H5LTread_dataset(file, "reduction_state", H5T_NATIVE_CHAR,
                                      reduction_state));
    total_size = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_reduction_index; i++) {
      if (OPS_instance::getOPSInstance()->OPS_reduction_list[i]->initialized == 1) {
        memcpy(OPS_instance::getOPSInstance()->OPS_reduction_list[i]->data, &reduction_state[total_size],
               OPS_instance::getOPSInstance()->OPS_reduction_list[i]->size);
        total_size += OPS_instance::getOPSInstance()->OPS_reduction_list[i]->size;
      }
    }
    free(reduction_state);
    check_hdf5_error(H5Fclose(file));
    ops_timers_core(&cpu, &t2);
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_printf("\nRestored in %g seconds, continuing normal execution...\n",
                 t2 - t1);
  }

  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_BEGIN) {
    ops_backup_point = ops_call_counter;
#ifdef OPS_CHK_THREAD
    // if spin-off thread, and it is still working, we need to wait for it to
    // finish
    if (OPS_instance::getOPSInstance()->ops_thread_offload) {
      if (OPS_instance::getOPSInstance()->OPS_diags > 2 && ops_ramdisk_ctrl_finish)
        printf("Main thread waiting for previous checkpoint completion\n");
      while (ops_ramdisk_ctrl_finish)
        usleep(10000);
    }
#endif

    // Remove previous files, create new ones
    ops_checkpoint_prepare_files();

    // write all control
    ops_checkpointing_save_control(file);
    if (ops_duplicate_backup) {
      ops_checkpointing_save_control(file_dup);
    }

#ifdef OPS_CHK_THREAD
    if (OPS_instance::getOPSInstance()->ops_thread_offload) {
      if (ops_best_backup_point_size == 0) {
        ops_dat_entry *item, *tmp_item;
        int dat_count = 0;
        for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
          tmp_item = TAILQ_NEXT(item, entries);
          if (OPS_instance::getOPSInstance()->OPS_dat_ever_written[item->dat->index]) {
            long size = item->dat->elem_size;
            for (int d = 0; d < item->dat->block->dims; d++)
              size *= item->dat->size[d];
            ops_best_backup_point_size += size;
            dat_count++;
          }
        }
        ops_best_backup_point_size = ops_best_backup_point_size / 5l;
        if (OPS_instance::getOPSInstance()->OPS_diags > 5)
          printf("Approximated ramdisk size %ld\n", ops_best_backup_point_size);
      }
      if (ops_duplicate_backup)
        ops_ramdisk_init(2l * ops_best_backup_point_size +
                         (2l * ops_best_backup_point_size) / 5l);
      else
        ops_ramdisk_init(ops_best_backup_point_size +
                         ops_best_backup_point_size / 5l);
    }
#endif

    // write datasets
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT &&
          OPS_instance::getOPSInstance()->OPS_dat_ever_written[args[i].dat->index] &&
          OPS_instance::getOPSInstance()->OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
          args[i].acc != OPS_WRITE && args[i].opt == 1) {
        // write it to disk
        save_dat(args[i].dat);
      } else if (args[i].argtype == OPS_ARG_DAT &&
                 OPS_instance::getOPSInstance()->OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
                 args[i].acc == OPS_WRITE && args[i].opt == 1) {
        save_dat_partial(args[i].dat, range);
      }
    }

    // Check if we are done
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_IN_PROCESS;
    bool done = true;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      if (OPS_instance::getOPSInstance()->OPS_dat_status[item->dat->index] == OPS_UNDECIDED &&
          OPS_instance::getOPSInstance()->OPS_dat_ever_written[item->dat->index]) {
        done = false;
      }
    }
    if (done)
      OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_END;
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS) {
    // when we have already begun backing up, but there are a few datasets that
    // are undecided (whether or not they should be backed up)
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT &&
          OPS_instance::getOPSInstance()->OPS_dat_ever_written[args[i].dat->index] &&
          OPS_instance::getOPSInstance()->OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
          args[i].acc != OPS_WRITE && args[i].opt == 1) {
        save_dat(args[i].dat);
      } else if (args[i].argtype == OPS_ARG_DAT &&
                 OPS_instance::getOPSInstance()->OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
                 args[i].acc == OPS_WRITE && args[i].opt == 1) {
        save_dat_partial(args[i].dat, range);
      }
    }
    bool done = true;
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      if (OPS_instance::getOPSInstance()->OPS_dat_status[item->dat->index] == OPS_UNDECIDED &&
          OPS_instance::getOPSInstance()->OPS_dat_ever_written[item->dat->index]) {
        done = false;
      }
    }
    // if there are no undecided datasets left
    if (done)
      OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_END;
  }

  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_END) {
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype == OPS_ARG_DAT &&
          OPS_instance::getOPSInstance()->OPS_dat_ever_written[args[i].dat->index] &&
          OPS_instance::getOPSInstance()->OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
          args[i].acc != OPS_WRITE && args[i].opt == 1) {
        save_dat(args[i].dat);
      } else if (args[i].argtype == OPS_ARG_DAT &&
                 OPS_instance::getOPSInstance()->OPS_dat_status[args[i].dat->index] == OPS_UNDECIDED &&
                 args[i].acc == OPS_WRITE && args[i].opt == 1) {
        save_dat_partial(args[i].dat, range);
      }
    }

    // either timed out or ended, if it's the former, back up everything left
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      if (OPS_instance::getOPSInstance()->OPS_dat_status[item->dat->index] == OPS_UNDECIDED &&
          OPS_instance::getOPSInstance()->OPS_dat_ever_written[item->dat->index]) {
        save_dat(item->dat);
        if (OPS_instance::getOPSInstance()->OPS_diags > 4)
          printf("Timeout, force saving %s\n", item->dat->name);
      }
    }

    if (OPS_instance::getOPSInstance()->OPS_diags > 6) {
      for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
        tmp_item = TAILQ_NEXT(item, entries);
        ops_printf("Ever written %s %d\n", item->dat->name,
                   OPS_instance::getOPSInstance()->OPS_dat_ever_written[item->dat->index]);
      }
    }

    ops_checkpoint_complete();

    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_printf("\nCheckpoint created %d bytes reduction data\n",
                 OPS_chk_red_offset);
    // finished backing up, reset everything, prepare to be backed up at a later
    // point
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_GATHER;
    for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      OPS_instance::getOPSInstance()->OPS_dat_status[item->dat->index] = OPS_UNDECIDED;
    }
  }
  ops_timers_core(&cpu, &t2);
  OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;
  return true;
}

void ops_checkpointing_exit() {
  if (OPS_instance::getOPSInstance()->backup_state != OPS_NONE) {
    ops_statistics_exit();

#ifdef OPS_CHK_THREAD
    if (OPS_instance::getOPSInstance()->ops_thread_offload && ops_ramdisk_initialised) {
      ops_ramdisk_ctrl_exit = 1;
      pthread_join(thread, NULL);
      free(ops_ramdisk_buffer);
      free(ops_ramdisk_item_queue);
    }
#endif

    if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory) {
      for (int i = ops_ramdisk_item_queue_tail; i < ops_ramdisk_item_queue_head;
           i++)
        free(ops_ramdisk_item_queue[i].data);
      free(ops_ramdisk_item_queue);
    }

    if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS) {
      check_hdf5_error(H5Fclose(file));
      if (ops_duplicate_backup)
        check_hdf5_error(H5Fclose(file_dup));
    }

    remove(filename);
    if (ops_duplicate_backup)
      remove(filename_dup);

    if (OPS_instance::getOPSInstance()->ops_lock_file)
      ops_create_lock_done(filename);

    if (diagnostics) {
      fprintf(diagf, "FINISHED\n");
      fclose(diagf);
    }

    if (OPS_instance::getOPSInstance()->OPS_diags > 2) {
      double moments_time[2] = {0.0};
      ops_compute_moment(ops_chk_save, &moments_time[0], &moments_time[1]);
      moments_time[1] =
          sqrt(moments_time[1] - moments_time[0] * moments_time[0]);
      double moments_time1[2] = {0.0};
      ops_compute_moment(ops_chk_dup, &moments_time1[0], &moments_time1[1]);
      moments_time1[1] =
          sqrt(moments_time1[1] - moments_time1[0] * moments_time1[0]);
      double moments_time2[2] = {0.0};
      ops_compute_moment(ops_chk_write, &moments_time2[0], &moments_time2[1]);
      moments_time2[1] =
          sqrt(moments_time2[1] - moments_time2[0] * moments_time2[0]);
      ops_printf(
          "Time spent in save_dat: %g (%g) (dup %g (%g), hdf5 write %g (%g)\n",
          moments_time[0], moments_time[1], moments_time1[0], moments_time1[1],
          moments_time2[0], moments_time2[1]);
#ifdef OPS_CHK_THREAD
      if (OPS_instance::getOPSInstance()->ops_thread_offload && ops_ramdisk_initialised) {
        double moments_time[2] = {0.0};
        ops_compute_moment(ops_chk_thr_queue, &moments_time[0],
                           &moments_time[1]);
        moments_time[1] =
            sqrt(moments_time[1] - moments_time[0] * moments_time[0]);
        ops_printf("Time spend in copy to ramdisk %g (%g)\n", moments_time[0],
                   moments_time[1]);
      }
#endif
    }
    ops_call_counter = 0;
    ops_backup_point = -1;
    ops_checkpoint_interval = -1;
    ops_last_checkpoint = -1;
    ops_pre_backup_phase = false;
    OPS_instance::getOPSInstance()->backup_state = OPS_NONE;
    ops_loop_max = 0;
    free(ops_loops_hashmap);
    ops_loops_hashmap = NULL;
    free(OPS_instance::getOPSInstance()->OPS_dat_ever_written);
    OPS_instance::getOPSInstance()->OPS_dat_ever_written = NULL;
    free(OPS_instance::getOPSInstance()->OPS_dat_status);
    OPS_instance::getOPSInstance()->OPS_dat_status = NULL;
    OPS_partial_buffer_size = 0;
    free(OPS_partial_buffer);
    OPS_partial_buffer = NULL;
    free(ops_inm_ctrl.reduction_state);
    free(ops_inm_ctrl.OPS_checkpointing_payload);
    free(ops_inm_ctrl.OPS_chk_red_storage);

    // filename =  "";
  }
}

#ifdef __cplusplus
}
#endif

#else

class OPS_instance_checkpointing {};

#ifdef __cplusplus
extern "C" {
#endif

bool ops_checkpointing_init(const char *filename, double interval,
                            int options) {
  return false;
}
bool ops_checkpointing_before(ops_arg *args, int nargs, int *range,
                              int loop_id) {
  return true;
}
bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range,
                                   const char *s) {
  return true;
}
void ops_checkpointing_exit() {}

void ops_checkpointing_reduction(ops_reduction red) {
  ops_execute_reduction(red);
}

void ops_checkpointing_initphase_done() {}

void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist) {}
bool ops_checkpointing_fastfw(int nbytes, char *payload) { return false; }
bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist,
                                             int nbytes, char *payload) {
  return false;
}

bool ops_checkpointing_manual_datlist_fastfw_trigger(int ndats,
                                                     ops_dat *datlist,
                                                     int nbytes,
                                                     char *payload) {
  return false;
}
#ifdef __cplusplus
}
#endif

#endif
