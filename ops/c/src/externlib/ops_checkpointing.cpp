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
#include <ops_exceptions.h>

#include <chrono>
#include <thread>

#ifdef CHECKPOINTING

#ifndef defaultTimeout
#define defaultTimeout 2.0
#endif
#define ROUND64L(x) ((((x)-1l) / 64l + 1l) * 64l)

// TODO: True temp dat support

// Internal function definitions
void ops_download_dat(ops_dat dat);
void ops_upload_dat(ops_dat dat);
bool ops_checkpointing_filename(const char *file_name, std::string &filename_out,
                                std::string &filename_out2);
void ops_checkpointing_calc_range(ops_dat dat, const int *range,
                                  int *saved_range);
void ops_checkpointing_duplicate_data(ops_dat dat, int my_type, int my_nelems,
                                      char *my_data, int *my_range,
                                      int *rm_type, int *rm_elems,
                                      char **rm_data, int **rm_range);


void ops_usleep(size_t length)
{
   std::chrono::microseconds us(length);
   std::this_thread::sleep_for(us);
}


struct ops_strat_data;
#include "ops_checkpointing_class.h"

OPS_instance_checkpointing::OPS_instance_checkpointing() {
    ops_call_counter = 0;
    ops_backup_point_g = -1;
    ops_best_backup_point_g = -1;
    ops_best_backup_point_size = 0;
    ops_checkpoint_interval = -1;
    ops_last_checkpoint = -1;
    ops_pre_backup_phase = false;
    ops_duplicate_backup = false;
    ops_loop_max = 0;
    ops_loops_hashmap = NULL;
    OPS_chk_red_size = 0;
    OPS_chk_red_offset_g = 0;
    OPS_chk_red_storage_g = NULL;

    OPS_partial_buffer_size = 0;
    OPS_partial_buffer = NULL;

    OPS_checkpointing_payload_nbytes_g = 0;
    OPS_checkpointing_payload_g = NULL;

    ops_reduction_counter = 0;
    ops_sync_frequency = 50;
    ops_reduction_avg_time = 0.0;

    ops_checkpointing_options = 0;
    ops_chk_write = 0.0;
    ops_chk_dup = 0.0;
    ops_chk_save = 0.0;

    diagnostics = 0;

    ops_strat = NULL;
  }

#define ops_call_counter OPS_instance::getOPSInstance()->checkpointing_instance->ops_call_counter
#define ops_backup_point_g OPS_instance::getOPSInstance()->checkpointing_instance->ops_backup_point_g
#define ops_best_backup_point_g OPS_instance::getOPSInstance()->checkpointing_instance->ops_best_backup_point_g
#define ops_best_backup_point_size OPS_instance::getOPSInstance()->checkpointing_instance->ops_best_backup_point_size
#define ops_checkpoint_interval OPS_instance::getOPSInstance()->checkpointing_instance->ops_checkpoint_interval
#define ops_last_checkpoint OPS_instance::getOPSInstance()->checkpointing_instance->ops_last_checkpoint
#define ops_pre_backup_phase OPS_instance::getOPSInstance()->checkpointing_instance->ops_pre_backup_phase
#define ops_duplicate_backup OPS_instance::getOPSInstance()->checkpointing_instance->ops_duplicate_backup
#define ops_loop_max OPS_instance::getOPSInstance()->checkpointing_instance->ops_loop_max
#define ops_loops_hashmap OPS_instance::getOPSInstance()->checkpointing_instance->ops_loops_hashmap
#define OPS_chk_red_size OPS_instance::getOPSInstance()->checkpointing_instance->OPS_chk_red_size
#define OPS_chk_red_offset_g OPS_instance::getOPSInstance()->checkpointing_instance->OPS_chk_red_offset_g
#define OPS_chk_red_storage_g OPS_instance::getOPSInstance()->checkpointing_instance->OPS_chk_red_storage_g
#define OPS_partial_buffer_size OPS_instance::getOPSInstance()->checkpointing_instance->OPS_partial_buffer_size
#define OPS_partial_buffer OPS_instance::getOPSInstance()->checkpointing_instance->OPS_partial_buffer
#define OPS_checkpointing_payload_nbytes_g OPS_instance::getOPSInstance()->checkpointing_instance->OPS_checkpointing_payload_nbytes_g
#define OPS_checkpointing_payload_g OPS_instance::getOPSInstance()->checkpointing_instance->OPS_checkpointing_payload_g
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
void __check_hdf5_error(herr_t err, const char *_file, const int line) {
  if (err < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: " << _file << ":" << line << " OPS_HDF5_error() Runtime API error " << err << " " << params;
    throw ex;
  }
}

bool file_exists(const std::string &file_name) {
  if (FILE *_file = fopen(file_name.c_str(), "r")) {
    fclose(_file);
    return true;
  }
  return false;
}

void ops_create_lock(const std::string &fname) {
   std::string buffer = fname + ".lock";
  //sprintf(buffer, "%s.lock", fname);
  if (FILE *lock = fopen(buffer.c_str(), "w+")) {
    fclose(lock);
  } else {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      OPS_instance::getOPSInstance()->ostream() << "Warning: lock file already exists\n";
  }
}

void ops_create_lock_done(const std::string &fname) {
   std::string buffer = fname + ".done";
  if (FILE *lock = fopen(buffer.c_str(), "w+")) {
    fclose(lock);
  } else {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      OPS_instance::getOPSInstance()->ostream() <<  "Warning: lock done file already exists\n";
  }
}

void ops_pack_chk(const char *__restrict src, char *__restrict dest,
                  const int count, const int blocklength, const int stride) {
  for (int i = 0; i < count; i++) {
    memcpy(dest, src, blocklength);
    src += stride;
    dest += blocklength;
  }
}

void ops_unpack_chk(char *__restrict dest, const char *__restrict src,
                    const int count, const int blocklength, const int stride) {
  for (int i = 0; i < count; i++) {
    memcpy(dest, src, blocklength);
    src += blocklength;
    dest += stride;
  }
}

typedef struct {
  ops_dat dat;
  hid_t outfile;
  size_t size;
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
volatile size_t ops_ramdisk_size = 0;
volatile size_t ops_ramdisk_head = 0;
volatile size_t ops_ramdisk_tail = 0;

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
          OPS_instance::getOPSInstance()->ostream() << "Thread saving partial " <<
                 ops_ramdisk_item_queue[tail].dat->name << '\n';
        save_to_hdf5_partial(ops_ramdisk_item_queue[tail].dat,
                             ops_ramdisk_item_queue[tail].outfile,
                             ops_ramdisk_item_queue[tail].size /
                                 (ops_ramdisk_item_queue[tail].dat->elem_size /
                                  ops_ramdisk_item_queue[tail].dat->dim),
                             ops_ramdisk_item_queue[tail].saved_range,
                             ops_ramdisk_item_queue[tail].data);
      } else {
        if (OPS_instance::getOPSInstance()->OPS_diags > 5)
          OPS_instance::getOPSInstance()->ostream() << "Thread saving full " <<
                 ops_ramdisk_item_queue[tail].dat->name << '\n';
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
        ops_usleep((size_t)(ops_checkpoint_interval * 300000.0));
      }
      ops_usleep(100000);
    }
  }
  pthread_exit(NULL); // Perhaps return whether everything was properly saved
}

void OPS_reallocate_ramdisk(size_t size) {
  // Wait for the queue to drain
  if (OPS_instance::getOPSInstance()->OPS_diags > 2 &&
      (ops_ramdisk_item_queue_head != ops_ramdisk_item_queue_tail))
    OPS_instance::getOPSInstance()->ostream() << "Main thread waiting for ramdisk reallocation head "
        << ops_ramdisk_item_queue_head << " tail " << ops_ramdisk_item_queue_tail << '\n';
  while (ops_ramdisk_item_queue_head != ops_ramdisk_item_queue_tail)
    ops_usleep(20000);
  ops_ramdisk_size = ROUND64L(size);
  ops_ramdisk_buffer =
      (char *)ops_realloc(ops_ramdisk_buffer, ops_ramdisk_size * sizeof(char));
  ops_ramdisk_tail = 0;
  ops_ramdisk_head = 0;
}

void ops_ramdisk_init(size_t size) {
  if (ops_ramdisk_initialised)
    return;
  ops_ramdisk_size = ROUND64L(size);
  ops_ramdisk_buffer = (char *)ops_malloc(ops_ramdisk_size * sizeof(char));
  ops_ramdisk_tail = 0;
  ops_ramdisk_head = 0;
  ops_ramdisk_item_queue = (ops_ramdisk_item *)ops_calloc(
      3 * OPS_instance::getOPSInstance()->OPS_dat_index , sizeof(ops_ramdisk_item));
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
  size_t sizeround64 = ROUND64L(size);
  if (ops_ramdisk_size < sizeround64)
    OPS_reallocate_ramdisk(
        ROUND64L(3l * (size_t)sizeround64 + (size_t)sizeround64 / 5l + 64));
  // Copy data to ramdisk
  size_t tail = ops_ramdisk_tail;
  size_t head = ops_ramdisk_head;
  while ((head < tail && head + sizeround64 >= tail) ||
         (head + sizeround64 >= ops_ramdisk_size && sizeround64 >= tail)) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf2(OPS_instance::getOPSInstance(), "Main thread waiting for ramdisk room %ld < %ld && %ld + %ld >= "
             "%ld or %ld + %ld  >= %ld && %ld >= %ld\n",
             head, tail, head, sizeround64, tail, head, sizeround64,
             ops_ramdisk_size, sizeround64, tail);
    ops_usleep(10000);
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
    OPS_instance::getOPSInstance()->ostream() << "Main thread waiting for ramdisk item queue room " <<
           item_idx_next << "==" << ops_ramdisk_item_queue_tail << '\n';
  while (item_idx_next == ops_ramdisk_item_queue_tail)
    ops_usleep(10000);

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

  std::string buf = dat->name;
  buf += " saved range";
  dims[0] = 2 * OPS_MAX_DIM;
  check_hdf5_error(
      H5LTmake_dataset(outfile, buf.c_str(), 1, dims, H5T_NATIVE_INT, saved_range));
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
    OPS_instance::getOPSInstance()->ostream() << "Backed up " << dat->name << '\n';
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
  size_t prod[OPS_MAX_DIM + 1];
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
    OPS_instance::getOPSInstance()->ostream() << "Backed up " << dat->name << " (partial)\n";
  ops_timers_core(&c1, &t2);
  ops_chk_save += t2 - t1;
}

void ops_restore_dataset(ops_dat dat) {
  if (!H5LTfind_dataset(file, dat->name))
    return;
  std::string buf = dat->name;
  buf += "_saved_range";
  hsize_t dims[1] = {0};
  dims[0] = 2 * OPS_MAX_DIM;
  // if no range specified, just read back the entire dataset
  if (!H5LTfind_dataset(file, buf.c_str())) {
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
      OPS_instance::getOPSInstance()->ostream() << "Restored " << dat->name << '\n';
  } else { // if a range is specified,
    int saved_range[OPS_MAX_DIM * 2];
    check_hdf5_error(H5LTread_dataset(file, buf.c_str(), H5T_NATIVE_INT, saved_range));
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
      OPS_instance::getOPSInstance()->ostream() << "Restored " << dat->name << " (partial)\n";
  }
}

void ops_checkpoint_prepare_files() {
  if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 5)
      if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "New inmemory checkpoint, purging previous\n";
    for (int i = ops_ramdisk_item_queue_tail; i < ops_ramdisk_item_queue_head;
         i++) {
      ops_free(ops_ramdisk_item_queue[i].data);
      ops_ramdisk_item_queue[i].data = NULL;
      ops_ramdisk_item_queue[i].dat = NULL;
    }
    ops_ramdisk_item_queue_tail = 0;
    ops_ramdisk_item_queue_head = 0;
  } else {
    double cpu, t3, t4;
    ops_timers_core(&cpu, &t3);
    if (file_exists(filename))
      remove(filename.c_str());
    if (ops_duplicate_backup && file_exists(filename_dup))
      remove(filename_dup.c_str());
    ops_timers_core(&cpu, &t4);
    if (OPS_instance::getOPSInstance()->OPS_diags > 5)
      OPS_instance::getOPSInstance()->ostream() << "Removed previous file " << t4 - t3 << "s\n";

    // where we start backing up stuff
    ops_timers_core(&cpu, &t3);
    file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (ops_duplicate_backup)
      file_dup =
          H5Fcreate(filename_dup.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ops_timers_core(&cpu, &t4);
    if (OPS_instance::getOPSInstance()->OPS_diags > 5)
      OPS_instance::getOPSInstance()->ostream() << "Opened new file " << t4 - t3 << "s\n";
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

struct ops_checkpoint_inmemory_control {
  int ops_backup_point;
  int ops_best_backup_point;
  char *reduction_state;
  int reduction_state_size;
  char *OPS_checkpointing_payload;
  int OPS_checkpointing_payload_nbytes;
  int OPS_chk_red_offset;
  char *OPS_chk_red_storage;
};

ops_checkpoint_inmemory_control ops_inm_ctrl;

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
  ops_free(ops_inm_ctrl.reduction_state);
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
    OPS_instance::getOPSInstance()->ostream() << "Error: OPS received unknown signal " << signo << '\n';
    return;
  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS) {
    OPS_instance::getOPSInstance()->ostream() <<  "Error: failed in checkpointing region, aborting checkpoint\n";
    return;
  }
  if (OPS_instance::getOPSInstance()->OPS_diags > 1)
    OPS_instance::getOPSInstance()->ostream() <<  "Process received SIGINT, dumping checkpoint from memory...\n";

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
    ops_free(ops_ramdisk_item_queue[i].data);
  }
  ops_free(ops_ramdisk_item_queue);

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
    if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "//\n// OPS Checkpointing -- Backup mode\n//\n";
    return false;
  } else {
    file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_LEADIN;
    if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() <<  "//\n// OPS Checkpointing -- Restore mode\n//\n";

    if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      // we are not restoring anything during lead-in and will be resuming
      // execution the first time we encounter the API call
    } else {
      double cpu, t1, t2;
      ops_timers_core(&cpu, &t1);
      // read backup point
      check_hdf5_error(H5LTread_dataset(file, "ops_backup_point",
                                        H5T_NATIVE_INT, &ops_backup_point_g));
      check_hdf5_error(H5LTread_dataset(file, "ops_best_backup_point",
                                        H5T_NATIVE_INT,
                                        &ops_best_backup_point_g));

      // loading of datasets is postponed until we reach the restore point

      // restore reduction storage
      check_hdf5_error(H5LTread_dataset(file, "OPS_chk_red_offset",
                                        H5T_NATIVE_INT, &OPS_chk_red_offset_g));
      OPS_chk_red_storage_g =
          (char *)ops_malloc(OPS_chk_red_offset_g * sizeof(char));
      OPS_chk_red_size = OPS_chk_red_offset_g;
      OPS_chk_red_offset_g = 0;
      check_hdf5_error(H5LTread_dataset(file, "OPS_chk_red_storage",
                                        H5T_NATIVE_CHAR, OPS_chk_red_storage_g));
      ops_timers_core(&cpu, &t2);
      OPS_instance::getOPSInstance()->OPS_checkpointing_time += t2 - t1;
    }
    return true;
  }
}

/**
* Initialises checkpointing using the given filename
*/
OPS_FTN_INTEROP
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
      OPS_instance::getOPSInstance()->ostream() << "Warning: in-memory checkpointing and thread-offload " <<
                 "checkpointing are mutually exclusive. Thread-offload will be " <<
                 "switched off\n";
      OPS_instance::getOPSInstance()->ops_thread_offload = 0;
    }
#endif
    if (signal(SIGINT, ops_chkp_sig_handler) == SIG_ERR) {
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS can't catch SIGINT - disable in-memory checkpointing");
    }
    ops_ramdisk_item_queue = (ops_ramdisk_item *)ops_calloc(
        3 * OPS_instance::getOPSInstance()->OPS_dat_index , sizeof(ops_ramdisk_item));
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
  OPS_instance::getOPSInstance()->OPS_dat_status = (ops_checkpoint_types *)ops_calloc(
      OPS_instance::getOPSInstance()->OPS_dat_index , sizeof(ops_checkpoint_types));

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

OPS_FTN_INTEROP
void ops_checkpointing_initphase_done() {
  if (!OPS_instance::getOPSInstance()->OPS_enable_checkpointing)
    return;
  if (OPS_instance::getOPSInstance()->backup_state == OPS_NONE) {
    ops_checkpointing_initstate();
  }
}

void ops_checkpointing_save_control(hid_t file_out) {
  // write control variables and all initialized reduction handles

  ops_inm_ctrl.ops_backup_point = ops_backup_point_g;
  ops_inm_ctrl.ops_best_backup_point = ops_best_backup_point_g;

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
    ops_free(ops_inm_ctrl.reduction_state);
  ops_inm_ctrl.reduction_state = reduction_state;
  ops_inm_ctrl.reduction_state_size = total_size;

  // Save payload if specified by user
  if ((ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    ops_inm_ctrl.OPS_checkpointing_payload =
        (char *)ops_realloc(ops_inm_ctrl.OPS_checkpointing_payload,
                            OPS_checkpointing_payload_nbytes_g * sizeof(char));
    memcpy(ops_inm_ctrl.OPS_checkpointing_payload, OPS_checkpointing_payload_g,
           OPS_checkpointing_payload_nbytes_g * sizeof(char));
    ops_inm_ctrl.OPS_checkpointing_payload_nbytes =
        OPS_checkpointing_payload_nbytes_g;
  }

  // save reduction history
  ops_inm_ctrl.OPS_chk_red_offset = OPS_chk_red_offset_g;
  ops_inm_ctrl.OPS_chk_red_storage = (char *)ops_realloc(
      ops_inm_ctrl.OPS_chk_red_storage, OPS_chk_red_offset_g * sizeof(char));
  memcpy(ops_inm_ctrl.OPS_chk_red_storage, OPS_chk_red_storage_g,
         OPS_chk_red_offset_g * sizeof(char));

  // write to file
  ops_ctrldump(file_out);
}

OPS_FTN_INTEROP
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

      ops_backup_point_g = ops_call_counter;
#ifdef OPS_CHK_THREAD
      // if spin-off thread, and it is still working, we need to wait for it to
      // finish
      if (OPS_instance::getOPSInstance()->ops_thread_offload) {
        if (OPS_instance::getOPSInstance()->OPS_diags > 2 && ops_ramdisk_ctrl_finish)
          OPS_instance::getOPSInstance()->ostream() << "Main thread waiting for previous checkpoint completion\n";
        while (ops_ramdisk_ctrl_finish)
          ops_usleep(10000);
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
        size_t cum_size = 0;
        for (int i = 0; i < ndats; i++) {
          size_t size = datlist[i]->elem_size;
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
        OPS_instance::getOPSInstance()->ostream() << "Written new file " << t4 - t3 << '\n';

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
        if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "\nCheckpoint created (manual datlist) in " <<
                   t2 - t1 << " seconds\n";
    }
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    // do nothing, will get triggered anyway
  }
}

OPS_FTN_INTEROP
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
      OPS_checkpointing_payload_nbytes_g = nbytes;
      OPS_checkpointing_payload_g = payload;
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
      if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "\nRestored at fast-forward point (in "<<t2 - now <<" seconds), continuing "
                 << "normal execution...\n";
    return true;
  }
  return false;
}

OPS_FTN_INTEROP
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
      OPS_checkpointing_payload_nbytes_g = nbytes;
      OPS_checkpointing_payload_g = payload;
      ops_checkpointing_manual_datlist(ndats, datlist);
    }
  } else if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    ops_checkpointing_fastfw(nbytes, payload);
    return true;
  }
  return false;
}

OPS_FTN_INTEROP
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
      if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "Warning: ops_checkpointing_manual_datlist_fastfw_trigger "
                 << "called, but checkpointing options do not match\n";
    return false;
  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
    ops_pre_backup_phase = true;
    OPS_checkpointing_payload_nbytes_g = nbytes;
    OPS_checkpointing_payload_g = payload;
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
  if (OPS_chk_red_offset_g + red->size > OPS_chk_red_size &&
      !(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
    OPS_chk_red_size *= 2;
    OPS_chk_red_size = MAX(OPS_chk_red_size, 100 * red->size);
    OPS_chk_red_storage_g = (char *)ops_realloc(OPS_chk_red_storage_g,
                                              OPS_chk_red_size * sizeof(char));
  }
  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_LEADIN) {
    if (!(ops_checkpointing_options & OPS_CHECKPOINT_FASTFW)) {
      memcpy(red->data, &OPS_chk_red_storage_g[OPS_chk_red_offset_g], red->size);
      OPS_chk_red_offset_g += red->size;
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
      memcpy(&OPS_chk_red_storage_g[OPS_chk_red_offset_g], red->data, red->size);
      OPS_chk_red_offset_g += red->size;
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
        if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "ops_sync_frequency " << ops_sync_frequency << '\n';
      if (timing[1] == 1.0) {
        ops_reduction_counter = 0;
        if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_GATHER) {
          if (OPS_instance::getOPSInstance()->OPS_diags > 4)
            if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "\nIt's time to checkpoint... "<< red->name << '\n';
          ops_last_checkpoint = now;
          if (ops_pre_backup_phase == true &&
              !(ops_checkpointing_options &
                (OPS_CHECKPOINT_FASTFW | OPS_CHECKPOINT_MANUAL_DATLIST))) {
            if (OPS_instance::getOPSInstance()->OPS_diags > 1)
              if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << 
                  "Double timeout for checkpointing forcing immediate begin\n";
            OPS_instance::getOPSInstance()->backup_state = OPS_BACKUP_BEGIN;
            ops_pre_backup_phase = false;
          } else
            ops_pre_backup_phase = true;
        } else {
          if (OPS_instance::getOPSInstance()->OPS_diags > 4)
            if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "\nTimeout for checkpoint region...\n";
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

  if (ops_call_counter == ops_backup_point_g &&
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
    ops_free(reduction_state);
    check_hdf5_error(H5Fclose(file));
    ops_timers_core(&cpu, &t2);
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "\nRestored in "<<t2 - t1
        <<" seconds, continuing normal execution...\n";
  }

  if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_BEGIN) {
    ops_backup_point_g = ops_call_counter;
#ifdef OPS_CHK_THREAD
    // if spin-off thread, and it is still working, we need to wait for it to
    // finish
    if (OPS_instance::getOPSInstance()->ops_thread_offload) {
      if (OPS_instance::getOPSInstance()->OPS_diags > 2 && ops_ramdisk_ctrl_finish)
        OPS_instance::getOPSInstance()->ostream() << "Main thread waiting for previous checkpoint completion\n";
      while (ops_ramdisk_ctrl_finish)
        ops_usleep(10000);
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
            size_t size = item->dat->elem_size;
            for (int d = 0; d < item->dat->block->dims; d++)
              size *= item->dat->size[d];
            ops_best_backup_point_size += size;
            dat_count++;
          }
        }
        ops_best_backup_point_size = ops_best_backup_point_size / 5l;
        if (OPS_instance::getOPSInstance()->OPS_diags > 5)
          OPS_instance::getOPSInstance()->ostream() << "Approximated ramdisk size " << ops_best_backup_point_size << '\n';
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
          if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "Timeout, force saving " << item->dat->name << '\n';
      }
    }

    if (OPS_instance::getOPSInstance()->OPS_diags > 6) {
      for (item = TAILQ_FIRST(&OPS_instance::getOPSInstance()->OPS_dat_list); item != NULL; item = tmp_item) {
        tmp_item = TAILQ_NEXT(item, entries);
        if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "Ever written " << item->dat->name << " " <<
                   OPS_instance::getOPSInstance()->OPS_dat_ever_written[item->dat->index] << '\n';
      }
    }

    ops_checkpoint_complete();

    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      if (ops_is_root()) OPS_instance::getOPSInstance()->ostream() << "\nCheckpoint created "<< OPS_chk_red_offset_g <<" bytes reduction data\n";
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

void ops_checkpointing_exit(OPS_instance *instance) {
  if (OPS_instance::getOPSInstance()->backup_state != OPS_NONE) {
    ops_statistics_exit();

#ifdef OPS_CHK_THREAD
    if (OPS_instance::getOPSInstance()->ops_thread_offload && ops_ramdisk_initialised) {
      ops_ramdisk_ctrl_exit = 1;
      pthread_join(thread, NULL);
      ops_free(ops_ramdisk_buffer);
      ops_free(ops_ramdisk_item_queue);
    }
#endif

    if (OPS_instance::getOPSInstance()->ops_checkpoint_inmemory) {
      for (int i = ops_ramdisk_item_queue_tail; i < ops_ramdisk_item_queue_head;
           i++)
        ops_free(ops_ramdisk_item_queue[i].data);
      ops_free(ops_ramdisk_item_queue);
    }

    if (OPS_instance::getOPSInstance()->backup_state == OPS_BACKUP_IN_PROCESS) {
      check_hdf5_error(H5Fclose(file));
      if (ops_duplicate_backup)
        check_hdf5_error(H5Fclose(file_dup));
    }

    remove(filename.c_str());
    if (ops_duplicate_backup)
      remove(filename_dup.c_str());

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
      ops_printf2(OPS_instance::getOPSInstance(),
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
        ops_printf2(OPS_instance::getOPSInstance(),"Time spend in copy to ramdisk %g (%g)\n", moments_time[0],
                   moments_time[1]);
      }
#endif
    }
    ops_call_counter = 0;
    ops_backup_point_g = -1;
    ops_checkpoint_interval = -1;
    ops_last_checkpoint = -1;
    ops_pre_backup_phase = false;
    OPS_instance::getOPSInstance()->backup_state = OPS_NONE;
    ops_loop_max = 0;
    ops_free(ops_loops_hashmap);
    ops_loops_hashmap = NULL;
    ops_free(OPS_instance::getOPSInstance()->OPS_dat_ever_written);
    OPS_instance::getOPSInstance()->OPS_dat_ever_written = NULL;
    ops_free(OPS_instance::getOPSInstance()->OPS_dat_status);
    OPS_instance::getOPSInstance()->OPS_dat_status = NULL;
    OPS_partial_buffer_size = 0;
    ops_free(OPS_partial_buffer);
    OPS_partial_buffer = NULL;
    ops_free(ops_inm_ctrl.reduction_state);
    ops_free(ops_inm_ctrl.OPS_checkpointing_payload);
    ops_free(ops_inm_ctrl.OPS_chk_red_storage);

    // filename =  "";
  }
}


#else

class OPS_instance_checkpointing {};

bool ops_checkpointing_before(ops_arg *args, int nargs, int *range,
                              int loop_id) {
    (void)args;(void)nargs;(void)range;(void)loop_id;
  return true;
}
bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range,
                                   const char *s) {
    (void)args;(void)nargs;(void)range;(void)s;
  return true;
}
void ops_checkpointing_exit(OPS_instance* instance) { (void)instance; }

void ops_checkpointing_reduction(ops_reduction red) {
  ops_execute_reduction(red);
}



bool ops_checkpointing_init(const char *filename, double interval,
                            int options) {
    (void)filename;(void)interval;(void)options;
  return false;
}
void ops_checkpointing_initphase_done() {}

void ops_checkpointing_manual_datlist(int ndats, ops_dat* datlist) { (void)ndats;(void)datlist; }
bool ops_checkpointing_fastfw(int nbytes, char* payload) { (void)nbytes;(void)payload;return false; }
bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist,
                                             int nbytes, char *payload) {
    (void)ndats;(void)datlist;(void)nbytes;(void)payload;
  return false;
}

bool ops_checkpointing_manual_datlist_fastfw_trigger(int ndats,
                                                     ops_dat *datlist,
                                                     int nbytes,
                                                     char *payload) {
    (void)ndats;(void)datlist;(void)nbytes;(void)payload;
  return false;
}

#endif
