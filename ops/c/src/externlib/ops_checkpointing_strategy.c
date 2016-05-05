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

#ifdef CHECKPOINTING

#ifndef defaultTimeout
#define defaultTimeout 10.0
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ops_get_dat_full_range(ops_dat dat, int **full_range);

extern char *OPS_dat_ever_written;
extern int ops_call_counter;
extern int ops_best_backup_point;
extern int ops_best_backup_point_size;

int ops_strat_max_loop_counter = 0;
long *ops_strat_min_saved = NULL;
long *ops_strat_max_saved = NULL;
long *ops_strat_avg_saved = NULL;
long *ops_strat_saved_counter = NULL;
int *ops_strat_timescalled = NULL;
int *ops_strat_maxcalled = NULL;
int *ops_strat_lastcalled = NULL;
int **ops_strat_dat_status = NULL;
int *ops_strat_in_progress = NULL;

long ops_strat_calc_saved_amount_full(ops_dat dat) {

  int *full_range;
  ops_get_dat_full_range(dat, &full_range);
  int this_saved = dat->elem_size;
  for (int d = 0; d < dat->block->dims; d++)
    this_saved *= full_range[d];
  return this_saved;
}

long ops_strat_calc_saved_amount_partial(ops_dat dat, int *range) {
  int *full_range;
  ops_get_dat_full_range(dat, &full_range);
  int this_saved = 0;
  for (int d = 0; d < dat->block->dims; d++) {
    int itersize = full_range[d] - (range[2 * d + 1] - range[2 * d + 0]);
    for (int d2 = 0; d2 < dat->block->dims; d2++)
      if (d2 != d)
        itersize *= full_range[d2];
    this_saved += dat->elem_size * itersize;
  }
  int this_saved_full = dat->elem_size;
  for (int d = 0; d < dat->block->dims; d++)
    this_saved_full *= full_range[d];

  return MIN(this_saved, this_saved_full);
}

// TODO: in a real multiblock setting, the same loop (with the same id) will be
// called several times
// on different datasets and blocks. We need to differentiate between these
// calls to determine the stating point

// TODO: no timeout considerations here

void ops_strat_gather_statistics(ops_arg *args, int nargs, int loop_id,
                                 int *range) {
  if (ops_best_backup_point != -1)
    return;

  if (loop_id >= ops_strat_max_loop_counter) {
    int ops_strat_max_loop_counter_old = ops_strat_max_loop_counter;
    ops_strat_max_loop_counter = MAX(50,loop_id*2);
    ops_strat_min_saved = (long*)ops_realloc(ops_strat_min_saved, ops_strat_max_loop_counter*sizeof(long));
    ops_strat_max_saved = (long*)ops_realloc(ops_strat_max_saved, ops_strat_max_loop_counter*sizeof(long));
    ops_strat_avg_saved = (long*)ops_realloc(ops_strat_avg_saved, ops_strat_max_loop_counter*sizeof(long));
    ops_strat_saved_counter = (long*)ops_realloc(ops_strat_saved_counter, ops_strat_max_loop_counter*sizeof(long));

    ops_strat_timescalled = (int*)ops_realloc(ops_strat_timescalled, ops_strat_max_loop_counter*sizeof(int));
    ops_strat_maxcalled = (int*)ops_realloc(ops_strat_maxcalled, ops_strat_max_loop_counter*sizeof(int));
    ops_strat_lastcalled = (int*)ops_realloc(ops_strat_lastcalled, ops_strat_max_loop_counter*sizeof(int));
    ops_strat_in_progress = (int*)ops_realloc(ops_strat_in_progress, ops_strat_max_loop_counter*sizeof(int));
    ops_strat_dat_status = (int**)ops_realloc(ops_strat_dat_status, ops_strat_max_loop_counter*sizeof(int*));

    for (int i = ops_strat_max_loop_counter_old; i < ops_strat_max_loop_counter; i++) {
      ops_strat_min_saved[i] = LONG_MAX;
      ops_strat_max_saved[i] = 0;
      ops_strat_avg_saved[i] = 0;
      ops_strat_saved_counter[i] = 0;
      ops_strat_timescalled[i] = 0;
      ops_strat_maxcalled[i] = 0;
      ops_strat_lastcalled[i] = 0;
      ops_strat_dat_status[i] = (int *)ops_malloc(OPS_dat_index * sizeof(int));
      ops_strat_in_progress[i] = 0;
    }
  }

  if (!ops_strat_in_progress[loop_id]) {
    for (int i = 0; i < OPS_dat_index; i++)
      ops_strat_dat_status[loop_id][i] = OPS_UNDECIDED;
    ops_strat_saved_counter[loop_id] = 0;
    ops_strat_in_progress[loop_id] = MAX(1, ops_call_counter);
  }

  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OPS_ARG_DAT &&
        OPS_dat_ever_written[args[i].dat->index] && args[i].acc != OPS_WRITE &&
        args[i].opt == 1) {

      // if dataset is not written, it will have to be saved
      long tosave = ops_strat_calc_saved_amount_full(args[i].dat);
      for (int loop = 0; loop < ops_strat_max_loop_counter; ++loop) {
        if (ops_strat_in_progress[loop] &&
            ops_strat_dat_status[loop][args[i].dat->index] == OPS_UNDECIDED) {
          ops_strat_dat_status[loop][args[i].dat->index] = OPS_SAVED;
          ops_strat_saved_counter[loop] += tosave;
        }
      }
    } else if (args[i].argtype == OPS_ARG_DAT &&
               OPS_dat_ever_written[args[i].dat->index] &&
               args[i].acc == OPS_WRITE && args[i].opt == 1) {

      // if dataset is written, only a part of it (the edges that are not) has
      // to be saved
      long tosave = ops_strat_calc_saved_amount_partial(args[i].dat, range);
      for (int loop = 0; loop < ops_strat_max_loop_counter; ++loop) {
        if (ops_strat_in_progress[loop] &&
            ops_strat_dat_status[loop][args[i].dat->index] == OPS_UNDECIDED) {
          ops_strat_dat_status[loop][args[i].dat->index] = OPS_SAVED;
          ops_strat_saved_counter[loop] += tosave;
        }
      }
    }
  }

  for (int loop = 0; loop < ops_strat_max_loop_counter; ++loop) {
    if (ops_strat_in_progress[loop] == 0)
      continue;

    int done = 1;
    for (int i = 0; i < OPS_dat_index; i++) {
      if (OPS_dat_ever_written[i] &&
          ops_strat_dat_status[loop][i] == OPS_UNDECIDED) {
        done = 0;
        break;
      }
    }

    if (!done &&
        (ops_call_counter - ops_strat_in_progress[loop]) >
            300) { // TODO: this is pretty arbitrary
      ops_dat_entry *item, *tmp_item;
      for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
        tmp_item = TAILQ_NEXT(item, entries);
        ops_dat dat = item->dat;
        if (OPS_dat_ever_written[dat->index] &&
            ops_strat_dat_status[loop][dat->index] == OPS_UNDECIDED) {
          ops_strat_saved_counter[loop] +=
              ops_strat_calc_saved_amount_full(dat);
        }
      }
      done = 1;
    }

    if (done) {
      ops_strat_min_saved[loop] =
          MIN(ops_strat_min_saved[loop], ops_strat_saved_counter[loop]);
      ops_strat_max_saved[loop] =
          MAX(ops_strat_max_saved[loop], ops_strat_saved_counter[loop]);
      ops_strat_timescalled[loop]++;
      ops_strat_avg_saved[loop] =
          ((ops_strat_timescalled[loop] - 1) * ops_strat_avg_saved[loop] +
           ops_strat_saved_counter[loop]) /
          ops_strat_timescalled[loop];
      ops_strat_maxcalled[loop] =
          MAX(ops_strat_maxcalled[loop],
              ops_call_counter - ops_strat_in_progress[loop]);
      ops_strat_lastcalled[loop] = ops_call_counter;
      ops_strat_in_progress[loop] = 0;
    }
  }
}

typedef struct {
  long key;
  int value;
} ops_keyvalue;

int comp2(const void *a, const void *b) {
  if (((ops_keyvalue *)a)->key < ((ops_keyvalue *)b)->key)
    return -1;
  if (((ops_keyvalue *)a)->key == ((ops_keyvalue *)b)->key)
    return 0;
  if (((ops_keyvalue *)a)->key > ((ops_keyvalue *)b)->key)
    return 1;
  return 0;
}

bool ops_strat_should_backup(ops_arg *args, int nargs, int loop_id,
                             int *range) {
  if (ops_best_backup_point == -1) {
    ops_keyvalue *kv = (ops_keyvalue*)ops_malloc(ops_strat_max_loop_counter*sizeof(ops_keyvalue));
    for (int i = 0; i < ops_strat_max_loop_counter; i++) {
      if (ops_strat_timescalled[i] > 2) {
        kv[i].key = ops_strat_avg_saved[i];
        kv[i].value = i;
      } else {
        kv[i].key = LONG_MAX;
        kv[i].value = -1;
      }
    }
    qsort(kv, ops_strat_max_loop_counter, sizeof(ops_keyvalue), comp2);
    int kern = 0;
    while (true) {
      int idx = kv[kern].value;
      if (MAX(ops_strat_maxcalled[idx],
              ops_call_counter - ops_strat_lastcalled[idx]) <
          ops_call_counter / 10) {
        if (OPS_diags > 2)
          ops_printf(
              "Using kernel %d (%s) as backup point, will save ~%d kBytes\n",
              idx, OPS_kernels[idx].name, ops_strat_avg_saved[idx] / 1024);
        // ops_best_backup_point_size = ops_strat_avg_saved[idx];
        // for (int m = kern; m < OPS_kern_max; m++) {
        //   idx = kv[m].value;
        //   if (OPS_diags>2) ops_printf("Other candidates were %d (%s) as
        //   backup point, will save ~%d kBytes\n", idx, OPS_kernels[idx].name,
        //   ops_strat_avg_saved[idx]/1024);
        // }
        kern = idx;
        break;
      } else {
        if (OPS_diags > 2)
          ops_printf("Discarding candidate %d (%s) as backup point %d kBytes\n",
                     idx, OPS_kernels[idx].name,
                     ops_strat_avg_saved[idx] / 1024);
      }
      kern++;
      if (kern == OPS_kern_max) {
        ops_printf("Error: No suitable backup point found!\n");
        exit(-1);
      }
    }
    ops_best_backup_point = kern;
  }
  return (loop_id == ops_best_backup_point);
}

void ops_statistics_exit() {
  for (int i = 0; i < ops_strat_max_loop_counter; i++)
    free(ops_strat_dat_status[i]);
  free(ops_strat_dat_status);
  free(ops_strat_min_saved);
  free(ops_strat_max_saved);
  free(ops_strat_timescalled);
  free(ops_strat_avg_saved);
  free(ops_strat_maxcalled);
  free(ops_strat_in_progress);
  free(ops_strat_lastcalled);
  free(ops_strat_saved_counter);
  ops_strat_max_loop_counter = 0;
  ops_best_backup_point = -1;
}

#ifdef __cplusplus
}
#endif

#else

#ifdef __cplusplus
extern "C" {
#endif

void ops_strat_gather_statistics(ops_arg *args, int nargs, int loop_id,
                                 int *range) {}
bool ops_strat_should_backup(ops_arg *args, int nargs, int loop_id,
                             int *range) {
  return false;
}
void ops_statistics_exit() {}
#ifdef __cplusplus
}
#endif

#endif
