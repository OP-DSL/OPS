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

/** @brief OPS core library functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the core library functions utilized by all OPS
 * backends
  */
#include <stdlib.h>
#include <sys/time.h>
#include "ops_lib_core.h"
#include "ops_hdf5.h"

#if defined(_OPENMP)
#include <omp.h>
#else

inline int omp_get_max_threads() {
  if (getenv("OMP_NUM_THREADS"))
    return atoi(getenv("OMP_NUM_THREADS"));
  else
    return 1;
}
#endif

#include <vector>
using namespace std;

int ops_loop_over_blocks = 0;
int *ops_loop_over_blocks_predicate = NULL;
int ops_loop_over_blocks_condition = 0;
extern std::vector<ops_kernel_descriptor *> ops_kernel_list;
std::vector<int> replicated;
std::vector<char *> orig_ptrs;
std::vector<char *> new_ptrs;
std::vector<ops_dat> free_queue;

void replicate_dats() {
  replicated.resize(MAX((int)replicated.size(), OPS_dat_index));
  orig_ptrs.resize(MAX((int)replicated.size(), OPS_dat_index));
  fill(replicated.begin(), replicated.end(), 0);

  //Go through all kernels, and all their arguments, see if a dataset is modified
  // If so, we need a replica for each thread
  for (unsigned int k = 0; k < ops_kernel_list.size(); k++) {
    for (int i = 0; i < ops_kernel_list[k]->nargs; i++) {
      ops_arg *args = ops_kernel_list[k]->args;
      if (args[i].opt && (args[i].argtype == OPS_ARG_DAT || args[i].argtype == OPS_ARG_PROLONG || 
          args[i].argtype == OPS_ARG_RESTRICT || args[i].argtype == OPS_ARG_DAT2)) {
        if (args[i].acc != OPS_READ && args[i].dat->amr == 0)
          replicated[args[i].dat->index] = 1;
      }
    }
  }

  //Go through all datasets, create replicas if flagged
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    ops_dat dat = item->dat;
    if (replicated[dat->index] == 1) {
      printf("replicating %s\n",dat->name);
      char *tmp = (char*)malloc(dat->mem*omp_get_max_threads());
      #pragma omp parallel for
      for (int t = 0; t < omp_get_max_threads(); t++)
        memcpy(tmp+dat->mem*t,dat->data,dat->mem);
      orig_ptrs[dat->index] = dat->data;
      dat->data = tmp;
    }
  }

  //Go through kernels and ops_args, replace pointers
  for (unsigned int k = 0; k < ops_kernel_list.size(); k++) {
    for (int i = 0; i < ops_kernel_list[k]->nargs; i++) {
      ops_arg *args = ops_kernel_list[k]->args;
      if (args[i].opt && (args[i].argtype == OPS_ARG_DAT || args[i].argtype == OPS_ARG_PROLONG || 
          args[i].argtype == OPS_ARG_RESTRICT || args[i].argtype == OPS_ARG_DAT2)) {
        if (replicated[args[i].dat->index]==1)
          args[i].data = args[i].dat->data;
      }
    }
  }
}

void restore_dat_ptrs() {
 ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    ops_dat dat = item->dat;
    if (replicated[dat->index] == 1) {
      free(dat->data);
      //TODO: I should not need to copy back from the replicas... these dats should strictly be temporaries
      dat->data = orig_ptrs[dat->index];
    }
  }
  fill(replicated.begin(), replicated.end(), 0);
}

void ops_execute_amr() {
  replicate_dats();
  #pragma omp parallel for
  for (int i = 0; i < ops_loop_over_blocks; i++) {
    if (ops_loop_over_blocks_predicate != NULL &&
        ops_loop_over_blocks_predicate[i] != ops_loop_over_blocks_condition) continue;
    for (unsigned int k = 0; k < ops_kernel_list.size(); k++) {
      //WARNING: fortran block index
      ops_kernel_list[k]->function(ops_kernel_list[k]->name, ops_kernel_list[k]->block, i+1,
          ops_kernel_list[k]->dim, ops_kernel_list[k]->range, ops_kernel_list[k]->nargs, ops_kernel_list[k]->args);
    }
  }
  ops_kernel_list.clear();
  restore_dat_ptrs();
}

extern "C" {

  void ops_par_loop_blocks_all(int nblocks) {
    ops_loop_over_blocks = nblocks;
    ops_loop_over_blocks_predicate = NULL;
  }
  void ops_par_loop_blocks_int_1cond(int *arr, int nblocks, int pred) {
    ops_loop_over_blocks = nblocks;
    ops_loop_over_blocks_predicate = arr;
    ops_loop_over_blocks_condition = pred;
  }
  void ops_par_loop_blocks_end() {
    ops_execute_amr();
    ops_loop_over_blocks = 0;
    ops_loop_over_blocks_predicate = NULL;
    ops_loop_over_blocks_condition = 0;
    for (unsigned int i = 0; i < free_queue.size(); i++) ops_free_dat(free_queue[i]);
    free_queue.clear();
  }
  int ops_amr_lazy_offset(ops_dat dat);
  void ops_queue_free_dat(ops_dat dat);
}

int ops_amr_lazy_offset_idx() {
#if defined(_OPENMP)
  if (ops_loop_over_blocks) return omp_get_thread_num();
  else return 0;
#else
  return 0;
#endif
}

int ops_amr_lazy_offset(ops_dat dat) {
#if defined(_OPENMP)
  if (ops_loop_over_blocks && replicated[dat->index]==1) return omp_get_thread_num();
  else return 0;
#else
  return 0;
#endif
}
void ops_queue_free_dat(ops_dat dat) {
  free_queue.push_back(dat);
}

void ops_amr_reduction_size(int *count, int *stride, int size) {
  *stride = ((size-1)/64+1)*64;
  *count = MAX(omp_get_max_threads(),1);
}

template <typename T>
T ops_red_op(T a, T b, int op) {
  if (op == OPS_INC)
    return a+b;
  else if (op == OPS_MIN)
    return a<b?a:b;
  else if (op == OPS_MAX)
    return a>b?a:b;
  else return a;
}

void ops_amr_reduction_result(ops_reduction handle) {
  if (handle->multithreaded) {
    int stride, count;
    ops_amr_reduction_size(&count, &stride, handle->size);
    char *v0 = handle->data;
    for (int i = 1; i < count; i++) {
      char *val = handle->data + i * stride;
      if (!strcmp(handle->type,"double")) {
        int dim = handle->size/sizeof(double);
        for (int d = 0; d < dim; d++)
          ((double*)v0)[d] = ops_red_op(((double*)val)[d], ((double*)v0)[d],handle->acc);
      }
      if (!strcmp(handle->type,"float")) {
        int dim = handle->size/sizeof(float);
        for (int d = 0; d < dim; d++)
          ((float*)v0)[d] = ops_red_op(((float*)val)[d], ((float*)v0)[d],handle->acc);
      }
      if (!strcmp(handle->type,"int")) {
        int dim = handle->size/sizeof(int);
        for (int d = 0; d < dim; d++)
          ((int*)v0)[d] = ops_red_op(((int*)val)[d], ((int*)v0)[d],handle->acc);
      }
    }
  }
}
