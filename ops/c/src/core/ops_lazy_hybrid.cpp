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

/** @brief Execution schemes for hybrid execution
  * @author Gihan Mudalige, Istvan Reguly
  */
#include <stdlib.h>
#include <sys/time.h>
#include "ops_lib_core.h"
#include "ops_hdf5.h"

#include <cuda_runtime.h>
#include <ops_cuda_rt_support.h>

#ifdef OPS_NVTX
#include <nvToolsExt.h>
#endif

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

cudaStream_t cpu_stream, gpu_stream;
cudaEvent_t ev1, ev2;

extern "C" {
long ops_get_base_index_dim(ops_dat dat, int dim);
int intersection(int range1_beg, int range1_end, int range2_beg,
                 int range2_end, int *intersect_begin);
int union_range(int range1_beg, int range1_end, int range2_beg,
                 int range2_end, int *union_begin);
void ops_download_dat_range(ops_dat, int, int, cudaStream_t);
void ops_upload_dat_range(ops_dat, int, int, cudaStream_t);
}

#include <vector>
using namespace std;

struct ops_hybrid_dirty {
  ops_dat dat;
  int dirty_from_d;
  int dirty_from_h;
  int dirty_to_d;
  int dirty_to_h;
  long index_offset;
};

struct ops_kernel_stats {
  int prev_split;
  double prev_balance;
  double gpu_time;
  double cpu_time;
};

vector<ops_hybrid_dirty> dirtyflags(0);
vector<ops_kernel_stats> stats(0);

bool ops_hybrid_initialised = false;

int ops_hybrid_gbl_split = 0;

void ops_hybrid_initialise() {
  dirtyflags.resize(OPS_dat_index);
  int max_size = 0;
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    int i = item->dat->index;
    dirtyflags[i].dat = item->dat;
    //Device is initially clean
    dirtyflags[i].dirty_from_d = 0;
    dirtyflags[i].dirty_to_d = 0;
    //Host is initially clean
    dirtyflags[i].dirty_from_h = 0;
    dirtyflags[i].dirty_to_h = 0;
    //how many bytes into the array is 0 along last dimension
    //   -- this is similar to base_offset, except it's 0 only along last dim
    dirtyflags[i].index_offset = ops_get_base_index_dim(item->dat, item->dat->block->dims-1); 
    max_size = max(max_size, item->dat->size[item->dat->block->dims-1]);
  }
  ops_hybrid_gbl_split = max_size/4;
  ops_hybrid_initialised = true;
  cutilSafeCall(cudaStreamCreateWithFlags(&cpu_stream, cudaStreamNonBlocking));
  cutilSafeCall(cudaStreamCreate(&gpu_stream));
  cutilSafeCall(cudaEventCreate(&ev1));
  cutilSafeCall(cudaEventCreate(&ev2));
}

void ops_hybrid_record_kernel_stats(int index, double cpu_time, double gpu_time) {
  stats[index].gpu_time = gpu_time;
  stats[index].cpu_time = cpu_time;
}

int ops_hybrid_get_split(ops_kernel_descriptor *desc, int from, int to) {
  if (desc->index >= stats.size()) {
    int old_size = stats.size();
    stats.resize(desc->index+10);
    for (int i = old_size; i < desc->index+10; i++) {
      stats[i].gpu_time = -1;
      stats[i].cpu_time = -1;
      stats[i].prev_balance = 0.25;
    }
  }
//  return 700;
  //If 1 line or not seen before, then first guess
  if (to-from == 1 || stats[desc->index].gpu_time == -1) {
    return ops_hybrid_gbl_split;
  }

  //Boundary loop - stick to a split that needs the least data transfer
  if (desc->range[1]-desc->range[0] == 1 || desc->range[3]-desc->range[2] == 1) {
    int num_arg = 0;
    int avg_split = 0;
    for (int arg = 0; arg < desc->nargs; arg++) {
      if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc != OPS_WRITE) {
        ops_dat dat = desc->args[arg].dat;
        if (dirtyflags[dat->index].dirty_to_d-dirtyflags[dat->index].dirty_from_d != 0 &&
            dirtyflags[dat->index].dirty_to_h-dirtyflags[dat->index].dirty_from_h != 0) {
          num_arg++;
          avg_split += (dirtyflags[dat->index].dirty_to_d + dirtyflags[dat->index].dirty_from_h)/2;
        }
      }
    }
    if (num_arg==0) 
    {
      stats[desc->index].prev_balance = (ops_hybrid_gbl_split-from)/(to-from);
      return ops_hybrid_gbl_split;
    }
    stats[desc->index].prev_balance = (avg_split/num_arg-from)/(to-from);
    return max(0,avg_split/num_arg);
  }

  if (stats[desc->index].gpu_time == 0 || stats[desc->index].cpu_time == 0) {
    printf("!!!!WUT!!! %s\n", desc->name);
    return ops_hybrid_gbl_split;
  }

   double new_balance = stats[desc->index].prev_balance * stats[desc->index].gpu_time / 
            ((1.0 - stats[desc->index].prev_balance) * stats[desc->index].cpu_time +
             stats[desc->index].prev_balance * stats[desc->index].gpu_time);
   double smoothed_balance = (2.0*stats[desc->index].prev_balance + new_balance)/3.0;
   stats[desc->index].prev_balance = smoothed_balance;
   stats[desc->index].prev_split = from + (to-from)*smoothed_balance;
   return stats[desc->index].prev_split;
}

//Assume arg is dat, and OPS_READ
void ops_hybrid_calc_clean(ops_arg *arg, int from, int to, int split, int &from_d, int &to_d, int &from_h, int &to_h) {
  ops_dat dat = arg->dat;
  int d = dat->block->dims-1;
  //max stencil offset in positive and negative directions
  int max_pos = 0;
  int max_neg = 0;
  for (int i = 0; i < arg->stencil->points; i++) {
    max_pos = max(max_pos, arg->stencil->stencil[i*arg->stencil->dims+d]);
    max_neg = min(max_neg, arg->stencil->stencil[i*arg->stencil->dims+d]);
  }

  //Intersection of full execution range with the GPU's execution range
  int start;
  int len = intersection(from, to, split, INT_MAX, &start);
  //Intersection of the dependency range of GPU's execution with GPU dirty region
  if (len > 0) {
    len = intersection(start+max_neg, start+len+max_pos,
                       dirtyflags[dat->index].dirty_from_d,dirtyflags[dat->index].dirty_to_d,
                       &start);
    if (len > 0) {
      from_d = start;
      to_d = start + len;
      //update dirty region
      if (to_d == dirtyflags[dat->index].dirty_to_d) //No gap
        dirtyflags[dat->index].dirty_to_d = from_d;
      else {
        //This can happen if we have a really narrow iteration range
        if (OPS_diags>4) ops_printf("Note: hybrid execution - cleaned region does not extend to end of dirty region\n");
      }
    }
  }

  len = intersection(from, to, 0, split, &start);
  //Intersection of the dependency range of CPU's execution with CPU dirty region
  if (len > 0) {
    len = intersection(start+max_neg, start+len+max_pos,
                       dirtyflags[dat->index].dirty_from_h,dirtyflags[dat->index].dirty_to_h,
                       &start);
    if (len > 0) {
      from_h = start;
      to_h = start + len;
      //update dirty region
      if (from_h == dirtyflags[dat->index].dirty_from_h) //No gap
        dirtyflags[dat->index].dirty_from_h = to_h;
      else {
        //This can happen if we have a really narrow iteration range
        if (OPS_diags>4) ops_printf("Note: hybrid execution - cleaned region does not extend to beginning of dirty region\n");
      }
    }
  }
  printf("%s up/downloading from_h %d to_h %d, from_d %d, to_d %d, new dirty cpu %d-%d gpu %d-%d\n", dat->name, from_h, to_h, from_d, to_d, dirtyflags[dat->index].dirty_from_h, dirtyflags[dat->index].dirty_to_h, dirtyflags[dat->index].dirty_from_d, dirtyflags[dat->index].dirty_to_d);
}

//Assume arg is dat, and OPS_WRITE
void ops_hybrid_report_dirty(ops_arg *arg, int from, int to, int split) {
  ops_dat dat = arg->dat;

  //Intersection of full execution range with the CPU's execution range
  int cpu_start;
  int cpu_len = intersection(from, to, 0, split, &cpu_start);
  if (cpu_len > 0) {
    //Intersection of CPU execution range with CPU dirty region
    int cpu_clean_start;
    int cpu_clean_len = intersection(cpu_start, cpu_start + cpu_len, 
        dirtyflags[dat->index].dirty_from_h, dirtyflags[dat->index].dirty_to_h, &cpu_clean_start);
    
    //If there is a remainder on the left, clean it
    if (cpu_clean_len > 0 && cpu_clean_start > dirtyflags[dat->index].dirty_from_h) 
      ops_download_dat_range(arg->dat, dirtyflags[dat->index].dirty_from_h, cpu_clean_start, cpu_stream);

    //Start of dirty region is at least end of CPU execution region
    dirtyflags[dat->index].dirty_from_h = max(dirtyflags[dat->index].dirty_from_h, cpu_start + cpu_len);
    if (dirtyflags[dat->index].dirty_from_h > dirtyflags[dat->index].dirty_to_h)
      dirtyflags[dat->index].dirty_to_h = dirtyflags[dat->index].dirty_from_h;

    //If this is an edge dat dat has size 1 in the last dim, and the CPU wrote it, it's clean
    if (dat->size[dat->block->dims-1] == 1) {
      dirtyflags[dat->index].dirty_from_h = 0;
      dirtyflags[dat->index].dirty_to_h = 0;
      //Should be dirty on the GPU (unless cleaned below)
      dirtyflags[dat->index].dirty_from_d = 0;
      dirtyflags[dat->index].dirty_to_d = 1;
    }
  }

  //Intersection of full execution range with the GPU's execution range
  int gpu_start;
  int gpu_len = intersection(from, to, split, INT_MAX, &gpu_start);
  if (gpu_len > 0) {
    //CPU dirty region is the union of the gpu execution range with the previous dirty region
    int dirty_cpu_start;
    int dirty_cpu_len = union_range(dirtyflags[dat->index].dirty_from_h, dirtyflags[dat->index].dirty_to_h,
                                    gpu_start, gpu_start+gpu_len, &dirty_cpu_start);
    if (dirty_cpu_len > 0 && dat->size[dat->block->dims-1] != 1) {
      dirtyflags[dat->index].dirty_from_h = dirty_cpu_start;
      dirtyflags[dat->index].dirty_to_h   = dirty_cpu_start + dirty_cpu_len;
    }

    //Intersection of GPU execution range with GPU dirty region
    int gpu_clean_start;
    int gpu_clean_len = intersection(gpu_start, gpu_start + gpu_len, 
        dirtyflags[dat->index].dirty_from_d, dirtyflags[dat->index].dirty_to_d, &gpu_clean_start);
    
    //If there is a remainder on the right, clean it
    if (gpu_clean_len > 0 && gpu_clean_start + gpu_clean_len < dirtyflags[dat->index].dirty_to_d) 
      ops_upload_dat_range(arg->dat, gpu_clean_start + gpu_clean_len, dirtyflags[dat->index].dirty_to_d, gpu_stream);

    //End of dirty region is iat most the start of GPU execution region
    dirtyflags[dat->index].dirty_to_d = min(dirtyflags[dat->index].dirty_to_d, gpu_start);
    if (dirtyflags[dat->index].dirty_from_d > dirtyflags[dat->index].dirty_to_d)
      dirtyflags[dat->index].dirty_from_d = dirtyflags[dat->index].dirty_to_d;

    //If this is an edge dat dat has size 1 in the last dim, and the CPU wrote it, it's clean
    if (dat->size[dat->block->dims-1] == 1) {
      dirtyflags[dat->index].dirty_from_d = 0;
      dirtyflags[dat->index].dirty_to_d = 0;
    }

  }
  if (cpu_len > 0) {
    //GPU dirty region is the union of the cpu execution range with the previous dirty region
    int dirty_gpu_start;
    int dirty_gpu_len = union_range(dirtyflags[dat->index].dirty_from_d, dirtyflags[dat->index].dirty_to_d,
                                    cpu_start, cpu_start+cpu_len, &dirty_gpu_start);
    if (dirty_gpu_len > 0 && dat->size[dat->block->dims-1] != 1) {
      dirtyflags[dat->index].dirty_from_d = dirty_gpu_start;
      dirtyflags[dat->index].dirty_to_d   = dirty_gpu_start + dirty_gpu_len;
    }
  }

  printf("%s marking dirty cpu %d-%d gpu %d-%d\n", dat->name, dirtyflags[dat->index].dirty_from_h, dirtyflags[dat->index].dirty_to_h, dirtyflags[dat->index].dirty_from_d, dirtyflags[dat->index].dirty_to_d);
}

void ops_hybrid_clean(ops_kernel_descriptor * desc, int split) {
  int *range = desc->range;
  for (int arg = 0; arg < desc->nargs; arg++) {
    if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc != OPS_WRITE) {
      int range_from = range[2*(desc->args[arg].dat->block->dims-1)];
      int range_to = range[2*(desc->args[arg].dat->block->dims-1)+1];
      int from_d, to_d, from_h, to_h;
      //calculate range->index into array
      int this_from  = range_from + dirtyflags[desc->args[arg].dat->index].index_offset;
      int this_to    = range_to   + dirtyflags[desc->args[arg].dat->index].index_offset;
      int this_split = split      + dirtyflags[desc->args[arg].dat->index].index_offset; 
      from_d = to_d = from_h = to_h = 0;
      ops_hybrid_calc_clean(&(desc->args[arg]),
          this_from, this_to, this_split,
          from_d, to_d, from_h, to_h);
      ops_download_dat_range(desc->args[arg].dat, from_h, to_h, cpu_stream);
      ops_upload_dat_range(desc->args[arg].dat, from_d, to_d, gpu_stream);
    }
  }
}

void ops_hybrid_after(ops_kernel_descriptor * desc, int split) {
  int *range = desc->range;
  for (int arg = 0; arg < desc->nargs; arg++) {
    if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc != OPS_READ) {
      int range_from = range[2*(desc->args[arg].dat->block->dims-1)];
      int range_to = range[2*(desc->args[arg].dat->block->dims-1)+1];
      int this_from  = range_from + dirtyflags[desc->args[arg].dat->index].index_offset;
      int this_to    = range_to   + dirtyflags[desc->args[arg].dat->index].index_offset;
      int this_split = split      + dirtyflags[desc->args[arg].dat->index].index_offset; 
      ops_hybrid_report_dirty(&(desc->args[arg]),
          this_from, this_to, this_split);
    }
  }
}

extern "C" void ops_download_dat_hybrid(ops_dat dat) {
  ops_download_dat_range(dat, dirtyflags[dat->index].dirty_from_h,
                               dirtyflags[dat->index].dirty_to_h, 0);
}

void ops_hybrid_execute(ops_kernel_descriptor *desc) {
  if (!ops_hybrid_initialised) ops_hybrid_initialise();
  int from = desc->range[2*(desc->dim-1)];
  int to = desc->range[2*(desc->dim-1)+1];
  int split = ops_hybrid_get_split(desc, from, to);
/*  if (desc->index == 2) {
    split = 305;
  }*/
//  printf("%s with split %d\n", desc->name, split);
  ops_hybrid_clean(desc, split);

  double c,t1=0,t2=0;

  cudaDeviceSynchronize();
  //Launch GPU bit
  if (split<to) {
    desc->range[2*(desc->dim-1)] = max(desc->range[2*(desc->dim-1)],split);
    desc->device = 1;
    cudaEventRecord(ev1,0);
    desc->function(desc);
    cudaEventRecord(ev2,0);
  }
  //Launch CPU bit
  if (from < split) {
    desc->range[2*(desc->dim-1)] = from;
    desc->range[2*(desc->dim-1)+1] = min(desc->range[2*(desc->dim-1)+1],split);
    desc->device = 0;
#ifdef OPS_NVTX
    nvtxRangePushA(desc->name);
#endif
    ops_timers_core(&c,&t1);
    desc->function(desc);
    ops_timers_core(&c,&t2);
#ifdef OPS_NVTX
    nvtxRangePop();
#endif
  }
  cudaEventSynchronize(ev2);
  float gpu_elapsed=0;
  if (split < to) cudaEventElapsedTime(&gpu_elapsed, ev1, ev2);
  printf("%s balance %g GPU time %g CPU time %g wait %g\n", stats[desc->index].prev_balance, desc->name, gpu_elapsed, (t2-t1)*1000.0, gpu_elapsed-(t2-t1)*1000.0);
  ops_hybrid_record_kernel_stats(desc->index, (t2-t1)*1000.0, gpu_elapsed);
  desc->range[2*(desc->dim-1)]   = from;
  desc->range[2*(desc->dim-1)+1] = to;
  ops_hybrid_after(desc, split);
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

extern "C" void ops_reduction_result_hybrid(ops_reduction handle) {
  if (ops_hybrid) {
    char *v0 = handle->data;
    for (int i = 1; i < 2; i++) {
      char *val = handle->data + i * handle->size;
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
