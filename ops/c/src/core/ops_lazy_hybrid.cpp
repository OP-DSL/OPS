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

struct ops_hybrid_dirty {
  ops_dat dat;
  int dirty_from_d;
  int dirty_from_h;
  int dirty_to_d;
  int dirty_to_h;
};

vector<ops_hybrid_dirty> dirtyflags(0);

bool ops_hybrid_initalised = false;

void ops_init_hybrid() {
  dirtyflags.resize(OPS_dat_index);
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    int i = item->dat->index;
    dirtyflags[i].dat = item->dat;
    //Device is initially clean
    dirtyflags[i].dirty_from_d = 0;
    dirtyflags[i].dirty_to_d = 0;
    //Host is initially dirty
    dirtyflags[i].dirty_from_h = 0;
    dirtyflags[i].dirty_to_h = item->dat->size[item->dat->block->dims-1];
  }
  ops_hybrid_initialised = true;
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

  int len = intersection(from, to, 0, split, &start);
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
}

//Assume arg is dat, and OPS_WRITE
void ops_hybrid_report_dirty(ops_arg *arg, int from, int to, int split) {
  ops_dat dat = arg->dat;

  //If the GPU is executing anything
  if (to > split) {
    dirtyflags[dat->index].dirty_to_h = max(dirtyflags[dat->index].dirty_to_h, to);
    dirtyflags[dat->index].dirty_from_h = min(dirtyflags[dat->index].dirty_from_h, max(split,from));
  }

  //If the CPU is executing anything
  if (from < split) {
    dirtyflags[dat->index].dirty_from_d = min(dirtyflags[dat->index].dirty_from_d, from);
    dirtyflags[dat->index].dirty_to_d = max(dirtyflags[dat->index].dirty_to_d, min(split,to));
  }
}

void ops_hybrid_clean(ops_kernel_descriptor * desc) {
  int *range = desc->range;
  for (int i = 0; i < desc->nargs; i++) {
    if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc == OPS_READ) {
      int from_d, to_d, from_h, to_h;
      int split = 500;
      //TODO: calculate range->index into array
      from_d = to_d = from_h = to_h = 0;
      ops_hybrid_calc_clean(&(desc->args[arg]),
          range[2*(desc->args[arg].dat->block->dims-1)],
          range[2*(desc->args[arg].dat->block->dims-1)+1],
          split, from_d, to_d, from_h, to_h);
      ops_download_dat_range(desc->args[arg].dat, from_h, to_h);
      ops_upload_dat_range(desc->args[arg].dat, from_d, to_d);
    }
  }
}

void ops_hybrid_after(ops_kernel_descriptor * desc) {
  int *range = desc->range;
  for (int i = 0; i < desc->nargs; i++) {
    if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc != OPS_READ) {
      ops_hybrid_report_dirty(&(desc->args[arg]),
          range[2*(desc->args[arg].dat->block->dims-1)],
          range[2*(desc->args[arg].dat->block->dims-1)+1],
          split);
    }
  }
}

void ops_hybrid_execute(ops_kernel_descriptor *desc) {
  if (!ops_hybrid_initialised) ops_hybrid_initialise();
  ops_hybrid_clean(desc);
  int from = desc->range[2*(desc->dim-1)];
  int to = desc->range[2*(desc->dim-1)+1];
  //Launch GPU bit
  if (split<to) {
    desc->range[2*(desc->dim-1)] = max(desc->range[2*(desc->dim-1)],split);
    desc->device = 1;
    desc->function(desc);
  }
  //Launch CPU bit
  if (from < split) {
    desc->range[2*(desc->dim-1)] = from;
    desc->range[2*(desc->dim-1)+1] = min(desc->range[2*(desc->dim-1)+1],split);
    desc->device = 0;
    desc->function(desc);
  }
  desc->range[2*(desc->dim-1)]   = from;
  desc->range[2*(desc->dim-1)+1] = to;
  ops_hybrid_after(desc);
}

