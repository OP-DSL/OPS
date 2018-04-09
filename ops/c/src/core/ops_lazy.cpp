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
  * @brief OPS core library functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the core library functions utilized by all OPS
 * backends
  */
#include <stdlib.h>
#include <sys/time.h>
#include "ops_lib_core.h"
#include "ops_hdf5.h"
#include <ops_exceptions.h>

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


extern int ops_loop_over_blocks;
double ops_tiled_halo_exchange_time = 0.0;

/////////////////////////////////////////////////////////////////////////
// Data structures
/////////////////////////////////////////////////////////////////////////
//Tiling plan & storage
struct tiling_plan {
  int nloops;
  std::vector<unsigned long> loop_sequence;
  int ntiles;
  std::vector<std::vector<int> > tiled_ranges; // ranges for each loop
  std::vector<ops_dat> dats_to_exchange;
  std::vector<int> depths_to_exchange;
};

class OPS_instance_tiling {
public:
  OPS_instance_tiling() : TILE1D(-1), TILE2D(-1), TILE3D(-1), ops_dims_tiling_internal(1) {}
  std::vector<ops_kernel_descriptor *> ops_kernel_list;

  // Tiling
  std::vector<std::vector<int> >
      data_read_deps; // latest data dependencies for each dataset
  std::vector<std::vector<int> >
      data_write_deps; // latest data dependencies for each dataset

  std::vector<std::vector<int> >
      data_read_deps_edge; // latest data dependencies for each dataset around the edges

  std::vector<tiling_plan> tiling_plans;

  // tile sizes
  int TILE1D;
  int TILE2D;
  int TILE3D;

  // dimensionality of blocks used throughout
  int ops_dims_tiling_internal;

};
#define TILE4D -1
#define TILE5D -1

#define ops_kernel_list OPS_instance::getOPSInstance()->tiling_instance->ops_kernel_list
#define data_read_deps OPS_instance::getOPSInstance()->tiling_instance->data_read_deps
#define data_write_deps OPS_instance::getOPSInstance()->tiling_instance->data_write_deps
#define data_read_deps_edge OPS_instance::getOPSInstance()->tiling_instance->data_read_deps_edge
#define tiling_plans OPS_instance::getOPSInstance()->tiling_instance->tiling_plans
#define TILE1D OPS_instance::getOPSInstance()->tiling_instance->TILE1D
#define TILE2D OPS_instance::getOPSInstance()->tiling_instance->TILE2D
#define TILE3D OPS_instance::getOPSInstance()->tiling_instance->TILE3D
#define ops_dims_tiling_internal OPS_instance::getOPSInstance()->tiling_instance->ops_dims_tiling_internal
#define LOOPARG ops_kernel_list[loop]->args[arg]
#define LOOPRANGE ops_kernel_list[loop]->range

/////////////////////////////////////////////////////////////////////////
// Helper functions
/////////////////////////////////////////////////////////////////////////

//Computes intersection of two ranges
inline int intersection(int range1_beg, int range1_end, int range2_beg,
                 int range2_end, int *intersect_begin) {
  if (range1_beg >= range1_end || range2_beg >= range2_end) return 0;
  int i_min = MAX(range1_beg, range2_beg);
  int i_max = MIN(range1_end, range2_end);
  *intersect_begin = i_min;
  return i_max > i_min ? i_max - i_min : 0;
}

//Queries L3 cache size
size_t ops_internal_get_cache_size() {
  if (OPS_instance::getOPSInstance()->OPS_hybrid_gpu) return 0;
  FILE *p = 0;
  p = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
  unsigned int i = 0;
  if (p) {
    fscanf(p, "%d", &i);
    fclose(p);
  }
  return i;
}

void ops_execute();

/////////////////////////////////////////////////////////////////////////
// Enqueueing loops
// - if tiling enabled, add to the list
// - if tiling disabled, execute immediately on CPU
/////////////////////////////////////////////////////////////////////////

void ops_enqueue_kernel(ops_kernel_descriptor *desc) {
  if (OPS_instance::getOPSInstance()->ops_enable_tiling && OPS_instance::getOPSInstance()->tiling_instance == NULL)
    OPS_instance::getOPSInstance()->tiling_instance = new OPS_instance_tiling();

  if (OPS_instance::getOPSInstance()->ops_enable_tiling || ops_loop_over_blocks)
    ops_kernel_list.push_back(desc);
  else {
    //Prepare the local execution ranges
    int start[OPS_MAX_DIM], end[OPS_MAX_DIM], arg_idx[OPS_MAX_DIM];
    if (compute_ranges(desc->args, desc->nargs,desc->block, desc->range, start, end, arg_idx) < 0) return;
    for (int d = 0; d < desc->block->dims; d++){
      desc->range[2*d+0] = start[d];
      desc->range[2*d+1] = end[d];
    }
    //If not tiling, I have to do the halo exchanges here
    double t1,t2,c;
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_timers_core(&c,&t1);

    //Halo exchanges
    if (desc->device) ops_H_D_exchanges_device(desc->args,desc->nargs);
    else ops_H_D_exchanges_host(desc->args,desc->nargs);
    ops_halo_exchanges(desc->args,desc->nargs,desc->orig_range);
    if (!desc->device) ops_H_D_exchanges_host(desc->args,desc->nargs);

    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_timers_core(&c,&t2);
    //Run the kernel
    desc->function(desc->name, desc->block, desc->blockidx, desc->dim, desc->range, desc->nargs, desc->args);

    //Dirtybits
    if (desc->device) ops_set_dirtybit_device(desc->args,desc->nargs);
    else ops_set_dirtybit_host(desc->args,desc->nargs);
    for (int arg = 0; arg < desc->nargs; arg++) {
      if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc != OPS_READ)
        ops_set_halo_dirtybit3(&desc->args[arg], desc->orig_range);
    }
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      OPS_instance::getOPSInstance()->OPS_kernels[desc->index].mpi_time += t2-t1;

    for (int i = 0; i < desc->nargs; i++)
      if (desc->args[i].argtype == OPS_ARG_GBL && desc->args[i].acc == OPS_READ)
        free(desc->args[i].data);
    free(desc->args);
    free(desc);
  }
//ops_execute();
}

/////////////////////////////////////////////////////////////////////////
// Computing dependencies across MPI
/////////////////////////////////////////////////////////////////////////

void ops_compute_mpi_dependencies(int loop, int d, int *start, int *end, int *biggest_range) {
  //If loop range starts before my left boundary, my left neighbour's end index
  // is either my start index or the end index of the loop (whichever is smaller)
  int left_neighbour_end = LOOPRANGE[2*d] < biggest_range[2*d] ? 
  MIN(LOOPRANGE[2*d+1],start[d]) : -INT_MAX;
  //If loop range ends after my right boundary, my right neighbour's start index
  // is either my end index or the start index of the loop (whichever is greater)
  int right_neighbour_start = LOOPRANGE[2*d+1] >= biggest_range[2*d+1] ? 
  MAX(LOOPRANGE[2*d],end[d]) : INT_MAX;

  if (LOOPRANGE[2 * d + 0] != start[d]) {
    for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
      // For any dataset written (i.e. not read)
      if (LOOPARG.argtype == OPS_ARG_DAT &&
        LOOPARG.opt == 1 &&
        LOOPARG.acc != OPS_READ) {

        //Left neighbour's end index
        int intersect_begin = 0;
        //Take intersection of execution range with (my left boundary-1) and prior data dependency
        int intersect_len = intersection(biggest_range[2*d]-1,data_read_deps_edge[LOOPARG.dat->index][2 * d],
                                         LOOPRANGE[2 * d + 0], LOOPRANGE[2 * d + 1], &intersect_begin);
        if (intersect_len > 0)
          left_neighbour_end = MAX(left_neighbour_end,intersect_begin + intersect_len);
      }
    }
  }
  if (LOOPRANGE[2 * d + 1] != end[d]) {
  // Look at read dependencies of datasets being written
    for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
  // For any dataset written (i.e. not read)
      if (LOOPARG.argtype == OPS_ARG_DAT &&
        LOOPARG.opt == 1 &&
        LOOPARG.acc != OPS_READ) {

        int intersect_begin = 0;
  //Take intersection of execution range with (my right boundary+1) and prior data dependency
        int intersect_len = intersection(data_read_deps_edge[LOOPARG.dat->index][2 * d + 1], biggest_range[2*d+1]+1,
          LOOPRANGE[2 * d + 0], LOOPRANGE[2 * d + 1], &intersect_begin);
        if (intersect_len > 0)
          right_neighbour_start = MIN(right_neighbour_start,intersect_begin);
      }
    }
  }

  // Update read dependencies of neighbours
  for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          // For any dataset read (i.e. not write-only)
    if (LOOPARG.argtype == OPS_ARG_DAT &&
      LOOPARG.opt == 1 &&
      LOOPARG.acc != OPS_WRITE) {
      int d_m_min = 0; // Find biggest positive/negative direction stencil
                           // point for this dimension
      int d_p_max = 0;
      for (int p = 0; p < LOOPARG.stencil->points; p++) {
          d_m_min = MIN(d_m_min,
            LOOPARG.stencil->stencil[LOOPARG.stencil->dims * p + d]);
          d_p_max = MAX(d_p_max,
            LOOPARG.stencil->stencil[LOOPARG.stencil->dims * p + d]);
      }
          //If this is the first tile update left neighbour's read dependency
      if (left_neighbour_end+d_p_max > biggest_range[2*d]) {
        data_read_deps_edge[LOOPARG.dat->index][2 * d + 0] =
          MAX(data_read_deps_edge[LOOPARG.dat->index][2 * d + 0],
          left_neighbour_end + d_p_max);
      }
          //If this is the last tile update right neighbour's read dependency
      if (right_neighbour_start+d_m_min < biggest_range[2*d+1]) {
        data_read_deps_edge[LOOPARG.dat->index][2 * d + 1] =
          MIN(data_read_deps_edge[LOOPARG.dat->index][2 * d + 1],
          right_neighbour_start + d_m_min);
      }
    }
  }
}


/////////////////////////////////////////////////////////////////////////
// Creating a new tiling plan
/////////////////////////////////////////////////////////////////////////

int ops_construct_tile_plan() {
  // Create new tiling plan
  double t1, t2, c1, c2;
  ops_timers_core(&c1, &t1);

  //
  // Set up pointers
  //

  tiling_plans.resize(tiling_plans.size() + 1);
  std::vector<std::vector<int> > &tiled_ranges =
      tiling_plans[tiling_plans.size() - 1].tiled_ranges;
  std::vector<ops_dat> &dats_to_exchange = 
      tiling_plans[tiling_plans.size() - 1].dats_to_exchange;
  std::vector<int> &depths_to_exchange = 
      tiling_plans[tiling_plans.size() - 1].depths_to_exchange;

  tiling_plans[tiling_plans.size() - 1].nloops = ops_kernel_list.size();
  tiling_plans[tiling_plans.size() - 1].loop_sequence.resize(
      ops_kernel_list.size());
  for (unsigned int i = 0; i < ops_kernel_list.size(); i++)
    tiling_plans[tiling_plans.size() - 1].loop_sequence[i] =
        ops_kernel_list[i]->hash;

  //
  // Compute biggest range
  //
  int dims = ops_kernel_list[ops_kernel_list.size() - 1]->block->dims;
  int biggest_range[2 * OPS_MAX_DIM];
  for (int d = 0; d < dims; d++) {
    biggest_range[2*d+0] = INT_MAX;
    biggest_range[2*d+1] = -INT_MAX;
  }
  // TODO: mixed dim blocks, currently it's just the last loop's block
  ops_dims_tiling_internal = dims;
  for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
    int start[OPS_MAX_DIM], end[OPS_MAX_DIM], disp[OPS_MAX_DIM];
    ops_get_abs_owned_range(ops_kernel_list[i]->block, ops_kernel_list[i]->range, start, end, disp);
    // TODO: handling non-owned blocks
    for (int d = 0; d < dims; d++) {
      biggest_range[2 * d] =
          MIN(biggest_range[2 * d], start[d]);
      biggest_range[2 * d + 1] =
          MAX(biggest_range[2 * d + 1], end[d]);
    }
    for (int d = dims; d < OPS_MAX_DIM; d++) {
      biggest_range[2 * d] = 1;
      biggest_range[2 * d + 1] = 1;
    }
  }
  
  for (int d = 0; d < dims; d++) {
    if (biggest_range[2*d] > biggest_range[2*d+1])
      biggest_range[2*d] = biggest_range[2*d+1];
  }

  size_t full_owned_size = 1;
  for (int d = 0; d < dims; d++) {
    full_owned_size *= (biggest_range[2 * d + 1] - biggest_range[2 * d]);
		if (OPS_instance::getOPSInstance()->OPS_diags>5) printf("Proc %d dim %d biggest range %d-%d\n",ops_get_proc(), d, biggest_range[2 * d], biggest_range[2 * d+1]);
  }

  //
  // Get tile sizes
  //
  if (getenv("T1"))
    TILE1D = atoi(getenv("T1"));
  else
    TILE1D = -1;
  if (getenv("T2"))
    TILE2D = atoi(getenv("T2"));
  else
    TILE2D = -1;
  if (getenv("T3"))
    TILE3D = atoi(getenv("T3"));
  else
    TILE3D = -1;
  int tile_sizes[5] = {TILE1D, TILE2D, TILE3D, TILE4D, TILE5D};
  // Initialise tiling datasets
  tiled_ranges.resize(ops_kernel_list.size());

  //
  // If no tile sizes specified, compute it
  //
  if (OPS_instance::getOPSInstance()->ops_cache_size == 0)
    OPS_instance::getOPSInstance()->ops_cache_size = ops_internal_get_cache_size() / 1000;
  // If tile sizes undefined, make an educated guess
  if (tile_sizes[0] == -1 && tile_sizes[1] == -1 && tile_sizes[2] == -1 &&
      OPS_instance::getOPSInstance()->ops_cache_size != 0) {
    // Figure out which datasets are being accessed in these loops
    std::vector<int> datasets_touched(OPS_instance::getOPSInstance()->OPS_dat_index, 0);
    for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
      for (int arg = 0; arg < ops_kernel_list[i]->nargs; arg++)
        if (ops_kernel_list[i]->args[arg].argtype == OPS_ARG_DAT)
          datasets_touched[ops_kernel_list[i]->args[arg].dat->index] = 1;
    }
    size_t total_mem = 0;
    ops_dat_entry *item;
    TAILQ_FOREACH(item, &OPS_instance::getOPSInstance()->OPS_dat_list, entries) {
      if (datasets_touched[item->dat->index] == 1)
        total_mem += item->dat->mem;
    }

    double data_per_point = (double)total_mem / (double)full_owned_size;
    int points_per_tile = (double)OPS_instance::getOPSInstance()->ops_cache_size * 1000000.0 / data_per_point;
    if (dims == 2) {
      // aim for an X size twice as much as the Y size, and the Y size an
      // integer multiple of the #of threads
      int M = sqrt(points_per_tile /
                   (3 * omp_get_max_threads() * omp_get_max_threads()));
      tile_sizes[0] = 3 * M * omp_get_max_threads();
      tile_sizes[1] = M * omp_get_max_threads();
      // Sanity check
      if (tile_sizes[0] <= 0 || tile_sizes[1] <= 0)
        tile_sizes[0] = tile_sizes[1] = -1;
    } else if (dims == 3) {
      // determine X size so at least 10*#of max threads is left for Y*Z
      tile_sizes[0] = biggest_range[1] - biggest_range[0];
      while ((double)points_per_tile / (double)tile_sizes[0] <
             10.0 * omp_get_max_threads())
        tile_sizes[0] = tile_sizes[0] / 2;
      tile_sizes[2] = sqrt((double)points_per_tile / (double)tile_sizes[0]);
      tile_sizes[1] = points_per_tile / (tile_sizes[0] * tile_sizes[2]);
      // Sanity check
      if (tile_sizes[0] <= 0 || tile_sizes[1] <= 0 || tile_sizes[2] <= 0)
        tile_sizes[0] = tile_sizes[1] = tile_sizes[2] = -1;
    }
    if (OPS_instance::getOPSInstance()->OPS_diags > 3)
      ops_printf("Defaulting to the following tile size: %dx%dx%d\n",
                 tile_sizes[0], tile_sizes[1], tile_sizes[2]);
  }

  //
  // Compute max number of tiles in each dimension
  //
  int ntiles[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++)
    ntiles[d] = 1;
  if (tile_sizes[0] > 0)
    ntiles[0]=(biggest_range[2*0+1]-biggest_range[2*0]-1)/tile_sizes[0]+1;
  if (tile_sizes[1] > 0)
    ntiles[1]=(biggest_range[2*1+1]-biggest_range[2*1]-1)/tile_sizes[1]+1;
  if (tile_sizes[2] > 0)
    ntiles[2]=(biggest_range[2*2+1]-biggest_range[2*2]-1)/tile_sizes[2]+1;
#if OPS_MAX_DIM > 3
  if (tile_sizes[3] > 0)
    ntiles[3]=(biggest_range[2*3+1]-biggest_range[2*3]-1)/tile_sizes[3]+1;
#endif
#if OPS_MAX_DIM > 4
  if (tile_sizes[4] > 0)
    ntiles[4]=(biggest_range[2*4+1]-biggest_range[2*4]-1)/tile_sizes[4]+1;
#endif
  if (OPS_MAX_DIM > 5) {
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: Tiling not supported dims>5");
  }

  int tiles_prod[OPS_MAX_DIM + 1];
  tiles_prod[0] = 1;
  for (int d = 1; d < OPS_MAX_DIM + 1; d++)
    tiles_prod[d] = tiles_prod[d - 1] * ntiles[d - 1];

  // Compute grand total number of tiles
  int total_tiles = tiles_prod[OPS_MAX_DIM];
  tiling_plans[tiling_plans.size() - 1].ntiles = total_tiles;

  //
  // Initialise storage
  //

  // Allocate room to store the range of each tile for each loop
  for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
    tiled_ranges[i].resize(total_tiles * OPS_MAX_DIM * 2);
  }

  // Initialise dataset dependencies
  data_read_deps.resize(OPS_instance::getOPSInstance()->OPS_dat_index);
  data_write_deps.resize(OPS_instance::getOPSInstance()->OPS_dat_index);
  data_read_deps_edge.resize(OPS_instance::getOPSInstance()->OPS_dat_index);
  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_dat_index; i++) {
    data_read_deps[i].resize(total_tiles * OPS_MAX_DIM * 2);
    data_write_deps[i].resize(total_tiles * OPS_MAX_DIM * 2);
    data_read_deps_edge[i].resize(OPS_MAX_DIM * 2);
    for (int d = 0; d < total_tiles * OPS_MAX_DIM; d++) {
      data_read_deps[i][2 * d + 0] = INT_MAX;   // Anything will be less
      data_read_deps[i][2 * d + 1] = -INT_MAX;  // Anything will be more
      data_write_deps[i][2 * d + 0] = INT_MAX;  // Anything will be less
      data_write_deps[i][2 * d + 1] = -INT_MAX; // Anything will be more
    }
    for (int d = 0; d < OPS_MAX_DIM; d++) {
      data_read_deps_edge[i][2 * d + 0] = -INT_MAX;   // Anything will be more
      data_read_deps_edge[i][2 * d + 1] = INT_MAX;  // Anything will be less
    }
  }

  //
  // Main tiling dependency analysis loop
  //
  for (int loop = (int)ops_kernel_list.size() - 1; loop >= 0; loop--) {
    int start[OPS_MAX_DIM], end[OPS_MAX_DIM], disp[OPS_MAX_DIM];
    ops_get_abs_owned_range(ops_kernel_list[loop]->block, LOOPRANGE, start, end, disp);

    for (int d = 0; d < dims; d++) {

      ops_compute_mpi_dependencies(loop, d, start, end, biggest_range);

      for (int tile = 0; tile < total_tiles; tile++) {

        
        // If this tile is the first on this process in this dimension
        if ((tile / tiles_prod[d]) % ntiles[d] == 0) {
          //If this is the leftmost process for this loop, then start index is the
          // same as the original start index
          if (LOOPRANGE[2 * d + 0] == start[d])
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = start[d];
          else {
            //If this is not the leftmost process, look at read dependencies
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = INT_MAX;
            // Look at read dependencies of datasets being written
              for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
              // For any dataset written (i.e. not read)
                if (LOOPARG.argtype == OPS_ARG_DAT &&
                  LOOPARG.opt == 1 &&
                  LOOPARG.acc != OPS_READ) {
                // Start index is the smallest across all of the dependencies, but
                // no smaller than the loop range
                  int intersect_begin = 0;
                  //Take intersection of execution range with prior data dependency and my end range
                  int intersect_len = intersection(data_read_deps[LOOPARG.dat->index][tile * OPS_MAX_DIM * 2 + 2 * d + 0], 
                                                   biggest_range[2*d+1],
                                                   LOOPRANGE[2 * d + 0], LOOPRANGE[2 * d + 1], &intersect_begin);
                  if (intersect_len > 0)
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = 
                      MIN(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],intersect_begin);
              }
            }

            //If no prior dependencies, set to normal start index
            if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] == INT_MAX) 
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = 
                  MIN(LOOPRANGE[2*d+1],start[d]);
          }
        }
        else // Otherwise begin range is end of previous tile's
          tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] =
              tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile - tiles_prod[d]) +
                                 2 * d + 1];

        // End index, if last tile in the dimension and if
        // this is the rightmost process involved in the loop
        if ((tile / tiles_prod[d]) % ntiles[d] == ntiles[d] - 1 &&
             LOOPRANGE[2 * d + 1] == end[d]) {
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] =
                LOOPRANGE[2 * d + 1];
        }
        // Otherwise it depends on data dependencies
        else {
          //By default make end index = start index, then extend upwards from there
          tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0];

					//
					// Extend by checking dependencies
					//
          // Look at read dependencies of datasets being written
          for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
            // For any dataset written (i.e. not read)
            if (LOOPARG.argtype == OPS_ARG_DAT &&
                LOOPARG.opt == 1 &&
                LOOPARG.acc != OPS_READ) {
              // End index is the greatest across all of the dependencies, but
              // no greater than the loop range
              int intersect_begin = 0;
              //Take intersection of execution range with tile start index and prior data dependency
              int intersect_len = intersection(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
                                               data_read_deps[LOOPARG.dat->index][tile * OPS_MAX_DIM * 2 + 2 * d + 1],
                                               LOOPRANGE[2 * d + 0], LOOPRANGE[2 * d + 1], &intersect_begin);
              if (intersect_len > 0)
                tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = 
                  MAX(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1],intersect_begin + intersect_len);
            }
          }
        }
        }
        for (int tile = 0; tile < total_tiles; tile++) {

          //Keep tile begin indices consistent
          if ((tile / tiles_prod[d]) % ntiles[d] != 0) 
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] =
              tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile - tiles_prod[d]) + 2 * d + 1];
          if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] > tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1])
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0];

          //If this is not the last tile, we need to check write dependencies too
          if ((tile / tiles_prod[d]) % ntiles[d] != ntiles[d] - 1 &&
            //unless the next tile shrunk to 0, in which case this is the last tile
              tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile + tiles_prod[d]) + 2 * d + 1] -
              tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile + tiles_prod[d]) + 2 * d + 0] > 0) {
            // Look at write dependencies of datasets being accessed
            for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
              if (LOOPARG.argtype == OPS_ARG_DAT &&
                  LOOPARG.opt == 1 &&
                  data_write_deps[LOOPARG.dat->index]
                                 [tile * OPS_MAX_DIM * 2 + 2 * d + 1] != -INT_MAX ) {
                int d_m_min = 0;  // Find biggest positive/negative direction
                                 // stencil point for this dimension
                int d_p_max = 0;
                for (int p = 0;
                     p < LOOPARG.stencil->points; p++) {
                  d_m_min = MIN(d_m_min,
                      LOOPARG.stencil->stencil
                          [LOOPARG.stencil->dims * p + d]);
                  d_p_max = MAX(d_p_max,
                      LOOPARG.stencil->stencil
                          [LOOPARG.stencil->dims * p + d]);
                }
                // End index is the greatest across all of the dependencies, but
                // no greater than the loop range
                int intersect_begin = 0;
                //Take intersection of execution range with tile start index and write data dependency + stencil width
                int intersect_len = intersection(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
                                                 data_write_deps[LOOPARG.dat->index][tile * OPS_MAX_DIM * 2 + 2 * d + 1]-d_m_min,
                                                 LOOPRANGE[2 * d + 0], LOOPRANGE[2 * d + 1], &intersect_begin);
                if (intersect_len > 0) {
                  tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = 
                    MAX(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1],intersect_begin + intersect_len);
                  //If we overshot the next tile's end index - due to different skewing factors
                  // that means this tile is now the last one, and we don't need to worry about
                  // write dependencies beyond that point
                  if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] >
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile + tiles_prod[d]) + 2 * d + 1])
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = 
                      tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile + tiles_prod[d]) + 2 * d + 1];
                }
              }
            }
          }

          // If no prior dependencies, end index is leftmost range + (tile index
          // + 1) * tile size, or end index if not tiled in this dimension
          if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] ==
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0]) {
            if (tile_sizes[d] <= 0)
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = end[d];
            else
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = 
                MAX(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
											MIN(end[d],
                          biggest_range[2 * d + 0] +
                              ((tile / tiles_prod[d]) % ntiles[d] + 1) *
                                  tile_sizes[d]));
          }

        if (OPS_instance::getOPSInstance()->OPS_diags > 5 && tile_sizes[d] != -1)
          printf("Proc %d, %s tile %d dim %d: exec range is: %d-%d\n",
                 ops_get_proc(), ops_kernel_list[loop]->name, tile, d,
                 tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
                 tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1]);
      }



      for (int tile = 0; tile < total_tiles; tile++) {

        int intersect_begin = 0;
        int intersect_len = intersection(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
                                         tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1],
                                         LOOPRANGE[2 * d + 0], LOOPRANGE[2 * d + 1], &intersect_begin);

//TODO: check if I intersect in other dimensions as well - but this may cause issues over MPI boundaries?
// also, this might ake adjacent tiles have slightly different skewing - again a problem over MPI?
//currently not an issue due to edge loops only having stencils in the direction where they are edges


				//If invalid/no range, skip updating dependencies
        if (intersect_len <= 0)
          continue;
 
        // Update read dependencies based on current iteration range
        for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          // For any dataset read (i.e. not write-only)
          if (LOOPARG.argtype == OPS_ARG_DAT &&
              LOOPARG.opt == 1 &&
              LOOPARG.acc != OPS_WRITE) {

            // Find biggest positive/negative direction stencil
            // point for this dimension
            int d_m_min = 0;                  
            int d_p_max = 0;
            for (int p = 0; p < LOOPARG.stencil->points; p++) {
              d_m_min = MIN(d_m_min,
                  LOOPARG.stencil->stencil[LOOPARG.stencil->dims * p + d]);
              d_p_max = MAX(d_p_max,
                  LOOPARG.stencil->stencil[LOOPARG.stencil->dims * p + d]);
            }
          
            // Extend dependency range with stencil
            data_read_deps[LOOPARG.dat->index]
                [tile * OPS_MAX_DIM * 2 + 2 * d + 0] = MIN(
                    data_read_deps[LOOPARG.dat->index]
                                  [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] +
                        d_m_min);
            data_read_deps[LOOPARG.dat->index]
                [tile * OPS_MAX_DIM * 2 + 2 * d + 1] = MAX(
                    data_read_deps[LOOPARG.dat->index]
                                  [tile * OPS_MAX_DIM * 2 + 2 * d + 1],
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] +
                        d_p_max);

            if (OPS_instance::getOPSInstance()->OPS_diags > 5 && tile_sizes[d] != -1)
              printf("Dataset read %s dependency dim %d set to %d %d\n",
                     LOOPARG.dat->name, d,
                     data_read_deps[LOOPARG.dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                     data_read_deps[LOOPARG.dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 1]);
          }
        }

        // Update write dependencies based on current iteration range
        for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          // For any dataset read (i.e. not write-only)
          if (LOOPARG.argtype == OPS_ARG_DAT &&
              LOOPARG.opt == 1 &&
              LOOPARG.acc != OPS_READ) {
            // Extend dependency range with stencil
            data_write_deps[LOOPARG.dat->index]
                [tile * OPS_MAX_DIM * 2 + 2 * d + 0] = MIN(
                    data_write_deps[LOOPARG.dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0]);
            data_write_deps[LOOPARG.dat->index]
                [tile * OPS_MAX_DIM * 2 + 2 * d + 1] = MAX(
                    data_write_deps[LOOPARG.dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 1],
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1]);

            if (OPS_instance::getOPSInstance()->OPS_diags > 5 && tile_sizes[d] != -1)
              printf(
                  "Dataset write %s dependency dim %d set to %d %d\n",
                  LOOPARG.dat->name, d,
                  data_write_deps[LOOPARG.dat->index]
                                 [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                  data_write_deps[LOOPARG.dat->index]
                                 [tile * OPS_MAX_DIM * 2 + 2 * d + 1]);
          }
        }
      }
    }

    //Subtract base index displacements over MPI
    for (int d = 0; d < dims; d++) {
      for (int tile = 0; tile < total_tiles; tile++) {
        tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] -= disp[d];
        tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] -= disp[d];
      }
    }
  }

  //Figure out which datasets need halo exchange - based on whether written or read first
  std::vector<int> datasets_accessed(OPS_instance::getOPSInstance()->OPS_dat_index, -1);
  for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
    for (int arg = 0; arg < ops_kernel_list[i]->nargs; arg++)
      if (ops_kernel_list[i]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[i]->args[arg].opt == 1 && datasets_accessed[ops_kernel_list[i]->args[arg].dat->index] == -1) {
        datasets_accessed[ops_kernel_list[i]->args[arg].dat->index] = (ops_kernel_list[i]->args[arg].acc == OPS_WRITE ? 0 : 1);
        if (ops_kernel_list[i]->args[arg].acc != OPS_WRITE)
          dats_to_exchange.push_back(ops_kernel_list[i]->args[arg].dat);
         if (OPS_instance::getOPSInstance()->OPS_diags > 5)
              ops_printf("First access to dataset %s is %d (0-write, 1-read)\n",ops_kernel_list[i]->args[arg].dat->name, datasets_accessed[ops_kernel_list[i]->args[arg].dat->index]);
      }
  }

  //Register halo depths needed
  depths_to_exchange.resize(OPS_MAX_DIM*4*dats_to_exchange.size()); //left send, left recv, right send, right recv
  for (int i = 0; i < (int)dats_to_exchange.size(); i++) {
    for (int d = 0; d < dims; d++) {
      if (data_read_deps_edge[dats_to_exchange[i]->index][2*d] == -INT_MAX)
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 0] = 0;
      else
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 0] = MAX(0,data_read_deps_edge[dats_to_exchange[i]->index][2*d]-biggest_range[2*d]);

      //Left recv depth is the read dependency range of the first tile, extending beyond the left owned range
      //TODO: use owned ranges instead of biggest range, even though they are the same for non-edge processes
      if (data_read_deps[dats_to_exchange[i]->index][2*d] == INT_MAX)
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 1] = 0;
      else
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 1] = MAX(0,biggest_range[2*d]-data_read_deps[dats_to_exchange[i]->index][2*d]);

      if (data_read_deps_edge[dats_to_exchange[i]->index][2*d+1] == INT_MAX)
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 2] = 0;
      else
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 2] = MAX(0,biggest_range[2*d+1]-data_read_deps_edge[dats_to_exchange[i]->index][2*d+1]);

      //Right recv depth is the read dependency range of the last tile, extending beyond the right owned range
      //Since the last tiles may disappear at the first loop (due to different skewing slopes), I need to check all the tiles and
      //take the max
      
      int right_read_dep = -INT_MAX;
      for (int tile = 0; tile < ntiles[d]; tile++)
        right_read_dep = MAX(right_read_dep,data_read_deps[dats_to_exchange[i]->index][tile*tiles_prod[d]*OPS_MAX_DIM*2+2*d+1]);
      if (right_read_dep == -INT_MAX) {
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 3] = 0;
      } else {
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 3] = MAX(0,right_read_dep-biggest_range[2*d+1]);
      }

      if (OPS_instance::getOPSInstance()->OPS_diags > 5)
        printf("Proc %d Dataset %s, dim %d, left send: %d, left recv: %d, right send: %d, right recv: %d\n", ops_get_proc(),dats_to_exchange[i]->name, d, depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 0],depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 1],depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 2],depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 3]);
    }
  }

  ops_timers_core(&c2, &t2);
  if (OPS_instance::getOPSInstance()->OPS_diags > 2)
    printf("Created tiling plan for %d loops in %g seconds, with tile size: %dx%dx%d\n", int(ops_kernel_list.size()), t2 - t1, tile_sizes[0], tile_sizes[1], tile_sizes[2]);

  // return index to newly created tiling plan
  return tiling_plans.size() - 1;
}

////////////////////////////////////////////////////////////////////
// Execute tiling plan
////////////////////////////////////////////////////////////////////
void ops_execute() {
  if (!OPS_instance::getOPSInstance()->ops_enable_tiling) return;
  if (OPS_instance::getOPSInstance()->tiling_instance == NULL)
    OPS_instance::getOPSInstance()->tiling_instance = new OPS_instance_tiling();
  if (ops_kernel_list.size() == 0)
    return;

  // Try to find an existing tiling plan for this sequence of loops
  int match = -1;
  for (unsigned int i = 0; i < tiling_plans.size(); i++) {
    if (int(ops_kernel_list.size()) == tiling_plans[i].nloops) {
      int count = 0;
      for (unsigned int j = 0; j < ops_kernel_list.size(); j++) {
        if (ops_kernel_list[j]->hash == tiling_plans[i].loop_sequence[j])
          count++;
        else
          break;
      }
      if (count == int(ops_kernel_list.size())) {
        match = i;
        break;
      }
    }
  }

  // If not found, construct one
  if (match == -1)
    match = ops_construct_tile_plan();
  std::vector<std::vector<int> > &tiled_ranges =
      tiling_plans[match].tiled_ranges;
  int total_tiles = tiling_plans[match].ntiles;

  //Do halo exchanges
  double c,t1,t2;
  if (OPS_instance::getOPSInstance()->OPS_diags>1)
    ops_timers_core(&c,&t1);
  
  ops_halo_exchanges_datlist(&tiling_plans[match].dats_to_exchange[0],
                             tiling_plans[match].dats_to_exchange.size(),
                             &tiling_plans[match].depths_to_exchange[0]);

  if (OPS_instance::getOPSInstance()->OPS_diags>1) {
    ops_timers_core(&c,&t2);
    OPS_instance::getOPSInstance()->ops_tiled_halo_exchange_time += t2-t1;
  }

  if (OPS_instance::getOPSInstance()->OPS_diags>3)
    ops_printf("Executing tiling plan for %d loops\n", ops_kernel_list.size());

  //Execute tiles
  for (int tile = 0; tile < total_tiles; tile++) {
    for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {

      if (tiled_ranges[i][OPS_MAX_DIM * 2 * tile + 1] -
                  tiled_ranges[i][OPS_MAX_DIM * 2 * tile + 0] ==
              0 ||
          (ops_dims_tiling_internal > 1 &&
           tiled_ranges[i][OPS_MAX_DIM * 2 * tile + 3] -
                   tiled_ranges[i][OPS_MAX_DIM * 2 * tile + 2] ==
               0) ||
          (ops_dims_tiling_internal > 2 &&
           tiled_ranges[i][OPS_MAX_DIM * 2 * tile + 5] -
                   tiled_ranges[i][OPS_MAX_DIM * 2 * tile + 4] ==
               0))
        continue;

      memcpy(&ops_kernel_list[i]->range[0],
             &tiled_ranges[i][OPS_MAX_DIM * 2 * tile],
             OPS_MAX_DIM * 2 * sizeof(int));
      if (OPS_instance::getOPSInstance()->OPS_diags > 4)
        printf("Proc %d Executing %s %d-%d %d-%d %d-%d\n", ops_get_proc(), ops_kernel_list[i]->name,
               ops_kernel_list[i]->range[0], ops_kernel_list[i]->range[1],
               ops_kernel_list[i]->range[2], ops_kernel_list[i]->range[3],
               ops_kernel_list[i]->range[4], ops_kernel_list[i]->range[5]);
      ops_kernel_list[i]->function(ops_kernel_list[i]->name, ops_kernel_list[i]->block, ops_kernel_list[i]->blockidx, ops_kernel_list[i]->dim, ops_kernel_list[i]->range, ops_kernel_list[i]->nargs, ops_kernel_list[i]->args);
    }
  }

  //Set dirtybits
  for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
    for (int arg = 0; arg < ops_kernel_list[i]->nargs; arg++) {
      if (ops_kernel_list[i]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[i]->args[arg].acc != OPS_READ)
        ops_set_halo_dirtybit3(&ops_kernel_list[i]->args[arg], ops_kernel_list[i]->orig_range);
    }
    if (ops_kernel_list[i]->device) ops_set_dirtybit_device(ops_kernel_list[i]->args,ops_kernel_list[i]->nargs);
    else ops_set_dirtybit_host(ops_kernel_list[i]->args,ops_kernel_list[i]->nargs);
  }

  for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
    for (int j = 0; j < ops_kernel_list[i]->nargs; j++)
      if (ops_kernel_list[i]->args[j].argtype == OPS_ARG_GBL && 
          ops_kernel_list[i]->args[j].acc == OPS_READ)
        free(ops_kernel_list[i]->args[j].data);
    free(ops_kernel_list[i]->args);
    free(ops_kernel_list[i]);
  }
  ops_kernel_list.clear();
}

extern "C" {

static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)calloc(len+16, sizeof(char));
  return strncpy(dest, src, len);
}

ops_kernel_descriptor * ops_create_kernel_descriptor(const char *name, ops_block block, int blockidx, int idx, int dim, int *range, int nargs, ops_arg *args, void (*fun)(const char*, ops_block, int, int, int*, int, ops_arg*)) {
   ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
   desc->name = copy_str(name);
   if (block == NULL) {
     desc->block = NULL;
     for (int i = 0; i < nargs; i++)
       if (args[i].opt == 1 && (args[i].argtype == OPS_ARG_DAT || args[i].argtype == OPS_ARG_PROLONG || args[i].argtype == OPS_ARG_RESTRICT || args[i].argtype == OPS_ARG_DAT2)) {
         desc->block = args[i].dat->block;
         break;
       }

   } else
     desc->block = block;
   desc->dim = dim;
   desc->device = 0;
   desc->index = idx;
   desc->blockidx = blockidx;
   desc->hash = 5381;
   desc->hash = ((desc->hash << 5) + desc->hash) + idx;
   for (int i = 0; i < dim*2; i++) {
     desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
     desc->range[i] = range[i];
     desc->orig_range[i] = range[i];
   }
   desc->nargs = nargs;
   desc->args = (ops_arg*)malloc(nargs*sizeof(ops_arg));
   char *tmp;
   for (int i = 0; i < nargs; i++) {
     desc->args[i] = args[i];
     if (args[i].opt && (args[i].argtype == OPS_ARG_DAT || args[i].argtype == OPS_ARG_PROLONG || args[i].argtype == OPS_ARG_RESTRICT || args[i].argtype == OPS_ARG_DAT2)) 
       desc->hash = ((desc->hash << 5) + desc->hash) + args[i].dat->index;
     else if (args[i].argtype == OPS_ARG_GBL and args[i].acc == OPS_READ) {
       tmp = (char*)malloc(args[i].dim * args[i].typesize);
       memcpy(tmp, args[i].data, args[i].dim * args[i].typesize);
       args[i].data = tmp;
     }
   }
   desc->function = fun;
   return desc;
}

void ops_enqueue_f(const char *name, ops_block block, int idx, int dim, int *range, int nargs, ops_arg *args, void (*fun)(const char*, ops_block, int, int, int*, int, ops_arg*)) {
  ops_kernel_descriptor * desc = ops_create_kernel_descriptor(name, block, 0, idx, dim, range, nargs, args, fun);
  ops_enqueue_kernel(desc);
  //fun(name, block, 0, dim, range, nargs, args);
}

void ops_enqueue_amr_f(const char *name, int blockidx, int idx, int dim, int *range, int nargs, ops_arg *args, void (*fun)(const char*, ops_block, int, int, int*, int, ops_arg*)) {
  ops_kernel_descriptor * desc = ops_create_kernel_descriptor(name, NULL, blockidx, idx, dim, range, nargs, args, fun);
  ops_enqueue_kernel(desc);
  //fun(name, NULL, blockidx, dim, range, nargs, args);
}
}
