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
#if defined(_OPENMP)
#include <omp.h>
#else

#include "ops_lib_core.h"
#include "ops_hdf5.h"

inline int omp_get_max_threads() {
  if (getenv("OMP_NUM_THREADS"))
    return atoi(getenv("OMP_NUM_THREADS"));
  else
    return 1;
}
#endif

#include <vector>
using namespace std;

extern int ops_enable_tiling;
extern int ops_cache_size;

double ops_tiled_halo_exchange_time = 0.0;

std::vector<ops_kernel_descriptor *> ops_kernel_list;

// Tiling
std::vector<std::vector<int> >
    data_read_deps; // latest data dependencies for each dataset
std::vector<std::vector<int> >
    data_write_deps; // latest data dependencies for each dataset

std::vector<std::vector<int> >
    data_read_deps_edge; // latest data dependencies for each dataset around the edges



struct tiling_plan {
  int nloops;
  std::vector<int> loop_sequence;
  int ntiles;
  std::vector<std::vector<int> > tiled_ranges; // ranges for each loop
  std::vector<ops_dat> dats_to_exchange;
  std::vector<int> depths_to_exchange;
};

std::vector<tiling_plan> tiling_plans(0);

void ops_execute();

void ops_enqueue_kernel(ops_kernel_descriptor *desc) {
  if (ops_enable_tiling)
    ops_kernel_list.push_back(desc);
  else {
    //Prepare the local execution ranges
    int start[OPS_MAX_DIM], end[OPS_MAX_DIM], disp[OPS_MAX_DIM];
    if (!ops_get_abs_owned_range(desc->block, desc->range, start, end, disp)) return;
    for (int d = 0; d < desc->block->dims; d++){
      desc->range[2*d+0] = start[d] - disp[d];
      desc->range[2*d+1] = end[d] - disp[d];
    }
    //If not tiling, I have to do the halo exchanges here
    double t1,t2,c;
    if (OPS_diags > 1)
      ops_timers_core(&c,&t1);
    ops_halo_exchanges(desc->args,desc->nargs,desc->orig_range);
    if (OPS_diags > 1)
      ops_timers_core(&c,&t2);
    //Run the kernel
    desc->function(desc);
    for (int arg = 0; arg < desc->nargs; arg++) {
      if (desc->args[arg].argtype == OPS_ARG_DAT && desc->args[arg].acc != OPS_READ)
        ops_set_halo_dirtybit3(&desc->args[arg], desc->orig_range);
    }
    if (OPS_diags > 1)
      OPS_kernels[desc->index].mpi_time += t2-t1;
  }
}

//#define TILE1D -1
//#define TILE2D 12
int TILE1D = -1;
int TILE2D = -1;
int TILE3D = -1;
#define TILE4D -1
#define TILE5D -1

int ops_dims_tiling_internal = 1;

size_t ops_internal_get_cache_size() {
  FILE *p = 0;
  p = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
  unsigned int i = 0;
  if (p) {
    fscanf(p, "%d", &i);
    fclose(p);
  }
  return i;
}

int ops_construct_tile_plan() {
  // Create new tiling plan
  if (OPS_diags > 2)
    ops_printf("Creating new tiling plan for %d loops\n",
               ops_kernel_list.size());
  double t1, t2, c1, c2;
  ops_timers_core(&c1, &t1);
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
  for (int i = 0; i < ops_kernel_list.size(); i++)
    tiling_plans[tiling_plans.size() - 1].loop_sequence[i] =
        ops_kernel_list[i]->index;

  // Get tile sizes
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

  // Compute biggest range
  int dims = ops_kernel_list[ops_kernel_list.size() - 1]->block->dims;
  int biggest_range[2 * OPS_MAX_DIM];
  for (int d = 0; d < dims; d++) {
    biggest_range[2*d+0] = INT_MAX;
    biggest_range[2*d+1] = -INT_MAX;
  }
  // TODO: mixed dim blocks, currently it's just the last loop's block
  ops_dims_tiling_internal = dims;
  for (int i = 0; i < ops_kernel_list.size(); i++) {
    int start[OPS_MAX_DIM], end[OPS_MAX_DIM], disp[OPS_MAX_DIM];
    bool owned = ops_get_abs_owned_range(ops_kernel_list[i]->block, ops_kernel_list[i]->range, start, end, disp);
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

  size_t full_owned_size = 1;
  for (int d = 0; d < dims; d++) {
    full_owned_size *= (biggest_range[2 * d + 1] - biggest_range[2 * d]);
  }

  if (ops_cache_size == 0)
    ops_cache_size = ops_internal_get_cache_size() / 1000;
  // If tile sizes undefined, make an educated guess
  if (tile_sizes[0] == -1 && tile_sizes[1] == -1 && tile_sizes[2] == -1 &&
      ops_cache_size != 0) {
    // Figure out which datasets are being accessed in these loops
    std::vector<int> datasets_touched(OPS_dat_index, 0);
    for (int i = 0; i < ops_kernel_list.size(); i++) {
      for (int arg = 0; arg < ops_kernel_list[i]->nargs; arg++)
        if (ops_kernel_list[i]->args[arg].argtype == OPS_ARG_DAT)
          datasets_touched[ops_kernel_list[i]->args[arg].dat->index] = 1;
    }
    size_t total_mem = 0;
    ops_dat_entry *item;
    TAILQ_FOREACH(item, &OPS_dat_list, entries) {
      if (datasets_touched[item->dat->index] == 1)
        total_mem += item->dat->mem;
    }

    double data_per_point = (double)total_mem / (double)full_owned_size;
    int points_per_tile = (double)ops_cache_size * 1000000.0 / data_per_point;
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
    if (OPS_diags > 2)
      ops_printf("Defaulting to the following tile size: %dx%dx%d\n",
                 tile_sizes[0], tile_sizes[1], tile_sizes[2]);
  }

  // Compute max number of tiles in each dimension
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
    printf("Error, tiling currently not equipped to handle > 5 dims\n");
    exit(-1);
  }

  int tiles_prod[OPS_MAX_DIM + 1];
  tiles_prod[0] = 1;
  for (int d = 1; d < OPS_MAX_DIM + 1; d++)
    tiles_prod[d] = tiles_prod[d - 1] * ntiles[d - 1];

  // Compute grand total number of tiles
  int total_tiles = tiles_prod[OPS_MAX_DIM];
  tiling_plans[tiling_plans.size() - 1].ntiles = total_tiles;

  // Allocate room to store the range of each tile for each loop
  for (int i = 0; i < ops_kernel_list.size(); i++) {
    tiled_ranges[i].resize(total_tiles * OPS_MAX_DIM * 2);
  }

  // Initialise dataset dependencies
  data_read_deps.resize(OPS_dat_index);
  data_write_deps.resize(OPS_dat_index);
  data_read_deps_edge.resize(OPS_dat_index);
  for (int i = 0; i < OPS_dat_index; i++) {
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

  // Loop over ops_par_loops, backward
  for (int loop = ops_kernel_list.size() - 1; loop >= 0; loop--) {
    int start[OPS_MAX_DIM], end[OPS_MAX_DIM], disp[OPS_MAX_DIM];
    bool owned = ops_get_abs_owned_range(ops_kernel_list[loop]->block, ops_kernel_list[loop]->range, start, end, disp);

    for (int d = 0; d < dims; d++) {
      for (int tile = 0; tile < total_tiles; tile++) {

        int left_neighbour_end = ops_kernel_list[loop]->range[2*d] < biggest_range[2*d] ? start[d] : -INT_MAX;
        int right_neighbour_start = ops_kernel_list[loop]->range[2*d+1] >= biggest_range[2*d+1] ? end[d] : INT_MAX;

        // If this tile is the first on this process in this dimension
        if ((tile / tiles_prod[d]) % ntiles[d] == 0) {
          //If this is the leftmost process for this loop, then start index is the
          // same as the original start index
          if (ops_kernel_list[loop]->range[2 * d + 0] == start[d])
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = start[d];
          else {
            //If this is not the leftmost process, look at read dependencies
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = INT_MAX;
            // Look at read dependencies of datasets being written
              for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
              // For any dataset written (i.e. not read)
                if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT &&
                  ops_kernel_list[loop]->args[arg].opt == 1 &&
                  ops_kernel_list[loop]->args[arg].acc != OPS_READ) {
                // Start index is the smallest across all of the dependencies, but
                // no smaller than the loop range
                  tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = MAX(
                    ops_kernel_list[loop]->range[2 * d + 0],
                    MIN(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
                      data_read_deps
                      [ops_kernel_list[loop]->args[arg].dat->index]
                      [tile * OPS_MAX_DIM * 2 + 2 * d + 0]));

                //Left neighbour's end index
                if (tile == 0) {
                  left_neighbour_end = MIN(
                    ops_kernel_list[loop]->range[2 * d + 1],
                    MAX(left_neighbour_end,
                        data_read_deps_edge
                            [ops_kernel_list[loop]->args[arg].dat->index][2 * d]));
                }
              }
            }

            //If no prior dependencies, set to normal start index
            if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] == INT_MAX) 
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] = start[d];
          }
        }
        else // Otherwise begin range is end of previous tile's
          tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] =
              tiled_ranges[loop][OPS_MAX_DIM * 2 * (tile - tiles_prod[d]) +
                                 2 * d + 1];

        // End index, if last tile in the dimension and if
        // this is the rightmost process involved in the loop
        if ((tile / tiles_prod[d]) % ntiles[d] == ntiles[d] - 1 &&
             ops_kernel_list[loop]->range[2 * d + 1] == end[d]) {
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] =
                ops_kernel_list[loop]->range[2 * d + 1];
        }
        // Otherwise it depends on data dependencies
        else {
          tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = -INT_MAX;
          // Look at read dependencies of datasets being written
          for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
            // For any dataset written (i.e. not read)
            if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT &&
                ops_kernel_list[loop]->args[arg].opt == 1 &&
                ops_kernel_list[loop]->args[arg].acc != OPS_READ) {
              // End index is the greatest across all of the dependencies, but
              // no greater than the loop range
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = MIN(
                  ops_kernel_list[loop]->range[2 * d + 1],
                  MAX(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1],
                      data_read_deps
                          [ops_kernel_list[loop]->args[arg].dat->index]
                          [tile * OPS_MAX_DIM * 2 + 2 * d + 1]));

              if (tile == total_tiles-1) {
                right_neighbour_start = MAX(
                  ops_kernel_list[loop]->range[2 * d + 0],
                  MIN(right_neighbour_start,
                      data_read_deps_edge
                          [ops_kernel_list[loop]->args[arg].dat->index][2 * d + 1]));
              }
            }
          }
          //If this is not the last tile, we need to check write dependencies too
          if ((tile / tiles_prod[d]) % ntiles[d] != ntiles[d] - 1) {
            // Look at write dependencies of datasets being accessed
            for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
              if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT &&
                  ops_kernel_list[loop]->args[arg].opt == 1 &&
                  data_write_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                 [tile * OPS_MAX_DIM * 2 + 2 * d + 1] !=
                      -INT_MAX ) {
                int d_m_min = 0;  // Find biggest positive/negative direction
                                 // stencil point for this dimension
                int d_p_max = 0;
                for (int p = 0;
                     p < ops_kernel_list[loop]->args[arg].stencil->points; p++) {
                  d_m_min = MIN(
                      d_m_min,
                      ops_kernel_list[loop]->args[arg].stencil->stencil
                          [ops_kernel_list[loop]->args[arg].stencil->dims * p +
                           d]);
                  d_p_max = MAX(
                      d_p_max,
                      ops_kernel_list[loop]->args[arg].stencil->stencil
                          [ops_kernel_list[loop]->args[arg].stencil->dims * p +
                           d]);
                }
                // End index is the greatest across all of the dependencies, but
                // no greater than the loop range
                tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] = MIN(
                    ops_kernel_list[loop]->range[2 * d + 1],
                    MAX(tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1],
                        data_write_deps
                                [ops_kernel_list[loop]->args[arg].dat->index]
                                [tile * OPS_MAX_DIM * 2 + 2 * d + 1] -
                            d_m_min));
              }
            }
          }
          // If no prior dependencies, end index is leftmost range + (tile index
          // + 1) * tile size, or end index if not tiled in this dimension
          if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] ==
              -INT_MAX) {
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] =
                tile_sizes[d] <= 0
                    ? end[d] //ops_kernel_list[loop]->range[2 * d + 1]
                    : MIN(end[d], //ops_kernel_list[loop]->range[2 * d + 1],
                          biggest_range[2 * d + 0] +
                              ((tile / tiles_prod[d]) % ntiles[d] + 1) *
                                  tile_sizes[d]);
          }
          // But in an edge case, if begin range is larger than the computed end
          // range, just set the two to the same value
          if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] <
              ops_kernel_list[loop]->range[2 * d + 0])
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] =
                ops_kernel_list[loop]->range[2 * d + 0];
        }

        if (OPS_diags > 5 && tile_sizes[d] != -1)
          printf("%s tile %d dim %d: exec range is: %d-%d\n",
                 ops_kernel_list[loop]->name, tile, d,
                 tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0],
                 tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1]);

        
        // Update read dependencies based on current iteration range
        for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          // For any dataset read (i.e. not write-only)
          if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT &&
              ops_kernel_list[loop]->args[arg].opt == 1 &&
              ops_kernel_list[loop]->args[arg].acc != OPS_WRITE) {
            int d_m_min = 0; // Find biggest positive/negative direction stencil
                             // point for this dimension
            int d_p_max = 0;
            for (int p = 0;
                 p < ops_kernel_list[loop]->args[arg].stencil->points; p++) {
              d_m_min = MIN(
                  d_m_min,
                  ops_kernel_list[loop]->args[arg].stencil->stencil
                      [ops_kernel_list[loop]->args[arg].stencil->dims * p + d]);
              d_p_max = MAX(
                  d_p_max,
                  ops_kernel_list[loop]->args[arg].stencil->stencil
                      [ops_kernel_list[loop]->args[arg].stencil->dims * p + d]);
            }
            //If there is actually an execution range
            if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] !=
              tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0]) {
              // Extend dependency range with stencil
              data_read_deps
                  [ops_kernel_list[loop]->args[arg].dat->index]
                  [tile * OPS_MAX_DIM * 2 + 2 * d + 0] = MIN(
                      data_read_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                    [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                      tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0] +
                          d_m_min);
              data_read_deps
                  [ops_kernel_list[loop]->args[arg].dat->index]
                  [tile * OPS_MAX_DIM * 2 + 2 * d + 1] = MAX(
                      data_read_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                    [tile * OPS_MAX_DIM * 2 + 2 * d + 1],
                      tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] +
                          d_p_max);
            }
            //If this is the first tile update left neighbour's read dependency
            if (tile==0 && left_neighbour_end >= biggest_range[2*d]) {
              data_read_deps_edge
                  [ops_kernel_list[loop]->args[arg].dat->index][2 * d + 0] = MAX(
                      data_read_deps_edge[ops_kernel_list[loop]->args[arg].dat->index][2 * d + 0],
                      left_neighbour_end + d_p_max);
            }
            //If this is the last tile update right neighbour's read dependency
            if (tile==total_tiles-1 && right_neighbour_start <= biggest_range[2*d+1]) {
              data_read_deps_edge
                  [ops_kernel_list[loop]->args[arg].dat->index][2 * d + 1] = MIN(
                      data_read_deps_edge[ops_kernel_list[loop]->args[arg].dat->index][2 * d + 1],
                      right_neighbour_start + d_m_min);
            }

            if (OPS_diags > 5 && tile_sizes[d] != -1)
              printf("Dataset read %s dependency dim %d set to %d %d\n",
                     ops_kernel_list[loop]->args[arg].dat->name, d,
                     data_read_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                     data_read_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 1]);
          }
        }

        if (tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1] <=
            tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0])
          continue;

        // Update write dependencies based on current iteration range
        for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          // For any dataset read (i.e. not write-only)
          if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT &&
              ops_kernel_list[loop]->args[arg].opt == 1 &&
              ops_kernel_list[loop]->args[arg].acc != OPS_READ) {
            // Extend dependency range with stencil
            data_write_deps
                [ops_kernel_list[loop]->args[arg].dat->index]
                [tile * OPS_MAX_DIM * 2 + 2 * d + 0] = MIN(
                    data_write_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 0]);
            data_write_deps
                [ops_kernel_list[loop]->args[arg].dat->index]
                [tile * OPS_MAX_DIM * 2 + 2 * d + 1] = MAX(
                    data_write_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                   [tile * OPS_MAX_DIM * 2 + 2 * d + 1],
                    tiled_ranges[loop][OPS_MAX_DIM * 2 * tile + 2 * d + 1]);

            if (OPS_diags > 5 && tile_sizes[d] != -1)
              printf(
                  "Dataset write %s dependency dim %d set to %d %d\n",
                  ops_kernel_list[loop]->args[arg].dat->name, d,
                  data_write_deps[ops_kernel_list[loop]->args[arg].dat->index]
                                 [tile * OPS_MAX_DIM * 2 + 2 * d + 0],
                  data_write_deps[ops_kernel_list[loop]->args[arg].dat->index]
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
  std::vector<int> datasets_accessed(OPS_dat_index, -1);
  for (int i = 0; i < ops_kernel_list.size(); i++) {
    for (int arg = 0; arg < ops_kernel_list[i]->nargs; arg++)
      //TODO: this is not safe - what if only a small part of it is written (e.g. halo)?
      if (ops_kernel_list[i]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[i]->args[arg].opt == 1 && datasets_accessed[ops_kernel_list[i]->args[arg].dat->index] == -1) {
        datasets_accessed[ops_kernel_list[i]->args[arg].dat->index] = (ops_kernel_list[i]->args[arg].acc == OPS_WRITE ? 0 : 1);
        if (ops_kernel_list[i]->args[arg].acc != OPS_WRITE)
          dats_to_exchange.push_back(ops_kernel_list[i]->args[arg].dat);
         if (OPS_diags > 5)
              ops_printf("First access to dataset %s is %d (0-write, 1-read)\n",ops_kernel_list[i]->args[arg].dat->name, datasets_accessed[ops_kernel_list[i]->args[arg].dat->index]);
      }
  }
  //Register halo depths needed
  depths_to_exchange.resize(OPS_MAX_DIM*4*dats_to_exchange.size()); //left send, left recv, right send, right recv
  for (int i = 0; i < dats_to_exchange.size(); i++) {
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
      if (data_read_deps[dats_to_exchange[i]->index][(total_tiles-1)*OPS_MAX_DIM*2+2*d+1] == -INT_MAX)
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 3] = 0;
      else
        depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 3] = MAX(0,data_read_deps[dats_to_exchange[i]->index][(total_tiles-1)*OPS_MAX_DIM*2+2*d+1]-biggest_range[2*d+1]);
      if (OPS_diags > 5)
        printf("Dataset %s, dim %d, left send: %d, left recv: %d, right send: %d, right recv: %d\n",dats_to_exchange[i]->name, d, depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 0],depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 1],depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 2],depths_to_exchange[i*OPS_MAX_DIM*4 + d*4 + 3]);
    }
  }

  ops_timers_core(&c2, &t2);
  if (OPS_diags > 2)
    printf("Created tiling plan %g seconds\n", t2 - t1);
  // return index to newly created tiling plan
  return tiling_plans.size() - 1;
}
void ops_execute() {
  if (ops_kernel_list.size() == 0)
    return;

  // Try to find an existing tiling plan for this sequence of loops
  int match = -1;
  for (int i = 0; i < tiling_plans.size(); i++) {
    if (ops_kernel_list.size() == tiling_plans[i].nloops) {
      int count = 0;
      for (int j = 0; j < ops_kernel_list.size(); j++) {
        if (ops_kernel_list[j]->index == tiling_plans[i].loop_sequence[j])
          count++;
        else
          break;
      }
      if (count == ops_kernel_list.size()) {
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
  if (OPS_diags>1)
    ops_timers_core(&c,&t1);
  
  ops_halo_exchanges_datlist(&tiling_plans[match].dats_to_exchange[0],
                             tiling_plans[match].dats_to_exchange.size(),
                             &tiling_plans[match].depths_to_exchange[0]);
  if (OPS_diags>1) {
    ops_timers_core(&c,&t2);
    ops_tiled_halo_exchange_time += t2-t1;
  }

  //Execute tiles
  for (int tile = 0; tile < total_tiles; tile++) {
    for (int i = 0; i < ops_kernel_list.size(); i++) {

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
      if (OPS_diags > 5)
        printf("Proc %d Executing %s %d-%d %d-%d %d-%d\n", ops_get_proc(), ops_kernel_list[i]->name,
               ops_kernel_list[i]->range[0], ops_kernel_list[i]->range[1],
               ops_kernel_list[i]->range[2], ops_kernel_list[i]->range[3],
               ops_kernel_list[i]->range[4], ops_kernel_list[i]->range[5]);
      ops_kernel_list[i]->function(ops_kernel_list[i]);
    }
  }

  //Set dirtybits
  for (int i = 0; i < ops_kernel_list.size(); i++) {
    for (int arg = 0; arg < ops_kernel_list[i]->nargs; arg++) {
      if (ops_kernel_list[i]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[i]->args[arg].acc != OPS_READ)
        ops_set_halo_dirtybit3(&ops_kernel_list[i]->args[arg], ops_kernel_list[i]->orig_range);
    }
  }

  for (int i = 0; i < ops_kernel_list.size(); i++) {
    // free(ops_kernel_list[i]->args);
    // free(ops_kernel_list[i]);
  }
  ops_kernel_list.clear();
}
