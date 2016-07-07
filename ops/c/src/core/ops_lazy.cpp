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

#include <sys/time.h>
#include <stdlib.h>
#include "ops_lib_core.h"
#include <sys/time.h>

#include <vector>
using namespace std;

std::vector<ops_kernel_descriptor *> ops_kernel_list;

//Tiling
std::vector<std::vector<int> > tiled_ranges; //ranges for each loop
std::vector<std::vector<int> > data_read_deps; //latest data dependencies for each dataset
std::vector<std::vector<int> > data_write_deps; //latest data dependencies for each dataset


void ops_execute();

void ops_enqueue_kernel(ops_kernel_descriptor *desc) {
  ops_kernel_list.push_back(desc);
}

//#define TILE1D -1
//#define TILE2D 12
int TILE1D = -1;
int TILE2D = -1;
#define TILE3D -1
#define TILE4D -1
#define TILE5D -1


void ops_execute() {
  if (ops_kernel_list.size()==0) return;
  TILE1D = atoi(getenv ("T1"));
  TILE2D = atoi(getenv ("T2"));
  int tile_sizes[5] = {TILE1D, TILE2D, TILE3D, TILE4D, TILE5D};  
  //Initialise tiling datasets
  tiled_ranges.resize(ops_kernel_list.size());

  //Compute biggest range
  int biggest_range[2*OPS_MAX_DIM];
  memcpy(biggest_range,ops_kernel_list[0]->range,2*OPS_MAX_DIM*sizeof(int));
  //TODO: mixed dim blocks, currently it's jsut the last loop's block
  int dims = ops_kernel_list[ops_kernel_list.size()-1]->block->dims;
  for (int i = 1; i < ops_kernel_list.size(); i++) {
    for (int d = 0; d < dims; d++) {
      biggest_range[2*d] = MIN(biggest_range[2*d], ops_kernel_list[i]->range[2*d]);
      biggest_range[2*d+1] = MAX(biggest_range[2*d+1], ops_kernel_list[i]->range[2*d+1]);
    }
  }

  //Compute max number of tiles in each dimension
  int ntiles[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++) ntiles[d] = 1;
  if (TILE1D>0) ntiles[0] = (biggest_range[2*0+1]-biggest_range[2*0]-1)/TILE1D+1;
  if (TILE2D>0) ntiles[1] = (biggest_range[2*1+1]-biggest_range[2*1]-1)/TILE2D+1;
  if (TILE3D>0) ntiles[2] = (biggest_range[2*2+1]-biggest_range[2*2]-1)/TILE3D+1;
  if (TILE4D>0) ntiles[3] = (biggest_range[2*3+1]-biggest_range[2*3]-1)/TILE4D+1;
  if (TILE5D>0) ntiles[4] = (biggest_range[2*4+1]-biggest_range[2*4]-1)/TILE5D+1;
  if (OPS_MAX_DIM>5) {printf("Error, tiling currently not equipped to handle > 5 dims\n");exit(-1);}

  int tiles_prod[OPS_MAX_DIM+1];
  tiles_prod[0] = 1;
  for (int d = 1; d < OPS_MAX_DIM+1; d++) tiles_prod[d] = tiles_prod[d-1]*ntiles[d-1];

  //Compute grand total number of tiles
  int total_tiles = tiles_prod[OPS_MAX_DIM];

  //Allocate room to store the range of each tile for each loop
  for (int i = 0; i < ops_kernel_list.size(); i++) {
    tiled_ranges[i].resize(total_tiles*OPS_MAX_DIM*2);
  }

  //Initialise dataset dependencies
  data_read_deps.resize(OPS_dat_index);
  data_write_deps.resize(OPS_dat_index);
  for (int i = 0; i < OPS_dat_index; i++) {
    data_read_deps[i].resize(total_tiles*OPS_MAX_DIM*2);
    data_write_deps[i].resize(total_tiles*OPS_MAX_DIM*2);
    for (int d = 0; d < total_tiles*OPS_MAX_DIM; d++) {
      data_read_deps[i][2*d+0] = INT_MAX; //Anything will be less
      data_read_deps[i][2*d+1] = -INT_MAX; //Anythign will be more
      data_write_deps[i][2*d+0] = INT_MAX; //Anything will be less
      data_write_deps[i][2*d+1] = -INT_MAX; //Anythign will be more
    }
  }

  //Loop over ops_par_loops, backward
  for (int loop = ops_kernel_list.size()-1; loop >= 0; loop--) {
    for (int d = 0; d < dims; d++) {
      for (int tile = 0; tile < total_tiles; tile++) {
        
        //If this tile is the first in this dimension, then start index is the same as the original start index
        if ((tile/tiles_prod[d])%ntiles[d]==0)
          tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 0] = ops_kernel_list[loop]->range[2*d + 0];
        else //Otherwise begin range is end of previous tile's
          tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 0] = tiled_ranges[loop][OPS_MAX_DIM*2*(tile-tiles_prod[d]) + 2*d + 1];

        //End index, if last tile in the dimension
        if ((tile/tiles_prod[d])%ntiles[d]==ntiles[d]-1)
          tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] = ops_kernel_list[loop]->range[2*d + 1];
        //Otherwise it depends on data dependencies
        else {
          tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] = -INT_MAX;
          //Look at read dependencies of datasets being written
          for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
            //For any dataset written (i.e. not read)
            if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[loop]->args[arg].acc != OPS_READ) {
              //End index is the greatest across all of the dependencies, but no greater than the loop range
              tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] = 
                MIN(ops_kernel_list[loop]->range[2*d + 1],
                  MAX(tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1],
                    data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1]));
            }
          }
          //Look at write dependencies of datasets being read
          for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
            //For any dataset written (i.e. not read)
            if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[loop]->args[arg].acc != OPS_WRITE) {
              int d_m_min = 0; //Find biggest positive/negative direction stencil point for this dimension
              int d_p_max = 0;
              for (int p = 0; p < ops_kernel_list[loop]->args[arg].stencil->points; p++) {
                d_m_min = MIN(d_m_min,ops_kernel_list[loop]->args[arg].stencil->stencil[ops_kernel_list[loop]->args[arg].stencil->dims*p+d]);
                d_p_max = MAX(d_p_max,ops_kernel_list[loop]->args[arg].stencil->stencil[ops_kernel_list[loop]->args[arg].stencil->dims*p+d]);
              }
              //End index is the greatest across all of the dependencies, but no greater than the loop range
              tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] = 
                MIN(ops_kernel_list[loop]->range[2*d + 1],
                  MAX(tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1],
                    data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1] - d_m_min));
            }
          }
          //If no prior dependencies, end index is leftmost range + (tile index + 1) * tile size, or end index if not tiled in this dimension
          if (tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] == -INT_MAX) {
            tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] = 
              tile_sizes[d]<=0 ? ops_kernel_list[loop]->range[2*d + 1] :
                MIN(ops_kernel_list[loop]->range[2*d + 1], biggest_range[2*d+0] + ((tile/tiles_prod[d])%ntiles[d]+1)*tile_sizes[d]);
          }
          //But in an edge case, if begin range is larger than the computed end range, just set the two to the same value
          if (tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] < ops_kernel_list[loop]->range[2*d + 0])
              tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] = ops_kernel_list[loop]->range[2*d + 0];
        }

        if (OPS_diags>5)
          printf("%s tile %d dim %d: exec range is: %d-%d\n", ops_kernel_list[loop]->name, tile, d, tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 0], tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1]);

        if (tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] <= tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 0])
          continue;

        //Update read dependencies based on current iteration range
        for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          //For any dataset read (i.e. not write-only)
          if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[loop]->args[arg].acc != OPS_WRITE) {
            int d_m_min = 0; //Find biggest positive/negative direction stencil point for this dimension
            int d_p_max = 0;
            for (int p = 0; p < ops_kernel_list[loop]->args[arg].stencil->points; p++) {
              d_m_min = MIN(d_m_min,ops_kernel_list[loop]->args[arg].stencil->stencil[ops_kernel_list[loop]->args[arg].stencil->dims*p+d]);
              d_p_max = MAX(d_p_max,ops_kernel_list[loop]->args[arg].stencil->stencil[ops_kernel_list[loop]->args[arg].stencil->dims*p+d]);
            }
            //Extend dependency range with stencil
            data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+0] = 
              MIN(data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+0],
                tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 0] + d_m_min);
            data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1] = 
              MAX(data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1],
                tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1] + d_p_max);
            if (OPS_diags>5)
              printf("Dataset read %s dependency dim %d set to %d %d\n", ops_kernel_list[loop]->args[arg].dat->name, d, data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+0], data_read_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1]);
          }
        }
        //Update write dependencies based on current iteration range
        for (int arg = 0; arg < ops_kernel_list[loop]->nargs; arg++) {
          //For any dataset read (i.e. not write-only)
          if (ops_kernel_list[loop]->args[arg].argtype == OPS_ARG_DAT && ops_kernel_list[loop]->args[arg].acc != OPS_READ) {
            //Extend dependency range with stencil
            data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+0] = 
              MIN(data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+0],
                tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 0]);
            data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1] = 
              MAX(data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1],
                tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1]);
//            if (-2147483647 == tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1]) printf("%d %d\n",tiled_ranges[loop][OPS_MAX_DIM*2*tile + 2*d + 1], )
            if (OPS_diags>5)
              printf("Dataset write %s dependency dim %d set to %d %d\n", ops_kernel_list[loop]->args[arg].dat->name, d, data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+0], data_write_deps[ops_kernel_list[loop]->args[arg].dat->index][tile*OPS_MAX_DIM*2+2*d+1]);
          }
        }
      }
    }
  }

  for (int tile = 0; tile < total_tiles; tile++) {
    for (int i = 0; i < ops_kernel_list.size(); i++) {      
      memcpy(&ops_kernel_list[i]->range[0], &tiled_ranges[i][OPS_MAX_DIM*2*tile], OPS_MAX_DIM*2*sizeof(int));
      if (OPS_diags>5)
        printf("Executing %s %d-%d %d-%d\n",ops_kernel_list[i]->name, ops_kernel_list[i]->range[0],ops_kernel_list[i]->range[1],ops_kernel_list[i]->range[2],ops_kernel_list[i]->range[3]);
      ops_kernel_list[i]->function(ops_kernel_list[i]);
    }
  }
  for (int i = 0; i < ops_kernel_list.size(); i++) {
    free(ops_kernel_list[i]->args);
    free(ops_kernel_list[i]);
  }
  ops_kernel_list.clear();
}
