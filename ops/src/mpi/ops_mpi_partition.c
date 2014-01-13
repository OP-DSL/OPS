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

/** @brief ops mpi partitioning
  * @author Gihan Mudalige, adopted from Parallel OCCF routines by Mike Giles
  * @details Implements the single block structured mesh partitioning
  * for distributed memort (MPI) parallelization
  */

#include <mpi.h>
#include <ops_lib_cpp.h>
#include <ops_mpi_core.h>

MPI_Comm OPS_MPI_WORLD; // comm world for a single block
                        // -- need to have a communicator for each block in multi-block
int ops_comm_size;
int ops_my_rank;

sub_block_list *OPS_sub_block_list;// pointer to list holding sub-block
                                   // geometries

void ops_decomp(ops_block block, int g_ndim, int* g_sizes)
{
  //g_dim  - global number of dimensions .. will be the same on each local mpi process
  //g_sizes - global dimension sizes, i.e. size in each dimension of the global mesh

/** ---- create cartesian processor grid ---- **/

  MPI_Comm OPS_CART_COMM; // cartesian comm world
                          // -- agian need to have and store
                          // a comm world for each block in multi-block
  int ndim = g_ndim;
  int *pdims = (int *) xmalloc(ndim*sizeof(int));
  int *periodic = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++) {
    pdims[n] = 0;
    periodic[n] = 0; //false .. for now
  }

  MPI_Dims_create(ops_comm_size, ndim, pdims);
  MPI_Cart_create( OPS_MPI_WORLD,  ndim,  pdims,  periodic,
    1,  &OPS_CART_COMM);

/** ---- determine subgrid dimensions and displacements ---- **/

  int my_cart_rank;
  int *coords = (int *) xmalloc(ndim*sizeof(int));
  int *disps = (int *) xmalloc(ndim*sizeof(int));
  int *sizes = (int *) xmalloc(ndim*sizeof(int));


  MPI_Comm_rank(OPS_CART_COMM, &my_cart_rank);
  MPI_Cart_coords( OPS_CART_COMM, my_cart_rank, ndim, coords);

  for(int n=0; n<ndim; n++){
    disps[n] = (coords[n] * g_sizes[n])/pdims[n];
    sizes[n]  = ((coords[n]+1)*g_sizes[n])/pdims[n] - disps[n];
    g_sizes[n] = sizes[n];
  }

/** ---- get IDs of neighbours ---- **/

  int *id_m = (int *) xmalloc(ndim*sizeof(int));
  int *id_p = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++)
    MPI_Cart_shift(OPS_CART_COMM, n, 1, &id_m[n], &id_p[n]);

/** ---- calculate subgrid start and end indicies ---- **/

  int *istart = (int *) xmalloc(ndim*sizeof(int));
  int *iend = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++){
    istart[n] = disps[n];
    iend[n] = istart[n]+sizes[n]-1;
  }

/** ---- Store subgrid decomposition geometries ---- **/

  sub_block_list sb_list= (sub_block_list)xmalloc(sizeof(sub_block));
  sb_list->block = block;
  sb_list->ndim = ndim;
  sb_list->coords = coords;
  sb_list->id_m = id_m;
  sb_list->id_p = id_p;
  sb_list->sizes = sizes;
  sb_list->disps = disps;
  sb_list->istart = istart;
  sb_list->iend = iend;

  OPS_sub_block_list[block->index] = sb_list;

  MPI_Barrier(OPS_MPI_WORLD);
  ops_printf("block \"%s\" decomposed on to a processor grid of ",block->name);
  for(int n=0; n<ndim; n++){
    ops_printf("%d ",pdims[n]);
    n == ndim-1? ops_printf(" ") : ops_printf("x ");
  }
  ops_printf("\n");
}


void ops_partition(int g_ndim, int* g_sizes, char* routine)
{
  //create list to hold sub-grid decomposition geometries for each mpi process
  OPS_sub_block_list = (sub_block_list *)xmalloc(OPS_block_index*sizeof(sub_block_list));

  for(int b=0; b<OPS_block_index; b++){ //for each block
    ops_block block=OPS_block_list[b];
    ops_decomp(block, g_ndim, g_sizes); //for now there is only one block

    sub_block_list sb_list = OPS_sub_block_list[block->index];

    printf(" ===========================================================================\n" );
    printf(" rank %d (",ops_my_rank);
    for(int n=0; n<sb_list->ndim; n++)
      printf("%d ",sb_list->coords[n]);
    printf(")\n");
    printf( " ------------------------------\n" );
    printf(" %5s  :  %9s  :  %9s  :  %5s  :  %5s  :  %5s  :  %5s\n",
      "dim", "prev_rank", "next_rank", "disp", "size","start",  "end");
    for(int n=0; n<sb_list->ndim; n++)
    printf(" %5d  :  %9d  :  %9d  :  %5d  :  %5d  :  %5d  :  %5d\n",
      n, sb_list->id_m[n], sb_list->id_p[n], sb_list->disps[n], sb_list->sizes[n],
      sb_list->istart[n], sb_list->iend[n]);
    printf("\n");
  }
  ops_printf("Finished block decomposition\n");

}

//ops_dats will be declared with max halo depths in both directions in a dimension
//use the same depth when allocating the local ops_dats on each MPI process

//for each ops_dat declared on a decomposed ops_block
  //for each dimension of the block
    //for each direction in dimension
      //create a send buffer and a receive buffer

//halo exchange routine should aim to be one Isend/Ireceive per block dimension
//Then this routine can be called for each dimension as in the decompose() routine, with a waitall after each dimension
//might need MPI_CART_RANK to convert the cartecian coordinates of sender and receiver to actual MPI ranks -- seems no need for this


//special case where iterating in 2D and accessing 1D edge, then all procs will need to
//have a new special halo created... this will only be known at loop runtime
//and perhaps will need to be allocated on-the-fly.
