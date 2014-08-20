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
#include <ops_mpi_core.h>
#include <math.h>

extern int ops_buffer_size;
extern char *ops_buffer_send_1;
extern char *ops_buffer_recv_1;
extern char *ops_buffer_send_2;
extern char *ops_buffer_recv_2;
extern int ops_buffer_send_1_size;
extern int ops_buffer_recv_1_size;
extern int ops_buffer_send_2_size;
extern int ops_buffer_recv_2_size;



MPI_Comm OPS_MPI_WORLD; // comm world for a single block
                        // -- need to have a communicator for each block in multi-block

MPI_Comm OPS_CART_COMM; // cartesian comm world
// -- agian need to have and store
// a comm world for each block in multi-block


/*
* Lists of sub-blocks and sub-dats declared in an OPS programs -- for MPI backends
*/

int ops_comm_size;
int ops_my_rank;

sub_block_list *OPS_sub_block_list;// pointer to list holding sub-block
                                   // geometries


sub_dat_list *OPS_sub_dat_list;// pointer to list holding sub-dat
                                 // details


void ops_decomp(ops_block block, int g_ndim, int* g_sizes)
{
  //g_dim  - global number of dimensions .. will be the same on each local mpi process
  //g_sizes - global dimension sizes, i.e. size in each dimension of the global mesh

  sub_block *sb= (sub_block *)xmalloc(sizeof(sub_block));
  sb->block = block;
  sb->ndim = g_ndim;

/** ---- create cartesian processor grid ---- **/

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

  MPI_Comm_rank(OPS_CART_COMM, &my_cart_rank);
  MPI_Cart_coords( OPS_CART_COMM, my_cart_rank, ndim, sb->coords);

  for(int n=0; n<ndim; n++){
    sb->decomp_disp[n] = (sb->coords[n] * g_sizes[n])/pdims[n];
    sb->decomp_size[n]  = ((sb->coords[n]+1)*g_sizes[n])/pdims[n] - sb->decomp_disp[n];
    g_sizes[n] = sb->decomp_size[n];
  }

/** ---- get IDs of neighbours ---- **/

  for(int n=0; n<ndim; n++)
    MPI_Cart_shift(OPS_CART_COMM, n, 1, &(sb->id_m[n]), &(sb->id_p[n]));


/** ---- Store subgrid decomposition geometries ---- **/

  OPS_sub_block_list[block->index] = sb;

  MPI_Barrier(OPS_MPI_WORLD);
  ops_printf("block \"%s\" decomposed on to a processor grid of ",block->name);
  for(int n=0; n<ndim; n++){
    ops_printf("%d ",pdims[n]);
    n == ndim-1? ops_printf(" ") : ops_printf("x ");
  }
  ops_printf("\n");
  free(pdims);
  free(periodic);
}

void ops_decomp_dats(sub_block *sb) {
  ops_block block = sb->block;
  ops_dat_entry *item, *tmp_item;
  for (item = TAILQ_FIRST(&(OPS_block_list[block->index].datasets)); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;
    sub_dat *sd = OPS_sub_dat_list[dat->index];

    //aggregate size and prod array
    int *prod_t = (int *) xmalloc((sb->ndim+1)*sizeof(int));
    int *prod = &prod_t[1];
    prod[-1] = 1;
    sd->prod = prod;

    for (int d = 0; d < block->dims; d++) {
      sd->gbl_base[d] = dat->base[d];
      sd->gbl_size[d] = dat->size[d];

      //special treatment if it's an edge dataset in this direction
      if (dat->e_dat && (dat->size[d] == 1)) {
        if (dat->base[d]!=0) {printf("Dataset %s is an edge dataset, but has a non-0 base\n", dat->name); exit(-1);}
        prod[d] = prod[d-1];
        sd->decomp_disp[d] = 0;
        sd->decomp_size[d] = 1;
        continue;
      }

      int zerobase_gbl_size = dat->size[d] + dat->d_m[d] + dat->d_p[d] + dat->base[d];
      sd->decomp_disp[d] = sb->decomp_disp[d];
      sd->decomp_size[d] = MAX(0,MIN(sb->decomp_size[d], zerobase_gbl_size - sb->decomp_disp[d]));
      if(sb->id_m[d] != MPI_PROC_NULL) {
        //if not negative end, then base should be 0
        dat->base[d] = 0;
      }
      dat->size[d] = sd->decomp_size[d] - dat->base[d] - dat->d_m[d] - dat->d_p[d]; //TODO: block halo doubles as intra-block halo
      prod[d] = prod[d-1]*dat->size[d];
    }

    //Allocate datasets
    //TODO: read HDF5, what if it was already allocated - re-distribute
    dat->data = (char *)calloc(prod[sb->ndim-1]*dat->elem_size,1);
    ops_cpHostToDevice ( (void**)&(dat->data_d), (void**)&(dat->data_d), prod[sb->ndim-1]*dat->elem_size);

    //TODO: halo exchanges should not include the block halo part for partitions that are on the edge of a block
    sd->mpidat = (MPI_Datatype *) xmalloc(sizeof(MPI_Datatype)*sb->ndim * MAX_DEPTH);

    MPI_Datatype new_type_p; //create generic type for MPI comms
    MPI_Type_contiguous(dat->elem_size, MPI_CHAR, &new_type_p);
    MPI_Type_commit(&new_type_p);
    sd->halos=(ops_int_halo *)malloc(MAX_DEPTH*sb->ndim*sizeof(ops_int_halo));

    for(int n = 0; n<sb->ndim; n++) {
      for(int d = 0; d<MAX_DEPTH; d++) {
        MPI_Type_vector(prod[sb->ndim - 1]/prod[n], d*prod[n-1],
                        prod[n], new_type_p, &(sd->mpidat[MAX_DEPTH*n+d]));
        MPI_Type_commit(&(sd->mpidat[MAX_DEPTH*n+d]));
        sd->halos[MAX_DEPTH*n+d].count = prod[sb->ndim - 1]/prod[n];
        sd->halos[MAX_DEPTH*n+d].blocklength = d*prod[n-1] * dat->elem_size;
        sd->halos[MAX_DEPTH*n+d].stride = prod[n] * dat->elem_size;
        //printf("Datatype: %d %d %d\n", prod[sb->ndim - 1]/prod[n], prod[n-1], prod[n]);
      }
    }
  }
}

void ops_partition(char* routine)
{
  //create list to hold sub-grid decomposition geometries for each mpi process
  OPS_sub_block_list = (sub_block_list *)xmalloc(OPS_block_index*sizeof(sub_block_list));

  int max_block_dim = 0;
  int max_block_dims = 0;

  for(int b=0; b<OPS_block_index; b++){ //for each block
    ops_block block=OPS_block_list[b].block;
    int max_sizes[OPS_MAX_DIM] = {0};
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&(OPS_block_list[block->index].datasets)); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      for (int d = 0; d < block->dims; d++)
        max_sizes[d] = MAX(item->dat->size[d]+item->dat->base[d]+item->dat->d_m[d]+item->dat->d_p[d],max_sizes[d]);
    }
    //decompose based on maximum dataset size defined on this block
    ops_decomp(block, block->dims, max_sizes); //for now there is only one block

    sub_block *sb = OPS_sub_block_list[block->index];

    ops_decomp_dats(sb);

    printf(" ===========================================================================\n" );
    printf(" rank %d (",ops_my_rank);
    for(int n=0; n<sb->ndim; n++)
      printf("%d ",sb->coords[n]);
    printf(")\n");
    printf( " ------------------------------\n" );
    printf(" %5s  :  %9s  :  %9s  :  %5s  :  %5s  :  %5s  :  %5s\n",
      "dim", "prev_rank", "next_rank", "disp", "size","start",  "end");
    for(int n=0; n<sb->ndim; n++)
    printf(" %5d  :  %9d  :  %9d  :  %5d  :  %5d\n",
      n, sb->id_m[n], sb->id_p[n], sb->decomp_disp[n], sb->decomp_size[n]);
    printf("\n");

    max_block_dims = MAX(max_block_dims,sb->ndim);
    for(int n=0; n<sb->ndim; n++) max_block_dim = MAX(max_block_dim,sb->decomp_size[n]);
  }
  ops_printf("Finished block decomposition\n");

  //allocate send/recv buffer (double, 8 args, maximum depth)
  ops_buffer_size = 8*8*MAX_DEPTH*pow(2*MAX_DEPTH+max_block_dim,max_block_dims-1);
  ops_comm_realloc(&ops_buffer_send_1,ops_buffer_size*sizeof(char),0);
  ops_comm_realloc(&ops_buffer_recv_1,ops_buffer_size*sizeof(char),0);
  ops_comm_realloc(&ops_buffer_send_2,ops_buffer_size*sizeof(char),0);
  ops_comm_realloc(&ops_buffer_recv_2,ops_buffer_size*sizeof(char),0);
  ops_buffer_send_1_size = ops_buffer_size;
  ops_buffer_recv_1_size = ops_buffer_size;
  ops_buffer_send_2_size = ops_buffer_size;
  ops_buffer_recv_2_size = ops_buffer_size;
}

//special case where iterating in 2D and accessing 1D edge, then all procs will need to
//have a new special halo created... this will only be known at loop runtime
//and perhaps will need to be allocated on-the-fly.


void ops_mpi_exit()
{
  ops_dat_entry *item;
  int i;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    i = (item->dat)->index;
    free(&OPS_sub_dat_list[i]->prod[-1]);
    free(OPS_sub_dat_list[i]->halos);
    for(int n = 0; n<OPS_sub_dat_list[i]->dat->block->dims; n++) {
      for(int d = 0; d<MAX_DEPTH; d++) {
        MPI_Type_free(&(OPS_sub_dat_list[i]->mpidat[MAX_DEPTH*n+d]));
      }
    }
    free(OPS_sub_dat_list[i]->mpidat);
    free(OPS_sub_dat_list[i]->dirty_dir_send);
    free(OPS_sub_dat_list[i]->dirty_dir_recv);
    free(OPS_sub_dat_list[i]);
  }
  free(OPS_sub_dat_list);
  OPS_sub_dat_list = NULL;
}
