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



MPI_Comm OPS_MPI_GLOBAL; // comm world
ops_mpi_halo *OPS_mpi_halo_list = NULL;
ops_mpi_halo_group *OPS_mpi_halo_group_list = NULL;

/*
* Lists of sub-blocks and sub-dats declared in an OPS programs -- for MPI backends
*/

int ops_comm_global_size;
int ops_my_global_rank;

sub_block_list *OPS_sub_block_list;// pointer to list holding sub-block
                                   // geometries


sub_dat_list *OPS_sub_dat_list;// pointer to list holding sub-dat
                                 // details

/*
 * Returns a CSR-like array, listing the processes assigned to each block and describing their sub-dimensions
 * This one is just a really primitive initial implementation
 */
void ops_partition_blocks(int **processes, int **proc_offsets, int **proc_disps, int **proc_sizes, int **proc_dimsplit) {
  *processes =    (int *)malloc(            OPS_block_index *sizeof(int));
  *proc_offsets = (int *)malloc(         (1+OPS_block_index)*sizeof(int));
  *proc_disps =   (int *)malloc(OPS_MAX_DIM*OPS_block_index *sizeof(int));
  *proc_sizes =   (int *)malloc(OPS_MAX_DIM*OPS_block_index *sizeof(int));
  *proc_dimsplit= (int *)malloc(OPS_MAX_DIM*OPS_block_index *sizeof(int));

  for (int i = 0; i < OPS_block_index; i++) {
    (*processes)[i] = i%ops_comm_global_size;
    (*proc_offsets)[i] = i;
    ops_block block=OPS_block_list[i].block;
    int max_sizes[OPS_MAX_DIM] = {0};
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&(OPS_block_list[block->index].datasets)); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      for (int d = 0; d < block->dims; d++)
        max_sizes[d] = MAX(item->dat->size[d]+item->dat->base[d]+item->dat->d_m[d]+item->dat->d_p[d],max_sizes[d]);
    }

    for (int j = 0; j < OPS_MAX_DIM; j++) {
      (*proc_disps)[OPS_MAX_DIM*i+j] = 0;
      (*proc_sizes)[OPS_MAX_DIM*i+j] = max_sizes[j];
      (*proc_dimsplit)[OPS_MAX_DIM*i+j] = 1;
    }
  }
  (*proc_offsets)[OPS_block_index] = OPS_block_index;
}

void ops_decomp(ops_block block, int num_proc, int *processes, int *proc_disps, int *proc_sizes, int *proc_dimsplit)
{

  int g_ndim = block->dims;
  //g_dim  - global number of dimensions .. will be the same on each local mpi process
  //g_sizes - global dimension sizes, i.e. size in each dimension of the global mesh

  sub_block *sb= (sub_block *)xmalloc(sizeof(sub_block));
  sb->block = block;
  sb->ndim = g_ndim;

/** ---- create cartesian processor grid ---- **/

  int ndim = g_ndim;
  int *pdims = proc_dimsplit;
  int *periodic = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++) {
    periodic[n] = 0; //false .. for now
  }
  MPI_Group global;
  MPI_Comm_group(OPS_MPI_GLOBAL, &global);
  MPI_Group_incl(global, num_proc, processes, &sb->grp);
  MPI_Comm_create(OPS_MPI_GLOBAL, sb->grp, &sb->comm1);
  MPI_Cart_create(sb->comm1,  ndim,  pdims,  periodic,
    1,  &sb->comm);

  sb->owned = 0;
  for (int i = 0; i < num_proc; i++) sb->owned = sb->owned || (processes[i] == ops_my_global_rank);

/** ---- determine subgrid dimensions and displacements ---- **/

  for(int n=0; n<ndim; n++){
    sb->decomp_disp[n] = proc_disps[n];//(sb->coords[n] * g_sizes[n])/pdims[n];
    sb->decomp_size[n] = proc_sizes[n];  //((sb->coords[n]+1)*g_sizes[n])/pdims[n] - sb->decomp_disp[n];
  }

/** ---- get IDs of neighbours ---- **/

  if (sb->owned) {
    int my_cart_rank;
    MPI_Comm_rank(sb->comm, &my_cart_rank);
    MPI_Cart_coords( sb->comm, my_cart_rank, ndim, sb->coords);

    for(int n=0; n<ndim; n++)
      MPI_Cart_shift(sb->comm, n, 1, &(sb->id_m[n]), &(sb->id_p[n]));
  }


/** ---- Store subgrid decomposition geometries ---- **/

  OPS_sub_block_list[block->index] = sb;

  ops_printf("block \"%s\" decomposed on to a processor grid of ",block->name);
  for(int n=0; n<ndim; n++){
    ops_printf("%d ",pdims[n]);
    n == ndim-1? ops_printf(" ") : ops_printf("x ");
  }
  ops_printf("\n");
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

    if (!sb->owned) continue;

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

int intersection(int range1_beg, int range1_end, int range2_beg, int range2_end) {
  int i_min = MAX(range1_beg,range2_beg);
  int i_max = MIN(range1_end,range2_end);
  return i_max>i_min ? i_max-i_min : 0;
}

void ops_partition_halos(int *processes, int *proc_offsets, int *proc_disps, int *proc_sizes, int *proc_dimsplit) {
  for (int i = 0; i < OPS_halo_index; i++) {
    ops_halo halo = OPS_halo_list[i];
    if (!OPS_sub_block_list[halo->from->block->index]->owned &&
        !OPS_sub_block_list[halo->to  ->block->index]->owned) {
      OPS_mpi_halo_list[i].nproc_from = 0;
      OPS_mpi_halo_list[i].nproc_to   = 0;
      OPS_mpi_halo_list[i].index      = i;
      OPS_mpi_halo_list[i].proclist = NULL;
      OPS_mpi_halo_list[i].local_from_base = NULL;
      OPS_mpi_halo_list[i].local_to_base = NULL;
      OPS_mpi_halo_list[i].local_iter_range = NULL;
      continue;
    }
    sub_block *sb_from = OPS_sub_block_list[halo->from->block->index];
    sub_dat   *sd_from = OPS_sub_dat_list[halo->from->index];
    sub_block *sb_to   = OPS_sub_block_list[halo->to->block->index];
    sub_dat   *sd_to   = OPS_sub_dat_list[halo->to->index];
    OPS_mpi_halo_list[i].nproc_from = 0;
    OPS_mpi_halo_list[i].nproc_to = 0;
    OPS_mpi_halo_list[i].index      = i;
    OPS_mpi_halo_list[i].proclist = NULL;
    OPS_mpi_halo_list[i].local_from_base = NULL;
    OPS_mpi_halo_list[i].local_to_base = NULL;
    OPS_mpi_halo_list[i].local_iter_range = NULL;

    //Map out all halo regions where the current process has the "from" part
    if (sb_from->owned) {
      int all_dims = 1;
      int intersection_range[OPS_MAX_DIM]; //size of my part of the halo
      int relative_base[OPS_MAX_DIM]; //of my part of the halo into the full halo
      //first, compute the intersection and location of the halo within my owned part of the dataset
      for (int d = 0; d < sb_from->ndim; d++) {
        int intersection_local = intersection(sd_from->decomp_disp[d] + sd_from->dat->base[d], //check block halo??
                                              sd_from->decomp_disp[d] + sd_from->decomp_size[d],
                                              halo->from_base[d],
                                              halo->from_base[d]+halo->iter_size[abs(halo->from_dir[d])-1]);
        int d2 = 0;
        while (d2 != abs(halo->from_dir[d])-1) d2++;
        intersection_range[d2] = intersection_local;
        //where my part of the halo begins in the full halo, with the coordinate ordering of iter_range
        relative_base[d2] = (sd_from->decomp_disp[d] + sd_from->dat->base[d] - halo->from_base[d]) > 0 ?
                            (sd_from->decomp_disp[d] + sd_from->dat->base[d] - halo->from_base[d]) : 0;
        all_dims = all_dims && (intersection_local>0);
      }
      //it there is an actual intersection, discover all target partitions that connect to my bit of the halo
      if (all_dims) {
        //There is going to be at least one destination, at most the number of processes that hold parts of the destination dataset
        int max_dest = proc_offsets[halo->to->block->index+1] - proc_offsets[halo->to->block->index];
        OPS_mpi_halo_list[i].proclist = (int *)malloc(max_dest*sizeof(int));
        OPS_mpi_halo_list[i].local_from_base = (int *)malloc(OPS_MAX_DIM*max_dest*sizeof(int));
        OPS_mpi_halo_list[i].local_to_base = (int *)malloc(OPS_MAX_DIM*max_dest*sizeof(int));
        OPS_mpi_halo_list[i].local_iter_range = (int *)malloc(OPS_MAX_DIM*max_dest*sizeof(int));

        //find intersecting destination partitions
        for (int j = proc_offsets[halo->to->block->index];
                 j < proc_offsets[halo->to->block->index+1]; ++j) {
          all_dims = 0;
          int to_base[OPS_MAX_DIM], from_relative_base[OPS_MAX_DIM], owned_range_for_proc[OPS_MAX_DIM];
          for (int d = 0; d < sb_to->ndim; ++d) {
            int left_pad  = sd_to->dat->base[d] + (proc_disps[j*OPS_MAX_DIM+d]==0 ? sd_to->dat->d_m[d] : 0);
            int right_pad = (proc_disps[j*OPS_MAX_DIM+d] + proc_sizes[j*OPS_MAX_DIM+d]) >= sd_to->gbl_size[d] ? (-sd_to->dat->d_p[d]) : 0;
            int intersection_local = intersection(sd_to->decomp_disp[d] + left_pad,
                                              sd_to->decomp_disp[d] + sd_to->decomp_size[d] + right_pad,
                                              halo->to_base[d] + relative_base[abs(halo->to_dir[d])-1],
                                              halo->to_base[d] + relative_base[abs(halo->to_dir[d])-1]+intersection_range[abs(halo->to_dir[d])-1]);
            int d2 = 0;
            while (d2 != abs(halo->to_dir[d])-1) d2++;
            owned_range_for_proc[d2] = intersection_local;
            to_base[d] = (halo->to_base[d] + relative_base[abs(halo->to_dir[d])-1] - sd_to->decomp_disp[d]) > left_pad ?
                         (halo->to_base[d] + relative_base[abs(halo->to_dir[d])-1] - sd_to->decomp_disp[d]) : 0;
            from_relative_base[d2] = (sd_to->decomp_disp[d] + left_pad - halo->to_base[d] - relative_base[abs(halo->to_dir[d])-1]) > 0 ?
                                     (sd_to->decomp_disp[d] + left_pad - halo->to_base[d] - relative_base[abs(halo->to_dir[d])-1]) : 0;
            all_dims = all_dims && (intersection_local>0);
          }
          if (all_dims) {
            //set up entry
            int entry = OPS_mpi_halo_list[i].nproc_from;
            OPS_mpi_halo_list[i].proclist[entry] = processes[j];
            for (int d = 0; d < sb_from->ndim; d++) {
              OPS_mpi_halo_list[i].local_iter_range[OPS_MAX_DIM*entry + d] = owned_range_for_proc[d];
              OPS_mpi_halo_list[i].local_from_base[OPS_MAX_DIM*entry + d] =
                 ((halo->from_base[d] - sd_from->decomp_disp[d]) > sd_from->dat->base[d] ?
                  (halo->from_base[d] - sd_from->decomp_disp[d]) > sd_from->dat->base[d] : 0) + from_relative_base[abs(halo->from_dir[d])-1];
              OPS_mpi_halo_list[i].local_to_base[OPS_MAX_DIM*entry + d] = to_base[d]; //This isn't really relevant, but good for debug (if both blocks on same proc)
            }
            OPS_mpi_halo_list[i].nproc_from++;
          }
        }
      }
    }

    //Map out all halo regions where the current process has the "to" part
    if (sb_to->owned) {
      int all_dims = 1;
      int intersection_range[OPS_MAX_DIM]; //size of my part of the halo
      int relative_base[OPS_MAX_DIM]; //of my part of the halo into the full halo
      //first, compute the intersection and location of the halo within my owned part of the dataset
      for (int d = 0; d < sb_from->ndim; d++) {
        int left_pad  = sd_to->dat->base[d] + (sd_to->decomp_disp[d]==0 ? sd_to->dat->d_m[d] : 0);
        int right_pad = (sd_to->decomp_disp[d] + sd_to->decomp_size[d]) >= sd_to->gbl_size[d] ? (-sd_to->dat->d_p[d]) : 0;
        int intersection_local = intersection(sd_to->decomp_disp[d] + left_pad,
                                              sd_to->decomp_disp[d] + sd_to->decomp_size[d] + right_pad,
                                              halo->to_base[d],
                                              halo->to_base[d]+halo->iter_size[abs(halo->to_dir[d])-1]);
        int d2 = 0;
        while (d2 != abs(halo->to_dir[d])-1) d2++;
        intersection_range[d2] = intersection_local;
        //where my part of the halo begins in the full halo, with the coordinate ordering of iter_range
        relative_base[d2] = (sd_to->decomp_disp[d] + left_pad - halo->to_base[d]) > 0 ?
                            (sd_to->decomp_disp[d] + left_pad - halo->to_base[d]) : 0;
        all_dims = all_dims && (intersection_local>0);
      }
      //it there is an actual intersection, discover all target partitions that connect to my bit of the halo
      if (all_dims) {
        //There is going to be at least one source, at most the number of processes that hold parts of the source dataset
        int max_src = proc_offsets[halo->from->block->index+1] - proc_offsets[halo->from->block->index];
        int existing = OPS_mpi_halo_list[i].nproc_from;
        OPS_mpi_halo_list[i].proclist = (int *)realloc(OPS_mpi_halo_list[i].proclist, (max_src+existing)*sizeof(int));
        OPS_mpi_halo_list[i].local_from_base = (int *)realloc(OPS_mpi_halo_list[i].local_from_base, OPS_MAX_DIM*(max_src+existing)*sizeof(int));
        OPS_mpi_halo_list[i].local_to_base = (int *)realloc(OPS_mpi_halo_list[i].local_to_base, OPS_MAX_DIM*(max_src+existing)*sizeof(int));
        OPS_mpi_halo_list[i].local_iter_range = (int *)realloc(OPS_mpi_halo_list[i].local_iter_range, OPS_MAX_DIM*(max_src+existing)*sizeof(int));

        //find intersecting destination partitions
        for (int j = proc_offsets[halo->from->block->index];
                 j < proc_offsets[halo->from->block->index+1]; ++j) {
          all_dims = 0;
          int from_base[OPS_MAX_DIM], to_relative_base[OPS_MAX_DIM], owned_range_for_proc[OPS_MAX_DIM];
          for (int d = 0; d < sb_from->ndim; ++d) {
            int intersection_local = intersection(sd_from->decomp_disp[d] + sd_from->dat->base[d], //check block halo??
                                              sd_from->decomp_disp[d] + sd_from->decomp_size[d],
                                              halo->from_base[d] + relative_base[abs(halo->from_dir[d])-1],
                                              halo->from_base[d] + relative_base[abs(halo->from_dir[d])-1]+intersection_range[abs(halo->from_dir[d])-1]);
            int d2 = 0;
            while (d2 != abs(halo->from_dir[d])-1) d2++;
            owned_range_for_proc[d2] = intersection_local;
            from_base[d] = (halo->from_base[d] + relative_base[abs(halo->from_dir[d])-1] - sd_from->decomp_disp[d]) > sd_from->dat->base[d] ?
                           (halo->from_base[d] + relative_base[abs(halo->from_dir[d])-1] - sd_from->decomp_disp[d]) : 0;
            to_relative_base[d2] = (sd_from->decomp_disp[d] + sd_from->dat->base[d] - halo->from_base[d] - relative_base[abs(halo->from_dir[d])-1]) > 0 ?
                                   (sd_from->decomp_disp[d] + sd_from->dat->base[d] - halo->from_base[d] - relative_base[abs(halo->from_dir[d])-1]) : 0;
            all_dims = all_dims && (intersection_local>0);
          }
          if (all_dims) {
            //set up entry
            int entry = OPS_mpi_halo_list[i].nproc_from + OPS_mpi_halo_list[i].nproc_to;
            OPS_mpi_halo_list[i].proclist[entry] = processes[j];
            for (int d = 0; d < sb_to->ndim; d++) {
              OPS_mpi_halo_list[i].local_iter_range[OPS_MAX_DIM*entry + d] = owned_range_for_proc[d];
              OPS_mpi_halo_list[i].local_to_base[OPS_MAX_DIM*entry + d] =
                 ((halo->to_base[d] - sd_to->decomp_disp[d]) > sd_to->dat->base[d] ?
                  (halo->to_base[d] - sd_to->decomp_disp[d]) > sd_to->dat->base[d] : 0) + to_relative_base[abs(halo->to_dir[d])-1];
              OPS_mpi_halo_list[i].local_from_base[OPS_MAX_DIM*entry + d] = from_base[d]; //This isn't really relevant, but good for debug (if both blocks on same proc)
            }
            OPS_mpi_halo_list[i].nproc_to++;
          }
        }
      }
    }
  }
  for (int i = 0; i < OPS_halo_group_index; i++) {
    ops_halo_group group = OPS_halo_group_list[i];
    int owned = 0;
    for (int j = 0; j < group->nhalos; j++) {
      if (OPS_mpi_halo_list[group->halos[j]->index].nproc_from > 0 || OPS_mpi_halo_list[group->halos[j]->index].nproc_to > 0) owned++;
    }
    OPS_mpi_halo_group_list[i].group = group;
    OPS_mpi_halo_group_list[i].nhalos = owned;
    OPS_mpi_halo_group_list[i].index = i;
    OPS_mpi_halo_group_list[i].mpi_halos = (ops_mpi_halo **)malloc(owned*sizeof(ops_mpi_halo*));
    owned = 0;
    for (int j = 0; j < group->nhalos; j++) {
      if (OPS_mpi_halo_list[group->halos[j]->index].nproc_from > 0 || OPS_mpi_halo_list[group->halos[j]->index].nproc_to > 0) {
        OPS_mpi_halo_group_list[i].mpi_halos[owned++] = &OPS_mpi_halo_list[group->halos[j]->index];
      }
    }
    //TODO: create stored list by destination/source partition for aggregation
  }
}

void ops_partition(char* routine)
{
  //create list to hold sub-grid decomposition geometries for each mpi process
  OPS_sub_block_list = (sub_block_list *)xmalloc(OPS_block_index*sizeof(sub_block_list));

  int max_block_dim = 0;
  int max_block_dims = 0;

  //Distribute blocks amongst processes
  int *processes, *proc_offsets, *proc_disps, *proc_sizes, *proc_dimsplit;
  ops_partition_blocks(&processes, &proc_offsets, &proc_disps, &proc_sizes, &proc_dimsplit);

  for(int b=0; b<OPS_block_index; b++){ //for each block
    //decompose this block
    int num_proc = proc_offsets[b+1] - proc_offsets[b];
    ops_block block = OPS_block_list[b].block;
    ops_decomp(block, num_proc, &processes[proc_offsets[b]], &proc_disps[OPS_MAX_DIM*proc_offsets[b]],
               &proc_sizes[OPS_MAX_DIM*proc_offsets[b]], &proc_dimsplit[OPS_MAX_DIM*b]);

    sub_block *sb = OPS_sub_block_list[block->index];

    ops_decomp_dats(sb);

    printf(" ===========================================================================\n" );
    printf(" rank %d (",ops_my_global_rank);
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

  OPS_mpi_halo_list = (ops_mpi_halo *)malloc(OPS_halo_index * sizeof(ops_mpi_halo));
  OPS_mpi_halo_group_list = (ops_mpi_halo_group *)malloc(OPS_halo_group_index * sizeof(ops_mpi_halo_group));
  ops_partition_halos(processes, proc_offsets, proc_disps, proc_sizes, proc_dimsplit);

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

  free(processes);
  free(proc_offsets);
  free(proc_disps);
  free(proc_sizes);
  free(proc_dimsplit);
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

  for (int i = 0; i < OPS_halo_index; i++) {
    if (OPS_mpi_halo_list[i].nproc_from > 0 || OPS_mpi_halo_list[i].nproc_to > 0) {
      free(OPS_mpi_halo_list[i].proclist);
      free(OPS_mpi_halo_list[i].local_from_base);
      free(OPS_mpi_halo_list[i].local_to_base);
      free(OPS_mpi_halo_list[i].local_iter_range);
    }
  }
  free(OPS_mpi_halo_list);
  for (int i = 0; i < OPS_halo_group_index; i++) {
    if (OPS_mpi_halo_group_list[i].nhalos > 0) {
      free(OPS_mpi_halo_group_list[i].mpi_halos);
    }
  }
  free(OPS_mpi_halo_group_list);
}
