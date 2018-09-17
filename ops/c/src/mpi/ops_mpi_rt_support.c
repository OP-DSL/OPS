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

/** @brief ops mpi run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi backend
  */

#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

extern int OPS_diags;

// Timing
double t1, t2, c1, c2;

#define AGGREGATE
int ops_buffer_size = 0;
char *ops_buffer_send_1 = NULL;
char *ops_buffer_recv_1 = NULL;
char *ops_buffer_send_2 = NULL;
char *ops_buffer_recv_2 = NULL;
int ops_buffer_send_1_size = 0;
int ops_buffer_recv_1_size = 0;
int ops_buffer_send_2_size = 0;
int ops_buffer_recv_2_size = 0;
int *mpi_neigh_size = NULL;

extern double ops_gather_time;
extern double ops_scatter_time;
extern double ops_sendrecv_time;

int intersection(int range1_beg, int range1_end, int range2_beg,
                 int range2_end) {
  int i_min = MAX(range1_beg, range2_beg);
  int i_max = MIN(range1_end, range2_end);
  return i_max > i_min ? i_max - i_min : 0;
}

int contains(int point, int *range) {
  return (point >= range[0] && point < range[1]);
}

int ops_compute_intersections(ops_dat dat, int d_pos, int d_neg, 
                              int *iter_range, int dim,
                              int *left_send_depth, int *left_recv_depth,
                              int *right_send_depth, int *right_recv_depth) {

  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];

  int range_intersect[OPS_MAX_DIM] = {0};

  int ndim = sb->ndim;

  for (int dim = 0; dim < ndim; dim++) {
    range_intersect[dim] = intersection(
        iter_range[2 * dim] + d_neg, iter_range[2 * dim + 1] + d_pos,
        sd->decomp_disp[dim],
        (sd->decomp_disp[dim] + sd->decomp_size[dim])); // i.e. the intersection
                                                        // of the dependency
                                                        // range with my full
                                                        // range
  }

  if (d_pos > 0) {
    *left_send_depth =
        contains(sd->decomp_disp[dim] - 1, &iter_range[2 * dim])
            ? // if my left neighbor's last point is in the iteration range
            d_pos
            : // then it needs full depth required by the stencil
            (iter_range[2 * dim + 1] < sd->decomp_disp[dim]
                 ? // otherwise if range ends somewhere before my range begins
                 MAX(0,
                     d_pos - (sd->decomp_disp[dim] - iter_range[2 * dim + 1]))
                 :   // the dependency may still reach into my range
                 0); // otherwise 0

    *right_recv_depth =
        contains((sd->decomp_disp[dim] + sd->decomp_size[dim]) - 1,
                 &iter_range[2 * dim])
            ? // if my last point is in the iteration range
            d_pos
            : // then I need full depth required by the stencil
            (iter_range[2 * dim + 1] <
                     (sd->decomp_disp[dim] + sd->decomp_size[dim])
                 ? // otherwise if range ends somewhere before my range ends
                 MAX(0, d_pos - ((sd->decomp_disp[dim] + sd->decomp_size[dim]) -
                                 iter_range[2 * dim + 1]))
                 :   // the dependency may still reach into my neighbor's range
                 0); // otherwise 0
  }

  if (d_neg < 0) {
    *left_recv_depth =
        contains(sd->decomp_disp[dim], &iter_range[2 * dim])
            ? // if my first point is in the iteration range
            -d_neg
            : // then I need full depth required by the stencil
            (iter_range[2 * dim] > sd->decomp_disp[dim]
                 ? // otherwise if range starts somewhere after my range begins
                 MAX(0, -d_neg - (iter_range[2 * dim] - sd->decomp_disp[dim]))
                 : // the dependency may still reach into my left neighbor's
                 // range
                 0); // otherwise 0

    *right_send_depth =
        contains((sd->decomp_disp[dim] + sd->decomp_size[dim]),
                 &iter_range[2 * dim])
            ? // if my neighbor's first point is in the iteration range
            -d_neg
            : // then it needs full depth required by the stencil
            (iter_range[2 * dim] > (sd->decomp_disp[dim] + sd->decomp_size[dim])
                 ? // otherwise if range starts somewhere after my neighbor's
                 // range begins
                 MAX(0, -d_neg - (iter_range[2 * dim] - (sd->decomp_disp[dim] +
                                                         sd->decomp_size[dim])))
                 :   // the dependency may still reach into my range
                 0); // otherwise 0
  }

  // decide whether we intersect in all other dimensions
  int other_dims = 1;
  for (int d2 = 0; d2 < ndim; d2++)
    if (d2 != dim)
      other_dims =
          other_dims && (range_intersect[d2] > 0 || dat->size[d2] == 1);

  if (other_dims == 0)
    return 0;
  return 1;

}
void ops_exchange_halo_packer(ops_dat dat, int d_pos, int d_neg,
                              int *iter_range, int dim,
                              int *send_recv_offsets) {
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_send_depth = 0;
  int left_recv_depth = 0;
  int right_send_depth = 0;
  int right_recv_depth = 0;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++) {
    d_m[d] = dat->d_m[d] + sd->d_im[d];
    d_p[d] = dat->d_p[d] + sd->d_ip[d];
  }

  int *prod = sd->prod;

  if (!ops_compute_intersections(dat, d_pos,d_neg, iter_range, dim,
          &left_send_depth, &left_recv_depth, &right_send_depth, &right_recv_depth)) return;

  //
  // negative direction
  //

  // decide actual depth based on dirtybits
  int actual_depth_send = 0;
  for (int d = 0; d <= left_send_depth; d++)
    if (sd->dirty_dir_send[2 * MAX_DEPTH * dim + d] == 1)
      actual_depth_send = d;

  int actual_depth_recv = 0;
  for (int d = 0; d <= right_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d] == 1)
      actual_depth_recv = d;

  if (actual_depth_recv > abs(d_m[dim])) {
    printf("Error: trying to exchange a %d-deep halo for %s, but halo is only %d deep. Please set d_m and d_p accordingly\n",actual_depth_recv, dat->name, abs(d_m[dim]));
    MPI_Abort(sb->comm, -1);
  }

  // set up initial pointers
  int i2 = (-d_m[dim]) * prod[dim - 1];
  // int i4 = (prod[dim]/prod[dim-1] - (d_p[dim])    ) * prod[dim-1];
  // printf("block %s, dat %s, prod[dim-1] %d, prod[dim]
  // %d\n",dat->block->name,dat->name, prod[dim-1],prod[dim]);

  if (OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_m[dim], 665, &they_send,
                 1, MPI_INT, sb->id_p[dim], 665, sb->comm, &status);
    if (sb->id_p[dim] >= 0 && actual_depth_recv != they_send) {
      printf("Error: Right recv mismatch\n");
      MPI_Abort(sb->comm, -1);
    }
  }

  // Compute size of packed data
  int send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  int recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[0] + send_size > ops_buffer_send_1_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_send_1\n");
    ops_buffer_send_1 = (char *)ops_realloc_fast(ops_buffer_send_1, send_recv_offsets[0],
                                        send_recv_offsets[0] + 4 * send_size);
    ops_buffer_send_1_size = send_recv_offsets[0] + 4 * send_size;
  }
  if (send_recv_offsets[1] + recv_size > ops_buffer_recv_1_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_recv_1\n");
    ops_buffer_recv_1 = (char *)ops_realloc_fast(ops_buffer_recv_1, send_recv_offsets[1],
                                        send_recv_offsets[1] + 4 * recv_size);
    ops_buffer_recv_1_size = send_recv_offsets[0] + 4 * recv_size;
  }

  // Pack data
  if (actual_depth_send > 0)
    ops_pack(dat, i2, ops_buffer_send_1 + send_recv_offsets[0],
             &sd->halos[MAX_DEPTH * dim + actual_depth_send]);

  // if (actual_depth_send>0)
  //   ops_printf("%s send neg %d\n",dat->name, actual_depth_send);

  // increase offset
  send_recv_offsets[0] += send_size;
  send_recv_offsets[1] += recv_size;

  // clear dirtybits
  for (int d = 0; d <= actual_depth_send; d++)
    sd->dirty_dir_send[2 * MAX_DEPTH * dim + d] = 0;

  //
  // similarly for positive direction
  //

  // decide actual depth based on dirtybits
  actual_depth_send = 0;
  for (int d = 0; d <= right_send_depth; d++)
    if (sd->dirty_dir_send[2 * MAX_DEPTH * dim + MAX_DEPTH + d] == 1)
      actual_depth_send = d;

  actual_depth_recv = 0;
  for (int d = 0; d <= left_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + d] == 1)
      actual_depth_recv = d;

  if (actual_depth_recv > d_p[dim]) {
    printf("Error: trying to exchange a %d-deep halo for %s, but halo is only %d deep. Please set d_m and d_p accordingly\n",actual_depth_recv, dat->name, d_p[dim]);
    MPI_Abort(sb->comm, -1);
  }
      
  // set up initial pointers
  // int i1 = (-d_m[dim] - actual_depth_recv) * prod[dim-1];
  int i3 = (prod[dim] / prod[dim - 1] - (d_p[dim]) - actual_depth_send) *
           prod[dim - 1];

  if (OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_p[dim], 666, &they_send,
                 1, MPI_INT, sb->id_m[dim], 666, sb->comm, &status);
    if (sb->id_m[dim] >= 0 && actual_depth_recv != they_send) {
      printf("Error: Left recv mismatch\n");
      MPI_Abort(sb->comm, -1);
    }
  }

  // Compute size of packed data
  send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[2] + send_size > ops_buffer_send_2_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_send_2\n");
    ops_buffer_send_2 = (char *)ops_realloc_fast(ops_buffer_send_2,  send_recv_offsets[2],
                                        send_recv_offsets[2] + 4 * send_size);
    ops_buffer_send_2_size = send_recv_offsets[2] + 4 * send_size;
  }
  if (send_recv_offsets[3] + recv_size > ops_buffer_recv_2_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_recv_2\n");
    ops_buffer_recv_2 = (char *)ops_realloc_fast(ops_buffer_recv_2,  send_recv_offsets[3],
                                        send_recv_offsets[3] + 4 * recv_size);
    ops_buffer_recv_2_size = send_recv_offsets[3] + 4 * recv_size;
  }

  // pack data
  if (actual_depth_send > 0)
    ops_pack(dat, i3, ops_buffer_send_2 + send_recv_offsets[2],
             &sd->halos[MAX_DEPTH * dim + actual_depth_send]);

  // if (actual_depth_send>0)
  //   ops_printf("%s send pos %d\n",dat->name, actual_depth_send);

  // increase offset
  send_recv_offsets[2] += send_size;
  send_recv_offsets[3] += recv_size;

  // clear dirtybits
  for (int d = 0; d <= actual_depth_send; d++)
    sd->dirty_dir_send[2 * MAX_DEPTH * dim + MAX_DEPTH + d] = 0;
}

void ops_exchange_halo_packer_given(ops_dat dat, int *depths, int dim,
                              int *send_recv_offsets) {
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_send_depth = depths[0];
  int left_recv_depth = depths[1];
  int right_send_depth = depths[2];
  int right_recv_depth = depths[3];

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++) {
    d_m[d] = dat->d_m[d] + sd->d_im[d];
    d_p[d] = dat->d_p[d] + sd->d_ip[d];
  }

  if (sb->id_m[dim] != MPI_PROC_NULL && sd->d_im[dim] > -left_recv_depth) {
    printf("Error, requested %d depth halo exchange, only has %d deep halo. Please set OPS_TILING_MAXDEPTH.\n", left_recv_depth, -sd->d_im[dim]);
    MPI_Abort(sb->comm,-1);
  }
  if (sb->id_p[dim] != MPI_PROC_NULL && sd->d_ip[dim] < right_recv_depth) {
    printf("Error, requested %d depth halo exchange, only has %d deep halo. Please set OPS_TILING_MAXDEPTH.\n", right_recv_depth, sd->d_ip[dim]);
    MPI_Abort(sb->comm,-1);
  }

  int *prod = sd->prod;

  //
  // negative direction
  //

  // decide actual depth based on dirtybits
  int actual_depth_send = 0;
  for (int d = 0; d <= left_send_depth; d++)
    if (sd->dirty_dir_send[2 * MAX_DEPTH * dim + d] == 1)
      actual_depth_send = d;

  int actual_depth_recv = 0;
  for (int d = 0; d <= right_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d] == 1)
      actual_depth_recv = d;

  if (sb->id_m[dim] == MPI_PROC_NULL) {
    actual_depth_send = 0;
  }
  if (sb->id_p[dim] == MPI_PROC_NULL) {
    actual_depth_recv = 0;
  }

  // set up initial pointers
  int i2 = (-d_m[dim]) * prod[dim - 1];
  // int i4 = (prod[dim]/prod[dim-1] - (d_p[dim])    ) * prod[dim-1];
  // printf("block %s, dat %s, prod[dim-1] %d, prod[dim]
  // %d\n",dat->block->name,dat->name, prod[dim-1],prod[dim]);

  if (OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_m[dim], 665, &they_send,
                 1, MPI_INT, sb->id_p[dim], 665, sb->comm, &status);
    if (sb->id_p[dim] >= 0 && actual_depth_recv != they_send) {
      printf("Error: Proc %d dat %s dim %d: Right recv mismatch %d sends %d, I receive %d, originally asked to receive %d\n",ops_get_proc(), dat->name, dim, sb->id_p[dim], they_send, actual_depth_recv, right_recv_depth);
      MPI_Abort(sb->comm, -1);
    }
  }

  // Compute size of packed data
  int send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  int recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[0] + send_size > ops_buffer_send_1_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_send_1\n");
    ops_buffer_send_1 = (char *)ops_realloc_fast(ops_buffer_send_1,  send_recv_offsets[0],
                                        send_recv_offsets[0] + 4 * send_size);
    ops_buffer_send_1_size = send_recv_offsets[0] + 4 * send_size;
  }
  if (send_recv_offsets[1] + recv_size > ops_buffer_recv_1_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_recv_1\n");
    ops_buffer_recv_1 = (char *)ops_realloc_fast(ops_buffer_recv_1, send_recv_offsets[1],
                                        send_recv_offsets[1] + 4 * recv_size);
    ops_buffer_recv_1_size = send_recv_offsets[0] + 4 * recv_size;
  }

  // Pack data
  if (actual_depth_send > 0)
    ops_pack(dat, i2, ops_buffer_send_1 + send_recv_offsets[0],
             &sd->halos[MAX_DEPTH * dim + actual_depth_send]);

  // increase offset
  send_recv_offsets[0] += send_size;
  send_recv_offsets[1] += recv_size;

  // clear dirtybits
  for (int d = 0; d <= actual_depth_send; d++)
    sd->dirty_dir_send[2 * MAX_DEPTH * dim + d] = 0;

  //
  // similarly for positive direction
  //

  // decide actual depth based on dirtybits
  actual_depth_send = 0;
  for (int d = 0; d <= right_send_depth; d++)
    if (sd->dirty_dir_send[2 * MAX_DEPTH * dim + MAX_DEPTH + d] == 1)
      actual_depth_send = d;

  actual_depth_recv = 0;
  for (int d = 0; d <= left_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + d] == 1)
      actual_depth_recv = d;

  if (sb->id_p[dim] == MPI_PROC_NULL) {
    actual_depth_send = 0;
  }
  if (sb->id_m[dim] == MPI_PROC_NULL) {
    actual_depth_recv = 0;
  }

  // set up initial pointers
  // int i1 = (-d_m[dim] - actual_depth_recv) * prod[dim-1];
  int i3 = (prod[dim] / prod[dim - 1] - (d_p[dim]) - actual_depth_send) *
           prod[dim - 1];

  if (OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_p[dim], 666, &they_send,
                 1, MPI_INT, sb->id_m[dim], 666, sb->comm, &status);
    if (sb->id_m[dim] != MPI_PROC_NULL && actual_depth_recv != they_send) {
      printf("Error: Proc %d dat %s dim %d Left recv mismatch %d sends %d, I receive %d originally asked to recv %d\n",ops_get_proc(), dat->name, dim, sb->id_m[dim], they_send, actual_depth_recv, left_recv_depth);
      MPI_Abort(sb->comm, -1);
    }
  }

  // Compute size of packed data
  send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[2] + send_size > ops_buffer_send_2_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_send_2\n");
    ops_buffer_send_2 = (char *)ops_realloc_fast(ops_buffer_send_2,  send_recv_offsets[2],
                                        send_recv_offsets[2] + 4 * send_size);
    ops_buffer_send_2_size = send_recv_offsets[2] + 4 * send_size;
  }
  if (send_recv_offsets[3] + recv_size > ops_buffer_recv_2_size) {
    if (OPS_diags > 4)
      printf("Realloc ops_buffer_recv_2\n");
    ops_buffer_recv_2 = (char *)ops_realloc_fast(ops_buffer_recv_2, send_recv_offsets[3],
                                        send_recv_offsets[3] + 4 * recv_size);
    ops_buffer_recv_2_size = send_recv_offsets[3] + 4 * recv_size;
  }

  // pack data
  if (actual_depth_send > 0)
    ops_pack(dat, i3, ops_buffer_send_2 + send_recv_offsets[2],
             &sd->halos[MAX_DEPTH * dim + actual_depth_send]);

  // increase offset
  send_recv_offsets[2] += send_size;
  send_recv_offsets[3] += recv_size;

  // clear dirtybits
  for (int d = 0; d <= actual_depth_send; d++)
    sd->dirty_dir_send[2 * MAX_DEPTH * dim + MAX_DEPTH + d] = 0;
}


void ops_exchange_halo_unpacker(ops_dat dat, int d_pos, int d_neg,
                                int *iter_range, int dim,
                                int *send_recv_offsets) {
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_recv_depth = 0;
  int right_recv_depth = 0;
  int left_send_depth = 0;
  int right_send_depth = 0;
  int *prod = sd->prod;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++) {
    d_m[d] = dat->d_m[d] + sd->d_im[d];
    d_p[d] = dat->d_p[d] + sd->d_ip[d];
  }

  if (!ops_compute_intersections(dat, d_pos,d_neg, iter_range, dim,
          &left_send_depth, &left_recv_depth, &right_send_depth, &right_recv_depth)) return;

  //
  // negative direction
  //

  // decide actual depth based on dirtybits
  int actual_depth_recv = 0;
  for (int d = 0; d <= right_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d] == 1)
      actual_depth_recv = d;

  // set up initial pointers
  int i4 = (prod[dim] / prod[dim - 1] - (d_p[dim])) * prod[dim - 1];

  // Compute size of packed data
  int recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  // Unpack data
  if (actual_depth_recv > 0)
    ops_unpack(dat, i4, ops_buffer_recv_1 + send_recv_offsets[1],
               &sd->halos[MAX_DEPTH * dim + actual_depth_recv]);
  // increase offset
  send_recv_offsets[1] += recv_size;
  // clear dirtybits
  for (int d = 0; d <= actual_depth_recv; d++)
    sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d] = 0;

  //
  // similarly for positive direction
  //

  // decide actual depth based on dirtybits
  actual_depth_recv = 0;
  for (int d = 0; d <= left_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + d] == 1)
      actual_depth_recv = d;

  // set up initial pointers
  int i1 = (-d_m[dim] - actual_depth_recv) * prod[dim - 1];

  // Compute size of packed data
  recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  // Unpack data
  if (actual_depth_recv > 0)
    ops_unpack(dat, i1, ops_buffer_recv_2 + send_recv_offsets[3],
               &sd->halos[MAX_DEPTH * dim + actual_depth_recv]);
  // increase offset
  send_recv_offsets[3] += recv_size;
  // clear dirtybits
  for (int d = 0; d <= actual_depth_recv; d++)
    sd->dirty_dir_recv[2 * MAX_DEPTH * dim + d] = 0;
}


void ops_exchange_halo_unpacker_given(ops_dat dat, int *depths, int dim,
                              int *send_recv_offsets) {
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_send_depth = depths[0];
  int left_recv_depth = depths[1];
  int right_send_depth = depths[2];
  int right_recv_depth = depths[3];

  int *prod = sd->prod;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++) {
    d_m[d] = dat->d_m[d] + sd->d_im[d];
    d_p[d] = dat->d_p[d] + sd->d_ip[d];
  }
  //
  // negative direction
  //

  // decide actual depth based on dirtybits
  int actual_depth_recv = 0;
  for (int d = 0; d <= right_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d] == 1)
      actual_depth_recv = d;

  if (sb->id_p[dim] == MPI_PROC_NULL) {
    actual_depth_recv = 0;
  }

  // set up initial pointers
  int i4 = (prod[dim] / prod[dim - 1] - (d_p[dim])) * prod[dim - 1];

  // Compute size of packed data
  int recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  // Unpack data
  if (actual_depth_recv > 0)
    ops_unpack(dat, i4, ops_buffer_recv_1 + send_recv_offsets[1],
               &sd->halos[MAX_DEPTH * dim + actual_depth_recv]);
  // increase offset
  send_recv_offsets[1] += recv_size;
  // clear dirtybits
  for (int d = 0; d <= actual_depth_recv; d++)
    sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d] = 0;

  //
  // similarly for positive direction
  //

  // decide actual depth based on dirtybits
  actual_depth_recv = 0;
  for (int d = 0; d <= left_recv_depth; d++)
    if (sd->dirty_dir_recv[2 * MAX_DEPTH * dim + d] == 1)
      actual_depth_recv = d;

  if (sb->id_m[dim] == MPI_PROC_NULL) {
    actual_depth_recv = 0;
  }

  // set up initial pointers
  int i1 = (-d_m[dim] - actual_depth_recv) * prod[dim - 1];

  // Compute size of packed data
  recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  // Unpack data
  if (actual_depth_recv > 0)
    ops_unpack(dat, i1, ops_buffer_recv_2 + send_recv_offsets[3],
               &sd->halos[MAX_DEPTH * dim + actual_depth_recv]);
  // increase offset
  send_recv_offsets[3] += recv_size;
  // clear dirtybits
  for (int d = 0; d <= actual_depth_recv; d++)
    sd->dirty_dir_recv[2 * MAX_DEPTH * dim + d] = 0;

}

void ops_halo_exchanges(ops_arg *args, int nargs, int *range) {
  // double c1,c2,t1,t2;
  // printf("*************** range[i] %d %d %d %d\n",range[0],range[1],range[2],
  // range[3]);
  int send_recv_offsets[4]; //{send_1, recv_1, send_2, recv_2}, for the two
                            // directions, negative then positive
  MPI_Comm comm = MPI_COMM_NULL;

  for (int dim = 0; dim < OPS_MAX_DIM; dim++) {
    // ops_timers_core(&c1,&t1);
    int id_m = -1, id_p = -1;
    int other_dims = 1;
    for (int i = 0; i < 4; i++)
      send_recv_offsets[i] = 0;
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype != OPS_ARG_DAT ||
          !(args[i].acc == OPS_READ || args[i].acc == OPS_RW) ||
          args[i].opt == 0)
        continue;
      ops_dat dat = args[i].dat;
      int dat_ndim = OPS_sub_block_list[dat->block->index]->ndim;
      if (dat_ndim <= dim || dat->size[dim] <= 1)
        continue; // dimension of the sub-block is less than current dim OR has
                  // a size of 1 (edge dat)
      comm = OPS_sub_block_list[dat->block->index]
                 ->comm; // use communicator for this sub-block

      // check if there is an intersection of dependency range with my full
      // range
      // in *other* dimensions (i.e. any other dimension d2 ,but the current one
      // dim)
      for (int d2 = 0; d2 < dat_ndim; d2++) {
        if (dim != d2)
          other_dims =
              other_dims &&
              (dat->size[d2] == 1 ||
               intersection(range[2 * d2] - MAX_DEPTH,
                            range[2 * d2 + 1] + MAX_DEPTH,
                            OPS_sub_dat_list[dat->index]->decomp_disp[d2],
                            OPS_sub_dat_list[dat->index]->decomp_disp[d2] +
                                OPS_sub_dat_list[dat->index]
                                    ->decomp_size[d2])); // i.e. the
                                                         // intersection of the
                                                         // dependency range
                                                         // with my full range
      }
      if (other_dims == 0)
        break;
      id_m = OPS_sub_block_list[dat->block->index]
                 ->id_m[dim]; // neighbor in negative direction
      id_p = OPS_sub_block_list[dat->block->index]
                 ->id_p[dim]; // neighbor in possitive direction
      int d_pos = 0, d_neg = 0;
      for (int p = 0; p < args[i].stencil->points; p++) {
        d_pos = MAX(d_pos, args[i].stencil->stencil[dat_ndim * p + dim]);
        d_neg = MIN(d_neg, args[i].stencil->stencil[dat_ndim * p + dim]);
      }
      if (d_pos > 0 || d_neg < 0)
        ops_exchange_halo_packer(dat, d_pos, d_neg, range, dim,
                                 send_recv_offsets);
    }
    //  ops_timers_core(&c2,&t2);
    //  ops_gather_time += t2-t1;

    // early exit - if one of the args does not have an intersection in other
    // dims
    // then none of the args will have an intersection - as all dats (except
    // edge dats)
    // are defined on the whole domain
    if (other_dims == 0 || comm == MPI_COMM_NULL)
      continue;

    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, send_recv_offsets[0], MPI_BYTE,
              send_recv_offsets[0] > 0 ? id_m : MPI_PROC_NULL, dim, comm,
              &request[0]);
    MPI_Isend(ops_buffer_send_2, send_recv_offsets[2], MPI_BYTE,
              send_recv_offsets[2] > 0 ? id_p : MPI_PROC_NULL,
              OPS_MAX_DIM + dim, comm, &request[1]);
    MPI_Irecv(ops_buffer_recv_1, send_recv_offsets[1], MPI_BYTE,
              send_recv_offsets[1] > 0 ? id_p : MPI_PROC_NULL, dim, comm,
              &request[2]);
    MPI_Irecv(ops_buffer_recv_2, send_recv_offsets[3], MPI_BYTE,
              send_recv_offsets[3] > 0 ? id_m : MPI_PROC_NULL,
              OPS_MAX_DIM + dim, comm, &request[3]);

    MPI_Status status[4];
    MPI_Waitall(2, &request[2], &status[2]);

    //  ops_timers_core(&c1,&t1);
    //  ops_sendrecv_time += t1-t2;

    for (int i = 0; i < 4; i++)
      send_recv_offsets[i] = 0;
    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype != OPS_ARG_DAT ||
          !(args[i].acc == OPS_READ || args[i].acc == OPS_RW) ||
          args[i].opt == 0)
        continue;
      ops_dat dat = args[i].dat;
      int dat_ndim = OPS_sub_block_list[dat->block->index]->ndim;
      if (dat_ndim <= dim || dat->size[dim] <= 1)
        continue;
      int d_pos = 0, d_neg = 0;
      for (int p = 0; p < args[i].stencil->points; p++) {
        d_pos = MAX(d_pos, args[i].stencil->stencil[dat_ndim * p + dim]);
        d_neg = MIN(d_neg, args[i].stencil->stencil[dat_ndim * p + dim]);
      }
      if (d_pos > 0 || d_neg < 0)
        ops_exchange_halo_unpacker(dat, d_pos, d_neg, range, dim,
                                   send_recv_offsets);
    }

    MPI_Waitall(2, &request[0], &status[0]);
    //  ops_timers_core(&c2,&t2);
    //  ops_scatter_time += t2-t1;
  }
}

void ops_halo_exchanges_datlist(ops_dat *dats, int ndats, int *depths) {
  // double c1,c2,t1,t2;
  int send_recv_offsets[4]; //{send_1, recv_1, send_2, recv_2}, for the two
                            // directions, negative then positive
  MPI_Comm comm = MPI_COMM_NULL;

  for (int dim = 0; dim < OPS_MAX_DIM; dim++) {
    // ops_timers_core(&c1,&t1);
    int id_m = -1, id_p = -1;
    
    for (int i = 0; i < 4; i++)
      send_recv_offsets[i] = 0;
    for (int i = 0; i < ndats; i++) {
      ops_dat dat = dats[i];
      int dat_ndim = OPS_sub_block_list[dat->block->index]->ndim;
      if (dat_ndim <= dim || dat->size[dim] <= 1)
        continue; // dimension of the sub-block is less than current dim OR has
                  // a size of 1 (edge dat)
      comm = OPS_sub_block_list[dat->block->index]
                 ->comm; // use communicator for this sub-block

      id_m = OPS_sub_block_list[dat->block->index]
                 ->id_m[dim]; // neighbor in negative direction
      id_p = OPS_sub_block_list[dat->block->index]
                 ->id_p[dim]; // neighbor in possitive direction
      
      ops_exchange_halo_packer_given(dat, &depths[OPS_MAX_DIM*4*i + dim*4], dim,
                                 send_recv_offsets);
    }
    //  ops_timers_core(&c2,&t2);
    //  ops_gather_time += t2-t1;

    // early exit
    if (comm == MPI_COMM_NULL)
      continue;

    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, send_recv_offsets[0], MPI_BYTE,
              send_recv_offsets[0] > 0 ? id_m : MPI_PROC_NULL, dim, comm,
              &request[0]);
    MPI_Isend(ops_buffer_send_2, send_recv_offsets[2], MPI_BYTE,
              send_recv_offsets[2] > 0 ? id_p : MPI_PROC_NULL,
              OPS_MAX_DIM + dim, comm, &request[1]);
    MPI_Irecv(ops_buffer_recv_1, send_recv_offsets[1], MPI_BYTE,
              send_recv_offsets[1] > 0 ? id_p : MPI_PROC_NULL, dim, comm,
              &request[2]);
    MPI_Irecv(ops_buffer_recv_2, send_recv_offsets[3], MPI_BYTE,
              send_recv_offsets[3] > 0 ? id_m : MPI_PROC_NULL,
              OPS_MAX_DIM + dim, comm, &request[3]);

    MPI_Status status[4];
    MPI_Waitall(2, &request[2], &status[2]);

    //  ops_timers_core(&c1,&t1);
    //  ops_sendrecv_time += t1-t2;

    for (int i = 0; i < 4; i++)
      send_recv_offsets[i] = 0;
    for (int i = 0; i < ndats; i++) {
      ops_dat dat = dats[i];
      int dat_ndim = OPS_sub_block_list[dat->block->index]->ndim;
      if (dat_ndim <= dim || dat->size[dim] <= 1)
        continue;
      ops_exchange_halo_unpacker_given(dat, &depths[OPS_MAX_DIM*4*i + dim*4], dim,
                                   send_recv_offsets);
    }

    MPI_Waitall(2, &request[0], &status[0]);
    //  ops_timers_core(&c2,&t2);
    //  ops_scatter_time += t2-t1;
  }
}

void ops_mpi_reduce_double(ops_arg *arg, double *data) {
  (void)data;
  // if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
  double result[arg->dim * ops_comm_global_size];

  if (arg->acc == OPS_INC) // global reduction
    MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_SUM,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_MAX) // global maximum
    MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_MAX,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_MIN) // global minimum
    MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_MIN,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_WRITE) { // any
    MPI_Allgather((double *)arg->data, arg->dim, MPI_DOUBLE, result, arg->dim,
                  MPI_DOUBLE, OPS_MPI_GLOBAL);
    for (int i = 1; i < ops_comm_global_size; i++) {
      for (int j = 0; j < arg->dim; j++) {
        if (result[i * arg->dim + j] != 0.0)
          result[j] = result[i * arg->dim + j];
      }
    }
  }
  memcpy(arg->data, result, sizeof(double) * arg->dim);
  //}
}

void ops_mpi_reduce_float(ops_arg *arg, float *data) {
  (void)data;

  // if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
  float result[arg->dim * ops_comm_global_size];

  if (arg->acc == OPS_INC) // global reduction
    MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_SUM,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_MAX) // global maximum
    MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_MAX,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_MIN) // global minimum
    MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_MIN,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_WRITE) { // any
    MPI_Allgather((float *)arg->data, arg->dim, MPI_FLOAT, result, arg->dim,
                  MPI_FLOAT, OPS_MPI_GLOBAL);
    for (int i = 1; i < ops_comm_global_size; i++) {
      for (int j = 0; j < arg->dim; j++) {
        if (result[i * arg->dim + j] != 0.0f)
          result[j] = result[i * arg->dim + j];
      }
    }
  }
  memcpy(arg->data, result, sizeof(float) * arg->dim);
  //}
}

void ops_mpi_reduce_int(ops_arg *arg, int *data) {
  (void)data;

  // if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
  int result[arg->dim * ops_comm_global_size];

  if (arg->acc == OPS_INC) // global reduction
    MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT, MPI_SUM,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_MAX) // global maximum
    MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT, MPI_MAX,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_MIN) // global minimum
    MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT, MPI_MIN,
                  OPS_MPI_GLOBAL);
  else if (arg->acc == OPS_WRITE) { // any
    MPI_Allgather((int *)arg->data, arg->dim, MPI_INT, result, arg->dim,
                  MPI_INT, OPS_MPI_GLOBAL);
    for (int i = 1; i < ops_comm_global_size; i++) {
      for (int j = 0; j < arg->dim; j++) {
        if (result[i * arg->dim + j] != 0)
          result[j] = result[i * arg->dim + j];
      }
    }
  }
  memcpy(arg->data, result, sizeof(int) * arg->dim);
  //}
}

void ops_execute_reduction(ops_reduction handle) {
  char local[handle->size];
  memcpy(local, handle->data, handle->size);
  if (strcmp(handle->type, "int") == 0 || strcmp(handle->type, "int(4)") == 0 ||
      strcmp(handle->type, "integer(4)") == 0 ||
      strcmp(handle->type, "integer") == 0) {
    for (int i = 1; i < OPS_block_index; i++) {
      if (!OPS_sub_block_list[i]->owned)
        continue;
      if (handle->acc == OPS_MAX)
        for (int d = 0; d < handle->size / sizeof(int); d++)
          ((int *)local)[d] = MAX(
              ((int *)local)[d], ((int *)(handle->data + i * handle->size))[d]);
      if (handle->acc == OPS_MIN)
        for (int d = 0; d < handle->size / sizeof(int); d++)
          ((int *)local)[d] = MIN(
              ((int *)local)[d], ((int *)(handle->data + i * handle->size))[d]);
      if (handle->acc == OPS_INC)
        for (int d = 0; d < handle->size / sizeof(int); d++)
          ((int *)local)[d] += ((int *)(handle->data + i * handle->size))[d];
      if (handle->acc == OPS_WRITE)
        for (int d = 0; d < handle->size / sizeof(int); d++)
          ((int *)local)[d] =
              ((int *)(handle->data + i * handle->size))[d] != 0
                  ? ((int *)(handle->data + i * handle->size))[d]
                  : ((int *)local)[d];
    }
    ops_arg arg;
    arg.data = local;
    arg.acc = handle->acc;
    arg.dim = handle->size / sizeof(int);
    ops_mpi_reduce_int(&arg, (int *)local);
  }
  if (strcmp(handle->type, "float") == 0 || strcmp(handle->type, "real") == 0) {
    for (int i = 1; i < OPS_block_index; i++) {
      if (!OPS_sub_block_list[i]->owned)
        continue;
      if (handle->acc == OPS_MAX)
        for (int d = 0; d < handle->size / sizeof(float); d++)
          ((float *)local)[d] =
              MAX(((float *)local)[d],
                  ((float *)(handle->data + i * handle->size))[d]);
      if (handle->acc == OPS_MIN)
        for (int d = 0; d < handle->size / sizeof(float); d++)
          ((float *)local)[d] =
              MIN(((float *)local)[d],
                  ((float *)(handle->data + i * handle->size))[d]);
      if (handle->acc == OPS_INC)
        for (int d = 0; d < handle->size / sizeof(float); d++)
          ((float *)local)[d] +=
              ((float *)(handle->data + i * handle->size))[d];
      if (handle->acc == OPS_WRITE)
        for (int d = 0; d < handle->size / sizeof(float); d++)
          ((float *)local)[d] =
              ((float *)(handle->data + i * handle->size))[d] != 0.0f
                  ? ((float *)(handle->data + i * handle->size))[d]
                  : ((float *)local)[d];
    }
    ops_arg arg;
    arg.data = local;
    arg.acc = handle->acc;
    arg.dim = handle->size / sizeof(float);
    ops_mpi_reduce_float(&arg, (float *)local);
  }
  if (strcmp(handle->type, "double") == 0 ||
      strcmp(handle->type, "real(8)") == 0 ||
      strcmp(handle->type, "double precision") == 0) {
    for (int i = 1; i < OPS_block_index; i++) {
      if (!OPS_sub_block_list[i]->owned)
        continue;
      if (handle->acc == OPS_MAX)
        for (int d = 0; d < handle->size / sizeof(double); d++)
          ((double *)local)[d] =
              MAX(((double *)local)[d],
                  ((double *)(handle->data + i * handle->size))[d]);
      if (handle->acc == OPS_MIN)
        for (int d = 0; d < handle->size / sizeof(double); d++)
          ((double *)local)[d] =
              MIN(((double *)local)[d],
                  ((double *)(handle->data + i * handle->size))[d]);
      if (handle->acc == OPS_INC)
        for (int d = 0; d < handle->size / sizeof(double); d++)
          ((double *)local)[d] +=
              ((double *)(handle->data + i * handle->size))[d];
      if (handle->acc == OPS_WRITE)
        for (int d = 0; d < handle->size / sizeof(double); d++)
          ((double *)local)[d] =
              ((double *)(handle->data + i * handle->size))[d] != 0.0
                  ? ((double *)(handle->data + i * handle->size))[d]
                  : ((double *)local)[d];
    }
    ops_arg arg;
    arg.data = local;
    arg.acc = handle->acc;
    arg.dim = handle->size / sizeof(double);
    ops_mpi_reduce_double(&arg, (double *)local);
  }
  memcpy(handle->data, local, handle->size);
}

void ops_set_halo_dirtybit(ops_arg *arg) {
  if (arg->opt == 0)
    return;
  sub_dat_list sd = OPS_sub_dat_list[arg->dat->index];
  sd->dirtybit = 1;
  for (int i = 0; i < 2 * arg->dat->block->dims * MAX_DEPTH; i++)
    sd->dirty_dir_send[i] = 1;
  for (int i = 0; i < 2 * arg->dat->block->dims * MAX_DEPTH; i++)
    sd->dirty_dir_recv[i] = 1;
}

void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range) {
  // TODO: account for base
  if (arg->opt == 0)
    return;
  ops_dat dat = arg->dat;
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_boundary_modified[OPS_MAX_DIM] = {0};
  int left_halo_modified[OPS_MAX_DIM] = {0};
  int right_boundary_modified[OPS_MAX_DIM] = {0};
  int right_halo_modified[OPS_MAX_DIM] = {0};

  int range_intersect[OPS_MAX_DIM] = {0};

  int ndim = sb->ndim;

  for (int dim = 0; dim < ndim; dim++) {
    range_intersect[dim] = intersection(
        iter_range[2 * dim], iter_range[2 * dim + 1], sd->decomp_disp[dim],
        (sd->decomp_disp[dim] + sd->decomp_size[dim])); // i.e. the intersection
                                                        // of the execution
                                                        // range with my full
                                                        // range

    left_boundary_modified[dim] = intersection(
        iter_range[2 * dim], iter_range[2 * dim + 1], sd->decomp_disp[dim],
        sd->decomp_disp[dim] + MAX_DEPTH - 1); // i.e. the intersection of the
                                               // execution range with my left
                                               // boundary
    right_halo_modified[dim] =
        intersection(iter_range[2 * dim], iter_range[2 * dim + 1],
                     (sd->decomp_disp[dim] + sd->decomp_size[dim]),
                     (sd->decomp_disp[dim] + sd->decomp_size[dim]) + MAX_DEPTH -
                         1); // i.e. the intersection of the execution range
                             // with the my right neighbour's boundary
    right_boundary_modified[dim] = intersection(
        iter_range[2 * dim], iter_range[2 * dim + 1],
        (sd->decomp_disp[dim] + sd->decomp_size[dim]) - MAX_DEPTH + 1,
        (sd->decomp_disp[dim] + sd->decomp_size[dim]));
    left_halo_modified[dim] = intersection(
        iter_range[2 * dim], iter_range[2 * dim + 1],
        sd->decomp_disp[dim] - MAX_DEPTH + 1, sd->decomp_disp[dim]);
  }

  sd->dirtybit = 1;
  for (int dim = 0; dim < ndim; dim++) {
    int other_dims = 1;
    for (int d2 = 0; d2 < ndim; d2++)
      if (d2 != dim)
        other_dims =
            other_dims && (range_intersect[d2] > 0 || dat->size[d2] == 1);

    if (left_boundary_modified[dim] > 0 && other_dims) {
      int beg = 1 + (iter_range[2 * dim] >= sd->decomp_disp[dim]
                         ? iter_range[2 * dim] - sd->decomp_disp[dim]
                         : 0);
      for (int d2 = beg; d2 < beg + left_boundary_modified[dim];
           d2++) { // we shifted dirtybits, [1] is the first layer not the
                   // second
        sd->dirty_dir_send[2 * MAX_DEPTH * dim + d2] = 1;
      }
    }
    if (left_halo_modified[dim] > 0 && other_dims) {
      int beg =
          iter_range[2 * dim] >= sd->decomp_disp[dim] - MAX_DEPTH + 1
              ? iter_range[2 * dim] - (sd->decomp_disp[dim] - MAX_DEPTH + 1)
              : 0;
      for (int d2 = beg; d2 < beg + left_halo_modified[dim]; d2++) {
        sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH - d2 - 1] = 1;
      }
    }
    if (right_boundary_modified[dim] > 0 && other_dims) {
      int beg =
          iter_range[2 * dim] >=
                  (sd->decomp_disp[dim] + sd->decomp_size[dim]) - MAX_DEPTH + 1
              ? iter_range[2 * dim] -
                    ((sd->decomp_disp[dim] + sd->decomp_size[dim]) - MAX_DEPTH +
                     1)
              : 0;
      for (int d2 = beg; d2 < beg + right_boundary_modified[dim]; d2++) {
        sd->dirty_dir_send[2 * MAX_DEPTH * dim + 2 * MAX_DEPTH - d2 - 1] = 1;
      }
    }
    if (right_halo_modified[dim] > 0 && other_dims) {
      int beg = 1 + (iter_range[2 * dim] >=
                             (sd->decomp_disp[dim] + sd->decomp_size[dim])
                         ? iter_range[2 * dim] -
                               (sd->decomp_disp[dim] + sd->decomp_size[dim])
                         : 0);
      for (int d2 = beg; d2 < beg + right_halo_modified[dim]; d2++) {
        sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d2] = 1;
      }
    }
  }
}

void ops_halo_transfer(ops_halo_group group) {
  ops_execute();
  ops_mpi_halo_group *mpi_group = &OPS_mpi_halo_group_list[group->index];
  if (mpi_group->nhalos == 0)
    return;

  // Reset offset counters
  mpi_neigh_size[0] = 0;
  for (int i = 1; i < mpi_group->num_neighbors_send; i++)
    mpi_neigh_size[i] = mpi_neigh_size[i - 1] + mpi_group->send_sizes[i - 1];

  // Loop over all the halos we own in the group
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_halo *halo = mpi_group->mpi_halos[h];
    sub_dat *sd = OPS_sub_dat_list[halo->halo->from->index];

    // Loop over all the send fragments and pack into buffer
    for (int f = 0; f < halo->nproc_from; f++) {
      int ranges[OPS_MAX_DIM * 2];
      int step[OPS_MAX_DIM];
      int buf_strides[OPS_MAX_DIM];
      int fragment_size = halo->halo->from->elem_size;
      for (int i = 0; i < OPS_MAX_DIM; i++) {
        if (halo->halo->from_dir[i] > 0) {
          ranges[2 * i] =
              halo->local_from_base[f * OPS_MAX_DIM + i] -
              sd->d_im[i]; // Need to account for intra-block halo padding
          ranges[2 * i + 1] =
              ranges[2 * i] +
              halo->local_iter_size[f * OPS_MAX_DIM +
                                    abs(halo->halo->from_dir[i]) - 1];
          step[i] = 1;
        } else {
          ranges[2 * i + 1] =
              halo->local_from_base[f * OPS_MAX_DIM + i] - 1 - sd->d_im[i];
          ranges[2 * i] =
              ranges[2 * i + 1] +
              halo->local_iter_size[f * OPS_MAX_DIM +
                                    abs(halo->halo->from_dir[i]) - 1];
          step[i] = -1;
        }
        buf_strides[i] = 1;
        for (int j = 0; j != abs(halo->halo->from_dir[i]) - 1; j++)
          buf_strides[i] *= halo->local_iter_size[f * OPS_MAX_DIM + j];
        fragment_size *= halo->local_iter_size[f * OPS_MAX_DIM + i];
      }
      int process = halo->proclist[f];
      int proc_grp_idx = 0;
      while (process != mpi_group->neighbors_send[proc_grp_idx])
        proc_grp_idx++;
      ops_halo_copy_tobuf(ops_buffer_send_1, mpi_neigh_size[proc_grp_idx],
                          halo->halo->from, ranges[0], ranges[1], ranges[2],
                          ranges[3], ranges[4], ranges[5], step[0], step[1],
                          step[2], buf_strides[0], buf_strides[1],
                          buf_strides[2]);
      mpi_neigh_size[proc_grp_idx] += fragment_size;
    }
  }

  mpi_neigh_size[0] = 0;
  for (int i = 1; i < mpi_group->num_neighbors_send; i++)
    mpi_neigh_size[i] = mpi_neigh_size[i - 1] + mpi_group->send_sizes[i - 1];
  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    MPI_Isend(&ops_buffer_send_1[mpi_neigh_size[i]], mpi_group->send_sizes[i],
              MPI_BYTE, mpi_group->neighbors_send[i], 100 + mpi_group->index,
              OPS_MPI_GLOBAL, &mpi_group->requests[i]);

  mpi_neigh_size[0] = 0;
  for (int i = 1; i < mpi_group->num_neighbors_recv; i++)
    mpi_neigh_size[i] = mpi_neigh_size[i - 1] + mpi_group->recv_sizes[i - 1];
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    MPI_Irecv(&ops_buffer_recv_1[mpi_neigh_size[i]], mpi_group->recv_sizes[i],
              MPI_BYTE, mpi_group->neighbors_recv[i], 100 + mpi_group->index,
              OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  // Loop over all the halos we own in the group
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_halo *halo = mpi_group->mpi_halos[h];
    sub_dat *sd = OPS_sub_dat_list[halo->halo->to->index];

    // Loop over all the recv fragments and pack into buffer
    for (int f = halo->nproc_from; f < halo->nproc_from + halo->nproc_to; f++) {
      int ranges[OPS_MAX_DIM * 2];
      int step[OPS_MAX_DIM];
      int buf_strides[OPS_MAX_DIM];
      int fragment_size = halo->halo->to->elem_size;
      for (int i = 0; i < OPS_MAX_DIM; i++) {
        if (halo->halo->to_dir[i] > 0) {
          ranges[2 * i] =
              halo->local_to_base[f * OPS_MAX_DIM + i] - sd->d_im[i];
          ranges[2 * i + 1] =
              ranges[2 * i] +
              halo->local_iter_size[f * OPS_MAX_DIM +
                                    abs(halo->halo->to_dir[i]) - 1];
          step[i] = 1;
        } else {
          ranges[2 * i + 1] =
              halo->local_to_base[f * OPS_MAX_DIM + i] - 1 - sd->d_im[i];
          ranges[2 * i] = ranges[2 * i + 1] +
                          halo->local_iter_size[f * OPS_MAX_DIM +
                                                abs(halo->halo->to_dir[i]) - 1];
          step[i] = -1;
        }
        buf_strides[i] = 1;
        for (int j = 0; j != abs(halo->halo->to_dir[i]) - 1; j++)
          buf_strides[i] *= halo->local_iter_size[f * OPS_MAX_DIM + j];
        fragment_size *= halo->local_iter_size[f * OPS_MAX_DIM + i];
      }
      int process = halo->proclist[f];
      int proc_grp_idx = 0;
      while (process != mpi_group->neighbors_recv[proc_grp_idx])
        proc_grp_idx++;
      ops_halo_copy_frombuf(halo->halo->to, ops_buffer_recv_1,
                            mpi_neigh_size[proc_grp_idx], ranges[0], ranges[1],
                            ranges[2], ranges[3], ranges[4], ranges[5], step[0],
                            step[1], step[2], buf_strides[0], buf_strides[1],
                            buf_strides[2]);
      mpi_neigh_size[proc_grp_idx] += fragment_size;
    }
  }
  MPI_Waitall(mpi_group->num_neighbors_send, &mpi_group->requests[0],
              &mpi_group->statuses[0]);
}

void ops_force_halo_exchange(ops_dat dat, ops_stencil stencil) {
  ops_arg arg = ops_arg_dat(dat, dat->dim, stencil, dat->type, OPS_READ);
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int range[2*OPS_MAX_DIM];
  for (int i = 0; i < dat->block->dims; i++) {
    range[2*i] = sd->gbl_base[i];
    range[2*i+1] = sd->gbl_size[i];
  }
  ops_halo_exchanges(&arg, 1, range);
}

int ops_dat_get_local_npartitions(ops_dat dat) {
  return 1;
}

void ops_dat_get_local_extents(ops_dat dat, int part, int *sizes) {
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  for (int d = 0; d < dat->block->dims; d++)
    sizes[d] = dat->size[d] + dat->d_m[d] + sd->d_im[d] - dat->d_p[d] - sd->d_ip[d];
}

char* ops_dat_get_raw_pointer(ops_dat dat, int part, ops_stencil stencil, int *stride) {
  ops_get_data(dat);
  ops_force_halo_exchange(dat, stencil);
  if (stride != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      stride[d] = dat->size[d];
  return dat->data + dat->base_offset;
}

void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc) {
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  if (acc != OPS_READ)
  dat->dirty_hd = 1; 
  sd->dirtybit = 1;
  for (int i = 0; i < 2 * dat->block->dims * MAX_DEPTH; i++) {
    sd->dirty_dir_send[i] = 1;
    sd->dirty_dir_recv[i] = 1;
  }
}

void ops_dat_fetch_data(ops_dat dat, int part, char *data) {
  ops_get_data(dat);
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int lsize[OPS_MAX_DIM] = {0};
  ops_dat_get_local_extents(dat, part, lsize);
  lsize[0] *= dat->elem_size/dat->dim; //now in bytes
  if (dat->block->dims>3) {ops_printf("Error, ops_dat_fetch_data not implemented for dims>3\n"); exit(-1);}
  if (OPS_soa && dat->dim > 1) {ops_printf("Error, ops_dat_fetch_data not implemented for SoA\n"); exit(-1);}

  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      memcpy(&data[k*lsize[0]*lsize[1]+j*lsize[0]],
             &dat->data[((j-dat->d_m[1]-sd->d_im[1] + (k-dat->d_m[2]-sd->d_im[2])*dat->size[1])*dat->size[0] - dat->d_m[0] - sd->d_im[0])* dat->elem_size],
             lsize[0]);
}

void ops_dat_set_data(ops_dat dat, int part, char *data) {
  int lsize[OPS_MAX_DIM] = {0};
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  ops_dat_get_local_extents(dat, part, lsize);
  lsize[0] *= dat->elem_size/dat->dim; //now in bytes
  if (dat->block->dims>3) {ops_printf("Error, ops_dat_set_data not implemented for dims>3\n"); exit(-1);}
  if (OPS_soa && dat->dim > 1) {ops_printf("Error, ops_dat_set_data not implemented for SoA\n"); exit(-1);}

  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      memcpy(&dat->data[((j-dat->d_m[1]-sd->d_im[1] + (k-dat->d_m[2]-sd->d_im[2])*dat->size[1])*dat->size[0] - dat->d_m[0] - sd->d_im[0])* dat->elem_size],
             &data[k*lsize[0]*lsize[1]+j*lsize[0]],
             lsize[0]);

  dat->dirty_hd = 1;
  sd->dirtybit = 1;
  for (int i = 0; i < 2 * dat->block->dims * MAX_DEPTH; i++) {
    sd->dirty_dir_send[i] = 1;
    sd->dirty_dir_recv[i] = 1;
  }
}

int ops_dat_get_global_npartitions(ops_dat dat) {
  //TODO: lower-rank datasets?
  int nranks = 0;
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  if (sb->owned)
    MPI_Comm_size(sb->comm, &nranks);

  return nranks;
}

void ops_dat_get_global_extents(ops_dat dat, int part, int *disp, int *size) {
  //TODO: part?? local or global?
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  for (int d = 0; d < dat->block->dims; d++) {
    disp[d] = sd->decomp_disp[d] - dat->d_m[d];
    size[d] = sd->decomp_size[d] + dat->d_m[d] - dat->d_p[d];
  }
}

