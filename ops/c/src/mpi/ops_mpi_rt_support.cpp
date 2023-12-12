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
  * @brief OPS mpi run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi backend
  */
#include <vector>
#include <ops_lib_core.h>
#include "ops_util.h"
#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_exceptions.h>
#include <cassert>

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

  size_t *prod = sd->prod;

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
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: trying to exchange a " << actual_depth_recv << "-deep halo for " << dat->name << ", but halo is only " << abs(d_m[dim]) << " deep. Please set d_m and d_p accordingly";
    throw ex;
  }
  if (actual_depth_send > sd->decomp_size[dim]) {
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: overpartitioning! Trying to exchange a " << actual_depth_send << "-deep halo for " << dat->name << ", but dataset is only " << sd->decomp_size[dim] << " wide on this process.";
    throw ex;
  }

  // set up initial pointers
  int i2 = (-d_m[dim]) * prod[dim - 1];
  // int i4 = (prod[dim]/prod[dim-1] - (d_p[dim])    ) * prod[dim-1];
  // printf("block %s, dat %s, prod[dim-1] %d, prod[dim]
  // %d\n",dat->block->name,dat->name, prod[dim-1],prod[dim]);

  if (OPS_instance::getOPSInstance()->OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_m[dim], 665, &they_send,
                 1, MPI_INT, sb->id_p[dim], 665, sb->comm, &status);
    if (sb->id_p[dim] >= 0 && actual_depth_recv != they_send) {
      throw OPSException(OPS_INTERNAL_ERROR, "Error: Right recv mismatch");
    }
  }

  // Compute size of packed data
  int send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  int recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[0] + send_size > ops_buffer_send_1_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_send_1\n");
    ops_buffer_send_1 = (char *)OPS_realloc_fast(ops_buffer_send_1, send_recv_offsets[0],
                                        send_recv_offsets[0] + 4 * send_size);
    ops_buffer_send_1_size = send_recv_offsets[0] + 4 * send_size;
  }
  if (send_recv_offsets[1] + recv_size > ops_buffer_recv_1_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_recv_1\n");
    ops_buffer_recv_1 = (char *)OPS_realloc_fast(ops_buffer_recv_1, send_recv_offsets[1],
                                        send_recv_offsets[1] + 4 * recv_size);
    ops_buffer_recv_1_size = send_recv_offsets[1] + 4 * recv_size;
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
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: trying to exchange a " << actual_depth_recv << "-deep halo for " << dat->name << ", but halo is only " << d_p[dim] << " deep. Please set d_m and d_p accordingly";
    throw ex;
  }
  if (actual_depth_send > sd->decomp_size[dim]) {
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: overpartitioning! Trying to exchange a " << actual_depth_send << "-deep halo for " << dat->name << ", but dataset is only " << sd->decomp_size[dim] << " wide on this process.";
    throw ex;
  }


  // set up initial pointers
  // int i1 = (-d_m[dim] - actual_depth_recv) * prod[dim-1];
  int i3 = (prod[dim] / prod[dim - 1] - (d_p[dim]) - actual_depth_send) *
           prod[dim - 1];

  if (OPS_instance::getOPSInstance()->OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_p[dim], 666, &they_send,
                 1, MPI_INT, sb->id_m[dim], 666, sb->comm, &status);
    if (sb->id_m[dim] >= 0 && actual_depth_recv != they_send) {
      printf("Name: %s actual_depth_recv = %d, they_send = %d\n", dat->name, actual_depth_recv, they_send);
      throw OPSException(OPS_INTERNAL_ERROR, "Error: Right recv mismatch");
    }
  }

  // Compute size of packed data
  send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[2] + send_size > ops_buffer_send_2_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_send_2\n");
    ops_buffer_send_2 = (char *)OPS_realloc_fast(ops_buffer_send_2,  send_recv_offsets[2],
                                        send_recv_offsets[2] + 4 * send_size);
    ops_buffer_send_2_size = send_recv_offsets[2] + 4 * send_size;
  }
  if (send_recv_offsets[3] + recv_size > ops_buffer_recv_2_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_recv_2\n");
    ops_buffer_recv_2 = (char *)OPS_realloc_fast(ops_buffer_recv_2,  send_recv_offsets[3],
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
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: trying to exchange a " << left_recv_depth << "-deep halo for " << dat->name << ", but halo is only " << -sd->d_im[dim] << " deep. Please set OPS_TILING_MAXDEPTH accordingly";
    throw ex;
  }

  if (sb->id_p[dim] != MPI_PROC_NULL && sd->d_ip[dim] < right_recv_depth) {
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: trying to exchange a " << right_recv_depth << "-deep halo for " << dat->name << ", but halo is only " << sd->d_ip[dim] << " deep. Please set OPS_TILING_MAXDEPTH accordingly";
    throw ex;
  }

  if (left_send_depth > sd->decomp_size[dim]) {
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: overpartitioning! Trying to exchange a " << left_send_depth << "-deep halo for " << dat->name << ", but dataset is only " << sd->decomp_size[dim] << " wide on this process.";
    throw ex;
  }
  if (right_send_depth > sd->decomp_size[dim]) {
    OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
    ex << "Error: overpartitioning! Trying to exchange a " << right_send_depth << "-deep halo for " << dat->name << ", but dataset is only " << sd->decomp_size[dim] << " wide on this process.";
    throw ex;
  }

  size_t *prod = sd->prod;

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

  if (OPS_instance::getOPSInstance()->OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_m[dim], 665, &they_send,
                 1, MPI_INT, sb->id_p[dim], 665, sb->comm, &status);
    if (sb->id_p[dim] >= 0 && actual_depth_recv != they_send) {
      printf("Name: %s actual_depth_recv = %d, they_send = %d\n", dat->name, actual_depth_recv, they_send);
      throw OPSException(OPS_INTERNAL_ERROR, "Error: Right recv mismatch");
    }
  }

  // Compute size of packed data
  int send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  int recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
                  sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[0] + send_size > ops_buffer_send_1_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_send_1\n");
    ops_buffer_send_1 = (char *)OPS_realloc_fast(ops_buffer_send_1,  send_recv_offsets[0],
                                        send_recv_offsets[0] + 4 * send_size);
    ops_buffer_send_1_size = send_recv_offsets[0] + 4 * send_size;
  }
  if (send_recv_offsets[1] + recv_size > ops_buffer_recv_1_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_recv_1\n");
    ops_buffer_recv_1 = (char *)OPS_realloc_fast(ops_buffer_recv_1, send_recv_offsets[1],
                                        send_recv_offsets[1] + 4 * recv_size);
    ops_buffer_recv_1_size = send_recv_offsets[1] + 4 * recv_size;
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

  if (OPS_instance::getOPSInstance()->OPS_diags > 5) { // Consistency checking
    int they_send;
    MPI_Status status;
    MPI_Sendrecv(&actual_depth_send, 1, MPI_INT, sb->id_p[dim], 666, &they_send,
                 1, MPI_INT, sb->id_m[dim], 666, sb->comm, &status);
    if (sb->id_m[dim] != MPI_PROC_NULL && actual_depth_recv != they_send) {
      throw OPSException(OPS_INTERNAL_ERROR, "Error: Left recv mismatch");
    }
  }

  // Compute size of packed data
  send_size = sd->halos[MAX_DEPTH * dim + actual_depth_send].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_send].count * dat->dim;
  recv_size = sd->halos[MAX_DEPTH * dim + actual_depth_recv].blocklength *
              sd->halos[MAX_DEPTH * dim + actual_depth_recv].count * dat->dim;

  if (send_recv_offsets[2] + send_size > ops_buffer_send_2_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_send_2\n");
    ops_buffer_send_2 = (char *)OPS_realloc_fast(ops_buffer_send_2,  send_recv_offsets[2],
                                        send_recv_offsets[2] + 4 * send_size);
    ops_buffer_send_2_size = send_recv_offsets[2] + 4 * send_size;
  }
  if (send_recv_offsets[3] + recv_size > ops_buffer_recv_2_size) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 4)
      printf("Realloc ops_buffer_recv_2\n");
    ops_buffer_recv_2 = (char *)OPS_realloc_fast(ops_buffer_recv_2, send_recv_offsets[3],
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
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_recv_depth = 0;
  int right_recv_depth = 0;
  int left_send_depth = 0;
  int right_send_depth = 0;
  size_t *prod = sd->prod;

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
  int left_recv_depth = depths[1];
  int right_recv_depth = depths[3];

  size_t *prod = sd->prod;

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

void ops_halo_exchanges(ops_arg* args, int nargs, int *range_in) {
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
          (args[i].acc == OPS_WRITE || args[i].acc == OPS_MAX || args[i].acc == OPS_MIN) ||
          args[i].opt == 0)
        continue;

      if(dim >= args[i].stencil->dims)
        continue;

      ops_dat dat = args[i].dat;
      int dat_ndim = OPS_sub_block_list[dat->block->index]->ndim;
      if (args[i].argtype == OPS_ARG_DAT &&
          (args[i].acc == OPS_READ || args[i].acc == OPS_RW || args[i].acc == OPS_INC) &&
          args[i].stencil->points == 1 &&
          args[i].stencil->stencil[dim] == 0)
        continue;

      if (dat_ndim <= dim || dat->size[dim] <= 1)
        continue; // dimension of the sub-block is less than current dim OR has
                  // a size of 1 (edge dat)
      comm = OPS_sub_block_list[dat->block->index]
                 ->comm; // use communicator for this sub-block

      int range[2*OPS_MAX_DIM];
      for (int d2 = 0; d2 < dat_ndim; d2++) {
        if (args[i].stencil->type ==1) {
          range[2*d2+0] = range_in[2*d2+0]/args[i].stencil->mgrid_stride[d2];
          range[2*d2+1] = (range_in[2*d2+1]-1)/args[i].stencil->mgrid_stride[d2]+1;
        } else if (args[i].stencil->type ==2) {
          range[2*d2+0] = range_in[2*d2+0]*args[i].stencil->mgrid_stride[d2];
          range[2*d2+1] = range_in[2*d2+1]*args[i].stencil->mgrid_stride[d2];
        } else {
          range[2*d2+0] = range_in[2*d2+0];
          range[2*d2+1] = range_in[2*d2+1];
        }
      }
      //check if there is an intersection of dependency range with my full range
      //in *other* dimensions (i.e. any other dimension d2 ,but the current one dim)
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
                 ->id_p[dim]; // neighbor in positive direction
      int d_pos = 0, d_neg = 0;
      for (int p = 0; p < args[i].stencil->points; p++) {
        d_pos = MAX(d_pos, args[i].stencil->stencil[dat_ndim * p + dim]);
        d_neg = MIN(d_neg, args[i].stencil->stencil[dat_ndim * p + dim]);
      }

      if (args[i].stencil->type == 1) d_neg--;

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

      int range[2*OPS_MAX_DIM];
      for (int d2 = 0; d2 < dat_ndim; d2++) {
        if (args[i].stencil->type ==1) {
          range[2*d2+0] = range_in[2*d2+0]/args[i].stencil->mgrid_stride[d2];
          range[2*d2+1] = (range_in[2*d2+1]-1)/args[i].stencil->mgrid_stride[d2]+1;
        } else if (args[i].stencil->type ==2) {
          range[2*d2+0] = range_in[2*d2+0]*args[i].stencil->mgrid_stride[d2];
          range[2*d2+1] = range_in[2*d2+1]*args[i].stencil->mgrid_stride[d2];
        } else {
          range[2*d2+0] = range_in[2*d2+0];
          range[2*d2+1] = range_in[2*d2+1];
        }
      }

      int d_pos=0,d_neg=0;
      for (int p = 0; p < args[i].stencil->points; p++) {
        d_pos = MAX(d_pos, args[i].stencil->stencil[dat_ndim * p + dim]);
        d_neg = MIN(d_neg, args[i].stencil->stencil[dat_ndim * p + dim]);
      }
      if (args[i].stencil->type == 1) d_neg--;
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
                 ->id_p[dim]; // neighbor in positive direction

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
    //  printf("1 %g %d\n", t1-t2, send_recv_offsets[0] + send_recv_offsets[2]);
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
    //  printf("2 %g %d\n", t2-t1, send_recv_offsets[3] + send_recv_offsets[1]);
  }
}

#define ops_reduce_gen(type, mpi_type, zero) \
void ops_mpi_reduce_##type (ops_arg *arg, type *data) { \
  std::vector< type > result(arg->dim * ops_comm_global_size); \
\
  if (arg->acc == OPS_INC) \
    MPI_Allreduce((type *)arg->data, result.data(), arg->dim, mpi_type, MPI_SUM, \
                  OPS_MPI_GLOBAL); \
  else if (arg->acc == OPS_MAX) \
    MPI_Allreduce((type *)arg->data, result.data(), arg->dim, mpi_type, MPI_MAX, \
                  OPS_MPI_GLOBAL); \
  else if (arg->acc == OPS_MIN) \
    MPI_Allreduce((type *)arg->data, result.data(), arg->dim, mpi_type, MPI_MIN, \
                  OPS_MPI_GLOBAL); \
  else if (arg->acc == OPS_WRITE) { \
    MPI_Allgather((type *)arg->data, arg->dim, mpi_type, result.data(), arg->dim, \
                  MPI_DOUBLE, OPS_MPI_GLOBAL); \
    for (int i = 1; i < ops_comm_global_size; i++) { \
      for (int j = 0; j < arg->dim; j++) { \
        if (result[i * arg->dim + j] != zero) \
          result[j] = result[i * arg->dim + j]; \
      } \
    } \
  } \
  memcpy(arg->data, result.data(), sizeof(type) * arg->dim); \
}

ops_reduce_gen(double, MPI_DOUBLE, 0.0)
ops_reduce_gen(float, MPI_FLOAT, 0.0f)
ops_reduce_gen(int, MPI_INT, 0)
ops_reduce_gen(char, MPI_CHAR, 0)
ops_reduce_gen(short, MPI_SHORT, 0)
ops_reduce_gen(long, MPI_LONG, 0)
ops_reduce_gen(ll, MPI_LONG_LONG, 0)
ops_reduce_gen(ull, MPI_UNSIGNED_LONG_LONG, 0)
ops_reduce_gen(ul, MPI_UNSIGNED_LONG, 0)
ops_reduce_gen(uint, MPI_UNSIGNED, 0)
ops_reduce_gen(complexd, MPI_C_DOUBLE_COMPLEX, complexd(0.0,0.0))
ops_reduce_gen(complexf, MPI_C_FLOAT_COMPLEX, complexf(0.0f,0.0f))



#define ops_reduce_exec(type, zero) \
    std::vector< type > local(handle->size); \
    memcpy(local.data(), handle->data, handle->size); \
    for (int i = 1; i < OPS_instance::getOPSInstance()->OPS_block_index; i++) { \
      if (!OPS_sub_block_list[i]->owned) \
        continue; \
      if (handle->acc == OPS_MAX) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] = MAX( \
              local[d], ((type *)(handle->data + i * handle->size))[d]); \
      if (handle->acc == OPS_MIN) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] = MIN( \
              local[d], ((type *)(handle->data + i * handle->size))[d]); \
      if (handle->acc == OPS_INC) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] += ((type *)(handle->data + i * handle->size))[d]; \
      if (handle->acc == OPS_WRITE) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] = \
              ((type *)(handle->data + i * handle->size))[d] != zero \
                  ? ((type *)(handle->data + i * handle->size))[d] \
                  : local[d];\
    } \
    ops_arg arg; \
    arg.data = (char*)local.data(); \
    arg.acc = handle->acc; \
    arg.dim = handle->size / sizeof(type); \
    ops_mpi_reduce_##type (&arg, local.data()); \
    memcpy(handle->data, local.data(), handle->size);

#define MINC(a,b) (std::abs(a) < std::abs(b) ? (a) : (b))
#define MAXC(a,b) (std::abs(a) < std::abs(b) ? (b) : (a))

#define ops_reduce_exec_complex(type, zero) \
    std::vector< type > local(handle->size); \
    memcpy(local.data(), handle->data, handle->size); \
    for (int i = 1; i < OPS_instance::getOPSInstance()->OPS_block_index; i++) { \
      if (!OPS_sub_block_list[i]->owned) \
        continue; \
      if (handle->acc == OPS_MAX) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] = MAXC( \
              local[d], ((type *)(handle->data + i * handle->size))[d]); \
      if (handle->acc == OPS_MIN) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] = MINC( \
              local[d], ((type *)(handle->data + i * handle->size))[d]); \
      if (handle->acc == OPS_INC) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] += ((type *)(handle->data + i * handle->size))[d]; \
      if (handle->acc == OPS_WRITE) \
        for (unsigned int d = 0; d < handle->size / sizeof(type); d++) \
          local[d] = \
              ((type *)(handle->data + i * handle->size))[d] != zero \
                  ? ((type *)(handle->data + i * handle->size))[d] \
                  : local[d];\
    } \
    ops_arg arg; \
    arg.data = (char*)local.data(); \
    arg.acc = handle->acc; \
    arg.dim = handle->size / sizeof(type); \
    ops_mpi_reduce_##type (&arg, local.data()); \
    memcpy(handle->data, local.data(), handle->size);

void ops_execute_reduction(ops_reduction handle) {
  if (strcmp(handle->type, "int") == 0 || strcmp(handle->type, "int(4)") == 0 ||
      strcmp(handle->type, "integer(4)") == 0 ||
      strcmp(handle->type, "integer") == 0) {
    ops_reduce_exec(int, 0)
  }
  if (strcmp(handle->type, "float") == 0 || strcmp(handle->type, "real") == 0) {
    ops_reduce_exec(float, 0.0f)
  }
  if (strcmp(handle->type, "double") == 0 ||
      strcmp(handle->type, "real(8)") == 0 ||
      strcmp(handle->type, "double precision") == 0) {
    ops_reduce_exec(double, 0.0)
  }
  if (strcmp(handle->type, "complexd") == 0) {
    ops_reduce_exec_complex(complexd, complexd(0.0,0.0))
  }
  if (strcmp(handle->type, "complexf") == 0) {
    ops_reduce_exec_complex(complexf, complexf(0.0f,0.0f))
  }
  if (strcmp(handle->type, "char") == 0) {
    ops_reduce_exec(char, 0)
  }
  if (strcmp(handle->type, "short") == 0) {
    ops_reduce_exec(short, 0)
  }
  if (strcmp(handle->type, "long") == 0) {
    ops_reduce_exec(long, 0)
  }
  if ((strcmp(handle->type, "long long") == 0) || (strcmp(handle->type, "ll") == 0)) {
    ops_reduce_exec(ll, 0)
  }
  if ((strcmp(handle->type, "unsigned long long") == 0) || (strcmp(handle->type, "ull") == 0)) {
    ops_reduce_exec(ull, 0)
  }
  if ((strcmp(handle->type, "unsigned long") == 0) || (strcmp(handle->type, "ul") == 0)) {
    ops_reduce_exec(ul, 0)
  }
  if ((strcmp(handle->type, "unsigned int") == 0) || (strcmp(handle->type, "uint") == 0)) {
    ops_reduce_exec(uint, 0)
  }


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
    range_intersect[dim] = intersection( iter_range[2 * dim], iter_range[2 * dim + 1],
                                         sd->decomp_disp[dim], (sd->decomp_disp[dim] + sd->decomp_size[dim]));
    // i.e. the intersection of the execution range with my full range

    left_boundary_modified[dim] = intersection( iter_range[2 * dim], iter_range[2 * dim + 1],
                                               sd->decomp_disp[dim], sd->decomp_disp[dim] + MAX_DEPTH - 1);
    // i.e. the intersection of the execution range with my left boundary

    right_halo_modified[dim] = intersection(iter_range[2 * dim], iter_range[2 * dim + 1],
                  (sd->decomp_disp[dim] + sd->decomp_size[dim]), (sd->decomp_disp[dim] + sd->decomp_size[dim]) + MAX_DEPTH - 1);
    // i.e. the intersection of the execution range with the my right neighbour's boundary

    right_boundary_modified[dim] = intersection( iter_range[2 * dim], iter_range[2 * dim + 1],
       (sd->decomp_disp[dim] + sd->decomp_size[dim]) - MAX_DEPTH + 1, (sd->decomp_disp[dim] + sd->decomp_size[dim]));

    left_halo_modified[dim] = intersection( iter_range[2 * dim], iter_range[2 * dim + 1],
                           sd->decomp_disp[dim] - MAX_DEPTH + 1, sd->decomp_disp[dim]);
  }

  int left_bnd_beg=0, left_bnd_end=0, left_halo_beg=0, left_halo_end=0;
  int right_bnd_beg=0, right_bnd_end=0, right_halo_beg=0, right_halo_end=0;

  sd->dirtybit = 1;
  for (int dim = 0; dim < ndim; dim++) {
    int other_dims = 1;
    for (int d2 = 0; d2 < ndim; d2++)
      if (d2 != dim)
        other_dims = other_dims && (range_intersect[d2] > 0 || dat->size[d2] == 1);

    if (left_boundary_modified[dim] > 0 && other_dims) {
      int beg = 1 + (iter_range[2 * dim] >= sd->decomp_disp[dim]
                         ? iter_range[2 * dim] - sd->decomp_disp[dim]
                         : 0);
      left_bnd_beg = beg + arg->left_boundary_cleanUpTo[dim];  left_bnd_end = beg + left_boundary_modified[dim];
      for (int d2 = beg + arg->left_boundary_cleanUpTo[dim]; d2 < beg + left_boundary_modified[dim]; d2++) { 
        // we shifted dirtybits, [1] is the first layer not the second
        sd->dirty_dir_send[2 * MAX_DEPTH * dim + d2] = 1;
      }
    }
    if (left_halo_modified[dim] > 0 && other_dims) {
      int beg = iter_range[2 * dim] >= sd->decomp_disp[dim] - MAX_DEPTH + 1
              ? iter_range[2 * dim] - (sd->decomp_disp[dim] - MAX_DEPTH + 1)
              : 0;
      left_halo_beg = beg + arg->left_halo_cleanUpTo[dim]; left_halo_end = beg + left_halo_modified[dim];
      for (int d2 = beg + arg->left_halo_cleanUpTo[dim]; d2 < beg + left_halo_modified[dim]; d2++) {
        sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH - d2 - 1] = 1;
      }
    }
    if (right_boundary_modified[dim] > 0 && other_dims) {
      int beg = iter_range[2 * dim] >= (sd->decomp_disp[dim] + sd->decomp_size[dim]) - MAX_DEPTH + 1
              ? iter_range[2 * dim] - ((sd->decomp_disp[dim] + sd->decomp_size[dim]) - MAX_DEPTH + 1)
              : 0;
      right_bnd_beg = beg + arg->right_boundary_cleanUpTo[dim]; right_bnd_end = beg + right_boundary_modified[dim];
      for (int d2 = beg + arg->right_boundary_cleanUpTo[dim]; d2 < beg + right_boundary_modified[dim]; d2++) {
        sd->dirty_dir_send[2 * MAX_DEPTH * dim + 2 * MAX_DEPTH - d2 - 1] = 1;
      }
    }
    if (right_halo_modified[dim] > 0 && other_dims) {
      int beg = 1 + (iter_range[2 * dim] >= (sd->decomp_disp[dim] + sd->decomp_size[dim])
                   ? iter_range[2 * dim] - (sd->decomp_disp[dim] + sd->decomp_size[dim])
                   : 0);
      right_halo_beg = beg + arg->right_halo_cleanUpTo[dim]; right_halo_end = beg + right_halo_modified[dim];
      for (int d2 = beg + arg->right_halo_cleanUpTo[dim]; d2 < beg + right_halo_modified[dim]; d2++) {
        sd->dirty_dir_recv[2 * MAX_DEPTH * dim + MAX_DEPTH + d2] = 1;
      }
    }
  }

  OPS_instance *instance = OPS_instance::getOPSInstance();
  for (int dim = 0; dim < ndim; dim++)
    if (instance->OPS_diags>5) printf2(instance, "Proc %d dim %d name %s dirtybit set left-boundary %d-%d, left-halo %d-%d, right-boundary %d-%d, right-halo %d-%d \n", ops_get_proc(), dim, dat->name, left_bnd_beg, left_bnd_end, left_halo_beg, left_halo_end, right_bnd_beg, right_bnd_end, right_halo_beg, right_halo_end);

  // Reset the dirtybit cleanUpTo arrays
  for (int dim = 0; dim < ndim; dim++) {
    arg->left_boundary_cleanUpTo[dim] = 0;
    arg->left_halo_cleanUpTo[dim] = 0;
    arg->right_boundary_cleanUpTo[dim] = 0;
    arg->right_halo_cleanUpTo[dim] = 0;
  }
}

void ops_halo_transfer(ops_halo_group group) {
  ops_execute(group->instance);
  ops_mpi_halo_group *mpi_group = &OPS_mpi_halo_group_list[group->index];
  if (mpi_group->nhalos == 0)
    return;

  double c, t1, t2;
  ops_timers_core(&c, &t1);
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

  ops_timers_core(&c, &t2);
  group->instance->ops_user_halo_exchanges_time += t2 - t1;
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
  if (OPS_sub_block_list[dat->block->index]->owned == 1)
    return 1;
  else return 0;
}

void ops_dat_get_raw_metadata(ops_dat dat, int part, int *disp, int *size, int *stride, int *d_m, int *d_p) {
  ops_dat_get_extents(dat, part, disp, size);
  if (stride != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      stride[d] = dat->size[d];
  if (d_m != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      d_m[d] = dat->d_m[d];
  if (d_p != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      d_p[d] = dat->d_p[d];

}

char* ops_dat_get_raw_pointer(ops_dat dat, int part, ops_stencil stencil, ops_memspace *memspace) {
  ops_force_halo_exchange(dat, stencil);
  if (dat->dirty_hd == OPS_DEVICE || *memspace == OPS_DEVICE) {
    if(dat->data_d == NULL) {
      OPSException ex(OPS_RUNTIME_ERROR);
      ex << "Error: ops_dat_get_raw_pointer() sees a NULL device buffer when the buffer should not "
        "be NULL.   This should never happen.  ops_dat name: " << dat->name;
      throw ex;
    }
  }

  if (*memspace == OPS_HOST) {
    if (dat->dirty_hd == OPS_DEVICE) {
      // User wants raw pointer on host, and data is dirty on the device
      // Fetch the data from device to host
      ops_get_data(dat);
    }
    else {
      // Data is diry on host - nothing to do
    }
  } else if (*memspace == OPS_DEVICE) {
    if (dat->dirty_hd == OPS_HOST) {
      // User wants raw pointer on device, and data is dirty on the host
      // Upload the data from host to device
      ops_put_data(dat);
    }
    else {
      // Data is dirty on device - nothing to do
    }
  }
  else {
    // User has not specified where they want the pointer
    // We need a default behaviour:
    if (dat->dirty_hd == OPS_DEVICE)        *memspace = OPS_DEVICE;
    else if (dat->dirty_hd == OPS_HOST)     *memspace = OPS_HOST;
    else if (dat->data_d != NULL)           *memspace = OPS_DEVICE;
    else                                    *memspace = OPS_HOST;
  }

  assert(*memspace==OPS_HOST || *memspace==OPS_DEVICE);
  // Lock the ops_dat with the current memspace
  dat->locked_hd = *memspace;
  return (*memspace == OPS_HOST ? dat->data : dat->data_d) + dat->base_offset;
}

void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc) {
  if (dat->locked_hd==0) {
    // Dat is unlocked
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: ops_dat_release_raw_data() called, but with no matching ops_dat_get_raw_pointer() beforehand: " << dat->name;
    throw ex;
  }
  if (acc != OPS_READ) {
    dat->dirty_hd = dat->locked_hd; // dirty on host or device depending on where the pointer was obtained
    sub_dat_list sd = OPS_sub_dat_list[dat->index];
    sd->dirtybit = 1;
    for (int i = 0; i < 2 * dat->block->dims * MAX_DEPTH; i++) {
      sd->dirty_dir_send[i] = 1;
      sd->dirty_dir_recv[i] = 1;
    }
  }
  dat->locked_hd = 0;
}

void ops_dat_release_raw_data_memspace(ops_dat dat, int part, ops_access acc, ops_memspace *memspace) {
  if (dat->locked_hd==0) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: ops_dat_release_raw_data_memspace() called, but with no matching ops_dat_get_raw_pointer() beforehand: " << dat->name;
    throw ex;
  }
  if (acc != OPS_READ) {
    dat->dirty_hd = *memspace; // dirty on host or device depending on argument
    sub_dat_list sd = OPS_sub_dat_list[dat->index];
    sd->dirtybit = 1;
    for (int i = 0; i < 2 * dat->block->dims * MAX_DEPTH; i++) {
      sd->dirty_dir_send[i] = 1;
      sd->dirty_dir_recv[i] = 1;
    }
  }
  dat->locked_hd = 0;
}

void ops_dat_fetch_data_host(ops_dat dat, int part, char *data) {
      throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
}
void ops_dat_fetch_data_slab_host(ops_dat dat, int part, char *data, int *range) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
}

void ops_dat_fetch_data(ops_dat dat, int part, char *data) {
  ops_execute(dat->block->instance);
  ops_get_data(dat);
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int lsize[OPS_MAX_DIM] = {1};
  int ldisp[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(dat, part, ldisp, lsize);
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    lsize[d] = 1;
    ldisp[d] = 0;
  }
  lsize[0] *= dat->elem_size; //now in bytes
  if (dat->block->dims>3) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for dims>3");
  if (OPS_instance::getOPSInstance()->OPS_soa && dat->dim > 1) throw OPSException(OPS_NOT_IMPLEMENTED, "Error, missing OPS implementation: ops_dat_fetch_data not implemented for SoA");

  for (int k = 0; k < lsize[2]; k++)
    for (int j = 0; j < lsize[1]; j++)
      memcpy(&data[k*lsize[0]*lsize[1]+j*lsize[0]],
             &dat->data[((j-dat->d_m[1]-sd->d_im[1] + (k-dat->d_m[2]-sd->d_im[2])*dat->size[1])*dat->size[0] - dat->d_m[0] - sd->d_im[0])* dat->elem_size],
             lsize[0]);
}

void ops_dat_set_data_host(ops_dat dat, int part, char *data) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
}
void ops_dat_set_data_slab_host(ops_dat dat, int part, char *local_buf,
                                int *local_range) {
  (void)part;
  sub_dat *sd = OPS_sub_dat_list[dat->index];
  ops_execute(dat->block->instance);
  int local_buf_size[OPS_MAX_DIM] = {1};
  int range_max_dim[2 * OPS_MAX_DIM] = {0};
  int d_m[OPS_MAX_DIM]{0};
  for (int d = 0; d < dat->block->dims; d++) {
    local_buf_size[d] = local_range[2 * d + 1] - local_range[2 * d + 0];
    range_max_dim[2 * d] = local_range[2 * d];
    range_max_dim[2 * d + 1] = local_range[2 * d + 1];
  }
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    local_buf_size[d] = 1;
    range_max_dim[2 * d] = 0;
    range_max_dim[2 * d + 1] = 1;
  }

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    d_m[d] = sd->d_im[d] + dat->d_m[d];
  }

  if (dat->block->dims > 5)
    throw OPSException(OPS_NOT_IMPLEMENTED,
                       "Error, missing OPS implementation: ops_dat_fetch_data "
                       "not implemented for dims>5");

  set_loop_slab(local_buf, dat->data, local_buf_size, dat->size, d_m,
                dat->elem_size, dat->dim, range_max_dim);

  dat->dirty_hd = 1;
  sd->dirtybit = 1;
  for (int i = 0; i < 2 * dat->block->dims * MAX_DEPTH; i++) {
    sd->dirty_dir_send[i] = 1;
    sd->dirty_dir_recv[i] = 1;
  }
}

void ops_dat_set_data(ops_dat dat, int part, char *data) {
  ops_execute(dat->block->instance);
  const sub_dat *sd = OPS_sub_dat_list[dat->index];
  const int space_dim{dat->block->dims};
  int *local_range{new int(2 * space_dim)};
  int *range{new int(2 * space_dim)};
  for (int d = 0; d < space_dim; d++) {
    range[2 * d] = sd->gbl_d_m[d];
    range[2 * d + 1] = sd->gbl_size[d] + sd->gbl_d_m[d];
  }
  determine_local_range(dat, range, local_range);
  ops_dat_set_data_slab_host(dat, 0, data, local_range);
  delete range;
  delete local_range;
}

size_t ops_dat_get_slab_extents(ops_dat dat, int part, int *disp, int *size, int *slab) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
/*  int sizel[OPS_MAX_DIM], displ[OPS_MAX_DIM];
  ops_dat_get_extents(dat, part, displ, sizel);
  size_t bytes = dat->elem_size;
  for (int d = 0; d < dat->block->dims; d++) {
    if (slab[2*d]<0 || slab[2*d+1]>sizel[d]) {
      OPSException ex(OPS_RUNTIME_ERROR);
      ex << "Error: ops_dat_get_slab_extents() called on " << dat->name << " with slab ranges in dimension "<<d<<": "<<slab[2*d]<<"-"<<slab[2*d+1]<<" beyond data size 0-" <<sizel[d];
    throw ex;
    }
    disp[d] = slab[2*d];
    size[d] = slab[2*d+1]-slab[2*d];
    bytes *= size[d];
  }
  return bytes;*/
}


int ops_dat_get_global_npartitions(ops_dat dat) {
  //TODO: lower-rank datasets?
  int nranks = 0;
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  if (sb->owned)
    MPI_Comm_size(sb->comm, &nranks);

  return nranks;
}

void ops_dat_get_extents(ops_dat dat, int part, int *disp, int *size) {
  //TODO: part?? local or global?
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  if (disp != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      disp[d] = sd->decomp_disp[d] - dat->d_m[d];
  if (size != NULL)
    for (int d = 0; d < dat->block->dims; d++)
      size[d] = sd->decomp_size[d] + dat->d_m[d] - dat->d_p[d];
}
