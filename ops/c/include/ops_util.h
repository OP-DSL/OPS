#ifndef __OP_UTIL_H
#define __OP_UTIL_H
#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
* THIS SOFTWARE IS PROVIDED BY Mike Giles and AUTHORS ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles and AUTHORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file
 *  @brief Header file for the utility functions used in op_util.c
 *  @author Gihan R. Mudalige, (Started 23-08-2013)
 */

void *xmalloc(size_t size);

void *xrealloc(void *ptr, size_t size);

void* xcalloc (size_t number, size_t size);

//int min(int array[], int size);

int binary_search(int a[], int value, int low, int high);

int linear_search(int a[], int value, int low, int high);

void quickSort(int arr[], int left, int right);

int removeDups(int a[], int array_size);

int file_exist(char const *filename);

inline int mult2(int *size, int dim) {
  int result = 1;
  if (dim > 0) {
    for (int i = 0; i < dim; i++)
      result *= size[i];
  }
  return result;
}

inline int add2(int *coords, int *size, int dim) {
  int result = coords[0];
  for (int i = 1; i <= dim; i++)
    result += coords[i] * mult2(size, i);
  return result;
}

inline int address2(int ndim, int dat_size, int *start, int *size, int *stride,
                    int *off) {
  int base = 0;
  for (int i = 0; i < ndim; i++) {
    base = base + dat_size * mult2(size, i) * (start[i] * stride[i] - off[i]);
  }

  /* for 2D the code generator hard codes the following */
  // base = base + dat_size * 1       * (ps[0] * std[0] - off[0]);
  // base = base + dat_size * size[0] * (ps[1] * std[1] - off[1]);

  return base;
}

inline int off2D(int dim, int *start, int *end, int *size, int *stride) {
  int i = 0;
  int c1[2];
  int c2[2];

  for (i = 0; i <= dim; i++)
    c1[i] = start[i] + 1;
  for (i = dim + 1; i < 2; i++)
    c1[i] = start[i];

  for (i = 0; i < dim; i++)
    c2[i] = end[i];
  for (i = dim; i < 2; i++)
    c2[i] = start[i];

  for (i = 0; i < 2; i++) {
    c1[i] *= stride[i];
    c2[i] *= stride[i];
  }
  int off = add2(c1, size, dim) - add2(c2, size, dim);
  return off;
}

inline int off3D(int dim, int *start, int *end, int *size, int *stride) {
  int i = 0;
  int c1[3];
  int c2[3];

  for (i = 0; i <= dim; i++)
    c1[i] = start[i] + 1;
  for (i = dim + 1; i < 3; i++)
    c1[i] = start[i];

  for (i = 0; i < dim; i++)
    c2[i] = end[i];
  for (i = dim; i < 3; i++)
    c2[i] = start[i];

  for (i = 0; i < 3; i++) {
    c1[i] *= stride[i];
    c2[i] *= stride[i];
  }
  int off = add2(c1, size, dim) - add2(c2, size, dim);
  return off;
}


/// @brief set the local memory of a ops_dat from a buf
/// @param buf pointer to the buf which is always assumed to be in AoS layout
/// @param dat pointer to the ops_dat data
/// @param buf_size the size of the buf
/// @param dat_size
/// @param d_m
/// @param elem_size the number of bits per gird point
/// @param dat_dim the number of elements per grid point
/// e.g., for a multi_dim (d) int ops_dat elem_size=4*d
/// @param range_max_dim  the range of slab
void set_loop_slab(char *buf, char *dat, const int *buf_size,
                   const int *dat_size, const int *d_m, int elem_size,
                   int dat_dim, const int *range_max_dim);

/// @brief copy the local data of a ops_dat to a buf
/// @param buf pointer to the buf which is always assumed to be in AoS layout
/// @param dat pointer to the ops_dat data
/// @param buf_size the size of the buf
/// @param dat_size
/// @param d_m
/// @param elem_size the number of bits per gird point
/// @param dat_dim the number of elements per grid point
/// e.g., for a multi_dim (d) int ops_dat elem_size=4*d
/// @param range_max_dim  the range of slab
void fetch_loop_slab(char *buf, char *dat, const int *buf_size,
                     const int *dat_size, const int *d_m, int elem_size,
                     int dat_dim, const int *range_max_dim);

/// @brief determine the range of a ops_dat at a local rank
/// @param dat a ops_dat
/// @param global_range
/// @param local_range
void determine_local_range(const ops_dat dat, const int *global_range,
                           int *local_range);
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* __OP_UTIL_H */
