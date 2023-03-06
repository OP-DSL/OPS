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
  * @brief Some utility functions for the OPS
  * @author Gihan Mudalige, Istvan Reguly
  * @details Some utility functions for the OPS
  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <ops_util.h>
#include <ops_lib_core.h>

#include <vector>

/*******************************************************************************
* Wrapper for malloc from www.gnu.org/
*******************************************************************************/
void *xmalloc(size_t size) {
  if (size == 0)
    return (void *)NULL;
  void *value = malloc(size);
  if (value == 0)
    printf("Virtual memory exhausted at malloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for realloc from www.gnu.org/
*******************************************************************************/
void *xrealloc(void *ptr, size_t size) {
  if (size == 0) {
    free(ptr);
    return (void *)NULL;
  }

  void *value = realloc(ptr, size);
  if (value == 0)
    printf("Virtual memory exhausted at realloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for calloc from www.gnu.org/
*******************************************************************************/
void *xcalloc(size_t number, size_t size) {
  if (size == 0)
    return (void *)NULL;

  void *value = calloc(number, size);
  if (value == 0)
    printf("Virtual memory exhausted at calloc\n");
  return value;
}

/*******************************************************************************
* Return the index of the min value in an array
*******************************************************************************/
int min_element(int *array, int size) {
  int min = INT_MAX;
  int index = -1;
  for (int i = 0; i < size; i++) {
    if (array[i] < min) {
      index = i;
      min = array[i];
    }
  }
  return index;
}

/*******************************************************************************
* Binary search an array for a given value
*******************************************************************************/
int binary_search(int a[], int value, int low, int high) {
  if (high < low)
    return -1; // not found

  int mid = low + (high - low) / 2;
  if (a[mid] > value)
    return binary_search(a, value, low, mid - 1);
  else if (a[mid] < value)
    return binary_search(a, value, mid + 1, high);
  else
    return mid; // found
}

/*******************************************************************************
* Linear search an array for a given value
*******************************************************************************/
int linear_search(int a[], int value, int low, int high) {
  for (int i = low; i <= high; i++) {
    if (a[i] == value)
      return i;
  }
  return -1;
}

/*******************************************************************************
* Quicksort an array
*******************************************************************************/
void quickSort(int arr[], int left, int right) {
  int i = left;
  int j = right;
  int tmp;
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)
      i++;
    while (arr[j] > pivot)
      j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
      i++;
      j--;
    }
  };
  // recursion
  if (left < j)
    quickSort(arr, left, j);
  if (i < right)
    quickSort(arr, i, right);
}

/*******************************************************************************
* Remove duplicates in an array
*******************************************************************************/
int removeDups(int a[], int array_size) {
  int i, j;
  j = 0;
  // Remove the duplicates ...
  for (i = 1; i < array_size; i++) {
    if (a[i] != a[j]) {
      j++;
      a[j] = a[i]; // Move it to the front
    }
  }
  // The new array size..
  array_size = (j + 1);
  return array_size;
}

/*******************************************************************************
* Check if a file exists
*******************************************************************************/
int file_exist(char const *filename) {
  struct stat buffer;
  return (stat(filename, &buffer) == 0);
}

/*******************************************************************************
* Transpose an array, with potentially different padding - out has to be no smaller
*******************************************************************************/
//double conv_time = 0;
void ops_transpose_data(char *in, char* out, int type_size, int ndim, int* size_in, int *size_out, int* dim_perm) {

//  double t1 = omp_get_wtime();
  int total = 1;
  int count_in[OPS_MAX_DIM];
  int prod_out[OPS_MAX_DIM+1];
  int prod_in[OPS_MAX_DIM+1];
  prod_out[0] = 1;
  prod_in[0] = 1;
  for (int d = 0; d < ndim; d++) {
    total *= size_in[d];
    count_in[d] = size_in[d];
    prod_out[d+1] = prod_out[d]*size_out[d];
    prod_in[d+1] = prod_in[d]*size_in[d];
  }

  if (ndim == 2) {
#pragma omp parallel for OMP_COLLAPSE(2)
    for (int j = 0; j < size_in[1]; j++) {
      for (int i = 0; i < size_in[0]; i++) {
        int idx_out = i*prod_out[dim_perm[0]] +
          j*prod_out[dim_perm[1]];
        int idx_in = i + j * size_in[0];
        memcpy(out + type_size * idx_out,
            in  + type_size * idx_in,
            type_size);

      }
    }
    //conv_time += omp_get_wtime()-t1;
    return;
  }
  if (ndim == 3) {
#pragma omp parallel for OMP_COLLAPSE(3)
    for (int k = 0; k < size_in[2]; k++) {
      for (int j = 0; j < size_in[1]; j++) {
        for (int i = 0; i < size_in[0]; i++) {
          int idx_out = i*prod_out[dim_perm[0]] +
            j*prod_out[dim_perm[1]] +
            k*prod_out[dim_perm[2]];
          int idx_in = i + j * size_in[0] + k * size_in[0] * size_in[1];
          memcpy(out + type_size * idx_out,
              in  + type_size * idx_in,
              type_size);

        }
      }
    }
    //conv_time += omp_get_wtime()-t1;
    return;
  }
  if (ndim == 4) {
#pragma omp parallel for OMP_COLLAPSE(4)
    for (int u = 0; u < size_in[3]; u++) {
      for (int k = 0; k < size_in[2]; k++) {
        for (int j = 0; j < size_in[1]; j++) {
          for (int i = 0; i < size_in[0]; i++) {
            int idx_out = i*prod_out[dim_perm[0]] +
              j*prod_out[dim_perm[1]] +
              k*prod_out[dim_perm[2]] +
              u*prod_out[dim_perm[3]];
            int idx_in = i + j * size_in[0] + k * size_in[0] * size_in[1] +
              u * size_in[0] * size_in[1] * size_in[2];
            memcpy(out + type_size * idx_out,
                in  + type_size * idx_in,
                type_size);

          }
        }
      }
    }
    //conv_time += omp_get_wtime()-t1;
    return;
  }

  for (int i = 0; i < total; i++) {


    int idx_out = 0;
    for (int d = 0; d < ndim; d++) {
      int idx_in_d = size_in[d]-count_in[d];
      int d_out = dim_perm[d];
      idx_out += idx_in_d * prod_out[d_out];
    }

    memcpy(out+type_size*idx_out, in+type_size*i, type_size);

    count_in[0]--;
    int m = 0;
    while(count_in[m] == 0 && i < total-1) { //Can skip the very last iteration
      count_in[m] = size_in[m];
      m++;
      count_in[m]--;
    }

  }
    //conv_time += omp_get_wtime()-t1;
}

void ops_convert_layout(char *in, char *out, ops_block block, int size, int *dat_size, int *dat_size_orig, int type_size, int hybrid_layout) {

      const int num_dims = block->dims + ((size>1)?1:0) + /*(block->count>1?1:0)*/ + (hybrid_layout>0?1:0);
      std::vector<int> size_in(num_dims);
      std::vector<int> size_out(num_dims);
      std::vector<int> dim_perm(num_dims);
      
      int s1 = (size>1 && !block->instance->OPS_soa)?1:0;
      int s2 = (size>1)?1:0;

      if (size>1) {
        size_in[0] = size;
        int idx_dim = (block->instance->OPS_soa)?
            (/*(block->count>1?1:0)+*/block->dims):0;
        dim_perm[0] = idx_dim; 
        size_out[idx_dim] = size;
      }

      for (int d = 0; d < block->dims/*block->batchdim*/; d++) {
        size_in[s2+d] = dat_size_orig[d];
        size_out[s1+d] = dat_size[d];
        dim_perm[s2+d] = s1+d;
      }
      /*if (block->count>1) {
        size_in[s2+block->dims] = block->count;
        size_out[s1+block->batchdim] = block->count;
        dim_perm[s2+block->dims] = s1+block->batchdim;
      }
      for (int d = block->batchdim; d < block->dims; d++) {
        size_in[s2+d] = dat_size_orig[d];
        size_out[s1+d+1] = dat_size_orig[d];
        dim_perm[s2+d] = s1+d+1;
      }

      //Split batchdim in two
      if (hybrid_layout && block->count>1) {
        size_in[s2+block->dims] = hybrid_layout;
        size_in[s2+block->dims+1] = block->count/hybrid_layout;
        size_out[s1+block->batchdim] = hybrid_layout;
        size_out[s2+block->dims+1] = block->count/hybrid_layout;
        dim_perm[s2+block->dims+1] = s2+block->dims+1;
        if (size_in[s2+block->dims] * size_in[s2+block->dims+1] != block->count)
          throw OPSException(OPS_INVALID_ARGUMENT, "Error:  ops_decl_dat -- when batching, the number of systems must be a multiple of the batch size ");
      }*/

      ops_transpose_data(in, out, type_size, num_dims, size_in.data(), size_out.data(), dim_perm.data());
}


void ops_init_zero(char *data, size_t bytes) {
#pragma omp parallel for
  for (long long i = 0; i < (long long)bytes; i++) {
    data[i] = 0;
  }
}
