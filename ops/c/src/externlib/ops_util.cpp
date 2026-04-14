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
#if defined(__linux__)
#include <unistd.h>
#endif

#ifdef OPS_MPI
#include <ops_mpi_core.h>
#else
#include <ops_lib_core.h>
#endif

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <ops_internal2.h>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <mutex>
#include <set>
#include <random>

#ifdef OPS_ML_XGBOOST
#include <xgboost/c_api.h>
#endif

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

void fetch_loop_slab(char *buf, char *dat, const int *buf_size,
                     const int *dat_size, const int *d_m, int elem_size,
                     int dat_dim, const int *range_max_dim) {
  // TODO: add OpenMP here if needed

#if OPS_MAX_DIM > 4
  for (int m = 0; m < buf_size[4]; m++) {
#endif
#if OPS_MAX_DIM > 3
    for (int l = 0; l < buf_size[3]; l++) {
#endif
      for (int k = 0; k < buf_size[2]; k++) {
        for (int j = 0; j < buf_size[1]; j++) {
          size_t buf_index{0}, dat_index{0};
          size_t moff_buf{0}, moff_dat{0}, loff_buf{0}, loff_dat{0};
#if OPS_MAX_DIM > 4
          moff_buf = m * buf_size[0] * buf_size[1] * buf_size[2] * buf_size[3];
          moff_dat = (range_max_dim[2 * 4] + m - d_m[4]) * dat_size[3] *
                     dat_size[2] * dat_size[1] * dat_size[0];
#endif
#if OPS_MAX_DIM > 3
          loff_buf = l * buf_size[0] * buf_size[1] * buf_size[2];
          loff_dat = (range_max_dim[2 * 3] + l - d_m[3]) * dat_size[2] *
                     dat_size[1] * dat_size[0];
#endif
          if (OPS_instance::getOPSInstance()->OPS_soa == 1) {
            for (int i = 0; i < buf_size[0]; i++) {
              for (int d = 0; d < dat_dim; d++) {
                const int type_bits{elem_size / dat_dim};
                size_t doff_dat{static_cast<size_t>(d)};
#if OPS_MAX_DIM > 4
                doff_dat *= (dat_size[4]);
#endif
#if OPS_MAX_DIM > 3
                doff_dat *= (dat_size[3]);
#endif
                doff_dat *= (dat_size[2] * dat_size[1] * dat_size[0]);
                buf_index =
                    (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                     j * buf_size[0] + i) *
                        elem_size +
                    d * type_bits;
                dat_index = (doff_dat + moff_dat + loff_dat +
                             (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                                 dat_size[0] +
                             (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                             range_max_dim[2 * 0] + i - d_m[0]) *
                            type_bits;
                memcpy(&buf[buf_index], &dat[dat_index], type_bits);
              } // d
            }   // i
          } else {
            buf_index = (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                         j * buf_size[0]) *
                        elem_size;
            dat_index = (moff_dat + loff_dat +
                         (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                             dat_size[0] +
                         (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                         range_max_dim[2 * 0] - d_m[0]) *
                        elem_size;
            memcpy(&buf[buf_index], &dat[dat_index], buf_size[0] * elem_size);
          } // OPS_SOA
        }   // j
      }     // k

#if OPS_MAX_DIM > 3
    } // l
#endif
#if OPS_MAX_DIM > 4
  } // m
#endif
}

void set_loop_slab(char *buf, char *dat, const int *buf_size,
                   const int *dat_size, const int *d_m, int elem_size,
                   int dat_dim, const int *range_max_dim) {
  // TODO: add OpenMP here if needed

#if OPS_MAX_DIM > 4
  for (int m = 0; m < buf_size[4]; m++) {
#endif
#if OPS_MAX_DIM > 3
    for (int l = 0; l < buf_size[3]; l++) {
#endif
      for (int k = 0; k < buf_size[2]; k++) {
        for (int j = 0; j < buf_size[1]; j++) {
          size_t buf_index{0}, dat_index{0};
          size_t moff_buf{0}, moff_dat{0}, loff_buf{0}, loff_dat{0};
#if OPS_MAX_DIM > 4
          moff_buf = m * buf_size[0] * buf_size[1] * buf_size[2] * buf_size[3];
          moff_dat = (range_max_dim[2 * 4] + m - d_m[4]) * dat_size[3] *
                     dat_size[2] * dat_size[1] * dat_size[0];
#endif
#if OPS_MAX_DIM > 3
          loff_buf = l * buf_size[0] * buf_size[1] * buf_size[2];
          loff_dat = (range_max_dim[2 * 3] + l - d_m[3]) * dat_size[2] *
                     dat_size[1] * dat_size[0];
#endif
          if (OPS_instance::getOPSInstance()->OPS_soa == 1) {
            for (int i = 0; i < buf_size[0]; i++) {
              for (int d = 0; d < dat_dim; d++) {
                const int type_bits{elem_size / dat_dim};
                size_t doff_dat{static_cast<size_t>(d)};
#if OPS_MAX_DIM > 4
                doff_dat *= (dat_size[4]);
#endif
#if OPS_MAX_DIM > 3
                doff_dat *= (dat_size[3]);
#endif
                doff_dat *= (dat_size[2] * dat_size[1] * dat_size[0]);
                buf_index =
                    (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                     j * buf_size[0] + i) *
                        elem_size +
                    d * type_bits;
                dat_index = (doff_dat + moff_dat + loff_dat +
                             (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                                 dat_size[0] +
                             (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                             range_max_dim[2 * 0] + i - d_m[0]) *
                            type_bits;
                memcpy(&dat[dat_index], &buf[buf_index], type_bits);
              } // d
            }   // i
          } else {
            buf_index = (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                         j * buf_size[0]) *
                        elem_size;
            dat_index = (moff_dat + loff_dat +
                         (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                             dat_size[0] +
                         (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                         range_max_dim[2 * 0] - d_m[0]) *
                        elem_size;
            memcpy(&dat[dat_index], &buf[buf_index], buf_size[0] * elem_size);
          } // OPS_SOA
        }   // j
      }     // k

#if OPS_MAX_DIM > 3
    } // l
#endif
#if OPS_MAX_DIM > 4
  } // m
#endif
}

void determine_local_range(const ops_dat dat, const int *global_range,
                           int *local_range) {
  ops_arg dat_arg;
  const int space_dim{dat->block->dims};
  if (space_dim == 3) {
    int s3D_000[]{0, 0, 0};
    ops_stencil S3D_000{ops_decl_stencil(3, 1, s3D_000, "000")};
    dat_arg = ops_arg_dat(dat, dat->dim, S3D_000, dat->type, OPS_READ);
  }
  else if (space_dim == 2) {
    int s2D_00[]{0, 0};
    ops_stencil S2D_00{ops_decl_stencil(2, 1, s2D_00, "00")};
    dat_arg = ops_arg_dat(dat, dat->dim, S2D_00, dat->type, OPS_READ);
  }
  else if (space_dim == 1) {
    int s1D_0[]{0};
    ops_stencil S1D_0{ops_decl_stencil(1, 1, s1D_0, "0")};
    dat_arg = ops_arg_dat(dat, dat->dim, S1D_0, dat->type, OPS_READ);
  }
  else {
    OPSException ex(OPS_NOT_IMPLEMENTED);
    ex << "Error: determine_local_range -- not implemented for dim >3";
    throw ex;
  }

  int *arg_idx{new int[space_dim]};

  int *local_start{new int[space_dim]};
  int *local_end{new int[space_dim]};
  if (compute_ranges(&dat_arg, 1, dat->block, (int *)global_range, local_start,
                     local_end, arg_idx) < 0) {
    return;
  }
  for (int i = 0; i < space_dim; i++) {
    local_range[2 * i] = local_start[i];
    local_range[2 * i + 1] = local_end[i];
  }
  // printf(
  //     "At Rank = %d istart=%d iend=%d  jstart=%d jend=%d  kstart=%d kend=%d\n",
  //     ops_my_global_rank, local_range[0], local_range[1], local_range[2],
  //     local_range[3], local_range[4], local_range[5]);
  delete[] arg_idx;
  delete[] local_start;
  delete[] local_end;
}



// ------------------------------
// Autotuning state and utilities
// ------------------------------
namespace {
  // Structure to hold array (ops_dat) metadata
  struct ArrayInfo {
    int arg_idx;           // Index of the argument
    std::string name;      // Name of the dataset
    std::string type;      // Data type (e.g., "double", "float")
    int dim;               // Elements per grid point
    int elem_size;         // Bytes per grid point
    int size[OPS_MAX_DIM]; // Size in each dimension (including halo)
    int d_m[OPS_MAX_DIM];  // Halo depth negative direction
    int d_p[OPS_MAX_DIM];  // Halo depth positive direction
    int stride[OPS_MAX_DIM]; // Stride in each dimension
    size_t total_bytes;    // Total memory in bytes
    ops_access access;     // Access type (READ, WRITE, RW, etc.)
  };

  struct TuneState {
    std::vector<ops::dim3> candidates;
    size_t next_idx{0};
    bool decided{false};
    double best_time{1e300};
    double default_time{1e300};
    ops::dim3 best{1,1,1};
    ops::dim3 last_served{1,1,1};
    long long last_points{0};
    long long best_points{0};
    // Metadata captured once per kernel
    int nargs_meta{0};
    int nstencil_args{0};
    int widest_radius{0};
    int widest_radius_x{0};
    int widest_radius_y{0};
    int widest_radius_z{0};
    std::string stencil_sig{};
    bool meta_set{false};
    // Array metadata
    std::vector<ArrayInfo> arrays;
    int ndims_kernel{0};  // Kernel dimensionality
    int max_threads_per_block{1024};  // Real limit per kernel from cudaFuncGetAttributes
    int x_extent{0}, y_extent{0}, z_extent{0}; // Grid extents for this kernel
    float ml_features[43]{};
    bool ml_features_set{false};

    ops::dim3 default_block{0,0,0};  // dimension-adjusted default block

    // Mode 5: ML-guided exploration (same candidate space as Mode 1)
    bool explore_started{false};     // true after candidates are built
    bool explore_converged{false};   // true when all candidates tested
    std::vector<ops::dim3> explore_candidates; // all valid candidates, ML-ranked
    size_t explore_idx{0};           // next candidate to serve
    ops::dim3 explore_best{0,0,0};   // best block found so far
    double explore_best_time{1e300}; // best time found
    double explore_pred_time{-1.0};  // time of ML top-1 prediction (candidate #0)
    int explore_best_rank{-1};       // rank of actual best in ML ordering (1-based)

  };

  std::unordered_map<int, TuneState> g_tune;
  std::ofstream g_log;
  std::once_flag g_log_once;
  std::mutex g_mutex;
  std::ofstream g_best_log;
  std::once_flag g_best_log_once;
  long long g_tuning_rows = 0;
  long long g_best_rows = 0;

  // Cache for precomputed block sizes from CSV (mode=2)
  std::unordered_map<int, ops::dim3> g_precomputed_blocks;
  bool g_precomputed_loaded = false;


  void load_precomputed_blocks() {
    if (g_precomputed_loaded) return;
    g_precomputed_loaded = true;
    
    const char* csv_path = std::getenv("OPS_BLOCKSIZE_CSV");
    if (!csv_path) {
      printf("[OPS] WARNING: OPS_AUTOTUNE_MODE=2 but OPS_BLOCKSIZE_CSV not set\n");
      return;
    }
    
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      printf("[OPS] WARNING: Cannot open CSV file: %s\n", csv_path);
      return;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#' || line[0] == 'A') continue; // Skip empty, comments, metadata
      std::istringstream ss(line);
      std::string token;
      
      try {
        std::getline(ss, token, ','); int kernel_id = std::stoi(token);
        std::getline(ss, token, ','); unsigned int bx = std::stoul(token);
        std::getline(ss, token, ','); unsigned int by = std::stoul(token);
        std::getline(ss, token, ','); unsigned int bz = std::stoul(token);
        
        g_precomputed_blocks[kernel_id] = ops::dim3(bx, by, bz);
      } catch (...) {
        continue; // Skip malformed lines
      }
    }
    
    printf("[OPS] Loaded %zu precomputed block sizes from %s\n",
           g_precomputed_blocks.size(), csv_path);
  }


  // =========================================================================
  // ML Inference (Mode 4: XGBoost multi-output, Mode 6: XGBoost single-output) — inline block size prediction
  // =========================================================================

  // Number of features must match the trained model (BASE + STENCIL + DERIVED)
  static const int ML_NUM_BASE_FEATURES     = 20;
  static const int ML_NUM_STENCIL_FEATURES  = 10;
  static const int ML_NUM_DERIVED_FEATURES  = 13;
  static const int ML_NUM_FEATURES_EXPECTED = 43;  // 20 + 10 + 13


  // ---- XGBoost Multi-Output Model (3 boosters: bx, by, bz) ----
#ifdef OPS_ML_XGBOOST

  // Per-axis class values (must match Python BX_CLASSES, BY_CLASSES, BZ_CLASSES)
  static const int BX_CLASSES[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  static const int BY_CLASSES[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  static const int BZ_CLASSES[] = {1, 2, 4, 8, 16, 32, 64};
  static const int NUM_BX = 11;
  static const int NUM_BY = 11;
  static const int NUM_BZ = 7;
  static const int XGB_MIN_THREADS = 32;
  static const int XGB_MAX_THREADS = 1024;

  struct XGBMultiModel {
    BoosterHandle booster_bx{nullptr};
    BoosterHandle booster_by{nullptr};
    BoosterHandle booster_bz{nullptr};
    bool loaded{false};
  };

  XGBMultiModel g_xgb_model;

  // Mode 5: Exploration-based online training state
  struct ExploreTrainSample {
    float features[43];
    int label_bx;            // class index for best BX found by exploration
    int label_by;            // class index for best BY
    int label_bz;            // class index for best BZ
    double explore_best_time;
    double default_time;
  };

  struct ExploreLearningState {
    std::vector<ExploreTrainSample> buffer;
    int update_interval{200};    // flush buffer & update boosters every N samples
    int total_samples{0};
    int total_updates{0};
    bool initialized{false};
  };

  ExploreLearningState g_explore_learn;

  // Find class index for a block value in a class array (exact or nearest)
  static int find_class_index(const int* classes, int num_classes, int value) {
    for (int i = 0; i < num_classes; i++) {
      if (classes[i] == value) return i;
    }
    // Nearest match
    int best_idx = 0;
    int best_dist = std::abs(classes[0] - value);
    for (int i = 1; i < num_classes; i++) {
      int dist = std::abs(classes[i] - value);
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = i;
      }
    }
    return best_idx;
  }

  static void init_explore_learning() {
    if (g_explore_learn.initialized) return;
    const char* interval_str = std::getenv("OPS_ONLINE_UPDATE_INTERVAL");
    if (interval_str) {
      g_explore_learn.update_interval = std::atoi(interval_str);
      if (g_explore_learn.update_interval < 1) g_explore_learn.update_interval = 200;
    }
    g_explore_learn.initialized = true;
    printf("[OPS-EXPLORE] Training initialized: update_interval=%d\n",
           g_explore_learn.update_interval);
  }

  // Update all 3 boosters using buffered exploration results as training labels
  static void update_boosters_from_exploration() {
    if (g_explore_learn.buffer.empty()) return;
    if (!g_xgb_model.loaded) return;

    int n = (int)g_explore_learn.buffer.size();
    int ncols = 43;

    // Build feature matrix and label arrays
    std::vector<float> feat_mat(n * ncols);
    std::vector<float> labels_bx(n), labels_by(n), labels_bz(n);
    std::vector<float> weights(n);

    for (int i = 0; i < n; i++) {
      auto& s = g_explore_learn.buffer[i];
      std::copy(s.features, s.features + ncols, feat_mat.begin() + i * ncols);
      labels_bx[i] = (float)s.label_bx;
      labels_by[i] = (float)s.label_by;
      labels_bz[i] = (float)s.label_bz;
      // Weight: ratio of default/explore time (higher weight = bigger speedup found)
      if (s.explore_best_time > 0.0 && s.default_time > 0.0) {
        weights[i] = (float)(s.default_time / s.explore_best_time);
      } else {
        weights[i] = 1.0f;
      }
    }

    auto update_one = [&](BoosterHandle booster, const float* labels, const char* name) {
      DMatrixHandle dmat;
      XGDMatrixCreateFromMat(feat_mat.data(), n, ncols, NAN, &dmat);
      XGDMatrixSetFloatInfo(dmat, "label", labels, n);
      XGDMatrixSetFloatInfo(dmat, "weight", weights.data(), n);
      int ret = XGBoosterUpdateOneIter(booster, g_explore_learn.total_updates, dmat);
      if (ret != 0) {
        printf("[OPS-EXPLORE] ERROR: XGBoosterUpdateOneIter(%s) failed: %s\n",
               name, XGBGetLastError());
      } else {
        printf("[OPS-EXPLORE] Updated %s booster with %d samples (update #%d)\n",
               name, n, g_explore_learn.total_updates + 1);
      }
      XGDMatrixFree(dmat);
    };

    update_one(g_xgb_model.booster_bx, labels_bx.data(), "bx");
    update_one(g_xgb_model.booster_by, labels_by.data(), "by");
    update_one(g_xgb_model.booster_bz, labels_bz.data(), "bz");

    g_explore_learn.total_updates++;
    g_explore_learn.total_samples += n;
    g_explore_learn.buffer.clear();

    printf("[OPS-EXPLORE] Training: total_samples=%d total_updates=%d\n",
           g_explore_learn.total_samples, g_explore_learn.total_updates);
  }

  // Buffer a training sample when a kernel's exploration converges
  static void buffer_explore_sample(TuneState& st, int kernel_id) {
    init_explore_learning();

    // Determine label: use explored best if it beats default, otherwise default
    ops::dim3 label_block;
    if (st.explore_best_time < st.default_time && st.default_time < 1e100) {
      label_block = st.explore_best;
      printf("[OPS-EXPLORE] kernel %d: explore beat default (%.6f < %.6f) -> label (%u,%u,%u)\n",
             kernel_id, st.explore_best_time, st.default_time,
             label_block.x, label_block.y, label_block.z);
    } else {
      // Default block is best or default time unknown
      OPS_instance* inst = OPS_instance::getOPSInstance();
      if (inst) {
        label_block = ops::dim3(inst->OPS_block_size_x, inst->OPS_block_size_y, inst->OPS_block_size_z);
      } else {
        label_block = st.explore_best;
      }
      printf("[OPS-EXPLORE] kernel %d: default kept (explore=%.6f vs default=%.6f) -> label (%u,%u,%u)\n",
             kernel_id, st.explore_best_time, st.default_time,
             label_block.x, label_block.y, label_block.z);
    }

    ExploreTrainSample sample;
    std::copy(st.ml_features, st.ml_features + 43, sample.features);
    sample.label_bx = find_class_index(BX_CLASSES, NUM_BX, label_block.x);
    sample.label_by = find_class_index(BY_CLASSES, NUM_BY, label_block.y);
    sample.label_bz = find_class_index(BZ_CLASSES, NUM_BZ, label_block.z);
    sample.explore_best_time = st.explore_best_time;
    sample.default_time = st.default_time;

    g_explore_learn.buffer.push_back(sample);

    // Flush buffer when it reaches update_interval
    if ((int)g_explore_learn.buffer.size() >= g_explore_learn.update_interval) {
      update_boosters_from_exploration();
    }
  }

  static bool load_one_booster(const char* path, const char* name, BoosterHandle* out) {
    int ret = XGBoosterCreate(nullptr, 0, out);
    if (ret != 0) {
      printf("[OPS] ERROR: XGBoosterCreate(%s) failed: %s\n", name, XGBGetLastError());
      return false;
    }
    ret = XGBoosterLoadModel(*out, path);
    if (ret != 0) {
      printf("[OPS] ERROR: XGBoosterLoadModel(%s) failed: %s\n", name, XGBGetLastError());
      XGBoosterFree(*out); *out = nullptr;
      return false;
    }
    printf("[OPS] XGBoost %s model loaded: %s\n", name, path);
    return true;
  }

  bool load_xgboost_model() {
    if (g_xgb_model.loaded) return true;

    const char* bx_path = std::getenv("OPS_XGBOOST_BX");
    const char* by_path = std::getenv("OPS_XGBOOST_BY");
    const char* bz_path = std::getenv("OPS_XGBOOST_BZ");

    if (!bx_path || !by_path || !bz_path) {
      printf("[OPS] ERROR: Mode 4 requires OPS_XGBOOST_BX, OPS_XGBOOST_BY, OPS_XGBOOST_BZ\n");
      return false;
    }

    if (!load_one_booster(bx_path, "bx", &g_xgb_model.booster_bx)) return false;
    if (!load_one_booster(by_path, "by", &g_xgb_model.booster_by)) return false;
    if (!load_one_booster(bz_path, "bz", &g_xgb_model.booster_bz)) return false;

    g_xgb_model.loaded = true;
    return true;
  }

  // Predict argmax class from one booster
  static int xgb_predict_axis(BoosterHandle booster, DMatrixHandle dmat) {
    bst_ulong out_len;
    const float* out_result;
    int ret = XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result);
    if (ret != 0) {
      printf("[OPS] ERROR: XGBoosterPredict failed: %s\n", XGBGetLastError());
      return 0;
    }
    int best = 0;
    float best_prob = out_result[0];
    for (bst_ulong i = 1; i < out_len; i++) {
      if (out_result[i] > best_prob) {
        best_prob = out_result[i];
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  // Enforce MIN_THREADS <= bx*by*bz <= max_thr
  // max_thr defaults to XGB_MAX_THREADS (1024) but can be per-kernel limit from cudaFuncGetAttributes
  static void enforce_thread_constraint(int& bx, int& by, int& bz, int max_thr = XGB_MAX_THREADS) {
    int prod = bx * by * bz;
    // Scale down
    while (prod > max_thr) {
      if (bx >= by && bx >= bz && bx > 1) bx /= 2;
      else if (by >= bz && by > 1) by /= 2;
      else if (bz > 1) bz /= 2;
      else break;
      prod = bx * by * bz;
    }
    // Scale up
    while (prod < XGB_MIN_THREADS) {
      if (bx <= by && bx <= bz && bx * 2 <= max_thr) bx *= 2;
      else if (by <= bz && by * 2 <= max_thr) by *= 2;
      else if (bz * 2 <= 64) bz *= 2;
      else if (bx * 2 <= max_thr) bx *= 2;
      else if (by * 2 <= max_thr) by *= 2;
      else break;
      prod = bx * by * bz;
    }
  }

  // Multi-output prediction: returns (bx, by, bz) directly
  ops::dim3 xgboost_predict_multi(const float* features, int num_features) {
    DMatrixHandle dmat;
    int ret = XGDMatrixCreateFromMat(features, 1, num_features, NAN, &dmat);
    if (ret != 0) {
      printf("[OPS] ERROR: XGDMatrixCreateFromMat failed: %s\n", XGBGetLastError());
      return ops::dim3(32, 1, 1);
    }

    int idx_bx = xgb_predict_axis(g_xgb_model.booster_bx, dmat);
    int idx_by = xgb_predict_axis(g_xgb_model.booster_by, dmat);
    int idx_bz = xgb_predict_axis(g_xgb_model.booster_bz, dmat);
    XGDMatrixFree(dmat);

    int bx = BX_CLASSES[idx_bx < NUM_BX ? idx_bx : 0];
    int by = BY_CLASSES[idx_by < NUM_BY ? idx_by : 0];
    int bz = BZ_CLASSES[idx_bz < NUM_BZ ? idx_bz : 0];

    enforce_thread_constraint(bx, by, bz);
    return ops::dim3(bx, by, bz);
  }

  // ---- Mode 6: XGBoost single-output (combined classes) ----

  // Class mapping loaded at runtime from file (OPS_XGBOOST_SINGLE_CLASSES env var)
  static int g_single_num_classes = 0;
  static std::vector<int> g_single_class_bx;
  static std::vector<int> g_single_class_by;
  static std::vector<int> g_single_class_bz;

  // Global state for single-output model
  static BoosterHandle g_xgb_single_booster = nullptr;
  static bool g_xgb_single_loaded = false;

  static bool load_single_class_mapping(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
      printf("[OPS] ERROR: Cannot open class mapping file: %s\n", path);
      return false;
    }
    g_single_class_bx.clear();
    g_single_class_by.clear();
    g_single_class_bz.clear();
    char line[256];
    while (fgets(line, sizeof(line), f)) {
      if (line[0] == '#' || line[0] == '\n') continue;
      int bx, by, bz;
      if (sscanf(line, "%d %d %d", &bx, &by, &bz) == 3) {
        g_single_class_bx.push_back(bx);
        g_single_class_by.push_back(by);
        g_single_class_bz.push_back(bz);
      }
    }
    fclose(f);
    g_single_num_classes = (int)g_single_class_bx.size();
    printf("[OPS] Loaded %d single-output classes from %s\n", g_single_num_classes, path);
    return g_single_num_classes > 0;
  }

  bool load_xgboost_single_model() {
    if (g_xgb_single_loaded) return true;
    const char* model_path = std::getenv("OPS_XGBOOST_SINGLE");
    if (!model_path) {
      printf("[OPS] ERROR: Mode 6 requires OPS_XGBOOST_SINGLE env var\n");
      return false;
    }
    const char* classes_path = std::getenv("OPS_XGBOOST_SINGLE_CLASSES");
    if (!classes_path) {
      printf("[OPS] ERROR: Mode 6 requires OPS_XGBOOST_SINGLE_CLASSES env var\n");
      return false;
    }
    if (!load_single_class_mapping(classes_path)) return false;
    if (!load_one_booster(model_path, "single", &g_xgb_single_booster)) return false;
    g_xgb_single_loaded = true;
    return true;
  }

  ops::dim3 xgboost_predict_single(const float* features, int num_features) {
    DMatrixHandle dmat;
    int ret = XGDMatrixCreateFromMat(features, 1, num_features, NAN, &dmat);
    if (ret != 0) {
      printf("[OPS] ERROR: XGDMatrixCreateFromMat failed: %s\n", XGBGetLastError());
      return ops::dim3(32, 1, 1);
    }

    // Predict: returns SINGLE_NUM_CLASSES probabilities
    bst_ulong out_len;
    const float* out_result;
    ret = XGBoosterPredict(g_xgb_single_booster, dmat, 0, 0, 0, &out_len, &out_result);
    XGDMatrixFree(dmat);
    if (ret != 0) {
      printf("[OPS] ERROR: XGBoosterPredict(single) failed: %s\n", XGBGetLastError());
      return ops::dim3(32, 1, 1);
    }

    // Argmax over classes
    int best = 0;
    float best_prob = out_result[0];
    for (bst_ulong i = 1; i < out_len && i < (bst_ulong)g_single_num_classes; i++) {
      if (out_result[i] > best_prob) {
        best_prob = out_result[i];
        best = static_cast<int>(i);
      }
    }

    int bx = g_single_class_bx[best];
    int by = g_single_class_by[best];
    int bz = g_single_class_bz[best];
    enforce_thread_constraint(bx, by, bz);
    return ops::dim3(bx, by, bz);
  }

  // ---- Mode 5: ML-guided exploration helpers ----

  // Get full probability vector for one axis booster
  static std::vector<float> xgb_get_probs(BoosterHandle booster, DMatrixHandle dmat, int num_classes) {
    bst_ulong out_len;
    const float* out_result;
    int ret = XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result);
    std::vector<float> probs(num_classes, 1.0f / num_classes); // uniform fallback
    if (ret == 0 && (int)out_len >= num_classes) {
      for (int i = 0; i < num_classes; i++) probs[i] = out_result[i];
    }
    return probs;
  }

  // Build ALL valid candidates using EXACTLY the same logic as Mode 1 (build_candidates_1d/2d/3d)
  // Then rank them by ML probability: score = P(bx) * P(by) * P(bz)
  static void build_explore_candidates(TuneState& st, const float* features, int num_features,
                                        int ndims, int max_thr) {
    // Step 1: Build candidates using Mode 1's exact logic
    // (powers of 2, same extents, same constraints)
    st.explore_candidates.clear();
    const int max_threads = std::min(1024, max_thr);
    const int max_x = std::min(st.x_extent > 0 ? st.x_extent : 1, max_threads);
    const int max_y = std::min(st.y_extent > 0 ? st.y_extent : 1, max_threads);
    const int max_z = std::min(std::min(st.z_extent > 0 ? st.z_extent : 1, max_threads), 64);

    std::vector<ops::dim3> all_candidates;
    if (ndims <= 1) {
      for (int bx = 1; bx <= max_x; bx <<= 1) {
        if (bx <= max_threads && bx >= 32) all_candidates.push_back(ops::dim3(bx, 1, 1));
        if (bx == 0) break;
      }
    } else if (ndims == 2) {
      for (int by = 1; by <= max_y; by <<= 1) {
        for (int bx = 1; bx <= max_x; bx <<= 1) {
          int prod = bx * by;
          if (prod <= max_threads && prod >= 32)
            all_candidates.push_back(ops::dim3(bx, by, 1));
          if (bx == 0) break;
        }
        if (by == 0) break;
      }
    } else {
      for (int bz = 1; bz <= max_z; bz <<= 1) {
        for (int by = 1; by <= max_y; by <<= 1) {
          for (int bx = 1; bx <= max_x; bx <<= 1) {
            long long prod = 1LL * bx * by * bz;
            if (prod <= max_threads && prod >= 32)
              all_candidates.push_back(ops::dim3(bx, by, bz));
            if (bx == 0) break;
          }
          if (by == 0) break;
        }
        if (bz == 0) break;
      }
    }

    // Add default block if not present
    OPS_instance* inst = OPS_instance::getOPSInstance();
    if (inst) {
      unsigned int gx = inst->OPS_block_size_x;
      unsigned int gy = (ndims > 1) ? inst->OPS_block_size_y : 1;
      unsigned int gz = (ndims > 2) ? inst->OPS_block_size_z : 1;
      ops::dim3 def_block(gx, gy, gz);
      bool found = false;
      for (const auto& c : all_candidates) {
        if (c.x == def_block.x && c.y == def_block.y && c.z == def_block.z) { found = true; break; }
      }
      if (!found) all_candidates.push_back(def_block);
    }

    // Step 2: Get ML probabilities and score each candidate
    DMatrixHandle dmat;
    XGDMatrixCreateFromMat(features, 1, num_features, NAN, &dmat);
    std::vector<float> prob_bx = xgb_get_probs(g_xgb_model.booster_bx, dmat, NUM_BX);
    std::vector<float> prob_by = xgb_get_probs(g_xgb_model.booster_by, dmat, NUM_BY);
    std::vector<float> prob_bz = xgb_get_probs(g_xgb_model.booster_bz, dmat, NUM_BZ);
    XGDMatrixFree(dmat);

    struct ScoredBlock {
      ops::dim3 block;
      float score;
    };
    std::vector<ScoredBlock> scored;
    for (const auto& c : all_candidates) {
      int ibx = find_class_index(BX_CLASSES, NUM_BX, c.x);
      int iby = find_class_index(BY_CLASSES, NUM_BY, c.y);
      int ibz = find_class_index(BZ_CLASSES, NUM_BZ, c.z);
      float score = prob_bx[ibx] * prob_by[iby] * prob_bz[ibz];
      scored.push_back({c, score});
    }

    // Sort by ML score descending
    std::sort(scored.begin(), scored.end(),
              [](const ScoredBlock& a, const ScoredBlock& b) {
                return a.score > b.score;
              });

    for (size_t i = 0; i < scored.size(); i++) {
      st.explore_candidates.push_back(scored[i].block);
    }
    st.explore_idx = 0;

    printf("[OPS-EXPLORE] kernel: %zu valid candidates (same as Mode 1), ML-ranked\n",
           st.explore_candidates.size());
  }

#endif  // OPS_ML_XGBOOST

  // ---- Feature extraction (shared by XGBoost modes) ----

  void extract_kernel_features(int kernel_id, int ndims,
                                int* local_range, int nargs, ops_arg* args,
                                int max_threads_per_block,
                                float* features) {
    // Feature order MUST match Python's ALL_FEATURE_COLS:
    //   BASE_FEATURES (20) + STENCIL_FEATURES (10) + DERIVED_FEATURES (13)

    // ---- Aggregate array metadata (like data_loader.py::aggregate_metadata) ----
    double total_bytes = 0;
    int max_dim = 0, max_elem_size = 0;
    int max_size_x = 0, max_size_y = 0, max_size_z = 0;
    int min_dm_x = 0, min_dm_y = 0, min_dm_z = 0;
    int max_dp_x = 0, max_dp_y = 0, max_dp_z = 0;
    int num_read = 0, num_write = 0, num_rw = 0;
    int nstencil_args = 0;
    bool first_arr = true;

    for (int i = 0; i < nargs; i++) {
      ops_arg &a = args[i];
      if (a.argtype == OPS_ARG_DAT && a.dat != nullptr) {
        total_bytes += static_cast<double>(a.dat->mem);
        if (a.dat->dim > max_dim) max_dim = a.dat->dim;
        if (a.dat->elem_size > max_elem_size) max_elem_size = a.dat->elem_size;
        if (a.dat->size[0] > max_size_x) max_size_x = a.dat->size[0];
        if (a.dat->size[1] > max_size_y) max_size_y = a.dat->size[1];
        if (a.dat->size[2] > max_size_z) max_size_z = a.dat->size[2];

        if (first_arr) {
          min_dm_x = a.dat->d_m[0]; min_dm_y = a.dat->d_m[1]; min_dm_z = a.dat->d_m[2];
          max_dp_x = a.dat->d_p[0]; max_dp_y = a.dat->d_p[1]; max_dp_z = a.dat->d_p[2];
          first_arr = false;
        } else {
          if (a.dat->d_m[0] < min_dm_x) min_dm_x = a.dat->d_m[0];
          if (a.dat->d_m[1] < min_dm_y) min_dm_y = a.dat->d_m[1];
          if (a.dat->d_m[2] < min_dm_z) min_dm_z = a.dat->d_m[2];
          if (a.dat->d_p[0] > max_dp_x) max_dp_x = a.dat->d_p[0];
          if (a.dat->d_p[1] > max_dp_y) max_dp_y = a.dat->d_p[1];
          if (a.dat->d_p[2] > max_dp_z) max_dp_z = a.dat->d_p[2];
        }

        if (a.acc == OPS_READ)       num_read++;
        else if (a.acc == OPS_WRITE)  num_write++;
        else if (a.acc == OPS_RW)     num_rw++;
      }
    }

    // Grid extents from local_range
    const float grid_x = ndims >= 1 ? static_cast<float>(std::max(1, local_range[1] - local_range[0])) : 1.0f;
    const float grid_y = ndims >= 2 ? static_cast<float>(std::max(1, local_range[3] - local_range[2])) : 1.0f;
    const float grid_z = ndims >= 3 ? static_cast<float>(std::max(1, local_range[5] - local_range[4])) : 1.0f;

    // ---- Stencil features (like mlp_model.py::parse_stencil_sig) ----
    int n_args_x_offset = 0, n_args_y_offset = 0, n_args_z_offset = 0;
    int n_args_face = 0, n_args_stride = 0, n_args_none = 0;
    int max_radius_x = 0, max_radius_y = 0, max_radius_z = 0;
    int total_offset_points = 0;

    for (int i = 0; i < nargs; i++) {
      ops_arg &a = args[i];

      if (a.stencil == nullptr) {
        n_args_none++;
        continue;
      }

      nstencil_args++;
      const char* sname = a.stencil->name;

      // Face stencil (name starts with 'f')
      if (sname && sname[0] == 'f') {
        n_args_face++;
        continue;
      }

      // Stride stencil
      if (sname && strstr(sname, "stride") != nullptr) {
        n_args_stride++;
        if (strstr(sname, "stride3D_x") || strstr(sname, "P100") || strstr(sname, "M100")) {
          n_args_x_offset++;
          if (1 > max_radius_x) max_radius_x = 1;
          total_offset_points++;
        }
        if (strstr(sname, "stride3D_y") || strstr(sname, "0P10") || strstr(sname, "0M10")) {
          n_args_y_offset++;
          if (1 > max_radius_y) max_radius_y = 1;
          total_offset_points++;
        }
        if (strstr(sname, "stride3D_z") || strstr(sname, "00P1") || strstr(sname, "00M1")) {
          n_args_z_offset++;
          if (1 > max_radius_z) max_radius_z = 1;
          total_offset_points++;
        }
        continue;
      }

      // Regular stencil: parse raw offsets, skip p=0 (base point)
      int sdims = a.stencil->dims > 0 ? a.stencil->dims : ndims;
      for (int p = 1; p < a.stencil->points; p++) {
        int x_off = a.stencil->stencil[p * sdims + 0];
        int y_off = sdims > 1 ? a.stencil->stencil[p * sdims + 1] : 0;
        int z_off = sdims > 2 ? a.stencil->stencil[p * sdims + 2] : 0;

        if (x_off != 0) {
          n_args_x_offset++;
          int ax = x_off < 0 ? -x_off : x_off;
          if (ax > max_radius_x) max_radius_x = ax;
        }
        if (y_off != 0) {
          n_args_y_offset++;
          int ay = y_off < 0 ? -y_off : y_off;
          if (ay > max_radius_y) max_radius_y = ay;
        }
        if (z_off != 0) {
          n_args_z_offset++;
          int az = z_off < 0 ? -z_off : z_off;
          if (az > max_radius_z) max_radius_z = az;
        }
        total_offset_points++;
      }
    }

    // ---- Derived features (like mlp_model.py::add_derived_features) ----
    const double sz_x = static_cast<double>(max_size_x);
    const double sz_y = static_cast<double>(std::max(max_size_y, 1));
    const double sz_z = static_cast<double>(std::max(max_size_z, 1));
    const double total_points = sz_x * sz_y * sz_z;
    const double ratio_xy = sz_x / sz_y;
    const double ratio_xz = sz_x / sz_z;
    const double bytes_per_point = total_points > 0 ? total_bytes / total_points : 0.0;
    const float is_2d = (max_size_z <= 1) ? 1.0f : 0.0f;

    // Dominant stencil axis: 0=none, 1=X, 2=Y, 3=Z
    int stencil_counts[3] = {n_args_x_offset, n_args_y_offset, n_args_z_offset};
    int max_sc = *std::max_element(stencil_counts, stencil_counts + 3);
    int dominant = 0;
    if (max_sc > 0) {
      // argmax + 1: 1=X, 2=Y, 3=Z
      dominant = static_cast<int>(std::max_element(stencil_counts, stencil_counts + 3) - stencil_counts) + 1;
    }
    const float collapse_x = (dominant == 1) ? 1.0f : 0.0f;
    const float collapse_y = (dominant == 2) ? 1.0f : 0.0f;
    const float collapse_z = (dominant == 3) ? 1.0f : 0.0f;
    const float is_face_access = (n_args_face > 0) ? 1.0f : 0.0f;
    const float is_high_complexity = (nstencil_args >= 12) ? 1.0f : 0.0f;
    const float face_low_args  = (n_args_face > 0 && nstencil_args <= 5) ? 1.0f : 0.0f;
    const float face_high_args = (n_args_face > 0 && nstencil_args >= 12) ? 1.0f : 0.0f;

    // ---- Fill feature vector (exact order matching Python ALL_FEATURE_COLS) ----
    int fi = 0;
    // BASE_FEATURES (20)
    features[fi++] = static_cast<float>(nstencil_args);       //  0: nstencil_args
    features[fi++] = static_cast<float>(total_bytes);          //  1: total_bytes
    features[fi++] = static_cast<float>(max_dim);              //  2: dim
    features[fi++] = static_cast<float>(max_elem_size);        //  3: elem_size
    features[fi++] = static_cast<float>(max_size_x);           //  4: size_x
    features[fi++] = static_cast<float>(max_size_y);           //  5: size_y
    features[fi++] = static_cast<float>(max_size_z);           //  6: size_z
    features[fi++] = static_cast<float>(min_dm_x);             //  7: d_m_x
    features[fi++] = static_cast<float>(min_dm_y);             //  8: d_m_y
    features[fi++] = static_cast<float>(min_dm_z);             //  9: d_m_z
    features[fi++] = static_cast<float>(max_dp_x);             // 10: d_p_x
    features[fi++] = static_cast<float>(max_dp_y);             // 11: d_p_y
    features[fi++] = static_cast<float>(max_dp_z);             // 12: d_p_z
    features[fi++] = static_cast<float>(num_read);             // 13: num_read
    features[fi++] = static_cast<float>(num_write);            // 14: num_write
    features[fi++] = static_cast<float>(num_rw);               // 15: num_rw
    features[fi++] = grid_x;                                   // 16: grid_x
    features[fi++] = grid_y;                                   // 17: grid_y
    features[fi++] = grid_z;                                   // 18: grid_z
    features[fi++] = static_cast<float>(max_threads_per_block);// 19: max_threads

    // STENCIL_FEATURES (10)
    features[fi++] = static_cast<float>(n_args_x_offset);     // 20: n_args_x_offset
    features[fi++] = static_cast<float>(n_args_y_offset);     // 21: n_args_y_offset
    features[fi++] = static_cast<float>(n_args_z_offset);     // 22: n_args_z_offset
    features[fi++] = static_cast<float>(n_args_face);          // 23: n_args_face
    features[fi++] = static_cast<float>(n_args_stride);        // 24: n_args_stride
    features[fi++] = static_cast<float>(n_args_none);          // 25: n_args_none
    features[fi++] = static_cast<float>(max_radius_x);        // 26: max_radius_x
    features[fi++] = static_cast<float>(max_radius_y);        // 27: max_radius_y
    features[fi++] = static_cast<float>(max_radius_z);        // 28: max_radius_z
    features[fi++] = static_cast<float>(total_offset_points); // 29: total_offset_points

    // DERIVED_FEATURES (13)
    features[fi++] = static_cast<float>(total_points);         // 30: total_points
    features[fi++] = static_cast<float>(ratio_xy);             // 31: ratio_xy
    features[fi++] = static_cast<float>(ratio_xz);             // 32: ratio_xz
    features[fi++] = static_cast<float>(bytes_per_point);      // 33: bytes_per_point
    features[fi++] = is_2d;                                    // 34: is_2d
    features[fi++] = static_cast<float>(dominant);             // 35: dominant_stencil_axis
    features[fi++] = collapse_x;                               // 36: collapse_x
    features[fi++] = collapse_y;                               // 37: collapse_y
    features[fi++] = collapse_z;                               // 38: collapse_z
    features[fi++] = is_face_access;                           // 39: is_face_access
    features[fi++] = is_high_complexity;                       // 40: is_high_complexity
    features[fi++] = face_low_args;                            // 41: face_low_args
    features[fi++] = face_high_args;                           // 42: face_high_args
    // fi == 43 == ML_NUM_FEATURES_EXPECTED
  }

  // ---- ML prediction dispatcher ----

  ops::dim3 ml_predict_block_size(int mode, int kernel_id, int ndims,
                                   int* local_range, int nargs, ops_arg* args,
                                   int max_threads) {
    // Extract features
    float features[ML_NUM_FEATURES_EXPECTED];
    extract_kernel_features(kernel_id, ndims, local_range, nargs, args,
                            max_threads, features);

    // Store features in TuneState for CSV export
    auto& tune_st = g_tune[kernel_id];
    std::memcpy(tune_st.ml_features, features, sizeof(features));
    tune_st.ml_features_set = true;

    int predicted_class = -1;

    if (mode == 4) {
#ifdef OPS_ML_XGBOOST
      // XGBoost multi-output inference
      if (!load_xgboost_model()) {
        printf("[OPS] WARNING: XGBoost model not available, falling back to default\n");
        return ops::dim3(
          OPS_instance::getOPSInstance()->OPS_block_size_x,
          ndims > 1 ? OPS_instance::getOPSInstance()->OPS_block_size_y : 1,
          ndims > 2 ? OPS_instance::getOPSInstance()->OPS_block_size_z : 1);
      }
      ops::dim3 result = xgboost_predict_multi(features, ML_NUM_FEATURES_EXPECTED);
      printf("[OPS-XGB] kernel %d -> (%u,%u,%u)\n",
             kernel_id, result.x, result.y, result.z);
      return result;
#else
      printf("[OPS] ERROR: XGBoost mode (4) not compiled. Rebuild with -DOPS_ML_XGBOOST and link -lxgboost\n");
      return ops::dim3(
        OPS_instance::getOPSInstance()->OPS_block_size_x,
        ndims > 1 ? OPS_instance::getOPSInstance()->OPS_block_size_y : 1,
        ndims > 2 ? OPS_instance::getOPSInstance()->OPS_block_size_z : 1);
#endif

    } else if (mode == 6) {
#ifdef OPS_ML_XGBOOST
      // XGBoost single-output inference (231 combined classes)
      if (!load_xgboost_single_model()) {
        printf("[OPS] WARNING: XGBoost single model not available, falling back to default\n");
        return ops::dim3(
          OPS_instance::getOPSInstance()->OPS_block_size_x,
          ndims > 1 ? OPS_instance::getOPSInstance()->OPS_block_size_y : 1,
          ndims > 2 ? OPS_instance::getOPSInstance()->OPS_block_size_z : 1);
      }
      ops::dim3 result = xgboost_predict_single(features, ML_NUM_FEATURES_EXPECTED);
      printf("[OPS-XGB-SINGLE] kernel %d -> (%u,%u,%u)\n",
             kernel_id, result.x, result.y, result.z);
      return result;
#else
      printf("[OPS] ERROR: XGBoost mode (6) not compiled. Rebuild with -DOPS_ML_XGBOOST and link -lxgboost\n");
      return ops::dim3(
        OPS_instance::getOPSInstance()->OPS_block_size_x,
        ndims > 1 ? OPS_instance::getOPSInstance()->OPS_block_size_y : 1,
        ndims > 2 ? OPS_instance::getOPSInstance()->OPS_block_size_z : 1);
#endif
    }

    // Should not reach here
    return ops::dim3(32, 1, 1);
  }

  // =========================================================================
  // End ML Inference
  // =========================================================================


  std::string get_app_name() {
  std::string app_name = "unknown";
  #if defined(__linux__)
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
      exe_path[len] = '\0';
      std::string full(exe_path);
      size_t pos = full.find_last_of('/');
      app_name = (pos == std::string::npos) ? full : full.substr(pos + 1);
    }
  #endif
  return app_name;
}


bool is_autotune(){
  OPS_instance *inst = OPS_instance::getOPSInstance();
  return (!inst || inst->OPS_autotune_mode != 0);
}


void ensure_dir_exists(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    if (S_ISDIR(st.st_mode)) return;   

    printf("[OPS] WARNING: path exists but is not a directory: %s\n", path.c_str());
    return;
  }
  // No existe: intentar crearlo
  if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
    printf("[OPS] ERROR: failed to create directory %s: %s\n", path.c_str(), strerror(errno));

  }
}

  void ensure_log_open() {
    std::call_once(g_log_once, [](){
      OPS_instance *inst = OPS_instance::getOPSInstance();
      bool autotune = is_autotune();
      std::string app_name = get_app_name();
      // Use OPS_LOGS_DIR env var if set, otherwise default to /home/kevin/OPS_LOGS
      const char* env_logs_dir = std::getenv("OPS_LOGS_DIR");
      std::string base_dir = env_logs_dir ? std::string(env_logs_dir) : "/home/kevin/OPS_LOGS";
      std::string app_dir;
      if (env_logs_dir) {
        // Custom dir: write directly to base_dir/autotune_on or autotune_off
        app_dir = base_dir + (autotune ? "/autotune_on" : "/autotune_off");
        ensure_dir_exists(base_dir);
        ensure_dir_exists(app_dir);
      } else {
        // Default: write to base_dir/app_name/autotune_on or autotune_off
        std::string app_root = base_dir + "/" + app_name;
        app_dir = app_root + (autotune ? "/autotune_on" : "/autotune_off");
        ensure_dir_exists(base_dir);
        ensure_dir_exists(app_root);
        ensure_dir_exists(app_dir);
      }
      // printf("[OPS] Logging opening %s\n", app_dir.c_str());
      g_log.open(app_dir + "/ops_blocksize_tuning_logmod.csv",
                std::ios::out | std::ios::app);
      if (g_log.tellp() == 0) {
        g_log << "kernel_id,bx,by,bz,"
                "nstencil_args,total_bytes,dim,elem_size,"
                "size_x,size_y,size_z,"
                "d_m_x,d_m_y,d_m_z,d_p_x,d_p_y,d_p_z,"
                "num_read,num_write,num_rw,"
                "grid_x,grid_y,grid_z,max_threads,"
                "n_args_x_offset,n_args_y_offset,n_args_z_offset,"
                "n_args_face,n_args_stride,n_args_none,"
                "max_radius_x,max_radius_y,max_radius_z,total_offset_points,"
                "total_points,ratio_xy,ratio_xz,bytes_per_point,"
                "is_2d,dominant_stencil_axis,"
                "collapse_x,collapse_y,collapse_z,"
                "is_face_access,is_high_complexity,face_low_args,face_high_args,"
                "stencil_sig,execution_time,default_time,best_time,label,pred_time,pred_rank"
              << std::endl;
      }
    });
  }

  void ensure_best_log_open() {
    std::call_once(g_best_log_once, [](){
      OPS_instance *inst = OPS_instance::getOPSInstance();
      bool autotune = is_autotune();
      std::string app_name = get_app_name();
      // Use OPS_LOGS_DIR env var if set, otherwise default to /home/kevin/OPS_LOGS
      const char* env_logs_dir = std::getenv("OPS_LOGS_DIR");
      std::string base_dir = env_logs_dir ? std::string(env_logs_dir) : "/home/kevin/OPS_LOGS";
      std::string app_dir;
      if (env_logs_dir) {
        // Custom dir: write directly to base_dir/autotune_on or autotune_off
        app_dir = base_dir + (autotune ? "/autotune_on" : "/autotune_off");
        ensure_dir_exists(base_dir);
        ensure_dir_exists(app_dir);
      } else {
        // Default: write to base_dir/app_name/autotune_on or autotune_off
        std::string app_root = base_dir + "/" + app_name;
        app_dir = app_root + (autotune ? "/autotune_on" : "/autotune_off");
        ensure_dir_exists(base_dir);
        ensure_dir_exists(app_root);
        ensure_dir_exists(app_dir);
      }

      g_best_log.open(app_dir + "/ops_blocksize_best_logmod.csv",
                std::ios::out | std::ios::app);
      if (g_best_log.tellp() == 0) {
      
        g_best_log << "kernel_id,bx,by,bz,best_time,default_time,points,gpoints_per_s,nargs,nstencil_args,widest_radius,widest_radius_x,widest_radius_y,widest_radius_z,stencil_sig,max_threads_per_block" << std::endl;
      }
    });
  }

  // Arrays metadata log
  std::ofstream g_arrays_log;
  std::once_flag g_arrays_log_once;

  void ensure_arrays_log_open() {
    std::call_once(g_arrays_log_once, [](){
      bool autotune = is_autotune();
      const char* env_logs_dir = std::getenv("OPS_LOGS_DIR");
      std::string base_dir = env_logs_dir ? std::string(env_logs_dir) : "/home/kevin/OPS_LOGS";
      std::string app_dir;
      if (env_logs_dir) {
        app_dir = base_dir + (autotune ? "/autotune_on" : "/autotune_off");
      } else {
        std::string app_name = get_app_name();
        std::string app_root = base_dir + "/" + app_name;
        app_dir = app_root + (autotune ? "/autotune_on" : "/autotune_off");
      }
      ensure_dir_exists(app_dir);
      
      g_arrays_log.open(app_dir + "/ops_arrays_metadata.csv",
                std::ios::out | std::ios::trunc);
      g_arrays_log << "kernel_id,arg_idx,array_name,data_type,dim,elem_size,"
                   << "size_x,size_y,size_z,d_m_x,d_m_y,d_m_z,d_p_x,d_p_y,d_p_z,"
                   << "stride_x,stride_y,stride_z,total_bytes,access_type" << std::endl;
    });
  }

  void write_arrays_metadata(int kernel_id, const std::vector<ArrayInfo>& arrays) {
    ensure_arrays_log_open();
    if (!g_arrays_log.is_open()) return;
    
    const char* access_names[] = {"READ", "WRITE", "RW", "INC", "MIN", "MAX"};
    
    for (const auto& arr : arrays) {
      const char* acc_str = (arr.access >= 0 && arr.access <= 5) ? access_names[arr.access] : "UNKNOWN";
      g_arrays_log << kernel_id << ","
                   << arr.arg_idx << ","
                   << arr.name << ","
                   << arr.type << ","
                   << arr.dim << ","
                   << arr.elem_size << ","
                   << arr.size[0] << "," << arr.size[1] << "," << arr.size[2] << ","
                   << arr.d_m[0] << "," << arr.d_m[1] << "," << arr.d_m[2] << ","
                   << arr.d_p[0] << "," << arr.d_p[1] << "," << arr.d_p[2] << ","
                   << arr.stride[0] << "," << arr.stride[1] << "," << arr.stride[2] << ","
                   << arr.total_bytes << ","
                   << acc_str << std::endl;
    }
  }

  // Arrays metadata log for BEST kernels only
  std::ofstream g_arrays_best_log;
  std::once_flag g_arrays_best_log_once;

  void ensure_arrays_best_log_open() {
    std::call_once(g_arrays_best_log_once, [](){
      bool autotune = is_autotune();
      const char* env_logs_dir = std::getenv("OPS_LOGS_DIR");
      std::string base_dir = env_logs_dir ? std::string(env_logs_dir) : "/home/kevin/OPS_LOGS";
      std::string app_dir;
      if (env_logs_dir) {
        app_dir = base_dir + (autotune ? "/autotune_on" : "/autotune_off");
      } else {
        std::string app_name = get_app_name();
        std::string app_root = base_dir + "/" + app_name;
        app_dir = app_root + (autotune ? "/autotune_on" : "/autotune_off");
      }
      ensure_dir_exists(app_dir);
      
      g_arrays_best_log.open(app_dir + "/ops_arrays_metadata_best.csv",
                std::ios::out | std::ios::trunc);
      g_arrays_best_log << "kernel_id,arg_idx,array_name,data_type,dim,elem_size,"
                   << "size_x,size_y,size_z,d_m_x,d_m_y,d_m_z,d_p_x,d_p_y,d_p_z,"
                   << "stride_x,stride_y,stride_z,total_bytes,access_type" << std::endl;
    });
  }

  void write_arrays_metadata_best(int kernel_id, const std::vector<ArrayInfo>& arrays) {
    ensure_arrays_best_log_open();
    if (!g_arrays_best_log.is_open()) return;
    
    const char* access_names[] = {"READ", "WRITE", "RW", "INC", "MIN", "MAX"};
    
    for (const auto& arr : arrays) {
      const char* acc_str = (arr.access >= 0 && arr.access <= 5) ? access_names[arr.access] : "UNKNOWN";
      g_arrays_best_log << kernel_id << ","
                   << arr.arg_idx << ","
                   << arr.name << ","
                   << arr.type << ","
                   << arr.dim << ","
                   << arr.elem_size << ","
                   << arr.size[0] << "," << arr.size[1] << "," << arr.size[2] << ","
                   << arr.d_m[0] << "," << arr.d_m[1] << "," << arr.d_m[2] << ","
                   << arr.d_p[0] << "," << arr.d_p[1] << "," << arr.d_p[2] << ","
                   << arr.stride[0] << "," << arr.stride[1] << "," << arr.stride[2] << ","
                   << arr.total_bytes << ","
                   << acc_str << std::endl;
    }
    g_arrays_best_log.flush();
  }

  // Flag to require bx > by (and bx > bz in 3D) - read from OPS_AUTOTUNE_BX_PRIORITY env var
  static bool g_bx_priority_checked = false;
  static bool g_bx_priority = false;
  
  bool get_bx_priority() {
    if (!g_bx_priority_checked) {
      const char* env = std::getenv("OPS_AUTOTUNE_BX_PRIORITY");
      g_bx_priority = (env && std::string(env) == "1");
      g_bx_priority_checked = true;
    }
    return g_bx_priority;
  }

  inline int clamp_positive(int v) { return v < 1 ? 1 : v; }

  void build_candidates_1d(TuneState &st, int x_extent, int max_threads_hw) {
    const int max_threads = std::min(1024, max_threads_hw);
    const int max_x = std::min(x_extent, max_threads);
    for (int bx = 1; bx <= max_x; bx <<= 1) {
      if (bx <= max_threads && bx >= 32) st.candidates.emplace_back(bx, 1, 1);
      if (bx == 0) break; // safety for overflow
    }
  }

  void build_candidates_2d(TuneState &st, int x_extent, int y_extent, int max_threads_hw) {
    const int max_threads = std::min(1024, max_threads_hw);
    const int max_x = std::min(x_extent, max_threads);
    const int max_y = std::min(y_extent, max_threads);
    const bool bx_priority = get_bx_priority();
    for (int by = 1; by <= max_y; by <<= 1) {
      for (int bx = 1; bx <= max_x; bx <<= 1) {
        const int prod = bx * by;
        bool shape_ok = !bx_priority || (bx >= by);
        if (prod <= max_threads && prod >= 32 && shape_ok)
          st.candidates.emplace_back(bx, by, 1);
        if (bx == 0) break; // safety for overflow
      }
      if (by == 0) break; // safety for overflow
    }
  }

  void build_candidates_3d(TuneState &st, int x_extent, int y_extent, int z_extent, int max_threads_hw) {
    const int max_threads = std::min(1024, max_threads_hw);
    const int max_x = std::min(x_extent, max_threads);
    const int max_y = std::min(y_extent, max_threads);
    const int max_z = std::min(std::min(z_extent, max_threads), 64);
    const bool bx_priority = get_bx_priority();
    for (int bz = 1; bz <= max_z; bz <<= 1) {
      for (int by = 1; by <= max_y; by <<= 1) {
        for (int bx = 1; bx <= max_x; bx <<= 1) {
          const long long prod = 1LL * bx * by * bz;
          bool shape_ok = !bx_priority || (bx >= by && bx >= bz);
          if (prod <= max_threads && prod >= 32 && shape_ok)
            st.candidates.emplace_back(bx, by, bz);
          if (bx == 0) break; // safety for overflow
        }
        if (by == 0) break; // safety for overflow
      }
      if (bz == 0) break; // safety for overflow
    }
  }
}

ops::dim3 ops_get_kernel_block_size(int kernel_id, int ndims, int *local_range, int nargs, ops_arg* args, int max_threads, int registers) {


  (void)nargs; (void)args; (void)registers;
  //std::lock_guard<std::mutex> lock(g_mutex);
  ensure_log_open();


  auto &st = g_tune[kernel_id]; //replicate (instead of find)
  st.max_threads_per_block = max_threads;  // Store real limit per kernel

  // Compute extents from local_range
  const int x_extent = ndims >= 1 ? clamp_positive(local_range[1] - local_range[0]) : 1;
  const int y_extent = ndims >= 2 ? clamp_positive(local_range[3] - local_range[2]) : 1;
  const int z_extent = ndims >= 3 ? clamp_positive(local_range[5] - local_range[4]) : 1;
  st.x_extent = x_extent;
  st.y_extent = y_extent;
  st.z_extent = z_extent;

  // Capture kernel metadata once (argument count, stencil usage, widest stencil, signature, array info)
  if (!st.meta_set) {
    st.nargs_meta = nargs;
    st.nstencil_args = 0;
    st.widest_radius = 0;
    st.widest_radius_x = 0;
    st.widest_radius_y = 0;
    st.widest_radius_z = 0;
    st.ndims_kernel = ndims;
    // Store dimension-adjusted default block
    {
      OPS_instance *def_inst = OPS_instance::getOPSInstance();
      st.default_block = ops::dim3(
          def_inst->OPS_block_size_x,
          (ndims > 1) ? def_inst->OPS_block_size_y : 1,
          (ndims > 2) ? def_inst->OPS_block_size_z : 1);
    }
    std::ostringstream ss;
    for (int i = 0; i < nargs; ++i) {
      ops_arg &a = args[i];
      
      // Capture array (ops_dat) metadata if this is a DAT argument
      if (a.argtype == OPS_ARG_DAT && a.dat != nullptr) {
        ArrayInfo arr_info;
        arr_info.arg_idx = i;
        arr_info.name = a.dat->name ? a.dat->name : "unnamed";
        arr_info.type = a.dat->type ? a.dat->type : "unknown";
        arr_info.dim = a.dat->dim;
        arr_info.elem_size = a.dat->elem_size;
        arr_info.total_bytes = a.dat->mem;
        arr_info.access = a.acc;
        
        // Copy size, halo depths, and compute strides
        for (int d = 0; d < OPS_MAX_DIM; ++d) {
          arr_info.size[d] = a.dat->size[d];
          arr_info.d_m[d] = a.dat->d_m[d];
          arr_info.d_p[d] = a.dat->d_p[d];
          arr_info.stride[d] = a.dat->stride[d];
        }
        
        st.arrays.push_back(arr_info);
      }
      
      if (a.stencil != nullptr) {
        st.nstencil_args++;
        int dims = ndims; // default to kernel dims
        // If ops_stencil exposes dims, prefer it
        // NOTE: using pointer checks to avoid UB if fields differ across builds
        // Compute maximal Chebyshev radius among stencil points
        int max_rad = 0;
        int max_rad_per_axis[3] = {0, 0, 0};
        if (a.stencil->points > 0 && a.stencil->stencil != nullptr) {
          // Heuristic: try a.stencil->dims if available via sizeof trick is not possible here; assume ndims
          for (int p = 0; p < a.stencil->points; ++p) {
            for (int d = 0; d < dims; ++d) {
              int off = a.stencil->stencil[p*dims + d];
              int abso = off < 0 ? -off : off;
              if (abso > max_rad) max_rad = abso;
              if (d < 3 && abso > max_rad_per_axis[d]) max_rad_per_axis[d] = abso;
            }
          }
        }
        if (max_rad > st.widest_radius) st.widest_radius = max_rad;
        if (max_rad_per_axis[0] > st.widest_radius_x) st.widest_radius_x = max_rad_per_axis[0];
        if (max_rad_per_axis[1] > st.widest_radius_y) st.widest_radius_y = max_rad_per_axis[1];
        if (max_rad_per_axis[2] > st.widest_radius_z) st.widest_radius_z = max_rad_per_axis[2];
        if (a.stencil->name && a.stencil->name[0] != '\0') {
          ss << a.stencil->name;
        } else {
          ss << "rad" << max_rad;
        }
      } else {
        ss << "none";
      }
      if (i + 1 < nargs) ss << ';';
    }
    st.stencil_sig = ss.str();
    st.meta_set = true;
    
    // Write arrays metadata to CSV
    if (!st.arrays.empty()) {
      write_arrays_metadata(kernel_id, st.arrays);
    }
  }

  if (st.candidates.empty()) {
    OPS_instance *inst = OPS_instance::getOPSInstance();
    const int mode = inst ? inst->OPS_autotune_mode : 1;

    // Always extract ML features for CSV logging (all modes, not just ML modes)
    if (!st.ml_features_set) {
      float features[ML_NUM_FEATURES_EXPECTED];
      extract_kernel_features(kernel_id, ndims, local_range, nargs, args,
                              max_threads, features);
      std::memcpy(st.ml_features, features, sizeof(features));
      st.ml_features_set = true;
    }

    // printf("mode : %d\n", mode);
    if (mode == 0) {
      // printf("[OPS] using default block sizes ...\n");
      // Modo bloques por defecto: usar solo OPS_BLOCK_SIZE_* como candidato único
      unsigned int gx = OPS_instance::getOPSInstance()->OPS_block_size_x;
      unsigned int gy = (ndims > 1) ? OPS_instance::getOPSInstance()->OPS_block_size_y : 1;
      unsigned int gz = (ndims > 2) ? OPS_instance::getOPSInstance()->OPS_block_size_z : 1;
      st.candidates.clear();
      st.candidates.emplace_back(gx, gy, gz);
      st.best = st.candidates.front();
      st.next_idx = 0;
    } else if (mode == 2) {
      // Mode 2: Use precomputed block sizes from CSV
      load_precomputed_blocks();
      auto it = g_precomputed_blocks.find(kernel_id);
      if (it != g_precomputed_blocks.end()) {
        st.candidates.emplace_back(it->second);
        st.best = it->second;
        st.decided = true;
      } else {
        // Fallback to default if kernel not in CSV
        unsigned int gx = inst->OPS_block_size_x;
        unsigned int gy = (ndims > 1) ? inst->OPS_block_size_y : 1;
        unsigned int gz = (ndims > 2) ? inst->OPS_block_size_z : 1;
        st.candidates.emplace_back(gx, gy, gz);
        st.best = st.candidates.front();
        st.decided = true;
        printf("[OPS] WARNING: kernel %d not found in CSV, using default (%u,%u,%u)\n", 
               kernel_id, gx, gy, gz);
      }
      st.next_idx = 0;
    } else if (mode == 4 || mode == 6) {
      // Mode 4: XGBoost multi-output | Mode 6: XGBoost single-output
      // The model receives the 43 features directly and predicts (bx, by, bz)
      ops::dim3 predicted = ml_predict_block_size(
        mode, kernel_id, ndims, local_range, nargs, args, max_threads);
      st.candidates.emplace_back(predicted);
      st.best = predicted;
      st.decided = true;
      st.next_idx = 0;
    } else if (mode == 5) {
#ifdef OPS_ML_XGBOOST
      // Mode 5: ML-guided exploration — same candidate space as Mode 1, ML-ranked
      // Guard: only build candidates ONCE per kernel (explore_started stays true)
      if (!st.explore_started) {
        if (!load_xgboost_model()) {
          // Fallback to default
          unsigned int gx = inst->OPS_block_size_x;
          unsigned int gy = (ndims > 1) ? inst->OPS_block_size_y : 1;
          unsigned int gz = (ndims > 2) ? inst->OPS_block_size_z : 1;
          st.candidates.emplace_back(gx, gy, gz);
          st.best = st.candidates.front();
          st.decided = true;
          st.next_idx = 0;
        } else {
          // Build all valid candidates (powers of 2, same as Mode 1) ranked by ML probability
          build_explore_candidates(st, st.ml_features, ML_NUM_FEATURES_EXPECTED, ndims, st.max_threads_per_block);
          st.explore_started = true;
        }
      }
#else
      printf("[OPS] ERROR: Mode 5 requires XGBoost. Rebuild with -DOPS_ML_XGBOOST\n");
      unsigned int gx = inst->OPS_block_size_x;
      unsigned int gy = (ndims > 1) ? inst->OPS_block_size_y : 1;
      unsigned int gz = (ndims > 2) ? inst->OPS_block_size_z : 1;
      st.candidates.emplace_back(gx, gy, gz);
      st.best = st.candidates.front();
      st.decided = true;
      st.next_idx = 0;
#endif
    } else {
      // Mode 1: Autotune — build all candidates and try each one
      if (ndims <= 1) {
        build_candidates_1d(st, x_extent, max_threads);
      } else if (ndims == 2) {
        build_candidates_2d(st, x_extent, y_extent, max_threads);
      } else {
        build_candidates_3d(st, x_extent, y_extent, z_extent, max_threads);
      }

      // Always include default block as a candidate so the autotune
      // guarantees the best is at least as fast as the default
      if (inst) {
        unsigned int gx = inst->OPS_block_size_x;
        unsigned int gy = (ndims > 1) ? inst->OPS_block_size_y : 1;
        unsigned int gz = (ndims > 2) ? inst->OPS_block_size_z : 1;
        ops::dim3 def_block(gx, gy, gz);
        bool already_present = false;
        for (const auto &c : st.candidates) {
          if (c.x == def_block.x && c.y == def_block.y && c.z == def_block.z) {
            already_present = true;
            break;
          }
        }
        if (!already_present) {
          st.candidates.emplace_back(def_block);
        }
      }
    }
  }

#ifdef OPS_ML_XGBOOST
  // Mode 5: serve candidates in ML-ranked order, all valid blocks (same space as Mode 1)
  {
    OPS_instance *mode_inst = OPS_instance::getOPSInstance();
    if (mode_inst && mode_inst->OPS_autotune_mode == 5 && st.explore_started) {
      if (st.explore_converged) {
        // Already done — serve best
        st.last_served = st.explore_best;
        st.last_points = 1LL * x_extent * y_extent * z_extent;
        return st.explore_best;
      }

      if (st.explore_idx < st.explore_candidates.size()) {
        // Serve next ML-ranked candidate
        st.last_served = st.explore_candidates[st.explore_idx++];
        st.last_points = 1LL * x_extent * y_extent * z_extent;
        return st.last_served;
      }

      // All candidates tested — converge with the best found
      st.explore_converged = true;
      st.decided = true;
      st.best = st.explore_best;
      st.best_time = st.explore_best_time;

      // Find rank of actual best in ML ordering (1-based)
      st.explore_best_rank = 0;
      for (size_t ri = 0; ri < st.explore_candidates.size(); ri++) {
        if (st.explore_candidates[ri].x == st.explore_best.x &&
            st.explore_candidates[ri].y == st.explore_best.y &&
            st.explore_candidates[ri].z == st.explore_best.z) {
          st.explore_best_rank = (int)ri + 1;
          break;
        }
      }

      printf("[OPS-EXPLORE] kernel %d DONE -> (%u,%u,%u) time=%.6f (%zu candidates tested)\n",
             kernel_id, st.explore_best.x, st.explore_best.y, st.explore_best.z,
             st.explore_best_time, st.explore_candidates.size());
      printf("[OPS-EXPLORE] kernel %d ML_PRED=(%u,%u,%u) pred_time=%.6f best_rank=%d/%zu\n",
             kernel_id,
             st.explore_candidates[0].x, st.explore_candidates[0].y, st.explore_candidates[0].z,
             st.explore_pred_time, st.explore_best_rank, st.explore_candidates.size());

      // Write best CSV — only if best is different from default block
      {
        bool is_default = (st.best.x == st.default_block.x &&
                           st.best.y == st.default_block.y &&
                           st.best.z == st.default_block.z);
        if (!is_default) {
          ensure_best_log_open();
          const double best_gpoints = (st.best_time > 0.0 && st.best_points > 0)
                                      ? (st.best_points / st.best_time) / 1e9 : 0.0;
          if (g_best_log.good()) {
            g_best_log << kernel_id << ',' << st.best.x << ',' << st.best.y << ',' << st.best.z << ','
                       << st.best_time << ','
                       << st.default_time << ','
                       << st.best_points << ',' << best_gpoints << ','
                       << st.nargs_meta << ',' << st.nstencil_args << ','
                       << st.widest_radius << ',' << st.widest_radius_x << ',' << st.widest_radius_y << ',' << st.widest_radius_z << ','
                       << '"' << st.stencil_sig << '"' << ',' << st.max_threads_per_block << std::endl;
            g_best_log.flush();
            g_best_rows++;
          }
        }
      }

      // Buffer training sample for online model update
      buffer_explore_sample(st, kernel_id);
      st.last_served = st.explore_best;
      st.last_points = 1LL * x_extent * y_extent * z_extent;
      return st.explore_best;
    }
  }
#endif

  if (st.decided) {
    st.last_served = st.best;
    st.last_points = 1LL * x_extent * y_extent * z_extent;
    return st.best;
  }

  // Serve next candidate
  if (st.next_idx < st.candidates.size()) {
    st.last_served = st.candidates[st.next_idx++];
    st.last_points = 1LL * x_extent * y_extent * z_extent;
    return st.last_served;
  }

  // If ran out (should not happen without performance records), fall back to best or first
  st.decided = true;
  if (st.best.x == 0) st.best = st.candidates.front();
  st.last_served = st.best;
  st.last_points = 1LL * x_extent * y_extent * z_extent;
  return st.best;
}


// New API variant that can record both total and MPI (remote) time per kernel.
// Existing callers can continue to use ops_record_kernel_performance, which
// delegates here with seconds_mpi = 0.0.
void ops_record_kernel_performance(int kernel_id, double seconds_total) {

  //std::lock_guard<std::mutex> lock(g_mutex);
  ensure_log_open();
  auto it = g_tune.find(kernel_id);
  auto &st = it->second;
  const ops::dim3 bs = st.last_served;
  const long long points = st.last_points > 0 ? st.last_points : 0;
  const double gpoints = (seconds_total > 0.0 && points > 0) ? (points / seconds_total) / 1e9 : 0.0;


  // CSV log: kernel_id,bx,by,bz,execution_time,mpi_time,mpi_fraction,points,gpoints_per_s
  if (g_log.good()) {
    g_log << kernel_id << ',' << bs.x << ',' << bs.y << ',' << bs.z;
    // 43 ML features (same order as enriched autotune CSVs)
    for (int fi = 0; fi < 43; ++fi) {
      g_log << ',' << st.ml_features[fi];
    }
    g_log << ',' << '"' << st.stencil_sig << '"'
          << ',' << seconds_total
          << ',' << st.default_time
          << ',' << st.best_time
          << ',' << (st.decided && st.best_time < 1e100 ? 1 : 0)
          << ',' << st.explore_pred_time
          << ',' << st.explore_best_rank
          << std::endl;
    g_log.flush();
    g_tuning_rows++;
  }

  // Track default time using dimension-adjusted default block
  if (st.default_block.x == bs.x &&
      st.default_block.y == bs.y &&
      st.default_block.z == bs.z) {
      if (seconds_total > 0.0 && (st.default_time <= 0.0 || st.default_time > 1e100 || seconds_total < st.default_time)) {
          st.default_time = seconds_total;
      }
  }

  // Track best - only update if this is faster
  if (seconds_total > 0.0 && (st.best_time <= 0.0 || seconds_total < st.best_time)) {
    st.best_time = seconds_total;
    st.best = bs;
    st.best_points = points;

  }

#ifdef OPS_ML_XGBOOST
  // Mode 5: track exploration timing
  {
    OPS_instance *ex_inst = OPS_instance::getOPSInstance();
    if (ex_inst && ex_inst->OPS_autotune_mode == 5 && st.explore_started && !st.explore_converged) {
      // Track time of ML top-1 prediction (first candidate served = explore_idx was 1 after serving)
      if (st.explore_idx == 1 && seconds_total > 0.0) {
        st.explore_pred_time = seconds_total;
      }
      if (seconds_total > 0.0 && seconds_total < st.explore_best_time) {
        st.explore_best_time = seconds_total;
        st.explore_best = bs;
      }
      printf("[OPS-EXPLORE] kernel %d [%zu/%zu] (%u,%u,%u) time=%.6f %s\n",
             kernel_id,
             st.explore_idx, st.explore_candidates.size(),
             bs.x, bs.y, bs.z, seconds_total,
             (seconds_total > 0.0 && seconds_total <= st.explore_best_time) ? "<= BEST" : "");
    }
  }
#endif

  // Decide when all candidates have been tested (Mode 1 only, Mode 5 handles its own convergence)
  if (!st.decided && !st.explore_started && st.next_idx >= st.candidates.size()) {
    st.decided = true;
    // Optional: brief stdout summary
    // printf("[OPS autotune] v1.2 kernel %d best block = (%u,%u,%u) time = %.6f s \n",
    //        kernel_id, st.best.x, st.best.y, st.best.z, st.best_time,
    //        (st.best_time > 0.0 ? st.best_time : 0.0));

    // Write best-only CSV summary — only if best is strictly better than default
    bool is_default_block = (st.best.x == st.default_block.x &&
        st.best.y == st.default_block.y &&
        st.best.z == st.default_block.z);

    {
      ensure_best_log_open();
      const double best_gpoints = (st.best_time > 0.0 && st.best_points > 0)
                                  ? (st.best_points / st.best_time) / 1e9 : 0.0;

      if (g_best_log.good()) {
        g_best_log << kernel_id << ',' << st.best.x << ',' << st.best.y << ',' << st.best.z << ','
                   << st.best_time << ','
                   << st.default_time << ','
                   << st.best_points << ',' << best_gpoints << ','
                   << st.nargs_meta << ',' << st.nstencil_args << ','
                   << st.widest_radius << ',' << st.widest_radius_x << ',' << st.widest_radius_y << ',' << st.widest_radius_z << ','
                   << '"' << st.stencil_sig << '"' << ',' << st.max_threads_per_block << std::endl;
        g_best_log.flush();
        g_best_rows++;
      }
    }
    
    // Write arrays metadata for best kernels only
    if (!st.arrays.empty()) {
      write_arrays_metadata_best(kernel_id, st.arrays);
    }
  }

}




void ops_flush_autotune_logs() {
  
  std::string app_name = get_app_name();
  bool autotune = is_autotune();


  if (g_tuning_rows > 0 && g_log.good()) {
    g_log << std::endl << std::endl << std::endl;
    g_log << "Autotune -- " << autotune << std::endl;
    g_log << "total of rows -- " << g_tuning_rows << std::endl;
    g_log << "app name -- " << app_name << std::endl;
    g_log.flush();
  }

  if (g_best_rows > 0 && g_best_log.good()) {
    double default_total_time = 0.0;
    double best_total_time = 0.0;
    for (auto& kv : g_tune) {
      int kernel_id = kv.first;
      auto& st = kv.second;
      // Only count kernels with valid times (not the initial 1e300)
      if (st.default_time < 1e100 && st.best_time < 1e100 && 
          kernel_id < OPS_instance::getOPSInstance()->OPS_kern_max) {
        int count = OPS_instance::getOPSInstance()->OPS_kernels[kernel_id].count;
        default_total_time += count * st.default_time;
        best_total_time += count * st.best_time;
      }
    }
    
    g_best_log << std::endl << std::endl << std::endl;
    g_best_log << "Autotune -- " << autotune << std::endl;
    g_best_log << "total of rows -- " << g_best_rows << std::endl;
    g_best_log << "app name -- " << app_name << std::endl;
    g_best_log << "default total time -- " << default_total_time << std::endl;
    g_best_log << "best total time -- " << best_total_time << std::endl;
    g_best_log.flush();
  }
  g_best_log.close();
  g_log.close();

#ifdef OPS_ML_XGBOOST
  // Mode 5: flush remaining training buffer and save updated models
  {
    OPS_instance *flush_inst = OPS_instance::getOPSInstance();
    if (flush_inst && flush_inst->OPS_autotune_mode == 5 && g_xgb_model.loaded) {
      // Flush remaining buffer
      if (!g_explore_learn.buffer.empty()) {
        printf("[OPS-EXPLORE] Flushing %zu remaining training samples\n",
               g_explore_learn.buffer.size());
        update_boosters_from_exploration();
      }

      // Save updated models to disk
      const char* bx_path = std::getenv("OPS_XGBOOST_BX");
      const char* by_path = std::getenv("OPS_XGBOOST_BY");
      const char* bz_path = std::getenv("OPS_XGBOOST_BZ");

      auto save_one = [](BoosterHandle booster, const char* path, const char* name) {
        if (!booster || !path) return;
        int ret = XGBoosterSaveModel(booster, path);
        if (ret != 0) {
          printf("[OPS-EXPLORE] ERROR saving %s model to %s: %s\n",
                 name, path, XGBGetLastError());
        } else {
          printf("[OPS-EXPLORE] Saved updated %s model to %s\n", name, path);
        }
      };

      save_one(g_xgb_model.booster_bx, bx_path, "bx");
      save_one(g_xgb_model.booster_by, by_path, "by");
      save_one(g_xgb_model.booster_bz, bz_path, "bz");

      printf("[OPS-EXPLORE] Training complete: %d total samples, %d model updates\n",
             g_explore_learn.total_samples, g_explore_learn.total_updates);
    }
  }
#endif
}
