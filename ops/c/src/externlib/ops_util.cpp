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
#include <ops_internal2.h>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <mutex>

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
  struct TuneState {
    std::vector<ops::dim3> candidates;
    size_t next_idx{0};
    bool decided{false};
    double best_time{1e300};
    ops::dim3 best{1,1,1};
    ops::dim3 last_served{1,1,1};
    long long last_points{0};
    long long best_points{0};
    // Metadata captured once per kernel
    int nargs_meta{0};
    int nstencil_args{0};
    int widest_radius{0};
    std::string stencil_sig{};
    bool meta_set{false};
   

  };

  std::unordered_map<int, TuneState> g_tune;
  std::ofstream g_log;
  std::once_flag g_log_once;
  std::mutex g_mutex;
  std::ofstream g_best_log;
  std::once_flag g_best_log_once;
  long long g_tuning_rows = 0;
  long long g_best_rows = 0;


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
      std::string base_dir = "/home/kevin/OPS_LOGS";
      std::string app_root = base_dir + "/" + app_name; // /home/.../OPS_LOGS/lattboltz2d_cuda
      std::string app_dir = base_dir + "/" + app_name + (autotune ? "/autotune_on" : "/autotune_off");

      ensure_dir_exists(base_dir);
      ensure_dir_exists(app_root); 
      ensure_dir_exists(app_dir);
      // printf("[OPS] Logging opening %s\n", app_dir.c_str());
      g_log.open(app_dir + "/ops_blocksize_tuning_logmod.csv",
                std::ios::out | std::ios::app);
      if (g_log.tellp() == 0) {
        g_log << "kernel_id,bx,by,bz,execution_time,points,gpoints_per_s,"
                "nargs,nstencil_args,widest_radius,stencil_sig"
              << std::endl;
      }
    });
  }

  void ensure_best_log_open() {
    std::call_once(g_best_log_once, [](){
      OPS_instance *inst = OPS_instance::getOPSInstance();
      bool autotune = is_autotune();
      std::string app_name = get_app_name();
      std::string base_dir = "/home/kevin/OPS_LOGS";
      std::string app_root = base_dir + "/" + app_name; // /home/.../OPS_LOGS/lattboltz2d_cuda
      std::string app_dir = base_dir + "/" + app_name + (autotune ? "/autotune_on" : "/autotune_off");

      g_best_log.open(app_dir + "/ops_blocksize_best_logmod.csv",
                std::ios::out | std::ios::app);
      if (g_best_log.tellp() == 0) {
      
        g_best_log << "kernel_id,bx,by,bz,best_time,points,gpoints_per_s,nargs,nstencil_args,widest_radius,stencil_sig" << std::endl;
      }
    });
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
    for (int by = 1; by <= max_y; by <<= 1) {
      for (int bx = 1; bx <= max_x; bx <<= 1) {
        const int prod = bx * by;
        if (prod <= max_threads && prod >= 32 /*&& bx > by*/)
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
    for (int bz = 1; bz <= max_z; bz <<= 1) {
      for (int by = 1; by <= max_y; by <<= 1) {
        for (int bx = 1; bx <= max_x; bx <<= 1) {
          const long long prod = 1LL * bx * by * bz;
          if (prod <= max_threads && prod >= 32 /*&& bx > by && bx > bz*/)
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
  // ensure_log_open();

  auto &st = g_tune[kernel_id]; //replicate (instead of find)

  // Compute extents from local_range
  const int x_extent = ndims >= 1 ? clamp_positive(local_range[1] - local_range[0]) : 1;
  const int y_extent = ndims >= 2 ? clamp_positive(local_range[3] - local_range[2]) : 1;
  const int z_extent = ndims >= 3 ? clamp_positive(local_range[5] - local_range[4]) : 1;

  // Capture kernel metadata once (argument count, stencil usage, widest stencil, signature)
  if (!st.meta_set) {
    st.nargs_meta = nargs;
    st.nstencil_args = 0;
    st.widest_radius = 0;
    std::ostringstream ss;
    for (int i = 0; i < nargs; ++i) {
      ops_arg &a = args[i];
      if (a.stencil != nullptr) {
        st.nstencil_args++;
        int dims = ndims; // default to kernel dims
        // If ops_stencil exposes dims, prefer it
        // NOTE: using pointer checks to avoid UB if fields differ across builds
        // Compute maximal Chebyshev radius among stencil points
        int max_rad = 0;
        if (a.stencil->points > 0 && a.stencil->stencil != nullptr) {
          // Heuristic: try a.stencil->dims if available via sizeof trick is not possible here; assume ndims
          for (int p = 0; p < a.stencil->points; ++p) {
            for (int d = 0; d < dims; ++d) {
              int off = a.stencil->stencil[p*dims + d];
              int abso = off < 0 ? -off : off;
              if (abso > max_rad) max_rad = abso;
            }
          }
        }
        if (max_rad > st.widest_radius) st.widest_radius = max_rad;
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
  }

  if (st.candidates.empty()) {
    OPS_instance *inst = OPS_instance::getOPSInstance();
    const int mode = inst ? inst->OPS_autotune_mode : 1;
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
    } else {
      // printf("[OPS] building candidates ...\n");
      // Modo autotune completo: construir candidatos
      if (ndims <= 1) {
        build_candidates_1d(st, x_extent, max_threads);
      } else if (ndims == 2) {
        build_candidates_2d(st, x_extent, y_extent, max_threads);

      } else {
        build_candidates_3d(st, x_extent, y_extent, z_extent, max_threads);
      }

      // Fallback a bloque global si no hay candidatos
      if (st.candidates.empty() && inst) {
        unsigned int gx = inst->OPS_block_size_x;
        unsigned int gy = (ndims > 1) ? inst->OPS_block_size_y : 1;
        unsigned int gz = (ndims > 2) ? inst->OPS_block_size_z : 1;
        if (gz > 64) gz = 64; // enforce bz <= 64
        if (ndims > 1 && gy >= gx) {
          if (gx > 1) gy = gx / 2;
          else gy = 1;
        }
        st.candidates.emplace_back(gx, gy, gz);
      }
    }
  }

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
    g_log << kernel_id << ',' << bs.x << ',' << bs.y << ',' << bs.z << ','
          << seconds_total << ',' 
          << points << ',' << gpoints << ','
          << st.nargs_meta << ',' << st.nstencil_args << ','
          << st.widest_radius << ',' << '"' << st.stencil_sig << '"' << std::endl;
    g_log.flush();
    g_tuning_rows++;
  }

  // Track best
  if (seconds_total > 0.0 && seconds_total) {
    st.best_time = seconds_total;
    st.best = bs;
    st.best_points = points;
  }

  // Decide when all candidates have been tested
  if (!st.decided && st.next_idx >= st.candidates.size()) {
    st.decided = true;
    // Optional: brief stdout summary
    // printf("[OPS autotune] v1.2 kernel %d best block = (%u,%u,%u) time = %.6f s \n",
    //        kernel_id, st.best.x, st.best.y, st.best.z, st.best_time,
    //        (st.best_time > 0.0 ? st.best_time : 0.0));

    // Write best-only CSV summary
    // ensure_best_log_open();
    const double best_gpoints = (st.best_time > 0.0 && st.best_points > 0)
                                ? (st.best_points / st.best_time) / 1e9 : 0.0;
   
    if (g_best_log.good()) {
      g_best_log << kernel_id << ',' << st.best.x << ',' << st.best.y << ',' << st.best.z << ','
                 << st.best_time << ',' 
                 << st.best_points << ',' << best_gpoints << ','
                 << st.nargs_meta << ',' << st.nstencil_args << ','
                 << st.widest_radius << ',' << '"' << st.stencil_sig << '"' << std::endl;
      g_best_log.flush();
      g_best_rows++;
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
    g_best_log << std::endl << std::endl << std::endl;
    g_best_log << "Autotune -- " << autotune << std::endl;
    g_best_log << "total of rows -- " << g_best_rows << std::endl;
    g_best_log << "app name -- " << app_name << std::endl;
    g_best_log.flush();
  }
}
