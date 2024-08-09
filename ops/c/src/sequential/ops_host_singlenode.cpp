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
  * @brief OPS cuda and single-process specific functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements cuda backend runtime support functions applicable to non-MPI backend
  */

#include <ops_lib_core.h>


ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size, int *base,
                          int *d_m, int *d_p, int *stride, char *data, int type_size,
                          char const *type, char const *name) {

  /** ----             allocate an empty dat             ---- **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p,
                                       stride, data, type_size, type, name);

  if (data != NULL && !block->instance->OPS_realloc) {
    dat->user_managed =
        1; // will be reset to 0 if called from ops_decl_dat_hdf5()
    dat->is_hdf5 = 0;
    dat->hdf5_file = "none"; // will be set to an hdf5 file if called from
                             // ops_decl_dat_hdf5()
    size_t bytes = size * type_size;
    for (int i = 0; i < block->dims; i++)
      bytes = bytes * dat->size[i];
    dat->mem = bytes;
  } else {
    // Allocate memory immediately
    size_t bytes = size * type_size;

    // Compute    padding x-dim for vectorization
    int x_pad = (1+((dat->size[0]-1)/SIMD_VEC))*SIMD_VEC - dat->size[0];
    dat->size[0] += x_pad;
    dat->d_p[0] += x_pad;
    dat->x_pad = x_pad;
    // printf("\nPadded size is %d total size =%d \n",x_pad,dat->size[0]);

    for (int i = 0; i < block->dims; i++)
      bytes = bytes * dat->size[i];
    dat->data = (char *)ops_malloc(bytes);
    dat->user_managed = 0;
    dat->mem = bytes;
    if (data != NULL && block->instance->OPS_realloc) {
      ops_convert_layout(data, dat->data, block, size,
          dat->size, dat_size, type_size, 0);
//          dat->size, dat_size_orig, type_size, 0);
//          block->instance->OPS_hybrid_layout ? //TODO: comes in when batching
//          block->instance->ops_batch_size : 0);
    } else
      ops_init_zero(dat->data, bytes);
  }

  // Compute offset in bytes to the base index
  dat->base_offset = 0;
  size_t cumsize = 1;
  for (int i = 0; i < block->dims; i++) {
    dat->base_offset +=
        (block->instance->OPS_soa ? dat->type_size : dat->elem_size)
        * cumsize * (-dat->base[i] - dat->d_m[i]);
    cumsize *= dat->size[i];
  }

  return dat;
}

char *get_buffer_ptr(char *ops_halo_buffer, int i, int j, int k, int l, int m, int d, int elem_size, int *ranges, int *step, int *buf_strides, int buf_type_size) {
  return ops_halo_buffer +
          (
        #if OPS_MAX_DIM > 4
          (m - ranges[8]) * step[4] * buf_strides[4] +
        #endif
        #if OPS_MAX_DIM > 3
          (l - ranges[6]) * step[3] * buf_strides[3] +
        #endif
        #if OPS_MAX_DIM > 2
          (k - ranges[4]) * step[2] * buf_strides[2] +
        #endif
        #if OPS_MAX_DIM > 1
          (j - ranges[2]) * step[1] * buf_strides[1] +
        #endif
          (i - ranges[0]) * step[0] * buf_strides[0]) *
              elem_size + d * buf_type_size;
}

char *get_data_ptr(ops_dat dat, int i, int j, int k, int l, int m, int d, int OPS_soa) {
  return dat->data +
      (OPS_soa ?
        (
          #if OPS_MAX_DIM > 4
          m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3] +
          #endif
          #if OPS_MAX_DIM > 3
          l * dat->size[0] * dat->size[1] * dat->size[2] +
          #endif
          #if OPS_MAX_DIM > 2
          k * dat->size[0] * dat->size[1] +
          #endif
          #if OPS_MAX_DIM > 1
          j * dat->size[0] +
          #endif
          i +
          d * dat->size[0]
            #if OPS_MAX_DIM > 4
            * dat->size[4]
            #endif
            #if OPS_MAX_DIM > 3
            * dat->size[3]
            #endif
            #if OPS_MAX_DIM > 2
            * dat->size[2]
            #endif
            #if OPS_MAX_DIM > 1
            * dat->size[1]
            #endif
        ) * dat->type_size
      :(
        #if OPS_MAX_DIM > 4
        m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3] +
        #endif
        #if OPS_MAX_DIM > 3
        l * dat->size[0] * dat->size[1] * dat->size[2] +
        #endif
        #if OPS_MAX_DIM > 2
        k * dat->size[0] * dat->size[1] +
        #endif
        #if OPS_MAX_DIM > 1
        j * dat->size[0] +
        #endif
        i) *
            dat->elem_size + d * dat->type_size);
}

void ops_halo_transfer(ops_halo_group group) {
  ops_execute(group->instance);
  // Test contents of halo group
  /*ops_halo halo;
  for(int i = 0; i<group->nhalos; i++) {
    halo = group->halos[i];
    printf("halo->from->name = %s, halo->to->name %s\n",halo->from->name,
  halo->to->name);
    for (int i = 0; i < halo->from->block->dims; i++) {
      printf("halo->iter_size[i] %d ",halo->iter_size[i]);
      printf("halo->from_base[i] %d ",halo->from_base[i]);
      printf("halo->to_base[i] %d ",halo->to_base[i]);
      printf("halo->from_dir[i] %d ",halo->from_dir[i]);
      printf("halo->to_dir[i] %d \n",halo->to_dir[i]);
    }
  }
  //return;*/
  // printf("group->nhalos %d\n",group->nhalos);
  for (int h = 0; h < group->nhalos; h++) {
    ops_halo halo = group->halos[h];
    int size = halo->from->elem_size * halo->iter_size[0];
    for (int i = 1; i < halo->from->block->dims; i++)
      size *= halo->iter_size[i];
    if (size > group->instance->ops_halo_buffer_size) {
      group->instance->ops_halo_buffer = (char *)ops_realloc(group->instance->ops_halo_buffer, size);
      group->instance->ops_halo_buffer_size = size;
    }

    // copy to linear buffer from source
    int ranges[OPS_MAX_DIM * 2] = {0};
    int step[OPS_MAX_DIM] = {0};
    int buf_strides[OPS_MAX_DIM] = {0};
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      if (halo->from_dir[i] > 0) {
        ranges[2 * i] =
            halo->from_base[i] - halo->from->d_m[i] - halo->from->base[i];
        ranges[2 * i + 1] =
            ranges[2 * i] + halo->iter_size[abs(halo->from_dir[i]) - 1];
        step[i] = 1;
      } else {
        ranges[2 * i + 1] =
            halo->from_base[i] - 1 - halo->from->d_m[i] - halo->from->base[i];
        ranges[2 * i] =
            ranges[2 * i + 1] + halo->iter_size[abs(halo->from_dir[i]) - 1];
        step[i] = -1;
      }
      buf_strides[i] = 1;
      for (int j = 0; j != abs(halo->from_dir[i]) - 1; j++)
        buf_strides[i] *= halo->iter_size[j];
    }
    int OPS_soa = group->instance->OPS_soa;
    char *ops_halo_buffer =  group->instance->ops_halo_buffer;
    int storage_type_size = halo->from->type_size < halo->to->type_size ? halo->from->type_size : halo->to->type_size;
    bool mixed_exchange = halo->from->type_size!=halo->to->type_size &&
                    (strcmp(halo->from->type, "float") == 0 || strcmp(halo->from->type, "double") == 0 || strcmp(halo->from->type, "half") == 0) &&
                    (strcmp(halo->to->type, "float") == 0 || strcmp(halo->to->type, "double") == 0 || strcmp(halo->to->type, "half") == 0);
  #if OPS_MAX_DIM>4
    #if OPS_MAX_DIM == 5
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(5)
    #endif
    #endif
    for (int m = MIN(ranges[8], ranges[9] + 1);
         m < MAX(ranges[8] + 1, ranges[9]); m++) {
  #else
    int m = 0;
    {
  #endif
    #if OPS_MAX_DIM>3
      #if OPS_MAX_DIM == 4
      #ifdef _OPENMP
      #pragma omp parallel for OMP_COLLAPSE(4)
      #endif
      #endif
      for (int l = MIN(ranges[6], ranges[7] + 1);
           l < MAX(ranges[6] + 1, ranges[7]); l++) {
    #else
      int l = 0;
      {
    #endif
      #if OPS_MAX_DIM>2
        #if OPS_MAX_DIM == 3
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(3)
        #endif
        #endif
        for (int k = MIN(ranges[4], ranges[5] + 1);
             k < MAX(ranges[4] + 1, ranges[5]); k++) {
      #else
        int k = 0;
        {
      #endif
        #if OPS_MAX_DIM>1
          #if OPS_MAX_DIM == 2
          #ifdef _OPENMP
          #pragma omp parallel for OMP_COLLAPSE(2)
          #endif
          #endif
          for (int j = MIN(ranges[2], ranges[3] + 1);
               j < MAX(ranges[2] + 1, ranges[3]); j++) {
        #else
          int j = 0;
          {
        #endif
            for (int i = MIN(ranges[0], ranges[1] + 1);
                 i < MAX(ranges[0] + 1, ranges[1]); i++) {
              for (int d = 0; d < halo->from->dim; d++) {
                if (mixed_exchange) {
                  if (storage_type_size == 4) {
                    float value = 0;
                    if (halo->from->type_size == 4) {
                      value = *((float *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    } else if (halo->from->type_size == 8) {
                      value = *((double *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    } 
                    else if (halo->from->type_size == 2) {
                      value = *((half *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    }
                    memcpy(get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, 4, ranges, step, buf_strides, 4), &value, 4);
                  } else if (storage_type_size == 8) {
                    double value = 0;
                    if (halo->from->type_size == 4) {
                      value = *((float *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    } else if (halo->from->type_size == 8) {
                      value = *((double *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    } 
                    else if (halo->from->type_size == 2) {
                      value = *((half *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    }
                    memcpy(get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, 8, ranges, step, buf_strides, 8), &value, 8);
                  } 
                  else if (storage_type_size == 2) {
                    half value = 0;
                    if (halo->from->type_size == 4) {
                      value = *((float *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    } else if (halo->from->type_size == 8) {
                      value = *((double *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    } else if (halo->from->type_size == 2) {
                      value = *((half *)get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa));
                    }
                    memcpy(get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, 2, ranges, step, buf_strides, 2), &value, 2);
                  }
                } else {
                  char *from = get_data_ptr(halo->from, i, j, k, l, m, d, OPS_soa);
                  char *to = get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, halo->from->elem_size, ranges, step, buf_strides, halo->from->type_size);
                  memcpy(to, from, halo->from->type_size);
                  
                }
              }
            }
          }
        }
      }
    }

    // copy from linear buffer to target
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      if (halo->to_dir[i] > 0) {
        ranges[2 * i] = halo->to_base[i] - halo->to->d_m[i] - halo->to->base[i];
        ranges[2 * i + 1] =
            ranges[2 * i] + halo->iter_size[abs(halo->to_dir[i]) - 1];
        step[i] = 1;
      } else {
        ranges[2 * i + 1] =
            halo->to_base[i] - 1 - halo->to->d_m[i] - halo->to->base[i];
        ranges[2 * i] =
            ranges[2 * i + 1] + halo->iter_size[abs(halo->to_dir[i]) - 1];
        step[i] = -1;
      }
      buf_strides[i] = 1;
      for (int j = 0; j != abs(halo->to_dir[i]) - 1; j++)
        buf_strides[i] *= halo->iter_size[j];
    }
    OPS_soa = group->instance->OPS_soa;
    ops_halo_buffer =  group->instance->ops_halo_buffer;
  #if OPS_MAX_DIM>4
    #if OPS_MAX_DIM == 5
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(5)
    #endif
    #endif
    for (int m = MIN(ranges[8], ranges[9] + 1);
         m < MAX(ranges[8] + 1, ranges[9]); m++) {
  #else
    int m = 0;
    {
  #endif
    #if OPS_MAX_DIM>3
      #if OPS_MAX_DIM == 4
      #ifdef _OPENMP
      #pragma omp parallel for OMP_COLLAPSE(4)
      #endif
      #endif
      for (int l = MIN(ranges[6], ranges[7] + 1);
           l < MAX(ranges[6] + 1, ranges[7]); l++) {
    #else
      int l = 0;
      {
    #endif
      #if OPS_MAX_DIM>2
        #if OPS_MAX_DIM == 3
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(3)
        #endif
        #endif
        for (int k = MIN(ranges[4], ranges[5] + 1);
             k < MAX(ranges[4] + 1, ranges[5]); k++) {
      #else
        int k = 0;
        {
      #endif
        #if OPS_MAX_DIM>1
          #if OPS_MAX_DIM == 2
          #ifdef _OPENMP
          #pragma omp parallel for OMP_COLLAPSE(2)
          #endif
          #endif
          for (int j = MIN(ranges[2], ranges[3] + 1);
               j < MAX(ranges[2] + 1, ranges[3]); j++) {
        #else
          int j = 0;
          {
        #endif
            for (int i = MIN(ranges[0], ranges[1] + 1);
                 i < MAX(ranges[0] + 1, ranges[1]); i++) {
              for (int d = 0; d < halo->to->dim; d++) {
                if (mixed_exchange) {
                  if (storage_type_size == 4) {
                    float value = 0;
                    memcpy(&value, get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, 4, ranges, step, buf_strides, 4), 4);
                    if (halo->to->type_size == 4) {
                      *((float *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    } else if (halo->to->type_size == 8) {
                      *((double *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    } 
                    else if (halo->to->type_size == 2) {
                      *((half *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    }
                  } else if (storage_type_size == 8) {
                    double value = 0;
                    memcpy(&value, get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, 8, ranges, step, buf_strides, 8), 8);
                    if (halo->to->type_size == 4) {
                      *((float *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    } else if (halo->to->type_size == 8) {
                      *((double *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    } 
                    else if (halo->to->type_size == 2) {
                      *((half *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    }
                  } 
                  else if (storage_type_size == 2) {
                    half value = 0;
                    memcpy(&value, get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, 2, ranges, step, buf_strides, 2), 2);
                    if (halo->to->type_size == 4) {
                      *((float *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    } else if (halo->to->type_size == 8) {
                      *((double *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    } else if (halo->to->type_size == 2) {
                      *((half *)get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa)) = value;
                    }
                  }
                } else {
                  char *from = get_buffer_ptr(ops_halo_buffer, i, j, k, l, m, d, halo->to->elem_size, ranges, step, buf_strides, halo->to->type_size);
                  char *to = get_data_ptr(halo->to, i, j, k, l, m, d, OPS_soa);
                  memcpy(to, from, halo->to->type_size);
                }

              }
            }
          }
        }
      }
    }
  }
}

void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  (void)memspace;
  ops_dat_fetch_data_host(dat, part, data);
}

void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
    (void)memspace;
  ops_dat_fetch_data_slab_host(dat, part, data, range);
}
void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
    (void)memspace;
  ops_dat_set_data_host(dat, part, data);
}

void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
    (void)memspace;
  ops_dat_set_data_slab_host(dat, part, data, range);
}
