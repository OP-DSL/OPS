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

/** @brief ops sequential backend implementation
  * @author Gihan Mudalige
  * @details Implements the OPS API calls for the sequential backend
  */

#include <ops_lib_core.h>
char *ops_halo_buffer = NULL;
int ops_halo_buffer_size = 0;
int posix_memalign(void **memptr, size_t alignment, size_t size); 


void ops_init(int argc, char **argv, int diags) {
  ops_init_core(argc, argv, diags);
}

void ops_exit() { ops_exit_core(); }

ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size, int *base,
                          int *d_m, int *d_p, char *data, int type_size,
                          char const *type, char const *name) {

  /** ----             allocate an empty dat             ---- **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p,
                                       data, type_size, type, name);

  if (data != NULL) {
    // printf("Data read in from HDF5 file or is allocated by the user\n");
    dat->user_managed =
        1; // will be reset to 0 if called from ops_decl_dat_hdf5()
    dat->is_hdf5 = 0;
    dat->hdf5_file = "none"; // will be set to an hdf5 file if called from
                             // ops_decl_dat_hdf5()
  } else {
    // Allocate memory immediately
    int bytes = size * type_size;
#ifdef __INTEL_COMPILER
    //On intel, make x size a multiple of 128 bytes by extending d_p
    int oldsize = dat->size[0];
    //Compute least common multiple - type_size is a multiple of 2, I need to remove all factors of 2 from size
    int size_non_2 = size;
    while (size_non_2%2==0 && size>1) size_non_2 /= 2;
    int least_common_multiple = 128/type_size * size_non_2;
    dat->size[0] = ((dat->size[0]-1)/(least_common_multiple )+1)*least_common_multiple;
    dat->d_p[0] += (dat->size[0] - oldsize);
#endif
    for (int i = 0; i < block->dims; i++)
      bytes = bytes * dat->size[i];
#ifdef __INTEL_COMPILER
//    dat->data = (char *)_mm_malloc(bytes, 2*1024*1024); // initialize data bits to 0
    posix_memalign((void**)&(dat->data), 2*1024*1024, bytes);
#else
    dat->data = (char *)calloc(bytes, 1); // initialize data bits to 0
#endif
    dat->user_managed = 0;
    dat->mem = bytes;
  }

  // Compute offset in bytes to the base index
  dat->base_offset = 0;
  long cumsize = 1;
  for (int i = 0; i < block->dims; i++) {
    dat->base_offset +=
        dat->elem_size * cumsize * (-dat->base[i] - dat->d_m[i]);
    cumsize *= dat->size[i];
  }

  return dat;
}

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(from, to, iter_size, from_base, to_base, from_dir,
                            to_dir);
}

void ops_halo_transfer(ops_halo_group group) {
  ops_execute();
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
    if (size > ops_halo_buffer_size) {
      ops_halo_buffer = (char *)realloc(ops_halo_buffer, size);
      ops_halo_buffer_size = size;
    }

    // copy to linear buffer from source
    int ranges[OPS_MAX_DIM * 2];
    int step[OPS_MAX_DIM];
    int buf_strides[OPS_MAX_DIM];
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
#pragma omp parallel for collapse(3)
    for (int k = MIN(ranges[4], ranges[5] + 1);
         k < MAX(ranges[4] + 1, ranges[5]); k++) {
      for (int j = MIN(ranges[2], ranges[3] + 1);
           j < MAX(ranges[2] + 1, ranges[3]); j++) {
        for (int i = MIN(ranges[0], ranges[1] + 1);
             i < MAX(ranges[0] + 1, ranges[1]); i++) {
          memcpy(ops_halo_buffer +
                     ((k - ranges[4]) * step[2] * buf_strides[2] +
                      (j - ranges[2]) * step[1] * buf_strides[1] +
                      (i - ranges[0]) * step[0] * buf_strides[0]) *
                         halo->from->elem_size,
                 halo->from->data +
                     (k * halo->from->size[0] * halo->from->size[1] +
                      j * halo->from->size[0] + i) *
                         halo->from->elem_size,
                 halo->from->elem_size);
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
#pragma omp parallel for collapse(3)
    for (int k = MIN(ranges[4], ranges[5] + 1);
         k < MAX(ranges[4] + 1, ranges[5]); k++) {
      for (int j = MIN(ranges[2], ranges[3] + 1);
           j < MAX(ranges[2] + 1, ranges[3]); j++) {
        for (int i = MIN(ranges[0], ranges[1] + 1);
             i < MAX(ranges[0] + 1, ranges[1]); i++) {
          memcpy(halo->to->data +
                     (k * halo->to->size[0] * halo->to->size[1] +
                      j * halo->to->size[0] + i) *
                         halo->to->elem_size,
                 ops_halo_buffer +
                     ((k - ranges[4]) * step[2] * buf_strides[2] +
                      (j - ranges[2]) * step[1] * buf_strides[1] +
                      (i - ranges[0]) * step[0] * buf_strides[0]) *
                         halo->to->elem_size,
                 halo->to->elem_size);
        }
      }
    }
  }
}

ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc) {
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  return temp;
}

ops_arg ops_arg_dat_opt(ops_dat dat, int dim, ops_stencil stencil,
                        char const *type, ops_access acc, int flag) {
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  (&temp)->opt = flag;
  return temp;
}

ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc) {
  return ops_arg_gbl_core(data, dim, size, acc);
}

void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr) {
  ops_execute();
  ops_checkpointing_reduction(handle);
  memcpy(ptr, handle->data, handle->size);
  handle->initialized = 0;
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name) {
  ops_print_dat_to_txtfile_core(dat, file_name);
}

void ops_get_data(ops_dat dat) {
  // data already on the host .. do nothing
  (void)dat;
}

void ops_decl_const_char(int dim, char const *type, int typeSize, char *data,
                         char const *name) {
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

void ops_partition(const char *routine) {
  (void)routine;
  // printf("Partitioning ops_dats\n");
}

void ops_H_D_exchanges_host(ops_arg *args, int nargs) {
  (void)nargs;
  (void)args;
}

void ops_H_D_exchanges_device(ops_arg *args, int nargs) {
  (void)nargs;
  (void)args;
}

void ops_cpHostToDevice(void **data_d, void **data_h, int size) {
  (void)data_d;
  (void)data_h;
  (void)size;
}

void ops_download_dat(ops_dat dat) { (void)dat; }

void ops_upload_dat(ops_dat dat) { (void)dat; }

void ops_timers(double *cpu, double *et) { ops_timers_core(cpu, et); }
