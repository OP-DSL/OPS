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
  * @brief common functions for GPU backends
  * @author Gihan Mudalige
  * @details Implements common functions from the various GPU backends for the sequential
  * (no MPI) versions
  */

#include "ops_lib_core.h"
#include <ops_exceptions.h>
#include <string>
#include <assert.h>

ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size, int *base,
                          int *d_m, int *d_p, int *stride, char *data, int type_size,
                          char const *type, char const *name) {

  /** ----             allocate an empty dat             ---- **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p,
                                       stride, data, type_size, type, name);

  size_t bytes = size * type_size;
  for (int i = 0; i < block->dims; i++)
    bytes = bytes * dat->size[i];

  if (data != NULL && !block->instance->OPS_realloc) {
    // printf("Data read in from HDF5 file or is allocated by the user\n");
    dat->user_managed =
        1; // will be reset to 0 if called from ops_decl_dat_hdf5()
    dat->is_hdf5 = 0;
    dat->mem = bytes;
    dat->hdf5_file = "none"; // will be set to an hdf5 file if called from
    ops_cpHostToDevice ( block->instance, ( void ** ) &( dat->data_d ),
            ( void ** ) &( dat->data ), bytes );
                             // ops_decl_dat_hdf5()
  } else {
    // Allocate memory immediately
    dat->data = (char*) ops_malloc(bytes);
    dat->user_managed = 0;
    dat->mem = bytes;
    dat->data_d = NULL;
    if (data != NULL && block->instance->OPS_realloc) {
      ops_convert_layout(data, dat->data, block, size,
          dat->size, dat_size, type_size, 0);
//          dat->size, dat_size_orig, type_size, 0);
//          block->instance->OPS_hybrid_layout ? //TODO: comes in when batching
//          block->instance->ops_batch_size : 0);
    } else
      ops_init_zero(dat->data, bytes);

    ops_cpHostToDevice ( block->instance, ( void ** ) &( dat->data_d ),
            ( void ** ) &(data), bytes );
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


  dat->x_pad = 0; // no padding for data alignment

  return dat;
}

void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_fetch_data_slab_host(dat, part, data, range);
  else {
    ops_execute(dat->block->instance);
    int range2[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range2[2*i] = range[2*i];
      range2[2*i+1] = range[2*i+1];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range2[2*i] = 0;
      range2[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_put_data(dat);
      dat->dirty_hd = 0;
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    size_t prod = 1;
    for (int d = 0; d < OPS_MAX_DIM; d++) {
      target->size[d] = range2[2*d+1]-range2[2*d];
      target->base_offset -= target->elem_size*prod*range2[2*d];
      prod *= target->size[d];
    }
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    strcpy(desc->name, "ops_internal_copy_device\0");
    desc->isdevice = 1;
    desc->func = ops_internal_copy_device;
    ops_internal_copy_device(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
  } 

}

void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_set_data_slab_host(dat, part, data, range);
  else {
    ops_execute(dat->block->instance);
    int range2[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range2[2*i] = range[2*i];
      range2[2*i+1] = range[2*i+1];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range2[2*i] = 0;
      range2[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_put_data(dat);
      dat->dirty_hd = 0;
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    size_t prod = 1;
    for (int d = 0; d < OPS_MAX_DIM; d++) {
      target->size[d] = range2[2*d+1]-range2[2*d];
      target->base_offset -= target->elem_size*prod*range2[2*d];
      prod *= target->size[d];
    }
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    strcpy(desc->name, "ops_internal_copy_device_reverse\0");
    desc->isdevice = 1;
    desc->func = ops_internal_copy_device;
    ops_internal_copy_device(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
    dat->dirty_hd = 2;
  }

}


void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_fetch_data_host(dat, part, data);
  else {
    ops_execute(dat->block->instance);
    int disp[OPS_MAX_DIM], size[OPS_MAX_DIM];
    ops_dat_get_extents(dat, 0, disp, size);
    int range[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range[2*i] = dat->base[i];
      range[2*i+1] = range[2*i] + size[i];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range[2*i] = 0;
      range[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1) {
      ops_put_data(dat);
      dat->dirty_hd = 0;
    }
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    for (int d = 0; d < OPS_MAX_DIM; d++) target->size[d] = size[d];
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    strcpy(desc->name, "ops_internal_copy_device\0");
    desc->isdevice = 1;
    desc->func = ops_internal_copy_device;
    ops_internal_copy_device(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
  } 
}

void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  if (memspace == OPS_HOST) ops_dat_set_data_host(dat, part, data);
  else {
    ops_execute(dat->block->instance);
    int disp[OPS_MAX_DIM], size[OPS_MAX_DIM];
    ops_dat_get_extents(dat, 0, disp, size);
    int range[2*OPS_MAX_DIM];
    for (int i = 0; i < dat->block->dims; i++) {
      range[2*i] = dat->base[i];
      range[2*i+1] = range[2*i] + size[i];
    }
    for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
      range[2*i] = 0;
      range[2*i+1] = 1;
    }
    if (dat->dirty_hd == 1)
      ops_put_data(dat);
    ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    target->data_d = data;
    target->elem_size = dat->elem_size;
    target->base_offset = 0;
    for (int d = 0; d < OPS_MAX_DIM; d++) target->size[d] = size[d];
    ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
    strcpy(desc->name, "ops_internal_copy_device_reverse\0");
    desc->isdevice = 1;
    desc->func = ops_internal_copy_device;
    ops_internal_copy_device(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
    dat->dirty_hd = 2;
  } 
}

void ops_halo_transfer(ops_halo_group group) {


  int storage_type_size = group->halos[0]->from->type_size < group->halos[0]->to->type_size ? group->halos[0]->from->type_size : group->halos[0]->to->type_size;
  bool mixed_exchange = group->halos[0]->from->type_size!=group->halos[0]->to->type_size &&
                  (strcmp(group->halos[0]->from->type, "float") == 0 || strcmp(group->halos[0]->from->type, "double") == 0 || strcmp(group->halos[0]->from->type, "half") == 0) &&
                  (strcmp(group->halos[0]->to->type, "float") == 0 || strcmp(group->halos[0]->to->type, "double") == 0 || strcmp(group->halos[0]->to->type, "half") == 0);


  for (int h = 0; h < group->nhalos; h++) {
    ops_halo halo = group->halos[h];
    int size = std::min(halo->from->elem_size,halo->to->elem_size) * halo->iter_size[0];
    for (int i = 1; i < halo->from->block->dims; i++)
      size *= halo->iter_size[i];
    if (size > group->instance->ops_halo_buffer_size) {
      ops_device_free(group->instance, (void**)&group->instance->ops_halo_buffer_d);
      ops_device_malloc(group->instance, (void **)&group->instance->ops_halo_buffer_d, size);
      group->instance->ops_halo_buffer_size = size;
      //deviceSafeCall(cudaDeviceSynchronize());
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

    if (halo->from->dirty_hd == 1) {
      ops_put_data(halo->from);
      halo->from->dirty_hd = 0;
    }
    ops_halo_copy_tobuf(group->instance->ops_halo_buffer_d, 0, halo->from, ranges[0], ranges[1],
                        ranges[2], ranges[3], ranges[4], ranges[5], step[0],
                        step[1], step[2], buf_strides[0], buf_strides[1],
                        buf_strides[2],mixed_exchange, storage_type_size);

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

    if (halo->to->dirty_hd == 1) {
      ops_put_data(halo->to);
      halo->to->dirty_hd = 0;
    }
    ops_halo_copy_frombuf(halo->to, group->instance->ops_halo_buffer_d, 0, ranges[0], ranges[1],
                          ranges[2], ranges[3], ranges[4], ranges[5], step[0],
                          step[1], step[2], buf_strides[0], buf_strides[1],
                          buf_strides[2],mixed_exchange, storage_type_size);

    halo->to->dirty_hd = 2;
  }
}
