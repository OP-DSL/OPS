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
  * @brief OPS mpi+cuda/opencl host-side run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi backend
  */
#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest,
              const ops_int_halo *__restrict halo) {
  if (OPS_soa) {
    const char *__restrict src = dat->data + src_offset * dat->type_size;
  #pragma omp parallel for collapse(3) shared(src,dest)
    for (unsigned int i = 0; i < halo->count; i++) {
      for (int d = 0; d < dat->dim; d++)
        for (int v = 0; v < halo->blocklength/dat->type_size; v++)
          memcpy(dest+i*halo->blocklength*dat->dim+ v*dat->type_size*dat->dim + d*dat->type_size,
                 src+i*halo->stride + v*dat->type_size + d*(dat->size[0]*dat->size[1]*dat->size[2])*dat->type_size, dat->type_size);
    }
  } else {
    const char *__restrict src = dat->data + src_offset * dat->elem_size;
  #pragma omp parallel for shared(src,dest)
    for (unsigned int i = 0; i < halo->count; i++) {
      memcpy(dest+i*halo->blocklength*dat->dim, src+i*halo->stride*dat->dim, halo->blocklength*dat->dim);
      
    }
  }
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src,
                const ops_int_halo *__restrict halo) {
  if (OPS_soa) {
  char *__restrict dest = dat->data + dest_offset * dat->type_size;
  #pragma omp parallel for collapse(3) shared(src,dest)
    for (unsigned int i = 0; i < halo->count; i++) {
      for (int d = 0; d < dat->dim; d++)
        for (int v = 0; v < halo->blocklength/dat->type_size; v++)
          memcpy(dest+i*halo->stride + v*dat->type_size + d*(dat->size[0]*dat->size[1]*dat->size[2])*dat->type_size, 
            src+i*halo->blocklength*dat->dim + v*dat->type_size*dat->dim + d*dat->type_size, dat->type_size);
    }
  } else {
    char *__restrict dest = dat->data + dest_offset * dat->elem_size;
  #pragma omp parallel for shared(src,dest)
    for (unsigned int i = 0; i < halo->count; i++) {
      memcpy(dest+i*halo->stride*dat->dim, src+i*halo->blocklength*dat->dim, halo->blocklength*dat->dim);
    }
  }
}

char *ops_realloc_fast(char *ptr, size_t olds, size_t news) {
  return (char*)ops_realloc(ptr, news);
}

void ops_H_D_exchanges_cuda(ops_arg *args, int nargs) {
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit_cuda(ops_arg *args, int nargs) {
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit_opencl(ops_arg *args, int nargs) {
  (void)nargs;
  (void)args;
}

void ops_cpHostToDevice(void **data_d, void **data_h, int size) {
  (void)data_d;
  (void)data_h;
  (void)size;
}

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {
#pragma omp parallel for collapse(3)
  for (int k = MIN(rz_s,rz_e+1); k < MAX(rz_s+1,rz_e); k ++) {
    for (int j = MIN(ry_s,ry_e+1); j < MAX(ry_s+1,ry_e); j ++) {
      for (int i = MIN(rx_s,rx_e+1); i < MAX(rx_s+1,rx_e); i ++) {
        for (int d = 0; d < src->dim; d++) 
        memcpy(dest + dest_offset +
                   ((k - rz_s) * z_step * buf_strides_z +
                    (j - ry_s) * y_step * buf_strides_y +
                    (i - rx_s) * x_step * buf_strides_x) *
                       src->elem_size + d*src->type_size,
               src->data +
                   (OPS_soa ? ((k * src->size[0] * src->size[1] + j * src->size[0] + i) 
                            + d * src->size[0] * src->size[1] * src->size[2]) * src->type_size
                          : ((k * src->size[0] * src->size[1] + j * src->size[0] + i) *
                            src->elem_size + d*src->type_size)),
               src->type_size);
      }
    }
  }
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {
#pragma omp parallel for collapse(3)
  for (int k = MIN(rz_s,rz_e+1); k < MAX(rz_s+1,rz_e); k ++) {
    for (int j = MIN(ry_s,ry_e+1); j < MAX(ry_s+1,ry_e); j ++) {
      for (int i = MIN(rx_s,rx_e+1); i < MAX(rx_s+1,rx_e); i ++) {
        for (int d = 0; d < dest->dim; d++) 
        memcpy(dest->data +
                   (OPS_soa ? ((k * dest->size[0] * dest->size[1] + j * dest->size[0] + i)
                        + d * dest->size[0] * dest->size[1] * dest->size[2]) * dest->type_size
                       : ((k * dest->size[0] * dest->size[1] + j * dest->size[0] + i) *
                       dest->elem_size + d*dest->type_size)),
               src + src_offset +
                   ((k - rz_s) * z_step * buf_strides_z +
                    (j - ry_s) * y_step * buf_strides_y +
                    (i - rx_s) * x_step * buf_strides_x) *
                       dest->elem_size + d*dest->type_size,
               dest->type_size);
      }
    }
  }
}

void ops_download_dat(ops_dat dat) { (void)dat; }

void ops_upload_dat(ops_dat dat) { (void)dat; }
