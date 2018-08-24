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

/** @brief ops mpi+cuda run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi+cuda
 * backend
  */

#include <ops_cuda_rt_support.h>
#include <ops_mpi_core.h>

#ifdef __cplusplus
extern "C" {
#endif

int halo_buffer_size = 0;
char *halo_buffer_d = NULL;

__global__ void ops_cuda_packer_1(const char *__restrict src,
                                  char *__restrict dest, int count, int len,
                                  int stride, int hybrid_offset) {
  int idx = hybrid_offset + blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    dest[idx] = src[stride * block + idx % len];
  }
}

__global__ void ops_cuda_packer_1_soa(const char *__restrict src,
                                  char *__restrict dest, int count, int len,
                                  int stride, int hybrid_offset, int dim, int size) {
  int idx = hybrid_offset + blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    for (int d=0; d<dim; d++) {   
      dest[idx*dim+d] = src[stride * block + idx % len + d * size];
    }
  }
}

__global__ void ops_cuda_unpacker_1(const char *__restrict src,
                                    char *__restrict dest, int count, int len,
                                    int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    dest[stride * block + idx % len] = src[idx];
  }
}

__global__ void ops_cuda_unpacker_1_soa(const char *__restrict src,
                                    char *__restrict dest, int count, int len,
                                    int stride, int dim, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    for (int d=0; d<dim; d++) {   
      dest[stride * block + idx % len + d * size] = src[idx*dim + d];
    }
  }
}


__global__ void ops_cuda_packer_4(const int *__restrict src,
                                  int *__restrict dest, int count, int len,
                                  int stride, int hybrid_offset) {
  int idx = hybrid_offset + blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    dest[idx] = src[stride * block + idx % len];
  }
}

__global__ void ops_cuda_unpacker_4(const int *__restrict src,
                                    int *__restrict dest, int count, int len,
                                    int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    dest[stride * block + idx % len] = src[idx];
  }
}

extern "C" int ops_hybrid_get_clean_cpu(ops_dat dat);
extern "C" void ops_pack_hybrid_cpu(ops_dat dat, const int src_offset, char *__restrict dest,
               int halo_blocklength, int halo_stride, int new_count);
extern "C" void ops_unpack_hybrid_cpu(ops_dat dat, const int dest_offset, const char *__restrict src,
                 int halo_blocklength, int halo_stride, int new_count);

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest,
              const ops_int_halo *__restrict halo) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
    dat->dirty_hd = 0;
  }
  //printf("Halo exchange of %s: offset %d, count %d, blocklen %d stride %d\n",dat->name, src_offset, halo->count, halo->blocklength, halo->stride);

  int hybrid_offset = 0; //if hybrid, where is the split betwen cpu and gpu. -1 if last dim 

  //With hybrid, we need to split the halo gather based on who has what
  //CPU has up to dirty_from_h. With overlapping clean regions, should we prefer CPU or GPU??
  if (ops_hybrid) {
    if (OPS_gpu_direct) {ops_printf("Error: hybrid execution and GPU Direct combination unsupported\n"); exit(-1);}
    int cpu_clean = ops_hybrid_get_clean_cpu(dat);

    //Find what dim we are working in, and what depth
    int dim=-1;
    sub_dat_list sd = OPS_sub_dat_list[dat->index];
    for (int dim2 = 0; dim2 < dat->block->dims; dim2++) {
      for (int depth2 = 0; depth2 < MAX_DEPTH; depth2++) {
        if (halo == &sd->halos[MAX_DEPTH * dim2 + depth2]) {
          dim = dim2;
          break;
        }
      }
      if (dim != -1) break;
    }
    if (dim == -1) {printf("Error, hybrid could not find halo depth/dim\n"); exit(-1);}

    //if last dim, then left face on CPU, right face on GPU
    if (dim == dat->block->dims-1) {
      hybrid_offset = 0;

      //Need to determine if left or right face
      int d_m = dat->d_m[dim] + sd->d_im[dim];
      if (src_offset == (-d_m) * sd->prod[dim - 1]) { //this is the formula for the left face offset in op_mpi_rt_support.c
        ops_pack_hybrid_cpu(dat, src_offset, dest, halo->blocklength, halo->stride, halo->count);
        return;
      } // For the right face, do nothing, code below will work as usual

    } else { //otherwise need to split based on cpu_clean
      //number of halo blocks in lower dimensions
      int lowdim_count = halo->count/dat->size[dat->block->dims-1];
      hybrid_offset = lowdim_count * cpu_clean;
    }
  }

  const char *__restrict src = dat->data_d + src_offset * (OPS_soa ? dat->type_size : dat->elem_size);
  if (halo_buffer_size < halo->count * halo->blocklength * dat->dim && !OPS_gpu_direct) {
    if (halo_buffer_d != NULL)
      cutilSafeCall(cudaFree(halo_buffer_d));
    cutilSafeCall(cudaMalloc((void **)&halo_buffer_d,
                             halo->count * halo->blocklength * dat->dim * 4));
    halo_buffer_size = halo->count * halo->blocklength * dat->dim * 4;
  }
  char *device_buf = NULL;
  if (OPS_gpu_direct)
    device_buf = dest;
  else
    device_buf = halo_buffer_d;

  if (OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * (halo->count-hybrid_offset)) - 1) / num_threads + 1;
    ops_cuda_packer_1_soa<<<num_blocks, num_threads>>>(
        src, device_buf, halo->count, halo->blocklength, halo->stride, hybrid_offset * halo->blocklength,
        dat->dim, dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size);

  } else if (halo->blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo->blocklength / 4) * (halo->count-hybrid_offset)) - 1) / num_threads + 1;
    ops_cuda_packer_4<<<num_blocks, num_threads>>>(
        (const int *)src, (int *)device_buf, halo->count, halo->blocklength*dat->dim / 4,
        halo->stride*dat->dim / 4, hybrid_offset * (dat->dim * halo->blocklength / 4));
  } else {
    int num_threads = 128;
    int num_blocks = ((dat->dim * halo->blocklength * (halo->count-hybrid_offset)) - 1) / num_threads + 1;
    ops_cuda_packer_1<<<num_blocks, num_threads>>>(
        src, device_buf, halo->count, halo->blocklength*dat->dim,
        halo->stride*dat->dim, hybrid_offset * dat->dim * halo->blocklength);
  }

  if (!OPS_gpu_direct)
    cutilSafeCall(cudaMemcpyAsync(dest + (hybrid_offset * halo->blocklength * dat->dim),
                             halo_buffer_d + (hybrid_offset * halo->blocklength * dat->dim),
                             (halo->count-hybrid_offset) * halo->blocklength * dat->dim,
                             cudaMemcpyDeviceToHost, 0));

  if (ops_hybrid && hybrid_offset != 0)
    ops_pack_hybrid_cpu(dat, src_offset, dest, halo->blocklength, halo->stride, hybrid_offset);

  cutilSafeCall(cudaDeviceSynchronize());
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src,
                const ops_int_halo *__restrict halo) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
    dat->dirty_hd = 0;
  }
  char *__restrict dest = dat->data_d + dest_offset * (OPS_soa ? dat->type_size : dat->elem_size);
  if (halo_buffer_size < halo->count * halo->blocklength * dat->dim && !OPS_gpu_direct) {
    if (halo_buffer_d != NULL)
      cutilSafeCall(cudaFree(halo_buffer_d));
    cutilSafeCall(cudaMalloc((void **)&halo_buffer_d,
                             halo->count * halo->blocklength * dat->dim * 4));
    halo_buffer_size = halo->count * halo->blocklength * dat->dim * 4;
  }

  const char *device_buf = NULL;
  if (OPS_gpu_direct)
    device_buf = src;
  else
    device_buf = halo_buffer_d;

  if (!OPS_gpu_direct)
    cutilSafeCall(cudaMemcpyAsync(halo_buffer_d, src,
                             halo->count * halo->blocklength * dat->dim,
                             cudaMemcpyHostToDevice,0));
  if (!ops_hybrid) cutilSafeCall(cudaDeviceSynchronize());

  if (OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo->blocklength * halo->count) - 1) / num_threads + 1;
    ops_cuda_unpacker_1_soa<<<num_blocks, num_threads>>>(
        device_buf, dest, halo->count, halo->blocklength, halo->stride,
        dat->dim, dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size);
  } else if (halo->blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo->blocklength / 4) * halo->count) - 1) / num_threads + 1;
    ops_cuda_unpacker_4<<<num_blocks, num_threads>>>(
        (const int *)device_buf, (int *)dest, halo->count,
        halo->blocklength*dat->dim / 4, halo->stride*dat->dim / 4);
  } else {
    int num_threads = 128;
    int num_blocks = ((dat->dim * halo->blocklength * halo->count) - 1) / num_threads + 1;
    ops_cuda_unpacker_1<<<num_blocks, num_threads>>>(
        device_buf, dest, halo->count, halo->blocklength*dat->dim, halo->stride*dat->dim);
  }

  dat->dirty_hd = 2;

  if (ops_hybrid) {
    ops_unpack_hybrid_cpu(dat, dest_offset, src, halo->blocklength, halo->stride, halo->count);
    cutilSafeCall(cudaDeviceSynchronize());
  }
}

char* ops_realloc_fast(char *ptr, size_t olds, size_t news) {
  if (OPS_gpu_direct) {
    if (ptr == NULL) {
      cutilSafeCall(cudaMalloc((void **)&ptr, news));
      return ptr;
    } else {
      if (OPS_diags>3) printf("Warning: cuda cache realloc\n");
      char *ptr2;
      cutilSafeCall(cudaMalloc((void **)&ptr2, news));
      cutilSafeCall(cudaMemcpy(ptr2, ptr, olds, cudaMemcpyDeviceToDevice));
      cutilSafeCall(cudaFree(ptr));
      return ptr2;
    }
  } else {
    char *ptr2;
    cutilSafeCall(cudaMallocHost((void**)&ptr2,news));
    if (olds > 0)
  	  memcpy(ptr2, ptr, olds);
    if (ptr != NULL) cutilSafeCall(cudaFreeHost(ptr));
    return ptr2;
  }
}

__global__ void copy_kernel_tobuf(char *dest, char *src, int rx_s, int rx_e,
                                  int ry_s, int ry_e, int rz_s, int rz_e,
                                  int x_step, int y_step, int z_step,
                                  int size_x, int size_y, int size_z,
                                  int buf_strides_x, int buf_strides_y,
                                  int buf_strides_z, int type_size, int dim, int OPS_soa) {

  int idx_z = rz_s + z_step * (blockDim.z * blockIdx.z + threadIdx.z);
  int idx_y = ry_s + y_step * (blockDim.y * blockIdx.y + threadIdx.y);
  int idx_x = rx_s + x_step * (blockDim.x * blockIdx.x + threadIdx.x);

  if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
      (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
      (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

    if (OPS_soa) src += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size;
    else src += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size * dim;
    dest += ((idx_z - rz_s) * z_step * buf_strides_z +
             (idx_y - ry_s) * y_step * buf_strides_y +
             (idx_x - rx_s) * x_step * buf_strides_x) *
            type_size * dim ;
    for (int d = 0; d < dim; d++) {
      memcpy(dest+d*type_size, src, type_size);
      if (OPS_soa) src += size_x * size_y * size_z * type_size;
      else src += type_size;
    }
  }
}

__global__ void copy_kernel_frombuf(char *dest, char *src, int rx_s, int rx_e,
                                    int ry_s, int ry_e, int rz_s, int rz_e,
                                    int x_step, int y_step, int z_step,
                                    int size_x, int size_y, int size_z,
                                    int buf_strides_x, int buf_strides_y,
                                    int buf_strides_z, int type_size, int dim, int OPS_soa) {

  int idx_z = rz_s + z_step * (blockDim.z * blockIdx.z + threadIdx.z);
  int idx_y = ry_s + y_step * (blockDim.y * blockIdx.y + threadIdx.y);
  int idx_x = rx_s + x_step * (blockDim.x * blockIdx.x + threadIdx.x);

  if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
      (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
      (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

    if (OPS_soa) dest += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size;
    else dest += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size * dim;
    src += ((idx_z - rz_s) * z_step * buf_strides_z +
            (idx_y - ry_s) * y_step * buf_strides_y +
            (idx_x - rx_s) * x_step * buf_strides_x) *
           type_size * dim;
    for (int d = 0; d < dim; d++) {
      memcpy(dest, src + d * type_size, type_size);
      if (OPS_soa) dest += size_x * size_y * size_z * type_size;
      else dest += type_size;
    }
  }
}

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {

  dest += dest_offset;
  int thr_x = abs(rx_s - rx_e);
  int blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s - ry_e);
  int blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s - rz_e);
  int blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  int size =
      abs(src->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));
  char *gpu_ptr;
  if (OPS_gpu_direct)
    gpu_ptr = dest;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL)
        cutilSafeCall(cudaFree(halo_buffer_d));
      cutilSafeCall(cudaMalloc((void **)&halo_buffer_d, size * sizeof(char)));
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
  }

  if (src->dirty_hd == 1) {
    ops_upload_dat(src);
    src->dirty_hd = 0;
  }

  dim3 grid(blk_x, blk_y, blk_z);
  dim3 tblock(thr_x, thr_y, thr_z);
  copy_kernel_tobuf<<<grid, tblock>>>(
      gpu_ptr, src->data_d, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, src->size[0], src->size[1], src->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, src->type_size, src->dim, OPS_soa);

  if (!OPS_gpu_direct)
    cutilSafeCall(cudaMemcpy(dest, halo_buffer_d, size * sizeof(char),
                             cudaMemcpyDeviceToHost));
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {

  src += src_offset;
  int thr_x = abs(rx_s - rx_e);
  int blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s - ry_e);
  int blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s - rz_e);
  int blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  int size =
      abs(dest->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));
  char *gpu_ptr;
  if (OPS_gpu_direct)
    gpu_ptr = src;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL)
        cutilSafeCall(cudaFree(halo_buffer_d));
      cutilSafeCall(cudaMalloc((void **)&halo_buffer_d, size * sizeof(char)));
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
    cutilSafeCall(cudaMemcpy(halo_buffer_d, src, size * sizeof(char),
                             cudaMemcpyHostToDevice));
  }

  if (dest->dirty_hd == 1) {
    ops_upload_dat(dest);
    dest->dirty_hd = 0;
  }

  dim3 grid(blk_x, blk_y, blk_z);
  dim3 tblock(thr_x, thr_y, thr_z);
  copy_kernel_frombuf<<<grid, tblock>>>(
      dest->data_d, gpu_ptr, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, dest->size[0], dest->size[1], dest->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, dest->type_size, dest->dim, OPS_soa);
  dest->dirty_hd = 2;
}

#ifdef __cplusplus
}
#endif
