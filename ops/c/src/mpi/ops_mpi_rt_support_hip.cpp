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
  * @brief OPS mpi+cuda run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi+cuda
 * backend
  */
#include <ops_hip_rt_support.h>

int halo_buffer_size = 0;
char *halo_buffer_d = NULL;

void ops_exit_device(OPS_instance *instance) {
  if (halo_buffer_d != NULL)
    ops_device_free(instance, (void**)&halo_buffer_d);

}

__global__ void ops_hip_packer_1(const char *__restrict src,
                                  char *__restrict dest, int count, int len,
                                  int stride) {
  int idx =  hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int block = idx / len;
  if (idx < count * len) {
    dest[idx] = src[stride * block + idx % len];
  }
}

__global__ void ops_hip_packer_1_soa(const char *__restrict src,
                                  char *__restrict dest, int count, int len,
                                  int stride, int dim, int size) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int block = idx / len;
  if (idx < count * len) {
    for (int d=0; d<dim; d++) {   
      dest[idx*dim+d] = src[stride * block + idx % len + d * size];
    }
  }
}

__global__ void ops_hip_unpacker_1(const char *__restrict src,
                                    char *__restrict dest, int count, int len,
                                    int stride) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int block = idx / len;
  if (idx < count * len) {
    dest[stride * block + idx % len] = src[idx];
  }
}

__global__ void ops_hip_unpacker_1_soa(const char *__restrict src,
                                    char *__restrict dest, int count, int len,
                                    int stride, int dim, int size) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int block = idx / len;
  if (idx < count * len) {
    for (int d=0; d<dim; d++) {   
      dest[stride * block + idx % len + d * size] = src[idx*dim + d];
    }
  }
}


__global__ void ops_hip_packer_4(const int *__restrict src,
                                  int *__restrict dest, int count, int len,
                                  int stride) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int block = idx / len;
  if (idx < count * len) {
    dest[idx] = src[stride * block + idx % len];
  }
}

__global__ void ops_hip_unpacker_4(const int *__restrict src,
                                    int *__restrict dest, int count, int len,
                                    int stride) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int block = idx / len;
  if (idx < count * len) {
    dest[stride * block + idx % len] = src[idx];
  }
}

void ops_pack_hip_internal(ops_dat dat, const int src_offset, char *__restrict dest,
              const int halo_blocklength, const int halo_stride, const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_put_data(dat);
    dat->dirty_hd = 0;
  }

  const char *__restrict src = dat->data_d + src_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size);
  if (halo_buffer_size < halo_count * halo_blocklength * dat->dim && !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (halo_buffer_d != NULL)
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipFree(halo_buffer_d));
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMalloc((void **)&halo_buffer_d,
                             halo_count * halo_blocklength * dat->dim * 4));
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }
  char *device_buf = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    device_buf = dest;
  else
    device_buf = halo_buffer_d;

  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
    hipLaunchKernelGGL(ops_hip_packer_1_soa, num_blocks, num_threads, 0, 0,
        src, device_buf, halo_count, halo_blocklength, halo_stride,
        dat->dim, dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size);
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());

  } else if (halo_blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo_blocklength / 4) * halo_count) - 1) / num_threads + 1;
    hipLaunchKernelGGL(ops_hip_packer_4, num_blocks, num_threads, 0, 0,
        (const int *)src, (int *)device_buf, halo_count, halo_blocklength*dat->dim / 4,
        halo_stride*dat->dim / 4);
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  } else {
    int num_threads = 128;
    int num_blocks = ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
    hipLaunchKernelGGL(ops_hip_packer_1, num_blocks, num_threads, 0, 0,//ops_hip_packer_1<<<num_blocks, num_threads>>>(
        src, device_buf, halo_count, halo_blocklength*dat->dim, halo_stride*dat->dim);
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  }
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct)
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpy(dest, halo_buffer_d,
                             halo_count * halo_blocklength * dat->dim,
                             hipMemcpyDeviceToHost));
  else
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipDeviceSynchronize());
}

void ops_unpack_hip_internal(ops_dat dat, const int dest_offset, const char *__restrict src,
                const int halo_blocklength, const int halo_stride, const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_put_data(dat);
    dat->dirty_hd = 0;
  }
  char *__restrict dest = dat->data_d + dest_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size);
  if (halo_buffer_size < halo_count * halo_blocklength * dat->dim && !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (halo_buffer_d != NULL)
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipFree(halo_buffer_d));
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMalloc((void **)&halo_buffer_d,
                             halo_count * halo_blocklength * dat->dim * 4));
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }

  const char *device_buf = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    device_buf = src;
  else
    device_buf = halo_buffer_d;

  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct)
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpy(halo_buffer_d, src,
                             halo_count * halo_blocklength * dat->dim,
                             hipMemcpyHostToDevice));
  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
    hipLaunchKernelGGL(ops_hip_unpacker_1_soa, num_blocks, num_threads, 0, 0,//ops_hip_unpacker_1_soa<<<num_blocks, num_threads>>>(
        device_buf, dest, halo_count, halo_blocklength, halo_stride,
        dat->dim, dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size);
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  } else if (halo_blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo_blocklength / 4) * halo_count) - 1) / num_threads + 1;
    hipLaunchKernelGGL(ops_hip_unpacker_4, num_blocks, num_threads, 0, 0,//6<<<num_blocks, num_threads>>>(
        (const int *)device_buf, (int *)dest, halo_count,
        halo_blocklength*dat->dim / 4, halo_stride*dat->dim / 4);
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  } else {
    int num_threads = 128;
    int num_blocks = ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
    hipLaunchKernelGGL(ops_hip_unpacker_1, num_blocks, num_threads, 0, 0,//<<<num_blocks, num_threads>>>(
        device_buf, dest, halo_count, halo_blocklength*dat->dim, halo_stride*dat->dim);
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  }

  dat->dirty_hd = 2;
}

char* OPS_realloc_fast(char *ptr, size_t olds, size_t news) {
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (ptr == NULL) {
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMalloc((void **)&ptr, news));
      return ptr;
    } else {
      if (OPS_instance::getOPSInstance()->OPS_diags>3) printf("Warning: hip cache realloc\n");
      char *ptr2;
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMalloc((void **)&ptr2, news));
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpy(ptr2, ptr, olds, hipMemcpyDeviceToDevice));
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipFree(ptr));
      return ptr2;
    }
  } else {
    char *ptr2;
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipHostMalloc((void**)&ptr2,news)); //TODO: is this aligned??
    if (olds > 0)
  	  memcpy(ptr2, ptr, olds);
    if (ptr != NULL) hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipHostFree(ptr));
    return ptr2;
  }
}

__global__ void copy_kernel_tobuf(char *dest, char *src, int rx_s, int rx_e,
                                  int ry_s, int ry_e, int rz_s, int rz_e,
                                  int x_step, int y_step, int z_step,
                                  int size_x, int size_y, int size_z,
                                  int buf_strides_x, int buf_strides_y,
                                  int buf_strides_z, int type_size, int dim, int OPS_soa) {

  int idx_z = rz_s + z_step * (hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z);
  int idx_y = ry_s + y_step * (hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y);
  int idx_x = rx_s + x_step * (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x);

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

  int idx_z = rz_s + z_step * (hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z);
  int idx_y = ry_s + y_step * (hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y);
  int idx_x = rx_s + x_step * (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x);

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
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    gpu_ptr = dest;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL)
        hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipFree(halo_buffer_d));
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMalloc((void **)&halo_buffer_d, size * sizeof(char)));
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
  }

  if (src->dirty_hd == 1) {
    ops_put_data(src);
    src->dirty_hd = 0;
  }

  dim3 grid(blk_x, blk_y, blk_z);
  dim3 tblock(thr_x, thr_y, thr_z);
  hipLaunchKernelGGL(copy_kernel_tobuf, grid, tblock, 0, 0, 
      gpu_ptr, src->data_d, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, src->size[0], src->size[1], src->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, src->type_size, src->dim, OPS_instance::getOPSInstance()->OPS_soa);
  hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());

  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct)
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpy(dest, halo_buffer_d, size * sizeof(char),
                             hipMemcpyDeviceToHost));
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
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    gpu_ptr = src;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL)
        hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipFree(halo_buffer_d));
      hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMalloc((void **)&halo_buffer_d, size * sizeof(char)));
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpy(halo_buffer_d, src, size * sizeof(char),
                             hipMemcpyHostToDevice));
  }

  if (dest->dirty_hd == 1) {
    ops_put_data(dest);
    dest->dirty_hd = 0;
  }

  dim3 grid(blk_x, blk_y, blk_z);
  dim3 tblock(thr_x, thr_y, thr_z);
  hipLaunchKernelGGL(copy_kernel_frombuf,grid, tblock, 0, 0, 
      dest->data_d, gpu_ptr, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, dest->size[0], dest->size[1], dest->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, dest->type_size, dest->dim, OPS_instance::getOPSInstance()->OPS_soa);
  hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  dest->dirty_hd = 2;
}

__global__ void ops_internal_copy_hip_kernel(char * dat0_p, char *dat1_p,
         int s0, int start0, int end0,
#if OPS_MAX_DIM>1
        int s1, int start1, int end1,
#if OPS_MAX_DIM>2
        int s2, int start2, int end2,
#if OPS_MAX_DIM>3
        int s3, int start3, int end3,
#if OPS_MAX_DIM>4
        int s4, int start4, int end4,
#endif
#endif
#endif
#endif
        int dim, int type_size,
        int OPS_soa) {
  int i = start0 + hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
  int j = start1 + hipThreadIdx_y + hipBlockIdx_y*hipBlockDim_y;
  int rest = hipThreadIdx_z + hipBlockIdx_z*hipBlockDim_z;
  int mult = OPS_soa ? type_size : dim*type_size;

    long fullsize = s0;
    long idx = i*mult;
#if OPS_MAX_DIM>1
    fullsize *= s1;
    idx += j * s0 * mult;
#endif
#if OPS_MAX_DIM>2
    fullsize *= s2;
    int k = start2+rest%s2;
    idx += k * s0 * s1 * mult;
#endif
#if OPS_MAX_DIM>3
    fullsize *= s3;
    int l = start3+rest/s2;
    idx += l * s0 * s1 * s2 * mult;
#endif
#if OPS_MAX_DIM>3
    fullsize *= s4;
    int m = start4+rest/(s2*s3);
    idx += m * s0 * s1 * s2 * s3 * mult;
#endif
    if (i<end0
#if OPS_MAX_DIM>1
        && j < end1
#if OPS_MAX_DIM>2
        && k < end2
#if OPS_MAX_DIM>3
        && l < end3
#if OPS_MAX_DIM>4
        && m < end4
#endif
#endif
#endif
#endif
       )

    if (OPS_soa)
      for (int d = 0; d < dim; d++)
        for (int c = 0; c < type_size; c++)
          dat1_p[idx+d*fullsize*type_size+c] = dat0_p[idx+d*fullsize*type_size+c];
    else
      for (int d = 0; d < dim*type_size; d++)
        dat1_p[idx+d] = dat0_p[idx+d];

}


void ops_internal_copy_hip(ops_kernel_descriptor *desc) {
  int range[2*OPS_MAX_DIM]={0};
  for (int d = 0; d < desc->dim; d++) {
    range[2*d] = desc->range[2*d];
    range[2*d+1] = desc->range[2*d+1];
  }
  for (int d = desc->dim; d < OPS_MAX_DIM; d++) {
    range[2*d] = 0;
    range[2*d+1] = 1;
  }
  ops_dat dat0 = desc->args[0].dat;
  double __t1 = 0.,__t2 = 0.,__c1 = 0.,__c2 = 0.;
  if (dat0->block->instance->OPS_diags>1) {
    dat0->block->instance->OPS_kernels[-1].count++;
    ops_timers_core(&__c1,&__t1);
  }
  char *dat0_p = desc->args[0].data_d + desc->args[0].dat->base_offset;
  char *dat1_p = desc->args[1].data_d + desc->args[1].dat->base_offset;
  int s0 = dat0->size[0];
#if OPS_MAX_DIM>1
  int s1 = dat0->size[1];
#if OPS_MAX_DIM>2
  int s2 = dat0->size[2];
#if OPS_MAX_DIM>3
  int s3 = dat0->size[3];
#if OPS_MAX_DIM>4
  int s4 = dat0->size[4];
#endif
#endif
#endif
#endif

  dim3 grid((range[2*0+1]-range[2*0] - 1) / dat0->block->instance->OPS_block_size_x + 1,
            (range[2*1+1]-range[2*1] - 1) / dat0->block->instance->OPS_block_size_y + 1,
           ((range[2*2+1]-range[2*2] - 1) / dat0->block->instance->OPS_block_size_z + 1) *
            (range[2*3+1]-range[2*3]) *
            (range[2*4+1]-range[2*4]));
  dim3 tblock(dat0->block->instance->OPS_block_size_x,
              dat0->block->instance->OPS_block_size_y,
              dat0->block->instance->OPS_block_size_z);

  if (grid.x>0 && grid.y>0 && grid.z>0) {
  hipLaunchKernelGGL(ops_internal_copy_hip_kernel,grid, tblock, 0, 0, 
        dat0_p,
        dat1_p,
        s0, range[2*0], range[2*0+1],
#if OPS_MAX_DIM>1
        s1, range[2*1], range[2*1+1],
#if OPS_MAX_DIM>2
        s2, range[2*2], range[2*2+1],
#if OPS_MAX_DIM>3
        s3, range[2*3], range[2*3+1],
#if OPS_MAX_DIM>4
        s4, range[2*4], range[2*4+1],
#endif
#endif
#endif
#endif
        dat0->dim, dat0->type_size,
        dat0->block->instance->OPS_soa
        );
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipGetLastError());
  }
  if (dat0->block->instance->OPS_diags>1) {
    hipSafeCall(dat0->block->instance->ostream(), hipDeviceSynchronize());
    ops_timers_core(&__c2,&__t2);
    int start[OPS_MAX_DIM];
    int end[OPS_MAX_DIM];
    for ( int n=0; n<desc->dim; n++ ){
      start[n] = range[2*n];end[n] = range[2*n+1];
    }
    dat0->block->instance->OPS_kernels[-1].time += __t2-__t1;
    dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[0]);
    dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[1]);
  }

}

void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
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
    desc->name = "ops_internal_copy_hip";
    desc->device = 1;
    desc->function = ops_internal_copy_hip;
    ops_internal_copy_hip(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
  }

}

void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
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
    desc->name = "ops_internal_copy_hip_reverse";
    desc->device = 1;
    desc->function = ops_internal_copy_hip;
    ops_internal_copy_hip(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
    dat->dirty_hd = 2;
  }

}


void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
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
    desc->name = "ops_internal_copy_hip";
    desc->device = 1;
    desc->function = ops_internal_copy_hip;
    ops_internal_copy_hip(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
  }
}

void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
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
    desc->name = "ops_internal_copy_hip_reverse";
    desc->device = 1;
    desc->function = ops_internal_copy_hip;
    ops_internal_copy_hip(desc);
    target->data_d = NULL;
    ops_free(target);
    ops_free(desc->args);
    ops_free(desc);
    dat->dirty_hd = 2;
  }
}
