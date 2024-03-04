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
  * @brief OPS mpi+sycl run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi+sycl
 * backend
  */
#include <ops_sycl_rt_support.h>
#include <ops_device_rt_support.h>

int halo_buffer_size = 0;
char *halo_buffer_d = NULL;

void ops_exit_device(OPS_instance *instance) {
  if (halo_buffer_d != NULL)
    ops_device_free(instance, (void**)&halo_buffer_d);
  delete instance->sycl_instance->queue;
  delete instance->sycl_instance;
}

void ops_pack_sycl_internal(ops_dat dat, const int src_offset,
                            char *__restrict dest, const int halo_blocklength,
                            const int halo_stride, const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_put_data(dat);
    dat->dirty_hd = 0;
  }

  const char *__restrict src = dat->data_d;
  size_t src_offset2 =
      src_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size
                                                            : dat->elem_size);
  int datdim = dat->dim;
  if (halo_buffer_size < halo_count * halo_blocklength * dat->dim &&
      !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (halo_buffer_d != NULL) {
      ops_device_free(dat->block->instance, (void**)&halo_buffer_d);
    }
    ops_device_malloc(dat->block->instance, (void**)&halo_buffer_d, halo_count * halo_blocklength * dat->dim * 4);
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }
  char *dest_buff = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    dest_buff = dest;
  else
    dest_buff = halo_buffer_d;


  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
    size_t datsize =
        dat->size[0] * dat->size[1] * dat->size[2] * dat->type_size;

    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {
          // parloop
          cgh.parallel_for<class ops_sycl_packer_1_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                int global_x_id = item.get_global_id()[0];
                int block = global_x_id / halo_blocklength;
                if (global_x_id < halo_count * halo_blocklength) {
                  for (int d = 0; d < datdim; d++) {
                    dest_buff[global_x_id * datdim + d] =
                        src[src_offset2 + halo_stride * block +
                                global_x_id % halo_blocklength + d * datsize];
                  }
                }
              });
        });

  } else if (halo_blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo_blocklength / 4) * halo_count) - 1) / num_threads +
        1;
    dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler
                                                               &cgh) {

      // parloop
      cgh.parallel_for<class ops_sycl_packer_4>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads),
                                cl::sycl::range<1>(num_threads)),
          [=](cl::sycl::nd_item<1> item) {
            int global_x_id = item.get_global_id()[0];
            int block =
                global_x_id / (halo_blocklength * datdim / 4);
            if (global_x_id < halo_count * (halo_blocklength * datdim / 4)) {
              memcpy(&dest_buff[global_x_id * 4],
                     &src[src_offset2 +
                              ((halo_stride * datdim / 4) * block +
                               global_x_id % (halo_blocklength * datdim / 4)) *
                                  4],
                     4);
            }
          });
    });
  } else {
    int num_threads = 128;
    int num_blocks =
        ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {

          // parloop
          cgh.parallel_for<class ops_sycl_packer_1>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                int global_x_id = item.get_global_id()[0];
                int block =
                    global_x_id / (halo_blocklength * datdim);
                if (global_x_id < halo_count * (halo_blocklength * datdim)) {
                  dest_buff[global_x_id] =
                      src[src_offset2 + (halo_stride * datdim) * block +
                              global_x_id % (halo_blocklength * datdim)];
                }
              });
        });
  }
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    ops_device_memcpy_d2h(dat->block->instance, (void**)&dest, (void**)&halo_buffer_d, halo_count * halo_blocklength * dat->dim);
  } else
    ops_device_sync(dat->block->instance);
}

void ops_unpack_sycl_internal(ops_dat dat, const int dest_offset,
                              const char *__restrict src,
                              const int halo_blocklength, const int halo_stride,
                              const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_put_data(dat);
    dat->dirty_hd = 0;
  }
  char *__restrict dest = dat->data_d;
  size_t dest_offset2 =
      dest_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size
                                                             : dat->elem_size);
  int datdim = dat->dim;
  if (halo_buffer_size < halo_count * halo_blocklength * dat->dim &&
      !OPS_instance::getOPSInstance()->OPS_gpu_direct) {

    if (halo_buffer_d != NULL) {
      ops_device_free(dat->block->instance, (void**)&halo_buffer_d);
    }
    ops_device_malloc(dat->block->instance, (void**)&halo_buffer_d, halo_count * halo_blocklength * dat->dim * 4);
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }

  const char *src_buff = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    src_buff = src;
  else
    src_buff = halo_buffer_d;

  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    ops_device_memcpy_h2d(dat->block->instance, (void**)&halo_buffer_d, (void**)&src, halo_count * halo_blocklength * dat->dim);
  }
  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
    size_t datsize =
        dat->size[0] * dat->size[1] * dat->size[2] * dat->type_size;
    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {
          // parloop
          cgh.parallel_for<class ops_sycl_unpacker_1_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                int global_x_id = item.get_global_id()[0];
                int block = global_x_id / halo_blocklength;
                if (global_x_id < halo_count * halo_blocklength) {
                  for (int d = 0; d < datdim; d++) {
                    dest[dest_offset2 + halo_stride * block +
                             global_x_id % halo_blocklength + d * datsize] =
                        src_buff[global_x_id * datdim + d];
                  }
                }
              });
        });
  } else if (halo_blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo_blocklength / 4) * halo_count) - 1) / num_threads +
        1;
    dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler
                                                               &cgh) {

      // parloop
      cgh.parallel_for<class ops_sycl_unpacker_4>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads),
                                cl::sycl::range<1>(num_threads)),
          [=](cl::sycl::nd_item<1> item) {
            int global_x_id = item.get_global_id()[0];
            int block =
                global_x_id / (halo_blocklength * datdim / 4);
            if (global_x_id < halo_count * (halo_blocklength * datdim / 4)) {
              memcpy(&dest[dest_offset2 +
                               ((halo_stride * datdim / 4) * block +
                                global_x_id % (halo_blocklength * datdim / 4)) *
                                   4],
                     &src_buff[global_x_id * 4], 4);
            }
          });
    });
  } else {
    int num_threads = 128;
    int num_blocks =
        ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {

          // parloop
          cgh.parallel_for<class ops_sycl_unpacker_1>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                int global_x_id = item.get_global_id()[0];
                int block =
                    global_x_id / (halo_blocklength * datdim);
                if (global_x_id < halo_count * (halo_blocklength * datdim)) {
                  dest[dest_offset2 + (halo_stride * dat->dim) * block +
                           global_x_id % (halo_blocklength * datdim)] =
                      src_buff[global_x_id];
                }
              });
        });
  }

  dat->dirty_hd = 2;
}

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {

  ops_block block = src->block;

  int size =
      abs(src->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));

  char *gpu_ptr;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    gpu_ptr = dest;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL) {
        ops_device_free(src->block->instance, (void**)&halo_buffer_d);
      }
      ops_device_malloc(src->block->instance, (void**)&halo_buffer_d, size);
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
  }

  if (src->dirty_hd == 1) {
    ops_put_data(src);
    src->dirty_hd = 0;
  }

  int size_x = src->size[0];
  int size_y = src->size[1];
  int size_z = src->size[2];
  int type_size = src->type_size;
  int dim = src->dim;
  int OPS_soa = block->instance->OPS_soa;
  int dest_offset_local = 0;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    dest_offset_local = dest_offset;

  char *src_buff = src->data_d;

  block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

    cgh.parallel_for<class copy_tobuf>(
        cl::sycl::range<3>(rz_e - rz_s, ry_e - ry_s, rx_e - rx_s),
        [=](cl::sycl::id<3> item) {
          int d_offset = dest_offset_local;
          int s_offset = 0;

          int idx_z = rz_s + z_step * item.get(0);
          int idx_y = ry_s + y_step * item.get(1);
          int idx_x = rx_s + x_step * item.get(2);
          if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
              (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
              (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

            if (OPS_soa)
              s_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) *
                          type_size;
            else
              s_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) *
                          type_size * dim;
            d_offset += ((idx_z - rz_s) * z_step * buf_strides_z +
                         (idx_y - ry_s) * y_step * buf_strides_y +
                         (idx_x - rx_s) * x_step * buf_strides_x) *
                        type_size * dim;
            for (int d = 0; d < dim; d++) {
              memcpy(&gpu_ptr[d_offset + d * type_size], &src_buff[s_offset],
                     type_size);
              if (OPS_soa)
                s_offset += size_x * size_y * size_z * type_size;
              else
                s_offset += type_size;
            }
          }
        });
  });
  ops_device_sync(block->instance);
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    void *dest_ptr = dest+dest_offset;
    ops_device_memcpy_d2h(block->instance, (void**)&dest_ptr, (void**)&halo_buffer_d, size);
  }
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {

  ops_block block = dest->block;

  int size =
      abs(dest->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));
  char *gpu_ptr;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    gpu_ptr = src;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL) {
        ops_device_free(dest->block->instance, (void**)&halo_buffer_d);
      }
      ops_device_malloc(dest->block->instance, (void**)&halo_buffer_d, size);
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
    void *src_ptr = src+src_offset;
    ops_device_memcpy_h2d(dest->block->instance, (void**)&halo_buffer_d, (void**)&(src_ptr), size);
  }

  if (dest->dirty_hd == 1) {
    ops_put_data(dest);
    dest->dirty_hd = 0;
  }

  int size_x = dest->size[0];
  int size_y = dest->size[1];
  int size_z = dest->size[2];
  int type_size = dest->type_size;
  int dim = dest->dim;
  int OPS_soa = block->instance->OPS_soa;
  int src_offset_local = 0;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    src_offset_local = src_offset;

  char *dest_buff = dest->data_d;

  block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

    cgh.parallel_for<class copy_frombuf>(
        cl::sycl::range<3>(rz_e - rz_s, ry_e - ry_s, rx_e - rx_s),
        [=](cl::sycl::id<3> item) {
          int d_offset = 0;
          int s_offset = src_offset_local;

          int idx_z = rz_s + z_step * item.get(0);
          int idx_y = ry_s + y_step * item.get(1);
          int idx_x = rx_s + x_step * item.get(2);
          if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
              (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
              (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

            if (OPS_soa)
              d_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) *
                          type_size;
            else
              d_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) *
                          type_size * dim;
            s_offset += ((idx_z - rz_s) * z_step * buf_strides_z +
                         (idx_y - ry_s) * y_step * buf_strides_y +
                         (idx_x - rx_s) * x_step * buf_strides_x) *
                        type_size * dim;
            for (int d = 0; d < dim; d++) {
              memcpy(&dest_buff[d_offset], &gpu_ptr[s_offset + d * type_size],
                     type_size);
              if (OPS_soa)
                d_offset += size_x * size_y * size_z * type_size;
              else
                d_offset += type_size;
            }
          }
        });
  });

  ops_device_sync(dest->block->instance);

  dest->dirty_hd = 2;
}

void ops_internal_copy_device_kernel(char * dat0_p, char *dat1_p,
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
        int OPS_soa, cl::sycl::nd_item<3> item) {
  int i = start0 + item.get_global_id()[2];
  int j = start1 + item.get_global_id()[1];
  int rest = item.get_global_id()[0];
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
       ) {

    if (OPS_soa) {
      for (int d = 0; d < dim; d++)
        for (int c = 0; c < type_size; c++)
          dat1_p[idx+d*fullsize*type_size+c] = dat0_p[idx+d*fullsize*type_size+c];
    } else {
      for (int d = 0; d < dim*type_size; d++)
        dat1_p[idx+d] = dat0_p[idx+d];
    }
  }

}


void ops_internal_copy_device(ops_kernel_descriptor *desc) {
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

  int blk_x = (range[2*0+1]-range[2*0] - 1) / dat0->block->instance->OPS_block_size_x + 1;
  int blk_y =  (range[2*1+1]-range[2*1] - 1) / dat0->block->instance->OPS_block_size_y + 1;
  int blk_z = ((range[2*2+1]-range[2*2] - 1) / dat0->block->instance->OPS_block_size_z + 1) *
            (range[2*3+1]-range[2*3]) *
            (range[2*4+1]-range[2*4]);
  int thr_x = dat0->block->instance->OPS_block_size_x;
  int thr_y = dat0->block->instance->OPS_block_size_y;
  int thr_z = dat0->block->instance->OPS_block_size_z;

  if (blk_x>0 && blk_y>0 && blk_z>0) {
      dat0->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<class copy_frombuf1>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
      ops_internal_copy_device_kernel(
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
        dat0->block->instance->OPS_soa, item
        );});
    });
  }
  if (dat0->block->instance->OPS_diags>1) {
    ops_device_sync(dat0->block->instance);
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


void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest,
              const ops_int_halo *__restrict halo) {
  ops_pack_sycl_internal(dat,  src_offset, dest, halo->blocklength, halo->stride, halo->count);
}
void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src,
                const ops_int_halo *__restrict halo) {
  ops_unpack_sycl_internal(dat,  dest_offset, src, halo->blocklength, halo->stride, halo->count);
}


char* OPS_realloc_fast(char *ptr, size_t olds, size_t news) {
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (ptr == NULL) {
      ops_device_malloc(OPS_instance::getOPSInstance(), (void **)&ptr, news);
      return ptr;
    } else {
      if (OPS_instance::getOPSInstance()->OPS_diags>3) printf("Warning: SYCL cache realloc\n");
      char *ptr2;
      ops_device_malloc(OPS_instance::getOPSInstance(), (void **)&ptr2, news);
      ops_device_memcpy_d2d(OPS_instance::getOPSInstance(), (void **)&ptr2, (void **)&ptr, olds);
      ops_device_free(OPS_instance::getOPSInstance(), (void **)&ptr);
      return ptr2;
    }
  } else {
    char *ptr2;
    ops_device_mallochost(OPS_instance::getOPSInstance(), (void **)&ptr2, news);
    if (olds > 0)
  	  memcpy(ptr2, ptr, olds);
    if (ptr != NULL) ops_device_free(OPS_instance::getOPSInstance(), (void **)&ptr);
    return ptr2;
  }
}
