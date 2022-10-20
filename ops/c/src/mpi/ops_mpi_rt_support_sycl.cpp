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

#include <ops_sycl_rt_support.h>

int halo_buffer_size = 0;
char *halo_buffer_d = NULL;

void ops_pack_sycl_internal(ops_dat dat, const int src_offset,
                            char *__restrict dest, const int halo_blocklength,
                            const int halo_stride, const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
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
    #ifdef SYCL_USM
      cl::sycl::free((void*)halo_buffer_d, *dat->block->instance->sycl_instance->queue);
    }
    halo_buffer_d = (char*)cl::sycl::malloc_device(halo_count * halo_blocklength * dat->dim * 4,
                            *dat->block->instance->sycl_instance->queue);
    #else
      cl::sycl::buffer<char, 1> *halo_buffer_sycl =
          static_cast<cl::sycl::buffer<char, 1> *>((void *)halo_buffer_d);
      delete halo_buffer_sycl;
    }
    cl::sycl::buffer<char, 1> *halo_buffer_sycl = new cl::sycl::buffer<char, 1>(
        cl::sycl::range<1>(halo_count * halo_blocklength * dat->dim * 4));
    halo_buffer_d = (char *)halo_buffer_sycl;
    #endif
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }
  char *device_buf = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    device_buf = dest;
  else
    device_buf = halo_buffer_d;

  cl::sycl::buffer<char, 1> *src_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)src);
  #ifndef SYCL_USM
  cl::sycl::buffer<char, 1> *dest_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)device_buf);
  #endif

  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
    size_t datsize =
        dat->size[0] * dat->size[1] * dat->size[2] * dat->type_size;

    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {
          // Accessors
          #ifdef SYCL_USM
          char *dest_acc = device_buf;
          #else
          auto dest_acc =
              dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
          #endif
          auto src_acc =
              src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);

          // parloop
          cgh.parallel_for<class ops_sycl_packer_1_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                cl::sycl::cl_int global_x_id = item.get_global_id()[0];
                cl::sycl::cl_int block = global_x_id / halo_blocklength;
                if (global_x_id < halo_count * halo_blocklength) {
                  for (int d = 0; d < datdim; d++) {
                    dest_acc[global_x_id * datdim + d] =
                        src_acc[src_offset2 + halo_stride * block +
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
      // Accessors
      #ifdef SYCL_USM
      char *dest_acc = device_buf;
      #else
      auto dest_acc =
          dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
      #endif
      auto src_acc =
          src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);

      // parloop
      cgh.parallel_for<class ops_sycl_packer_4>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads),
                                cl::sycl::range<1>(num_threads)),
          [=](cl::sycl::nd_item<1> item) {
            cl::sycl::cl_int global_x_id = item.get_global_id()[0];
            cl::sycl::cl_int block =
                global_x_id / (halo_blocklength * datdim / 4);
            if (global_x_id < halo_count * (halo_blocklength * datdim / 4)) {
              memcpy(&dest_acc[global_x_id * 4],
                     &src_acc[src_offset2 +
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
          #ifdef SYCL_USM
          char *dest_acc = device_buf;
          #else
          auto dest_acc =
              dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
          #endif
          auto src_acc =
              src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);

          // parloop
          cgh.parallel_for<class ops_sycl_packer_1>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                cl::sycl::cl_int global_x_id = item.get_global_id()[0];
                cl::sycl::cl_int block =
                    global_x_id / (halo_blocklength * datdim);
                if (global_x_id < halo_count * (halo_blocklength * datdim)) {
                  dest_acc[global_x_id] =
                      src_acc[src_offset2 + (halo_stride * datdim) * block +
                              global_x_id % (halo_blocklength * datdim)];
                }
              });
        });
  }
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    #ifdef SYCL_USM
//if the queue is out-of-order, the following wait is needed
//    dat->block->instance->sycl_instance->queue->wait();
    dat->block->instance->sycl_instance->queue->memcpy(dest, halo_buffer_d, halo_count * halo_blocklength * dat->dim);
    dat->block->instance->sycl_instance->queue->wait();
    #else
    ops_sycl_memcpyDeviceToHost(dat->block->instance, dest_buff, dest,
                                halo_count * halo_blocklength * dat->dim);
    #endif
  } else
    dat->block->instance->sycl_instance->queue->wait();
}

void ops_unpack_sycl_internal(ops_dat dat, const int dest_offset,
                              const char *__restrict src,
                              const int halo_blocklength, const int halo_stride,
                              const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
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
    #ifdef SYCL_USM
      cl::sycl::free((void*)halo_buffer_d, *dat->block->instance->sycl_instance->queue);
    }
    halo_buffer_d = (char*)cl::sycl::malloc_device(halo_count * halo_blocklength * dat->dim * 4,
                            *dat->block->instance->sycl_instance->queue);
    #else
      cl::sycl::buffer<char, 1> *halo_buffer_sycl =
          static_cast<cl::sycl::buffer<char, 1> *>((void *)halo_buffer_d);
      delete halo_buffer_sycl;
    }
    cl::sycl::buffer<char, 1> *halo_buffer_sycl = new cl::sycl::buffer<char, 1>(
        cl::sycl::range<1>(halo_count * halo_blocklength * dat->dim * 4));
    halo_buffer_d = (char *)halo_buffer_sycl;
    #endif
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }

  const char *device_buf = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    device_buf = src;
  else
    device_buf = halo_buffer_d;

  #ifndef SYCL_USM
  cl::sycl::buffer<char, 1> *src_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)device_buf);
  #endif
  cl::sycl::buffer<char, 1> *dest_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)dest);

  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    #ifdef SYCL_USM
    dat->block->instance->sycl_instance->queue->memcpy(halo_buffer_d, src, halo_count * halo_blocklength * dat->dim);
    dat->block->instance->sycl_instance->queue->wait();
    #else
    ops_sycl_memcpyHostToDevice(dat->block->instance, src_buff, src,
                                halo_count * halo_blocklength * dat->dim);
    #endif
  }
  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
    size_t datsize =
        dat->size[0] * dat->size[1] * dat->size[2] * dat->type_size;
    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {
          // Accessors
          auto dest_acc =
              dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
          #ifdef SYCL_USM
          const char *src_acc = device_buf;
          #else
          auto src_acc =
              src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
          #endif

          // parloop
          cgh.parallel_for<class ops_sycl_unpacker_1_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                cl::sycl::cl_int global_x_id = item.get_global_id()[0];
                cl::sycl::cl_int block = global_x_id / halo_blocklength;
                if (global_x_id < halo_count * halo_blocklength) {
                  for (int d = 0; d < datdim; d++) {
                    dest_acc[dest_offset2 + halo_stride * block +
                             global_x_id % halo_blocklength + d * datsize] =
                        src_acc[global_x_id * datdim + d];
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
      // Accessors
      auto dest_acc =
          dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
      #ifdef SYCL_USM
      const char *src_acc = device_buf;
      #else
      auto src_acc =
          src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
      #endif

      // parloop
      cgh.parallel_for<class ops_sycl_unpacker_4>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads),
                                cl::sycl::range<1>(num_threads)),
          [=](cl::sycl::nd_item<1> item) {
            cl::sycl::cl_int global_x_id = item.get_global_id()[0];
            cl::sycl::cl_int block =
                global_x_id / (halo_blocklength * datdim / 4);
            if (global_x_id < halo_count * (halo_blocklength * datdim / 4)) {
              memcpy(&dest_acc[dest_offset2 +
                               ((halo_stride * datdim / 4) * block +
                                global_x_id % (halo_blocklength * datdim / 4)) *
                                   4],
                     &src_acc[global_x_id * 4], 4);
            }
          });
    });
  } else {
    int num_threads = 128;
    int num_blocks =
        ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
    dat->block->instance->sycl_instance->queue->submit(
        [&](cl::sycl::handler &cgh) {
          // Accessors
          auto dest_acc =
              dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
          #ifdef SYCL_USM
          const char *src_acc = device_buf;
          #else
          auto src_acc =
              src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
          #endif

          // parloop
          cgh.parallel_for<class ops_sycl_unpacker_1>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(num_blocks * num_threads),
                  cl::sycl::range<1>(num_threads)),
              [=](cl::sycl::nd_item<1> item) {
                cl::sycl::cl_int global_x_id = item.get_global_id()[0];
                cl::sycl::cl_int block =
                    global_x_id / (halo_blocklength * datdim);
                if (global_x_id < halo_count * (halo_blocklength * datdim)) {
                  dest_acc[dest_offset2 + (halo_stride * dat->dim) * block +
                           global_x_id % (halo_blocklength * datdim)] =
                      src_acc[global_x_id];
                }
              });
        });
  }

  dat->dirty_hd = 2;
}

char *OPS_realloc_fast(char *ptr, size_t olds, size_t news) {
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (ptr == NULL) {
      #ifdef SYCL_USM
      ptr = (char*)cl::sycl::malloc_device(news,
                            *OPS_instance::getOPSInstance()->sycl_instance->queue);
      OPS_instance::getOPSInstance()->sycl_instance->queue->wait();
      #else
      cl::sycl::buffer<char, 1> *ptr_buff =
          new cl::sycl::buffer<char, 1>(cl::sycl::range<1>(news));
      ptr = (char *)ptr_buff;
      #endif
      return ptr;
    } else {
      if (OPS_instance::getOPSInstance()->OPS_diags > 3)
        printf("Warning: SYCL cache realloc\n");
      char *ptr2;
      #ifdef SYCL_USM
      ptr2 = (char*)cl::sycl::malloc_device(news,
                            *OPS_instance::getOPSInstance()->sycl_instance->queue);
      OPS_instance::getOPSInstance()->sycl_instance->queue->memcpy(ptr2, ptr, olds);
      cl::sycl::free(ptr, *OPS_instance::getOPSInstance()->sycl_instance->queue);
      #else
      cl::sycl::buffer<char, 1> *ptr2_buff =
          new cl::sycl::buffer<char, 1>(cl::sycl::range<1>(news));
      ptr2 = (char *)ptr2_buff;
      cl::sycl::buffer<char, 1> *ptr_buff =
          static_cast<cl::sycl::buffer<char, 1> *>((void *)ptr);
      ops_sycl_memcpyDeviceToDevice(OPS_instance::getOPSInstance(), ptr_buff,
                                    ptr2_buff, olds);
      delete ptr_buff;
      #endif
      return ptr2;
    }
  } else {
    // TODO: pinned memory
    return (char *)ops_realloc((void *)ptr, news);
  }
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
      #ifdef SYCL_USM
        cl::sycl::free((void*)halo_buffer_d, *src->block->instance->sycl_instance->queue);
      }
      halo_buffer_d = (char*)cl::sycl::malloc_device(size,
                              *src->block->instance->sycl_instance->queue);
      #else
        cl::sycl::buffer<char, 1> *halo_buffer_sycl =
            static_cast<cl::sycl::buffer<char, 1> *>((void *)halo_buffer_d);
        delete halo_buffer_sycl;
      }
      cl::sycl::buffer<char, 1> *halo_buffer_sycl =
          new cl::sycl::buffer<char, 1>(cl::sycl::range<1>(size));
      halo_buffer_d = (char *)halo_buffer_sycl;
      #endif
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
  }

  if (src->dirty_hd == 1) {
    ops_upload_dat(src);
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

  #ifndef SYCL_USM
  cl::sycl::buffer<char, 1> *dest_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)gpu_ptr);
  #endif
  cl::sycl::buffer<char, 1> *src_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)src->data_d);

  memset(dest + dest_offset, 0, size);
  block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
    #ifdef SYCL_USM
    char *dest_acc = gpu_ptr;
    #else
    auto dest_acc =
        dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
    #endif
    auto src_acc =
        src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<class copy_tobuf>(
        cl::sycl::range<3>(rz_e - rz_s, ry_e - ry_s, rx_e - rx_s),
        [=](cl::sycl::id<3> item) {
          cl::sycl::cl_int d_offset = dest_offset_local;
          cl::sycl::cl_int s_offset = 0;

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
              memcpy(&dest_acc[d_offset + d * type_size], &src_acc[s_offset],
                     type_size);
              if (OPS_soa)
                s_offset += size_x * size_y * size_z * type_size;
              else
                s_offset += type_size;
            }
          }
        });
  });
  src->block->instance->sycl_instance->queue->wait();
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    #ifdef SYCL_USM
    block->instance->sycl_instance->queue->memcpy(dest + dest_offset, halo_buffer_d, size);
    block->instance->sycl_instance->queue->wait();
    #else
    ops_sycl_memcpyDeviceToHost(block->instance, dest_buff, dest + dest_offset,
                                size * sizeof(char));
    #endif
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
      #ifdef SYCL_USM
        cl::sycl::free((void*)halo_buffer_d, *dest->block->instance->sycl_instance->queue);
      }
      halo_buffer_d = (char*)cl::sycl::malloc_device(size,
                              *dest->block->instance->sycl_instance->queue);
      #else
        cl::sycl::buffer<char, 1> *halo_buffer_sycl =
            static_cast<cl::sycl::buffer<char, 1> *>((void *)halo_buffer_d);
        delete halo_buffer_sycl;
      }
      cl::sycl::buffer<char, 1> *halo_buffer_sycl =
          new cl::sycl::buffer<char, 1>(cl::sycl::range<1>(size));
      halo_buffer_d = (char *)halo_buffer_sycl;
      #endif
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
    #ifdef SYCL_USM
    dest->block->instance->sycl_instance->queue->memcpy(halo_buffer_d, src + src_offset, size);
    dest->block->instance->sycl_instance->queue->wait();
    #else
    cl::sycl::buffer<char, 1> *src_buff =
        static_cast<cl::sycl::buffer<char, 1> *>((void *)gpu_ptr);
    ops_sycl_memcpyHostToDevice(block->instance, src_buff, src + src_offset,
                                size * sizeof(char));
    #endif
  }

  if (dest->dirty_hd == 1) {
    ops_upload_dat(dest);
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

  #ifndef SYCL_USM
  cl::sycl::buffer<char, 1> *src_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)gpu_ptr);
  #endif
  cl::sycl::buffer<char, 1> *dest_buff =
      static_cast<cl::sycl::buffer<char, 1> *>((void *)dest->data_d);

  block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
    // Accessors
    auto dest_acc =
        dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
    #ifdef SYCL_USM
    char *src_acc = gpu_ptr;
    #else
    auto src_acc =
        src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
    #endif

    cgh.parallel_for<class copy_frombuf>(
        cl::sycl::range<3>(rz_e - rz_s, ry_e - ry_s, rx_e - rx_s),
        [=](cl::sycl::id<3> item) {
          cl::sycl::cl_int d_offset = 0;
          cl::sycl::cl_int s_offset = src_offset_local;

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
              memcpy(&dest_acc[d_offset], &src_acc[s_offset + d * type_size],
                     type_size);
              if (OPS_soa)
                d_offset += size_x * size_y * size_z * type_size;
              else
                d_offset += type_size;
            }
          }
        });
  });

  dest->block->instance->sycl_instance->queue->wait();

  dest->dirty_hd = 2;
}

void ops_internal_copy_sycl(ops_kernel_descriptor *desc) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
  /*
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

    dim3 grid((range[2*0+1]-range[2*0] - 1) /
  dat0->block->instance->OPS_block_size_x + 1, (range[2*1+1]-range[2*1] - 1) /
  dat0->block->instance->OPS_block_size_y + 1,
             ((range[2*2+1]-range[2*2] - 1) /
  dat0->block->instance->OPS_block_size_z + 1) * (range[2*3+1]-range[2*3]) *
              (range[2*4+1]-range[2*4]));
    dim3 tblock(dat0->block->instance->OPS_block_size_x,
                dat0->block->instance->OPS_block_size_y,
                dat0->block->instance->OPS_block_size_z);

    if (grid.x>0 && grid.y>0 && grid.z>0) {
      ops_internal_copy_sycl_kernel<<<grid,tblock>>>(
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
      cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
    }
    if (dat0->block->instance->OPS_diags>1) {
      cutilSafeCall(dat0->block->instance->ostream(), cudaDeviceSynchronize());
      ops_timers_core(&__c2,&__t2);
      int start[OPS_MAX_DIM];
      int end[OPS_MAX_DIM];
      for ( int n=0; n<desc->dim; n++ ){
        start[n] = range[2*n];end[n] = range[2*n+1];
      }
      dat0->block->instance->OPS_kernels[-1].time += __t2-__t1;
      dat0->block->instance->OPS_kernels[-1].transfer +=
  ops_compute_transfer(desc->dim, start, end, &desc->args[0]);
      dat0->block->instance->OPS_kernels[-1].transfer +=
  ops_compute_transfer(desc->dim, start, end, &desc->args[1]);
    }
  */
}

void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data,
                                      int *range, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
  /*
    if (memspace == OPS_HOST) ops_dat_fetch_data_slab_host(dat, part, data,
    range); else { ops_execute(dat->block->instance); int range2[2*OPS_MAX_DIM];
      for (int i = 0; i < dat->block->dims; i++) {
        range2[2*i] = range[2*i];
        range2[2*i+1] = range[2*i+1];
      }
      for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
        range2[2*i] = 0;
        range2[2*i+1] = 1;
      }
      if (dat->dirty_hd == 1) {
        ops_upload_dat(dat);
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
      desc->name = "ops_internal_copy_cuda";
      desc->device = 1;
      desc->function = ops_internal_copy_sycl;
      ops_internal_copy_sycl(desc);
      target->data_d = NULL;
      ops_free(target);
      ops_free(desc->args);
      ops_free(desc);
    }
  */
}

void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data,
                                    int *range, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
  /*
    if (memspace == OPS_HOST) ops_dat_set_data_slab_host(dat, part, data,
    range); else { ops_execute(dat->block->instance); int range2[2*OPS_MAX_DIM];
      for (int i = 0; i < dat->block->dims; i++) {
        range2[2*i] = range[2*i];
        range2[2*i+1] = range[2*i+1];
      }
      for (int i = dat->block->dims; i < OPS_MAX_DIM; i++) {
        range2[2*i] = 0;
        range2[2*i+1] = 1;
      }
      if (dat->dirty_hd == 1) {
        ops_upload_dat(dat);
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
      desc->name = "ops_internal_copy_cuda_reverse";
      desc->device = 1;
      desc->function = ops_internal_copy_sycl;
      ops_internal_copy_sycl(desc);
      target->data_d = NULL;
      ops_free(target);
      ops_free(desc->args);
      ops_free(desc);
      dat->dirty_hd = 2;
    }
  */
}

void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data,
                                 ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
  /*
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
        ops_upload_dat(dat);
        dat->dirty_hd = 0;
      }
      ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
      target->data_d = data;
      target->elem_size = dat->elem_size;
      target->base_offset = 0;
      for (int d = 0; d < OPS_MAX_DIM; d++) target->size[d] = size[d];
      ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
      desc->name = "ops_internal_copy_cuda";
      desc->device = 1;
      desc->function = ops_internal_copy_sycl;
      ops_internal_copy_sycl(desc);
      target->data_d = NULL;
      ops_free(target);
      ops_free(desc->args);
      ops_free(desc);
    }
  */
}

void ops_dat_set_data_memspace(ops_dat dat, int part, char *data,
                               ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
  /*
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
        ops_upload_dat(dat);
      ops_dat target = (ops_dat)ops_malloc(sizeof(ops_dat_core));
      target->data_d = data;
      target->elem_size = dat->elem_size;
      target->base_offset = 0;
      for (int d = 0; d < OPS_MAX_DIM; d++) target->size[d] = size[d];
      ops_kernel_descriptor *desc = ops_dat_deep_copy_core(target, dat, range);
      desc->name = "ops_internal_copy_cuda_reverse";
      desc->device = 1;
      desc->function = ops_internal_copy_sycl;
      ops_internal_copy_sycl(desc);
      target->data_d = NULL;
      ops_free(target);
      ops_free(desc->args);
      ops_free(desc);
      dat->dirty_hd = 2;
    }
  */
}
