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
  * @brief OPS cuda specific runtime support functions
  * @author Gihan Mudalige
  * @details Implements cuda backend runtime support functions
  */

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ops_device_rt_support.h>
#include <ops_sycl_rt_support.h>
#include <ops_lib_core.h>


void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {
  ops_block block = src->block;

  // dest += dest_offset; <- a kernelen belül kell
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

  //dim3 grid(blk_x, blk_y, blk_z);
  //dim3 tblock(thr_x, thr_y, thr_z);
  int size_x = src->size[0];
  int size_y = src->size[1];
  int size_z = src->size[2];
  int type_size = src->type_size;
  int dim = src->dim;
  int OPS_soa = block->instance->OPS_soa;

  char *src_buff = src->data_d;

  block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {    //Queue->Submit

    //nd_range elso argumentume a teljes méret, nem a blokkok száma: https://docs.oneapi.com/versions/latest/dpcpp/iface/nd_range.html
    cgh.parallel_for<class copy_tobuf>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
      //get x dimension id
      int global_x_id = item.get_global_id()[2];
      //get y dimension id
      int global_y_id = item.get_global_id()[1];
      //get z dimension id
      int global_z_id = item.get_global_id()[0];

      int d_offset = dest_offset;
      int s_offset = 0;

      int idx_z = rz_s + z_step * global_z_id;
      int idx_y = ry_s + y_step * global_y_id;
      int idx_x = rx_s + x_step * global_x_id;

      if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
         (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
         (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

        if (OPS_soa) s_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size;

        else s_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size * dim;
        d_offset += ((idx_z - rz_s) * z_step * buf_strides_z +
               (idx_y - ry_s) * y_step * buf_strides_y +
               (idx_x - rx_s) * x_step * buf_strides_x) *
               type_size * dim;
        for (int d = 0; d < dim; d++) {
          memcpy(&dest[d_offset + d*type_size],
               &src_buff[s_offset],
               type_size);
          if (OPS_soa) s_offset += size_x * size_y * size_z * type_size;
          else s_offset += type_size;
        }
      }
    });
  });
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {


  ops_block block = dest->block;

  // src += src_offset;
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
  
  int size_x = dest->size[0];
  int size_y = dest->size[1];
  int size_z = dest->size[2];
  int type_size = dest->type_size;
  int dim = dest->dim;
  int OPS_soa = block->instance->OPS_soa;
  
  char* dest_buff = dest->data_d;
  
  block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
    //Accessors
    cgh.parallel_for<class copy_frombuf1>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
      //get x dimension id
      int global_x_id = item.get_global_id()[2];
      //get y dimension id
      int global_y_id = item.get_global_id()[1];
      //get z dimension id
      int global_z_id = item.get_global_id()[0];
      
      int d_offset = 0;
      int s_offset = src_offset;
      
      int idx_z = rz_s + z_step * global_z_id;
      int idx_y = ry_s + y_step * global_y_id;
      int idx_x = rx_s + x_step * global_x_id;
      
      if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
         (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
         (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {
        
        if (OPS_soa) d_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size;
        else d_offset += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size * dim;
        s_offset += ((idx_z - rz_s) * z_step * buf_strides_z + (idx_y - ry_s) * y_step * buf_strides_y + (idx_x - rx_s) * x_step * buf_strides_x) * type_size * dim;
        for (int d = 0; d < dim; d++) {
          memcpy(&dest_buff[d_offset], &src[s_offset + d*type_size], type_size);
          if (OPS_soa) d_offset += size_x * size_y * size_z * type_size;
          else d_offset += type_size;
        }
      }
    });
  });
  
  dest->dirty_hd = 2;
}


template <int dir>
void ops_internal_copy_device_kernel(char * dat0_p, char *dat1_p,
         int s0, int s01, int start0, int end0,
#if OPS_MAX_DIM>1
        int s1, int s11, int start1, int end1,
#if OPS_MAX_DIM>2
        int s2, int s21, int start2, int end2,
#if OPS_MAX_DIM>3
        int s3, int s31, int start3, int end3,
#if OPS_MAX_DIM>4
        int s4, int s41, int start4, int end4,
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
    long fullsize1 = s01;
    long idx = i*mult;
    long idx1 = i*mult;
#if OPS_MAX_DIM>1
    fullsize *= s1;
    fullsize1 *= s11;
    idx += j * s0 * mult;
    idx1 += j * s01 * mult;
#endif
#if OPS_MAX_DIM>2
    fullsize *= s2;
    fullsize1 *= s21;
    const int sz2 = end2-start2;
    int nextSize = rest / sz2;
    int k = start2+rest - nextSize*sz2;
    rest = nextSize;
    idx += k * s0 * s1 * mult;
    idx1 += k * s01 * s11 * mult;
#endif
#if OPS_MAX_DIM>3
    fullsize *= s3;
    fullsize1 *= s31;
    const int sz3 = end3-start3;
    nextSize = rest / sz3;
    int l = start3 + rest - nextSize*sz3;
    rest = nextSize;
    idx += l * s0 * s1 * s2 * mult;
    idx1 += l * s01 * s11 * s21 * mult;
#endif
#if OPS_MAX_DIM>4
    fullsize *= s4;
    fullsize1 *= s41;
    const int sz4 = end4-start4;
    nextSize = rest / sz4;
    int m = start4 + rest - nextSize*sz4;
    idx += m * s0 * s1 * s2 * s3 * mult;
    idx1 += m * s01 * s11 * s21 * s31 * mult;
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
        for (int c = 0; c < type_size; c++) {
          if (dir == 0)
            dat1_p[idx1+d*fullsize1*type_size+c] = dat0_p[idx+d*fullsize*type_size+c];
          else
            dat0_p[idx+d*fullsize*type_size+c] = dat1_p[idx1+d*fullsize1*type_size+c];
        }
    } else {
      for (int d = 0; d < dim*type_size; d++) {
        if (dir == 0)
          dat1_p[idx1+d] = dat0_p[idx+d];
        else
          dat0_p[idx+d] = dat1_p[idx1+d];
      }
    }
  }

}


void ops_internal_copy_device(ops_kernel_descriptor *desc) {
  int reverse = strcmp(desc->name, "ops_internal_copy_device_reverse")==0;
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
  ops_dat dat1 = desc->args[1].dat;
  double __t1=0.0,__t2=0.0,__c1,__c2;
  if (dat0->block->instance->OPS_diags>1) {
    dat0->block->instance->OPS_kernels[-1].count++;
    ops_timers_core(&__c1,&__t1);
  }
  char *dat0_p = desc->args[0].data_d + desc->args[0].dat->base_offset;
  char *dat1_p = desc->args[1].data_d + desc->args[1].dat->base_offset;
  int s0 = dat0->size[0];
  int s01 = dat1->size[0];
#if OPS_MAX_DIM>1
  int s1 = dat0->size[1];
  int s11 = dat1->size[1];
#if OPS_MAX_DIM>2
  int s2 = dat0->size[2];
  int s21 = dat1->size[2];
#if OPS_MAX_DIM>3
  int s3 = dat0->size[3];
  int s31 = dat1->size[3];
#if OPS_MAX_DIM>4
  int s4 = dat0->size[4];
  int s41 = dat1->size[4];
#endif
#endif
#endif
#endif

  int dim = dat0->dim;
  int type_size = dat0->type_size;
  int OPS_soa = dat0->block->instance->OPS_soa;

  int blk_x = (range[2*0+1]-range[2*0] - 1) / dat0->block->instance->OPS_block_size_x + 1;
  int blk_y = (range[2*1+1]-range[2*1] - 1) / dat0->block->instance->OPS_block_size_y + 1;
  int blk_z = ((range[2*2+1]-range[2*2] - 1) / dat0->block->instance->OPS_block_size_z + 1) *
            (range[2*3+1]-range[2*3]) *
            (range[2*4+1]-range[2*4]);
  int thr_x = dat0->block->instance->OPS_block_size_x;
  int thr_y = dat0->block->instance->OPS_block_size_y;
  int thr_z = dat0->block->instance->OPS_block_size_z;

  if (blk_x>0 && blk_y>0 && blk_z>0) {
    if (reverse)
      dat0->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<class copy_frombuf2>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
          ops_internal_copy_device_kernel<1>(
        dat0_p,
        dat1_p,
        s0,s01, range[2*0], range[2*0+1],
#if OPS_MAX_DIM>1
        s1,s11, range[2*1], range[2*1+1],
#if OPS_MAX_DIM>2
        s2,s21, range[2*2], range[2*2+1],
#if OPS_MAX_DIM>3
        s3,s31, range[2*3], range[2*3+1],
#if OPS_MAX_DIM>4
        s4,s41, range[2*4], range[2*4+1],
#endif
#endif
#endif
#endif
        dim, type_size,
        OPS_soa, item);
        });
    });



    else
      dat0->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<class copy_frombuf3>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
          ops_internal_copy_device_kernel<0>(
        dat0_p,
        dat1_p,
        s0,s01, range[2*0], range[2*0+1],
#if OPS_MAX_DIM>1
        s1,s11, range[2*1], range[2*1+1],
#if OPS_MAX_DIM>2
        s2,s21, range[2*2], range[2*2+1],
#if OPS_MAX_DIM>3
        s3,s31, range[2*3], range[2*3+1],
#if OPS_MAX_DIM>4
        s4,s41, range[2*4], range[2*4+1],
#endif
#endif
#endif
#endif
        dim, type_size,
        OPS_soa, item
        );});});
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
