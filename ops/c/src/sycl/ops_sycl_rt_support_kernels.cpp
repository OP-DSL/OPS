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
  * @brief OPS sycl specific runtime support functions
  * @author Gabor Daniel Balogh
  * @details Implements sycl backend runtime support functions
  */

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ops_sycl_rt_support.h>
#include <ops_lib_core.h>

#include <iostream>


/* ######################-NEM KELL INNENTŐL-#########################
__global__ void copy_kernel(char *dest, char *src, int size ) {
  int tid = blockIdx.x;
  memcpy(&dest[tid],&src[tid],size);
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
            type_size * dim;
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
      memcpy(dest, src + d*type_size, type_size);
      if (OPS_soa) dest += size_x * size_y * size_z * type_size;
      else dest += type_size;
    }
  }
}

######################-IDÁIG-######################### */

 void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                          int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                          int x_step, int y_step, int z_step, int buf_strides_x,
                          int buf_strides_y, int buf_strides_z) {
	//throw OPSException(OPS_INTERNAL_ERROR, "Error: halo exchange for SYCL unimplemented");


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

	cl::sycl::buffer<char,1> *dest_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)dest);
  //ez csak src->data_d
	cl::sycl::buffer<char,1> *src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)src->data_d);

	block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {		//Queue->Submit
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		//auto out = cl::sycl::stream(1024, 768, cgh);

		//nd_range elso argumentume a teljes méret, nem a blokkok száma: https://docs.oneapi.com/versions/latest/dpcpp/iface/nd_range.html
		cgh.parallel_for<class copy_tobuf>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
			//get x dimension id
			cl::sycl::cl_int global_x_id = item.get_global_id()[2];
			//get y dimension id
			cl::sycl::cl_int global_y_id = item.get_global_id()[1];
			//get z dimension id
			cl::sycl::cl_int global_z_id = item.get_global_id()[0];

			cl::sycl::cl_int local_x_id = item.get_local_id()[2];
			cl::sycl::cl_int local_y_id = item.get_local_id()[1];
			cl::sycl::cl_int local_z_id = item.get_local_id()[0];

			cl::sycl::cl_int linear_id = item.get_local_linear_id();

			cl::sycl::cl_int d_offset = dest_offset;
			cl::sycl::cl_int s_offset = 0;

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
					memcpy(&dest_acc[d_offset + d*type_size],
						   &src_acc[s_offset],
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
	//throw OPSException(OPS_INTERNAL_ERROR, "Error: halo exchange for SYCL unimplemented");

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
	
	//dim3 grid(blk_x, blk_y, blk_z);
	//dim3 tblock(thr_x, thr_y, thr_z);
	
	int size_x = dest->size[0];
	int size_y = dest->size[1];
	int size_z = dest->size[2];
	int type_size = dest->type_size;
	int dim = dest->dim;
	int OPS_soa = block->instance->OPS_soa;
	
	cl::sycl::buffer<char,1> *src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)src);
	
	cl::sycl::buffer<char,1> *dest_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)dest->data_d);
	
	block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		cgh.parallel_for<class copy_frombuf>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
			//get x dimension id
			cl::sycl::cl_int global_x_id = item.get_global_id()[2];
			//get y dimension id
			cl::sycl::cl_int global_y_id = item.get_global_id()[1];
			//get z dimension id
			cl::sycl::cl_int global_z_id = item.get_global_id()[0];
			// get linear id
			cl::sycl::cl_int linear_id = item.get_local_linear_id();
			
			cl::sycl::cl_int d_offset = 0;
			cl::sycl::cl_int s_offset = src_offset;
			
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
					memcpy(&dest_acc[d_offset], &src_acc[s_offset + d*type_size], type_size);
					if (OPS_soa) d_offset += size_x * size_y * size_z * type_size;
					else d_offset += type_size;
				}
			}
		});
	});
	
	
	/*copy_kernel_frombuf<<<grid, tblock>>>(
		dest->data_d, src, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
		z_step, dest->size[0], dest->size[1], dest->size[2], buf_strides_x,
		buf_strides_y, buf_strides_z, dest->type_size, dest->dim, dest->block->instance->OPS_soa);
	cutilSafeCall(dest->block->instance->ostream(),cudaGetLastError());
	dest->dirty_hd = 2;*/
}

void ops_internal_copy_sycl(ops_kernel_descriptor *desc) {
	//throw OPSException(OPS_INTERNAL_ERROR, "Error: internal copy for SYCL unimplemented");
	int reverse = strcmp(desc->name, "ops_internal_copy_sycl_reverse") == 0;
	int range[2 * OPS_MAX_DIM] = {0};
	for (int d = 0; d < desc->dim; d++) {
		range[2 * d] = desc->range[2 * d];
		range[2 * d + 1] = desc->range[2 * d + 1];
	}
	for (int d = desc->dim; d < OPS_MAX_DIM; d++) {
		range[2 * d] = 0;
		range[2 * d + 1] = 1;
	}
	ops_dat dat0 = desc->args[0].dat;
	ops_dat dat1 = desc->args[1].dat;
	double __t1 = 0.0, __t2 = 0.0, __c1, __c2;
	if (desc->block->instance->OPS_diags > 1) {
		desc->block->instance->OPS_kernels[-1].count++;
		ops_timers_core(&__c1, &__t1);
	}
	int s0 = dat0->size[0];
	int s01 = dat1->size[0];
	#if OPS_MAX_DIM > 1
		int s1 = dat0->size[1];
		int s11 = dat1->size[1];
		#if OPS_MAX_DIM > 2
			int s2 = dat0->size[2];
			int s21 = dat1->size[2];
			#if OPS_MAX_DIM > 3
				int s3 = dat0->size[3];
				int s31 = dat1->size[3];
				#if OPS_MAX_DIM > 4
					int s4 = dat0->size[4];
					int s41 = dat1->size[4];
				#endif
			#endif
		#endif
	#endif
	auto *dat0_buf = static_cast<cl::sycl::buffer<char, 1> *>((void*)dat0->data_d);
	auto *dat1_buf = static_cast<cl::sycl::buffer<char, 1> *>((void*)dat1->data_d);
	assert(false && "implement deep copy for not whole copies");
	if (!reverse) {
		#ifdef SYCL_COPY
			desc->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
				auto acc0 = (*dat0_buf).template get_access<cl::sycl::access::mode::read>(cgh);
				auto acc1 =	(*dat1_buf).template get_access<cl::sycl::access::mode::write>(cgh);
				cgh.copy(acc0, acc1);
			});
			desc->block->instance->sycl_instance->queue->wait();
		#else
			auto HostAccessor0 = (*dat0_buf).get_access<cl::sycl::access::mode::read>();
			auto HostAccessor1 = (*dat1_buf).get_access<cl::sycl::access::mode::write>();
			for (size_t i = 0; i < dat0->mem; i++)
			HostAccessor1[i] = HostAccessor0[i];
		#endif
	}
	else {
		#ifdef SYCL_COPY
			desc->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
				auto acc0 =	(*dat0_buf).template get_access<cl::sycl::access::mode::write>(cgh);
				auto acc1 =	(*dat1_buf).template get_access<cl::sycl::access::mode::read>(cgh);
				cgh.copy(acc1, acc0);
			});
			desc->block->instance->sycl_instance->queue->wait();
		#else
			auto HostAccessor0 = (*dat0_buf).get_access<cl::sycl::access::mode::write>();
			auto HostAccessor1 = (*dat1_buf).get_access<cl::sycl::access::mode::read>();
			for (size_t i = 0; i < dat0->mem; i++)
			HostAccessor0[i] = HostAccessor1[i];
		#endif
	}
	if (dat0->block->instance->OPS_diags > 1) {
		ops_timers_core(&__c2, &__t2);
		int start[OPS_MAX_DIM];
		int end[OPS_MAX_DIM];
		for (int n = 0; n < desc->dim; n++) {
			start[n] = range[2 * n];
			end[n] = range[2 * n + 1];
		}
		dat0->block->instance->OPS_kernels[-1].time += __t2 - __t1;
		dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[0]);
		dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[1]);
	}
}
