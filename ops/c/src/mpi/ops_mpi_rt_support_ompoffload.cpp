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
 * @brief OPS mpi+ompoffload run-time support routines
 * @author Gihan Mudalige, Istvan Reguly
 * @details Implements the runtime support routines for the OPS mpi+ompoffload
 * backend
 */
#include <ops_lib_core.h>
#include <ops_device_rt_support.h>

int halo_buffer_size = 0;
char *halo_buffer_d = NULL;

void ops_exit_device(OPS_instance *instance) {
	if (halo_buffer_d != NULL)
		ops_device_free(instance, (void**)&halo_buffer_d);
}

void ops_pack_ompoffload_internal(ops_dat dat, const int src_offset,
		char *__restrict dest, const int halo_blocklength,
		const int halo_stride, const int halo_count) {

	if (dat->dirty_hd == 1) {
		ops_put_data(dat);
		dat->dirty_hd = 0;
	}

	size_t src_offset2 = src_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size);
	const char *__restrict src = (char*) (dat->data_d + src_offset2);

	if (halo_buffer_size < halo_count * halo_blocklength * dat->dim && !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
		if (halo_buffer_d != NULL) {
			ops_device_free(dat->block->instance, (void**)&halo_buffer_d);
		}
		ops_device_malloc(dat->block->instance, (void**)&halo_buffer_d, halo_count * halo_blocklength * dat->dim * 4);
		halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
	}

	char *dest_buff = NULL;
	if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
		dest_buff = dest;
	else {
		dest_buff = (char*) halo_buffer_d;
	}

	if (OPS_instance::getOPSInstance()->OPS_soa) {
		size_t dat_size = dat->size[0] * dat->size[1] * dat->size[2] * dat->type_size;
		size_t datdim = dat->dim;
		int len = halo_blocklength;
		int stride = halo_stride;

		#pragma omp target teams distribute parallel for 
		for (int idx = 0; idx < halo_count * len; idx++) {
			int block = idx / len;
			for (int d = 0; d < datdim; d++) {
				dest_buff[idx * datdim + d] = src[stride * block + idx % len + d * dat_size];
			}
		}
	} else if (halo_blocklength % 4 == 0) {
		int len = halo_blocklength * dat->dim / 4;
		int stride = halo_stride * dat->dim / 4;
		int* dest_buffer = (int*)dest_buff;
		const int* src_buffer = (const int*)src;

		#pragma omp target teams distribute parallel for
		for (int idx = 0; idx < halo_count * len; idx++) {
			int block = idx / len;
			dest_buffer[idx] = src_buffer[stride * block + idx % len];
		}
	} else {
		int len = halo_blocklength * dat->dim;
		int stride = halo_stride * dat->dim;

		#pragma omp target teams distribute parallel for
		for (int idx = 0; idx < halo_count * len; idx++) {
			int block = idx / len;
			dest_buff[idx] = src[stride * block + idx % len];
		}
	}
	if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
		ops_device_memcpy_d2h(dat->block->instance, (void**)&dest, (void**)&halo_buffer_d, halo_count * halo_blocklength * dat->dim);
	}
}

void ops_unpack_ompoffload_internal(ops_dat dat, const int dest_offset, const char *__restrict src,
		const int halo_blocklength, const int halo_stride, const int halo_count) {

	if (dat->dirty_hd == 1) {
		ops_put_data(dat);
		dat->dirty_hd = 0;
	}

	size_t dest_offset2 = dest_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size);
	char *__restrict dest = (char*) (dat->data_d + dest_offset2);

	if (halo_buffer_size < halo_count * halo_blocklength * dat->dim && !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
		if (halo_buffer_d != NULL) {
			ops_device_free(dat->block->instance, (void**)&halo_buffer_d);
		}
		ops_device_malloc(dat->block->instance, (void**)&halo_buffer_d, halo_count * halo_blocklength * dat->dim * 4);
		halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
	}

	const char *src_buff = NULL;
	if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
		src_buff = src;
	else {
		src_buff = (char*) halo_buffer_d;
	}

	if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
		ops_device_memcpy_h2d(dat->block->instance, (void**)&halo_buffer_d, (void**)&src, halo_count * halo_blocklength * dat->dim);
	}

	if (OPS_instance::getOPSInstance()->OPS_soa) {
		size_t dat_size = dat->size[0] * dat->size[1] * dat->size[2] * dat->type_size;
		size_t datdim = dat->dim;
		int len = halo_blocklength;
		int stride = halo_stride;

		#pragma omp target teams distribute parallel for
		for (int idx = 0; idx < halo_count * len; idx++) {
			int block = idx / len;
			for (int d = 0; d < datdim; d++) {
				dest[stride * block + idx % len + d * dat_size] = src_buff[idx * datdim + d];
			}
		}
	} else if (halo_blocklength % 4 == 0) {
		int len = halo_blocklength * dat->dim / 4;
		int stride = halo_stride * dat->dim / 4;
		int* dest_buffer = (int*)dest;
		const int* src_buffer = (const int*)src_buff;

		#pragma omp target teams distribute parallel for
		for (int idx = 0; idx < halo_count * len; idx++) {
			int block = idx / len;
			dest_buffer[stride * block + idx % len] = src_buffer[idx];
		}
	} else {
		int len = halo_blocklength * dat->dim;
		int stride = halo_stride * dat->dim;

		#pragma omp target teams distribute parallel for
		for (int idx = 0; idx < halo_count * len; idx++) {
			int block = idx / len;
			dest[stride * block + idx % len] = src_buff[idx];
		}
	}

	dat->dirty_hd = 2;
}

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
		int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
		int x_step, int y_step, int z_step, int buf_strides_x,
		int buf_strides_y, int buf_strides_z) {

	dest += dest_offset;
	int thr_x = abs(rx_s - rx_e);
	int thr_y = abs(ry_s - ry_e);
	int thr_z = abs(rz_s - rz_e);
	int size = abs(src->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));

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
		gpu_ptr = (char*) halo_buffer_d;
	}

	if (src->dirty_hd == 1) {
		ops_put_data(src);
		src->dirty_hd = 0;
	}

	char *src_buff = (char*) src->data_d;
	int size_x = src->size[0];
	int size_y = src->size[1];
	int size_z = src->size[2];
	int type_size = src->type_size;
	int dim = src->dim;
	int OPS_soa = src->block->instance->OPS_soa;


	#pragma omp target teams distribute parallel for collapse(3)
	for (int k = 0; k < thr_z; k++) {
		for (int j = 0; j < thr_y; j++) {
			for (int i = 0; i < thr_x; i++) {
				if (i > abs(rx_s - rx_e) || j > abs(ry_s - ry_e) || k > abs(rz_s - rz_e))
					continue;
				int idx_z = rz_s + z_step * k;
				int idx_y = ry_s + y_step * j;
				int idx_x = rx_s + x_step * i;

				if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
						(y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
						(z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

					if (OPS_soa) {
						src_buff += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size;
					} else {
						src_buff += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size * dim;
					}
					gpu_ptr += ((idx_z - rz_s) * z_step * buf_strides_z +
							(idx_y - ry_s) * y_step * buf_strides_y +
							(idx_x - rx_s) * x_step * buf_strides_x) *
						type_size * dim;
					for (int d = 0; d < dim; d++) {
                        // memcpy(gpu_ptr + d * type_size, src_buff, type_size);
                        char *dest_buff = gpu_ptr + d * type_size;
                        for (int it = 0; it < type_size; it++)
                            dest_buff[it] = src_buff[it];
						if (OPS_soa) src_buff += size_x * size_y * size_z * type_size;
						else src_buff += type_size;
					}
				}
			}
		}
	}

	if (!OPS_instance::getOPSInstance()->OPS_gpu_direct) {
		ops_device_memcpy_d2h(src->block->instance, (void**)&dest, (void**)&halo_buffer_d, size);
	}
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
		int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
		int x_step, int y_step, int z_step,
		int buf_strides_x, int buf_strides_y,
		int buf_strides_z) {

	src += src_offset;
	int thr_x = abs(rx_s - rx_e);
	int thr_y = abs(ry_s - ry_e);
	int thr_z = abs(rz_s - rz_e);
	int size = abs(dest->elem_size * (rx_e - rx_s) * (ry_e - ry_s) * (rz_e - rz_s));

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
		gpu_ptr = (char*) halo_buffer_d;
		ops_device_memcpy_h2d(dest->block->instance, (void**)&halo_buffer_d, (void**)&src, size);
	}

	if (dest->dirty_hd == 1) {
		ops_put_data(dest);
		dest->dirty_hd = 0;
	}

	char *dest_buff = (char*) dest->data_d;
	int size_x = dest->size[0];
	int size_y = dest->size[1];
	int size_z = dest->size[2];
	int type_size = dest->type_size;
	int dim = dest->dim;
	int OPS_soa = dest->block->instance->OPS_soa;

	#pragma omp target teams distribute parallel for collapse(3)
	for (int k = 0; k < thr_z; k++) {
		for (int j = 0; j < thr_y; j++) {
			for (int i = 0; i < thr_x; i++) {
				if (i > abs(rx_s - rx_e) || j > abs(ry_s - ry_e) || k > abs(rz_s - rz_e))
					continue;
				int idx_z = rz_s + z_step * k;
				int idx_y = ry_s + y_step * j;
				int idx_x = rx_s + x_step * i;

				if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
						(y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
						(z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

					if (OPS_soa) {
						dest_buff += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size;
					} else {
						dest_buff += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * type_size * dim;
					}
					gpu_ptr += ((idx_z - rz_s) * z_step * buf_strides_z +
							(idx_y - ry_s) * y_step * buf_strides_y +
							(idx_x - rx_s) * x_step * buf_strides_x) *
						type_size * dim;
					for (int d = 0; d < dim; d++) {
                        // memcpy(dest_buff, gpu_ptr + d * type_size, type_size);
                        char *src_buff = gpu_ptr + d * type_size;
                        for (int it = 0; it < type_size; it++)
                            dest_buff[it] = src_buff[it];
						if (OPS_soa) dest_buff += size_x * size_y * size_z * type_size;
						else dest_buff += type_size;
					}
				}
			}
		}
	}
	dest->dirty_hd = 2;
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
	char *dat0_p = (char*) (desc->args[0].data_d + desc->args[0].dat->base_offset);
	char *dat1_p = (char*) (desc->args[1].data_d + desc->args[1].dat->base_offset);
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
        int thr_x = dat0->block->instance->OPS_block_size_x * (range[2*0+1]-range[2*0]);
	int thr_y = dat0->block->instance->OPS_block_size_y * (range[2*1+1]-range[2*1]);
	int thr_z = dat0->block->instance->OPS_block_size_z * (range[2*2+1]-range[2*2]) *
		(range[2*3+1]-range[2*3]) *
		(range[2*4+1]-range[2*4]);

	int dim = dat0->dim;
	int type_size = dat0->type_size;
	int OPS_soa = dat0->block->instance->OPS_soa;
	int start0 = range[2*0], end0 = range[2*0+1];
#if OPS_MAX_DIM>1
	int start1 = range[2*1], end1 = range[2*1+1];
#if OPS_MAX_DIM>2
	int start2 = range[2*2], end2 = range[2*2+1];
#if OPS_MAX_DIM>3
	int start3 = range[2*3], end3 = range[2*3+1];
#if OPS_MAX_DIM>4
	int start4 = range[2*4], end4 = range[2*4+1];
#endif
#endif
#endif
#endif

	if (thr_x > 0 && thr_y > 0 && thr_z > 0) {
		#pragma omp target teams distribute parallel for collapse(3)
		for (int z = 0; z < thr_z; z++) {
			for (int y = 0; y < thr_y; y++) {
				for (int x = 0; x < thr_x; x++) {
					int i = start0 + x;
					int j = start1 + y;
					int rest = z;
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
			}
		}
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
	ops_pack_ompoffload_internal(dat,  src_offset, dest, halo->blocklength, halo->stride, halo->count);
}
void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src,
		const ops_int_halo *__restrict halo) {
	ops_unpack_ompoffload_internal(dat,  dest_offset, src, halo->blocklength, halo->stride, halo->count);
}


char* OPS_realloc_fast(char *ptr, size_t olds, size_t news) {
	if (OPS_instance::getOPSInstance()->OPS_gpu_direct) {
		if (ptr == NULL) {
			ops_device_malloc(OPS_instance::getOPSInstance(), (void **)&ptr, news);
			return ptr;
		} else {
			if (OPS_instance::getOPSInstance()->OPS_diags>3) printf("Warning: OpenMP Offload cache realloc\n");
			char *ptr2;
			ops_device_malloc(OPS_instance::getOPSInstance(), (void **)&ptr2, news);
			ops_device_memcpy_d2d(OPS_instance::getOPSInstance(), (void **)&ptr2, (void **)&ptr, olds);
			ops_device_free(OPS_instance::getOPSInstance(), (void **)&ptr);
			return ptr2;
		}
	} else {
		char *ptr2;
		ops_device_mallochost(OPS_instance::getOPSInstance(), (void **)&ptr2, news);
		if (olds > 0) memcpy(ptr2, ptr, olds);
		if (ptr != NULL) ops_device_free(OPS_instance::getOPSInstance(), (void **)&ptr);
		return ptr2;
	}
}
