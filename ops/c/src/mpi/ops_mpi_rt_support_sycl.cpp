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

/*
__global__ void ops_cuda_packer_1(const char *__restrict src,
                                  char *__restrict dest, int count, int len,
                                  int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int block = idx / len;
  if (idx < count * len) {
    dest[idx] = src[stride * block + idx % len];
  }
}

__global__ void ops_cuda_packer_1_soa(const char *__restrict src,
                                  char *__restrict dest, int count, int len,
                                  int stride, int dim, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
                                  int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
*/

void ops_pack_sycl_internal(ops_dat dat, const int src_offset, char *__restrict dest,
              const int halo_blocklength, const int halo_stride, const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
    dat->dirty_hd = 0;
  }

  const char *__restrict src = dat->data_d + src_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size);
  if (halo_buffer_size < halo_count * halo_blocklength * dat->dim && !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (halo_buffer_d != NULL){
	  cl::sycl::buffer<char,1> * halo_buffer_sycl = static_cast<cl::sycl::buffer<char,1> *>((void*)halo_buffer_d);
	  delete halo_buffer_sycl;
      //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaFree(halo_buffer_d));
	}
	cl::sycl::buffer<char,1> * halo_buffer_sycl = new cl::sycl::buffer<char,1>(cl::sycl::range<1>(halo_count * halo_blocklength * dat->dim * 4));
	halo_buffer_d = (char*) halo_buffer_sycl;
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMalloc((void **)&halo_buffer_d, halo_count * halo_blocklength * dat->dim * 4));
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }
  char *device_buf = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    device_buf = dest;
  else
    device_buf = halo_buffer_d;

  cl::sycl::buffer<char,1> * src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)src);
  cl::sycl::buffer<char,1> * dest_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)device_buf);
  
  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
	
	//ide kernel kell!! tartalma az ops_cuda_packer_1_soa
	dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		
		//parloop
		cgh.parallel_for<class ops_sycl_packer_1_soa>(cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads), cl::sycl::range<1>(num_threads)), [=](cl::sycl::nd_item<1> item) {
			cl::sycl::cl_int global_x_id = item.get_global_id()[0];
			cl::sycl::cl_int block = global_x_id / halo_blocklength;
			if(global_x_id < halo_count * halo_blocklength) {
				for (int d=0; d<dat->dim; d++) {
					dest_acc[global_x_id * dat->dim + d] = src_acc[halo_stride * block + global_x_id % halo_blocklength + d * dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size];
				}
			}
		});
	});
    /*
	ops_cuda_packer_1_soa<<<num_blocks, num_threads>>>(
        src, device_buf, halo_count, halo_blocklength, halo_stride,
        dat->dim, dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size);
	*/
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());

  } else if (halo_blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo_blocklength / 4) * halo_count) - 1) / num_threads + 1;
	//ide kernel kell!! tartalma az ops_cuda_packer_4
	dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		
		//parloop
		cgh.parallel_for<class ops_sycl_packer_4>(cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads), cl::sycl::range<1>(num_threads)), [=](cl::sycl::nd_item<1> item) {
			cl::sycl::cl_int global_x_id = item.get_global_id()[0];
			cl::sycl::cl_int block = global_x_id / halo_blocklength*dat->dim / 4;
			if(global_x_id < halo_count * halo_blocklength*dat->dim / 4) {
				dest_acc[global_x_id] = src_acc[halo_stride*dat->dim / 4 * block + global_x_id % halo_blocklength*dat->dim / 4];
			}
		});
	});
	/*
    ops_cuda_packer_4<<<num_blocks, num_threads>>>(
        (const int *)src, (int *)device_buf, halo_count, halo_blocklength*dat->dim / 4,
        halo_stride*dat->dim / 4);
	*/
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
  } else {
    int num_threads = 128;
    int num_blocks = ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
	//ide kernel kell!! tartalma az ops_cuda_packer_1
	dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		
		//parloop
		cgh.parallel_for<class ops_sycl_packer_1>(cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads), cl::sycl::range<1>(num_threads)), [=](cl::sycl::nd_item<1> item) {
			cl::sycl::cl_int global_x_id = item.get_global_id()[0];
			cl::sycl::cl_int block = global_x_id / halo_blocklength*dat->dim;
			if(global_x_id < halo_count * halo_blocklength*dat->dim) {
				dest_acc[global_x_id] = src_acc[halo_stride*dat->dim * block + global_x_id % halo_blocklength*dat->dim];
			}
		});
	});
	/*
    ops_cuda_packer_1<<<num_blocks, num_threads>>>(
        src, device_buf, halo_count, halo_blocklength*dat->dim, halo_stride*dat->dim);
    */
	//cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
  }
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct)
	ops_sycl_memcpyDeviceToHost(dat->block->instance, dest_buff, dest, halo_blocklength * dat->dim);
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMemcpy(dest, halo_buffer_d, halo_count * halo_blocklength * dat->dim, cudaMemcpyDeviceToHost));
  else
	dat->block->instance->sycl_instance->queue->wait();
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaDeviceSynchronize());
}

void ops_unpack_sycl_internal(ops_dat dat, const int dest_offset, const char *__restrict src,
                const int halo_blocklength, const int halo_stride, const int halo_count) {

  if (dat->dirty_hd == 1) {
    ops_upload_dat(dat);
    dat->dirty_hd = 0;
  }
  char *__restrict dest = dat->data_d + dest_offset * (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size);
  if (halo_buffer_size < halo_count * halo_blocklength * dat->dim && !OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (halo_buffer_d != NULL){
      cl::sycl::buffer<char,1> * halo_buffer_sycl = static_cast<cl::sycl::buffer<char,1> *>((void*)halo_buffer_d);
	  delete halo_buffer_sycl;
	  //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaFree(halo_buffer_d));
	}
    cl::sycl::buffer<char,1> * halo_buffer_sycl = new cl::sycl::buffer<char,1>(cl::sycl::range<1>(halo_count * halo_blocklength * dat->dim * 4));
	halo_buffer_d = (char*) halo_buffer_sycl;
	//cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMalloc((void **)&halo_buffer_d, halo_count * halo_blocklength * dat->dim * 4));
    halo_buffer_size = halo_count * halo_blocklength * dat->dim * 4;
  }

  const char *device_buf = NULL;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    device_buf = src;
  else
    device_buf = halo_buffer_d;

  cl::sycl::buffer<char,1> * src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)device_buf);
  cl::sycl::buffer<char,1> * dest_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)dest);
  
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct){
    ops_sycl_memcpyHostToDevice(dat->block->instance, src_buff, halo_buffer_d, halo_count * halo_blocklength * dat->dim);
	//cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMemcpy(halo_buffer_d, src, halo_count * halo_blocklength * dat->dim, cudaMemcpyHostToDevice));
  }
  if (OPS_instance::getOPSInstance()->OPS_soa) {
    int num_threads = 128;
    int num_blocks = ((halo_blocklength * halo_count) - 1) / num_threads + 1;
	//ide kernel kell!! tartalma az ops_cuda_unpacker_1_soa
    dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		
		//parloop
		cgh.parallel_for<class ops_sycl_packer_1_soa>(cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads), cl::sycl::range<1>(num_threads)), [=](cl::sycl::nd_item<1> item) {
			cl::sycl::cl_int global_x_id = item.get_global_id()[0];
			cl::sycl::cl_int block = global_x_id / halo_blocklength;
			if(global_x_id < halo_count * halo_blocklength) {
				for (int d=0; d<dat->dim; d++) {
					dest_acc[halo_stride * block + global_x_id % halo_blocklength + d * dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size] = src_acc[global_x_id * dat->dim + d];
				}
			}
		});
	});
	/*
	ops_cuda_unpacker_1_soa<<<num_blocks, num_threads>>>(
        device_buf, dest, halo_count, halo_blocklength, halo_stride,
        dat->dim, dat->size[0]*dat->size[1]*dat->size[2]*dat->type_size);
    cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
	*/
  } else if (halo_blocklength % 4 == 0) {
    int num_threads = 128;
    int num_blocks =
        (((dat->dim * halo_blocklength / 4) * halo_count) - 1) / num_threads + 1;
	//ide kernel kell!! tartalma az ops_cuda_unpacker_4
    dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		
		//parloop
		cgh.parallel_for<class ops_sycl_packer_4>(cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads), cl::sycl::range<1>(num_threads)), [=](cl::sycl::nd_item<1> item) {
			cl::sycl::cl_int global_x_id = item.get_global_id()[0];
			cl::sycl::cl_int block = global_x_id / halo_blocklength*dat->dim / 4;
			if(global_x_id < halo_count * halo_blocklength*dat->dim / 4) {
				dest_acc[halo_stride*dat->dim / 4 * block + global_x_id % halo_blocklength*dat->dim / 4] = src_acc[global_x_id];
			}
		});
	});
	/*
	ops_cuda_unpacker_4<<<num_blocks, num_threads>>>(
        (const int *)device_buf, (int *)dest, halo_count,
        halo_blocklength*dat->dim / 4, halo_stride*dat->dim / 4);
    cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
	*/
  } else {
    int num_threads = 128;
    int num_blocks = ((dat->dim * halo_blocklength * halo_count) - 1) / num_threads + 1;
	//ide kernel kell!! tartalma az ops_cuda_unpacker_1
    dat->block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		
		//parloop
		cgh.parallel_for<class ops_sycl_packer_1>(cl::sycl::nd_range<1>(cl::sycl::range<1>(num_blocks * num_threads), cl::sycl::range<1>(num_threads)), [=](cl::sycl::nd_item<1> item) {
			cl::sycl::cl_int global_x_id = item.get_global_id()[0];
			cl::sycl::cl_int block = global_x_id / halo_blocklength*dat->dim;
			if(global_x_id < halo_count * halo_blocklength*dat->dim) {
				dest_acc[halo_stride*dat->dim * block + global_x_id % halo_blocklength*dat->dim] = src_acc[global_x_id];
			}
		});
	});
	/*
	ops_cuda_unpacker_1<<<num_blocks, num_threads>>>(
        device_buf, dest, halo_count, halo_blocklength*dat->dim, halo_stride*dat->dim);
    cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
	*/
  }

  dat->dirty_hd = 2;
}

//ITT MI VAN?
char* OPS_realloc_fast(char *ptr, size_t olds, size_t news) {
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct) {
    if (ptr == NULL) {
	  cl::sycl::buffer<char,1> * ptr_buff = new cl::sycl::buffer<char,1>(cl::sycl::range<1>(news));
	  ptr = (char*) ptr_buff;
      //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMalloc((void **)&ptr, news));
      return ptr;
    } else {
      if (OPS_instance::getOPSInstance()->OPS_diags>3) printf("Warning: cuda cache realloc\n");
      char *ptr2;
	  cl::sycl::buffer<char,1> * ptr2_buff = new cl::sycl::buffer<char,1>(cl::sycl::range<1>(news));
	  ptr2 = (char*) ptr2_buff;
	  cl::sycl::buffer<char,1> * ptr_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)ptr);
      //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMalloc((void **)&ptr2, news));
      ops_sycl_memcpyDeviceToDevice(OPS_instance::getOPSInstance(), ptr_buff, ptr2_buff, olds);
	  //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMemcpy(ptr2, ptr, olds, cudaMemcpyDeviceToDevice));
	  delete ptr_buff;
      //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaFree(ptr));
      return ptr2;
    }
  } else {
    /*char *ptr2;
    cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),
	cudaMallocHost((void**)&ptr2,news)); //TODO: is this aligned??
    if (olds > 0)
  	  memcpy(ptr2, ptr, olds);
    if (ptr != NULL) cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaFreeHost(ptr));
    return ptr2;
	*/
	return (char*) ops_realloc((void*) ptr, news);
  }
}
/*
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
*/
/*
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
*/
void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {

  
  ops_block block = src->block;
  
  //dest += dest_offset;
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
  
  //EZ MI?(A)
  char *gpu_ptr;
  if (OPS_instance::getOPSInstance()->OPS_gpu_direct)
    gpu_ptr = dest;
  else {
    if (halo_buffer_size < size) {
      if (halo_buffer_d != NULL){
		cl::sycl::buffer<char,1> * halo_buffer_sycl = static_cast<cl::sycl::buffer<char,1> *>((void*)halo_buffer_d);
		delete halo_buffer_sycl;
		//cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaFree(halo_buffer_d));
	  }
	  cl::sycl::buffer<char,1> * halo_buffer_sycl = new cl::sycl::buffer<char,1>(cl::sycl::range<1>(size));
	  halo_buffer_d = (char*) halo_buffer_sycl;
      //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMalloc((void **)&halo_buffer_d, size * sizeof(char)));
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
  }
  //EZ(A)

  if (src->dirty_hd == 1) {
    ops_upload_dat(src);
    src->dirty_hd = 0;
  }

  //dim3 grid(blk_x, blk_y, blk_z);
  //dim3 tblock(thr_x, thr_y, thr_z);
	int size_x = src->size[0];
	int size_y = src->size[1];
	int size_z = src->size[2];
	int type_size = src->type_size;
	int dim = src->dim;
	int OPS_soa = block->instance->OPS_soa;
	
	cl::sycl::buffer<char,1> *dest_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)gpu_ptr);
	cl::sycl::buffer<char,1> *src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)src->data_d);
	
	
	block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {		//Queue->Submit
		//Accessors
		auto dest_acc = dest_buff->get_access<cl::sycl::access::mode::read_write>(cgh);
		auto src_acc = src_buff->get_access<cl::sycl::access::mode::read_write>(cgh);

		//nd_range elso argumentume a teljes méret, nem a blokkok száma: https://docs.oneapi.com/versions/latest/dpcpp/iface/nd_range.html
		cgh.parallel_for<class copy_tobuf>(cl::sycl::nd_range<3>(cl::sycl::range<3>(blk_z*thr_z,blk_y*thr_y,blk_x*thr_x),cl::sycl::range<3>(thr_z,thr_y,thr_x)), [=](cl::sycl::nd_item<3> item) {
			//get x dimension id
			cl::sycl::cl_int global_x_id = item.get_global_id()[2];
			//get y dimension id
			cl::sycl::cl_int global_y_id = item.get_global_id()[1];
			//get z dimension id
			cl::sycl::cl_int global_z_id = item.get_global_id()[0];

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
/*  copy_kernel_tobuf<<<grid, tblock>>>(
      gpu_ptr, src->data_d, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, src->size[0], src->size[1], src->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, src->type_size, src->dim, OPS_instance::getOPSInstance()->OPS_soa);
*/

  //EZ MI?(B)
  if (!OPS_instance::getOPSInstance()->OPS_gpu_direct)
	ops_sycl_memcpyDeviceToHost(block->instance, dest_buff, dest, size * sizeof(char));
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMemcpy(dest, halo_buffer_d, size * sizeof(char), cudaMemcpyDeviceToHost));
  //EZ(B)
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {

  ops_block block = dest->block;
  
  //src += src_offset;
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
      if (halo_buffer_d != NULL){
		cl::sycl::buffer<char,1> * halo_buffer_sycl = static_cast<cl::sycl::buffer<char,1> *>((void*)halo_buffer_d);
		delete halo_buffer_sycl;
		//cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaFree(halo_buffer_d));
		
	  }
	  cl::sycl::buffer<char,1> * halo_buffer_sycl = new cl::sycl::buffer<char,1>(cl::sycl::range<1>(size));
	  halo_buffer_d = (char*) halo_buffer_sycl;
      //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMalloc((void **)&halo_buffer_d, size * sizeof(char)));
      halo_buffer_size = size;
    }
    gpu_ptr = halo_buffer_d;
	cl::sycl::buffer<char,1> *src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)gpu_ptr);
	ops_sycl_memcpyHostToDevice(block->instance, src_buff, halo_buffer_d, size * sizeof(char));
    //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaMemcpy(halo_buffer_d, src, size * sizeof(char), cudaMemcpyHostToDevice));
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
	
	cl::sycl::buffer<char,1> *src_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)gpu_ptr);
	
	cl::sycl::buffer<char,1> *dest_buff = static_cast<cl::sycl::buffer<char,1> *>((void*)dest->data_d);

  //dim3 grid(blk_x, blk_y, blk_z);
  //dim3 tblock(thr_x, thr_y, thr_z);
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
      dest->data_d, gpu_ptr, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, dest->size[0], dest->size[1], dest->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, dest->type_size, dest->dim, OPS_instance::getOPSInstance()->OPS_soa);
  */
  //cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),cudaGetLastError());
  dest->dirty_hd = 2;
}

/*
__global__ void ops_internal_copy_sycl_kernel(char * dat0_p, char *dat1_p,
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
  int i = start0 + threadIdx.x + blockIdx.x*blockDim.x;
  int j = start1 + threadIdx.y + blockIdx.y*blockDim.y;
  int rest = threadIdx.z + blockIdx.z*blockDim.z;
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
*/

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

  dim3 grid((range[2*0+1]-range[2*0] - 1) / dat0->block->instance->OPS_block_size_x + 1,
            (range[2*1+1]-range[2*1] - 1) / dat0->block->instance->OPS_block_size_y + 1,
           ((range[2*2+1]-range[2*2] - 1) / dat0->block->instance->OPS_block_size_z + 1) *
            (range[2*3+1]-range[2*3]) *
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
    dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[0]);
    dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[1]);
  }
*/
}

void ops_dat_fetch_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
/*
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

void ops_dat_set_data_slab_memspace(ops_dat dat, int part, char *data, int *range, ops_memspace memspace) {
  throw OPSException(OPS_NOT_IMPLEMENTED, "Error: Not implemented");
/*
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


void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
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

void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace) {
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

