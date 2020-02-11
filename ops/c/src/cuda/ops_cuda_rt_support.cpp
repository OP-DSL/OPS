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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <ops_cuda_rt_support.h>
#include <ops_lib_core.h>
#include <ops_exceptions.h>

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct cudaDeviceProp cudaDeviceProp_t;

//
// CUDA utility functions
//

void __cudaSafeCall(std::ostream &stream, cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf2(stream, "%s(%i) : cutilSafeCall() Runtime API error : %s.\n", file,
            line, cudaGetErrorString(err));
    if (err == cudaErrorNoKernelImageForDevice) 
    throw OPSException(OPS_RUNTIME_ERROR, "Please make sure the OPS CUDA/MPI+CUDA backends were compiled for your GPU");
    else throw OPSException(OPS_RUNTIME_ERROR, cudaGetErrorString(err));
  }
}

void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;
  int deviceCount;
  cutilSafeCall(instance->ostream(), cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available CUDA devices");
  }

  // Test we have access to a device

  float *test;
  int my_id = ops_get_proc();
  instance->OPS_hybrid_gpu = 0;
  for (int i = 0; i < deviceCount; i++) {
    cudaError_t err = cudaSetDevice((i+my_id)%deviceCount);
    if (err == cudaSuccess) {
      cudaError_t err = cudaMalloc((void **)&test, sizeof(float));
      if (err == cudaSuccess) {
        instance->OPS_hybrid_gpu = 1;
        break;
      }
    }
  }
  if (instance->OPS_hybrid_gpu) {
    cudaFree(test);

    int deviceId = -1;
    cudaGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    cutilSafeCall(instance->ostream(), cudaGetDeviceProperties(&deviceProp, deviceId));
    if (instance->OPS_diags>2) instance->ostream() << "\n Using CUDA device: " <<
      deviceId << " " << deviceProp.name;
  } else {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available CUDA devices");
  }
}

void ops_cpHostToDevice(OPS_instance *instance, void **data_d, void **data_h, size_t size) {

  if ( *data_d == NULL )
      cutilSafeCall(instance->ostream(), cudaMalloc(data_d, size));

  if (data_h == NULL || *data_h == NULL) {
    cutilSafeCall(instance->ostream(), cudaMalloc(data_d, size));
    cutilSafeCall(instance->ostream(), cudaMemset(*data_d, 0, size));
    return;
  }

  /*static void* stage = NULL;
  static size_t stage_size = 0;

  void *src = NULL;
  if ( size < 4*1024*1024 ) {
      if ( (size_t)size > stage_size ) {
          if ( stage ) cudaFreeHost(stage);
          stage_size = size;
          cutilSafeCall(instance->ostream(), cudaMallocHost(&stage, stage_size));
      }

      memcpy(stage, *data_h, size);
      src = stage;
  } else {
      src = *data_h;
  }*/

  cutilSafeCall(instance->ostream(), cudaMemcpy(*data_d, *data_h, size, cudaMemcpyHostToDevice));
}

void ops_download_dat(ops_dat dat) {

  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  // printf("downloading to host from device %d bytes\n",bytes);
  cutilSafeCall(dat->block->instance->ostream(), 
      cudaMemcpy(dat->data, dat->data_d, bytes, cudaMemcpyDeviceToHost));
}

void ops_upload_dat(ops_dat dat) {

  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  // printf("uploading to device from host %d bytes\n",bytes);
  cutilSafeCall(dat->block->instance->ostream(), 
      cudaMemcpy(dat->data_d, dat->data, bytes, cudaMemcpyHostToDevice));
}

void ops_H_D_exchanges_host(ops_arg *args, int nargs) {
  // printf("in ops_H_D_exchanges\n");
  for (int n = 0; n < nargs; n++)
    if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 2) {
      ops_download_dat(args[n].dat);
      // printf("halo exchanges on host\n");
      args[n].dat->dirty_hd = 0;
    }
}

void ops_H_D_exchanges_device(ops_arg *args, int nargs) {
  for (int n = 0; n < nargs; n++)
    if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 1) {
      ops_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
}

void ops_set_dirtybit_device(ops_arg *args, int nargs) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].argtype == OPS_ARG_DAT) &&
        (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE ||
         args[n].acc == OPS_RW)) {
      args[n].dat->dirty_hd = 2;
    }
  }
}


//set dirty bit for single ops_arg dat
void ops_set_dirtybit_device_dat(ops_dat dat) {
  dat->dirty_hd = 2;
}

//
// routine to fetch data from GPU to CPU (with transposing SoA to AoS if needed)
//

void ops_cuda_get_data(ops_dat dat) {
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  cutilSafeCall(dat->block->instance->ostream(), 
      cudaMemcpy(dat->data, dat->data_d, bytes, cudaMemcpyDeviceToHost));
  cutilSafeCall(dat->block->instance->ostream(), cudaDeviceSynchronize());
}

//
// routine to upload data from CPU to GPU (with transposing SoA to AoS if needed)
//

void ops_cuda_put_data(ops_dat dat) {
  if (dat->dirty_hd == 1)
    dat->dirty_hd = 0;
  else
    return;
  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  cutilSafeCall(dat->block->instance->ostream(), 
      cudaMemcpy(dat->data_d, dat->data, bytes, cudaMemcpyHostToDevice));
  cutilSafeCall(dat->block->instance->ostream(), cudaDeviceSynchronize());
}


//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(OPS_instance *instance, int consts_bytes) {
  if (consts_bytes > instance->OPS_consts_bytes) {
    if (instance->OPS_consts_bytes > 0) {
      free(instance->OPS_consts_h);
      cudaFreeHost(instance->OPS_gbl_prev);
      cutilSafeCall(instance->ostream(), cudaFree(instance->OPS_consts_d));
    }
    instance->OPS_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
    cudaMallocHost((void **)&instance->OPS_gbl_prev, instance->OPS_consts_bytes);
    instance->OPS_consts_h = (char *)ops_malloc(instance->OPS_consts_bytes);
    cutilSafeCall(instance->ostream(), cudaMalloc((void **)&instance->OPS_consts_d, instance->OPS_consts_bytes));
  }
}

void reallocReductArrays(OPS_instance *instance, int reduct_bytes) {
  if (reduct_bytes > instance->OPS_reduct_bytes) {
    if (instance->OPS_reduct_bytes > 0) {
      free(instance->OPS_reduct_h);
      cutilSafeCall(instance->ostream(), cudaFree(instance->OPS_reduct_d));
    }
    instance->OPS_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
    instance->OPS_reduct_h = (char *)ops_malloc(instance->OPS_reduct_bytes);
    cutilSafeCall(instance->ostream(), cudaMalloc((void **)&instance->OPS_reduct_d, instance->OPS_reduct_bytes));
  }
}

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice(OPS_instance *instance, int consts_bytes) {
  instance->OPS_gbl_changed = 0;
  for (int i = 0; i < consts_bytes; i++) {
    if (instance->OPS_consts_h[i] != instance->OPS_gbl_prev[i])
      instance->OPS_gbl_changed = 1;
  }
  if (instance->OPS_gbl_changed) {
    // memcpy(instance->OPS_gbl_prev,instance->OPS_consts_h,consts_bytes);
    // cutilSafeCall ( cudaMemcpyAsync ( instance->OPS_consts_d, instance->OPS_gbl_prev,
    // consts_bytes,
    //                             cudaMemcpyHostToDevice ) );
    cutilSafeCall(instance->ostream(), cudaMemcpy(instance->OPS_consts_d, instance->OPS_consts_h, consts_bytes,
                             cudaMemcpyHostToDevice));
    memcpy(instance->OPS_gbl_prev, instance->OPS_consts_h, consts_bytes);
  }
}

void mvReductArraysToDevice(OPS_instance *instance, int reduct_bytes) {
  cutilSafeCall(instance->ostream(), cudaMemcpy(instance->OPS_reduct_d, instance->OPS_reduct_h, reduct_bytes,
                           cudaMemcpyHostToDevice));
}

void mvReductArraysToHost(OPS_instance *instance, int reduct_bytes) {
  cutilSafeCall(instance->ostream(), cudaMemcpy(instance->OPS_reduct_h, instance->OPS_reduct_d, reduct_bytes,
                           cudaMemcpyDeviceToHost));
}

void ops_cuda_exit(OPS_instance *instance) {
  if (!instance->OPS_hybrid_gpu)
    return;
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &instance->OPS_dat_list, entries) {
    cutilSafeCall(instance->ostream(), cudaFree((item->dat)->data_d));
  }
  if (instance->OPS_consts_bytes > 0) {
    free(instance->OPS_consts_h);
    cudaFreeHost(instance->OPS_gbl_prev);
    cutilSafeCall(instance->ostream(), cudaFree(instance->OPS_consts_d));
  }
  if (instance->OPS_reduct_bytes > 0) {
    free(instance->OPS_reduct_h);
    cutilSafeCall(instance->ostream(), cudaFree(instance->OPS_reduct_d));
  }
}
