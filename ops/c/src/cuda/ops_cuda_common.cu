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
  * @brief OPS common cuda-specific functions (non-MPI and MPI)
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the CUDA-specific routines shared between single-GPU
  * and MPI+CUDA backends
  */

#include <ops_cuda_rt_support.h>

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

void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  if ((instance->OPS_block_size_x * instance->OPS_block_size_y * instance->OPS_block_size_z) > 1024) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_*");
  }
  cutilDeviceInit(instance, argc, argv);
  instance->OPS_hybrid_gpu = 1;
  cutilSafeCall(instance->ostream(),cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  cutilSafeCall(instance->ostream(), cudaMalloc(ptr, bytes));
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  cutilSafeCall(instance->ostream(), cudaMallocHost(ptr, bytes));
}

void ops_device_free(OPS_instance *instance, void** ptr) {
  cutilSafeCall(instance->ostream(),cudaFree(*ptr));
  *ptr = nullptr;
}

void ops_device_freehost(OPS_instance *instance, void** ptr) {
  cutilSafeCall(instance->ostream(),cudaFreeHost(*ptr));
  *ptr = nullptr;
}

void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size) {
    cutilSafeCall(instance->ostream(), cudaMemcpy(*to, *from, size, cudaMemcpyHostToDevice));
}

void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size) {
    cutilSafeCall(instance->ostream(), cudaMemcpy(*to, *from, size, cudaMemcpyDeviceToHost));
}

void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size) {
    cutilSafeCall(instance->ostream(), cudaMemcpy(*to, *from, size, cudaMemcpyDeviceToDevice));
}

void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size) {
    cutilSafeCall(instance->ostream(), cudaMemset(*ptr, val, size));
}

void ops_device_sync(OPS_instance *instance) {
  cutilSafeCall(instance->ostream(), cudaDeviceSynchronize());
}

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct cudaDeviceProp cudaDeviceProp_t;

void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;
  int deviceCount;
  cutilSafeCall(instance->ostream(), cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available CUDA devices");
  }

  // Test we have access to a device

  float *test = 0;
  int my_id = ops_get_proc();
  instance->OPS_hybrid_gpu = 0;
  for (int i = 0; i < deviceCount; i++) {
    cudaError_t err = cudaSetDevice((i+my_id)%deviceCount);
    if (err == cudaSuccess) {
      cudaError_t err2 = cudaMalloc((void **)&test, sizeof(float));
      if (err2 == cudaSuccess) {
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
    if (instance->OPS_diags>=1) instance->ostream() << "\n Using CUDA device: " <<
      deviceId << " " << deviceProp.name <<"\n";
  } else {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available CUDA devices");
  }
}
