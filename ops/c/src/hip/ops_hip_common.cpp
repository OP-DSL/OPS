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
  * @details Implements the HIP-specific routines shared between single-GPU 
  * and MPI+HIP backends
  */

#include <ops_hip_rt_support.h>

#include <hip/hiprand.h>
hiprandGenerator_t ops_rand_gen;
int hiprand_initialised = 0;

//
// HIP utility functions
//

void __hipSafeCall(std::ostream &stream, hipError_t err, const char *file, const int line) {
  if (hipSuccess != err) {
    fprintf2(stream, "%s(%i) : hipSafeCall() Runtime API error : %s.\n", file,
            line, hipGetErrorString(err));
    if (err == hipErrorNoBinaryForGpu) 
    throw OPSException(OPS_RUNTIME_ERROR, "Please make sure the OPS HIP/MPI+HIP backends were compiled for your GPU");
    else throw OPSException(OPS_RUNTIME_ERROR, hipGetErrorString(err));
  }
}

void __hiprandSafeCall(std::ostream &stream, hiprandStatus_t err, const char *file, const int line) {
  if (HIPRAND_STATUS_SUCCESS != err) {
    fprintf2(stream, "%s(%i) : hiprandSafeCall() Runtime API error : %s.\n", file,
            line, hipGetErrorString(hipGetLastError()));
    throw OPSException(OPS_RUNTIME_ERROR, "Please make sure the OPS HIP/MPI+HIP backends were compiled with hiprand for your GPU");
  }
}

void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  if ((instance->OPS_block_size_x * instance->OPS_block_size_y * instance->OPS_block_size_z) > 1024) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_*");
  }
  cutilDeviceInit(instance, argc, argv);
  instance->OPS_hybrid_gpu = 1;
  //hipSafeCall(instance->ostream(),hipDeviceSetCacheConfig(hipFuncCachePreferL1));

}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  hipSafeCall(instance->ostream(), hipMalloc(ptr, bytes));
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  hipSafeCall(instance->ostream(), hipMallocHost(ptr, bytes));
}

void ops_device_free(OPS_instance *instance, void** ptr) {
  hipSafeCall(instance->ostream(),hipFree(*ptr));
  *ptr = nullptr;
}

void ops_device_freehost(OPS_instance *instance, void** ptr) {
  hipSafeCall(instance->ostream(),hipHostFree(*ptr));
  *ptr = nullptr;
}

void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size) {
    hipSafeCall(instance->ostream(), hipMemcpy(*to, *from, size, hipMemcpyHostToDevice));
}

void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size) {
    hipSafeCall(instance->ostream(), hipMemcpy(*to, *from, size, hipMemcpyDeviceToHost));
}

void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size) {
    hipSafeCall(instance->ostream(), hipMemcpy(*to, *from, size, hipMemcpyDeviceToDevice));
}

void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size) {
    hipSafeCall(instance->ostream(), hipMemset(*ptr, val, size));
}

void ops_device_sync(OPS_instance *instance) {
  hipSafeCall(instance->ostream(), hipDeviceSynchronize());
}

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct hipDeviceProp_t cudaDeviceProp_t;

void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;
  int deviceCount;
  hipSafeCall(instance->ostream(), hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available HIP devices");
  }

  // Test we have access to a device

  float *test = 0;
  int my_id = ops_get_proc();
  instance->OPS_hybrid_gpu = 0;
  for (int i = 0; i < deviceCount; i++) {
    hipError_t err = hipSetDevice((i+my_id)%deviceCount);
    if (err == hipSuccess) {
      hipError_t err2 = hipMalloc((void **)&test, sizeof(float));
      if (err2 == hipSuccess) {
        instance->OPS_hybrid_gpu = 1;
        break;
      }
    }
  }
  if (instance->OPS_hybrid_gpu) {
    hipFree(test);

    int deviceId = -1;
    hipGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    hipSafeCall(instance->ostream(), hipGetDeviceProperties(&deviceProp, deviceId));
    if (instance->OPS_diags>2) instance->ostream() << "\n Using HIP device: " <<
      deviceId << " " << deviceProp.name;
  } else {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available HIP devices");
  }
}

void ops_randomgen_init(unsigned int seed, int options) {
  OPS_instance *instance = OPS_instance::getOPSInstance();

  /* Create pseudo-random number generator */
  hiprandSafeCall(instance->ostream(), hiprandCreateGenerator(&ops_rand_gen, HIPRAND_RNG_PSEUDO_DEFAULT));

  /* Set seed */
  int comm_global_size = ops_num_procs();
  int my_global_rank = ops_get_proc();

  if(comm_global_size == 0)
    hiprandSafeCall(instance->ostream(), hiprandSetPseudoRandomGeneratorSeed(ops_rand_gen, seed));
  else
    hiprandSafeCall(instance->ostream(), hiprandSetPseudoRandomGeneratorSeed(ops_rand_gen, seed*my_global_rank+my_global_rank));

  hiprand_initialised = 1;
}

void ops_fill_random_uniform(ops_dat dat) {
  OPS_instance *instance = OPS_instance::getOPSInstance();

  size_t cumsize = dat->dim;
  const char *type = dat->type;

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    cumsize *= dat->size[d];
  }

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 || strcmp(type, "real(kind=8)") == 0) {
    hiprandSafeCall(instance->ostream(), hiprandGenerateUniformDouble(ops_rand_gen, (double *)dat->data_d, cumsize));
  }
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0) {
    hiprandSafeCall(instance->ostream(), hiprandGenerateUniform(ops_rand_gen, (float *)dat->data_d, cumsize));
  }
  else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0 || strcmp(type, "integer(kind=4)") == 0) {
    hiprandSafeCall(instance->ostream(), hiprandGenerate(ops_rand_gen, (unsigned int *)dat->data_d, cumsize));
  }
  else {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: uniform random number generation not implemented for data type: "<<dat->type;
    throw ex;
  }

  dat->dirty_hd = 2;
  // set halo
  ops_stencil stencil = ops_dat_create_zeropt_stencil(dat);
  ops_arg arg = ops_arg_dat(dat, dat->dim, stencil, dat->type, OPS_WRITE);
  int *iter_range = new int[dat->block->dims*2];
  for ( int n = 0; n < dat->block->dims; n++) {
    iter_range[2*n] = 0;
    iter_range[2*n+1] = dat->size[n];
  }
  ops_set_halo_dirtybit3(&arg, iter_range);
}

void ops_fill_random_normal(ops_dat dat) {
  OPS_instance *instance = OPS_instance::getOPSInstance();

  size_t cumsize = dat->dim;
  const char *type = dat->type;

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    cumsize *= dat->size[d];
  }

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 || strcmp(type, "real(kind=8)") == 0) {
    hiprandSafeCall(instance->ostream(), hiprandGenerateNormalDouble(ops_rand_gen, (double *)dat->data_d, cumsize, 0.0, 1.0));
  }
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0) {
    hiprandSafeCall(instance->ostream(), hiprandGenerateNormal(ops_rand_gen, (float *)dat->data_d, cumsize, 0.0f, 1.0f));
  }
  else {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: normal random number generation not implemented for data type: "<<dat->type;
    throw ex;
  }

  dat->dirty_hd = 2;
  // set halo
  ops_stencil stencil = ops_dat_create_zeropt_stencil(dat);
  ops_arg arg = ops_arg_dat(dat, dat->dim, stencil, dat->type, OPS_WRITE);
  int *iter_range = new int[dat->block->dims*2];
  for ( int n = 0; n < dat->block->dims; n++) {
    iter_range[2*n] = 0;
    iter_range[2*n+1] = dat->size[n];
  }
  ops_set_halo_dirtybit3(&arg, iter_range);
}

void ops_randomgen_exit() {
  OPS_instance *instance = OPS_instance::getOPSInstance();

  if(hiprand_initialised) {
    hiprand_initialised = 0;
    hiprandSafeCall(instance->ostream(), hiprandDestroyGenerator(ops_rand_gen));
  }
}
