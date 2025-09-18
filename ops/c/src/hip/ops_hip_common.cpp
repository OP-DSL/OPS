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

#include <hiprand/hiprand.h>
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
  hipDeviceSetCacheConfig(hipFuncCachePreferL1);
}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  hipSafeCall(instance->ostream(), hipMalloc(ptr, bytes));
  instance->ops_device_memory_allocated_bytes += bytes;
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  hipSafeCall(instance->ostream(), hipHostMalloc(ptr, bytes));
  instance->ops_host_memory_allocated_bytes += bytes;
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
  
  if (instance->OPS_device_id >= 0) {
    // User specified a device ID
    if (instance->OPS_device_id >= deviceCount) {
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: specified HIP device ID exceeds available device count");
    }
    hipError_t err = hipSetDevice(instance->OPS_device_id);
    if (err == hipSuccess) {
      hipError_t err2 = hipMalloc((void **)&test, sizeof(float));
      if (err2 == hipSuccess) {
        instance->OPS_hybrid_gpu = 1;
      } else {
        throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: specified HIP device is not accessible");
      }
    } else {
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: failed to set specified HIP device");
    }
  } else {
    // Auto-select device
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
  }
  if (instance->OPS_hybrid_gpu) {
    hipFree(test);

    int deviceId = -1;
    hipGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    hipSafeCall(instance->ostream(), hipGetDeviceProperties(&deviceProp, deviceId));
    if (instance->OPS_diags>=1) instance->ostream() << "\n Using HIP device: " <<
      deviceId << " " << deviceProp.name <<"\n";

    // Initialize GPU power measurement
    _ops_reset_gpu_power_counters(instance);
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

  if(comm_global_size == 1)
    hiprandSafeCall(instance->ostream(), hiprandSetPseudoRandomGeneratorSeed(ops_rand_gen, seed));
  else
    hiprandSafeCall(instance->ostream(), hiprandSetPseudoRandomGeneratorSeed(ops_rand_gen, seed + my_global_rank * 2654435761u));

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
  ops_arg arg = ops_arg_dat(dat, dat->dim, instance->OPS_internal_0[dat->block->dims -1], dat->type, OPS_WRITE);
  int *iter_range{new int[dat->block->dims*2]};
  for ( int n = 0; n < dat->block->dims; n++) {
    iter_range[2*n] = 0;
    iter_range[2*n+1] = dat->size[n];
  }
  ops_set_halo_dirtybit3(&arg, iter_range);
  delete[] iter_range;
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
  ops_arg arg = ops_arg_dat(dat, dat->dim, instance->OPS_internal_0[dat->block->dims -1], dat->type, OPS_WRITE);
  int *iter_range{new int[dat->block->dims*2]};
  for ( int n = 0; n < dat->block->dims; n++) {
    iter_range[2*n] = 0;
    iter_range[2*n+1] = dat->size[n];
  }
  ops_set_halo_dirtybit3(&arg, iter_range);
  delete[] iter_range;
}

void ops_randomgen_exit() {
  OPS_instance *instance = OPS_instance::getOPSInstance();

  if(hiprand_initialised) {
    hiprand_initialised = 0;
    hiprandSafeCall(instance->ostream(), hiprandDestroyGenerator(ops_rand_gen));
  }
}

/*
 * GPU Power Measurement Functions using ROCm SMI
 */

#if OPS_ENABLE_ROCM_SMI
#include <rocm_smi/rocm_smi.h>

// ROCm SMI device handle for current GPU
static uint32_t rocm_device_id = 0;
static bool rocm_smi_initialized = false;
static uint32_t rocm_num_devices = 0;

void __rocmSmiSafeCall(rsmi_status_t result, const char *file, int line) {
    if (result != RSMI_STATUS_SUCCESS) {
        fprintf(stderr, "ROCm SMI error at %s:%d - error code %d\n", file, line, result);
        // Don't throw exception, just disable GPU power measurement
        rocm_smi_initialized = false;
    }
}

#define rocmSmiSafeCall(call) __rocmSmiSafeCall(call, __FILE__, __LINE__)

#endif // OPS_ENABLE_ROCM_SMI

void _ops_init_gpu_power_measurement(OPS_instance *instance) {
#if OPS_ENABLE_ROCM_SMI
    rsmi_status_t result;
    
    // Initialize ROCm SMI
    result = rsmi_init(0);
    if (result != RSMI_STATUS_SUCCESS) {
        if (instance->OPS_diags > 4) {
            instance->ostream() << "Warning: Failed to initialize ROCm SMI for GPU power measurement: error " 
                               << result << std::endl;
        }
        rocm_smi_initialized = false;
        return;
    }
    
    // Get number of monitoring devices
    result = rsmi_num_monitor_devices(&rocm_num_devices);
    if (result != RSMI_STATUS_SUCCESS || rocm_num_devices == 0) {
        if (instance->OPS_diags > 4) {
            instance->ostream() << "Warning: No ROCm SMI monitoring devices found" << std::endl;
        }
        rocm_smi_initialized = false;
        return;
    }
    
    // Get current HIP device (assumes it matches ROCm SMI device index)
    int deviceId = -1;
    hipError_t hip_result = hipGetDevice(&deviceId);
    if (hip_result != hipSuccess || deviceId < 0 || (uint32_t)deviceId >= rocm_num_devices) {
        if (instance->OPS_diags > 4) {
            instance->ostream() << "Warning: Failed to get current HIP device for power measurement" << std::endl;
        }
        rocm_smi_initialized = false;
        return;
    }
    
    rocm_device_id = (uint32_t)deviceId;
    rocm_smi_initialized = true;
    
    if (instance->OPS_diags > 4) {
        instance->ostream() << "GPU power measurement initialized using ROCm SMI for device " 
                           << rocm_device_id << std::endl;
    }
#else
    (void)instance; // Suppress unused parameter warning
    rocm_smi_initialized = false;
    if (instance->OPS_diags > 4) {
        instance->ostream() << "GPU power measurement not available (ROCm SMI disabled)" << std::endl;
    }
#endif
}

void _ops_get_gpu_power(OPS_instance *instance, unsigned int *power_watts) {
    (void)instance; // Suppress unused parameter warning
    
    *power_watts = 0; // Default to 0 if measurement fails
    
#if OPS_ENABLE_ROCM_SMI
    if (!rocm_smi_initialized) {
        return;
    }
    
    uint64_t power_microwatts = 0;
    rsmi_status_t result;
    
    // Use newer API for ROCm 6.0+ or fall back to older API
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 6)
    // ROCm 6.0+ - use rsmi_dev_power_get
    RSMI_POWER_TYPE power_type;
    result = rsmi_dev_power_get(rocm_device_id, &power_microwatts, &power_type);
    if (result != RSMI_STATUS_SUCCESS || power_type == RSMI_INVALID_POWER) {
        if (instance->OPS_diags > 3) {
            instance->ostream() << "Warning: Failed to get GPU power usage from ROCm SMI (6+ API): error " 
                               << result << std::endl;
        }
        *power_watts = 0;
        return;
    }
#else
    // ROCm 5.x and earlier - use rsmi_dev_power_ave_get
    result = rsmi_dev_power_ave_get(rocm_device_id, 0, &power_microwatts);
    if (result != RSMI_STATUS_SUCCESS) {
        if (instance->OPS_diags > 3) {
            instance->ostream() << "Warning: Failed to get GPU power usage from ROCm SMI (5.x API): error " 
                               << result << std::endl;
        }
        *power_watts = 0;
        return;
    }
#endif
    
    // Convert from microwatts to watts
    *power_watts = (unsigned int)(power_microwatts / 1000000);
#endif
}

void _ops_finalize_gpu_power_measurement(OPS_instance *instance) {
    (void)instance; // Suppress unused parameter warning
    
#if OPS_ENABLE_ROCM_SMI
    if (rocm_smi_initialized) {
        rsmi_shut_down();
        rocm_smi_initialized = false;
        rocm_device_id = 0;
        rocm_num_devices = 0;
    }
#endif
}
