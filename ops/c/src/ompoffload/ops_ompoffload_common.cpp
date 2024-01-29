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
  * @details Implements the OpenMP Offload-specific routines shared between single-GPU 
  * and MPI+HIP backends
  */

#include <omp.h>
#include <ops_lib_core.h>

void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;
  (void)instance;
}

void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  if ((instance->OPS_block_size_x * instance->OPS_block_size_y * instance->OPS_block_size_z) > 1024) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_*");
  }
  cutilDeviceInit(instance, argc, argv);
  int my_id = ops_get_proc();
  int no_of_devices = omp_get_num_devices();
  omp_set_default_device(my_id % no_of_devices);
  instance->OPS_hybrid_gpu = 1;
}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  *ptr = ops_malloc(bytes);
  int device = omp_get_default_device();

  void* device_ptr = omp_target_alloc(bytes, device);
  omp_target_associate_ptr(*ptr, device_ptr, bytes, 0, device);
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  *ptr = ops_malloc(bytes);
}

void ops_device_free(OPS_instance *instance, void** ptr) {
  int device = omp_get_default_device();

  void* device_ptr = omp_get_mapped_ptr(*ptr, device);
  omp_target_disassociate_ptr(*ptr, device);
  omp_target_free(device_ptr, device);
  ops_free(*ptr);
  *ptr = nullptr;
}

void ops_device_freehost(OPS_instance *instance, void** ptr) {
  ops_free(*ptr);
  *ptr = nullptr;
}

void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size) {
  int host = omp_get_initial_device();
  int device = omp_get_default_device();

  void* device_ptr = omp_get_mapped_ptr(*to, device);
  omp_target_memcpy(device_ptr, *from, size, 0, 0, device, host);
}

void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size) {
  int host = omp_get_initial_device();
  int device = omp_get_default_device();

  void* device_ptr = omp_get_mapped_ptr(*from, device);
  omp_target_memcpy(*to, device_ptr, size, 0, 0, host, device);
}

void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size) {
  int device, device2;
  int no_devices = omp_get_num_devices();

  for(int i = 0; i < no_devices; i++) {
    if (omp_target_is_present(*from, i)) {
	device = i;
    }
    if (omp_target_is_present(*to, i)) {
	device2 = i;
    }
  }

  void* device_ptr = omp_get_mapped_ptr(*from, device);
  void* device_ptr2 = omp_get_mapped_ptr(*to, device);
  omp_target_memcpy(device_ptr2, device_ptr, size, 0, 0, device2, device);
}

void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size) {
  int device = omp_get_default_device();

  char* ptr2 = (char*) *ptr;
  #pragma omp target teams distribute parallel for //map(from:ptr2[0:size])
  for (int i = 0; i < size; i++) {
    ptr2[i] = (char)val;
  }
}

void ops_device_sync(OPS_instance *instance) {
}
