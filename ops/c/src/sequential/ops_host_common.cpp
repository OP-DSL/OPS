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

#include <ops_device_rt_support.h>

void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  instance->OPS_hybrid_gpu = 0;
}

void ops_exit_device(OPS_instance *instance) {
  (void)instance;
}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_free(OPS_instance *instance, void** ptr) {
  *ptr = nullptr;
  throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_freehost(OPS_instance *instance, void** ptr) {
  *ptr = nullptr;
  throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

void ops_device_sync(OPS_instance *instance) {
  (void)instance;
}

void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;
  (void)instance;
}

void ops_internal_copy_device(ops_kernel_descriptor *desc) {
  throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: should not have ended up here for host backend");
}

__attribute__((weak)) void ops_decl_const_char(OPS_instance *instance, int dim, char const *type, int size, char *dat, char const *name){
  (void)instance;
  (void)dim;
  (void)type;
  (void)size;
  (void)dat;
  (void)name;
}
