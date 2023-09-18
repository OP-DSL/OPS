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
  * @brief OPS common HIP-specific functions (non-MPI and MPI)
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the HIP-specific routines shared between single-GPU
  * and MPI+HIP backends
  */

#include <ops_device_rt_support.h>
#include <ops_sycl_rt_support.h>


void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  cutilDeviceInit(instance, argc, argv);
  instance->OPS_hybrid_gpu = 1;
}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  *ptr = cl::sycl::malloc_device(bytes, *instance->sycl_instance->queue);
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  *ptr = cl::sycl::malloc_host(bytes, *instance->sycl_instance->queue);
}

void ops_device_free(OPS_instance *instance, void** ptr) {
  cl::sycl::free(*ptr, *instance->sycl_instance->queue);
  *ptr = nullptr;
}

void ops_device_freehost(OPS_instance *instance, void** ptr) {
  cl::sycl::free(*ptr, *instance->sycl_instance->queue);
  *ptr = nullptr;
}

void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size) {
    //in-order queue, no wait needed here
    instance->sycl_instance->queue->memcpy(*to, *from, size);
    instance->sycl_instance->queue->wait();
}

void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size) {
    instance->sycl_instance->queue->memcpy(*to, *from, size);
    instance->sycl_instance->queue->wait();
}

void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size) {
    instance->sycl_instance->queue->memcpy(*to, *from, size);
    instance->sycl_instance->queue->wait();
}

void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size) {
    instance->sycl_instance->queue->memset(*ptr, val, size);
    instance->sycl_instance->queue->wait();
}

void ops_device_sync(OPS_instance *instance) {
  instance->sycl_instance->queue->wait();
}

void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  char temp[64];
  const char *pch;

  int OPS_sycl_device = 3;
  for (int i = 0; i < argc; ++i) {
    pch = strstr(argv[i], "OPS_SYCL_DEVICE=");
    if (pch != NULL) {
      snprintf(temp, 64, "%s", pch);
      if (strcmp(temp + strlen("OPS_SYCL_DEVICE="), "host") == 0)
        OPS_sycl_device = 0;
      else if (strcmp(temp + strlen("OPS_SYCL_DEVICE="), "cpu") == 0)
        OPS_sycl_device = 1;
      else if (strcmp(temp + strlen("OPS_SYCL_DEVICE="), "gpu") == 0)
        OPS_sycl_device = 2;
      else {
        int val = atoi(temp + strlen("OPS_SYCL_DEVICE="));
        OPS_sycl_device=4+val;
      }
    }
  }
  instance->sycl_instance = new OPS_instance_sycl();
  switch (OPS_sycl_device) {
  case 0:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::host_selector(), cl::sycl::property::queue::in_order());
    break;
  case 1:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::cpu_selector(), cl::sycl::property::queue::in_order());
    break;
  case 2:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::gpu_selector(), cl::sycl::property::queue::in_order());
    break;
  case 3:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::default_selector(), cl::sycl::property::queue::in_order());
    break;
  default:
    std::vector<cl::sycl::device> devices;
    devices = cl::sycl::device::get_devices();
    int devid = OPS_sycl_device - 4;
    if (devid < 0 || devid >= devices.size()) {
      ops_printf("Error, unrecognised SYCL device selection. Available devices (%d)\n",devices.size());
      for (int i = 0; i < devices.size(); i++)
      {
        auto platform = devices[i].get_platform();
        ops_printf("%d: [%s] %s\n", i, platform.get_info<cl::sycl::info::platform::name>().c_str(), devices[i].get_info<cl::sycl::info::device::name>().c_str());
      }
      exit(-1);
    }
    instance->sycl_instance->queue =
        new cl::sycl::queue(devices[devid], cl::sycl::property::queue::in_order());
  }

  instance->OPS_hybrid_gpu = 1;
  auto platform = instance->sycl_instance->queue->get_device().get_platform();
  if (instance->OPS_diags>=1) instance->ostream() << "Running on " << instance->sycl_instance->queue->get_device().get_info<cl::sycl::info::device::name>() << " platform " << platform.get_info<cl::sycl::info::platform::name>() << "\n";
}
