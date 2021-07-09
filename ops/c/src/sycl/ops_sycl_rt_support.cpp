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

#include <ops_sycl_rt_support.h>
#include <ops_lib_core.h>
#include <ops_exceptions.h>


//
// SYCL utility functions
//

void syclDeviceInit(OPS_instance *instance, const int argc,
                    const char *const argv[]) {
  int OPS_sycl_device = 3;
  for (int i = 0; i < argc; ++i) {
    if (strncmp(argv[i], "OPS_SYCL_DEVICE=", 16) == 0) {
      if (strcmp(argv[i] + strlen("OPS_SYCL_DEVICE="), "host") == 0)
        OPS_sycl_device = 0;
      else if (strcmp(argv[i] + strlen("OPS_SYCL_DEVICE="), "cpu") == 0)
        OPS_sycl_device = 1;
      else if (strcmp(argv[i] + strlen("OPS_SYCL_DEVICE="), "gpu") == 0)
        OPS_sycl_device = 2;
      break;
    }
  }
  instance->sycl_instance = new OPS_instance_sycl();
  switch (OPS_sycl_device) {
  case 0:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::host_selector());
    break;
  case 1:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::cpu_selector());
    break;
  case 2:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::gpu_selector());
    break;
  case 3:
    instance->sycl_instance->queue =
        new cl::sycl::queue(cl::sycl::default_selector());
    break;
  default: ops_printf("Error, unrecognised SYCL device selection\n"); exit(-1);
  }

  instance->OPS_hybrid_gpu = 1;
  instance->ostream() << "Running on " << instance->sycl_instance->queue->get_device().get_info<cl::sycl::info::device::name>() << "\n";
}

void *ops_sycl_register_const(void *old_p, void *new_p) {
  if (old_p == NULL) OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts.push_back(new_p);
  else {
    for (size_t i = 0; i < OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts.size(); i++)
      if (OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts[i]==old_p) {
        delete reinterpret_cast<cl::sycl::buffer<char,1>*>(OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts[i]);
        OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts[i]=new_p;
      }
  }
  return new_p;
}

void ops_sycl_memcpyHostToDevice(OPS_instance *instance,
                                 cl::sycl::buffer<char, 1> *data_d,
                                 char *data_h, size_t bytes) {
  // create sub buffer
  cl::sycl::buffer<char, 1> buffer(*data_d, cl::sycl::id<1>(0), cl::sycl::range<1>(bytes));
#ifdef SYCL_COPY
  instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
    auto acc = buffer.template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.copy(data_h, acc);
  });
  instance->sycl_instance->queue->wait();
#else
  auto HostAccessor = buffer.get_host_access(cl::sycl::write_only);
  for (size_t i = 0; i < bytes; i++)
    HostAccessor[i] = data_h[i];
#endif
}

void ops_sycl_memcpyDeviceToHost(OPS_instance *instance,
                                 cl::sycl::buffer<char, 1> *data_d,
                                 char *data_h, size_t bytes) {
  // create sub buffer
  cl::sycl::buffer<char, 1> buffer(*data_d, cl::sycl::id<1>(0), cl::sycl::range<1>(bytes));
#ifdef SYCL_COPY
  instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
    auto acc = buffer.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.copy(acc, data_h);
  });
  instance->sycl_instance->queue->wait();
#else
  auto HostAccessor = buffer.get_host_access(cl::sycl::read_only);
  for (size_t i = 0; i < bytes; i++)
    data_h[i] = HostAccessor[i];
#endif
}

void ops_cpHostToDevice(OPS_instance *instance, void **data_d, void **data_h,
                        size_t size) {
  if (!instance->OPS_hybrid_gpu) return;
  if (*data_d != nullptr) {
    delete reinterpret_cast<cl::sycl::buffer<char, 1> *>(*data_d);
  }
  auto *buffer = new cl::sycl::buffer<char, 1>(cl::sycl::range<1>(size));
  *data_d = (void *)buffer;
  char *data = (char *)(*data_h);
  ops_sycl_memcpyHostToDevice(instance, buffer, data, size);
}

void ops_download_dat(ops_dat dat) {
  ops_sycl_memcpyDeviceToHost(
      dat->block->instance,
      reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d), dat->data,
      dat->mem);
}

void ops_upload_dat(ops_dat dat) {
  ops_sycl_memcpyHostToDevice(
      dat->block->instance,
      reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d), dat->data,
      dat->mem);
}

void ops_H_D_exchanges_host(ops_arg *args, int nargs) {
  // printf("in ops_H_D_exchanges\n");
  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OPS_ARG_DAT &&
        args[n].dat->locked_hd > 0) {
      OPSException ex(OPS_RUNTIME_ERROR, "ERROR: ops_par_loops involving datasets for which raw pointers have not been released are not allowed");
      throw ex;
    }
    if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 2) {
      ops_download_dat(args[n].dat);
      // printf("halo exchanges on host\n");
      args[n].dat->dirty_hd = 0;
    }
  }
}

void ops_H_D_exchanges_device(ops_arg *args, int nargs) {
  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OPS_ARG_DAT &&
        args[n].dat->locked_hd > 0) {
      OPSException ex(OPS_RUNTIME_ERROR, "ERROR: ops_par_loops involving datasets for which raw pointers have not been released are not allowed");
      throw ex;
    }
    if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 1) {
      ops_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
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

void ops_sycl_get_data(ops_dat dat) {
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  ops_download_dat(dat);
  dat->block->instance->sycl_instance->queue->wait();
}

//
// routine to upload data from CPU to GPU (with transposing SoA to AoS if needed)
//

void ops_sycl_put_data(ops_dat dat) {
  if (dat->dirty_hd == 1)
    dat->dirty_hd = 0;
  else
    return;
  ops_upload_dat(dat);
  dat->block->instance->sycl_instance->queue->wait();
}


//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(OPS_instance *instance, int consts_bytes) {
  if (consts_bytes > instance->OPS_consts_bytes) {
    if (instance->OPS_consts_bytes > 0) {
      ops_free(instance->OPS_consts_h);
      ops_free(instance->OPS_gbl_prev);
      delete reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)instance->OPS_consts_d);
    }
    instance->OPS_consts_bytes =
        4 * consts_bytes; // 4 is arbitrary, more than needed
    instance->OPS_gbl_prev = (char *)ops_malloc(instance->OPS_consts_bytes);
    instance->OPS_consts_h = (char *)ops_malloc(instance->OPS_consts_bytes);
    instance->OPS_consts_d = (char *)new cl::sycl::buffer<char, 1>(
        cl::sycl::range<1>(instance->OPS_consts_bytes));
  }
}

void reallocReductArrays(OPS_instance *instance, int reduct_bytes) {
  if (reduct_bytes > instance->OPS_reduct_bytes) {
    if (instance->OPS_reduct_bytes > 0) {
      ops_free(instance->OPS_reduct_h);
      delete reinterpret_cast<cl::sycl::buffer<char, 1> *>(
          (void *)instance->OPS_reduct_d);
    }
    instance->OPS_reduct_bytes =
        4 * reduct_bytes; // 4 is arbitrary, more than needed
    instance->OPS_reduct_h = (char *)ops_malloc(instance->OPS_reduct_bytes);
    instance->OPS_reduct_d =
        reinterpret_cast<char *>(new cl::sycl::buffer<char, 1>(
            cl::sycl::range<1>(instance->OPS_reduct_bytes)));
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
    ops_sycl_memcpyHostToDevice(
        instance,
        reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)instance->OPS_consts_d),
        instance->OPS_consts_h, consts_bytes);
    memcpy(instance->OPS_gbl_prev, instance->OPS_consts_h, consts_bytes);
  }
}

void mvReductArraysToDevice(OPS_instance *instance, int reduct_bytes) {
  ops_sycl_memcpyHostToDevice(
      instance,
      reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)instance->OPS_reduct_d),
      instance->OPS_reduct_h, reduct_bytes);
}

void mvReductArraysToHost(OPS_instance *instance, int reduct_bytes) {
  ops_sycl_memcpyDeviceToHost(
      instance,
      reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)instance->OPS_reduct_d),
      instance->OPS_reduct_h, reduct_bytes);
}

void ops_sycl_exit(OPS_instance *instance) {
  if (!instance->OPS_hybrid_gpu)
    return;
  if (instance->OPS_consts_bytes > 0) {
    ops_free(instance->OPS_consts_h);
    ops_free(instance->OPS_gbl_prev);
    delete reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)instance->OPS_consts_d);
  }
  if (instance->OPS_reduct_bytes > 0) {
    ops_free(instance->OPS_reduct_h);
    delete reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)instance->OPS_reduct_d);
  }
  for (size_t i = 0; i < OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts.size(); i++)
    delete reinterpret_cast<cl::sycl::buffer<char,1>*>(OPS_instance::getOPSInstance()->sycl_instance->ops_sycl_consts[i]);
}

void ops_free_dat(ops_dat dat) {
  delete dat;
}

// _ops_free_dat is called directly from ~ops_dat_core
void _ops_free_dat(ops_dat dat) {
  delete reinterpret_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d);
  dat->data_d = nullptr;
  ops_free_dat_core(dat);
}
