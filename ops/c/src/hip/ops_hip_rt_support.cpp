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
  * @brief OPS hip specific runtime support functions
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

#include <hip/hip_runtime.h>

#include <ops_hip_rt_support.h>
#include <ops_lib_core.h>
#include <ops_exceptions.h>

//
// HIP utility functions
//

void __hipSafeCall(std::ostream &stream, hipError_t err, const char *file, const int line) {//??hiperror->megtaláltam neten
  if (hipSuccess != err) {//hipsuccess->elvileg megtaláltam neten
    fprintf2(stream, "%s(%i) : hipSafeCall() Runtime API error : %s.\n", file,
            line, hipGetErrorString(err));//cudaGEt.-> megtaláltam neten
    throw OPSException(OPS_RUNTIME_ERROR, hipGetErrorString(err));
  }
}

void hipDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;
  int deviceCount;
  hipSafeCall(instance->ostream(), hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available HIP devices");
  }

  // Test we have access to a device

  float *test;
  int my_id = ops_get_proc();
  instance->OPS_hybrid_gpu = 0;
  for (int i = 0; i < deviceCount; i++) {
    hipError_t err = hipSetDevice((i+my_id)%deviceCount);//hipsetdevice is elvileg van
    if (err == hipSuccess) {
      hipError_t err = hipMalloc((void **)&test, sizeof(float));
      if (err == hipSuccess) {
        instance->OPS_hybrid_gpu = 1;
        break;
      }
    }
  }
  if (instance->OPS_hybrid_gpu) {
    hipFree(test);//Ilyen is van 

    int deviceId = -1;
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProp;
    hipSafeCall(instance->ostream(), hipGetDeviceProperties(&deviceProp, deviceId));
    if (instance->OPS_diags>2) instance->ostream() << "\n Using HIP device: " <<
      deviceId << " " << deviceProp.name;
  } else {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: no available HIP devices");
  }
}

void ops_cpHostToDevice(OPS_instance *instance, void **data_d, void **data_h, size_t size) {

  if ( *data_d == NULL )
      hipSafeCall(instance->ostream(), hipMalloc(data_d, size));

  if (data_h == NULL || *data_h == NULL) {
    hipSafeCall(instance->ostream(), hipMalloc(data_d, size));
    hipSafeCall(instance->ostream(), hipMemset(*data_d, 0, size));//ilyen is van 
    return;
  }

  /*static void* stage = NULL;
  static size_t stage_size = 0;

  void *src = NULL;
  if ( size < 4*1024*1024 ) {
      if ( (size_t)size > stage_size ) {
          if ( stage ) cudaFreeHost(stage);
          stage_size = size;
          hipSafeCall(instance->ostream(), cudaMallocHost(&stage, stage_size));
      }

      memcpy(stage, *data_h, size);
      src = stage;
  } else {
      src = *data_h;
  }*/

  hipSafeCall(instance->ostream(), hipMemcpy(*data_d, *data_h, size, hipMemcpyHostToDevice));
}

void ops_download_dat(ops_dat dat) {

  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  // printf("downloading to host from device %d bytes\n",bytes);
  hipSafeCall(dat->block->instance->ostream(), 
      hipMemcpy(dat->data, dat->data_d, bytes,hipMemcpyDeviceToHost));
}

void ops_upload_dat(ops_dat dat) {

  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  // printf("uploading to device from host %d bytes\n",bytes);
  hipSafeCall(dat->block->instance->ostream(), 
      hipMemcpy(dat->data_d, dat->data, bytes, hipMemcpyHostToDevice));
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

void ops_hip_get_data(ops_dat dat) {
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  hipSafeCall(dat->block->instance->ostream(), 
      hipMemcpy(dat->data, dat->data_d, bytes, hipMemcpyDeviceToHost));
  hipSafeCall(dat->block->instance->ostream(), hipDeviceSynchronize());
}

//
// routine to upload data from CPU to GPU (with transposing SoA to AoS if needed)
//

void ops_hip_put_data(ops_dat dat) {
  if (dat->dirty_hd == 1)
    dat->dirty_hd = 0;
  else
    return;
  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  hipSafeCall(dat->block->instance->ostream(), 
      hipMemcpy(dat->data_d, dat->data, bytes, hipMemcpyHostToDevice));
  hipSafeCall(dat->block->instance->ostream(), hipDeviceSynchronize());
}


//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(OPS_instance *instance, int consts_bytes) {
  if (consts_bytes > instance->OPS_consts_bytes) {
    if (instance->OPS_consts_bytes > 0) {
      free(instance->OPS_consts_h);
      hipFreeHost(instance->OPS_gbl_prev);
      hipSafeCall(instance->ostream(), hipFree(instance->OPS_consts_d));
    }
    instance->OPS_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
    hipMallocHost((void **)&instance->OPS_gbl_prev, instance->OPS_consts_bytes);
    instance->OPS_consts_h = (char *)ops_malloc(instance->OPS_consts_bytes);
    hipSafeCall(instance->ostream(), hipMalloc((void **)&instance->OPS_consts_d, instance->OPS_consts_bytes));
  }
}

void reallocReductArrays(OPS_instance *instance, int reduct_bytes) {
  if (reduct_bytes > instance->OPS_reduct_bytes) {
    if (instance->OPS_reduct_bytes > 0) {
      free(instance->OPS_reduct_h);
      hipSafeCall(instance->ostream(), hipFree(instance->OPS_reduct_d));
    }
    instance->OPS_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
    instance->OPS_reduct_h = (char *)ops_malloc(instance->OPS_reduct_bytes);
    hipSafeCall(instance->ostream(), hipMalloc((void **)&instance->OPS_reduct_d, instance->OPS_reduct_bytes));
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
    // hipSafeCall ( hipMemcpyAsync ( instance->OPS_consts_d, instance->OPS_gbl_prev,
    // consts_bytes,
    //                            hipMemcpyHostToDevice ) );
    hipSafeCall(instance->ostream(), hipMemcpy(instance->OPS_consts_d, instance->OPS_consts_h, consts_bytes,
                             hipMemcpyHostToDevice));
    memcpy(instance->OPS_gbl_prev, instance->OPS_consts_h, consts_bytes);//Ez kell, hogy hip legyen?
  }
}

void mvReductArraysToDevice(OPS_instance *instance, int reduct_bytes) {
  hipSafeCall(instance->ostream(), hipMemcpy(instance->OPS_reduct_d, instance->OPS_reduct_h, reduct_bytes,
                           hipMemcpyHostToDevice));
}

void mvReductArraysToHost(OPS_instance *instance, int reduct_bytes) {
  hipSafeCall(instance->ostream(), hipMemcpy(instance->OPS_reduct_h, instance->OPS_reduct_d, reduct_bytes,
                           hipMemcpyDeviceToHost));
}

void ops_hip_exit(OPS_instance *instance) {
  if (!instance->OPS_hybrid_gpu)
    return;
  if (instance->OPS_consts_bytes > 0) {
    free(instance->OPS_consts_h);
    hipFreeHost(instance->OPS_gbl_prev);
    hipSafeCall(instance->ostream(), hipFree(instance->OPS_consts_d));
  }
  if (instance->OPS_reduct_bytes > 0) {
    free(instance->OPS_reduct_h);
    hipSafeCall(instance->ostream(), hipFree(instance->OPS_reduct_d));
  }
}

void ops_free_dat(ops_dat dat) {
  delete dat;
}

// _ops_free_dat is called directly from ~ops_dat_core
void _ops_free_dat(ops_dat dat) {
  hipSafeCall(dat->block->instance->ostream(), hipFree(dat->data_d));
  ops_free_dat_core(dat);
}
