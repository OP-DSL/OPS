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
  * @brief OPS common OpenCL-specific functions (non-MPI and MPI)
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the OpenCL-specific routines shared between single-GPU 
  * and MPI+OpenCL backends
  */


// Turn off warnings about deprecated APIs
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

//#include <math_constants.h>

#include <ops_lib_core.h>
#include <ops_mpi_core.h>
#include <ops_opencl_rt_support.h>
#include <ops_exceptions.h>

#include <random>
std::default_random_engine ops_rand_gen;
//std::mt19937 ops_rand_gen;

//
// Get return (error) messages from OpenCL run-time
//
char *clGetErrorString(cl_int err) {
  switch (err) {
  case CL_SUCCESS:
    return (char *)"Success!";
  case CL_DEVICE_NOT_FOUND:
    return (char *)"Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:
    return (char *)"Device not available";
  case CL_COMPILER_NOT_AVAILABLE:
    return (char *)"Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return (char *)"Memory object allocation failure";
  case CL_OUT_OF_RESOURCES:
    return (char *)"Out of resources";
  case CL_OUT_OF_HOST_MEMORY:
    return (char *)"Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return (char *)"Profiling information not available";
  case CL_MEM_COPY_OVERLAP:
    return (char *)"Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH:
    return (char *)"Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return (char *)"Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE:
    return (char *)"Program build failure";
  case CL_MAP_FAILURE:
    return (char *)"Map failure";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return (char *)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return (char *)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_INVALID_VALUE:
    return (char *)"Invalid value";
  case CL_INVALID_DEVICE_TYPE:
    return (char *)"Invalid device type";
  case CL_INVALID_PLATFORM:
    return (char *)"Invalid platform";
  case CL_INVALID_DEVICE:
    return (char *)"Invalid device";
  case CL_INVALID_CONTEXT:
    return (char *)"Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES:
    return (char *)"Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE:
    return (char *)"Invalid command queue";
  case CL_INVALID_HOST_PTR:
    return (char *)"Invalid host pointer";
  case CL_INVALID_MEM_OBJECT:
    return (char *)"Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return (char *)"Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE:
    return (char *)"Invalid image size";
  case CL_INVALID_SAMPLER:
    return (char *)"Invalid sampler";
  case CL_INVALID_BINARY:
    return (char *)"Invalid binary";
  case CL_INVALID_BUILD_OPTIONS:
    return (char *)"Invalid build options";
  case CL_INVALID_PROGRAM:
    return (char *)"Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return (char *)"Invalid program executable";
  case CL_INVALID_KERNEL_NAME:
    return (char *)"Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION:
    return (char *)"Invalid kernel definition";
  case CL_INVALID_KERNEL:
    return (char *)"Invalid kernel";
  case CL_INVALID_ARG_INDEX:
    return (char *)"Invalid argument index";
  case CL_INVALID_ARG_VALUE:
    return (char *)"Invalid argument value";
  case CL_INVALID_ARG_SIZE:
    return (char *)"Invalid argument size";
  case CL_INVALID_KERNEL_ARGS:
    return (char *)"Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION:
    return (char *)"Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE:
    return (char *)"Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE:
    return (char *)"Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET:
    return (char *)"Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST:
    return (char *)"Invalid event wait list";
  case CL_INVALID_EVENT:
    return (char *)"Invalid event";
  case CL_INVALID_OPERATION:
    return (char *)"Invalid operation";
  case CL_INVALID_GL_OBJECT:
    return (char *)"Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE:
    return (char *)"Invalid buffer size";
  case CL_INVALID_MIP_LEVEL:
    return (char *)"Invalid mip-map level";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return (char *)"Invalid global work size";
  case CL_INVALID_PROPERTY:
    return (char *)"Invalid property";
  default:
    return (char *)"Unknown";
  }
}

//
// OpenCL utility functions
//

void __clSafeCall(cl_int ret, const char *file, const int line) {
  if (CL_SUCCESS != ret) {
    OPSException ex(OPS_OPENCL_ERROR);
    ex << "Error: " << file << "(" << line << "): clSafeCall() Runtime API error : " << clGetErrorString(ret);
    throw ex;
  }
}

void ops_init_device(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  if ((instance->OPS_block_size_x * instance->OPS_block_size_y * instance->OPS_block_size_z) > 1024) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_*");
  }
  for (int n = 1; n < argc; n++) {
    if (strncmp(argv[n], "OPS_CL_DEVICE=", 14) == 0) {
      instance->OPS_cl_device = atoi(argv[n] + 14);
      instance->ostream() << "\n OPS_cl_device = " << instance->OPS_cl_device << '\n';
    }
  }

  instance->opencl_instance = new OPS_instance_opencl();
  instance->opencl_instance->copy_tobuf_kernel = NULL;
  instance->opencl_instance->copy_frombuf_kernel = NULL;
  instance->opencl_instance->copy_opencl_kernel = NULL;
  instance->opencl_instance->isbuilt_copy_tobuf_kernel = false;
  instance->opencl_instance->isbuilt_copy_frombuf_kernel = false;
  instance->opencl_instance->isbuilt_copy_opencl_kernel = false;

  cutilDeviceInit(instance, argc, argv);
  instance->OPS_hybrid_gpu = 1;
}

void ops_device_malloc(OPS_instance *instance, void** ptr, size_t bytes) {
  cl_int ret = 0;
  *ptr = (char*)clCreateBuffer(instance->opencl_instance->OPS_opencl_core.context,
      CL_MEM_READ_WRITE, bytes, NULL, &ret);
  clSafeCall(ret);
  clSafeCall(clFinish(instance->opencl_instance->OPS_opencl_core.command_queue));
}

void ops_device_mallochost(OPS_instance *instance, void** ptr, size_t bytes) {
  *ptr = ops_malloc(bytes);
}

void ops_device_free(OPS_instance *instance, void** ptr) {
  clReleaseMemObject((cl_mem)(*ptr));
  *ptr = nullptr;
}

void ops_device_freehost(OPS_instance *instance, void** ptr) {
  ops_free(*ptr);
  *ptr = nullptr;
}

void ops_device_memcpy_h2d(OPS_instance *instance, void** to, void **from, size_t size) {
  clSafeCall(clEnqueueWriteBuffer(instance->opencl_instance->OPS_opencl_core.command_queue,
                                  (cl_mem)*to, CL_TRUE, 0, size, *from, 0,
                                  NULL, NULL));
  clSafeCall(clFinish(instance->opencl_instance->OPS_opencl_core.command_queue));
}

void ops_device_memcpy_d2h(OPS_instance *instance, void** to, void **from, size_t size) {
  clSafeCall(clEnqueueReadBuffer(instance->opencl_instance->OPS_opencl_core.command_queue,
                                 (cl_mem)*from, CL_TRUE, 0, size,
                                 *to, 0, NULL, NULL));
  clSafeCall(clFinish(instance->opencl_instance->OPS_opencl_core.command_queue));
}

void ops_device_memcpy_d2d(OPS_instance *instance, void** to, void **from, size_t size) {
  clSafeCall(clEnqueueCopyBuffer(instance->opencl_instance->OPS_opencl_core.command_queue,
                                 (cl_mem)*from, (cl_mem)*to, 0, 0, size,
                                  0, NULL, NULL));
  clSafeCall(clFinish(instance->opencl_instance->OPS_opencl_core.command_queue));
}

void ops_device_memset(OPS_instance *instance, void** ptr, int val, size_t size) {
    cl_int val2 = val;
    clSafeCall(clEnqueueFillBuffer(instance->opencl_instance->OPS_opencl_core.command_queue,
                                  (cl_mem)*ptr, (const void*)&val2, sizeof(int), 0, size, 0, NULL, NULL));
}

void ops_device_sync(OPS_instance *instance) {
  clSafeCall(clFinish(instance->opencl_instance->OPS_opencl_core.command_queue));
}

/**adapted from ocl_tools.c by Dan Curran (dancrn.com)*/
void cutilDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]) {
  (void)argc;
  (void)argv;

  char *dev_name;
  instance->opencl_instance->OPS_opencl_core.n_constants = 0;
  instance->opencl_instance->OPS_opencl_core.n_platforms = 0;
  instance->opencl_instance->OPS_opencl_core.n_devices = 0;
  instance->opencl_instance->OPS_opencl_core.platform_id = NULL;
  instance->opencl_instance->OPS_opencl_core.devices = NULL;
  cl_int ret = 0;
  cl_uint dev_type_flag = CL_DEVICE_TYPE_CPU;

  // determine the user requested device type
  int device_type = instance->OPS_cl_device;
  switch (device_type) {
  case 0: // CPU:
    dev_type_flag = CL_DEVICE_TYPE_CPU;
    break;
  case 1: // GPU:
    dev_type_flag = CL_DEVICE_TYPE_GPU;
    break;
  case 2: // Phi:
    dev_type_flag = CL_DEVICE_TYPE_ACCELERATOR;
    break;
  }

  // get number of platforms on current system (cast is intentional)
  clSafeCall(clGetPlatformIDs(0, NULL, &instance->opencl_instance->OPS_opencl_core.n_platforms));
  instance->ostream() << "Number of OpenCL platforms = " <<
         (int)instance->opencl_instance->OPS_opencl_core.n_platforms << '\n';

  // alloc space for platform ids
  instance->opencl_instance->OPS_opencl_core.platform_id = (cl_platform_id *)ops_calloc(
      instance->opencl_instance->OPS_opencl_core.n_platforms, sizeof(cl_platform_id));

  // read in platform ids from runtime
  clSafeCall(clGetPlatformIDs(instance->opencl_instance->OPS_opencl_core.n_platforms,
                              instance->opencl_instance->OPS_opencl_core.platform_id, NULL));

  for (unsigned int p = 0; p < instance->opencl_instance->OPS_opencl_core.n_platforms; p++) {
    // search for requested device : CPU, GPUs and ACCELERATORS (i.e Xeon Phi)
    // get number of devices on this platform
    ret = clGetDeviceIDs(instance->opencl_instance->OPS_opencl_core.platform_id[p], dev_type_flag, 0, NULL,
                         &instance->opencl_instance->OPS_opencl_core.n_devices);

    // this platform may not have the requested type of devices
    if (CL_DEVICE_NOT_FOUND == ret || 0 == instance->opencl_instance->OPS_opencl_core.n_devices)
      continue;
    else
      instance->ostream() << "Number of devices on platform "<<p<<" = " <<
             instance->opencl_instance->OPS_opencl_core.n_devices << '\n';

    // alloc space for device ids
    instance->opencl_instance->OPS_opencl_core.devices = (cl_device_id *)ops_calloc(
        instance->opencl_instance->OPS_opencl_core.n_devices, sizeof(cl_device_id));

    // get device IDs for this platform
    clSafeCall(clGetDeviceIDs(instance->opencl_instance->OPS_opencl_core.platform_id[p], dev_type_flag,
                              instance->opencl_instance->OPS_opencl_core.n_devices,
                              instance->opencl_instance->OPS_opencl_core.devices, NULL));

    for (unsigned int d = 0; d < instance->opencl_instance->OPS_opencl_core.n_devices; d++) {
      // attempt to create context from device id
      instance->opencl_instance->OPS_opencl_core.context = clCreateContext(
          NULL, 1, &instance->opencl_instance->OPS_opencl_core.devices[d], NULL, NULL, &ret);

      // check other devices if it failed
      if (CL_SUCCESS != ret)
        continue;

      // attempt to create a command queue
      instance->opencl_instance->OPS_opencl_core.command_queue = clCreateCommandQueue(
          instance->opencl_instance->OPS_opencl_core.context, instance->opencl_instance->OPS_opencl_core.devices[d], 0, &ret);
      if (CL_SUCCESS != ret) {
        // if we've failed, release the context we just acquired
        clReleaseContext(instance->opencl_instance->OPS_opencl_core.context);
        instance->opencl_instance->OPS_opencl_core.context = NULL;
        continue;
      }

      // this is definitely the device id we'll be using
      instance->opencl_instance->OPS_opencl_core.device_id = instance->opencl_instance->OPS_opencl_core.devices[d];

      // ops_free the rest of them.
      ops_free(instance->opencl_instance->OPS_opencl_core.devices);

      size_t dev_name_len = 0;
      ret = clGetDeviceInfo(instance->opencl_instance->OPS_opencl_core.device_id, CL_DEVICE_NAME, 0, NULL,
                            &dev_name_len);

      // it's unlikely this will happen
      if (ret != CL_SUCCESS) {
        // cleanup after ourselves
        clReleaseCommandQueue(instance->opencl_instance->OPS_opencl_core.command_queue);
        clReleaseContext(instance->opencl_instance->OPS_opencl_core.context);

        instance->opencl_instance->OPS_opencl_core.context = NULL;
        instance->opencl_instance->OPS_opencl_core.command_queue = NULL;

        instance->ostream() << "Error: Unable to get device name length.\n";
        clSafeCall(ret);
        return;
      }

      // alloc space for device name and '\0'
      dev_name = (char *)ops_calloc(dev_name_len + 1, sizeof(char));
      // attempt to get device name
      ret = clGetDeviceInfo(instance->opencl_instance->OPS_opencl_core.device_id, CL_DEVICE_NAME,
                            dev_name_len, dev_name, NULL);
      if (CL_SUCCESS != ret) {
        // cleanup after ourselves
        clReleaseCommandQueue(instance->opencl_instance->OPS_opencl_core.command_queue);
        clReleaseContext(instance->opencl_instance->OPS_opencl_core.context);

        instance->opencl_instance->OPS_opencl_core.context = NULL;
        instance->opencl_instance->OPS_opencl_core.command_queue = NULL;

        instance->ostream() << "Error: Unable to get device name.\n";
        clSafeCall(ret);
        return;
      }

      // at this point, we've got a device, it's name, a context and a queue
      instance->ostream() << "OpenCL Running on platform "<<p<<" device "<<d<<" : "<<dev_name<<"\n";
      return;
    }

    instance->opencl_instance->OPS_opencl_core.devices = NULL;
    instance->ostream() << "\n";
  }

  instance->ostream() << "Error: No available devices found.\n";
  return;
}

void ops_randomgen_init(unsigned int seed, int options) {
  /* Set seed */
  int comm_global_size = ops_num_procs();
  int my_global_rank = ops_get_proc();

  if(comm_global_size == 0)
    ops_rand_gen.seed(seed);
  else
    ops_rand_gen.seed(seed*my_global_rank+my_global_rank);
}

void ops_fill_random_uniform(ops_dat dat) {
  size_t cumsize = dat->dim;
  const char *type = dat->type;

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    cumsize *= dat->size[d];
  }

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 || strcmp(type, "real(kind=8)") == 0) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i =0 ; i < cumsize; i++) {
      ((double *)dat->data)[i] = distribution(ops_rand_gen);
    }
  }
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0) {
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i =0 ; i < cumsize; i++) {
      ((float *)dat->data)[i] = distribution(ops_rand_gen);
    }
  }
  else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0 || strcmp(type, "integer(kind=4)") == 0) {
    std::uniform_int_distribution<int> distribution(0, INT_MAX);
    for (int i =0 ; i < cumsize; i++) {
      ((int *)dat->data)[i] = distribution(ops_rand_gen);
    }
  }
  else {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: uniform random number generation not implemented for data type: "<<dat->type;
    throw ex;
  }

  dat->dirty_hd = 1;
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
  size_t cumsize = dat->dim;
  const char *type = dat->type;

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    cumsize *= dat->size[d];
  }

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 || strcmp(type, "real(kind=8)") == 0) {
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i =0 ; i < cumsize; i++) {
      ((double *)dat->data)[i] = distribution(ops_rand_gen);
    }
  }
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0) {
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (int i =0 ; i < cumsize; i++) {
      ((float *)dat->data)[i] = distribution(ops_rand_gen);
    }
  }
  else {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: normal random number generation not implemented for data type: "<<dat->type;
    throw ex;
  }

  dat->dirty_hd = 1;
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
}
