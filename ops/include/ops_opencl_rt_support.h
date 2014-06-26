#ifndef __OPS_OPENCL_RT_SUPPORT_H
#define __OPS_OPENCL_RT_SUPPORT_H
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
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
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

/** @brief ops cuda specific runtime support functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements cuda backend runtime support functions
  */

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//
// OpenCL specific data structure
//

typedef struct{
  cl_platform_id *platform_id;
  cl_device_id device_id;
  cl_device_id *subdev_id;
  cl_uint n_devices;
  cl_uint n_platforms;
  cl_command_queue command_queue;
  cl_kernel *kernel;
  cl_program program;
  cl_context context;
  cl_uint n_kernels;
  cl_mem *constant;
  cl_uint n_constants;
  //cl_mem *data_d; // cl_mem struct corresponding to ops_core_dat char* data_d
} ops_opencl_core;

extern int OPS_cl_device;


#include <ops_lib_cpp.h>
#include <ops_lib_core.h>


/* define CUDA warpsize for OPS */

#define OPS_WARPSIZE 32

#ifdef __cplusplus
extern "C" {
#endif

/*
* Global variables actually defined in the corresponding c file
*/
extern char * OPS_consts_h,
            * OPS_consts_d,
            * OPS_reduct_h,
            * OPS_reduct_d;

extern int OPS_block_size_x;
extern int OPS_block_size_y;



#define clSafeCall(ret) __clSafeCall(ret, __FILE__,__LINE__)


void openclDeviceInit( int argc, char ** argv);
void __clSafeCall( cl_int ret, const char * file, const int line );
void ops_cpHostToDevice(void ** data_d, void ** data_h, int size );
void ops_opencl_get_data( ops_dat dat );
void reallocConstArrays( int consts_bytes );
void reallocReductArrays( int reduct_bytes );
void mvConstArraysToDevice( int consts_bytes );
void mvReductArraysToDevice( int reduct_bytes );
void mvReductArraysToHost( int reduct_bytes );
void ops_opencl_exit( );

#ifdef __cplusplus
}
#endif

#endif /* __OPS_OPENCL_RT_SUPPORT_H */
