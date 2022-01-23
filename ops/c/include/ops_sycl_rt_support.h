#ifndef __OPS_SYCL_RT_SUPPORT_H
#define __OPS_SYCL_RT_SUPPORT_H
#ifndef DOXYGEN_SHOULD_SKIP_THIS
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

/** @file
  * @brief OPS sycl specific runtime support functions
  * @author Gabor Daniel Balogh
  * @details Implements sycl backend runtime support functions
  */

#include <CL/sycl.hpp>

#include <ops_lib_core.h>

class OPS_instance_sycl  {
public:
  std::vector<void*> ops_sycl_consts;
  cl::sycl::queue *queue;
};


void syclDeviceInit(OPS_instance *instance, const int argc, const char * const argv[]);
void ops_sycl_get_data(ops_dat dat);
void ops_sycl_put_data(ops_dat dat);
void reallocConstArrays(OPS_instance *instance,int consts_bytes);
void reallocReductArrays(OPS_instance *instance,int reduct_bytes);
void mvConstArraysToDevice(OPS_instance *instance,int consts_bytes);
void mvReductArraysToDevice(OPS_instance *instance,int reduct_bytes);
void mvReductArraysToHost(OPS_instance *instance,int reduct_bytes);
void ops_sycl_exit(OPS_instance *instance);
void ops_upload_dat(ops_dat dat);
void ops_download_dat(ops_dat dat);
void ops_internal_copy_sycl(ops_kernel_descriptor *desc);
void *ops_sycl_register_const(void *old_p, void *new_p);
void ops_sycl_memcpyHostToDevice(OPS_instance *instance, cl::sycl::buffer<char, 1> *data_d, char *data_h, size_t bytes);
void ops_sycl_memcpyDeviceToHost(OPS_instance *instance, cl::sycl::buffer<char, 1> *data_d, char *data_h, size_t bytes);
void ops_sycl_memcpyDeviceToDevice(OPS_instance *instance,
 cl::sycl::buffer<char, 1> *data_d_src, cl::sycl::buffer<char, 1> *data_d_dest, size_t bytes);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* __OPS_SYCL_RT_SUPPORT_H */
