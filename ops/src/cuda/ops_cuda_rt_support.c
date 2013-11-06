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

/** @brief ops cuda specific runtime support functions
  * @author Gihan Mudalige
  * @details Implements cuda backend runtime support functions
  */


//
// header files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <ops_lib_core.h>
#include <ops_cuda_rt_support.h>


//
// CUDA utility functions
//

void __cudaSafeCall ( cudaError_t err, const char * file, const int line )
{
  if ( cudaSuccess != err ) {
    fprintf ( stderr, "%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
              file, line, cudaGetErrorString ( err ) );
    exit ( -1 );
  }
}

void cutilDeviceInit( int argc, char ** argv )
{
  (void)argc;
  (void)argv;
  int deviceCount;
  cutilSafeCall( cudaGetDeviceCount ( &deviceCount ) );
  if ( deviceCount == 0 ) {
    printf ( "cutil error: no devices supporting CUDA\n" );
    exit ( -1 );
  }

  // Test we have access to a device
  float *test;
  cudaError_t err = cudaMalloc((void **)&test, sizeof(float));
  if (err != cudaSuccess) {
    OPS_hybrid_gpu = 0;
  } else {
    OPS_hybrid_gpu = 1;
  }
  if (OPS_hybrid_gpu) {
    cudaFree(test);

    cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int deviceId = -1;
    cudaGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    cutilSafeCall ( cudaGetDeviceProperties ( &deviceProp, deviceId ) );
    printf ( "\n Using CUDA device: %d %s\n",deviceId, deviceProp.name );
  } else {
    printf ( "\n Using CPU\n" );
  }
}

void ops_cpHostToDevice ( void ** data_d, void ** data_h, int size )
{
  //if (!OPS_hybrid_gpu) return;
  cutilSafeCall ( cudaMalloc ( data_d, size ) );
  cutilSafeCall ( cudaMemcpy ( *data_d, *data_h, size,
                               cudaMemcpyHostToDevice ) );
  cutilSafeCall ( cudaDeviceSynchronize ( ) );
}



void op_upload_dat(op_dat dat) {
  if (!OP_hybrid_gpu) return;
  int set_size = dat->set->size;
  if (strstr( dat->type, ":soa")!= NULL) {
    char *temp_data = (char *)op_malloc(dat->size*set_size*sizeof(char));
    int element_size = dat->size/dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size*i*set_size + element_size*j + c] = dat->data[dat->size*j+element_size*i+c];
        }
      }
    }
    cutilSafeCall( cudaMemcpy(dat->data_d, temp_data, set_size*dat->size, cudaMemcpyHostToDevice));
    op_free(temp_data);
  } else {
    cutilSafeCall( cudaMemcpy(dat->data_d, dat->data, set_size*dat->size, cudaMemcpyHostToDevice));
  }
}

void ops_download_dat(op_dat dat) {

  //if (!OP_hybrid_gpu) return;
  int bytes = dat->size;
  for (int i=0; i<dat->block->dims; i++) bytes = bytes*dat->block->block_size[i];
  cutilSafeCall( cudaMemcpy(dat->data, dat->data_d, bytes, cudaMemcpyDeviceToHost));

}

void ops_upload_dat(op_dat dat) {

  //if (!OP_hybrid_gpu) return;
  int bytes = dat->size;
  for (int i=0; i<dat->block->dims; i++) bytes = bytes*dat->block->block_size[i];
  cutilSafeCall( cudaMemcpy(dat->data_d, dat->data , bytes, cudaMemcpyHostToDevice));

}

void ops_halo_exchanges(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++)
    if(args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 2) {
      ops_download_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
}

void ops_halo_exchanges_cuda(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++)
    if(args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 1) {
      ops_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
}

void ops_set_dirtybit(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++) {
    if((args[n].argtype == OPS_ARG_DAT) &&
       (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE || args[n].acc == OPS_RW)) {
      args[n].dat->dirty_hd = 1;
    }
  }
}

void ops_set_dirtybit_cuda(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++) {
    if((args[n].argtype == OPS_ARG_DAT) &&
       (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE || args[n].acc == OPS_RW)) {
      args[n].dat->dirty_hd = 2;
    }
  }
}
