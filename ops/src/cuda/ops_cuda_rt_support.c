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


// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct cudaDeviceProp cudaDeviceProp_t;

int OPS_consts_bytes = 0,
    OPS_reduct_bytes = 0;

char * OPS_consts_h,
     * OPS_consts_d,
     * OPS_reduct_h,
     * OPS_reduct_d;

int OPS_gbl_changed = 1;
char *OPS_gbl_prev = NULL;

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
    //printf ( "\n Using CPU\n" );
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


void ops_download_dat(ops_dat dat) {

  //if (!OPS_hybrid_gpu) return;
  int bytes = dat->elem_size;
  for (int i=0; i<dat->block->dims; i++) bytes = bytes * dat->size[i];
  //printf("downloading to host from device %d bytes\n",bytes);
  cutilSafeCall( cudaMemcpy(dat->data, dat->data_d, bytes, cudaMemcpyDeviceToHost));

}

void ops_upload_dat(ops_dat dat) {

  //if (!OPS_hybrid_gpu) return;
  int bytes = dat->elem_size;
  for (int i=0; i<dat->block->dims; i++) bytes = bytes * dat->size[i];
  //printf("uploading to device from host %d bytes\n",bytes);
  cutilSafeCall( cudaMemcpy(dat->data_d, dat->data , bytes, cudaMemcpyHostToDevice));

}

void ops_H_D_exchanges_host(ops_arg *args, int nargs)
{
  //printf("in ops_H_D_exchanges\n");
  for (int n=0; n<nargs; n++)
    if(args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 2) {
      ops_download_dat(args[n].dat);
      //printf("halo exchanges on host\n");
      args[n].dat->dirty_hd = 0;
    }
}

void ops_H_D_exchanges_device(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++)
    if(args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 1) {
      ops_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
}

void ops_set_dirtybit_device(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++) {
    if((args[n].argtype == OPS_ARG_DAT) &&
       (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE || args[n].acc == OPS_RW) ) {
      args[n].dat->dirty_hd = 2;
    }
  }
}

//
// routine to fetch data from GPU to CPU (with transposing SoA to AoS if needed)
//

void ops_cuda_get_data( ops_dat dat )
{
  //if (!OPS_hybrid_gpu) return;
  if (dat->dirty_hd == 2) dat->dirty_hd = 0;
  else return;
  int bytes = dat->elem_size;
  for (int i=0; i<dat->block->dims; i++) bytes = bytes * dat->size[i];
  cutilSafeCall ( cudaMemcpy ( dat->data, dat->data_d,
                               bytes,
                               cudaMemcpyDeviceToHost ) );
  cutilSafeCall ( cudaDeviceSynchronize ( ) );

}


//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays ( int consts_bytes )
{
  if ( consts_bytes > OPS_consts_bytes ) {
    if ( OPS_consts_bytes > 0 ) {
      free ( OPS_consts_h );
      cudaFreeHost ( OPS_gbl_prev );
      cutilSafeCall ( cudaFree ( OPS_consts_d ) );
    }
    OPS_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
    cudaMallocHost ( (void**)&OPS_gbl_prev, OPS_consts_bytes );
    OPS_consts_h = ( char * ) malloc ( OPS_consts_bytes );
    cutilSafeCall ( cudaMalloc ( ( void ** ) &OPS_consts_d,
                                 OPS_consts_bytes ) );
  }
}

void reallocReductArrays ( int reduct_bytes )
{
  if ( reduct_bytes > OPS_reduct_bytes ) {
    if ( OPS_reduct_bytes > 0 ) {
      free ( OPS_reduct_h );
      cutilSafeCall ( cudaFree ( OPS_reduct_d ) );
    }
    OPS_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
    OPS_reduct_h = ( char * ) malloc ( OPS_reduct_bytes );
    cutilSafeCall ( cudaMalloc ( ( void ** ) &OPS_reduct_d,
                                 OPS_reduct_bytes ) );
  }
}

//
// routines to move constant/reduct arrays
//



void mvConstArraysToDevice ( int consts_bytes )
{
  OPS_gbl_changed=0;
  for(int i = 0; i<consts_bytes; i++)
  {
    if (OPS_consts_h[i] != OPS_gbl_prev[i]) OPS_gbl_changed=1;
  }
  if (OPS_gbl_changed) {
    //memcpy(OPS_gbl_prev,OPS_consts_h,consts_bytes);
    //cutilSafeCall ( cudaMemcpyAsync ( OPS_consts_d, OPS_gbl_prev, consts_bytes,
    //                             cudaMemcpyHostToDevice ) );
    cutilSafeCall ( cudaMemcpy ( OPS_consts_d, OPS_consts_h, consts_bytes,
                                 cudaMemcpyHostToDevice ) );
    memcpy(OPS_gbl_prev,OPS_consts_h,consts_bytes);
  }
}

void mvReductArraysToDevice ( int reduct_bytes )
{
  cutilSafeCall ( cudaMemcpy ( OPS_reduct_d, OPS_reduct_h, reduct_bytes,
                               cudaMemcpyHostToDevice ) );
  cutilSafeCall ( cudaDeviceSynchronize ( ) );
}

void mvReductArraysToHost ( int reduct_bytes )
{
  cutilSafeCall ( cudaMemcpy ( OPS_reduct_h, OPS_reduct_d, reduct_bytes,
                               cudaMemcpyDeviceToHost ) );
  cutilSafeCall ( cudaDeviceSynchronize ( ) );
}


void ops_cuda_exit ( )
{
  if (!OPS_hybrid_gpu) return;
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries)
  {
    cutilSafeCall (cudaFree((item->dat)->data_d));
  }

  cudaDeviceReset ( );
}

