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

/** @brief ops openmp4 specific runtime support functions
  * @author Alessio De Rango
  * @details Implements openmp4 backend runtime support functions
  */

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <ops_lib_core.h>
#include <ops_openmp4_rt_support.h>
// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct cudaDeviceProp cudaDeviceProp_t;

int OPS_consts_bytes = 0, OPS_reduct_bytes = 0;

char *OPS_consts_h, *OPS_consts_d, *OPS_reduct_h, *OPS_reduct_d;

int OPS_gbl_changed = 1;
char *OPS_gbl_prev = NULL;

//
// CUDA utility functions
//


void cutilDeviceInit(int argc, char **argv) {
  (void)argc;
  (void)argv;
  // copy one scalar to initialize OpenMP env. 
  // Improvement: later we can set default device.
  int tmp=0;
  #pragma omp target enter data map(to:tmp)

  OPS_hybrid_gpu = 1;
}

void ops_mvHostToDevice(void **map, int size) {
  if (!OPS_hybrid_gpu)
    return;
  char *temp = (char*)*map;
  #pragma omp target enter data map(to: temp[:size])
  #pragma omp target update to(temp[:size])
  //TODO test
}

void ops_cpHostToDevice(void **data_d, void **data_h, int size) {
  if (!OPS_hybrid_gpu)
    return;
  //TODO jo igy? decl miatt kell az enter data elm.
  #pragma omp target enter data map(to: data_d[:size])
  #pragma omp target update to(data_d[:size])
}

void ops_download_dat(ops_dat dat) {
  if (!OPS_hybrid_gpu)
    return;
  int tot = 1;
  for (int i = 0; i < dat->block->dims; i++)
    tot = tot * dat->size[i];
  
  #pragma omp target update from(dat->data[:tot])

  // printf("downloading to host from device %d bytes\n",bytes);

}

void ops_upload_dat(ops_dat dat) {

  int tot = 1;
  for (int i = 0; i < dat->block->dims; i++)
    tot = tot * dat->size[i];
  #pragma omp target update to(dat->data[:tot])

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

//
// routine to fetch data from GPU to CPU (with transposing SoA to AoS if needed)
//

void ops_cuda_get_data(ops_dat dat) {
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  int tot = 1;
  for (int i = 0; i < dat->block->dims; i++)
    tot = tot * dat->size[i];
  #pragma omp target update from(dat->data[:tot])
}

//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(int consts_bytes) {
 (void) consts_bytes;
}

void reallocReductArrays(int reduct_bytes) {
 (void) reduct_bytes;
}

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice(int consts_bytes) {
  OPS_gbl_changed = 0;
  if (OPS_gbl_prev != NULL)
  for (int i = 0; i < consts_bytes; i++) {
    if (OPS_consts_h[i] != OPS_gbl_prev[i])
      OPS_gbl_changed = 1;
  }
  else {
    OPS_gbl_changed = 0;
    OPS_gbl_prev = (char *)malloc(consts_bytes);
    #pragma omp target enter data map(to: OPS_consts_h[0:consts_bytes]);
    memcpy(OPS_gbl_prev, OPS_consts_h, consts_bytes);
  } 
		
  
  if (OPS_gbl_changed) {
  /* if (OPS_consts_d == NULL)
    OPS_consts_d = (char *)malloc(4 * consts_bytes);
   for (int i = 0; i < consts_bytes; i++) {
    OPS_consts_d[i] = OPS_consts_h[i];
   }*/
   //int size = consts_bytes/sizeof(char);
   #pragma omp target update to( OPS_consts_h[0:consts_bytes]) 
    //cutilSafeCall(cudaMemcpy(OPS_consts_d, OPS_consts_h, consts_bytes,
    //                         cudaMemcpyHostToDevice));
    memcpy(OPS_gbl_prev, OPS_consts_h, consts_bytes);
  }
}

void mvReductArraysToDevice(int reduct_bytes) {
 (void) reduct_bytes;
}

void mvReductArraysToHost(int reduct_bytes) {
 (void) reduct_bytes;
}

void ops_cuda_exit() {
 if (!OPS_hybrid_gpu)
    return;
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    #pragma omp target exit data map(from: (item->dat)->data)
    free((item->dat)->data);
  }
}
