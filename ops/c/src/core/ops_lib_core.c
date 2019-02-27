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

/** @brief OPS core library functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the core library functions utilized by all OPS
  * backends
  */

#include "ops_lib_core.h"
#include <float.h>
#include <limits.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/time.h>

int OPS_diags = 0;

int OPS_block_index = 0, OPS_block_max = 0;
int OPS_stencil_index = 0, OPS_stencil_max = 0;
int OPS_dat_index = 0;
int OPS_kern_max = 0, OPS_kern_curr = 0;
ops_kernel *OPS_kernels = NULL;
ops_arg *OPS_curr_args = NULL;
const char *OPS_curr_name = NULL;
int OPS_hybrid_gpu = 0, OPS_gpu_direct = 0;
int OPS_halo_group_index = 0, OPS_halo_group_max = 0, OPS_halo_index = 0,
    OPS_halo_max = 0, OPS_reduction_index = 0, OPS_reduction_max = 0;
ops_reduction *OPS_reduction_list = NULL;
int OPS_enable_checkpointing = 0;
int ops_thread_offload = 0;
int ops_checkpoint_inmemory = 0;
int ops_lock_file = 0;
int ops_enable_tiling = 0;
int ops_cache_size = 0;
int OPS_soa = 0;
int ops_tiling_mpidepth = -1;
extern double ops_tiled_halo_exchange_time;
int ops_force_decomp[OPS_MAX_DIM] = {0};
int OPS_realloc = 0;

//int ops_hdf5_chunk_size[OPS_MAX_DIM] = {0};

/*
* Lists of blocks and dats declared in an OPS programs
*/

ops_block_descriptor *OPS_block_list = NULL;
ops_halo *OPS_halo_list = NULL;
ops_halo_group *OPS_halo_group_list = NULL;
ops_stencil *OPS_stencil_list = NULL;
Double_linked_list OPS_dat_list; // Head of the double linked list

int OPS_block_size_x = 32;
int OPS_block_size_y = 4;
int OPS_block_size_z = 1;

double ops_gather_time = 0.0;
double ops_scatter_time = 0.0;
double ops_sendrecv_time = 0.0;
/*
* Utility functions
*/
static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)ops_calloc(len, sizeof(char));
  return strncpy(dest, src, len);
}

int compare_blocks(ops_block block1, ops_block block2) {
  if (block1->dims == block2->dims && block1->index == block2->index &&
      strcmp(block1->name, block2->name) == 0)
    return 1;
  else
    return 0;
}

ops_dat search_dat(ops_block block, int elem_size, int *dat_size, int *offset,
                   char const *type, char const *name) {
  ops_dat_entry *item;
  ops_dat_entry *tmp_item;
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat item_dat = item->dat;

    if (strcmp(item_dat->name, name) ==
            0 && /* there are other components to compare*/
        (item_dat->elem_size) == elem_size &&
        compare_blocks(item_dat->block, block) == 1 &&
        strcmp(item_dat->type, type) == 0) {
      return item_dat;
    }
  }

  return NULL;
}

/* Special function only called by fortran backend to get
commandline arguments as argv is not easy to pass through from
frotran to C
*/
void ops_set_args(const int argc, const char *argv) {

  char temp[64];
  char *pch;
  pch = strstr(argv, "OPS_BLOCK_SIZE_X=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OPS_block_size_x = atoi(temp + 17);
    ops_printf("\n OPS_block_size_x = %d \n", OPS_block_size_x);
  }
  pch = strstr(argv, "OPS_BLOCK_SIZE_Y=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OPS_block_size_y = atoi(temp + 17);
    ops_printf("\n OPS_block_size_y = %d \n", OPS_block_size_y);
  }
  pch = strstr(argv, "OPS_BLOCK_SIZE_Z=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OPS_block_size_z = atoi(temp + 17);
    ops_printf("\n OPS_block_size_z = %d \n", OPS_block_size_z);
  }
  pch = strstr(argv, "-gpudirect");
  if (pch != NULL) {
    OPS_gpu_direct = 1;
    ops_printf("\n GPU Direct enabled\n");
  }
  pch = strstr(argv, "-OPS_DIAGS=");
  if (pch != NULL) {
    strncpy(temp, pch, 12);
    OPS_diags = atoi(temp + 11);
    ops_printf("\n OPS_diags = %d \n", OPS_diags);
  }
  pch = strstr(argv, "OPS_CACHE_SIZE=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    ops_cache_size = atoi(temp + 15);
    ops_printf("\n Cache size per process = %d \n", ops_cache_size);
  }
  pch = strstr(argv, "OPS_REALLOC=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OPS_realloc = atoi(temp + 12);
    ops_printf("\n Reallocating = %d \n", OPS_realloc);
  }

  pch = strstr(argv, "OPS_TILING");
  if (pch != NULL) {
    ops_enable_tiling = 1;
    ops_printf("\n Tiling enabled\n");
  }
	pch = strstr(argv, "OPS_TILING_MAXDEPTH=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    ops_tiling_mpidepth = atoi(temp + 20);
    ops_printf("\n Max tiling depth across processes = %d \n", ops_tiling_mpidepth);
  }
  pch = strstr(argv, "OPS_FORCE_DECOMP_X=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    ops_force_decomp[0] = atoi(temp + 19);
    ops_printf("\n Forced decomposition in x direction = %d \n", ops_force_decomp[0]);
  }
  pch = strstr(argv, "OPS_FORCE_DECOMP_Y=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    ops_force_decomp[1] = atoi(temp + 19);
    ops_printf("\n Forced decomposition in y direction = %d \n", ops_force_decomp[1]);
  }
  pch = strstr(argv, "OPS_FORCE_DECOMP_Z=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    ops_force_decomp[2] = atoi(temp + 19);
    ops_printf("\n Forced decomposition in z direction = %d \n", ops_force_decomp[2]);
  }

  if (strstr(argv, "OPS_CHECKPOINT_INMEMORY") != NULL) {
    ops_checkpoint_inmemory = 1;
    ops_printf("\n OPS Checkpointing in memory\n");
  } else if (strstr(argv, "OPS_CHECKPOINT_LOCKFILE") != NULL) {
    ops_lock_file = 1;
    ops_printf("\n OPS Checkpointing creating lockfiles\n");
  } else if (strstr(argv, "OPS_CHECKPOINT_THREAD") != NULL) {
    ops_thread_offload = 1;
    ops_printf("\n OPS Checkpointing on a separate thread\n");
  } else if (strstr(argv, "OPS_CHECKPOINT=") != NULL) {
    pch = strstr(argv, "OPS_CHECKPOINT=");
    OPS_enable_checkpointing = 2;
    strncpy(temp, pch, 20);
    OPS_ranks_per_node = atoi(temp + 15);
    ops_printf("\n OPS Checkpointing with mirroring offset %d\n",
               OPS_ranks_per_node);
  } else if (strstr(argv, "OPS_CHECKPOINT") != NULL) {
    OPS_enable_checkpointing = 1;
    ops_printf("\n OPS Checkpointing enabled\n");
  }

  /*pch = strstr(argv, "OPS_HDF5_CHUNK_SIZE=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    for (int i = 0; i < OPS_MAX_DIM; i++)
      ops_hdf5_chunk_size[i] = MAX(atoi(temp + 20),1);
    ops_printf("\n HDF5 write chunck size = (%d)^%s \n",
      ops_hdf5_chunk_size[0],OPS_MAX_DIM);
  }*/


}

/*
* OPS core functions
*/
void ops_init_core(const int argc, const char **argv, const int diags) {
  OPS_diags = diags;
  for (int d = 0; d < OPS_MAX_DIM; d++) ops_force_decomp[d] = 0;
  for (int n = 1; n < argc; n++) {

    ops_set_args(argc, argv[n]);

    /*if ( strncmp ( argv[n], "OPS_BLOCK_SIZE_X=", 17 ) == 0 )
    {
      OPS_block_size_x = atoi ( argv[n] + 17 );
      ops_printf ( "\n OPS_block_size_x = %d \n", OPS_block_size_x );
    }

    if ( strncmp ( argv[n], "OPS_BLOCK_SIZE_Y=", 17 ) == 0 )
    {
      OPS_block_size_y = atoi ( argv[n] + 17 );
      ops_printf ( "\n OPS_block_size_y = %d \n", OPS_block_size_y );
    }

    if ( strncmp ( argv[n], "-gpudirect", 10 ) == 0 )
    {
      OPS_gpu_direct = 1;
      ops_printf ( "\n GPU Direct enabled\n" );
    }
    if ( strncmp ( argv[n], "OPS_DIAGS=", 10 ) == 0 )
    {
      OPS_diags = atoi ( argv[n] + 10 );
      ops_printf ( "\n OPS_diags = %d \n", OPS_diags );
    }
    if ( strncmp ( argv[n], "OPS_CHECKPOINT=", 15 ) == 0 )
    {
      OPS_enable_checkpointing = 2;
      OPS_ranks_per_node = atoi ( argv[n] + 15 );
      ops_printf ( "\n OPS Checkpointing with mirroring offset %d\n",
    OPS_ranks_per_node);
    }
    else if ( strncmp ( argv[n], "OPS_CHECKPOINT_THREAD", 21 ) == 0 )
    {
      ops_thread_offload = 1;
      ops_printf ( "\n OPS Checkpointing on a separate thread\n");
    }
    else if ( strncmp ( argv[n], "OPS_CHECKPOINT_INMEMORY", 23 ) == 0 )
    {
      ops_checkpoint_inmemory = 1;
      ops_printf ( "\n OPS Checkpointing in memory\n");
    }
    else if ( strncmp ( argv[n], "OPS_CHECKPOINT_LOCKFILE", 23 ) == 0 )
    {
      ops_lock_file = 1;
      ops_printf ( "\n OPS Checkpointing creating lockfiles\n");
    }
    else if ( strncmp ( argv[n], "OPS_CHECKPOINT", 14 ) == 0 )
    {
      OPS_enable_checkpointing = 1;
      ops_printf ( "\n OPS Checkpointing enabled\n");
    }*/
  }

  /*Initialize the double linked list to hold ops_dats*/
  TAILQ_INIT(&OPS_dat_list);
}

void ops_exit_core() {
  ops_checkpointing_exit();
  ops_dat_entry *item;
  // free storage and pointers for blocks
  for (int i = 0; i < OPS_block_index; i++) {
    free((char *)(OPS_block_list[i].block->name));
    while ((item = TAILQ_FIRST(&(OPS_block_list[i].datasets)))) {
      TAILQ_REMOVE(&(OPS_block_list[i].datasets), item, entries);
      free(item);
    }
    free(OPS_block_list[i].block);
  }
  free(OPS_block_list);
  OPS_block_list = NULL;

  /*free doubly linked list holding the ops_dats */

  while ((item = TAILQ_FIRST(&OPS_dat_list))) {
    if ((item->dat)->user_managed == 0)
//#ifdef __INTEL_COMPILER
//      _mm_free((item->dat)->data);
//#else
      free((item->dat)->data);
//#endif
    free((char *)(item->dat)->name);
    free((char *)(item->dat)->type);
    TAILQ_REMOVE(&OPS_dat_list, item, entries);
    free(item->dat);
    free(item);
  }

  // free stencills
  for (int i = 0; i < OPS_stencil_index; i++) {
    free((char *)OPS_stencil_list[i]->name);
    free(OPS_stencil_list[i]->stencil);
    free(OPS_stencil_list[i]->stride);
    free(OPS_stencil_list[i]);
  }
  free(OPS_stencil_list);
  OPS_stencil_list = NULL;

  for (int i = 0; i < OPS_halo_index; i++) {
    free(OPS_halo_list[i]);
  }

  // free block halos
  for (int i = 0; i < OPS_halo_group_index; i++) {
    free(OPS_halo_group_list[i]->halos);
    free(OPS_halo_group_list[i]);
  }

  // reset initial values
  OPS_block_index = 0;
  OPS_dat_index = 0;
  OPS_block_max = 0;
}

/*ops_block ops_decl_block(int dims, const char *name) {
  if (dims < 0) {
    printf(
        "Error: ops_decl_block -- negative/zero dimension size for block: %s\n",
        name);
    exit(-1);
  }

  if (OPS_block_index == OPS_block_max) {
    if (OPS_block_max > 0) printf("Warning: potential realloc issue in ops_lib_core.c detected, please modify ops_decl_block to allocate more blocks initially!\n");
    OPS_block_max += 15;
    OPS_block_list = (ops_block_descriptor *)realloc(
        OPS_block_list, OPS_block_max * sizeof(ops_block_descriptor));

    if (OPS_block_list == NULL) {
      printf("Error: ops_decl_block -- error reallocating memory\n");
      exit(-1);
    }
  }

  ops_block block = (ops_block)ops_malloc(sizeof(ops_block_core));
  block->index = OPS_block_index;
  block->dims = dims;
  block->name = copy_str(name);
  OPS_block_list[OPS_block_index].block = block;
  OPS_block_list[OPS_block_index].num_datasets = 0;
  TAILQ_INIT(&(OPS_block_list[OPS_block_index].datasets));
  OPS_block_index++;

  return block;
}*/

ops_block ops_decl_block(int dims, const char *name) {
  if (dims < 0) {
    printf(
        "Error: ops_decl_block -- negative/zero dimension size for block: %s\n",
        name);
    exit(-1);
  }

  if (OPS_block_index == OPS_block_max) {
    if (OPS_block_max > 0) printf("Warning: potential realloc issue in ops_lib_core.c detected, please modify ops_decl_block to allocate more blocks initially!\n");

    OPS_block_max += 20;
    ops_block_descriptor *OPS_block_list_new = (ops_block_descriptor *)xmalloc(
        OPS_block_max * sizeof(ops_block_descriptor));
    if (OPS_block_list_new == NULL) {
      printf("Error: ops_decl_block -- error reallocating memory\n");
      exit(-1);
    }

    //copy old blocks
    for (int i = 0; i < OPS_block_index; i++) {
      OPS_block_list_new[i].block = OPS_block_list[i].block;

      TAILQ_INIT(&(OPS_block_list_new[i].datasets));
      //remove ops_dats from old queue and add to new queue
      ops_dat_entry *item;
      while ((item = TAILQ_FIRST(&(OPS_block_list[i].datasets)))) {
        TAILQ_REMOVE(&(OPS_block_list[i].datasets), item, entries);
        TAILQ_INSERT_TAIL(&OPS_block_list_new[i].datasets, item, entries);
      }

      OPS_block_list_new[i].num_datasets = OPS_block_list[i].num_datasets;

    }
    free(OPS_block_list);
    OPS_block_list = OPS_block_list_new;

  }

  ops_block block = (ops_block)xmalloc(sizeof(ops_block_core));
  block->index = OPS_block_index;
  block->dims = dims;
  block->name = copy_str(name);
  OPS_block_list[OPS_block_index].block = block;
  OPS_block_list[OPS_block_index].num_datasets = 0;
  TAILQ_INIT(&(OPS_block_list[OPS_block_index].datasets));
  OPS_block_index++;

  return block;
}


void ops_decl_const_core(int dim, char const *type, int typeSize, char *data,
                         char const *name) {
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

ops_dat ops_decl_dat_core(ops_block block, int dim, int *dataset_size,
                          int *base, int *d_m, int *d_p, char *data,
                          int type_size, char const *type, char const *name) {
  if (block == NULL) {
    printf("Error: ops_decl_dat -- invalid block for data: %s\n", name);
    exit(-1);
  }

  if (dim <= 0) {
    printf("Error: ops_decl_dat -- negative/zero number of items per grid "
           "point in data: %s\n",
           name);
    exit(-1);
  }

  ops_dat dat = (ops_dat)ops_malloc(sizeof(ops_dat_core));
  dat->index = OPS_dat_index;
  dat->block = block;
  dat->dim = dim;

  dat->type_size = type_size;
  // note here that the element size is taken to
  // be the type_size in bytes multiplied by the dimension of an element
  dat->elem_size = type_size * dim;

  dat->e_dat = 0; // default to non-edge dat

  for (int n = 0; n < block->dims; n++) {
    if (dataset_size[n] != 1) {
      // compute total size - which includes the block halo
      dat->size[n] = dataset_size[n] - d_m[n] + d_p[n];
    } else {
      dat->size[n] = 1;
      dat->e_dat = 1;
    }
  }

  for (int n = 0; n < block->dims; n++)
    dat->base[n] = base[n];

  for (int n = 0; n < block->dims; n++) {
    if (d_m[n] <= 0)
      dat->d_m[n] = d_m[n];
    else {
      ops_printf("Error: Non negative d_m during declaration of %s\n", name);
      exit(2);
    }
  }

  for (int n = 0; n < block->dims; n++) {
    if (d_p[n] >= 0)
      dat->d_p[n] = d_p[n];
    else {
      ops_printf("Error: Non positive d_p during declaration of %s\n", name);
      exit(2);
    }
  }

  // set the size of higher dimensions to 1
  for (int n = block->dims; n < OPS_MAX_DIM; n++) {
    dat->size[n] = 1;
    dat->base[n] = 0;
    dat->d_m[n] = 0;
    dat->d_p[n] = 0;
  }

  dat->data = (char *)data;
  dat->data_d = NULL;
  dat->user_managed = 1;
  dat->dirty_hd = 0;
  dat->is_hdf5 = 0;
  dat->hdf5_file = "none";
  dat->type = copy_str(type);
  dat->name = copy_str(name);
  dat->x_pad = 0; // initialize padding for data alignment to zero

  /* Create a pointer to an item in the ops_dats doubly linked list */
  ops_dat_entry *item;

  // add the newly created ops_dat to list
  item = (ops_dat_entry *)ops_malloc(sizeof(ops_dat_entry));
  if (item == NULL) {
    printf("Error: op_decl_dat -- error allocating memory to double linked "
           "list entry\n");
    exit(-1);
  }
  item->dat = dat;

  // add item to the end of the list
  TAILQ_INSERT_TAIL(&OPS_dat_list, item, entries);
  OPS_dat_index++;

  // Another entry for a different list
  item = (ops_dat_entry *)ops_malloc(sizeof(ops_dat_entry));
  if (item == NULL) {
    printf("Error: op_decl_dat -- error allocating memory to double linked "
           "list entry\n");
    exit(-1);
  }
  item->dat = dat;
  TAILQ_INSERT_TAIL(&OPS_block_list[block->index].datasets, item, entries);
  OPS_block_list[block->index].num_datasets++;

  return dat;
}

ops_dat ops_decl_dat_temp_core(ops_block block, int dim, int *dataset_size,
                               int *base, int *d_m, int *d_p, char *data,
                               int type_size, char const *type,
                               char const *name) {
  // Check if this dat already exists in the double linked list
  ops_dat found_dat = search_dat(block, dim, dataset_size, base, type, name);
  if (found_dat != NULL) {
    printf("Error: ops_dat with name %s already exists, cannot create "
           "temporary ops_dat\n ",
           name);
    exit(2);
  }
  // if not found ...
  return ops_decl_dat_core(block, dim, dataset_size, base, d_m, d_p, data,
                           type_size, type, name);
}

void ops_free_dat(ops_dat dat) {
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &OPS_dat_list, entries) {
    if (item->dat->index == dat->index) {
      TAILQ_REMOVE(&OPS_dat_list, item, entries);
      break;
    }
  }
  TAILQ_FOREACH(item, &(OPS_block_list[dat->block->index].datasets), entries) {
    if (item->dat->index == dat->index) {
      TAILQ_REMOVE(&(OPS_block_list[dat->block->index].datasets), item, entries);
      break;
    }
  }
}

ops_stencil ops_decl_stencil(int dims, int points, int *sten,
                             char const *name) {

  if (OPS_stencil_index == OPS_stencil_max) {
    OPS_stencil_max += 10;
    OPS_stencil_list = (ops_stencil *)ops_realloc(
        OPS_stencil_list, OPS_stencil_max * sizeof(ops_stencil));

    if (OPS_stencil_list == NULL) {
      printf("Error: ops_decl_stencil -- error reallocating memory\n");
      exit(-1);
    }
  }

  ops_stencil stencil = (ops_stencil)ops_malloc(sizeof(ops_stencil_core));
  stencil->index = OPS_stencil_index;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);

  stencil->stencil = (int *)ops_malloc(dims * points * sizeof(int));
  memcpy(stencil->stencil, sten, sizeof(int) * dims * points);

  stencil->stride = (int *)ops_malloc(dims * sizeof(int));
  for (int i = 0; i < dims; i++)
    stencil->stride[i] = 1;

  OPS_stencil_list[OPS_stencil_index++] = stencil;

  return stencil;
}

ops_stencil ops_decl_strided_stencil(int dims, int points, int *sten,
                                     int *stride, char const *name) {

  if (OPS_stencil_index == OPS_stencil_max) {
    OPS_stencil_max += 10;
    OPS_stencil_list = (ops_stencil *)ops_realloc(
        OPS_stencil_list, OPS_stencil_max * sizeof(ops_stencil));

    if (OPS_stencil_list == NULL) {
      printf("Error: ops_decl_stencil -- error reallocating memory\n");
      exit(-1);
    }
  }

  ops_stencil stencil = (ops_stencil)ops_malloc(sizeof(ops_stencil_core));
  stencil->index = OPS_stencil_index;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);

  stencil->stencil = (int *)ops_malloc(dims * points * sizeof(int));
  memcpy(stencil->stencil, sten, sizeof(int) * dims * points);

  stencil->stride = (int *)ops_malloc(dims * sizeof(int));
  memcpy(stencil->stride, stride, sizeof(int) * dims);

  OPS_stencil_list[OPS_stencil_index++] = stencil;
  return stencil;
}

ops_arg ops_arg_reduce_core(ops_reduction handle, int dim, const char *type,
                            ops_access acc) {
  ops_arg arg;
  arg.argtype = OPS_ARG_GBL;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = dim;
  arg.data = (char *)handle;
  arg.acc = acc;
  if (handle->initialized == 0) {
    handle->initialized = 1;
    handle->acc = acc;
    if (acc == OPS_INC)
      memset(handle->data, 0, handle->size);
    if (strcmp(type, "double") == 0 ||
        strcmp(type, "real(8)") == 0) { // TODO: handle other types
      if (acc == OPS_MIN)
        for (int i = 0; i < handle->size / 8; i++)
          ((double *)handle->data)[i] = DBL_MAX;
      if (acc == OPS_MAX)
        for (int i = 0; i < handle->size / 8; i++)
          ((double *)handle->data)[i] = -1.0 * DBL_MAX;
    } else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0) {
      if (acc == OPS_MIN)
        for (int i = 0; i < handle->size / 4; i++)
          ((float *)handle->data)[i] = FLT_MAX;
      if (acc == OPS_MAX)
        for (int i = 0; i < handle->size / 4; i++)
          ((float *)handle->data)[i] = -1.0f * FLT_MAX;
    } else if (strcmp(type, "int") == 0 || strcmp(type, "integer") == 0) {
      if (acc == OPS_MIN)
        for (int i = 0; i < handle->size / 4; i++)
          ((int *)handle->data)[i] = INT_MAX;
      if (acc == OPS_MAX)
        for (int i = 0; i < handle->size / 4; i++)
          ((int *)handle->data)[i] = -1 * INT_MAX;
    } else {
      printf("Warning: reduction type not recognised, please add in "
             "ops_lib_core.c !\n");
    }
  } else if (handle->acc != acc) {
    printf("Error: ops_reduction handle %s was aleady used with a different "
           "access type\n",
           handle->name);
    exit(-1);
  }
  return arg;
}

ops_halo_group ops_decl_halo_group(int nhalos, ops_halo halos[nhalos]) {
  if (OPS_halo_group_index == OPS_halo_group_max) {
    OPS_halo_group_max += 10;
    OPS_halo_group_list = (ops_halo_group *)ops_realloc(
        OPS_halo_group_list, OPS_halo_group_max * sizeof(ops_halo_group));

    if (OPS_halo_group_list == NULL) {
      printf("Error: ops_decl_halo_group -- error reallocating memory\n");
      exit(-1);
    }
  }

  ops_halo_group grp = (ops_halo_group)ops_malloc(sizeof(ops_halo_group_core));
  grp->nhalos = nhalos;

  //make a copy
  ops_halo* halos_temp = (ops_halo *)ops_malloc(nhalos*sizeof(ops_halo));
  memcpy(halos_temp, &halos[0], nhalos*sizeof(ops_halo));
  grp->halos = halos_temp;

  grp->index = OPS_halo_group_index;
  OPS_halo_group_list[OPS_halo_group_index++] = grp;

  return grp;
}

ops_halo_group ops_decl_halo_group_elem(int nhalos, ops_halo *halos,
                                        ops_halo_group grp) {

  if (OPS_halo_group_index == OPS_halo_group_max) {
    OPS_halo_group_max += 10;
    OPS_halo_group_list = (ops_halo_group *)ops_realloc(
        OPS_halo_group_list, OPS_halo_group_max * sizeof(ops_halo_group));

    if (OPS_halo_group_list == NULL) {
      printf("Error: ops_decl_halo_group -- error reallocating memory\n");
      exit(-1);
    }
  }

  // Test contents of halo group
  /*ops_halo halo;
  halo = halos[0];
  printf("%d halo->from->name = %s, halo->to->name %s\n",nhalos,
  halo->from->name, halo->to->name);
  for (int i = 0; i < halo->from->block->dims; i++) {
    printf("halo->iter_size[%d] %d ", i, halo->iter_size[i]);
    printf("halo->from_base[%d] %d ", i, halo->from_base[i]);
    printf("halo->to_base[%d] %d ", i, halo->to_base[i]);
    printf("halo->from_dir[%d] %d ", i, halo->from_dir[i]);
    printf("halo->to_dir[%d] %d \n", i, halo->to_dir[i]);
  }*/

  if (grp == NULL) {
    grp = (ops_halo_group)ops_malloc(sizeof(ops_halo_group_core));
    grp->nhalos = 0;
    // printf("null grp, grp->nhalos = %d\n",grp->nhalos);
    if (nhalos != 0) {
      ops_halo *halos_temp = (ops_halo *)ops_malloc(1 * sizeof(ops_halo_core));
      memcpy(halos_temp, halos, 1 * sizeof(ops_halo_core));
      grp->halos = halos_temp;
      grp->nhalos++;
    }
    grp->index = OPS_halo_group_index;
    OPS_halo_group_list[OPS_halo_group_index++] = grp;
  } else {
    // printf("NON null grp, grp->nhalos = %d\n",grp->nhalos);
    grp->halos = (ops_halo *)ops_realloc(grp->halos, (grp->nhalos + 1) *
                                                         sizeof(ops_halo_core));
    memcpy(&grp->halos[grp->nhalos], &halos[0], 1 * sizeof(ops_halo_core));
    grp->nhalos++;
  }
  return grp;
}

ops_halo ops_decl_halo_core(ops_dat from, ops_dat to, int *iter_size,
                            int *from_base, int *to_base, int *from_dir,
                            int *to_dir) {
  if (OPS_halo_index == OPS_halo_max) {
    OPS_halo_max += 10;
    OPS_halo_list =
        (ops_halo *)ops_realloc(OPS_halo_list, OPS_halo_max * sizeof(ops_halo));

    if (OPS_halo_list == NULL) {
      printf("Error: ops_decl_halo_core -- error reallocating memory\n");
      exit(-1);
    }
  }

  ops_halo halo = (ops_halo)ops_malloc(sizeof(ops_halo_core));
  halo->index = OPS_halo_index;
  halo->from = from;
  halo->to = to;
  for (int i = 0; i < from->block->dims; i++) {
    halo->iter_size[i] = iter_size[i];
    halo->from_base[i] = from_base[i];
    halo->to_base[i] = to_base[i];
    halo->from_dir[i] = from_dir[i];
    halo->to_dir[i] = to_dir[i];
  }
  for (int i = from->block->dims; i < OPS_MAX_DIM; i++) {
    halo->iter_size[i] = 1;
    halo->from_base[i] = 0;
    halo->to_base[i] = 0;
    halo->from_dir[i] = i + 1;
    halo->to_dir[i] = i + 1;
  }

  OPS_halo_list[OPS_halo_index++] = halo;
  return halo;
}

ops_arg ops_arg_dat_core(ops_dat dat, ops_stencil stencil, ops_access acc) {
  ops_arg arg;
  arg.argtype = OPS_ARG_DAT;
  arg.dat = dat;
  arg.stencil = stencil;
  if (acc == OPS_WRITE && stencil->points != 1) {
    printf("Error: OPS does not support OPS_WRITE arguments with a non (0,0,0) "
           "stencil due to potential race conditions\n");
    exit(-1);
  }
  if (dat != NULL) {
    arg.data = dat->data;
    arg.data_d = dat->data_d;
  } else {
    arg.data = NULL;
    arg.data_d = NULL;
  }
  arg.acc = acc;
  arg.opt = 1;
  return arg;
}

ops_arg ops_arg_gbl_core(char *data, int dim, int size, ops_access acc) {
  ops_arg arg;
  arg.argtype = OPS_ARG_GBL;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = dim;
  arg.data = data;
  arg.acc = acc;
  return arg;
}

ops_arg ops_arg_idx() {
  ops_arg arg;
  arg.argtype = OPS_ARG_IDX;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;
}

ops_reduction ops_decl_reduction_handle_core(int size, const char *type,
                                             const char *name) {
  if (OPS_reduction_index == OPS_reduction_max) {
    OPS_reduction_max += 10;
    OPS_reduction_list = (ops_reduction *)ops_realloc(
        OPS_reduction_list, OPS_reduction_max * sizeof(ops_reduction));

    if (OPS_reduction_list == NULL) {
      printf("Error: ops_decl_reduction_handle -- error reallocating memory\n");
      exit(-1);
    }
  }

  ops_reduction red = (ops_reduction)ops_malloc(sizeof(ops_reduction_core));
  red->initialized = 0;
  red->size = size;
  red->data = (char *)ops_malloc(size * sizeof(char));
  red->name = copy_str(name);
  red->type = copy_str(type);
  OPS_reduction_list[OPS_reduction_index] = red;
  red->index = OPS_reduction_index++;
  return red;
}

void ops_diagnostic_output() {
  if (OPS_diags > 2) {
    printf("\n OPS diagnostic output\n");
    printf(" --------------------\n");

    printf("\n block dimension\n");
    printf(" -------------------\n");
    for (int n = 0; n < OPS_block_index; n++) {
      printf(" %15s %15dD ", OPS_block_list[n].block->name,
             OPS_block_list[n].block->dims);
      printf("\n");
    }

    // printf ("\n dats item/point [block_size] [base] [d_m] [d_p]
    // memory(MBytes) block\n" );
    printf("\n %15s %15s %15s %10s %10s %10s %15s %15s\n", "data", "item/point",
           "[block_size]", "[base]", "[d_m]", "[d_p]", "memory(MBytes)",
           "block");

    printf(" ------------------------------------------------------------------"
           "-------\n");
    ops_dat_entry *item;
    double tot_memory = 0.0;
    TAILQ_FOREACH(item, &OPS_dat_list, entries) {
      printf(" %15s %15d ", (item->dat)->name, (item->dat)->dim);
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf("[%d]", (item->dat)->size[i]);
      printf(" ");
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf("[%d]", (item->dat)->base[i]);
      printf(" ");
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf("[%d]", (item->dat)->d_m[i]);
      printf(" ");
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf("[%d]", (item->dat)->d_p[i]);
      printf(" ");

      printf(" %15.3lf ", (item->dat)->mem / (1024.0 * 1024.0));
      tot_memory += (item->dat)->mem;
      printf(" %15s\n", (item->dat)->block->name);
    }
    printf("\n");
    printf("Total Memory Allocated for ops_dats (GBytes) : %.3lf\n",
           tot_memory / (1024.0 * 1024.0 * 1024.0));
    printf("\n");
  }
}

void ops_dump3(ops_dat dat, const char *name) {
  // TODO: this has to be backend-specific
  /*  char str[100];
    strcpy(str,"./dump/");
    strcat(str,name);
    strcat(str,"_");
    strcat(str,dat->name);
    //const char* file_name = dat->name;
    FILE *fp;
    if ( (fp = fopen(str,"w")) == NULL) {
      printf("can't open file %s\n",str);
      exit(2);
    }
    int x_end = dat->tail[0]==-3 ? dat->block_size[0]+dat->tail[0] :
    dat->block_size[0]+dat->tail[0]-1;
    int y_end = dat->tail[1]==-3 ? dat->block_size[1]+dat->tail[1] :
    dat->block_size[1]+dat->tail[1]-1;
    int z_end = dat->tail[2]==-3 ? dat->block_size[2]+dat->tail[2] :
    dat->block_size[2]+dat->tail[2]-1;
    for (int z = -dat->offset[2]; z < z_end; z++) {
      for (int y = -dat->offset[1]; y < y_end; y++) {
        for (int x = -dat->offset[0]; x < x_end; x++) {
          fprintf(fp,"%d %d %d
    %.17g\n",x+dat->offset[0],y+dat->offset[1],z+dat->offset[2],
            *(double*)(dat->data+8*(x+dat->block_size[0]*y+dat->block_size[1]*dat->block_size[0]*z)));

        }
      }
    }
    fclose(fp);
    */
}

void ops_print_dat_to_txtfile_core(ops_dat dat, const char *file_name) {
  // printf("file %s, name %s type = %s\n",file_name, dat->name, dat->type);

  // TODO: this has to be backend-specific
  FILE *fp;
  if ((fp = fopen(file_name, "a")) == NULL) {
    printf("Error: can't open file %s\n", file_name);
    exit(2);
  }

  if (fprintf(fp, "ops_dat:  %s \n", dat->name) < 0) {
    printf("Error: error writing to %s\n", file_name);
    exit(2);
  }
  if (fprintf(fp, "ops_dat dim:  %d \n", dat->dim) < 0) {
    printf("error writing to %s\n", file_name);
    exit(2);
  }

  if (fprintf(fp, "block Dims : %d ", dat->block->dims) < 0) {
    printf("error writing to %s\n", file_name);
    exit(2);
  }

  for (int i = 0; i < dat->block->dims; i++) {
    if (fprintf(fp, "[%d]", dat->size[i]) < 0) {
      printf("Error: error writing to %s\n", file_name);
      exit(2);
    }
  }
  fprintf(fp, "\n");

  if (fprintf(fp, "elem size %d \n", dat->elem_size) < 0) {
    printf("Error: error writing to %s\n", file_name);
    exit(2);
  }

  size_t prod[OPS_MAX_DIM+1];
  prod[0] = dat->size[0];
  for (int d = 1; d < OPS_MAX_DIM; d++) {
    prod[d] = prod[d-1] * dat->size[d];
  }
  for (int d = OPS_MAX_DIM; d <= OPS_MAX_DIM; d++)
    prod[d] = prod[d-1];

#if OPS_MAX_DIM > 5
  for (int n = 0; n < dat->size[5]; n++) {
#else
  {
  int n = 0;
#endif
  #if OPS_MAX_DIM > 4
    for (int m = 0; m < dat->size[4]; m++) {
  #else
    {
    int m = 0;
  #endif
    #if OPS_MAX_DIM > 3
      for (int l = 0; l < dat->size[3]; l++) {
    #else
      {
      int l = 0;
    #endif
      #if OPS_MAX_DIM > 2
        for (int k = 0; k < dat->size[2]; k++) {
      #else
        {
        int k = 0;
      #endif
        #if OPS_MAX_DIM > 1
          for (int j = 0; j < dat->size[1]; j++) {
        #else
          {
          int j = 0;
        #endif
          #if OPS_MAX_DIM > 0
            for (int i = 0; i < dat->size[0]; i++) {
          #else
            {
            int i = 0;
          #endif

              for (int d = 0; d < dat->dim; d++) {

                size_t offset = OPS_soa ?
                        (n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i + d * prod[5])
                      :((n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i)*dat->dim + d);
                if (strcmp(dat->type, "double") == 0 || strcmp(dat->type, "real(8)") == 0 ||
                    strcmp(dat->type, "double precision") == 0) {
                  if (fprintf(fp, " %3.10lf", ((double *)dat->data)[offset]) < 0) {
                    printf("Error: error writing to %s\n", file_name);
                    exit(2);
                  }
                } else if (strcmp(dat->type, "float") == 0 ||
                           strcmp(dat->type, "real") == 0) {
                  if (fprintf(fp, "%e ", ((float *)dat->data)[offset]) < 0) {
                    printf("Error: error writing to %s\n", file_name);
                    exit(2);
                  }
                } else if (strcmp(dat->type, "int") == 0 ||
                           strcmp(dat->type, "integer") == 0 ||
                           strcmp(dat->type, "integer(4)") == 0 ||
                          strcmp(dat->type, "int(4)") == 0) {
                  if (fprintf(fp, "%d ", ((int *)dat->data)[offset]) < 0) {
                    printf("Error: error writing to %s\n", file_name);
                    exit(2);
                  }
                } else {
                  printf("Error: Unknown type %s, cannot be written to file %s\n",
                         dat->type, file_name);
                  exit(2);
                }
              } //d
            } //i
            if (dat->size[0] > 1) fprintf(fp, "\n");
          }//j
          if (dat->size[1] > 1) fprintf(fp, "\n");
        }//k
        if (dat->size[2] > 1) fprintf(fp, "\n");
      }//l
      if (dat->size[3] > 1) fprintf(fp, "\n");
    }//m
    if (dat->size[4] > 1) fprintf(fp, "\n");
  }//n
  if (dat->size[5] > 1) fprintf(fp, "\n");

  fclose(fp);
}

void ops_timing_output_stdout() { ops_timing_output(stdout); }

void ops_timing_output(FILE *stream) {
  if (stream == NULL)
    stream = stdout;

  if (OPS_diags > 1)
    if (OPS_enable_checkpointing)
      ops_printf("\nTotal time spent in checkpointing: %g seconds\n",
                 OPS_checkpointing_time);
  if (OPS_diags > 1) {
    int maxlen = 0;
    for (int i = 0; i < OPS_kern_max; i++) {
      if (OPS_kernels[i].count > 0)
        maxlen = MAX(maxlen, strlen(OPS_kernels[i].name));
      if (OPS_kernels[i].count > 0 && strlen(OPS_kernels[i].name) > 50) {
        printf("Too long\n");
      }
    }
    char *buf = (char *)ops_malloc((maxlen + 180) * sizeof(char));
    char buf2[180];
    sprintf(buf, "Name");
    for (int i = 4; i < maxlen; i++)
      strcat(buf, " ");
    ops_fprintf(stream, "\n\n%s  Count Time     MPI-time     Bandwidth(GB/s)\n",
                buf);

    sprintf(buf, "");
    for (int i = 0; i < maxlen + 31; i++)
      strcat(buf, "-");
    ops_fprintf(stream, "%s\n", buf);
    double sumtime = 0.0f;
    double sumtime_mpi = 0.0f;
    for (int k = 0; k < OPS_kern_max; k++) {
      double moments_mpi_time[2] = {0.0};
      double moments_time[2] = {0.0};
      ops_compute_moment(OPS_kernels[k].time, &moments_time[0],
                         &moments_time[1]);
      ops_compute_moment(OPS_kernels[k].mpi_time, &moments_mpi_time[0],
                         &moments_mpi_time[1]);

      if (OPS_kernels[k].count < 1)
        continue;
      sprintf(buf, "%s", OPS_kernels[k].name);
      for (int i = strlen(OPS_kernels[k].name); i < maxlen + 2; i++)
        strcat(buf, " ");

      sprintf(
          buf2, "%-5d %-6f (%-6f) %-6f (%-6f)  %-13.2f", OPS_kernels[k].count,
          moments_time[0],
          sqrt(moments_time[1] - moments_time[0] * moments_time[0]),
          moments_mpi_time[0],
          sqrt(moments_mpi_time[1] - moments_mpi_time[0] * moments_mpi_time[0]),
          OPS_kernels[k].transfer / ((moments_time[0]) * 1024 * 1024 * 1024));

      // sprintf(buf2,"%-5d %-6f  %-6f  %-13.2f", OPS_kernels[k].count,
      // OPS_kernels[k].time,
      //  OPS_kernels[k].mpi_time,
      //  OPS_kernels[k].transfer/OPS_kernels[k].time/1000/1000/1000);
      ops_fprintf(stream, "%s%s\n", buf, buf2);
      sumtime += moments_time[0];
      sumtime_mpi += moments_mpi_time[0];
    }
    ops_fprintf(stream, "Total kernel time: %g\n", sumtime);
    if (ops_tiled_halo_exchange_time > 0.0) {
      double moments_time[2] = {0.0};
      ops_compute_moment(ops_tiled_halo_exchange_time, &moments_time[0], &moments_time[1]);
      ops_fprintf(stream, "Total tiled halo exchange time: %g\n", moments_time[0]);
    } else if (sumtime_mpi > 0) {
      ops_fprintf(stream, "Total halo exchange time: %g\n", sumtime_mpi);
    }
    // printf("Times: %g %g %g\n",ops_gather_time, ops_sendrecv_time,
    // ops_scatter_time);
    free(buf);
  }
}

void ops_timers_core(double *cpu, double *et) {
  (void)cpu;
  struct timeval t;

  gettimeofday(&t, (struct timezone *)0);
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
}

void ops_timing_realloc(int kernel, const char *name) {
  int OPS_kern_max_new;
  OPS_kern_curr = kernel;

  if (kernel >= OPS_kern_max) {
    OPS_kern_max_new = kernel + 10;
    OPS_kernels = (ops_kernel *)ops_realloc(
        OPS_kernels, OPS_kern_max_new * sizeof(ops_kernel));
    if (OPS_kernels == NULL) {
      printf("Error: ops_timing_realloc error \n");
      exit(-1);
    }

    for (int n = OPS_kern_max; n < OPS_kern_max_new; n++) {
      OPS_kernels[n].count = -1;
      OPS_kernels[n].time = 0.0f;
      OPS_kernels[n].transfer = 0.0f;
      OPS_kernels[n].mpi_time = 0.0f;
    }
    OPS_kern_max = OPS_kern_max_new;
  }

  if (OPS_kernels[kernel].count == -1) {
    OPS_kernels[kernel].name =
        (char *)ops_malloc((strlen(name) + 1) * sizeof(char));
    strcpy(OPS_kernels[kernel].name, name);
    OPS_kernels[kernel].count = 0;
  }
}

float ops_compute_transfer(int dims, int *start, int *end, ops_arg *arg) {
  float size = 1.0f;
  for (int i = 0; i < dims; i++) {
    if (arg->stencil->stride[i] != 0 && (end[i] - start[i]) > 0)
      size *= end[i] - start[i];
  }
  size *=
      arg->dat->elem_size *
      ((arg->acc == OPS_READ || arg->acc == OPS_WRITE) ? 1.0f : 2.0f);
  return size;
}

void ops_compute_transfer_f(int dims, int *start, int *end, ops_arg *arg,
                            float *value) {
  *value = ops_compute_transfer(dims, start, end, arg);
}

void ops_register_args(ops_arg *args, const char *name) {
  OPS_curr_args = args;
  OPS_curr_name = name;
}

int ops_stencil_check_1d(int arg_idx, int idx0, int dim0) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[1 * i] == idx0) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d) not found in declaration %s in loop %s "
             "arg %d : %s\n",
             idx0, OPS_curr_args[arg_idx].stencil->name, OPS_curr_name, arg_idx,
             OPS_curr_args[arg_idx].dat->name);
      exit(-1);
    }
  }
  return idx0;
}

int ops_stencil_check_1d_md(int arg_idx, int idx0, int mult_d, int d) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[1 * i] == idx0) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d) not found in declaration %s in loop %s "
             "arg %d : %s\n",
             idx0, OPS_curr_args[arg_idx].stencil->name, OPS_curr_name, arg_idx,
             OPS_curr_args[arg_idx].dat->name);
      exit(-1);
    }
  }
  return idx0 * mult_d + d;
}

int ops_stencil_check_2d(int arg_idx, int idx0, int idx1, int dim0, int dim1) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[2 * i] == idx0 &&
          OPS_curr_args[arg_idx].stencil->stencil[2 * i + 1] == idx1) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d,%d) not found in declaration %s in loop "
             "%s arg %d\n",
             idx0, idx1, OPS_curr_args[arg_idx].stencil->name, OPS_curr_name,
             arg_idx);
      exit(-1);
    }
  }
  return idx0 + dim0 * (idx1);
}
int ops_stencil_check_3d(int arg_idx, int idx0, int idx1, int idx2, int dim0,
                         int dim1) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[3 * i] == idx0 &&
          OPS_curr_args[arg_idx].stencil->stencil[3 * i + 1] == idx1 &&
          OPS_curr_args[arg_idx].stencil->stencil[3 * i + 2] == idx2) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d,%d,%d) not found in declaration %s in "
             "loop %s arg %d\n",
             idx0, idx1, idx2, OPS_curr_args[arg_idx].stencil->name,
             OPS_curr_name, arg_idx);
      exit(-1);
    }
  }
  return idx0 + dim0 * (idx1) + dim0 * dim1 * (idx2);
}

int ops_stencil_check_4d(int arg_idx, int idx0, int idx1, int idx2, int idx3, int dim0,
                         int dim1, int dim2) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[4 * i] == idx0 &&
          OPS_curr_args[arg_idx].stencil->stencil[4 * i + 1] == idx1 &&
          OPS_curr_args[arg_idx].stencil->stencil[4 * i + 2] == idx2 &&
          OPS_curr_args[arg_idx].stencil->stencil[4 * i + 3] == idx3) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d,%d,%d,%d) not found in declaration %s in "
             "loop %s arg %d\n",
             idx0, idx1, idx2, idx3, OPS_curr_args[arg_idx].stencil->name,
             OPS_curr_name, arg_idx);
      exit(-1);
    }
  }
  return idx0 + dim0 * (idx1) + dim0 * dim1 * (idx2) + dim0 * dim1 * dim2 * idx3;
}

int ops_stencil_check_5d(int arg_idx, int idx0, int idx1, int idx2, int idx3, int idx4, int dim0,
                         int dim1, int dim2, int dim3) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[5 * i] == idx0 &&
          OPS_curr_args[arg_idx].stencil->stencil[5 * i + 1] == idx1 &&
          OPS_curr_args[arg_idx].stencil->stencil[5 * i + 2] == idx2 &&
          OPS_curr_args[arg_idx].stencil->stencil[5 * i + 3] == idx3 &&
          OPS_curr_args[arg_idx].stencil->stencil[5 * i + 4] == idx4) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d,%d,%d,%d,%d) not found in declaration %s in "
             "loop %s arg %d\n",
             idx0, idx1, idx2, idx3, idx4, OPS_curr_args[arg_idx].stencil->name,
             OPS_curr_name, arg_idx);
      exit(-1);
    }
  }
  return idx0 + dim0 * (idx1) + dim0 * dim1 * (idx2) + dim0 * dim1 * dim2 * idx3 + dim0 * dim1 * dim2 * dim3 * idx4;
}


void ops_NaNcheck_core(ops_dat dat) {

  size_t prod[OPS_MAX_DIM+1];
  prod[0] = dat->size[0];
  for (int d = 1; d < OPS_MAX_DIM; d++) {
    prod[d] = prod[d-1] * dat->size[d];
  }
  for (int d = OPS_MAX_DIM; d <= OPS_MAX_DIM; d++)
    prod[d] = prod[d-1];

#if OPS_MAX_DIM > 5
  for (int n = 0; n < dat->size[5]; n++) {
#else
  {
  int n = 0;
#endif
  #if OPS_MAX_DIM > 4
    for (int m = 0; m < dat->size[4]; m++) {
  #else
    {
    int m = 0;
  #endif
    #if OPS_MAX_DIM > 3
      for (int l = 0; l < dat->size[3]; l++) {
    #else
      {
      int l = 0;
    #endif
      #if OPS_MAX_DIM > 2
        for (int k = 0; k < dat->size[2]; k++) {
      #else
        {
        int k = 0;
      #endif
        #if OPS_MAX_DIM > 1
          for (int j = 0; j < dat->size[1]; j++) {
        #else
          {
          int j = 0;
        #endif
          #if OPS_MAX_DIM > 0
            for (int i = 0; i < dat->size[0]; i++) {
          #else
            {
            int i = 0;
          #endif

              for (int d = 0; d < dat->dim; d++) {

                size_t offset = OPS_soa ?
                        (n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i + d * prod[5])
                      :((n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i)*dat->dim + d);
                if (strcmp(dat->type, "double") == 0 || strcmp(dat->type, "real(8)") == 0 ||
                    strcmp(dat->type, "double precision") == 0) {
                  if (  isnan(((double *)dat->data)[offset])  ) {
                    printf("Error: NaN detected at element %d\n", offset);
                    exit(2);
                  }
                } else if (strcmp(dat->type, "float") == 0 ||
                           strcmp(dat->type, "real") == 0) {
                  if (  isnan(((float *)dat->data)[offset])  ) {
                    printf("Error: NaN detected at element %d\n", offset);
                    exit(2);
                  }
                } else if (strcmp(dat->type, "int") == 0 ||
                           strcmp(dat->type, "integer") == 0 ||
                           strcmp(dat->type, "integer(4)") == 0 ||
                          strcmp(dat->type, "int(4)") == 0) {
                  // do nothing
                } else {
                  printf("Error: Unknown type %s, cannot check for NaNs\n", dat->type);
                  exit(2);
                }
              } //d
            } //i
          }//j
        }//k
      }//l
    }//m
  }//n
}


/* Called from Fortran to set the indices to C*/
ops_halo ops_decl_halo_convert(ops_dat from, ops_dat to, int *iter_size,
                               int *from_base, int *to_base, int *from_dir,
                               int *to_dir) {

  for (int i = 0; i < from->block->dims; i++) {
    from_base[i]--;
    to_base[i]--;
  }

  ops_halo temp = ops_decl_halo_core(from, to, iter_size, from_base, to_base,
                                     from_dir, to_dir);

  for (int i = 0; i < from->block->dims; i++) {
    from_base[i]++;
    to_base[i]++;
  }

  return temp;
}

void setKernelTime(int id, char name[], double kernelTime, double mpiTime,
                   float transfer, int count) {
  ops_timing_realloc(id, name);

  OPS_kernels[id].count += count;
  OPS_kernels[id].time += (float)kernelTime;
  OPS_kernels[id].mpi_time += (float)mpiTime;
  OPS_kernels[id].transfer += transfer;
}

void *ops_malloc(size_t size) {
#ifdef __INTEL_COMPILER
  // return _mm_malloc(size, OPS_ALIGNMENT);
  return memalign(OPS_ALIGNMENT, size);
#else
  return xmalloc(size);
#endif
}

void *ops_calloc(size_t num, size_t size) {
#ifdef __INTEL_COMPILER
  // void * ptr = _mm_malloc(num*size, OPS_ALIGNMENT);
  void *ptr = memalign(OPS_ALIGNMENT, num * size);
  memset(ptr, 0, num * size);
  return ptr;
#else
  return xcalloc(num, size);
#endif
}

void *ops_realloc(void *ptr, size_t size) {
#ifdef __INTEL_COMPILER
  void *newptr = xrealloc(ptr, size);
  if (((unsigned long)newptr & (OPS_ALIGNMENT - 1)) != 0) {
    void *newptr2 = memalign(OPS_ALIGNMENT, size);
    // void *newptr2 = _mm_malloc(size, OPS_ALIGNMENT);
    memcpy(newptr2, newptr, size);
    free(newptr);
    return newptr2;
  } else {
    return newptr;
  }
#else
  return xrealloc(ptr, size);
#endif
}

void ops_free(void *ptr) {
#ifdef __INTEL_COMPILER
  //_mm_free(ptr);
  free(ptr);
#else
  free(ptr);
#endif
}
