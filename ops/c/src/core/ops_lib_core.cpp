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
  * @brief OPS core library functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the core library functions utilized by all OPS
  * backends
  */

#include <ops_lib_core.h>
#include <ops_exceptions.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

#include <assert.h>
#include <string>
#include <memory>
#if __cplusplus>=201103L
#include <chrono>
#else
#ifdef __unix__
#include <sys/time.h>
#elif defined (_WIN32) || defined(WIN32)
#include <windows.h>
#endif
#endif

#if defined (_WIN32) || defined(WIN32)
#include <malloc.h>
int posix_memalign(void **memptr, size_t alignment, size_t size) {
  *memptr = _aligned_malloc(size, alignment);
  return *memptr == NULL;
}
#endif
/*
* Utility functions
*/
static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)ops_calloc(len, sizeof(char));
  snprintf(dest, len, "%s", src);
  return dest;
}

std::vector<int> splitStringInt(std::string probsString, char sep) {
    std::stringstream ss(probsString);
    std::string arg;
    std::vector<int> params;
    for (char i; ss >> i;) {
        arg.push_back(i);
        if (ss.peek() == sep || ss.peek() == '\n') {
            if (arg.length() > 0 && isdigit(arg[0])) {
                params.push_back(atoi(arg.c_str()));
                arg.clear();
            }
            ss.ignore();
            if (ss.peek() == '\n') break;
        }
    }
    if (arg.length() > 0 && isdigit(arg[0])) {
        params.push_back(atoi(arg.c_str()));
        arg.clear();
    }
    return params;
}

void _ops_set_args(OPS_instance *instance, const char *argv) {
  char temp[64];
  const char *pch;
  pch = strstr(argv, "OPS_BLOCK_SIZE_X=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->OPS_block_size_x = atoi(temp + 17);
    if (instance->is_root()) instance->ostream() << "\n OPS_block_size_x = " << instance->OPS_block_size_x << '\n';
  }
  pch = strstr(argv, "OPS_BLOCK_SIZE_Y=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->OPS_block_size_y = atoi(temp + 17);
    if (instance->is_root()) instance->ostream() <<"\n OPS_block_size_y = " << instance->OPS_block_size_y  << '\n';
  }
  pch = strstr(argv, "OPS_BLOCK_SIZE_Z=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->OPS_block_size_z = atoi(temp + 17);
    if (instance->is_root()) instance->ostream() << "\n OPS_block_size_z = " << instance->OPS_block_size_z  << '\n';
  }
  pch = strstr(argv, "-gpudirect");
  if (pch != NULL) {
    instance->OPS_gpu_direct = 1;
    if (instance->is_root()) instance->ostream() << "\n GPU Direct enabled\n";
  }
  pch = strstr(argv, "-OPS_DIAGS=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->OPS_diags = atoi(temp + strlen("-OPS_DIAGS="));
    if (instance->is_root()) instance->ostream() << "\n OPS_diags = " << instance->OPS_diags << '\n';
  }
  pch = strstr(argv, "OPS_CACHE_SIZE=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->ops_cache_size = atoi(temp + 15);
    if (instance->is_root()) instance->ostream() << "\n Cache size per process = " << instance->ops_cache_size << '\n';
  }
  pch = strstr(argv, "OPS_REALLOC=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->OPS_realloc = atoi(temp + 12);
    if (instance->is_root()) instance->ostream() << "\n Reallocating = " << instance->OPS_realloc << '\n';
  }

  pch = strstr(argv, "OPS_TILING");
  if (pch != NULL) {
    instance->ops_enable_tiling = 1;
    if (instance->is_root()) instance->ostream() << "\n Tiling enabled\n";
  }
	pch = strstr(argv, "OPS_TILING_MAXDEPTH=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->ops_tiling_mpidepth = atoi(temp + 20);
    if (instance->is_root()) instance->ostream() << "\n Max tiling depth across processes = " << instance->ops_tiling_mpidepth << '\n';
  }
  pch = strstr(argv, "OPS_PROCESSES_PER_BLOCK=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    std::string proclist(temp+24);
    instance->processes_per_block = splitStringInt(proclist, ',');
    if (instance->is_root()) instance->ostream() << "\n number of processes assigned to blocks = " << proclist << '\n';
  }
	pch = strstr(argv, "OPS_TILESIZE_X=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->tilesize_x = atoi(temp + 15);
    if (instance->is_root()) instance->ostream() << "\n Tile size in X = " << instance->tilesize_x << '\n';
  }
	pch = strstr(argv, "OPS_TILESIZE_Y=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->tilesize_y = atoi(temp + 15);
    if (instance->is_root()) instance->ostream() << "\n Tile size in Y = " << instance->tilesize_y << '\n';
  }
	pch = strstr(argv, "OPS_TILESIZE_Z=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    instance->tilesize_z = atoi(temp + 15);
    if (instance->is_root()) instance->ostream() << "\n Tile size in Z = " << instance->tilesize_z << '\n';
  }

  pch = strstr(argv, "OPS_FORCE_DECOMP_X=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    std::string counts(temp+19);
    instance->ops_force_decomp_x = splitStringInt(counts, ',');
    if (instance->is_root()) instance->ostream() << "\n Forced decomposition in x direction = " << counts << '\n';
  }
  pch = strstr(argv, "OPS_FORCE_DECOMP_Y=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    std::string counts(temp+19);
    instance->ops_force_decomp_y = splitStringInt(counts, ',');
    if (instance->is_root()) instance->ostream() << "\n Forced decomposition in y direction = " << counts << '\n';
  }
  pch = strstr(argv, "OPS_FORCE_DECOMP_Z=");
  if (pch != NULL) {
    snprintf(temp, 64, "%s", pch);
    std::string counts(temp+19);
    instance->ops_force_decomp_z = splitStringInt(counts, ',');
    if (instance->is_root()) instance->ostream() << "\n Forced decomposition in z direction = " << counts << '\n';
  }

  if (strstr(argv, "OPS_CHECKPOINT_INMEMORY") != NULL) {
    instance->ops_checkpoint_inmemory = 1;
    if (instance->is_root()) instance->ostream() << "\n OPS Checkpointing in memory\n";
  } else if (strstr(argv, "OPS_CHECKPOINT_LOCKFILE") != NULL) {
    instance->ops_lock_file = 1;
    if (instance->is_root()) instance->ostream() << "\n OPS Checkpointing creating lockfiles\n";
  } else if (strstr(argv, "OPS_CHECKPOINT_THREAD") != NULL) {
    instance->ops_thread_offload = 1;
    if (instance->is_root()) instance->ostream() << "\n OPS Checkpointing on a separate thread\n";
  } else if (strstr(argv, "OPS_CHECKPOINT=") != NULL) {
    pch = strstr(argv, "OPS_CHECKPOINT=");
    instance->OPS_enable_checkpointing = 2;
    snprintf(temp, 64, "%s", pch);
    instance->OPS_ranks_per_node = atoi(temp + 15);
    if (instance->is_root()) instance->ostream() << "\n OPS Checkpointing with mirroring offset " <<
               instance->OPS_ranks_per_node << '\n';
  } else if (strstr(argv, "OPS_CHECKPOINT") != NULL) {
    instance->OPS_enable_checkpointing = 1;
    if (instance->is_root()) instance->ostream() << "\n OPS Checkpointing enabled\n";
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

extern "C" void ops_set_args_ftn(char *argv, int len) {
  argv[len]='\0';
  _ops_set_args(OPS_instance::getOPSInstance(), argv);
}

/* Special function only called by fortran backend to get
commandline arguments as argv is not easy to pass through from
Fortran to C
*/
extern "C" void ops_set_args(const char *argv) {
  _ops_set_args(OPS_instance::getOPSInstance(), argv);
}

/*
* OPS core functions
*/
void ops_init_core(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  instance->OPS_diags = diags;
  for (int n = 1; n < argc; n++) {
    _ops_set_args(instance, argv[n]);
  }

  /*Initialize the double linked list to hold ops_dats*/
  TAILQ_INIT(&instance->OPS_dat_list);
}

void _ops_init(OPS_instance *instance, const int argc, const char * const argv[], const int diags) {
  ops_init_core(instance, argc, argv, diags);
  ops_init_device(instance, argc, argv, diags);
}

void ops_init(const int argc, const char *const argv[], const int diags) {
  _ops_init(OPS_instance::getOPSInstance(), argc, argv, diags);
}

void ops_exit_core(OPS_instance *instance) {
  ops_checkpointing_exit(instance);
  ops_exit_lazy(instance);
  ops_dat_entry *item = TAILQ_FIRST(&instance->OPS_dat_list);

  /*free doubly linked list holding the ops_dats */

  while (item) {
    TAILQ_REMOVE(&instance->OPS_dat_list, item, entries);
    delete item->dat;
    ops_free(item);
    item = TAILQ_FIRST(&instance->OPS_dat_list);
  }

  // free storage and pointers for blocks
  for (int i = 0; i < instance->OPS_block_index; i++) {
    ops_free((char *)(instance->OPS_block_list[i].block->name));
    item = TAILQ_FIRST(&(instance->OPS_block_list[i].datasets));
    while (item) {
      TAILQ_REMOVE(&(instance->OPS_block_list[i].datasets), item, entries);
      ops_free(item);
      item = TAILQ_FIRST(&(instance->OPS_block_list[i].datasets));
    }
    ops_free(instance->OPS_block_list[i].block);
  }
  ops_free(instance->OPS_block_list);
  instance->OPS_block_list = NULL;


  // free stencils
  for (int i = 0; i < instance->OPS_stencil_index; i++) {
    ops_free((char *)instance->OPS_stencil_list[i]->name);
    ops_free(instance->OPS_stencil_list[i]->stencil);
    ops_free(instance->OPS_stencil_list[i]->stride);
    ops_free(instance->OPS_stencil_list[i]->mgrid_stride);
    ops_free(instance->OPS_stencil_list[i]);
  }
  ops_free(instance->OPS_stencil_list);
  instance->OPS_stencil_list = NULL;

  for (int i = 0; i < instance->OPS_halo_index; i++) {
    ops_free(instance->OPS_halo_list[i]);
  }
  ops_free(instance->OPS_halo_list);

  for (int i = 0; i < instance->OPS_halo_group_index; i++) {
    ops_free(instance->OPS_halo_group_list[i]->halos);
    ops_free(instance->OPS_halo_group_list[i]);
  }
  ops_free(instance->OPS_halo_group_list);

  for (int i = 0; i < instance->OPS_reduction_index; i++) {
    ops_free(instance->OPS_reduction_list[i]->data);
    ops_free(instance->OPS_reduction_list[i]->type);
    ops_free(instance->OPS_reduction_list[i]->name);
    ops_free(instance->OPS_reduction_list[i]);
  }
  ops_free(instance->OPS_reduction_list);

  // reset initial values
  instance->OPS_block_index = 0;
  instance->OPS_dat_index = 0;
  instance->OPS_block_max = 0;

  if (instance->OPS_kern_max != 0) {
     assert(instance->OPS_kernels);
     for (int n = -1; n < instance->OPS_kern_max; n++) {
        if (instance->OPS_kernels[n].name != NULL)
           ops_free(instance->OPS_kernels[n].name);
     }
     ops_free(instance->OPS_kernels-1);
     instance->OPS_kern_max=0;
  }

  instance->is_initialised = 0;
}

ops_block _ops_decl_block(OPS_instance *instance, int dims, const char *name) {
  if (dims <= 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_block -- negative/zero dimension size for block: " << name;
      throw ex;
  }
  if (dims > OPS_MAX_DIM) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_block -- too large dimension for block: " << name << " please change OPS_MAX_DIM in ops_lib_core.h and recompile OPS.";
      throw ex;
  }

  if (instance->OPS_block_index == instance->OPS_block_max) {
    instance->OPS_block_max += 20;

    ops_block_descriptor *OPS_block_list_new = (ops_block_descriptor *)ops_calloc(1,
        instance->OPS_block_max * sizeof(ops_block_descriptor));

    if (OPS_block_list_new == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_block -- error reallocating memory");
    }

    //copy old blocks
    for (int i = 0; i < instance->OPS_block_index; i++) {
      OPS_block_list_new[i].block = instance->OPS_block_list[i].block;

      TAILQ_INIT(&(OPS_block_list_new[i].datasets));
      //remove ops_dats from old queue and add to new queue
      ops_dat_entry *item= TAILQ_FIRST(&(instance->OPS_block_list[i].datasets));
      while (item) {
        TAILQ_REMOVE(&(instance->OPS_block_list[i].datasets), item, entries);
        TAILQ_INSERT_TAIL(&OPS_block_list_new[i].datasets, item, entries);
        item = TAILQ_FIRST(&(instance->OPS_block_list[i].datasets));
      }

      OPS_block_list_new[i].num_datasets = instance->OPS_block_list[i].num_datasets;

    }
    ops_free(instance->OPS_block_list);
    instance->OPS_block_list = OPS_block_list_new;

  }

  ops_block block = (ops_block)ops_calloc(1, sizeof(ops_block_core));
  block->index = instance->OPS_block_index;
  block->dims = dims;
  block->name = copy_str(name);
  block->instance = instance;
  instance->OPS_block_list[instance->OPS_block_index].block = block;
  instance->OPS_block_list[instance->OPS_block_index].num_datasets = 0;
  TAILQ_INIT(&(instance->OPS_block_list[instance->OPS_block_index].datasets));
  instance->OPS_block_index++;

  return block;
}

ops_block ops_decl_block(int dims, const char *name) {
  return _ops_decl_block(OPS_instance::getOPSInstance(), dims, name);
}

void ops_decl_const_core(int dim, char const *type, int typeSize, char *data,
                         char const *name) {
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

void ops_dat_init_metadata_core(
      ops_dat dat,   // Input, the ops_dat whose metadata needs updating. Was previously created by call to
                     // ops_dat_alloc_and_link_core so internal linked lists are already valid
      // The metadata to update
      int dim, int *dataset_size,
      int *base, int *d_m, int *d_p, int *stride, char *data,
      int type_size, char const *type, const char *name)
{
  if (dim <= 0) {
    OPSException ex(OPS_INVALID_ARGUMENT);
    ex << "Error: ops_decl_dat -- negative/zero number of items per grid point in dataset: " << name;
    throw ex;
  }
  ops_block block = dat->block;
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
    if (dat->size[n] == 1)
      dat->base[n] = 0;
    else
      dat->base[n] = base[n];

  for (int n = 0; n < block->dims; n++) {
    if (d_m[n] <= 0)
      dat->d_m[n] = d_m[n];
    else {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: ops_decl_dat -- Non negative d_m during declaration of: " << name;
      throw ex;
    }
  }

  for (int n = 0; n < block->dims; n++) {
    if (d_p[n] >= 0)
      dat->d_p[n] = d_p[n];
    else {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: ops_decl_dat -- Non positive d_p during declaration of: " << name;
      throw ex;
    }
  }
  for(int n=0;n<block->dims;n++)
    dat->stride[n] = stride[n];

  // set the size of higher dimensions to 1
  for (int n = block->dims; n < OPS_MAX_DIM; n++) {
    dat->size[n] = 1;
    dat->base[n] = 0;
    dat->d_m[n] = 0;
    dat->d_p[n] = 0;
  }

  dat->data = (char *)data;
  dat->user_managed = 1;
  dat->is_hdf5 = 0;
  dat->x_pad = 0; // initialize padding for data alignment to zero
  if( dat->hdf5_file == nullptr ) {
     dat->hdf5_file = "none";
  }
  if( dat->type != nullptr) {
     ops_free(  (void*)dat->type );
     dat->type = nullptr;
  }
  if(type != nullptr ) {
     dat->type = copy_str(type);
  }
  if(dat->name != nullptr) {
     ops_free( (void*)dat->name);
     dat->name = nullptr;
  }
  if(name != nullptr) {
     dat->name = copy_str(name);
  }
  // NOTE: this function does NOT initialise 
  //   dat->data_d         device memory pointer
  //   dat->mem            host memory buffer size is uninitialised
  //   dat->base_offset    base offset is uninitialised
  //   dat->dirty_hd
  //   dat->locked_hd
  // These quantities are computed differently for different backends
}


/**
 * Allocate an ops_dat on a given ops_block and insert into internal linked lists.
 * Return the ops_dat.  The ops_dat has a valid index and block, but nothing else.
 */
ops_dat ops_dat_alloc_core(ops_block block)
{
  if (block == NULL) {
    OPSException ex(OPS_INVALID_ARGUMENT);
    ex << "Error: ops_dat_alloc_core -- invalid block passed in";
    throw ex;
  }

  // This will memset(0) the entire ops_dat_core
  ops_dat dat = new ops_dat_core;
  dat->index = block->instance->OPS_dat_index++;
  dat->block = block;

  /* Create a pointer to an item in the ops_dats doubly linked list */
  ops_dat_entry *item;

  // add the newly created ops_dat to list
  item = (ops_dat_entry *)ops_calloc(1, sizeof(ops_dat_entry));
  if (item == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, op_decl_dat -- error allocating memory to double linked list entry");
  }
  item->dat = dat;

  // add item to the end of the list
  TAILQ_INSERT_TAIL(&block->instance->OPS_dat_list, item, entries);

  // Another entry for a different list
  item = (ops_dat_entry *)ops_calloc(1, sizeof(ops_dat_entry));
  if (item == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, op_decl_dat -- error allocating memory to double linked list entry");
  }
  item->dat = dat;
  TAILQ_INSERT_TAIL(&block->instance->OPS_block_list[block->index].datasets, item, entries);
  block->instance->OPS_block_list[block->index].num_datasets++;

  return dat;
}


ops_dat ops_decl_dat_core(ops_block block, int dim, int *dataset_size,
                          int *base, int *d_m, int *d_p, int *stride, char *data,
                          int type_size, char const *type, char const *name) 
{
   ops_dat dat = ops_dat_alloc_core(block);
   ops_dat_init_metadata_core(dat, dim, dataset_size, base, d_m, d_p, stride, data, type_size, type, name);
   return dat;
}

ops_dat ops_decl_dat_temp_core(ops_block block, int dim, int *dataset_size,
                               int *base, int *d_m, int *d_p, int *stride, char *data,
                               int type_size, char const *type,
                               char const *name) {
  return ops_decl_dat_core(block, dim, dataset_size, base, d_m, d_p, stride, data,
                           type_size, type, name);
}

void ops_free_dat_core(ops_dat dat) {
  ops_dat_entry *item;
  TAILQ_FOREACH(item, &dat->block->instance->OPS_dat_list, entries) {
    if (item->dat->index == dat->index) {
      TAILQ_REMOVE(&dat->block->instance->OPS_dat_list, item, entries);
      ops_free(item);
      break;
    }
  }
  TAILQ_FOREACH(item, &(dat->block->instance->OPS_block_list[dat->block->index].datasets), entries) {
    if (item->dat->index == dat->index) {
      TAILQ_REMOVE(&(dat->block->instance->OPS_block_list[dat->block->index].datasets), item, entries);
      ops_free(item);
      break;
    }
  }
  if(dat->user_managed == 0) {
      ops_free(dat->data);
      dat->data = nullptr;
  }
  ops_free((char*)dat->name);
  dat->name = nullptr;
  ops_free((char*)dat->type);
  dat->type = nullptr;
}

ops_stencil _ops_decl_stencil(OPS_instance *instance, int dims, int points, int *sten,
                             char const *name) {
  if (dims <= 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- negative/zero dimension size for stencil: " << name;
      throw ex;
  }
  if (dims > OPS_MAX_DIM) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- too large dimension for stencil: " << name << " please change OPS_MAX_DIM in ops_lib_core.h and recompile OPS.";
      throw ex;
  }

  if (instance->OPS_stencil_index == instance->OPS_stencil_max) {
    instance->OPS_stencil_max += 10;
    instance->OPS_stencil_list = (ops_stencil *)ops_realloc(
        instance->OPS_stencil_list, instance->OPS_stencil_max * sizeof(ops_stencil));

    if (instance->OPS_stencil_list == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_stencil -- error reallocating memory");
    }
  }

  ops_stencil stencil = (ops_stencil)ops_calloc(1, sizeof(ops_stencil_core));
  instance->OPS_stencil_list[instance->OPS_stencil_index] = stencil;
  stencil->index = instance->OPS_stencil_index++;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);

  stencil->stencil = (int *)ops_malloc(dims * points * sizeof(int));
  memcpy(stencil->stencil, sten, sizeof(int) * dims * points);

  stencil->stride = (int *)ops_malloc(dims * sizeof(int));
  stencil->mgrid_stride = 0;
  for (int i = 0; i < dims; i++)
    stencil->stride[i] = 1;

  stencil->type = 0;


  return stencil;
}

ops_stencil ops_decl_stencil(int dims, int points, int *sten,
                             char const *name) {
  return _ops_decl_stencil(OPS_instance::getOPSInstance(), dims, points, sten, name);
}

ops_stencil _ops_decl_strided_stencil(OPS_instance *instance, int dims, int points, int *sten,
                                     int *stride, char const *name) {
  if (dims <= 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- negative/zero dimension size for stencil: " << name;
      throw ex;
  }
  if (dims > OPS_MAX_DIM) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- too large dimension for stencil: " << name << " please change OPS_MAX_DIM in ops_lib_core.h and recompile OPS.";
      throw ex;
  }


  if (instance->OPS_stencil_index == instance->OPS_stencil_max) {
    instance->OPS_stencil_max += 10;
    instance->OPS_stencil_list = (ops_stencil *)ops_realloc(
        instance->OPS_stencil_list, instance->OPS_stencil_max * sizeof(ops_stencil));

    if (instance->OPS_stencil_list == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_stencil -- error reallocating memory");
    }
  }

  ops_stencil stencil = (ops_stencil)ops_calloc(1, sizeof(ops_stencil_core));
  instance->OPS_stencil_list[instance->OPS_stencil_index] = stencil;
  stencil->index = instance->OPS_stencil_index++;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);

  stencil->stencil = (int *)ops_malloc(dims * points * sizeof(int));
  memcpy(stencil->stencil, sten, sizeof(int) * dims * points);

  stencil->stride = (int *)ops_malloc(dims * sizeof(int));
  memcpy(stencil->stride, stride, sizeof(int) * dims);

  stencil->mgrid_stride = (int *)ops_malloc(dims*sizeof(int));
  for (int i = 0; i < dims; i++) stencil->mgrid_stride[i] = 1;


  stencil->type = 0;

  return stencil;
}

ops_stencil ops_decl_strided_stencil(int dims, int points, int *sten,
                                     int *stride, char const *name) {
  return _ops_decl_strided_stencil(OPS_instance::getOPSInstance(), dims, points, sten, stride, name);
}

ops_stencil _ops_decl_restrict_stencil ( OPS_instance *instance, int dims, int points, int *sten, int *stride, char const * name)
{
  if (dims <= 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- negative/zero dimension size for stencil: " << name;
      throw ex;
  }
  if (dims > OPS_MAX_DIM) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- too large dimension for stencil: " << name << " please change OPS_MAX_DIM in ops_lib_core.h and recompile OPS.";
      throw ex;
  }


  if ( instance->OPS_stencil_index == instance->OPS_stencil_max ) {
    instance->OPS_stencil_max += 10;
    instance->OPS_stencil_list = (ops_stencil *) realloc(instance->OPS_stencil_list,instance->OPS_stencil_max * sizeof(ops_stencil));

    if ( instance->OPS_stencil_list == NULL ) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_stencil -- error reallocating memory");
    }
  }

  ops_stencil stencil = (ops_stencil)ops_malloc(sizeof(ops_stencil_core));
  instance->OPS_stencil_list[instance->OPS_stencil_index] = stencil;
  stencil->index = instance->OPS_stencil_index++;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);

  stencil->stencil = (int *)ops_malloc(dims*points*sizeof(int));
  memcpy(stencil->stencil,sten,sizeof(int)*dims*points);

  stencil->stride = (int *)ops_malloc(dims*sizeof(int));
  for (int i = 0; i < dims; i++) stencil->stride[i] = 1;

  stencil->mgrid_stride = (int *)ops_malloc(dims*sizeof(int));
  memcpy(stencil->mgrid_stride,stride,sizeof(int)*dims);

  stencil->type = 2;


  return stencil;
}

ops_stencil ops_decl_restrict_stencil ( int dims, int points, int *sten, int *stride, char const * name) {
  return _ops_decl_restrict_stencil(OPS_instance::getOPSInstance(), dims, points, sten, stride, name);
}

ops_stencil _ops_decl_prolong_stencil ( OPS_instance *instance, int dims, int points, int *sten, int *stride, char const * name)
{
  if (dims <= 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- negative/zero dimension size for stencil: " << name;
      throw ex;
  }
  if (dims > OPS_MAX_DIM) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error:  ops_decl_stencil -- too large dimension for stencil: " << name << " please change OPS_MAX_DIM in ops_lib_core.h and recompile OPS.";
      throw ex;
  }

  if ( instance->OPS_stencil_index == instance->OPS_stencil_max ) {
    instance->OPS_stencil_max += 10;
    instance->OPS_stencil_list = (ops_stencil *) realloc(instance->OPS_stencil_list,instance->OPS_stencil_max * sizeof(ops_stencil));

    if ( instance->OPS_stencil_list == NULL ) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_stencil -- error reallocating memory");
    }
  }

  ops_stencil stencil = (ops_stencil)ops_calloc(1, sizeof(ops_stencil_core));
  instance->OPS_stencil_list[instance->OPS_stencil_index] = stencil;
  stencil->index = instance->OPS_stencil_index++;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);

  stencil->stencil = (int *)ops_malloc(dims*points*sizeof(int));
  memcpy(stencil->stencil,sten,sizeof(int)*dims*points);

  stencil->stride = (int *)ops_malloc(dims*sizeof(int));
  for (int i = 0; i < dims; i++) stencil->stride[i] = 1;

  stencil->mgrid_stride = (int *)ops_malloc(dims*sizeof(int));
  memcpy(stencil->mgrid_stride,stride,sizeof(int)*dims);

  stencil->type = 1;

  return stencil;
}

ops_stencil ops_decl_prolong_stencil ( int dims, int points, int *sten, int *stride, char const * name) {
  return _ops_decl_prolong_stencil(OPS_instance::getOPSInstance(), dims, points, sten, stride, name);
}

#define UINT_MIN 0
#define ULONG_MIN 0
#define ULLONG_MIN 0

#define OPS_RED_INIT(type,token) \
      if (acc == OPS_MIN) \
        for (unsigned int i = 0; i < handle->size / sizeof(type); i++) \
          ((type *)handle->data)[i] = token##_MAX; \
      if (acc == OPS_MAX) \
        for (unsigned int i = 0; i < handle->size / sizeof(type); i++) \
          ((type *)handle->data)[i] = - token##_MAX;

#define OPS_RED_INIT2(type,token) \
      if (acc == OPS_MIN) \
        for (unsigned int i = 0; i < handle->size / sizeof(type); i++) \
          ((type *)handle->data)[i] = token##_MAX; \
      if (acc == OPS_MAX) \
        for (unsigned int i = 0; i < handle->size / sizeof(type); i++) \
          ((type *)handle->data)[i] = token##_MIN;

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
      OPS_RED_INIT(double,DBL)
    } else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0) {
      OPS_RED_INIT(float,FLT)
    } else if (strcmp(type, "int") == 0 || strcmp(type, "integer") == 0) {
      OPS_RED_INIT2(int,INT)
    } else if (strcmp(type, "long") == 0) {
      OPS_RED_INIT2(long,LONG)
    } else if (strcmp(type, "char") == 0) {
      OPS_RED_INIT2(char,CHAR)
    } else if (strcmp(type, "short") == 0) {
      OPS_RED_INIT2(short,SHRT)
    } else if (strcmp(type, "long long") == 0 || strcmp(type, "ll") == 0) {
      OPS_RED_INIT2(long long,LLONG)
    } else if (strcmp(type, "unsigned long long") == 0 || strcmp(type, "ull") == 0) {
      OPS_RED_INIT2(unsigned long long,ULLONG)
    } else if (strcmp(type, "unsigned long") == 0 || strcmp(type, "ul") == 0) {
      OPS_RED_INIT2(unsigned long,ULONG)
    } else if (strcmp(type, "unsigned int") == 0 || strcmp(type, "uint") == 0) {
      OPS_RED_INIT2(unsigned int,UINT)
    }
    else if (strcmp(type, "complexf") == 0) {
      if (acc == OPS_MIN)
        for (int i = 0; i < 2 * handle->size / 4; i++)
          ((complexf *)handle->data)[i] = complexf(FLT_MAX, FLT_MAX);
      if (acc == OPS_MAX)
        for (int i = 0; i < 2 * handle->size / 4; i++)
          ((complexf *)handle->data)[i] = complexf(0,0);
    } else if (strcmp(type, "complexd") == 0) {
      if (acc == OPS_MIN)
        for (int i = 0; i < 2 * handle->size / 8; i++)
          ((complexd *)handle->data)[i] = complexd(DBL_MAX,DBL_MAX);
      if (acc == OPS_MAX)
        for (int i = 0; i < 2 * handle->size / 8; i++)
          ((complexd *)handle->data)[i] = complexd(0,0);
    } else {
      throw OPSException(OPS_NOT_IMPLEMENTED, "Error, reduction type not recognised, please add in ops_lib_core.c");
    }
  } else if (handle->acc != acc) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: ops_reduction handle " << handle->name << " was already used with a different access type";
      throw ex;
  }
  return arg;
}

ops_halo_group _ops_decl_halo_group(OPS_instance *instance, int nhalos, ops_halo halos[]) {
  if (instance->OPS_halo_group_index == instance->OPS_halo_group_max) {
    instance->OPS_halo_group_max += 10;
    instance->OPS_halo_group_list = (ops_halo_group *)ops_realloc(
        instance->OPS_halo_group_list, instance->OPS_halo_group_max * sizeof(ops_halo_group));

    if (instance->OPS_halo_group_list == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_halo_group -- error reallocating memory");
    }
  }

  ops_halo_group grp = (ops_halo_group)ops_calloc(1, sizeof(ops_halo_group_core));
  grp->nhalos = nhalos;

  //make a copy
  ops_halo* halos_temp = (ops_halo *)ops_calloc(1, nhalos*sizeof(ops_halo));
  memcpy(halos_temp, &halos[0], nhalos*sizeof(ops_halo));
  grp->halos = halos_temp;
  grp->instance = instance;

  instance->OPS_halo_group_list[instance->OPS_halo_group_index] = grp;
  grp->index = instance->OPS_halo_group_index++;

  return grp;
}

ops_halo_group ops_decl_halo_group(int nhalos, ops_halo halos[]) {
  return _ops_decl_halo_group(OPS_instance::getOPSInstance(), nhalos, halos);
}


ops_halo ops_decl_halo_core(OPS_instance *instance, ops_dat from, ops_dat to, int *iter_size,
                            int *from_base, int *to_base, int *from_dir,
                            int *to_dir) {
  if (instance->OPS_halo_index == instance->OPS_halo_max) {
    instance->OPS_halo_max += 10;
    instance->OPS_halo_list =
        (ops_halo *)ops_realloc(instance->OPS_halo_list, instance->OPS_halo_max * sizeof(ops_halo));

    if (instance->OPS_halo_list == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_halo_core -- error reallocating memory");
    }
  }

  ops_halo halo = (ops_halo)ops_calloc(1, sizeof(ops_halo_core));
  instance->OPS_halo_list[instance->OPS_halo_index] = halo;
  halo->index = instance->OPS_halo_index++;
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

  return halo;
}


ops_halo _ops_decl_halo(OPS_instance *instance, ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(instance, from, to, iter_size, from_base, to_base, from_dir,
                            to_dir);
}

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(OPS_instance::getOPSInstance(), from, to, iter_size, from_base, to_base, from_dir,
                            to_dir);
}

ops_arg ops_arg_dat_core(ops_dat dat, ops_stencil stencil, ops_access acc) {
  ops_arg arg;
  // In the absence of a constructor ...
  memset(&arg, 0, sizeof(ops_arg));
  arg.argtype = OPS_ARG_DAT;
  arg.dat = dat;
  arg.stencil = stencil;
  if (acc == OPS_WRITE && stencil->points != 1) {
      throw OPSException(OPS_INVALID_ARGUMENT, "Error: OPS does not support OPS_WRITE arguments with a non (0,0,0) stencil due to potential race conditions");
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
  // In the absence of a constructor ...
  memset(&arg, 0, sizeof(ops_arg));
  arg.argtype = OPS_ARG_GBL;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = dim;
  arg.data = data;
  arg.acc = acc;
  arg.elem_size = size;
  return arg;
}

ops_arg ops_arg_idx() {
  ops_arg arg;
  // In the absence of a constructor ...
  memset(&arg, 0, sizeof(ops_arg));
  arg.argtype = OPS_ARG_IDX;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;
}

ops_arg ops_arg_dat(ops_dat dat, int dim, ops_stencil stencil, char const *type,
                    ops_access acc) {
    (void)type;
  // return ops_arg_dat_core( dat, stencil, acc );
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  return temp;
}

ops_arg ops_arg_dat_opt(ops_dat dat, int dim, ops_stencil stencil,
                        char const *type, ops_access acc, int flag) {
    (void)type;(void)dim;
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->opt = flag;
  return temp;
}

ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc) {
  return ops_arg_gbl_core(data, dim, size, acc);
}

ops_reduction ops_decl_reduction_handle_core(OPS_instance *instance, int size, const char *type,
                                             const char *name) {
  if (instance->OPS_reduction_index == instance->OPS_reduction_max) {
    instance->OPS_reduction_max += 10;
    instance->OPS_reduction_list = (ops_reduction *)ops_realloc(
        instance->OPS_reduction_list, instance->OPS_reduction_max * sizeof(ops_reduction));

    if (instance->OPS_reduction_list == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_reduction_handle -- error reallocating memory");
    }
  }

  ops_reduction red = (ops_reduction)ops_calloc(1, sizeof(ops_reduction_core));
  red->initialized = 0;
  red->size = size;
  red->data = (char *)ops_calloc(size, sizeof(char));
  red->name = copy_str(name);
  red->type = copy_str(type);
  red->instance = instance;
  instance->OPS_reduction_list[instance->OPS_reduction_index] = red;
  red->index = instance->OPS_reduction_index++;
  return red;
}

void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr) {
    (void)type_size;
  ops_execute(handle->instance);
  ops_checkpointing_reduction(handle);
  memcpy(ptr, handle->data, handle->size);
  handle->initialized = 0;
}


void _ops_diagnostic_output(OPS_instance *instance) {
  if (instance->OPS_diags > 2) {
    printf2(instance, "\n OPS diagnostic output\n");
    printf2(instance, " --------------------\n");

    printf2(instance, "\n block dimension\n");
    printf2(instance, " -------------------\n");
    for (int n = 0; n < instance->OPS_block_index; n++) {
      printf2(instance, " %15s %15dD ", instance->OPS_block_list[n].block->name,
             instance->OPS_block_list[n].block->dims);
      printf2(instance, "\n");
    }

    // printf ("\n dats item/point [block_size] [base] [d_m] [d_p]
    // memory(MBytes) block\n" );
    printf2(instance, "\n %15s %15s %15s %10s %10s %10s %15s %15s\n", "data", "item/point",
           "[block_size]", "[base]", "[d_m]", "[d_p]", "memory(MBytes)",
           "block");

    printf2(instance, " ------------------------------------------------------------------"
           "-------\n");
    ops_dat_entry *item;
    double tot_memory = 0.0;
    TAILQ_FOREACH(item, &instance->OPS_dat_list, entries) {
      printf2(instance, " %15s %15d ", (item->dat)->name, (item->dat)->dim);
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf2(instance, "[%d]", (item->dat)->size[i]);
      printf2(instance, " ");
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf2(instance, "[%d]", (item->dat)->base[i]);
      printf2(instance, " ");
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf2(instance, "[%d]", (item->dat)->d_m[i]);
      printf2(instance, " ");
      for (int i = 0; i < (item->dat)->block->dims; i++)
        printf2(instance, "[%d]", (item->dat)->d_p[i]);
      printf2(instance, " ");

      printf2(instance, " %15.3lf ", (item->dat)->mem / (1024.0 * 1024.0));
      tot_memory += (item->dat)->mem;
      printf2(instance, " %15s\n", (item->dat)->block->name);
    }
    printf2(instance, "\n");
    printf2(instance, "Total Memory Allocated for ops_dats (GBytes) : %.3lf\n",
           tot_memory / (1024.0 * 1024.0 * 1024.0));
    printf2(instance, "\n");
  }
}

void ops_diagnostic_output() {
  _ops_diagnostic_output(OPS_instance::getOPSInstance());
}

void ops_dump3(ops_dat dat, const char *name) {
  // TODO: this has to be backend-specific
    (void)name;
    (void)dat;
}

bool ops_checkpointing_filename(const char *file_name, std::string &filename_out,
                                std::string &filename_out2);


void ops_print_dat_to_txtfile_core(ops_dat dat, const char* file_name_in)
{
  //printf("file %s, name %s type = %s\n",file_name, dat->name, dat->type);
   std::string file_name, ignored;
  ops_checkpointing_filename(file_name_in, file_name, ignored);
  FILE *fp;
  if (fopen_s(&fp,file_name.c_str(), "a") != 0) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: can't open file " << file_name;
    throw ex;
  }

  if (fprintf(fp, "ops_dat:  %s \n", dat->name) < 0) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: error writing to file " << file_name;
    throw ex;
  }
  if (fprintf(fp, "ops_dat dim:  %d \n", dat->dim) < 0) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: error writing to file " << file_name;
    throw ex;
  }

  if (fprintf(fp, "block Dims : %d ", dat->block->dims) < 0) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: error writing to file " << file_name;
    throw ex;
  }

  for (int i = 0; i < dat->block->dims; i++) {
    if (fprintf(fp, "[%d]", dat->size[i]) < 0) {
      OPSException ex(OPS_RUNTIME_ERROR);
      ex << "Error: error writing to file " << file_name;
      throw ex;
    }
  }
  fprintf(fp, "\n");

  if (fprintf(fp, "elem size %d \n", dat->elem_size) < 0) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: error writing to file " << file_name;
    throw ex;
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

                size_t offset = dat->block->instance->OPS_soa ?
                        (n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i + d * prod[5])
                      :((n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i)*dat->dim + d);
                if (strcmp(dat->type, "double") == 0 || strcmp(dat->type, "real(8)") == 0 ||
                    strcmp(dat->type, "double precision") == 0) {
                  if (fprintf(fp, " %3.10lf", ((double *)dat->data)[offset]) < 0) {
                    OPSException ex(OPS_RUNTIME_ERROR);
                    ex << "Error: error writing to file " << file_name;
                    throw ex;
                  }
                } else if (strcmp(dat->type, "float") == 0 ||
                           strcmp(dat->type, "real") == 0) {
                  if (fprintf(fp, "%e ", ((float *)dat->data)[offset]) < 0) {
                    OPSException ex(OPS_RUNTIME_ERROR);
                    ex << "Error: error writing to file " << file_name;
                    throw ex;
                  }
                } else if (strcmp(dat->type, "int") == 0 ||
                           strcmp(dat->type, "integer") == 0 ||
                           strcmp(dat->type, "integer(4)") == 0 ||
                           strcmp(dat->type, "int(4)") == 0 /* ||
                           strcmp(dat->type, "long") == 0 ||
                           strcmp(dat->type, "long long") == 0 ||
                           strcmp(dat->type, "ll") == 0 ||
                           strcmp(dat->type, "short") == 0 ||
                           strcmp(dat->type, "char") == 0*/) {
                  if (fprintf(fp, "%d ", ((int *)dat->data)[offset]) < 0) {
                    OPSException ex(OPS_RUNTIME_ERROR);
                    ex << "Error: error writing to file " << file_name;
                    throw ex;
                  }
                } else if (strcmp(dat->type, "complexf") == 0) {
                  if (fprintf(fp, "%e+%ei ", ((float *)dat->data)[2*offset],((float *)dat->data)[2*offset+1]) < 0) {
                    OPSException ex(OPS_RUNTIME_ERROR);
                    ex << "Error: error writing to file " << file_name;
                    throw ex;
                  }
                } else if (strcmp(dat->type, "complexd") == 0) {
                  if (fprintf(fp, "%3.10lf+%3.10lfi ", ((double *)dat->data)[2*offset],((double *)dat->data)[2*offset+1]) < 0) {
                    OPSException ex(OPS_RUNTIME_ERROR);
                    ex << "Error: error writing to file " << file_name;
                    throw ex;
                  }
                } else {
                    OPSException ex(OPS_NOT_IMPLEMENTED);
                    ex << "Error: Unknown type " << dat->type << " cannot be written to file " << file_name;
                    throw ex;
                }
              } //d
            } //i
#if OPS_MAX_DIM > 0
            if (dat->size[0] > 1) fprintf(fp, "\n");
#endif
          }//j
#if OPS_MAX_DIM > 1
          if (dat->size[1] > 1) fprintf(fp, "\n");
#endif
        }//k
#if OPS_MAX_DIM > 2
        if (dat->size[2] > 1) fprintf(fp, "\n");
#endif
      }//l
#if OPS_MAX_DIM > 3
      if (dat->size[3] > 1) fprintf(fp, "\n");
#endif
    }//m
#if OPS_MAX_DIM > 4
    if (dat->size[4] > 1) fprintf(fp, "\n");
#endif
  }//n
#if OPS_MAX_DIM > 5
  if (dat->size[5] > 1) fprintf(fp, "\n");
#endif

  fclose(fp);
}

void ops_timing_output_stdout() { ops_timing_output(std::cout); }

void _ops_timing_output(OPS_instance *instance, std::ostream &stream) {

  if (instance->OPS_diags > 1)
    if (instance->OPS_enable_checkpointing)
      ops_fprintf2(stream, "\nTotal time spent in checkpointing: %g seconds\n",
                 instance->OPS_checkpointing_time);
  if (instance->OPS_diags > 1 && instance->OPS_kernels != NULL) {
    size_t maxlen = 0;
    for (int i = -1; i < instance->OPS_kern_max; i++) {
      if (instance->OPS_kernels[i].count > 0)
        maxlen = MAX(maxlen, strlen(instance->OPS_kernels[i].name));
      if (instance->OPS_kernels[i].count > 0 && strlen(instance->OPS_kernels[i].name) > 50) {
        ops_printf2(instance, "Too long\n");
      }
    }
    char *buf = (char *)ops_malloc((maxlen + 180) * sizeof(char));
    char buf2[180];
    snprintf(buf, 5, "Name");
    for (unsigned int i = 4; i < maxlen; i++)
      strncat_s(buf, (maxlen + 180), " ", 1);
    ops_fprintf2(stream, "\n\n%s  Count Time     MPI-time     Bandwidth(GB/s)\n",
                buf);

    for (unsigned int i = 0; i < maxlen + 31; i++)
      strncat_s(buf, (maxlen + 180), "-", 1);
    ops_fprintf2(stream, "%s\n", buf);
    double sumtime = 0.0f;
    double sumtime_mpi = 0.0f;
    for (int k = -1; k < instance->OPS_kern_max; k++) {
      double moments_mpi_time[2] = {0.0};
      double moments_time[2] = {0.0};
      ops_compute_moment(instance->OPS_kernels[k].time, &moments_time[0],
                         &moments_time[1]);
      ops_compute_moment(instance->OPS_kernels[k].mpi_time, &moments_mpi_time[0],
                         &moments_mpi_time[1]);

      if (instance->OPS_kernels[k].count < 1)
        continue;
      snprintf(buf, (maxlen + 180), "%s", instance->OPS_kernels[k].name);
      for (size_t i = strlen(instance->OPS_kernels[k].name); i < maxlen + 2; i++)
        strncat_s(buf, (maxlen + 180), " ",1);

      snprintf(
          buf2, 180, "%-5d %-6f (%-6f) %-6f (%-6f)  %-13.2f", instance->OPS_kernels[k].count,
          moments_time[0],
          sqrt(moments_time[1] - moments_time[0] * moments_time[0]),
          moments_mpi_time[0],
          sqrt(moments_mpi_time[1] - moments_mpi_time[0] * moments_mpi_time[0]),
          instance->OPS_kernels[k].transfer / ((moments_time[0]) * 1024 * 1024 * 1024));

      // sprintf(buf2,"%-5d %-6f  %-6f  %-13.2f", instance->OPS_kernels[k].count,
      // instance->OPS_kernels[k].time,
      //  instance->OPS_kernels[k].mpi_time,
      //  instance->OPS_kernels[k].transfer/instance->OPS_kernels[k].time/1000/1000/1000);
      ops_fprintf2(stream, "%s%s\n", buf, buf2);
      sumtime += moments_time[0];
      sumtime_mpi += moments_mpi_time[0];
    }
    ops_fprintf2(stream, "Total kernel time: %g\n", sumtime);
    if (instance->ops_tiled_halo_exchange_time > 0.0) {
      double moments_time[2] = {0.0};
      ops_compute_moment(instance->ops_tiled_halo_exchange_time, &moments_time[0], &moments_time[1]);
      ops_fprintf2(stream, "Total tiled halo exchange time: %g\n", moments_time[0]);
    } else if (sumtime_mpi > 0) {
      ops_fprintf2(stream, "Total halo exchange time: %g\n", sumtime_mpi);
    }
    // printf("Times: %g %g %g\n",ops_gather_time, ops_sendrecv_time,
    // ops_scatter_time);
    ops_free(buf);
  }
}

void ops_timing_output(std::ostream &stream) {
  _ops_timing_output(OPS_instance::getOPSInstance(), stream);
}

void ops_timers_core(double *cpu, double *et) {
#if __cplusplus>=201103L
  (void)cpu;
  *et = (double)std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count()/1000000.0;
#else
#ifdef __unix__
  (void)cpu;
  struct timeval t;

  gettimeofday(&t, (struct timezone *)0);
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
#elif defined(_WIN32) || defined(WIN32)
  (void)cpu;
  DWORD time = GetTickCount();
  *et = ((double)time)/1000.0;
#endif
#endif
}

void ops_timing_realloc(OPS_instance *instance, int kernel, const char *name) {
  int OPS_kern_max_new;
  instance->OPS_kern_curr = kernel;

  if (kernel >= instance->OPS_kern_max || instance->OPS_kern_max == 0) {
    OPS_kern_max_new = MAX(0,kernel) + 10;
    if (instance->OPS_kern_max == 0) {
      instance->OPS_kernels = ((ops_kernel *)ops_calloc(
        OPS_kern_max_new+1, sizeof(ops_kernel)))+1;
      instance->OPS_kernels[-1].count = -1;
    } else
      instance->OPS_kernels = ((ops_kernel *)ops_realloc(
        instance->OPS_kernels-1, (OPS_kern_max_new+1) * sizeof(ops_kernel)))+1;
    if (instance->OPS_kernels == NULL) {
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_timing_realloc -- error reallocating memory");
    }

    for (int n = instance->OPS_kern_max; n < OPS_kern_max_new; n++) {
      instance->OPS_kernels[n].count = -1;
      instance->OPS_kernels[n].name = NULL;
      instance->OPS_kernels[n].time = 0.0f;
      instance->OPS_kernels[n].transfer = 0.0f;
      instance->OPS_kernels[n].mpi_time = 0.0f;
    }
    instance->OPS_kern_max = OPS_kern_max_new;
  }

  if (instance->OPS_kernels[kernel].count == -1) {
    instance->OPS_kernels[kernel].name = (char *)ops_malloc((strlen(name) + 1) * sizeof(char));
    strcpy_s(instance->OPS_kernels[kernel].name, strlen(name)+1, name);
    instance->OPS_kernels[kernel].count = 0;
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

extern "C" void ops_compute_transfer_f(int dims, int *start, int *end, ops_arg *arg,
                            float *value) {
  *value = ops_compute_transfer(dims, start, end, arg);
}

void ops_register_args(OPS_instance *instance, ops_arg *args, const char *name) {
  if (instance != OPS_instance::getOPSInstance()) throw OPSException(OPS_RUNTIME_ERROR, "OPS_DEBUG mode only supported with a single OPS instance!");
  instance->OPS_curr_args = args;
  instance->OPS_curr_name = name;
}

int ops_stencil_check_1d(int arg_idx, int idx0, int dim0) {
  (void)dim0;
  if (OPS_instance::getOPSInstance()->OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[1 * i] == idx0) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: stencil point (" << idx0 << ") not found in declaration " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->name
         << " in loop " << OPS_instance::getOPSInstance()->OPS_curr_name << " arg " << arg_idx << " : " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].dat->name;
      throw ex;
    }
  }
  return idx0;
}

int ops_stencil_check_1d_md(int arg_idx, int idx0, int mult_d, int d) {
  if (OPS_instance::getOPSInstance()->OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[1 * i] == idx0) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: stencil point (" << idx0 << ") not found in declaration " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->name
         << " in loop " << OPS_instance::getOPSInstance()->OPS_curr_name << " arg " << arg_idx << " : " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].dat->name;
      throw ex;
    }
  }
  return idx0 * mult_d + d;
}

int ops_stencil_check_2d(int arg_idx, int idx0, int idx1, int dim0, int dim1) {
  (void)dim1;
  if (OPS_instance::getOPSInstance()->OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[2 * i] == idx0 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[2 * i + 1] == idx1) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: stencil point (" << idx0 << ", "<< idx1 << ") not found in declaration " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->name
         << " in loop " << OPS_instance::getOPSInstance()->OPS_curr_name << " arg " << arg_idx;
      throw ex;
    }
  }
  return idx0 + dim0 * (idx1);
}
int ops_stencil_check_3d(int arg_idx, int idx0, int idx1, int idx2, int dim0,
                         int dim1) {
  if (OPS_instance::getOPSInstance()->OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[3 * i] == idx0 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[3 * i + 1] == idx1 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[3 * i + 2] == idx2) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: stencil point (" << idx0 << ", "<< idx1 << ", " << idx2 << ") not found in declaration " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->name
         << " in loop " << OPS_instance::getOPSInstance()->OPS_curr_name << " arg " << arg_idx;
      throw ex;
    }
  }
  return idx0 + dim0 * (idx1) + dim0 * dim1 * (idx2);
}

int ops_stencil_check_4d(int arg_idx, int idx0, int idx1, int idx2, int idx3, int dim0,
                         int dim1, int dim2) {
  if (OPS_instance::getOPSInstance()->OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[4 * i] == idx0 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[4 * i + 1] == idx1 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[4 * i + 2] == idx2 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[4 * i + 3] == idx3) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: stencil point (" << idx0 << ", "<< idx1 << ", " << idx2 << ", " << idx3 << ") not found in declaration " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->name
         << " in loop " << OPS_instance::getOPSInstance()->OPS_curr_name << " arg " << arg_idx;
      throw ex;
    }
  }
  return idx0 + dim0 * (idx1) + dim0 * dim1 * (idx2) + dim0 * dim1 * dim2 * idx3;
}

int ops_stencil_check_5d(int arg_idx, int idx0, int idx1, int idx2, int idx3, int idx4, int dim0,
                         int dim1, int dim2, int dim3) {
  if (OPS_instance::getOPSInstance()->OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[5 * i] == idx0 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[5 * i + 1] == idx1 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[5 * i + 2] == idx2 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[5 * i + 3] == idx3 &&
          OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->stencil[5 * i + 4] == idx4) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      OPSException ex(OPS_INVALID_ARGUMENT);
      ex << "Error: stencil point (" << idx0 << ", "<< idx1 << ", " << idx2 << ", " << idx3 << ", " << idx4 << ") not found in declaration " << OPS_instance::getOPSInstance()->OPS_curr_args[arg_idx].stencil->name
         << " in loop " << OPS_instance::getOPSInstance()->OPS_curr_name << " arg " << arg_idx;
      throw ex;
    }
  }
  return idx0 + dim0 * (idx1) + dim0 * dim1 * (idx2) + dim0 * dim1 * dim2 * idx3 + dim0 * dim1 * dim2 * dim3 * idx4;
}

void ops_NaNcheck_core(ops_dat dat, char *buffer) {

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

                size_t offset = dat->block->instance->OPS_soa ?
                        (n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i + d * prod[5])
                      :((n * prod[4] + m * prod[3] + l * prod[2] + k * prod[1] + j * prod[0] + i)*dat->dim + d);
                if (strcmp(dat->type, "double") == 0 || strcmp(dat->type, "real(8)") == 0 ||
                    strcmp(dat->type, "double precision") == 0) {
                  if (  std::isnan(((double *)dat->data)[offset])  ) {
                    printf("%sError: NaN detected at element %zu\n", buffer, offset);
                    exit(2);
                  }
                } else if (strcmp(dat->type, "float") == 0 ||
                           strcmp(dat->type, "real") == 0) {
                  if (  std::isnan(((float *)dat->data)[offset])  ) {
                    printf("%sError: NaN detected at element %zu\n", buffer, offset);
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
extern "C" ops_halo ops_decl_halo_convert(ops_dat from, ops_dat to, int *iter_size,
                               int *from_base, int *to_base, int *from_dir,
                               int *to_dir) {

  for (int i = 0; i < from->block->dims; i++) {
    from_base[i]--;
    to_base[i]--;
  }

  ops_halo temp = ops_decl_halo_core(from->block->instance, from, to, iter_size, from_base, to_base,
                                     from_dir, to_dir);

  for (int i = 0; i < from->block->dims; i++) {
    from_base[i]++;
    to_base[i]++;
  }

  return temp;
}

extern "C" void setKernelTime(int id, char name[], double kernelTime, double mpiTime,
                   float transfer, int count) {
  ops_timing_realloc(OPS_instance::getOPSInstance(),id, name);

  OPS_instance::getOPSInstance()->OPS_kernels[id].count += count;
  OPS_instance::getOPSInstance()->OPS_kernels[id].time += (float)kernelTime;
  OPS_instance::getOPSInstance()->OPS_kernels[id].mpi_time += (float)mpiTime;
  OPS_instance::getOPSInstance()->OPS_kernels[id].transfer += transfer;
}

extern "C" ops_halo_group ops_decl_halo_group_elem(int nhalos, ops_halo *halos,
                                        ops_halo_group grp) {

  OPS_instance *instance = OPS_instance::getOPSInstance();
  if (instance->OPS_halo_group_index == instance->OPS_halo_group_max) {
    instance->OPS_halo_group_max += 10;
    instance->OPS_halo_group_list = (ops_halo_group *)ops_realloc(
        instance->OPS_halo_group_list, instance->OPS_halo_group_max * sizeof(ops_halo_group));

    if (instance->OPS_halo_group_list == NULL) {
      printf( "Error, ops_decl_halo_group -- error reallocating memory");
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
    grp = (ops_halo_group)ops_calloc(1, sizeof(ops_halo_group_core));
    grp->nhalos = 0;
    if (nhalos != 0) {
      ops_halo *halos_temp = (ops_halo *)ops_calloc(1 , sizeof(ops_halo_core));
      memcpy(halos_temp, halos, 1 * sizeof(ops_halo_core));
      grp->halos = halos_temp;
      grp->nhalos++;
    }
    grp->instance = instance;
    instance->OPS_halo_group_list[instance->OPS_halo_group_index] = grp;
    grp->index = instance->OPS_halo_group_index++;
  } else {
    grp->halos = (ops_halo *)ops_realloc(grp->halos, (grp->nhalos + 1) *
                                                         sizeof(ops_halo_core));
    memcpy(&grp->halos[grp->nhalos], &halos[0], 1 * sizeof(ops_halo_core));
    grp->nhalos++;
  }
  return grp;
}


void *ops_malloc(size_t size) {
  void *ptr = NULL;
  if( posix_memalign((void**)&(ptr), OPS_ALIGNMENT, size) ) {
      OPSException ex(OPS_INTERNAL_ERROR);
      ex << "Error, posix_memalign() returned an error.";
      throw ex;
  }
  return ptr;
}

void *ops_calloc(size_t num, size_t size) {
//#ifdef __INTEL_COMPILER
  // void * ptr = _mm_malloc(num*size, OPS_ALIGNMENT);
  void *ptr=NULL;
  if( posix_memalign((void**)&(ptr), OPS_ALIGNMENT, num*size) ) {
      OPSException ex(OPS_INTERNAL_ERROR);
      ex << "Error, posix_memalign() returned an error.";
      throw ex;
  }
  memset(ptr, 0, num * size);
  return ptr;
//#else
//  return xcalloc(num, size);
//#endif
}

void *ops_realloc(void *ptr, size_t size) {
//#ifdef __INTEL_COMPILER
#if defined (_WIN32) || defined(WIN32)
  void* newptr = _aligned_realloc(ptr, size, OPS_ALIGNMENT);
#else
  void *newptr = realloc(ptr, size);
#endif
  static_assert(sizeof(size_t) == sizeof(void*), "size_t is not big enough to hold pointer address");
  if (((size_t)newptr & (OPS_ALIGNMENT - 1)) != 0) {
    void *newptr2=NULL;
    if( posix_memalign((void**)&(newptr2), OPS_ALIGNMENT, size) ) {
        OPSException ex(OPS_INTERNAL_ERROR);
        ex << "Error, posix_memalign() returned an error.";
        throw ex;
    }
    // void *newptr2 = _mm_malloc(size, OPS_ALIGNMENT);
    memcpy(newptr2, newptr, size);
    ops_free(newptr);
    return newptr2;
  } else {
    return newptr;
  }
//#else
//  return xrealloc(ptr, size);
//#endif
}

void ops_free(void *ptr) {
#if defined (_WIN32) || defined(WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void ops_internal_copy_seq(ops_kernel_descriptor *desc) {
    ops_dat dat0 = desc->args[0].dat;
    double __t1=0,__t2=0,__c1,__c2;
    if (dat0->block->instance->OPS_diags>1) {
        dat0->block->instance->OPS_kernels[-1].count++;
        ops_timers_core(&__c1,&__t1);
    }
    int range[2*OPS_MAX_DIM]={0};
    long fullsize = 1;
    for (int d = 0; d < desc->dim; d++) {
        range[2*d] = desc->range[2*d];
        range[2*d+1] = desc->range[2*d+1];
        fullsize *= dat0->size[d];
    }
    for (int d = desc->dim; d < OPS_MAX_DIM; d++) {
        range[2*d] = 0;
        range[2*d+1] = 1;
        fullsize *= dat0->size[d];
    }
    char *dat0_p = desc->args[0].data + desc->args[0].dat->base_offset;
    char *dat1_p = desc->args[1].data + desc->args[1].dat->base_offset;
    int mult = dat0->block->instance->OPS_soa ? dat0->type_size : dat0->dim*dat0->type_size;


#if OPS_MAX_DIM>4
#pragma omp parallel for OMP_COLLAPSE(5)
    for (long m = (long)range[2*4]; m < (long)range[2*4+1]; m++)
#else
        long m = 0;
#endif
#if OPS_MAX_DIM>3
#if OPS_MAX_DIM==4
#pragma omp parallel for OMP_COLLAPSE(4)
#endif
    for (long l = (long)range[2*3]; l < (long)range[2*3+1]; l++)
#else
        long l = 0;
#endif
#if OPS_MAX_DIM>2
#if OPS_MAX_DIM==3
#pragma omp parallel for OMP_COLLAPSE(3)
#endif
    for (long k = (long)range[2*2]; k < (long)range[2*2+1]; k++)
#else
        long k = 0;
#endif
#if OPS_MAX_DIM>1
#if OPS_MAX_DIM==2
#pragma omp parallel for OMP_COLLAPSE(2)
#endif
    for (long j = (long)range[2*1]; j < (long)range[2*1+1]; j++)
#else
        long j = 0;
#endif
#if OPS_MAX_DIM==1
#pragma omp parallel for
#endif
    for (long i = (long)range[2*0]; i < (long)range[2*0+1]; i++) {
        long idx = i*mult;
#if OPS_MAX_DIM>1
        idx += j * dat0->size[0] * mult;
#endif
#if OPS_MAX_DIM>2
        idx += k * dat0->size[0] * dat0->size[1] * mult;
#endif
#if OPS_MAX_DIM>3
        idx += l * dat0->size[0] * dat0->size[1] * dat0->size[2] * mult;
#endif
#if OPS_MAX_DIM>3
        idx += m * dat0->size[0] * dat0->size[1] * dat0->size[2] * dat0->size[3] * mult;
#endif
        if (dat0->block->instance->OPS_soa) {
            for (int d = 0; d < dat0->dim; d++) {
                for (int c = 0; c < dat0->type_size; d++) {
                    dat1_p[idx+d*fullsize*dat0->type_size+c] = dat0_p[idx+d*fullsize*dat0->type_size+c];
                }
            }
        }
        else {
            for (int d = 0; d < dat0->dim*dat0->type_size; d++) {
                dat1_p[idx+d] = dat0_p[idx+d];
            }
        }

    }
    if (dat0->block->instance->OPS_diags>1) {
        ops_timers_core(&__c2,&__t2);
        int start[OPS_MAX_DIM];
        int end[OPS_MAX_DIM];
        for ( int n=0; n<desc->dim; n++ ) {
            start[n] = range[2*n];end[n] = range[2*n+1];
        }
        dat0->block->instance->OPS_kernels[-1].time += __t2-__t1;
        dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[0]);
        dat0->block->instance->OPS_kernels[-1].transfer += ops_compute_transfer(desc->dim, start, end, &desc->args[1]);
    }
}

ops_kernel_descriptor * ops_dat_deep_copy_core(ops_dat target, ops_dat orig_dat, int *range) 
{
    ops_kernel_descriptor *desc =
        (ops_kernel_descriptor *)ops_calloc(1, sizeof(ops_kernel_descriptor));
    //  desc->name = "ops_internal_copy_seq";
    desc->block = orig_dat->block;
    desc->dim = orig_dat->block->dims;
    desc->device = 0;
    desc->index = -1;
    ops_timing_realloc(orig_dat->block->instance, -1, "ops_dat_deep_copy");
    desc->hash = 5381;
    desc->hash = ((desc->hash << 5) + desc->hash) + 1;
    for (int i = 0; i < 2*OPS_MAX_DIM; i++) {
        desc->range[i] = range[i];
        desc->orig_range[i] = range[i];
        desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
    }
    desc->nargs = 2;
    desc->args = (ops_arg *)ops_malloc(2 * sizeof(ops_arg));
    desc->args[0] = ops_arg_dat(orig_dat, orig_dat->dim, desc->block->instance->OPS_internal_0[orig_dat->block->dims -1], orig_dat->type, OPS_READ);
    desc->hash = ((desc->hash << 5) + desc->hash) + desc->args[0].dat->index;
    desc->args[1] = ops_arg_dat(target, orig_dat->dim, desc->block->instance->OPS_internal_0[orig_dat->block->dims -1], orig_dat->type, OPS_WRITE);
    desc->hash = ((desc->hash << 5) + desc->hash) + desc->args[1].dat->index;
    return desc;
}


/**
 * This copies metadata from orig_dat to target_dat.  Metadata is defined as everything that
 * is not the actual data.  If target_dat has different size than orig_dat, this method will
 * deallocate target_dat.mem and reallocate them to the correct size.  This function will also
 * change the name of target_dat to orig_dat->name + "_copy"
 *
 * @return 1 if target.mem was reallocated, 0 otherwise
 *
 * Constraint : target_dat and orig_dat have to be defined on the same ops_block
 *            : ops_dat can't be an HDF5 file
 */
int ops_dat_copy_metadata_core(ops_dat target, ops_dat orig_dat) 
{
   // I'm assuming this is a sufficient test to determine whether two blocks are equal
  if( target->block->index != orig_dat->block->index ) {
     // Blocks don't match, throw an exception
     OPSException ex(OPS_INVALID_ARGUMENT);
     ex << "Error: ops_dat_copy_metadata_core -- the target and source blocks are not equal.  Cannot copy ops_dats "
        "between two different blocks";
     throw ex;
  }
  if( orig_dat->is_hdf5 ) {
     OPSException ex(OPS_INVALID_ARGUMENT);
     ex << "Error: ops_dat_copy_metadata_core -- the target ops_dat is an HDF5 file.  HDF5 files can't be copied";
     throw ex;
  }

  // Check if source and target memory buffers are identical
  int realloc = 0;
  if( target->mem != orig_dat->mem ) 
  {
     // We need to reallocate
     realloc = 1;
     ops_free(target->data);
     target->data = nullptr;
     target->data = (char*)ops_malloc(orig_dat->mem);
     target->mem = orig_dat->mem;
  }


  // Compute the original, unadjusted size of the dat (i.e. subtract padding)
  int dat_size[OPS_MAX_DIM];
  for (int i = 0; i < orig_dat->block->dims; i++) {
    dat_size[i] = orig_dat->size[i] + orig_dat->d_m[i] - orig_dat->d_p[i];
  }
  char *name2 = (char*)ops_malloc(strlen(orig_dat->name)+5+1);
  snprintf(name2, strlen(orig_dat->name) + 5, "%s_copy", orig_dat->name);

  ops_dat_init_metadata_core(target, orig_dat->dim, dat_size,
                                      orig_dat->base, orig_dat->d_m, orig_dat->d_p,
                                      orig_dat->stride, 
                                      target->data,    // buffer to write
                                      orig_dat->type_size,
                                      orig_dat->type, name2);
  ops_free(name2);

  // Following data on target is uninitialised:
  //   dat->data_d      -  backend specialization will take care
  //   dat->base_offset 
  //   dat->dirty_hd
  //   dat->locked_hd
  target->user_managed = 0;
  target->is_hdf5=0;
  target->hdf5_file = "none";
  target->base_offset = orig_dat->base_offset;
  target->dirty_hd = orig_dat->dirty_hd;
  // Is this correct??
  target->locked_hd = orig_dat->locked_hd;
  return realloc;
}

void ops_cpHostToDevice(OPS_instance *instance, void **data_d, void **data_h, size_t size) {
  if ( *data_d == NULL ) {
    ops_device_malloc(instance, data_d, size);
  }
  if (data_h == NULL || *data_h == NULL) {
    ops_device_memset(instance, data_d, 0, size);
    return;
  }
  ops_device_memcpy_h2d(instance, data_d, data_h, size);
}


void ops_H_D_exchanges_host(ops_arg *args, int nargs) {
  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OPS_ARG_DAT &&
        args[n].dat->locked_hd > 0) {
      OPSException ex(OPS_RUNTIME_ERROR, "ERROR: ops_par_loops involving datasets for which raw pointers have not been released are not allowed");
      throw ex;
    }
    if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 2) {
      ops_get_data(args[n].dat);
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
      ops_put_data(args[n].dat);
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

void ops_get_data(ops_dat dat) {
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  ops_device_memcpy_d2h(dat->block->instance, (void**)&dat->data, (void**)&dat->data_d, bytes);
  ops_device_sync(dat->block->instance);
}

//
// routine to upload data from CPU to GPU (with transposing SoA to AoS if needed)
//

void ops_put_data(ops_dat dat) {
  if (dat->dirty_hd == 1)
    dat->dirty_hd = 0;
  else
    return;
  size_t bytes = dat->elem_size;
  for (int i = 0; i < dat->block->dims; i++)
    bytes = bytes * dat->size[i];
  ops_device_memcpy_h2d(dat->block->instance, (void**)&dat->data_d, (void**)&dat->data, bytes);
  ops_device_sync(dat->block->instance);
}


//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(OPS_instance *instance, int consts_bytes) {
  if (consts_bytes > instance->OPS_consts_bytes) {
    if (instance->OPS_consts_bytes > 0) {
      ops_free(instance->OPS_consts_h);
      ops_device_freehost(instance, (void**)&instance->OPS_gbl_prev);
      ops_device_free(instance, (void**)&instance->OPS_consts_d);
    }
    instance->OPS_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
    ops_device_mallochost(instance, (void **)&instance->OPS_gbl_prev, instance->OPS_consts_bytes);
    instance->OPS_consts_h = (char *)ops_malloc(instance->OPS_consts_bytes);
    ops_device_malloc(instance, (void **)&instance->OPS_consts_d, instance->OPS_consts_bytes);
  }
}

void reallocReductArrays(OPS_instance *instance, int reduct_bytes) {
  if (reduct_bytes > instance->OPS_reduct_bytes) {
    if (instance->OPS_reduct_bytes > 0) {
      ops_free(instance->OPS_reduct_h);
      ops_device_free(instance, (void**)&instance->OPS_reduct_d);
    }
    instance->OPS_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
    instance->OPS_reduct_h = (char *)ops_malloc(instance->OPS_reduct_bytes);
    ops_device_malloc(instance, (void **)&instance->OPS_reduct_d, instance->OPS_reduct_bytes);
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
    ops_device_memcpy_h2d(instance, (void**)&instance->OPS_consts_d, (void**)&instance->OPS_consts_h, consts_bytes);
    memcpy(instance->OPS_gbl_prev, instance->OPS_consts_h, consts_bytes);
  }
}

void mvReductArraysToDevice(OPS_instance *instance, int reduct_bytes) {
  ops_device_memcpy_h2d(instance, (void**)&instance->OPS_reduct_d, (void**)&instance->OPS_reduct_h, reduct_bytes);
}

void mvReductArraysToHost(OPS_instance *instance, int reduct_bytes) {
  ops_device_memcpy_d2h(instance, (void**)&instance->OPS_reduct_h, (void**)&instance->OPS_reduct_d, reduct_bytes);
}

void ops_upload_gbls(ops_arg* args, int nargs) {
    int consts_bytes = 0;
    ops_block block = NULL;
    for (int n = 0; n < nargs; n++) {
      if (args[n].argtype == OPS_ARG_GBL && args[n].acc == OPS_READ)
        consts_bytes += ROUND_UP(args[n].dim * args[n].elem_size);
      if (args[n].argtype == OPS_ARG_DAT)
        block = args[n].dat->block;
    }
    if (consts_bytes) {
      reallocConstArrays(block->instance,consts_bytes);
      consts_bytes = 0;
      for (int n = 0; n < nargs; n++) {
        if (args[n].argtype == OPS_ARG_GBL && args[n].acc == OPS_READ) {
          for (int i = 0; i < args[n].dim*args[n].elem_size; i++)
            (block->instance->OPS_consts_h + consts_bytes)[i] = args[n].data[i];
          args[n].data_d = block->instance->OPS_consts_d + consts_bytes;
          consts_bytes += ROUND_UP(args[n].dim * args[n].elem_size);
        }
      }
      mvConstArraysToDevice(block->instance,consts_bytes);
    }
}

void ops_free_dat(ops_dat dat) {
  delete dat;
}

void _ops_free_dat(ops_dat dat) {
  if (dat->data_d) ops_device_free(dat->block->instance, (void**)&dat->data_d);
  ops_free_dat_core(dat);
}


/************* Functions only use in the Fortran Backend ************/

extern "C" int getOPS_block_size_x() { return OPS_instance::getOPSInstance()->OPS_block_size_x; }
extern "C" int getOPS_block_size_y() { return OPS_instance::getOPSInstance()->OPS_block_size_y; }
extern "C" int getOPS_block_size_z() { return OPS_instance::getOPSInstance()->OPS_block_size_z; }
