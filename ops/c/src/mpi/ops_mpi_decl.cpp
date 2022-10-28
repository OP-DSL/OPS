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
  * @brief OPS mpi declaration
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the OPS API calls for the mpi+X backend
  */

#include <mpi.h>
#include <ops_mpi_core.h>

extern char *halo_buffer_d;
extern char *ops_buffer_send_1;
extern char *ops_buffer_recv_1;
extern char *ops_buffer_send_2;
extern char *ops_buffer_recv_2;

// The one and only non-threads safe global OPS_instance
// Declared in ops_instance.cpp
extern OPS_instance *global_ops_instance;


void _ops_init(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  //We do not have thread safety across MPI
  if ( OPS_instance::numInstances() != 1 ) {
    OPSException ex(OPS_RUNTIME_ERROR, "ERROR: multiple OPS instances are not supported over MPI");
    throw ex;
  }
  void *v;
  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init((int *)(&argc), (char ***)&argv);
  }

  //Splitting up the communication world for MPMD apps
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &v, &flag);

  if (!flag) {
	  MPI_Comm_dup(MPI_COMM_WORLD, &OPS_MPI_GLOBAL);
  }
  else {
	  int appnum = * (int *) v;
	  int rank;

	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	  MPI_Comm_split(MPI_COMM_WORLD,appnum, rank, &OPS_MPI_GLOBAL);
  }

  MPI_Comm_rank(OPS_MPI_GLOBAL, &ops_my_global_rank);
  MPI_Comm_size(OPS_MPI_GLOBAL, &ops_comm_global_size);

  // So the MPI backend is not thread safe - that's fine.  It currently does not pass 
  // OPS_instance pointers around, but rather uses the global instance exclusively.
  // The only feasible use-case for MPI is that there is one OPS_instance.  Ideally that
  // would not be created via thread-safe API, but we kinda want to support that.  So we 
  // drill a back door into our own safety system: provided there is only one OPS_instance,
  // we assign that to the global non-thread safe var and carry on.
  global_ops_instance = instance;

  ops_init_core(instance, argc, argv, diags);
  ops_init_device(instance, argc, argv, diags);
}

void ops_init(const int argc, const char *const argv[], const int diags) {
  _ops_init(OPS_instance::getOPSInstance(), argc, argv, diags);
}

void _ops_exit(OPS_instance *instance) {
  if (instance->is_initialised == 0) return;
  if (instance->ops_halo_buffer!=NULL) ops_free(instance->ops_halo_buffer);
  if (instance->OPS_consts_bytes > 0) {
    ops_free(instance->OPS_consts_h);
    if (instance->OPS_gbl_prev!=NULL) ops_device_freehost(instance, (void**)&instance->OPS_gbl_prev);
    if (instance->OPS_consts_d!=NULL) ops_device_free(instance, (void**)&instance->OPS_consts_d);
  }
  if (instance->OPS_reduct_bytes > 0) {
    ops_free(instance->OPS_reduct_h);
    if (instance->OPS_reduct_d!=NULL) ops_device_free(instance, (void**)&instance->OPS_reduct_d);
  }
  
  ops_mpi_exit(instance);

  if (halo_buffer_d != NULL)
    ops_device_free(instance, (void**)&halo_buffer_d);
  if (instance->OPS_gpu_direct && ops_buffer_send_1 != NULL) {
    ops_device_free(instance, (void**)&ops_buffer_send_1);
    ops_device_free(instance, (void**)&ops_buffer_recv_1);
    ops_device_free(instance, (void**)&ops_buffer_send_2);
    ops_device_free(instance, (void**)&ops_buffer_recv_2);
  } else if (ops_buffer_send_1 != NULL)
  {
    ops_device_freehost(instance, (void**)&ops_buffer_send_1);
    ops_device_freehost(instance, (void**)&ops_buffer_recv_1);
    ops_device_freehost(instance, (void**)&ops_buffer_send_2);
    ops_device_freehost(instance, (void**)&ops_buffer_recv_2);
  }
  int flag = 0;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
  ops_exit_device(instance);
  ops_exit_core(instance);
}

void ops_exit() {
  _ops_exit(OPS_instance::getOPSInstance());
}

ops_dat ops_decl_dat_char(ops_block block, int size, int *dat_size, int *base,
                          int *d_m, int *d_p, int *stride, char *data, int type_size,
                          char const *type, char const *name) {

  /** ---- allocate an empty dat based on the local array sizes computed
           above on each MPI process                                      ----
     **/

  ops_dat dat = ops_decl_dat_temp_core(block, size, dat_size, base, d_m, d_p,
                                       stride, data, type_size, type, name);

  dat->user_managed = 0;
  dat->is_hdf5 = 0;

  // note that currently we assume replicated dats are read only or initialized
  // just once
  // what to do if not ?? How will the halos be handled
  // TODO: proper allocation and TAILQ
  // create list to hold sub-grid decomposition geometries for each mpi process
  OPS_sub_dat_list = (sub_dat_list *)ops_realloc(
      OPS_sub_dat_list, OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(sub_dat_list));

  // store away product array prod[] and MPI_Types for this ops_dat
  sub_dat_list sd = (sub_dat_list)ops_calloc(1, sizeof(sub_dat));
  sd->dat = dat;
  sd->dirtybit = 1;
  sd->dirty_dir_send =
      (int *)ops_malloc(sizeof(int) * 2 * block->dims * MAX_DEPTH);
  for (int i = 0; i < 2 * block->dims * MAX_DEPTH; i++)
    sd->dirty_dir_send[i] = 1;
  sd->dirty_dir_recv =
      (int *)ops_malloc(sizeof(int) * 2 * block->dims * MAX_DEPTH);
  for (int i = 0; i < 2 * block->dims * MAX_DEPTH; i++)
    sd->dirty_dir_recv[i] = 1;
  for (int i = 0; i < OPS_MAX_DIM; i++) {
    sd->d_ip[i] = 0;
    sd->d_im[i] = 0;
  }
  OPS_sub_dat_list[dat->index] = sd;

  return dat;
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name) {
  if (OPS_sub_block_list[dat->block->index]->owned == 1) {
    ops_get_data(dat);
    ops_print_dat_to_txtfile_core(dat, file_name);
  }
}

void ops_NaNcheck(ops_dat dat) {
  if (OPS_sub_block_list[dat->block->index]->owned == 1) {
    ops_get_data(dat);
    char buffer[30];
    sprintf(buffer, "On rank %d \t", ops_my_global_rank);
    ops_NaNcheck_core(dat, buffer);
  }
}

ops_dat ops_dat_copy(ops_dat orig_dat) {
  ops_dat dat = ops_dat_copy_mpi_core(orig_dat);
  ops_dat_deep_copy(dat, orig_dat);
  return dat;
}

void ops_dat_deep_copy(ops_dat target, ops_dat source) {
  int realloc = ops_dat_copy_metadata_core(target, source);
  if(realloc && source->block->instance->OPS_hybrid_gpu) {
    if(target->data_d != nullptr) {
      ops_device_free(source->block->instance, (void**)&(target->data_d);
      target->data_d = nullptr;
    }
    ops_device_malloc(source->block->instance, (void**)&(target->data_d), target->mem);
  }

  ops_kernel_descriptor *desc = ops_dat_deep_copy_mpi_core(target, source);
  if (source->block->instance->OPS_hybrid_gpu) {
    desc->name = "ops_internal_copy_device";
    desc->device = 1;
    desc->function = ops_internal_copy_device;
  } else {
    desc->name = "ops_internal_copy_seq";
    desc->device = 0;
    desc->function = ops_internal_copy_seq;
  }
  ops_enqueue_kernel(desc);
}
