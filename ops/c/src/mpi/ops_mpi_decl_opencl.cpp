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
  * @brief OPS mpi+opencl declaration
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the OPS API calls for the mpi+cuda backend
  */

#include <mpi.h>
#include <ops_mpi_core.h>

//#include <math_constants.h>
#include <ops_opencl_rt_support.h>
#include <ops_exceptions.h>

extern char *halo_buffer_d;
extern char *ops_buffer_send_1;
extern char *ops_buffer_recv_1;
extern char *ops_buffer_send_2;
extern char *ops_buffer_recv_2;

void ops_init_opencl(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  ops_init_core(instance, argc, argv, diags);

  if ((instance->OPS_block_size_x * instance->OPS_block_size_y * instance->OPS_block_size_z) > 1024) {
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_*");
      OPSException ex(OPS_RUNTIME_CONFIGURATION_ERROR);
      ex <<  "Error: OPS_block_size_x*OPS_block_size_y*OPS_block_size_z should be less than 1024 -- error OPS_block_size_* ";
      ex << " Current settings: " << instance->OPS_block_size_x << " " << instance->OPS_block_size_y << " " << instance->OPS_block_size_z;
      throw ex;
  }
  for (int n = 1; n < argc; n++) {
    if (strncmp(argv[n], "OPS_CL_DEVICE=", 14) == 0) {
      instance->OPS_cl_device = atoi(argv[n] + 14);
      printf("\n OPS_CL_DEVICE = %d \n", instance->OPS_cl_device);
    }
  }

  instance->opencl_instance = new OPS_instance_opencl();
  instance->opencl_instance->copy_tobuf_kernel = NULL;
  instance->opencl_instance->copy_frombuf_kernel = NULL;
  instance->opencl_instance->isbuilt_copy_tobuf_kernel = false;
  instance->opencl_instance->isbuilt_copy_frombuf_kernel = false;
  openclDeviceInit(instance, argc, argv);
}


// The one and only non-threads safe global OPS_instance
// Declared in ops_instance.cpp
extern OPS_instance *global_ops_instance;


void _ops_init(OPS_instance *instance, const int argc, const char *const argv[], const int diags) {
  //We do not have thread safety across MPI
  if ( OPS_instance::numInstances() != 1 ) {
    OPSException ex(OPS_RUNTIME_ERROR, "ERROR: multiple OPS instances are not supported over MPI");
    throw ex;
  }
  // So the MPI backend is not thread safe - that's fine.  It currently does not pass 
  // OPS_instance pointers around, but rather uses the global instance exclusively.
  // The only feasible use-case for MPI is that there is one OPS_instance.  Ideally that
  // would not be created via thread-safe API, but we kinda want to support that.  So we 
  // drill a back door into our own safety system: provided there is only one OPS_instance,
  // we assign that to the global non-thread safe var and carry on.
  global_ops_instance = instance;
  void* v;
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

  ops_init_opencl(instance, argc, argv, diags);
}

void ops_init(const int argc, const char *const argv[], const int diags) {
  _ops_init(OPS_instance::getOPSInstance(), argc, argv, diags);
}


void _ops_exit(OPS_instance *instance) {
  if (instance->is_initialised == 0) return;
  ops_mpi_exit(instance);
  if (halo_buffer_d != NULL)
    clReleaseMemObject((cl_mem)(halo_buffer_d));
  ops_free(ops_buffer_send_1);
  ops_free(ops_buffer_recv_1);
  ops_free(ops_buffer_send_2);
  ops_free(ops_buffer_recv_2);

  int flag = 0;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
  ops_opencl_exit(instance);
  delete instance->opencl_instance;
  ops_exit_core(instance);
}

void ops_exit() {
  _ops_exit(OPS_instance::getOPSInstance());
}

ops_dat ops_dat_copy(ops_dat orig_dat) {
  ops_dat dat = ops_dat_copy_mpi_core(orig_dat);
  ops_dat_deep_copy(dat, orig_dat);
  return dat;
}

void ops_dat_deep_copy(ops_dat target, ops_dat source) {
  // Copy the metadata.  This will reallocate target->data if necessary
  int realloc = ops_dat_copy_metadata_core(target, source);
  if(realloc) {
    if(target->data_d != nullptr) {
      clSafeCall(clReleaseMemObject((cl_mem)target->data_d));
      target->data_d = nullptr;
    }
    cl_int ret = 0;
    target->data_d = (char*)clCreateBuffer(target->block->instance->opencl_instance->OPS_opencl_core.context,
        CL_MEM_READ_WRITE, target->mem, NULL, &ret);
    clSafeCall(ret);
    clSafeCall(clFinish(target->block->instance->opencl_instance->OPS_opencl_core.command_queue));
  }

  ops_kernel_descriptor *desc = ops_dat_deep_copy_mpi_core(target, source);
  desc->name = "ops_internal_copy_opencl";
  desc->isdevice = 1;
  desc->func = ops_internal_copy_opencl;
  ops_enqueue_kernel(desc);
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

  // note that currently we assume replicated dats are read only or initialized
  // just once
  // what to do if not ?? How will the halos be handled

  // TODO: proper allocation and TAILQ
  // create list to hold sub-grid decomposition geometries for each mpi process
  OPS_sub_dat_list = (sub_dat_list *)ops_realloc(
      OPS_sub_dat_list, block->instance->OPS_dat_index * sizeof(sub_dat_list));

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

void ops_reduction_result_char(ops_reduction handle, int type_size, char *ptr) {
  ops_execute(handle->instance);
  ops_checkpointing_reduction(handle);
  memcpy(ptr, handle->data, handle->size);
  handle->initialized = 0;
}

ops_halo _ops_decl_halo(OPS_instance *instance, ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(instance, from, to, iter_size, from_base, to_base, from_dir,
                            to_dir);
}

ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int *from_base,
                       int *to_base, int *from_dir, int *to_dir) {
  return ops_decl_halo_core(from->block->instance, from, to, iter_size, from_base, to_base, from_dir,
                            to_dir);
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name) {
  if (OPS_sub_block_list[dat->block->index]->owned == 1) {
    ops_opencl_get_data(dat);
    ops_print_dat_to_txtfile_core(dat, file_name);
  }
}

void ops_NaNcheck(ops_dat dat) {
  if (OPS_sub_block_list[dat->block->index]->owned == 1) {
    ops_opencl_get_data(dat);
    char buffer[20];
    sprintf(buffer, "On rank %d \t", ops_my_global_rank);
    ops_NaNcheck_core(dat, buffer);
  }
}


// routine to fetch data from device
void ops_get_data(ops_dat dat) { ops_opencl_get_data(dat); }
void ops_put_data(ops_dat dat) { ops_opencl_put_data(dat); }
