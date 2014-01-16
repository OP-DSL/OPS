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

/** @brief ops mpi declaration
  * @author Gihan Mudalige
  * @details Implements the OPS API calls for the mpi backend
  */

#include <mpi.h>
#include <ops_lib_cpp.h>
#include <ops_mpi_core.h>


void
ops_init ( int argc, char ** argv, int diags )
{
  int flag = 0;
  MPI_Initialized(&flag);
  if(!flag) {
    MPI_Init(&argc, &argv);
  }

  MPI_Comm_dup(MPI_COMM_WORLD, &OPS_MPI_WORLD);
  MPI_Comm_rank(OPS_MPI_WORLD, &ops_my_rank);
  MPI_Comm_size(OPS_MPI_WORLD, &ops_comm_size);

  ops_init_core ( argc, argv, diags );
}

void ops_exit()
{
  //ops_mpi_exit();
  //ops_rt_exit();
  ops_exit_core();

  int flag = 0;
  MPI_Finalized(&flag);
  if(!flag) MPI_Finalize();
}

ops_dat ops_decl_dat_char (ops_block block, int size, int *block_size,
                           int* offset,  char* data, int type_size,
                           char const * type, char const * name )
{
  ops_dat dat;

  if(data != NULL) {
    dat = ops_decl_dat_core(block, size, block_size, offset, data, type_size, type, name);
  }
  else {
    dat = ops_decl_dat_temp_core (block, size, block_size, offset,
                                           data, type_size, type, name );
    int bytes = size*type_size;
    for (int i=0; i<block->dims; i++) bytes = bytes*block_size[i];
    dat->data = (char*) calloc(bytes, 1); //initialize data bits to 0
    //dat->data = (char*) malloc(bytes); //initialize data bits to 0
    dat->user_managed = 0;
  }

  return dat;
}


ops_dat ops_decl_dat_mpi_char(ops_block block, int size, int *dat_size, int* offset,
                           int* tail, char* data,
                           int type_size, char const * type, char const * name )
{
  int edge_dat = 0; //flag indicating that this is an edge dat

  sub_block_list sb = OPS_sub_block_list[block->index]; //get sub-block geometries

  int* sub_size = (int *)xmalloc(sizeof(int) * sb->ndim);

  for(int n=0;n<sb->ndim;n++){
    if(dat_size[n] != 1) { //i.e. this dat is a regular data block that needs to decomposed
      //compute the local array sizes for this dat for this dimension
      //including max halo depths
      sub_size[n] = sb->sizes[n] - offset[n] - tail[n];
    }
    else { // this dat is a an edge data block that needs to be replicated on each MPI process
      //apply the size as 1 for this dimension, later to be replicated on each process
      sub_size[n] = 1;
      edge_dat = 1;
    }
  }

/** ---- allocate an empty dat based on the local array sizes computed
         above on each MPI process                                      ---- **/
  ops_dat dat = ops_decl_dat_temp_core(block, size, sub_size, offset, data, type_size, type, name );

  int bytes = size*type_size;
  for (int i=0; i<sb->ndim; i++) bytes = bytes*sub_size[i];
  dat->data = (char*) calloc(bytes, 1); //initialize data bits to 0
  dat->user_managed = 0;

  for(int j = 0; j<sub_size[1]; j++) {
    for(int i = 0; i<sub_size[0]; i++) {
      dat->data[dat->size * (j* sub_size[0] + i) ] = (double )(j * sub_size[0] + i);
    }
  }

  //note that currently we assume replicated dats are read only or initialized just once
  //what to do if not ?? How will the halos be handled

  /** ---- Create MPI data types for halo exchange ---- **/
  if(!edge_dat) {

    int *prod_t = (int *) xmalloc((sb->ndim+1)*sizeof(int));
    int *prod = &prod_t[1];

    prod[-1] = 1;
    for(int n = 0; n<sb->ndim; n++) {
      prod[n] = prod[n-1]*(sb->sizes[n] -offset[n] - tail[n]);
    }

    MPI_Datatype* stride = (MPI_Datatype *) xmalloc(sizeof(MPI_Datatype)*sb->ndim * MAX_DEPTH);

    for(int n = 0; n<sb->ndim; n++) { //need to make MPI_DOUBLE_PRECISION general
      for(int d = 0; d<MAX_DEPTH; d++) {
        MPI_Type_vector(prod[sb->ndim - 1]/prod[n], d*prod[n-1], prod[n], MPI_DOUBLE_PRECISION, &stride[MAX_DEPTH*n+d]);
        MPI_Type_commit(&stride[MAX_DEPTH*n+d]);
        //printf("Datatype: %d %d %d\n", prod[sb->ndim - 1]/prod[n], prod[n-1], prod[n]);
      }
    }

    //store away product array prod[] and MPI_Types for this ops_dat
    sb->prod = prod;
    sb->mpidat = stride;
    sb->offset = offset;
    sb->tail = tail;
  }

  return dat;
}

ops_arg ops_arg_dat( ops_dat dat, ops_stencil stencil, char const * type, ops_access acc )
{
  return ops_arg_dat_core( dat, stencil, acc );
}

ops_arg ops_arg_gbl_char( char * data, int dim, int size, ops_access acc )
{
  return ops_arg_gbl_core( data, dim, size, acc );
}

void ops_print_dat_to_txtfile(ops_dat dat, const char *file_name)
{
  ops_print_dat_to_txtfile_core(dat, file_name);
}

void ops_decl_const_char( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

void ops_timers(double * cpu, double * et)
{
    ops_timers_core(cpu,et);
}

void ops_printf(const char* format, ...)
{
  if(ops_my_rank==MPI_ROOT) {
    va_list argptr;
    va_start(argptr, format);
    vprintf(format, argptr);
    va_end(argptr);
  }
}

void ops_fprintf(FILE *stream, const char *format, ...)
{
  if(ops_my_rank==MPI_ROOT) {
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stream, format, argptr);
    va_end(argptr);
  }
}
