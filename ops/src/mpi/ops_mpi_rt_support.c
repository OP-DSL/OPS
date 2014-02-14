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

/** @brief ops mpi run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi backend
  */

#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

// Timing
double t1,t2,c1,c2;

int ops_is_root()
{
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  return (my_rank==MPI_ROOT);
}

void ops_exchange_halo(ops_arg* arg, int d /*depth*/)
{
  ops_dat dat = arg->dat;
  //printf("Exchanging halos for %s\n",dat->name);

  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];

  int i1,i2,i3,i4; //indicies for halo and boundary of the dat
  int* d_m = sd->d_m;
  int* d_p = sd->d_p;
  int* prod = sd->prod;
  int size = dat->size;
  MPI_Status status;

  for(int n=0;n<sb->ndim;n++){
    if(dat->block_size[n] > 1 && d > 0) {

      i1 = (-d_m[n] - d) * prod[n-1];
      i2 = (-d_m[n]    ) * prod[n-1];
      i3 = (prod[n]/prod[n-1] - (-d_p[n]) - d) * prod[n-1];
      i4 = (prod[n]/prod[n-1] - (-d_p[n])    ) * prod[n-1];

      //send in positive direction, receive from negative direction
      //printf("Exchaning 1 From:%d To: %d\n", i3, i1);
      MPI_Sendrecv(&dat->data[i3*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_p[n],0,
                   &dat->data[i1*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_m[n],0,
                   OPS_CART_COMM, &status);

      //send in negative direction, receive from positive direction
      //printf("Exchaning 2 From:%d To: %d\n", i2, i4);
      MPI_Sendrecv(&dat->data[i2*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_m[n],1,
                   &dat->data[i4*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_p[n],1,
                   OPS_CART_COMM, &status);
    }
  }
}


void ops_mpi_reduce_double(ops_arg* arg, double* data)
{
  (void)data;
  ops_timers_core(&c1, &t1);

  if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
    double *result = (double *) calloc (arg->dim, sizeof (double));

    if(arg->acc == OPS_INC)//global reduction
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_SUM, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MAX)//global maximum
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_MAX, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MIN)//global minimum
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_MIN, OPS_MPI_WORLD);
    else if(arg->acc == OPS_WRITE) {//any
      result = (double *) xrealloc (result,arg->dim*ops_comm_size*sizeof(double));
      MPI_Allgather((double *)arg->data, arg->dim, MPI_DOUBLE, result, arg->dim, MPI_DOUBLE, OPS_MPI_WORLD);
      for (int i = 1; i < ops_comm_size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0.0)
            result[j] = result[i*arg->dim+j];
        }
      }
    }
    memcpy(arg->data, result, sizeof(double)*arg->dim);
    free(result);
  }
}


/*void ops_mpi_reduce_double(ops_arg* arg, double* data)
{
  (void)data;
  ops_timers_core(&c1, &t1);
  if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ)
  {
    double result_static;
    double *result;
    if (arg->dim > 1 && arg->acc != OPS_WRITE)
      result = (double *) calloc (arg->dim, sizeof (double));
    else result = &result_static;

    if(arg->acc == OPS_INC)//global reduction
    {
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE,
          MPI_SUM, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(double)*arg->dim);
    }
    else if(arg->acc == OPS_MAX)//global maximum
    {
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE,
          MPI_MAX, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(double)*arg->dim);;
    }
    else if(arg->acc == OPS_MIN)//global minimum
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_DOUBLE,
          MPI_MIN, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(double)*arg->dim);
    }
    else if(arg->acc == OPS_WRITE)//any
    {
      int size;
      MPI_Comm_size(OPS_MPI_WORLD, &size);
      result = (double *) calloc (arg->dim*size, sizeof (double));
      MPI_Allgather((double *)arg->data, arg->dim, MPI_DOUBLE,
                    result, arg->dim, MPI_DOUBLE,
                    OPS_MPI_WORLD);
      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0.0)
            result[j] = result[i*arg->dim+j];
        }
      }
      memcpy(arg->data, result, sizeof(double)*arg->dim);
      if (arg->dim == 1) free(result);
    }
    if (arg->dim > 1) free (result);
  }
  ops_timers_core(&c2, &t2);
}*/


void ops_mpi_reduce_float(ops_arg* arg, float* data)
{
  (void)data;
  ops_timers_core(&c1, &t1);
  if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ)
  {
    float result_static;
    float *result;
    if (arg->dim > 1 && arg->acc != OPS_WRITE)
      result = (float *) calloc (arg->dim, sizeof (float));
    else result = &result_static;

    if(arg->acc == OPS_INC)//global reduction
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT,
          MPI_SUM, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(float)*arg->dim);
    }
    else if(arg->acc == OPS_MAX)//global maximum
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT,
          MPI_MAX, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(float)*arg->dim);;
    }
    else if(arg->acc == OPS_MIN)//global minimum
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT,
          MPI_MIN, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(float)*arg->dim);
    }
    else if(arg->acc == OPS_WRITE)//any
    {
      int size;
      MPI_Comm_size(OPS_MPI_WORLD, &size);
      result = (float *) calloc (arg->dim*size, sizeof (float));
      MPI_Allgather((float *)arg->data, arg->dim, MPI_FLOAT,
                    result, arg->dim, MPI_FLOAT,
                    OPS_MPI_WORLD);
      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0.0)
            result[j] = result[i*arg->dim+j];
        }
      }
      memcpy(arg->data, result, sizeof(float)*arg->dim);
      if (arg->dim == 1) free(result);
    }
    if (arg->dim > 1) free (result);
  }
  ops_timers_core(&c2, &t2);
}


void ops_mpi_reduce_int(ops_arg* arg, int* data)
{
  (void)data;
  ops_timers_core(&c1, &t1);
  if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ)
  {
    int result_static;
    int *result;
    if (arg->dim > 1 && arg->acc != OPS_WRITE)
      result = (int *) calloc (arg->dim, sizeof (int));
    else result = &result_static;

    if(arg->acc == OPS_INC)//global reduction
    {
      MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT,
          MPI_SUM, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(int)*arg->dim);
    }
    else if(arg->acc == OPS_MAX)//global maximum
    {
      MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT,
          MPI_MAX, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(int)*arg->dim);;
    }
    else if(arg->acc == OPS_MIN)//global minimum
    {
      MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT,
          MPI_MIN, OPS_MPI_WORLD);
      memcpy(arg->data, result, sizeof(int)*arg->dim);
    }
    else if(arg->acc == OPS_WRITE)//any
    {
      int size;
      MPI_Comm_size(OPS_MPI_WORLD, &size);
      result = (int *) calloc (arg->dim*size, sizeof (int));
      MPI_Allgather((int *)arg->data, arg->dim, MPI_INT,
                    result, arg->dim, MPI_INT,
                    OPS_MPI_WORLD);
      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0.0)
            result[j] = result[i*arg->dim+j];
        }
      }
      memcpy(arg->data, result, sizeof(int)*arg->dim);
      if (arg->dim == 1) free(result);
    }
    if (arg->dim > 1) free (result);
  }
  ops_timers_core(&c2, &t2);
}
