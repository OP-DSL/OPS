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

int ops_is_root()
{
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  return (my_rank==MPI_ROOT);
}

void ops_exchange_halo(ops_arg* arg, int d /*depth*/)
{
  ops_dat dat = arg->dat;

  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];

  int i1,i2,i3,i4; //indicies for halo and boundary of the dat
  int* d_m = sd->d_m;
  int* d_p = sd->d_p;
  int* prod = sd->prod;
  MPI_Status *status;
  int size = dat->size;

  for(int n=0;n<sb->ndim;n++){

    MPI_Status status;
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
