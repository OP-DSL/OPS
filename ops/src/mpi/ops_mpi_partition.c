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

/** @brief ops mpi partitioning
  * @author Gihan Mudalige, adopted from Parallel OCCF routines by Mike Giles
  * @details Implements the single block structured mesh partitioning
  * for distributed memort (MPI) parallelization
  */

#include <mpi.h>
#include <ops_lib_cpp.h>
#include <ops_mpi_core.h>

MPI_Comm OPS_MPI_WORLD; // comm world for a single block
int comm_size;
int my_rank;


void ops_partition(int ndim2, int* dims2, char* routine)
{
  /**create cartesian processor grid**/

  MPI_Comm OPS_CART_COMM; //comm world
  int ndim = ndim2;
  int *pdims = (int *) xmalloc(ndim*sizeof(int));
  int *periodic = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++) {
    pdims[n] = 0;
    periodic[n] = 0; //false .. for now
  }

  MPI_Dims_create(comm_size, ndim, pdims);

  if(my_rank == MPI_ROOT){
    printf("proc grid: ");
    for(int n=0; n<ndim; n++)
      printf("%d ",pdims[n]);
    printf("\n");
  }

  MPI_Cart_create( OPS_MPI_WORLD,  ndim,  pdims,  periodic,
    1,  &OPS_CART_COMM);


  /**determine subgrid dimensions and displacements**/

  int my_cart_rank;
  int *coords = (int *) xmalloc(ndim*sizeof(int));
  int *disps = (int *) xmalloc(ndim*sizeof(int));
  int *dims = (int *) xmalloc(ndim*sizeof(int));


  MPI_Comm_rank(OPS_CART_COMM, &my_cart_rank);
  MPI_Cart_coords( OPS_CART_COMM, my_cart_rank, ndim, coords);

  printf("Coordinates of rank %d : (",my_rank);
  for(int n=0; n<ndim; n++)
    printf("%d ",coords[n]);
  printf(")\n");

  for(int n=0; n<ndim; n++){
    disps[n] = (coords[n] * dims2[n])/pdims[n];
    dims[n]  = ((coords[n]+1)*dims2[n])/pdims[n] - disps[n];
    dims2[n] = dims[n];
  }

  /**get IDs of neighbours**/

  int *id_m = (int *) xmalloc(ndim*sizeof(int));
  int *id_p = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++)
    MPI_Cart_shift(OPS_CART_COMM, n, 1, id_m, id_p);

  /**calculate subgrid start and end indicies**/
  int *ibeg = (int *) xmalloc(ndim*sizeof(int));
  int *iend = (int *) xmalloc(ndim*sizeof(int));
  for(int n=0; n<ndim; n++){
    ibeg[n] = disps[n];
    iend[n] = ibeg[n]+dims[n]-1;
  }

  printf("rank %d \n",my_rank);
    printf("%5s  :  %5s  :  %5s  :  %5s  :  %5s\n","dim","disp","size","start","end");
  for(int n=0; n<ndim; n++)
    printf("%5d  :  %5d  :  %5d  :  %5d  :  %5d\n",n , disps[n], dims[n], ibeg[n], iend[n]);
  printf("\n");

  printf("Finished block decomposition\n");

}
