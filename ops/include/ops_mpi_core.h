#ifndef __OPS_MPI_CORE_H
#define __OPS_MPI_CORE_H
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

/** @brief core header file for the ops MPI backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Headderfile for OPS MPI backend
  */

#include <mpi.h>

/** Define the root MPI process **/
#ifdef MPI_ROOT
#undef MPI_ROOT
#endif
#define MPI_ROOT 0

//
//Struct for holding the decomposition details of a block on an MPI process
//
typedef struct {
  // the decomposition is for this block
  ops_block block;
  //number of dimensions;
  int ndim;
  // my MPI rank in each dimension (in cart cords)
  int* coords;
  // previous neighbor in each dimension (in cart cords)
  int* id_m;
  // next neighbor in each dimension (in cart cords)
  int* id_p;
  // the size of the local sub-block in each dimension
  int* sizes;
  // the displacement from the start of the block in each dimension
  int* disps;
  // the global index of the starting element of the local sub-block in each dimension
  int* istart;
  // the global index of the starting element of the local sub-block
  int* iend;

  //might need to hold explicitly the local istart and iend which are 0 and sizes[n]-1

} sub_block;

typedef sub_block * sub_block_list;


//
//MPI Communicator for halo creation and exchange
//

extern MPI_Comm OPS_MPI_WORLD;
extern int ops_comm_size;
extern int ops_my_rank;

//
// list holding sub-block geometries
//
extern sub_block_list *OPS_sub_block_list;


#endif /*__OPS_MPI_CORE_H*/
