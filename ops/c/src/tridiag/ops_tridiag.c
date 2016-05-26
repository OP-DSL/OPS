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

/* @brief OPS API calls and wrapper routins for Tridiagonal solvers
* @author Gihan Mudalige, Istvan Reguly
* @details Implementations of the OPS API calls, wrapper routines and other
* functions for interfacing with external Tridiagonal libraries
*/

#include <ops_lib_core.h>
#include "trid_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void ops_initTridMultiDimBatchSolve(int ndim, int *dims) {
  //dummy routine for non-GPU backends
}

void ops_tridMultiDimBatch(
  int ndim, //number of dimsnsions, ndim <= MAXDIM = 8
  int solvedim, // user chosen dimension to perform solve
  int* dims, // array containing the sizes of each ndim dimensions
  ops_dat a, ops_dat b, ops_dat c,//left hand side coefficients of a
                                  //multidimensional problem. An array containing
                                  //A matrices of individual problems
  ops_dat d, // right hand side coefficients of a multidimensional problem. An
             // array containing d column vectors of individual problems
  ops_dat u,
  int* opts // indicates different algorithms to use -- not used for CPU backends
) {

  tridDmtsvStridedBatch((const double *)a->data,
    (const double *)b->data,
    (const double *)c->data,
    (double *)d->data, (double *)u->data, ndim, solvedim, dims, dims);



/* Assuming tridMultiDimBatchSolve() can do all 3 dimensions of the solve

1. copy over ops_dat->data to memaligned, x-dim padded blocks of memory
   - should we strip block halos ?
   (under MPI strip intra-block, i.e. MPI halos)
2. Perform tridiagonal solve in given dimension
3. copy back solution to ops_dat->data (i.e. strip the x-dim padding)

The problem is that we do not want to be copying over and over, especially as this
will be called within a iterative loop

If we increase the d_p[0] (the xdimension possitive block halo) so that size[0]
becomes a multiple of 32 (giving us padded memory) then atleast for single node
parallelizations we can simply use the ops_dat->data allocated by ops without copys
(is this true ?)

For MPI the above can be done, but will lead to large MPI halos in x-dimension
Is this somthing we can work with ?
*/

}

void ops_tridMultiDimBatch_Inc(
  int ndim, //number of dimsnsions, ndim <= MAXDIM = 8
  int solvedim, // user chosen dimension to perform solve
  int* dims, // array containing the sizes of each ndim dimensions
  ops_dat a, ops_dat b, ops_dat c,//left hand side coefficients of a
                                  //multidimensional problem. An array containing
                                  //A matrices of individual problems
  ops_dat d, // right hand side coefficients of a multidimensional problem. An
             // array containing d column vectors of individual problems
  ops_dat u,
  int* opts // indicates different algorithms to use -- not used for CPU backends
) {

  tridDmtsvStridedBatchInc((const double *)a->data,
    (const double *)b->data,
    (const double *)c->data,
    (double *)d->data, (double *)u->data, ndim, solvedim, dims, dims);

}
#ifdef __cplusplus
}
#endif