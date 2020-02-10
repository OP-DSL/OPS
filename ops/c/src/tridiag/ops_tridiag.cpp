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
  * @brief OPS API calls and wrapper routins for Tridiagonal solvers
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the OPS API calls, wrapper routines and other
  * functions for interfacing with external Tridiagonal libraries
  */

#include <ops_lib_core.h>
#include "trid_cpu.h"
#include <ops_exceptions.h>

#ifdef __cplusplus
extern "C" {
#endif

void ops_initTridMultiDimBatchSolve(int ndim, int *dims) {
  // dummy routine for non-GPU backends
}

void ops_tridMultiDimBatch(
    int ndim,      // number of dimsnsions, ndim <= MAXDIM = 8
    int solvedim,  // user chosen dimension to perform solve
    int *dims,     // array containing the sizes of each ndim dimensions
    ops_dat a, ops_dat b, ops_dat c,  // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat
        d,  // right hand side coefficients of a multidimensional problem. An
            // array containing d column vectors of individual problems
    ops_dat u
    ) {
  if (a->mem != b->mem || b->mem != c->mem || c->mem != d->mem) {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets all need to be the same size");
  }
  tridDmtsvStridedBatch((const double *)a->data, (const double *)b->data,
                        (const double *)c->data, (double *)d->data,
                        (double *)u->data, ndim, solvedim, dims, a->size);

  /* Right now, we are simply using the same memory allocated by OPS
  as can be seen by the use of a->data, b->data, c->data etc.

  These data is currently not padded to be 32 or 64 bit aligned
  in the x-lines and so is inefficient.

  In the ADI example currently the mesh size is 256^3 and so we are
  32/54 bit alighed, thus we do not see any performance definiencies
  but other sizes will show this issue

  As such we will need to think on how to pad arrays.
  The problem is that on apps like Cloverleaf we see poorer performance
  due to extra x dim padding.
  */
}

void ops_tridMultiDimBatch_Inc(
    int ndim,      // number of dimsnsions, ndim <= MAXDIM = 8
    int solvedim,  // user chosen dimension to perform solve
    int *dims,     // array containing the sizes of each ndim dimensions
    ops_dat a, ops_dat b, ops_dat c,  // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat
        d,  // right hand side coefficients of a multidimensional problem. An
            // array containing d column vectors of individual problems
    ops_dat u
    ) {
  if (a->mem != b->mem || b->mem != c->mem || c->mem != d->mem || d->mem != u->mem) {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets all need to be the same size");
  }
  tridDmtsvStridedBatchInc((const double *)a->data, (const double *)b->data,
                           (const double *)c->data, (double *)d->data,
                           (double *)u->data, ndim, solvedim, dims, a->size);

}
#ifdef __cplusplus
}
#endif
