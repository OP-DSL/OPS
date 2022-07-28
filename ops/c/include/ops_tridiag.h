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
  * @brief OPS tridiagonal API header file
  * @author Gihan Mudalige
  * @details This header file should be included by all C++
  * OPS applications using the OPS Tridiagonal API
  */

#ifndef __OPS_TRIDIAG_H
#define __OPS_TRIDIAG_H

#include "ops_lib_core.h"

class ops_tridsolver_params {
public:
  enum SolveStrategy {
    GATHER_SCATTER = 0, // Gather boundaries on first nodes solve reduced system
                        // and scatter results
    ALLGATHER,          // Gather boundaries and solve reduced on all nodes
    JACOBI,             // Use Jacobi iterations to solve the reduced system
    PCR,                // Use PCR to solve the reduced system
    LATENCY_HIDING_INTERLEAVED, // Perform solves in mini-batches. Do forward
                                // run of the current mini-batch start
                                // communication and finish the previous
                                // mini-batch
    LATENCY_HIDING_TWO_STEP     // Perform solves in min-batches. First step:
                            // forwards and start communication, second step:
                            // wait for ready requests and finish mini-batches
  };

  ops_tridsolver_params(ops_block block);
  ops_tridsolver_params(ops_block block, SolveStrategy strategy);
  ~ops_tridsolver_params();

  void set_jacobi_params(double rtol, double atol, int maxiter);
  void set_batch_size(int batch_size);
  void set_cuda_opts(int opt_x, int opt_y, int opt_z);
  void set_cuda_sync(int sync);

  void *tridsolver_params;
};

OPS_FTN_INTEROP
void ops_tridMultiDimBatch(int ndim, int solvedim, int* range, ops_dat a,
                           ops_dat b, ops_dat c, ops_dat d,
                           ops_tridsolver_params *tridsolver_ctx);

OPS_FTN_INTEROP
void ops_tridMultiDimBatch_Inc(int ndim, int solvedim, int* range, ops_dat a,
                               ops_dat b, ops_dat c, ops_dat d, ops_dat u,
                               ops_tridsolver_params *tridsolver_ctx);

#endif
