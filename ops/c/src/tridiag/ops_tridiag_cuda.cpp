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
  * @brief OPS API calls and wrapper routines for Tridiagonal solvers
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the OPS API calls, wrapper routines and other
  * functions for interfacing with external Tridiagonal libraries
  */

#include <ops_cuda_rt_support.h>
#include <ops_lib_core.h>
#include <ops_tridiag.h>

#include <tridsolver.h>

ops_tridsolver_params::ops_tridsolver_params(ops_block block) {
  TridParams *tp = new TridParams();
  tp->opts[0] = 0;
  tp->opts[1] = 0;
  tp->opts[2] = 0;
  tp->sync    = 0;
  if (block->instance->OPS_diags > 1)
    tp->sync = 1;
  tridsolver_params = (void *)tp;
}

ops_tridsolver_params::ops_tridsolver_params(ops_block block,
                                             SolveStrategy strategy) {
  TridParams *tp = new TridParams();
  tp->opts[0] = 0;
  tp->opts[1] = 0;
  tp->opts[2] = 0;
  tp->sync    = 0;
  if (block->instance->OPS_diags > 1)
    tp->sync = 1;
  tridsolver_params = (void *)tp;
}

ops_tridsolver_params::~ops_tridsolver_params() {
  delete (TridParams *)tridsolver_params;
}

void ops_tridsolver_params::set_jacobi_params(double rtol, double atol,
                                              int maxiter) {
  // N/A for non-MPI code
}

void ops_tridsolver_params::set_batch_size(int batch_size) {
  // N/A for non-MPI code
}

void ops_tridsolver_params::set_cuda_opts(int opt_x, int opt_y, int opt_z) {
  TridParams *tp = (TridParams *)tridsolver_params;
  tp->opts[0] = opt_x;
  tp->opts[1] = opt_y;
  tp->opts[2] = opt_z;
}

void ops_tridsolver_params::set_cuda_sync(int sync) {
  TridParams *tp = (TridParams *)tridsolver_params;
  tp->sync = sync;
}

void ops_tridMultiDimBatch(
    int ndim,     // number of dimensions, ndim <= MAXDIM = 8
    int solvedim, // user chosen dimension to perform solve
    int *dims,    // array containing the sizes of each ndim dimensions
    ops_dat a, ops_dat b, ops_dat c, // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat d, // right hand side coefficients of a multidimensional problem. An
               // array containing d column vectors of individual problems
    ops_dat u,
    ops_tridsolver_params *tridsolver_ctx
    ) {

  // check if sizes match
  for (int i = 0; i < ndim; i++) {
    if (a->size[i] != b->size[i] || b->size[i] != c->size[i] ||
        c->size[i] != d->size[i] || d->size[i] != u->size[i]) {
      throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets all need to be the same size");
    }
  }

  int device = OPS_DEVICE;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  const double *a_ptr = (double *)ops_dat_get_raw_pointer(a, 0, S3D_000, &device);
  const double *b_ptr = (double *)ops_dat_get_raw_pointer(b, 0, S3D_000, &device);
  const double *c_ptr = (double *)ops_dat_get_raw_pointer(c, 0, S3D_000, &device);
  double *d_ptr = (double *)ops_dat_get_raw_pointer(d, 0, S3D_000, &device);
  double *u_ptr = (double *)ops_dat_get_raw_pointer(u, 0, S3D_000, &device);

  tridDmtsvStridedBatch((TridParams *)tridsolver_ctx->tridsolver_params, a_ptr,
                        b_ptr, c_ptr, d_ptr, u_ptr, ndim, solvedim, dims,
                        a->size);

  ops_dat_release_raw_data(u, 0, OPS_READ);
  ops_dat_release_raw_data(d, 0, OPS_RW);
  ops_dat_release_raw_data(c, 0, OPS_READ);
  ops_dat_release_raw_data(b, 0, OPS_READ);
  ops_dat_release_raw_data(a, 0, OPS_READ);
}

void ops_tridMultiDimBatch_Inc(
    int ndim,     // number of dimensions, ndim <= MAXDIM = 8
    int solvedim, // user chosen dimension to perform solve
    int *dims,    // array containing the sizes of each ndim dimensions
    ops_dat a, ops_dat b, ops_dat c, // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat d, // right hand side coefficients of a multidimensional problem. An
               // array containing d column vectors of individual problems
    ops_dat u,
    ops_tridsolver_params *tridsolver_ctx
    ) {

  // check if sizes match
  for (int i = 0; i < ndim; i++) {
    if (a->size[i] != b->size[i] || b->size[i] != c->size[i] ||
        c->size[i] != d->size[i] || d->size[i] != u->size[i]) {
      throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets all need to be the same size");
    }
  }

  int device = OPS_DEVICE;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  const double *a_ptr = (double *)ops_dat_get_raw_pointer(a, 0, S3D_000, &device);
  const double *b_ptr = (double *)ops_dat_get_raw_pointer(b, 0, S3D_000, &device);
  const double *c_ptr = (double *)ops_dat_get_raw_pointer(c, 0, S3D_000, &device);
  double *d_ptr = (double *)ops_dat_get_raw_pointer(d, 0, S3D_000, &device);
  double *u_ptr = (double *)ops_dat_get_raw_pointer(u, 0, S3D_000, &device);

  tridDmtsvStridedBatchInc((TridParams *)tridsolver_ctx->tridsolver_params,
                           a_ptr, b_ptr, c_ptr, d_ptr, u_ptr, ndim, solvedim,
                           dims, a->size);

  ops_dat_release_raw_data(u, 0, OPS_RW);
  ops_dat_release_raw_data(d, 0, OPS_READ);
  ops_dat_release_raw_data(c, 0, OPS_READ);
  ops_dat_release_raw_data(b, 0, OPS_READ);
  ops_dat_release_raw_data(a, 0, OPS_READ);
}
