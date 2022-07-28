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

#include <string>

#include <ops_lib_core.h>
#include <ops_tridiag.h>
#include <ops_exceptions.h>

#include <tridsolver.h>

ops_tridsolver_params::ops_tridsolver_params(ops_block block) {
  tridsolver_params = (void *)new TridParams();
}

ops_tridsolver_params::ops_tridsolver_params(ops_block block,
                                             SolveStrategy strategy) {
  tridsolver_params = (void *)new TridParams();
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
  // N/A for everything except single node CUDA
}

void ops_tridsolver_params::set_cuda_sync(int sync) {
  // N/A for everything except single node CUDA
}

void ops_tridMultiDimBatch(
    int ndim,      // number of dimensions, ndim <= MAXDIM = 8
    int solvedim,  // user chosen dimension to perform solve
    int *range,    // array containing the range over which to solve
    ops_dat a, ops_dat b, ops_dat c,  // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat
        d,  // right hand side coefficients of a multidimensional problem. An
            // array containing d column vectors of individual problems
    ops_tridsolver_params *tridsolver_ctx
    ) {

  for (int i = 0; i < ndim; i++) {
    if (a->size[i] != b->size[i] || b->size[i] != c->size[i] ||
        c->size[i] != d->size[i]) {
      throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets all need to be the same size");
    }
  }

  if(strcmp(a->type, b->type) != 0 || strcmp(b->type, c->type) != 0 ||
     strcmp(c->type, d->type) != 0) {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets must all be of the same type");
  }

  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  // Calculate dimension of area being solved and the starting offset from the
  // origin of the dat
  int dims[3];
  int offset = 0;
  for(int i = 0; i < ndim; i++) {
    dims[i] = range[2 * i + 1] - range[2 * i];
    if(i == 0)
      offset += range[2 * i];
    if(i == 1)
      offset += range[2 * i] * a->size[0];
    if(i == 2)
      offset += range[2 * i] * a->size[1] * a->size[0];
  }

  if(strcmp(a->type, "double") == 0) {
    const double *a_ptr = (double *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const double *b_ptr = (double *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const double *c_ptr = (double *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    double *d_ptr = (double *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);

    tridDmtsvStridedBatch((TridParams *)tridsolver_ctx->tridsolver_params,
                          a_ptr + offset, b_ptr + offset, c_ptr + offset,
                          d_ptr + offset, ndim, solvedim, dims, a->size);

    ops_dat_release_raw_data(d, 0, OPS_RW);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else if(strcmp(a->type, "float") == 0) {
    const float *a_ptr = (float *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const float *b_ptr = (float *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const float *c_ptr = (float *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    float *d_ptr = (float *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);

    tridSmtsvStridedBatch((TridParams *)tridsolver_ctx->tridsolver_params,
                          a_ptr + offset, b_ptr + offset, c_ptr + offset,
                          d_ptr + offset, ndim, solvedim, dims, a->size);

    ops_dat_release_raw_data(d, 0, OPS_RW);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: unsupported type. Dataset type must either be \"double\" or \"float\"");
  }

  /* Right now, we are simply using the same memory allocated by OPS
  as can be seen by the use of a->data, b->data, c->data etc.

  These data is currently not padded to be 32 or 64 bit aligned
  in the x-lines and so is inefficient.

  In the ADI example currently the mesh size is 256^3 and so we are
  32/64 bit aligned, thus we do not see any performance deficiencies
  but other sizes will show this issue

  As such we will need to think on how to pad arrays.
  The problem is that on apps like Cloverleaf we see poorer performance
  due to extra x dim padding.
  */
}

void ops_tridMultiDimBatch_Inc(
    int ndim,      // number of dimensions, ndim <= MAXDIM = 8
    int solvedim,  // user chosen dimension to perform solve
    int *range,    // array containing the range over which to solve
    ops_dat a, ops_dat b, ops_dat c,  // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat
        d,  // right hand side coefficients of a multidimensional problem. An
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

  if(strcmp(a->type, b->type) != 0 || strcmp(b->type, c->type) != 0 ||
     strcmp(c->type, d->type) != 0 || strcmp(d->type, u->type) != 0 ) {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets must all be of the same type");
  }

  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  // Calculate dimension of area being solved and the starting offset from the
  // origin of the dat
  int dims[3];
  int offset = 0;
  for(int i = 0; i < ndim; i++) {
    dims[i] = range[2 * i + 1] - range[2 * i];
    if(i == 0)
      offset += range[2 * i];
    if(i == 1)
      offset += range[2 * i] * a->size[0];
    if(i == 2)
      offset += range[2 * i] * a->size[1] * a->size[0];
  }

  if(strcmp(a->type, "double") == 0) {
    const double *a_ptr = (double *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const double *b_ptr = (double *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const double *c_ptr = (double *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    double *d_ptr = (double *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);
    double *u_ptr = (double *)ops_dat_get_raw_pointer(u, 0, S3D_000, &host);

    tridDmtsvStridedBatchInc((TridParams *)tridsolver_ctx->tridsolver_params,
                             a_ptr + offset, b_ptr + offset, c_ptr + offset,
                             d_ptr + offset, u_ptr + offset, ndim, solvedim,
                             dims, a->size);

    ops_dat_release_raw_data(u, 0, OPS_RW);
    ops_dat_release_raw_data(d, 0, OPS_READ);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else if(strcmp(a->type, "float") == 0) {
    const float *a_ptr = (float *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const float *b_ptr = (float *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const float *c_ptr = (float *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    float *d_ptr = (float *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);
    float *u_ptr = (float *)ops_dat_get_raw_pointer(u, 0, S3D_000, &host);

    tridSmtsvStridedBatchInc((TridParams *)tridsolver_ctx->tridsolver_params,
                             a_ptr + offset, b_ptr + offset, c_ptr + offset,
                             d_ptr + offset, u_ptr + offset, ndim, solvedim,
                             dims, a->size);

    ops_dat_release_raw_data(u, 0, OPS_RW);
    ops_dat_release_raw_data(d, 0, OPS_READ);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: unsupported type. Dataset type must either be \"double\" or \"float\"");
  }
}
