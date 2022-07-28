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
  * @author Gihan Mudalige, Istvan Reguly, Toby Flynn
  * @details Implementations of the OPS API calls, wrapper routines and other
  * functions for interfacing with external Tridiagonal libraries
  */

#include <string>

#include <ops_lib_core.h>
#include <ops_mpi_core.h>
#include <ops_exceptions.h>
#include <ops_tridiag.h>

#include <tridsolver.h>
#include <trid_mpi_solver_params.hpp>

ops_tridsolver_params::ops_tridsolver_params(ops_block block) {
  TridParams *tp = new TridParams();
  sub_block *sb = OPS_sub_block_list[block->index];
  MpiSolverParams::MPICommStrategy strat = MpiSolverParams::PCR;
  MpiSolverParams *trid_mpi_params = new MpiSolverParams(sb->comm, sb->ndim,
                                                         sb->pdims, strat);
  tp->mpi_params    = (void *)trid_mpi_params;
  tridsolver_params = (void *)tp;
}

ops_tridsolver_params::ops_tridsolver_params(ops_block block,
                                             SolveStrategy strategy) {
  TridParams *tp = new TridParams();
  sub_block *sb = OPS_sub_block_list[block->index];
  MpiSolverParams::MPICommStrategy strat;
  switch (strategy) {
    case GATHER_SCATTER:
      strat = MpiSolverParams::GATHER_SCATTER;
      break;
    case ALLGATHER:
      strat = MpiSolverParams::ALLGATHER;
      break;
    case LATENCY_HIDING_TWO_STEP:
      strat = MpiSolverParams::LATENCY_HIDING_TWO_STEP;
      break;
    case LATENCY_HIDING_INTERLEAVED:
      strat = MpiSolverParams::LATENCY_HIDING_INTERLEAVED;
      break;
    case JACOBI:
      strat = MpiSolverParams::JACOBI;
      break;
    case PCR:
      strat = MpiSolverParams::PCR;
      break;
    default:
      throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: Unrecognised solving strategy");
  }

  MpiSolverParams *trid_mpi_params = new MpiSolverParams(sb->comm, sb->ndim,
                                                         sb->pdims, strat);
  tp->mpi_params    = (void *)trid_mpi_params;
  tridsolver_params = (void *)tp;
}

ops_tridsolver_params::~ops_tridsolver_params() {
  delete (MpiSolverParams *)((TridParams *)tridsolver_params)->mpi_params;
  delete (TridParams *)tridsolver_params;
}

void ops_tridsolver_params::set_jacobi_params(double rtol, double atol,
                                              int maxiter) {
  TridParams *tp = (TridParams *)tridsolver_params;
  MpiSolverParams *params = (MpiSolverParams *)tp->mpi_params;
  params->jacobi_rtol     = rtol;
  params->jacobi_atol     = atol;
  params->jacobi_maxiter  = maxiter;
}

void ops_tridsolver_params::set_batch_size(int batch_size) {
  TridParams *tp = (TridParams *)tridsolver_params;
  MpiSolverParams *params = (MpiSolverParams *)tp->mpi_params;
  params->mpi_batch_size  = batch_size;
}

void ops_tridsolver_params::set_cuda_opts(int opt_x, int opt_y, int opt_z) {
  // N/A for everything except single node CUDA
}

void ops_tridsolver_params::set_cuda_sync(int sync) {
  // N/A for everything except single node CUDA
}

void ops_tridMultiDimBatch(
    int ndim,     // number of dimensions, ndim <= MAXDIM = 8
    int solvedim, // user chosen dimension to perform solve
    int *range,   // array containing the range over which to solve
    ops_dat a, ops_dat b, ops_dat c, // left hand side coefficients of a
    // multidimensional problem. An array containing
    // A matrices of individual problems
    ops_dat d, // right hand side coefficients of a multidimensional problem. An
               // array containing d column vectors of individual problems
    ops_tridsolver_params *tridsolver_ctx
    ) {

  // check if sizes match
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

  // Will break if range exlcudes an entire MPI process
  int offset = 0;
  int dims_calc[3];
  sub_dat_list sd = OPS_sub_dat_list[a->index];
  for(int i = 0; i < ndim; i++) {
    bool minRangeInSubDat = range[2 * i] >= sd->decomp_disp[i] && range[2 * i] < sd->decomp_disp[i] + sd->decomp_size[i];
    bool maxRangeinSubDat = range[2 * i + 1] > sd->decomp_disp[i] && range[2 * i + 1] <= sd->decomp_disp[i] + sd->decomp_size[i];
    // Check that range is within this sub_dat's global elements
    // (otherwise no offset from sub_dat's origin needed for this dimension)
    if(minRangeInSubDat) {
      if(i == 0)
        offset += range[2 * i];
      if(i == 1)
        offset += range[2 * i] * a->size[0];
      if(i == 2)
        offset += range[2 * i] * a->size[1] * a->size[0];
    }

    int pads_m = -1 * (a->d_m[i] + sd->d_im[i]);
    int pads_p = a->d_p[i] + sd->d_ip[i];
    dims_calc[i] = a->size[i] - pads_m - pads_p;

    // Edit size of solve to include range information
    if(minRangeInSubDat && maxRangeinSubDat) {
      dims_calc[i] = range[2 * i + 1] - range[2 * i];
    } else if(minRangeInSubDat) {
      dims_calc[i] -= range[2 * i];
    } else if(maxRangeinSubDat) {
      int global_size = sd->gbl_size[i] - sd->gbl_d_p[i] + sd->gbl_d_m[i];
      dims_calc[i] -= global_size - range[2 * i + 1];
    }
  }

  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  if(strcmp(a->type, "double") == 0) {
    // Get raw pointer access to data held by OPS
    // Points to element 0, skipping MPI halo
    const double *a_ptr = (double *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const double *b_ptr = (double *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const double *c_ptr = (double *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    double *d_ptr = (double *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);

    tridDmtsvStridedBatch((TridParams *)tridsolver_ctx->tridsolver_params,
                          a_ptr + offset, b_ptr + offset, c_ptr + offset,
                          d_ptr + offset, ndim, solvedim, dims_calc, a->size);

    // Release pointer access back to OPS
    ops_dat_release_raw_data(d, 0, OPS_RW);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else if(strcmp(a->type, "float") == 0) {
    // Get raw pointer access to data held by OPS
    // Points to element 0, skipping MPI halo
    const float *a_ptr = (float *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const float *b_ptr = (float *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const float *c_ptr = (float *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    float *d_ptr = (float *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);

    tridSmtsvStridedBatch((TridParams *)tridsolver_ctx->tridsolver_params,
                          a_ptr + offset, b_ptr + offset, c_ptr + offset,
                          d_ptr + offset, ndim, solvedim, dims_calc, a->size);

    // Release pointer access back to OPS
    ops_dat_release_raw_data(d, 0, OPS_RW);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: unsupported type. Dataset type must either be \"double\" or \"float\"");
  }
}

void ops_tridMultiDimBatch_Inc(
    int ndim,     // number of dimensions, ndim <= MAXDIM = 8
    int solvedim, // user chosen dimension to perform solve
    int *range,   // array containing the range over which to solve
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

  if(strcmp(a->type, b->type) != 0 || strcmp(b->type, c->type) != 0 ||
     strcmp(c->type, d->type) != 0 || strcmp(d->type, u->type) != 0 ) {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: the a,b,c,d datasets must all be of the same type");
  }

  // Will break if range exlcudes an entire MPI process
  int offset = 0;
  int dims_calc[3];
  sub_dat_list sd = OPS_sub_dat_list[a->index];
  for(int i = 0; i < ndim; i++) {
    bool minRangeInSubDat = range[2 * i] >= sd->decomp_disp[i] && range[2 * i] < sd->decomp_disp[i] + sd->decomp_size[i];
    bool maxRangeinSubDat = range[2 * i + 1] > sd->decomp_disp[i] && range[2 * i + 1] <= sd->decomp_disp[i] + sd->decomp_size[i];
    // Check that range is within this sub_dat's global elements
    // (otherwise no offset from sub_dat's origin needed for this dimension)
    if(minRangeInSubDat) {
      if(i == 0)
        offset += range[2 * i];
      if(i == 1)
        offset += range[2 * i] * a->size[0];
      if(i == 2)
        offset += range[2 * i] * a->size[1] * a->size[0];
    }

    int pads_m = -1 * (a->d_m[i] + sd->d_im[i]);
    int pads_p = a->d_p[i] + sd->d_ip[i];
    dims_calc[i] = a->size[i] - pads_m - pads_p;

    // Edit size of solve to include range information
    if(minRangeInSubDat && maxRangeinSubDat) {
      dims_calc[i] = range[2 * i + 1] - range[2 * i];
    } else if(minRangeInSubDat) {
      dims_calc[i] -= range[2 * i];
    } else if(maxRangeinSubDat) {
      int global_size = sd->gbl_size[i] - sd->gbl_d_p[i] + sd->gbl_d_m[i];
      dims_calc[i] -= global_size - range[2 * i + 1];
    }
  }

  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  if(strcmp(a->type, "double") == 0) {
    const double *a_ptr = (double *)ops_dat_get_raw_pointer(a, 0, S3D_000, &host);
    const double *b_ptr = (double *)ops_dat_get_raw_pointer(b, 0, S3D_000, &host);
    const double *c_ptr = (double *)ops_dat_get_raw_pointer(c, 0, S3D_000, &host);
    double *d_ptr = (double *)ops_dat_get_raw_pointer(d, 0, S3D_000, &host);
    double *u_ptr = (double *)ops_dat_get_raw_pointer(u, 0, S3D_000, &host);

    // For now do not consider adding padding
    tridDmtsvStridedBatchInc((TridParams *)tridsolver_ctx->tridsolver_params,
                             a_ptr + offset, b_ptr + offset, c_ptr + offset,
                             d_ptr + offset, u_ptr + offset, ndim, solvedim,
                             dims_calc, a->size);

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

    // For now do not consider adding padding
    tridSmtsvStridedBatchInc((TridParams *)tridsolver_ctx->tridsolver_params,
                             a_ptr + offset, b_ptr + offset, c_ptr + offset,
                             d_ptr + offset, u_ptr + offset, ndim, solvedim,
                             dims_calc, a->size);

    ops_dat_release_raw_data(u, 0, OPS_RW);
    ops_dat_release_raw_data(d, 0, OPS_READ);
    ops_dat_release_raw_data(c, 0, OPS_READ);
    ops_dat_release_raw_data(b, 0, OPS_READ);
    ops_dat_release_raw_data(a, 0, OPS_READ);
  } else {
    throw OPSException(OPS_RUNTIME_ERROR, "Tridsolver error: unsupported type. Dataset type must either be \"double\" or \"float\"");
  }
}
