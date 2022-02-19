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

/** @brief An example of implementing the fourth-order compact scheme
 *  @author Jianping Meng
 *  @details An example of implementing the fourth-order compact scheme using
 *the tridiagonal solver. A field variable u is initialised with
 *sin(x)*sin(2*y)*sin(3*z) and the its derivatives, u_x, u_y and u_z, are
 *calculated using the compact scheme.
 **/

#include <cmath>
#include <string>
#include <vector>
#define OPS_3D
#include "ops_seq_v2.h"
#include "data.h"
// Kernel functions for setting the initial conditions
#include "init_kernel.h"
// Kernel functions for preparing for a,b,c,d
#include "preproc_kernel.h"

// Defining the computational problem domain. As a test, we use the
// simplest grid See the document for the meaning of variables r1 and r2.
double xyzRange[2]{0, 2 * 3.1415926535897};
int nx{32};
int ny{nx};
int nz{nx};
double dx{(xyzRange[1] - xyzRange[0]) / (nx - 1)};
double dy{(xyzRange[1] - xyzRange[0]) / (ny - 1)};
double dz{(xyzRange[1] - xyzRange[0]) / (nz - 1)};
double h{dx};
double left{h / 3};
double right{h / 3};
double present{4 * h / 3};

// Utility function for writting out the data
void WriteDataToH5(const std::string &fileName, const ops_block &block,
                   const std::vector<ops_dat> &dataList) {
  ops_fetch_block_hdf5_file(block, fileName.c_str());
  for (auto data : dataList) {
    ops_fetch_dat_hdf5_file(data, fileName.c_str());
  }
}

int main(int argc, char *argv[]) {
  // OPS initialisation
  ops_init(argc, argv, 2);
  /**--- OPS declarations----**/
  // declare block
  ops_block compact3d = ops_decl_block(3, "Compact3D");
  // declare data on blocks
  // max haloDepth depths for the dat in the positive direction
  const int haloDepth{1};
  int d_p[3]{haloDepth, haloDepth, haloDepth};
  // max haloDepth depths for the dat in the negative direction*/
  int d_m[3]{-haloDepth, -haloDepth, -haloDepth};

  // size of the dat
  int size[3]{nx, ny, nz};
  int base[3]{0, 0, 0};
  double *temp = NULL;

  // Define u, u_x, u_y and u_z
  ops_dat u{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "u")};
  ops_dat ux{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "ux")};
  ops_dat uy{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "uy")};
  ops_dat uz{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "uz")};

  ops_dat a{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "a")};
  ops_dat b{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "b")};
  ops_dat c{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "c")};
  ops_dat d{
      ops_decl_dat(compact3d, 1, size, base, d_m, d_p, temp, "double", "d")};

  const std::vector<ops_dat> resList{u, ux, uy, uz};

  // declare stencils
  int s3D_000[]{0, 0, 0};
  ops_stencil S3D_000{ops_decl_stencil(3, 1, s3D_000, "000")};

  int s3D_7pt[]{0, 0, 0, -1, 0, 0, 1,  0, 0, 0, -1,
                0, 0, 1, 0,  0, 0, -1, 0, 0, 1};
  ops_stencil S3D_7PT{ops_decl_stencil(3, 7, s3D_7pt, "3d7Point")};
  int iterRange[]{0, nx, 0, ny, 0, nz};
  int initRange[]{-1, nx + 1, -1, ny + 1, -1, nz + 1};

  // declare constants
  ops_decl_const("nx", 1, "int", &nx);
  ops_decl_const("ny", 1, "int", &ny);
  ops_decl_const("nz", 1, "int", &nz);
  ops_decl_const("left", 1, "double", &left);
  ops_decl_const("right", 1, "double", &right);
  ops_decl_const("present", 1, "double", &present);
  ops_decl_const("h", 1, "double", &h);
  // decompose the block
  ops_partition("Compact3D");

  ops_diagnostic_output();

  // initialize Tridiagonal Library
  ops_tridsolver_params *trid_ctx = new ops_tridsolver_params(compact3d);

  //-------- Initialize-----
  ops_par_loop(initKernel, "initKernel", compact3d, 3, initRange,
               ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(ux, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(uy, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(uz, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());

  ops_par_loop(preprocessX, "preprocessX", compact3d, 3, iterRange,
               ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(d, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());

  // Get the u_x, note that the solver will add result to ux so that ux must be // zero before the call
  ops_tridMultiDimBatch_Inc(3, 0, size, a, b, c, d, ux, trid_ctx);

  ops_par_loop(preprocessY, "preprocessY", compact3d, 3, iterRange,
               ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(d, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());


  // Get the u_y, note that the solver will add result to ux so that uy must be // zero before the call
  ops_tridMultiDimBatch_Inc(3, 1, size, a, b, c, d, uy, trid_ctx);

  ops_par_loop(preprocessZ, "preprocessZ", compact3d, 3, iterRange,
               ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(d, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());

  // Get the u_z, note that the solver will add result to ux so that ux must be // zero before the call
  ops_tridMultiDimBatch_Inc(3, 2, size, a, b, c, d, uz, trid_ctx);

  WriteDataToH5("Compact3D.h5", compact3d, resList);
  delete trid_ctx;

  ops_exit();
}
