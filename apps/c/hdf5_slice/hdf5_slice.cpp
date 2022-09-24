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

/**
 * @brief for testing slice output through HDF5
 *  @author Jianping Meng
 *  @details This application is to test the slice output through HDF5 when
 *           writting the full array is too expensive.
 */

#include <cmath>
#include <string>
#include <vector>
#define OPS_3D
#include "data.h"
#include "ops_seq_v2.h"
// Kernel functions for setting the initial conditions
#include "init_kernel.h"

// Defining the computational problem domain. As a test, we use the
// simplest grid See the document for the meaning of variables r1 and r2.
double xyzRange[2]{0, 1};
int nx{64};
int ny{64};
int nz{64};
double h{(xyzRange[1] - xyzRange[0]) / (nx - 1)};

int main(int argc, char *argv[]) {
  // OPS initialisation
  ops_init(argc, argv, 1);
  //--- OPS declarations----
  // declare block
  ops_block slice3Du{ops_decl_block(3, "slice3Du")};
  ops_block slice3Dv{ops_decl_block(3, "slice3Dv")};
  // declare data on blocks
  // max haloDepth depths for the dat in the positive direction
  const int haloDepth{2};
  int d_p[3]{haloDepth, haloDepth, haloDepth};
  // max haloDepth depths for the dat in the negative direction*/
  int d_m[3]{-haloDepth, -haloDepth, -haloDepth};

  // size of the dat
  int size[3]{nx, ny, nz};
  int base[3]{0, 0, 0};
  double *temp = NULL;
  int *int_tmp = NULL;

  ops_dat u{
      ops_decl_dat(slice3Du, 1, size, base, d_m, d_p, temp, "double", "u")};
  ops_dat velo{
      ops_decl_dat(slice3Du, 3, size, base, d_m, d_p, temp, "double", "u")};
  ops_dat v{
      ops_decl_dat(slice3Dv, 1, size, base, d_m, d_p, int_tmp, "int", "v")};

  // declare stencils
  int s3D_000[]{0, 0, 0};
  ops_stencil S3D_000{ops_decl_stencil(3, 1, s3D_000, "000")};

  int iterRange[]{0, nx, 0, ny, 0, nz};

  // declare constants
  ops_decl_const("nx", 1, "int", &nx);
  ops_decl_const("ny", 1, "int", &ny);
  ops_decl_const("nz", 1, "int", &nz);
  ops_decl_const("h", 1, "double", &h);

  // decompose the block
  ops_partition("slice3D");
  ops_diagnostic_output();

  //-------- Initialize-------

  ops_par_loop(initKernelU, "initKernelU", slice3Du, 3, iterRange,
               ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE), ops_arg_idx());
  ops_par_loop(initKernelvelo, "initKernelvelo", slice3Du, 3, iterRange,
               ops_arg_dat(velo, 3, S3D_000, "double", OPS_WRITE),
               ops_arg_idx());
  ops_par_loop(initKernelV, "initKernelV", slice3Dv, 3, iterRange,
               ops_arg_dat(v, 1, S3D_000, "int", OPS_WRITE), ops_arg_idx());

  double ct0, ct1, et0, et1;
  double total1{0}, total2{0}, total3{0};
  ops_timers(&ct0, &et0);
  ops_write_plane_group_hdf5({{1, 16}, {0, 1}, {2, 16}}, "1",
                             {{u, v}, {u, v}, {u, v}});
  ops_timers(&ct1, &et1);
  total1 += et1 - et0;
  ops_timers(&ct0, &et0);
  ops_write_plane_group_hdf5({{1, 16}, {0, 1}, {2, 16}}, "2",
                             {{u, v}, {u, v}, {u, v}});
  ops_timers(&ct1, &et1);
  total2 += et1 - et0;

  ops_timers(&ct0, &et0);
  ops_write_plane_group_hdf5({{1, 8}, {0, 4}, {2, 15}}, "0",
                             {{velo}, {velo}, {velo}});
  ops_timers(&ct1, &et1);
  total3 += et1 - et0;
  ops_printf("The time write 1 series is %f\n", total1);
  ops_printf("The time write 2 series is %f\n", total2);
  ops_printf("The time write velo series is %f\n", total3);
  // ops_fetch_block_hdf5_file(slice3Du, "slice3Du.h5");
  // ops_fetch_dat_hdf5_file(u, "slice3Du.h5");
  // ops_fetch_block_hdf5_file(slice3Dv, "slice3Dv.h5");
  // ops_fetch_dat_hdf5_file(v, "slice3Dv.h5");
  ops_exit();
}
