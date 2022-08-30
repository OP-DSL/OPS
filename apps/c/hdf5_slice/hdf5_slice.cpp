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
#include "ops_seq_v2.h"
#include "data.h"
// Kernel functions for setting the initial conditions
#include "init_kernel.h"



// Defining the computational problem domain. As a test, we use the
// simplest grid See the document for the meaning of variables r1 and r2.
double xyzRange[2]{0, 1};
int nx{32};
int ny{32};
int nz{32};
double h{(xyzRange[1] - xyzRange[0]) / (nx - 1)};

int main(int argc, char *argv[]) {
    // OPS initialisation
    ops_init(argc, argv, 4);
    //--- OPS declarations----
    // declare block
    ops_block slice3D = ops_decl_block(3, "slice3D");
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

    ops_dat u{
        ops_decl_dat(slice3D, 1, size, base, d_m, d_p, temp, "double", "u")};

    // declare stencils
    int s3D_000[]{0, 0, 0};
    ops_stencil S3D_000{ops_decl_stencil(3, 1, s3D_000, "000")};

    int s3D_7pt[]{0, 0, 0, -1, 0, 0, 1,  0, 0, 0, -1,
                  0, 0, 1, 0,  0, 0, -1, 0, 0, 1};
    ops_stencil S3D_7PT{ops_decl_stencil(3, 7, s3D_7pt, "3d7Point")};
    int iterRange[]{0, nx, 0, ny, 0, nz};

    // declare constants
    ops_decl_const("nx", 1, "int", &nx);
    ops_decl_const("ny", 1, "int", &ny);
    ops_decl_const("nz", 1, "int", &nz);
    ops_decl_const("h", 1, "double", &h);


    // decompose the block
    ops_partition("slice3D");


    printf("\nLocal dimensions at rank %d: %d x %d x %d\n", ops_get_proc(),
           u->size[0], u->size[1], u->size[2]);
    ops_diagnostic_output();

    //-------- Initialize-------

    ops_par_loop(initKernel, "initKernel", slice3D, 3, iterRange,
                 ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_idx());

    ops_write_dataslice_hdf5("test.h5", u, 1, 16);
    // ops_fetch_block_hdf5_file(slice3D, "fetch.h5");
    // ops_fetch_dat_hdf5_file(u, "fetch.h5");
    ops_exit();
}
