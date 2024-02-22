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

/** @brief 2D Burgers' equation solved using ADI
 *  @author Jianping Meng
 *  @details PDE is solved with the ADI (Alternating Direction Implicit) method
 *  uses the Scalar tridiagonal solver for CPU and GPU written by Endre. Lazslo
 **/
#include <cmath>

// OPS header file
#define OPS_2D

#include "data.h"
#include "ops_seq_v2.h"
#include "init_kernel.h"
#include "preproc_kernel.h"

// int opts[3], pads[3], synch;
// configure the problem and computational domain
double Re{100};
// Computational domain. As a test, we use the simplest grid
// See the document for the meaning of variables r1 and r2.
double xyRange[2]{0, 1};
int nx{256};
int ny{nx};
double dx{(xyRange[1] - xyRange[0]) / (nx - 1)};
double dy{(xyRange[1] - xyRange[0]) / (ny - 1)};
double h{dx};
double dt{0.0001};
double r1{dt / (4 * h)};
double r2{dt / (2 * Re * h * h)};
int iter{1};
int main(int argc, const char **argv) {
    /**--- Initialisation----**/
    // OPS initialisation
    ops_init(argc, argv, 2);

    /**--- OPS declarations----**/
    // declare block
    ops_block burger2D = ops_decl_block(2, "Burger2D");
    // define the number of halo points
    const int haloDepth{1};

    // declare data on blocks
    // max halo depths for the dat in the possitive direction
    int d_p[2]{haloDepth, haloDepth};
    // max halo depths for the dat in the negative direction
    int d_m[2]{-haloDepth, -haloDepth};
    // size of the dat -identical to the block where it is defined on
    int size[2]{nx, ny};
    int pads[2]{nx, ny};
    int base[2]{0, 0};
    double *temp{nullptr};

    ops_dat u{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "u")};
    ops_dat uStar{ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp,
                               "double", "uStar")};
    ops_dat v{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "v")};
    ops_dat vStar{ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp,
                               "double", "vStar")};
    ops_dat a{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "a")};

    ops_dat b{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "b")};

    ops_dat c{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "c")};

    ops_dat du{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "du")};
    ops_dat dv{
        ops_decl_dat(burger2D, 1, size, base, d_m, d_p, temp, "double", "dv")};

    // declare stencils the local point
    int s2D_00[]{0, 0};
    ops_stencil S2D_00{ops_decl_stencil(2, 1, s2D_00, "00")};
    // declare stencils for the central differencing
    int s2D_5pt[]{0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
    ops_stencil S2D_5PT{ops_decl_stencil(2, 5, s2D_5pt, "2d5Point")};

    // declare constants
    ops_decl_const("nx", 1, "int", &nx);
    ops_decl_const("ny", 1, "int", &ny);
    ops_decl_const("Re", 1, "double", &Re);
    ops_decl_const("h", 1, "double", &h);
    ops_decl_const("r1", 1, "double", &r1);
    ops_decl_const("r2", 1, "double", &r2);

    // decompose the block
    ops_partition("2D_BLOCK_DECOMPSE");

    double ct0, ct1, et0, et1, ct2, et2, ct3, et3;

    ops_printf("\nGrid dimensions: %d x %d\n", nx, ny);
    ops_diagnostic_output();

    /**-------- Initialize-------**/
    int iterRange[]{0, nx, 0, ny};
    // int iterRangeWithHalo[]{-haloDepth, nx + haloDepth, -haloDepth,
    //                         ny + haloDepth};
    // // Without corner points
    // int iterTopHalo[]{0, nx, ny, ny + haloDepth};
    // int iterBotHalo[]{0, nx, -haloDepth, 0};
    // // Considering corner points
    // int iterLeftHalo[]{-haloDepth, 0, -haloDepth, ny + haloDepth};
    // int iterRightHalo[]{nx, nx + haloDepth, -haloDepth, ny + haloDepth};

    ops_par_loop(initKernel, "initKernel", burger2D, 2, iterRange,
                 ops_arg_dat(u, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(v, 1, S2D_00, "double", OPS_WRITE), ops_arg_idx());
    ops_fetch_block_hdf5_file(burger2D, "adi_burger2D_init.h5");
    ops_fetch_dat_hdf5_file(u, "adi_burger2D_init.h5");
    ops_fetch_dat_hdf5_file(v, "adi_burger2D_init.h5");

    ops_timers(&ct0, &et0);

    ops_tridsolver_params *trid_ctx = new ops_tridsolver_params(burger2D);

    for (int it = 1; it <= iter; it++) {  // Start main iteration loop
        double time{it * dt};
        ops_timers(&ct2, &et2);
        ops_par_loop(preprocessX, "preprocessX", burger2D, 2, iterRange,
                     ops_arg_dat(u, 1, S2D_5PT, "double", OPS_READ),
                     ops_arg_dat(v, 1, S2D_5PT, "double", OPS_READ),
                     ops_arg_gbl(&time, 1, "double", OPS_READ),
                     ops_arg_dat(a, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(b, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(c, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(du, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(dv, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(uStar, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(vStar, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_idx());
        ops_timers(&ct3, &et3);
        ops_printf("Elapsed preproc X (sec): %lf (s)\n", et3 - et2);
        ops_fetch_block_hdf5_file(burger2D, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(u, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(v, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(a, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(b, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(c, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(du, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(dv, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(uStar, "adi_burger2D_proc.h5");
        ops_fetch_dat_hdf5_file(vStar, "adi_burger2D_proc.h5");

        /**---- perform tri-diagonal solves in x-direction--**/
        ops_timers(&ct2, &et2);
        ops_tridMultiDimBatch_Inc(2, 0, iterRange, a, b, c, du, uStar, trid_ctx);
        ops_tridMultiDimBatch_Inc(2, 0, iterRange, a, b, c, dv, vStar, trid_ctx);
        ops_fetch_block_hdf5_file(burger2D, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(u, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(v, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(a, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(b, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(c, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(du, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(dv, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(uStar, "adi_burger2D_X.h5");
        ops_fetch_dat_hdf5_file(vStar, "adi_burger2D_X.h5");
        ops_timers(&ct3, &et3);
        ops_printf("Elapsed trid_x (sec): %lf (s)\n", et3 - et2);

        ops_timers(&ct2, &et2);
        ops_par_loop(preprocessY, "preprocessY", burger2D, 2, iterRange,
                     ops_arg_dat(uStar, 1, S2D_5PT, "double", OPS_READ),
                     ops_arg_dat(vStar, 1, S2D_5PT, "double", OPS_READ),
                     ops_arg_gbl(&time, 1, "double", OPS_READ),
                     ops_arg_dat(a, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(b, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(c, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(du, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(dv, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(u, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(v, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_idx());
        ops_timers(&ct3, &et3);
        ops_printf("Elapsed preproc Y (sec): %lf (s)\n", et3 - et2);

        /**---- perform tri-diagonal solves in y-direction--**/
        ops_timers(&ct2, &et2);
        ops_tridMultiDimBatch_Inc(2, 1, iterRange, a, b, c, du, u, trid_ctx);
        ops_tridMultiDimBatch_Inc(2, 1, iterRange, a, b, c, dv, v, trid_ctx);
        ops_timers(&ct3, &et3);
        ops_printf("Elapsed trid_y (sec): %lf (s)\n", et3 - et2);
        ops_printf("Finish time=%lf\n", time);
    }  // End main iteration loop

    delete trid_ctx;

    ops_timers(&ct1, &et1);

    /**---- dump solution to HDF5 file with OPS-**/
    ops_fetch_block_hdf5_file(burger2D, "adi_burger2D.h5");
    ops_fetch_dat_hdf5_file(u, "adi_burger2D.h5");
    ops_fetch_dat_hdf5_file(v, "adi_burger2D.h5");

    ops_printf("\nTotal Wall time %lf\n", et1 - et0);
    ops_printf("\nSuccessful exit from OPS!\n");
    ops_exit();
}
