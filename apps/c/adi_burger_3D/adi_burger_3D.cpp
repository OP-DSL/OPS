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

/** @brief 3D Burgers' equation solved using ADI and explicit schemes
 *  @author Jianping Meng
 *  @details The 3D Burgers' equation is solved with the ADI
 * (Alternating Direction Implicit) method and then compared
 * with explicit schemes
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

double Re{1};
// Defining the computational problem domain. As a test, we use the
// simplest grid See the document for the meaning of variables r1 and r2.
double xyzRange[2]{0, 1};
int nx{64};
int ny{nx};
int nz{nx};
double dx{(xyzRange[1] - xyzRange[0]) / (nx - 1)};
double dy{(xyzRange[1] - xyzRange[0]) / (ny - 1)};
double dz{(xyzRange[1] - xyzRange[0]) / (nz - 1)};
double h{dx};
double dt{0.0001};
double r1{dt / (4 * h)};
double r2{dt / (2 * Re * h * h)};
int iter{100};
// Utility functions mainly debug
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
    ops_block burger3D = ops_decl_block(3, "Burger3D");
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
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "u")};
    ops_dat v{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "v")};
    ops_dat w{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "w")};

    ops_dat uStar{ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp,
                               "double", "uStar")};
    ops_dat vStar{ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp,
                               "double", "vStar")};
    ops_dat wStar{ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp,
                               "double", "wStar")};

    ops_dat a{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "a")};
    ops_dat b{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "b")};
    ops_dat c{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "c")};

    ops_dat du{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "du")};
    ops_dat dv{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "dv")};
    ops_dat dw{
        ops_decl_dat(burger3D, 1, size, base, d_m, d_p, temp, "double", "dw")};

    const std::vector<ops_dat> resList{u, v, w};
    const std::vector<ops_dat> debugList{u,     v,     w,     du, dv, dw,
                                         uStar, vStar, wStar, a,  b,  c};

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
    ops_decl_const("Re", 1, "double", &Re);
    ops_decl_const("h", 1, "double", &h);
    ops_decl_const("r1", 1, "double", &r1);
    ops_decl_const("r2", 1, "double", &r2);

    // decompose the block
    ops_partition("BurgersEquation3D");

    double ct0, ct1, et0, et1, ct2, et2, ct3, et3;
    double totalPreprocX{0}, totalPreprocY{0}, totalPreprocZ{0}, totalX{0},
        totalY{0}, totalZ{0};

    ops_printf("\nNumber of iterations: %d\n", iter);
    ops_printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);
    ops_printf("\nh=%f r1=%f r2=%f\n", h, r1, r2);
    printf("\nLocal dimensions at rank %d: %d x %d x %d\n", ops_get_proc(),
           u->size[0], u->size[1], u->size[2]);
    ops_diagnostic_output();

    /**-------- Initialize-------**/

    ops_par_loop(initKernel, "initKernel", burger3D, 3, iterRange,
                 ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(v, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(w, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_idx());

    ops_tridsolver_params *trid_ctx = new ops_tridsolver_params(burger3D);

    ops_timers(&ct0, &et0);
    for (int it = 1; it <= iter; it++) {
        double time{it * dt};

        ops_par_loop(CopyUVW, "CopyUVW", burger3D, 3, iterRange,
                     ops_arg_dat(u, 1, S3D_000, "double", OPS_READ),
                     ops_arg_dat(v, 1, S3D_000, "double", OPS_READ),
                     ops_arg_dat(w, 1, S3D_000, "double", OPS_READ),
                     ops_arg_dat(uStar, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(vStar, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(wStar, 1, S3D_000, "double", OPS_WRITE));
        // Calculate a,b,c,du,dv,dw for the X-direction
        ops_timers(&ct2, &et2);
        ops_par_loop(preprocessX, "preprocessX", burger3D, 3, iterRange,
                     ops_arg_dat(uStar, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_dat(vStar, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_dat(wStar, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_gbl(&time, 1, "double", OPS_READ),
                     ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(du, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(dv, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(dw, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(v, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(w, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_idx());
        ops_timers(&ct3, &et3);
        totalPreprocX += et3 - et2;
        // WriteDataToH5("Burger3DProcX.h5", burger3D, debugList);
        // ops_printf("Elapsed preproc (sec): %lf (s)\n", et3 - et2);
        // perform tri-diagonal solves in x-direction
        ops_timers(&ct2, &et2);
        ops_tridMultiDimBatch_Inc(3, 0, iterRange, a, b, c, du, u, trid_ctx);
        ops_tridMultiDimBatch_Inc(3, 0, iterRange, a, b, c, dv, v, trid_ctx);
        ops_tridMultiDimBatch_Inc(3, 0, iterRange, a, b, c, dw, w, trid_ctx);
        ops_timers(&ct3, &et3);
        totalX += et3 - et2;
        // WriteDataToH5("Burger3DX.h5", burger3D, debugList);
        // Calculate a,b,c,du,dv,dw for the Y-direction
        ops_timers(&ct2, &et2);
        ops_par_loop(preprocessY, "preprocessY", burger3D, 3, iterRange,
                     ops_arg_dat(u, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_dat(v, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_dat(w, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_gbl(&time, 1, "double", OPS_READ),
                     ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(du, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(dv, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(dw, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(uStar, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(vStar, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(wStar, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_idx());
        ops_timers(&ct3, &et3);
        totalPreprocY += et3 - et2;
        // WriteDataToH5("Burger3DProcY.h5", burger3D, debugList);
        // perform tri-diagonal solves in y-direction
        ops_timers(&ct2, &et2);
        ops_tridMultiDimBatch_Inc(3, 1, iterRange, a, b, c, du, uStar, trid_ctx);
        ops_tridMultiDimBatch_Inc(3, 1, iterRange, a, b, c, dv, vStar, trid_ctx);
        ops_tridMultiDimBatch_Inc(3, 1, iterRange, a, b, c, dw, wStar, trid_ctx);
        ops_timers(&ct3, &et3);
        totalY += et3 - et2;
        // WriteDataToH5("Burger3DY.h5", burger3D, debugList);
        // Calculate a,b,c,du,dv,dw for the Z-direction
        ops_timers(&ct2, &et2);
        ops_par_loop(preprocessZ, "preprocessZ", burger3D, 3, iterRange,
                     ops_arg_dat(uStar, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_dat(vStar, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_dat(wStar, 1, S3D_7PT, "double", OPS_READ),
                     ops_arg_gbl(&time, 1, "double", OPS_READ),
                     ops_arg_dat(a, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(b, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(c, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(du, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(dv, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(dw, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(v, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_dat(w, 1, S3D_000, "double", OPS_WRITE),
                     ops_arg_idx());
        ops_timers(&ct3, &et3);
        totalPreprocZ += et3 - et2;
        // WriteDataToH5("Burger3DProcZ.h5", burger3D, debugList);
        // ops_printf("Elapsed preproc (sec): %lf (s)\n", et3 - et2);
        // perform tri-diagonal solves in Z-direction
        ops_timers(&ct2, &et2);
        ops_tridMultiDimBatch_Inc(3, 2, iterRange, a, b, c, du, u, trid_ctx);
        ops_tridMultiDimBatch_Inc(3, 2, iterRange, a, b, c, dv, v, trid_ctx);
        ops_tridMultiDimBatch_Inc(3, 2, iterRange, a, b, c, dw, w, trid_ctx);
        ops_timers(&ct3, &et3);
        totalZ += et3 - et2;
        //WriteDataToH5("Burger3DZ.h5", burger3D, debugList);
    }  // End main iteration loop
    ops_timers(&ct1, &et1);

    delete trid_ctx;

    const double time{iter * dt};
    double uSqrSum, vSqrSum, wSqrSum, uDiffSqrSum, vDiffSqrSum, wDiffSqrSum;
    ops_reduction uSqr;
    //, vSqr, wSqr, uDiffSqr, vDiffSqr, wDiffSqr;
    //ops_printf("Calculating the error at time %f!\n", time);
    // ops_par_loop(calculateL2NormError, "calculateL2NormError", burger3D, 3,
    //              iterRange, ops_arg_dat(u, 1, S3D_000, "double", OPS_READ),
    //              ops_arg_dat(v, 1, S3D_000, "double", OPS_READ),
    //              ops_arg_dat(w, 1, S3D_000, "double", OPS_READ),
    //              ops_arg_gbl(&time, 1, "double", OPS_READ),
    //              ops_arg_reduce(uSqr, 1, "double", OPS_INC),
    //              ops_arg_reduce(vSqr, 1, "double", OPS_INC),
    //              ops_arg_reduce(wSqr, 1, "double", OPS_INC),
    //              ops_arg_reduce(uDiffSqr, 1, "double", OPS_INC),
    //              ops_arg_reduce(vDiffSqr, 1, "double", OPS_INC),
    //              ops_arg_reduce(wDiffSqr, 1, "double", OPS_INC),
    //              ops_arg_idx());

    // ops_printf("Reducing...!\n");
    //ops_reduction_result(uSqr, &uSqrSum);
    // ops_reduction_result(vSqr, &vSqrSum);
    // ops_reduction_result(wSqr, &wSqrSum);
    // ops_reduction_result(uDiffSqr, &uDiffSqrSum);
    // ops_reduction_result(vDiffSqr, &vDiffSqrSum);
    // ops_reduction_result(wDiffSqr, &wDiffSqrSum);
    WriteDataToH5("Burger3DRes.h5", burger3D, resList);
    ops_printf("\nTotal Wall time (s): %lf\n", et1 - et0);
    ops_printf("Preproc total time at X (s): %lf\n", totalPreprocX);
    ops_printf("Preproc total time at Y (s): %lf\n", totalPreprocY);
    ops_printf("Preproc total time at Z (s): %lf\n", totalPreprocZ);
    ops_printf("X Dim total time (s): %lf\n", totalX);
    ops_printf("Y Dim total time (s): %lf\n", totalY);
    ops_printf("Z Dim total time (s): %lf\n", totalZ);
    // ops_printf("Error at U: %lf\n", sqrt(uDiffSqrSum / uSqrSum));
    // ops_printf("Error at V: %lf\n", sqrt(vDiffSqrSum / vSqrSum));
    // ops_printf("Error at w: %lf\n", sqrt(wDiffSqrSum / wSqrSum));
    ops_printf("\nSuccessful exit from OPS!\n");
    ops_exit();
    return 0;
}
