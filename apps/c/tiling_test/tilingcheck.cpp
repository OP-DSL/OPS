#include <stdlib.h>
#include <stdio.h>

//Including main OPS header file, and setting 2D
#define OPS_1D
#include <ops_seq_v2.h>
//Including applicaiton-specific "user kernels"
#include "user_kernels.h"

int main(int argc, const char** argv)
{

//  Initialise the OPS library, passing runtime args, and setting diagnostics level
    ops_init(argc, argv, 6);

//  Size along x
    int nxglbl = 64;
    int nhalox = 5;

    double *temp = NULL;
    int *temp_int = NULL;

//  The 1D block
    ops_block block = ops_decl_block(1, "my_grid");

    int base[] = {0};
    int size[1], d_m[1], d_p[1];

//  Dats without halos
    size[0] = {nxglbl};    d_m[0] = {0};    d_p[0] = {0};
    
    ops_dat store1 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store1");
    ops_dat store2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store2");
    ops_dat store3 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store3");
    ops_dat store4 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store4");
    ops_dat store5 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store5");
    ops_dat store6 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store6");
    ops_dat divm = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "divm");

    ops_dat ucor = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "ucor");
    ops_dat vcor = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vcor");
    ops_dat wcor = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wcor");

    ops_dat wd1x = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wd1x");
    ops_dat pd1x = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "pd1x");
    ops_dat td1x = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "td1x");

    ops_dat wd2x = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wd2x");
    ops_dat pd2x = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "pd2x");
    ops_dat td2x = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "td2x");

    ops_dat ufxl = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "ufxl");
    ops_dat vfxl = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vfxl");
    ops_dat wfxl = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wfxl");    

    ops_dat drun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "drun");
    ops_dat urun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "urun");
    ops_dat vrun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vrun");
    ops_dat wrun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wrun");
    ops_dat erun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "erun");

    ops_dat derr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "derr");
    ops_dat uerr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "uerr");
    ops_dat verr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "verr");
    ops_dat werr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "werr");
    ops_dat eerr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "eerr");

//  multi-dim dats without halos
    size[0] = {nxglbl};    d_m[0] = {0};    d_p[0] = {0};

    ops_dat yrun[2];
    yrun[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yrun1");
    yrun[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yrun2");

    ops_dat yerr[2];
    yerr[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yerr1");
    yerr[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yerr2");

    ops_dat rrte[2];
    rrte[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "rrte1");
    rrte[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "rrte2");

    ops_dat rate[2];
    rate[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "rate1");
    rate[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "rate2");


//  Dats with halos
    size[0] = {nxglbl};    d_m[0] = {-nhalox};    d_p[0] = {nhalox};
    
    ops_dat itndex[2];
    itndex[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp_int, "int", "itndex1");
    itndex[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp_int, "int", "itndex2");

    ops_dat yrhs[2];
    yrhs[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yrhs1");
    yrhs[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yrhs2");

    ops_dat drhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "drhs");
    ops_dat erhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "erhs");
    ops_dat urhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "urhs");
    ops_dat vrhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vrhs");
    ops_dat wrhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wrhs");


    ops_dat utmp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "utmp");
    ops_dat vtmp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vtmp");
    ops_dat wtmp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wtmp");
    ops_dat prun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "prun");
    ops_dat trun = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "trun");
    ops_dat transp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "transp");
    ops_dat store7 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store7");

//----------------------------------------------------------------------------------------------
    int s1d_0[] = {0};
    ops_stencil S1D_0 = ops_decl_stencil(1,1,s1d_0,"0");

    int s1d_11pt[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    ops_stencil S1D_11pt = ops_decl_stencil(1,11,s1d_11pt,"11pt");

//-----------------------------------------------------------------------------------------------
    ops_partition("");

/*  ---------------------------- RHSCAL ---------------------------- */
    int iter_range[2];    

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqM, "eq_M temper 143", block, 1, iter_range,
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    for (int ispec = 0; ispec < 2; ispec++ )
    { 
        int iindex = 1 + (ispec-1)/15;
        ops_par_loop(eqN, "eq_N temper 169", block, 1, iter_range,
                    ops_arg_dat(transp, 1, S1D_0, "double", OPS_RW),
                    ops_arg_dat(itndex[iindex], 1, S1D_0, "double", OPS_RW),
                    ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                    ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ));
    }

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqO, "eq_O temper 186", block, 1, iter_range,
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(prun, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqD, "eq_D rhscal dfbydx 342", block, 1, iter_range,
                ops_arg_dat(trun, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqP, "eq_P rhscal 426", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqD, "eq_D rhscal d2fdx2 571", block, 1, iter_range,
                ops_arg_dat(trun, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqP, "eq_P chrate 211 rhscal 621", block, 1, iter_range,
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ));

/*
    iter_range[0] = 0;   iter_range[1] = nxglbl;
    for(int ispec = 0; ispec < 2; ispec++) {
        ops_par_loop(eqB1, "eq_B rhscal 625", block, 1, iter_range,
                ops_arg_dat(rrte[ispec], 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_READ));
    }

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqH, "eq_H rhscal 713", block, 1, iter_range,
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqH, "eq_H rhscal 721", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_WRITE));
*/
    for(int ispec = 0; ispec < 2; ispec++) {
/*
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(eqB, "eq_B rhscal 816", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqG, "eq_G rhscal 828", block, 1, iter_range,
            ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(divm, 1, S1D_0, "double", OPS_READ));
*/
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(eqG1, "eq_G rhscal 845", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 850", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));
/*
        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqA, "eq_A rhscal 874", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 897", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqI, "eq_I rhscal 983", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(eqB1, "eq_B rhscal 1013", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqG, "eq_G rhscal 1116", block, 1, iter_range,
            ops_arg_dat(ucor, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ));

        ops_par_loop(eqG, "eq_G rhscal 1121", block, 1, iter_range,
            ops_arg_dat(vcor, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ));

        ops_par_loop(eqG, "eq_G rhscal 1126", block, 1, iter_range,
            ops_arg_dat(wcor, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));
*/
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        int iindex = 1 + (ispec-1)/2;
        ops_par_loop(eqC, "eq_C rhscal 1151", block, 1, iter_range,
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(itndex[iindex], 1, S1D_0, "int", OPS_READ));
/*
        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 1254", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqE, "eq_E rhscal 1270", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store5, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store6, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqI1, "eq_I1 rhscal 1295", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store5, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store6, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 1318", block, 1, iter_range,
                ops_arg_dat(utmp, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqI1, "eq_I1 rhscal 1349", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store5, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store6, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqB1, "eq_B rhscal 1455", block, 1, iter_range,
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqD, "eq_D rhscal d2fdx2 1468", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));
*/

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqE1, "eq_E1 rhscal 1487", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ));
/*
        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqJ, "eq_J rhscal 1510", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ));
*/
    } // End of ispec loop

/*
    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqD, "eq_D rhscal dfbydx 2553", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqB1, "eq_B rhscal 2585", block, 1, iter_range,
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqI2, "eq_I2 rhscal 2604", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(ucor, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vcor, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wcor, 1, S1D_0, "double", OPS_READ));
*/
/*
   for(int ispec = 0; ispec < 2; ispec++) { 

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 2693", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(eqF, "eq_F rhscal 2713", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(ucor, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vcor, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wcor, 1, S1D_0, "double", OPS_READ));
    }
*/

/*  un-comment to make fall ranges within limit and avoid false dependency going from rhsvel to rhscal

#ifdef OPS_LAZY
    ops_execute();
#endif

un-comment till here */

/*  ----------------- RHSVEL --------------------- */
/*
    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqK, "eq_K rhsvel 58", block, 1, iter_range,
            ops_arg_dat(utmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));
*/
    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqK, "eq_K rhsvel 63", block, 1, iter_range,
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));
/*
    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqK, "eq_K rhsvel 68", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqG1, "eq_G rhsvel 159", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqD, "eq_D rhsvel dfbydx 166", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));
*/

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqG1, "eq_G rhsvel 179", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqD, "eq_D rhsvel dfbydx 187", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store5, 1, S1D_0, "double", OPS_WRITE));

/*
    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqL, "eq_L rhsvel 196", block, 1, iter_range,
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(eqG1, "eq_G rhsvel 207", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(eqD, "eq_D rhsvel dfbydx 215", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store6, 1, S1D_0, "double", OPS_WRITE));
*/
#ifdef OPS_LAZY
    ops_execute();
#endif

    ops_exit();
    return 0;
}
