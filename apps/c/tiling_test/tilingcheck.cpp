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
    int imax = 64;
    int nhalox = 5;

    double *temp = NULL;

//  The 1D block
    ops_block block = ops_decl_block(1, "my_grid");

    int base[] = {0};
    int size[1], d_m[1], d_p[1];

    size[0] = {imax};    d_m[0] = {-nhalox};    d_p[0] = {nhalox};
    ops_dat utmp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "utmp");
    ops_dat vtmp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vtmp");
    ops_dat wtmp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wtmp");
    ops_dat store7 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store7");
    ops_dat transp = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "transp");

    ops_dat drhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "drhs");
    ops_dat urhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "urhs");
    ops_dat vrhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "vrhs");
    ops_dat wrhs = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wrhs");

    ops_dat yrhs[2];
    yrhs[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yrhs1");
    yrhs[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "yrhs2");

    ops_dat rate[2];
    rate[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "rate1");
    rate[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "rate2");

//  Dats without halos
    size[0] = {imax};    d_m[0] = {0};    d_p[0] = {0};
    ops_dat store1 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store1");
    ops_dat store2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store2");
    ops_dat store3 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store3");
    ops_dat store4 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "store4");
    ops_dat divm = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "divm");

    int s1d_0[] = {0};
    ops_stencil S1D_0 = ops_decl_stencil(1,1,s1d_0,"0");

    int s1d_11pt[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    ops_stencil S1D_11pt = ops_decl_stencil(1,11,s1d_11pt,"11pt");

    ops_partition("");

//  ops_par_loops
    int iter_range[2];    


/*  ---------------------------- RHSCAL ---------------------------- */
    for(int ispec = 0; ispec < 2; ispec++) {

        iter_range[0] = -nhalox;   iter_range[1] = imax+nhalox;
        ops_par_loop(eqB, "eq_B rhscal 816", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqG, "eq_G rhscal 828", block, 1, iter_range,
            ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(divm, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = imax+nhalox;
        ops_par_loop(eqG1, "eq_G rhscal 845", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 850", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqA, "eq_A rhscal 874", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 897", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqA, "eq_A rhscal 983", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = imax+nhalox;
        ops_par_loop(eqB1, "eq_B rhscal 1013", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = imax+nhalox;
        ops_par_loop(eqC, "eq_C rhscal 1151", block, 1, iter_range,
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 1254", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqE, "eq_E rhscal 1270", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 1318", block, 1, iter_range,
                ops_arg_dat(utmp, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqB1, "eq_B rhscal 1455", block, 1, iter_range,
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqD, "eq_D rhscal d2fdx2 1468", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqE, "eq_E rhscal 1487", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ));

    } // End of ispec loop

    iter_range[0] = 0;   iter_range[1] = imax;
    ops_par_loop(eqD, "eq_D rhscal dfbydx 2553", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = imax;
    ops_par_loop(eqB1, "eq_B rhscal 2585", block, 1, iter_range,
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ));

   for(int ispec = 0; ispec < 2; ispec++) { 

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqD, "eq_D rhscal dfbydx 2553", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = imax;
        ops_par_loop(eqF, "eq_F rhscal 2713", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));
    }

/*  ----------------- RHSVEL --------------------- */
    iter_range[0] = -nhalox;   iter_range[1] = imax+nhalox;
    ops_par_loop(eqB1, "eq_B rhsvel 58", block, 1, iter_range,
            ops_arg_dat(utmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ));

    ops_par_loop(eqB1, "eq_B rhsvel 63", block, 1, iter_range,
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ));

    ops_par_loop(eqB1, "eq_B rhsvel 68", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ));

    ops_par_loop(eqG1, "eq_G rhsvel 159", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = imax;
    ops_par_loop(eqD, "eq_D rhsvel dfbydx 166", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

#ifdef OPS_LAZY
    ops_execute();
#endif

    ops_exit();
    return 0;
}
