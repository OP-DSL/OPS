#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    int nspec = 9;
    int nspimx = 15;
    char buf[6];

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

    ops_dat yrun[nspec], yerr[nspec], rrte[nspec], rate[nspec];
    for (int ispec = 0; ispec < nspec; ispec++) {
        strcpy(buf, "\0");
        sprintf(buf, "yrun%d\0", ispec+1);
        yrun[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);

        strcpy(buf, "\0");
        sprintf(buf, "yerr%d\0", ispec+1);
        yerr[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);

        strcpy(buf, "\0");
        sprintf(buf, "rrte%d\0", ispec+1);
        rrte[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);

        strcpy(buf, "\0");
        sprintf(buf, "rate%d\0", ispec+1);
        rate[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);
    }
    
//  Dats with halos
    size[0] = {nxglbl};    d_m[0] = {-nhalox};    d_p[0] = {nhalox};
    
    ops_dat tcoeff = ops_decl_dat(block, 6, size, base, d_m, d_p, temp, "double", "tcoeff");
    ops_dat tderiv = ops_decl_dat(block, 5, size, base, d_m, d_p, temp, "double", "tderiv");

    ops_dat itndex[2];
    itndex[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp_int, "int", "itndex1");
    itndex[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp_int, "int", "itndex2");

    ops_dat yrhs[nspec];
    for (int ispec = 0; ispec < nspec; ispec++) {
        strcpy(buf, "\0");
        sprintf(buf, "yrhs%d\0", ispec+1);
        yrhs[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);
    }

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
    ops_par_loop(equationQ, "equationQ temper 75", block, 1, iter_range,
                ops_arg_dat(tcoeff, 6, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_READ));

/*  11 july 2024 - currently not in use in senga  
    for (int icp = 0; icp < 5; icp++) {
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationB, "equationB temper 85", block, 1, iter_range,
                ops_arg_dat(tcoeff, 6, S1D_0, "double", OPS_RW));
    }
*/

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationC, "equationC temper 90", block, 1, iter_range,
                ops_arg_dat(tderiv, 5, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationA, "equationA temper 95", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE));

    for (int ispec = 0; ispec < nspec; ispec++ ) {
        int iindex = 1 + (ispec-1)/nspimx;
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationM, "equationM temper 111", block, 1, iter_range,
                    ops_arg_dat(tcoeff, 6, S1D_0, "double", OPS_INC),
                    ops_arg_dat(tderiv, 5, S1D_0, "double", OPS_RW),
                    ops_arg_dat(itndex[iindex], 1, S1D_0, "double", OPS_READ),
                    ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ));

        ops_par_loop(equationG, "equationG temper 129", block, 1, iter_range,
                    ops_arg_dat(store7, 1, S1D_0, "double", OPS_INC),
                    ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ));
    }

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationZ, "equationZ temper 144", block, 1, iter_range,
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(tcoeff, 6, S1D_0, "double", OPS_READ),
                ops_arg_dat(tderiv, 5, S1D_0, "double", OPS_READ),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_READ));

    for (int iindex = 0; iindex < 2; iindex++) {
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationA, "equationA temper 158", block, 1, iter_range,
                    ops_arg_dat(itndex[iindex], 1, S1D_0, "double", OPS_WRITE));
    }

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationA, "equationA temper 162", block, 1, iter_range,
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_WRITE));

    for (int ispec = 0; ispec < nspec; ispec++ ) { 
        int iindex = 1 + (ispec-1)/nspimx;
        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationL, "equationL temper 170", block, 1, iter_range,
                    ops_arg_dat(transp, 1, S1D_0, "double", OPS_INC),
                    ops_arg_dat(itndex[iindex], 1, S1D_0, "double", OPS_INC),
                    ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                    ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ));
    }
    
    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationO, "equationO temper 187", block, 1, iter_range,
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(prun, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhscal dfbydx 145", block, 1, iter_range,
                ops_arg_dat(urhs, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationK, "equationK rhscal 151", block, 1, iter_range,
                ops_arg_dat(divm, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationF, "equationF rhscal 246", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationH, "equationH rhscal 315", block, 1, iter_range,
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(divm, 1, S1D_0, "double", OPS_READ));
 
    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationH, "equationH rhscal 328", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhscal dfbydx 333", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));
    
    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationG, "equationG rhsvel 357", block, 1, iter_range,
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhscal dfbydx 370", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationAA, "equationAA rhscal 376", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhscal dfbydx 396", block, 1, iter_range,
                ops_arg_dat(trun, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationI, "equationI rhscal 480", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhscal d2fdx2 646", block, 1, iter_range,
                ops_arg_dat(trun, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationI, "equationI chrate 212 rhscal 696", block, 1, iter_range,
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    for(int ispec = 0; ispec < nspec; ispec++) {
        ops_par_loop(equationE, "equationE rhscal 700", block, 1, iter_range,
                ops_arg_dat(rrte[ispec], 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_READ));
    }

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationA, "equationA rhscal 788", block, 1, iter_range,
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationA, "equationA rhscal 796", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_WRITE));

    for(int ispec = 0; ispec < nspec; ispec++) {

        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationF, "equationF rhscal 892", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_RW),
                ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));
    }

    for(int ispec = 0; ispec < nspec; ispec++) {

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationJ, "equationJ rhscal 908", block, 1, iter_range,
            ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(divm, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationH, "equationH rhscal 925", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationD, "equationD rhscal dfbydx 930", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationN, "equationN rhscal 954", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationD, "equationD rhscal dfbydx 977", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationU, "equationU rhscal 1099", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        ops_par_loop(equationE, "equationE rhscal 1129", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(transp, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationJ_fused, "equationJ rhscal 1284", block, 1, iter_range,
            ops_arg_dat(ucor, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(vcor, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(wcor, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
        int iindex = 1 + (ispec-1)/nspimx;
        ops_par_loop(equationP, "equationP rhscal 1313", block, 1, iter_range,
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(trun, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(yrhs[ispec], 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(itndex[iindex], 1, S1D_0, "int", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationD, "equationD rhscal dfbydx 1416", block, 1, iter_range,
                ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationW, "equationW rhscal 1432", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store5, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store6, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationV, "equationV rhscal 1457", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store5, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store6, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationD, "equationD rhscal dfbydx 1480", block, 1, iter_range,
                ops_arg_dat(utmp, 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationV, "equationV rhscal 1511", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store5, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store6, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationE, "equationE rhscal 1617", block, 1, iter_range,
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
                ops_arg_dat(store7, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationD, "equationD rhscal d2fdx2 1630", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationT, "equationT rhscal 1649", block, 1, iter_range,
                ops_arg_dat(rate[ispec], 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationS, "equationS rhscal 1672", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ));

    } // End of ispec loop

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhscal dfbydx 2742", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationE, "equationE rhscal 2774", block, 1, iter_range,
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationY, "equationY rhscal 2793", block, 1, iter_range,
                ops_arg_dat(erhs, 1, S1D_0, "double", OPS_INC),
                ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store2, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store3, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(store4, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(ucor, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(vcor, 1, S1D_0, "double", OPS_READ),
                ops_arg_dat(wcor, 1, S1D_0, "double", OPS_READ));

   for(int ispec = 0; ispec < nspec; ispec++) { 

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationD, "equationD rhscal dfbydx 2901", block, 1, iter_range,
                ops_arg_dat(yrhs[ispec], 1, S1D_11pt, "double", OPS_READ),
                ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

        iter_range[0] = 0;   iter_range[1] = nxglbl;
        ops_par_loop(equationX, "equationX rhscal 2921", block, 1, iter_range,
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


/*  un-comment to make fall ranges within limit and avoid false dependency going from rhsvel to rhscal

#ifdef OPS_LAZY
    ops_execute();
#endif

un-comment till here */

/*  ----------------- RHSVEL --------------------- */

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationH_fused, "equationH_fused rhsvel 72", block, 1, iter_range,
            ops_arg_dat(utmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(vrhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(wrhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(drhs, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 120", block, 1, iter_range,
            ops_arg_dat(vtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 153", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 211", block, 1, iter_range,
            ops_arg_dat(wtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 235", block, 1, iter_range,
            ops_arg_dat(utmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 284", block, 1, iter_range,
            ops_arg_dat(utmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 308", block, 1, iter_range,
            ops_arg_dat(vtmp, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationH, "equationH rhsvel 341", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(utmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 348", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationH, "equationH rhsvel 361", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(vtmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 369", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store5, 1, S1D_0, "double", OPS_WRITE));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationG, "equationG rhsvel 378", block, 1, iter_range,
            ops_arg_dat(store4, 1, S1D_0, "double", OPS_INC),
            ops_arg_dat(store1, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = -nhalox;   iter_range[1] = nxglbl+nhalox;
    ops_par_loop(equationH, "equationH rhsvel 389", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_0, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S1D_0, "double", OPS_READ),
            ops_arg_dat(wtmp, 1, S1D_0, "double", OPS_READ));

    iter_range[0] = 0;   iter_range[1] = nxglbl;
    ops_par_loop(equationD, "equationD rhsvel dfbydx 397", block, 1, iter_range,
            ops_arg_dat(store7, 1, S1D_11pt, "double", OPS_READ),
            ops_arg_dat(store6, 1, S1D_0, "double", OPS_WRITE));

#ifdef OPS_LAZY
    ops_execute();
#endif

    ops_exit();
    return 0;
}
