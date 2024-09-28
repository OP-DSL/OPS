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

//  Size along x,y,z
    int nxglbl = 504, nyglbl = 252, nzglbl = 252;
    int nhalox = 5, nhaloy = 5, nhaloz = 5;
    int nspec = 9;
    int nspimx = 15;
    int nsnull = 0, nsperi = 1;
    int nsbci1 = 11, nsbci2 = 12, nsbci3 = 13, nsbci4 = 14;
    int nsbco1 = 21, nsbco2 = 22, nsbco3 = 23, nsbco4 = 24;
    int nsbcw1 = 31, nsbcw2 = 32, nsbcw3 = 33, nsbcw4 = 34;

    int nsbcxl = 21, nsbcxr = 21, nsbcyl = 1, nsbcyr = 1, nsbczl = 1, nsbczr = 1;

    char buf[6];

//  Flags controlling loop sequence
    int flmavt = 1;
    int fxlcnv = 1, fxrcnv = 1;
    int flmixp = 1;

    double *temp = NULL;
    int *temp_int = NULL;

//  The 1D block
    ops_block block = ops_decl_block(3, "my_grid");

    int base[3] = {0,0,0};
    int size[3], d_m[3], d_p[3];

//  Dats without halos
    size[0] = {nxglbl};     size[1] = {nyglbl};     size[2] = {nzglbl};
    d_m[0] = {0};           d_m[1] = {0};           d_m[2] = {0};
    d_p[0] = {0};           d_p[1] = {0};           d_p[2] = {0};
    
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

    ops_dat pd1y = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "pd1y");

    ops_dat pd2y = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "pd2y");

    ops_dat pd1z = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "pd1z");

    ops_dat pd2z = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "pd2z");

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
    size[0] = {nxglbl};     size[1] = {nyglbl};     size[2] = {nzglbl};
    d_m[0] = {0};           d_m[1] = {0};           d_m[2] = {0};
    d_p[0] = {0};           d_p[1] = {0};           d_p[2] = {0}; 

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
    size[0] = {nxglbl};     size[1] = {nyglbl};     size[2] = {nzglbl};
    d_m[0] = {-nhalox};     d_m[1] = {-nhaloy};     d_m[2] = {-nhaloz};
    d_p[0] = {nhalox};      d_p[1] = {nhaloy};      d_p[2] = {nhaloz};
    
    ops_dat tcoeff = ops_decl_dat(block, 6, size, base, d_m, d_p, temp, "double", "tcoeff");
    ops_dat tderiv = ops_decl_dat(block, 5, size, base, d_m, d_p, temp, "double", "tderiv");

    ops_dat itndex[2];
    itndex[0] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp_int, "int", "itndex1");
    itndex[1] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp_int, "int", "itndex2");

    ops_dat yrhs[nspec], ctrans[nspec];
    for (int ispec = 0; ispec < nspec; ispec++) {
        strcpy(buf, "\0");
        sprintf(buf, "yrhs%d\0", ispec+1);
        yrhs[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);

        strcpy(buf, "\0");
        sprintf(buf, "ctrans%d\0", ispec+1);
        ctrans[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);
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

    ops_dat wmomix = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "wmomix");
    ops_dat difmix = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "difmix");

    ops_dat combo1 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "combo1");
    ops_dat combo2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "combo2");
    ops_dat combo3 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "combo3");

//  Edge dataset
    size[0] = {1};     size[1] = {nyglbl};     size[2] = {nzglbl};
    d_m[0] = {0};           d_m[1] = {0};           d_m[2] = {0};
    d_p[0] = {0};           d_p[1] = {0};           d_p[2] = {0};

    ops_dat strdxl = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "strdxl");
    ops_dat bcl2xl = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "bcl2xl");
    ops_dat strdxr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "strdxr");
    ops_dat bcl2xr = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "bcl2xr");

    ops_dat ratexl[nspec], ratexr[nspec];
    for (int ispec = 0; ispec < nspec; ispec++) {
        strcpy(buf, "\0");
        sprintf(buf, "ratexl%d\0", ispec+1);
        ratexl[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);

        strcpy(buf, "\0");
        sprintf(buf, "ratexr%d\0", ispec+1);
        ratexr[ispec] = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", buf);
    }

//----------------------------------------------------------------------------------------------
    int s3d_000[] = {0,0,0};
    ops_stencil S3D_000 = ops_decl_stencil(3,1,s3d_000,"0");

    int s3d_p500_to_m500_x[] = {5,0,0, 4,0,0, 3,0,0, 2,0,0, 1,0,0, 0,0,0, -1,0,0, -2,0,0, -3,0,0, -4,0,0, -5,0,0};
    ops_stencil S3D_p500_to_m500_x = ops_decl_stencil(3,11,s3d_p500_to_m500_x,"5,0,0 to -5,0,0");

    int s3d_p050_to_m050_y[] = {0,5,0, 0,4,0, 0,3,0, 0,2,0, 0,1,0, 0,0,0, 0,-1,0, 0,-2,0, 0,-3,0, 0,-4,0, 0,-5,0};
    ops_stencil S3D_p050_to_m050_y = ops_decl_stencil(3,11,s3d_p050_to_m050_y,"0,5,0 to 0,-5,0");

    int s3d_p005_to_m005_z[] = {0,0,5, 0,0,4, 0,0,3, 0,0,2, 0,0,1, 0,0,0, 0,0,-1, 0,0,-2, 0,0,-3, 0,0,-4, 0,0,-5};
    ops_stencil S3D_p005_to_m005_z = ops_decl_stencil(3,11,s3d_p005_to_m005_z,"0,0,5 to 0,0,-5");


//-----------------------------------------------------------------------------------------------
    ops_partition("");

/*  ---------------------------- RHSCAL ---------------------------- */
    int extended_iter_range[] = {-nhalox,nxglbl+nhalox,-nhaloy,nyglbl+nhaloy,-nhaloz,nzglbl+nhaloz};
    int internal_iter_range[] = {0,nxglbl,0,nyglbl,0,nzglbl};

    int temper_iter_range[] = {-nhalox,nxglbl+nhalox,-nhaloy,nyglbl+nhaloy,-nhaloz,nzglbl+nhaloz};

    // validates to true currently due to nsbcxl == nsbco1
    if(nsbcxl == nsbco1 || nsbcxl == nsbci1 || nsbcxl == nsbci2 ||
       nsbcxl == nsbci3 || nsbcxl == nsbcw1 || nsbcxl == nsbcw2)  temper_iter_range[0] = 1;

    // validates to true currently due to nsbcxr == nsbco1
    if (nsbcxr == nsbco1 || nsbcxr == nsbci1 || nsbcxr == nsbci2 || 
        nsbcxr == nsbci3 || nsbcxr == nsbcw1 || nsbcxr == nsbcw2) temper_iter_range[1] = nxglbl;

    if (nsbcyl == nsbco1 || nsbcyl == nsbci1 || nsbcyl == nsbci2 || 
        nsbcyl == nsbci3 || nsbcyl == nsbcw1 || nsbcyl == nsbcw2) temper_iter_range[2] = 1;

    if (nsbcyr == nsbco1 || nsbcyr == nsbci1 || nsbcyr == nsbci2 || 
        nsbcyr == nsbci3 || nsbcyr == nsbcw1 || nsbcyr == nsbcw2) temper_iter_range[3] = nyglbl;

    if (nsbczl == nsbco1 || nsbczl == nsbci1 || nsbczl == nsbci2 || 
        nsbczl == nsbci3 || nsbczl == nsbcw1 || nsbczl == nsbcw2) temper_iter_range[4] = 1;

    if (nsbczr == nsbco1 || nsbczr == nsbci1 || nsbczr == nsbci2 || 
        nsbczr == nsbci3 || nsbczr == nsbcw1 || nsbczr == nsbcw2) temper_iter_range[5] = nzglbl;

    ops_par_loop(equationQ, "equationQ temper 75", block, 1, temper_iter_range,
            ops_arg_dat(tcoeff, 6, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationC, "equationC temper 90", block, 1, temper_iter_range,
            ops_arg_dat(tderiv, 5, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationA, "equationA temper 95", block, 1, temper_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE));

    for (int ispec = 0; ispec < nspec; ispec++ ) {
        int iindex = 1 + (ispec-1)/nspimx;
        ops_par_loop(equationM, "equationM temper 111", block, 1, temper_iter_range,
                ops_arg_dat(tcoeff, 6, S3D_000, "double", OPS_INC),
                ops_arg_dat(tderiv, 5, S3D_000, "double", OPS_RW),
                ops_arg_dat(itndex[iindex], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationG, "equationG temper 129", block, 1, temper_iter_range,
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ));
    }

    ops_par_loop(equationZ, "equationZ temper 144", block, 1, temper_iter_range,
            ops_arg_dat(trun, 1, S3D_000, "double", OPS_RW),
            ops_arg_dat(tcoeff, 6, S3D_000, "double", OPS_READ),
            ops_arg_dat(tderiv, 5, S3D_000, "double", OPS_READ),
            ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_READ));

    for (int iindex = 0; iindex < 2; iindex++) {
        ops_par_loop(equationA, "equationA temper 158", block, 1, temper_iter_range,
                ops_arg_dat(itndex[iindex], 1, S3D_000, "double", OPS_WRITE));
    }

    ops_par_loop(equationA, "equationA temper 162", block, 1, temper_iter_range,
                ops_arg_dat(transp, 1, S3D_000, "double", OPS_WRITE));

    for (int ispec = 0; ispec < nspec; ispec++ ) { 
        int iindex = 1 + (ispec-1)/nspimx;
        ops_par_loop(equationL, "equationL temper 170", block, 1, temper_iter_range,
                ops_arg_dat(transp, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(itndex[iindex], 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(trun, 1, S3D_000, "double", OPS_READ));
    }
    
    ops_par_loop(equationO, "equationO temper 187", block, 1, temper_iter_range,
            ops_arg_dat(transp, 1, S3D_000, "double", OPS_RW),
            ops_arg_dat(prun, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(trun, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDX, "equationDX rhscal dfbydx 146", block, 1, internal_iter_range,
            ops_arg_dat(urhs, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhscal dfbydy 147", block, 1, internal_iter_range,
            ops_arg_dat(vrhs, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 148", block, 1, internal_iter_range,
            ops_arg_dat(wrhs, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationK, "equationK rhscal 151", block, 1, internal_iter_range,
            ops_arg_dat(divm, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationF, "equationF rhscal 246", block, 1, temper_iter_range,
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_RW),
            ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationH, "equationH rhscal 313", block, 1, internal_iter_range,
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(divm, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationH, "equationH rhscal 328", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDX, "equationDX rhscal dfbydx 333", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationH, "equationH rhscal 338", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDY, "equationDY rhscal dfbydy 343", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationH, "equationH rhscal 348", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 353", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationN, "equationN rhsvel 357", block, 1, internal_iter_range,
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_INC),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDX, "equationDX rhscal dfbydx 370", block, 1, internal_iter_range,
            ops_arg_dat(erhs, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhscal dfbydy 371", block, 1, internal_iter_range,
            ops_arg_dat(erhs, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 372", block, 1, internal_iter_range,
            ops_arg_dat(erhs, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationAA, "equationAA rhscal 376", block, 1, internal_iter_range,
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDX, "equationDX rhscal dfbydx 396", block, 1, internal_iter_range,
            ops_arg_dat(trun, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhscal dfbydy 397", block, 1, internal_iter_range,
            ops_arg_dat(trun, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 398", block, 1, internal_iter_range,
            ops_arg_dat(trun, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationI, "equationI rhscal 480", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(transp, 1, S3D_000, "double", OPS_RW),
            ops_arg_dat(trun, 1, S3D_000, "double", OPS_READ));

    if(flmavt) {
        ops_par_loop(equationE, "equationE rhscal 494", block, 1, extended_iter_range,
                ops_arg_dat(transp, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(trun, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationA, "equationA rhscal 499", block, 1, extended_iter_range,
                ops_arg_dat(combo1, 1, S3D_000, "double", OPS_WRITE));
        ops_par_loop(equationA, "equationA rhscal 501", block, 1, extended_iter_range,
                ops_arg_dat(combo2, 1, S3D_000, "double", OPS_WRITE));
        ops_par_loop(equationA, "equationA rhscal 503", block, 1, extended_iter_range,
                ops_arg_dat(combo3, 1, S3D_000, "double", OPS_WRITE));
        for (int ispec = 0; ispec < nspec; ispec++ ) {
            ops_par_loop(equationAB, "equationAB rhscal 508", block, 1, extended_iter_range,
                    ops_arg_dat(combo1, 1, S3D_000, "double", OPS_RW),
                    ops_arg_dat(combo2, 1, S3D_000, "double", OPS_RW),
                    ops_arg_dat(combo3, 1, S3D_000, "double", OPS_RW),
                    ops_arg_dat(transp, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ));
        }

        ops_par_loop(equationAC, "equationAC rhscal 521", block, 1, extended_iter_range,
                ops_arg_dat(combo1, 1, S3D_000, "double", OPS_RW),
                ops_arg_dat(combo2, 1, S3D_000, "double", OPS_RW),
                ops_arg_dat(combo3, 1, S3D_000, "double", OPS_RW),
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(wmomix, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ));
    }

    ops_par_loop(equationDX, "equationDX rhscal dfbydx 558", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 559", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store5, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 560", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store6, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationAD, "equationS rhscal 577", block, 1, internal_iter_range,
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store5, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store6, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDX, "equationDX rhscal d2fdx2 654", block, 1, internal_iter_range,
            ops_arg_dat(trun, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhscal d2fdy2 655", block, 1, internal_iter_range,
            ops_arg_dat(trun, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhscal d2fdz2 656", block, 1, internal_iter_range,
            ops_arg_dat(trun, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationI, "equationI chrate 212 rhscal 696", block, 1, internal_iter_range,
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(trun, 1, S3D_000, "double", OPS_READ));

    for(int ispec = 0; ispec < nspec; ispec++) {
        ops_par_loop(equationE, "equationE rhscal 708", block, 1, internal_iter_range,
                ops_arg_dat(rrte[ispec], 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_READ));
    }


/*  needs correction due to converting test app from 1D to 3D  
    if(fxlcnv) {
        internal_iter_range[0] = 0;   internal_iter_range[1] = 1;
        for(int ispec = 0; ispec < nspec; ispec++) {
            ops_par_loop(equationE, "equationE rhscal 724", block, 1, internal_iter_range,
                ops_arg_dat(ratexl[ispec], 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_READ));
        }
    }

    if(fxrcnv) {
        internal_iter_range[0] = nxglbl-1;   internal_iter_range[1] = nxglbl;
        for(int ispec = 0; ispec < nspec; ispec++) {
            ops_par_loop(equationE, "equationE rhscal 733", block, 1, internal_iter_range,
                ops_arg_dat(ratexr[ispec], 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_READ));
        }
    }
*/

    ops_par_loop(equationA, "equationA rhscal 787", block, 1, internal_iter_range,
            ops_arg_dat(ucor, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationA, "equationA rhscal 790", block, 1, internal_iter_range,
            ops_arg_dat(vcor, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationA, "equationA rhscal 7963", block, 1, internal_iter_range,
            ops_arg_dat(wcor, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationA, "equationA rhscal 796", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationA, "equationA rhscal 804", block, 1, extended_iter_range,
            ops_arg_dat(wtmp, 1, S3D_000, "double", OPS_WRITE));

    for(int ispec = 0; ispec < nspec; ispec++) {
        ops_par_loop(equationF, "equationF rhscal 900", block, 1, extended_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_RW),
                ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ));
    }

    for(int ispec = 0; ispec < nspec; ispec++) {

        ops_par_loop(equationJ, "equationJ rhscal 916", block, 1, internal_iter_range,
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(divm, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationH, "equationH rhscal 933", block, 1, extended_iter_range,
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationDX, "equationDX rhscal dfbydx 938", block, 1, internal_iter_range,
                ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationH, "equationH rhscal 943", block, 1, extended_iter_range,
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationDY, "equationDY rhscal dfbydy 948", block, 1, internal_iter_range,
                ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE)); 

        ops_par_loop(equationH, "equationH rhscal 953", block, 1, extended_iter_range,
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 958", block, 1, internal_iter_range,
                ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationN, "equationN rhscal 962", block, 1, internal_iter_range,
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationDX, "equationDX rhscal dfbydx 985", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDY, "equationDY rhscal dfbydy 986", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p050_to_m050_y, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 987", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p005_to_m005_z, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationU, "equationU rhscal 1107", block, 1, internal_iter_range,
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationE, "equationE rhscal 1137", block, 1, extended_iter_range,
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(transp, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationJ_fused, "equationJ rhscal 1297", block, 1, internal_iter_range,
                ops_arg_dat(ucor, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(vcor, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(wcor, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ));

        int iindex = 1 + (ispec-1)/nspimx;
        ops_par_loop(equationP, "equationP rhscal 1326", block, 1, extended_iter_range,
                ops_arg_dat(utmp, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(wtmp, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(trun, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(itndex[iindex], 1, S3D_000, "int", OPS_READ));

        ops_par_loop(equationDX, "equationDX rhscal dfbydx 1429", block, 1, internal_iter_range,
                ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDY, "equationDY rhscal dfbydy 1430", block, 1, internal_iter_range,
                ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
                ops_arg_dat(store5, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 1431", block, 1, internal_iter_range,
                ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
                ops_arg_dat(store6, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationW, "equationW rhscal 1445", block, 1, internal_iter_range,
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store5, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store6, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationV, "equationV rhscal 1470", block, 1, internal_iter_range,
                ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(utmp, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store5, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store6, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationDX, "equationDX rhscal dfbydx 1493", block, 1, internal_iter_range,
                ops_arg_dat(utmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDY, "equationDY rhscal dfbydy 1494", block, 1, internal_iter_range,
                ops_arg_dat(utmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
                ops_arg_dat(store5, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 1495", block, 1, internal_iter_range,
                ops_arg_dat(utmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
                ops_arg_dat(store6, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationV, "equationV rhscal 1524", block, 1, internal_iter_range,
                ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store5, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store6, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationE, "equationE rhscal 1630", block, 1, internal_iter_range,
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationDX, "equationDX rhscal d2fdx2 1643", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDY, "equationDY rhscal d2fdy2 1644", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p050_to_m050_y, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDZ, "equationDZ rhscal d2fdz2 1645", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p005_to_m005_z, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationT, "equationT rhscal 1662", block, 1, internal_iter_range,
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ));

        ops_par_loop(equationS, "equationS rhscal 1685", block, 1, internal_iter_range,
                ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
                ops_arg_dat(utmp, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ));

        if( flmixp ) {

            ops_par_loop(equationAG, "equationAG rhscal 1981", block, 1, extended_iter_range,
                    ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
                    ops_arg_dat(difmix, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(wmomix, 1, S3D_000, "double", OPS_READ));

            ops_par_loop(equationJ_fused, "equationJ rhscal 2007", block, 1, internal_iter_range,
                    ops_arg_dat(ucor, 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(vcor, 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(wcor, 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd1x, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd1y, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd1z, 1, S3D_000, "double", OPS_READ));

            ops_par_loop(equationDX, "equationDX rhscal dfbydx 2022", block, 1, internal_iter_range,
                    ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
                    ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

            ops_par_loop(equationDY, "equationDY rhscal dfbydy 2023", block, 1, internal_iter_range,
                    ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
                    ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

            ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 2024", block, 1, internal_iter_range,
                    ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
                    ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

            ops_par_loop(equationV, "equationV rhscal 2106", block, 1, internal_iter_range,
                    ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd1x, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(store5, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd1y, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(store6, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd1z, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ));

            ops_par_loop(equationT, "equationT rhscal 2217", block, 1, internal_iter_range,
                    ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(pd2x, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd2y, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd2z, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ));
    
            ops_par_loop(equationS, "equationS rhscal 2240", block, 1, internal_iter_range,
                    ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
                    ops_arg_dat(pd2x, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd2y, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(pd2z, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(store7, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(utmp, 1, S3D_000, "double", OPS_READ));
        } // if (flmixp)

    } // End of ispec loop

    ops_par_loop(equationDX, "equationDX rhscal dfbydx 2755", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhscal dfbydy 2756", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 2757", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationE, "equationE rhscal 2787", block, 1, internal_iter_range,
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationY, "equationY rhscal 2806", block, 1, internal_iter_range,
            ops_arg_dat(erhs, 1, S3D_000, "double", OPS_INC),
            ops_arg_dat(wtmp, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(ucor, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vcor, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wcor, 1, S3D_000, "double", OPS_READ));

    if(flmavt) {
        for(int ispec = 0; ispec < nspec; ispec++) {
            ops_par_loop(equationE, "equationE rhscal 2834", block, 1, extended_iter_range,
                    ops_arg_dat(ctrans[ispec], 1, S3D_000, "double", OPS_WRITE),
                    ops_arg_dat(transp, 1, S3D_000, "double", OPS_READ));
        }

        ops_par_loop(equationA, "equationA rhscal 2843", block, 1, extended_iter_range,
                ops_arg_dat(combo1, 1, S3D_000, "double", OPS_WRITE));

        for(int ispec = 0; ispec < nspec; ispec++) {
            ops_par_loop(equationA, "equationA rhscal 2847", block, 1, extended_iter_range,
                    ops_arg_dat(combo2, 1, S3D_000, "double", OPS_WRITE));

            for(int jspec = 0; jspec < nspec; jspec++) {
                ops_par_loop(equationAE, "equationAE rhscal 2851", block, 1, extended_iter_range,
                        ops_arg_dat(combo2, 1, S3D_000, "double", OPS_RW),
                        ops_arg_dat(ctrans[ispec], 1, S3D_000, "double", OPS_READ),
                        ops_arg_dat(ctrans[jspec], 1, S3D_000, "double", OPS_READ),
                        ops_arg_dat(yrhs[jspec], 1, S3D_000, "double", OPS_READ));

            }

            ops_par_loop(equationAE, "equationAE rhscal 2863", block, 1, extended_iter_range,
                    ops_arg_dat(combo1, 1, S3D_000, "double", OPS_RW),
                    ops_arg_dat(ctrans[ispec], 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(combo2, 1, S3D_000, "double", OPS_READ),
                    ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_READ));
        }
        ops_par_loop(equationE, "equationE rhscal 2873", block, 1, extended_iter_range,
                    ops_arg_dat(difmix, 1, S3D_000, "double", OPS_WRITE),
                    ops_arg_dat(combo1, 1, S3D_000, "double", OPS_READ));
    }

    for(int ispec = 0; ispec < nspec; ispec++) {

        ops_par_loop(equationDX, "equationDX rhscal dfbydx 2920", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDY, "equationDY rhscal dfbydy 2921", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p050_to_m050_y, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationDZ, "equationDZ rhscal dfbydz 2922", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_p005_to_m005_z, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_WRITE));

        ops_par_loop(equationX, "equationX rhscal 2940", block, 1, internal_iter_range,
                ops_arg_dat(yrhs[ispec], 1, S3D_000, "double", OPS_RW),
                ops_arg_dat(rate[ispec], 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store2, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store3, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(ucor, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(vcor, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(wcor, 1, S3D_000, "double", OPS_READ));
    }


/*  needs correction due to converting test app from 1D to 3D 
        if(fxlcnv || fxrcnv) {
        internal_iter_range[0] = 0;   internal_iter_range[1] = nxglbl;
        ops_par_loop(equationDX, "equationDX rhscal dfbydx 2975", block, 1, internal_iter_range,
                ops_arg_dat(drhs, 1, S3D_p500_to_m500_x, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

        if(fxlcnv) {
            internal_iter_range[0] = 0;   internal_iter_range[1] = 1;
            ops_par_loop(equationAF, "equationAF rhscal 2979", block, 1, internal_iter_range,
                ops_arg_dat(strdxl, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(bcl2xl, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ));
        }

        if(fxrcnv) {
            internal_iter_range[0] = nxglbl-1;   internal_iter_range[1] = nxglbl;
            ops_par_loop(equationAF, "equationAF rhscal 2988", block, 1, internal_iter_range,
                ops_arg_dat(strdxr, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(bcl2xr, 1, S3D_000, "double", OPS_WRITE),
                ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ),
                ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ));
        }
    }
*/

/*  ----------------- RHSVEL --------------------- */

    ops_par_loop(equationH_fused, "equationH_fused rhsvel 72", block, 1, extended_iter_range,
            ops_arg_dat(utmp, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(wtmp, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(drhs, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 87", block, 1, internal_iter_range,
            ops_arg_dat(utmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 88", block, 1, internal_iter_range,
            ops_arg_dat(utmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 120", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 121", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 153", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 154", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 186", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 187", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 211", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 212", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 235", block, 1, internal_iter_range,
            ops_arg_dat(utmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 236", block, 1, internal_iter_range,
            ops_arg_dat(utmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 259", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 260", block, 1, internal_iter_range,
            ops_arg_dat(wtmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 284", block, 1, internal_iter_range,
            ops_arg_dat(utmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 285", block, 1, internal_iter_range,
            ops_arg_dat(utmp, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 308", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 309", block, 1, internal_iter_range,
            ops_arg_dat(vtmp, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationH, "equationH rhsvel 341", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(utmp, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 348", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationH, "equationH rhsvel 361", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 368", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 369", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store5, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationG, "equationG rhsvel 378", block, 1, internal_iter_range,
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_INC),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationH, "equationH rhsvel 389", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(urhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wtmp, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 396", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDX, "equationDX rhsvel dfbydx 397", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p500_to_m500_x, "double", OPS_READ),
            ops_arg_dat(store6, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationG, "equationG rhsvel 428", block, 1, internal_iter_range,
            ops_arg_dat(store4, 1, S3D_000, "double", OPS_INC),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationH, "equationH rhsvel 416", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(vtmp, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 422", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationG, "equationG rhsvel 428", block, 1, internal_iter_range,
            ops_arg_dat(store5, 1, S3D_000, "double", OPS_INC),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationH, "equationH rhsvel 439", block, 1, extended_iter_range,
            ops_arg_dat(store7, 1, S3D_000, "double", OPS_WRITE),
            ops_arg_dat(vrhs, 1, S3D_000, "double", OPS_READ),
            ops_arg_dat(wtmp, 1, S3D_000, "double", OPS_READ));

    ops_par_loop(equationDZ, "equationDZ rhsvel dfbydz 446", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p005_to_m005_z, "double", OPS_READ),
            ops_arg_dat(store1, 1, S3D_000, "double", OPS_WRITE));

    ops_par_loop(equationDY, "equationDY rhsvel dfbydy 447", block, 1, internal_iter_range,
            ops_arg_dat(store7, 1, S3D_p050_to_m050_y, "double", OPS_READ),
            ops_arg_dat(store2, 1, S3D_000, "double", OPS_WRITE));


#ifdef OPS_LAZY
    ops_execute();
#endif

    ops_exit();
    return 0;
}
