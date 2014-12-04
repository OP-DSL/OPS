/** @brief SHSGC top level program
  * @author Satya P. Jammy, converted to OPS by Gihan Mudalige
  * @details
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing multi-block Structured mesh applications
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"


/******************************************************************************
* OPS variables
/******************************************************************************/

// ops blocks
ops_block shsgc_grid;

//ops dats
ops_dat x;
ops_dat rho_old, rho_new, rho_res;
ops_dat rhou_old, rhou_new, rhou_res;
ops_dat rhov_old, rhov_new;
ops_dat rhoE_old, rhoE_new, rhoE_res;
ops_dat rhoin;
ops_dat r, al, alam, gt, tht, ep2, cmp, cf, eff, s;
ops_dat readvar;

ops_reduction rms;

//
//Declare commonly used stencils
//
ops_stencil S1D_0, S1D_01, S1D_0M1;
ops_stencil S1D_0M1M2P1P2;

/******************************************************************************
* Initialize Global constants and variables
/******************************************************************************/


/**----------shsgc Vars/Consts--------------**/

int nxp = 204;
int nyp = 5;
int xhalo = 2;
int yhalo = 2;
double xmin = -5.0;
double ymin = 0;
double xmax = 5.0;
double ymax = 0.5;
double dx = (xmax-xmin)/(nxp-(1 + 2*xhalo));
double dy = (ymax-ymin)/(nyp-1);
double pl = 10.333f;
double pr = 1.0f;
double rhol = 3.857143;
double rhor = 1.0f;
double ul = 2.6293690 ;
double ur = 0.0f;
double gam = 1.4;
double gam1=gam - 1.0;
double eps = 0.2;
double lambda = 5.0;
double a1[3];
double a2[3];
double dt=0.0002;
double del2 = 1e-8;
double akap2 = 0.40;
double tvdsmu = 0.25f;
double con = pow (tvdsmu,2.f);

FILE *fp;

//
//kernles
//
#include "initialize_kernel.h"
#include "save_kernel.h"
#include "zerores_kernel.h"
#include "drhoudx_kernel.h"
#include "drhouupdx_kernel.h"
#include "drhoEpudx_kernel.h"
#include "updateRK3_kernel.h"
#include "Riemann_kernel.h"
#include "limiter_kernel.h"
#include "tvd_kernel.h"
#include "vars_kernel.h"
#include "calupwindeff_kernel.h"
#include "fact_kernel.h"
#include "update_kernel.h"
#include "test_kernel.h"


/******************************************************************************
* Main program
/******************************************************************************/

int main(int argc, char **argv) {

  double totaltime =0.0f;
  // Initialize rk3 co-efficient's
  a1[0] = 2.0/3.0;
  a1[1] = 5.0/12.0;
  a1[2] = 3.0/5.0;
  a2[0] = 1.0/4.0;
  a2[1] = 3.0/20.0;
  a2[2] = 3.0/5.0;

  /**-------------------------- OPS Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,1);

  //
  //declare 1D block
  //
  shsgc_grid = ops_decl_block(1, "shsgc grid");

  //
  //declare data on block
  //

  int d_p[1] = {2}; //max block halo depths for the dat in the possitive direction
  int d_m[1] = {-2}; //max block halo depths for the dat in the negative direction
  int size[1] = {nxp}; //size of 1D dat
  int base[1] = {0};
  double* temp = NULL;

  x = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "x");

  // Conservative variables definition
  rho_old = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rho_old");
  rho_new = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rho_new");
  rho_res = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rho_res");

  rhou_old = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhou_old");
  rhou_new = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhou_new");
  rhou_res = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhou_res");

  rhov_old = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhov_old");
  rhov_new = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhov_new");

  rhoE_old = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_old");
  rhoE_new = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_new");
  rhoE_res = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_res");

  //extra dat for rhoin
  rhoin = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoin");

  // TVD scheme variables
  r     =  ops_decl_dat(shsgc_grid, 9, size, base, d_m, d_p, temp, "double", "r");
  al    = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "al");
  alam  = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "alam");
  gt    = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "gt");
  tht   = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "tht");
  ep2   = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "ep2");
  cmp   = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "cmp");
  cf    = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "cf");
  eff   = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "eff");
  s     = ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, "double", "s");

  //read in referance solution
  /*if ((fp = fopen("Rho", "r")) == NULL) {
    printf("can't open file %s\n","Rho");
    exit(2);
  }

  readvar     = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "readvar");
  temp = (double *)malloc(sizeof(double)*nxp);
  for (int i=0; i<nxp; i++){
	  fscanf(fp,"%lf",&temp[i]);
  }
  //manual copy of read values -- ****** need to be removed later ******
  memcpy(readvar->data, temp, sizeof(double)*nxp);
  free(temp);*/


  //reduction handle for rms variable
  rms = ops_decl_reduction_handle(sizeof(double), "double", "rms");

  //
  //Declare commonly used stencils
  //
  int s1D_0[]   = {0};
  S1D_0         = ops_decl_stencil( 1, 1, s1D_0, "0");
  int s1D_0M1M2P1P2[] = {0,-1,-2,1,2};
  S1D_0M1M2P1P2 = ops_decl_stencil( 1, 5, s1D_0M1M2P1P2, "0,-1,-2,1,2");

  int s1D_01[]   = {0,1};
  S1D_01         = ops_decl_stencil( 1, 2, s1D_01, "0,1");

  int s1D_0M1[]   = {0,-1};
  S1D_0M1         = ops_decl_stencil( 1, 2, s1D_01, "0,-1");

  ops_partition("1D_BLOCK_DECOMPOSE");


  //initialize global constants
  ops_decl_const("nxp", 1, "int", &nxp );
  ops_decl_const("nyp", 1, "int", &nyp );
  ops_decl_const("xhalo", 1, "int", &xhalo );
  ops_decl_const("yhalo", 1, "int", &yhalo );
  ops_decl_const("xmin", 1, "double", &xmin );
  ops_decl_const("ymin", 1, "double", &ymin );
  ops_decl_const("xmax", 1, "double", &xmax );
  ops_decl_const("ymax", 1, "double", &ymax );
  ops_decl_const("dx", 1, "double", &dx );
  ops_decl_const("dy", 1, "double", &dy );
  ops_decl_const("pl", 1, "double", &pl );
  ops_decl_const("pr", 1, "double", &pr );
  ops_decl_const("rhol", 1, "double", &rhol );
  ops_decl_const("rhor", 1, "double", &rhor );
  ops_decl_const("ul", 1, "double", &ul );
  ops_decl_const("ur", 1, "double", &ur );
  ops_decl_const("gam", 1, "double", &gam );
  ops_decl_const("gam1", 1, "double", &gam1 );
  ops_decl_const("eps", 1, "double", &eps );
  ops_decl_const("lambda", 1, "double", &lambda );
  ops_decl_const("dt", 1, "double", &dt );
  ops_decl_const("del2", 1, "double", &del2 );
  ops_decl_const("akap2", 1, "double", &akap2 );
  ops_decl_const("tvdsmu", 1, "double", &tvdsmu );
  ops_decl_const("con", 1, "double", &con );


  //
  // Initialize with the test case
  //

  int nxp_range[] = {0,nxp};
  ops_par_loop(initialize_kernel, "initialize_kernel", shsgc_grid, 1, nxp_range,
               ops_arg_dat(x, 1, S1D_0, "double", OPS_WRITE),
               ops_arg_dat(rho_new, 1, S1D_0, "double", OPS_WRITE),
               ops_arg_dat(rhou_new, 1, S1D_0, "double", OPS_WRITE),
               ops_arg_dat(rhoE_new, 1, S1D_0, "double", OPS_WRITE),
               ops_arg_dat(rhoin, 1, S1D_0, "double", OPS_WRITE),
               ops_arg_idx());

  //ops_print_dat_to_txtfile(rhoin, "shsgc.dat");


  //
  //main iterative loop
  //

  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);

  int niter = 9005;
  for (int iter = 0; iter <niter;  iter++){

    // Save previous data arguments
    ops_par_loop(save_kernel, "save_kernel", shsgc_grid, 1, nxp_range,
             ops_arg_dat(rho_old, 1, S1D_0, "double", OPS_WRITE),
             ops_arg_dat(rhou_old, 1, S1D_0, "double", OPS_WRITE),
             ops_arg_dat(rhoE_old, 1, S1D_0, "double", OPS_WRITE),
             ops_arg_dat(rho_new, 1, S1D_0, "double", OPS_READ),
             ops_arg_dat(rhou_new, 1, S1D_0, "double", OPS_READ),
             ops_arg_dat(rhoE_new, 1, S1D_0, "double", OPS_READ));

    //rk3 loop
    for (int nrk=0; nrk <3; nrk++){

      // make residue equal to zero
      ops_par_loop(zerores_kernel, "zerores_kernel", shsgc_grid, 1, nxp_range,
              ops_arg_dat(rho_res, 1, S1D_0, "double", OPS_WRITE),
              ops_arg_dat(rhou_res, 1, S1D_0, "double",OPS_WRITE),
              ops_arg_dat(rhoE_res, 1, S1D_0, "double",OPS_WRITE));

      // computations of convective derivatives
      //TODO

      // calculate drhou/dx
      int nxp_range_1[] = {2,nxp-2};
      ops_par_loop(drhoudx_kernel, "drhoudx_kernel", shsgc_grid, 1, nxp_range_1,
              ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rho_res, 1, S1D_0, "double",OPS_WRITE));

      // calculate d(rhouu + p)/dx
      ops_par_loop(drhouupdx_kernel, "drhouupdx_kernel", shsgc_grid, 1, nxp_range_1,
              ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rho_new,  1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rhoE_new, 1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rhou_res,  1, S1D_0, "double",OPS_WRITE));

      // Energy equation derivative d(rhoE+p)u/dx
      ops_par_loop(drhoEpudx_kernel, "drhoEpudx_kernel", shsgc_grid, 1, nxp_range_1,
              ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rho_new,  1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rhoE_new, 1, S1D_0M1M2P1P2, "double",OPS_READ),
              ops_arg_dat(rhoE_res,  1, S1D_0, "double",OPS_WRITE));

      //if (nrk == 1) {
      //  ops_print_dat_to_txtfile(rhoE_res, "shsgc.dat");
      //  exit(0);
      //}

      //update use rk3 co-efficient's
      int nxp_range_2[] = {3,nxp-2};
      ops_par_loop(updateRK3_kernel, "updateRK3_kernel", shsgc_grid, 1, nxp_range_2,
                   ops_arg_dat(rho_new,  1, S1D_0, "double",OPS_WRITE),
                   ops_arg_dat(rhou_new, 1, S1D_0, "double",OPS_WRITE),
                   ops_arg_dat(rhoE_new, 1, S1D_0, "double",OPS_WRITE),
                   ops_arg_dat(rho_old,  1, S1D_0, "double",OPS_RW),
                   ops_arg_dat(rhou_old, 1, S1D_0, "double",OPS_RW),
                   ops_arg_dat(rhoE_old, 1, S1D_0, "double",OPS_RW),
                   ops_arg_dat(rho_res,  1, S1D_0, "double",OPS_READ),
                   ops_arg_dat(rhou_res, 1, S1D_0, "double",OPS_READ),
                   ops_arg_dat(rhoE_res, 1, S1D_0, "double",OPS_READ),
                   ops_arg_gbl(&a1[nrk], 1, "double", OPS_READ),
                   ops_arg_gbl(&a2[nrk], 1, "double", OPS_READ));
    }

    //
    // TVD scheme
    //

    // Riemann invariants
    int nxp_range_3[] = {0,nxp-1};
    ops_par_loop(Riemann_kernel, "Riemann_kernel", shsgc_grid, 1, nxp_range_3,
                 ops_arg_dat(rho_new,  1, S1D_01, "double",OPS_READ),
                 ops_arg_dat(rhou_new,  1, S1D_01, "double",OPS_READ),
                 ops_arg_dat(rhoE_new,  1, S1D_01, "double",OPS_READ),
                 ops_arg_dat(alam,  3, S1D_01, "double",OPS_WRITE),
                 ops_arg_dat(r,  9, S1D_01, "double",OPS_WRITE),
                 ops_arg_dat(al, 3, S1D_01, "double",OPS_WRITE));

    // limiter function
    int nxp_range_4[] = {1,nxp};
    ops_par_loop(limiter_kernel, "limiter_kernel", shsgc_grid, 1, nxp_range_4,
                 ops_arg_dat(al, 3, S1D_0M1, "double",OPS_READ),
                 ops_arg_dat(tht,3, S1D_0, "double",OPS_WRITE),
                 ops_arg_dat(gt, 3, S1D_0, "double",OPS_WRITE));

    // Second order tvd dissipation
    ops_par_loop(tvd_kernel, "tvd_kernel", shsgc_grid, 1, nxp_range_3,
                 ops_arg_dat(tht, 3, S1D_0M1, "double",OPS_READ),
                 ops_arg_dat(ep2, 3, S1D_0, "double",OPS_WRITE));

    // vars
    ops_par_loop(vars_kernel, "vars_kernel", shsgc_grid, 1, nxp_range_3,
                 ops_arg_dat(alam,3, S1D_0, "double",OPS_READ),
                 ops_arg_dat(al,  3, S1D_0, "double",OPS_READ),
                 ops_arg_dat(gt,  3, S1D_01, "double",OPS_READ),
                 ops_arg_dat(cmp, 3, S1D_0, "double",OPS_WRITE),
                 ops_arg_dat(cf,  3, S1D_0, "double",OPS_WRITE));


    // cal upwind eff
    ops_par_loop(calupwindeff_kernel, "calupwindeff_kernel", shsgc_grid, 1, nxp_range_3,
                 ops_arg_dat(cmp,3, S1D_0, "double",OPS_READ),
                 ops_arg_dat(gt, 3, S1D_01, "double",OPS_READ),
                 ops_arg_dat(cf, 3, S1D_0, "double",OPS_READ),
                 ops_arg_dat(al, 3, S1D_0, "double",OPS_READ),
                 ops_arg_dat(ep2,3, S1D_0, "double",OPS_READ),
                 ops_arg_dat(r,  9, S1D_0, "double",OPS_READ),
                 ops_arg_dat(eff,3, S1D_0, "double",OPS_WRITE));

    //fact
    ops_par_loop(fact_kernel, "fact_kernel", shsgc_grid, 1, nxp_range_4,
                 ops_arg_dat(eff,  3, S1D_0M1, "double",OPS_READ),
                 ops_arg_dat(s,    3, S1D_0,   "double",OPS_WRITE));


    // update loop
    int nxp_range_5[] = {3,nxp-3};
    ops_par_loop(update_kernel, "update_kernel", shsgc_grid, 1, nxp_range_5,
                 ops_arg_dat(rho_new,  1, S1D_0, "double",OPS_RW),
                 ops_arg_dat(rhou_new, 1, S1D_0, "double",OPS_RW),
                 ops_arg_dat(rhoE_new, 1, S1D_0, "double",OPS_RW),
                 ops_arg_dat(s,        3, S1D_0, "double",OPS_READ));

    totaltime = totaltime + dt;
    ops_printf("%d \t %f\n", iter, totaltime);

  }

  ops_timers_core(&ct1, &et1);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  //compare solution to referance solution
  double local_rms = 0.0;
  ops_par_loop(test_kernel, "test_kernel", shsgc_grid, 1, nxp_range,
               ops_arg_dat(rho_new,  1, S1D_0, "double",OPS_READ),
               ops_arg_reduce(rms, 1, "double", OPS_INC));
  //ops_arg_dat(readvar,  1, S1D_0, "double",OPS_READ),

  ops_reduction_result(rms, &local_rms);
  //printf("\nthe RMS between C and Fortran is %lf\n" , sqrt(local_rms)/nxp);

  ops_printf("\nRMS %lf\n" , sqrt(local_rms)/nxp); //Correct RMS = 0.233689

  //ops_print_dat_to_txtfile(alam, "shsgc.dat");
  ops_print_dat_to_txtfile(rho_new, "shsgc.dat");
  //ops_print_dat_to_txtfile(al, "shsgc.dat");
  ops_exit();

}