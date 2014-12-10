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

// ops blocks


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
double Mach = 3;
double Pr=0.72;
double Re = 100;
double omega = 0.7;
FILE *fp;
//
// Subroutines
void conv();
void visc();
//
//Headers
//
#include "vars.h"
#include "ops_data.h"
#include "shsgc_ops_vars.h"
//
//kernles
//
#include "initialize_kernel.h"


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

  int d_p[1] = {0}; //max block halo depths for the dat in the possitive direction
  int d_m[1] = {0}; //max block halo depths for the dat in the negative direction
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
  viscu_res = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhou_res");

  fn = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "fn");
  dfn = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "dfn");

  rhoE_old = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_old");
  rhoE_new = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_new");
  rhoE_res = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_res");
  viscE_res = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoE_res");

  //extra dat for rhoin
  rhoin = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "rhoin");
  
  u = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "u");
  u_x = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "u_x");
  u_xx = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "u_xx");
  
  T = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "T");
  T_x = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "T_x");
  T_xx = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "T_xx");
  
  mu = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "mu");
  mu_x = ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, "double", "mu_x");
  

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
  
  
  //reduction handle for rms variable
  rms = ops_decl_reduction_handle(sizeof(double), "double", "rms");
  
  //
  //Declare commonly used stencils
  //
  int s1D_0[]   = {0};
  S1D_0         = ops_decl_stencil( 1, 1, s1D_0, "0");
  int s1D_0M1M2P1P2[] = {0,-1,-2,1,2};
  S1D_0M1M2P1P2 = ops_decl_stencil( 5, 1, s1D_0M1M2P1P2, "0,-1,-2,1,2");

  int s1D_01[]   = {0,1};
  S1D_01         = ops_decl_stencil( 2, 1, s1D_01, "0,1");
  
  int s1D_0M1[]   = {0,-1};
  S1D_0M1         = ops_decl_stencil( 2, 1, s1D_01, "0,-1");
  
  ops_partition("1D_BLOCK_DECOMPOSE");
  

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

 
    //rk3 loop
  for (int nrk=0; nrk <3; nrk++){

    conv();
 	  visc();

      
}

 
    totaltime = totaltime + dt;
    printf("%d \t %f\n", iter, totaltime);
   
  }
  
  ops_timers_core(&ct1, &et1);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);
  
  //compare solution to referance solution
 
  ops_exit();
    
}
