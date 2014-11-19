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

//
//Declare commonly used stencils
//
ops_stencil S1D_0;

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


//#define OPS_ACC0_1D(x) (x)

//
//kernles
//
#include "initialize_kernel.h"


/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{  
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
  int size[1] = {0}; //size of 1D dat
  int base[1] = {nxp};
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
  
  
  //
  //Declare commonly used stencils
  //
  int s1D_0[]   = {0};
  S1D_0         = ops_decl_stencil( 1, 1, s1D_0, "0");
  
  ops_partition("1D_BLOCK_DECOMPOSE");
  printf("here\n");
  //
  // Initialize with the test case
  //
  
  int nxp_range[] = {0,nxp};
  ops_par_loop(initialize_kernel, "initialize_kernel", shsgc_grid, 1, nxp_range,
               ops_arg_dat(x, 1, S1D_0, "double", OPS_WRITE),
               ops_arg_idx());
  
  ops_print_dat_to_txtfile(x, "shsgc.dat");
  
}