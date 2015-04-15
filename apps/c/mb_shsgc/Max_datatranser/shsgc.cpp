/** @brief SHSGC top level program
  * @author Satya P. Jammy
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


/******************************************************************************
* Initialize Global constants and variables
/******************************************************************************/
int scale = 1;
int nxp = 2500;
int nyp = 5;
int xhalo = 2;
double xmin = -5.0;
double xmax = 5.0;
double dx = (xmax-xmin)/(nxp-1);
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
double xt = 0;
ops_stencil S1D_0, S1D_01,S1D_0M1M2P1P2, S1D_0M1;

/**----------shsgc kernels --------------**/

#include "gridgen_kernel.h"
#include "init_kernel.h"
#include "conv_kernel.h"
#include "tvdx_kernel.h"


/******************************************************************************
* Main program
/******************************************************************************/

int main(int argc, char **argv) {
  // rk3 variables
  a1[0] = 2.0/3.0;
  a1[1] = 5.0/12.0;
  a1[2] = 3.0/5.0;
  a2[0] = 1.0/4.0;
  a2[1] = 3.0/20.0;
  a2[2] = 3.0/5.0;

  ops_init(argc,argv,1);
  int nblock = 1;
  if(nxp%nblock != 0)
  ops_printf("wrong input\n");
  ops_printf("Simulation details are:\n");
  ops_printf("-----------------------------------------\n");
  ops_printf("Scale factor is %d \n", scale);
  nxp =nxp * scale;
  ops_printf("Number of gridpoints are %d\n",nxp);
  ops_printf("Time step is %lf\n",dt);
  ops_printf("Number of Blocks are %d\n",nblock);
  ops_printf("-----------------------------------------\n");
  // ops_blocks declaration

  ops_block *shsgc_grid = (ops_block *)malloc(nblock*sizeof(ops_block*));

  //ops dats
  ops_dat *x            = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rho_old      = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rho_new      = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rho_res      = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhou_old     = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhou_new     = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhou_res     = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhoE_old     = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhoE_new     = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhoE_res     = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *der1         = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *der2         = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *der3         = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *rhoin        = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *workarray1   = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *workarray2   = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *workarray3   = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
//   tvd scheme arrays
  ops_dat *r    = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *al   = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *alam = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *gt   = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *tht  = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *ep2  = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *cmp  = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *cf   = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *eff  = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
  ops_dat *s    = (ops_dat *)malloc(nblock*sizeof(ops_dat*));
//   check output array
  ops_dat *rhout    = (ops_dat *)malloc(nblock*sizeof(ops_dat*));



//   Tvd declaration end
  char buf[50];
  int d_p[1]   = {2}; //max block halo depths for the dat in the possitive
  int d_m[1]   = {-2}; //max block halo depths for the dat in the negative
  int size[1]  = {nxp/nblock};
  int base[1]  = {0};
  double* temp = NULL;
  int *sizes = (int*)malloc(2*nblock*sizeof(int));
  // declare blocks and data files
  for(int i=0; i<nblock; i++){

    sprintf(buf,"shsgc_block[%d]",i);
    shsgc_grid[i] = ops_decl_block(1, buf);
    sprintf(buf,"x[%d]",i);
    x[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rho_old[%d]",i);
    rho_old[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rho_new[%d]",i);
    rho_new[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rho_res[%d]",i);
    rho_res[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rhou_old[%d]",i);
    rhou_old[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rhou_new[%d]",i);
    rhou_new[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rhou_res[%d]",i);
    rhou_res[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rhoE_old[%d]",i);
    rhoE_old[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rhoE_new[%d]",i);
    rhoE_new[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"rhoE_res[%d]",i);
    rhoE_res[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);

//   work array declaration
    sprintf(buf,"workarray1[%d]",i);
    workarray1[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"workarray2[%d]",i);
    workarray2[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"workarray3[%d]",i);
    workarray3[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"der1[%d]",i);
    der1[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"der2[%d]",i);
    der2[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"der3[%d]",i);
    der3[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
//     cross check initialization
    sprintf(buf,"rhoin[%d]",i);
    rhoin[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
//     check the output post shock
    sprintf(buf,"rhout[%d]",i);
    rhout[i] = ops_decl_dat(shsgc_grid[i], 1, size, base, d_m, d_p, temp, "double", buf);
//     TVD scheme arrays
    sprintf(buf,"r[%d]",i);
    r[i]     =  ops_decl_dat(shsgc_grid[i], 9, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"al[%d]",i);
    al[i]    = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"alam[%d]",i);
    alam[i]  = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"gt[%d]",i);
    gt[i]    = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"tht[%d]",i);
    tht[i]   = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"ep2[%d]",i);
    ep2[i]   = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"cmp[%d]",i);
    cmp[i]   = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"cf[%d]",i);
    cf[i]    = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"eff[%d]",i);
    eff[i]   = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    sprintf(buf,"s[%d]",i);
    s[i]     = ops_decl_dat(shsgc_grid[i], 3, size, base, d_m, d_p, temp, "double", buf);
    // sizes should be changed at a later date if no of grid points in X is not a factor of nblock
    sizes[2*i]   = 0;
    sizes[(2*i)+1] = size[0];

  }
  // decalre commonly used stencils
  int s1D_0[]   = {0};
  S1D_0         = ops_decl_stencil( 1, 1, s1D_0, "0");
  int s1D_0M1M2P1P2[] = {0,-1,-2,1,2};
  S1D_0M1M2P1P2 = ops_decl_stencil( 1, 5, s1D_0M1M2P1P2, "0,-1,-2,1,2");

  int s1D_01[]   = {0,1};
  S1D_01         = ops_decl_stencil( 1, 2, s1D_01, "0,1");

  int s1D_0M1[]   = {0,-1};
  S1D_0M1         = ops_decl_stencil( 1, 2, s1D_0M1, "0,-1");

  // loop control parameters
  int niter =9000;
  // initialise

//   Declare halo groups TODO
  ops_halo *rhohalo = (ops_halo *)malloc(2*(nblock-1)*sizeof(ops_halo *));
  ops_halo *rhouhalo = (ops_halo *)malloc(2*(nblock-1)*sizeof(ops_halo *));
  ops_halo *rhoEhalo = (ops_halo *)malloc((2*nblock-1)*sizeof(ops_halo *));
  int offrho  = 0;
  int offrhou = 0;
  int offrhoE = 0;
  for(int i=0; i<nblock; i++){
    if( i>0){
      int halo_iter[] = {2};
      int base_from[] = {sizes[2*i-1]-xhalo};
      int base_to[] = {sizes[2*i]-xhalo};
      int dir[] = {1,2};
//       ops_printf("i==j%d %d\n ", base_from[0],base_to[0]);
      rhohalo[offrho++] = ops_decl_halo(rho_new[i-1], rho_new[i], halo_iter, base_from, base_to, dir, dir);
      rhouhalo[offrhou++] = ops_decl_halo(rhou_new[i-1], rhou_new[i], halo_iter, base_from, base_to, dir, dir);
      rhoEhalo[offrhoE++] = ops_decl_halo(rhoE_new[i-1], rhoE_new[i], halo_iter, base_from, base_to, dir, dir);
      base_from[0] = sizes[2*i]; base_to[0] = sizes[2*(i)-1];
      rhohalo[offrho++] = ops_decl_halo(rho_new[i], rho_new[i-1], halo_iter, base_from, base_to, dir, dir);
      rhouhalo[offrhou++] = ops_decl_halo(rhou_new[i], rhou_new[i-1], halo_iter, base_from, base_to, dir, dir);
      rhoEhalo[offrhoE++] = ops_decl_halo(rhoE_new[i], rhoE_new[i-1], halo_iter, base_from, base_to, dir, dir);
//       ops_printf("%d %d\n ", base_from[0],base_to[0]);
    }
  }
//   ops_printf("halo %d %d %d\n", offrho, offrhou, offrhoE);
  ops_halo_group rho_halos = ops_decl_halo_group(offrho,rhohalo);
  ops_halo_group rhou_halos = ops_decl_halo_group(offrhou,rhouhalo);
  ops_halo_group rhoE_halos = ops_decl_halo_group(offrhoE,rhoEhalo);
  // testing halo implementations
//   ops_halo_transfer(rho_halos);
//   end TODO

// ops reductions
  ops_reduction post_err = ops_decl_reduction_handle(sizeof(double), "double", "err");
  ops_reduction pre_err = ops_decl_reduction_handle(sizeof(double), "double", "err1");
  ops_reduction num_pre = ops_decl_reduction_handle(sizeof(int), "int", "err2");


  ops_partition("");

  ops_decl_const( "nxp", 1, "int", &nxp );
  ops_decl_const( "nyp", 1, "int", &nyp );
  ops_decl_const( "xhalo", 1, "int", &xhalo );
  ops_decl_const( "xmin", 1, "double", &xmin );
  ops_decl_const( "xmax", 1, "double", &xmax );
  ops_decl_const( "dx", 1, "double", &dx );
  ops_decl_const( "pl", 1, "double", &pl);
  ops_decl_const( "pr", 1, "double", &pr);
  ops_decl_const( "rhol", 1, "double", &rhol);
  ops_decl_const( "rhor", 1, "double", &rhor);
  ops_decl_const( "ul", 1, "double", &ul);
  ops_decl_const( "ur", 1, "double", &ur);
  ops_decl_const( "gam", 1, "double", &gam);
  ops_decl_const( "gam1", 1, "double", &gam1);
  ops_decl_const( "eps", 1, "double", &eps);
  ops_decl_const( "lambda", 1, "double", &lambda);
  ops_decl_const( "dt", 1, "double", &dt);
  ops_decl_const( "del2", 1, "double", &del2);
  ops_decl_const( "akap2", 1, "double", &akap2);
  ops_decl_const( "tvdsmu", 1, "double", &tvdsmu);
  ops_decl_const( "con", 1, "double", &con);
  ops_decl_const( "Mach", 1, "double", &Mach);
  ops_decl_const( "xt", 1, "double", &xt);
  // scale factor for domain scaling
  ops_decl_const( "scale", 1, "int", &scale );


  // Generate grid for the 1d domain later change this to read from file
  for(int i=0; i<nblock; i++){
    int range[] = {sizes[2*(i)]-xhalo,sizes[2*(i)+1]+xhalo};
    xt = xmin + dx*i*nxp/nblock;
    ops_update_const( "xt", 1, "double", &xt);
    ops_par_loop(gridgen_kernel, "gridgen_kernel", shsgc_grid[i], 1, range,
               ops_arg_dat(x[i], 1, S1D_0, "double", OPS_WRITE),
               ops_arg_idx());
  }

  // initialize the domain with
  for(int i=0; i<nblock; i++){
    int range[] = {sizes[2*(i)]-xhalo,sizes[2*(i)+1]+xhalo};
    ops_par_loop(init_kernel, "init_kernel", shsgc_grid[i], 1, range,
                 ops_arg_dat(x[i], 1, S1D_0, "double", OPS_READ),
                 ops_arg_dat(rho_new[i], 1, S1D_0, "double", OPS_WRITE),
                 ops_arg_dat(rhou_new[i], 1, S1D_0, "double", OPS_WRITE),
                 ops_arg_dat(rhoE_new[i], 1, S1D_0, "double", OPS_WRITE),
                 ops_arg_dat(rhoin[i], 1, S1D_0, "double", OPS_WRITE),
                 ops_arg_dat(rho_old[i], 1, S1D_0, "double", OPS_WRITE),
                 ops_arg_dat(rhou_old[i], 1, S1D_0, "double", OPS_WRITE),
                 ops_arg_dat(rhoE_old[i], 1, S1D_0, "double", OPS_WRITE));
  }

  //   wirte the grid and initial data to files based on nblocks
  for(int i=0; i<nblock; i++){
    sprintf(buf,"x%d",i+1);
    ops_print_dat_to_txtfile(x[i], buf);
    sprintf(buf,"rhoin%d",i+1);
    ops_print_dat_to_txtfile(rhoin[i], buf);
  }

//   main iteration
  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);
  for(int iter =0; iter<niter; iter++){
//     ops_printf("iteration is %d\n",iter);
    if(nblock>0){
      ops_halo_transfer(rho_halos);
      ops_halo_transfer(rhou_halos);
      ops_halo_transfer(rhoE_halos);
    }
    for(int i=0; i<nblock; i++){
      //     save previous data
      int range1[] = {sizes[(2*i)]-xhalo,sizes[(2*i)+1]+xhalo};
      ops_par_loop(save_kernel, "save_kernel", shsgc_grid[i], 1, range1,
                   ops_arg_dat(rho_old[i],  1, S1D_0, "double", OPS_WRITE),
                   ops_arg_dat(rhou_old[i], 1, S1D_0, "double", OPS_WRITE),
                   ops_arg_dat(rhoE_old[i], 1, S1D_0, "double", OPS_WRITE),
                   ops_arg_dat(rho_new[i],  1, S1D_0, "double", OPS_READ ),
                   ops_arg_dat(rhou_new[i], 1, S1D_0, "double", OPS_READ ),
                   ops_arg_dat(rhoE_new[i], 1, S1D_0, "double", OPS_READ ));
    }
    // rk3 loop
    for(int nrk=0; nrk<3; nrk++){

      //   convective terms evaluation
      for(int i=0; i<nblock; i++){
        int range[] = {sizes[(2*i)]-xhalo,sizes[(2*i)+1]+xhalo};
        ops_par_loop(calvar_kernel, "calvar_kernel", shsgc_grid[i], 1, range,
                     ops_arg_dat(rho_new[i]   , 1, S1D_0, "double", OPS_READ ),
                     ops_arg_dat(rhou_new[i]  , 1, S1D_0, "double", OPS_READ ),
                     ops_arg_dat(rhoE_new[i]  , 1, S1D_0, "double", OPS_READ ),
                     ops_arg_dat(workarray2[i], 1, S1D_0, "double", OPS_WRITE),
                     ops_arg_dat(workarray3[i], 1, S1D_0, "double", OPS_WRITE));

        // evaluate derivative of rhou, workarray2[i], workarray3[i] save in der1[i],der2[i], der3
        int rangeder[] = {sizes[(2*i)], sizes[(2*i)+1]};
        ops_par_loop(xder1_kernel, "xder1kernel", shsgc_grid[i], 1, rangeder,
                     ops_arg_dat(rhou_new[i], 1, S1D_0M1M2P1P2, "double", OPS_READ),
                     ops_arg_dat(der1[i]    , 1, S1D_0        , "double", OPS_WRITE));

        ops_par_loop(xder1_kernel, "xder1kernel", shsgc_grid[i], 1, rangeder,
                     ops_arg_dat(workarray2[i], 1, S1D_0M1M2P1P2, "double", OPS_READ),
                     ops_arg_dat(der2[i]      , 1, S1D_0        , "double", OPS_WRITE));

        ops_par_loop(xder1_kernel, "xder1kernel", shsgc_grid[i], 1, rangeder,
                     ops_arg_dat(workarray3[i], 1, S1D_0M1M2P1P2, "double", OPS_READ),
                     ops_arg_dat(der3[i], 1, S1D_0, "double", OPS_WRITE));
        //evaluate res[i]idues
        int rangeupdate[] = {sizes[(2*i)],sizes[(2*i)+1]};
        ops_par_loop(residue_eval, "residue_eval", shsgc_grid[i], 1, rangeupdate,
                     ops_arg_dat(der1[i], 1, S1D_0, "double", OPS_READ),
                     ops_arg_dat(der2[i], 1, S1D_0, "double", OPS_READ),
                     ops_arg_dat(der3[i], 1, S1D_0, "double", OPS_READ),
                     ops_arg_dat(rho_res[i], 1, S1D_0, "double", OPS_WRITE),
                     ops_arg_dat(rhou_res[i], 1, S1D_0, "double", OPS_WRITE),
                     ops_arg_dat(rhoE_res[i], 1, S1D_0, "double", OPS_WRITE));

        ops_par_loop(updateRK3_kernel, "updateRK3_kernel", shsgc_grid[i], 1, rangeupdate,
                     ops_arg_dat(rho_new[i],  1, S1D_0, "double",OPS_WRITE),
                     ops_arg_dat(rhou_new[i], 1, S1D_0, "double",OPS_WRITE),
                     ops_arg_dat(rhoE_new[i], 1, S1D_0, "double",OPS_WRITE),
                     ops_arg_dat(rho_old[i],  1, S1D_0, "double",OPS_RW),
                     ops_arg_dat(rhou_old[i], 1, S1D_0, "double",OPS_RW),
                     ops_arg_dat(rhoE_old[i], 1, S1D_0, "double",OPS_RW),
                     ops_arg_dat(rho_res[i],  1, S1D_0, "double",OPS_RW),
                     ops_arg_dat(rhou_res[i], 1, S1D_0, "double",OPS_RW),
                     ops_arg_dat(rhoE_res[i], 1, S1D_0, "double",OPS_RW),
                     ops_arg_gbl(&a1[nrk], 1, "double", OPS_READ),
                     ops_arg_gbl(&a2[nrk], 1, "double", OPS_READ));
      }
//       for (int i=0; i<nblock; i++){
        if(nblock>0){
          ops_halo_transfer(rho_halos);
          ops_halo_transfer(rhou_halos);
          ops_halo_transfer(rhoE_halos);
        }
//       }
    }
      // rk loop ends here
//       evaluate tvdx scheme
// TODO decalre variables for tvd scheme also code for halo exchange

   for(int i=0; i<nblock; i++){
      int riemman_range[] = {sizes[(2*i)]-xhalo,sizes[(2*i)+1]-1+xhalo};
      ops_par_loop(Riemann_kernel, "Riemann_kernel", shsgc_grid[i], 1, riemman_range,
                   ops_arg_dat(rho_new[i],  1, S1D_01, "double",OPS_READ),
                   ops_arg_dat(rhou_new[i], 1, S1D_01, "double",OPS_READ),
                   ops_arg_dat(rhoE_new[i], 1, S1D_01, "double",OPS_READ),
                   ops_arg_dat(alam[i],     3, S1D_01, "double",OPS_WRITE),
                   ops_arg_dat(r[i],        9, S1D_01, "double",OPS_WRITE),
                   ops_arg_dat(al[i],       3, S1D_01, "double",OPS_WRITE));

      // limiter function
      int tvd_limiter_range[] = {sizes[(2*i)]+1-xhalo,sizes[(2*i)+1]+xhalo-1};
      ops_par_loop(limiter_kernel, "limiter_kernel", shsgc_grid[i], 1, tvd_limiter_range,
                   ops_arg_dat(al[i],  3, S1D_0M1, "double",OPS_READ),
                   ops_arg_dat(tht[i], 3, S1D_0, "double",OPS_WRITE),
                   ops_arg_dat(gt[i],  3, S1D_0, "double",OPS_WRITE));

      // Second order tvd dissipation
      ops_par_loop(tvd_kernel, "tvd_kernel", shsgc_grid[i], 1, riemman_range,
                   ops_arg_dat(tht[i], 3, S1D_01, "double",OPS_READ),
                   ops_arg_dat(ep2[i], 3, S1D_0, "double",OPS_WRITE));

      // vars
      ops_par_loop(vars_kernel, "vars_kernel", shsgc_grid[i], 1, riemman_range,
                   ops_arg_dat(alam[i],3, S1D_0, "double",OPS_READ),
                   ops_arg_dat(al[i],  3, S1D_0, "double",OPS_READ),
                   ops_arg_dat(gt[i],  3, S1D_01, "double",OPS_READ),
                   ops_arg_dat(cmp[i], 3, S1D_0, "double",OPS_WRITE),
                   ops_arg_dat(cf[i],  3, S1D_0, "double",OPS_WRITE));


      // cal upwind eff
      ops_par_loop(calupwindeff_kernel, "calupwindeff_kernel", shsgc_grid[i], 1, riemman_range,
                   ops_arg_dat(cmp[i],3, S1D_0, "double",OPS_READ),
                   ops_arg_dat(gt[i], 3, S1D_01, "double",OPS_READ),
                   ops_arg_dat(cf[i], 3, S1D_0, "double",OPS_READ),
                   ops_arg_dat(al[i], 3, S1D_0, "double",OPS_READ),
                   ops_arg_dat(ep2[i],3, S1D_0, "double",OPS_READ),
                   ops_arg_dat(r[i],  9, S1D_0, "double",OPS_READ),
                   ops_arg_dat(eff[i],3, S1D_0, "double",OPS_WRITE));

      //fact
      ops_par_loop(fact_kernel, "fact_kernel", shsgc_grid[i], 1, tvd_limiter_range,
                   ops_arg_dat(eff[i],  3, S1D_0M1, "double",OPS_READ),
                   ops_arg_dat(s[i],    3, S1D_0,   "double",OPS_WRITE));


      // update loop
      int rangeupdate[] = {sizes[(2*i)],sizes[(2*i)+1] };
      ops_par_loop(update_kernel, "update_kernel", shsgc_grid[i], 1, rangeupdate,
                   ops_arg_dat(rho_new[i],  1, S1D_0, "double",OPS_RW),
                   ops_arg_dat(rhou_new[i], 1, S1D_0, "double",OPS_RW),
                   ops_arg_dat(rhoE_new[i], 1, S1D_0, "double",OPS_RW),
                   ops_arg_dat(s[i],        3, S1D_0, "double",OPS_READ));
    }
    if(nblock>0){
      ops_halo_transfer(rho_halos);
      ops_halo_transfer(rhou_halos);
      ops_halo_transfer(rhoE_halos);
    }

  }
  ops_timers(&ct1, &et1);
  ops_printf("\nTimings are:\n");
  ops_printf("-----------------------------------------\n");
  ops_printf("Total Wall time %lf\n",et1-et0);
  ops_printf("Wall time per iteration is %g \n",(et1-et0)/niter);
  ops_printf("-----------------------------------------\n");
//   write output to the file
  for(int i=0; i<nblock; i++){
    sprintf(buf,"rhoout%d",i+1);
    ops_print_dat_to_txtfile(rho_new[i], buf);
  }
//   check the output

  double err = 0.0;
  double err1 = 0.0;
  int nump = 0;

  for(int i=0; i<nblock; i++){
    int range_all[] = {sizes[(2*i)],sizes[(2*i)+1]};
    ops_par_loop(checkop_kernel, "checkop_kernel", shsgc_grid[i], 1, range_all,
                 ops_arg_dat(rho_new[i],  1, S1D_0, "double",OPS_READ),
                 ops_arg_dat(x[i],  1, S1D_0, "double",OPS_READ),
                 ops_arg_dat(rhoin[i], 1, S1D_0, "double",OPS_READ),
                 ops_arg_reduce(pre_err, 1, "double", OPS_INC),
                 ops_arg_reduce(post_err, 1, "double", OPS_INC),
                 ops_arg_reduce(num_pre, 1, "int", OPS_INC));
  }
  // error square before shock
  ops_reduction_result(pre_err,&err);
  // error square after shock
  ops_reduction_result(post_err,&err1);
  // number of points after shock
  ops_reduction_result(num_pre,&nump);
  err1 = sqrt(err1)/nump;
  ops_printf("\nErros are:\n");
  ops_printf("-----------------------------------------\n");
  //ops_printf("Pre shock error is: %g\n",sqrt(err)/(nxp-nump));
  ops_printf("Pre shock error is: %g\n",sqrt(err)/(nxp-nump));
  ops_printf("Post shock error is: %g\n",(err1));
  if(err1 -0.0003206 < 1e-6 && nxp == 200)
    ops_printf("Error is correct for 200 gridpoitns\n");
  else if(err1 < 0.001 && nxp != 200)
    ops_printf("Post shock Error is acceptable\n");
  else
    ops_printf("Something is wrong\n");
//   ops_printf("Post shock num is is: %d\n",(err2));
  ops_printf("-----------------------------------------\n");
  ops_exit();
}