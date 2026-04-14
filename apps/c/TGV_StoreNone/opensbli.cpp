#include <stdlib.h> 
#include <string.h> 
#include <math.h>
#ifdef PROFILE_ITT
#include <ittnotify.h>
#endif
int block0np0;
int block0np1;
int block0np2;
double Delta0block0;
double Delta1block0;
double Delta2block0;
int niter;
double dt;
double gama;
double Minf;
double Re;
double Pr;
double inv_0;
double inv_1;
double inv_2;
double inv_3;
double inv_4;
double inv_5;
double rc6;
double rc7;
double rcinv8;
double rc9;
double rc10;
double rcinv11;
double rcinv12;
double rcinv13;
double rc14;
#define OPS_3D
#include "ops_seq.h"
#include "opensbliblock00_kernels.h"
int main(int argc, char **argv) 
{
  if (argc < 4) { printf("Error, requires 3 arguments for the mesh size\n"); exit(-1); }
  block0np0 = atoi(argv[1]);
  block0np1 = atoi(argv[2]);
  block0np2 = atoi(argv[3]);
  Delta0block0 = M_PI/(block0np0-1);
  Delta1block0 = M_PI/(block0np1-1);
  Delta2block0 = M_PI/(block0np2-1);
  niter = 500;
  double rkold[] = {1.0/4.0, 3.0/20.0, 3.0/5.0};
  double rknew[] = {2.0/3.0, 5.0/12.0, 3.0/5.0};
  dt = 0.005;
  gama = 1.4;
  Minf = 0.1;
  Re = 800.0;
  Pr = 0.71;
  inv_0 = 1.0/Delta0block0;
  inv_1 = 1.0/Delta2block0;
  inv_2 = 1.0/Delta1block0;
  inv_3 = pow(Delta0block0, -2);
  inv_4 = pow(Delta1block0, -2);
  inv_5 = pow(Delta2block0, -2);
  rc6 = 1.0/2.0;
  rc7 = 1.0/12.0;
  rcinv8 = 1.0/Re;
  rc9 = 1.0/3.0;
  rc10 = 4.0/3.0;
  rcinv11 = 1.0/Pr;
  rcinv12 = pow(Minf, -2);
  rcinv13 = 1.0/(gama - 1);
  rc14 = 2.0/3.0;
  // Initializing OPS 
  ops_init(argc,argv,1);
  ops_decl_const("block0np0" , 1, "int", &block0np0);
  ops_decl_const("block0np1" , 1, "int", &block0np1);
  ops_decl_const("block0np2" , 1, "int", &block0np2);
  ops_decl_const("Delta0block0" , 1, "double", &Delta0block0);
  ops_decl_const("Delta1block0" , 1, "double", &Delta1block0);
  ops_decl_const("Delta2block0" , 1, "double", &Delta2block0);
  ops_decl_const("niter" , 1, "int", &niter);
  ops_decl_const("dt" , 1, "double", &dt);
  ops_decl_const("gama" , 1, "double", &gama);
  ops_decl_const("Minf" , 1, "double", &Minf);
  ops_decl_const("Re" , 1, "double", &Re);
  ops_decl_const("Pr" , 1, "double", &Pr);
  ops_decl_const("inv_0" , 1, "double", &inv_0);
  ops_decl_const("inv_1" , 1, "double", &inv_1);
  ops_decl_const("inv_2" , 1, "double", &inv_2);
  ops_decl_const("inv_3" , 1, "double", &inv_3);
  ops_decl_const("inv_4" , 1, "double", &inv_4);
  ops_decl_const("inv_5" , 1, "double", &inv_5);
  ops_decl_const("rc6" , 1, "double", &rc6);
  ops_decl_const("rc7" , 1, "double", &rc7);
  ops_decl_const("rcinv8" , 1, "double", &rcinv8);
  ops_decl_const("rc9" , 1, "double", &rc9);
  ops_decl_const("rc10" , 1, "double", &rc10);
  ops_decl_const("rcinv11" , 1, "double", &rcinv11);
  ops_decl_const("rcinv12" , 1, "double", &rcinv12);
  ops_decl_const("rcinv13" , 1, "double", &rcinv13);
  ops_decl_const("rc14" , 1, "double", &rc14);
  // Define and Declare OPS Block
  ops_block opensbliblock00 = ops_decl_block(3, "opensbliblock00");
#include "defdec_data_set.h"
  // Define and declare stencils
  int stencil_0_00temp[] = {-2, 0, 0, -1, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0};
  ops_stencil stencil_0_00 = ops_decl_stencil(3,13,stencil_0_00temp,"stencil_0_00temp");
  int stencil_0_01temp[] = {-2, 0, 0, -1, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, -2, 0, 0, -1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0};
  ops_stencil stencil_0_01 = ops_decl_stencil(3,12,stencil_0_01temp,"stencil_0_01temp");
  int stencil_0_02temp[] = {0, -2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0};
  ops_stencil stencil_0_02 = ops_decl_stencil(3,5,stencil_0_02temp,"stencil_0_02temp");
  int stencil_0_03temp[] = {0, 0, 0};
  ops_stencil stencil_0_03 = ops_decl_stencil(3,1,stencil_0_03temp,"stencil_0_03temp");
  int stencil_0_04temp[] = {-2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0};
  ops_stencil stencil_0_04 = ops_decl_stencil(3,5,stencil_0_04temp,"stencil_0_04temp");
  int stencil_0_05temp[] = {0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 2};
  ops_stencil stencil_0_05 = ops_decl_stencil(3,5,stencil_0_05temp,"stencil_0_05temp");
  int stencil_0_06temp[] = {0, -2, 0, 0, -1, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0};
  ops_stencil stencil_0_06 = ops_decl_stencil(3,9,stencil_0_06temp,"stencil_0_06temp");
  int stencil_0_07temp[] = {-2, 0, 0, -1, 0, 0, 1, 0, 0, 2, 0, 0};
  ops_stencil stencil_0_07 = ops_decl_stencil(3,4,stencil_0_07temp,"stencil_0_07temp");
  int stencil_0_08temp[] = {0, 0, -2, 0, 0, -1, 0, 0, 1, 0, 0, 2};
  ops_stencil stencil_0_08 = ops_decl_stencil(3,4,stencil_0_08temp,"stencil_0_08temp");
  int stencil_0_09temp[] = {0, -2, 0, 0, -1, 0, 0, 1, 0, 0, 2, 0};
  ops_stencil stencil_0_09 = ops_decl_stencil(3,4,stencil_0_09temp,"stencil_0_09temp");
  // Init OPS partition
  ops_partition("");
	ops_diagnostic_output();
  ops_printf("OpenSBLI TGV case SN version size %dx%dx%d\n", block0np0, block0np1, block0np2);

  int iteration_range_32_block0[] = {-5, block0np0 + 5, -5, block0np1 + 5, -5, block0np2 + 5};
  ops_par_loop(opensbliblock00Kernel032, "Grid_based_initialisation0", opensbliblock00, 3, iteration_range_32_block0,
      ops_arg_dat(rhoE_B0, 1, stencil_0_03, "double", OPS_WRITE),
      ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_WRITE),
      ops_arg_dat(rhou0_B0, 1, stencil_0_03, "double", OPS_WRITE),
      ops_arg_dat(rhou2_B0, 1, stencil_0_03, "double", OPS_WRITE),
      ops_arg_dat(rhou1_B0, 1, stencil_0_03, "double", OPS_WRITE),
      ops_arg_idx());

  opensbliblock00->instance->reset_power_counters();
#ifdef PROFILE_ITT
  __itt_resume();
#endif
  double cpu_start0, elapsed_start0;
  ops_timers(&cpu_start0, &elapsed_start0);
  for(int iter=0; iter<=niter - 1; iter++)
  {
    int iteration_range_26_block0[] = {0, 1, -2, block0np1 + 2, -2, block0np2 + 2};
    ops_par_loop(opensbliblock00Kernel026, "Symmetry boundary dir0 side0", opensbliblock00, 3, iteration_range_26_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rho_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rhou0_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rhou2_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rhou1_B0, 1, stencil_0_07, "double", OPS_RW));


    int iteration_range_27_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
    ops_par_loop(opensbliblock00Kernel027, "Symmetry boundary dir0 side1", opensbliblock00, 3, iteration_range_27_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rho_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rhou0_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rhou2_B0, 1, stencil_0_07, "double", OPS_RW),
        ops_arg_dat(rhou1_B0, 1, stencil_0_07, "double", OPS_RW));


    int iteration_range_28_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
    ops_par_loop(opensbliblock00Kernel028, "Symmetry boundary dir1 side0", opensbliblock00, 3, iteration_range_28_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rho_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rhou0_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rhou2_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rhou1_B0, 1, stencil_0_09, "double", OPS_RW));


    int iteration_range_29_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1, -2, block0np2 + 2};
    ops_par_loop(opensbliblock00Kernel029, "Symmetry boundary dir1 side1", opensbliblock00, 3, iteration_range_29_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rho_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rhou0_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rhou2_B0, 1, stencil_0_09, "double", OPS_RW),
        ops_arg_dat(rhou1_B0, 1, stencil_0_09, "double", OPS_RW));


    int iteration_range_30_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, 0, 1};
    ops_par_loop(opensbliblock00Kernel030, "Symmetry boundary dir2 side0", opensbliblock00, 3, iteration_range_30_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rho_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rhou0_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rhou2_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rhou1_B0, 1, stencil_0_08, "double", OPS_RW));


    int iteration_range_31_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, block0np2 - 1, block0np2};
    ops_par_loop(opensbliblock00Kernel031, "Symmetry boundary dir2 side1", opensbliblock00, 3, iteration_range_31_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rho_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rhou0_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rhou2_B0, 1, stencil_0_08, "double", OPS_RW),
        ops_arg_dat(rhou1_B0, 1, stencil_0_08, "double", OPS_RW));


    int iteration_range_33_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
    ops_par_loop(opensbliblock00Kernel033, "Save equations", opensbliblock00, 3, iteration_range_33_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_03, "double", OPS_READ),
        ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_READ),
        ops_arg_dat(rhou0_B0, 1, stencil_0_03, "double", OPS_READ),
        ops_arg_dat(rhou2_B0, 1, stencil_0_03, "double", OPS_READ),
        ops_arg_dat(rhou1_B0, 1, stencil_0_03, "double", OPS_READ),
        ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_03, "double", OPS_WRITE),
        ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_03, "double", OPS_WRITE),
        ops_arg_dat(rho_RKold_B0, 1, stencil_0_03, "double", OPS_WRITE),
        ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_03, "double", OPS_WRITE),
        ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_03, "double", OPS_WRITE));


    for(int stage=0; stage<=2; stage++)
    {
      int iteration_range_1_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel001, "CRu0_B0", opensbliblock00, 3, iteration_range_1_block0,
          ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_5_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel005, "CRu1_B0", opensbliblock00, 3, iteration_range_5_block0,
          ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_9_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel009, "CRu2_B0", opensbliblock00, 3, iteration_range_9_block0,
          ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_17_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel017, "CRp_B0", opensbliblock00, 3, iteration_range_17_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(p_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_18_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel018, "CRT_B0", opensbliblock00, 3, iteration_range_18_block0,
          ops_arg_dat(p_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(T_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_0_block0[] = {0, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel000, "Derivative evaluation CD u0_B0 x0 ", opensbliblock00, 3, iteration_range_0_block0,
          ops_arg_dat(u0_B0, 1, stencil_0_07, "double", OPS_READ),
          ops_arg_dat(wk0_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_2_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel002, "Derivative evaluation CD u0_B0 x2 ", opensbliblock00, 3, iteration_range_2_block0,
          ops_arg_dat(u0_B0, 1, stencil_0_08, "double", OPS_READ),
          ops_arg_dat(wk1_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_3_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel003, "Derivative evaluation CD u0_B0 x1 ", opensbliblock00, 3, iteration_range_3_block0,
          ops_arg_dat(u0_B0, 1, stencil_0_09, "double", OPS_READ),
          ops_arg_dat(wk2_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_4_block0[] = {0, block0np0, -2, block0np1 + 2, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel004, "Derivative evaluation CD u1_B0 x0 ", opensbliblock00, 3, iteration_range_4_block0,
          ops_arg_dat(u1_B0, 1, stencil_0_07, "double", OPS_READ),
          ops_arg_dat(wk3_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_6_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel006, "Derivative evaluation CD u1_B0 x2 ", opensbliblock00, 3, iteration_range_6_block0,
          ops_arg_dat(u1_B0, 1, stencil_0_08, "double", OPS_READ),
          ops_arg_dat(wk4_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_7_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel007, "Derivative evaluation CD u1_B0 x1 ", opensbliblock00, 3, iteration_range_7_block0,
          ops_arg_dat(u1_B0, 1, stencil_0_09, "double", OPS_READ),
          ops_arg_dat(wk5_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_8_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel008, "Derivative evaluation CD u2_B0 x0 ", opensbliblock00, 3, iteration_range_8_block0,
          ops_arg_dat(u2_B0, 1, stencil_0_07, "double", OPS_READ),
          ops_arg_dat(wk6_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_10_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel010, "Derivative evaluation CD u2_B0 x2 ", opensbliblock00, 3, iteration_range_10_block0,
          ops_arg_dat(u2_B0, 1, stencil_0_08, "double", OPS_READ),
          ops_arg_dat(wk7_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_11_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel011, "Derivative evaluation CD u2_B0 x1 ", opensbliblock00, 3, iteration_range_11_block0,
          ops_arg_dat(u2_B0, 1, stencil_0_09, "double", OPS_READ),
          ops_arg_dat(wk8_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_24_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel024, "Convective terms", opensbliblock00, 3, iteration_range_24_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_04, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_05, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_02, "double", OPS_READ),
          ops_arg_dat(wk7_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(p_B0, 1, stencil_0_01, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(wk5_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(wk0_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(Residual4_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(Residual3_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(Residual1_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(Residual0_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(Residual2_B0, 1, stencil_0_03, "double", OPS_WRITE));


      int iteration_range_25_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel025, "Viscous terms", opensbliblock00, 3, iteration_range_25_block0,
          ops_arg_dat(wk8_B0, 1, stencil_0_05, "double", OPS_READ),
          ops_arg_dat(wk3_B0, 1, stencil_0_02, "double", OPS_READ),
          ops_arg_dat(wk2_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(wk1_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(wk4_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(wk7_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(T_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_00, "double", OPS_READ),
          ops_arg_dat(wk5_B0, 1, stencil_0_05, "double", OPS_READ),
          ops_arg_dat(wk0_B0, 1, stencil_0_06, "double", OPS_READ),
          ops_arg_dat(wk6_B0, 1, stencil_0_05, "double", OPS_READ),
          ops_arg_dat(Residual4_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(Residual1_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(Residual3_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(Residual2_B0, 1, stencil_0_03, "double", OPS_RW));


      int iteration_range_35_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel035, "Sub stage advancement", opensbliblock00, 3, iteration_range_35_block0,
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual3_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual1_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual0_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual4_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual2_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhoE_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(rho_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(rhou0_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(rhou2_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_dat(rhou1_B0, 1, stencil_0_03, "double", OPS_WRITE),
          ops_arg_gbl(&rknew[stage], 1, "double", OPS_READ));


      int iteration_range_34_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(opensbliblock00Kernel034, "Temporal solution advancement", opensbliblock00, 3, iteration_range_34_block0,
          ops_arg_dat(Residual3_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual1_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual0_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual4_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(Residual2_B0, 1, stencil_0_03, "double", OPS_READ),
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_03, "double", OPS_RW),
          ops_arg_gbl(&rkold[stage], 1, "double", OPS_READ));


      int iteration_range_26_block0[] = {0, 1, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel026, "Symmetry boundary dir0 side0", opensbliblock00, 3, iteration_range_26_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_07, "double", OPS_RW));


      int iteration_range_27_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel027, "Symmetry boundary dir0 side1", opensbliblock00, 3, iteration_range_27_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_07, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_07, "double", OPS_RW));


      int iteration_range_28_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel028, "Symmetry boundary dir1 side0", opensbliblock00, 3, iteration_range_28_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_09, "double", OPS_RW));


      int iteration_range_29_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1, -2, block0np2 + 2};
      ops_par_loop(opensbliblock00Kernel029, "Symmetry boundary dir1 side1", opensbliblock00, 3, iteration_range_29_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_09, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_09, "double", OPS_RW));


      int iteration_range_30_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, 0, 1};
      ops_par_loop(opensbliblock00Kernel030, "Symmetry boundary dir2 side0", opensbliblock00, 3, iteration_range_30_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_08, "double", OPS_RW));


      int iteration_range_31_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, block0np2 - 1, block0np2};
      ops_par_loop(opensbliblock00Kernel031, "Symmetry boundary dir2 side1", opensbliblock00, 3, iteration_range_31_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_08, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_08, "double", OPS_RW));


    }
    ops_execute(opensbliblock00->instance);
  }
  double cpu_end0, elapsed_end0;
  ops_timers(&cpu_end0, &elapsed_end0);
#ifdef PROFILE_ITT
  __itt_pause();
#endif
  ops_timing_output(std::cout);
  ops_printf("\nTimings are:\n");
  ops_printf("-----------------------------------------\n");
  ops_printf("Total Wall time %lf\n",elapsed_end0-elapsed_start0);
  ops_exit();
  //Main program end 
}
