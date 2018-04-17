#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int imax, jmax;
double pi  = 2.0 * asin(1.0);

//Including main OPS header file, and setting 2D
#define OPS_2D
#include <ops_seq.h>
//Including applicaiton-specific "user kernels"
#include "laplace_kernels.h" 

int main(int argc, char** argv)
{
  //Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
  ops_init(argc, argv,1);

  //Size along y
  jmax = 4094;
  //Size along x
  imax = 4094;
  int iter_max = 100;

  const double tol = 1.0e-6;
  double error     = 1.0;

  double *A=NULL;
  double *Anew=NULL;

  //
  //Declare & define key data structures
  //
  
  //The 2D block
  ops_block block = ops_decl_block(2, "my_grid");
  //The two datasets
  int size[] = {imax, jmax};
  int base[] = {0,0};
  int d_m[] = {-1,-1};
  int d_p[] = {1,1};
  ops_dat d_A    = ops_decl_dat(block, 1, size, base,
                               d_m, d_p, A,    "double", "A");
  ops_dat d_Anew = ops_decl_dat(block, 1, size, base,
                               d_m, d_p, Anew, "double", "Anew");
  //Two stencils, a 1-point, and a 5-point
  int s2d_00[] = {0,0};
  ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
  int s2d_5pt[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
  ops_stencil S2D_5pt = ops_decl_stencil(2,5,s2d_5pt,"5pt");

  //Reduction handle
  ops_reduction h_err = ops_decl_reduction_handle(sizeof(double), "double", "error");

  //declare and define global constants
  ops_decl_const("imax",1,"int",&imax);
  ops_decl_const("jmax",1,"int",&jmax);
  ops_decl_const("pi",1,"double",&pi);

  ops_partition("");

  // set boundary conditions
  int bottom_range[] = {-1, imax+1, -1, 0};
  ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
      ops_arg_dat(d_A, 1, S2D_00, "double", OPS_WRITE));

  int top_range[] = {-1, imax+1, jmax, jmax+1};
  ops_par_loop(set_zero, "set_zero", block, 2, top_range,
      ops_arg_dat(d_A, 1, S2D_00, "double", OPS_WRITE));

  int left_range[] = {-1, 0, -1, jmax+1};
  ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
      ops_arg_dat(d_A, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_idx());

  int right_range[] = {imax, imax+1, -1, jmax+1};
  ops_par_loop(right_bndcon, "right_bndcon", block, 2, right_range,
      ops_arg_dat(d_A, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_idx());

  ops_printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

  int iter = 0;

  ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
      ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE));

  ops_par_loop(set_zero, "set_zero", block, 2, top_range,
      ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE));

  ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
      ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_idx());

  ops_par_loop(right_bndcon, "right_bndcon", block, 2, right_range,
      ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_idx());


  while ( error > tol && iter < iter_max )
  {
    int interior_range[] = {0,imax,0,jmax};
    ops_par_loop(apply_stencil, "apply_stencil", block, 2, interior_range,
        ops_arg_dat(d_A,    1, S2D_5pt, "double", OPS_READ),
        ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_reduce(h_err, 1, "double", OPS_MAX));
    ops_reduction_result(h_err, &error);

    ops_par_loop(copy, "copy", block, 2, interior_range,
        ops_arg_dat(d_A,    1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_READ));

    if(iter % 10 == 0) ops_printf("%5d, %0.6f\n", iter, error);        
    iter++;
  }

  ops_printf("%5d, %0.6f\n", iter, error);        

  ops_timing_output(stdout);

  double err_diff = fabs((100.0*(error/2.421354960840227e-03))-100.0);
  printf("Total error is within %3.15E %% of the expected error\n",err_diff);
  if(err_diff < 0.001)
    printf("This run is considered PASSED\n");
  else
    printf("This test is considered FAILED\n");

  //Finalising the OPS library
  ops_exit();
  free(A);
  free(Anew);
  return 0;
}

