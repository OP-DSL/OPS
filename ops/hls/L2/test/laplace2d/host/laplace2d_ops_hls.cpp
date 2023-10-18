
// Auto-generated at 2023-10-02 14:24:24.741093 by ops-translator

extern void ops_init_backend(int argc, const char** argv);

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int imax, jmax;
//#pragma acc declare create(imax)
//#pragma acc declare create(jmax)
float pi  = 2.0 * asin(1.0);
//#pragma acc declare create(pi)

//Including main OPS header file, and setting 2D
#define OPS_2D
#include <ops_hls_rt_support.h>
//Including applicaiton-specific "user kernels"
/* ops_par_loop declarations */

void ops_par_loop_set_zero(int, int*, ops::hls::Grid<float>&);

void ops_par_loop_left_bndcon(int, int*, ops::hls::Grid<float>&, const float, const int);

void ops_par_loop_right_bndcon(int, int*, ops::hls::Grid<float>&, const float, const int);

void ops_par_loop_apply_stencil(int, int*, ops::hls::Grid<float>&, ops::hls::Grid<float>&);

void ops_par_loop_copy(int, int*, ops::hls::Grid<float>&, ops::hls::Grid<float>&);

#include "hls_kernels.hpp"

int main(int argc, const char** argv)
{
  //Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
	ops_init_backend(argc, argv);


  //Size along y
  jmax = 10;
  //Size along x
  imax = 10;
  int iter_max = 100;

  const float tol = 1.0e-6;
  float error     = 1.0;

  float *A=NULL;
  float *Anew=NULL;

  //
  //Declare & define key data structures
  //
  
  // //The 2D block
  // ops_block block = ops_decl_block(2, "my_grid");
  //The two datasets
  int size[] = {imax, jmax};
  int base[] = {0,0};
  int d_m[] = {-1,-1};
  int d_p[] = {1,1};
  ops::hls::Grid<float> d_A = createGrid(2, size, base, d_m, d_p, A);
  ops::hls::Grid<float> d_Anew = createGrid(2, size, base, d_m, d_p, Anew);

  //Two stencils, a 1-point, and a 5-point
  // int s2d_00[] = {0,0};
  // ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
  // int s2d_5pt[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
  // ops_stencil S2D_5pt = ops_decl_stencil(2,5,s2d_5pt,"5pt");

  //Reduction handle
  // ops_reduction h_err = ops_decl_reduction_handle(sizeof(float), "float", "error");

  //declare and define global constants
// #pragma acc update device(imax)
// #pragma acc update device(jmax)
// #pragma acc update device(pi)

  // ops_partition("");

//   set boundary conditions
  int bottom_range[] = {-1, imax+1, -1, 0};
  ops_par_loop_set_zero(2, bottom_range,
      d_A);

  int top_range[] = {-1, imax+1, jmax, jmax+1};
  ops_par_loop_set_zero(2, top_range,
      d_A);

  int left_range[] = {-1, 0, -1, jmax+1};
  ops_par_loop_left_bndcon(2, left_range,
      d_A, pi, jmax);

  int right_range[] = {imax-1, imax, -1, jmax+1};
  ops_par_loop_right_bndcon(2, right_range,
      d_A, pi, jmax);

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

//  int iter = 0;

  ops_par_loop_set_zero(2, bottom_range,
      d_Anew);

  ops_par_loop_set_zero( 2, top_range,
      d_Anew);

  ops_par_loop_left_bndcon(2, left_range,
      d_Anew, pi, jmax);

  ops_par_loop_right_bndcon(2, right_range,
      d_Anew, pi, jmax);

  getGrid(d_A);
  getGrid(d_Anew);

  int num_iter = 100;
  for (int iter = 0; iter < num_iter; iter++)
  {
    int interior_range[] = {0,imax,0,jmax};
    ops_par_loop_apply_stencil(2, interior_range,
        d_A,
        d_Anew);

    ops_par_loop_copy(2, interior_range,
        d_Anew,
        d_A);

    if(iter % 10 == 0) printf("%5d\n", iter);
  }

  // ops_printf("%5d, %0.6f\n", iter, error);        

  // ops_timing_output(std::cout);

  // float err_diff = fabs((100.0*(error/2.421354960840227e-03))-100.0);
  // printf("Total error is within %3.15E %% of the expected error\n",err_diff);
  // if(err_diff < 0.001)
  //   printf("This run is considered PASSED\n");
  // else
  //   printf("This test is considered FAILED\n");

  //Finalising the OPS library
  // ops_exit();
  // free(A);
  // free(Anew);
  getGrid(d_A);
  ops::hls::FPGA::getInstance()->finish();
  printGrid2D<float>(d_A, "d_A");
  return 0;
}
