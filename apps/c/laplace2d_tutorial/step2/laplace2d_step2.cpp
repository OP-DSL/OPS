#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

//Including main OPS header file, and setting 2D
#define OPS_2D
#include <ops_seq_v2.h>
//Including applicaiton-specific "user kernels"
#include "laplace_kernels.h" 

int main(int argc, const char** argv)
{
  //Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
  ops_init(argc, argv,1);

  //Size along y
  int jmax = 4094;
  //Size along x
  int imax = 4094;
  int iter_max = 100;

  double pi  = 2.0 * asin(1.0);
  const double tol = 1.0e-6;
  double error     = 1.0;

  double *A;
  double *Anew;

  A    = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
  Anew = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));

  memset(A, 0, (imax+2) * (jmax+2) * sizeof(double));

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

  // set boundary conditions
  for (int i = 0; i < imax+2; i++)
    A[(0)*(imax+2)+i]   = 0.0;

  for (int i = 0; i < imax+2; i++)
    A[(jmax+1)*(imax+2)+i] = 0.0;

  for (int j = 0; j < jmax+2; j++)
  {
    A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1));
  }

  for (int j = 0; j < imax+2; j++)
  {
    A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1))*exp(-pi);
  }

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

  int iter = 0;

  for (int i = 1; i < imax+2; i++)
    Anew[(0)*(imax+2)+i]   = 0.0;

  for (int i = 1; i < imax+2; i++)
    Anew[(jmax+1)*(imax+2)+i] = 0.0;

  for (int j = 1; j < jmax+2; j++)
    Anew[(j)*(imax+2)+0]   = sin(pi * j / (jmax+1));

  for (int j = 1; j < jmax+2; j++)
    Anew[(j)*(imax+2)+jmax+1] = sin(pi * j / (jmax+1))*expf(-pi);


  while ( error > tol && iter < iter_max )
  {
    error = 0.0;
    for( int j = 1; j < jmax+1; j++ )
    {
      for( int i = 1; i < imax+1; i++)
      {
        Anew[(j)*(imax+2)+i] = 0.25f * ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
            + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);
        error = fmax( error, fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]));
      }
    }

    for( int j = 1; j < jmax+1; j++ )
    {
      for( int i = 1; i < imax+1; i++)
      {
        A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];    
      }
    }
    if(iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);        
    iter++;
  }

  printf("%5d, %0.6f\n", iter, error);        

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

