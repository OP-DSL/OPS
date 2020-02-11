#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Global constants in the equations are
double rkold[3];
double c0;
double rknew[3];
double rc0;
double rc1;
double rc2;
double rc3;
int nx0;
double deltai0;
double deltat;
// OPS header file
#define OPS_1D
#include "ops_seq_v2.h"
#include "complex_numbers_block_0_kernel.h"

// main program start
int main (int argc, char **argv) 
{

   c0 = 0.500000000000000;
   rc0 = 1.0/280.0;
   rc1 = 4.0/105.0;
   rc2 = 1.0/5.0;
   rc3 = 4.0/5.0;
   nx0 = 1000;
   deltai0 = 0.00100000000000000;
   deltat = 0.000400000000000000;
   rkold[0] = 1.0/4.0;
   rkold[1] = 3.0/20.0;
   rkold[2] = 3.0/5.0;
   rknew[0] = 2.0/3.0;
   rknew[1] = 5.0/12.0;
   rknew[2] = 3.0/5.0;

   // Initializing OPS 
   ops_init(argc,argv,1);

   ops_decl_const("c0" , 1, "double", &c0);
   ops_decl_const("rc0" , 1, "double", &rc0);
   ops_decl_const("rc1" , 1, "double", &rc1);
   ops_decl_const("rc2" , 1, "double", &rc2);
   ops_decl_const("rc3" , 1, "double", &rc3);
   ops_decl_const("nx0" , 1, "int", &nx0);
   ops_decl_const("deltai0" , 1, "double", &deltai0);
   ops_decl_const("deltat" , 1, "double", &deltat);

   // Defining block in OPS Format
   ops_block complex_numbers_block;

   // Initialising block in OPS Format
   complex_numbers_block = ops_decl_block(1, "complex_numbers_block");

   // Define dataset
   ops_dat phi;
   ops_dat phi_old;
   ops_dat wk0;
   ops_dat wk1;

   // Initialise/allocate OPS dataset.
   int halo_p[] = {4};
   int halo_m[] = {-4};
   int size[] = {nx0};
   int base[] = {0};
   double* val = NULL;
   phi = ops_decl_dat(complex_numbers_block, 1, size, base, halo_m, halo_p, val, "double", "phi");
   phi_old = ops_decl_dat(complex_numbers_block, 1, size, base, halo_m, halo_p, val, "double", "phi_old");
   wk0 = ops_decl_dat(complex_numbers_block, 1, size, base, halo_m, halo_p, val, "double", "wk0");
   wk1 = ops_decl_dat(complex_numbers_block, 1, size, base, halo_m, halo_p, val, "double", "wk1");

   // Declare all the stencils used 
   int stencil1_temp[] = {0};
   ops_stencil stencil1 = ops_decl_stencil(1,1,stencil1_temp,"0");
   int stencil0_temp[] = {-4,-3,-2,-1,1,2,3,4};
   ops_stencil stencil0 = ops_decl_stencil(1,8,stencil0_temp,"-4,-3,-2,-1,1,2,3,4");

   ops_reduction real = ops_decl_reduction_handle(sizeof(double), "double", "reduction_real");
   ops_reduction imaginary = ops_decl_reduction_handle(sizeof(double), "double", "reduction_imaginary");

   // Boundary condition exchange code
   ops_halo_group halo_exchange0 ;
   {
      int halo_iter[] = {4};
      int from_base[] = {0};
      int to_base[] = {nx0};
      int dir[] = {1};
      ops_halo halo0 = ops_decl_halo(phi, phi, halo_iter, from_base, to_base, dir, dir);
      ops_halo grp[] = {halo0};
      halo_exchange0 = ops_decl_halo_group(1,grp);
   }
   // Boundary condition exchange code
   ops_halo_group halo_exchange1 ;
   {
      int halo_iter[] = {4};
      int from_base[] = {nx0 - 4};
      int to_base[] = {-4};
      int dir[] = {1};
      ops_halo halo0 = ops_decl_halo(phi, phi, halo_iter, from_base, to_base, dir, dir);
      ops_halo grp[] = {halo0};
      halo_exchange1 = ops_decl_halo_group(1,grp);
   }

   // Init OPS partition
   ops_partition("");

   int iter_range5[] = {-4, nx0 + 4};
   ops_par_loop(complex_numbers_block0_5_kernel, "Initialisation", complex_numbers_block, 1, iter_range5,
   ops_arg_dat(phi, 1, stencil1, "double", OPS_WRITE),
   ops_arg_idx());



   // Boundary condition exchange calls
   ops_halo_transfer(halo_exchange0);
   // Boundary condition exchange calls
   ops_halo_transfer(halo_exchange1);

   double cpu_start, elapsed_start;
   ops_timers(&cpu_start, &elapsed_start);
   
   for (int iteration=0; iteration<1; iteration++){


      int iter_range4[] = {-4, nx0 + 4};
      ops_par_loop(complex_numbers_block0_4_kernel, "Save equations", complex_numbers_block, 1, iter_range4,
      ops_arg_dat(phi, 1, stencil1, "double", OPS_READ),
      ops_arg_dat(phi_old, 1, stencil1, "double", OPS_WRITE));



      for (int stage=0; stage<3; stage++){


         int iter_range0[] = {0, nx0};
         ops_par_loop(complex_numbers_block0_0_kernel, "D(phi[x0 t] x0)", complex_numbers_block, 1, iter_range0,
         ops_arg_dat(phi, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk0, 1, stencil1, "double", OPS_WRITE));


         int iter_range1[] = {0, nx0};
         ops_par_loop(complex_numbers_block0_1_kernel, "Residual of equation", complex_numbers_block, 1, iter_range1,
         ops_arg_dat(wk0, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(wk1, 1, stencil1, "double", OPS_WRITE));


         int iter_range2[] = {-4, nx0 + 4};
         ops_par_loop(complex_numbers_block0_2_kernel, "RK new (subloop) update", complex_numbers_block, 1, iter_range2,
         ops_arg_dat(phi_old, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(wk1, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(phi, 1, stencil1, "double", OPS_WRITE),
         ops_arg_gbl(&rknew[stage], 1, "double", OPS_READ));


         int iter_range3[] = {-4, nx0 + 4};
         ops_par_loop(complex_numbers_block0_3_kernel, "RK old update", complex_numbers_block, 1, iter_range3,
         ops_arg_dat(wk1, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(phi_old, 1, stencil1, "double", OPS_RW),
         ops_arg_gbl(&rkold[stage], 1, "double", OPS_READ));

         // Boundary condition exchange calls
         ops_halo_transfer(halo_exchange0);
         // Boundary condition exchange calls
         ops_halo_transfer(halo_exchange1);

      }

      int iter_range0[] = {0, nx0};
      ops_par_loop(complex_numbers_block0_0_kernel, "D(phi[x0 t] x0)",
                   complex_numbers_block, 1, iter_range0,
                   ops_arg_dat(phi, 1, stencil0, "double", OPS_READ),
                   ops_arg_dat(wk0, 1, stencil1, "double", OPS_WRITE));

      int iter_range1[] = {0, nx0};
      ops_par_loop(complex_numbers_block0_1_kernel, "Residual of equation",
                   complex_numbers_block, 1, iter_range1,
                   ops_arg_dat(wk0, 1, stencil1, "double", OPS_READ),
                   ops_arg_dat(wk1, 1, stencil1, "double", OPS_WRITE));

      int iter_range2[] = {-4, nx0 + 4};
      ops_par_loop(complex_numbers_block0_2_kernel, "RK new (subloop) update",
                   complex_numbers_block, 1, iter_range2,
                   ops_arg_dat(phi_old, 1, stencil1, "double", OPS_READ),
                   ops_arg_dat(wk1, 1, stencil1, "double", OPS_READ),
                   ops_arg_dat(phi, 1, stencil1, "double", OPS_WRITE),
                   ops_arg_gbl(&rknew[stage], 1, "double", OPS_READ));

      int iter_range3[] = {-4, nx0 + 4};
      ops_par_loop(complex_numbers_block0_3_kernel, "RK old update",
                   complex_numbers_block, 1, iter_range3,
                   ops_arg_dat(wk1, 1, stencil1, "double", OPS_READ),
                   ops_arg_dat(phi_old, 1, stencil1, "double", OPS_RW),
                   ops_arg_gbl(&rkold[stage], 1, "double", OPS_READ));

      // Boundary condition exchange calls
      ops_halo_transfer(halo_exchange0);
      // Boundary condition exchange calls
      ops_halo_transfer(halo_exchange1);
    }

    int iter_range0[] = {0, nx0};
    ops_par_loop(complex_numbers_block0_cn_kernel, "Complex numbers",
                 complex_numbers_block, 1, iter_range0,
                 ops_arg_dat(phi, 1, stencil0, "double", OPS_READ),
                 ops_arg_reduce(real, 1, "double", OPS_INC),
                 ops_arg_reduce(imaginary, 1, "double", OPS_INC));
  }

  double cpu_end, elapsed_end;
  ops_timers(&cpu_end, &elapsed_end);

  ops_printf("\nTimings are:\n");
  ops_printf("-----------------------------------------\n");
  ops_printf("Total Wall time %lf\n", elapsed_end - elapsed_start);

  ops_fetch_block_hdf5_file(complex_numbers_block, "complex_numbers_2500.h5");
  ops_fetch_dat_hdf5_file(phi, "complex_numbers_2500.h5");

  // Exit OPS
  ops_exit();

}
