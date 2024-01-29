/* STEP 4 - Handling complex stencil and periodic boundaries
*/
#include <iostream>
#include <fstream>
#include <cstdlib>

double OMEGA = 1.0;
double rho0 = 1.0;
double deltaUX = 10e-6;

// Including main OPS header file, and setting 2D
#define OPS_2D
#include <ops_seq_v2.h>
// Including applicaiton-specific "elemental kernels"
#include "lattice_kernels.h"

static inline int mod(int v, int m) {
    int val = v%m;
    if (val<0) val = m+val;
    return val;
}

int main(int argc, char ** argv) {
    // Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
    ops_init(argc, argv, 1);

    const int NX = 128;
    const int NY = 128;

    const double W[] = {4.0/9.0,1.0/9.0,1.0/36.0,1.0/9.0,1.0/36.0,1.0/9.0,1.0/36.0,1.0/9.0,1.0/36.0};
    const int cx[] = {0,0,1,1, 1, 0,-1,-1,-1};
    const int cy[] = {0,1,1,0,-1,-1,-1, 0, 1};

    const int opposite[] = {0,5,6,7,8,1,2,3,4};

    double energy;

    double ct0,et0,ct1,et1; //timer variables

    int *temp_int = NULL;
    double *temp_dbl = NULL;

    //====================================
    // Declare & define key data structures
    //====================================

    // The 2D block
    ops_block lb_block = ops_decl_block(2, "lattice-boltzmann_grid");

    int size[] = {NX, NY};
    int base[] = {0,0};     // this is in C indexing - start from 0

    int d_m[]  = {0,0};     // max boundary depths for the dat in the negative direction
    int d_p[]  = {0,0};     // max boundary depths for the dat in the possitive direction
    // Single dim dats
    ops_dat d_SOLID = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_int, "int", "SOLID");
    ops_dat d_rho   = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl, "double", "rho"); 
    ops_dat d_ux    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl, "double", "ux");
    ops_dat d_uy    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl, "double", "uy");

    // Multi dim dats
    ops_dat d_N_SOLID   = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl, "double", "N_SOLID");

    d_m[0] = -1; d_m[1] = -1;
    d_p[0] = 1;  d_p[1] = 1;
    ops_dat d_N         = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl, "double", "N");
    ops_dat d_workArray = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl, "double", "workArray");

    // Declare stencils
    int s2d_00[] = {0,0};
    ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
    int s2d_9pt[] = {-1,-1, -1,0, -1,1, 0,-1, 0,0, 0,1, 1,-1, 1,0, 1,1}; //9-point stencil
    ops_stencil S2D_9pt = ops_decl_stencil(2,9,s2d_9pt,"9pt");

    // Declare and define global constants
    ops_decl_const("OMEGA",1,"double",&OMEGA);
    ops_decl_const("rho0",1,"double",&rho0);
    ops_decl_const("deltaUX",1,"double",&deltaUX);

    // Reduction handle
    ops_reduction h_energy = ops_decl_reduction_handle(sizeof(double), "double", "error");

    int dir_from[]  = {1,2};
    int dir_to[]    = {1,2};

    // Periodic Boundaries 
    ops_halo *halos_x = (ops_halo*)malloc(2*sizeof(ops_halo));
    ops_halo *halos_y = (ops_halo*)malloc(2*sizeof(ops_halo));

    {
        int iter_size[] = {1,NY+2};
        int base_from[] = {0,-1};
        int base_to[]   = {NX,-1};
        halos_x[0] = ops_decl_halo(d_workArray, d_workArray, iter_size, base_from, base_to, dir_from, dir_to);
    }
    {
        int iter_size[] = {1,NY+2};
        int base_from[] = {NX-1,-1};
        int base_to[]   = {-1,-1};
        halos_x[1] = ops_decl_halo(d_workArray, d_workArray, iter_size, base_from, base_to, dir_from, dir_to);
    }

    {
        int iter_size[] = {NX+2,1};
        int base_from[] = {-1,0};
        int base_to[]   = {-1,NY};
        halos_y[0] = ops_decl_halo(d_workArray, d_workArray, iter_size, base_from, base_to, dir_from, dir_to);
    }
    {
        int iter_size[] = {NX+2,1};
        int base_from[] = {-1,NY-1};
        int base_to[]   = {-1,-1};
        halos_y[1] = ops_decl_halo(d_workArray, d_workArray, iter_size, base_from, base_to, dir_from, dir_to);
    }

    ops_halo_group halos_group_x = ops_decl_halo_group(2,halos_x);
    ops_halo_group halos_group_y = ops_decl_halo_group(2,halos_y);

    ops_partition("");

    // Range
    int full_range[] = {-1,NX+1, -1,NY+1};
    int interior_range[] = {0,NX, 0,NY};

    // Generate obstacles based on grid positions
    ops_par_loop(init_solid, "Generate random obstacles", lb_block, 2, interior_range, 
                ops_arg_dat(d_SOLID, 1, S2D_00, "int", OPS_WRITE),
                ops_arg_idx());

    // Initial values
    ops_par_loop(init_n, "Initial values", lb_block, 2, interior_range,
                ops_arg_dat(d_N, 9, S2D_00, "double", OPS_WRITE),
                ops_arg_gbl(W, 9, "double", OPS_READ));

    //Start timer
    ops_timers(&ct0, &et0);

    // Main time loop
    for (int t = 0; t < 4000; t++) {

        // Backup values
        ops_par_loop(timeloop_eqA, "equation_A", lb_block, 2, full_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ));

        // Gather neighbour values
        ops_halo_transfer(halos_group_x);
        ops_halo_transfer(halos_group_y);

        ops_par_loop(timeloop_eqB, "equation_B", lb_block, 2, interior_range,
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_workArray, 9, S2D_9pt, "double", OPS_READ),
                    ops_arg_gbl(cx, 9, "int", OPS_READ),
                    ops_arg_gbl(cy, 9, "int", OPS_READ));

        // Bounce back from solids, no collision
        ops_par_loop(timeloop_eqC, "equation_C", lb_block, 2, interior_range,
                    ops_arg_dat(d_N_SOLID, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(opposite, 9, "int", OPS_READ));

        ops_par_loop(timeloop_eqD, "equation_D", lb_block, 2, interior_range,
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_INC),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ));
    
        ops_par_loop(timeloop_eqE, "equation_E", lb_block, 2, interior_range,
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(cx, 9, "int", OPS_READ));

        ops_par_loop(timeloop_eqF, "equation_F", lb_block, 2, interior_range,
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(cy, 9, "int", OPS_READ));
   
        ops_par_loop(timeloop_eqG, "equation_G", lb_block, 2, interior_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(cx, 9, "int", OPS_READ),
                    ops_arg_gbl(cy, 9, "int", OPS_READ));

        ops_par_loop(timeloop_eqH, "equation_H", lb_block, 2, interior_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_RW));

        ops_par_loop(timeloop_eqI, "equation_I", lb_block, 2, interior_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_READ));

        ops_par_loop(timeloop_eqJ, "equation_J", lb_block, 2, interior_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(W, 9, "double", OPS_READ));

        ops_par_loop(timeloop_eqK, "equation_K", lb_block, 2, full_range,
                     ops_arg_dat(d_N, 9, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_READ));

        ops_par_loop(timeloop_eqL, "equation_L", lb_block, 2, interior_range,
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_N_SOLID, 9, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_SOLID, 1, S2D_00, "int", OPS_READ));

        // Calculate kinetic energy
        energy = 0.0;
        ops_par_loop(timeloop_eqM, "equation_M", lb_block, 2, interior_range,
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_READ),
                    ops_arg_reduce(h_energy, 1, "double", OPS_INC));
        
        ops_reduction_result(h_energy, &energy);

        if (t%100==0) 
            ops_printf(" %d  %10.5e \n", t, energy);            
        if (t==3999 && NX == 128 && NY == 128) {
          double diff = fabs(((energy - 0.0000111849)/0.0000111849));
          if (diff < 0.00001) {
            ops_printf("Energy : %10.5e diff: %10.5e %s\n", energy, diff, "Test PASSED");
          } else {
            ops_printf("Energy : %10.5e diff:  %10.5e %s\n", energy, diff, "Test FAILED");
          }
        }

    } // End of main time loop

    //End timer
    ops_timers(&ct1, &et1);
    ops_printf("\nTotal Wall time %lf seconds\n",et1-et0);

    /*if (true) {
        std::ofstream myfile;
        myfile.open ("output_velocity.txt");
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                myfile << SOLID[j*NX+i] << " " << ux[j*NX+i] << " " << uy[j*NX+i] << std::endl;
            }
        }
        myfile.close();
    }*/

    // Finalising the OPS library
    ops_exit();
    
}// End of main function

