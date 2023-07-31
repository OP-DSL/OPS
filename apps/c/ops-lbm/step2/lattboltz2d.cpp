/* STEP 2 - Handling loops with globals
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

    int* SOLID = new int[NX*NY];
    double* N = new double[NX*NY*9];

    // Work arrays
    double* workArray = new double[NX*NY*9];
    double* N_SOLID = new double[NX*NY*9];
    double* rho = new double[NX*NY];
    double* ux = new double[NX*NY];
    double* uy = new double[NX*NY];

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
    ops_dat d_SOLID = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, SOLID, "int", "SOLID");
    ops_dat d_rho   = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, rho, "double", "rho"); 
    ops_dat d_ux    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, ux, "double", "ux");
    ops_dat d_uy    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, uy, "double", "uy");

    // Multi dim dats
    ops_dat d_N         = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, N, "double", "N");
    ops_dat d_N_SOLID   = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, N_SOLID, "double", "N_SOLID");
    ops_dat d_workArray = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, workArray, "double", "workArray");

    // Declare stencils
    int s2d_00[] = {0,0};
    ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");

    // Declare and define global constants
    ops_decl_const("OMEGA",1,"double",&OMEGA);
    ops_decl_const("rho0",1,"double",&rho0);
    ops_decl_const("deltaUX",1,"double",&deltaUX);

    // Range
    int full_range[] = {0,NX, 0,NY};

    // Generate obstacles based on grid positions
    ops_par_loop(init_solid, "Generate random obstacles", lb_block, 2, full_range, 
                ops_arg_dat(d_SOLID, 1, S2D_00, "int", OPS_WRITE),
                ops_arg_idx());

    // Initial values
    ops_par_loop(init_n, "Initial values", lb_block, 2, full_range,
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
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 1; f < 9; f++) {
                    N[(j*NX+i)*9 + f] = workArray[(mod(j-cy[f],NY)*NX+mod(i-cx[f],NX))*9 + f];
                }
            }
        }

        // Bounce back from solids, no collision
        ops_par_loop(timeloop_eqC, "equation_C", lb_block, 2, full_range,
                    ops_arg_dat(d_N_SOLID, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(opposite, 9, "int", OPS_READ));

        ops_par_loop(timeloop_eqD, "equation_D", lb_block, 2, full_range,
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_INC),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ));
    
        ops_par_loop(timeloop_eqE, "equation_E", lb_block, 2, full_range,
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(cx, 9, "int", OPS_READ));

        ops_par_loop(timeloop_eqF, "equation_F", lb_block, 2, full_range,
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(cy, 9, "int", OPS_READ));
   
        ops_par_loop(timeloop_eqG, "equation_G", lb_block, 2, full_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(cx, 9, "int", OPS_READ),
                    ops_arg_gbl(cy, 9, "int", OPS_READ));

        ops_par_loop(timeloop_eqH, "equation_H", lb_block, 2, full_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_RW));

        ops_par_loop(timeloop_eqI, "equation_I", lb_block, 2, full_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_ux, 1, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_uy, 1, S2D_00, "double", OPS_READ));

        ops_par_loop(timeloop_eqJ, "equation_J", lb_block, 2, full_range,
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_rho, 1, S2D_00, "double", OPS_READ),
                    ops_arg_gbl(W, 9, "double", OPS_READ));

        ops_par_loop(timeloop_eqK, "equation_K", lb_block, 2, full_range,
                     ops_arg_dat(d_N, 9, S2D_00, "double", OPS_RW),
                    ops_arg_dat(d_workArray, 9, S2D_00, "double", OPS_READ));

        ops_par_loop(timeloop_eqL, "equation_L", lb_block, 2, full_range,
                    ops_arg_dat(d_N, 9, S2D_00, "double", OPS_WRITE),
                    ops_arg_dat(d_N_SOLID, 9, S2D_00, "double", OPS_READ),
                    ops_arg_dat(d_SOLID, 1, S2D_00, "int", OPS_READ));

        // Calculate kinetic energy
        energy = 0.0;
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                energy += ux[j*NX+i]*ux[j*NX+i]+uy[j*NX+i]*uy[j*NX+i];// reduction 
            }
        }

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

    if (true) {
        std::ofstream myfile;
        myfile.open ("output_velocity.txt");
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                myfile << SOLID[j*NX+i] << " " << ux[j*NX+i] << " " << uy[j*NX+i] << std::endl;
            }
        }
        myfile.close();
    }

    // Finalising the OPS library
    ops_exit();
    delete[] SOLID;      SOLID = nullptr;
    delete[] N;          N = nullptr;
    delete[] workArray;  workArray = nullptr;
    delete[] N_SOLID;    N_SOLID = nullptr;
    delete[] rho;        rho = nullptr;
    delete[] ux;         ux = nullptr;
    delete[] uy;         uy = nullptr;
    
}// End of main function

