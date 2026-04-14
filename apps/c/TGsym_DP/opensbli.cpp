#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#ifdef PROFILE_ITT
#include <ittnotify.h>
#endif
#include "constants.h"
#define OPS_3D
#define OPS_API 2
#include "ops_seq.h"
#include "opensbliblock00_kernels.h"
#include "io.h"
#ifndef mult
#define mult 1
#endif
int main(int argc, char **argv) 
{
// Initializing OPS 
ops_init(argc,argv,2);
// Set restart to 1 to restart the simulation from HDF5 file
restart = 0;
// User defined constant values
block0np0 = mult*320+1;
block0np1 = 321;
block0np2 = 321;
Delta0block0 = mult*M_PI/(block0np0-1);
Delta1block0 = M_PI/(block0np1-1);
Delta2block0 = M_PI/(block0np2-1);
niter = 50;
double rkold[] = {(1.0/4.0), (3.0/20.0), (3.0/5.0)};
double rknew[] = {(2.0/3.0), (5.0/12.0), (3.0/5.0)};
dt = 0.00125;
write_output_file = 400;
HDF5_timing = 0;
gama = 1.4;
Minf = 0.5;
Pr = 0.71;
Re = 800.0;
inv2Delta0block0 = 1.0/(Delta0block0*Delta0block0);
inv2Delta1block0 = 1.0/(Delta1block0*Delta1block0);
inv2Delta2block0 = 1.0/(Delta2block0*Delta2block0);
inv2Minf = 1.0/(Minf*Minf);
invDelta0block0 = 1.0/(Delta0block0);
invDelta1block0 = 1.0/(Delta1block0);
invDelta2block0 = 1.0/(Delta2block0);
invPr = 1.0/(Pr);
invRe = 1.0/(Re);
inv_gamma_m1 = 1.0/((-1 + gama));
ops_decl_const("Delta0block0" , 1, "double", &Delta0block0);
ops_decl_const("Delta1block0" , 1, "double", &Delta1block0);
ops_decl_const("Delta2block0" , 1, "double", &Delta2block0);
ops_decl_const("HDF5_timing" , 1, "int", &HDF5_timing);
ops_decl_const("Minf" , 1, "double", &Minf);
ops_decl_const("Pr" , 1, "double", &Pr);
ops_decl_const("Re" , 1, "double", &Re);
ops_decl_const("block0np0" , 1, "int", &block0np0);
ops_decl_const("block0np1" , 1, "int", &block0np1);
ops_decl_const("block0np2" , 1, "int", &block0np2);
ops_decl_const("dt" , 1, "double", &dt);
ops_decl_const("gama" , 1, "double", &gama);
ops_decl_const("inv2Delta0block0" , 1, "double", &inv2Delta0block0);
ops_decl_const("inv2Delta1block0" , 1, "double", &inv2Delta1block0);
ops_decl_const("inv2Delta2block0" , 1, "double", &inv2Delta2block0);
ops_decl_const("inv2Minf" , 1, "double", &inv2Minf);
ops_decl_const("invDelta0block0" , 1, "double", &invDelta0block0);
ops_decl_const("invDelta1block0" , 1, "double", &invDelta1block0);
ops_decl_const("invDelta2block0" , 1, "double", &invDelta2block0);
ops_decl_const("invPr" , 1, "double", &invPr);
ops_decl_const("invRe" , 1, "double", &invRe);
ops_decl_const("inv_gamma_m1" , 1, "double", &inv_gamma_m1);
ops_decl_const("niter" , 1, "int", &niter);
ops_decl_const("simulation_time" , 1, "double", &simulation_time);
ops_decl_const("start_iter" , 1, "int", &start_iter);
ops_decl_const("write_output_file" , 1, "int", &write_output_file);
// Define and Declare OPS Block
ops_block opensbliblock00 = ops_decl_block(3, "opensbliblock00");
#include "defdec_data_set.h"
// Define and declare stencils
#include "stencils.h"
// Init OPS partition
double partition_start0, elapsed_partition_start0, partition_end0, elapsed_partition_end0;
ops_timers(&partition_start0, &elapsed_partition_start0);
ops_partition("");
ops_timers(&partition_end0, &elapsed_partition_end0);
ops_printf("-----------------------------------------\n MPI partition and reading input file time: %lf\n -----------------------------------------\n", elapsed_partition_end0-elapsed_partition_start0);
fflush(stdout);

// Restart procedure
ops_printf("\033[1;32m");
if (restart == 1){
ops_printf("OpenSBLI is restarting from the input file: restart.h5\n");
}
else {
ops_printf("OpenSBLI is starting from the initial condition.\n");
}
ops_printf("\033[0m");
// Constants from HDF5 restart file
if (restart == 1){
ops_get_const_hdf5("simulation_time", 1, "double", (char*)&simulation_time, "restart.h5");
ops_get_const_hdf5("iter", 1, "int", (char*)&start_iter, "restart.h5");
}
else {
simulation_time = 0.0;
start_iter = 0;
}
tstart = simulation_time;

if (restart == 0){
int iteration_range_87_block0[] = {-5, block0np0 + 5, -5, block0np1 + 5, -5, block0np2 + 5};
ops_par_loop(opensbliblock00Kernel087, "Grid_based_initialisation0", opensbliblock00, 3, iteration_range_87_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());
}

  opensbliblock00->instance->reset_power_counters();
#ifdef PROFILE_ITT
  __itt_resume();
#endif

// Initialize loop timers
double cpu_start0, elapsed_start0, cpu_end0, elapsed_end0;
ops_timers(&cpu_start0, &elapsed_start0);
for(iter=start_iter; iter<=start_iter+niter - 1; iter++)
{
simulation_time = tstart + dt*((iter - start_iter)+1);

int iteration_range_81_block0[] = {0, 1, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel081, "Symmetry boundary dir0 side0", opensbliblock00, 3, iteration_range_81_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW));

int iteration_range_82_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel082, "Symmetry boundary dir0 side1", opensbliblock00, 3, iteration_range_82_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW));

int iteration_range_83_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel083, "Symmetry boundary dir1 side0", opensbliblock00, 3, iteration_range_83_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW));

int iteration_range_84_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel084, "Symmetry boundary dir1 side1", opensbliblock00, 3, iteration_range_84_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW));

int iteration_range_85_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, 0, 1};
ops_par_loop(opensbliblock00Kernel085, "Symmetry boundary dir2 side0", opensbliblock00, 3, iteration_range_85_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW));

int iteration_range_86_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, block0np2 - 1, block0np2};
ops_par_loop(opensbliblock00Kernel086, "Symmetry boundary dir2 side1", opensbliblock00, 3, iteration_range_86_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW));

int iteration_range_88_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel088, "Save equations", opensbliblock00, 3, iteration_range_88_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

for(stage=0; stage<=2; stage++)
{
int iteration_range_2_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel002, "CRu0_B0", opensbliblock00, 3, iteration_range_2_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_24_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel024, "CRu1_B0", opensbliblock00, 3, iteration_range_24_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_37_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel037, "CRu2_B0", opensbliblock00, 3, iteration_range_37_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_8_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel008, "CRp_B0", opensbliblock00, 3, iteration_range_8_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_79_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel079, "CRT_B0", opensbliblock00, 3, iteration_range_79_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(T_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_0_block0[] = {-2, block0np0 + 2, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel000, "Convective terms group 0", opensbliblock00, 3, iteration_range_0_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk39_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk40_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk41_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk42_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk43_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk44_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_3_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel003, "Convective CD rhou1_B0*u0_B0 x0 ", opensbliblock00, 3, iteration_range_3_block0,
ops_arg_dat(wk39_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_5_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel005, "Convective CD rhou0_B0 x0 ", opensbliblock00, 3, iteration_range_5_block0,
ops_arg_dat(rhou0_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_7_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel007, "Convective CD rhou2_B0*u0_B0 x0 ", opensbliblock00, 3, iteration_range_7_block0,
ops_arg_dat(wk40_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_9_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel009, "Convective CD p_B0 x0 ", opensbliblock00, 3, iteration_range_9_block0,
ops_arg_dat(p_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_11_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel011, "Convective CD rhoE_B0*u0_B0 x0 ", opensbliblock00, 3, iteration_range_11_block0,
ops_arg_dat(wk41_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_12_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel012, "Convective CD rhoE_B0 x0 ", opensbliblock00, 3, iteration_range_12_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_13_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel013, "Convective CD rhou1_B0 x0 ", opensbliblock00, 3, iteration_range_13_block0,
ops_arg_dat(rhou1_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_14_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel014, "Convective CD p_B0*u0_B0 x0 ", opensbliblock00, 3, iteration_range_14_block0,
ops_arg_dat(wk42_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_16_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel016, "Convective CD rho_B0*u0_B0 x0 ", opensbliblock00, 3, iteration_range_16_block0,
ops_arg_dat(wk43_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_17_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel017, "Convective CD u0_B0 x0 ", opensbliblock00, 3, iteration_range_17_block0,
ops_arg_dat(u0_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_18_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel018, "Convective CD rhou2_B0 x0 ", opensbliblock00, 3, iteration_range_18_block0,
ops_arg_dat(rhou2_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_19_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel019, "Convective CD rhou0_B0*u0_B0 x0 ", opensbliblock00, 3, iteration_range_19_block0,
ops_arg_dat(wk44_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk11_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_20_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel020, "Convective CD rho_B0 x0 ", opensbliblock00, 3, iteration_range_20_block0,
ops_arg_dat(rho_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk12_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_21_block0[] = {0, block0np0, -2, block0np1 + 2, 0, block0np2};
ops_par_loop(opensbliblock00Kernel021, "Convective terms group 1", opensbliblock00, 3, iteration_range_21_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk39_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk40_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk41_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk42_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk43_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk44_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_22_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel022, "Convective CD rhou1_B0 x1 ", opensbliblock00, 3, iteration_range_22_block0,
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk13_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_23_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel023, "Convective CD rho_B0 x1 ", opensbliblock00, 3, iteration_range_23_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk14_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_25_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel025, "Convective CD rhou1_B0*u1_B0 x1 ", opensbliblock00, 3, iteration_range_25_block0,
ops_arg_dat(wk39_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk15_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_26_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel026, "Convective CD rho_B0*u1_B0 x1 ", opensbliblock00, 3, iteration_range_26_block0,
ops_arg_dat(wk40_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk16_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_27_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel027, "Convective CD rhou2_B0 x1 ", opensbliblock00, 3, iteration_range_27_block0,
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk17_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_28_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel028, "Convective CD p_B0 x1 ", opensbliblock00, 3, iteration_range_28_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk18_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_29_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel029, "Convective CD rhou0_B0*u1_B0 x1 ", opensbliblock00, 3, iteration_range_29_block0,
ops_arg_dat(wk41_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk19_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_30_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel030, "Convective CD rhoE_B0 x1 ", opensbliblock00, 3, iteration_range_30_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk20_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_31_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel031, "Convective CD rhou0_B0 x1 ", opensbliblock00, 3, iteration_range_31_block0,
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk21_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_32_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel032, "Convective CD p_B0*u1_B0 x1 ", opensbliblock00, 3, iteration_range_32_block0,
ops_arg_dat(wk42_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk22_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_33_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel033, "Convective CD u1_B0 x1 ", opensbliblock00, 3, iteration_range_33_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk23_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_34_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel034, "Convective CD rhoE_B0*u1_B0 x1 ", opensbliblock00, 3, iteration_range_34_block0,
ops_arg_dat(wk43_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk24_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_35_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel035, "Convective CD rhou2_B0*u1_B0 x1 ", opensbliblock00, 3, iteration_range_35_block0,
ops_arg_dat(wk44_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk25_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_36_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel036, "Convective terms group 2", opensbliblock00, 3, iteration_range_36_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk39_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk40_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk41_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk42_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk43_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk44_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_38_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel038, "Convective CD rhou2_B0*u2_B0 x2 ", opensbliblock00, 3, iteration_range_38_block0,
ops_arg_dat(wk39_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk26_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_39_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel039, "Convective CD rhoE_B0 x2 ", opensbliblock00, 3, iteration_range_39_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk27_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_40_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel040, "Convective CD rhoE_B0*u2_B0 x2 ", opensbliblock00, 3, iteration_range_40_block0,
ops_arg_dat(wk40_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk28_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_41_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel041, "Convective CD rhou0_B0 x2 ", opensbliblock00, 3, iteration_range_41_block0,
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk29_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_42_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel042, "Convective CD rho_B0*u2_B0 x2 ", opensbliblock00, 3, iteration_range_42_block0,
ops_arg_dat(wk41_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk30_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_43_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel043, "Convective CD rhou0_B0*u2_B0 x2 ", opensbliblock00, 3, iteration_range_43_block0,
ops_arg_dat(wk42_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk31_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_44_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel044, "Convective CD rhou1_B0 x2 ", opensbliblock00, 3, iteration_range_44_block0,
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk32_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_45_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel045, "Convective CD p_B0 x2 ", opensbliblock00, 3, iteration_range_45_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk33_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_46_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel046, "Convective CD rhou2_B0 x2 ", opensbliblock00, 3, iteration_range_46_block0,
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk34_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_47_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel047, "Convective CD p_B0*u2_B0 x2 ", opensbliblock00, 3, iteration_range_47_block0,
ops_arg_dat(wk43_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk35_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_48_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel048, "Convective CD rhou1_B0*u2_B0 x2 ", opensbliblock00, 3, iteration_range_48_block0,
ops_arg_dat(wk44_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk36_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_49_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel049, "Convective CD rho_B0 x2 ", opensbliblock00, 3, iteration_range_49_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk37_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_50_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel050, "Convective CD u2_B0 x2 ", opensbliblock00, 3, iteration_range_50_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk38_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_51_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel051, "Convective residual ", opensbliblock00, 3, iteration_range_51_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk12_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk13_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk14_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk15_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk16_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk17_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk18_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk19_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk20_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk21_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk22_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk23_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk24_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk25_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk26_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk27_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk28_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk29_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk30_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk31_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk32_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk33_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk34_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk35_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk36_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk37_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk38_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_53_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel053, "Viscous CD u0_B0 x0 x0 ", opensbliblock00, 3, iteration_range_53_block0,
ops_arg_dat(u0_B0, 1, stencil_0_22_00_00_11, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_55_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel055, "Viscous CD T_B0 x0 x0 ", opensbliblock00, 3, iteration_range_55_block0,
ops_arg_dat(T_B0, 1, stencil_0_22_00_00_11, "double", OPS_READ),
ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_57_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel057, "Viscous CD u2_B0 x1 ", opensbliblock00, 3, iteration_range_57_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_58_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel058, "Viscous CD u0_B0 x1 ", opensbliblock00, 3, iteration_range_58_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_59_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel059, "Viscous CD u2_B0 x1 x1 ", opensbliblock00, 3, iteration_range_59_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_22_00_11, "double", OPS_READ),
ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_60_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel060, "Viscous CD u1_B0 x1 x1 ", opensbliblock00, 3, iteration_range_60_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_22_00_11, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_61_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel061, "Viscous CD u1_B0 x2 x2 ", opensbliblock00, 3, iteration_range_61_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_63_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel063, "Viscous CD u1_B0 x2 ", opensbliblock00, 3, iteration_range_63_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk11_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_64_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel064, "Viscous CD u1_B0 x0 x0 ", opensbliblock00, 3, iteration_range_64_block0,
ops_arg_dat(u1_B0, 1, stencil_0_22_00_00_11, "double", OPS_READ),
ops_arg_dat(wk12_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_65_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel065, "Viscous CD u2_B0 x2 x2 ", opensbliblock00, 3, iteration_range_65_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk13_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_66_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel066, "Viscous CD u2_B0 x0 x0 ", opensbliblock00, 3, iteration_range_66_block0,
ops_arg_dat(u2_B0, 1, stencil_0_22_00_00_11, "double", OPS_READ),
ops_arg_dat(wk14_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_67_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel067, "Viscous CD u0_B0 x2 x2 ", opensbliblock00, 3, iteration_range_67_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk15_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_68_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel068, "Viscous CD u0_B0 x1 x1 ", opensbliblock00, 3, iteration_range_68_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_22_00_11, "double", OPS_READ),
ops_arg_dat(wk16_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_69_block0[] = {0, block0np0, -2, block0np1 + 2, 0, block0np2};
ops_par_loop(opensbliblock00Kernel069, "Viscous CD u1_B0 x0 ", opensbliblock00, 3, iteration_range_69_block0,
ops_arg_dat(u1_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk17_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_70_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel070, "Viscous CD u1_B0 x1 ", opensbliblock00, 3, iteration_range_70_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk18_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_73_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel073, "Viscous CD T_B0 x1 x1 ", opensbliblock00, 3, iteration_range_73_block0,
ops_arg_dat(T_B0, 1, stencil_0_00_22_00_11, "double", OPS_READ),
ops_arg_dat(wk21_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_74_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel074, "Viscous CD u2_B0 x0 ", opensbliblock00, 3, iteration_range_74_block0,
ops_arg_dat(u2_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk22_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_75_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel075, "Viscous CD u0_B0 x2 ", opensbliblock00, 3, iteration_range_75_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk23_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_76_block0[] = {0, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel076, "Viscous CD u0_B0 x0 ", opensbliblock00, 3, iteration_range_76_block0,
ops_arg_dat(u0_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
ops_arg_dat(wk24_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_77_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel077, "Viscous CD u2_B0 x2 ", opensbliblock00, 3, iteration_range_77_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk25_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_78_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel078, "Viscous CD T_B0 x2 x2 ", opensbliblock00, 3, iteration_range_78_block0,
ops_arg_dat(T_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk26_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_52_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel052, "Viscous CD CD u0_B0 x0 x2 ", opensbliblock00, 3, iteration_range_52_block0,
ops_arg_dat(wk24_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_54_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel054, "Viscous CD CD u0_B0 x0 x1 ", opensbliblock00, 3, iteration_range_54_block0,
ops_arg_dat(wk24_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_56_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel056, "Viscous CD CD u2_B0 x1 x2 ", opensbliblock00, 3, iteration_range_56_block0,
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_62_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel062, "Viscous CD CD u2_B0 x0 x2 ", opensbliblock00, 3, iteration_range_62_block0,
ops_arg_dat(wk22_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_71_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel071, "Viscous CD CD u1_B0 x0 x1 ", opensbliblock00, 3, iteration_range_71_block0,
ops_arg_dat(wk17_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
ops_arg_dat(wk19_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_72_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel072, "Viscous CD CD u1_B0 x1 x2 ", opensbliblock00, 3, iteration_range_72_block0,
ops_arg_dat(wk18_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk20_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_80_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel080, "Viscous residual", opensbliblock00, 3, iteration_range_80_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk12_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk13_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk14_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk15_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk16_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk17_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk18_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk19_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk20_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk21_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk22_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk23_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk24_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk25_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk26_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

int iteration_range_90_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel090, "Sub stage advancement", opensbliblock00, 3, iteration_range_90_block0,
ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_gbl(&rknew[stage], 1, "double", OPS_READ));

int iteration_range_89_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel089, "Temporal solution advancement", opensbliblock00, 3, iteration_range_89_block0,
ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_gbl(&rkold[stage], 1, "double", OPS_READ));

int iteration_range_81_block0[] = {0, 1, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel081, "Symmetry boundary dir0 side0", opensbliblock00, 3, iteration_range_81_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW));

int iteration_range_82_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel082, "Symmetry boundary dir0 side1", opensbliblock00, 3, iteration_range_82_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW));

int iteration_range_83_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel083, "Symmetry boundary dir1 side0", opensbliblock00, 3, iteration_range_83_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW));

int iteration_range_84_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel084, "Symmetry boundary dir1 side1", opensbliblock00, 3, iteration_range_84_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW));

int iteration_range_85_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, 0, 1};
ops_par_loop(opensbliblock00Kernel085, "Symmetry boundary dir2 side0", opensbliblock00, 3, iteration_range_85_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW));

int iteration_range_86_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, block0np2 - 1, block0np2};
ops_par_loop(opensbliblock00Kernel086, "Symmetry boundary dir2 side1", opensbliblock00, 3, iteration_range_86_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_22_8, "double", OPS_RW));

}
ops_execute(opensbliblock00->instance);
}
ops_timers(&cpu_end0, &elapsed_end0);
ops_timing_output(std::cout);
ops_printf("\nTimings are:\n");
ops_printf("-----------------------------------------\n");
ops_printf("Total Wall time %lf\n",elapsed_end0-elapsed_start0);
#ifdef PROFILE_ITT
  __itt_pause();
#endif
ops_exit();
//Main program end 
}
