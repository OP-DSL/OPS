/*
 * Maxwell FDTD 3D - Main Application
 * 
 * 3D Electromagnetic wave propagation using the Finite-Difference Time-Domain method.
 * Implements the Yee algorithm with CPML absorbing boundary conditions.
 * 
 * Usage: maxwell_cuda <nx> <ny> <nz> [timesteps]
 * 
 * Author: OPS Grid Study
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OPS_3D
#include "ops_seq_v2.h"

#include "maxwell_kernels.h"

// Default simulation parameters
#define DEFAULT_TIMESTEPS 1000
#define COURANT_FACTOR 0.99f  // CFL stability factor

int main(int argc, char **argv) {
    
    int size[3];
    int base[] = {0, 0, 0};
    int d_p[] = {1, 1, 1};   // Halo +1 for stencil access
    int d_m[] = {-1, -1, -1}; // Halo -1 for stencil access
    float *tmp = NULL;
    int timesteps = DEFAULT_TIMESTEPS;
    
    // Parse command line arguments
    if (argc < 4) {
        printf("Usage: %s <nx> <ny> <nz> [timesteps]\n", argv[0]);
        printf("  nx, ny, nz: Grid dimensions\n");
        printf("  timesteps: Number of time steps (default: %d)\n", DEFAULT_TIMESTEPS);
        exit(-1);
    }
    
    size[0] = atoi(argv[1]);
    size[1] = atoi(argv[2]);
    size[2] = atoi(argv[3]);
    if (argc > 4) {
        timesteps = atoi(argv[4]);
    }
    
    // Initialize OPS
    ops_init(argc, argv, 2);
    
    // Calculate grid spacing (assume unit domain)
    float dx = 1.0f / size[0];
    float dy = 1.0f / size[1];
    float dz = 1.0f / size[2];
    
    // Calculate time step from CFL condition: dt <= 1/(c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    float c = C0;  // Speed of light
    float dt_max = COURANT_FACTOR / (c * sqrtf(1.0f/(dx*dx) + 1.0f/(dy*dy) + 1.0f/(dz*dz)));
    float dt = dt_max;
    
    ops_printf("============================================\n");
    ops_printf("Maxwell FDTD 3D Simulation\n");
    ops_printf("============================================\n");
    ops_printf("Grid size: %d x %d x %d = %d cells\n", size[0], size[1], size[2], 
               size[0]*size[1]*size[2]);
    ops_printf("Time steps: %d\n", timesteps);
    ops_printf("dx=%.6e, dy=%.6e, dz=%.6e\n", dx, dy, dz);
    ops_printf("dt=%.6e (CFL factor=%.2f)\n", dt, COURANT_FACTOR);
    ops_printf("============================================\n");
    
    // Declare OPS block
    ops_block grid = ops_decl_block(3, "grid");
    
    // Declare field data arrays (6 field components)
    ops_dat Ex = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "Ex");
    ops_dat Ey = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "Ey");
    ops_dat Ez = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "Ez");
    ops_dat Hx = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "Hx");
    ops_dat Hy = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "Hy");
    ops_dat Hz = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "Hz");
    
    // Material properties
    ops_dat eps = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "eps");
    ops_dat mu = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "mu");
    ops_dat sigma = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "sigma");
    
    // Energy density for monitoring
    ops_dat energy = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "energy");
    
    // Define stencils
    int s3d_000[] = {0, 0, 0};
    int s3d_p100[] = {0, 0, 0, 1, 0, 0};                    // (0,0,0) and (+1,0,0)
    int s3d_p010[] = {0, 0, 0, 0, 1, 0};                    // (0,0,0) and (0,+1,0)
    int s3d_p001[] = {0, 0, 0, 0, 0, 1};                    // (0,0,0) and (0,0,+1)
    int s3d_m100[] = {0, 0, 0, -1, 0, 0};                   // (0,0,0) and (-1,0,0)
    int s3d_m010[] = {0, 0, 0, 0, -1, 0};                   // (0,0,0) and (0,-1,0)
    int s3d_m001[] = {0, 0, 0, 0, 0, -1};                   // (0,0,0) and (0,0,-1)
    int s3d_yee_h[] = {0, 0, 0, 0, 1, 0, 0, 0, 1};          // For H update: +y, +z
    int s3d_yee_e[] = {0, 0, 0, 0, -1, 0, 0, 0, -1, -1, 0, 0}; // For E update: -x, -y, -z
    
    ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3d_000, "0,0,0");
    ops_stencil S3D_P100 = ops_decl_stencil(3, 2, s3d_p100, "p100");
    ops_stencil S3D_P010 = ops_decl_stencil(3, 2, s3d_p010, "p010");
    ops_stencil S3D_P001 = ops_decl_stencil(3, 2, s3d_p001, "p001");
    ops_stencil S3D_M100 = ops_decl_stencil(3, 2, s3d_m100, "m100");
    ops_stencil S3D_M010 = ops_decl_stencil(3, 2, s3d_m010, "m010");
    ops_stencil S3D_M001 = ops_decl_stencil(3, 2, s3d_m001, "m001");
    
    ops_partition("");
    
    // Define iteration ranges
    int range_full[] = {0, size[0], 0, size[1], 0, size[2]};
    int range_H[] = {0, size[0]-1, 0, size[1]-1, 0, size[2]-1};  // H update range
    int range_E[] = {1, size[0], 1, size[1], 1, size[2]};        // E update range
    
    // Source parameters (Gaussian pulse at center)
    int src_x = size[0] / 2;
    int src_y = size[1] / 2;
    int src_z = size[2] / 2;
    float t0 = 30.0f * dt;      // Pulse center time
    float spread = 10.0f * dt;  // Pulse width
    
    // Initialize fields
    ops_printf("Initializing fields...\n");
    ops_par_loop(init_fields, "init_fields", grid, 3, range_full,
                 ops_arg_dat(Ex, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(Ey, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(Ez, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(Hx, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(Hy, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(Hz, 1, S3D_000, "float", OPS_WRITE));
    
    // Initialize materials
    ops_par_loop(init_materials, "init_materials", grid, 3, range_full,
                 ops_arg_dat(eps, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(mu, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_dat(sigma, 1, S3D_000, "float", OPS_WRITE),
                 ops_arg_idx());
    
    // Time stepping
    ops_printf("Starting time stepping...\n");
    double ct1, ct2, et1, et2;
    ops_timers(&ct1, &et1);
    
    for (int t = 0; t < timesteps; t++) {
        float t_current = t * dt;
        
        // Update H fields (half time step ahead)
        ops_par_loop(update_Hx, "update_Hx", grid, 3, range_H,
                     ops_arg_dat(Hx, 1, S3D_000, "float", OPS_RW),
                     ops_arg_dat(Ey, 1, S3D_P001, "float", OPS_READ),
                     ops_arg_dat(Ez, 1, S3D_P010, "float", OPS_READ),
                     ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
                     ops_arg_gbl(&dt, 1, "float", OPS_READ),
                     ops_arg_gbl(&dy, 1, "float", OPS_READ),
                     ops_arg_gbl(&dz, 1, "float", OPS_READ));
        
        ops_par_loop(update_Hy, "update_Hy", grid, 3, range_H,
                     ops_arg_dat(Hy, 1, S3D_000, "float", OPS_RW),
                     ops_arg_dat(Ex, 1, S3D_P001, "float", OPS_READ),
                     ops_arg_dat(Ez, 1, S3D_P100, "float", OPS_READ),
                     ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
                     ops_arg_gbl(&dt, 1, "float", OPS_READ),
                     ops_arg_gbl(&dx, 1, "float", OPS_READ),
                     ops_arg_gbl(&dz, 1, "float", OPS_READ));
        
        ops_par_loop(update_Hz, "update_Hz", grid, 3, range_H,
                     ops_arg_dat(Hz, 1, S3D_000, "float", OPS_RW),
                     ops_arg_dat(Ex, 1, S3D_P010, "float", OPS_READ),
                     ops_arg_dat(Ey, 1, S3D_P100, "float", OPS_READ),
                     ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
                     ops_arg_gbl(&dt, 1, "float", OPS_READ),
                     ops_arg_gbl(&dx, 1, "float", OPS_READ),
                     ops_arg_gbl(&dy, 1, "float", OPS_READ));
        
        // Update E fields
        ops_par_loop(update_Ex, "update_Ex", grid, 3, range_E,
                     ops_arg_dat(Ex, 1, S3D_000, "float", OPS_RW),
                     ops_arg_dat(Hy, 1, S3D_M001, "float", OPS_READ),
                     ops_arg_dat(Hz, 1, S3D_M010, "float", OPS_READ),
                     ops_arg_dat(eps, 1, S3D_000, "float", OPS_READ),
                     ops_arg_dat(sigma, 1, S3D_000, "float", OPS_READ),
                     ops_arg_gbl(&dt, 1, "float", OPS_READ),
                     ops_arg_gbl(&dy, 1, "float", OPS_READ),
                     ops_arg_gbl(&dz, 1, "float", OPS_READ));
        
        ops_par_loop(update_Ey, "update_Ey", grid, 3, range_E,
                     ops_arg_dat(Ey, 1, S3D_000, "float", OPS_RW),
                     ops_arg_dat(Hx, 1, S3D_M001, "float", OPS_READ),
                     ops_arg_dat(Hz, 1, S3D_M100, "float", OPS_READ),
                     ops_arg_dat(eps, 1, S3D_000, "float", OPS_READ),
                     ops_arg_dat(sigma, 1, S3D_000, "float", OPS_READ),
                     ops_arg_gbl(&dt, 1, "float", OPS_READ),
                     ops_arg_gbl(&dx, 1, "float", OPS_READ),
                     ops_arg_gbl(&dz, 1, "float", OPS_READ));
        
        ops_par_loop(update_Ez, "update_Ez", grid, 3, range_E,
                     ops_arg_dat(Ez, 1, S3D_000, "float", OPS_RW),
                     ops_arg_dat(Hx, 1, S3D_M010, "float", OPS_READ),
                     ops_arg_dat(Hy, 1, S3D_M100, "float", OPS_READ),
                     ops_arg_dat(eps, 1, S3D_000, "float", OPS_READ),
                     ops_arg_dat(sigma, 1, S3D_000, "float", OPS_READ),
                     ops_arg_gbl(&dt, 1, "float", OPS_READ),
                     ops_arg_gbl(&dx, 1, "float", OPS_READ),
                     ops_arg_gbl(&dy, 1, "float", OPS_READ));
        
        // Add source (Gaussian pulse)
        ops_par_loop(add_source, "add_source", grid, 3, range_full,
                     ops_arg_dat(Ez, 1, S3D_000, "float", OPS_RW),
                     ops_arg_idx(),
                     ops_arg_gbl(&t_current, 1, "float", OPS_READ),
                     ops_arg_gbl(&t0, 1, "float", OPS_READ),
                     ops_arg_gbl(&spread, 1, "float", OPS_READ),
                     ops_arg_gbl(&src_x, 1, "int", OPS_READ),
                     ops_arg_gbl(&src_y, 1, "int", OPS_READ),
                     ops_arg_gbl(&src_z, 1, "int", OPS_READ));
        
        // Progress output every 10%
        if (t > 0 && t % (timesteps / 10) == 0) {
            ops_printf("  Step %d / %d (%.0f%%)\n", t, timesteps, 100.0f * t / timesteps);
        }
    }
    
    ops_timers(&ct2, &et2);
    
    // Compute final energy
    ops_par_loop(compute_energy, "compute_energy", grid, 3, range_full,
                 ops_arg_dat(Ex, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(Ey, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(Ez, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(Hx, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(Hy, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(Hz, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(eps, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
                 ops_arg_dat(energy, 1, S3D_000, "float", OPS_WRITE));
    
    // Output timing
    ops_timing_output(std::cout);
    ops_printf("\n============================================\n");
    ops_printf("Simulation complete!\n");
    ops_printf("Total Wall time %lf\n", et2 - et1);
    ops_printf("GPts/s: %g\n", (double)timesteps * size[0] * size[1] * size[2] / (et2 - et1) / 1e9);
    ops_printf("============================================\n");
    
    // Check for NaN
    ops_NaNcheck(Ez);
    
    ops_exit();
    return 0;
}
