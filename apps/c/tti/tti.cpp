#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sys/time.h"

#define OPS_3D
#include "ops_seq_v2.h"

float r9;
float r10;

#include "tti_kernels.h"


float dt = 0.001, start = 0, stop = 0.1;  // time variables
int space_order = 8;                     // Space order


int main(int argc, char **argv) {

    int T_intervals;
    int size[3];
    int base[] = {0, 0, 0};
    int d_p[] = {space_order,space_order,space_order};
    int d_m[] = {-space_order, -space_order, -space_order};
    int d_p2[] = {space_order/2,space_order/2,space_order/2};
    int d_m2[] = {-space_order/2, -space_order/2, -space_order/2};
    float *tmp = NULL;
    T_intervals = ceil((stop - start + dt) / dt);
    
    // Read input
    if (argc < 2) {
        printf("Inform grid size\n");
        exit(-1);
    }

    size[0] = atoi(argv[1]);
    size[1] = atoi(argv[2]);
    size[2] = atoi(argv[3]);

    ops_init(argc, argv, 2);

    // Declare global constant
    ops_printf("TII size = %dx%dx%d, %d steps\n", size[0], size[1], size[2], T_intervals);

    // Declare ops_block
    ops_block grid = ops_decl_block(3, "grid");

    // Declare ops_dat objects
    ops_dat u[3];
    u[0] = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "ut0");
    u[1] = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "ut1");
    u[2] = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "ut2");
    ops_dat v[3];
    v[0] = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "vt0");
    v[1] = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "vt1");
    v[2] = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "vt2");

    ops_dat vp = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "vp");
    ops_dat damp = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "damp");
    ops_dat epsilon = ops_decl_dat(grid, 1, size, base, d_m, d_p, tmp, "float", "epsilon");

    ops_dat r2 = ops_decl_dat(grid, 1, size, base, d_m2, d_p2, tmp, "float", "r2");
    ops_dat r3 = ops_decl_dat(grid, 1, size, base, d_m2, d_p2, tmp, "float", "r3");
    ops_dat r4 = ops_decl_dat(grid, 1, size, base, d_m2, d_p2, tmp, "float", "r4");
    ops_dat r5 = ops_decl_dat(grid, 1, size, base, d_m2, d_p2, tmp, "float", "r5");
    ops_dat r11 = ops_decl_dat(grid, 1, size, base, d_m2, d_p2, tmp, "float", "r11");
    ops_dat r12 = ops_decl_dat(grid, 1, size, base, d_m2, d_p2, tmp, "float", "r12");

    int s3d_000[] = {0, 0, 0};
    int s3d_so8[] = {-3, 0, 0, 0, -4, 0,  1, 0, 0,  0, 2, 0,  4, 0, 0,  0, -3, 0, 0, 0,  -1, 0, 0, -2, 0,
                      0,  4, 0, 0, -3, -2, 0, 0, -1, 0, 0, 2,  0, 0, 0,  0, 3,  0, 0, -4, 0,  0, 2, 0,  3,
                      0,  0, 0, 1, -4, 0,  0, 0, 0,  0, 0, -1, 0, 0, -2, 0, 0,  4, 0, 3,  0,  0, 0, 1,  0};
    int s3d_so4[] = {-2,0,0, 2,0,0, 0,-2,0, 0,2,0, 0,0,-2, 0,0,2};
    ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3d_000, "0,0,0");
    ops_stencil S3D_SO8 = ops_decl_stencil(3, 25, s3d_so8, "so8");
    ops_stencil S3D_SO4 = ops_decl_stencil(3, 6, s3d_so4, "so4");

    r9 = 1.0F/(dt*dt);
    r10 = 1.0F/dt;

    ops_decl_const("r9", 1, "float", &r9);
    ops_decl_const("r10", 1, "float", &r10);

    ops_partition("");
  

    double et1, et2, c;
    int range_full[6] = {-space_order, size[0]+space_order, -space_order, size[1]+space_order, -space_order, size[2]+space_order};
    ops_par_loop(init1, "init1", grid, 3, range_full,
                    ops_arg_dat(vp, 1, S3D_000, "float", OPS_WRITE),
                    ops_arg_dat(damp, 1, S3D_000, "float", OPS_WRITE));

    for (int time = 0, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= T_intervals; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3)) {
        if (time == 1) ops_timers(&c, &et1);
        int range_2[6] = {-2, size[0]+2, -2, size[1]+2, -2, size[2]+2};
        ops_par_loop(tti_kernel1, "tti_kernel1", grid, 3, range_2,
                    ops_arg_dat(r11, 1, S3D_000, "float", OPS_WRITE),
                    ops_arg_dat(r12, 1, S3D_000, "float", OPS_WRITE),
                    ops_arg_dat(u[t0], 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(v[t0], 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(r3, 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(r4, 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(r5, 1, S3D_000, "float", OPS_READ));

        int range[6] = {0, size[0], 0, size[1], 0, size[2]};
        ops_par_loop(tti_kernel2, "tti_kernel2", grid, 3, range,
                    ops_arg_dat(r11, 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(r12, 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(u[t0], 1, S3D_SO8, "float", OPS_READ),
                    ops_arg_dat(v[t0], 1, S3D_SO8, "float", OPS_READ),
                    ops_arg_dat(r2, 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(r3, 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(r4, 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(r5, 1, S3D_SO4, "float", OPS_READ),
                    ops_arg_dat(vp, 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(damp, 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(epsilon, 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(u[t1], 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(v[t1], 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(u[t2], 1, S3D_000, "float", OPS_WRITE),
                    ops_arg_dat(v[t2], 1, S3D_000, "float", OPS_WRITE));
    }
    ops_timers(&c, &et2);
    ops_timing_output(std::cout);
    ops_printf("\nTotal Wall time %lf\n",et2-et1);
    ops_printf("GPts/s: %g\n", double(T_intervals)*size[0]*size[1]*size[2]/(et2-et1)/1e9);

    ops_NaNcheck(u[0]);

    ops_exit();
    return 0;
}
