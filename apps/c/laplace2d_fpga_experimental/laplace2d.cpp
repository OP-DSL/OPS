#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int imax, jmax;
float pi  = 2.0 * asin(1.0);

//Including main OPS header file, and setting 2D
#define OPS_2D
#define OPS_CPP_API
// #define OPS_FPGA
#define VERIFICATION
#include <ops_seq_v2.h>
//Including applicaiton-specific "user kernels"
#include "laplace_kernels.h"
#include "laplace2d_cpu_verification.hpp"

int main(int argc, const char** argv)
{
    //Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
    ops_init(argc, argv,1);

    //Size along y
    jmax = 4094;
    //Size along x
    imax = 4094;
    unsigned int iter_max = 100;

//   const float tol = 1.0e-6;
//   float error     = 1.0;

    float *A=NULL;
    float *Anew=NULL;
#ifdef VERIFICATION
    float *Acpu=NULL;
    float *AnewCpu=NULL;
#endif
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
                                d_m, d_p, A,    "float", "A");
    ops_dat d_Anew = ops_decl_dat(block, 1, size, base,
                                d_m, d_p, Anew, "float", "Anew");

#ifdef VERIFICATION
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    Acpu = (float*) malloc(sizeof(float) * grid_size_x * grid_size_y);
    AnewCpu = (float*) malloc(sizeof(float) * grid_size_x * grid_size_y);
#endif

    //Two stencils, a 1-point, and a 5-point
    int s2d_00[] = {0,0};
    ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
    int s2d_5pt[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
    ops_stencil S2D_5pt = ops_decl_stencil(2,5,s2d_5pt,"5pt");

    //Reduction handle
    //   ops_reduction h_err = ops_decl_reduction_handle(sizeof(float), "float", "error");

    //declare and define global constants
    ops_decl_const("imax",1,"int",&imax);
    ops_decl_const("jmax",1,"int",&jmax);
    ops_decl_const("pi",1,"float",&pi);

    ops_partition("");

    // set boundary conditions
    int bottom_range[] = {-1, imax+1, -1, 0};
    ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
        ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE));

    int top_range[] = {-1, imax+1, jmax, jmax+1};
    ops_par_loop(set_zero, "set_zero", block, 2, top_range,
        ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE));

    int left_range[] = {-1, 0, -1, jmax+1};
    ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
        ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE),
        ops_arg_idx());

    int right_range[] = {imax, imax+1, -1, jmax+1};
    ops_par_loop(right_bndcon, "right_bndcon", block, 2, right_range,
        ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE),
        ops_arg_idx());

    ops_printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

    int iter = 0;

    ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
        ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE));

    ops_par_loop(set_zero, "set_zero", block, 2, top_range,
        ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE));

    ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
        ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE),
        ops_arg_idx());

    ops_par_loop(right_bndcon, "right_bndcon", block, 2, right_range,
        ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE),
        ops_arg_idx());

#ifdef VERIFICATION
    A = (float*)ops_dat_get_raw_pointer(d_A, 0, S2D_00, OPS_HOST);
    Anew = (float*)ops_dat_get_raw_pointer(d_Anew, 0, S2D_00, OPS_HOST);

    if(verify(A, Anew, size, d_m, d_p))
        std::cout << "verification of d_A and d_Anew" << "[PASSED]" << std::endl;
    else
        std::cerr << "verification of d_A and d_Anew" << "[FAILED]" << std::endl;

    initilizeGrid(Acpu, size, d_m, d_p, pi, jmax);
    copyGrid(AnewCpu, Acpu, size, d_m, d_p);

    if (verify(Acpu, A, size, d_m, d_p))
        std::cout << "verification of Acpu and A" << "[PASSED]" << std::endl;
    else
        std::cerr << "verification of Acpu and A" << "[FAILED]" << std::endl;
    //  printGrid2D<float>(d_A, "A");
    //  printGrid2D<float>(Acpu, d_A.originalProperty, "Acpu");

    if (verify(AnewCpu, Anew, size, d_m, d_p))
        std::cout << "verification of AnewCpu and Anew" << "[PASSED]" << std::endl;
    else
        std::cerr << "verification of AnewCpu and Anew" << "[FAILED]" << std::endl;
#endif

    int interior_range[] = {0,imax,0,jmax};
    #ifndef OPS_FPGA
    for (unsigned int i = 0; i < iter_max; i++)
    {
        
        ops_par_loop(apply_stencil, "apply_stencil", block, 2, interior_range,
            ops_arg_dat(d_A,    1, S2D_5pt, "float", OPS_READ),
            ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE));

        // ops_dat_deep_copy(d_A, d_Anew);
        ops_par_loop(copy, "copy", block, 2, interior_range,
            ops_arg_dat(d_A,    1, S2D_00, "float", OPS_WRITE),
            ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_READ));

        // if(iter % 10 == 0) ops_printf("%5d, %0.6f\n", iter, error);        
        // iter++;
    }
    #else
        ops_iter_par_loop(iter_max, 
            ops_par_loop(apply_stencil, "apply_stencil", block, 2, interior_range,
                ops_arg_dat(d_A,    1, S2D_5pt, "float", OPS_READ),
                ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE)), 
            ops_par_copy<float>(d_A, d_Anew));
    #endif

    #ifdef VERIFICATION
        A = (float*)d_A->get_raw_pointer(0, S2D_00, OPS_HOST);
        Anew = (float*)d_Anew->get_raw_pointer(0, S2D_00, OPS_HOST);

		for (int iter = 0; iter < iter_max; iter++)
		{
			calcGrid(Acpu, AnewCpu, size, d_m, d_p);
			copyGrid(Acpu, AnewCpu, size, d_m, d_p);
		}

		if (verify(A, Acpu, size, d_m, d_p))
			std::cout << "verification of A and Acpu after calc" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of A and Acpu after calc" << "[FAILED]" << std::endl;

		if (verify(Anew, AnewCpu, size, d_m, d_p))
			std::cout << "verification of Anew and AnewCpu after calc" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of Anew and AnewCpu after calc" << "[FAILED]" << std::endl;

		// printGrid2D<float>(d_A, "d_A");
//		printGrid2D<float>(Acpu, d_A.originalProperty, "d_Acpu");

		// printGrid2D<float>(d_Anew, "d_Anew");
//		printGrid2D<float>(AnewCpu, d_Anew.originalProperty, "d_AnewCpu");

		free(Acpu);
		free(AnewCpu);
    #endif
    //   ops_printf("%5d, %0.6f\n", iter, error);        

    //   ops_timing_output(std::cout);

    //   float err_diff = fabs((100.0*(error/2.421354960840227e-03))-100.0);
    //   printf("Total error is within %3.15E %% of the expected error\n",err_diff);
    //   if(err_diff < 0.001)
    //     printf("This run is considered PASSED\n");
    //   else
    //     printf("This test is considered FAILED\n");

    //Finalising the OPS library
    ops_exit();
    return 0;
}

