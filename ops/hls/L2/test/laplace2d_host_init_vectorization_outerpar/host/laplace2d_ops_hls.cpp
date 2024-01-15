
// Auto-generated at 2023-10-02 14:24:24.741093 by ops-translator

//extern void ops_init_backend(int argc, const char** argv);

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define OPS_HLS_V2
//#define DEBUG_LOG

#include <ops_hls_rt_support.h>

#define VERIFICATION
//#define PROFILE

int imax, jmax;
//#pragma acc declare create(imax)
//#pragma acc declare create(jmax)
float pi  = 2.0 * asin(1.0);
//#pragma acc declare create(pi)

//Including main OPS header file, and setting 2D
#define OPS_2D
#include <laplace2d_cpu_verification.hpp>


#ifdef PROFILE
	#include <chrono>
#endif

//Including applicaiton-specific "user kernels"
/* ops_par_loop declarations */

void ops_par_loop_set_zero(int, int*, ops::hls::Grid<float>&);

void ops_par_loop_left_bndcon(int, int*, ops::hls::Grid<float>&);

void ops_par_loop_right_bndcon(int, int*, ops::hls::Grid<float>&);

// void ops_par_loop_apply_stencil(int, int*, ops::hls::Grid<float>&, ops::hls::Grid<float>&);

// void ops_par_loop_copy(int, int*, ops::hls::Grid<float>&, ops::hls::Grid<float>&);
void ops__itr_par_loop_outerloop_1(int, int*, ops::hls::Grid<float>&, ops::hls::Grid<float>&, int, bool = false, int* = nullptr);


#include "hls_kernels.hpp"

int main(int argc, const char** argv)
{
	//Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
	ops_init_backend(argc, argv);
	unsigned int batches = 1;

#ifdef PROFILE
	double init_runtime[batches];
	double main_loop_runtime[batches];
#endif

	for (unsigned int bat = 0; bat < batches; bat++)
	{
		//Size along y
		jmax = 5;
		//Size along x
		imax = 5;
		int iter_max = 60000;

		const float tol = 1.0e-6;
		float error     = 1.0;

		float *A=nullptr;
		float *Anew=nullptr;
#ifdef VERIFICATION
		float *Acpu= nullptr;
		float *AnewCpu = nullptr;
#endif
		//
		//Declare & define key data structures
		//

		// //The 2D block
		ops::hls::Block block = ops_hls_decl_block(2, "my_grid");
		//The two datasets
		int size[] = {imax, jmax};
		int base[] = {0,0};
		int d_m[] = {-1,-1};
		int d_p[] = {1,1};
		auto d_A = ops_hls_decl_dat(block, 1, size, base, d_m, d_p, A, "float", "A");
		auto d_Anew = ops_hls_decl_dat(block, 1, size, base, d_m, d_p, Anew, "float", "Anew");

#ifdef VERIFICATION
		// Naive CPU grid initialization
		A = d_A.hostBuffer.data();
		Anew = d_Anew.hostBuffer.data();
		Acpu = (float*) malloc(sizeof(float) * d_A.originalProperty.grid_size[0] * d_A.originalProperty.grid_size[1]);
		AnewCpu = (float*) malloc(sizeof(float) * d_Anew.originalProperty.grid_size[0] * d_Anew.originalProperty.grid_size[1]);
#endif

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

#ifdef PROFILE
		auto init_start_clk_point = std::chrono::high_resolution_clock::now();
#endif

		//   set boundary conditions
		int bottom_range[] = {-1, imax+1, -1, 0};
		ops_par_loop_set_zero(2, bottom_range,
				d_A);

		int top_range[] = {-1, imax+1, jmax, jmax+1};
		ops_par_loop_set_zero(2, top_range,
				d_A);

		int left_range[] = {-1, 0, -1, jmax+1};
		ops_par_loop_left_bndcon(2, left_range,
				d_A);

		int right_range[] = {imax, imax+1, -1, jmax+1};
		ops_par_loop_right_bndcon(2, right_range,
				d_A);

#ifdef PROFILE
		auto init_end_clk_point = std::chrono::high_resolution_clock::now();
		init_runtime[bat] = std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif

		printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);


#ifdef PROFILE
		init_start_clk_point = std::chrono::high_resolution_clock::now();
#endif

		ops_par_loop_set_zero(2, bottom_range,
				d_Anew);

		ops_par_loop_set_zero( 2, top_range,
				d_Anew);

		ops_par_loop_left_bndcon(2, left_range,
				d_Anew);

		ops_par_loop_right_bndcon(2, right_range,
				d_Anew);

#ifdef PROFILE
		init_end_clk_point = std::chrono::high_resolution_clock::now();
		init_runtime[bat] += std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif

#ifdef VERIFICATION
		  getGrid(d_A);
		  getGrid(d_Anew);


		if(verify(A, Anew, d_A.originalProperty))
			std::cout << "verification of d_A and d_Anew" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of d_A and d_Anew" << "[FAILED]" << std::endl;

		initilizeGrid(Acpu, d_A.originalProperty, pi, jmax);
		copyGrid(AnewCpu, Acpu, d_A.originalProperty);

		if (verify(Acpu, A, d_A.originalProperty))
			std::cout << "verification of Acpu and A" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of Acpu and A" << "[FAILED]" << std::endl;
		//  printGrid2D<float>(d_A, "A");
		//  printGrid2D<float>(Acpu, d_A.originalProperty, "Acpu");

		if (verify(AnewCpu, Anew, d_A.originalProperty))
			std::cout << "verification of AnewCpu and Anew" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of AnewCpu and Anew" << "[FAILED]" << std::endl;
#endif

//		  getGrid(d_A);
//		  testInitGrid(d_A.hostBuffer.data(), d_A.originalProperty);
//		  d_A.isHostBufDirty = true;
//		  sendGrid(d_A);
//		  printGrid2D<float>(d_A, "d_A_test");

#ifdef PROFILE
		auto main_loop_start_clk_point = std::chrono::high_resolution_clock::now();
#endif

		int interior_range[] = {0,imax,0,jmax};
		int swapMap[] = {0, 1};
		ops__itr_par_loop_outerloop_1(2, interior_range, d_A, d_Anew, iter_max, true, swapMap);
// 		for (int iter = 0; iter < iter_max; iter++)
// 		{
// 			int interior_range[] = {0,imax,0,jmax};
// 			ops_par_loop_apply_stencil(2, interior_range,
// 				d_A,
// 				d_Anew);

// 			ops_par_loop_copy(2, interior_range,
// 				d_Anew,
// 				d_A);

// //			if(iter % 10 == 0) printf("%5d\n", iter);
// 		}

#ifdef PROFILE
		auto main_loop_end_clk_point = std::chrono::high_resolution_clock::now();
		main_loop_runtime[bat] = std::chrono::duration<double, std::micro>(main_loop_end_clk_point - main_loop_start_clk_point).count();
#endif

#ifdef VERIFICATION
		for (int iter = 0; iter < iter_max; iter++)
		{
			calcGrid(Acpu, AnewCpu, d_A.originalProperty);
			copyGrid(Acpu, AnewCpu, d_A.originalProperty);
		}

		getGrid(d_A);
		getGrid(d_Anew);

		if (verify(A, Acpu, d_A.originalProperty))
			std::cout << "verification of A and Acpu after calc" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of A and Acpu after calc" << "[FAILED]" << std::endl;

		if (verify(Anew, AnewCpu, d_Anew.originalProperty))
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
	}

#ifdef PROFILE
	std::cout << std::endl;
	std::cout << "******************************************************" << std::endl;
	std::cout << "**                runtime summary                   **" << std::endl;
	std::cout << "******************************************************" << std::endl;

	double avg_main_loop_runtime = 0;
	double max_main_loop_runtime = 0;
	double min_main_loop_runtime = 0;
	double avg_init_runtime = 0;
	double max_init_runtime = 0;
	double min_init_runtime = 0;
	double main_loop_std = 0;
	double init_std = 0;
	double total_std = 0;

	for (unsigned int bat = 0; bat < batches; bat++)
	{
		std::cout << "run: "<< bat << "| total runtime: " << main_loop_runtime[bat] + init_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> init runtime: " << init_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> main loop runtime: " << main_loop_runtime[bat] << "(us)" << std::endl;
		avg_init_runtime += init_runtime[bat];
		avg_main_loop_runtime += main_loop_runtime[bat];

		if (bat == 0)
		{
			max_main_loop_runtime = main_loop_runtime[bat];
			min_main_loop_runtime = main_loop_runtime[bat];
			max_init_runtime = init_runtime[bat];
			min_init_runtime = init_runtime[bat];
		}
		else
		{
			max_main_loop_runtime = std::max(max_main_loop_runtime, main_loop_runtime[bat]);
			min_main_loop_runtime = std::min(min_main_loop_runtime, main_loop_runtime[bat]);
			max_init_runtime = std::max(max_init_runtime, init_runtime[bat]);
			min_init_runtime = std::min(min_init_runtime, init_runtime[bat]);
		}
	}

	avg_init_runtime /= batches;
	avg_main_loop_runtime /= batches;

	for (unsigned int bat = 0; bat < batches; bat++)
	{
		main_loop_std += std::pow(main_loop_runtime[bat] - avg_main_loop_runtime, 2);
		init_std += std::pow(init_runtime[bat] - avg_init_runtime, 2);
		total_std += std::pow(main_loop_runtime[bat] + init_runtime[bat] - avg_init_runtime - avg_main_loop_runtime, 2);
	}

	main_loop_std = std::sqrt(main_loop_std / batches);
	init_std = std::sqrt(init_std / batches);
	total_std = std::sqrt(total_std / batches);

	std::cout << "Total runtime (AVG): " << avg_main_loop_runtime + avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << avg_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime (MIN): " << min_main_loop_runtime + min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << min_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime (MAX): " << max_main_loop_runtime + max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << max_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Standard Deviation init: " << init_std << std::endl;
	std::cout << "Standard Deviation main loop: " << main_loop_std << std::endl;
	std::cout << "Standard Deviation total: " << total_std << std::endl;
	std::cout << "======================================================" << std::endl;
#endif
	ops_exit_backend();
	//  free(A);
	//  free(Anew);
	std::cout << "Exit properly" << std::endl;
	return 0;
}
