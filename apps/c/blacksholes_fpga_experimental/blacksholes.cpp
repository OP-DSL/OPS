
/** @brief Implementation to run OPS implementation as standalone with contrast to blacksholes_app.
  * @author Beniel Thileepan
  * @details
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing multi-block Structured mesh applications.
  *  Coded in C API.
  */

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "blacksholes_utils.h"
#include "blacksholes_cpu.h"

#define OPS_1D
#define OPS_HLS_V2
// #define OPS_FPGA
#define VERIFICATION
#define DEBUG_VERBOSE
#define PROFILE
#include <ops_seq_v2.h>
#include "blacksholes_kernels.h"

extern const unsigned short mem_vector_factor;

int main(int argc, const char **argv)
{
    // OPS initialisation
    ops_init(argc,argv,1);

	GridParameter gridProp;
	gridProp.logical_size_x = 200;
	gridProp.logical_size_y = 1;
	gridProp.batch = 1;
	gridProp.num_iter = 6000;

    	// setting grid parameters given by user
	const char* pch;

	for ( int n = 1; n < argc; n++ )
	{
		pch = strstr(argv[n], "-sizex=");

		if(pch != NULL)
		{
			gridProp.logical_size_x = atoi ( argv[n] + 7 ); continue;
		}

		pch = strstr(argv[n], "-iters=");

		if(pch != NULL)
		{
			gridProp.num_iter = atoi ( argv[n] + 7 ); continue;
		}
		pch = strstr(argv[n], "-batch=");

		if(pch != NULL)
		{
			gridProp.batch = atoi ( argv[n] + 7 ); continue;
		}
	}

    	printf("Grid: %dx1 , %d iterations, %d batches\n", gridProp.logical_size_x, gridProp.num_iter, gridProp.batch);

	//adding halo
	gridProp.act_size_x = gridProp.logical_size_x+2;
	gridProp.act_size_y = 1;

	//padding each row as multiple of vectorization factor
	gridProp.grid_size_x = (gridProp.act_size_x % mem_vector_factor) != 0 ?
			(gridProp.act_size_x/mem_vector_factor + 1) * mem_vector_factor :
			gridProp.act_size_x;
	gridProp.grid_size_y = gridProp.act_size_y;

	//allocating memory buffer
	unsigned int data_size_bytes = gridProp.grid_size_x * gridProp.grid_size_y * sizeof(float) * gridProp.batch;

	if(data_size_bytes >= 4000000000)
	{
		std::cerr << "Maximum buffer size is exceeded!" << std::endl;
		return -1;
	}

#ifdef PROFILE
	double init_cpu_runtime[gridProp.batch];
	double main_loop_cpu_runtime[gridProp.batch];
	double init_runtime[gridProp.batch];
	double main_loop_runtime[gridProp.batch];
#endif
	std::vector<BlacksholesParameter> calcParam(gridProp.batch); //multiple blacksholes calculations

	//First calculation for test value

	calcParam[0].spot_price = 16;
	calcParam[0].strike_price = 10;
	calcParam[0].time_to_maturity = 0.25;
	calcParam[0].volatility = 0.4;
	calcParam[0].risk_free_rate = 0.1;
	calcParam[0].N = gridProp.num_iter;
	calcParam[0].K = gridProp.logical_size_x;
	calcParam[0].SMaxFactor = 3;
	calcParam[0].delta_t = calcParam[0].time_to_maturity / calcParam[0].N;
	calcParam[0].delta_S = calcParam[0].strike_price * calcParam[0].SMaxFactor/ (calcParam[0].K);
	calcParam[0].stable = stencil_stability(calcParam[0]);

	std::random_device dev;
	std::mt19937 rndGen(dev());
	std::uniform_real_distribution<> dis(0.0, 0.05);

	for (int i = 1; i < gridProp.batch; i++)
	{
		calcParam[i].spot_price = 16 + dis(rndGen);
		calcParam[i].strike_price = 10 + dis(rndGen);
		calcParam[i].time_to_maturity = 0.25 + dis(rndGen);
		calcParam[i].volatility = 0.4 + dis(rndGen);
		calcParam[i].risk_free_rate = 0.1 + dis(rndGen);
		calcParam[i].N = gridProp.num_iter;
		calcParam[i].K = gridProp.logical_size_x;
		calcParam[i].SMaxFactor = 3;
		calcParam[i].delta_t = calcParam[i].time_to_maturity / calcParam[i].N;
		calcParam[i].delta_S = calcParam[i].strike_price * calcParam[i].SMaxFactor/ (calcParam[i].K);
		calcParam[i].stable = stencil_stability(calcParam[i]);

		if (not calcParam[i].stable)
		{
			std::cerr << "Calc job: " << i << " is unstable" << std::endl;
		}
	}

	//ops_block
	ops_block grid1D = ops_decl_block(1, "grid1D");

	//ops_data
	int size[] = {static_cast<int>(gridProp.logical_size_x)};
	int base[] = {0};
	int d_m[] = {-1};
	int d_p[] = {1};

	ops_dat dat_current[gridProp.batch]; // = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, current,"float", "dat_current");
	ops_dat dat_next[gridProp.batch];// = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, next,"float", "dat_next");
	ops_dat dat_a[gridProp.batch]; // = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, a, "float", "dat_a");
	ops_dat dat_b[gridProp.batch]; // = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, b, "float", "dat_b");
	ops_dat dat_c[gridProp.batch]; // = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, c, "float", "dat_c");

#ifdef VERIFICATION
    float * grid_u1_cpu[gridProp.batch]; // = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_cpu[gridProp.batch]; // = (float*) aligned_alloc(4096, data_size_bytes);
#endif
	//Allocation
	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		float *current = nullptr, *next = nullptr;
		float *a = nullptr, *b = nullptr, *c = nullptr;

		std::string name = std::string("current_") + std::to_string(bat);
		dat_current[bat] = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, current,"float", name.c_str());
		name = std::string("next_") + std::to_string(bat);
		dat_next[bat] = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, next,"float", name.c_str());
		name = std::string("a_") + std::to_string(bat);
		dat_a[bat] = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, a, "float", name.c_str());
		name = std::string("b_") + std::to_string(bat);
		dat_b[bat] = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, a, "float", name.c_str());
		name = std::string("x_") + std::to_string(bat);
		dat_c[bat] = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, a, "float", name.c_str());
#ifdef VERIFICATION
		grid_u1_cpu[bat] = (float*) aligned_alloc(4096, data_size_bytes);
		grid_u2_cpu[bat] = (float*) aligned_alloc(4096, data_size_bytes);
#endif
	}

#ifdef VERIFICATION
	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
	#ifdef PROFILE
		auto init_start_clk_point = std::chrono::high_resolution_clock::now();
	#endif

		intialize_grid(grid_u1_cpu[bat], gridProp, calcParam[bat]);
		copy_grid(grid_u1_cpu[bat], grid_u2_cpu[bat], gridProp);

	#ifdef PROFILE
		auto init_stop_clk_point = std::chrono::high_resolution_clock::now();
		init_cpu_runtime[bat] = std::chrono::duration<double, std::micro> (init_stop_clk_point - init_start_clk_point).count();
	#endif
	}
	
	#ifdef DEBUG_VERBOSE
	std::cout << std::endl;
	std::cout << "*********************************************"  << std::endl;
	std::cout << "**            intial grid values           **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		std::cout << "Batch [" << bat << "]" << std::endl;
		for (unsigned int i = 0; i < gridProp.act_size_x; i++)
		{
			std::cout << "index: " << i << " initial_val: " << grid_u1_cpu[bat][i]<< std::endl;
		}
	}
	std::cout << "============================================="  << std::endl << std::endl;
	#endif
#endif
	//defining the stencils
	int s1d_3pt[] = {-1, 0, 1};
	int s1d_1pt[] = {0};

	ops_stencil S1D_3pt = ops_decl_stencil(1, 3, s1d_3pt, "3pt");
	ops_stencil S1D_1pt = ops_decl_stencil(1, 1, s1d_1pt, "1pt");

	//partition
	ops_partition("1D_BLOCK_DECOMPOSE");

	int lower_Pad_range[] = {-1,0};
	int upper_pad_range[] = {gridProp.logical_size_x, gridProp.logical_size_x + 1};
	int interior_range[] = {0, gridProp.logical_size_x};
	int full_range[] = {-1, gridProp.logical_size_x + 1};

	//initializing data
	for (int bat = 0; bat < gridProp.batch; bat++)
	{
#ifdef PROFILE
		auto grid_init_start_clk_point = std::chrono::high_resolution_clock::now();
#endif
		ops_par_loop(ops_krnl_zero_init, "ops_zero_init", grid1D, 1, lower_Pad_range,
				ops_arg_dat(dat_current[bat], 1, S1D_1pt, "float", OPS_WRITE));

		
		float sMax = calcParam[bat].SMaxFactor * calcParam[bat].strike_price;

		ops_par_loop(ops_krnl_const_init, "ops_const_init", grid1D, 1, upper_pad_range,
				ops_arg_dat(dat_current[bat], 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_gbl(&sMax, 1, "float", OPS_READ));
		
		float* delta_s = &calcParam[bat].delta_S;
		float* strike_price = &calcParam[bat].strike_price;

		ops_par_loop(ops_krnl_interior_init, "interior_init", grid1D, 1, interior_range,
				ops_arg_dat(dat_current[bat], 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_idx(),
				ops_arg_gbl(delta_s, 1, "float", OPS_READ),
				ops_arg_gbl(strike_price, 1,"float", OPS_READ));


		ops_par_loop(ops_krnl_copy, "init_dat_next", grid1D, 1, full_range,
				ops_arg_dat(dat_next[bat], 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_current[bat], 1, S1D_1pt, "float", OPS_READ));

		//blacksholes calc
		float alpha = calcParam[bat].volatility * calcParam[bat].volatility * calcParam[bat].delta_t;
		float beta = calcParam[bat].risk_free_rate * calcParam[bat].delta_t;

		ops_par_loop(ops_krnl_calc_coefficient, "calc_coefficient", grid1D, 1, interior_range,
				ops_arg_dat(dat_a[bat], 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_b[bat], 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_c[bat], 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_gbl(&(alpha), 1, "float", OPS_READ),
				ops_arg_gbl(&(beta), 1 , "float", OPS_READ),
				ops_arg_idx());

#ifdef PROFILE
		auto grid_init_stop_clk_point = std::chrono::high_resolution_clock::now();
		init_runtime[bat] = std::chrono::duration<double, std::micro>(grid_init_stop_clk_point - grid_init_start_clk_point).count();
		auto blacksholes_calc_start_clk_point = grid_init_stop_clk_point;
#endif
#ifndef OPS_FPGA
		for (int iter = 0 ; iter < calcParam[bat].N; iter++)
		{
			ops_par_loop(ops_krnl_blacksholes, "blacksholes_1", grid1D, 1, interior_range,
					ops_arg_dat(dat_current[bat], 1, S1D_1pt, "float", OPS_WRITE),
					ops_arg_dat(dat_next[bat], 1, S1D_3pt, "float", OPS_READ),
					ops_arg_dat(dat_a[bat], 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_b[bat], 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_c[bat], 1, S1D_1pt, "float", OPS_READ));
		}
#else
        ops_iter_par_loop("ops_iter_par_loop_0", calcParam[bat].N,
			ops_par_loop(ops_krnl_blacksholes, "blacksholes_1", grid1D, 1, interior_range,
					ops_arg_dat(dat_current[bat], 1, S1D_1pt, "float", OPS_WRITE),
					ops_arg_dat(dat_next[bat], 1, S1D_3pt, "float", OPS_READ),
					ops_arg_dat(dat_a[bat], 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_b[bat], 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_c[bat], 1, S1D_1pt, "float", OPS_READ)),
            ops_par_copy<float>(dat_current[bat], dat_next[bat]));
#endif

#ifdef PROFILE
		auto blacksholes_calc_stop_clk_point = std::chrono::high_resolution_clock::now();
		main_loop_runtime[bat] = std::chrono::duration<double, std::micro>(blacksholes_calc_stop_clk_point - blacksholes_calc_start_clk_point).count();
#endif 
}

#ifdef VERIFICATION
    for (unsigned int bat = 0; bat < gridProp.batch; bat++)
    { 
	#ifdef PROFILE
		auto naive_start_clk_point = std::chrono::high_resolution_clock::now();
	#endif
		//golden computation
		bs_explicit1(grid_u1_cpu[bat], grid_u2_cpu[bat], gridProp, calcParam[bat]);
	
	#ifdef PROFILE
		auto naive_stop_clk_point = std::chrono::high_resolution_clock::now();
		main_loop_cpu_runtime[bat] = std::chrono::duration<double, std::micro> (naive_stop_clk_point - naive_start_clk_point).count(); 
	#endif
	}

	std::cout << std::endl;
	std::cout << "*********************************************"  << std::endl;
	std::cout << "**      Debug info after calculations      **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		//explicit blacksholes test
    	float direct_calc_value = test_blacksholes_call_option(calcParam[bat]]);

		float* current_raw = (float*)ops_dat_get_raw_pointer(dat_current[bat], 0, S1D_1pt, OPS_HOST);
		float* next_raw = (float*)ops_dat_get_raw_pointer(dat_next[bat], 0, S1D_1pt, OPS_HOST);

	#ifdef DEBUG_VERBOSE

		printGrid2D<float>(next_raw, dat_next[bat].originalProperty, "next after computation");
		printGrid2D<float>(grid_u2_cpu[bat], dat_next[bat].originalProperty, "next_Acpu after computation");
	#endif
        if(verify(next_raw, grid_u2_cpu[bat], size, d_m, d_p, full_range))
            std::cout << "[BATCH - " << bat << "] verification of current after calculation" << "[PASSED]" << std::endl;
        else
            std::cout << "[BATCH - " << bat << "] verification of current after calculation" << "[FAILED]" << std::endl;
		std::cout << "[BATCH - " << bat << "] call option price from cpu direct calc method: " << direct_calc_value << std::endl;	
		std::cout << "[BATCH - " << bat << "] call option price from cpu explicit iter method: " << get_call_option(grid_u2_cpu[bat], calcParam[bat]) << std::endl;
		std::cout << "[BATCH - " << bat << "] call option price from ops explicit method: " << get_call_option(next_raw, calcParam[bat]) << std::endl;
	}
	std::cout << "============================================="  << std::endl << std::endl;
#endif
	
	    //cleaning
    for (unsigned int bat = 0; bat < gridProp.batch; bat++)
    {
        ops_free_dat(dat_current[bat]);
        ops_free_dat(dat_next[bat]);
        ops_free_dat(dat_a[bat]);
        ops_free_dat(dat_b[bat]);
		ops_free_dat(dat_c[bat]);
#ifdef VERIFICATION
        free(grid_u1_cpu[bat]);
        free(grid_u2_cpu[bat]);
#endif
    }
//	for (int i = 0; i < gridProp.logical_size_x; i++)
//	{
//		std::cout << "idx: " << i << ", dat_current: " << grid_ops_result[i] << std::endl;
//	}
//	ops_print_dat_to_txtfile_core(dat_current, "dat_current.txt");
//	ops_print_dat_to_txtfile_core(dat_next, "dat_next.txt");
	
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

	#ifdef VERIFICATION
	double cpu_avg_main_loop_runtime = 0;
	double cpu_max_main_loop_runtime = 0;
	double cpu_min_main_loop_runtime = 0;
	double cpu_avg_init_runtime = 0;
	double cpu_max_init_runtime = 0;
	double cpu_min_init_runtime = 0;
	double cpu_main_loop_std = 0;
	double cpu_init_std = 0;
	double cpu_total_std = 0;
	#endif

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		std::cout << "run: "<< bat << "| total runtime (DEVICE): " << main_loop_runtime[bat] + init_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> init runtime: " << init_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> main loop runtime: " << main_loop_runtime[bat] << "(us)" << std::endl;
	#ifdef VERIFICATION	
		std::cout << "| total runtime (CPU-golden): " << main_loop_cpu_runtime[bat] + init_cpu_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> init runtime: " << init_cpu_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> main loop runtime: " << main_loop_cpu_runtime[bat] << "(us)" << std::endl;
	#endif
		avg_init_runtime += init_runtime[bat];
		avg_main_loop_runtime += main_loop_runtime[bat];
	#ifdef VERIFICATION
		cpu_avg_init_runtime += init_cpu_runtime[bat];
		cpu_avg_main_loop_runtime += main_loop_cpu_runtime[bat];
	#endif

		if (bat == 0)
		{
			max_main_loop_runtime = main_loop_runtime[bat];
			min_main_loop_runtime = main_loop_runtime[bat];
			max_init_runtime = init_runtime[bat];
			min_init_runtime = init_runtime[bat];
	#ifdef VERIFICATION
			cpu_max_main_loop_runtime = main_loop_cpu_runtime[bat];
			cpu_min_main_loop_runtime = main_loop_cpu_runtime[bat];
			cpu_max_init_runtime = init_cpu_runtime[bat];
			cpu_min_init_runtime = init_cpu_runtime[bat];
	#endif
		}
		else
		{
			max_main_loop_runtime = std::max(max_main_loop_runtime, main_loop_runtime[bat]);
			min_main_loop_runtime = std::min(min_main_loop_runtime, main_loop_runtime[bat]);
			max_init_runtime = std::max(max_init_runtime, init_runtime[bat]);
			min_init_runtime = std::min(min_init_runtime, init_runtime[bat]);
	#ifdef VERIFICATION
			cpu_max_main_loop_runtime = std::max(cpu_max_main_loop_runtime, main_loop_cpu_runtime[bat]);
			cpu_min_main_loop_runtime = std::min(cpu_min_main_loop_runtime, main_loop_cpu_runtime[bat]);
			cpu_max_init_runtime = std::max(cpu_max_init_runtime, init_cpu_runtime[bat]);
			cpu_min_init_runtime = std::min(cpu_min_init_runtime, init_cpu_runtime[bat]);
	#endif
		}
	}

	avg_init_runtime /= gridProp.batch;
	avg_main_loop_runtime /= gridProp.batch;

	#ifdef VERIFICATION
	cpu_avg_init_runtime /= gridProp.batch;
	cpu_avg_main_loop_runtime /= gridProp.batch;
	#endif

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		main_loop_std += std::pow(main_loop_runtime[bat] - avg_main_loop_runtime, 2);
		init_std += std::pow(init_runtime[bat] - avg_init_runtime, 2);
		total_std += std::pow(main_loop_runtime[bat] + init_runtime[bat] - avg_init_runtime - avg_main_loop_runtime, 2);
	#ifdef VERIFICATION
		cpu_main_loop_std += std::pow(main_loop_cpu_runtime[bat] - cpu_avg_main_loop_runtime, 2);
		cpu_init_std += std::pow(init_cpu_runtime[bat] - cpu_avg_init_runtime, 2);
		cpu_total_std += std::pow(main_loop_cpu_runtime[bat] + init_cpu_runtime[bat] - cpu_avg_init_runtime - cpu_avg_main_loop_runtime, 2);
	#endif
	}

	main_loop_std = std::sqrt(main_loop_std / gridProp.batch);
	init_std = std::sqrt(init_std / gridProp.batch);
	total_std = std::sqrt(total_std / gridProp.batch);
	#ifdef VERIFICATION
	cpu_main_loop_std = std::sqrt(cpu_main_loop_std / gridProp.batch);
	cpu_init_std = std::sqrt(cpu_init_std / gridProp.batch);
	cpu_total_std = std::sqrt(cpu_total_std / gridProp.batch);
	#endif
	std::cout << "Total runtime - DEVICE (AVG): " << avg_main_loop_runtime + avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << avg_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime - DEVICE (MIN): " << min_main_loop_runtime + min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << min_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime - DEVICE (MAX): " << max_main_loop_runtime + max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << max_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Standard Deviation init - DEVICE: " << init_std << std::endl;
	std::cout << "Standard Deviation main loop - DEVICE: " << main_loop_std << std::endl;
	std::cout << "Standard Deviation total - DEVICE: " << total_std << std::endl;
	#ifdef VERIFICATION
	std::cout << "Total runtime - CPU (AVG): " << cpu_avg_main_loop_runtime + cpu_avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << cpu_avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << cpu_avg_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime - CPU (MIN): " << cpu_min_main_loop_runtime + cpu_min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << cpu_min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << cpu_min_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime - CPU (MAX): " << cpu_max_main_loop_runtime + cpu_max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << cpu_max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << cpu_max_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Standard Deviation init - CPU: " << cpu_init_std << std::endl;
	std::cout << "Standard Deviation main loop - CPU: " << cpu_main_loop_std << std::endl;
	std::cout << "Standard Deviation total - CPU: " << cpu_total_std << std::endl;
	#endif
	std::cout << "======================================================" << std::endl;
#endif

	//Finalizing the OPS library

    ops_exit();

 	std::cout << "Exit properly" << std::endl;
    return 0;
}