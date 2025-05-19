// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <string>
#include <ops_hls_defs.hpp>
#include <ops_hls_utils.hpp>
#include <ops_hls_fpga.hpp>
#include "poisson_cpu.h"



/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{

    //setting Mesh default parameters
    struct ops::hls::GridPropertyCore data_g;
    data_g.size[0] = 20;
    data_g.size[1] = 20;
    
    unsigned int batch = 2;
    unsigned int vector_factor = 8;

    // number of iterations
    int n_iter = 10;

    // setting grid parameters given by user
    const char* pch;
    for ( int n = 1; n < argc; n++ ) {
        pch = strstr(argv[n], "-sizex=");
        if(pch != NULL) {
        data_g.size[0] = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-sizey=");
        if(pch != NULL) {
        data_g.size[1] = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-iters=");
        if(pch != NULL) {
        n_iter = atoi ( argv[n] + 7 ); continue;
        }

        pch = strstr(argv[n], "-batch=");
        if(pch != NULL) {
        batch = atoi ( argv[n] + 7 ); continue;
        }
    }

    printf("Grid: %dx%d , %d iterations, %d batches\n", data_g.size[0], data_g.size[1], n_iter, batch);

    // adding boundary
    data_g.d_m[0] = 1;
    data_g.d_m[1] = 1;
    data_g.d_p[0] = 1;
    data_g.d_m[1] = 1;
    data_g.actual_size[0] = data_g.size[0] + data_g.d_m[0] + data_g.d_p[0];
    data_g.actual_size[1] = data_g.size[1] + data_g.d_m[0] + data_g.d_p[0];
    unsigned short shift_bits = LOG2(vector_factor);
    data_g.xblocks = (data_g.actual_size[0] + vector_factor - 1) >> shift_bits;
    data_g.grid_size[0] =  data_g.xblocks << shift_bits;
    data_g.grid_size[1] = data_g.actual_size[1];
    data_g.total_itr = (data_g.actual_size[1] + 1) * data_g.xblocks;  //(data_g.actual_size[1] + p/2) * xblocks.
    data_g.outer_loop_limit = (data_g.actual_size[1] + 1);

    
    // allocating memory for host program and FPGA buffers
    unsigned int data_size_bytes = data_g.grid_size[0] * data_g.grid_size[1] * sizeof(float) * batch;
    data_size_bytes = (data_size_bytes % 16 != 0) ? (data_size_bytes/16 +1)*16 : data_size_bytes;

    if(data_size_bytes >= 4000000000){
        printf("Maximum buffer size is exceeded!\n");
        return 0;
    }
    float * grid_u1 = (float*)aligned_alloc(4096, data_size_bytes);
    float * grid_u2 = (float*)aligned_alloc(4096, data_size_bytes);

    float * grid_u1_d = (float*)aligned_alloc(4096, data_size_bytes);
    float * grid_u2_d = (float*)aligned_alloc(4096, data_size_bytes);

    // setting boundary value and copying to FPGA buffer
    initialise_grid(grid_u1, data_g);
    copy_grid(grid_u1, grid_u1_d, data_g);



    //OPENCL HOST CODE AREA START
        std::string binaryFile(argv[1]);
        unsigned int device_id = 0;

        ops::hls::FPGA fpga(device_id);
        fpga.xclbin(binaryFile);
        

        auto devices = xcl::get_xil_devices();
        auto device = devices[0];

        OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));


        //Create Program and Kernel
        auto fileBuf = xcl::read_binary_file(binaryFile);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
        devices.resize(1);
        OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
        OCL_CHECK(err, cl::Kernel krnl_slr0(program, "stencil_SLR0", &err));
        OCL_CHECK(err, cl::Kernel krnl_slr1(program, "stencil_SLR1", &err));
        OCL_CHECK(err, cl::Kernel krnl_slr2(program, "stencil_SLR2", &err));



        //Allocate Buffer in Global Memory
        OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u1_d, &err));
        OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, data_size_bytes, grid_u2_d, &err));


        //Set the Kernel Arguments
        int narg = 0;
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, buffer_input));
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, buffer_output));
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_x));
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_y));
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.grid_size_x));
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, n_iter));
        OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.batch));

        narg = 0;
        OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_x));
        OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_y));
        OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.grid_size_x));
        OCL_CHECK(err, err = krnl_slr1.setArg(narg++, n_iter));
        OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.batch));

        narg = 0;
        OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_x));
        OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_y));
        OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.grid_size_x));
        OCL_CHECK(err, err = krnl_slr2.setArg(narg++, n_iter));
        OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.batch));

        //Copy input data to device global memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/));

        auto start = std::chrono::high_resolution_clock::now();




        //Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_slr0));
        OCL_CHECK(err, err = q.enqueueTask(krnl_slr1));
        OCL_CHECK(err, err = q.enqueueTask(krnl_slr2));
        q.finish();


        auto finish = std::chrono::high_resolution_clock::now();

        //Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, CL_MIGRATE_MEM_OBJECT_HOST));

        q.finish();


    // golden stencil computation on host
    for(int itr = 0; itr < n_iter*60; itr++){
        stencil_computation(grid_u1, grid_u2, data_g);
        stencil_computation(grid_u2, grid_u1, data_g);
    }
        
    std::chrono::duration<double> elapsed = finish - start;

    printf("Runtime on FPGA is %f seconds\n", elapsed.count());
    double error = square_error(grid_u1, grid_u1_d, data_g);
    float bandwidth = (data_g.logical_size_x * data_g.logical_size_y * sizeof(float) * 4.0 * n_iter * data_g.batch)/(elapsed.count() * 1000 * 1000 * 1000);
    printf("\nMean Square error is  %f\n\n", error/(data_g.logical_size_x * data_g.logical_size_y));
    printf("\nOPS Bandwidth is %f\n", bandwidth);

    free(grid_u1);
    free(grid_u2);
    free(grid_u1_d);
    free(grid_u2_d);

    return 0;
}
