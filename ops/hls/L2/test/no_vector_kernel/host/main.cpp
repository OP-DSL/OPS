
#include <iostream>
#include <stdlib.h>
#include <string>
#include <ops_hls_fpga.hpp>
#include <ops_hls_defs.hpp>
#include <xcl2.hpp>
#include <cassert>
#include <iomanip>

void getRangeAdjustedGridProp(ops::hls::GridPropertyCore& original, 
		ops::hls::AccessRange& range, ops::hls::GridPropertyCore& adjusted, const unsigned short vector_factor=8)
{
	assert(range.dim == original.dim);

	for (int i = 0; i < range.dim; i++)
	{
		assert(range.end[i] <= original.actual_size[i] - original.d_p[i]);
		assert(range.start[i] >= 0);
	}

	adjusted.dim = original.dim;

	for (int i = 0; i < range.dim; i++)
	{
		adjusted.size[i] = range.end[i] - range.start[i];
		adjusted.d_m[i] = original.d_m[i];
		adjusted.d_p[i] = original.d_p[i];
		adjusted.actual_size[i] = adjusted.size[i] + adjusted.d_p[i] + adjusted.d_m[i];
		if (i == 0)
		{
			adjusted.xblocks = (adjusted.actual_size[0] + vector_factor - 1) / vector_factor;
			adjusted.grid_size[0] = adjusted.xblocks * vector_factor;
		}
		else
			adjusted.grid_size[i] = adjusted.actual_size[i];
	}
}

template <typename T>
void printGrid2D(T* data, ops::hls::GridPropertyCore& gridProp)
{
	for (int j = 0; gridProp.grid_size[1]; j++)
	{
		for (int i = 0; gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			std::cout << std::setw(10) << data[index];
		}
		std::cout << std::endl;
	}
}

void printGridProp(ops::hls::GridPropertyCore& gridProp)
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "         grid properties       " << std::endl;
	std::cout << "-------------------------------" << std::endl;
	std::cout << std::setw(15) << std::right << "dim: "  << gridProp.dim << std::endl;
	std::cout << std::setw(15) << std::right << "d_m: " << "(" << gridProp.d_m[0] 
				<< ", " << gridProp.d_m[1] << ", " << gridProp.d_m[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "d_p: " << "(" << gridProp.d_p[0] 
				<< ", " << gridProp.d_p[1] << ", " << gridProp.d_p[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "logical size: " << "(" << gridProp.size[0] 
				<< ", " << gridProp.size[1] << ", " << gridProp.size[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "actual size: " << "(" << gridProp.actual_size[0] 
				<< ", " << gridProp.actual_size[1] << ", " << gridProp.actual_size[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "grid size: " << "(" << gridProp.grid_size[0] 
				<< ", " << gridProp.grid_size[1] << ", " << gridProp.grid_size[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "xblocks: " << gridProp.xblocks << std::endl;
	std::cout << std::setw(15) << std::right << "total iterations: " << gridProp.total_itr << std::endl;
	std::cout << std::setw(15) << std::right << "outer limit: " << gridProp.outer_loop_limit << std::endl;
	std::cout << "-------------------------------" << std::endl;
}

int main(int argc, char **argv)
{
	unsigned int deviceId = 0;
	ops::hls::FPGA fpga(deviceId);

	std::string xclbinFile = argv[1];
	fpga.xclbin(xclbinFile);

	cl_int err;
	cl::Context context  = fpga.getContext();
	cl::Program program = fpga.getProgram();
	cl::CommandQueue queue = fpga.getCommandQueue();

	unsigned int vector_factor = 8;

	ops::hls::GridPropertyCore gridProp;
	gridProp.size[0] = 10;
	gridProp.size[1] = 10;
	gridProp.d_m[0] = 1;
	gridProp.d_m[1] = 1;
	gridProp.d_p[0] = 1;
	gridProp.d_p[1] = 1;
	gridProp.actual_size[0] = gridProp.size[0] + gridProp.d_p[0] + gridProp.d_m[0];
	gridProp.actual_size[1] = gridProp.size[1] + gridProp.d_p[1] + gridProp.d_m[1];
	gridProp.xblocks = (gridProp.actual_size[0] + vector_factor - 1) / vector_factor;
	gridProp.grid_size[0] = gridProp.xblocks * vector_factor;
	gridProp.grid_size[1] = gridProp.actual_size[1];
	gridProp.outer_loop_limit = gridProp.actual_size[1] + 1;
	gridProp.total_itr = gridProp.actual_size[1] * gridProp.xblocks;
	gridProp.dim = 2;

	host_buffer_t<float> u(gridProp.grid_size[0] * gridProp.grid_size[1]);
	host_buffer_t<ops::hls::GridPropertyCore> buffer_gridProp(1);
	auto buffer_u = fpga.createDeviceBuffer(CL_MEM_READ_ONLY, u);


	static const std::string kernelSimpleCopyName("kernel_simple_copy");
	static const std::string kernelDatamoverName("kernel_datamover_simple_copy");

	OCL_CHECK(err, cl::Kernel krnl_simpleCopy(program, kernelSimpleCopyName.c_str(), &err));
	OCL_CHECK(err, cl::Kernel krn_datamover(program, kernelDatamoverName.c_str(), &err));

	int narg = 0;
	const float value = 2.5;

	printGridProp(gridProp);

	ops::hls::AccessRange range;
	range.dim = 2;
	range.start[0] = 0;
	range.start[1] = 0;
	range.end[0] = gridProp.actual_size[0];
	range.end[1] = gridProp.actual_size[1];

	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,value));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.size[0]));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.size[1]));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.actual_size[0]));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.actual_size[1]));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.grid_size[0]));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.grid_size[1]));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.dim));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.xblocks));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.total_itr));
	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,gridProp.outer_loop_limit));

	narg = 0;
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.start[0]));
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.end[0]));
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.start[1]));
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.end[1]));
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, gridProp.grid_size[0]));
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, gridProp.grid_size[1]));
	OCL_CHECK(err, err = krn_datamover.setArg(narg++, buffer_u));


	OCL_CHECK(err, err = queue.enqueueTask(krnl_simpleCopy));
	OCL_CHECK(err, err = queue.enqueueTask(krn_datamover));

	queue.finish();

//	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_u}, CL_MIGRATE_MEM_OBJECT_HOST));
//
//	queue.finish();

//	printGrid2D(u.data(), gridProp);

	return 0;
}
