
#include <iostream>
#include <stdlib.h>
#include <string>
#include <ops_hls_fpga.hpp>
#include <ops_hls_defs.hpp>
#include <ops_hls_host_utils.hpp>
#include <xcl2.hpp>
#include <cassert>
#include <iomanip>
#include <random>

#define DEBUG_LOG

void getRangeAdjustedGridProp(ops::hls::GridPropertyCore& original, 
		ops::hls::AccessRange& range, ops::hls::GridPropertyCore& adjusted, const unsigned short vector_factor=8)
{
	assert(range.dim == original.dim);

	for (int i = 0; i < range.dim; i++)
	{
		assert(range.end[i] <= original.actual_size[i]);
		assert(range.start[i] >= 0);
	}

	adjusted.dim = original.dim;

	for (int i = 0; i < range.dim; i++)
	{
		adjusted.size[i] = range.end[i] - range.start[i];
		adjusted.d_m[i] = 0;
		adjusted.d_p[i] = 0;
		adjusted.actual_size[i] = adjusted.size[i] + adjusted.d_p[i] + adjusted.d_m[i];
		if (i == 0)
		{
			adjusted.xblocks = (adjusted.actual_size[0] + vector_factor - 1) / vector_factor;
			adjusted.grid_size[0] = adjusted.xblocks * vector_factor;
			adjusted.total_itr = adjusted.xblocks;
		}
		else
		{
			adjusted.grid_size[i] = adjusted.actual_size[i];
			adjusted.total_itr *= adjusted.actual_size[i];
		}
	}

	adjusted.outer_loop_limit = adjusted.actual_size[adjusted.dim - 1] + (adjusted.d_m[adjusted.dim-1] + adjusted.d_p[adjusted.dim -1])/2;


}

void printGrid2D(float* data, ops::hls::GridPropertyCore& gridProp, std::string prompt="")
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << " [DEBUG] grid values: " << prompt << std::endl;
	std::cout << "-------------------------------" << std::endl;

	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			std::cout << std::setw(12) << data[index];
		}
		std::cout << std::endl;
	}
}

void writeGrid(float* data, const float& val, ops::hls::GridPropertyCore& gridProp)
{
	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			data[index] = val;
		}
	}
}

void copyGrid(float* dst, float* src, ops::hls::GridPropertyCore& gridProp)
{
	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			dst[index] = src[index];
		}
	}
}

struct TaskDataHolder
{
	ops::hls::GridPropertyCore gridProperty;
	ops::hls::AccessRange range;
	host_buffer_t<float> host_buffer;
	cl::Event fill_event;
	cl::Event task_event;
	cl::Event data_event;
	cl::Event receive_event;
	cl::Buffer cl_buffer;
	double task_runtime;
	bool task_verification;
};


void writeGrid(TaskDataHolder& task, const float& val)
{
	writeGrid(task.host_buffer.data(), val, task.gridProperty);
}

void printGridProp(ops::hls::GridPropertyCore& gridProp, std::string prompt = "")
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "  grid properties: " << prompt << std::endl;
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

void printAccessRange(ops::hls::AccessRange& range)
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "          Access Range         " << std::endl;
	std::cout << "-------------------------------" << std::endl;
	std::cout << std::setw(15) << std::right << "dim: "  << range.dim << std::endl;
	std::cout << std::setw(15) << std::right << "start: " << "(" << range.start[0]
				<< ", " << range.start[1] << ", " << range.start[2] <<")"<< std::endl;
	std::cout << std::setw(15) << std::right << "end: " << "(" << range.end[0]
				<< ", " << range.end[1] << ", " << range.end[2] <<")"<< std::endl;
	std::cout << "-------------------------------" << std::endl;
}

bool verification(TaskDataHolder& task, const float& fill, const float& val)
{
	ops::hls::GridPropertyCore & gridProp = task.gridProperty;

	bool verified = true;
	for (int j = 0; j < gridProp.grid_size[1]; j++)
	{
		for (int i = 0; i < gridProp.grid_size[0]; i++)
		{
			int index = i + j * gridProp.grid_size[0];
			bool cond_x = (i >= task.range.start[0]) and (i < task.range.end[0]);
			bool cond_y = (j >= task.range.start[1]) and (j < task.range.end[1]);

			if (cond_x and cond_y)
			{
				if (task.host_buffer[index] != val)
				{
					verified = false;
					std::cerr << "[ERROR]: val missed at, i:" << i <<", j:" << j <<", with value:" << task.host_buffer[index] <<". expected: " << val << std::endl;
				}
			}
			else
			{
				if (task.host_buffer[index] != fill)
				{
					verified = false;
					std::cerr << "[ERROR]: fill missed at, i:" << i <<", j:" << j <<", with value:" << task.host_buffer[index] <<". expected: " << fill << std::endl;
				}
			}
		}
	}
	return verified;
}

int main(int argc, char **argv)
{
	std::cout << std::endl;
	std::cout << "***************************************************" << std::endl;
	std::cout << " TEST: simple copy kernel" << std::endl;
	std::cout << "***************************************************" << std::endl << std::endl;

	std::string xclbinFile = argv[1];
	cl_int err;

	auto devices = xcl::get_xil_devices();
	auto device = devices[0];

	OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	OCL_CHECK(err, cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
	//Create program and Kernel
	auto bins = xcl::import_binary_file(xclbinFile);

	OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));

	static const std::string kernelSimpleCopyName("kernel_simple_copy");
	static const std::string kernelDatamoverName("kernel_datamover_simple_copy");

	OCL_CHECK(err, cl::Kernel krnl_simpleCopy(program, kernelSimpleCopyName.c_str(), &err));
	OCL_CHECK(err, cl::Kernel krn_datamover(program, kernelDatamoverName.c_str(), &err));

	unsigned int vector_factor = 8;

	const unsigned int num_tests = 10;
	std::cout << "TOTAL NUMBER OF TESTS: " << num_tests << std::endl;

	//random generators
	std::random_device rd;
	unsigned int seed = 7;
	std::mt19937 mtSeeded(seed);
	std::mt19937 mtRandom(rd());
	std::uniform_int_distribution<unsigned short> disInt(5,15);
	std::normal_distribution<float> disFloat(100, 10);

	//summery data holders
	std::vector<TaskDataHolder> TaskData(num_tests);

	const float fill = 2.5;
	//creating grid property and fill
	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
		TaskDataHolder& taskdata = TaskData[test_itr];
		unsigned short x_size= disInt(mtSeeded);
		taskdata.gridProperty = createGridPropery(2, {x_size, x_size,1},
				{1,1,0},
				{1,1,0});

		unsigned int data_size = taskdata.gridProperty.grid_size[0] * taskdata.gridProperty.grid_size[1];
		unsigned int data_size_bytes = data_size * sizeof(float);

		taskdata.host_buffer.resize(data_size);

		OCL_CHECK(err, taskdata.cl_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, taskdata.host_buffer.data(), &err));
		writeGrid(taskdata, fill);

		std::uniform_int_distribution<unsigned short> disLow(0, x_size/3);
		std::uniform_int_distribution<unsigned short> disHigh(x_size * 2 / 3, x_size);

		ops::hls::AccessRange range;
		range.dim = 2;
		range.start[0] = disLow(mtRandom);
		range.start[1] = disLow(mtRandom);
		range.end[0] = disHigh(mtRandom);
		range.end[1] = disHigh(mtRandom);

		taskdata.range = range;
#ifdef DEBUG_LOG
        std::cout << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << " TEST " << test_itr << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << std::endl;

		printGridProp(taskdata.gridProperty, "original");
		printAccessRange(range);
#endif
	}


	const float value = 12.34;
	//Task push
	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
		TaskDataHolder& taskdata = TaskData[test_itr];
		ops::hls::GridPropertyCore adjustedGridProp;
		getRangeAdjustedGridProp(taskdata.gridProperty, taskdata.range, adjustedGridProp, vector_factor);

		int narg = 0;
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,value));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.size[0]));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.size[1]));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.actual_size[0]));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.actual_size[1]));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.grid_size[0]));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.grid_size[1]));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.dim));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.xblocks));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.total_itr));
		OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.outer_loop_limit));

		narg = 0;
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.range.start[0]));
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.range.end[0]));
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.range.start[1]));
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.range.end[1]));
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.gridProperty.grid_size[0]));
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.gridProperty.grid_size[1]));
		OCL_CHECK(err, err = krn_datamover.setArg(narg++, taskdata.cl_buffer));

		OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({taskdata.cl_buffer}, 0, nullptr, &taskdata.fill_event));
		std::vector<cl::Event> events;
		events.push_back(taskdata.fill_event);
		OCL_CHECK(err, err = queue.enqueueTask(krn_datamover, &events, &taskdata.data_event));
		OCL_CHECK(err, err = queue.enqueueTask(krnl_simpleCopy, &events, &taskdata.task_event));

		events.resize(0);
		events.push_back(taskdata.data_event);
		events.push_back(taskdata.task_event);
		//Receive event

		OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({taskdata.cl_buffer}, CL_MIGRATE_MEM_OBJECT_HOST, &events, &taskdata.receive_event));
	}


#ifdef DEBUG_LOG
	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
		TaskDataHolder& taskdata = TaskData[test_itr];
		taskdata.receive_event.wait();

		printGrid2D(taskdata.host_buffer.data(), taskdata.gridProperty, "After finished");

	}
#endif

	queue.finish();

    std::cout << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << " TEST SUMMARY " << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << std::endl;

	for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
	{
        std::cout << "TEST " << test_itr <<": ";

        if (verification(TaskData[test_itr], fill, value))
            std::cout << "PASSED";
        else
            std::cout << "FAILED";

        std::cout << std::endl;
	}

//	ops::hls::GridPropertyCore gridProp, adjustedGridProp;
//	gridProp.size[0] = 12;
//	gridProp.size[1] = 12;
//	gridProp.d_m[0] = 1;
//	gridProp.d_m[1] = 1;
//	gridProp.d_p[0] = 1;
//	gridProp.d_p[1] = 1;
//	gridProp.actual_size[0] = gridProp.size[0] + gridProp.d_p[0] + gridProp.d_m[0];
//	gridProp.actual_size[1] = gridProp.size[1] + gridProp.d_p[1] + gridProp.d_m[1];
//	gridProp.xblocks = (gridProp.actual_size[0] + vector_factor - 1) / vector_factor;
//	gridProp.grid_size[0] = gridProp.xblocks * vector_factor;
//	gridProp.grid_size[1] = gridProp.actual_size[1];
//	gridProp.outer_loop_limit = gridProp.actual_size[1] + 1;
//	gridProp.total_itr = gridProp.actual_size[1] * gridProp.xblocks;
//	gridProp.dim = 2;
//
//	unsigned int data_size_bytes = gridProp.grid_size[0] * gridProp.grid_size[1] * sizeof(float);
//	float * grid_u_d = (float*)aligned_alloc(4096, data_size_bytes);
////	host_buffer_t<float> u(gridProp.grid_size[0] * gridProp.grid_size[1]);
//
////	auto buffer_u = fpga.createDeviceBuffer(CL_MEM_READ_ONLY, u);
//
////	writeGrid(grid_u_d, 2.5, gridProp);
////	printGrid2D(grid_u_d, gridProp);
//
//	OCL_CHECK(err, cl::Buffer buffer_u(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u_d, &err));
//
//
//	static const std::string kernelSimpleCopyName("kernel_simple_copy");
//	static const std::string kernelDatamoverName("kernel_datamover_simple_copy");
//
//	OCL_CHECK(err, cl::Kernel krnl_simpleCopy(program, kernelSimpleCopyName.c_str(), &err));
//	OCL_CHECK(err, cl::Kernel krn_datamover(program, kernelDatamoverName.c_str(), &err));
//
//	int narg = 0;
//	const float value = 2.5;
//
//	printGridProp(gridProp);
//
//
//	ops::hls::AccessRange range;
//	range.dim = 2;
//	range.start[0] = 0;
//	range.start[1] = 0;
//	range.end[0] = gridProp.actual_size[0];
//	range.end[1] = gridProp.actual_size[1];
//
//	getRangeAdjustedGridProp(gridProp, range, adjustedGridProp, vector_factor);
//	printGridProp(adjustedGridProp);
//
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,value));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.size[0]));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.size[1]));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.actual_size[0]));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.actual_size[1]));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.grid_size[0]));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.grid_size[1]));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.dim));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.xblocks));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.total_itr));
//	OCL_CHECK(err, err = krnl_simpleCopy.setArg(narg++,adjustedGridProp.outer_loop_limit));
//
//	narg = 0;
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.start[0]));
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.end[0]));
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.start[1]));
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, range.end[1]));
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, gridProp.grid_size[0]));
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, gridProp.grid_size[1]));
//	OCL_CHECK(err, err = krn_datamover.setArg(narg++, buffer_u));
//
//
//	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_u}, 0));
//
//	queue.finish();
//
//	OCL_CHECK(err, err = queue.enqueueTask(krn_datamover));
//	OCL_CHECK(err, err = queue.enqueueTask(krnl_simpleCopy));
//
//
//	queue.finish();
//
//	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_u}, CL_MIGRATE_MEM_OBJECT_HOST));
//
//	queue.finish();
//
//	printGrid2D(grid_u_d, gridProp);
//	free(grid_u_d);
	return 0;
}
