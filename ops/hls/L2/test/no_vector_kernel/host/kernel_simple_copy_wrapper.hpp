#include <ops_hls_kernel.hpp>
#include <ops_hls_defs.hpp>

#include <ops_hls_host_utils.hpp>


class SimpleCopyWrapper : public ops::hls::Kernel
{
public:
//	SimpleCopyWrapper(ops::hls::FPGA* fpga) :
//		Kernel(fpga),
//		m_kernelName("kernel_simple_copy"),
//		m_datamoverKernelName("kernel_datamover_simple_copy")
//	{
//		cl_int err;
//		OCL_CHECK(err, m_krnl_simpleCopy = cl::Kernel(fpga->getProgram(), m_kernelName.c_str(), &err));
//		OCL_CHECK(err, m_krnl_datamover = cl::Kernel(fpga->getProgram(), m_datamoverKernelName.c_str(), &err));
//	}

	SimpleCopyWrapper() :
		Kernel(),
		m_kernelName("kernel_simple_copy"),
		m_datamoverKernelName("kernel_datamover_simple_copy")
	{
		cl_int err;
		OCL_CHECK(err, m_krnl_simpleCopy = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
		OCL_CHECK(err, m_krnl_datamover = cl::Kernel(m_fpga->getProgram(), m_datamoverKernelName.c_str(), &err));
	}

	void run(ops::hls::AccessRange& range, ops::hls::Grid<float>& grid, const float fillValue)
	{
		cl_int err;
		createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, grid.hostBuffer);
		ops::hls::GridPropertyCore adjustedGridProperty;
		getRangeAdjustedGridProp(grid.originalProperty, range, adjustedGridProperty, 1);

#ifdef DEBUG_LOG
		printGridProp(adjustedGridProperty, "Adjusted grid prop");
#endif
		int narg = 0;
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, fillValue));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.size[0]));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.size[1]));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.actual_size[0]));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.actual_size[1]));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.grid_size[0]));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.grid_size[1]));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.dim));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.xblocks));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.total_itr));
		OCL_CHECK(err, err = m_krnl_simpleCopy.setArg(narg++, adjustedGridProperty.outer_loop_limit));

		narg = 0;
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, range.start[0]));
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, range.end[0]));
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, range.start[1]));
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, range.end[1]));
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, grid.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, grid.originalProperty.grid_size[1]));
		OCL_CHECK(err, err = m_krnl_datamover.setArg(narg++, grid.deviceBuffer));

		cl::Event event_simpleCopy;
		cl::Event event_datamover;

		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_krnl_datamover, &grid.activeEvents, &event_simpleCopy));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_krnl_simpleCopy, &grid.activeEvents, &event_datamover));

		ops::hls::addEvent(grid, event_simpleCopy, m_kernelName);
		ops::hls::addEvent(grid, event_datamover, m_datamoverKernelName);

		grid.activeEvents.resize(0);
		grid.activeEvents.push_back(event_datamover);
		grid.activeEvents.push_back(event_simpleCopy);
	}

private:

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

	const std::string m_kernelName;
	const std::string m_datamoverKernelName;
	cl::Kernel m_krnl_simpleCopy;
	cl::Kernel m_krnl_datamover;
};
