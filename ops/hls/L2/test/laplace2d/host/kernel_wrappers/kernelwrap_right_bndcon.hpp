#pragma once
#include <ops_hls_rt_support.h>

class kernelwrap_right_bndcon : public ops::hls::Kernel
{
public:
    kernelwrap_right_bndcon():
        Kernel(),
        m_kernelName("kernel_right_bndcon"),
        m_datamoverName("datamover_right_bndcon")
    {
        cl_int err;
        OCL_CHECK(err, m_kernel_right_bndcon = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
        OCL_CHECK(err, m_datamover_right_bndcon = cl::Kernel(m_fpga->getProgram(), m_datamoverName.c_str(), &err));
    }

    void run(ops::hls::AccessRange& range, ops::hls::Grid<float>& arg0, const float const0, const int const1)
    {
        cl_int err;
        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0.hostBuffer);
        ops::hls::GridPropertyCore arg0_adjustedGridProp;
        getRangeAdjustedGridProp(arg0.originalProperty, range, arg0_adjustedGridProp, vector_factor);
#ifdef DEBUG_LOG
        printGridProp(arg0_adjustedGridProp, "arg0_adjustedGridProp");
#endif

        int narg = 0;
      
        OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.size[0]));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.size[1]));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.actual_size[0]));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.actual_size[1]));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.dim));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.xblocks));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.total_itr));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, arg0_adjustedGridProp.outer_loop_limit));
		OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, const0));
        OCL_CHECK(err, err = m_kernel_right_bndcon.setArg(narg++, const1));
        narg = 0;
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, range.start[0]));
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, range.end[0]));
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, range.start[1]));
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, range.end[1]));
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, arg0.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, arg0.originalProperty.grid_size[1]));
		OCL_CHECK(err, err = m_datamover_right_bndcon.setArg(narg++, arg0.deviceBuffer));

        cl::Event event_kernel;
        cl::Event event_datamover;

        std::vector<cl::Event> activeEvents(arg0.activeEvents.begin(), arg0.activeEvents.end());

        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_datamover_right_bndcon, &activeEvents, &event_kernel));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_right_bndcon, &activeEvents, &event_datamover));

#ifdef DEBUG_LOG
		ops::hls::addEvent(arg0, event_kernel, m_kernelName);
		ops::hls::addEvent(arg0, event_datamover, m_datamoverName);
#endif

		arg0.isDevBufDirty = true;
		arg0.activeEvents.resize(0);
		arg0.activeEvents.push_back(event_datamover);
		arg0.activeEvents.push_back(event_kernel);

//		event_datamover.wait();
//		event_kernel.wait();
    }

private:
    std::string m_kernelName;
    std::string m_datamoverName;
    cl::Kernel m_kernel_right_bndcon;
    cl::Kernel m_datamover_right_bndcon;
};

void ops_par_loop_right_bndcon(int dim , int* ops_range, ops::hls::Grid<float>& arg0)
{
	static kernelwrap_right_bndcon kernelwrap_right_bndconInst;

	ops::hls::AccessRange range;
	opsRange2hlsRange(dim, ops_range, range, arg0.originalProperty);

	kernelwrap_right_bndconInst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0);
	kernelwrap_right_bndconInst.run(range, arg0, pi, jmax);
//	kernelwrap_right_bndconInst.getGrid(arg0);
}
