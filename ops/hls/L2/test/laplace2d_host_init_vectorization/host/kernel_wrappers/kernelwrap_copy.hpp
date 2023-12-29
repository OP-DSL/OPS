#pragma once
#include <ops_hls_rt_support.h>

//static unsigned short vector_factor = 1;

class kernelwrap_copy : public ops::hls::Kernel
{
public:
    kernelwrap_copy():
        Kernel(),
        m_kernelName("kernel_copy"),
        m_datamoverName("datamover_copy")
    {
        cl_int err;
        OCL_CHECK(err, m_kernel_copy = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
        OCL_CHECK(err, m_datamover_copy = cl::Kernel(m_fpga->getProgram(), m_datamoverName.c_str(), &err));
    }

    void run(ops::hls::AccessRange& range, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1)
    {
        cl_int err;
//        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0.hostBuffer);
//        ops::hls::GridPropertyCore arg0_adjustedGridProp;
//        ops::hls::StencilConfigCore getStencilConfig(ops::hls::GridPropertyCoreV2& original, ops::hls::AccessRange& range, const unsigned short stencil_vector_factor=8,
//                const unsigned short mem_vector_factor=8, ops::hls::SizeType d_m=default_d_m, ops::hls::SizeType d_p=default_d_p)
        auto arg0_stencilConfig = getStencilConfig(arg0.originalProperty, range, vector_factor);
#ifdef DEBUG_LOG
        printGridProp(arg0.originalProperty, "arg0_originalGridProp");
        printStencilConfig(arg0_adjustedGridProp, "arg0_stencilConfig");
#endif

//        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg1.hostBuffer);
//        ops::hls::GridPropertyCore arg1_adjustedGridProp;
//        getRangeAdjustedGridProp(arg1.originalProperty, range, arg1_adjustedGridProp, vector_factor);

#ifdef DEBUG_LOG
        printGridProp(arg1.originalProperty, "arg1_originalGridProp");
#endif
        int narg = 0;
      
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.lower_limit[0]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.lower_limit[1]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.upper_limit[0]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.upper_limit[1]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.dim));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.outer_loop_limit));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_stencilConfig.total_itr));

        narg = 0;
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, range.start[0]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, range.end[0]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, range.start[1]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, range.end[1]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, arg0.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, arg0.originalProperty.grid_size[1]));
        OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, arg1.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, arg1.originalProperty.grid_size[1]));
		OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, arg0.deviceBuffer));
        OCL_CHECK(err, err = m_datamover_copy.setArg(narg++, arg1.deviceBuffer));

        cl::Event event_kernel;
        cl::Event event_datamover;

        std::vector<cl::Event> activeEvents(arg0.activeEvents.begin(), arg0.activeEvents.end());
        activeEvents.insert(activeEvents.end(), arg1.activeEvents.begin(), arg1.activeEvents.end());

        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_datamover_copy, &activeEvents, &event_datamover));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_copy, &activeEvents, &event_kernel));

#ifdef DEBUG_LOG
		ops::hls::addEvent(arg0, event_kernel, m_kernelName);
		ops::hls::addEvent(arg0, event_datamover, m_datamoverName);
        ops::hls::addEvent(arg1, event_kernel, m_kernelName);
		ops::hls::addEvent(arg1, event_datamover, m_datamoverName);
#endif

//        arg0.isDevBufDirty = true;
        arg1.isDevBufDirty = true;
		arg0.activeEvents.resize(0);
		arg0.activeEvents.push_back(event_datamover);
		arg0.activeEvents.push_back(event_kernel);
        arg1.activeEvents.resize(0);
		arg1.activeEvents.push_back(event_datamover);
		arg1.activeEvents.push_back(event_kernel);

		event_datamover.wait();
//		std::cout << "copy: datamover wait completed " << std::endl;
		event_kernel.wait();
//		std::cout << "copy: kernel wait completed " << std::endl;
    }

private:
    std::string m_kernelName;
    std::string m_datamoverName;
    cl::Kernel m_kernel_copy;
    cl::Kernel m_datamover_copy;
};



void ops_par_loop_copy(int dim , int* ops_range, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1)
{
	static kernelwrap_copy kernelwrap_copy_inst;

	ops::hls::AccessRange range;
	opsRange2hlsRange(dim, ops_range, range, arg0.originalProperty);

	kernelwrap_copy_inst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0);
	kernelwrap_copy_inst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg1);
	sendGrid(arg0);
	sendGrid(arg1);
	kernelwrap_copy_inst.run(range, arg0, arg1);
//	kernelwrap_copy_inst.getGrid(arg1);
}

