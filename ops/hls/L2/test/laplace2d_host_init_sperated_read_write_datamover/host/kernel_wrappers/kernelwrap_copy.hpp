#pragma once
#include <ops_hls_rt_support.h>

//static unsigned short vector_factor = 1;

class kernelwrap_copy : public ops::hls::Kernel
{
public:
    kernelwrap_copy():
        Kernel(),
        m_kernelName("kernel_copy"),
        m_datamoverReadName("datamover_copy_read"),
		m_datamoverWriteName("datamover_copy_write")
    {
        cl_int err;
        OCL_CHECK(err, m_kernel_copy = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
        OCL_CHECK(err, m_datamover_copy_read = cl::Kernel(m_fpga->getProgram(), m_datamoverReadName.c_str(), &err));
        OCL_CHECK(err, m_datamover_copy_write = cl::Kernel(m_fpga->getProgram(), m_datamoverWriteName.c_str(), &err));
    }

    void run(ops::hls::AccessRange& range, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1)
    {
        cl_int err;
        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0.hostBuffer);
        ops::hls::GridPropertyCore arg0_adjustedGridProp;
        getRangeAdjustedGridProp(arg0.originalProperty, range, arg0_adjustedGridProp, vector_factor);
#ifdef DEBUG_LOG
        printGridProp(arg0.originalProperty, "arg0_originalGridProp");
        printGridProp(arg0_adjustedGridProp, "arg0_adjustedGridProp");
#endif

        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg1.hostBuffer);
        ops::hls::GridPropertyCore arg1_adjustedGridProp;
        getRangeAdjustedGridProp(arg1.originalProperty, range, arg1_adjustedGridProp, vector_factor);

#ifdef DEBUG_LOG
        printGridProp(arg1.originalProperty, "arg1_originalGridProp");
        printGridProp(arg1_adjustedGridProp, "arg1_adjustedGridProp");
#endif
        int narg = 0;
      
        OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.size[0]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.size[1]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.actual_size[0]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.actual_size[1]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.dim));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.xblocks));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.total_itr));
		OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, arg0_adjustedGridProp.outer_loop_limit));
        OCL_CHECK(err, err = m_kernel_copy.setArg(narg++, getTotalBytes<stencil_type>(arg0_adjustedGridProp)));

        narg = 0;
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, range.start[0]));
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, range.end[0]));
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, range.start[1]));
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, range.end[1]));
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, arg0.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, arg0.originalProperty.grid_size[1]));
		OCL_CHECK(err, err = m_datamover_copy_read.setArg(narg++, arg0.deviceBuffer));

        narg = 0;
		OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, range.start[0]));
		OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, range.end[0]));
		OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, range.start[1]));
		OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, range.end[1]));
        OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, arg1.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, arg1.originalProperty.grid_size[1]));
        OCL_CHECK(err, err = m_datamover_copy_write.setArg(narg++, arg1.deviceBuffer));

        cl::Event event_kernel;
        cl::Event event_datamover_read, event_datamover_write;

        std::vector<cl::Event> activeEvents(arg0.activeEvents.begin(), arg0.activeEvents.end());
        activeEvents.insert(activeEvents.end(), arg1.activeEvents.begin(), arg1.activeEvents.end());

        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_datamover_copy_write, &arg1.activeEvents, &event_datamover_write));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_copy, &activeEvents, &event_kernel));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_datamover_copy_read, &arg0.activeEvents, &event_datamover_read));

#ifdef DEBUG_LOG
		ops::hls::addEvent(arg0, event_kernel, m_kernelName);
		ops::hls::addEvent(arg0, event_datamover_read, m_datamoverReadName);
        ops::hls::addEvent(arg1, event_kernel, m_kernelName);
		ops::hls::addEvent(arg1, event_datamover_write, m_datamoverWriteName);
#endif

//        arg0.isDevBufDirty = true;
        arg1.isDevBufDirty = true;
		arg0.activeEvents.resize(0);
		arg0.activeEvents.push_back(event_datamover_read);
		arg0.activeEvents.push_back(event_kernel);
        arg1.activeEvents.resize(0);
		arg1.activeEvents.push_back(event_datamover_write);
		arg1.activeEvents.push_back(event_kernel);

		event_datamover_write.wait();
//		std::cout << "copy: datamover wait completed " << std::endl;
//		event_kernel.wait();
//		std::cout << "copy: kernel wait completed " << std::endl;
    }

private:
    std::string m_kernelName;
    std::string m_datamoverReadName;
    std::string m_datamoverWriteName;
    cl::Kernel m_kernel_copy;
    cl::Kernel m_datamover_copy_read;
    cl::Kernel m_datamover_copy_write;
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

