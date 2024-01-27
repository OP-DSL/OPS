#pragma once
#include <ops_hls_rt_support.h>
#define DEBUG_LOG
//static unsigned short vector_factor = 1;

class kernelwrap_outerloop_1 : public ops::hls::Kernel
{
public:
    kernelwrap_outerloop_1():
        Kernel(),
        m_kernelName("kernel_outerloop_1"),
        m_datamoverName("datamover_outerloop_1")
    {
        cl_int err;
        OCL_CHECK(err, m_kernel_outerloop_1_1 = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
#ifdef MULTI_SLR
		OCL_CHECK(err, m_kernel_outerloop_1_2 = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
		OCL_CHECK(err, m_kernel_outerloop_1_3 = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
#endif
        OCL_CHECK(err, m_datamover_outerloop_1 = cl::Kernel(m_fpga->getProgram(), m_datamoverName.c_str(), &err));
    }

    void run(ops::hls::AccessRange& range, unsigned int outer_iter, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1)
    {
        cl_int err;
		ops::hls::SizeType s2d_5pt_d_m={1,1,0};
		ops::hls::SizeType s2d_5pt_d_p={1,1,0};
		ops::hls::SizeType s2d_1pt_d_m={0,0,0};
		ops::hls::SizeType s2d_1pt_d_p={0,0,0};

		auto s2d_5pt_stencilConfig = getStencilConfig(arg0.originalProperty, range, vector_factor, mem_vector_factor, s2d_5pt_d_m, s2d_5pt_d_p);
       	ops::hls::AccessRange read_range;
       	getAdjustedRange(arg0.originalProperty, range, read_range, s2d_5pt_d_m, s2d_5pt_d_p);

#ifdef DEBUG_LOG
        printAccessRange(range, "common access range");
        printGridProp(arg0.originalProperty, "arg0_originalGridProp");
        printStencilConfig(s2d_5pt_stencilConfig, "s2d_5pt_stencilConfig");
#endif

#ifndef MULTI_SLR
		unsigned int adjusted_outer_iter = (outer_iter + iter_par_factor - 1) / iter_par_factor;
#else
		unsigned int total_iter_par_factor = iter_par_factor * 3;
		unsigned int adjusted_outer_iter = (outer_iter + total_iter_par_factor - 1) / total_iter_par_factor;
#endif
//#ifdef DEBUG_LOG
//        printGridProp(arg1.originalProperty, "arg1_originalGridProp");
//        printStencilConfig(arg1_stencilConfig, "arg1_stencilConfig");
//#endif

        int narg = 0;
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, adjusted_outer_iter));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.dim));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.total_itr));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.lower_limit[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.lower_limit[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.upper_limit[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.upper_limit[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_1.setArg(narg++, s2d_5pt_stencilConfig.outer_loop_limit));

#ifdef MULTI_SLR
		narg = 0;	
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, adjusted_outer_iter));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.dim));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.total_itr));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.lower_limit[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.lower_limit[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.upper_limit[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.upper_limit[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_2.setArg(narg++, s2d_5pt_stencilConfig.outer_loop_limit));

		narg = 0;
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, adjusted_outer_iter));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.dim));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.total_itr));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.lower_limit[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.lower_limit[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.upper_limit[0]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.upper_limit[1]));
		OCL_CHECK(err, err = m_kernel_outerloop_1_3.setArg(narg++, s2d_5pt_stencilConfig.outer_loop_limit));
#endif

        narg = 0;
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, read_range.start[0]));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, read_range.end[0]));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, read_range.start[1]));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, read_range.end[1]));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, arg0.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, arg0.originalProperty.grid_size[1]));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, adjusted_outer_iter));
		OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, arg0.deviceBuffer));
        OCL_CHECK(err, err = m_datamover_outerloop_1.setArg(narg++, arg1.deviceBuffer));

        cl::Event event_kernel_1;
#ifdef MULTI_SLR
		cl::Event event_kernel_2;
		cl::Event event_kernel_3;
#endif
        cl::Event event_datamover;

        std::vector<cl::Event> activeEvents(arg0.activeEvents.begin(), arg0.activeEvents.end());
        activeEvents.insert(activeEvents.end(), arg1.activeEvents.begin(), arg1.activeEvents.end());

		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_outerloop_1_1, &activeEvents, &event_kernel_1));
#ifdef MULTI_SLR
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_outerloop_1_2, &activeEvents, &event_kernel_2));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_outerloop_1_3, &activeEvents, &event_kernel_3));
#endif
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_datamover_outerloop_1, &activeEvents, &event_datamover));

#ifdef DEBUG_LOG
		ops::hls::addEvent(arg0, event_kernel_1, m_kernelName);
		ops::hls::addEvent(arg0, event_datamover, m_datamoverName);
		ops::hls::addEvent(arg1, event_kernel_1, m_kernelName);
		ops::hls::addEvent(arg1, event_datamover, m_datamoverName);
	#ifdef MULTI_SLR
		ops::hls::addEvent(arg0, event_kernel_2, m_kernelName + std::string("_2"));
		ops::hls::addEvent(arg0, event_kernel_3, m_kernelName + std::string("_3"));
		ops::hls::addEvent(arg1, event_kernel_2, m_kernelName + std::string("_2"));
		ops::hls::addEvent(arg1, event_kernel_3, m_kernelName + std::string("_3"));
	#endif
#endif


        arg0.isDevBufDirty = true;
        arg1.isDevBufDirty = true;
		arg0.activeEvents.resize(0);
		arg0.activeEvents.push_back(event_datamover);
		arg0.activeEvents.push_back(event_kernel_1);
        arg1.activeEvents.resize(0);
		arg1.activeEvents.push_back(event_datamover);
		arg1.activeEvents.push_back(event_kernel_1);

#ifdef MULTI_SLR
		arg0.activeEvents.push_back(event_kernel_2);
		arg0.activeEvents.push_back(event_kernel_3);
		arg1.activeEvents.push_back(event_kernel_2);
		arg1.activeEvents.push_back(event_kernel_3);
#endif
		event_datamover.wait();
//		std::cout << "datamover wait completed " << std::endl;
		event_kernel_1.wait();
#ifdef MULTI_SLR
		event_kernel_2.wait();
		event_kernel_3.wait();
#endif
//		std::cout << "kernel wait completed " << std::endl;

		activeEvents.resize(0);
		activeEvents.insert(activeEvents.end(), arg0.activeEvents.begin(), arg0.activeEvents.end());

        cl::Event event_bufCpy;
        size_t total_bytes = arg0.originalProperty.grid_size[0] * arg0.originalProperty.grid_size[1] * arg0.originalProperty.grid_size[2] * sizeof(stencil_type);

        if (adjusted_outer_iter %2 == 0)
        {
        	OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueCopyBuffer(arg0.deviceBuffer, arg1.deviceBuffer, 0, 0, total_bytes, &activeEvents, &event_bufCpy));
        }
        else
        	OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueCopyBuffer(arg1.deviceBuffer, arg0.deviceBuffer, 0, 0, total_bytes, &activeEvents, &event_bufCpy));

        event_bufCpy.wait();
    }

private:
    std::string m_kernelName;
    std::string m_datamoverName;
    cl::Kernel m_kernel_outerloop_1_1;
#ifdef MULTI_SLR
	cl::Kernel m_kernel_outerloop_1_2;
	cl::Kernel m_kernel_outerloop_1_3;
#endif
    cl::Kernel m_datamover_outerloop_1;
};


void ops_itr_par_loop_outerloop_1(int dim , int* ops_range, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1, int outer_iter, bool is_copy, int* copy_map)
{
	static kernelwrap_outerloop_1 kernelwrap_outerloop_1_inst;

	ops::hls::AccessRange range;
	opsRange2hlsRange(dim, ops_range, range, arg0.originalProperty);

//	kernelwrap_apply_stencil_inst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0);
//	kernelwrap_apply_stencil_inst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg1);
	sendGrid(arg0);
//	sendGrid(arg1);
	kernelwrap_outerloop_1_inst.run(range, outer_iter, arg0, arg1);
//	kernelwrap_apply_stencil_inst.getGrid(arg1);
}

