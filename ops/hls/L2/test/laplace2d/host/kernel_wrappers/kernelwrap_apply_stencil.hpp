#pragma once
#include <ops_hls_rt_support.h>

//static unsigned short vector_factor = 1;
static ops::hls::SizeType s2d_5pt_d_m={1,1,0};
static ops::hls::SizeType s2d_5pt_d_p={1,1,0};
static ops::hls::SizeType s2d_1pt_d_m={0,0,0};
static ops::hls::SizeType s2d_1pt_d_p={0,0,0};

class kernelwrap_apply_stencil : public ops::hls::Kernel
{
public:
    kernelwrap_apply_stencil():
        Kernel(),
        m_kernelName("kernel_apply_stencil"),
        m_datamoverName("datamover_apply_stencil")
    {
        cl_int err;
        OCL_CHECK(err, m_kernel_apply_stencil = cl::Kernel(m_fpga->getProgram(), m_kernelName.c_str(), &err));
        OCL_CHECK(err, m_datamover_apply_stencil = cl::Kernel(m_fpga->getProgram(), m_datamoverName.c_str(), &err));
    }

    void run(ops::hls::AccessRange& range, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1)
    {
        cl_int err;
        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0.hostBuffer);
        ops::hls::GridPropertyCore arg0_adjustedGridProp;
        ops::hls::AccessRange read_range;
        getRangeAdjustedGridProp(arg0.originalProperty, range, arg0_adjustedGridProp, vector_factor, s2d_5pt_d_m, s2d_5pt_d_p);
        getAdjustedRange(arg0.originalProperty, range, read_range, s2d_5pt_d_m, s2d_5pt_d_p);

#ifdef DEBUG_LOG
        printGridProp(arg0.originalProperty, "arg0_originalGridProp");
        printGridProp(arg0_adjustedGridProp, "arg0_adjustedGridProp");
#endif

        createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg1.hostBuffer);
        ops::hls::GridPropertyCore arg1_adjustedGridProp;
        ops::hls::AccessRange write_range;
        getRangeAdjustedGridProp(arg1.originalProperty, range, arg1_adjustedGridProp, vector_factor, s2d_1pt_d_m, s2d_1pt_d_p);
        getAdjustedRange(arg1.originalProperty, range, write_range, s2d_1pt_d_m, s2d_1pt_d_p);
//        write_range.start[0] = read_range.start[0];
//        write_range.start[1] = read_range.start[1];
//        write_range.start[2] = read_range.start[2];
//        write_range.dim = read_range.dim;
//        write_range.end[0] = read_range.end[0];
//        write_range.end[1] = read_range.end[1] - 1;
//        write_range.end[2] = read_range.end[2];

#ifdef DEBUG_LOG
        printGridProp(arg1.originalProperty, "arg1_originalGridProp");
        printGridProp(arg1_adjustedGridProp, "arg1_adjustedGridProp");
#endif

        int narg = 0;
        OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.size[0]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.size[1]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.actual_size[0]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.actual_size[1]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.dim));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.xblocks));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.total_itr));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg0_adjustedGridProp.outer_loop_limit));
        OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.size[0]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.size[1]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.actual_size[0]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.actual_size[1]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.grid_size[0]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.grid_size[1]));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.dim));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.xblocks));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.total_itr));
		OCL_CHECK(err, err = m_kernel_apply_stencil.setArg(narg++, arg1_adjustedGridProp.outer_loop_limit));

        narg = 0;
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, read_range.start[0]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, read_range.end[0]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, read_range.start[1]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, read_range.end[1]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, write_range.start[0]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, write_range.end[0]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, write_range.start[1]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, write_range.end[1]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, arg0.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, arg0.originalProperty.grid_size[1]));
        OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, arg1.originalProperty.grid_size[0]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, arg1.originalProperty.grid_size[1]));
		OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, arg0.deviceBuffer));
        OCL_CHECK(err, err = m_datamover_apply_stencil.setArg(narg++, arg1.deviceBuffer));

        cl::Event event_kernel;
        cl::Event event_datamover;

        std::vector<cl::Event> activeEvents(arg0.activeEvents.begin(), arg0.activeEvents.end());
        activeEvents.insert(activeEvents.end(), arg1.activeEvents.begin(), arg1.activeEvents.end());

        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_datamover_apply_stencil, &activeEvents, &event_datamover));
		OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel_apply_stencil, &activeEvents, &event_kernel));

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
//		std::cout << "datamover wait completed " << std::endl;
		event_kernel.wait();
//		std::cout << "kernel wait completed " << std::endl;
    }

private:
    std::string m_kernelName;
    std::string m_datamoverName;
    cl::Kernel m_kernel_apply_stencil;
    cl::Kernel m_datamover_apply_stencil;
};



void ops_par_loop_apply_stencil(int dim , int* ops_range, ops::hls::Grid<float>& arg0, ops::hls::Grid<float>& arg1)
{
	static kernelwrap_apply_stencil kernelwrap_apply_stencil_inst;

	ops::hls::AccessRange range;
	opsRange2hlsRange(dim, ops_range, range, arg0.originalProperty);

	kernelwrap_apply_stencil_inst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg0);
	kernelwrap_apply_stencil_inst.createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arg1);
	sendGrid(arg0);
	kernelwrap_apply_stencil_inst.run(range, arg0, arg1);
//	kernelwrap_apply_stencil_inst.getGrid(arg1);
}

