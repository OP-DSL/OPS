
/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

/** @file
  * @brief kernel handle class
  * @author Beniel Thileepan (maintainer)
  * @details Abstract class definition to manage kernel with XOCL api as a utility 
  * to support host side and device side communication.
  */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#pragma once

#include <iostream>
#include <vector>
#include <regex>
#include <utility>
#include <string>
#include <cassert>
#include <unordered_map>
#include <ops_hls_defs.hpp>
#include <ops_hls_fpga.hpp>
// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"

namespace ops
{
namespace hls
{

template <typename T>
struct Grid
{
	GridPropertyCore originalProperty;
	unsigned short vector_factor = 8; //grid vector factor considered for adjustements
	host_buffer_t<T> hostBuffer;
	cl::Buffer deviceBuffer;
	std::vector<cl::Event> activeEvents;
	std::vector<std::pair<cl::Event, std::string>> allEvents;
	bool isHostBufDirty;
	bool isDevBufDirty;
};

template <typename T>
void addEvent(Grid<T>& p_grid, cl::Event& p_event, std::string prompt="")
{
	p_grid.allEvents.push_back(std::make_pair(p_event, prompt +
			std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count())));
}

template <typename T>
cl::Event& emplaceEvent(Grid<T>& p_grid, std::string prompt="")
{
	p_grid.allEvents.emplace(p_grid.allEvents.end());
	p_grid.allEvents[p_grid.allEvents.size()-1].second =  prompt +
			std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	return p_grid.allEvents[p_grid.allEvents.size()-1].first;
}

template <typename T>
void getGrid(Grid<T>& p_grid)
{
	if (p_grid.isDevBufDirty)
	{
		cl_int err;
		cl::Event event;
		OCL_CHECK(err, err = FPGA::getInstance()->getCommandQueue().enqueueMigrateMemObjects({p_grid.deviceBuffer}, CL_MIGRATE_MEM_OBJECT_HOST, &p_grid.activeEvents, &event));
		addEvent(p_grid, event, __func__);
		p_grid.activeEvents.resize(0);
		p_grid.activeEvents.push_back(event);
		p_grid.isDevBufDirty = false;
		p_grid.isHostBufDirty = false;
	}
}

template <typename T>
void sendGrid(Grid<T>& p_grid)
{
	if (p_grid.isHostBufDirty)
	{
		cl_int err;
		cl::Event event;
		OCL_CHECK(err, err = FPGA::getInstance()->getCommandQueue().enqueueMigrateMemObjects({p_grid.deviceBuffer}, 0, &p_grid.activeEvents, &event));
		addEvent(p_grid, event, __func__);
		p_grid.activeEvents.resize(0);
		p_grid.activeEvents.push_back(event);
		p_grid.isDevBufDirty = false;
		p_grid.isHostBufDirty = false;
	}
}


class Kernel
{
	public:
//		Kernel(FPGA* p_fpga = nullptr) : m_fpga(p_fpga) {}

		Kernel()
		{
			m_fpga = FPGA::getInstance();
		}

		void fpga(FPGA* p_fpga) { m_fpga = p_fpga; }

		// void getCU(const std::string& p_name)
		// {
		// 	cl_int err;
		// 	OCL_CHECK(err, m_kernel = cl::Kernel(m_fpga->getProgram(), p_name.c_str(), &err));
		// }

		// const cl::Kernel& operator()() const { return m_kernel; }
		// cl::Kernel& operator()() { return m_kernel; }

	//    void enqueueTask() const
	//    {
	//        cl_int err;
	//        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel));
	//    }

		void finish() const { m_fpga->finish(); }

		void run(){};

		void getBuffer(std::vector<cl::Memory>& h_m)
		{
			cl_int err;
			OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(h_m, CL_MIGRATE_MEM_OBJECT_HOST));
			finish();
		}

//		template <typename T>
//		void getGrid(Grid<T>& p_grid)
//		{
//			cl_int err;
//			cl::Event event;
//			OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({p_grid.deviceBuffer}, CL_MIGRATE_MEM_OBJECT_HOST, &p_grid.activeEvents, &event));
//			addEvent(p_grid, event, __func__);
//			p_grid.activeEvents.resize(0);
//			p_grid.activeEvents.push_back(event);
//		}

		void sendBuffer(std::vector<cl::Memory>& h_m)
		{
			cl_int err;
			OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(h_m, 0)); /* 0 means from host*/
			finish();
		}

//		template <typename T>
//		void sendGrid(Grid<T>& p_grid)
//		{
//			cl_int err;
//			cl::Event event;
//			OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({p_grid.deviceBuffer}, 0, &p_grid.activeEvents, &event));
//			addEvent(p_grid, event, __func__);
//			p_grid.activeEvents.resize(0);
//			p_grid.activeEvents.push_back(event);
//		}

		template <typename T>
		void sendGrid(std::vector<Grid<T>>& p_grid_vect)
		{
			for (auto p_grid : p_grid_vect)
			{
				sendGrid<T>(*p_grid);
			}
		}

		template <typename T>
		cl::Buffer createDeviceBuffer(cl_mem_flags p_flags, const host_buffer_t<T>& p_buffer) const
		{
			return m_fpga->createDeviceBuffer(p_flags, p_buffer);
		}

		template <typename T>
		void createDeviceBuffer(cl_mem_flags p_flags, Grid<T>& p_grid) const
		{
			p_grid.deviceBuffer = m_fpga->createDeviceBuffer(p_flags, p_grid.hostBuffer);
		}

		template <typename T>
		std::vector<cl::Buffer> createDeviceBuffer(cl_mem_flags p_flags, const std::vector<host_buffer_t<T> >& p_buffer) const
		{
			return m_fpga->createDeviceBuffer(p_flags, p_buffer);
		}

		template <typename T>
		void createDeviceBuffer(cl_mem_flags p_flags, std::vector<host_buffer_t<T> >& p_grid) const
		{
			for (auto it = p_grid.begin(); it = p_grid.end(); ++it)
			{
				it.deviceBuffer =  m_fpga->createDeviceBuffer(p_flags, it.hostBuffer);
			}
		}

		FPGA* m_fpga;
		cl::Kernel m_kernel;
};

}
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
