
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
#include <unordered_map>
#include <ops_hls_fpga.hpp>
// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"

namespace ops
{
namespace hls
{

class Kernel {
   public:
    Kernel(FPGA* p_fpga = nullptr) : m_fpga(p_fpga) {}

    void fpga(FPGA* p_fpga) { m_fpga = p_fpga; }

    void getCU(const string& p_name) {
        cl_int err;
        OCL_CHECK(err, m_kernel = cl::Kernel(m_fpga->getProgram(), p_name.c_str(), &err));
    }

    const cl::Kernel& operator()() const { return m_kernel; }
    cl::Kernel& operator()() { return m_kernel; }

    void enqueueTask() const {
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel));
    }

    void finish() const { m_fpga->finish(); }

    static double run(const vector<Kernel>& p_kernels) {
        auto start = chrono::high_resolution_clock::now();
        for (auto ker : p_kernels) ker.enqueueTask();
        for (auto ker : p_kernels) ker.finish();
        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = finish - start;
        double t_sec = elapsed.count();
        return t_sec;
    }

    void getBuffer(vector<cl::Memory>& h_m) {
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(h_m, CL_MIGRATE_MEM_OBJECT_HOST));
        finish();
    }

    void sendBuffer(vector<cl::Memory>& h_m) {
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(h_m, 0)); /* 0 means from host*/
        finish();
    }

    template <typename T>
    cl::Buffer createDeviceBuffer(cl_mem_flags p_flags, const host_buffer_t<T>& p_buffer) const {
        return m_fpga->createDeviceBuffer(p_flags, p_buffer);
    }

    template <typename T>
    vector<cl::Buffer> createDeviceBuffer(cl_mem_flags p_flags, const vector<host_buffer_t<T> >& p_buffer) const {
        return m_fpga->createDeviceBuffer(p_flags, p_buffer);
    }

    FPGA* m_fpga;
    cl::Kernel m_kernel;
};

}
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */