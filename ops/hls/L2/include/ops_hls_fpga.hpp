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
  * @brief FPGA handler class headerfile
  * @author Beniel Thileepan (maintainer)
  * @details This class manage FPGA platform interaction with XOCL API and wrapping related objects.
  */

#pragma once
#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <iostream>
#include <vector>
#include <regex>
#include <unordered_map>

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "../../ext/xcl2/xcl2.hpp"

template <typename T>
using host_buffer_t = std::vector<T, aligned_allocator<T> >;

namespace ops
{
namespace hls
{

/**
 * @brief This is a mock ops_block class to replace
 */
class Block
{
public:
	int dims;
	std::string name;
};

/**
 * @brief This is a singleton class indended to use for single FPGA device handlings with 
 * thread local usage. 
 * TODO: This can be further extended to support multi FPGAs.
*/
class FPGA {
   public:

    // Prevent cloning
    FPGA(FPGA &other) = delete;
    // Preventing assignment
    void operator=(const FPGA &) = delete;

    ~FPGA()
    {
    	m_bufferMaps.clear();
    	if (FPGA_)
    		delete(FPGA_);
    }

    static FPGA* getInstance();

//    const uint32_t next() const {
//        if (m_id == m_Devices.size() - 1) {
//            return 0;
//        }
//
//        setID(m_id + 1);
//        return (m_id + 1);
//    }

    void setID(uint32_t id) {
        m_id = id;
        if (m_id >= m_Devices.size()) {
            std::cout << "Device specified by id = " << m_id << " is not found." << std::endl;
            throw;
        }
        m_device = m_Devices[m_id];
    }

    bool xclbin(std::string binaryFile) {
        cl_int err;
        // get_xil_devices() is a utility API which will find the xilinx
        // platforms and will return list of devices connected to Xilinx platform

        // Creating Context
        OCL_CHECK(err, m_context = cl::Context(m_device, NULL, NULL, NULL, &err));

        // Creating Command Queue
        OCL_CHECK(err,
                  m_queue = cl::CommandQueue(m_context, m_device,
                                             CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        // read_binary_file() is a utility API which will load the binaryFile
        // and will return the pointer to file buffer.
        cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

        // Creating Program
        OCL_CHECK(err, m_program = cl::Program(m_context, {m_device}, bins, NULL, &err));
        return true;
    }
    const cl::Context& getContext() const { return m_context; }
    const cl::CommandQueue& getCommandQueue() const { return m_queue; }
    cl::CommandQueue& getCommandQueue() { return m_queue; }

    const cl::Program& getProgram() const { return m_program; }

    void finish() const { m_queue.finish(); }

    template <typename T>
    std::vector<cl::Buffer> createDeviceBuffer(cl_mem_flags p_flags, const std::vector<host_buffer_t<T> >& p_buffer) {
        size_t p_hbm_pc = p_buffer.size();
        std::vector<cl::Buffer> l_buffer(p_hbm_pc);
        for (int i = 0; i < p_hbm_pc; i++) {
            l_buffer[i] = createDeviceBuffer(p_flags, p_buffer[i]);
        }
        return l_buffer;
    }

    template <typename T>
    cl::Buffer createDeviceBuffer(cl_mem_flags p_flags, const host_buffer_t<T>& p_buffer) {
        const void* l_ptr = (const void*)p_buffer.data();
        if (exists(l_ptr)) return m_bufferMaps[l_ptr];

        size_t l_bufferSize = sizeof(T) * p_buffer.size();
        cl_int err;
        m_bufferMaps.insert(
            {l_ptr, cl::Buffer(m_context, p_flags, l_bufferSize, (void*)p_buffer.data(), &err)});
        if (err != CL_SUCCESS) {
            printf("Failed to allocate device buffer!\n");
            throw std::bad_alloc();
        }
        return m_bufferMaps[l_ptr];
    }

   protected:
    bool exists(const void* p_ptr) const {
        auto it = m_bufferMaps.find(p_ptr);
        return it != m_bufferMaps.end();
    }
    void getDevices(std::string deviceName) {
        cl_int err;
        auto devices = xcl::get_xil_devices();
        auto regexStr = std::regex(".*" + deviceName + ".*");
        for (auto device : devices) {
            std::string cl_device_name;
            OCL_CHECK(err, err = device.getInfo(CL_DEVICE_NAME, &cl_device_name));
            if (regex_match(cl_device_name, regexStr)) m_Devices.push_back(device);
        }
        if (0 == m_Devices.size()) {
            std::cout << "Device specified by name == " << deviceName << " is not found." << std::endl;
            throw;
        }
    }

    FPGA(std::string deviceName) {
        getDevices(deviceName);
        m_device = m_Devices[m_id];
        m_id = -1;
    }
    FPGA(unsigned int p_id = 0, std::string deviceName = "") {
        getDevices(deviceName);
        setID(p_id);
    }

    FPGA(unsigned int p_id, const std::vector<cl::Device>& devices) {
        m_id = p_id;
        m_Devices = devices;
        m_device = m_Devices[m_id];
    }

    static FPGA* FPGA_;

   private:
    unsigned int m_id;
    cl::Device m_device;
    std::vector<cl::Device> m_Devices;
    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Program m_program;
    std::unordered_map<const void*, cl::Buffer> m_bufferMaps;
};

}
}

void ops_init_backend(int argc, const char** argv, unsigned int devId = 0);
void ops_exit_backend();

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
