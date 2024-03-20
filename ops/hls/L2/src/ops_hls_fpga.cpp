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

#include <ops_hls_fpga.hpp>

ops::hls::FPGA* ops::hls::FPGA::FPGA_ = nullptr;

ops::hls::FPGA* ops::hls::FPGA::getInstance()
{
    if (FPGA_ == nullptr)
    {
        FPGA_ = new FPGA();
    }
    return FPGA_;
}

void ops_init_backend(int argc, const char** argv, unsigned int devId)
{
    std::string xclbinFile = argv[1];

    unsigned int deviceId = devId;

    ops::hls::FPGA * fpga = ops::hls::FPGA::getInstance();
    fpga->setID(deviceId);

    if(!fpga->xclbin(xclbinFile))
    {
        std::cerr << "[ERROR] Couldn't program fpga. exit" << std::endl;
		throw;
    }
}

template<typename _Period>
double ops_hls_get_execution_runtime(const std::string& kernel_name)
{
	ops::hls::FPGA * fpga = ops::hls::FPGA::getInstance();
	return fpga->getExecutionRuntime<_Period>(kernel_name);
}


template<typename _Period>
double ops_hls_get_execution_runtime(const char* kernel_name)
{
	auto name = std::string(kernel_name);
	return ops_hls_get_execution_runtime<_Period>(name);
}

void ops_exit_backend()
{
	  auto fpga_inst = ops::hls::FPGA::getInstance();
	  fpga_inst->finish();
}

template double ops_hls_get_execution_runtime<std::chrono::microseconds>(const std::string& kernel_name);
template double ops_hls_get_execution_runtime<std::chrono::nanoseconds>(const std::string& kernel_name);
template double ops_hls_get_execution_runtime<std::chrono::milliseconds>(const std::string& kernel_name);
template double ops_hls_get_execution_runtime<std::chrono::seconds>(const std::string& kernel_name);
