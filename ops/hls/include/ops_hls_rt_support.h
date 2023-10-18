#pragma once

 #ifndef DOXYGEN_SHOULD_SKIP_THIS 
/** @file
  * @brief OPS xilinx fpga specific runtime support functions
  * @author Beniel Thileepan
  * @details Implements vitis backend runtime support functions
  */

 // This extension file is required for stream APIs
#include <CL/cl_ext_xilinx.h>
// This file is required for OpenCL C++ wrapper APIs
#include "../ext/xcl2/xcl2.hpp"

#include <ap_int.h>
#include <ops_hls_defs.hpp>
#include <ops_hls_fpga.hpp>
#include <ops_hls_kernel.hpp>
#include <ops_hls_host_utils.hpp>

// typedef struct ops_hls_core
// {
//     ops::hls::FPGA* fpga_handle;
// };

// class OPS_instance_hls
// {
// public:
//   ops_hls_core OPS_hls_core;
//   bool isInit;
// };

// #include <ops_lib_core.h>
// #include <ops_device_rt_support.h>

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
