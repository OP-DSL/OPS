#pragma once

 #ifndef DOXYGEN_SHOULD_SKIP_THIS 
/** @file
  * @brief OPS xilinx fpga kernel build include
  * @author Beniel Thileepan
  * @details Implements vitis kernel build include as unified collection of all
  *     include files in the L1 library. 
  */

#include <ap_int.h>
#include "../../L1/include/ops_hls_datamover.hpp"
#include "../../L1/include/ops_hls_defs.hpp"
#include "../../L1/include/ops_hls_stencil_core_v2.hpp"
#include "../../L1/include/ops_hls_utils.hpp"
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
