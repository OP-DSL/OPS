
#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <ops_hls_rt_support.h>

//#define DEBUG_LOG

extern int imax, jmax;
extern float pi;
unsigned short vector_factor = 1;

void ops_init_backend(int argc, const char** argv)
{
    std::string xclbinFile = argv[1];

    unsigned int deviceId = 0;

    ops::hls::FPGA * fpga = ops::hls::FPGA::getInstance();
    fpga->setID(deviceId);

    if(!fpga->xclbin(xclbinFile))
    {
        std::cerr << "[ERROR] Couldn't program fpga. exit" << std::endl;
		throw;
    }
}

#include "kernelwrap_copy.hpp"
#include "kernelwrap_left_bndcon.hpp"
#include "kernelwrap_right_bndcon.hpp"
#include "kernelwrap_set_zero.hpp"
