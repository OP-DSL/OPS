# pragma once

#include "stencil2d.hpp"
#include <iostream>

void dut(ops::hls::GridPropertyCore& gridProp, Stencil2D & cross_stencil, stencil_type* data_in, stencil_type* data_out);
