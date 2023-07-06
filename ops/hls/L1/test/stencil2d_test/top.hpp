# pragma once

#include "stencil2d.hpp"

typedef float stencil_type;
constexpr unsigned short num_points = 5; //cross_stencil
constexpr unsigned short vector_factor = 8;
constexpr unsigned short coef_type = ops::hls::CoefTypes::CONST_COEF;
constexpr unsigned short stencil_size_p = 3;
constexpr unsigned short stencil_size_q = 3;

void dut();