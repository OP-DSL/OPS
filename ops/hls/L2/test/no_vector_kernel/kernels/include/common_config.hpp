#pragma once

#include <ops_hls_defs.hpp>

typedef float stencil_type;
constexpr unsigned int data_width = sizeof(stencil_type) * 8;
constexpr unsigned int mem_data_width = 32;
constexpr unsigned short vector_factor = 1;
constexpr unsigned short shift_bits = 0;
constexpr unsigned int axis_data_width = data_width * vector_factor;
constexpr unsigned short slr_p_stages = 10;

