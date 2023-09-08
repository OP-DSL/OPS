#pragma once

#include <ops_hls_defs.hpp>

typedef float stencil_type;
constexpr unsigned int data_width = sizeof(stencil_type) * 8;
constexpr unsigned int mem_data_width = 512;
constexpr unsigned short vector_factor = 8;
constexpr unsigned short shift_bits = 3;
constexpr unsigned int axis_data_width = data_width * vector_factor;
constexpr unsigned short slr_p_stages = 10;

constexpr unsigned int max_axi_depth = max_grid_size * max_grid_size * max_grid_size;

