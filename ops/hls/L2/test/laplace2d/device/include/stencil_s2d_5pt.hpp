#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>

static constexpr unsigned short num_points = 5;
static constexpr unsigned short stencil_size = 3;
static constexpr unsigned short stencil_dim = 2;

// class s2d_5pt : public ops::hls::StencilCore<stencil_type, num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF,
//     stencil_size
