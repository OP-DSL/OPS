#pragma once

typedef float stencil_type;
constexpr unsigned int data_width = sizeof(stencil_type) * 8;
constexpr unsigned int vector_factor = 1;
constexpr unsigned short mem_data_width = 32;
constexpr unsigned short shift_bits = 0;
constexpr unsigned short axis_data_width = data_width * vector_factor;

//#define DEBUG_LOG

#ifdef DEBUG_LOG
    #include <stdio.h>
#endif
