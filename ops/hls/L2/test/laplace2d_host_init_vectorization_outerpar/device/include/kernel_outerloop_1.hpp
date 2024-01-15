#include "PE_apply_stencil.hpp"
//#include "PE_copy.hpp"

extern "C" void kernel_outerloop_1(
        const unsigned int outer_itr,
        const unsigned short stencilConfig_grid_size_x,
        const unsigned short stencilConfig_grid_size_y,
        const unsigned short stencilConfig_dim,
        const unsigned int stencilConfig_total_itr,
        const unsigned short stencilConfig_lower_limit_x,
        const unsigned short stencilConfig_lower_limit_y,
        const unsigned short stencilConfig_upper_limit_x,
        const unsigned short stencilConfig_upper_limit_y,
        const unsigned short stencilConfig_outer_loop_limit,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out);
