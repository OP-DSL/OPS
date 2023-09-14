
#include <kernel_simple_copy.hpp>
#include <ops_hls_datamover.hpp>
#include <stdio.h>

// struct GridPropertyCore
// {
//     SizeType size;
//     SizeType d_p;
//     SizeType d_m;
//     SizeType grid_size;
//     SizeType actual_size;
//     SizeType offset;
//     unsigned short dim;
//     unsigned short xblocks;
//     unsigned int total_itr;
//     unsigned int outer_loop_limit;
// };
extern "C" void kernel_simple_copy(
    const float const_val,
    const unsigned short gridProp_size_x,
    const unsigned short gridProp_size_y,
    const unsigned short gridProp_actual_size_x,
    const unsigned short gridProp_actual_size_y,
    const unsigned short gridProp_grid_size_x,
    const unsigned short gridProp_grid_size_y,
    const unsigned short gridProp_dim,
    const unsigned short gridProp_xblocks,
    const unsigned int gridProp_total_itr,
    const unsigned int gridProp_outer_loop_limit,
	hls::stream <ap_axiu<axis_data_width,0,0,0>> &axis_out_u
)
{
    #pragma HLS INTERFACE s_axilite port = const_val bundle = control

    #pragma HLS INTERFACE s_axilite port = gridProp_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_size_y bundle = control 
    #pragma HLS INTERFACE s_axilite port = gridProp_actual_size_x bundle = control 
    #pragma HLS INTERFACE s_axilite port = gridProp_actual_size_y bundle = control 
    #pragma HLS INTERFACE s_axilite port = gridProp_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_xblocks bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_outer_loop_limit bundle = control

//	#pragma HLS DISAGGREGATE variable = range
//	#pragma HLS INTERFACE s_axilite port = range->start bundle = control
//	#pragma HLS INTERFACE s_axilite port = range->end bundle = control
//	#pragma HLS INTERFACE s_axilite port = range->dim bundle = control

	#pragma HLS INTERFACE axis port = axis_out_u register
	#pragma HLS INTERFACE ap_hls_chain port = return bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control

    static s2d_1pt_no_vect::widen_stream_dt widen_stream;
    static s2d_1pt_no_vect::mask_stream_dt mask_stream;

	#pragma HLS STREAM variable = widen_stream depth = max_depth_v8
	#pragma HLS STREAM variable = mask_stream depth = max_depth_v8

    ops::hls::GridPropertyCore gridProp;
    gridProp.dim = gridProp_dim;
    gridProp.size[0] = gridProp_size_x;
    gridProp.size[1] = gridProp_size_y;
    gridProp.actual_size[0] = gridProp_actual_size_x;
    gridProp.actual_size[1] = gridProp_actual_size_y;
    gridProp.grid_size[0] = gridProp_grid_size_x;
    gridProp.grid_size[1] = gridProp_grid_size_y;
    gridProp.xblocks = gridProp_xblocks;
    gridProp.total_itr = gridProp_total_itr;
    gridProp.outer_loop_limit = gridProp_outer_loop_limit;
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| starting kernel_simple_copy_PE\n", __func__);
#endif
    unsigned int total_bytes = gridProp.grid_size[0] * gridProp.grid_size[1] * sizeof(stencil_type);
    kernel_simple_copy_PE(gridProp, const_val, widen_stream, mask_stream);

//    template <unsigned int AXIS_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH>
//    void stream2axisMasked(::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& axis_out,
//    				::hls::stream<ap_uint<STREAM_DATA_WIDTH>>& strm_in,
//    				::hls::stream<ap_uint<STREAM_DATA_WIDTH/8>>& mask_in,
//    				unsigned int size)
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| starting stream2axisMasked\n", __func__);
#endif
    ops::hls::stream2axisMasked<256,256>(axis_out_u, widen_stream, mask_stream, total_bytes);
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| exiting.\n", __func__);
#endif
}
