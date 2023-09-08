
#include <datamover_simple_copy.hpp>
#include <ops_hls_datamover.hpp>

extern "C" void kernel_datamover_simple_copy(
    const unsigned short range_start_x,
	const unsigned short range_end_x,
	const unsigned short range_start_y,
	const unsigned short range_end_y,
	const unsigned short gridSize_x,
	const unsigned short gridSize_y,
    ap_uint<data_width>* arg0_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& stream0_in
)
{
	#pragma HLS INTERFACE s_axilite port = range_start_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_start_y bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_y bundle = control

	#pragma HLS INTERFACE s_axilite port = gridSize_x bundle = control
	#pragma HLS INTERFACE s_axilite port = gridSize_y bundle = control

    #pragma HLS INTERFACE mode=ma_axi bundle=gmem0 depth=max_axi_depth maxi_read_burst_length=64 maxi_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg0_out offset=slave
	#pragma HLS INTERFACE s_axilite port = arg0_out bundle = control
    #pragma HLS INTERFACE mode=axis port=stream0_in register
	#pragma HLS INTERFACE s_axilite port = return bundle = control

	ops::hls::SizeType gridSize = {gridSize_x, gridSize_y, 1};
	ops::hls::AccessRange range;
	range.start[0] = range_start_x;
	range.start[1] = range_start_y;
	range.end[0] = range_end_x;
	range.end[1] = range_end_y;
	range.dim = 2;
//	printf("[KERNEL_DEBUG]|%s| starting.\n", __func__);

    ops::hls::memWriteGrid<mem_data_width, axis_data_width, data_width>(arg0_out, stream0_in, gridSize, range);

//    printf("[KERNEL_DEBUG]|%s| exiting.\n", __func__);
}
