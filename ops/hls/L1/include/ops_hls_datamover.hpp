#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** @file
  * @brief Vitis HLS specific L1 data mover functions.
  * @author Beniel Thileepan
  * @details Implements of the templatised data mover functions with 
  * protcol conversion and data width conversions.
  */

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "ops_hls_defs.hpp"

namespace ops {
namespace hls {

/**
 * @brief 	convMemBeat2axisPkt reads a memory location with index and generate into AXI4-stream
 * 
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH>
static void convMemBeat2axisPkt(ap_uint<MEM_DATA_WIDTH>* mem_in,
					::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_out,
					const unsigned int& size,
					const unsigned int& index)
{	
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;
	
	ap_uint<MEM_DATA_WIDTH> tmp;
	tmp = mem_in[index];
	
	for (unsigned int pkt = 0; pkt < num_strm_pkts_per_beat; pkt++)
	{
	#pragma HLS PIPELINE II=num_strm_pkts_per_beat
	#pragma HLS LOOP_TRIPCOUNT min=min_strm_pkts_per_beat avg=avg_strm_pkts_per_beat max=max_strm_pkts_per_beat
		unsigned int byte_idx = index * bytes_per_beat + pkt * bytes_per_pkt;

		if (byte_idx < size)
		{
			ap_axiu<AXIS_DATA_WIDTH,0,0,0> tmp_pkt;
			tmp_pkt.data = tmp.range((pkt + 1) * AXIS_DATA_WIDTH - 1, pkt * AXIS_DATA_WIDTH);
			strm_out.write(tmp_pkt);
		}
	}
}

/**
 * @brief 	convAxisPkt2memBeat writes a memory location with index and from AXI4-stream
 * 
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH>
static void convAxisPkt2memBeat(ap_uint<MEM_DATA_WIDTH>* mem_out,
					::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
					unsigned int& size,
					unsigned int& index)
{	
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;
	
	ap_uint<MEM_DATA_WIDTH> tmp;
	
	for (unsigned int pkt = 0; pkt < num_strm_pkts_per_beat; pkt++)
	{
	#pragma HLS PIPELINE II=1
	#pragma HLS LOOP_TRIPCOUNT min=min_strm_pkts_per_beat avg=avg_strm_pkts_per_beat max=max_strm_pkts_per_beat
		unsigned int byte_idx = index * bytes_per_beat + pkt * bytes_per_pkt;

		if (byte_idx < size)
		{
			ap_axiu<AXIS_DATA_WIDTH,0,0,0> tmp_pkt;
			tmp_pkt = strm_in.read();
			tmp.range((pkt + 1) * AXIS_DATA_WIDTH - 1, pkt * AXIS_DATA_WIDTH) = tmp_pkt.data;
		}
	}
	mem_out[index] = tmp;
}

/**
 * @brief 	mem2axis reads from a memory location with to an AXI4 stream.
 *  		This is optimized to read from AXI4 with burst and to utilize width conversion in-between
 *  		AXI4 to AXI4-stream.
 * 
 * @tparam MEM_DATA_WIDTH : Data width of the AXI4 port
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam BURST_SIZE : Burst length of the AXI4 (max beats - 256)
 * 
 * @param mem_in : input memory port
 * @param stream_out : output AXI4-stream
 * @param size : Number of bytes of the data
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int BURST_SIZE=32>
void mem2axis(ap_uint<MEM_DATA_WIDTH>* mem_in,
				::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_out,
				unsigned int size)
{
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH % AXIS_DATA_WIDTH == 0, 
			"MEM_DATA_WIDTH has to be fully divided by AXIS_DATA_WIDTH");
	static_assert(MEM_DATA_WIDTH >= min_mem_data_width && MEM_DATA_WIDTH <= max_mem_data_width,
			"MEM_DATA_WIDTH failed limit check");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
	static_assert(BURST_SIZE >= min_burst_len && BURST_SIZE <= max_burst_len,
			" BURST_SIZE has failed limit check");
#endif

	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_bust_beats = num_beats % BURST_SIZE;
	
	unsigned int index = 0;

	for (unsigned int brst = 0; brst < num_bursts; brst++)
	{
	#pragma HLS LOOP_TRIPCOUNT avg=avg_num_of_bursts max=max_num_of_bursts

		for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
		{
		#pragma HLS PIPELINE II=num_strm_pkts_per_beat
		#pragma HLS LOOP_TRIPCOUNT min=min_burst_len avg=avg_burst_len max=max_burst_len

			convMemBeat2axisPkt<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_in, strm_out, size, index);
			index++;
		}
	}
	
	for (unsigned int beat = 0; beat < non_bust_beats; beat++)
	{
	#pragma HLS PIPELINE II=num_strm_pkts_per_beat
		convMemBeat2axisPkt<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_in, strm_out, size, index);
		index++;
	}
}

/**
 * @brief 	mem2axis reads from two memory location with a selector mux to an AXI4 stream.
 *  		This is optimized to read from AXI4 with burst and to utilize width conversion in-between
 *  		AXI4 to AXI4-stream
 * 
 * @tparam MEM_DATA_WIDTH : Data width of the AXI4 port
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam BURST_SIZE : Burst length of the AXI4 (max beats - 256)
 * 
 * @param mem_in0 : first memory port
 * @param mem_in1 : second memory port
 * @param stream_out : output AXI4-stream
 * @param size : Number of bytes of the data
 * @param selector : Memory location selector. 0 for mem_in0 and 1 for mem_in1
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int BURST_SIZE=32>
void mem2axis(ap_uint<MEM_DATA_WIDTH>* mem_in0, 
				ap_uint<MEM_DATA_WIDTH>* mem_in1,
				::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_out,
				unsigned int size,
				unsigned int selector)
{
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH % AXIS_DATA_WIDTH == 0, 
			"MEM_DATA_WIDTH has to be fully divided by AXIS_DATA_WIDTH");
	static_assert(MEM_DATA_WIDTH >= min_mem_data_width && MEM_DATA_WIDTH <= max_mem_data_width,
			"MEM_DATA_WIDTH failed limit check");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
	static_assert(BURST_SIZE >= min_burst_len && BURST_SIZE <= max_burst_len,
			" BURST_SIZE has failed limit check");
#endif
	
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_bust_beats = num_beats % BURST_SIZE;
	
	unsigned int index = 0;
	
	
	switch (selector)
	{
		case 0:
			for (unsigned int brst = 0; brst < num_bursts; brst++)
			{
			#pragma HLS LOOP_TRIPCOUNT avg=avg_num_of_bursts max=max_num_of_bursts

				for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
				{
				#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				#pragma HLS LOOP_TRIPCOUNT min=min_burst_len avg=avg_burst_len max=max_burst_len

					convMemBeat2axisPkt<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_in0, strm_out, size, index);
					index++;
				}
			}
			
			for (unsigned int beat = 0; beat < non_bust_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convMemBeat2axisPkt<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_in0, strm_out, size, index);
				index++;
			}
			
			break;
		case 1:
			for (unsigned int brst = 0; brst < num_bursts; brst++)
			{
			#pragma HLS LOOP_TRIPCOUNT avg=avg_num_of_bursts max=max_num_of_bursts

				for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
				{
				#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				#pragma HLS LOOP_TRIPCOUNT min=min_burst_len avg=avg_burst_len max=max_burst_len

					convMemBeat2axisPkt<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_in1, strm_out, size, index);
					index++;
				}
			}
			
			for (unsigned int beat = 0; beat < non_bust_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convMemBeat2axisPkt<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_in1, strm_out, size, index);
				index++;
			}
			
			break;	
	}
}

/**
 * @brief 	axis2mem reads from an AXI4 stream into a memory location.
 *  		This is optimized to write to AXI4 with burst and to utilize width conversion in-between
 *  		AXI4 to AXI4-stream
 *
 * @tparam MEM_DATA_WIDTH : Data width of the AXI4 port
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam BURST_SIZE : Burst length of the AXI4 (max beats - 256)
 *
 * @param mem_out : output memory port
 * @param stream_in : input AXI4-stream
 * @param size : Number of bytes of the data
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int BURST_SIZE=32>
void axis2mem(ap_uint<MEM_DATA_WIDTH>* mem_out,
				::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
				unsigned int size)
{
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH % AXIS_DATA_WIDTH == 0,
			"MEM_DATA_WIDTH has to be fully divided by AXIS_DATA_WIDTH");
	static_assert(MEM_DATA_WIDTH >= min_mem_data_width && MEM_DATA_WIDTH <= max_mem_data_width,
			"MEM_DATA_WIDTH failed limit check");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
	static_assert(BURST_SIZE >= min_burst_len && BURST_SIZE <= max_burst_len,
			" BURST_SIZE has failed limit check");
#endif

	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_bust_beats = num_beats % BURST_SIZE;

	unsigned int index = 0;

	for (unsigned int brst = 0; brst < num_bursts; brst++)
	{
	#pragma HLS LOOP_TRIPCOUNT avg=avg_num_of_bursts max=max_num_of_bursts

		for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
		{
		#pragma HLS PIPELINE II=num_strm_pkts_per_beat
		#pragma HLS LOOP_TRIPCOUNT min=min_burst_len avg=avg_burst_len max=max_burst_len

			convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out, strm_in, size, index);
			index++;
		}
	}

	for (unsigned int beat = 0; beat < non_bust_beats; beat++)
	{
	#pragma HLS PIPELINE II=num_strm_pkts_per_beat
		convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out, strm_in, size, index);
		index++;
	}

}

/**
 * @brief 	axis2mem reads from an AXI4 stream into a memory location with a selector mux to select out of two.
 *  		This is optimized to write to AXI4 with burst and to utilize width conversion in-between
 *  		AXI4 to AXI4-stream
 *
 * @tparam MEM_DATA_WIDTH : Data width of the AXI4 port
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam BURST_SIZE : Burst length of the AXI4 (max beats - 256)
 *
 * @param mem_out0 : first memory port
 * @param mem_out1 : second memory port
 * @param stream_in : input AXI4-stream
 * @param size : Number of bytes of the data
 * @param selector : Memory location selector. 0 for mem_out0 and 1 for mem_out1
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int BURST_SIZE=32>
void axis2mem(ap_uint<MEM_DATA_WIDTH>* mem_out0,
				ap_uint<MEM_DATA_WIDTH>* mem_out1,
				::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
				unsigned int size,
				unsigned int selector)
{
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH % AXIS_DATA_WIDTH == 0,
			"MEM_DATA_WIDTH has to be fully divided by AXIS_DATA_WIDTH");
	static_assert(MEM_DATA_WIDTH >= min_mem_data_width && MEM_DATA_WIDTH <= max_mem_data_width,
			"MEM_DATA_WIDTH failed limit check");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
	static_assert(BURST_SIZE >= min_burst_len && BURST_SIZE <= max_burst_len,
			" BURST_SIZE has failed limit check");
#endif

	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_bust_beats = num_beats % BURST_SIZE;

	unsigned int index = 0;

	switch (selector)
	{
		case 0:
			for (unsigned int brst = 0; brst < num_bursts; brst++)
			{
			#pragma HLS LOOP_TRIPCOUNT avg=avg_num_of_bursts max=max_num_of_bursts

				for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
				{
				#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				#pragma HLS LOOP_TRIPCOUNT min=min_burst_len avg=avg_burst_len max=max_burst_len

					convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out0, strm_in, size, index);
					index++;
				}
			}

			for (unsigned int beat = 0; beat < non_bust_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out0, strm_in, size, index);
				index++;
			}

			break;
		case 1:
			for (unsigned int brst = 0; brst < num_bursts; brst++)
			{
			#pragma HLS LOOP_TRIPCOUNT avg=avg_num_of_bursts max=max_num_of_bursts

				for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
				{
				#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				#pragma HLS LOOP_TRIPCOUNT min=min_burst_len avg=avg_burst_len max=max_burst_len

					convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out1, strm_in, size, index);
					index++;
				}
			}

			for (unsigned int beat = 0; beat < non_bust_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out1, strm_in, size, index);
				index++;
			}

			break;
	}
}

/**
 * @brief axis2stream converts AXI4-stream to hls-stream	
 *
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam STREAM_DATA_WIDTH : Data width of the HLS-stream port
 * 
 * @param axis_in : AXI4-stream input port
 * @param strm_out : HLS-stream output port
 * @param size : Number of the bytes of data
 */
template <unsigned int AXIS_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH>
void axis2stream(::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& axis_in,
				::hls::stream<ap_uint<STREAM_DATA_WIDTH>>& strm_out,
				unsigned int size)
{
#ifndef __SYTHESIS__
	static_assert(AXIS_DATA_WIDTH % STREAM_DATA_WIDTH == 0,
			"AXIS_DATA_WIDTH has to be fully divided by STREAM_DATA_WIDTH");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
	static_assert(STREAM_DATA_WIDTH >= min_hls_stream_data_width && AXIS_DATA_WIDTH <= max_hls_stream_data_width,
			"STREAM_DATA_WIDTH failed limit check");
#endif

	constexpr unsigned int num_hls_pkt_per_axis_pkt = AXIS_DATA_WIDTH / STREAM_DATA_WIDTH;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	
	const unsigned int num_axis_pkts = (size + bytes_per_pkt - 1) / bytes_per_pkt;

	for (unsigned int itr = 0; itr < num_axis_pkts; itr++)
	{
		ap_axiu<AXIS_DATA_WIDTH,0,0,0> axisPkt = axis_in.read();

		for (unsigned int j = 0; j < num_hls_pkt_per_axis_pkt; j++)
		{
		#pragma HLS PIPELINE II=1
			strm_out << axisPkt.data.range((j+1) * STREAM_DATA_WIDTH - 1, j * STREAM_DATA_WIDTH);
		}
	}
}

/**
 * @brief stream2axis converts hls-stream to AXI4-stream
 *
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam STREAM_DATA_WIDTH : Data width of the HLS-stream port
 * 
 * @param axis_out : AXI4-stream output port
 * @param strm_in : HLS-stream input port
 * @param size : Number of the bytes of data
 */

template <unsigned int AXIS_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH>
void stream2axis(::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& axis_out,
				::hls::stream<ap_uint<STREAM_DATA_WIDTH>>& strm_in,
				unsigned int size)
{
#ifndef __SYTHESIS__
	static_assert(AXIS_DATA_WIDTH % STREAM_DATA_WIDTH == 0,
			"AXIS_DATA_WIDTH has to be fully divided by STREAM_DATA_WIDTH");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
	static_assert(STREAM_DATA_WIDTH >= min_hls_stream_data_width && AXIS_DATA_WIDTH <= max_hls_stream_data_width,
			"STREAM_DATA_WIDTH failed limit check");
#endif

	constexpr unsigned int num_hls_pkt_per_axis_pkt = AXIS_DATA_WIDTH / STREAM_DATA_WIDTH;
	constexpr unsigned int bytes_per_pkt = AXIS_DATA_WIDTH / 8;
	
	const unsigned int num_axis_pkts = (size + bytes_per_pkt - 1) / bytes_per_pkt;

	for (unsigned int itr = 0; itr < num_axis_pkts; itr++)
	{
		ap_axiu<AXIS_DATA_WIDTH,0,0,0> axisPkt;

		for (unsigned int j = 0; j < num_hls_pkt_per_axis_pkt; j++)
		{
		#pragma HLS PIPELINE II=1
			axisPkt.data.range((j+1) * STREAM_DATA_WIDTH - 1, j * STREAM_DATA_WIDTH) = strm_in.read();
		}

		axis_out.write(axisPkt);
	}
}

}
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */