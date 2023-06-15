
#pragma once

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

namespace ops {
namespace hls {

/**
 * @brief 	convMemBeat2streamPkt reads a memory location with index and generate into AXI4-stream
 * 			with side channels. 
 * 
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH>
static void convMemBeat2streamPkt(ap_uint<MEM_DATA_WIDTH>* mem_in,
					::hls::stream<ap_axiu<STREAM_DATA_WIDTH,0,0,0>>& strm_out,
					const unsigned int& size,
					const unsigned int& index)
{	
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = STREAM_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / STREAM_DATA_WIDTH;
	
	ap_uint<MEM_DATA_WIDTH> tmp;
	tmp = mem_in[index];
	
	for (unsigned int pkt = 0; pkt < num_strm_pkts_per_beat; pkt++)
	{
	#pragma HLS PIPELINE II=num_strm_pkts_per_beat
		
		ap_axiu<STREAM_DATA_WIDTH,0,0,0> tmp_pkt;
		tmp_pkt.data = tmp.range((pkt + 1) * STREAM_DATA_WIDTH - 1, pkt * STREAM_DATA_WIDTH);

		/* TKEEP MACHANISM */
//		for (unsigned int i = 0; i < bytes_per_pkt; i++)
//		{
//		#pragma HLS UNROLL
//			tmp_pkt.keep.range((i+1), i) = index * bytes_per_beat + pkt * bytes_per_pkt + i < size ? 1 : 0;
//		}
		strm_out.write(tmp_pkt);
	}
}

/**
 * @brief 	readStream2Mem reads a memory location with index and generate into AXI4-stream
 * 			with side channels. 
 * 
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH>
static void convStreamPkt2memBeat(ap_uint<MEM_DATA_WIDTH>* mem_out,
					::hls::stream<ap_axiu<STREAM_DATA_WIDTH,0,0,0>>& strm_in,
					unsigned int& size,
					unsigned int& index)
{	
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = STREAM_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / STREAM_DATA_WIDTH;
	
	ap_uint<MEM_DATA_WIDTH> tmp;
	
	for (unsigned int pkt = 0; pkt < num_strm_pkts_per_beat; pkt++)
	{
	#pragma HLS PIPELINE II=num_strm_pkts_per_beat
		
		ap_axiu<STREAM_DATA_WIDTH,0,0,0> tmp_pkt;
		tmp_pkt = strm_in.read();

		/* TKEEP MACHANISM */
//		for (unsigned int i = 0; i < bytes_per_pkt; i++)
//		{
//		#pragma HLS UNROLL
//			tmp.range(pkt * STREAM_DATA_WIDTH + (i + 1) * 8, pkt * STREAM_DATA_WIDTH + i * 8) 
//					= tmp_pkt.keep.range((i+1), i) == 1 ? tmp_pkt.data.range((i + 1) * 8, i * 8);
//		}
		tmp.range((pkt + 1) * STREAM_DATA_WIDTH - 1, pkt * STREAM_DATA_WIDTH) = tmp_pkt.data;
	}
	mem_out[index] = tmp;
}

/**
 * @brief 	mem2stream reads from two memory location with a selector mux to an AXI4 stream.
 *  		This is optimized to read from AXI4 with burst and to utilize width conversion inbetween
 *  		AXI4 to AXI4-stream
 * 
 * @tparam MEM_DATA_WIDTH : Data width of the AXI4 port
 * @tparam STREAM_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam BURST_SIZE : Burst lenth of the AXI4 (max beats - 256)
 * 
 * @param mem_in0 : first memory port
 * @param mem_in1 : second memory port
 * @param stream_out : output AXI4-stream
 * @param size : Number of bytes of the data
 * @param selector : Memory location selector. 0 for mem_in0 and 1 for mem_in1
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void mem2stream(ap_uint<MEM_DATA_WIDTH>* mem_in0, 
				ap_uint<MEM_DATA_WIDTH>* mem_in1,
				::hls::stream<ap_axiu<STREAM_DATA_WIDTH,0,0,0>>& strm_out,
				unsigned int size,
				unsigned int selector)
{
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH % STREAM_DATA_WIDTH == 0, 
			"MEM_DATA_WIDTH has to be fully devided by STREAM_DATA_WIDTH");
	static_assert(MEM_DATA_WIDTH >= 128 && MEM_DATA_WIDTH <= 1024, 
			"MEM_DATA_WIDTH has to be >= 128 and <= 1024");
	static_assert(STREAM_DATA_WIDTH >= 64 && STREAM_DATA_WIDTH <= 1024,
			"STRAM_DATA_WIDTH has to be >= 64 and <= 1024");
	static_assert(BURST_SIZE >= 1 && BURST_SIZE <= 256,
			" BURST_SIZE has to be >= 1 and <= 256");
#endif
	
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_pkt = STREAM_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / STREAM_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_bust_beats = num_beats % BURST_SIZE;
	
	unsigned int index = 0;
	
	switch (selector)
	{
		case 0:
			for (unsigned int brst = 0; brst < num_bursts; brst++)
			{
				for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
				{
				#pragma HLS PIPELINE II=num_strm_pkts_per_beat
					convMemBeat2streamPkt<MEM_DATA_WIDTH, STREAM_DATA_WIDTH>(mem_in0, strm_out, size, index);
					index++;
				}
			}
			
			for (unsigned int beat = 0; beat < non_bust_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convMemBeat2streamPkt<MEM_DATA_WIDTH, STREAM_DATA_WIDTH>(mem_in0, strm_out, size, index);
				index++;
			}
			
			break;
		case 1:
			for (unsigned int brst = 0; brst < num_bursts; brst++)
			{
				for (unsigned int beat = 0; beat < BURST_SIZE; beat++)
				{
				#pragma HLS PIPELINE II=num_strm_pkts_per_beat
					convMemBeat2streamPkt<MEM_DATA_WIDTH, STREAM_DATA_WIDTH>(mem_in1, strm_out, size, index);
					index++;
				}
			}
			
			for (unsigned int beat = 0; beat < non_bust_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convMemBeat2streamPkt<MEM_DATA_WIDTH, STREAM_DATA_WIDTH>(mem_in1, strm_out, size, index);
				index++;
			}
			
			break;	
	}
}

///**
// * @brief stream2mem writes to two memory location with a selector mux from an AXI4 stream
// *
// * @tparam MEM_DATA_TYPE: data type of the memory
// * @tparam STREAM_DATA_TYPE: data type of hls::stream stream
// *
// * @param mem_in0: first memory port
// * @param mem_in1: second memory port
// * @param stream_out: output AXI4 stream
// */
//template <typename MEM_DATA_TYPE, typename STREAM_DATA_TYPE>
//void stream2mem(MEM_TYPE* mem_out0,
//				MEM_TYPE* mem_out1,
//				hls::stream<STREAM_DATA_TYPE>& strm_in,
//				unsigned int num,
//				unsigned int selector)
//{
//	switch (selector)
//	{
//		case 0:
//			for (unsigned int itr = 0; itr < num; itr++)
//			{
//			#pragma HLS PIPELINE
//				mem_out0[itr] = stream_in.read();
//			}
//			break;
//		case 1:
//			for (unsigned int itr = 0; itr < num; itr++)
//			{
//			#pragma HLS PIPELINE
//				mem_out1[itr] = stream_in.read();
//			}
//			break;
//	}
//}

}
}
