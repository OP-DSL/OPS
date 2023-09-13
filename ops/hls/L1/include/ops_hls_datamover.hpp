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
#include "ops_hls_utils.hpp"
#include <math.h>
#include <stdio.h>
//#define DEBUG_LOG

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
	#ifndef DEBUG_LOG_SIZE_OF
		#define DEBUG_LOG_SIZE_OF 4
	#endif
#endif
#endif

namespace ops {
namespace hls {

/**
 * @brief 	convMemBeat2axisPkt reads a memory location with index and generate into AXI4-stream. This works
 * with MEM_DATA_WIDTH >= AXIS_DATA_WIDTH.
 * 
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH>
static void convMemBeat2axisPkt(ap_uint<MEM_DATA_WIDTH>* mem_in,
					::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_out,
					const unsigned int& size,
					const unsigned int& index)
{	
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH >= AXIS_DATA_WIDTH,
			"MEM_DATA_WIDTH has to be grater that AXIS_DATA_WIDTH");
#endif
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
	printf("\n|HLS DEBUG_LOG| %s | bytes_per_beat: %d, bytes_per_axis_pkt: %d, num_strm_pkts_per_beat: %d\n"
			, __func__, bytes_per_beat, bytes_per_axis_pkt, num_strm_pkts_per_beat);
	printf("======================================================================================================\n");
#endif
#endif
	
	ap_uint<MEM_DATA_WIDTH> tmp;
	tmp = mem_in[index];
	
#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
	printf("   |HLS DEBUG_LOG| reading beat index: %d\n", index);
#endif
#endif

	for (unsigned int pkt = 0; pkt < num_strm_pkts_per_beat; pkt++)
	{
	#pragma HLS PIPELINE II=num_strm_pkts_per_beat
	#pragma HLS LOOP_TRIPCOUNT min=min_strm_pkts_per_beat avg=avg_strm_pkts_per_beat max=max_strm_pkts_per_beat
		unsigned int byte_idx = index * bytes_per_beat + pkt * bytes_per_axis_pkt;

		if (byte_idx < size)
		{
			ap_axiu<AXIS_DATA_WIDTH,0,0,0> tmp_pkt;
			tmp_pkt.data = tmp.range((pkt + 1) * AXIS_DATA_WIDTH - 1, pkt * AXIS_DATA_WIDTH);
			strm_out.write(tmp_pkt);

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
			printf("   |HLS DEBUG_LOG| sending axis pkt: %d, val=(", pkt);

			for (unsigned n = 0; n < bytes_per_axis_pkt/DEBUG_LOG_SIZE_OF; n++)
			{
				DataConv tmp;
				tmp.i = tmp_pkt.data.range((n+1) * DEBUG_LOG_SIZE_OF * 8 - 1, n * DEBUG_LOG_SIZE_OF * 8);
				printf("%f,", tmp.f);
			}
			printf(")\n");
#endif
#endif
		}
	}
}

/**
 * @brief 	convAxisPkt2memBeat writes a memory location with index and from AXI4-stream.
 * This works MEM_DATA_WIDTH >= AXIS_DATA_WIDTH
 * 
 */
template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH>
static void convAxisPkt2memBeat(ap_uint<MEM_DATA_WIDTH>* mem_out,
					::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
					unsigned int& size,
					unsigned int& index)
{	
#ifndef __SYTHESIS__
	static_assert(MEM_DATA_WIDTH >= AXIS_DATA_WIDTH,
			"MEM_DATA_WIDTH has to be grater that AXIS_DATA_WIDTH");
#endif
	constexpr unsigned int bytes_per_beat = MEM_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
	printf("\n|HLS DEBUG_LOG| %s | bytes_per_beat: %d, bytes_per_axis_pkt: %d, num_strm_pkts_per_beat: %d, beat index: %d\n"
			, __func__, bytes_per_beat, bytes_per_axis_pkt, num_strm_pkts_per_beat, index);
	printf("======================================================================================================\n");
#endif
#endif
	ap_uint<MEM_DATA_WIDTH> tmp;
	
	for (unsigned int pkt = 0; pkt < num_strm_pkts_per_beat; pkt++)
	{
	#pragma HLS PIPELINE II=1
	#pragma HLS LOOP_TRIPCOUNT min=min_strm_pkts_per_beat avg=avg_strm_pkts_per_beat max=max_strm_pkts_per_beat
		unsigned int byte_idx = index * bytes_per_beat + pkt * bytes_per_axis_pkt;

		if (byte_idx < size)
		{
			ap_axiu<AXIS_DATA_WIDTH,0,0,0> tmp_pkt;
			tmp_pkt = strm_in.read();
			tmp.range((pkt + 1) * AXIS_DATA_WIDTH - 1, pkt * AXIS_DATA_WIDTH) = tmp_pkt.data;

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
			printf("   |HLS DEBUG_LOG||%s| receiving axis pkt: %d, val=(", __func__, pkt);

			for (unsigned n = 0; n < bytes_per_axis_pkt/DEBUG_LOG_SIZE_OF; n++)
			{
				DataConv tmp;
				tmp.i = tmp_pkt.data.range((n+1) * DEBUG_LOG_SIZE_OF * 8 - 1, n * DEBUG_LOG_SIZE_OF * 8);
				printf("%f,", tmp.f);
			}
			printf(")\n");
#endif
#endif
		}
	}
	mem_out[index] = tmp;
}

/**
 * @brief 	convAxisPkt2memBeatMaksed writes a memory location with index and 
 * from AXI4-stream with strobe channels.
 *
 * @detaisl This works with DATA_WITDH < AXIS_DATA_WIDTH. This will read from
 * AXI4-stream packet with verifying strobe channel. If the TSTRB of of the first byte of a data
 * point with DATA_WIDH is LOW, it will omit writing it to memory.
 *
 */
template <unsigned int DATA_WIDTH, unsigned int AXIS_DATA_WIDTH>
static void convAxisPkt2memBeatMasked(ap_uint<DATA_WIDTH>* mem_out,
					::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
					unsigned int& size,
					unsigned int& index)
{
#ifndef __SYTHESIS__
	static_assert(DATA_WIDTH < AXIS_DATA_WIDTH,
			"MEM_DATA_WIDTH has to be less that AXIS_DATA_WIDTH");
#endif

	constexpr unsigned int bytes_per_data = DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_data_per_axis_pkt = AXIS_DATA_WIDTH / DATA_WIDTH;
	constexpr unsigned int shift_val = 1 << bytes_per_data;
	bool cond_write = 0;

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
	printf("\n|HLS DEBUG_LOG|%s| bytes_per_axis_pkt: %d, bytes_per_data: %d, num_data_per_axis_pkt: %d, beat index: %d\n"
			, __func__, bytes_per_axis_pkt, bytes_per_data, num_data_per_axis_pkt, index);
	printf("======================================================================================================\n");
#endif
#endif
	ap_axiu<AXIS_DATA_WIDTH,0,0,0> tmp_pkt;
	
	tmp_pkt = strm_in.read();

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
		printf("   |HLS DEBUG_LOG||%s| receiving axis , val=(", __func__);

		for (unsigned n = 0; n < bytes_per_axis_pkt/DEBUG_LOG_SIZE_OF; n++)
		{
			DataConv tmp;
			tmp.i = tmp_pkt.data.range((n+1) * DEBUG_LOG_SIZE_OF * 8 - 1, n * DEBUG_LOG_SIZE_OF * 8);
			printf("%f,", tmp.f);
		}

		printf(") strb=(%x)\n", tmp_pkt.strb);

#endif
#endif
	ap_uint<bytes_per_axis_pkt> strb_pos = 1;

	for (unsigned int idx = 0; idx < num_data_per_axis_pkt; idx++)
	{
#pragma HLS PIPELINE  II=1
#pragma HLS LOOP_TRIPCOUNT avg=avg_num_data_per_axis_pkt

		ap_uint<bytes_per_axis_pkt> cond_val = tmp_pkt.strb & strb_pos;
		cond_write = register_it((tmp_pkt.strb & strb_pos) == 0);
		ap_uint<DATA_WIDTH> write_val = tmp_pkt.data.range((idx+1) * DATA_WIDTH - 1, idx * DATA_WIDTH);

#ifdef DEBUG_LOG
		DataConv tmp;
		tmp.i = write_val;
		printf("   |HLS DEBUG_LOG||%s| writing to mem index: %d, strb_pos:%d, val: %f, cond: %d\n"
				, __func__, index+idx, strb_pos, tmp.f, cond_write);
#endif
		if (!cond_write)
		{
			mem_out[index + idx] = write_val;
		}
		strb_pos *= shift_val;
	}
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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_burst_beats = num_beats % BURST_SIZE;

#ifndef __SYTHESIS__
#ifdef DEBUG_LOG
	printf("|HLS DEBUG_LOG| %s | size: %d, num_beats: %d, num_burst: %d, non_burst_beats: %d\n"
			, __func__, size, num_beats, num_bursts, non_burst_beats);
	printf("====================================================================================\n");
#endif
#endif

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
	
	for (unsigned int beat = 0; beat < non_burst_beats; beat++)
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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_burst_beats = num_beats % BURST_SIZE;
	
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
			
			for (unsigned int beat = 0; beat < non_burst_beats; beat++)
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
			
			for (unsigned int beat = 0; beat < non_burst_beats; beat++)
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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_burst_beats = num_beats % BURST_SIZE;

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

	for (unsigned int beat = 0; beat < non_burst_beats; beat++)
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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int num_strm_pkts_per_beat = MEM_DATA_WIDTH / AXIS_DATA_WIDTH;

	const unsigned int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;
	const unsigned int num_bursts = num_beats / BURST_SIZE;
	const unsigned int non_burst_beats = num_beats % BURST_SIZE;

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

			for (unsigned int beat = 0; beat < non_burst_beats; beat++)
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

			for (unsigned int beat = 0; beat < non_burst_beats; beat++)
			{
			#pragma HLS PIPELINE II=num_strm_pkts_per_beat
				convAxisPkt2memBeat<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out1, strm_in, size, index);
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
template <unsigned int DATA_WIDTH, unsigned int AXIS_DATA_WIDTH>
void axis2memMasked(ap_uint<DATA_WIDTH>* mem_out,
				::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
				unsigned int size)
{
#ifndef __SYTHESIS__
	static_assert(AXIS_DATA_WIDTH % DATA_WIDTH == 0,
			"AXIS_DATA_WIDTH has to be fully divided by DATA_WIDTH");
	static_assert(AXIS_DATA_WIDTH >= min_axis_data_width && AXIS_DATA_WIDTH <= max_axis_data_width,
			"AXIS_DATA_WIDTH failed limit check");
#endif

	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int data_per_axis_pkt = AXIS_DATA_WIDTH / DATA_WIDTH;
	const unsigned int num_of_axis_pkt = (size + bytes_per_axis_pkt - 1) / bytes_per_axis_pkt;
//	const unsigned int num_bursts = num_beats / BURST_SIZE;
//	const unsigned int non_burst_beats = num_beats % BURST_SIZE;

	unsigned int index = 0;

	for (unsigned int brst = 0; brst < num_of_axis_pkt; brst++)
	{
	#pragma HLS PIPELINE II=data_per_axis_pkt
	#pragma HLS LOOP_TRIPCOUNT min=1 avg=8

		convAxisPkt2memBeatMasked<DATA_WIDTH,AXIS_DATA_WIDTH>(mem_out, strm_in, size, index);
		index += data_per_axis_pkt;
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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	
	const unsigned int num_axis_pkts = (size + bytes_per_axis_pkt - 1) / bytes_per_axis_pkt;

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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	
	const unsigned int num_axis_pkts = (size + bytes_per_axis_pkt - 1) / bytes_per_axis_pkt;

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

/**
 * @brief stream2axisMasked converts hls-stream to AXI4-stream with bit-mask for using
 * tsrb channel of axi4-stream protocol
 *
 * @tparam AXIS_DATA_WIDTH : Data width of the AXI4-stream port
 * @tparam STREAM_DATA_WIDTH : Data width of the HLS-stream port
 * 
 * @param axis_out : AXI4-stream output port
 * @param strm_in : HLS-stream input port
 * @param mask_in : mask HLS-stream port for mask bits to be coverted to strb channel
 * @param size : Number of the bytes of data
 */

template <unsigned int AXIS_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH>
void stream2axisMasked(::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& axis_out,
				::hls::stream<ap_uint<STREAM_DATA_WIDTH>>& strm_in,
				::hls::stream<ap_uint<STREAM_DATA_WIDTH/8>>& mask_in,
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
	constexpr unsigned int bytes_per_axis_pkt = AXIS_DATA_WIDTH / 8;
	constexpr unsigned int bytes_per_hls_pkt = STREAM_DATA_WIDTH / 8;
	
	const unsigned int num_axis_pkts = (size + bytes_per_axis_pkt - 1) / bytes_per_axis_pkt;

	#ifdef DEBUG_LOG
		printf("|HLS DEBUG_LOG|%s| stream to axis. size (bytes): %d, num_hls_pkt_per_axis_pkt: %d, bytes_per_axis_pkt: %d, bytes_per_hls_pkt:%d, num_axis_pkts:%d\n"
				, __func__, size, num_hls_pkt_per_axis_pkt, bytes_per_axis_pkt, bytes_per_hls_pkt, num_axis_pkts);
		printf("====================================================================================\n");
	#endif

	for (unsigned int itr = 0; itr < num_axis_pkts; itr++)
	{
		ap_axiu<AXIS_DATA_WIDTH,0,0,0> axisPkt;
#ifdef DEBUG_LOG
		printf("|HLS DEBUG_LOG|%s| flag 2. itr:%d\n",
							__func__, itr);
#endif
		for (unsigned int j = 0; j < num_hls_pkt_per_axis_pkt; j++)
		{
		#pragma HLS PIPELINE II=1
#ifdef DEBUG_LOG
			unsigned int hls_pkt_index = itr * num_hls_pkt_per_axis_pkt + j;
			printf("|HLS DEBUG_LOG|%s| reading hls pkt: %d\n",
					__func__, hls_pkt_index);
#endif
			axisPkt.data.range((j+1) * STREAM_DATA_WIDTH - 1, j * STREAM_DATA_WIDTH) = strm_in.read();
			axisPkt.strb.range((j+1) * bytes_per_hls_pkt - 1, j * bytes_per_hls_pkt) = mask_in.read();

		}
#ifdef DEBUG_LOG
		printf("|HLS DEBUG_LOG|%s| wrting axis pkt: %d\n",
				__func__, itr);
#endif
		axis_out.write(axisPkt);
	}
#ifdef DEBUG_LOG
	printf("|HLS DEBUG_LOG|%s| exiting.\n"
			, __func__);
#endif
}

template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int DATA_WIDTH=32>
void memReadGrid(ap_uint<DATA_WIDTH>* mem_in,
		::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_out,
		unsigned int size,
		SizeType gridSize,
		SizeType offset)
{
	unsigned int init_offset = offset[0] + offset[1] * gridSize[0] + offset[2] * gridSize[0] * gridSize[1];
	mem2axis<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>((ap_uint<MEM_DATA_WIDTH>*)(mem_in + init_offset), strm_out, size);
}

template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int DATA_WIDTH=32>
void memReadGrid(ap_uint<DATA_WIDTH>* mem_in,
		::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_out,
		SizeType gridSize,
		AccessRange& range)
{
	constexpr unsigned short vectorFactor = AXIS_DATA_WIDTH / DATA_WIDTH;
	unsigned short ShiftBits = LOG2(vectorFactor);
	unsigned short DataShiftBits = LOG2(DATA_WIDTH/8);
	unsigned short num_xblocks = (range.end[0] - range.start[0] + vectorFactor - 1) >> ShiftBits;
	unsigned short x_tile_size = num_xblocks << ShiftBits;
	unsigned int x_tile_size_bytes = x_tile_size << DataShiftBits;

	for (unsigned short k = range.start[2]; k < range.end[2]; k++)
	{
		for (unsigned short j = range.start[1]; j < range.end[1]; j++)
		{
			unsigned int offset = range.start[0] + j * gridSize[0] + k * gridSize[1] * gridSize[0];
			mem2axis<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>((ap_uint<MEM_DATA_WIDTH>*)(mem_in + offset), strm_out, x_tile_size_bytes);
		}
	}
}

template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int DATA_WIDTH=32>
void memWriteGrid(ap_uint<DATA_WIDTH>* mem_out,
		::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
		unsigned int size,
		SizeType gridSize,
		SizeType offset)
{
	unsigned int init_offset = offset[0] + offset[1] * gridSize[0] + offset[2] * gridSize[0] * gridSize[1];
	constexpr unsigned short mem_data_bytes = MEM_DATA_WIDTH / 8;
	constexpr unsigned short data_bytes = DATA_WIDTH /8;
	unsigned int num_whole_axi_blocks = size / mem_data_bytes;
	unsigned int whole_axi_size = num_whole_axi_blocks * mem_data_bytes;
	unsigned int partial_offset = whole_axi_size / data_bytes;
	unsigned int partial_axi_size = size - whole_axi_size;

	axis2mem<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>((ap_uint<MEM_DATA_WIDTH>*)(mem_out + init_offset), strm_in, whole_axi_size);
	axis2memMasked<DATA_WIDTH, AXIS_DATA_WIDTH>((mem_out + init_offset + partial_offset), strm_in, partial_axi_size);
}

template <unsigned int MEM_DATA_WIDTH, unsigned int AXIS_DATA_WIDTH, unsigned int DATA_WIDTH=32>
void memWriteGrid(ap_uint<DATA_WIDTH>* mem_out,
		::hls::stream<ap_axiu<AXIS_DATA_WIDTH,0,0,0>>& strm_in,
		SizeType gridSize,
		AccessRange& range)
{
//	unsigned int init_offset = offset[0] + offset[1] * gridSize[0] + offset[2] * gridSize[0] * gridSize[1];
//	constexpr unsigned short mem_data_bytes = MEM_DATA_WIDTH / 8;
//	constexpr unsigned short data_bytes = DATA_WIDTH /8;
//	unsigned int num_whole_axi_reads = size / mem_data_bytes;
//	unsigned int whole_axi_size = num_whole_axi_reads * mem_data_bytes;
//	unsigned int partial_offset = whole_axi_size / data_bytes;
//	unsigned int partial_axi_size = size - whole_axi_size;
	constexpr unsigned short vectorFactor = MEM_DATA_WIDTH / DATA_WIDTH;
	unsigned short ShiftBits = LOG2(vectorFactor);
	unsigned short DataShiftBits = LOG2(DATA_WIDTH/8);
	unsigned short xtile_size = (range.end[0] - range.start[0]);
	unsigned short num_whole_xblocks = xtile_size >> ShiftBits;
	unsigned short whole_xtile_size = num_whole_xblocks << ShiftBits;
	unsigned short whole_xtile_size_bytes = whole_xtile_size << DataShiftBits;
	unsigned short partial_xtile_size = xtile_size - whole_xtile_size;
	unsigned short partial_xtile_size_bytes = partial_xtile_size << DataShiftBits;
	unsigned short xtile_size_bytes = xtile_size << DataShiftBits;

	if (range.dim < 3)
	{
		range.start[2] = 0;
		range.end[2] = 1;
	}
	else if (range.dim < 2)
	{
		range.start[1] = 0;
		range.end[1] = 1;
	}

#ifdef DEBUG_LOG
	printf("|HLS DEBUG_LOG|%s| starting memWriteGrid. grid_size: (%d, %d, %d), range: (%d, %d, %d) --> (%d, %d, %d)\n whole_xtile_size:%d, partial_xtile_size:%d\n"
			, __func__, gridSize[0], gridSize[1], gridSize[2], range.start[0], range.start[1], range.start[2],
			range.end[0], range.end[1], range.end[2], whole_xtile_size, partial_xtile_size);
	printf("====================================================================================\n");
#endif
	for (unsigned short k = range.start[2]; k < range.end[2]; k++)
	{
		for (unsigned short j = range.start[1]; j < range.end[1]; j++)
		{
			unsigned short offset = k * gridSize[1];
			offset = j + offset;
			offset = offset * gridSize[0];
			offset = range.start[0] + offset;
#ifdef DEBUG_LOG
			printf("|HLS DEBUG_LOG|%s| writing. offset:%d, j:%d, k:%d\n"
					, __func__, offset, j, k);
#endif
			axis2mem<MEM_DATA_WIDTH, AXIS_DATA_WIDTH>((ap_uint<MEM_DATA_WIDTH>*)(mem_out + offset), strm_in, whole_xtile_size_bytes);
			axis2memMasked<DATA_WIDTH, AXIS_DATA_WIDTH>(mem_out + offset + whole_xtile_size, strm_in, partial_xtile_size_bytes);
		}
	}
#ifdef DEBUG_LOG
	printf("|HLS DEBUG_LOG|%s| exiting.\n"
			, __func__);
#endif
}

}
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
