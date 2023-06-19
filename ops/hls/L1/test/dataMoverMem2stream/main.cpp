#include <iostream>
#include <random>
#include "top.hpp"

// #define DEBUG_LOG 

const int num = 100; //22 * 22 * 22; //20 x 20 x 20 grid with (-1,+1) halo with 4-byte datatype in bytes
const int sizeBytes = num * sizeof(float);
const int bytes_per_beat = AXI_M_WIDTH / 8;
const int bytes_per_pkt = AXIS_WIDTH / 8;
const int data_per_beat = bytes_per_beat / sizeof(float); 
const int data_per_pkt = bytes_per_pkt / sizeof(float);
const int datasize_bits = sizeof(float) * 8;
const int num_beats = (sizeBytes + bytes_per_beat - 1) / bytes_per_beat;
const int num_pkts_per_beat = AXI_M_WIDTH / AXIS_WIDTH;


static bool testDut(ap_uint<AXI_M_WIDTH>* mem_in0,
		ap_uint<AXI_M_WIDTH>* mem_in1,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_out,
		unsigned int selector)
{
    dut(mem_in0, mem_in1, strm_out, sizeBytes, selector);

    ap_uint<AXI_M_WIDTH>* mem;
    mem = selector == 0 ? mem_in0 : mem_in1;

    bool no_error = true;

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
        {
            ap_axiu<AXIS_WIDTH, 0, 0, 0> inPkt = strm_out.read();

            for (int i = 0; i < data_per_pkt; i++)
            {
                int index = i + beat * data_per_beat + pkt * data_per_pkt;
                
                if (index >= num)
                    break;

                else if (mem[beat].range((i+1) * datasize_bits - 1 + pkt * AXIS_WIDTH, i * datasize_bits + pkt * AXIS_WIDTH) 
                                != inPkt.data.range((i+1) * datasize_bits - 1, i * datasize_bits))
                {
                    no_error = false;
                    std::cerr << "[ERROR] Value mismatch. Index: " << beat * data_per_beat + i 
                            << " mem val: " << mem[beat].range((i+1) * datasize_bits - 1 + pkt * AXIS_WIDTH, i * datasize_bits + pkt * AXIS_WIDTH)
                            << " stream val: " << inPkt.data.range((i+1) * datasize_bits - 1, i * datasize_bits) << std::endl;
                }

#ifdef DEBUG_LOG
                ops::hls::DataConv converter1, converter2;
                converter1.i = mem[beat].range((i+1) * datasize_bits - 1 + pkt * AXIS_WIDTH, i * datasize_bits + pkt * AXIS_WIDTH);
                converter2.i = inPkt.data.range((i+1) * datasize_bits - 1, i * datasize_bits);

                std::cout << "[DEBUG] Values at index: " << beat * data_per_beat + i 
                        << " mem val: " << converter1.f
                        << " stream val: " << converter2.f << std::endl;
#endif
            }
        }
    }

    if (no_error)
        return true;
    else
        return false;

    std::cout << std::endl;
}

int main()
{
    std::cout << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "TESTING: ops::hls::mem2stream" << std::endl;
    std::cout << "********************************************" << std::endl << std::endl;

    std::cout << "Size(Bytes): " << sizeBytes << std::endl;
    std::cout << "Number of total beats: " << num_beats << std::endl;

    ap_uint<AXI_M_WIDTH> mem0[num_beats];
    ap_uint<AXI_M_WIDTH> mem1[num_beats];
    hls::stream<ap_axiu<AXIS_WIDTH, 0, 0, 0>> stream;

    std::random_device rd;
    std::mt19937 e2(rd());

    std::uniform_real_distribution<> dist(0,100);

    ops::hls::DataConv converter;

#ifdef DEBUG_LOG
    std::cout << std:: endl << "[DEBUG] **** mem1 values ****" << std::endl; 
#endif

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int i = 0; i < data_per_beat; i++)
        {
            converter.f = dist(e2);
            mem0[beat].range((i+1)*datasize_bits - 1, i * datasize_bits) = converter.i;

#ifdef DEBUG_LOG
            int index =  i + beat * data_per_beat;
            std::cout << "[DEBUG] index: " << index << " value: " << converter.f << std::endl;
#endif
        }
    }

#ifdef DEBUG_LOG
    std::cout << std:: endl << "[DEBUG] **** mem1 values ****" << std::endl; 
#endif

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int i = 0; i < data_per_beat; i++)
        {
            converter.f = dist(e2);
            mem1[beat].range((i+1)*datasize_bits - 1, i * datasize_bits) = converter.i;

#ifdef DEBUG_LOG
            int index =  i + beat * data_per_beat;
            std::cout << "[DEBUG] index: " << index << " value: " << converter.f << std::endl;
#endif
        }
    }

    //calling test dut for selector = 0
    if (testDut(mem0, mem1, stream, 0))
        std::cout << "Selector 0: TEST PASSED" << std::endl;
    else
        std::cout << "Selector 0: TEST FAILED" << std::endl;

    std::cout << std::endl;

    //calling test dut for selector = 1
    if (testDut(mem0, mem1, stream, 1))
        std::cout << "Selector 1: TEST PASSED" << std::endl;
    else
        std::cout << "Selector 1: TEST FAILED" << std::endl;

    std::cout << std::endl;

    return 0;
}