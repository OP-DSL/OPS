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


static bool testDut(float* originalData,
                    ap_uint<AXI_M_WIDTH>* mem_out0,
                    ap_uint<AXI_M_WIDTH>* mem_out1,
		            hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm,
                    unsigned int selector)
{
    ops::hls::DataConv converter;

    ap_uint<AXI_M_WIDTH>* mem;
    mem = selector == 0 ? mem_out0 : mem_out1;

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
        {
            if (beat * data_per_beat + pkt * data_per_pkt >= num)
                break;

            ap_axiu<AXIS_WIDTH, 0, 0, 0> inPkt;

            for (int i = 0; i < data_per_pkt; i++)
            {
                unsigned int index = beat * data_per_beat + pkt * data_per_pkt + i;

                if (index >= sizeBytes)
                    break;

                converter.f = originalData[index];
                inPkt.data.range((i+1) * datasize_bits - 1, i * datasize_bits) = converter.i;
            }

            strm.write(inPkt);
        }
    }

    dut(mem_out0, mem_out1, strm, sizeBytes, selector);

    bool no_error = true;

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int i = 0; i < data_per_beat; i++)
        {
            int index = i + beat * data_per_beat;
                
            if (index >= num)
                break;

            converter.i = mem[beat].range((i+1) * datasize_bits - 1, i * datasize_bits);
            
            if (abs(converter.f - originalData[index]) >= EPSILON)
            {
                no_error = false;
                std::cerr << "[ERROR] Value mismatch. Index: " << beat * data_per_beat + i 
                        << " mem val: " << converter.f << " stream val: " << originalData[index] << std::endl;
            }

#ifdef DEBUG_LOG
            std::cout << "[DEBUG] Values at index: " << beat * data_per_beat + i 
                    << " mem val: " << converter.f << " stream val: " << originalData[index] << std::endl;
#endif
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
    std::cout << "TESTING: ops::hls::mem2stream 1 mem impl" << std::endl;
    std::cout << "********************************************" << std::endl << std::endl;

    std::cout << "Size(Bytes): " << sizeBytes << std::endl;
    std::cout << "Number of total beats: " << num_beats << std::endl;

    float data[num];
    ap_uint<AXI_M_WIDTH> mem0[num_beats];
    ap_uint<AXI_M_WIDTH> mem1[num_beats];
    hls::stream<ap_axiu<AXIS_WIDTH, 0, 0, 0>> stream;

    std::random_device rd;
    std::mt19937 e2(rd());

    std::uniform_real_distribution<> dist(0,100);

    ops::hls::DataConv converter;

#ifdef DEBUG_LOG
    std::cout << std:: endl << "[DEBUG] **** mem0 values ****" << std::endl; 
#endif

    for (unsigned int i = 0; i < num; i++)
    {
        data[i] = dist(e2);
#ifdef DEBUG_LOG
        std::cout << "[DEBUG] index: " << i << " value: " << data[i] << std::endl;
#endif
    }

    //calling test dut for selector = 0
    if (testDut(data, mem0, mem1, stream, 0))
        std::cout << "Selector 0: TEST PASSED" << std::endl;
    else
        std::cout << "Selector 0: TEST FAILED" << std::endl;

    std::cout << std::endl;

#ifdef DEBUG_LOG
    std::cout << std:: endl << "[DEBUG] **** mem1 values ****" << std::endl; 
#endif

    for (unsigned int i = 0; i < num; i++)
    {
        data[i] = dist(e2);
#ifdef DEBUG_LOG
        std::cout << "[DEBUG] index: " << i << " value: " << data[i] << std::endl;
#endif
    }
    
    //calling test dut for selector = 1
    if (testDut(data, mem0, mem1, stream, 1))
        std::cout << "Selector 1: TEST PASSED" << std::endl;
    else
        std::cout << "Selector 1: TEST FAILED" << std::endl;

    std::cout << std::endl;

    return 0;
}
