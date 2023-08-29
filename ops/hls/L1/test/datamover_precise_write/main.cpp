#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

//#define DEBUG_LOG

int main()
{
    std::cout << std::endl;
    std::cout << "*****************************************************************************" << std::endl;
    std::cout << "TESTING: ops::hls::mem2stream -> ops::hls::stream2mem smoke precise range (CSIM ONLY)" << std::endl;
    std::cout << "*****************************************************************************" << std::endl << std::endl;

    std::random_device rd;
    unsigned int seed = 7;
    std::mt19937 mtSeeded(seed);
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<> distInt(5, 30);
    std::normal_distribution<float> distFloat(100, 10);
    ops::hls::DataConv converter;

    const int num_tests  = 3;
    std::cout << "TOTAL NUMER OF TESTS: " << num_tests << std::endl;
    std::vector<bool> test_summary(10);

    for (int test_itr = 0; test_itr < num_tests; test_itr++)
    {
        std::cout << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << " TEST " << test_itr << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << std::endl;
        
        const int x_size = distInt(mtSeeded);
        const int num = x_size * x_size * x_size; 
        const int size = num * sizeof(float);
        const int data_per_axis_pkt = AXIS_WIDTH / DATA_WIDTH;
        const int num_of_axis_pkts = (num + data_per_axis_pkt -1 ) / data_per_axis_pkt;
//        const int bytes_per_beat = AXI_M_WIDTH / 8;
//        const int data_per_beat = bytes_per_beat / sizeof(float);
//const int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;

        std::cout << "x_size: " << x_size << std::endl;
        std::cout << "number of elem: " << num << std::endl;
        std::cout << "Size(Bytes): " << size << std::endl;
//        std::cout << "Number of total beats: " << num_beats << std::endl;

        ap_uint<DATA_WIDTH> mem0[num];
        ap_uint<DATA_WIDTH> mem1[num];

#ifdef DEBUG_LOG
        std::cout << std:: endl << "[DEBUG] **** mem values ****" << std::endl; 
#endif
        for (int index = 0; index < num; index++)
        {
			converter.f = distFloat(mtRandom);
			mem0[index] = converter.i;
#ifdef DEBUG_LOG

			std::cout << "index: " << index << " value: " << converter.f << std::endl;
#endif

        }

        hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>> axis_in;

        for (int pkt = 0; pkt < num_of_axis_pkts; pkt++)
        {
        	ap_axiu<AXIS_WIDTH,0,0,0> tmp;

        	for (int i = 0; i < data_per_axis_pkt; i++)
        	{
        		int index = pkt * data_per_axis_pkt + i;

        		if (index < num)
        		{
        			tmp.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH) = mem0[index];
        			tmp.strb.range((i+1) * sizeof(float) - 1, i * sizeof(float)) = -1;
        		}
        		else
        		{
        			tmp.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH) = 0;
        			tmp.strb.range((i+1) * sizeof(float) - 1, i * sizeof(float)) = 0;
        		}
        	}
        	axis_in.write(tmp);
        }

        //calling test dut
        dut(axis_in, mem1, size);

        bool no_error = true;

        for (unsigned int index = 0; index < num; index++)
        {
			ops::hls::DataConv tmp1, tmp2;

			tmp1.i = mem0[index];
			tmp2.i = mem1[index];
#ifdef DEBUG_LOG
            std::cout << "[INFO] Verification. Index: " << index
                    << " mem0 val: " << tmp1.f << " mem1 val: " << tmp2.f  << std::endl;
#endif
			if (mem0[index] != mem1[index])
			{
				no_error = false;

				std::cerr << "[ERROR] Value mismatch. Index: " << index
						<< " mem0 val: " << tmp1.f << " mem1 val: " << tmp2.f  << std::endl;
			}
        }
        if (no_error)
        {
            std::cout << "TEST PASSED." << std::endl;
            test_summary[test_itr] = true;
        }
        else
        {
            std::cout << "TEST FAILED." << std::endl;
            test_summary[test_itr] = false;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << " TEST SUMMARY " << std::endl;
    std::cout << "**********************************" << std::endl;
    std::cout << std::endl;

    for (unsigned int test_itr = 0; test_itr < num_tests; test_itr++)
    {
        std::cout << "TEST " << test_itr <<": ";
        
        if (test_summary[test_itr])
            std::cout << "PASSED";
        else
            std::cout << "FAILED";

        std::cout << std::endl;
    }

    std::cout << std::endl;
    return 0;
}
