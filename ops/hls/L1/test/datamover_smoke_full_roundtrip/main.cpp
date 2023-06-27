#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

int main()
{
    std::cout << std::endl;
    std::cout << "*****************************************************************************" << std::endl;
    std::cout << "TESTING: ops::hls::mem2stream -> ops::hls::stream2mem 2mem smoke (CSIM ONLY)" << std::endl;
    std::cout << "*****************************************************************************" << std::endl << std::endl;

    std::random_device rd;
    unsigned int seed = 7;
    std::mt19937 mtSeeded(seed);
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<> distInt(5, 40);
    std::normal_distribution<float> distFloat(100, 10);
    ops::hls::DataConv converter;

    const int num_tests  = 10;
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
        const int bytes_per_beat = AXI_M_WIDTH / 8;
        const int data_per_beat = AXI_M_WIDTH / sizeof(float); 
        const int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;

        std::cout << "x_size: " << x_size << std::endl;
        std::cout << "Size(Bytes): " << size << std::endl;
        std::cout << "Number of total beats: " << num_beats << std::endl;

        ap_uint<AXI_M_WIDTH> mem0[num_beats];
        ap_uint<AXI_M_WIDTH> mem1[num_beats];

#ifdef DEBUG_LOG
        std::cout << std:: endl << "[DEBUG] **** mem values ****" << std::endl; 
#endif
        for (int beat = 0; beat < num_beats; beat++)
        {
            for (int i = 0; i < data_per_beat; i++)
            {
                converter.f = distFloat(mtRandom);
                mem0[beat].range((i+1)*sizeof(float) - 1, i * sizeof(float)) = converter.i;

#ifdef DEBUG_LOG
                unsigned int index = beat * data_per_beat + i;
                std::cout << "index: " << index << " value: " << converter.f << std::endl; 
#endif
            }
        }

        //calling test dut
        dut(mem0, mem1, size);

        bool no_error = true;

        for (int beat = 0; beat < num_beats; beat++)
        {
            for (int i = 0; i < data_per_beat; i++)
            {
                int index = beat * data_per_beat + i;

                if (index < num)
                {
                    if (mem0[beat].range((i+1)*sizeof(float) - 1, i * sizeof(float)) != mem1[beat].range((i+1)*sizeof(float) - 1, i * sizeof(float)))
                    {
                        no_error = false;
                        std::cerr << "[ERROR] Value mismatch. Index: " << beat * data_per_beat + i 
                        << " mem0 val: " << (float) mem0[beat].range((i+1)*sizeof(float) - 1, i * sizeof(float))
                        << " mem1 val: " << (float) mem1[beat].range((i+1)*sizeof(float) - 1, i * sizeof(float)) << std::endl;
                    }
                }
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