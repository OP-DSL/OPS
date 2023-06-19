#include <iostream>
#include <stdlib.h>
#include "top.hpp"



int main()
{
    std::cout << std::endl;
    std::cout << "***********************************************************************" << std::endl;
    std::cout << "TESTING: ops::hls::mem2stream -> ops::hls::stream2mem round (CSIM ONLY)" << std::endl;
    std::cout << "***********************************************************************" << std::endl << std::endl;

    const int num = 22 * 22 * 22; //20 x 20 x 20 grid with (-1,+1) halo with 4-byte datatype in bytes
    const int size = num * sizeof(float);
    const int bytes_per_beat = AXI_M_WIDTH / 8;
    const int data_per_beat = AXI_M_WIDTH / sizeof(float); 
    const int num_beats = (size + bytes_per_beat - 1) / bytes_per_beat;

    std::cout << "Size(Bytes): " << size << std::endl;
    std::cout << "Number of total beats: " << num_beats << std::endl;

    ap_uint<AXI_M_WIDTH> mem0[num_beats];
    ap_uint<AXI_M_WIDTH> mem1[num_beats];
    hls::stream<ap_axiu<AXIS_WIDTH, 0, 0, 0>> stream;

    unsigned int seed = 7;
    srand(seed);

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int i = 0; i < data_per_beat; i++)
        {
            mem0[beat].range((i+1)*sizeof(float) - 1, i * sizeof(float)) = rand();
        }
    }

    //calling test dut
    dut(mem0, mem1, stream, size);

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
        std::cout << "TEST PASSED." << std::endl;
    else
        std::cout << "TEST FAILED." << std::endl;

    std::cout << std::endl;
    return 0;
}