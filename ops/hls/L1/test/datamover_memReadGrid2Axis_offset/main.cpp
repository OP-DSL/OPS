#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

// #define DEBUG_LOG 

static bool testDut(ap_uint<DATA_WIDTH>* mem,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_out,
		ops::hls::SizeType gridSize,
		unsigned short x_size)
{
	std::random_device rd;
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<unsigned short> distLow(0, x_size/3);
//    std::uniform_int_distribution<unsigned short> distHigh(x_size*2/3, x_size);

	ops::hls::SizeType offset = {distLow(mtRandom), distLow(mtRandom), distLow(mtRandom)};
	ops::hls::SizeType diff;
	diff[0] = gridSize[0] - offset[0];
	diff[1] = gridSize[1] - offset[1];
	diff[2] = gridSize[2] - offset[2];

	constexpr unsigned int num_pkts_per_beat = AXI_M_WIDTH / AXIS_WIDTH;
	constexpr unsigned int data_per_pkt = AXIS_WIDTH / DATA_WIDTH;
	constexpr unsigned int data_per_beat = AXI_M_WIDTH / DATA_WIDTH;
    constexpr unsigned int bytes_per_beat = AXI_M_WIDTH / 8;
    unsigned int num = diff[0] * diff[1] * diff[2];
    unsigned int sizeBytes = num * sizeof(float);
	const unsigned int num_beats = (sizeBytes + bytes_per_beat - 1) / bytes_per_beat;

	std::cout << "[INFO] " << "read offset: ("
			<< offset[0] << ", " << offset[1] << ", " << offset[2] << ")" << std::endl;
	std::cout << "Size (Bytes): " << sizeBytes << std::endl;

    dut(mem, strm_out, sizeBytes, gridSize, offset);

    ap_uint<DATA_WIDTH>* mem_offsetted = mem + offset[0] + offset[1] * gridSize[0] + offset[2] * gridSize[0] * gridSize[1];

    bool no_error = true;

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
        {
        	int pkt_idx = beat * data_per_beat + pkt * data_per_pkt;

        	ap_axiu<AXIS_WIDTH, 0, 0, 0> inPkt;
        	if (pkt_idx < num)
        		inPkt  = strm_out.read();
        	else
        		break;

            for (int i = 0; i < data_per_pkt; i++)
            {
                int index = i + pkt_idx;

                if (index >= num)
                    break;

                else if (mem_offsetted[index]
                                != inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH))
                {
                    no_error = false;
                    std::cerr << "[ERROR] Value mismatch. Index: " << beat * data_per_beat + i
                            << " mem val: " << mem_offsetted[index]
                            << " stream val: " << inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH) << std::endl;
                }

#ifdef DEBUG_LOG
                ops::hls::DataConv converter1, converter2;
                converter1.i = mem_offsetted[index];
                converter2.i = inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH);

                std::cout << "[DEBUG] Values at index: " << index
                        << " mem val: " << converter1.f
                        << " stream val: " << converter2.f << std::endl;
#endif
            }
        }
    }
    std::cout << std::endl;

    if (no_error)
        return true;
    else
        return false;
}

int main()
{
    std::cout << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "TESTING: ops::hls::memReadGrid2Axis impl" << std::endl;
    std::cout << "********************************************" << std::endl << std::endl;

    std::random_device rd;
    unsigned int seed = 7;
    std::mt19937 mtSeeded(seed);
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<unsigned short> distInt(5, 40);
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

        const unsigned short x_size = distInt(mtSeeded);
        const unsigned short bytes_per_beat = AXI_M_WIDTH / 8;
        const unsigned short data_per_beat = bytes_per_beat / sizeof(float);
        const unsigned short x_beats = (x_size + bytes_per_beat - 1) / bytes_per_beat;
        ops::hls::SizeType grid_size;
        grid_size[0] = x_beats * bytes_per_beat;
        grid_size[1] = x_size;
        grid_size[2] = x_size;
        const int num_elems = grid_size[0] * grid_size[1] * grid_size[2];
        const int grid_size_bytes = num_elems * sizeof(float);

        std::cout << "[INFO] x_size: " << x_size << std::endl;
        std::cout << "[INFO] grid size: (" << grid_size[0] <<", " << grid_size[1]
				<< ", " << grid_size[2] <<")" << std::endl;
        std::cout << "[INFO] Size(Bytes): " << grid_size_bytes << std::endl;

        ap_uint<DATA_WIDTH> mem0[num_elems];
        hls::stream<ap_axiu<AXIS_WIDTH, 0, 0, 0>> stream;

#ifdef DEBUG_LOG
        std::cout << std:: endl << "[DEBUG] **** mem values ****" << std::endl;
#endif
        for (int k = 0; k < grid_size[2]; k++)
        {
            for (int j = 0; j < grid_size[1]; j++)
            {
            	for (int i = 0; i < x_size; i++)
            	{
                unsigned int index = i + j * grid_size[0] + k * grid_size[0] * grid_size[1];

					converter.f = distFloat(mtRandom);
					mem0[index] = converter.i;
#ifdef DEBUG_LOG

					std::cout << "index: " << index << " value: " << converter.f << std::endl;
#endif

            	}
            }
        }


        bool no_error = testDut(mem0, stream, grid_size, x_size);


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
