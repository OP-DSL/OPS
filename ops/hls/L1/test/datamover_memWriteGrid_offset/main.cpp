#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

// #define DEBUG_LOG 


static bool testDut(float* data,
		ops::hls::SizeType gridSize,
		unsigned short x_size)
{
    static hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>> strm_in;
    ap_uint<DATA_WIDTH> mem[gridSize[0] * gridSize[1] * gridSize[2]];

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
    constexpr unsigned int data_bytes = DATA_WIDTH / 8;

    unsigned int num = diff[0] * diff[1] * diff[2];
    unsigned int sizeBytes = num * sizeof(float);
	const unsigned int num_beats = (sizeBytes + bytes_per_beat - 1) / bytes_per_beat;

	std::cout << "[INFO] read offset: ("
			<< offset[0] << ", " << offset[1] << ", " << offset[2] << ")" << std::endl;
    std::cout << "[INFO] number of beats: " << num_beats << std::endl;
	std::cout << "[INFO] Size (Bytes): " << sizeBytes << std::endl;

    unsigned int initial_offset = offset[0] + offset[1] * gridSize[0] + offset[2] * gridSize[0] * gridSize[1];

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
        {
        	int pkt_idx = beat * data_per_beat + pkt * data_per_pkt;

            ap_axiu<AXIS_WIDTH, 0, 0, 0> outPkt;

        	if (pkt_idx < num)
            {
        		outPkt.keep = -1;

                for (int i = 0; i < data_per_pkt; i++)
                {
                    int index = i + pkt_idx;

                    if (index < num)
                    {
                        ops::hls::DataConv converter;
                        converter.f = data[initial_offset + index];
                        outPkt.data.range((i+1)*DATA_WIDTH - 1, i * DATA_WIDTH) =  converter.i;
 #ifdef DEBUG_LOG
                        std::cout << "[DEBUG] writing val to stream. index: " 
                                << index << ", value: " << converter.f << std::endl;
 #endif
                        outPkt.strb.range((i+1) * data_bytes - 1, i * data_bytes) = -1;
                    }
                    else
                    {
                        outPkt.data.range((i+1)*DATA_WIDTH - 1, i * DATA_WIDTH) = 0;
#ifdef DEBUG_LOG
                        std::cout << "[DEBUG] writing val to stream. index: " 
                                << index << ", value: " << 0 << std::endl;
#endif
                        outPkt.strb.range((i+1) * data_bytes - 1, i * data_bytes) = 0;
                    }
                }
                strm_in.write(outPkt);
            }
        	else
        		break;
        }
    }

    dut(mem, strm_in, sizeBytes, gridSize, offset);

    ap_uint<DATA_WIDTH>* mem_offsetted = mem + offset[0] + offset[1] * gridSize[0] + offset[2] * gridSize[0] * gridSize[1];

    bool no_error = true;

    for (int beat = 0; beat < num_beats; beat++)
    {
        for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
        {
        	int pkt_idx = beat * data_per_beat + pkt * data_per_pkt;

        	if (pkt_idx < num)
            {
                for (int i = 0; i < data_per_pkt; i++)
                {
                    int index = i + pkt_idx;
                    ops::hls::DataConv converter;   
                    converter.i = mem_offsetted[index];

                    if (index < num)
                    {
                        if (converter.f != data[initial_offset + index])
                        {
                            no_error = false;
                            std::cerr << "[ERROR] Value mismatch. Index: " << index
                                    << " mem val: " << converter.f
                                    << " data val: " << data[initial_offset + index] << std::endl;
                        }
                    }
                    else
                    {
                        if (converter.f != 0.0)
                        {
                            no_error = false;
                            std::cerr << "[ERROR] Padded value has to be zero. Index: " << index
                                    << " mem val: " << converter.f << std::endl;
                        }
                    }

    #ifdef DEBUG_LOG

                    std::cout << "[DEBUG] Values at index: " << index
                            << " mem val: " << converter.f
                            << " data val: " << data[initial_offset + index] << std::endl;
    #endif
                }
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
        grid_size[0] = x_beats * data_per_beat;
        grid_size[1] = x_size;
        grid_size[2] = x_size;
        const int num_elems = grid_size[0] * grid_size[1] * grid_size[2];
        const int grid_size_bytes = num_elems * sizeof(float);

        std::cout << "[INFO] x_size: " << x_size << std::endl;
        std::cout << "[INFO] x_beats: " << x_beats << std::endl;
        std::cout << "[INFO] grid size: (" << grid_size[0] <<", " << grid_size[1]
				<< ", " << grid_size[2] <<")" << std::endl;
        std::cout << "[INFO] Size(Bytes): " << grid_size_bytes << std::endl;

        float data[num_elems];

#ifdef DEBUG_LOG
        std::cout << std:: endl << "[DEBUG] **** mem values ****" << std::endl;
#endif
        for (int k = 0; k < grid_size[2]; k++)
        {
            for (int j = 0; j < grid_size[1]; j++)
            {
            	for (int i = 0; i < grid_size[0]; i++)
            	{
                unsigned int index = i + j * grid_size[0] + k * grid_size[0] * grid_size[1];

					data[index] = distFloat(mtRandom);
#ifdef DEBUG_LOG

					std::cout << "index: " << index << " value: " << data[index] << std::endl;
#endif

            	}
            }
        }


        bool no_error = testDut(data, grid_size, x_size);


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
