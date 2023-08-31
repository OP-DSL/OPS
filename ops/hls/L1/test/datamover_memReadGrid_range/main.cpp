#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

//#define DEBUG_LOG

static bool testDut(ap_uint<DATA_WIDTH>* mem,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_out,
		ops::hls::SizeType gridSize,
		unsigned short x_size)
{
	std::random_device rd;
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<unsigned short> distLow(0, x_size/3);
    std::uniform_int_distribution<unsigned short> distHigh(x_size*2/3, x_size);

	ops::hls::SizeType start = {distLow(mtRandom), distLow(mtRandom), distLow(mtRandom)}; //{2,10,5}; //
	ops::hls::SizeType end = {distHigh(mtRandom), distHigh(mtRandom), distHigh(mtRandom)}; //{38,35,38}; //

	ops::hls::SizeType diff;
	diff[0] = end[0] - start[0];
	diff[1] = end[1] - start[1];
	diff[2] = end[2] - start[2];
	ops::hls::AccessRange range;
	range.dim = 3;
	range.start[0] = start[0]; range.start[1] = start[1]; range.start[2] = start[2];
	range.end[0] = end[0]; range.end[1] = end[1]; range.end[2] = end[2];

	constexpr unsigned short num_pkts_per_beat = AXI_M_WIDTH / AXIS_WIDTH;
	constexpr unsigned short data_per_pkt = AXIS_WIDTH / DATA_WIDTH;
	constexpr unsigned short data_per_beat = AXI_M_WIDTH / DATA_WIDTH;
    constexpr unsigned short bytes_per_beat = AXI_M_WIDTH / 8;
    unsigned short x_beats = (diff[0] + data_per_beat - 1) / data_per_beat;
    unsigned short x_pkts = (diff[0] + data_per_pkt - 1) / data_per_pkt;
	const unsigned int num_beats = x_beats * diff[1] * diff[2];
	unsigned int num = num_beats * data_per_beat;
	unsigned int sizeBytes = num * sizeof(float);

	std::cout << "[INFO] " << "read range: ("
			<< start[0] << ", " << start[1] << ", " << start[2] << ") --> ("
			<< end[0] << ", " << end[1] << ", " << end[2] << ")"<< std::endl;
	std::cout << "[INFO] " << "diff: ("
				<< diff[0] << ", " << diff[1] << ", " << diff[2] << ")" << std::endl;
	std::cout << "[INFO] x_beats: " << x_beats << std::endl;
	std::cout << "[INFO] x_pkts: " << x_pkts << std::endl;
	std::cout << "[INFO] Size (Bytes): " << sizeBytes << std::endl;

    dut(mem, strm_out, gridSize, range);

    ap_uint<DATA_WIDTH>* mem_offsetted = mem + start[0] + start[1] * gridSize[0] + start[2] * gridSize[0] * gridSize[1];

    bool no_error = true;

    for (unsigned short k = 0; k < diff[2]; k++)
    {
    	for (unsigned short j = 0; j < diff[1]; j++)
    	{
    		for (unsigned short x_beat = 0; x_beat < x_beats; x_beat++)
    		{
    			for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
				{
    				int pkt_id = pkt + x_beat * num_pkts_per_beat;

    				ap_axiu<AXIS_WIDTH, 0, 0, 0> inPkt;
    	        	if (pkt_id < x_pkts)
    	        		inPkt  = strm_out.read();
    	        	else
    	        		break;

    	            for (int i = 0; i < data_per_pkt; i++)
    	            {
    	            	int proper_i = i + pkt * data_per_pkt + x_beat * data_per_beat;
    	                int index = proper_i  + gridSize[0] * j + k * gridSize[0] * gridSize[1];

    	                if (index >= num)
    	                    break;

    	                else if (mem_offsetted[index]
    	                                != inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH))
    	                {
    	                    no_error = false;
    	                    ops::hls::DataConv converter1, converter2;
    	                    converter1.i = mem_offsetted[index];
							converter2.i = inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH);

    	                    std::cerr << "[ERROR] Value mismatch. Index: " << index << " (" << proper_i <<", " << j <<", " << k << ")"
    	                            << " mem val: " << converter1.f
    	                            << " stream val: " << converter2.f << std::endl;
    	                }

#ifdef DEBUG_LOG
    	                else
    	                {
							ops::hls::DataConv converter1, converter2;
							converter1.i = mem_offsetted[index];
							converter2.i = inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH);

							std::cout << "[DEBUG] Values at index: " << index << " (" << proper_i <<", " << j <<", " << k << ")"
									<< " mem val: " << converter1.f
									<< " stream val: " << converter2.f << std::endl;
    	                }
#endif
    	            }
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

        const unsigned short x_size = distInt(mtSeeded); //40; //
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

					converter.f = distFloat(mtRandom); //index; //
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
