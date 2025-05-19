#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

//#define DEBUG_LOG
bool first = true;

static bool testDut(ap_uint<DATA_WIDTH>* mem,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_out,
		ops::hls::SizeType gridSize,
		unsigned short x_size,
		bool full_Range = false)
{
	std::random_device rd;
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<unsigned short> distLow(0, x_size/3);
    std::uniform_int_distribution<unsigned short> distHigh(x_size*2/3, x_size);

	ops::hls::SizeType start;
	ops::hls::SizeType end;

	if (not full_Range)
	{
		start[0] = distLow(mtRandom);
		start[1] = distLow(mtRandom);
		start[2] = distLow(mtRandom); //{2,10,5}; //
		end[0] = distHigh(mtRandom);
		end[1] = distHigh(mtRandom);
		end[2] = distHigh(mtRandom); //{38,35,38}; //
	}
	else
	{
		start[0] = 0;
		start[1] = 0;
		start[2] = 0;
		end[0] = x_size;
		end[1] = gridSize[1];
		end[2] = gridSize[2];
	}

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

	constexpr unsigned short data_vector_factor = AXI_M_WIDTH / DATA_WIDTH;
	unsigned short ShiftBits = (unsigned short)LOG2(data_vector_factor);
	unsigned short DataShiftBits = (unsigned short)LOG2(DATA_WIDTH/8);
	unsigned short start_x = range.start[0] >> ShiftBits;
	unsigned short end_x = (range.end[0] + data_vector_factor - 1) >> ShiftBits;
	unsigned short grid_xblocks = gridSize[0] >> ShiftBits; //GridSize[0] has to be MEM_DATA_WIDTH aligned
	unsigned short num_xblocks = end_x - start_x;


    unsigned short x_beats = end_x - start_x;
    unsigned short x_pkts = x_beats * num_pkts_per_beat;
	const unsigned int num_beats = x_beats * diff[1] * diff[2];
	unsigned int num = num_beats * data_per_beat;
	unsigned int sizeBytes = num * sizeof(float);

	std::cout << "[INFO] " << "read range: ("
			<< start[0] << ", " << start[1] << ", " << start[2] << ") --> ("
			<< end[0] << ", " << end[1] << ", " << end[2] << ")"<< std::endl;
	std::cout << "[INFO] " << "start_x: " << start_x << ", end_x" << end_x << std::endl;
	std::cout << "[INFO] x_beats: " << x_beats << std::endl;
	std::cout << "[INFO] x_pkts: " << x_pkts << std::endl;
	std::cout << "[INFO] Size (Bytes): " << sizeBytes << std::endl;

    dut((ap_uint<AXI_M_WIDTH>*)mem, strm_out, gridSize, range);

    bool no_error = true;

    for (unsigned int k = start[2]; k < end[2]; k++)
    {
    	for (unsigned int j = start[1]; j < end[1]; j++)
    	{
    		for (unsigned int x_beat = start_x; x_beat < end_x; x_beat++)
    		{
    			for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
				{
    				int x_pkt_id = pkt + x_beat * num_pkts_per_beat;
//    				std::cout << "[INFO] x_beat: " << x_beat << ", j: " << j <<", k: " << k << "pkt: " << pkt << "x_pkt_id: " << x_pkt_id << std::endl;

    				ap_axiu<AXIS_WIDTH, 0, 0, 0> inPkt;
    	        	if (x_pkt_id < x_pkts)
    	        		inPkt  = strm_out.read();
    	        	else
    	        		break;

    	            for (int i = 0; i < data_per_pkt; i++)
    	            {
    	            	int proper_i = i + pkt * data_per_pkt + x_beat * data_per_beat;
    	                int index = proper_i  + gridSize[0] * j + k * gridSize[0] * gridSize[1];

    	                if (index >= num)
    	                    break;

    	                else if (mem[index]
    	                                != inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH))
    	                {
    	                    no_error = false;
    	                    ops::hls::DataConv converter1, converter2;
    	                    converter1.i = mem[index];
							converter2.i = inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH);

    	                    std::cerr << "[ERROR] Value mismatch. Index: " << index << " (proper_i:"
    	                    		<< proper_i <<", j:" << j <<", k:" << k << "), pkt: " << pkt << ", x_pkt_id: " << x_pkt_id << ", x_beat: " << x_beat
    	                            << " mem val: " << converter1.f
    	                            << " stream val: " << converter2.f << std::endl;
    	                }

//#ifdef DEBUG_LOG
    	                else if(first)
    	                {
							ops::hls::DataConv converter1, converter2;
							converter1.i = mem[index];
							converter2.i = inPkt.data.range((i+1) * DATA_WIDTH - 1, i * DATA_WIDTH);

							std::cout << "[DEBUG] Values at index: " << index << " (" << proper_i <<", " << j <<", " << k << ")"
									<< " mem val: " << converter1.f
									<< " stream val: " << converter2.f << std::endl;
    	                }
//#endif
    	            }
				}
    		}
    	}
    }
    std::cout << std::endl;
    first = false;
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
    std::uniform_int_distribution<unsigned short> distInt(5, 60);
    std::normal_distribution<float> distFloat(100, 10);
    ops::hls::DataConv converter;

    const int num_tests  = 10;
    std::cout << "TOTAL NUMER OF TESTS: " << num_tests << std::endl;
    std::vector<bool> test_summary(10);

    for (int test_itr = 0; test_itr < num_tests; test_itr++)
    {
        std::cout << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << " TEST " << test_itr << (test_itr >= num_tests/2 ? " Continuous range" :  "") << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << std::endl;

        const unsigned short x_size = distInt(mtSeeded); //40; //
        const unsigned short bytes_per_beat = AXI_M_WIDTH / 8;
        const unsigned short data_per_beat = bytes_per_beat / sizeof(float);
        const unsigned short x_beats = (x_size + data_per_beat - 1) / data_per_beat;
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
            	for (int i = 0; i < grid_size[0]; i++)
            	{
                unsigned int index = i + j * grid_size[0] + k * grid_size[0] * grid_size[1];

					converter.f = distFloat(mtRandom); //index; //
					mem0[index] = converter.i;
#ifdef DEBUG_LOG

					std::cout << "linear index: " << index << ", index: (" << i << ", " << j << ", " << k << "), value: " << converter.f << std::endl;
#endif

            	}
            }
        }


        bool no_error = testDut(mem0, stream, grid_size, x_size, test_itr << test_itr >= num_tests/2 ? true: false);


        if (no_error)
        {
            std::cout << "TEST PASSED." << std::endl;
            test_summary[test_itr] = true;
        }
        else
        {
			std::cout << std:: endl << "[DEBUG] **** mem values ****" << std::endl;

			for (int k = 0; k < grid_size[2]; k++)
			{
				for (int j = 0; j < grid_size[1]; j++)
				{
					for (int i = 0; i < grid_size[0]; i++)
					{
						unsigned int index = i + j * grid_size[0] + k * grid_size[0] * grid_size[1];
						converter.i = mem0[index];
						std::cout << "linear index: " << index << ", index: (" << i << ", " << j << ", " << k << "), value: " << converter.f << std::endl;
					}
				}
			}

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
