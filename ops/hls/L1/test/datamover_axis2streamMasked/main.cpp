#include <iostream>
#include <random>
#include <vector>
#include "top.hpp"

//#define DEBUG_LOG
static void initMem(ap_uint<DATA_WIDTH> * mem,
		const float initVal,
		ops::hls::SizeType gridSize)
{
	unsigned int numElem = gridSize[0] * gridSize[1] * gridSize[2];
	ops::hls::DataConv converter;
	converter.f = initVal;

	for (unsigned int i = 0; i < numElem; i++)
	{
		mem[i] = converter.i;
	}
}

static void printMem(ap_uint<DATA_WIDTH> * mem,
	ops::hls::SizeType gridSize,
	std::string printString)
{
#ifdef DEBUG_LOG
	unsigned int numElem = gridSize[0] * gridSize[1] * gridSize[2];
	ops::hls::DataConv converter;

    std::cout << std:: endl << "[DEBUG] **** mem value ****" << printString << std::endl;

	for (unsigned int i = 0; i < numElem; i++)
	{
		converter.i = mem[i];

		std::cout << " Index: " << i <<", val: " << converter.f << std::endl;
	}
#endif
}

static bool testDut(float* data,
		ops::hls::SizeType gridSize,
		unsigned short x_size)
{
    static hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>> strm_out;
    static hls::stream<ap_uint<STREAM_WIDTH>> data_in;
    static hls::stream<ap_uint<STREAM_WIDTH/8>> mask_in;
    ap_uint<DATA_WIDTH> mem[gridSize[0] * gridSize[1] * gridSize[2]];

    const float initVal = 2.0;
    initMem(mem, initVal, gridSize);

#ifdef DEBUG_LOG
    printMem(mem, gridSize, "After init");
#endif

	std::random_device rd;
    std::mt19937 mtRandom(rd());
    std::uniform_int_distribution<unsigned short> distLow(0, x_size/3);
    std::uniform_int_distribution<unsigned short> distHigh(x_size*2/3, x_size);
    std::uniform_int_distribution<unsigned short> distShift(0, 7);

	ops::hls::SizeType start = {distLow(mtRandom), distLow(mtRandom), distLow(mtRandom)}; //{5, 1, 5};
	ops::hls::SizeType end = {distHigh(mtRandom), distHigh(mtRandom), distHigh(mtRandom)}; //{15, 12, 10};

	ops::hls::SizeType diff;
	diff[0] = end[0] - start[0];
	diff[1] = end[1] - start[1];
	diff[2] = end[2] - start[2];

	ops::hls::AccessRange range;
	range.dim = 3;
	range.start[0] = start[0]; range.start[1] = start[1]; range.start[2] = start[2];
	range.end[0] = end[0]; range.end[1] = end[1]; range.end[2] = end[2];

	constexpr unsigned int num_pkts_per_beat = AXI_M_WIDTH / AXIS_WIDTH;
	constexpr unsigned int num_stream_pkts_per_beat = AXI_M_WIDTH / STREAM_WIDTH;
	constexpr unsigned int data_per_pkt = AXIS_WIDTH / DATA_WIDTH;
	constexpr unsigned int data_per_stram_pkt = STREAM_WIDTH / DATA_WIDTH;
	constexpr unsigned int data_per_beat = AXI_M_WIDTH / DATA_WIDTH;
    constexpr unsigned int bytes_per_beat = AXI_M_WIDTH / 8;
    constexpr unsigned int data_bytes = DATA_WIDTH / 8;
    unsigned short x_beats = (diff[0] + data_per_beat - 1) / data_per_beat;
    unsigned short x_pkts = (diff[0] + data_per_pkt - 1) / data_per_pkt;
    unsigned short x_stream_pkts = (diff[0] + data_per_stram_pkt - 1) / data_per_stram_pkt;
	const unsigned int num_beats = x_beats * diff[1] * diff[2];
	unsigned int num = num_beats * data_per_beat;
	unsigned int sizeBytes = num * sizeof(float);
	const unsigned int x_sizeBytes = x_pkts * data_per_pkt * data_bytes;
	ap_uint<data_bytes> zero = 0;
	ap_uint<data_bytes> one = -1;
	unsigned short shiftData = distShift(mtRandom);
	unsigned short shiftBytes = shiftData * data_bytes;

	std::cout << "[INFO] " << "write range: ("
			<< start[0] << ", " << start[1] << ", " << start[2] << ") --> ("
			<< end[0] << ", " << end[1] << ", " << end[2] << ")"<< std::endl;
	std::cout << "[INFO] " << "diff: ("
				<< diff[0] << ", " << diff[1] << ", " << diff[2] << ")" << std::endl;
	std::cout << "[INFO] x_beats: " << x_beats << std::endl;
	std::cout << "[INFO] x_pkts: " << x_pkts << std::endl;
	std::cout << "[INFO] shift: " << shiftData << std::endl;
	std::cout << "[INFO] Size (Bytes): " << sizeBytes << std::endl;

    unsigned int initial_offset = start[0] + start[1] * gridSize[0] + start[2] * gridSize[0] * gridSize[1];

    for (unsigned short k = 0; k <= diff[2]; k++)
    {
    	for (unsigned short j = 0; j <= diff[1]; j++)
    	{
    		for (unsigned short x_beat = 0; x_beat < x_beats; x_beat++)
    		{
    			for (int pkt = 0; pkt < num_pkts_per_beat; pkt++)
				{
    				int pkt_id = pkt + x_beat * num_pkts_per_beat;
    				ap_axiu<AXIS_WIDTH, 0, 0, 0> outPkt;

    				if (pkt_id < x_pkts)
    				{
        	            for (int data_i = 0; data_i < data_per_pkt; data_i++)
        	            {
        	            	int i = data_i + pkt * data_per_pkt + x_beat * data_per_beat;
        	                int index = i  + gridSize[0] * j + k * gridSize[0] * gridSize[1];
        	                ops::hls::DataConv converter;

        	                if (i < diff[0])
        	                {
        	                	converter.f = data[initial_offset + index];
        	                	outPkt.data.range((data_i+1) * DATA_WIDTH - 1, data_i * DATA_WIDTH) = converter.i;
        	                	outPkt.strb.range((data_i+1) * data_bytes - 1, data_i * data_bytes) = -1;
#ifdef DEBUG_LOG
                       std::cout << "[DEBUG] writing val to stream. i:" << i << ", j:" << j << ", k:" << k << ",index: "
                               << index << ", value: " << converter.f << std::endl;
#endif
        	                }
        	                else
        	                {
        	                	outPkt.data.range((data_i+1) * DATA_WIDTH - 1, data_i * DATA_WIDTH) = 0;
        	                	outPkt.strb.range((data_i+1) * data_bytes - 1, data_i * data_bytes) = 0;
#ifdef DEBUG_LOG
                        std::cout << "[DEBUG] writing val to stream with STRB OFF. index: "
                                << index << ", value: " << 0 << std::endl;
#endif
        	                }
        	            }
#ifdef DEBUG_LOG
						printf("   |HLS DEBUG_LOG||%s| write axis , val=(", __func__);

						for (unsigned n = 0; n < data_per_pkt; n++)
						{
							ops::hls::DataConv tmp;
							tmp.i = outPkt.data.range((n+1) * DATA_WIDTH - 1, n * DATA_WIDTH);
							printf("%f,", tmp.f);
						}

						printf(") strb=(%x)\n", outPkt.strb);
#endif
        	            strm_out.write(outPkt);
    				}
				}
			}
    	}
    }

    for (unsigned short k = 0; k <= diff[2]; k++)
    {
    	for (unsigned short j = 0; j <= diff[1]; j++)
    	{
    		dut(strm_out, data_in, mask_in, x_sizeBytes, shiftBytes);
    	}
    }

    printMem(mem, gridSize, "After mem write");

    ap_uint<DATA_WIDTH>* mem_offsetted = mem + initial_offset;

    bool no_error = true;

    for (unsigned short k = 0; k <= diff[2]; k++)
    {
    	for (unsigned short j = 0; j <= diff[1]; j++)
    	{
    		for (unsigned short x_beat = 0; x_beat < x_beats; x_beat++)
    		{
    			for (int pkt = 0; pkt < num_stream_pkts_per_beat; pkt++)
				{
    				int pkt_id = pkt + x_beat * num_stream_pkts_per_beat;
    				ap_uint<STREAM_WIDTH> dataPkt;
    				ap_uint<STREAM_WIDTH/8> maskPkt;

    				bool isFirstPkt = (x_beat==0 and pkt == 0);

    				if (isFirstPkt)
    				{
    					dataPkt = data_in.read();
    					maskPkt = mask_in.read();

        	            for (int data_i = 0; data_i < shiftData; data_i++)
        	            {
        	            	int i = data_i + pkt * data_per_stram_pkt + x_beat * data_per_beat;
        	                int index = i  + gridSize[0] * j + k * gridSize[0] * gridSize[1];
        	                ops::hls::DataConv converter;

							if (i < diff[0])
							{
								converter.i = dataPkt.range((data_i+1) * DATA_WIDTH - 1, data_i * DATA_WIDTH);

								if (converter.f != data[initial_offset + index])
								{
									no_error = false;
									std::cerr << "[ERROR]|" << __func__<< "|" << __LINE__ << "| " <<"Value mismatch. Index: " << index
											<< ", offseted_index: " << index + initial_offset
											<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
											<< ", stream val: " << converter.f
											<< ", data val: " << data[initial_offset + index] << std::endl;
								}
								if (maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes) != one)
								{
									no_error = false;
									std::cerr << "[ERROR]|" << __func__<< "|" << __LINE__ << "| " << "mask mismatch. it has to be all 1. Index: " << index
											<< ", offseted_index: " << index + initial_offset
											<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
											<< ", stream mask: " << maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes)
											<< ", data val: " << data[initial_offset + index] << std::endl;
								}

							}

#ifdef DEBUG_LOG
							std::cout << "[DEBUG] Index: " << index
									<< ", offseted_index: " << index + initial_offset
									<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
									<< ", stream mask: " << maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes)
									<< ", data val: " << data[initial_offset + index] << std::endl;
    #endif
        	            }
    				}

    				else if (pkt_id < x_pkts)
    				{
    					dataPkt = data_in.read();
    					maskPkt = mask_in.read();

        	            for (int data_i = 0; data_i < data_per_stram_pkt; data_i++)
        	            {
        	            	int i = data_i + pkt * data_per_stram_pkt + x_beat * data_per_beat;
        	                int index = i  + gridSize[0] * j + k * gridSize[0] * gridSize[1];
        	                ops::hls::DataConv converter;

        	                if (i < diff[0] + shiftData)
        	                {
        	                	converter.i = dataPkt.range((data_i+1) * DATA_WIDTH - 1, data_i * DATA_WIDTH);

        	                	if (converter.f != data[initial_offset + index - shiftData])
        	                	{
        	                		no_error = false;
									std::cerr << "[ERROR]|" << __func__<< "|" << __LINE__ << "| " << " Value mismatch. Index: " << index
											<< ", offseted_index: " << index + initial_offset
											<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
											<< ", stream val: " << converter.f
											<< ", data val: " << data[initial_offset + index - shiftData] << std::endl;
        	                	}
        	                	if (maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes) != one)
        	                	{
        	                		no_error = false;
									std::cerr << "[ERROR]|" << __func__<< "|" << __LINE__ << "| " << " mask mismatch. it has to be all 1. Index: " << index
											<< ", offseted_index: " << index + initial_offset
											<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
											<< ", stream mask: " << maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes)
											<< ", data val: " << data[initial_offset + index - shiftData] << std::endl;
        	                	}

        	                }
        	                else
        	                {
        	                	if (maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes) != zero)
        	                	{
        	                		no_error = false;
									std::cerr << "[ERROR] mask mismatch. it has to be all 0. Index: " << index
											<< ", offseted_index: " << index + initial_offset
											<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
											<< ", stream mask: " << maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes)
											<< ", data val: " << data[initial_offset + index] << std::endl;
        	                	}
        	                }
#ifdef DEBUG_LOG
							std::cout << "[DEBUG] Index: " << index
									<< ", offseted_index: " << index + initial_offset
									<< ", (i:"<< i << ", j:" << j <<", k:" << k << ")"
									<< ", stream mask: " << maskPkt.range((data_i + 1) * data_bytes -1, data_i * data_bytes)
									<< ", data val: " << data[initial_offset + index] << std::endl;
#endif
        	            }
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
    std::uniform_int_distribution<unsigned short> distInt(5, 30);
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

        const unsigned short x_size = distInt(mtSeeded); //16;
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

					data[index] = index; // distFloat(mtRandom);
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
