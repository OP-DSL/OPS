
#include "top.hpp"
//#define DEBUG_LOG

void dut(ops::hls::GridPropertyCore& gridProp, Stencil2D & cross_stencil, stencil_type* data_in, stencil_type* data_out)
{
    ops::hls::DataConv converterInput;
    typedef ap_uint<vector_factor * sizeof(stencil_type) * 8> widen_dt;
    typedef hls::stream<widen_dt> stream_dt;
    stencil_type coef[] = {0.125 , 0.125 , 0.5, 0.125, 0.125};

    static stream_dt in_stream, out_stream;
    static ::hls::stream<stencil_type> input_bus_1[vector_factor];
    static ::hls::stream<stencil_type> input_bus_2[vector_factor];
    static ::hls::stream<stencil_type> input_bus_3[vector_factor];
    static ::hls::stream<stencil_type> input_bus_4[vector_factor];
    static ::hls::stream<stencil_type> input_bus_5[vector_factor];
    static ::hls::stream<stencil_type> output_bus[vector_factor];
    static ::hls::stream<stencil_type> alt_bus[vector_factor];
    const unsigned int kernel_iterations = gridProp.total_itr;
    ::ops::hls::GridPropertyCore gridPropCpy = gridProp;


#pragma HLS STREAM variable = input_bus_1 depth = 8
#pragma HLS STREAM variable = input_bus_2 depth = 8
#pragma HLS STREAM variable = input_bus_3 depth = 8
#pragma HLS STREAM variable = input_bus_4 depth = 8
#pragma HLS STREAM variable = input_bus_5 depth = 8

#pragma HLS STREAM variable = output_bus depth = 8
#pragma HLS STREAM variable = alt_bus depth = 8
#pragma HLS STREAM variable= in_stream depth= max_depth_v8
#pragma HLS STREAM variable= out_stream depth= max_depth_v8

//#pragma HLS DATAFLOW
#ifdef DEBUG_LOG
                printf("[DEBUG] stream write from mem \n");
                printf("------------------------------\n\n");
#endif

    for (unsigned short row = 0; row < gridProp.grid_size[1]; row++)
    {
        for (unsigned short block = 0; block < gridProp.xblocks; block++)
        {
            widen_dt stream_pkt;

            for (unsigned short i = 0; i < vector_factor; i++)
            {
                unsigned short index = row * gridProp.grid_size[0] + block * vector_factor + i;
                converterInput.f = data_in[index];
                stream_pkt.range(cross_stencil.s_datatype_size * (i + 1) - 1, cross_stencil.s_datatype_size * i) = converterInput.i;
#ifdef DEBUG_LOG
                printf("[DEBUG][MEM2STREAM] row(%d), block(%d), i(%d): index(%d) - %f\n", row, block, i, index, converterInput.f);
#endif
            }

            in_stream.write(stream_pkt);
        }
    }

#ifdef DEBUG_LOG
                printf("[DEBUG] calling stencil kernel \n");
                printf("------------------------------\n\n");
#endif


    cross_stencil.stencilRead(in_stream, input_bus_1, input_bus_2, input_bus_3, input_bus_4, input_bus_5);
	kernel(coef, input_bus_1, input_bus_2, input_bus_3, input_bus_4, input_bus_5, output_bus, alt_bus, kernel_iterations);
	cross_stencil.stencilWrite(out_stream, output_bus, alt_bus);

    
#ifdef DEBUG_LOG
                printf("[DEBUG] mem write from stream \n");
                printf("------------------------------\n\n");
#endif

    for (unsigned short row = 0; row < gridProp.grid_size[1]; row++)
    {
        for (unsigned short block = 0; block < gridProp.xblocks; block++)
        {
            widen_dt stream_pkt = out_stream.read();

            for (unsigned short i = 0; i < vector_factor; i++)
            {
                unsigned short index = row * gridProp.grid_size[0] + block * vector_factor + i;
                converterInput.i = stream_pkt.range(cross_stencil.s_datatype_size * (i + 1) - 1, cross_stencil.s_datatype_size * i);
                data_out[index] = converterInput.f;

#ifdef DEBUG_LOG
                printf("[DEBUG][STREAM2MEM] row(%d), block(%d), i(%d): index(index) - %f\n", row, block, i, index, data_out[index]);
#endif
            }
        }
    }
    
}

void kernel(stencil_type * coef, 
		::hls::stream<stencil_type> input_bus_0[vector_factor],
		::hls::stream<stencil_type> input_bus_1[vector_factor],
	    ::hls::stream<stencil_type> input_bus_2[vector_factor],
	    ::hls::stream<stencil_type> input_bus_3[vector_factor],
	    ::hls::stream<stencil_type> input_bus_4[vector_factor],
        ::hls::stream<stencil_type> output_bus[vector_factor],
		::hls::stream<stencil_type> alt_bus[vector_factor],
		const unsigned int num_iter)
{
	for (unsigned int iter = 0; iter < num_iter; iter++)
	{
		for (unsigned int k = 0; k < vector_factor; k++)
		{
			stencil_type r1 = input_bus_0[k].read();
			stencil_type r2 = input_bus_1[k].read();
			stencil_type r3 = input_bus_2[k].read();
			stencil_type r4 = input_bus_3[k].read();
			stencil_type r5 = input_bus_4[k].read();

			r1 *= coef[0];
			r2 *= coef[1];
			r3 *= coef[2];
			r4 *= coef[3];
			r5 *= coef[4];

			stencil_type r6 = r1 + r2;
			stencil_type r7 = r3 + r4;
			stencil_type r8 = r6 + r7;
			stencil_type r = r5 + r8;

			output_bus[k].write(r);
			alt_bus[k].write(r3);
		}
	}
}
