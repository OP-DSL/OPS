
#include "top.hpp"
//#define DEBUG_LOG

void dut(const unsigned short gridProp_size_x,
	    const unsigned short gridProp_size_y,
	    const unsigned short gridProp_actual_size_x,
	    const unsigned short gridProp_actual_size_y,
	    const unsigned short gridProp_grid_size_x,
	    const unsigned short gridProp_grid_size_y,
	    const unsigned short gridProp_dim,
	    const unsigned short gridProp_xblocks,
	    const unsigned int gridProp_total_itr,
	    const unsigned int gridProp_outer_loop_limit,
		stencil_type* data_in,
		stencil_type* data_out)
{
#pragma TOP

#pragma HLS INTERFACE s_axilite port = gridProp_size_x bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_size_y bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_actual_size_x bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_actual_size_y bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_grid_size_x bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_grid_size_y bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_dim bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_xblocks bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_total_itr bundle = control
#pragma HLS INTERFACE s_axilite port = gridProp_outer_loop_limit bundle = control

#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

#pragma HLS INTERFACE mode=m_axi depth=max_depth_bytes max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=4 num_write_outstanding=4 port=data_in
#pragma HLS INTERFACE mode=s_axilite port=data_in bundle=control

#pragma HLS INTERFACE mode=m_axi depth=max_depth_bytes max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=4 num_write_outstanding=4 port=data_out
#pragma HLS INTERFACE mode=s_axilite port=data_out bundle=control
	/* Cross Stencil */
    Stencil2D cross_stencil;

    unsigned short points[] = {1,0, 0,1, 1,1, 2,1, 1,2};
    float coef[] = {0.125 , 0.125 , 0.5, 0.125, 0.125};
    unsigned short offset[] = {1,1};

    ops::hls::GridPropertyCore gridProp;
    gridProp.dim = gridProp_dim;
    gridProp.size[0] = gridProp_size_x;
    gridProp.size[1] = gridProp_size_y;
    gridProp.actual_size[0] = gridProp_actual_size_x;
    gridProp.actual_size[1] = gridProp_actual_size_y;
    gridProp.grid_size[0] = gridProp_grid_size_x;
    gridProp.grid_size[1] = gridProp_grid_size_y;
    gridProp.xblocks = gridProp_xblocks;
    gridProp.total_itr = gridProp_total_itr;
    gridProp.outer_loop_limit = gridProp_outer_loop_limit;

    cross_stencil.setGridProp(gridProp);
    cross_stencil.setPoints(points);


#ifndef __SYTHESIS__
    unsigned int stencil_sizes[2];
    unsigned short read_points[num_points * 2];

    cross_stencil.getPoints(read_points);

    std::cout << "SUCESSFUL INSTANTIATION OF STENCIL CORE" << std::endl;

    std::cout << "POINTS: ";

    for (int i = 0; i < num_points; i++)
    {
        std::cout << "(" << read_points[2 * i] << ", " << read_points[2 * i + 1] << ") ";
    }

    std::cout << std::endl << std::endl;
#endif

    ops::hls::DataConv converterInput;
    typedef ap_uint<vector_factor * sizeof(stencil_type) * 8> widen_dt;
    typedef hls::stream<widen_dt> stream_dt;

    static stream_dt in_stream, out_stream;
    static ::hls::stream<stencil_type> input_bus_1[vector_factor];
    static ::hls::stream<stencil_type> input_bus_2[vector_factor];
    static ::hls::stream<stencil_type> input_bus_3[vector_factor];
    static ::hls::stream<stencil_type> input_bus_4[vector_factor];
    static ::hls::stream<stencil_type> input_bus_5[vector_factor];
    static ::hls::stream<stencil_type> output_bus[vector_factor];
    static ::hls::stream<stencil_type> alt_bus[vector_factor];
    const unsigned int kernel_iterations = gridProp.outer_loop_limit * gridProp.xblocks;
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
