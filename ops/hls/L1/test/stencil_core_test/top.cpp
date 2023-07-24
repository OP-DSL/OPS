
#include "top.hpp"
//#define DEBUG_LOG

void dut(ops::hls::GridPropertyCore& gridProp, Stencil2D& cross_stencil, stencil_type* data_in, stencil_type* data_out)
{

    stencil_type coef[] = {0.125 , 0.125 , 0.5, 0.125, 0.125};

//#ifndef __SYTHESIS__
//    unsigned int stencil_sizes[2];
//    unsigned short read_points[num_points * 2];
//
//    cross_stencil.getPoints(read_points);
//
//    std::cout << "SUCESSFUL INSTANTIATION OF STENCIL CORE" << std::endl;
//
//    std::cout << "POINTS: ";
//
//    for (int i = 0; i < num_points; i++)
//    {
//        std::cout << "(" << read_points[2 * i] << ", " << read_points[2 * i + 1] << ") ";
//    }
//
//    std::cout << std::endl << std::endl;
//#endif

    ops::hls::DataConv converterInput;
    typedef ap_uint<vector_factor * sizeof(stencil_type) * 8> widen_dt;
    typedef hls::stream<widen_dt> stream_dt;

    static stream_dt in_stream, out_stream;

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

    cross_stencil.kernel(in_stream, out_stream, coef);


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
