
#pragma once

#include <ops_hls_stencil_core.hpp>
#include "common_config.hpp"

static constexpr unsigned short num_points = 5;
static constexpr unsigned short stencil_size[] = {3, 3};

class s2d_5pt : public ops::hls::StencilCore<stencil_type, 1, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
        stencil_size[0], stencil_size[1]>
{
    public:
        using ops::hls::StencilCore<stencil_type, 1, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
                stencil_size[0], stencil_size[1]>::m_gridProp;

        void stencilRead(stream_dt& rd_buffer,
            ::hls::stream<stencil_type> output_bus_0[vector_factor],
            ::hls::stream<stencil_type> output_bus_1[vector_factor],
            ::hls::stream<stencil_type> output_bus_2[vector_factor],
            ::hls::stream<stencil_type> output_bus_3[vector_factor],
            ::hls::stream<stencil_type> output_bus_4[vector_factor])
        {
//			#pragma HLS DEPENDENCE dependent=false distance=8 type=intra variable=m_rowArr_0
            unsigned short i = 0, j = 0;
            unsigned short i_l = 0; // Line buffer index
            unsigned short i_d = 0, j_d = 0;
            ::ops::hls::GridPropertyCore gridProp = m_gridProp;
            unsigned short grid_size_itr = gridProp.grid_size[1] * gridProp.xblocks;
            unsigned short total_itr = gridProp.total_itr;
            widen_dt read_val = 0;

            widen_dt m_stencilValues[num_points];
            #pragma HLS ARRAY_PARTITION variable = m_stencilValues dim=1 complete

            widen_dt m_buffer_0[max_depth_bytes/(sizeof(stencil_type) * vector_factor)];
            widen_dt m_buffer_1[max_depth_bytes/(sizeof(stencil_type) * vector_factor)];
            #pragma HLS BIND_STORAGE variable = m_buffer_0 type = ram_t2p impl=uram latency=2
            #pragma HLS BIND_STORAGE variable = m_buffer_1 type = ram_t2p impl=uram latency=2

            stencil_type m_rowArr_0[vector_factor + s_stencil_span_x];
            stencil_type m_rowArr_1[vector_factor + s_stencil_span_x];
            stencil_type m_rowArr_2[vector_factor + s_stencil_span_x];
            #pragma HLS ARRAY_PARTITION variable = m_rowArr_0 dim=1 complete
            #pragma HLS ARRAY_PARTITION variable = m_rowArr_1 dim=1 complete
            #pragma HLS ARRAY_PARTITION variable = m_rowArr_2 dim=1 complete

    //            const unsigned short row_center_indices[] = {0,2,4};
    //			#pragma HLS ARRAY_PARTITION variable=row_center_indices type=complete

            for (unsigned short itr = 0; itr < total_itr; itr++)
            {
    //            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
            #pragma HLS PIPELINE II=1

                i = i_d;
                j = j_d;

                spc_blocking_read:
                {
                    bool cond_x_terminate = (i == gridProp.xblocks - 1);
                    bool cond_y_terminate = (j == gridProp.outer_loop_limit - 1);

                    if (cond_x_terminate)
                    {
                        i_d = 0;
                    }
                    else
                    {
                        i_d++;
                    }

                    if (cond_x_terminate && cond_y_terminate)
                    {
                        j_d = 1;
                    }
                    else if(cond_x_terminate)
                    {
                        j_d++;
                    }
                    
                    bool cond_read = (itr < grid_size_itr);
                    if (cond_read){
                        read_val = rd_buffer.read();
                    }

                    m_stencilValues[0] = m_buffer_0[i_l]; // (i)
                    m_stencilValues[1] = m_stencilValues[2]; // (ii)
                    m_stencilValues[2] = m_stencilValues[3]; // (iii)
                    m_buffer_0[i_l] = m_stencilValues[1]; // (iv)
                    m_stencilValues[3] = m_buffer_1[i_l]; // (v)
                    m_stencilValues[4] = read_val; // (vi)
                    m_buffer_1[i_l] = m_stencilValues[4]; // (vii)

                    i_l++;

                    if(i_l >= (gridProp.xblocks - 1))
                    {
                        i_l = 0;
                    }

    #ifdef DEBUG_LOG
                    printf("[DEBUG][INTERNAL] loop params i(%d), j(%d), i_d(%d), j_d(%d), i_l(%d), itr(%d)\n", i, j, i_d, j_d, i_l, itr);
                    printf("[DEBUG][INTERNAL] --------------------------------------------------------\n\n");

                    printf("[DEBUG][INTERNAL] read values: (");
                    for (int ri = 0; ri < vector_factor; ri++)
                    {
                        ops::hls::DataConv tmpConverter;
                        tmpConverter.i = m_stencilValues[4].range((ri + 1)*s_datatype_size - 1, ri * s_datatype_size);
                        printf("%f ", tmpConverter.f);
                    }
                    printf(")\n");
    #endif
                }

                vec2arr: for (unsigned short k = 0; k < vector_factor; k++)
                {

                    ops::hls::DataConv tmpConverter_0, tmpConverter_1, tmpConverter_2;
                    tmpConverter_0.i = m_stencilValues[0].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
                    tmpConverter_1.i = m_stencilValues[2].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
                    tmpConverter_2.i = m_stencilValues[4].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);

                    m_rowArr_0[k + 1] = tmpConverter_0.f;
                    m_rowArr_1[k + 1] = tmpConverter_1.f;
                    m_rowArr_2[k + 1] = tmpConverter_2.f;
                }

                ops::hls::DataConv tmpConvRow1_0, tmpConvRow1_2;
                tmpConvRow1_0.i = m_stencilValues[1].range(s_datatype_size * (vector_factor) - 1, (vector_factor - 1) * s_datatype_size);
                tmpConvRow1_2.i = m_stencilValues[3].range(s_datatype_size - 1, 0);
                m_rowArr_1[0] = tmpConvRow1_0.f;
                m_rowArr_1[vector_factor + s_stencil_half_span_x] = tmpConvRow1_2.f;

                process: for (unsigned short k = 0; k < vector_factor; k++)
                {
                    output_bus_0[k].write(m_rowArr_0[k+1]);
                    output_bus_1[k].write(m_rowArr_1[k]);
                    output_bus_2[k].write(m_rowArr_1[k+1]);
                    output_bus_3[k].write(m_rowArr_1[k+2]);
                    output_bus_4[k].write(m_rowArr_2[k+1]);
                }
            }
        }

    private:
        static const unsigned short s_stencil_span_x = (s_size_x - 1);
        static const unsigned short s_stencil_half_span_x = s_stencil_span_x / 2;


};
