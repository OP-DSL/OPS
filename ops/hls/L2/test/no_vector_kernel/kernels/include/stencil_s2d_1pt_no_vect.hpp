#pragma once

#include <ops_hls_stencil_core.hpp>
#include "common_config.hpp"
#include <stdio.h>

static constexpr unsigned short num_points = 1;
static constexpr unsigned short stencil_size[] = {1, 1};

class s2d_1pt_no_vect : public ops::hls::StencilCore<stencil_type, 1, vector_factor, ops::hls::CoefTypes::CONST_COEF,
        stencil_size[0], stencil_size[1]>
{
    public:
        using ops::hls::StencilCore<stencil_type, 1, vector_factor, ops::hls::CoefTypes::CONST_COEF,
                stencil_size[0], stencil_size[1]>::m_gridProp;

//        void stencilRead(widen_stream_dt& rd_buffer, ::hls::stream<stencil_type> output_bus[1])
//        {
//            unsigned short i = 0, j = 0;
//            unsigned short i_l = 0; // Line buffer index
//            unsigned short i_d = 0, j_d = 0;
//            ::ops::hls::GridPropertyCore gridProp = m_gridProp;
//            unsigned short grid_size_itr = gridProp.grid_size[1] * gridProp.xblocks;
//            unsigned short total_itr = gridProp.total_itr;
//
//            widen_dt read_val = 0;
//
//            widen_dt stencilValues[num_points];
//            #pragma HLS ARRAY_PARTITION variable = stencilValues dim=1 complete
//
//            //No buffer
//
//            stencil_type rowArr_0[1 + s_stencil_span_x];
//			#pragma HLS ARRAY_PARTITION variable = rowArr_0 dim=1 complete
//
//            for (unsigned short itr = 0; itr < gridProp.total_itr; itr++)
//            {
//            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
//            #pragma HLS PIPELINE II=1
//
//                i = i_d;
//                j = j_d;
//
//                spc_blocking_read:
//                {
//                    bool cond_x_terminate = (i == gridProp.xblocks - 1);
//                    bool cond_y_terminate = (j == gridProp.outer_loop_limit - 1);
//
//                    if (cond_x_terminate)
//                    {
//                    	i_d = 0;
//                    }
//                    else
//                    {
//                    	i_d++;
//                    }
//
//                    if (cond_x_terminate && cond_y_terminate)
//                    {
//                    	j_d = 1;
//                    }
//                    else if(cond_x_terminate)
//                    {
//                    	j_d++;
//                    }
//
//                    bool cond_read = (itr < grid_size_itr);
//
//                    if(cond_read)
//                    {
//                        read_val = rd_buffer.read();
//                    }
//
//                    stencilValues[0] = read_val;
//
//                    i_l++;
//
//                    if(i_l >= (gridProp.xblocks - 1))
//                    {
//                    	i_l = 0;
//                    }
//                }
//
//                vec2arr: for (unsigned short k = 0; k < 1; k++)
//                {
//                    ops::hls::DataConv tmpConverter_0;
//                    tmpConverter_0.i = stencilValues[0].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
//                    rowArr_0[k + s_stencil_half_span_x] = tmpConverter_0.f;
//                }
//
//                process: for (unsigned short k = 0; k < 1; k++)
//                {
//                    output_bus[k].write(rowArr_0[k + s_stencil_half_span_x]);
//                }
//            }
//        }

        void stencilWrite(widen_stream_dt& wr_buffer, mask_stream_dt& strb_buffer, ::hls::stream<stencil_type> input_bus[vector_factor])
        {
            unsigned short i = 0, j = 0;
            unsigned short i_l = 0; // Line buffer index
            unsigned short i_d = 0, j_d = 0;
            ::ops::hls::GridPropertyCore gridProp = m_gridProp;
            unsigned short itr_limit = gridProp.outer_loop_limit * gridProp.xblocks;

            widen_dt updateValue;
            mask_dt maskValue;

            stencil_type m_memWrArr[1];
			#pragma HLS ARRAY_PARTITION variable = m_memWrArr dim=1 complete

            for (unsigned short itr = 0; itr < itr_limit; itr++)
            {
            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
            #pragma HLS PIPELINE II=1

                i = i_d;
                j = j_d;

                spc_blocking:
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
                    
                    i_l++;

                    if(i_l >= (gridProp.xblocks - 1))
                    {
                    	i_l = 0;
                    }
                }
                                
                process: for (unsigned short k = 0; k < vector_factor; k++)
                {  
                    stencil_type r = input_bus[k].read();
                    m_memWrArr[k] = r;
//                    printf("[KERNEL_DEBUG]|%s| reading, (i:%d, j:%d, k:%d), value:%f\n", __func__, i, j, k, r);
                }

                arr2vec: for (unsigned short k = 0; k < vector_factor; k++)
                {
                	unsigned short index = (i << shift_bits) + k;
                	bool cond_no_point_update = register_it(index < m_lowerLimits[0]
                								|| index >= m_upperLimits[0]
                								|| (j < (m_lowerLimits[1] + s_stencil_half_span_x))
                								|| (j >= (m_upperLimits[1] + s_stencil_half_span_x)));
                    ops::hls::DataConv tmpConv;
                    tmpConv.f = m_memWrArr[k];
                    if (cond_no_point_update)
                    {
                    	 maskValue.range((k + 1) * sizeof(stencil_type) - 1, k * sizeof(stencil_size)) = 0;
                    	 updateValue.range(s_datatype_size * (k + 1) - 1, k * s_datatype_size) = 0.1;
                    }
                    else
                    {
                    	maskValue.range((k + 1) * sizeof(stencil_type) - 1, k * sizeof(stencil_size)) = -1;
                    	updateValue.range(s_datatype_size * (k + 1) - 1, k * s_datatype_size) = tmpConv.i;
                    }
                }

                write:
                {
                    bool cond_write = ( j >= 1);

                    if (cond_write)
                    {
//                    	printf("[KERNEL_DEBUG]|%s| writing to axis. (i:%d, j:%d)\n", __func__, i, j);
                        wr_buffer << updateValue;
                        strb_buffer << maskValue;
                    }
                }
            }
        }

    private:

};
