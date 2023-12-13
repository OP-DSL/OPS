#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>

static constexpr unsigned short s2d_1pt_num_points = 1;
static constexpr unsigned short s2d_1pt_stencil_size = 1;
static constexpr unsigned short s2d_1pt_stencil_dim = 2;

class s2d_1pt : public ops::hls::StencilCore<stencil_type, s2d_1pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
        s2d_1pt_stencil_size, s2d_1pt_stencil_dim>
{
    public:
        using ops::hls::StencilCore<stencil_type, s2d_1pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
        s2d_1pt_stencil_size, s2d_1pt_stencil_dim>::m_gridProp;

        void idxRead(index_stream_dt idx_bus[vector_factor])
        {
            unsigned short i = 0, j = 0;
            unsigned short i_l = 0; // Line buffer index
            unsigned short i_d = 0, j_d = 0;
            ::ops::hls::GridPropertyCore gridProp = m_gridProp;
            unsigned short iter_limit = gridProp.outer_loop_limit * gridProp.xblocks;
            unsigned short total_itr = gridProp.total_itr;

            widen_dt read_val = 0;

            widen_dt stencilValues[s2d_1pt_num_points];
            #pragma HLS ARRAY_PARTITION variable = stencilValues dim=1 complete

            //No buffer
            stencil_type rowArr_0[vector_factor + s_stencil_span_x];
            #pragma HLS ARRAY_PARTITION variable = rowArr_0 dim=1 complete

            for (unsigned short itr = 0; itr < iter_limit; itr++)
            {
            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
            #pragma HLS PIPELINE II=1

                i = i_d;
                j = j_d;

                index_gen:
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
                        j_d = 1; //s_stencil_half_span_x;
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
                #pragma HLS UNROLL factor=vector_factor
                    unsigned short index = (i << shift_bits) + k;
                    index_dt indexPkt;
                    ops::hls::IndexConv indexConv;
                    indexConv.index[0] = index;
                    indexConv.index[1] = j;
                    indexPkt = indexConv.flatten;
                    idx_bus[k].write(indexPkt);
                }
            }
        } 

        void stencilRead(widen_stream_dt& rd_buffer, ::hls::stream<stencil_type> output_bus[vector_factor])
        {
            unsigned short i = 0, j = 0;
            unsigned short i_l = 0; // Line buffer index
            unsigned short i_d = 0, j_d = 0;
            ::ops::hls::GridPropertyCore gridProp = m_gridProp;
            unsigned short iter_limit = gridProp.outer_loop_limit * gridProp.xblocks;
            unsigned short total_itr = gridProp.total_itr;

            widen_dt read_val = 0;

            widen_dt stencilValues[s2d_1pt_num_points];
            #pragma HLS ARRAY_PARTITION variable = stencilValues dim=1 complete

            //No buffer
            stencil_type rowArr_0[vector_factor + s_stencil_span_x];
            #pragma HLS ARRAY_PARTITION variable = rowArr_0 dim=1 complete

            for (unsigned short itr = 0; itr < iter_limit; itr++)
            {
            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
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
                        j_d = 1; //s_stencil_half_span_x;
                    }
                    else if(cond_x_terminate)
                    {
                        j_d++;
                    }

                    bool cond_read = (itr < total_itr);

                    if(cond_read)
                    {
                        read_val = rd_buffer.read();
                    }

                    stencilValues[0] = read_val;

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
                    	tmpConverter.i = read_val.range((ri + 1)*s_datatype_size - 1, ri * s_datatype_size);
                    	printf("%f ", tmpConverter.f);
                    }
                    printf(")\n");
#endif
                }

                vec2arr: for (unsigned short k = 0; k < vector_factor; k++)
                {
                #pragma HLS UNROLL factor=vector_factor
                    ops::hls::DataConv tmpConverter_0;
                    tmpConverter_0.i = stencilValues[0].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
                    rowArr_0[k + s_stencil_half_span_x] = tmpConverter_0.f;
                }

                process: for (unsigned short k = 0; k < vector_factor; k++)
                {
                #pragma HLS UNROLL factor=vector_factor
                    output_bus[k].write(rowArr_0[k + s_stencil_half_span_x]);
                }
            }
        } 

        void stencilWrite(widen_stream_dt& wr_buffer, mask_stream_dt& strb_buffer, ::hls::stream<stencil_type> input_bus[vector_factor], unsigned short x_half_span = 0, unsigned short x_full_span = 0)
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
                                
                process_read: for (unsigned short k = 0; k < vector_factor; k++)
                {  
#pragma HLS UNROLL factor=vector_factor
                	unsigned short index = (i << shift_bits) + k;
                	bool cond_no_point_update = register_it((index < m_lowerLimits[0] + x_half_span)
                								|| (index >= m_upperLimits[0] + x_half_span)
                								|| (j < m_lowerLimits[1] + x_full_span)
                								|| (j >= (m_upperLimits[1] + x_full_span)));
                    ops::hls::DataConv tmpConv;
                    stencil_type r = input_bus[k].read();
                   	tmpConv.f = r;
#ifdef DEBUG_LOG
                    printf("[KERNEL_DEBUG]|%s| reading, (i:%d, j:%d, k:%d), limits(xl:>= %d, xh:< %d, yl:>= %d, yh:< %d), stencil_half_span: %d, value:%f, condition_update:%d\n",
                    		__func__, i, j, k, m_lowerLimits[0], m_upperLimits[0], (m_lowerLimits[1] + s_stencil_half_span_x), (m_upperLimits[1] + s_stencil_half_span_x), s_stencil_half_span_x, r, cond_no_point_update);
#endif
                    if (cond_no_point_update)
                    {
                    	 maskValue.range((k + 1) * sizeof(stencil_type) - 1, k * sizeof(stencil_type)) = 0;
                    	 updateValue.range(s_datatype_size * (k + 1) - 1, k * s_datatype_size) = 0.1;
                    }
                    else
                    {
                    	maskValue.range((k + 1) * sizeof(stencil_type) - 1, k * sizeof(stencil_type)) = -1;
                    	updateValue.range(s_datatype_size * (k + 1) - 1, k * s_datatype_size) = tmpConv.i;
                    }
                }

                write:
                {
                    bool cond_write =  register_it((j >= x_half_span) || (j < m_upperLimits[1] + x_half_span));

                    if (cond_write)
                    {
#ifdef DEBUG_LOG
                    	printf("[KERNEL_DEBUG]|%s| writing to axis. (i:%d, j:%d), val=(", __func__, i, j);

                    	for (unsigned short k = 0; k < vector_factor; k++)
						{
                			ops::hls::DataConv tmp;
                			tmp.i = updateValue.range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
                			printf("%f,", tmp.f);
                		}
                		printf(")\n");
#endif
                        wr_buffer << updateValue;
                        strb_buffer << maskValue;
                    }
                }
            }
#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| exiting.", __func__);
#endif
        }
       
};
