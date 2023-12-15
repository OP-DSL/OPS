#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>

void kernel_copy_core(stencil_type& arg0_in_0,
        stencil_type& arg1_out_0)
{
#pragma HLS INLINE
    arg1_out_0 = arg0_in_0;
#ifdef DEBUG_LOG
        	printf("[KERNEL_INTERNAL]|%s| read_val: (%f) write_val: (%f)\n", __func__, arg0_in_0, arg1_out_0);
#endif
}

// Reading stencil values are used to generate stencil class
static constexpr unsigned short s2d_1pt_num_points = 1;
static constexpr unsigned short s2d_1pt_stencil_size = 1;
static constexpr unsigned short s2d_1pt_stencil_dim = 2;

class kernel_copy_stencil : public ops::hls::StencilCore<stencil_type, s2d_1pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
        s2d_1pt_stencil_size, s2d_1pt_stencil_dim>
{
    public:
        using ops::hls::StencilCore<stencil_type, s2d_1pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
        s2d_1pt_stencil_size, s2d_1pt_stencil_dim>::m_gridProp;
    
    void run(widen_stream_dt& rd_buffer, 
    		widen_stream_dt& wr_buffer)
    {
        short i = -1, j = -s_stencil_half_span_x;
        unsigned short i_l = 0; // Line buffer index

        ::ops::hls::GridPropertyCore gridProp = m_gridProp;
        unsigned short itr_limit = gridProp.outer_loop_limit * gridProp.xblocks;
        unsigned short total_itr = gridProp.total_itr;
        widen_dt read_val = 0;

        widen_dt stencilValues[s2d_1pt_num_points];
        #pragma HLS ARRAY_PARTITION variable = stencilValues dim=1 complete
        widen_dt outputValues;
        
        //No buffer
        stencil_type rowArr_0[vector_factor + s_stencil_span_x];
        #pragma HLS ARRAY_PARTITION variable = rowArr_0 dim=1 complete

        for (unsigned short itr = 0; itr < itr_limit; itr++)
        {
        #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
        #pragma HLS PIPELINE II=1

            spc_blocking_read:
            {
                bool cond_x_terminate = (i == gridProp.xblocks - 1);
                bool cond_y_terminate = (j == gridProp.outer_loop_limit - 1);

                if (cond_x_terminate)
                {
                    i = 0;
                }
                else
                {
                    i++;
                }

                if (cond_x_terminate && cond_y_terminate)
                {
                    j = 1; //s_stencil_half_span_x;
                }
                else if(cond_x_terminate)
                {
                    j++;
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
                printf("[DEBUG][INTERNAL] loop params i(%d), j(%d), i(%d), j(%d), i_l(%d), itr(%d)\n", i, j, i, j, i_l, itr);
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
                unsigned short index = i * vector_factor + k;
                bool cond_no_send = register_it((index < m_lowerLimits[0])
                                            || (index >= m_upperLimits[0])
                                            || (j < m_lowerLimits[1])
                                            || (j >= m_upperLimits[1]));

                stencil_type read_val;
                
                if (not cond_no_send)
                {
                    kernel_copy_core(rowArr_0[k + s_stencil_half_span_x], read_val);
                }
                else
                {
                    read_val = rowArr_0[k + s_stencil_half_span_x];
                }

                ops::hls::DataConv tmpConv;
                tmpConv.f = read_val;
                outputValues.range((k + 1) * sizeof(stencil_type) * 8 - 1, k * sizeof(stencil_type) * 8) = tmpConv.i;
            }

            wr_buffer << outputValues;
        }
    }
};

void kernel_copy_PE(ops::hls::GridPropertyCore& gridProp,
        kernel_copy_stencil::widen_stream_dt& arg0_input_stream,
        kernel_copy_stencil::widen_stream_dt& arg1_output_stream)
{
    kernel_copy_stencil stencil;

    stencil.setGridProp(gridProp);
    stencil.run(arg0_input_stream, arg1_output_stream);
} 

extern "C" void kernel_copy(
        const unsigned short gridProp_size_x,
        const unsigned short gridProp_size_y,
        const unsigned short gridProp_actual_size_x,
        const unsigned short gridProp_actual_size_y,
        const unsigned short gridProp_grid_size_x,
        const unsigned short gridProp_grid_size_y,
        const unsigned short gridProp_dim,
        const unsigned short gridProp_xblocks,
        const unsigned int gridProp_total_itr,
        const unsigned int gridProp_outer_loop_limit,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out);
