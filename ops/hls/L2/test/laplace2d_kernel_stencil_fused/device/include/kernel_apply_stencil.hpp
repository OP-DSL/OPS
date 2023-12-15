#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>

void kernel_apply_stencil_core(stencil_type& arg0_in_0,
        stencil_type& arg0_in_1,
        stencil_type& arg0_in_2,
        stencil_type& arg0_in_3,
        stencil_type& arg0_in_4,
        stencil_type& arg1_out_0)
{
#pragma HLS INLINE
    // stencil_type r5 = arg0_in_0 + arg0_in_1;
    // stencil_type r6 = arg0_in_3 + arg0_in_4;
    // stencil_type r7 = r5 + r6;
    // arg1_out_0 = 0.25f * r7;

    arg1_out_0 = 0.25f * (arg0_in_0 + arg0_in_1 + arg0_in_3 + arg0_in_4);

#ifdef DEBUG_LOG
    printf("[KERNEL_INTERNAL]|%s| read_val: (%f, %f, %f, %f, %f) write_val: (%f)\n",
                __func__, arg0_in_0, arg0_in_1, arg0_in_2, arg0_in_3, arg0_in_4, arg1_out_0);
#endif
}

// Reading stencil values are used to generate stencil class
static constexpr unsigned short s2d_5pt_num_points = 5;
static constexpr unsigned short s2d_5pt_stencil_size = 3;
static constexpr unsigned short s2d_5pt_stencil_dim = 2;
// writing stencil values
static constexpr unsigned short s2d_1pt_num_points = 1;
static constexpr unsigned short s2d_1pt_stencil_size = 1;
static constexpr unsigned short s2d_1pt_stencil_dim = 2;

class kernel_apply_stencil_stencil : public ops::hls::StencilCore<stencil_type, s2d_5pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
        s2d_5pt_stencil_size, s2d_5pt_stencil_dim>
{
public:
    using ops::hls::StencilCore<stencil_type, s2d_5pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF, 
    s2d_5pt_stencil_size, s2d_5pt_stencil_dim>::m_gridProp;

    void run(widen_stream_dt& rd_buffer,
            widen_stream_dt& wr_buffer)
    {
//		#pragma HLS DEPENDENCE dependent=false distance=8 type=intra variable=rowArr_0
        short i = -1, j = -s_stencil_half_span_x;
        unsigned short i_l = 0; // Line buffer index

        ::ops::hls::GridPropertyCore gridProp = m_gridProp;
        unsigned short itr_limit = gridProp.outer_loop_limit * gridProp.xblocks;
        unsigned short total_itr = gridProp.total_itr;
        widen_dt read_val = 0;

        widen_dt stencilValues[s2d_5pt_num_points];
        #pragma HLS ARRAY_PARTITION variable = stencilValues dim=1 complete
        widen_dt outputValues;

        widen_dt buffer_0[max_depth_bytes/(sizeof(stencil_type) * vector_factor)];
        widen_dt buffer_1[max_depth_bytes/(sizeof(stencil_type) * vector_factor)];
        #pragma HLS BIND_STORAGE variable = buffer_0 type = ram_t2p impl=uram latency=2
        #pragma HLS BIND_STORAGE variable = buffer_1 type = ram_t2p impl=uram latency=2

        stencil_type rowArr_0[vector_factor + s_stencil_span_x];
        stencil_type rowArr_1[vector_factor + s_stencil_span_x];
        stencil_type rowArr_2[vector_factor + s_stencil_span_x];
        #pragma HLS ARRAY_PARTITION variable = rowArr_0 dim=1 complete
        #pragma HLS ARRAY_PARTITION variable = rowArr_1 dim=1 complete
        #pragma HLS ARRAY_PARTITION variable = rowArr_2 dim=1 complete

#ifdef DEBUG_LOG
        printf("[DEBUG][INTERNAL] itr_limt: %d, total_actual_itr: %d, grid_prop.xblocks: %d, grid_prop.outer_loop_limit: %d \n",
                itr_limit, total_itr, gridProp.xblocks, gridProp.outer_loop_limit);
#endif
//      const unsigned short row_center_indices[] = {0,2,4};
//		#pragma HLS ARRAY_PARTITION variable=row_center_indices type=complete

        for (unsigned short itr = 0; itr < itr_limit; itr++)
        {
//      #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
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
                    j = 0;
                }
                else if(cond_x_terminate)
                {
                    j++;
                }
                
                bool cond_read = (itr < total_itr);

                if (cond_read){
                    read_val = rd_buffer.read();
                }

                stencilValues[0] = buffer_0[i_l]; // (i)
                stencilValues[1] = stencilValues[2]; // (ii)
                stencilValues[2] = stencilValues[3]; // (iii)
                buffer_0[i_l] = stencilValues[1]; // (iv)
                stencilValues[3] = buffer_1[i_l]; // (v)
                stencilValues[4] = read_val; // (vi)
                buffer_1[i_l] = stencilValues[4]; // (vii)

                i_l++;

                if(i_l >= (gridProp.xblocks - 1))
                {
                    i_l = 0;
                }

#ifdef DEBUG_LOG
                printf("[DEBUG][INTERNAL] loop params i(%d), j(%d), i_l(%d), itr(%d)\n", i, j, i_l, itr);
                printf("[DEBUG][INTERNAL] --------------------------------------------------------\n\n");

                printf("[DEBUG][INTERNAL] read values: (");
                for (int ri = 0; ri < vector_factor; ri++)
                {
                    ops::hls::DataConv tmpConverter;
                    tmpConverter.i = stencilValues[4].range((ri + 1)*s_datatype_size - 1, ri * s_datatype_size);
                    printf("%f ", tmpConverter.f);
                }
                printf(")\n");
#endif
            }

            vec2arr: for (unsigned short k = 0; k < vector_factor; k++)
            {
#pragma HLS UNROLL factor=vector_factor
                ops::hls::DataConv tmpConverter_0, tmpConverter_1, tmpConverter_2;
                tmpConverter_0.i = stencilValues[0].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
                tmpConverter_1.i = stencilValues[2].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);
                tmpConverter_2.i = stencilValues[4].range(s_datatype_size * (k + 1) - 1, k * s_datatype_size);

                rowArr_0[k + s_stencil_half_span_x] = tmpConverter_0.f;
                rowArr_1[k + s_stencil_half_span_x] = tmpConverter_1.f;
                rowArr_2[k + s_stencil_half_span_x] = tmpConverter_2.f;
            }

            ops::hls::DataConv tmpConvRow1_0, tmpConvRow1_2;
            tmpConvRow1_0.i = stencilValues[1].range(s_datatype_size * (vector_factor) - 1, (vector_factor - 1) * s_datatype_size);
            tmpConvRow1_2.i = stencilValues[3].range(s_datatype_size - 1, 0);
            rowArr_1[0] = tmpConvRow1_0.f;
            rowArr_1[vector_factor + s_stencil_half_span_x] = tmpConvRow1_2.f;

            process: for (unsigned short k = 0; k < vector_factor; k++)
            {
#pragma HLS UNROLL factor=vector_factor
                unsigned short index = i * vector_factor + k;
                bool cond_no_send = register_it((index < m_lowerLimits[0])
                                            || (index >= m_upperLimits[0])
                                            || (j < m_lowerLimits[1])
                                            || (j >= m_upperLimits[1]));
#ifdef DEBUG_LOG
                printf("[DEBUG][INTERNAL] index=(%d, %d), lowerbound=(%d, %d), upperboud=(%d, %d), cond_no_send=%d\n", index, j,
                            m_lowerLimits[0], m_lowerLimits[1], m_upperLimits[0], m_upperLimits[1], cond_no_send);
                printf("[DEBUG][INTERNAL] values = (%f, %f, %f, %f, %f) \n\n", rowArr_0[k + s_stencil_half_span_x], rowArr_1[k], rowArr_1[k+1], rowArr_1[k+2], rowArr_2[k + s_stencil_half_span_x]);
#endif
                stencil_type read_val;
                
                if (not cond_no_send)
                {
                    kernel_apply_stencil_core(rowArr_0[k + s_stencil_half_span_x],
                            rowArr_1[k],
                            rowArr_1[k + 1],
                            rowArr_1[k + 2],
                            rowArr_2[k + s_stencil_half_span_x],
                            read_val);
                }
                else
                {
                    read_val = rowArr_1[k + 1];
                }
                ops::hls::DataConv tmpConv;
                tmpConv.f = read_val;
                outputValues.range((k + 1) * sizeof(stencil_type) * 8 - 1, k * sizeof(stencil_type) * 8) = tmpConv.i;
            }
#ifdef DEBUG_LOG
			printf("[DEBUG][INTERNAL]|%s| output val pkt: %d, val=(", __func__, i);

			for (unsigned short k = 0; k < vector_factor; k++)
			{
				ops::hls::DataConv tmp;
				tmp.i = outputValues.range((k + 1) * sizeof(stencil_type) * 8 - 1, k * sizeof(stencil_type) * 8);
				printf("%f,", tmp.f);
			}
			printf("), i:%d, j:%d\n",i,j);
#endif
            if (j >= 0)
            {
#ifdef DEBUG_LOG
            	printf("[DEBUG][INTERNAL]|%s| sending pkt\n", __func__);
#endif
            	wr_buffer << outputValues;
            }
        }
    }
};

void kernel_apply_stencil_PE(ops::hls::GridPropertyCore& gridProp,
        kernel_apply_stencil_stencil::widen_stream_dt& arg0_input_stream,
        kernel_apply_stencil_stencil::widen_stream_dt& arg1_output_stream)
{

#ifdef DEBUG_LOG
        	printf("[KERNEL_DEBUG]|%s| starting stencil kernel PE\n", __func__);
#endif
    kernel_apply_stencil_stencil stencil;
    stencil.setGridProp(gridProp);
    stencil.run(arg0_input_stream, arg1_output_stream);

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Ending stencil kernel PE\n", __func__);
#endif
} 

extern "C" void kernel_apply_stencil(
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
