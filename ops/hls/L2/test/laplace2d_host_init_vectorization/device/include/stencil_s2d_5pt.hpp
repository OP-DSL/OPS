#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core_v2.hpp>

static constexpr unsigned short s2d_5pt_num_points = 5;
static constexpr unsigned short s2d_5pt_stencil_size = 3;
static constexpr unsigned short s2d_5pt_stencil_dim = 2;

// class s2d_5pt : public ops::hls::StencilCore<stencil_type, s2d_5pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF,
//     s2d_5pt_stencil_size
class s2d_5pt : public ops::hls::StencilCoreV2<stencil_type, s2d_5pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF,
        s2d_5pt_stencil_size, s2d_5pt_stencil_dim>
{
    public:
        using ops::hls::StencilCoreV2<stencil_type, s2d_5pt_num_points, vector_factor, ops::hls::CoefTypes::CONST_COEF,
        s2d_5pt_stencil_size, s2d_5pt_stencil_dim>::m_stencilConfig;

        void stencilRead(widen_stream_dt& rd_buffer,
            ::hls::stream<stencil_type> output_bus_0[vector_factor],
            ::hls::stream<stencil_type> output_bus_1[vector_factor],
            ::hls::stream<stencil_type> output_bus_2[vector_factor],
            ::hls::stream<stencil_type> output_bus_3[vector_factor],
            ::hls::stream<stencil_type> output_bus_4[vector_factor],
			::hls::stream<bool> neg_cond_bus[vector_factor])
        {
#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| starting.", __func__);
#endif
//			#pragma HLS DEPENDENCE dependent=false distance=8 type=intra variable=rowArr_0
            short i = -s_stencil_half_span_x, j = -s_stencil_half_span_x;
            unsigned short i_l = 0; // Line buffer index

            ::ops::hls::StencilConfigCore stencilConfig = m_stencilConfig;
			#pragma HLS ARRAY_PARTITION variable = stencilConfig.lower_limit dim = 1 complete
			#pragma HLS ARRAY_PARTITION variable = stencilConfig.upper_limit dim = 1 complete

            unsigned short itr_limit = stencilConfig.outer_loop_limit * stencilConfig.grid_size[0];
            unsigned short total_itr = stencilConfig.total_itr;
            widen_dt read_val = 0;

            widen_dt stencilValues[s2d_5pt_num_points];
			#pragma HLS ARRAY_PARTITION variable = stencilValues dim=1 complete

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
            printf("[DEBUG][INTERNAL] itr_limt: %d, total_actual_itr: %d, grid_prop.xblocks(grid_size[0]): %d, grid_prop.outer_loop_limit: %d \n",
            		itr_limit, total_itr, stencilConfig.grid_size[0], stencilConfig.outer_loop_limit);
#endif
//            const unsigned short row_center_indices[] = {0,2,4};
//			#pragma HLS ARRAY_PARTITION variable=row_center_indices type=complete

            for (unsigned short itr = 0; itr < itr_limit; itr++)
            {
//            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
            #pragma HLS PIPELINE II=1

                spc_blocking_read:
                {
                    bool cond_x_terminate = (i == stencilConfig.grid_size[0] - 1);
                    bool cond_y_terminate = (j == stencilConfig.outer_loop_limit - 1);

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

                    if(i_l >= (stencilConfig.grid_size[0] - 1))
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
#pragma HLS UNROLL vector_factor
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
#pragma HLS UNROLL vector_factor
					unsigned short index = i * vector_factor + k;
					bool cond_no_send = register_it((index < stencilConfig.lower_limit[0])
												|| (index >= stencilConfig.upper_limit[0])
												|| (j < stencilConfig.lower_limit[1])
												|| (j >= stencilConfig.upper_limit[1]));
#ifdef DEBUG_LOG
					printf("[DEBUG][INTERNAL] index=(%d, %d), lowerbound=(%d, %d), upperbound=(%d, %d), cond_no_send=%d\n", index, j,
								stencilConfig.lower_limit[0], stencilConfig.lower_limit[1], stencilConfig.upper_limit[0], stencilConfig.upper_limit[1], cond_no_send);
					printf("[DEBUG][INTERNAL] values = (%f, %f, %f, %f, %f) \n\n", rowArr_0[k + s_stencil_half_span_x], rowArr_1[k], rowArr_1[k+1], rowArr_1[k+2], rowArr_2[k + s_stencil_half_span_x]);
#endif
					if (j >= 0)
					{
						output_bus_0[k].write(rowArr_0[k + s_stencil_half_span_x]);
						output_bus_1[k].write(rowArr_1[k]);
						output_bus_2[k].write(rowArr_1[k+1]);
						output_bus_3[k].write(rowArr_1[k+2]);
						output_bus_4[k].write(rowArr_2[k + s_stencil_half_span_x]);
						neg_cond_bus[k].write(cond_no_send);
					}
                }
            }
#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| exiting.", __func__);
#endif
        }
};
