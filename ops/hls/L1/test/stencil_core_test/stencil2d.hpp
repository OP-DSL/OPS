
#pragma once

#include <stdio.h>
#include "../../include/ops_hls_stencil_core.hpp"
#include "../../include/ops_hls_utils.hpp"
//#define DEBUG_LOG

typedef float stencil_type;
constexpr unsigned short num_points = 5; //cross_stencil
constexpr unsigned short vector_factor = 8;
constexpr unsigned short shift_bits = 3;
constexpr ops::hls::CoefTypes coef_type = ops::hls::CoefTypes::CONST_COEF;
constexpr unsigned short stencil_size_p = 3; //== stencil_size_q
constexpr unsigned short dim = 2;

 
// typedef ap_axiu<vector_factor * sizeof(stencil_type), 0, 0, 0> process_dt;
// typedef hls::stream<process_dt>  process_stream; 

class Stencil2D : public ops::hls::StencilCore<stencil_type, num_points, vector_factor, coef_type, stencil_size_p, 2>
{
    public:

        Stencil2D()
        {
        #pragma HLS ARRAY_PARTITION variable = m_rowArr_0 dim=1 complete
		#pragma HLS ARRAY_PARTITION variable = m_rowArr_1 dim=1 complete
		#pragma HLS ARRAY_PARTITION variable = m_rowArr_2 dim=1 complete
        #pragma HLS BIND_STORAGE variable = m_buffer_0 type = ram_t2p impl=uram latency=2
        #pragma HLS BIND_STORAGE variable = m_buffer_1 type = ram_t2p impl=uram latency=2
		#pragma HLS ARRAY_PARTITION variable = m_stencilValues complete
		#pragma HLS ARRAY_PARTITION variable = m_memWrArr complete
        }
        
        /**
         * This is sample stencil computation equalent to,
         * kernel(ACC<float> &current, ACC<float>& next)
         * {
         *      next(0,0) = 0.125 * curr(0,-1) + 0.125 * curr(-1,0) + 0.5 * curr(0,0) + 0.125 * curr(1,0) + 0.125 * curr(0,1)
         * }
         
                                    read                              
                                    |
                               (vi) |                                                                   index map
                                    v                                                   
                                +-------+                     (vii)                                     +-------+
                                |   4   | ----------------------------------------+                     |       |
                                |       |                                         |                     | (1,2) |
                        +-------+-------+-------+       +--------------------+    |             +-------+-------+-------+
                +------ |   1 <-+-- 2 <-+--  3  | <---- |  m_buffer_1[i_l]  | <--+             |       |       |       |
                |       |      (ii)    (iii)    |   (v) +--------------------+                  | (0,1) | (1,1) | (2,1) |
                |       +-------+-------+-------+                                               +-------+-------+-------+
                |               |   0   |                                                               |       |
                |               |       |                                                               | (1,0) |
                | (iv)          +-------+                                                               +-------+
                |                   ^
                |                   |  (i)
                |                   |
                |          +-------------------+
                +--------->|  m_buffer_0[i_l] |
                           +-------------------+                          
         */

        void kernel(widen_stream_dt& rd_buffer, widen_stream_dt& wr_buffer, stencil_type* coef)
        {
            unsigned short i = 0, j = 0;
            unsigned short i_l = 0; // Line buffer index
            unsigned short i_d = 0, j_d = 0;
            unsigned short grid_size_itr = m_gridProp.grid_size[1] * m_gridProp.xblocks;
            widen_dt read_val = 0;
//            const unsigned short row_center_indices[] = {0,2,4};
//			#pragma HLS ARRAY_PARTITION variable=row_center_indices type=complete

            for (unsigned short itr = 0; itr < m_gridProp.total_itr; itr++)
            {
            #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=avg_grid_size avg=avg_grid_size
            #pragma HLS PIPELINE II=1

                i = i_d;
                j = j_d;

                spc_blocking_read:
                {
                    bool cond_x_terminate = (i == m_gridProp.xblocks - 1);
                    bool cond_y_terminate = (j == m_gridProp.outer_loop_limit - 1);

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

                    if(i_l >= (m_gridProp.xblocks - 1))
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
                    unsigned short index = (i << shift_bits) + k;
                    stencil_type r1 = coef[0] * m_rowArr_0[k+1];
                    stencil_type r2 = coef[1] * m_rowArr_1[k];
                    stencil_type r3 = coef[2] * m_rowArr_1[k+1];
                    stencil_type r4 = coef[3] * m_rowArr_1[k+2];
                    stencil_type r5 = coef[4] * m_rowArr_2[k+1];

                    stencil_type r6 = r1 + r2;
                    stencil_type r7 = r3 + r4;
                    stencil_type r8 = r6 + r7;
                    stencil_type r = r5 + r8;

                    bool cond_no_point_update = register_it(index < m_lowerLimits[0] || index >= m_upperLimits[0]
															|| (j < (m_lowerLimits[1] + s_stencil_half_span_x))
															|| (j >= (m_upperLimits[1] + s_stencil_half_span_x)));

                    m_memWrArr[k] = cond_no_point_update ? m_rowArr_1[k+1] : r;
                }

                arr2vec: for (unsigned short k = 0; k < vector_factor; k++)
                {
                    ops::hls::DataConv tmpConv;
                    tmpConv.f = m_memWrArr[k];
                    m_updatedValue.range(s_datatype_size * (k + 1) - 1, k * s_datatype_size) = tmpConv.i;
                }

                write:
                {
                    bool cond_write = ( j >= 1);

                    if (cond_write)
                    {
                        wr_buffer << m_updatedValue;
                    }
                }
            }
        }

    private:
        static const unsigned short s_stencil_span_x = (stencil_size_p + 1) / 2;
        static const unsigned short s_stencil_half_span_x = s_stencil_span_x / 2;

        stencil_type m_rowArr_0[vector_factor + s_stencil_span_x];
        stencil_type m_rowArr_1[vector_factor + s_stencil_span_x];
        stencil_type m_rowArr_2[vector_factor + s_stencil_span_x];

        widen_dt m_buffer_0[max_depth_bytes/(sizeof(stencil_type) * vector_factor)];
        widen_dt m_buffer_1[max_depth_bytes/(sizeof(stencil_type) * vector_factor)];

        stencil_type m_memWrArr[vector_factor];
        widen_dt m_stencilValues[num_points];
        widen_dt m_updatedValue;
};

