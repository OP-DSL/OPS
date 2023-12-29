#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** @file
  * @brief Vitis HLS specific L1 abstract stencil core class
  * @author Beniel Thileepan
  * @details Implements of the template stencil class.
  * 
  */

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cstdarg>
#include <math.h>
#include "ops_hls_defs.hpp"
#include "ops_hls_utils.hpp"
#include <stdio.h>

/**
 * TODO: This version assume no reduction and arg_dat and arg_const passed 
 */
namespace ops
{
namespace hls
{

template <typename T, unsigned short NUM_POINTS, unsigned short VEC_FACTOR, CoefTypes COEF_TYPE,
        unsigned short STENCIL_SIZE_X, unsigned short STENCIL_DIM>
class StencilCoreV2
{
    public:
    	static constexpr unsigned short s_datatype_size = sizeof(T) * 8;
    	static constexpr unsigned short s_axis_width = VEC_FACTOR * s_datatype_size;
    	static constexpr unsigned short s_datatype_bytes = sizeof(T);
    	static constexpr unsigned short s_mask_width = VEC_FACTOR * s_datatype_bytes;
        static constexpr unsigned short s_singe_index_size = size_singleIndex;
        typedef ap_uint<s_axis_width> widen_dt;
        typedef ap_uint<s_mask_width> mask_dt;
        typedef ::hls::stream<widen_dt> widen_stream_dt;
        typedef ::hls::stream<mask_dt> mask_stream_dt;

        StencilCoreV2()
        {
		#pragma HLS ARRAY_PARTITION variable = m_sizes complete

#ifndef __SYTHESIS__
            static_assert(s_dim <= ops_max_dim, "Stencil cannot have more than maximum dimention supported by OPS_MAX_s_dim");
            static_assert(s_axis_width >= min_axis_data_width && s_axis_width <= max_axis_data_width,
			        "axis_width failed limit check. VEC_FACTOR and T should be within limits");
#endif        
            __init();
        }

        void setConfig(const StencilConfigCore& stencilConfig)
        {
            m_stencilConfig = stencilConfig;

#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| s_dim: %d, s_size_x: %d, s_stencil_span_x: %d, s_stencil_half_span_x: %d \n"
            		,__func__, s_dim, s_size_x, s_stencil_span_x, s_stencil_half_span_x);
#endif
        }

        // void setPoints(const unsigned short * stencilPoints)
        // {
        //     for (unsigned short i = 0; i < NUM_POINTS * s_dim; i++)
        //     {
        //         m_stencilPoints[i] = stencilPoints[i];
        //     }
        // }


#ifndef __SYTHESIS__

        // void getPoints(unsigned short* stencilPoints)
        // {
        //     for (unsigned short i = 0; i < NUM_POINTS * s_dim; i++)
        //     {
        //         stencilPoints[i] = m_stencilPoints[i];
        //     }
        // }

        void getCofig(StencilConfigCore& stencilConfig)
        {
            stencilConfig = m_stencilConfig;
        }
#endif

    private:

        inline void __init()
        {
            for (unsigned short i = 0; i < s_dim; i++)
            {
                m_sizes[i] = s_size_x;
            }
        }

    protected:
        static const unsigned short s_dim = STENCIL_DIM;
        static const unsigned short s_size_x = STENCIL_SIZE_X;
        static const unsigned short s_stencil_span_x = s_size_x - 1;
        static const unsigned short s_stencil_half_span_x = s_stencil_span_x / 2;

        StencilConfigCore m_stencilConfig;
        // unsigned short m_stencilPoints[NUM_POINTS * 2];
        unsigned short m_sizes[s_dim];

        // SizeType m_lowerLimits;
        // SizeType m_upperLimits;

};

/**
 * TODO: INDEX_STENCIL_V2 is not completed. Not required now.
*/
/*
template <unsigned short VEC_FACTOR, unsigned short STENCIL_DIM>
class INDEX_STENCIL_V2 : public ops::hls::StencilCoreV2<unsigned short, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF, 
        0, STENCIL_DIM>
{
public:
    static constexpr unsigned short s_index_size = size_IndexType;

    typedef ap_int<s_index_size> index_dt;
    using ops::hls::StencilCore<int, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF, 
    0, STENCIL_DIM>::m_stencilConfig;
    // typedef typename ops::hls::StencilCore<int, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF,
    //         0, STENCIL_DIM>::widen_dt widen_dt;
    typedef ::hls::stream<index_dt> index_stream_dt;    

    void idxRead(index_stream_dt idx_bus[vector_factor])
    {
        unsigned short i = 0;
#if(STENCIL_DIM > 1)
        unsigned short j = 0;
        unsigned short i_l = 0; // Line buffer index
#endif
#if(STENCIL_DIM > 2)
        unsigned short k = 0;
        unsigned short i_p = 0; // Plane buffer index
#endif
        
        ::ops::hls::StencilConfigCore stencilConfig = m_stencilConfig;
        unsigned short iter_limit = stencilConfig.total_itr;
// #if(STENCIL_DIM == 1)
//         unsigned short iter_limit = stencilConfig.total_itr;
// #elif(STENCIL_DIM == 2)      
//         unsigned short iter_limit = gridProp.outer_loop_limit * gridProp.xblocks;
// #else
//         unsigned short iter_limit = gridProp.outer_loop_limit *
//                 gridProp.grid_size[1] * gridProp.xblocks;
// #endif
        unsigned short total_itr = gridProp.total_itr;
        widen_dt read_val = 0;

        widen_dt stencilValues[1];

        //No buffer
        stencil_type rowArr_0[vector_factor];
        #pragma HLS ARRAY_PARTITION variable = rowArr_0 STENCIL_DIM=1 complete

        for (unsigned short itr = 0; itr < iter_limit; itr++)
        {
        #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
        #pragma HLS PIPELINE II=1

            index_gen:
            {
                bool cond_x_terminate = (i == gridProp.xblocks - 1); 
#if(STENCIL_DIM == 2)
                bool cond_y_terminate = (j == gridProp.outer_loop_limit - 1);
#elif(STENCIL_DIM == 3)
                bool cond_y_terminate = (j == gridProp.grid_size[1] - 1);
                bool cond_z_terminate = (k == gridProp.outer_loop_limit - 1);
#endif

                if (cond_x_terminate)
                    i = 0;
                else
                    i++;
#if(STENCIL_DIM == 2)
                if (cond_x_terminate && cond_y_terminate)
                    j = 0;
                else if (cond_x_terminate)
                    j++;
#elif(STENCIL_DIM == 3)
                if (cond_x_terminate && cond_y_terminate && cond_z_terminate)
                    k = 0;
                else if (cond_x_terminate && cond_y_terminate)
                    k++;
#endif

#if(STENCIL_DIM > 1)
                bool cond_end_of_line_buff = (i_l) >= (gridProp.xblocks - 1);
#endif
#if(STENCIL_DIM > 2)
                bool cond_end_of_plane_buff = (i_p) >= (gridProp.plane_diff);
#endif

#if(STENCIL_DIM > 1)
                if (cond_end_of_line_buff)
                    i_l = 0;
                else
                    i_l++;
#endif
#if(STENCIL_DIM > 2)
                if (cond_end_of_plane_buff)
                    i_p = 0;
                else
                    i_p++;
#endif
            }
            
            process: for (unsigned short x = 0; x < VEC_FACTOR; x++)
            {
#pragma HLS UNROLL factor=VEC_FACTOR

                unsigned short index = (i << LOG2(VEC_FACTOR)) + x;
                index_dt indexPkt;
                ops::hls::IndexConv indexConv;
                indexConv.index[0] = index;
#if(STENCIL_DIM > 1)
                indexConv.index[1] = j;
#endif
#if(STENCIL_DIM > 2)
                indexConv.index[2] = k;
#endif
                indexPkt = indexConv.flatten;
                idx_bus[x].write(indexPkt);
            }
        }
    }  
};
*/

}
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
