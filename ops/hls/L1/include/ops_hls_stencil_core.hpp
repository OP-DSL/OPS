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
        unsigned short SIZE_X, unsigned short DIM>
class StencilCore
{
    public:
    	static constexpr unsigned short s_datatype_size = sizeof(T) * 8;
    	static constexpr unsigned short s_axis_width = VEC_FACTOR * s_datatype_size;
    	static constexpr unsigned short s_datatype_bytes = sizeof(T);
    	static constexpr unsigned short s_mask_width = VEC_FACTOR * s_datatype_bytes;
        static constexpr unsigned short s_index_size = size_IndexType;
        static constexpr unsigned short s_singe_index_size = size_singleIndex;
        typedef ap_uint<s_axis_width> widen_dt;
        typedef ap_uint<s_mask_width> mask_dt;
        typedef ap_uint<s_index_size> index_dt;
        typedef ::hls::stream<widen_dt> widen_stream_dt;
        typedef ::hls::stream<mask_dt> mask_stream_dt;
        typedef ::hls::stream<index_dt> index_stream_dt; //TODO: This need to be removed as this is part of unique INDEX_STENCIL


        StencilCore()
        {
		#pragma HLS ARRAY_PARTITION variable = m_sizes complete

#ifndef __SYTHESIS__
            static_assert(s_dim <= ops_max_dim, "Stencil cannot have more than maximum dimention supported by OPS_MAX_s_dim");
            static_assert(s_axis_width >= min_axis_data_width && s_axis_width <= max_axis_data_width,
			        "axis_width failed limit check. VEC_FACTOR and T should be within limits");
#endif        
            __init();
        }

        void setGridProp(const GridPropertyCore& gridProp)
        {
            m_gridProp = gridProp;

            for (unsigned short i = 0; i < s_dim; i++)
            {
				unsigned short half_span = (m_sizes[i] + 1) >> 2;
				m_lowerLimits[i] = half_span;
				m_upperLimits[i] = m_gridProp.size[i] + half_span;
            }
#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| s_dim: %d, s_size_x: %d, s_stencil_span_x: %d, s_stencil_half_span_x: %d \n"
            		,__func__, s_dim, s_size_x, s_stencil_span_x, s_stencil_half_span_x);
#endif
        }

        void setPoints(const unsigned short * stencilPoints)
        {
            for (unsigned short i = 0; i < NUM_POINTS * s_dim; i++)
            {
                m_stencilPoints[i] = stencilPoints[i];
            }
        }


#ifndef __SYTHESIS__

        void getPoints(unsigned short* stencilPoints)
        {
            for (unsigned short i = 0; i < NUM_POINTS * s_dim; i++)
            {
                stencilPoints[i] = m_stencilPoints[i];
            }
        }

        void getGridProp(const GridPropertyCore& gridProp)
        {
            gridProp = m_gridProp;
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
        static const unsigned short s_dim = DIM;
        static const unsigned short s_size_x = SIZE_X;
        static const unsigned short s_stencil_span_x = s_size_x;
        static const unsigned short s_stencil_half_span_x = s_stencil_span_x / 2;

        GridPropertyCore m_gridProp;
        unsigned short m_stencilPoints[NUM_POINTS * 2];
        unsigned short m_sizes[s_dim];

        SizeType m_lowerLimits;
        SizeType m_upperLimits;

};

template <typename T, unsigned short VEC_FACTOR, unsigned short DIM>
class INDEX_STENCIL : public ops::hls::StencilCore<T, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF, 
        0, DIM>
{
public:
    using ops::hls::StencilCore<T, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF, 
    0, DIM>::m_gridProp;
    typedef typename ops::hls::StencilCore<T, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF,
        0, DIM>::index_stream_dt index_stream_dt;
    typedef typename ops::hls::StencilCore<T, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF,
            0, DIM>::widen_dt widen_dt;
    typedef typename ops::hls::StencilCore<T, 1, VEC_FACTOR, ops::hls::CoefTypes::CONST_COEF,
            0, DIM>::index_dt index_dt;


    void idxRead(index_stream_dt idx_bus[vector_factor])
    {
        unsigned short i = 0;
#if(DIM > 1)
        unsigned short j = 0;
        unsigned short i_l = 0; // Line buffer index
#endif
#if(DIM > 2)
        unsigned short k = 0;
        unsigned short i_p = 0; // Plane buffer index
#endif
        
        ::ops::hls::GridPropertyCore gridProp = m_gridProp;
#if(DIM == 1)
        unsigned short iter_limit = gridProp.total_itr;
#elif(DIM == 2)      
        unsigned short iter_limit = gridProp.outer_loop_limit * gridProp.xblocks;
#else
        unsigned short iter_limit = gridProp.outer_loop_limit *
                gridProp.grid_size[1] * gridProp.xblocks;
#endif
        unsigned short total_itr = gridProp.total_itr;
        widen_dt read_val = 0;

        widen_dt stencilValues[1];

        //No buffer
        stencil_type rowArr_0[vector_factor];
        #pragma HLS ARRAY_PARTITION variable = rowArr_0 dim=1 complete

        for (unsigned short itr = 0; itr < iter_limit; itr++)
        {
        #pragma HLS LOOP_TRIPCOUNT min=min_grid_size max=max_grid_size avg=avg_grid_size
        #pragma HLS PIPELINE II=1

            index_gen:
            {
                bool cond_x_terminate = (i == gridProp.xblocks - 1); 
#if(DIM == 2)
                bool cond_y_terminate = (j == gridProp.outer_loop_limit - 1);
#elif(DIM == 3)
                bool cond_y_terminate = (j == gridProp.grid_size[1] - 1);
                bool cond_z_terminate = (k == gridProp.outer_loop_limit - 1);
#endif

                if (cond_x_terminate)
                    i = 0;
                else
                    i++;
#if(DIM == 2)
                if (cond_x_terminate && cond_y_terminate)
                    j = 0;
                else if (cond_x_terminate)
                    j++;
#elif(DIM == 3)
                if (cond_x_terminate && cond_y_terminate && cond_z_terminate)
                    k = 0;
                else if (cond_x_terminate && cond_y_terminate)
                    k++;
#endif

#if(DIM > 1)
                bool cond_end_of_line_buff = (i_l) >= (gridProp.xblocks - 1);
#endif
#if(DIM > 2)
                bool cond_end_of_plane_buff = (i_p) >= (gridProp.plane_diff);
#endif

#if(DIM > 1)
                if (cond_end_of_line_buff)
                    i_l = 0;
                else
                    i_l++;
#endif
#if(DIM > 2)
                if (cond_end_of_plane_buff)
                    i_p = 0;
                else
                    i_p++;
#endif
            }
            
            process: for (unsigned short x = 0; x < VEC_FACTOR; x++)
            {
            #pragma HLS UNROLL complete
                unsigned short index = (i << LOG2(VEC_FACTOR)) + x;
                index_dt indexPkt;
                ops::hls::IndexConv indexConv;
                indexConv.index[0] = index;
#if(DIM > 1)
                indexConv.index[1] = j;
#endif
#if(DIM > 2)
                indexConv.index[2] = k;
#endif
                indexPkt = indexConv.flatten;
                idx_bus[x].write(indexPkt);
            }
        }
    }  
};


}
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
