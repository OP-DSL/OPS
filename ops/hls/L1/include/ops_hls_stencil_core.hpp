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
#include "ops_hls_defs.hpp"
#include "ops_hls_utils.hpp"

/**
 * TODO: This version assume no reduction and arg_dat and arg_const passed 
 */
namespace ops
{
namespace hls
{

template <typename T, unsigned short NUM_POINTS, unsigned short VEC_FACTOR, CoefTypes COEF_TYPE,
        unsigned short SIZE_X, unsigned short ...SIZES_REST>
class StencilCore
{
    public:
    	static constexpr unsigned short s_datatype_size = sizeof(T) * 8;
    	static constexpr unsigned short s_axis_width = VEC_FACTOR * s_datatype_size;
    	static constexpr unsigned short s_datatype_bytes = sizeof(T);
    	static constexpr unsigned short s_mask_width = VEC_FACTOR * s_datatype_bytes;
        typedef ap_uint<s_axis_width> widen_dt;
        typedef ap_uint<s_mask_width> mask_dt;
        typedef ::hls::stream<widen_dt> widen_stream_dt;
        typedef ::hls::stream<mask_dt> mask_stream_dt;


        StencilCore()
        {
		#pragma HLS ARRAY_PARTITION variable = m_sizes complete

#ifndef __SYTHESIS__
            static_assert(s_dim <= ops_max_dim, "Stencil cannot have more than maximum dimention supported by OPS_MAX_s_dim");
            static_assert(s_axis_width >= min_axis_data_width && s_axis_width <= max_axis_data_width,
			        "axis_width failed limit check. VEC_FACTOR and T should be within limits");
#endif        
            __init(SIZE_X, s_dim-1, SIZES_REST...);
        }

        void setGridProp(const GridPropertyCore& gridProp)
        {
            m_gridProp = gridProp;

            for (unsigned short i = 0; i < s_dim; i++)
            {
//            	if (i == 0)
//            	{
//            		m_lowerLimit[i] = s_stencil_half_span_x;
//            		m_upperLimit[i] = m_gridProp.actual_size[i];
//            	}
//            	else
//            	{
            		unsigned short half_span = (m_sizes[i] + 1) >> 2;
            		m_lowerLimits[i] = half_span;
            		m_upperLimits[i] = m_gridProp.size[i] + half_span;
//            	}
            }
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

        void __init(unsigned short size_x, int N_MIN_1, ...)
        {
            // m_sizes[0] = size_x;

            std::va_list args;
            va_start(args, N_MIN_1);

            for (unsigned short i = 0; i < N_MIN_1; i++)
            {
                m_sizes[i+1] = (unsigned short)va_arg(args, int);
            }

            va_end(args);

            m_sizes[0] = size_x;
        }

    protected:
        static const unsigned short s_dim = sizeof...(SIZES_REST) + 1;
        static const unsigned short s_size_x = SIZE_X;
        static const unsigned short s_stencil_span_x = (s_size_x + 1) / 2;
        static const unsigned short s_stencil_half_span_x = s_stencil_span_x / 2;

        GridPropertyCore m_gridProp;
        unsigned short m_stencilPoints[NUM_POINTS * 2];
        unsigned short m_sizes[s_dim];

        SizeType m_lowerLimits;
        SizeType m_upperLimits;

};


}
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
