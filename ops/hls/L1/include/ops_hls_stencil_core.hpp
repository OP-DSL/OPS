#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** @file
  * @brief Vitis HLS specific L1 abstract stencil core class
  * @author Beniel Thileepan
  * @details Implements of the templatised stencil class.
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

template <typename T, unsigned short NUM_POINTS, unsigned short VEC_FACTOR, unsigned short COEF_TYPE, 
        unsigned int ...SIZES>
class StencilCore
{
    public:
        
        static const int s_dim = sizeof...(SIZES);
        static const int s_datatype_size = sizeof(T) * 8;
        static const int s_axis_width = VEC_FACTOR * s_datatype_size;
        typedef ap_uint<s_axis_width> widen_dt;
        typedef ::hls::stream<widen_dt> stream_dt;


        StencilCore()
        {
        #pragma HLS ARRAY_PARTITION variable = m_coef complete
		#pragma HLS ARRAY_PARTITION variable = m_sizes complete

#ifndef __SYTHESIS__
            static_assert(s_dim <= ops_max_dim, "Stencil cannot have more than maximum dimention supported by OPS_MAX_s_dim");
            static_assert(s_axis_width >= min_axis_data_width && s_axis_width <= max_axis_data_width,
			        "axis_width failed limit check. VEC_FACTOR and T should be within limits");
#endif        
            __init(s_dim, SIZES...);
        }

    #if COEF_TYPE == 0
        void setCoef(const T* coef)
        {
            for (unsigned int i = 0; i < NUM_POINTS; i++)
            {
                m_coef[i] = coef[i];
            }
        }
    #endif

        void setGridProp(const GridPropertyCore& gridProp)
        {
            m_gridProp = gridProp;
        }

        void setPoints(const unsigned short * stencilPoints)
        {
            for (unsigned short i = 0; i < NUM_POINTS * s_dim; i++)
            {
                m_stencilPoints[i] = stencilPoints[i];
            }
        }

//        void getMemWr(T& memWrArr)
//        {
//            for (unsigned short i = 0; i < VEC_FACTOR; i++)
//            {
//                memWrArr[i] = m_memWrArr[i];
//            }
//        }
#ifndef __SYTHESIS__
        void getCoef(T* coef)
        {
            for (unsigned int i = 0; i < NUM_POINTS; i++)
            {
                coef[i] = m_coef[i];
            }
        }

        void getPoints(unsigned short* stencilPoints)
        {
            for (unsigned short i = 0; i < NUM_POINTS * s_dim; i++)
            {
                stencilPoints[i] = m_stencilPoints[i];
            }
        }
#endif

    private:

        void __init(int N, ...)
        {
            // m_sizes[0] = size_x;

            std::va_list args;
            va_start(args, N);

            for (unsigned short i = 0; i < N; i++)
            {
                m_sizes[i] = va_arg(args, unsigned int);
            }

            va_end(args);
        }

    protected:
        T m_coef[NUM_POINTS];
        GridPropertyCore m_gridProp;
        unsigned short m_stencilPoints[NUM_POINTS * 2];

        widen_dt m_updatedValue; 

        unsigned int m_sizes[s_dim];


};


}
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
