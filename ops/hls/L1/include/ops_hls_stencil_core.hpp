#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** @file
  * @brief Vitis HLS specific L1 2D stencil core class
  * @author Beniel Thileepan
  * @details Implements of the templatised stencil class.
  * 
  */

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cstdarg>
#include "ops_hls_defs.hpp"

/**
 * TODO: This version assume no reduction and arg_dat and arg_const passed 
 */
namespace ops
{
namespace hls
{

template <typename T, unsigned short NUM_POINTS, unsigned short VEC_FACTOR, unsigned short COEF_TYPE, 
        unsigned int ...SIZES>
class Stencil2DCore
{
    public:
        
        static const int DIM = sizeof...(SIZES);
        unsigned int m_sizes[DIM];

        Stencil2DCore()
        {
        #pragma HLS ARRAY_PARTITION variable = m_coef dim = 1 complete
        #pragma HLS ARRAY_PARTITION variable = m_rowArr dim = 0 complete
        #pragma HLS ARRAY_PARTITION variable = m_memWrArr dim = 1 complete

#ifndef __SYTHESIS__
            static_assert(DIM <= ops_max_dim, "Stencil cannot have more than maximum dimention supported by OPS_MAX_DIM");
#endif        
            __init(DIM, SIZES...);
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

        void setPoints(const short* stencilPoints)
        {
            for (unsigned short i = 0; i < NUM_POINTS * DIM; i++)
            {
                m_stencilPoints[i] = stencilPoints[i];
            }
        }

        void getMemWr(T& memWrArr)
        {
            for (unsigned short i = 0; i < VEC_FACTOR; i++)
            {
                memWrArr[i] = m_memWrArr[i]; 
            }
        }
#ifndef __SYTHESIS__
        void getCoef(T* coef)
        {
            for (unsigned int i = 0; i < NUM_POINTS; i++)
            {
                coef[i] = m_coef[i];
            }
        }

        void getPoints(short* stencilPoints)
        {
            for (unsigned short i = 0; i < NUM_POINTS * DIM; i++)
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
        short m_stencilPoints[NUM_POINTS * 2];
        T m_memWrArr[VEC_FACTOR];

};


}
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */