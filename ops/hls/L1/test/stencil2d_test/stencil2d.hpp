
#include "../../include/ops_hls_stencil_core.hpp"


template <typename T, unsigned short NUM_POINTS, unsigned short VEC_FACTOR, unsigned short COEF_TYPE, 
        unsigned int P, unsigned int Q>
class Stencil2D : public ops::hls::Stencil2DCore<T, NUM_POINTS, VEC_FACTOR, COEF_TYPE, P, Q>
{
};

