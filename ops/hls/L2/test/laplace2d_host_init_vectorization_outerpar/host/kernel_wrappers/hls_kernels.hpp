
#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <ops_hls_rt_support.h>

//#define DEBUG_LOG

extern int imax, jmax;
extern float pi;
unsigned short vector_factor = 8;
unsigned int iter_par_factor = 20;
unsigned short mem_vector_factor = 16;

#define MULTI_SLR

#include "kernelwrap_left_bndcon.hpp"
#include "kernelwrap_right_bndcon.hpp"
#include "kernelwrap_set_zero.hpp"
#include "kernelwrap_outerloop_1.hpp"
