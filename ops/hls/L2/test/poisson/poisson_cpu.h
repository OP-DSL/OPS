#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <ops_hls_defs.hpp>

int stencil_computation(float* current, float* next, ops::hls::GridPropertyCore data_g, unsigned int batch = 1);
double square_error(float* current, float* next, ops::hls::GridPropertyCore data_g, unsigned int batch = 1);
int copy_grid(float* grid_s, float* grid_d, ops::hls::GridPropertyCore data_g, unsigned int batch = 1);
int initialise_grid(float* grid, ops::hls::GridPropertyCore data_g, unsigned int batch = 1);