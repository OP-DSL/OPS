#pragma once

void ops_krnl_interior_init(ACC<float> & data, const int *idx, const float *deltaS, const float *strikePrice)
{
	float tmpVal = (idx[0] + 1)*(*deltaS) - (*strikePrice);
	data(0) = tmpVal > 0.0 ? tmpVal : 0.0;
}

void ops_krnl_zero_init(ACC<float> &data)
{
	data(0) = 0.0;
}

void ops_krnl_const_init(ACC<float> &data, const float *constant)
{
	data(0) = *constant;
}

void ops_krnl_copy(ACC<float> &data, const ACC<float>& data_new)
{
	data(0) = data_new(0);
}

void ops_krnl_blacksholes(ACC<float> & current, const ACC<float> & next, const ACC<float> & a, const ACC<float> & b, const ACC<float> & c)
{
	current(0) = a(0) * next (-1) + b(0) * next(0) + c(0) * next(1);
}

void ops_krnl_calc_coefficient(ACC<float> & a, ACC<float> & b, ACC<float> & c, const float * alpha, const float * beta, const int * idx)
{
	a(0) = 0.5 * ((*alpha) * (idx[0] + 1) * (idx[0] + 1) - (*beta) * (idx[0] + 1));
	b(0) = 1 - (*alpha) * (idx[0] + 1) * (idx[0] + 1) - (*beta);
	c(0) = 0.5 * ((*alpha) * (idx[0] + 1) * (idx[0] + 1) + (*beta) * (idx[0] + 1));
}