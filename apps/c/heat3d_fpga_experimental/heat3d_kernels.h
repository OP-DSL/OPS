//Kernels for heat3d demo app
//
#pragma once

void ops_krnl_interior_init(ACC<float>& data, const int * idx, const float * angle_res_x, const float * angle_res_y, const float * angle_res_z)
{
	data(0,0,0) = sin((*angle_res_x) * idx[0]) * sin((*angle_res_y) * idx[1]) * sin((*angle_res_z) * idx[2]);
}

void ops_krnl_zero_init(ACC<float>& data)
{
	data(0,0,0) = 0.0;
}

void ops_krnl_const_init(ACC<float>& data, const float *constant)
{
	data(0,0,0) = *constant;
}

void ops_krnl_copy(ACC<float> &data, const ACC<float>& data_new)
{
	data(0,0,0) = data_new(0,0,0);
}

void ops_krnl_heat3D(ACC<float> & next, const ACC<float> & current, const float * K, const int * idx)
{
	next(0,0,0) = (1 - 6 * (*K)) * current(0,0,0) 
			+ (*K) * (current(1,0,0) + current(-1,0,0) + current(0,1,0) + current(0,-1,0) + current(0,0,1) + current(0,0,-1));
}