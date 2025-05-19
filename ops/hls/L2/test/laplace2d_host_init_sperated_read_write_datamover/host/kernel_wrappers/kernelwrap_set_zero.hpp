#pragma once
#include <ops_hls_rt_support.h>

void kernel_set_zero(float& arg0_out_0)
{
	arg0_out_0 = 0.0;
}

void ops_par_loop_set_zero(int dim , int* ops_range, ops::hls::Grid<float>& arg0)
{
	ops::hls::AccessRange range;
	opsRange2hlsRange(dim, ops_range, range, arg0.originalProperty);

	for (unsigned short j = range.start[1]; j < range.end[1]; j++)
	{
		for (unsigned short i = range.start[0]; i < range.end[0]; i++)
		{
			auto index = i + j * arg0.originalProperty.grid_size[0];
			kernel_set_zero(arg0.hostBuffer[index]);
		}
	}
	arg0.isHostBufDirty = true;
}
