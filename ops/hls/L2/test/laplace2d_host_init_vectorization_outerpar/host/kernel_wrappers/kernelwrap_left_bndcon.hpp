#pragma once
#include <ops_hls_rt_support.h>

// extern int imax, jmax;
// extern float pi;

void kernel_left_bndcon(float& arg0_out_0, const int *idx)
{
	arg0_out_0 = sin(pi * (idx[1] + 1) / (jmax + 1));
}

void ops_par_loop_left_bndcon(int dim , int* ops_range, ops::hls::Grid<float>& arg0)
{
	ops::hls::AccessRange range;
	opsRange2hlsRange(dim, ops_range, range, arg0.originalProperty);
	constexpr int arg0_0_stencil_offset[] = {0,0,0};

	for (unsigned short j = range.start[1]; j < range.end[1]; j++)
	{
		for (unsigned short i = range.start[0]; i < range.end[0]; i++)
		{
			auto index = i + j * arg0.originalProperty.grid_size[0];
			ops::hls::IdxType idx({i - arg0.originalProperty.d_m[0], j - arg0.originalProperty.d_m[1], 0});
			kernel_left_bndcon(arg0.hostBuffer[getOffset(arg0_0_stencil_offset, arg0.originalProperty, i, j)], idx);
		}
	}

	arg0.isHostBufDirty = true;
}

