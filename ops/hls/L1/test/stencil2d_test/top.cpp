
#include "top.hpp"

void dut()
{
    Stencil2D<stencil_type, num_points, vector_factor, coef_type, stencil_size_p, stencil_size_q> cross_stencil;

    ops::hls::GridPropertyCore gridProp;

    gridProp.gridsize[0] = 100;
    gridProp.gridsize[1] = 100;
    gridProp.dim = 2;
    gridProp.d_p[0] = +1;
    gridProp.d_p[1] = +1;
    gridProp.d_m[0] = -1;
    gridProp.d_m[1] = -1;
    gridProp.offset[0] = 0;
    gridProp.offset[1] = 1;
    gridProp.xblocks = gridProp.gridsize[0] >> 3;

    short points[] = {0,-1, -1,0, 0,0, 1,0, 0,1};
    float coef[] = {0.25, 0.25, 1, 0.25, 0.25};

    cross_stencil.setGridProp(gridProp);
    cross_stencil.setPoints(points);
    cross_stencil.setCoef(coef);

#ifndef __SYTHESIS__
    unsigned int stencil_sizes[2];
    short read_points[num_points * 2];
    float read_coef[num_points];

    cross_stencil.getPoints(read_points);
    cross_stencil.getCoef(read_coef);
    stencil_sizes[0] = cross_stencil.m_sizes[0];
    stencil_sizes[1] = cross_stencil.m_sizes[1];

    std::cout << "SUCESSFUL INSTANTIATION OF STENCIL CORE" << std::endl;
    std::cout << "STENCIL SIZE: (" << stencil_sizes[0] << ", " << stencil_sizes[1] << ")" << std::endl;

    std::cout << "POINTS: ";

    for (int i = 0; i < num_points; i++)
    {
        std::cout << "(" << read_points[2 * i] << ", " << read_points[2 * i + 1] << ") ";
    }

    std::cout << std::endl;

    std::cout << "COEF: ";

    for (int i = 0; i < num_points; i++)
    {
        std::cout << read_coef[i] << (i == num_points - 1 ? "" :  ", ");
    }

    std::cout << std::endl << std::endl; 
#endif   
}