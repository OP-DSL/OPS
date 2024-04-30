#pragma once

#define EPSILON 0.0001
typedef float stencil_type;
extern float dx,dy,dx_2,dy_2,dx_2_dy_2,dx_2_plus_dy_2_mult_2;
extern const unsigned short mem_vector_factor;

void poisson_kernel_populate_cpu(stencil_type* u, stencil_type* f, stencil_type* ref, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            stencil_type x = dx * (stencil_type)(i + d_m[0]);
            stencil_type y = dy * (stencil_type)(j + d_m[1]);

            u[index] = myfun(sin(M_PI * x), cos(2.0 * M_PI * y))-1.0;
            f[index] = (-5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y)); // * dx_2_dy_2;
            ref[index] = sin(M_PI*x)*cos(2.0*M_PI*y);
        }
    }
}

void poisson_kernel_initialguess_cpu(stencil_type* d, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            d[index] = 0.0;
        }
    }
}

void poisson_kernel_stencil_cpu(stencil_type* u, stencil_type* f, stencil_type* u2, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;

//            u2[index] = ((u[index - 1] + u[index + 1]) * dy * dy
//                    + (u[index - grid_size_x] + u[index + grid_size_x]) * dx * dx
//                    - f[index]) / dx_2_plus_dy_2_mult_2;
            u2[index] = ((u[index - 1] + u[index + 1] + u[index + grid_size_x] + u[index - grid_size_x]) * 0.125  + (0.5 * u[index]));

//             printf("Verification:- i: %d, j: %d, index: %d, dx: %f, dy: %f, u2(0,0): %f, f: %f, u1(-1, -1): %f, u(-1,0): %f, u(0,0): %f, u(0,1): %f, u(1,1): %f\n",
//                 i, j, index, dx, dy, u2[index], f[index], u[index - grid_size_x], u[index - 1], u[index], u[index + 1], u[index + grid_size_x]);
        }
    }
}

void poisson_kernel_update_cpu(stencil_type* u2, stencil_type* u, int size[2], int d_m[2], int d_p[2], int range[4])
{
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
    int actual_size_x = size[0] - d_m[0] + d_p[0];

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;
            u2[index] = u[index];
        }
    }
}


bool verify(stencil_type * grid_data1, stencil_type *  grid_data2, int size[2], int d_m[2], int d_p[2], int range[4])
{
    bool passed = true;
    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    for (int j = range[2] - d_m[1]; j < range[3] -d_m[1]; j++)
    {
        for (int i = range[0] - d_m[0]; i < range[1] - d_m[0]; i++)
        {
            int index = j * grid_size_x + i;

            if (abs(grid_data1[index] - grid_data2[index]) > EPSILON)
            {
                std::cerr << "[ERROR] value Mismatch index: (" << i << ", " << j << "), grid_data1: "
						<< grid_data1[index] << ", and grid_data2: " << grid_data2[index] << std::endl;
                passed = false;
            }
        }
    }

    return passed;
}
