#ifndef block_0_KERNEL_H
#define block_0_KERNEL_H

void complex_numbers_block0_0_kernel(const double *phi , double *wk0)
{
wk0[OPS_ACC1(0)] = ((rc0)*phi[OPS_ACC0(-4)] - rc1*phi[OPS_ACC0(-3)] + (rc2)*phi[OPS_ACC0(-2)] - rc3*phi[OPS_ACC0(-1)] + (rc3)*phi[OPS_ACC0(1)] - rc2*phi[OPS_ACC0(2)] + (rc1)*phi[OPS_ACC0(3)] - rc0*phi[OPS_ACC0(4)])/deltai0;
}


void complex_numbers_block0_1_kernel(const double *wk0 , double *wk1)
{
wk1[OPS_ACC1(0)] = -c0*wk0[OPS_ACC0(0)];
}


void complex_numbers_block0_2_kernel(const double *phi_old , const double *wk1 , double *phi , const double *rknew)
{
phi[OPS_ACC2(0)] = deltat*rknew[0]*wk1[OPS_ACC1(0)] + phi_old[OPS_ACC0(0)];
}


void complex_numbers_block0_3_kernel(const double *wk1 , double *phi_old , const double *rkold)
{
phi_old[OPS_ACC1(0)] = deltat*rkold[0]*wk1[OPS_ACC0(0)] + phi_old[OPS_ACC1(0)];
}


void complex_numbers_block0_4_kernel(const double *phi , double *phi_old)
{
phi_old[OPS_ACC1(0)] = phi[OPS_ACC0(0)];
}


void complex_numbers_block0_5_kernel(double *phi , const int *idx)
{
phi[OPS_ACC0(0)] = sin(2*deltai0*idx[0]*M_PI);
}


void complex_numbers_block0_cn_kernel(const double *phi , double *real , double *imaginary)
{
    double __complex__ z = 1.0I;
    double __complex__ coeff = z;

    *real = *real + phi[OPS_ACC0(0)];
    coeff = coeff*z;
    *real = *real + __real__ coeff*phi[OPS_ACC0(0)];
    *imaginary = *imaginary + __imag__ coeff*phi[OPS_ACC0(0)];
}

#endif
