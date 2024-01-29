#ifndef block_0_KERNEL_H
#define block_0_KERNEL_H

void complex_numbers_block0_0_kernel(const ACC<double> &phi , ACC<double> &wk0)
{
wk0(0) = ((rc0)*phi(-4) - rc1*phi(-3) + (rc2)*phi(-2) - rc3*phi(-1) + (rc3)*phi(1) - rc2*phi(2) + (rc1)*phi(3) - rc0*phi(4))/deltai0;
}


void complex_numbers_block0_1_kernel(const ACC<double> &wk0 , ACC<double> &wk1)
{
wk1(0) = -c0*wk0(0);
}


void complex_numbers_block0_2_kernel(const ACC<double> &phi_old , const ACC<double> &wk1 , ACC<double> &phi , const double* rknew)
{
phi(0) = deltat*rknew[0]*wk1(0) + phi_old(0);
}


void complex_numbers_block0_3_kernel(const ACC<double> &wk1 , ACC<double> &phi_old , const double* rkold)
{
phi_old(0) = deltat*rkold[0]*wk1(0) + phi_old(0);
}


void complex_numbers_block0_4_kernel(const ACC<double> &phi , ACC<double> &phi_old)
{
phi_old(0) = phi(0);
}


void complex_numbers_block0_5_kernel(ACC<double> &phi , const int *idx)
{
phi(0) = sin(2*deltai0*idx[0]*M_PI);
}


void complex_numbers_block0_cn_kernel(const ACC<double> &phi , double *real , double *imaginary)
{
    double __complex__ z = 1.0I;
    double __complex__ coeff = z;

    *real = *real + phi(0);
    coeff = coeff*z;
    *real = *real + __real__ coeff*phi(0);
    *imaginary = *imaginary + __imag__ coeff*phi(0);
}

#endif
