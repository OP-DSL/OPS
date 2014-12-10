#ifndef ENERGY2_KERNEL_H
#define ENERGY2_KERNEL_H

#include "vars.h"


void energy2_kernel( double *mu, double *T_xx, double *viscE_res){ 
		double cq = 1.0 / (gam1 * pr * pow(Mach,2));
        double temp = T_xx[OPS_ACC1(0)]*mu[OPS_ACC0(0)] ;
		temp = temp * cq; 
		
		viscE_res[OPS_ACC2(0)] += temp ;
}

#endif
