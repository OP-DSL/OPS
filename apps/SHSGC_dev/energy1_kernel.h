#ifndef ENERGY1_KERNEL_H
#define ENERGY1_KERNEL_H

#include "vars.h"


void energy1_kernel( double *mu_x, double *T_x, double *viscE_res){ 
		double cq = 1.0 / (gam1 * pr * pow(Mach,2));
        double temp = T_x[OPS_ACC1(0)]*mu_x[OPS_ACC0(0)] ;
		temp = temp * cq;
		
		viscE_res[OPS_ACC2(0)] += temp ;
}

#endif
