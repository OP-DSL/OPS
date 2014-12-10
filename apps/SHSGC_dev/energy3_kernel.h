#ifndef ENERGY3_KERNEL_H
#define ENERGY3_KERNEL_H

#include "vars.h"


void energy3_kernel( double *u, double *u_x,double *mu, double *viscE_res){ 
	
		double temp = (4*mu[OPS_ACC2(0)]*u[OPS_ACC0(0)]*u_x[OPS_ACC1(0)])/3; 
		
		viscE_res[OPS_ACC3(0)] += temp ;
}

#endif
