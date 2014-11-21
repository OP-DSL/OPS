#ifndef TVD_KERNEL_H
#define TVD_KERNEL_H

#include "vars.h"


void tvd_kernel(const double *tht, double* ep2) {  	
    double maxim;
		for (int m=0; m < 3 ;m++) {
			if (tht[OPS_ACC_MD1(m,0)] > tht[OPS_ACC_MD1(m,1)]) 
				maxim = tht[OPS_ACC_MD1(m,0)];
			else
				maxim = tht[OPS_ACC_MD1(m,1)];
			ep2[OPS_ACC_MD1(m,0)] = akap2 * maxim;
		}
}

#endif