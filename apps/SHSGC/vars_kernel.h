#ifndef VARS_KERNEL_H
#define VARS_KERNEL_H

#include "vars.h"


void vars_kernel(const double* alam, const double* al, const double *gt, double* cmp,  double* cf) {
  
  double  anu, aaa, ga, qf, ww;
  for (int m=0; m < 3 ;m++) {
			anu = alam[OPS_ACC_MD0(m,0)];
			aaa = al[OPS_ACC_MD1(m,0)];
			ga = aaa * ( gt[OPS_ACC_MD2(m,1)] - gt[OPS_ACC_MD2(m,0)]) / (pow(aaa,2.f) + del2);
			qf = sqrt ( con + pow(anu,2.f));
			cmp[OPS_ACC_MD3(m,0)] = 0.50 * qf;
			ww = anu + cmp[OPS_ACC_MD3(m,0)] * ga; 
			qf = sqrt(con + pow(ww,2.f));
			cf[OPS_ACC_MD4(m,0)] = qf;
		}  
}

#endif