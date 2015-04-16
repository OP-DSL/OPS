#ifndef CALUPWINDEFF_KERNEL_H
#define CALUPWINDEFF_KERNEL_H

#include "vars.h"


void calupwindeff_kernel(const double* cmp, const double *gt, const double* cf,
                         const double* al, const double* ep2, const double* r, double* eff) {

		double e1 = (cmp[OPS_ACC_MD0(0,0)] * (gt[OPS_ACC_MD1(0,0)] + gt[OPS_ACC_MD1(0,1)]) - cf[OPS_ACC_MD2(0,0)] * al[OPS_ACC_MD3(0,0)]) * ep2[OPS_ACC_MD4(0,0)];
		double e2 = (cmp[OPS_ACC_MD0(1,0)] * (gt[OPS_ACC_MD1(1,0)] + gt[OPS_ACC_MD1(1,1)]) - cf[OPS_ACC_MD2(1,0)] * al[OPS_ACC_MD3(1,0)]) * ep2[OPS_ACC_MD4(1,0)];
		double e3 = (cmp[OPS_ACC_MD0(2,0)] * (gt[OPS_ACC_MD1(2,0)] + gt[OPS_ACC_MD1(2,1)]) - cf[OPS_ACC_MD2(2,0)] * al[OPS_ACC_MD3(2,0)]) * ep2[OPS_ACC_MD4(2,0)];

		eff[OPS_ACC_MD6(0,0)]=e1 * r[OPS_ACC_MD5(0,0)] + e2 * r[OPS_ACC_MD5(1,0)] + e3 * r[OPS_ACC_MD5(2,0)];
		eff[OPS_ACC_MD6(1,0)]=e1 * r[OPS_ACC_MD5(3,0)] + e2 * r[OPS_ACC_MD5(4,0)] + e3 * r[OPS_ACC_MD5(5,0)];
		eff[OPS_ACC_MD6(2,0)]=e1 * r[OPS_ACC_MD5(6,0)] + e2 * r[OPS_ACC_MD5(7,0)] + e3 * r[OPS_ACC_MD5(8,0)];
}

#endif