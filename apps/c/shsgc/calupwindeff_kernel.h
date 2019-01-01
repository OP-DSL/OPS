#ifndef CALUPWINDEFF_KERNEL_H
#define CALUPWINDEFF_KERNEL_H

#include "vars.h"


void calupwindeff_kernel(const ACC<double>& cmp, const ACC<double> &gt, const ACC<double>& cf,
                         const ACC<double>& al, const ACC<double>& ep2, const ACC<double>& r, ACC<double>& eff) {

		double e1 = (cmp(0,0) * (gt(0,0) + gt(0,1)) - cf(0,0) * al(0,0)) * ep2(0,0);
		double e2 = (cmp(1,0) * (gt(1,0) + gt(1,1)) - cf(1,0) * al(1,0)) * ep2(1,0);
		double e3 = (cmp(2,0) * (gt(2,0) + gt(2,1)) - cf(2,0) * al(2,0)) * ep2(2,0);

		eff(0,0)=e1 * r(0,0) + e2 * r(1,0) + e3 * r(2,0);
		eff(1,0)=e1 * r(3,0) + e2 * r(4,0) + e3 * r(5,0);
		eff(2,0)=e1 * r(6,0) + e2 * r(7,0) + e3 * r(8,0);
}
#endif
