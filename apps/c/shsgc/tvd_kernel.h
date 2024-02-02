#ifndef TVD_KERNEL_H
#define TVD_KERNEL_H

#include "vars.h"


void tvd_kernel(const ACC<double> &tht, ACC<double>& ep2) {
    double maxim;
		for (int m=0; m < 3 ;m++) {
			if (tht(m,0) > tht(m,1))
				maxim = tht(m,0);
			else
				maxim = tht(m,1);
			ep2(m,0) = akap2 * maxim;
		}
}
#endif
