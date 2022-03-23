#ifndef MULTIDIM_KERNEL_H
#define MULTIDIM_KERNEL_H

void multidim_kernel(ACC<double> &val, int *idx){
  val(0,0,0) = (double)(idx[0]);
  val(1,0,0) = (double)(idx[1]);
  // printf("%d %d: %p
  // %p\n",idx[0],idx[1],&val[OPS_ACC_MD0(0,0,0)],&val[OPS_ACC_MD0(1,0,0)]);
}

void KerSetCoordinates_test(ACC<double>& coordinates, const int* idx,
                         const double* coordX, const double* coordY) {

    coordinates(0, 0, 0) = 1;
    coordinates(1, 0, 0) = 4;
}

void KerSetCoordinates(ACC<double>& coordinates, const int* idx,
                         const double* coordX, const double* coordY) {

    coordinates(0, 0, 0) = coordX[idx[0]];
    coordinates(1, 0, 0) = coordY[idx[1]];
}

#endif //MULTIDIM_KERNEL_H
