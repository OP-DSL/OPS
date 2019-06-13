
void rms_kernel( const double* array, double* rms) {
  //*rms = *rms + array[OPS_ACC0(0,0,0)]*array[OPS_ACC0(0,0,0)];
  *rms += array[OPS_ACC0(0,0,0)];
}