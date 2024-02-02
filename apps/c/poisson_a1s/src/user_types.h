#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef OPS_FUN_PREFIX
#define OPS_FUN_PREFIX
#endif

static inline OPS_FUN_PREFIX double myfun(double a, double b) {
  return a*b+1.0;
}

