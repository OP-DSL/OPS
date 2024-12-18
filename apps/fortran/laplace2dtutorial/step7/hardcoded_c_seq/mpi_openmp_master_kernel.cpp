
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif

int imax;
int jmax;
double pi;

extern "C"
void ops_decl_const_f2c(char const *name, int dim, int size, char *dat) {
    if(!strcmp(name, "imax")) {
        imax = *(int *)dat;
//        printf("imax: %d\n", imax);
    }
    else
    if(!strcmp(name, "jmax")) {
        jmax = *(int *)dat;
//        printf("jmax: %d\n", jmax);
    }
    else
    if(!strcmp(name, "pi")) {
        pi = *(double *)dat;
//        printf("pi: %lf\n", pi);
    }
    else
    {
        throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
    }
}

void ops_init_backend(){}

#include "set_zero_kernel.cpp"
#include "left_bndcon_kernel.cpp"
#include "right_bndcon_kernel.cpp"
#include "apply_stencil_kernel.cpp"
#include "copy_kernel.cpp"
