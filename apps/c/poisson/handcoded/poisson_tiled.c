#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef WID
#define WID 256
#endif
#ifndef IT
#define IT 50 
#endif
typedef double real;
int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);//16384;
  int maxstep = atoi(argv[2])+atoi(argv[2])%IT;
  printf("Grid %d^2, WID %d IT %d, STEPS %d\n",N,WID,IT,maxstep);
  real * __restrict__ a1 __attribute__((aligned(64))) = (real*)_mm_malloc((N+2)*(N+2)*sizeof(real),64);
  real * __restrict__ a2 __attribute__((aligned(64))) = (real*)_mm_malloc((N+2)*(N+2)*sizeof(real),64);
  real *temp;
  real dt = 0.002;
  real h = 0.1;

  //initialise
    #pragma omp parallel for shared(a1,a2)
  for (int i = 0; i < N+2; i++) {
    for (int j = 0; j < N+2; j++) {
      a1[i*(N+2)+j] = 0.0;
      a2[i*(N+2)+j] = 0.0;
    }
  }
  a1[(1+N/2)*(N+2)+(1+N/2)] = 1.0f;

  double t1 = omp_get_wtime();
  __assume_aligned(a1,64);
  __assume_aligned(a2,64);
  for (int step = 0; step < maxstep; step+=IT) {
    //first tile
    for (int it = 0; it < IT; it++) {
    #pragma omp parallel for shared(a1,a2) //schedule(dynamic,1)
      for (int i = 1; i < WID-it+1; i++) {
        for (int j = 1; j < N+1; j++) {
          a2[i*(N+2)+j] = 0.125f*(a1[i*(N+2)+j+1] - 2.0f*a1[i*(N+2)+j] + a1[i*(N+2)+j-1]) +
                          0.125f*(a1[(i+1)*(N+2)+j] -2.0f*a1[i*(N+2)+j]+a1[(i-1)*(N+2)+j]) + a1[i*(N+2)+j];
//          a2[i*(N+2)+j] = a1[i*(N+2)+j] +
//            dt * (a1[i*(N+2)+j-1] + a1[(i-1)*(N+2)+j]
//                -4*a1[i*(N+2)+j] +
//                a1[i*(N+2)+j+1] + a1[(i+1)*(N+2)+j])/(h*h);
        }
      }
//SWAP!
    temp = a1;
    a1 = a2;
    a2 = temp;
    }

    //subsequent tiles
    for (int tile = 1; tile < ((N-1)/WID+1)-1; tile++) {
      for (int it = 0; it < IT; it++) {
    #pragma omp parallel for shared(a1,a2) //schedule(dynamic,1)
        for (int i = tile*WID-it+1; i < tile*WID-it+1+WID; i++) {
          for (int j = 1; j < N+1; j++) {
            a2[i*(N+2)+j] = 0.125f*(a1[i*(N+2)+j+1] - 2.0f*a1[i*(N+2)+j] + a1[i*(N+2)+j-1]) +
                          0.125f*(a1[(i+1)*(N+2)+j] -2.0f*a1[i*(N+2)+j]+a1[(i-1)*(N+2)+j]) + a1[i*(N+2)+j];
           // a2[i*(N+2)+j] = a1[i*(N+2)+j] +
           //   dt * (a1[i*(N+2)+j-1] + a1[(i-1)*(N+2)+j]
           //       -4*a1[i*(N+2)+j] +
           //       a1[i*(N+2)+j+1] + a1[(i+1)*(N+2)+j])/(h*h);
          }
        }
//SWAP
    temp = a1;
    a1 = a2;
    a2 = temp;
      }
    }

    //last tile
    for (int it = 0; it < IT; it++) {
    #pragma omp parallel for shared(a1,a2) //schedule(dynamic,1)
      for (int i = ((N-1)/WID)*WID-it+1; i < N+1; i++) {
        for (int j = 1; j < N+1; j++) {
          a2[i*(N+2)+j] = 0.125f*(a1[i*(N+2)+j+1] - 2.0f*a1[i*(N+2)+j] + a1[i*(N+2)+j-1]) +
                          0.125f*(a1[(i+1)*(N+2)+j] -2.0f*a1[i*(N+2)+j]+a1[(i-1)*(N+2)+j]) + a1[i*(N+2)+j];
          //a2[i*(N+2)+j] = a1[i*(N+2)+j] +
          //  dt * (a1[i*(N+2)+j-1] + a1[(i-1)*(N+2)+j]
          //      -4*a1[i*(N+2)+j] +
          //      a1[i*(N+2)+j+1] + a1[(i+1)*(N+2)+j])/(h*h);
        }
      }
//SWAP
    temp = a1;
    a1 = a2;
    a2 = temp;
    }

    //temp = a1;
    //a1 = a2;
    //a2 = temp;
  }
  double t2 = omp_get_wtime();
  real rms = 0.0;
  #pragma omp parallel for shared(a2) reduction(+:rms)
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      rms += a2[i*(N+2)+j]*a2[i*(N+2)+j];
    }
  }
#ifdef DEBUG
  FILE *fp,*fp2;
  fp = fopen("spa.txt", "w");
  fp2 = fopen("spa2.txt", "w");

  for (int i = 0; i < N+2; i++) {
    for (int j = 0; j < N+2; j++) {
      fprintf(fp, "%2.5f ", a1[i*(N+2)+j]);
      fprintf(fp2, "%2.5f ", a2[i*(N+2)+j]);
    }
    fprintf(fp, "\n");
    fprintf(fp2, "\n");
  }
  fclose(fp);
  fclose(fp2);
#endif
  printf("%g\n",rms);
  double moved = 2.0 * N*N*maxstep*sizeof(real)/1024.0/1024.0/1024.0;
  printf("Done in %g seconds %g GB/s\n",t2-t1, moved/(t2-t1));
  printf("%g\n",moved/(t2-t1));
  return 1;
}

