#include <ops_lib_core.h>
#include <cuda.h>

void fetch_test(ops_dat dat) {

  double slab2[] = {-1, -2, -3, -4};
  double *slab2_d;
  cudaMalloc((void**)&slab2_d, 4*sizeof(double));
  cudaMemcpy(slab2_d, slab2, 4*sizeof(double), cudaMemcpyHostToDevice);
  int slab2_range[] = {6,8,6,8};
  ops_dat_set_data_slab_memspace(dat, 0, (char*)slab2_d, slab2_range, OPS_DEVICE);

  int disp[OPS_MAX_DIM];
  int size[OPS_MAX_DIM];
  ops_dat_get_extents(dat, 0, disp, size);

  size_t bytes = sizeof(double) * size[0] * size[1];
  double *data_h = (double*)ops_malloc(bytes);
  double *data_d;
  cudaMalloc((void**)&data_d, bytes);

  ops_dat_fetch_data_memspace(dat, 0, (char*)data_d, OPS_DEVICE);

  cudaMemcpy(data_h, data_d, bytes, cudaMemcpyDeviceToHost);

  printf("Fetched data:\n");
  for (int j = 0; j < size[1]; j++) {
    for (int i = 0; i < size[0]; i++) {
      printf("%.1lf ", data_h[j*size[0]+i]);
    }
    printf("\n");
  }

  cudaFree(data_d);
  ops_free(data_h);
  
  double *slab_h = (double*)malloc(4*sizeof(double));
  double *slab_d;
  cudaMalloc((void**)&slab_d, 4*sizeof(double));
  int slab_range[] = {10,12,10,12};
  ops_dat_fetch_data_slab_memspace(dat, 0, (char*)slab_d, slab_range, OPS_DEVICE);
  cudaMemcpy(slab_h, slab_d, 4*sizeof(double), cudaMemcpyDeviceToHost);
  ops_printf("2D slab extracted on DEVICE:\n%g %g\n%g %g\n", slab_h[0], slab_h[1], slab_h[2], slab_h[3]);
  free(slab_h);
  cudaFree(slab_d);

}
