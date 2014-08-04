#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest, const ops_halo *__restrict halo) {
  const char * __restrict src = dat->data+src_offset*dat->size;
  for (unsigned int i = 0; i < halo->count; i ++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->stride;
    dest += halo->blocklength;
  }
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src, const ops_halo *__restrict halo) {
  char * __restrict dest = dat->data+dest_offset*dat->size;
  for (unsigned int i = 0; i < halo->count; i ++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->blocklength;
    dest += halo->stride;
  }
}

void ops_H_D_exchanges_cuda(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit_cuda(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit_opencl(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_comm_realloc(char **ptr, int size, int prev) {
  if (*ptr == NULL) {
    *ptr = (char *)malloc(size);
  } else {
    *ptr = (char*)realloc(*ptr, size);
  }
}