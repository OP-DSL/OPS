#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest, const ops_int_halo *__restrict halo) {
  const char * __restrict src = dat->data+src_offset*dat->elem_size;
  for (unsigned int i = 0; i < halo->count; i ++) {
    memcpy(dest, src, halo->blocklength);
    src += halo->stride;
    dest += halo->blocklength;
  }
}

void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src, const ops_int_halo *__restrict halo) {
  char * __restrict dest = dat->data+dest_offset*dat->elem_size;
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

void ops_cpHostToDevice(void ** data_d, void ** data_h, int size ) {
  (void)data_d;
  (void)data_h;
  (void)size;
}

void ops_halo_copy_tobuf(char * dest, int dest_offset, ops_dat src,
                        int rx_s, int rx_e,
                        int ry_s, int ry_e,
                        int rz_s, int rz_e,
                        int x_step, int y_step, int z_step,
                        int buf_strides_x, int buf_strides_y, int buf_strides_z) {
  for (int k = rz_s; (z_step==1 ? k < rz_e : k > rz_e); k += z_step) {
    for (int j = ry_s; (y_step==1 ? j < ry_e : j > ry_e); j += y_step) {
      for (int i = rx_s; (x_step==1 ? i < rx_e : i > rx_e); i += x_step) {
        memcpy(dest + dest_offset + ((k-rz_s)*z_step*buf_strides_z+ (j-ry_s)*y_step*buf_strides_y + (i-rx_s)*x_step*buf_strides_x)*src->elem_size,
               src->data + (k*src->size[0]*src->size[1]+j*src->size[0]+i)*src->elem_size, src->elem_size);
      }
    }
  }
}

void ops_halo_copy_frombuf(ops_dat dest,
                        char * src, int src_offset,
                        int rx_s, int rx_e,
                        int ry_s, int ry_e,
                        int rz_s, int rz_e,
                        int x_step, int y_step, int z_step,
                        int buf_strides_x, int buf_strides_y, int buf_strides_z) {
  for (int k = rz_s; (z_step==1 ? k < rz_e : k > rz_e); k += z_step) {
    for (int j = ry_s; (y_step==1 ? j < ry_e : j > ry_e); j += y_step) {
      for (int i = rx_s; (x_step==1 ? i < rx_e : i > rx_e); i += x_step) {
        memcpy(dest->data + (k*dest->size[0]*dest->size[1]+j*dest->size[0]+i)*dest->elem_size,
                src + src_offset + ((k-rz_s)*z_step*buf_strides_z+ (j-ry_s)*y_step*buf_strides_y + (i-rx_s)*x_step*buf_strides_x)*dest->elem_size, dest->elem_size);
      }
    }
  }
}