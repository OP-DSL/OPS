/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OPS distribution.
 *
 * Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * The name of Mike Giles may not be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file
 * @brief Utilities for copy and set the memory of a ops_dat
 * @author Jianping Meng
 * @details Utilities for copy and set the memory of a ops_dat
 */


#include "ops_lib_core.h"

void fetch_loop_slab(char *buf, char *dat, const int *buf_size,
                     const int *dat_size, const int *d_m, int elem_size,
                     int dat_dim, const int *range_max_dim) {
  // TODO: add OpenMP here if needed

#if OPS_MAX_DIM > 4
  for (int m = 0; m < buf_size[4]; m++) {
#endif
#if OPS_MAX_DIM > 3
    for (int l = 0; l < buf_size[3]; l++) {
#endif
      for (int k = 0; k < buf_size[2]; k++) {
        for (int j = 0; j < buf_size[1]; j++) {
          size_t buf_index{0}, dat_index{0};
          size_t moff_buf{0}, moff_dat{0}, loff_buf{0}, loff_dat{0};
#if OPS_MAX_DIM > 4
          moff_buf = m * buf_size[0] * buf_size[1] * buf_size[2] * buf_size[3];
          moff_dat = (range_max_dim[2 * 4] + m - d_m[4]) * dat_size[3] *
                     dat_size[2] * dat_size[1] * dat_size[0];
#endif
#if OPS_MAX_DIM > 3
          loff_buf = l * buf_size[0] * buf_size[1] * buf_size[2];
          loff_dat = (range_max_dim[2 * 3] + l - d_m[3]) * dat_size[2] *
                     dat_size[1] * dat_size[0];
#endif
          if (OPS_instance::getOPSInstance()->OPS_soa == 1) {
            for (int i = 0; i < buf_size[0]; i++) {
              for (int d = 0; d < dat_dim; d++) {
                const int type_bits{elem_size / dat_dim};
                size_t doff_dat{d};
#if OPS_MAX_DIM > 4
                doff_dat *= (dat_size[4]);
#endif
#if OPS_MAX_DIM > 3
                doff_dat *= (dat_size[3]);
#endif
                doff_dat *= (dat_size[2] * dat_size[1] * dat_size[0]);
                buf_index =
                    (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                     j * buf_size[0] + i) *
                        elem_size +
                    d * type_bits;
                dat_index = (doff_dat + moff_dat + loff_dat +
                             (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                                 dat_size[0] +
                             (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                             range_max_dim[2 * 0] + i - d_m[0]) *
                            type_bits;
                memcpy(&buf[buf_index], &dat[dat_index], type_bits);
              } // d
            }   // i
          } else {
            buf_index = (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                         j * buf_size[0]) *
                        elem_size;
            dat_index = (moff_dat + loff_dat +
                         (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                             dat_size[0] +
                         (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                         range_max_dim[2 * 0] - d_m[0]) *
                        elem_size;
            memcpy(&buf[buf_index], &dat[dat_index], buf_size[0] * elem_size);
          } // OPS_SOA
        }   // j
      }     // k

#if OPS_MAX_DIM > 3
    } // l
#endif
#if OPS_MAX_DIM > 4
  } // m
#endif
}

void set_loop_slab(char *buf, char *dat, const int *buf_size,
                   const int *dat_size, const int *d_m, int elem_size,
                   int dat_dim, const int *range_max_dim) {
  // TODO: add OpenMP here if needed

#if OPS_MAX_DIM > 4
  for (int m = 0; m < buf_size[4]; m++) {
#endif
#if OPS_MAX_DIM > 3
    for (int l = 0; l < buf_size[3]; l++) {
#endif
      for (int k = 0; k < buf_size[2]; k++) {
        for (int j = 0; j < buf_size[1]; j++) {
          size_t buf_index{0}, dat_index{0};
          size_t moff_buf{0}, moff_dat{0}, loff_buf{0}, loff_dat{0};
#if OPS_MAX_DIM > 4
          moff_buf = m * buf_size[0] * buf_size[1] * buf_size[2] * buf_size[3];
          moff_dat = (range_max_dim[2 * 4] + m - d_m[4]) * dat_size[3] *
                     dat_size[2] * dat_size[1] * dat_size[0];
#endif
#if OPS_MAX_DIM > 3
          loff_buf = l * buf_size[0] * buf_size[1] * buf_size[2];
          loff_dat = (range_max_dim[2 * 3] + l - d_m[3]) * dat_size[2] *
                     dat_size[1] * dat_size[0];
#endif
          if (OPS_instance::getOPSInstance()->OPS_soa == 1) {
            for (int i = 0; i < buf_size[0]; i++) {
              for (int d = 0; d < dat_dim; d++) {
                const int type_bits{elem_size / dat_dim};
                size_t doff_dat{d};
#if OPS_MAX_DIM > 4
                doff_dat *= (dat_size[4]);
#endif
#if OPS_MAX_DIM > 3
                doff_dat *= (dat_size[3]);
#endif
                doff_dat *= (dat_size[2] * dat_size[1] * dat_size[0]);
                buf_index =
                    (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                     j * buf_size[0] + i) *
                        elem_size +
                    d * type_bits;
                dat_index = (doff_dat + moff_dat + loff_dat +
                             (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                                 dat_size[0] +
                             (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                             range_max_dim[2 * 0] + i - d_m[0]) *
                            type_bits;
                memcpy(&dat[dat_index], &buf[buf_index], type_bits);
              } // d
            }   // i
          } else {
            buf_index = (moff_buf + loff_buf + k * buf_size[0] * buf_size[1] +
                         j * buf_size[0]) *
                        elem_size;
            dat_index = (moff_dat + loff_dat +
                         (range_max_dim[2 * 2] + k - d_m[2]) * dat_size[1] *
                             dat_size[0] +
                         (range_max_dim[2 * 1] + j - d_m[1]) * dat_size[0] +
                         range_max_dim[2 * 0] - d_m[0]) *
                        elem_size;
            memcpy(&dat[dat_index], &buf[buf_index], buf_size[0] * elem_size);
          } // OPS_SOA
        }   // j
      }     // k

#if OPS_MAX_DIM > 3
    } // l
#endif
#if OPS_MAX_DIM > 4
  } // m
#endif
}