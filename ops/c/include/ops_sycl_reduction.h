#ifndef OPS_SYCL_REDUCTION_H_
#define OPS_SYCL_REDUCTION_H_
#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
  * @brief Core header file for the ops sycl reductions - adapted from OP2
  * @author Gabor Daniel Balogh
  * @details This file provides an optimised implementation for reduction of
  * OPS global variables. It is separated from the op_sycl_rt_support.h file
  * because the reduction code is based on C++ templates, while the other file
  * only includes C routines.
  */

#include "ops_sycl_rt_support.h"


/**
 * @brief Performs reduction on a single value.
 *
 * @tparam reduction The reduction type, one of OPS_INC, OPS_MIN and OPS_MAX.
 * @tparam primitive type of the reduction, deduced from dat_l
 *
 * @param dat_g An accessor over the output array. Typically over instance->OPS_reduct_d.
 * @param offset Used only if linear_id == 0. Gives the index of the output in dar_g.
 * @param dat_l The local result of the reduction computed by the user kernel.
 * @param temp  Local memory used to store temporary values.
 * @param item_id The corresponding sycl item.
 */
template <ops_access reduction, class T, class item_type>
void ops_reduction_sycl(T *dat_g, T dat_l, T* temp, item_type &item_id, int group_size) {

  item_id.barrier(cl::sycl::access::fence_space::local_space); /* important to finish all previous activity */

  size_t linear_id = item_id.get_local_linear_id();
  temp[linear_id] = dat_l;
  item_id.barrier(cl::sycl::access::fence_space::local_space); /* important to finish all previous activity */

  int d0 = 1 << (31 - cl::sycl::clz(((int)(group_size) - 1)));

  for (size_t d = d0; d > 0; d >>= 1) {
    if (linear_id < d) {
      T dat_t = temp[linear_id + d];

      switch (reduction) {
        case OPS_INC:
          dat_l = dat_l + dat_t;
          break;
        case OPS_MIN:
          if (dat_t < dat_l)
            dat_l = dat_t;
          break;
        case OPS_MAX:
          if (dat_t > dat_l)
            dat_l = dat_t;
          break;
      }
      temp[linear_id] = dat_l;
    }
    item_id.barrier(cl::sycl::access::fence_space::local_space);
  }

  if (linear_id == 0) {
    switch (reduction) {
      case OPS_INC:
        dat_g[0] += dat_l;
        break;
      case OPS_MIN:
        if (dat_l < dat_g[0])
          dat_g[0] = dat_l;
        break;
      case OPS_MAX:
        if (dat_l > dat_g[0])
          dat_g[0] = dat_l;
        break;
    }
  }
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* OPS_SYCL_REDUCTION_H_ */
