#ifndef __OPS_CUDA_REDUCTION_H
#define __OPS_CUDA_REDUCTION_H
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
  * @brief Core header file for the ops cuda reductions - adapted from OP2
  * @author Gihan Mudalige, Istvan Reguly
  * @details This file provides an optimised implementation for reduction of
  * OPS global variables. It is separated from the op_cuda_rt_support.h file
  * because the reduction code is based on C++ templates, while the other file
  * only includes C routines.
  */

#include <cuda.h>

/*
 * reduction routine for arbitrary datatypes
 */

template <ops_access reduction, class T>
__inline__ __device__ void ops_reduction_cuda(volatile T *dat_g, T dat_l) {
  extern __shared__ volatile char temp2[];
  __shared__ volatile T *temp;
  temp = (T *)temp2;

  T dat_t;

  __syncthreads(); /* important to finish all previous activity */

  int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  temp[tid] = dat_l;

  // first, cope with blockDim.x perhaps not being a power of 2

  __syncthreads();

  int d = 1 << (31 - __clz(((int)(blockDim.x * blockDim.y * blockDim.z) - 1)));
  // d = blockDim.x/2 rounded up to nearest power of 2

  if (tid + d < blockDim.x * blockDim.y * blockDim.z) {
    dat_t = temp[tid + d];

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

    temp[tid] = dat_l;
  }

  // second, do reductions involving more than one warp

  for (d >>= 1; d > warpSize; d >>= 1) {
    __syncthreads();

    if (tid < d) {
      dat_t = temp[tid + d];

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

      temp[tid] = dat_l;
    }
  }

  // third, do reductions involving just one warp

  __syncthreads();

  if (tid < warpSize) {
    for (; d > 0; d >>= 1) {
      if (tid < d) {
        dat_t = temp[tid + d];

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

        temp[tid] = dat_l;
      }
    }

    // finally, update global reduction variable

    if (tid == 0) {
      switch (reduction) {
      case OPS_INC:
        *dat_g = *dat_g + dat_l;
        break;
      case OPS_MIN:
        if (dat_l < *dat_g)
          *dat_g = dat_l;
        break;
      case OPS_MAX:
        if (dat_l > *dat_g)
          *dat_g = dat_l;
        break;
      }
    }
  }
}

__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ < 600
// double atomicAdd
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template <ops_access reduction, class T>
__inline__ __device__ void ops_reduction_cuda(T *dat_g, T dat_l, int count, int stride, int out_stride) {
  extern __shared__ volatile char temp2[];
  __shared__ volatile T *temp;
  temp = (T *)temp2;

  T dat_t;

  __syncthreads(); /* important to finish all previous activity */

  int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  temp[tid] = dat_l;

  // first, cope with blockDim.x perhaps not being a power of 2

  __syncthreads();

  // d = blockDim.x/2 rounded up to nearest power of 2
  int d = 1 << (31 - __clz(((int)count - 1)));
  int batches = (blockDim.x * blockDim.y * blockDim.z)/count;
  int my_batch_lane = stride==1 ? tid%count : tid/stride;
  int my_batch = stride==1 ? tid/count : tid%stride;
  int my_batch_end = tid + (count-my_batch_lane)*stride;

  d *= stride;

  if (tid + d < my_batch_end ) {
    dat_t = temp[tid+d];

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

    temp[tid] = dat_l;
  }

  // second, do reductions involving more than one warp
  for (d >>= 1; d > MAX(warpSize,stride); d >>= 1) {
    __syncthreads();

    if (tid < d) {
      dat_t = temp[tid + d];

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

      temp[tid] = dat_l;
    }
  }

  // third, do reductions involving just one warp

  __syncthreads();

  if (tid < warpSize) {
    for (; d >= stride; d >>= 1) {
      if (tid < d) {
        dat_t = temp[tid + d];

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

        temp[tid] = dat_l;
      }
    }
    // finally, update global reduction variable

    if (my_batch_lane == 0) {
      dat_g += my_batch * out_stride;
      switch (reduction) {
      case OPS_INC:
        atomicAdd(dat_g, dat_l);
        //*dat_g = *dat_g + dat_l;
        break;
      case OPS_MIN:
        atomicMin(dat_g, dat_l);
        //if (dat_l < *dat_g)
          //*dat_g = dat_l;
        break;
      case OPS_MAX:
        atomicMax(dat_g, dat_l);
        //if (dat_l > *dat_g)
        //  *dat_g = dat_l;
        break;
      }
    }
  }
}



/*
 * reduction routine for arbitrary datatypes
 * (alternative version using just one warp)
 *
 */

template <ops_access reduction, class T>
__inline__ __device__ void ops_reduction_alt(volatile T *dat_g, T dat_l) {
  extern __shared__ volatile T temp[];
  T dat_t;

  __syncthreads(); /* important to finish all previous activity */

  int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  temp[tid] = dat_l;

  __syncthreads();

  // set number of active threads

  int d = warpSize;

  if (blockDim.x*blockDim.y*blockDim.z < warpSize)
    d = 1 << (31 - __clz((int)(blockDim.x * blockDim.y * blockDim.z)));
  // this gives blockDim.x rounded down to nearest power of 2

  if (tid < d) {

    // first, do reductions for each thread

    for (int t = tid + d; t < blockDim.x * blockDim.y * blockDim.z; t += d) {
      dat_t = temp[t];

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
    }

    temp[tid] = dat_l;

    // second, do reductions to combine thread reductions

    for (d >>= 1; d > 0; d >>= 1) {
      if (tid < d) {
        dat_t = temp[tid + d];

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

        temp[tid] = dat_l;
      }
    }

    // finally, update global reduction variable

    if (tid == 0) {
      switch (reduction) {
      case OPS_INC:
        *dat_g = *dat_g + dat_l;
        break;
      case OPS_MIN:
        if (dat_l < *dat_g)
          *dat_g = dat_l;
        break;
      case OPS_MAX:
        if (dat_l > *dat_g)
          *dat_g = dat_l;
        break;
      }
    }
  }
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* __OPS_CUDA_REDUCTION_H */
