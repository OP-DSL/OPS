#ifndef LIB_MPI_H
#define LIB_MPI_H
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

/** @brief header file declaring the functions for the ops mpi backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS API calls for the sequential backend
  */

#include "ops_lib_cpp.h"
#include "ops_mpi_core.h"

#ifndef OPS_ACC_MACROS
#ifndef OPS_DEBUG
#define OPS_ACC0(x,y) (x+xdim0*(y))
#define OPS_ACC1(x,y) (x+xdim1*(y))
#define OPS_ACC2(x,y) (x+xdim2*(y))
#define OPS_ACC3(x,y) (x+xdim3*(y))
#define OPS_ACC4(x,y) (x+xdim4*(y))
#define OPS_ACC5(x,y) (x+xdim5*(y))
#define OPS_ACC6(x,y) (x+xdim6*(y))
#define OPS_ACC7(x,y) (x+xdim7*(y))
#define OPS_ACC8(x,y) (x+xdim8*(y))
#define OPS_ACC9(x,y) (x+xdim9*(y))
#define OPS_ACC10(x,y) (x+xdim10*(y))
#define OPS_ACC11(x,y) (x+xdim11*(y))
#define OPS_ACC12(x,y) (x+xdim12*(y))
#define OPS_ACC13(x,y) (x+xdim13*(y))
#define OPS_ACC14(x,y) (x+xdim14*(y))
#define OPS_ACC15(x,y) (x+xdim15*(y))
#define OPS_ACC16(x,y) (x+xdim16*(y))
#define OPS_ACC17(x,y) (x+xdim17*(y))
#else

#define OPS_ACC0(x,y) (ops_stencil_check_2d(0, x, y, xdim0, -1))
#define OPS_ACC1(x,y) (ops_stencil_check_2d(1, x, y, xdim1, -1))
#define OPS_ACC2(x,y) (ops_stencil_check_2d(2, x, y, xdim2, -1))
#define OPS_ACC3(x,y) (ops_stencil_check_2d(3, x, y, xdim3, -1))
#define OPS_ACC4(x,y) (ops_stencil_check_2d(4, x, y, xdim4, -1))
#define OPS_ACC5(x,y) (ops_stencil_check_2d(5, x, y, xdim5, -1))
#define OPS_ACC6(x,y) (ops_stencil_check_2d(6, x, y, xdim6, -1))
#define OPS_ACC7(x,y) (ops_stencil_check_2d(7, x, y, xdim7, -1))
#define OPS_ACC8(x,y) (ops_stencil_check_2d(8, x, y, xdim8, -1))
#define OPS_ACC9(x,y) (ops_stencil_check_2d(9, x, y, xdim9, -1))
#define OPS_ACC10(x,y) (ops_stencil_check_2d(10, x, y, xdim10, -1))
#define OPS_ACC11(x,y) (ops_stencil_check_2d(11, x, y, xdim11, -1))
#define OPS_ACC12(x,y) (ops_stencil_check_2d(12, x, y, xdim12, -1))
#define OPS_ACC13(x,y) (ops_stencil_check_2d(13, x, y, xdim13, -1))
#define OPS_ACC14(x,y) (ops_stencil_check_2d(14, x, y, xdim14, -1))
#define OPS_ACC15(x,y) (ops_stencil_check_2d(15, x, y, xdim15, -1))
#define OPS_ACC16(x,y) (ops_stencil_check_2d(16, x, y, xdim16, -1))
#define OPS_ACC17(x,y) (ops_stencil_check_2d(17, x, y, xdim17, -1))
#endif
#endif

extern int xdim0;
extern int xdim1;
extern int xdim2;
extern int xdim3;
extern int xdim4;
extern int xdim5;
extern int xdim6;
extern int xdim7;
extern int xdim8;
extern int xdim9;
extern int xdim10;
extern int xdim11;
extern int xdim12;
extern int xdim13;
extern int xdim14;
extern int xdim15;
extern int xdim16;
extern int xdim17;



inline int mult2(int* s, int r)
{
  int result = 1;
  if(r > 0) {
    for(int i = 0; i<r;i++) result *= s[i];
  }
  return result;
}

inline int add2(int* co, int* s, int r)
{
  int result = co[0];
  for(int i = 1; i<=r;i++) result += co[i]*mult2(s,i);
  return result;
}


inline int off2(int ndim, int r, int* ps, int* pe, int* size, int* std)
{

  int i = 0;
  int c1[ndim];
  int c2[ndim];

  for(i=0; i<ndim; i++) c1[i] = ps[i];
  c1[r] = ps[r] + 1*std[r];

  for(i = 0; i<r; i++) std[i]!=0 ? c2[i] = pe[i]:c2[i] = ps[i]+1;
  for(i=r; i<ndim; i++) c2[i] = ps[i];

  int off =  add2(c1, size, r) - add2(c2, size, r);

  return off;
}

inline int address2(int ndim, int dat_size, int* ps, int* size, int* std, int* off)
{
  int base = 0;
  for(int i=0; i<ndim; i++) {
    base = base + dat_size * mult2(size, i) * (ps[i] * std[i] - off[i]);
  }
  return base;
}

#endif
