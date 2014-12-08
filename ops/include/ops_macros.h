#ifndef OPS_MACROS_H
#define OPS_MACROS_H
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
* THIS SOFTWARE IS PROVIDED BY Mike Giles and AUTHORS ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles and AUTHORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief header file declaring the functions for the ops mpi backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS macros
  */


/**--------------1-D ops_dats macros (one element per grid point)------------**/
#ifndef OPS_ACC_MACROS
  #define OPS_ACC_MACROS
  #ifdef OPS_3D //macros for 3D application
    #ifndef OPS_DEBUG //no debug checks
      #define OPS_ACC0(x,y,z) (x+xdim0*(y)+ydim0*xdim0*(z))
      #define OPS_ACC1(x,y,z) (x+xdim1*(y)+ydim1*xdim1*(z))
      #define OPS_ACC2(x,y,z) (x+xdim2*(y)+ydim2*xdim2*(z))
      #define OPS_ACC3(x,y,z) (x+xdim3*(y)+ydim3*xdim3*(z))
      #define OPS_ACC4(x,y,z) (x+xdim4*(y)+ydim4*xdim4*(z))
      #define OPS_ACC5(x,y,z) (x+xdim5*(y)+ydim5*xdim5*(z))
      #define OPS_ACC6(x,y,z) (x+xdim6*(y)+ydim6*xdim6*(z))
      #define OPS_ACC7(x,y,z) (x+xdim7*(y)+ydim7*xdim7*(z))
      #define OPS_ACC8(x,y,z) (x+xdim8*(y)+ydim8*xdim8*(z))
      #define OPS_ACC9(x,y,z) (x+xdim9*(y)+ydim9*xdim9*(z))
      #define OPS_ACC10(x,y,z) (x+xdim10*(y)+ydim10*xdim10*(z))
      #define OPS_ACC11(x,y,z) (x+xdim11*(y)+ydim11*xdim11*(z))
      #define OPS_ACC12(x,y,z) (x+xdim12*(y)+ydim12*xdim12*(z))
      #define OPS_ACC13(x,y,z) (x+xdim13*(y)+ydim13*xdim13*(z))
      #define OPS_ACC14(x,y,z) (x+xdim14*(y)+ydim14*xdim14*(z))
      #define OPS_ACC15(x,y,z) (x+xdim15*(y)+ydim15*xdim15*(z))
      #define OPS_ACC16(x,y,z) (x+xdim16*(y)+ydim16*xdim16*(z))
      #define OPS_ACC17(x,y,z) (x+xdim17*(y)+ydim17*xdim17*(z))
    #else //do debug checks
      #define OPS_ACC0(x,y,z) (ops_stencil_check_3d(0, x, y, z, xdim0, ydim0))
      #define OPS_ACC1(x,y,z) (ops_stencil_check_3d(1, x, y, z, xdim1, ydim1))
      #define OPS_ACC2(x,y,z) (ops_stencil_check_3d(2, x, y, z, xdim2, ydim2))
      #define OPS_ACC3(x,y,z) (ops_stencil_check_3d(3, x, y, z, xdim3, ydim3))
      #define OPS_ACC4(x,y,z) (ops_stencil_check_3d(4, x, y, z, xdim4, ydim4))
      #define OPS_ACC5(x,y,z) (ops_stencil_check_3d(5, x, y, z, xdim5, ydim5))
      #define OPS_ACC6(x,y,z) (ops_stencil_check_3d(6, x, y, z, xdim6, ydim6))
      #define OPS_ACC7(x,y,z) (ops_stencil_check_3d(7, x, y, z, xdim7, ydim7))
      #define OPS_ACC8(x,y,z) (ops_stencil_check_3d(8, x, y, z, xdim8, ydim8))
      #define OPS_ACC9(x,y,z) (ops_stencil_check_3d(9, x, y, z, xdim9, ydim9))
      #define OPS_ACC10(x,y,z) (ops_stencil_check_3d(10, x, y, z, xdim10, ydim10))
      #define OPS_ACC11(x,y,z) (ops_stencil_check_3d(11, x, y, z, xdim11, ydim11))
      #define OPS_ACC12(x,y,z) (ops_stencil_check_3d(12, x, y, z, xdim12, ydim12))
      #define OPS_ACC13(x,y,z) (ops_stencil_check_3d(13, x, y, z, xdim13, ydim13))
      #define OPS_ACC14(x,y,z) (ops_stencil_check_3d(14, x, y, z, xdim14, ydim14))
      #define OPS_ACC15(x,y,z) (ops_stencil_check_3d(15, x, y, z, xdim15, ydim15))
      #define OPS_ACC16(x,y,z) (ops_stencil_check_3d(16, x, y, z, xdim16, ydim16))
      #define OPS_ACC17(x,y,z) (ops_stencil_check_3d(17, x, y, z, xdim17, ydim17))
    #endif //end debug checks
  #elif defined OPS_2D //macros for 2D application
    #ifndef OPS_DEBUG //no debug checks
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
    #else //do debug checks
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
    #endif //end debug checks
  #else //macros for 1D application
    #ifndef OPS_DEBUG //no debug checks
      #define OPS_ACC0(x) (x)
      #define OPS_ACC1(x) (x)
      #define OPS_ACC2(x) (x)
      #define OPS_ACC3(x) (x)
      #define OPS_ACC4(x) (x)
      #define OPS_ACC5(x) (x)
      #define OPS_ACC6(x) (x)
      #define OPS_ACC7(x) (x)
      #define OPS_ACC8(x) (x)
      #define OPS_ACC9(x) (x)
      #define OPS_ACC10(x) (x)
      #define OPS_ACC11(x) (x)
      #define OPS_ACC12(x) (x)
      #define OPS_ACC13(x) (x)
      #define OPS_ACC14(x) (x)
      #define OPS_ACC15(x) (x)
      #define OPS_ACC16(x) (x)
      #define OPS_ACC17(x) (x)
    #else//do debug checks
      #define OPS_ACC0(x) (ops_stencil_check_1d(0, x, xdim0))
      #define OPS_ACC1(x) (ops_stencil_check_1d(1, x, xdim1))
      #define OPS_ACC2(x) (ops_stencil_check_1d(2, x, xdim2))
      #define OPS_ACC3(x) (ops_stencil_check_1d(3, x, xdim3))
      #define OPS_ACC4(x) (ops_stencil_check_1d(4, x, xdim4))
      #define OPS_ACC5(x) (ops_stencil_check_1d(5, x, xdim5))
      #define OPS_ACC6(x) (ops_stencil_check_1d(6, x, xdim6))
      #define OPS_ACC7(x) (ops_stencil_check_1d(7, x, xdim7))
      #define OPS_ACC8(x) (ops_stencil_check_1d(8, x, xdim8))
      #define OPS_ACC9(x) (ops_stencil_check_1d(9, x, xdim9))
      #define OPS_ACC10(x) (ops_stencil_check_1d(10, x, xdim10))
      #define OPS_ACC11(x) (ops_stencil_check_1d(11, x, xdim11))
      #define OPS_ACC12(x) (ops_stencil_check_1d(12, x, xdim12))
      #define OPS_ACC13(x) (ops_stencil_check_1d(13, x, xdim13))
      #define OPS_ACC14(x) (ops_stencil_check_1d(14, x, xdim14))
      #define OPS_ACC15(x) (ops_stencil_check_1d(15, x, xdim15))
      #define OPS_ACC16(x) (ops_stencil_check_1d(16, x, xdim16))
      #define OPS_ACC17(x) (ops_stencil_check_1d(17, x, xdim17))
    #endif//end debug checks
  #endif // end macros for 1D application
#endif //end OPS_ACC_MACROS



/**---------Multi-D ops_dats macros (multiple elements per grid point)-------**/
#ifndef OPS_ACC_MD_MACROS
  #define OPS_ACC_MACROS
  #ifdef OPS_3D //macros for 3D application
    #ifndef OPS_DEBUG //no debug checks
      #define OPS_ACC_MD0(d,x,y,z) ((x)*multi_d0+(d)+(xdim0*(y)*multi_d0)+(xdim0*ydim0*(z)*multi_d0))
    #else
      ///TODO #define OPS_ACC_MD0(x,y,z,d) (ops_stencil_check_3d(0, x, y, z, xdim0, ydim0))
    #endif
  #elif defined OPS_2D //macros for 2D application
    #ifndef OPS_DEBUG
      #define OPS_ACC_MD0(d,x,y) ((x)*multi_d0+(d)+(xdim0*(y)*multi_d0))
      #define OPS_ACC_MD1(d,x,y) ((x)*multi_d1+(d)+(xdim1*(y)*multi_d1))
      #define OPS_ACC_MD2(d,x,y) ((x)*multi_d2+(d)+(xdim2*(y)*multi_d2))
      #define OPS_ACC_MD3(d,x,y) ((x)*multi_d3+(d)+(xdim3*(y)*multi_d3))
      #define OPS_ACC_MD4(d,x,y) ((x)*multi_d4+(d)+(xdim4*(y)*multi_d4))
      #define OPS_ACC_MD5(d,x,y) ((x)*multi_d5+(d)+(xdim5*(y)*multi_d5))
      #define OPS_ACC_MD6(d,x,y) ((x)*multi_d6+(d)+(xdim6*(y)*multi_d6))
      #define OPS_ACC_MD7(d,x,y) ((x)*multi_d7+(d)+(xdim7*(y)*multi_d7))
      #define OPS_ACC_MD8(d,x,y) ((x)*multi_d8+(d)+(xdim8*(y)*multi_d8))
      #define OPS_ACC_MD9(d,x,y) ((x)*multi_d9+(d)+(xdim9*(y)*multi_d9))
      #define OPS_ACC_MD10(d,x,y) ((x)*multi_d10+(d)+(xdim10*(y)*multi_d10))
      #define OPS_ACC_MD11(d,x,y) ((x)*multi_d11+(d)+(xdim11*(y)*multi_d11))
      #define OPS_ACC_MD12(d,x,y) ((x)*multi_d12+(d)+(xdim12*(y)*multi_d12))
      #define OPS_ACC_MD13(d,x,y) ((x)*multi_d13+(d)+(xdim13*(y)*multi_d13))
      #define OPS_ACC_MD14(d,x,y) ((x)*multi_d14+(d)+(xdim14*(y)*multi_d14))
      #define OPS_ACC_MD15(d,x,y) ((x)*multi_d15+(d)+(xdim15*(y)*multi_d15))
      #define OPS_ACC_MD16(d,x,y) ((x)*multi_d16+(d)+(xdim16*(y)*multi_d16))
      #define OPS_ACC_MD17(d,x,y) ((x)*multi_d17+(d)+(xdim17*(y)*multi_d17))
    #else
      ///TODO #define OPS_ACC0(x,y) (ops_stencil_check_2d(0, x, -1, -1))
    #endif
  #else //macros for 1D application
    #ifndef OPS_DEBUG //no debug checks
      #define OPS_ACC_MD0(d,x) ((x)*multi_d0+d)
      #define OPS_ACC_MD1(d,x) ((x)*multi_d1+d)
      #define OPS_ACC_MD2(d,x) ((x)*multi_d2+d)
      #define OPS_ACC_MD3(d,x) ((x)*multi_d3+d)
      #define OPS_ACC_MD4(d,x) ((x)*multi_d4+d)
      #define OPS_ACC_MD5(d,x) ((x)*multi_d5+d)
      #define OPS_ACC_MD6(d,x) ((x)*multi_d6+d)
      #define OPS_ACC_MD7(d,x) ((x)*multi_d7+d)
      #define OPS_ACC_MD8(d,x) ((x)*multi_d8+d)
      #define OPS_ACC_MD9(d,x) ((x)*multi_d9+d)
      #define OPS_ACC_MD10(d,x) ((x)*multi_d10+d)
      #define OPS_ACC_MD11(d,x) ((x)*multi_d11+d)
      #define OPS_ACC_MD12(d,x) ((x)*multi_d12+d)
      #define OPS_ACC_MD13(d,x) ((x)*multi_d13+d)
      #define OPS_ACC_MD14(d,x) ((x)*multi_d14+d)
      #define OPS_ACC_MD15(d,x) ((x)*multi_d15+d)
      #define OPS_ACC_MD16(d,x) ((x)*multi_d16+d)
      #define OPS_ACC_MD17(d,x) ((x)*multi_d17+d)
    #else
      #define OPS_ACC_MD0(d,x) (ops_stencil_check_1d(0, x, xdim0))
      #define OPS_ACC_MD1(d,x) (ops_stencil_check_1d(1, x, xdim1))
      #define OPS_ACC_MD2(d,x) (ops_stencil_check_1d(2, x, xdim2))
      #define OPS_ACC_MD3(d,x) (ops_stencil_check_1d(3, x, xdim3))
      #define OPS_ACC_MD4(d,x) (ops_stencil_check_1d(4, x, xdim4))
      #define OPS_ACC_MD5(d,x) (ops_stencil_check_1d(5, x, xdim5))
      #define OPS_ACC_MD6(d,x) (ops_stencil_check_1d(6, x, xdim6))
      #define OPS_ACC_MD7(d,x) (ops_stencil_check_1d(7, x, xdim7))
      #define OPS_ACC_MD8(d,x) (ops_stencil_check_1d(8, x, xdim8))
      #define OPS_ACC_MD9(d,x) (ops_stencil_check_1d(9, x, xdim9))
      #define OPS_ACC_MD10(d,x) (ops_stencil_check_1d(10, x, xdim10))
      #define OPS_ACC_MD11(d,x) (ops_stencil_check_1d(11, x, xdim11))
      #define OPS_ACC_MD12(d,x) (ops_stencil_check_1d(12, x, xdim12))
      #define OPS_ACC_MD13(d,x) (ops_stencil_check_1d(13, x, xdim13))
      #define OPS_ACC_MD14(d,x) (ops_stencil_check_1d(14, x, xdim14))
      #define OPS_ACC_MD15(d,x) (ops_stencil_check_1d(15, x, xdim15))
      #define OPS_ACC_MD16(d,x) (ops_stencil_check_1d(16, x, xdim16))
      #define OPS_ACC_MD17(d,x) (ops_stencil_check_1d(17, x, xdim17))
    #endif
  #endif
#endif

/**--------------------------Set SIMD Vector lenght--------------------------**/
#ifndef SIMD_VEC
#define SIMD_VEC 4
#endif



#ifndef ROUND_DOWN
#define ROUND_DOWN(N,step) (((N)/(step))*(step))
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

#ifdef OPS_3D
extern int ydim0;
extern int ydim1;
extern int ydim2;
extern int ydim3;
extern int ydim4;
extern int ydim5;
extern int ydim6;
extern int ydim7;
extern int ydim8;
extern int ydim9;
extern int ydim10;
extern int ydim11;
extern int ydim12;
extern int ydim13;
extern int ydim14;
extern int ydim15;
extern int ydim16;
extern int ydim17;
#endif

extern int multi_d0;
extern int multi_d1;
extern int multi_d2;
extern int multi_d3;
extern int multi_d4;
extern int multi_d5;
extern int multi_d6;
extern int multi_d7;
extern int multi_d8;
extern int multi_d9;
extern int multi_d10;
extern int multi_d11;
extern int multi_d12;
extern int multi_d13;
extern int multi_d14;
extern int multi_d15;
extern int multi_d16;
extern int multi_d17;

#endif //OPS_MACROS_H
