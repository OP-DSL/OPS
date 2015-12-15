
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

/** @brief header file declaring the functions for the ops sequential backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS API calls for the sequential backend
  */

#include "ops_lib_cpp.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#ifndef OPS_ACC_MACROS
#ifdef OPS_3D
#ifndef OPS_DEBUG
#define OPS_ACC0(x,y,z) (x+xdim0*(y)+xdim0*ydim0*(z))
#define OPS_ACC1(x,y,z) (x+xdim1*(y)+xdim1*ydim1*(z))
#define OPS_ACC2(x,y,z) (x+xdim2*(y)+xdim2*ydim2*(z))
#define OPS_ACC3(x,y,z) (x+xdim3*(y)+xdim3*ydim3*(z))
#define OPS_ACC4(x,y,z) (x+xdim4*(y)+xdim4*ydim4*(z))
#define OPS_ACC5(x,y,z) (x+xdim5*(y)+xdim5*ydim5*(z))
#define OPS_ACC6(x,y,z) (x+xdim6*(y)+xdim6*ydim6*(z))
#define OPS_ACC7(x,y,z) (x+xdim7*(y)+xdim7*ydim7*(z))
#define OPS_ACC8(x,y,z) (x+xdim8*(y)+xdim8*ydim8*(z))
#define OPS_ACC9(x,y,z) (x+xdim9*(y)+xdim9*ydim9*(z))
#define OPS_ACC10(x,y,z) (x+xdim10*(y)+xdim10*ydim10*(z))
#define OPS_ACC11(x,y,z) (x+xdim11*(y)+xdim11*ydim11*(z))
#define OPS_ACC12(x,y,z) (x+xdim12*(y)+xdim12*ydim12*(z))
#define OPS_ACC13(x,y,z) (x+xdim13*(y)+xdim13*ydim13*(z))
#define OPS_ACC14(x,y,z) (x+xdim14*(y)+xdim14*ydim14*(z))
#define OPS_ACC15(x,y,z) (x+xdim15*(y)+xdim15*ydim15*(z))
#define OPS_ACC16(x,y,z) (x+xdim16*(y)+xdim16*ydim16*(z))
#define OPS_ACC17(x,y,z) (x+xdim17*(y)+xdim17*ydim17*(z))
#define OPS_ACC18(x,y,z) (x+xdim18*(y)+xdim18*ydim18*(z))
#define OPS_ACC19(x,y,z) (x+xdim19*(y)+xdim19*ydim19*(z))
#define OPS_ACC20(x,y,z) (x+xdim20*(y)+xdim20*ydim20*(z))
#define OPS_ACC21(x,y,z) (x+xdim21*(y)+xdim21*ydim21*(z))
#define OPS_ACC22(x,y,z) (x+xdim22*(y)+xdim22*ydim22*(z))
#define OPS_ACC23(x,y,z) (x+xdim23*(y)+xdim23*ydim23*(z))
#define OPS_ACC24(x,y,z) (x+xdim24*(y)+xdim24*ydim24*(z))
#define OPS_ACC25(x,y,z) (x+xdim25*(y)+xdim25*ydim25*(z))
#define OPS_ACC26(x,y,z) (x+xdim26*(y)+xdim26*ydim26*(z))
#define OPS_ACC27(x,y,z) (x+xdim27*(y)+xdim27*ydim27*(z))
#define OPS_ACC28(x,y,z) (x+xdim28*(y)+xdim28*ydim28*(z))
#define OPS_ACC29(x,y,z) (x+xdim29*(y)+xdim29*ydim29*(z))
#define OPS_ACC30(x,y,z) (x+xdim30*(y)+xdim30*ydim30*(z))
#define OPS_ACC31(x,y,z) (x+xdim31*(y)+xdim31*ydim31*(z))
#define OPS_ACC32(x,y,z) (x+xdim32*(y)+xdim32*ydim32*(z))
#define OPS_ACC33(x,y,z) (x+xdim33*(y)+xdim33*ydim33*(z))
#define OPS_ACC34(x,y,z) (x+xdim34*(y)+xdim34*ydim34*(z))
#define OPS_ACC35(x,y,z) (x+xdim35*(y)+xdim35*ydim35*(z))
#define OPS_ACC36(x,y,z) (x+xdim36*(y)+xdim36*ydim36*(z))
#define OPS_ACC37(x,y,z) (x+xdim37*(y)+xdim37*ydim37*(z))
#define OPS_ACC38(x,y,z) (x+xdim38*(y)+xdim38*ydim38*(z))
#define OPS_ACC39(x,y,z) (x+xdim39*(y)+xdim39*ydim39*(z))
#define OPS_ACC40(x,y,z) (x+xdim40*(y)+xdim40*ydim40*(z))
#define OPS_ACC41(x,y,z) (x+xdim41*(y)+xdim41*ydim41*(z))
#define OPS_ACC42(x,y,z) (x+xdim42*(y)+xdim42*ydim42*(z))
#define OPS_ACC43(x,y,z) (x+xdim43*(y)+xdim43*ydim43*(z))
#define OPS_ACC44(x,y,z) (x+xdim44*(y)+xdim44*ydim44*(z))
#define OPS_ACC45(x,y,z) (x+xdim45*(y)+xdim45*ydim45*(z))
#define OPS_ACC46(x,y,z) (x+xdim46*(y)+xdim46*ydim46*(z))
#define OPS_ACC47(x,y,z) (x+xdim47*(y)+xdim47*ydim47*(z))
#define OPS_ACC48(x,y,z) (x+xdim48*(y)+xdim48*ydim48*(z))
#define OPS_ACC49(x,y,z) (x+xdim49*(y)+xdim49*ydim49*(z))
#define OPS_ACC50(x,y,z) (x+xdim50*(y)+xdim50*ydim50*(z))
#else

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
#define OPS_ACC18(x,y,z) (ops_stencil_check_3d(18, x, y, z, xdim18, ydim18))
#define OPS_ACC19(x,y,z) (ops_stencil_check_3d(19, x, y, z, xdim19, ydim19))
#define OPS_ACC20(x,y,z) (ops_stencil_check_3d(20, x, y, z, xdim20, ydim20))
#define OPS_ACC21(x,y,z) (ops_stencil_check_3d(21, x, y, z, xdim21, ydim21))
#define OPS_ACC22(x,y,z) (ops_stencil_check_3d(22, x, y, z, xdim22, ydim22))
#define OPS_ACC23(x,y,z) (ops_stencil_check_3d(23, x, y, z, xdim23, ydim23))
#define OPS_ACC24(x,y,z) (ops_stencil_check_3d(24, x, y, z, xdim24, ydim24))
#define OPS_ACC25(x,y,z) (ops_stencil_check_3d(25, x, y, z, xdim25, ydim25))
#define OPS_ACC26(x,y,z) (ops_stencil_check_3d(26, x, y, z, xdim26, ydim26))
#define OPS_ACC27(x,y,z) (ops_stencil_check_3d(27, x, y, z, xdim27, ydim27))
#define OPS_ACC28(x,y,z) (ops_stencil_check_3d(28, x, y, z, xdim28, ydim28))
#define OPS_ACC29(x,y,z) (ops_stencil_check_3d(29, x, y, z, xdim29, ydim29))
#define OPS_ACC30(x,y,z) (ops_stencil_check_3d(30, x, y, z, xdim30, ydim30))
#define OPS_ACC31(x,y,z) (ops_stencil_check_3d(31, x, y, z, xdim31, ydim31))
#define OPS_ACC32(x,y,z) (ops_stencil_check_3d(32, x, y, z, xdim32, ydim32))
#define OPS_ACC33(x,y,z) (ops_stencil_check_3d(33, x, y, z, xdim33, ydim33))
#define OPS_ACC34(x,y,z) (ops_stencil_check_3d(34, x, y, z, xdim34, ydim34))
#define OPS_ACC35(x,y,z) (ops_stencil_check_3d(35, x, y, z, xdim35, ydim35))
#define OPS_ACC36(x,y,z) (ops_stencil_check_3d(36, x, y, z, xdim36, ydim36))
#define OPS_ACC37(x,y,z) (ops_stencil_check_3d(37, x, y, z, xdim37, ydim37))
#define OPS_ACC38(x,y,z) (ops_stencil_check_3d(38, x, y, z, xdim38, ydim38))
#define OPS_ACC39(x,y,z) (ops_stencil_check_3d(39, x, y, z, xdim39, ydim39))
#define OPS_ACC40(x,y,z) (ops_stencil_check_3d(40, x, y, z, xdim40, ydim40))
#define OPS_ACC41(x,y,z) (ops_stencil_check_3d(41, x, y, z, xdim41, ydim41))
#define OPS_ACC42(x,y,z) (ops_stencil_check_3d(42, x, y, z, xdim42, ydim42))
#define OPS_ACC43(x,y,z) (ops_stencil_check_3d(43, x, y, z, xdim43, ydim43))
#define OPS_ACC44(x,y,z) (ops_stencil_check_3d(44, x, y, z, xdim44, ydim44))
#define OPS_ACC45(x,y,z) (ops_stencil_check_3d(45, x, y, z, xdim45, ydim45))
#define OPS_ACC46(x,y,z) (ops_stencil_check_3d(46, x, y, z, xdim46, ydim46))
#define OPS_ACC47(x,y,z) (ops_stencil_check_3d(47, x, y, z, xdim47, ydim47))
#define OPS_ACC48(x,y,z) (ops_stencil_check_3d(48, x, y, z, xdim48, ydim48))
#define OPS_ACC49(x,y,z) (ops_stencil_check_3d(49, x, y, z, xdim49, ydim49))
#define OPS_ACC50(x,y,z) (ops_stencil_check_3d(50, x, y, z, xdim50, ydim50))
#endif
#else
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
#define OPS_ACC18(x,y) (x+xdim18*(y))
#define OPS_ACC19(x,y) (x+xdim19*(y))
#define OPS_ACC20(x,y) (x+xdim20*(y))
#define OPS_ACC21(x,y) (x+xdim21*(y))
#define OPS_ACC22(x,y) (x+xdim22*(y))
#define OPS_ACC23(x,y) (x+xdim23*(y))
#define OPS_ACC24(x,y) (x+xdim24*(y))
#define OPS_ACC25(x,y) (x+xdim25*(y))
#define OPS_ACC26(x,y) (x+xdim26*(y))
#define OPS_ACC27(x,y) (x+xdim27*(y))
#define OPS_ACC28(x,y) (x+xdim28*(y))
#define OPS_ACC29(x,y) (x+xdim29*(y))
#define OPS_ACC30(x,y) (x+xdim30*(y))
#define OPS_ACC31(x,y) (x+xdim31*(y))
#define OPS_ACC32(x,y) (x+xdim32*(y))
#define OPS_ACC33(x,y) (x+xdim33*(y))
#define OPS_ACC34(x,y) (x+xdim34*(y))
#define OPS_ACC35(x,y) (x+xdim35*(y))
#define OPS_ACC36(x,y) (x+xdim36*(y))
#define OPS_ACC37(x,y) (x+xdim37*(y))
#define OPS_ACC38(x,y) (x+xdim38*(y))
#define OPS_ACC39(x,y) (x+xdim39*(y))
#define OPS_ACC40(x,y) (x+xdim40*(y))
#define OPS_ACC41(x,y) (x+xdim41*(y))
#define OPS_ACC42(x,y) (x+xdim42*(y))
#define OPS_ACC43(x,y) (x+xdim43*(y))
#define OPS_ACC44(x,y) (x+xdim44*(y))
#define OPS_ACC45(x,y) (x+xdim45*(y))
#define OPS_ACC46(x,y) (x+xdim46*(y))
#define OPS_ACC47(x,y) (x+xdim47*(y))
#define OPS_ACC48(x,y) (x+xdim48*(y))
#define OPS_ACC49(x,y) (x+xdim49*(y))
#define OPS_ACC50(x,y) (x+xdim50*(y))
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
#define OPS_ACC18(x,y) (ops_stencil_check_2d(18, x, y, xdim18, -1))
#define OPS_ACC19(x,y) (ops_stencil_check_2d(19, x, y, xdim19, -1))
#define OPS_ACC20(x,y) (ops_stencil_check_2d(20, x, y, xdim20, -1))
#define OPS_ACC21(x,y) (ops_stencil_check_2d(21, x, y, xdim21, -1))
#define OPS_ACC22(x,y) (ops_stencil_check_2d(22, x, y, xdim22, -1))
#define OPS_ACC23(x,y) (ops_stencil_check_2d(23, x, y, xdim23, -1))
#define OPS_ACC24(x,y) (ops_stencil_check_2d(24, x, y, xdim24, -1))
#define OPS_ACC25(x,y) (ops_stencil_check_2d(25, x, y, xdim25, -1))
#define OPS_ACC26(x,y) (ops_stencil_check_2d(26, x, y, xdim26, -1))
#define OPS_ACC27(x,y) (ops_stencil_check_2d(27, x, y, xdim27, -1))
#define OPS_ACC28(x,y) (ops_stencil_check_2d(28, x, y, xdim28, -1))
#define OPS_ACC29(x,y) (ops_stencil_check_2d(29, x, y, xdim29, -1))
#define OPS_ACC30(x,y) (ops_stencil_check_2d(30, x, y, xdim30, -1))
#define OPS_ACC31(x,y) (ops_stencil_check_2d(31, x, y, xdim31, -1))
#define OPS_ACC32(x,y) (ops_stencil_check_2d(32, x, y, xdim32, -1))
#define OPS_ACC33(x,y) (ops_stencil_check_2d(33, x, y, xdim33, -1))
#define OPS_ACC34(x,y) (ops_stencil_check_2d(34, x, y, xdim34, -1))
#define OPS_ACC35(x,y) (ops_stencil_check_2d(35, x, y, xdim35, -1))
#define OPS_ACC36(x,y) (ops_stencil_check_2d(36, x, y, xdim36, -1))
#define OPS_ACC37(x,y) (ops_stencil_check_2d(37, x, y, xdim37, -1))
#define OPS_ACC38(x,y) (ops_stencil_check_2d(38, x, y, xdim38, -1))
#define OPS_ACC39(x,y) (ops_stencil_check_2d(39, x, y, xdim39, -1))
#define OPS_ACC40(x,y) (ops_stencil_check_2d(40, x, y, xdim40, -1))
#define OPS_ACC41(x,y) (ops_stencil_check_2d(41, x, y, xdim41, -1))
#define OPS_ACC42(x,y) (ops_stencil_check_2d(42, x, y, xdim42, -1))
#define OPS_ACC43(x,y) (ops_stencil_check_2d(43, x, y, xdim43, -1))
#define OPS_ACC44(x,y) (ops_stencil_check_2d(44, x, y, xdim44, -1))
#define OPS_ACC45(x,y) (ops_stencil_check_2d(45, x, y, xdim45, -1))
#define OPS_ACC46(x,y) (ops_stencil_check_2d(46, x, y, xdim46, -1))
#define OPS_ACC47(x,y) (ops_stencil_check_2d(47, x, y, xdim47, -1))
#define OPS_ACC48(x,y) (ops_stencil_check_2d(48, x, y, xdim48, -1))
#define OPS_ACC49(x,y) (ops_stencil_check_2d(49, x, y, xdim49, -1))
#define OPS_ACC50(x,y) (ops_stencil_check_2d(50, x, y, xdim50, -1))
#endif
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
extern int xdim18;
extern int xdim19;
extern int xdim20;
extern int xdim21;
extern int xdim22;
extern int xdim23;
extern int xdim24;
extern int xdim25;
extern int xdim26;
extern int xdim27;
extern int xdim28;
extern int xdim29;
extern int xdim30;
extern int xdim31;
extern int xdim32;
extern int xdim33;
extern int xdim34;
extern int xdim35;
extern int xdim36;
extern int xdim37;
extern int xdim38;
extern int xdim39;
extern int xdim40;
extern int xdim41;
extern int xdim42;
extern int xdim43;
extern int xdim44;
extern int xdim45;
extern int xdim46;
extern int xdim47;
extern int xdim48;
extern int xdim49;
extern int xdim50;
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
extern int ydim18;
extern int ydim19;
extern int ydim20;
extern int ydim21;
extern int ydim22;
extern int ydim23;
extern int ydim24;
extern int ydim25;
extern int ydim26;
extern int ydim27;
extern int ydim28;
extern int ydim29;
extern int ydim30;
extern int ydim31;
extern int ydim32;
extern int ydim33;
extern int ydim34;
extern int ydim35;
extern int ydim36;
extern int ydim37;
extern int ydim38;
extern int ydim39;
extern int ydim40;
extern int ydim41;
extern int ydim42;
extern int ydim43;
extern int ydim44;
extern int ydim45;
extern int ydim46;
extern int ydim47;
extern int ydim48;
extern int ydim49;
extern int ydim50;
#endif


static int arg_idx[OPS_MAX_DIM];

inline int mult(int* size, int dim)
{
  int result = 1;
  if(dim > 0) {
    for(int i = 0; i<dim;i++) result *= size[i];
  }
  return result;
}

inline int add(int* coords, int* size, int dim)
{
  int result = coords[0];
  for(int i = 1; i<=dim;i++) result += coords[i]*mult(size,i);
  return result;
}


inline int off(int ndim, int dim , int* start, int* end, int* size, int* stride)
{

  int i = 0;
  int c1[3];
  int c2[3];

  for(i=0; i<=dim; i++) c1[i] = start[i]+1;
  for(i=dim+1; i<ndim; i++) c1[i] = start[i];

  for(i = 0; i<dim; i++) c2[i] = end[i];
  for(i=dim; i<ndim; i++) c2[i] = start[i];

  for (i = 0; i < ndim; i++) {
    c1[i] *= stride[i];
    c2[i] *= stride[i];
  }
  int off =  add(c1, size, dim) - add(c2, size, dim);

  return off;
}

inline int address(int ndim, int dat_size, int* start, int* size, int* stride, int* base_off, int *d_m)
{
  int base = 0;
  for(int i=0; i<ndim; i++) {
    base = base + dat_size * mult(size, i) * (start[i] * stride[i] - base_off[i] - d_m[i]);
  }
  return base;
}

inline void stencil_depth(ops_stencil sten, int* d_pos, int* d_neg)
{
  for(int dim = 0;dim<sten->dims;dim++){
    d_pos[dim] = 0; d_neg[dim] = 0;
  }
  for(int p=0;p<sten->points; p++) {
    for(int dim = 0;dim<sten->dims;dim++){
    d_pos[dim] = MAX(d_pos[dim],sten->stencil[sten->dims*p + dim]);
    d_neg[dim] = MIN(d_neg[dim],sten->stencil[sten->dims*p + dim]);
    }
  }
}



//
//ops_par_loop routine for 1 arguments
//
template <class T0>
void ops_par_loop(void (*kernel)(T0*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0) {

  char *p_a[1];
  int  offs[1][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[1] = { arg0};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,1,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<1;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 1);
  ops_halo_exchanges(args,1,range);
  ops_H_D_exchanges_host(args, 1);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<1; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  ops_set_dirtybit_host(args, 1);
}


//
//ops_par_loop routine for 2 arguments
//
template <class T0,class T1>
void ops_par_loop(void (*kernel)(T0*, T1*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1) {

  char *p_a[2];
  int  offs[2][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[2] = { arg0, arg1};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,2,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<2;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 2; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 2);
  ops_halo_exchanges(args,2,range);
  ops_H_D_exchanges_host(args, 2);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<2; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  ops_set_dirtybit_host(args, 2);
}


//
//ops_par_loop routine for 3 arguments
//
template <class T0,class T1,class T2>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char *p_a[3];
  int  offs[3][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[3] = { arg0, arg1, arg2};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,3,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<3;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 3);
  ops_halo_exchanges(args,3,range);
  ops_H_D_exchanges_host(args, 3);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<3; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  ops_set_dirtybit_host(args, 3);
}


//
//ops_par_loop routine for 4 arguments
//
template <class T0,class T1,class T2,class T3>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {

  char *p_a[4];
  int  offs[4][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[4] = { arg0, arg1, arg2, arg3};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,4,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<4;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 4; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 4);
  ops_halo_exchanges(args,4,range);
  ops_H_D_exchanges_host(args, 4);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<4; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  ops_set_dirtybit_host(args, 4);
}


//
//ops_par_loop routine for 5 arguments
//
template <class T0,class T1,class T2,class T3,
class T4>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4) {

  char *p_a[5];
  int  offs[5][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[5] = { arg0, arg1, arg2, arg3,
                     arg4};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,5,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<5;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 5);
  ops_halo_exchanges(args,5,range);
  ops_H_D_exchanges_host(args, 5);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<5; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  ops_set_dirtybit_host(args, 5);
}


//
//ops_par_loop routine for 6 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5) {

  char *p_a[6];
  int  offs[6][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[6] = { arg0, arg1, arg2, arg3,
                     arg4, arg5};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,6,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<6;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 6; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 6);
  ops_halo_exchanges(args,6,range);
  ops_H_D_exchanges_host(args, 6);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<6; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  ops_set_dirtybit_host(args, 6);
}


//
//ops_par_loop routine for 7 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6) {

  char *p_a[7];
  int  offs[7][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[7] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,7,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<7;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 7; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 7);
  ops_halo_exchanges(args,7,range);
  ops_H_D_exchanges_host(args, 7);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<7; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  ops_set_dirtybit_host(args, 7);
}


//
//ops_par_loop routine for 8 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {

  char *p_a[8];
  int  offs[8][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[8] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,8,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<8;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 8);
  ops_halo_exchanges(args,8,range);
  ops_H_D_exchanges_host(args, 8);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<8; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  ops_set_dirtybit_host(args, 8);
}


//
//ops_par_loop routine for 9 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8) {

  char *p_a[9];
  int  offs[9][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[9] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,9,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<9;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 9; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 9);
  ops_halo_exchanges(args,9,range);
  ops_H_D_exchanges_host(args, 9);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<9; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  ops_set_dirtybit_host(args, 9);
}


//
//ops_par_loop routine for 10 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9) {

  char *p_a[10];
  int  offs[10][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[10] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,10,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<10;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 10; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 10);
  ops_halo_exchanges(args,10,range);
  ops_H_D_exchanges_host(args, 10);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<10; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  ops_set_dirtybit_host(args, 10);
}


//
//ops_par_loop routine for 11 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10) {

  char *p_a[11];
  int  offs[11][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[11] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,11,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<11;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 11; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 11);
  ops_halo_exchanges(args,11,range);
  ops_H_D_exchanges_host(args, 11);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<11; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  ops_set_dirtybit_host(args, 11);
}


//
//ops_par_loop routine for 12 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11) {

  char *p_a[12];
  int  offs[12][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[12] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,12,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<12;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 12; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 12);
  ops_halo_exchanges(args,12,range);
  ops_H_D_exchanges_host(args, 12);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<12; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  ops_set_dirtybit_host(args, 12);
}


//
//ops_par_loop routine for 13 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12) {

  char *p_a[13];
  int  offs[13][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[13] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,13,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<13;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 13; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 13);
  ops_halo_exchanges(args,13,range);
  ops_H_D_exchanges_host(args, 13);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<13; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  ops_set_dirtybit_host(args, 13);
}


//
//ops_par_loop routine for 14 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13) {

  char *p_a[14];
  int  offs[14][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[14] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,14,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<14;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 14; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 14);
  ops_halo_exchanges(args,14,range);
  ops_H_D_exchanges_host(args, 14);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<14; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  ops_set_dirtybit_host(args, 14);
}


//
//ops_par_loop routine for 15 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14) {

  char *p_a[15];
  int  offs[15][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[15] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,15,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<15;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 15; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 15);
  ops_halo_exchanges(args,15,range);
  ops_H_D_exchanges_host(args, 15);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<15; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  ops_set_dirtybit_host(args, 15);
}


//
//ops_par_loop routine for 16 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15) {

  char *p_a[16];
  int  offs[16][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[16] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,16,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<16;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 16; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 16);
  ops_halo_exchanges(args,16,range);
  ops_H_D_exchanges_host(args, 16);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<16; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  ops_set_dirtybit_host(args, 16);
}


//
//ops_par_loop routine for 17 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16) {

  char *p_a[17];
  int  offs[17][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[17] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,17,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<17;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 17; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 17);
  ops_halo_exchanges(args,17,range);
  ops_H_D_exchanges_host(args, 17);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<17; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  ops_set_dirtybit_host(args, 17);
}


//
//ops_par_loop routine for 18 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17) {

  char *p_a[18];
  int  offs[18][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[18] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,18,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<18;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 18; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 18);
  ops_halo_exchanges(args,18,range);
  ops_H_D_exchanges_host(args, 18);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<18; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  ops_set_dirtybit_host(args, 18);
}


//
//ops_par_loop routine for 19 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18) {

  char *p_a[19];
  int  offs[19][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[19] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,19,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<19;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 19; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 19);
  ops_halo_exchanges(args,19,range);
  ops_H_D_exchanges_host(args, 19);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<19; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  ops_set_dirtybit_host(args, 19);
}


//
//ops_par_loop routine for 20 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19) {

  char *p_a[20];
  int  offs[20][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[20] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,20,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<20;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 20; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 20);
  ops_halo_exchanges(args,20,range);
  ops_H_D_exchanges_host(args, 20);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<20; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  ops_set_dirtybit_host(args, 20);
}


//
//ops_par_loop routine for 21 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20) {

  char *p_a[21];
  int  offs[21][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[21] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,21,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<21;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 21; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 21);
  ops_halo_exchanges(args,21,range);
  ops_H_D_exchanges_host(args, 21);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<21; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  ops_set_dirtybit_host(args, 21);
}


//
//ops_par_loop routine for 22 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21) {

  char *p_a[22];
  int  offs[22][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[22] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,22,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<22;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 22; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 22);
  ops_halo_exchanges(args,22,range);
  ops_H_D_exchanges_host(args, 22);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<22; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  ops_set_dirtybit_host(args, 22);
}


//
//ops_par_loop routine for 23 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22) {

  char *p_a[23];
  int  offs[23][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[23] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,23,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<23;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 23; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 23);
  ops_halo_exchanges(args,23,range);
  ops_H_D_exchanges_host(args, 23);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<23; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  ops_set_dirtybit_host(args, 23);
}


//
//ops_par_loop routine for 24 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23) {

  char *p_a[24];
  int  offs[24][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[24] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,24,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<24;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 24; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 24);
  ops_halo_exchanges(args,24,range);
  ops_H_D_exchanges_host(args, 24);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<24; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  ops_set_dirtybit_host(args, 24);
}


//
//ops_par_loop routine for 25 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24) {

  char *p_a[25];
  int  offs[25][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[25] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,25,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<25;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 25; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 25);
  ops_halo_exchanges(args,25,range);
  ops_H_D_exchanges_host(args, 25);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<25; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  ops_set_dirtybit_host(args, 25);
}


//
//ops_par_loop routine for 26 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25) {

  char *p_a[26];
  int  offs[26][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[26] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,26,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<26;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 26; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 26);
  ops_halo_exchanges(args,26,range);
  ops_H_D_exchanges_host(args, 26);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<26; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  ops_set_dirtybit_host(args, 26);
}


//
//ops_par_loop routine for 27 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26) {

  char *p_a[27];
  int  offs[27][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[27] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,27,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<27;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 27; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 27);
  ops_halo_exchanges(args,27,range);
  ops_H_D_exchanges_host(args, 27);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<27; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  ops_set_dirtybit_host(args, 27);
}


//
//ops_par_loop routine for 28 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27) {

  char *p_a[28];
  int  offs[28][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[28] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,28,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<28;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 28; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 28);
  ops_halo_exchanges(args,28,range);
  ops_H_D_exchanges_host(args, 28);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<28; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  ops_set_dirtybit_host(args, 28);
}


//
//ops_par_loop routine for 29 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28) {

  char *p_a[29];
  int  offs[29][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[29] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,29,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<29;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 29; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 29);
  ops_halo_exchanges(args,29,range);
  ops_H_D_exchanges_host(args, 29);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<29; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  ops_set_dirtybit_host(args, 29);
}


//
//ops_par_loop routine for 30 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29) {

  char *p_a[30];
  int  offs[30][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[30] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,30,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<30;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 30; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 30);
  ops_halo_exchanges(args,30,range);
  ops_H_D_exchanges_host(args, 30);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<30; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  ops_set_dirtybit_host(args, 30);
}


//
//ops_par_loop routine for 31 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30) {

  char *p_a[31];
  int  offs[31][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[31] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,31,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<31;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 31; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 31);
  ops_halo_exchanges(args,31,range);
  ops_H_D_exchanges_host(args, 31);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<31; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  ops_set_dirtybit_host(args, 31);
}


//
//ops_par_loop routine for 32 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31) {

  char *p_a[32];
  int  offs[32][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[32] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,32,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<32;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 32; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 32);
  ops_halo_exchanges(args,32,range);
  ops_H_D_exchanges_host(args, 32);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<32; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  ops_set_dirtybit_host(args, 32);
}


//
//ops_par_loop routine for 33 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32) {

  char *p_a[33];
  int  offs[33][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[33] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,33,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<33;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 33; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 33);
  ops_halo_exchanges(args,33,range);
  ops_H_D_exchanges_host(args, 33);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<33; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  ops_set_dirtybit_host(args, 33);
}


//
//ops_par_loop routine for 34 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33) {

  char *p_a[34];
  int  offs[34][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[34] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,34,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<34;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 34; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 34);
  ops_halo_exchanges(args,34,range);
  ops_H_D_exchanges_host(args, 34);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<34; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  ops_set_dirtybit_host(args, 34);
}


//
//ops_par_loop routine for 35 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34) {

  char *p_a[35];
  int  offs[35][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[35] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,35,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<35;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 35; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 35);
  ops_halo_exchanges(args,35,range);
  ops_H_D_exchanges_host(args, 35);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<35; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  ops_set_dirtybit_host(args, 35);
}


//
//ops_par_loop routine for 36 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35) {

  char *p_a[36];
  int  offs[36][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[36] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,36,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<36;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 36; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 36);
  ops_halo_exchanges(args,36,range);
  ops_H_D_exchanges_host(args, 36);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<36; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  ops_set_dirtybit_host(args, 36);
}


//
//ops_par_loop routine for 37 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36) {

  char *p_a[37];
  int  offs[37][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[37] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,37,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<37;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 37; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 37);
  ops_halo_exchanges(args,37,range);
  ops_H_D_exchanges_host(args, 37);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<37; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  ops_set_dirtybit_host(args, 37);
}


//
//ops_par_loop routine for 38 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37) {

  char *p_a[38];
  int  offs[38][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[38] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,38,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<38;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 38; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 38);
  ops_halo_exchanges(args,38,range);
  ops_H_D_exchanges_host(args, 38);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<38; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  ops_set_dirtybit_host(args, 38);
}


//
//ops_par_loop routine for 39 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38) {

  char *p_a[39];
  int  offs[39][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[39] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,39,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<39;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 39; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 39);
  ops_halo_exchanges(args,39,range);
  ops_H_D_exchanges_host(args, 39);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<39; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  ops_set_dirtybit_host(args, 39);
}


//
//ops_par_loop routine for 40 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39) {

  char *p_a[40];
  int  offs[40][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[40] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,40,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<40;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 40; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 40);
  ops_halo_exchanges(args,40,range);
  ops_H_D_exchanges_host(args, 40);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<40; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  ops_set_dirtybit_host(args, 40);
}


//
//ops_par_loop routine for 41 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40) {

  char *p_a[41];
  int  offs[41][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[41] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,41,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<41;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 41; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 41);
  ops_halo_exchanges(args,41,range);
  ops_H_D_exchanges_host(args, 41);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<41; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  ops_set_dirtybit_host(args, 41);
}


//
//ops_par_loop routine for 42 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41) {

  char *p_a[42];
  int  offs[42][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[42] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,42,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<42;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 42; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 42);
  ops_halo_exchanges(args,42,range);
  ops_H_D_exchanges_host(args, 42);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<42; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  ops_set_dirtybit_host(args, 42);
}


//
//ops_par_loop routine for 43 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42) {

  char *p_a[43];
  int  offs[43][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[43] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,43,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<43;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 43; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 43);
  ops_halo_exchanges(args,43,range);
  ops_H_D_exchanges_host(args, 43);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<43; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  ops_set_dirtybit_host(args, 43);
}


//
//ops_par_loop routine for 44 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43) {

  char *p_a[44];
  int  offs[44][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[44] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,44,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<44;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 44; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 44);
  ops_halo_exchanges(args,44,range);
  ops_H_D_exchanges_host(args, 44);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<44; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  ops_set_dirtybit_host(args, 44);
}


//
//ops_par_loop routine for 45 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44) {

  char *p_a[45];
  int  offs[45][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[45] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,45,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<45;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 45; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 45);
  ops_halo_exchanges(args,45,range);
  ops_H_D_exchanges_host(args, 45);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<45; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  ops_set_dirtybit_host(args, 45);
}


//
//ops_par_loop routine for 46 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44,class T45>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*, T45*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44, ops_arg arg45) {

  char *p_a[46];
  int  offs[46][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[46] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44, arg45};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,46,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<46;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 46; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }
  if (args[45].argtype == OPS_ARG_DAT) {
    xdim45 = args[45].dat->size[0]*args[45].dat->dim;
    multi_d45 = args[45].dat->dim;
    #ifdef OPS_3D
    ydim45 = args[45].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 46);
  ops_halo_exchanges(args,46,range);
  ops_H_D_exchanges_host(args, 46);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44], (T45 *)p_a[45] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<46; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ) ops_dump3(args[45].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[45],range);
  ops_set_dirtybit_host(args, 46);
}


//
//ops_par_loop routine for 47 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44,class T45,class T46>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*, T45*, T46*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44, ops_arg arg45, ops_arg arg46) {

  char *p_a[47];
  int  offs[47][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[47] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44, arg45, arg46};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,47,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<47;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 47; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }
  if (args[45].argtype == OPS_ARG_DAT) {
    xdim45 = args[45].dat->size[0]*args[45].dat->dim;
    multi_d45 = args[45].dat->dim;
    #ifdef OPS_3D
    ydim45 = args[45].dat->size[1];
    #endif
  }
  if (args[46].argtype == OPS_ARG_DAT) {
    xdim46 = args[46].dat->size[0]*args[46].dat->dim;
    multi_d46 = args[46].dat->dim;
    #ifdef OPS_3D
    ydim46 = args[46].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 47);
  ops_halo_exchanges(args,47,range);
  ops_H_D_exchanges_host(args, 47);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44], (T45 *)p_a[45], (T46 *)p_a[46] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<47; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ) ops_dump3(args[45].dat,name);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ) ops_dump3(args[46].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[45],range);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[46],range);
  ops_set_dirtybit_host(args, 47);
}


//
//ops_par_loop routine for 48 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44,class T45,class T46,class T47>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*, T45*, T46*, T47*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44, ops_arg arg45, ops_arg arg46, ops_arg arg47) {

  char *p_a[48];
  int  offs[48][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[48] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44, arg45, arg46, arg47};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,48,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<48;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 48; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }
  if (args[45].argtype == OPS_ARG_DAT) {
    xdim45 = args[45].dat->size[0]*args[45].dat->dim;
    multi_d45 = args[45].dat->dim;
    #ifdef OPS_3D
    ydim45 = args[45].dat->size[1];
    #endif
  }
  if (args[46].argtype == OPS_ARG_DAT) {
    xdim46 = args[46].dat->size[0]*args[46].dat->dim;
    multi_d46 = args[46].dat->dim;
    #ifdef OPS_3D
    ydim46 = args[46].dat->size[1];
    #endif
  }
  if (args[47].argtype == OPS_ARG_DAT) {
    xdim47 = args[47].dat->size[0]*args[47].dat->dim;
    multi_d47 = args[47].dat->dim;
    #ifdef OPS_3D
    ydim47 = args[47].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 48);
  ops_halo_exchanges(args,48,range);
  ops_H_D_exchanges_host(args, 48);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44], (T45 *)p_a[45], (T46 *)p_a[46], (T47 *)p_a[47] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<48; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ) ops_dump3(args[45].dat,name);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ) ops_dump3(args[46].dat,name);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ) ops_dump3(args[47].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[45],range);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[46],range);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[47],range);
  ops_set_dirtybit_host(args, 48);
}


//
//ops_par_loop routine for 49 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44,class T45,class T46,class T47,
class T48>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*, T45*, T46*, T47*,
                           T48*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44, ops_arg arg45, ops_arg arg46, ops_arg arg47,
     ops_arg arg48) {

  char *p_a[49];
  int  offs[49][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[49] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44, arg45, arg46, arg47,
                     arg48};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,49,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<49;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 49; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }
  if (args[45].argtype == OPS_ARG_DAT) {
    xdim45 = args[45].dat->size[0]*args[45].dat->dim;
    multi_d45 = args[45].dat->dim;
    #ifdef OPS_3D
    ydim45 = args[45].dat->size[1];
    #endif
  }
  if (args[46].argtype == OPS_ARG_DAT) {
    xdim46 = args[46].dat->size[0]*args[46].dat->dim;
    multi_d46 = args[46].dat->dim;
    #ifdef OPS_3D
    ydim46 = args[46].dat->size[1];
    #endif
  }
  if (args[47].argtype == OPS_ARG_DAT) {
    xdim47 = args[47].dat->size[0]*args[47].dat->dim;
    multi_d47 = args[47].dat->dim;
    #ifdef OPS_3D
    ydim47 = args[47].dat->size[1];
    #endif
  }
  if (args[48].argtype == OPS_ARG_DAT) {
    xdim48 = args[48].dat->size[0]*args[48].dat->dim;
    multi_d48 = args[48].dat->dim;
    #ifdef OPS_3D
    ydim48 = args[48].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 49);
  ops_halo_exchanges(args,49,range);
  ops_H_D_exchanges_host(args, 49);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44], (T45 *)p_a[45], (T46 *)p_a[46], (T47 *)p_a[47],
           (T48 *)p_a[48] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<49; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ) ops_dump3(args[45].dat,name);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ) ops_dump3(args[46].dat,name);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ) ops_dump3(args[47].dat,name);
  if (args[48].argtype == OPS_ARG_DAT && args[48].acc != OPS_READ) ops_dump3(args[48].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[45],range);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[46],range);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[47],range);
  if (args[48].argtype == OPS_ARG_DAT && args[48].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[48],range);
  ops_set_dirtybit_host(args, 49);
}


//
//ops_par_loop routine for 50 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44,class T45,class T46,class T47,
class T48,class T49>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*, T45*, T46*, T47*,
                           T48*, T49*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44, ops_arg arg45, ops_arg arg46, ops_arg arg47,
     ops_arg arg48, ops_arg arg49) {

  char *p_a[50];
  int  offs[50][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[50] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44, arg45, arg46, arg47,
                     arg48, arg49};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,50,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<50;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 50; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }
  if (args[45].argtype == OPS_ARG_DAT) {
    xdim45 = args[45].dat->size[0]*args[45].dat->dim;
    multi_d45 = args[45].dat->dim;
    #ifdef OPS_3D
    ydim45 = args[45].dat->size[1];
    #endif
  }
  if (args[46].argtype == OPS_ARG_DAT) {
    xdim46 = args[46].dat->size[0]*args[46].dat->dim;
    multi_d46 = args[46].dat->dim;
    #ifdef OPS_3D
    ydim46 = args[46].dat->size[1];
    #endif
  }
  if (args[47].argtype == OPS_ARG_DAT) {
    xdim47 = args[47].dat->size[0]*args[47].dat->dim;
    multi_d47 = args[47].dat->dim;
    #ifdef OPS_3D
    ydim47 = args[47].dat->size[1];
    #endif
  }
  if (args[48].argtype == OPS_ARG_DAT) {
    xdim48 = args[48].dat->size[0]*args[48].dat->dim;
    multi_d48 = args[48].dat->dim;
    #ifdef OPS_3D
    ydim48 = args[48].dat->size[1];
    #endif
  }
  if (args[49].argtype == OPS_ARG_DAT) {
    xdim49 = args[49].dat->size[0]*args[49].dat->dim;
    multi_d49 = args[49].dat->dim;
    #ifdef OPS_3D
    ydim49 = args[49].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 50);
  ops_halo_exchanges(args,50,range);
  ops_H_D_exchanges_host(args, 50);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44], (T45 *)p_a[45], (T46 *)p_a[46], (T47 *)p_a[47],
           (T48 *)p_a[48], (T49 *)p_a[49] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<50; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ) ops_dump3(args[45].dat,name);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ) ops_dump3(args[46].dat,name);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ) ops_dump3(args[47].dat,name);
  if (args[48].argtype == OPS_ARG_DAT && args[48].acc != OPS_READ) ops_dump3(args[48].dat,name);
  if (args[49].argtype == OPS_ARG_DAT && args[49].acc != OPS_READ) ops_dump3(args[49].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[45],range);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[46],range);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[47],range);
  if (args[48].argtype == OPS_ARG_DAT && args[48].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[48],range);
  if (args[49].argtype == OPS_ARG_DAT && args[49].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[49],range);
  ops_set_dirtybit_host(args, 50);
}


//
//ops_par_loop routine for 51 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17,class T18,class T19,
class T20,class T21,class T22,class T23,
class T24,class T25,class T26,class T27,
class T28,class T29,class T30,class T31,
class T32,class T33,class T34,class T35,
class T36,class T37,class T38,class T39,
class T40,class T41,class T42,class T43,
class T44,class T45,class T46,class T47,
class T48,class T49,class T50>
void ops_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*, T18*, T19*,
                           T20*, T21*, T22*, T23*,
                           T24*, T25*, T26*, T27*,
                           T28*, T29*, T30*, T31*,
                           T32*, T33*, T34*, T35*,
                           T36*, T37*, T38*, T39*,
                           T40*, T41*, T42*, T43*,
                           T44*, T45*, T46*, T47*,
                           T48*, T49*, T50*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19,
     ops_arg arg20, ops_arg arg21, ops_arg arg22, ops_arg arg23,
     ops_arg arg24, ops_arg arg25, ops_arg arg26, ops_arg arg27,
     ops_arg arg28, ops_arg arg29, ops_arg arg30, ops_arg arg31,
     ops_arg arg32, ops_arg arg33, ops_arg arg34, ops_arg arg35,
     ops_arg arg36, ops_arg arg37, ops_arg arg38, ops_arg arg39,
     ops_arg arg40, ops_arg arg41, ops_arg arg42, ops_arg arg43,
     ops_arg arg44, ops_arg arg45, ops_arg arg46, ops_arg arg47,
     ops_arg arg48, ops_arg arg49, ops_arg arg50) {

  char *p_a[51];
  int  offs[51][OPS_MAX_DIM];

  int  count[dim];
  ops_arg args[51] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17, arg18, arg19,
                     arg20, arg21, arg22, arg23,
                     arg24, arg25, arg26, arg27,
                     arg28, arg29, arg30, arg31,
                     arg32, arg33, arg34, arg35,
                     arg36, arg37, arg38, arg39,
                     arg40, arg41, arg42, arg43,
                     arg44, arg45, arg46, arg47,
                     arg48, arg49, arg50};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,51,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<51;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[0], &end[0],
                         args[i].dat->size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 51; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];
  #endif //OPS_MPI
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->elem_size, &start[0],
        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,
        d_m);
    }
    else if (args[i].argtype == OPS_ARG_GBL) {
      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;
      else
  #ifdef OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;
  #else //OPS_MPI
        p_a[i] = ((ops_reduction)args[i].data)->data;
  #endif //OPS_MPI
    } else if (args[i].argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      p_a[i] = (char *)arg_idx;
    }
  }

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT) {
    xdim0 = args[0].dat->size[0]*args[0].dat->dim;
    multi_d0 = args[0].dat->dim;
    #ifdef OPS_3D
    ydim0 = args[0].dat->size[1];
    #endif
  }
  if (args[1].argtype == OPS_ARG_DAT) {
    xdim1 = args[1].dat->size[0]*args[1].dat->dim;
    multi_d1 = args[1].dat->dim;
    #ifdef OPS_3D
    ydim1 = args[1].dat->size[1];
    #endif
  }
  if (args[2].argtype == OPS_ARG_DAT) {
    xdim2 = args[2].dat->size[0]*args[2].dat->dim;
    multi_d2 = args[2].dat->dim;
    #ifdef OPS_3D
    ydim2 = args[2].dat->size[1];
    #endif
  }
  if (args[3].argtype == OPS_ARG_DAT) {
    xdim3 = args[3].dat->size[0]*args[3].dat->dim;
    multi_d3 = args[3].dat->dim;
    #ifdef OPS_3D
    ydim3 = args[3].dat->size[1];
    #endif
  }
  if (args[4].argtype == OPS_ARG_DAT) {
    xdim4 = args[4].dat->size[0]*args[4].dat->dim;
    multi_d4 = args[4].dat->dim;
    #ifdef OPS_3D
    ydim4 = args[4].dat->size[1];
    #endif
  }
  if (args[5].argtype == OPS_ARG_DAT) {
    xdim5 = args[5].dat->size[0]*args[5].dat->dim;
    multi_d5 = args[5].dat->dim;
    #ifdef OPS_3D
    ydim5 = args[5].dat->size[1];
    #endif
  }
  if (args[6].argtype == OPS_ARG_DAT) {
    xdim6 = args[6].dat->size[0]*args[6].dat->dim;
    multi_d6 = args[6].dat->dim;
    #ifdef OPS_3D
    ydim6 = args[6].dat->size[1];
    #endif
  }
  if (args[7].argtype == OPS_ARG_DAT) {
    xdim7 = args[7].dat->size[0]*args[7].dat->dim;
    multi_d7 = args[7].dat->dim;
    #ifdef OPS_3D
    ydim7 = args[7].dat->size[1];
    #endif
  }
  if (args[8].argtype == OPS_ARG_DAT) {
    xdim8 = args[8].dat->size[0]*args[8].dat->dim;
    multi_d8 = args[8].dat->dim;
    #ifdef OPS_3D
    ydim8 = args[8].dat->size[1];
    #endif
  }
  if (args[9].argtype == OPS_ARG_DAT) {
    xdim9 = args[9].dat->size[0]*args[9].dat->dim;
    multi_d9 = args[9].dat->dim;
    #ifdef OPS_3D
    ydim9 = args[9].dat->size[1];
    #endif
  }
  if (args[10].argtype == OPS_ARG_DAT) {
    xdim10 = args[10].dat->size[0]*args[10].dat->dim;
    multi_d10 = args[10].dat->dim;
    #ifdef OPS_3D
    ydim10 = args[10].dat->size[1];
    #endif
  }
  if (args[11].argtype == OPS_ARG_DAT) {
    xdim11 = args[11].dat->size[0]*args[11].dat->dim;
    multi_d11 = args[11].dat->dim;
    #ifdef OPS_3D
    ydim11 = args[11].dat->size[1];
    #endif
  }
  if (args[12].argtype == OPS_ARG_DAT) {
    xdim12 = args[12].dat->size[0]*args[12].dat->dim;
    multi_d12 = args[12].dat->dim;
    #ifdef OPS_3D
    ydim12 = args[12].dat->size[1];
    #endif
  }
  if (args[13].argtype == OPS_ARG_DAT) {
    xdim13 = args[13].dat->size[0]*args[13].dat->dim;
    multi_d13 = args[13].dat->dim;
    #ifdef OPS_3D
    ydim13 = args[13].dat->size[1];
    #endif
  }
  if (args[14].argtype == OPS_ARG_DAT) {
    xdim14 = args[14].dat->size[0]*args[14].dat->dim;
    multi_d14 = args[14].dat->dim;
    #ifdef OPS_3D
    ydim14 = args[14].dat->size[1];
    #endif
  }
  if (args[15].argtype == OPS_ARG_DAT) {
    xdim15 = args[15].dat->size[0]*args[15].dat->dim;
    multi_d15 = args[15].dat->dim;
    #ifdef OPS_3D
    ydim15 = args[15].dat->size[1];
    #endif
  }
  if (args[16].argtype == OPS_ARG_DAT) {
    xdim16 = args[16].dat->size[0]*args[16].dat->dim;
    multi_d16 = args[16].dat->dim;
    #ifdef OPS_3D
    ydim16 = args[16].dat->size[1];
    #endif
  }
  if (args[17].argtype == OPS_ARG_DAT) {
    xdim17 = args[17].dat->size[0]*args[17].dat->dim;
    multi_d17 = args[17].dat->dim;
    #ifdef OPS_3D
    ydim17 = args[17].dat->size[1];
    #endif
  }
  if (args[18].argtype == OPS_ARG_DAT) {
    xdim18 = args[18].dat->size[0]*args[18].dat->dim;
    multi_d18 = args[18].dat->dim;
    #ifdef OPS_3D
    ydim18 = args[18].dat->size[1];
    #endif
  }
  if (args[19].argtype == OPS_ARG_DAT) {
    xdim19 = args[19].dat->size[0]*args[19].dat->dim;
    multi_d19 = args[19].dat->dim;
    #ifdef OPS_3D
    ydim19 = args[19].dat->size[1];
    #endif
  }
  if (args[20].argtype == OPS_ARG_DAT) {
    xdim20 = args[20].dat->size[0]*args[20].dat->dim;
    multi_d20 = args[20].dat->dim;
    #ifdef OPS_3D
    ydim20 = args[20].dat->size[1];
    #endif
  }
  if (args[21].argtype == OPS_ARG_DAT) {
    xdim21 = args[21].dat->size[0]*args[21].dat->dim;
    multi_d21 = args[21].dat->dim;
    #ifdef OPS_3D
    ydim21 = args[21].dat->size[1];
    #endif
  }
  if (args[22].argtype == OPS_ARG_DAT) {
    xdim22 = args[22].dat->size[0]*args[22].dat->dim;
    multi_d22 = args[22].dat->dim;
    #ifdef OPS_3D
    ydim22 = args[22].dat->size[1];
    #endif
  }
  if (args[23].argtype == OPS_ARG_DAT) {
    xdim23 = args[23].dat->size[0]*args[23].dat->dim;
    multi_d23 = args[23].dat->dim;
    #ifdef OPS_3D
    ydim23 = args[23].dat->size[1];
    #endif
  }
  if (args[24].argtype == OPS_ARG_DAT) {
    xdim24 = args[24].dat->size[0]*args[24].dat->dim;
    multi_d24 = args[24].dat->dim;
    #ifdef OPS_3D
    ydim24 = args[24].dat->size[1];
    #endif
  }
  if (args[25].argtype == OPS_ARG_DAT) {
    xdim25 = args[25].dat->size[0]*args[25].dat->dim;
    multi_d25 = args[25].dat->dim;
    #ifdef OPS_3D
    ydim25 = args[25].dat->size[1];
    #endif
  }
  if (args[26].argtype == OPS_ARG_DAT) {
    xdim26 = args[26].dat->size[0]*args[26].dat->dim;
    multi_d26 = args[26].dat->dim;
    #ifdef OPS_3D
    ydim26 = args[26].dat->size[1];
    #endif
  }
  if (args[27].argtype == OPS_ARG_DAT) {
    xdim27 = args[27].dat->size[0]*args[27].dat->dim;
    multi_d27 = args[27].dat->dim;
    #ifdef OPS_3D
    ydim27 = args[27].dat->size[1];
    #endif
  }
  if (args[28].argtype == OPS_ARG_DAT) {
    xdim28 = args[28].dat->size[0]*args[28].dat->dim;
    multi_d28 = args[28].dat->dim;
    #ifdef OPS_3D
    ydim28 = args[28].dat->size[1];
    #endif
  }
  if (args[29].argtype == OPS_ARG_DAT) {
    xdim29 = args[29].dat->size[0]*args[29].dat->dim;
    multi_d29 = args[29].dat->dim;
    #ifdef OPS_3D
    ydim29 = args[29].dat->size[1];
    #endif
  }
  if (args[30].argtype == OPS_ARG_DAT) {
    xdim30 = args[30].dat->size[0]*args[30].dat->dim;
    multi_d30 = args[30].dat->dim;
    #ifdef OPS_3D
    ydim30 = args[30].dat->size[1];
    #endif
  }
  if (args[31].argtype == OPS_ARG_DAT) {
    xdim31 = args[31].dat->size[0]*args[31].dat->dim;
    multi_d31 = args[31].dat->dim;
    #ifdef OPS_3D
    ydim31 = args[31].dat->size[1];
    #endif
  }
  if (args[32].argtype == OPS_ARG_DAT) {
    xdim32 = args[32].dat->size[0]*args[32].dat->dim;
    multi_d32 = args[32].dat->dim;
    #ifdef OPS_3D
    ydim32 = args[32].dat->size[1];
    #endif
  }
  if (args[33].argtype == OPS_ARG_DAT) {
    xdim33 = args[33].dat->size[0]*args[33].dat->dim;
    multi_d33 = args[33].dat->dim;
    #ifdef OPS_3D
    ydim33 = args[33].dat->size[1];
    #endif
  }
  if (args[34].argtype == OPS_ARG_DAT) {
    xdim34 = args[34].dat->size[0]*args[34].dat->dim;
    multi_d34 = args[34].dat->dim;
    #ifdef OPS_3D
    ydim34 = args[34].dat->size[1];
    #endif
  }
  if (args[35].argtype == OPS_ARG_DAT) {
    xdim35 = args[35].dat->size[0]*args[35].dat->dim;
    multi_d35 = args[35].dat->dim;
    #ifdef OPS_3D
    ydim35 = args[35].dat->size[1];
    #endif
  }
  if (args[36].argtype == OPS_ARG_DAT) {
    xdim36 = args[36].dat->size[0]*args[36].dat->dim;
    multi_d36 = args[36].dat->dim;
    #ifdef OPS_3D
    ydim36 = args[36].dat->size[1];
    #endif
  }
  if (args[37].argtype == OPS_ARG_DAT) {
    xdim37 = args[37].dat->size[0]*args[37].dat->dim;
    multi_d37 = args[37].dat->dim;
    #ifdef OPS_3D
    ydim37 = args[37].dat->size[1];
    #endif
  }
  if (args[38].argtype == OPS_ARG_DAT) {
    xdim38 = args[38].dat->size[0]*args[38].dat->dim;
    multi_d38 = args[38].dat->dim;
    #ifdef OPS_3D
    ydim38 = args[38].dat->size[1];
    #endif
  }
  if (args[39].argtype == OPS_ARG_DAT) {
    xdim39 = args[39].dat->size[0]*args[39].dat->dim;
    multi_d39 = args[39].dat->dim;
    #ifdef OPS_3D
    ydim39 = args[39].dat->size[1];
    #endif
  }
  if (args[40].argtype == OPS_ARG_DAT) {
    xdim40 = args[40].dat->size[0]*args[40].dat->dim;
    multi_d40 = args[40].dat->dim;
    #ifdef OPS_3D
    ydim40 = args[40].dat->size[1];
    #endif
  }
  if (args[41].argtype == OPS_ARG_DAT) {
    xdim41 = args[41].dat->size[0]*args[41].dat->dim;
    multi_d41 = args[41].dat->dim;
    #ifdef OPS_3D
    ydim41 = args[41].dat->size[1];
    #endif
  }
  if (args[42].argtype == OPS_ARG_DAT) {
    xdim42 = args[42].dat->size[0]*args[42].dat->dim;
    multi_d42 = args[42].dat->dim;
    #ifdef OPS_3D
    ydim42 = args[42].dat->size[1];
    #endif
  }
  if (args[43].argtype == OPS_ARG_DAT) {
    xdim43 = args[43].dat->size[0]*args[43].dat->dim;
    multi_d43 = args[43].dat->dim;
    #ifdef OPS_3D
    ydim43 = args[43].dat->size[1];
    #endif
  }
  if (args[44].argtype == OPS_ARG_DAT) {
    xdim44 = args[44].dat->size[0]*args[44].dat->dim;
    multi_d44 = args[44].dat->dim;
    #ifdef OPS_3D
    ydim44 = args[44].dat->size[1];
    #endif
  }
  if (args[45].argtype == OPS_ARG_DAT) {
    xdim45 = args[45].dat->size[0]*args[45].dat->dim;
    multi_d45 = args[45].dat->dim;
    #ifdef OPS_3D
    ydim45 = args[45].dat->size[1];
    #endif
  }
  if (args[46].argtype == OPS_ARG_DAT) {
    xdim46 = args[46].dat->size[0]*args[46].dat->dim;
    multi_d46 = args[46].dat->dim;
    #ifdef OPS_3D
    ydim46 = args[46].dat->size[1];
    #endif
  }
  if (args[47].argtype == OPS_ARG_DAT) {
    xdim47 = args[47].dat->size[0]*args[47].dat->dim;
    multi_d47 = args[47].dat->dim;
    #ifdef OPS_3D
    ydim47 = args[47].dat->size[1];
    #endif
  }
  if (args[48].argtype == OPS_ARG_DAT) {
    xdim48 = args[48].dat->size[0]*args[48].dat->dim;
    multi_d48 = args[48].dat->dim;
    #ifdef OPS_3D
    ydim48 = args[48].dat->size[1];
    #endif
  }
  if (args[49].argtype == OPS_ARG_DAT) {
    xdim49 = args[49].dat->size[0]*args[49].dat->dim;
    multi_d49 = args[49].dat->dim;
    #ifdef OPS_3D
    ydim49 = args[49].dat->size[1];
    #endif
  }
  if (args[50].argtype == OPS_ARG_DAT) {
    xdim50 = args[50].dat->size[0]*args[50].dat->dim;
    multi_d50 = args[50].dat->dim;
    #ifdef OPS_3D
    ydim50 = args[50].dat->size[1];
    #endif
  }

  ops_H_D_exchanges_host(args, 51);
  ops_halo_exchanges(args,51,range);
  ops_H_D_exchanges_host(args, 51);
  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18], (T19 *)p_a[19],
           (T20 *)p_a[20], (T21 *)p_a[21], (T22 *)p_a[22], (T23 *)p_a[23],
           (T24 *)p_a[24], (T25 *)p_a[25], (T26 *)p_a[26], (T27 *)p_a[27],
           (T28 *)p_a[28], (T29 *)p_a[29], (T30 *)p_a[30], (T31 *)p_a[31],
           (T32 *)p_a[32], (T33 *)p_a[33], (T34 *)p_a[34], (T35 *)p_a[35],
           (T36 *)p_a[36], (T37 *)p_a[37], (T38 *)p_a[38], (T39 *)p_a[39],
           (T40 *)p_a[40], (T41 *)p_a[41], (T42 *)p_a[42], (T43 *)p_a[43],
           (T44 *)p_a[44], (T45 *)p_a[45], (T46 *)p_a[46], (T47 *)p_a[47],
           (T48 *)p_a[48], (T49 *)p_a[49], (T50 *)p_a[50] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<51; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);
      else if (args[i].argtype == OPS_ARG_IDX) {
        arg_idx[m]++;
  #ifdef OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
        for (int d = 0; d < m; d++) arg_idx[d] = start[d];
  #endif //OPS_MPI
      }
    }
  }

  #ifdef OPS_DEBUG_DUMP
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ) ops_dump3(args[0].dat,name);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ) ops_dump3(args[1].dat,name);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ) ops_dump3(args[2].dat,name);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ) ops_dump3(args[3].dat,name);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ) ops_dump3(args[4].dat,name);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ) ops_dump3(args[5].dat,name);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ) ops_dump3(args[6].dat,name);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ) ops_dump3(args[7].dat,name);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ) ops_dump3(args[8].dat,name);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ) ops_dump3(args[9].dat,name);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ) ops_dump3(args[10].dat,name);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ) ops_dump3(args[11].dat,name);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ) ops_dump3(args[12].dat,name);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ) ops_dump3(args[13].dat,name);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ) ops_dump3(args[14].dat,name);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ) ops_dump3(args[15].dat,name);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ) ops_dump3(args[16].dat,name);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ) ops_dump3(args[17].dat,name);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ) ops_dump3(args[18].dat,name);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ) ops_dump3(args[19].dat,name);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ) ops_dump3(args[20].dat,name);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ) ops_dump3(args[21].dat,name);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ) ops_dump3(args[22].dat,name);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ) ops_dump3(args[23].dat,name);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ) ops_dump3(args[24].dat,name);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ) ops_dump3(args[25].dat,name);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ) ops_dump3(args[26].dat,name);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ) ops_dump3(args[27].dat,name);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ) ops_dump3(args[28].dat,name);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ) ops_dump3(args[29].dat,name);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ) ops_dump3(args[30].dat,name);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ) ops_dump3(args[31].dat,name);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ) ops_dump3(args[32].dat,name);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ) ops_dump3(args[33].dat,name);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ) ops_dump3(args[34].dat,name);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ) ops_dump3(args[35].dat,name);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ) ops_dump3(args[36].dat,name);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ) ops_dump3(args[37].dat,name);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ) ops_dump3(args[38].dat,name);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ) ops_dump3(args[39].dat,name);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ) ops_dump3(args[40].dat,name);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ) ops_dump3(args[41].dat,name);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ) ops_dump3(args[42].dat,name);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ) ops_dump3(args[43].dat,name);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ) ops_dump3(args[44].dat,name);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ) ops_dump3(args[45].dat,name);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ) ops_dump3(args[46].dat,name);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ) ops_dump3(args[47].dat,name);
  if (args[48].argtype == OPS_ARG_DAT && args[48].acc != OPS_READ) ops_dump3(args[48].dat,name);
  if (args[49].argtype == OPS_ARG_DAT && args[49].acc != OPS_READ) ops_dump3(args[49].dat,name);
  if (args[50].argtype == OPS_ARG_DAT && args[50].acc != OPS_READ) ops_dump3(args[50].dat,name);
  #endif
  if (args[0].argtype == OPS_ARG_DAT && args[0].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[0],range);
  if (args[1].argtype == OPS_ARG_DAT && args[1].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[1],range);
  if (args[2].argtype == OPS_ARG_DAT && args[2].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[2],range);
  if (args[3].argtype == OPS_ARG_DAT && args[3].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[3],range);
  if (args[4].argtype == OPS_ARG_DAT && args[4].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[4],range);
  if (args[5].argtype == OPS_ARG_DAT && args[5].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[5],range);
  if (args[6].argtype == OPS_ARG_DAT && args[6].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[6],range);
  if (args[7].argtype == OPS_ARG_DAT && args[7].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[7],range);
  if (args[8].argtype == OPS_ARG_DAT && args[8].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[8],range);
  if (args[9].argtype == OPS_ARG_DAT && args[9].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[9],range);
  if (args[10].argtype == OPS_ARG_DAT && args[10].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[10],range);
  if (args[11].argtype == OPS_ARG_DAT && args[11].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[11],range);
  if (args[12].argtype == OPS_ARG_DAT && args[12].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[12],range);
  if (args[13].argtype == OPS_ARG_DAT && args[13].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[13],range);
  if (args[14].argtype == OPS_ARG_DAT && args[14].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[14],range);
  if (args[15].argtype == OPS_ARG_DAT && args[15].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[15],range);
  if (args[16].argtype == OPS_ARG_DAT && args[16].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[16],range);
  if (args[17].argtype == OPS_ARG_DAT && args[17].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[17],range);
  if (args[18].argtype == OPS_ARG_DAT && args[18].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[18],range);
  if (args[19].argtype == OPS_ARG_DAT && args[19].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[19],range);
  if (args[20].argtype == OPS_ARG_DAT && args[20].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[20],range);
  if (args[21].argtype == OPS_ARG_DAT && args[21].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[21],range);
  if (args[22].argtype == OPS_ARG_DAT && args[22].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[22],range);
  if (args[23].argtype == OPS_ARG_DAT && args[23].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[23],range);
  if (args[24].argtype == OPS_ARG_DAT && args[24].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[24],range);
  if (args[25].argtype == OPS_ARG_DAT && args[25].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[25],range);
  if (args[26].argtype == OPS_ARG_DAT && args[26].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[26],range);
  if (args[27].argtype == OPS_ARG_DAT && args[27].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[27],range);
  if (args[28].argtype == OPS_ARG_DAT && args[28].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[28],range);
  if (args[29].argtype == OPS_ARG_DAT && args[29].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[29],range);
  if (args[30].argtype == OPS_ARG_DAT && args[30].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[30],range);
  if (args[31].argtype == OPS_ARG_DAT && args[31].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[31],range);
  if (args[32].argtype == OPS_ARG_DAT && args[32].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[32],range);
  if (args[33].argtype == OPS_ARG_DAT && args[33].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[33],range);
  if (args[34].argtype == OPS_ARG_DAT && args[34].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[34],range);
  if (args[35].argtype == OPS_ARG_DAT && args[35].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[35],range);
  if (args[36].argtype == OPS_ARG_DAT && args[36].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[36],range);
  if (args[37].argtype == OPS_ARG_DAT && args[37].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[37],range);
  if (args[38].argtype == OPS_ARG_DAT && args[38].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[38],range);
  if (args[39].argtype == OPS_ARG_DAT && args[39].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[39],range);
  if (args[40].argtype == OPS_ARG_DAT && args[40].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[40],range);
  if (args[41].argtype == OPS_ARG_DAT && args[41].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[41],range);
  if (args[42].argtype == OPS_ARG_DAT && args[42].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[42],range);
  if (args[43].argtype == OPS_ARG_DAT && args[43].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[43],range);
  if (args[44].argtype == OPS_ARG_DAT && args[44].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[44],range);
  if (args[45].argtype == OPS_ARG_DAT && args[45].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[45],range);
  if (args[46].argtype == OPS_ARG_DAT && args[46].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[46],range);
  if (args[47].argtype == OPS_ARG_DAT && args[47].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[47],range);
  if (args[48].argtype == OPS_ARG_DAT && args[48].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[48],range);
  if (args[49].argtype == OPS_ARG_DAT && args[49].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[49],range);
  if (args[50].argtype == OPS_ARG_DAT && args[50].acc != OPS_READ)  ops_set_halo_dirtybit3(&args[50],range);
  ops_set_dirtybit_host(args, 51);
}
