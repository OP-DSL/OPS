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
#ifdef OPS_3D     // macros for 3D application
#ifndef OPS_DEBUG // no debug checks
#define OPS_ACC0(x, y, z) (x + xdim0 * (y) + ydim0 * xdim0 * (z))
#define OPS_ACC1(x, y, z) (x + xdim1 * (y) + ydim1 * xdim1 * (z))
#define OPS_ACC2(x, y, z) (x + xdim2 * (y) + ydim2 * xdim2 * (z))
#define OPS_ACC3(x, y, z) (x + xdim3 * (y) + ydim3 * xdim3 * (z))
#define OPS_ACC4(x, y, z) (x + xdim4 * (y) + ydim4 * xdim4 * (z))
#define OPS_ACC5(x, y, z) (x + xdim5 * (y) + ydim5 * xdim5 * (z))
#define OPS_ACC6(x, y, z) (x + xdim6 * (y) + ydim6 * xdim6 * (z))
#define OPS_ACC7(x, y, z) (x + xdim7 * (y) + ydim7 * xdim7 * (z))
#define OPS_ACC8(x, y, z) (x + xdim8 * (y) + ydim8 * xdim8 * (z))
#define OPS_ACC9(x, y, z) (x + xdim9 * (y) + ydim9 * xdim9 * (z))
#define OPS_ACC10(x, y, z) (x + xdim10 * (y) + ydim10 * xdim10 * (z))
#define OPS_ACC11(x, y, z) (x + xdim11 * (y) + ydim11 * xdim11 * (z))
#define OPS_ACC12(x, y, z) (x + xdim12 * (y) + ydim12 * xdim12 * (z))
#define OPS_ACC13(x, y, z) (x + xdim13 * (y) + ydim13 * xdim13 * (z))
#define OPS_ACC14(x, y, z) (x + xdim14 * (y) + ydim14 * xdim14 * (z))
#define OPS_ACC15(x, y, z) (x + xdim15 * (y) + ydim15 * xdim15 * (z))
#define OPS_ACC16(x, y, z) (x + xdim16 * (y) + ydim16 * xdim16 * (z))
#define OPS_ACC17(x, y, z) (x + xdim17 * (y) + ydim17 * xdim17 * (z))
#define OPS_ACC18(x, y, z) (x + xdim18 * (y) + ydim18 * xdim18 * (z))
#define OPS_ACC19(x, y, z) (x + xdim19 * (y) + ydim19 * xdim19 * (z))
#define OPS_ACC20(x, y, z) (x + xdim20 * (y) + ydim20 * xdim20 * (z))
#define OPS_ACC21(x, y, z) (x + xdim21 * (y) + ydim21 * xdim21 * (z))
#define OPS_ACC22(x, y, z) (x + xdim22 * (y) + ydim22 * xdim22 * (z))
#define OPS_ACC23(x, y, z) (x + xdim23 * (y) + ydim23 * xdim23 * (z))
#define OPS_ACC24(x, y, z) (x + xdim24 * (y) + ydim24 * xdim24 * (z))
#define OPS_ACC25(x, y, z) (x + xdim25 * (y) + ydim25 * xdim25 * (z))
#define OPS_ACC26(x, y, z) (x + xdim26 * (y) + ydim26 * xdim26 * (z))
#define OPS_ACC27(x, y, z) (x + xdim27 * (y) + ydim27 * xdim27 * (z))
#define OPS_ACC28(x, y, z) (x + xdim28 * (y) + ydim28 * xdim28 * (z))
#define OPS_ACC29(x, y, z) (x + xdim29 * (y) + ydim29 * xdim29 * (z))
#define OPS_ACC30(x, y, z) (x + xdim30 * (y) + ydim30 * xdim30 * (z))
#define OPS_ACC31(x, y, z) (x + xdim31 * (y) + ydim31 * xdim31 * (z))
#define OPS_ACC32(x, y, z) (x + xdim32 * (y) + ydim32 * xdim32 * (z))
#define OPS_ACC33(x, y, z) (x + xdim33 * (y) + ydim33 * xdim33 * (z))
#define OPS_ACC34(x, y, z) (x + xdim34 * (y) + ydim34 * xdim34 * (z))
#define OPS_ACC35(x, y, z) (x + xdim35 * (y) + ydim35 * xdim35 * (z))
#define OPS_ACC36(x, y, z) (x + xdim36 * (y) + ydim36 * xdim36 * (z))
#define OPS_ACC37(x, y, z) (x + xdim37 * (y) + ydim37 * xdim37 * (z))
#define OPS_ACC38(x, y, z) (x + xdim38 * (y) + ydim38 * xdim38 * (z))
#define OPS_ACC39(x, y, z) (x + xdim39 * (y) + ydim39 * xdim39 * (z))
#define OPS_ACC40(x, y, z) (x + xdim40 * (y) + ydim40 * xdim40 * (z))
#define OPS_ACC41(x, y, z) (x + xdim41 * (y) + ydim41 * xdim41 * (z))
#define OPS_ACC42(x, y, z) (x + xdim42 * (y) + ydim42 * xdim42 * (z))
#define OPS_ACC43(x, y, z) (x + xdim43 * (y) + ydim43 * xdim43 * (z))
#define OPS_ACC44(x, y, z) (x + xdim44 * (y) + ydim44 * xdim44 * (z))
#define OPS_ACC45(x, y, z) (x + xdim45 * (y) + ydim45 * xdim45 * (z))
#define OPS_ACC46(x, y, z) (x + xdim46 * (y) + ydim46 * xdim46 * (z))
#define OPS_ACC47(x, y, z) (x + xdim47 * (y) + ydim47 * xdim47 * (z))
#define OPS_ACC48(x, y, z) (x + xdim48 * (y) + ydim48 * xdim48 * (z))
#define OPS_ACC49(x, y, z) (x + xdim49 * (y) + ydim49 * xdim49 * (z))
#define OPS_ACC50(x, y, z) (x + xdim50 * (y) + ydim50 * xdim50 * (z))
#define OPS_ACC51(x, y, z) (x + xdim51 * (y) + ydim51 * xdim51 * (z))
#define OPS_ACC52(x, y, z) (x + xdim52 * (y) + ydim52 * xdim52 * (z))
#define OPS_ACC53(x, y, z) (x + xdim53 * (y) + ydim53 * xdim53 * (z))
#define OPS_ACC54(x, y, z) (x + xdim54 * (y) + ydim54 * xdim54 * (z))
#define OPS_ACC55(x, y, z) (x + xdim55 * (y) + ydim55 * xdim55 * (z))
#define OPS_ACC56(x, y, z) (x + xdim56 * (y) + ydim56 * xdim56 * (z))
#define OPS_ACC57(x, y, z) (x + xdim57 * (y) + ydim57 * xdim57 * (z))
#define OPS_ACC58(x, y, z) (x + xdim58 * (y) + ydim58 * xdim58 * (z))
#define OPS_ACC59(x, y, z) (x + xdim59 * (y) + ydim59 * xdim59 * (z))
#define OPS_ACC60(x, y, z) (x + xdim60 * (y) + ydim60 * xdim60 * (z))
#define OPS_ACC61(x, y, z) (x + xdim61 * (y) + ydim61 * xdim61 * (z))
#define OPS_ACC62(x, y, z) (x + xdim62 * (y) + ydim62 * xdim62 * (z))
#define OPS_ACC63(x, y, z) (x + xdim63 * (y) + ydim63 * xdim63 * (z))
#define OPS_ACC64(x, y, z) (x + xdim64 * (y) + ydim64 * xdim64 * (z))
#define OPS_ACC65(x, y, z) (x + xdim65 * (y) + ydim65 * xdim65 * (z))
#define OPS_ACC66(x, y, z) (x + xdim66 * (y) + ydim66 * xdim66 * (z))
#define OPS_ACC67(x, y, z) (x + xdim67 * (y) + ydim67 * xdim67 * (z))
#define OPS_ACC68(x, y, z) (x + xdim68 * (y) + ydim68 * xdim68 * (z))
#define OPS_ACC69(x, y, z) (x + xdim69 * (y) + ydim69 * xdim69 * (z))
#define OPS_ACC70(x, y, z) (x + xdim70 * (y) + ydim70 * xdim70 * (z))
#define OPS_ACC71(x, y, z) (x + xdim71 * (y) + ydim71 * xdim71 * (z))
#define OPS_ACC72(x, y, z) (x + xdim72 * (y) + ydim72 * xdim72 * (z))
#define OPS_ACC73(x, y, z) (x + xdim73 * (y) + ydim73 * xdim73 * (z))
#define OPS_ACC74(x, y, z) (x + xdim74 * (y) + ydim74 * xdim74 * (z))
#define OPS_ACC75(x, y, z) (x + xdim75 * (y) + ydim75 * xdim75 * (z))
#define OPS_ACC76(x, y, z) (x + xdim76 * (y) + ydim76 * xdim76 * (z))
#define OPS_ACC77(x, y, z) (x + xdim77 * (y) + xdim77 * ydim77 * (z))
#define OPS_ACC78(x, y, z) (x + xdim78 * (y) + xdim78 * ydim78 * (z))
#define OPS_ACC79(x, y, z) (x + xdim79 * (y) + xdim79 * ydim79 * (z))
#define OPS_ACC80(x, y, z) (x + xdim80 * (y) + xdim80 * ydim80 * (z))
#define OPS_ACC81(x, y, z) (x + xdim81 * (y) + xdim81 * ydim81 * (z))
#define OPS_ACC82(x, y, z) (x + xdim82 * (y) + xdim82 * ydim82 * (z))
#define OPS_ACC83(x, y, z) (x + xdim83 * (y) + xdim83 * ydim83 * (z))
#define OPS_ACC84(x, y, z) (x + xdim84 * (y) + xdim84 * ydim84 * (z))
#define OPS_ACC85(x, y, z) (x + xdim85 * (y) + xdim85 * ydim85 * (z))
#define OPS_ACC86(x, y, z) (x + xdim86 * (y) + xdim86 * ydim86 * (z))
#define OPS_ACC87(x, y, z) (x + xdim87 * (y) + xdim87 * ydim87 * (z))
#define OPS_ACC88(x, y, z) (x + xdim88 * (y) + xdim88 * ydim88 * (z))
#define OPS_ACC89(x, y, z) (x + xdim89 * (y) + xdim89 * ydim89 * (z))
#define OPS_ACC90(x, y, z) (x + xdim90 * (y) + xdim90 * ydim90 * (z))
#define OPS_ACC91(x, y, z) (x + xdim91 * (y) + xdim91 * ydim91 * (z))
#define OPS_ACC92(x, y, z) (x + xdim92 * (y) + xdim92 * ydim92 * (z))
#define OPS_ACC93(x, y, z) (x + xdim93 * (y) + xdim93 * ydim93 * (z))
#define OPS_ACC94(x, y, z) (x + xdim94 * (y) + xdim94 * ydim94 * (z))
#define OPS_ACC95(x, y, z) (x + xdim95 * (y) + xdim95 * ydim95 * (z))
#define OPS_ACC96(x, y, z) (x + xdim96 * (y) + xdim96 * ydim96 * (z))
#define OPS_ACC97(x, y, z) (x + xdim97 * (y) + xdim97 * ydim97 * (z))
#define OPS_ACC98(x, y, z) (x + xdim98 * (y) + xdim98 * ydim98 * (z))
#define OPS_ACC99(x, y, z) (x + xdim99 * (y) + xdim99 * ydim99 * (z))
#else // do debug checks
#define OPS_ACC0(x, y, z) (ops_stencil_check_3d(0, x, y, z, xdim0, ydim0))
#define OPS_ACC1(x, y, z) (ops_stencil_check_3d(1, x, y, z, xdim1, ydim1))
#define OPS_ACC2(x, y, z) (ops_stencil_check_3d(2, x, y, z, xdim2, ydim2))
#define OPS_ACC3(x, y, z) (ops_stencil_check_3d(3, x, y, z, xdim3, ydim3))
#define OPS_ACC4(x, y, z) (ops_stencil_check_3d(4, x, y, z, xdim4, ydim4))
#define OPS_ACC5(x, y, z) (ops_stencil_check_3d(5, x, y, z, xdim5, ydim5))
#define OPS_ACC6(x, y, z) (ops_stencil_check_3d(6, x, y, z, xdim6, ydim6))
#define OPS_ACC7(x, y, z) (ops_stencil_check_3d(7, x, y, z, xdim7, ydim7))
#define OPS_ACC8(x, y, z) (ops_stencil_check_3d(8, x, y, z, xdim8, ydim8))
#define OPS_ACC9(x, y, z) (ops_stencil_check_3d(9, x, y, z, xdim9, ydim9))
#define OPS_ACC10(x, y, z) (ops_stencil_check_3d(10, x, y, z, xdim10, ydim10))
#define OPS_ACC11(x, y, z) (ops_stencil_check_3d(11, x, y, z, xdim11, ydim11))
#define OPS_ACC12(x, y, z) (ops_stencil_check_3d(12, x, y, z, xdim12, ydim12))
#define OPS_ACC13(x, y, z) (ops_stencil_check_3d(13, x, y, z, xdim13, ydim13))
#define OPS_ACC14(x, y, z) (ops_stencil_check_3d(14, x, y, z, xdim14, ydim14))
#define OPS_ACC15(x, y, z) (ops_stencil_check_3d(15, x, y, z, xdim15, ydim15))
#define OPS_ACC16(x, y, z) (ops_stencil_check_3d(16, x, y, z, xdim16, ydim16))
#define OPS_ACC17(x, y, z) (ops_stencil_check_3d(17, x, y, z, xdim17, ydim17))
#define OPS_ACC18(x, y, z) (ops_stencil_check_3d(18, x, y, z, xdim18, ydim18))
#define OPS_ACC19(x, y, z) (ops_stencil_check_3d(19, x, y, z, xdim19, ydim19))
#define OPS_ACC20(x, y, z) (ops_stencil_check_3d(20, x, y, z, xdim20, ydim20))
#define OPS_ACC21(x, y, z) (ops_stencil_check_3d(21, x, y, z, xdim21, ydim21))
#define OPS_ACC22(x, y, z) (ops_stencil_check_3d(22, x, y, z, xdim22, ydim22))
#define OPS_ACC23(x, y, z) (ops_stencil_check_3d(23, x, y, z, xdim23, ydim23))
#define OPS_ACC24(x, y, z) (ops_stencil_check_3d(24, x, y, z, xdim24, ydim24))
#define OPS_ACC25(x, y, z) (ops_stencil_check_3d(25, x, y, z, xdim25, ydim25))
#define OPS_ACC26(x, y, z) (ops_stencil_check_3d(26, x, y, z, xdim26, ydim26))
#define OPS_ACC27(x, y, z) (ops_stencil_check_3d(27, x, y, z, xdim27, ydim27))
#define OPS_ACC28(x, y, z) (ops_stencil_check_3d(28, x, y, z, xdim28, ydim28))
#define OPS_ACC29(x, y, z) (ops_stencil_check_3d(29, x, y, z, xdim29, ydim29))
#define OPS_ACC30(x, y, z) (ops_stencil_check_3d(30, x, y, z, xdim30, ydim30))
#define OPS_ACC31(x, y, z) (ops_stencil_check_3d(31, x, y, z, xdim31, ydim31))
#define OPS_ACC32(x, y, z) (ops_stencil_check_3d(32, x, y, z, xdim32, ydim32))
#define OPS_ACC33(x, y, z) (ops_stencil_check_3d(33, x, y, z, xdim33, ydim33))
#define OPS_ACC34(x, y, z) (ops_stencil_check_3d(34, x, y, z, xdim34, ydim34))
#define OPS_ACC35(x, y, z) (ops_stencil_check_3d(35, x, y, z, xdim35, ydim35))
#define OPS_ACC36(x, y, z) (ops_stencil_check_3d(36, x, y, z, xdim36, ydim36))
#define OPS_ACC37(x, y, z) (ops_stencil_check_3d(37, x, y, z, xdim37, ydim37))
#define OPS_ACC38(x, y, z) (ops_stencil_check_3d(38, x, y, z, xdim38, ydim38))
#define OPS_ACC39(x, y, z) (ops_stencil_check_3d(39, x, y, z, xdim39, ydim39))
#define OPS_ACC40(x, y, z) (ops_stencil_check_3d(40, x, y, z, xdim40, ydim40))
#define OPS_ACC41(x, y, z) (ops_stencil_check_3d(41, x, y, z, xdim41, ydim41))
#define OPS_ACC42(x, y, z) (ops_stencil_check_3d(42, x, y, z, xdim42, ydim42))
#define OPS_ACC43(x, y, z) (ops_stencil_check_3d(43, x, y, z, xdim43, ydim43))
#define OPS_ACC44(x, y, z) (ops_stencil_check_3d(44, x, y, z, xdim44, ydim44))
#define OPS_ACC45(x, y, z) (ops_stencil_check_3d(45, x, y, z, xdim45, ydim45))
#define OPS_ACC46(x, y, z) (ops_stencil_check_3d(46, x, y, z, xdim46, ydim46))
#define OPS_ACC47(x, y, z) (ops_stencil_check_3d(47, x, y, z, xdim47, ydim47))
#define OPS_ACC48(x, y, z) (ops_stencil_check_3d(48, x, y, z, xdim48, ydim48))
#define OPS_ACC49(x, y, z) (ops_stencil_check_3d(49, x, y, z, xdim49, ydim49))
#define OPS_ACC50(x, y, z) (ops_stencil_check_3d(50, x, y, z, xdim50, ydim50))
#define OPS_ACC51(x, y, z) (ops_stencil_check_3d(51, x, y, z, xdim51, ydim51))
#define OPS_ACC52(x, y, z) (ops_stencil_check_3d(52, x, y, z, xdim52, ydim52))
#define OPS_ACC53(x, y, z) (ops_stencil_check_3d(53, x, y, z, xdim53, ydim53))
#define OPS_ACC54(x, y, z) (ops_stencil_check_3d(54, x, y, z, xdim54, ydim54))
#define OPS_ACC55(x, y, z) (ops_stencil_check_3d(55, x, y, z, xdim55, ydim55))
#define OPS_ACC56(x, y, z) (ops_stencil_check_3d(56, x, y, z, xdim56, ydim56))
#define OPS_ACC57(x, y, z) (ops_stencil_check_3d(57, x, y, z, xdim57, ydim57))
#define OPS_ACC58(x, y, z) (ops_stencil_check_3d(58, x, y, z, xdim58, ydim58))
#define OPS_ACC59(x, y, z) (ops_stencil_check_3d(59, x, y, z, xdim59, ydim59))
#define OPS_ACC60(x, y, z) (ops_stencil_check_3d(60, x, y, z, xdim60, ydim60))
#define OPS_ACC61(x, y, z) (ops_stencil_check_3d(61, x, y, z, xdim61, ydim61))
#define OPS_ACC62(x, y, z) (ops_stencil_check_3d(62, x, y, z, xdim62, ydim62))
#define OPS_ACC63(x, y, z) (ops_stencil_check_3d(63, x, y, z, xdim63, ydim63))
#define OPS_ACC64(x, y, z) (ops_stencil_check_3d(64, x, y, z, xdim64, ydim64))
#define OPS_ACC65(x, y, z) (ops_stencil_check_3d(65, x, y, z, xdim65, ydim65))
#define OPS_ACC66(x, y, z) (ops_stencil_check_3d(66, x, y, z, xdim66, ydim66))
#define OPS_ACC67(x, y, z) (ops_stencil_check_3d(67, x, y, z, xdim67, ydim67))
#define OPS_ACC68(x, y, z) (ops_stencil_check_3d(68, x, y, z, xdim68, ydim68))
#define OPS_ACC69(x, y, z) (ops_stencil_check_3d(69, x, y, z, xdim69, ydim69))
#define OPS_ACC70(x, y, z) (ops_stencil_check_3d(70, x, y, z, xdim70, ydim70))
#define OPS_ACC71(x, y, z) (ops_stencil_check_3d(71, x, y, z, xdim71, ydim71))
#define OPS_ACC72(x, y, z) (ops_stencil_check_3d(72, x, y, z, xdim72, ydim72))
#define OPS_ACC73(x, y, z) (ops_stencil_check_3d(73, x, y, z, xdim73, ydim73))
#define OPS_ACC74(x, y, z) (ops_stencil_check_3d(74, x, y, z, xdim74, ydim74))
#define OPS_ACC75(x, y, z) (ops_stencil_check_3d(75, x, y, z, xdim75, ydim75))
#define OPS_ACC76(x, y, z) (ops_stencil_check_3d(76, x, y, z, xdim76, ydim76))
#define OPS_ACC77(x, y, z) (ops_stencil_check_3d(77, x, y, z, xdim77, ydim77))
#define OPS_ACC78(x, y, z) (ops_stencil_check_3d(78, x, y, z, xdim78, ydim78))
#define OPS_ACC79(x, y, z) (ops_stencil_check_3d(79, x, y, z, xdim79, ydim79))
#define OPS_ACC80(x, y, z) (ops_stencil_check_3d(80, x, y, z, xdim80, ydim80))
#define OPS_ACC81(x, y, z) (ops_stencil_check_3d(81, x, y, z, xdim81, ydim81))
#define OPS_ACC82(x, y, z) (ops_stencil_check_3d(82, x, y, z, xdim82, ydim82))
#define OPS_ACC83(x, y, z) (ops_stencil_check_3d(83, x, y, z, xdim83, ydim83))
#define OPS_ACC84(x, y, z) (ops_stencil_check_3d(84, x, y, z, xdim84, ydim84))
#define OPS_ACC85(x, y, z) (ops_stencil_check_3d(85, x, y, z, xdim85, ydim85))
#define OPS_ACC86(x, y, z) (ops_stencil_check_3d(86, x, y, z, xdim86, ydim86))
#define OPS_ACC87(x, y, z) (ops_stencil_check_3d(87, x, y, z, xdim87, ydim87))
#define OPS_ACC88(x, y, z) (ops_stencil_check_3d(88, x, y, z, xdim88, ydim88))
#define OPS_ACC89(x, y, z) (ops_stencil_check_3d(89, x, y, z, xdim89, ydim89))
#define OPS_ACC90(x, y, z) (ops_stencil_check_3d(90, x, y, z, xdim90, ydim90))
#define OPS_ACC91(x, y, z) (ops_stencil_check_3d(91, x, y, z, xdim91, ydim91))
#define OPS_ACC92(x, y, z) (ops_stencil_check_3d(92, x, y, z, xdim92, ydim92))
#define OPS_ACC93(x, y, z) (ops_stencil_check_3d(93, x, y, z, xdim93, ydim93))
#define OPS_ACC94(x, y, z) (ops_stencil_check_3d(94, x, y, z, xdim94, ydim94))
#define OPS_ACC95(x, y, z) (ops_stencil_check_3d(95, x, y, z, xdim95, ydim95))
#define OPS_ACC96(x, y, z) (ops_stencil_check_3d(96, x, y, z, xdim96, ydim96))
#define OPS_ACC97(x, y, z) (ops_stencil_check_3d(97, x, y, z, xdim97, ydim97))
#define OPS_ACC98(x, y, z) (ops_stencil_check_3d(98, x, y, z, xdim98, ydim98))
#define OPS_ACC99(x, y, z) (ops_stencil_check_3d(99, x, y, z, xdim99, ydim99))
#endif               // end debug checks
#elif defined OPS_2D // macros for 2D application
#ifndef OPS_DEBUG    // no debug checks
#define OPS_ACC0(x, y) (x + xdim0 * (y))
#define OPS_ACC1(x, y) (x + xdim1 * (y))
#define OPS_ACC2(x, y) (x + xdim2 * (y))
#define OPS_ACC3(x, y) (x + xdim3 * (y))
#define OPS_ACC4(x, y) (x + xdim4 * (y))
#define OPS_ACC5(x, y) (x + xdim5 * (y))
#define OPS_ACC6(x, y) (x + xdim6 * (y))
#define OPS_ACC7(x, y) (x + xdim7 * (y))
#define OPS_ACC8(x, y) (x + xdim8 * (y))
#define OPS_ACC9(x, y) (x + xdim9 * (y))
#define OPS_ACC10(x, y) (x + xdim10 * (y))
#define OPS_ACC11(x, y) (x + xdim11 * (y))
#define OPS_ACC12(x, y) (x + xdim12 * (y))
#define OPS_ACC13(x, y) (x + xdim13 * (y))
#define OPS_ACC14(x, y) (x + xdim14 * (y))
#define OPS_ACC15(x, y) (x + xdim15 * (y))
#define OPS_ACC16(x, y) (x + xdim16 * (y))
#define OPS_ACC17(x, y) (x + xdim17 * (y))
#define OPS_ACC18(x, y) (x + xdim18 * (y))
#define OPS_ACC19(x, y) (x + xdim19 * (y))
#define OPS_ACC20(x, y) (x + xdim20 * (y))
#define OPS_ACC21(x, y) (x + xdim21 * (y))
#define OPS_ACC22(x, y) (x + xdim22 * (y))
#define OPS_ACC23(x, y) (x + xdim23 * (y))
#define OPS_ACC24(x, y) (x + xdim24 * (y))
#define OPS_ACC25(x, y) (x + xdim25 * (y))
#define OPS_ACC26(x, y) (x + xdim26 * (y))
#define OPS_ACC27(x, y) (x + xdim27 * (y))
#define OPS_ACC28(x, y) (x + xdim28 * (y))
#define OPS_ACC29(x, y) (x + xdim29 * (y))
#define OPS_ACC30(x, y) (x + xdim30 * (y))
#define OPS_ACC31(x, y) (x + xdim31 * (y))
#define OPS_ACC32(x, y) (x + xdim32 * (y))
#define OPS_ACC33(x, y) (x + xdim33 * (y))
#define OPS_ACC34(x, y) (x + xdim34 * (y))
#define OPS_ACC35(x, y) (x + xdim35 * (y))
#define OPS_ACC36(x, y) (x + xdim36 * (y))
#define OPS_ACC37(x, y) (x + xdim37 * (y))
#define OPS_ACC38(x, y) (x + xdim38 * (y))
#define OPS_ACC39(x, y) (x + xdim39 * (y))
#define OPS_ACC40(x, y) (x + xdim40 * (y))
#define OPS_ACC41(x, y) (x + xdim41 * (y))
#define OPS_ACC42(x, y) (x + xdim42 * (y))
#define OPS_ACC43(x, y) (x + xdim43 * (y))
#define OPS_ACC44(x, y) (x + xdim44 * (y))
#define OPS_ACC45(x, y) (x + xdim45 * (y))
#define OPS_ACC46(x, y) (x + xdim46 * (y))
#define OPS_ACC47(x, y) (x + xdim47 * (y))
#define OPS_ACC48(x, y) (x + xdim48 * (y))
#define OPS_ACC49(x, y) (x + xdim49 * (y))
#define OPS_ACC50(x, y) (x + xdim50 * (y))
#define OPS_ACC51(x, y) (x + xdim51 * (y))
#define OPS_ACC52(x, y) (x + xdim52 * (y))
#define OPS_ACC53(x, y) (x + xdim53 * (y))
#define OPS_ACC54(x, y) (x + xdim54 * (y))
#define OPS_ACC55(x, y) (x + xdim55 * (y))
#define OPS_ACC56(x, y) (x + xdim56 * (y))
#define OPS_ACC57(x, y) (x + xdim57 * (y))
#define OPS_ACC58(x, y) (x + xdim58 * (y))
#define OPS_ACC59(x, y) (x + xdim59 * (y))
#define OPS_ACC60(x, y) (x + xdim60 * (y))
#define OPS_ACC61(x, y) (x + xdim61 * (y))
#define OPS_ACC62(x, y) (x + xdim62 * (y))
#define OPS_ACC63(x, y) (x + xdim63 * (y))
#define OPS_ACC64(x, y) (x + xdim64 * (y))
#define OPS_ACC65(x, y) (x + xdim65 * (y))
#define OPS_ACC66(x, y) (x + xdim66 * (y))
#define OPS_ACC67(x, y) (x + xdim67 * (y))
#define OPS_ACC68(x, y) (x + xdim68 * (y))
#define OPS_ACC69(x, y) (x + xdim69 * (y))
#define OPS_ACC70(x, y) (x + xdim70 * (y))
#define OPS_ACC71(x, y) (x + xdim71 * (y))
#define OPS_ACC72(x, y) (x + xdim72 * (y))
#define OPS_ACC73(x, y) (x + xdim73 * (y))
#define OPS_ACC74(x, y) (x + xdim74 * (y))
#define OPS_ACC75(x, y) (x + xdim75 * (y))
#define OPS_ACC76(x, y) (x + xdim76 * (y))
#define OPS_ACC77(x, y) (x + xdim77 * (y))
#define OPS_ACC78(x, y) (x + xdim78 * (y))
#define OPS_ACC79(x, y) (x + xdim79 * (y))
#define OPS_ACC80(x, y) (x + xdim80 * (y))
#define OPS_ACC81(x, y) (x + xdim81 * (y))
#define OPS_ACC82(x, y) (x + xdim82 * (y))
#define OPS_ACC83(x, y) (x + xdim83 * (y))
#define OPS_ACC84(x, y) (x + xdim84 * (y))
#define OPS_ACC85(x, y) (x + xdim85 * (y))
#define OPS_ACC86(x, y) (x + xdim86 * (y))
#define OPS_ACC87(x, y) (x + xdim87 * (y))
#define OPS_ACC88(x, y) (x + xdim88 * (y))
#define OPS_ACC89(x, y) (x + xdim89 * (y))
#define OPS_ACC90(x, y) (x + xdim90 * (y))
#define OPS_ACC91(x, y) (x + xdim91 * (y))
#define OPS_ACC92(x, y) (x + xdim92 * (y))
#define OPS_ACC93(x, y) (x + xdim93 * (y))
#define OPS_ACC94(x, y) (x + xdim94 * (y))
#define OPS_ACC95(x, y) (x + xdim95 * (y))
#define OPS_ACC96(x, y) (x + xdim96 * (y))
#define OPS_ACC97(x, y) (x + xdim97 * (y))
#define OPS_ACC98(x, y) (x + xdim98 * (y))
#define OPS_ACC99(x, y) (x + xdim99 * (y))
#else // do debug checks
#define OPS_ACC0(x, y) (ops_stencil_check_2d(0, x, y, xdim0, -1))
#define OPS_ACC1(x, y) (ops_stencil_check_2d(1, x, y, xdim1, -1))
#define OPS_ACC2(x, y) (ops_stencil_check_2d(2, x, y, xdim2, -1))
#define OPS_ACC3(x, y) (ops_stencil_check_2d(3, x, y, xdim3, -1))
#define OPS_ACC4(x, y) (ops_stencil_check_2d(4, x, y, xdim4, -1))
#define OPS_ACC5(x, y) (ops_stencil_check_2d(5, x, y, xdim5, -1))
#define OPS_ACC6(x, y) (ops_stencil_check_2d(6, x, y, xdim6, -1))
#define OPS_ACC7(x, y) (ops_stencil_check_2d(7, x, y, xdim7, -1))
#define OPS_ACC8(x, y) (ops_stencil_check_2d(8, x, y, xdim8, -1))
#define OPS_ACC9(x, y) (ops_stencil_check_2d(9, x, y, xdim9, -1))
#define OPS_ACC10(x, y) (ops_stencil_check_2d(10, x, y, xdim10, -1))
#define OPS_ACC11(x, y) (ops_stencil_check_2d(11, x, y, xdim11, -1))
#define OPS_ACC12(x, y) (ops_stencil_check_2d(12, x, y, xdim12, -1))
#define OPS_ACC13(x, y) (ops_stencil_check_2d(13, x, y, xdim13, -1))
#define OPS_ACC14(x, y) (ops_stencil_check_2d(14, x, y, xdim14, -1))
#define OPS_ACC15(x, y) (ops_stencil_check_2d(15, x, y, xdim15, -1))
#define OPS_ACC16(x, y) (ops_stencil_check_2d(16, x, y, xdim16, -1))
#define OPS_ACC17(x, y) (ops_stencil_check_2d(17, x, y, xdim17, -1))
#define OPS_ACC18(x, y) (ops_stencil_check_2d(18, x, y, xdim18, -1))
#define OPS_ACC19(x, y) (ops_stencil_check_2d(19, x, y, xdim19, -1))
#define OPS_ACC20(x, y) (ops_stencil_check_2d(20, x, y, xdim20, -1))
#define OPS_ACC21(x, y) (ops_stencil_check_2d(21, x, y, xdim21, -1))
#define OPS_ACC22(x, y) (ops_stencil_check_2d(22, x, y, xdim22, -1))
#define OPS_ACC23(x, y) (ops_stencil_check_2d(23, x, y, xdim23, -1))
#define OPS_ACC24(x, y) (ops_stencil_check_2d(24, x, y, xdim24, -1))
#define OPS_ACC25(x, y) (ops_stencil_check_2d(25, x, y, xdim25, -1))
#define OPS_ACC26(x, y) (ops_stencil_check_2d(26, x, y, xdim26, -1))
#define OPS_ACC27(x, y) (ops_stencil_check_2d(27, x, y, xdim27, -1))
#define OPS_ACC28(x, y) (ops_stencil_check_2d(28, x, y, xdim28, -1))
#define OPS_ACC29(x, y) (ops_stencil_check_2d(29, x, y, xdim29, -1))
#define OPS_ACC30(x, y) (ops_stencil_check_2d(30, x, y, xdim30, -1))
#define OPS_ACC31(x, y) (ops_stencil_check_2d(31, x, y, xdim31, -1))
#define OPS_ACC32(x, y) (ops_stencil_check_2d(32, x, y, xdim32, -1))
#define OPS_ACC33(x, y) (ops_stencil_check_2d(33, x, y, xdim33, -1))
#define OPS_ACC34(x, y) (ops_stencil_check_2d(34, x, y, xdim34, -1))
#define OPS_ACC35(x, y) (ops_stencil_check_2d(35, x, y, xdim35, -1))
#define OPS_ACC36(x, y) (ops_stencil_check_2d(36, x, y, xdim36, -1))
#define OPS_ACC37(x, y) (ops_stencil_check_2d(37, x, y, xdim37, -1))
#define OPS_ACC38(x, y) (ops_stencil_check_2d(38, x, y, xdim38, -1))
#define OPS_ACC39(x, y) (ops_stencil_check_2d(39, x, y, xdim39, -1))
#define OPS_ACC40(x, y) (ops_stencil_check_2d(40, x, y, xdim40, -1))
#define OPS_ACC41(x, y) (ops_stencil_check_2d(41, x, y, xdim41, -1))
#define OPS_ACC42(x, y) (ops_stencil_check_2d(42, x, y, xdim42, -1))
#define OPS_ACC43(x, y) (ops_stencil_check_2d(43, x, y, xdim43, -1))
#define OPS_ACC44(x, y) (ops_stencil_check_2d(44, x, y, xdim44, -1))
#define OPS_ACC45(x, y) (ops_stencil_check_2d(45, x, y, xdim45, -1))
#define OPS_ACC46(x, y) (ops_stencil_check_2d(46, x, y, xdim46, -1))
#define OPS_ACC47(x, y) (ops_stencil_check_2d(47, x, y, xdim47, -1))
#define OPS_ACC48(x, y) (ops_stencil_check_2d(48, x, y, xdim48, -1))
#define OPS_ACC49(x, y) (ops_stencil_check_2d(49, x, y, xdim49, -1))
#define OPS_ACC50(x, y) (ops_stencil_check_2d(50, x, y, xdim50, -1))
#define OPS_ACC51(x, y) (ops_stencil_check_2d(51, x, y, xdim51, -1))
#define OPS_ACC52(x, y) (ops_stencil_check_2d(52, x, y, xdim52, -1))
#define OPS_ACC53(x, y) (ops_stencil_check_2d(53, x, y, xdim53, -1))
#define OPS_ACC54(x, y) (ops_stencil_check_2d(54, x, y, xdim54, -1))
#define OPS_ACC55(x, y) (ops_stencil_check_2d(55, x, y, xdim55, -1))
#define OPS_ACC56(x, y) (ops_stencil_check_2d(56, x, y, xdim56, -1))
#define OPS_ACC57(x, y) (ops_stencil_check_2d(57, x, y, xdim57, -1))
#define OPS_ACC58(x, y) (ops_stencil_check_2d(58, x, y, xdim58, -1))
#define OPS_ACC59(x, y) (ops_stencil_check_2d(59, x, y, xdim59, -1))
#define OPS_ACC60(x, y) (ops_stencil_check_2d(60, x, y, xdim60, -1))
#define OPS_ACC61(x, y) (ops_stencil_check_2d(61, x, y, xdim61, -1))
#define OPS_ACC62(x, y) (ops_stencil_check_2d(62, x, y, xdim62, -1))
#define OPS_ACC63(x, y) (ops_stencil_check_2d(63, x, y, xdim63, -1))
#define OPS_ACC64(x, y) (ops_stencil_check_2d(64, x, y, xdim64, -1))
#define OPS_ACC65(x, y) (ops_stencil_check_2d(65, x, y, xdim65, -1))
#define OPS_ACC66(x, y) (ops_stencil_check_2d(66, x, y, xdim66, -1))
#define OPS_ACC67(x, y) (ops_stencil_check_2d(67, x, y, xdim67, -1))
#define OPS_ACC68(x, y) (ops_stencil_check_2d(68, x, y, xdim68, -1))
#define OPS_ACC69(x, y) (ops_stencil_check_2d(69, x, y, xdim69, -1))
#define OPS_ACC70(x, y) (ops_stencil_check_2d(70, x, y, xdim70, -1))
#define OPS_ACC71(x, y) (ops_stencil_check_2d(71, x, y, xdim71, -1))
#define OPS_ACC72(x, y) (ops_stencil_check_2d(72, x, y, xdim72, -1))
#define OPS_ACC73(x, y) (ops_stencil_check_2d(73, x, y, xdim73, -1))
#define OPS_ACC74(x, y) (ops_stencil_check_2d(74, x, y, xdim74, -1))
#define OPS_ACC75(x, y) (ops_stencil_check_2d(75, x, y, xdim75, -1))
#define OPS_ACC76(x, y) (ops_stencil_check_2d(76, x, y, xdim76, -1))
#define OPS_ACC77(x, y) (ops_stencil_check_2d(77, x, y, xdim77, -1))
#define OPS_ACC78(x, y) (ops_stencil_check_2d(78, x, y, xdim78, -1))
#define OPS_ACC79(x, y) (ops_stencil_check_2d(79, x, y, xdim79, -1))
#define OPS_ACC80(x, y) (ops_stencil_check_2d(80, x, y, xdim80, -1))
#define OPS_ACC81(x, y) (ops_stencil_check_2d(81, x, y, xdim81, -1))
#define OPS_ACC82(x, y) (ops_stencil_check_2d(82, x, y, xdim82, -1))
#define OPS_ACC83(x, y) (ops_stencil_check_2d(83, x, y, xdim83, -1))
#define OPS_ACC84(x, y) (ops_stencil_check_2d(84, x, y, xdim84, -1))
#define OPS_ACC85(x, y) (ops_stencil_check_2d(85, x, y, xdim85, -1))
#define OPS_ACC86(x, y) (ops_stencil_check_2d(86, x, y, xdim86, -1))
#define OPS_ACC87(x, y) (ops_stencil_check_2d(87, x, y, xdim87, -1))
#define OPS_ACC88(x, y) (ops_stencil_check_2d(88, x, y, xdim88, -1))
#define OPS_ACC89(x, y) (ops_stencil_check_2d(89, x, y, xdim89, -1))
#define OPS_ACC90(x, y) (ops_stencil_check_2d(90, x, y, xdim90, -1))
#define OPS_ACC91(x, y) (ops_stencil_check_2d(91, x, y, xdim91, -1))
#define OPS_ACC92(x, y) (ops_stencil_check_2d(92, x, y, xdim92, -1))
#define OPS_ACC93(x, y) (ops_stencil_check_2d(93, x, y, xdim93, -1))
#define OPS_ACC94(x, y) (ops_stencil_check_2d(94, x, y, xdim94, -1))
#define OPS_ACC95(x, y) (ops_stencil_check_2d(95, x, y, xdim95, -1))
#define OPS_ACC96(x, y) (ops_stencil_check_2d(96, x, y, xdim96, -1))
#define OPS_ACC97(x, y) (ops_stencil_check_2d(97, x, y, xdim97, -1))
#define OPS_ACC98(x, y) (ops_stencil_check_2d(98, x, y, xdim98, -1))
#define OPS_ACC99(x, y) (ops_stencil_check_2d(99, x, y, xdim99, -1))
#endif            // end debug checks
#else             // macros for 1D application
#ifndef OPS_DEBUG // no debug checks
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
#define OPS_ACC18(x) (x)
#define OPS_ACC19(x) (x)
#define OPS_ACC20(x) (x)
#define OPS_ACC21(x) (x)
#define OPS_ACC22(x) (x)
#define OPS_ACC23(x) (x)
#define OPS_ACC24(x) (x)
#define OPS_ACC25(x) (x)
#define OPS_ACC26(x) (x)
#define OPS_ACC27(x) (x)
#define OPS_ACC28(x) (x)
#define OPS_ACC29(x) (x)
#define OPS_ACC30(x) (x)
#define OPS_ACC31(x) (x)
#define OPS_ACC32(x) (x)
#define OPS_ACC33(x) (x)
#define OPS_ACC34(x) (x)
#define OPS_ACC35(x) (x)
#define OPS_ACC36(x) (x)
#define OPS_ACC37(x) (x)
#define OPS_ACC38(x) (x)
#define OPS_ACC39(x) (x)
#define OPS_ACC40(x) (x)
#define OPS_ACC41(x) (x)
#define OPS_ACC42(x) (x)
#define OPS_ACC43(x) (x)
#define OPS_ACC44(x) (x)
#define OPS_ACC45(x) (x)
#define OPS_ACC46(x) (x)
#define OPS_ACC47(x) (x)
#define OPS_ACC48(x) (x)
#define OPS_ACC49(x) (x)
#define OPS_ACC50(x) (x)
#define OPS_ACC51(x) (x)
#define OPS_ACC52(x) (x)
#define OPS_ACC53(x) (x)
#define OPS_ACC54(x) (x)
#define OPS_ACC55(x) (x)
#define OPS_ACC56(x) (x)
#define OPS_ACC57(x) (x)
#define OPS_ACC58(x) (x)
#define OPS_ACC59(x) (x)
#define OPS_ACC60(x) (x)
#define OPS_ACC61(x) (x)
#define OPS_ACC62(x) (x)
#define OPS_ACC63(x) (x)
#define OPS_ACC64(x) (x)
#define OPS_ACC65(x) (x)
#define OPS_ACC66(x) (x)
#define OPS_ACC67(x) (x)
#define OPS_ACC68(x) (x)
#define OPS_ACC69(x) (x)
#define OPS_ACC70(x) (x)
#define OPS_ACC71(x) (x)
#define OPS_ACC72(x) (x)
#define OPS_ACC73(x) (x)
#define OPS_ACC74(x) (x)
#define OPS_ACC75(x) (x)
#define OPS_ACC76(x) (x)
#define OPS_ACC77(x) (x)
#define OPS_ACC78(x) (x)
#define OPS_ACC79(x) (x)
#define OPS_ACC80(x) (x)
#define OPS_ACC81(x) (x)
#define OPS_ACC82(x) (x)
#define OPS_ACC83(x) (x)
#define OPS_ACC84(x) (x)
#define OPS_ACC85(x) (x)
#define OPS_ACC86(x) (x)
#define OPS_ACC87(x) (x)
#define OPS_ACC88(x) (x)
#define OPS_ACC89(x) (x)
#define OPS_ACC90(x) (x)
#define OPS_ACC91(x) (x)
#define OPS_ACC92(x) (x)
#define OPS_ACC93(x) (x)
#define OPS_ACC94(x) (x)
#define OPS_ACC95(x) (x)
#define OPS_ACC96(x) (x)
#define OPS_ACC97(x) (x)
#define OPS_ACC98(x) (x)
#define OPS_ACC99(x) (x)
#else // do debug checks
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
#define OPS_ACC18(x) (ops_stencil_check_1d(18, x, xdim18))
#define OPS_ACC19(x) (ops_stencil_check_1d(19, x, xdim19))
#define OPS_ACC20(x) (ops_stencil_check_1d(20, x, xdim20))
#define OPS_ACC21(x) (ops_stencil_check_1d(21, x, xdim21))
#define OPS_ACC22(x) (ops_stencil_check_1d(22, x, xdim22))
#define OPS_ACC23(x) (ops_stencil_check_1d(23, x, xdim23))
#define OPS_ACC24(x) (ops_stencil_check_1d(24, x, xdim24))
#define OPS_ACC25(x) (ops_stencil_check_1d(25, x, xdim25))
#define OPS_ACC26(x) (ops_stencil_check_1d(26, x, xdim26))
#define OPS_ACC27(x) (ops_stencil_check_1d(27, x, xdim27))
#define OPS_ACC28(x) (ops_stencil_check_1d(28, x, xdim28))
#define OPS_ACC29(x) (ops_stencil_check_1d(29, x, xdim29))
#define OPS_ACC30(x) (ops_stencil_check_1d(30, x, xdim30))
#define OPS_ACC31(x) (ops_stencil_check_1d(31, x, xdim31))
#define OPS_ACC32(x) (ops_stencil_check_1d(32, x, xdim32))
#define OPS_ACC33(x) (ops_stencil_check_1d(33, x, xdim33))
#define OPS_ACC34(x) (ops_stencil_check_1d(34, x, xdim34))
#define OPS_ACC35(x) (ops_stencil_check_1d(35, x, xdim35))
#define OPS_ACC36(x) (ops_stencil_check_1d(36, x, xdim36))
#define OPS_ACC37(x) (ops_stencil_check_1d(37, x, xdim37))
#define OPS_ACC38(x) (ops_stencil_check_1d(38, x, xdim38))
#define OPS_ACC39(x) (ops_stencil_check_1d(39, x, xdim39))
#define OPS_ACC40(x) (ops_stencil_check_1d(40, x, xdim40))
#define OPS_ACC41(x) (ops_stencil_check_1d(41, x, xdim41))
#define OPS_ACC42(x) (ops_stencil_check_1d(42, x, xdim42))
#define OPS_ACC43(x) (ops_stencil_check_1d(43, x, xdim43))
#define OPS_ACC44(x) (ops_stencil_check_1d(44, x, xdim44))
#define OPS_ACC45(x) (ops_stencil_check_1d(45, x, xdim45))
#define OPS_ACC46(x) (ops_stencil_check_1d(46, x, xdim46))
#define OPS_ACC47(x) (ops_stencil_check_1d(47, x, xdim47))
#define OPS_ACC48(x) (ops_stencil_check_1d(48, x, xdim48))
#define OPS_ACC49(x) (ops_stencil_check_1d(49, x, xdim49))
#define OPS_ACC50(x) (ops_stencil_check_1d(50, x, xdim50))
#define OPS_ACC51(x) (ops_stencil_check_1d(51, x, xdim51))
#define OPS_ACC52(x) (ops_stencil_check_1d(52, x, xdim52))
#define OPS_ACC53(x) (ops_stencil_check_1d(53, x, xdim53))
#define OPS_ACC54(x) (ops_stencil_check_1d(54, x, xdim54))
#define OPS_ACC55(x) (ops_stencil_check_1d(55, x, xdim55))
#define OPS_ACC56(x) (ops_stencil_check_1d(56, x, xdim56))
#define OPS_ACC57(x) (ops_stencil_check_1d(57, x, xdim57))
#define OPS_ACC58(x) (ops_stencil_check_1d(58, x, xdim58))
#define OPS_ACC59(x) (ops_stencil_check_1d(59, x, xdim59))
#define OPS_ACC60(x) (ops_stencil_check_1d(60, x, xdim60))
#define OPS_ACC61(x) (ops_stencil_check_1d(61, x, xdim61))
#define OPS_ACC62(x) (ops_stencil_check_1d(62, x, xdim62))
#define OPS_ACC63(x) (ops_stencil_check_1d(63, x, xdim63))
#define OPS_ACC64(x) (ops_stencil_check_1d(64, x, xdim64))
#define OPS_ACC65(x) (ops_stencil_check_1d(65, x, xdim65))
#define OPS_ACC66(x) (ops_stencil_check_1d(66, x, xdim66))
#define OPS_ACC67(x) (ops_stencil_check_1d(67, x, xdim67))
#define OPS_ACC68(x) (ops_stencil_check_1d(68, x, xdim68))
#define OPS_ACC69(x) (ops_stencil_check_1d(69, x, xdim69))
#define OPS_ACC70(x) (ops_stencil_check_1d(70, x, xdim70))
#define OPS_ACC71(x) (ops_stencil_check_1d(71, x, xdim71))
#define OPS_ACC72(x) (ops_stencil_check_1d(72, x, xdim72))
#define OPS_ACC73(x) (ops_stencil_check_1d(73, x, xdim73))
#define OPS_ACC74(x) (ops_stencil_check_1d(74, x, xdim74))
#define OPS_ACC75(x) (ops_stencil_check_1d(75, x, xdim75))
#define OPS_ACC76(x) (ops_stencil_check_1d(76, x, xdim76))
#define OPS_ACC77(x) (ops_stencil_check_1d(77, x, xdim77))
#define OPS_ACC78(x) (ops_stencil_check_1d(78, x, xdim78))
#define OPS_ACC79(x) (ops_stencil_check_1d(79, x, xdim79))
#define OPS_ACC80(x) (ops_stencil_check_1d(80, x, xdim80))
#define OPS_ACC81(x) (ops_stencil_check_1d(81, x, xdim81))
#define OPS_ACC82(x) (ops_stencil_check_1d(82, x, xdim82))
#define OPS_ACC83(x) (ops_stencil_check_1d(83, x, xdim83))
#define OPS_ACC84(x) (ops_stencil_check_1d(84, x, xdim84))
#define OPS_ACC85(x) (ops_stencil_check_1d(85, x, xdim85))
#define OPS_ACC86(x) (ops_stencil_check_1d(86, x, xdim86))
#define OPS_ACC87(x) (ops_stencil_check_1d(87, x, xdim87))
#define OPS_ACC88(x) (ops_stencil_check_1d(88, x, xdim88))
#define OPS_ACC89(x) (ops_stencil_check_1d(89, x, xdim89))
#define OPS_ACC90(x) (ops_stencil_check_1d(90, x, xdim90))
#define OPS_ACC91(x) (ops_stencil_check_1d(91, x, xdim91))
#define OPS_ACC92(x) (ops_stencil_check_1d(92, x, xdim92))
#define OPS_ACC93(x) (ops_stencil_check_1d(93, x, xdim93))
#define OPS_ACC94(x) (ops_stencil_check_1d(94, x, xdim94))
#define OPS_ACC95(x) (ops_stencil_check_1d(95, x, xdim95))
#define OPS_ACC96(x) (ops_stencil_check_1d(96, x, xdim96))
#define OPS_ACC97(x) (ops_stencil_check_1d(97, x, xdim97))
#define OPS_ACC98(x) (ops_stencil_check_1d(98, x, xdim98))
#define OPS_ACC99(x) (ops_stencil_check_1d(99, x, xdim99))
#endif // end debug checks
#endif // end macros for 1D application
#endif // end OPS_ACC_MACROS

/**---------Multi-D ops_dats macros (multiple elements per grid point)-------**/
#ifndef OPS_ACC_MD_MACROS
#define OPS_ACC_MACROS
#ifdef OPS_3D     // macros for 3D application
#ifndef OPS_DEBUG // no debug checks
#ifndef OPS_SOA
#define OPS_ACC_MD0(d, x, y, z)                                                \
  ((x)*multi_d0 + (d) + (xdim0 * (y)*multi_d0) + (xdim0 * ydim0 * (z)*multi_d0))
#define OPS_ACC_MD1(d, x, y, z)                                                \
  ((x)*multi_d1 + (d) + (xdim1 * (y)*multi_d1) + (xdim1 * ydim1 * (z)*multi_d1))
#define OPS_ACC_MD2(d, x, y, z)                                                \
  ((x)*multi_d2 + (d) + (xdim2 * (y)*multi_d2) + (xdim2 * ydim2 * (z)*multi_d2))
#define OPS_ACC_MD3(d, x, y, z)                                                \
  ((x)*multi_d3 + (d) + (xdim3 * (y)*multi_d3) + (xdim3 * ydim3 * (z)*multi_d3))
#define OPS_ACC_MD4(d, x, y, z)                                                \
  ((x)*multi_d4 + (d) + (xdim4 * (y)*multi_d4) + (xdim4 * ydim4 * (z)*multi_d4))
#define OPS_ACC_MD5(d, x, y, z)                                                \
  ((x)*multi_d5 + (d) + (xdim5 * (y)*multi_d5) + (xdim5 * ydim5 * (z)*multi_d5))
#define OPS_ACC_MD6(d, x, y, z)                                                \
  ((x)*multi_d6 + (d) + (xdim6 * (y)*multi_d6) + (xdim6 * ydim6 * (z)*multi_d6))
#define OPS_ACC_MD7(d, x, y, z)                                                \
  ((x)*multi_d7 + (d) + (xdim7 * (y)*multi_d7) + (xdim7 * ydim7 * (z)*multi_d7))
#define OPS_ACC_MD8(d, x, y, z)                                                \
  ((x)*multi_d8 + (d) + (xdim8 * (y)*multi_d8) + (xdim8 * ydim8 * (z)*multi_d8))
#define OPS_ACC_MD9(d, x, y, z)                                                \
  ((x)*multi_d9 + (d) + (xdim9 * (y)*multi_d9) + (xdim9 * ydim9 * (z)*multi_d9))
#define OPS_ACC_MD10(d, x, y, z)                                               \
  ((x)*multi_d10 + (d) + (xdim10 * (y)*multi_d10) +                            \
   (xdim10 * ydim10 * (z)*multi_d10))
#define OPS_ACC_MD11(d, x, y, z)                                               \
  ((x)*multi_d11 + (d) + (xdim11 * (y)*multi_d11) +                            \
   (xdim11 * ydim11 * (z)*multi_d11))
#define OPS_ACC_MD12(d, x, y, z)                                               \
  ((x)*multi_d12 + (d) + (xdim12 * (y)*multi_d12) +                            \
   (xdim12 * ydim12 * (z)*multi_d12))
#define OPS_ACC_MD13(d, x, y, z)                                               \
  ((x)*multi_d13 + (d) + (xdim13 * (y)*multi_d13) +                            \
   (xdim13 * ydim13 * (z)*multi_d13))
#define OPS_ACC_MD14(d, x, y, z)                                               \
  ((x)*multi_d14 + (d) + (xdim14 * (y)*multi_d14) +                            \
   (xdim14 * ydim14 * (z)*multi_d14))
#define OPS_ACC_MD15(d, x, y, z)                                               \
  ((x)*multi_d15 + (d) + (xdim15 * (y)*multi_d15) +                            \
   (xdim15 * ydim15 * (z)*multi_d15))
#define OPS_ACC_MD16(d, x, y, z)                                               \
  ((x)*multi_d16 + (d) + (xdim16 * (y)*multi_d16) +                            \
   (xdim16 * ydim16 * (z)*multi_d16))
#define OPS_ACC_MD17(d, x, y, z)                                               \
  ((x)*multi_d17 + (d) + (xdim17 * (y)*multi_d17) +                            \
   (xdim17 * ydim17 * (z)*multi_d17))
#define OPS_ACC_MD18(d, x, y, z)                                               \
  ((x)*multi_d18 + (d) + (xdim18 * (y)*multi_d18) +                            \
   (xdim18 * ydim18 * (z)*multi_d18))
#define OPS_ACC_MD19(d, x, y, z)                                               \
  ((x)*multi_d19 + (d) + (xdim19 * (y)*multi_d19) +                            \
   (xdim19 * ydim19 * (z)*multi_d19))
#define OPS_ACC_MD20(d, x, y, z)                                               \
  ((x)*multi_d20 + (d) + (xdim20 * (y)*multi_d20) +                            \
   (xdim20 * ydim20 * (z)*multi_d20))
#define OPS_ACC_MD21(d, x, y, z)                                               \
  ((x)*multi_d21 + (d) + (xdim21 * (y)*multi_d21) +                            \
   (xdim21 * ydim21 * (z)*multi_d21))
#define OPS_ACC_MD22(d, x, y, z)                                               \
  ((x)*multi_d22 + (d) + (xdim22 * (y)*multi_d22) +                            \
   (xdim22 * ydim22 * (z)*multi_d22))
#define OPS_ACC_MD23(d, x, y, z)                                               \
  ((x)*multi_d23 + (d) + (xdim23 * (y)*multi_d23) +                            \
   (xdim23 * ydim23 * (z)*multi_d23))
#define OPS_ACC_MD24(d, x, y, z)                                               \
  ((x)*multi_d24 + (d) + (xdim24 * (y)*multi_d24) +                            \
   (xdim24 * ydim24 * (z)*multi_d24))
#define OPS_ACC_MD25(d, x, y, z)                                               \
  ((x)*multi_d25 + (d) + (xdim25 * (y)*multi_d25) +                            \
   (xdim25 * ydim25 * (z)*multi_d25))
#define OPS_ACC_MD26(d, x, y, z)                                               \
  ((x)*multi_d26 + (d) + (xdim26 * (y)*multi_d26) +                            \
   (xdim26 * ydim26 * (z)*multi_d26))
#define OPS_ACC_MD27(d, x, y, z)                                               \
  ((x)*multi_d27 + (d) + (xdim27 * (y)*multi_d27) +                            \
   (xdim27 * ydim27 * (z)*multi_d27))
#define OPS_ACC_MD28(d, x, y, z)                                               \
  ((x)*multi_d28 + (d) + (xdim28 * (y)*multi_d28) +                            \
   (xdim28 * ydim28 * (z)*multi_d28))
#define OPS_ACC_MD29(d, x, y, z)                                               \
  ((x)*multi_d29 + (d) + (xdim29 * (y)*multi_d29) +                            \
   (xdim29 * ydim29 * (z)*multi_d29))
#define OPS_ACC_MD30(d, x, y, z)                                               \
  ((x)*multi_d30 + (d) + (xdim30 * (y)*multi_d30) +                            \
   (xdim30 * ydim30 * (z)*multi_d30))
#define OPS_ACC_MD31(d, x, y, z)                                               \
  ((x)*multi_d31 + (d) + (xdim31 * (y)*multi_d31) +                            \
   (xdim31 * ydim31 * (z)*multi_d31))
#define OPS_ACC_MD32(d, x, y, z)                                               \
  ((x)*multi_d32 + (d) + (xdim32 * (y)*multi_d32) +                            \
   (xdim32 * ydim32 * (z)*multi_d32))
#define OPS_ACC_MD33(d, x, y, z)                                               \
  ((x)*multi_d33 + (d) + (xdim33 * (y)*multi_d33) +                            \
   (xdim33 * ydim33 * (z)*multi_d33))
#define OPS_ACC_MD34(d, x, y, z)                                               \
  ((x)*multi_d34 + (d) + (xdim34 * (y)*multi_d34) +                            \
   (xdim34 * ydim34 * (z)*multi_d34))
#define OPS_ACC_MD35(d, x, y, z)                                               \
  ((x)*multi_d35 + (d) + (xdim35 * (y)*multi_d35) +                            \
   (xdim35 * ydim35 * (z)*multi_d35))
#define OPS_ACC_MD36(d, x, y, z)                                               \
  ((x)*multi_d36 + (d) + (xdim36 * (y)*multi_d36) +                            \
   (xdim36 * ydim36 * (z)*multi_d36))
#define OPS_ACC_MD37(d, x, y, z)                                               \
  ((x)*multi_d37 + (d) + (xdim37 * (y)*multi_d37) +                            \
   (xdim37 * ydim37 * (z)*multi_d37))
#define OPS_ACC_MD38(d, x, y, z)                                               \
  ((x)*multi_d38 + (d) + (xdim38 * (y)*multi_d38) +                            \
   (xdim38 * ydim38 * (z)*multi_d38))
#define OPS_ACC_MD39(d, x, y, z)                                               \
  ((x)*multi_d39 + (d) + (xdim39 * (y)*multi_d39) +                            \
   (xdim39 * ydim39 * (z)*multi_d39))
#define OPS_ACC_MD40(d, x, y, z)                                               \
  ((x)*multi_d40 + (d) + (xdim40 * (y)*multi_d40) +                            \
   (xdim40 * ydim40 * (z)*multi_d40))
#define OPS_ACC_MD41(d, x, y, z)                                               \
  ((x)*multi_d41 + (d) + (xdim41 * (y)*multi_d41) +                            \
   (xdim41 * ydim41 * (z)*multi_d41))
#define OPS_ACC_MD42(d, x, y, z)                                               \
  ((x)*multi_d42 + (d) + (xdim42 * (y)*multi_d42) +                            \
   (xdim42 * ydim42 * (z)*multi_d42))
#define OPS_ACC_MD43(d, x, y, z)                                               \
  ((x)*multi_d43 + (d) + (xdim43 * (y)*multi_d43) +                            \
   (xdim43 * ydim43 * (z)*multi_d43))
#define OPS_ACC_MD44(d, x, y, z)                                               \
  ((x)*multi_d44 + (d) + (xdim44 * (y)*multi_d44) +                            \
   (xdim44 * ydim44 * (z)*multi_d44))
#define OPS_ACC_MD45(d, x, y, z)                                               \
  ((x)*multi_d45 + (d) + (xdim45 * (y)*multi_d45) +                            \
   (xdim45 * ydim45 * (z)*multi_d45))
#define OPS_ACC_MD46(d, x, y, z)                                               \
  ((x)*multi_d46 + (d) + (xdim46 * (y)*multi_d46) +                            \
   (xdim46 * ydim46 * (z)*multi_d46))
#define OPS_ACC_MD47(d, x, y, z)                                               \
  ((x)*multi_d47 + (d) + (xdim47 * (y)*multi_d47) +                            \
   (xdim47 * ydim47 * (z)*multi_d47))
#define OPS_ACC_MD48(d, x, y, z)                                               \
  ((x)*multi_d48 + (d) + (xdim48 * (y)*multi_d48) +                            \
   (xdim48 * ydim48 * (z)*multi_d48))
#define OPS_ACC_MD49(d, x, y, z)                                               \
  ((x)*multi_d49 + (d) + (xdim49 * (y)*multi_d49) +                            \
   (xdim49 * ydim49 * (z)*multi_d49))
#define OPS_ACC_MD50(d, x, y, z)                                               \
  ((x)*multi_d50 + (d) + (xdim50 * (y)*multi_d50) +                            \
   (xdim50 * ydim50 * (z)*multi_d50))
#define OPS_ACC_MD51(d, x, y, z)                                               \
  ((x)*multi_d51 + (d) + (xdim51 * (y)*multi_d51) +                            \
   (xdim51 * ydim51 * (z)*multi_d51))
#define OPS_ACC_MD52(d, x, y, z)                                               \
  ((x)*multi_d52 + (d) + (xdim52 * (y)*multi_d52) +                            \
   (xdim52 * ydim52 * (z)*multi_d52))
#define OPS_ACC_MD53(d, x, y, z)                                               \
  ((x)*multi_d53 + (d) + (xdim53 * (y)*multi_d53) +                            \
   (xdim53 * ydim53 * (z)*multi_d53))
#define OPS_ACC_MD54(d, x, y, z)                                               \
  ((x)*multi_d54 + (d) + (xdim54 * (y)*multi_d54) +                            \
   (xdim54 * ydim54 * (z)*multi_d54))
#define OPS_ACC_MD55(d, x, y, z)                                               \
  ((x)*multi_d55 + (d) + (xdim55 * (y)*multi_d55) +                            \
   (xdim55 * ydim55 * (z)*multi_d55))
#define OPS_ACC_MD56(d, x, y, z)                                               \
  ((x)*multi_d56 + (d) + (xdim56 * (y)*multi_d56) +                            \
   (xdim56 * ydim56 * (z)*multi_d56))
#define OPS_ACC_MD57(d, x, y, z)                                               \
  ((x)*multi_d57 + (d) + (xdim57 * (y)*multi_d57) +                            \
   (xdim57 * ydim57 * (z)*multi_d57))
#define OPS_ACC_MD58(d, x, y, z)                                               \
  ((x)*multi_d58 + (d) + (xdim58 * (y)*multi_d58) +                            \
   (xdim58 * ydim58 * (z)*multi_d58))
#define OPS_ACC_MD59(d, x, y, z)                                               \
  ((x)*multi_d59 + (d) + (xdim59 * (y)*multi_d59) +                            \
   (xdim59 * ydim59 * (z)*multi_d59))
#define OPS_ACC_MD60(d, x, y, z)                                               \
  ((x)*multi_d60 + (d) + (xdim60 * (y)*multi_d60) +                            \
   (xdim60 * ydim60 * (z)*multi_d60))
#define OPS_ACC_MD61(d, x, y, z)                                               \
  ((x)*multi_d61 + (d) + (xdim61 * (y)*multi_d61) +                            \
   (xdim61 * ydim61 * (z)*multi_d61))
#define OPS_ACC_MD62(d, x, y, z)                                               \
  ((x)*multi_d62 + (d) + (xdim62 * (y)*multi_d62) +                            \
   (xdim62 * ydim62 * (z)*multi_d62))
#define OPS_ACC_MD63(d, x, y, z)                                               \
  ((x)*multi_d63 + (d) + (xdim63 * (y)*multi_d63) +                            \
   (xdim63 * ydim63 * (z)*multi_d63))
#define OPS_ACC_MD64(d, x, y, z)                                               \
  ((x)*multi_d64 + (d) + (xdim64 * (y)*multi_d64) +                            \
   (xdim64 * ydim64 * (z)*multi_d64))
#define OPS_ACC_MD65(d, x, y, z)                                               \
  ((x)*multi_d65 + (d) + (xdim65 * (y)*multi_d65) +                            \
   (xdim65 * ydim65 * (z)*multi_d65))
#define OPS_ACC_MD66(d, x, y, z)                                               \
  ((x)*multi_d66 + (d) + (xdim66 * (y)*multi_d66) +                            \
   (xdim66 * ydim66 * (z)*multi_d66))
#define OPS_ACC_MD67(d, x, y, z)                                               \
  ((x)*multi_d67 + (d) + (xdim67 * (y)*multi_d67) +                            \
   (xdim67 * ydim67 * (z)*multi_d67))
#define OPS_ACC_MD68(d, x, y, z)                                               \
  ((x)*multi_d68 + (d) + (xdim68 * (y)*multi_d68) +                            \
   (xdim68 * ydim68 * (z)*multi_d68))
#define OPS_ACC_MD69(d, x, y, z)                                               \
  ((x)*multi_d69 + (d) + (xdim69 * (y)*multi_d69) +                            \
   (xdim69 * ydim69 * (z)*multi_d69))
#define OPS_ACC_MD70(d, x, y, z)                                               \
  ((x)*multi_d70 + (d) + (xdim70 * (y)*multi_d70) +                            \
   (xdim70 * ydim70 * (z)*multi_d70))
#define OPS_ACC_MD71(d, x, y, z)                                               \
  ((x)*multi_d71 + (d) + (xdim71 * (y)*multi_d71) +                            \
   (xdim71 * ydim71 * (z)*multi_d71))
#define OPS_ACC_MD72(d, x, y, z)                                               \
  ((x)*multi_d72 + (d) + (xdim72 * (y)*multi_d72) +                            \
   (xdim72 * ydim72 * (z)*multi_d72))
#define OPS_ACC_MD73(d, x, y, z)                                               \
  ((x)*multi_d73 + (d) + (xdim73 * (y)*multi_d73) +                            \
   (xdim73 * ydim73 * (z)*multi_d73))
#define OPS_ACC_MD74(d, x, y, z)                                               \
  ((x)*multi_d74 + (d) + (xdim74 * (y)*multi_d74) +                            \
   (xdim74 * ydim74 * (z)*multi_d74))
#define OPS_ACC_MD75(d, x, y, z)                                               \
  ((x)*multi_d75 + (d) + (xdim75 * (y)*multi_d75) +                            \
   (xdim75 * ydim75 * (z)*multi_d75))
#define OPS_ACC_MD76(d, x, y, z)                                               \
  ((x)*multi_d76 + (d) + (xdim76 * (y)*multi_d76) +                            \
   (xdim76 * ydim76 * (z)*multi_d76))
#define OPS_ACC_MD77(d, x, y, z)                                               \
  ((x)*multi_d77 + (d) + (xdim77 * (y)*multi_d77) +                            \
   (xdim77 * ydim77 * (z)*multi_d77))
#define OPS_ACC_MD78(d, x, y, z)                                               \
  ((x)*multi_d78 + (d) + (xdim78 * (y)*multi_d78) +                            \
   (xdim78 * ydim78 * (z)*multi_d78))
#define OPS_ACC_MD79(d, x, y, z)                                               \
  ((x)*multi_d79 + (d) + (xdim79 * (y)*multi_d79) +                            \
   (xdim79 * ydim79 * (z)*multi_d79))
#define OPS_ACC_MD80(d, x, y, z)                                               \
  ((x)*multi_d80 + (d) + (xdim80 * (y)*multi_d80) +                            \
   (xdim80 * ydim80 * (z)*multi_d80))
#define OPS_ACC_MD81(d, x, y, z)                                               \
  ((x)*multi_d81 + (d) + (xdim81 * (y)*multi_d81) +                            \
   (xdim81 * ydim81 * (z)*multi_d81))
#define OPS_ACC_MD82(d, x, y, z)                                               \
  ((x)*multi_d82 + (d) + (xdim82 * (y)*multi_d82) +                            \
   (xdim82 * ydim82 * (z)*multi_d82))
#define OPS_ACC_MD83(d, x, y, z)                                               \
  ((x)*multi_d83 + (d) + (xdim83 * (y)*multi_d83) +                            \
   (xdim83 * ydim83 * (z)*multi_d83))
#define OPS_ACC_MD84(d, x, y, z)                                               \
  ((x)*multi_d84 + (d) + (xdim84 * (y)*multi_d84) +                            \
   (xdim84 * ydim84 * (z)*multi_d84))
#define OPS_ACC_MD85(d, x, y, z)                                               \
  ((x)*multi_d85 + (d) + (xdim85 * (y)*multi_d85) +                            \
   (xdim85 * ydim85 * (z)*multi_d85))
#define OPS_ACC_MD86(d, x, y, z)                                               \
  ((x)*multi_d86 + (d) + (xdim86 * (y)*multi_d86) +                            \
   (xdim86 * ydim86 * (z)*multi_d86))
#define OPS_ACC_MD87(d, x, y, z)                                               \
  ((x)*multi_d87 + (d) + (xdim87 * (y)*multi_d87) +                            \
   (xdim87 * ydim87 * (z)*multi_d87))
#define OPS_ACC_MD88(d, x, y, z)                                               \
  ((x)*multi_d88 + (d) + (xdim88 * (y)*multi_d88) +                            \
   (xdim88 * ydim88 * (z)*multi_d88))
#define OPS_ACC_MD89(d, x, y, z)                                               \
  ((x)*multi_d89 + (d) + (xdim89 * (y)*multi_d89) +                            \
   (xdim89 * ydim89 * (z)*multi_d89))
#define OPS_ACC_MD90(d, x, y, z)                                               \
  ((x)*multi_d90 + (d) + (xdim90 * (y)*multi_d90) +                            \
   (xdim90 * ydim90 * (z)*multi_d90))
#define OPS_ACC_MD91(d, x, y, z)                                               \
  ((x)*multi_d91 + (d) + (xdim91 * (y)*multi_d91) +                            \
   (xdim91 * ydim91 * (z)*multi_d91))
#define OPS_ACC_MD92(d, x, y, z)                                               \
  ((x)*multi_d92 + (d) + (xdim92 * (y)*multi_d92) +                            \
   (xdim92 * ydim92 * (z)*multi_d92))
#define OPS_ACC_MD93(d, x, y, z)                                               \
  ((x)*multi_d93 + (d) + (xdim93 * (y)*multi_d93) +                            \
   (xdim93 * ydim93 * (z)*multi_d93))
#define OPS_ACC_MD94(d, x, y, z)                                               \
  ((x)*multi_d94 + (d) + (xdim94 * (y)*multi_d94) +                            \
   (xdim94 * ydim94 * (z)*multi_d94))
#define OPS_ACC_MD95(d, x, y, z)                                               \
  ((x)*multi_d95 + (d) + (xdim95 * (y)*multi_d95) +                            \
   (xdim95 * ydim95 * (z)*multi_d95))
#define OPS_ACC_MD96(d, x, y, z)                                               \
  ((x)*multi_d96 + (d) + (xdim96 * (y)*multi_d96) +                            \
   (xdim96 * ydim96 * (z)*multi_d96))
#define OPS_ACC_MD97(d, x, y, z)                                               \
  ((x)*multi_d97 + (d) + (xdim97 * (y)*multi_d97) +                            \
   (xdim97 * ydim97 * (z)*multi_d97))
#define OPS_ACC_MD98(d, x, y, z)                                               \
  ((x)*multi_d98 + (d) + (xdim98 * (y)*multi_d98) +                            \
   (xdim98 * ydim98 * (z)*multi_d98))
#define OPS_ACC_MD99(d, x, y, z)                                               \
  ((x)*multi_d99 + (d) + (xdim99 * (y)*multi_d99) +                            \
   (xdim99 * ydim99 * (z)*multi_d99))
#else
#define OPS_ACC_MD0(d, x, y, z)                                                \
  ((x) + (xdim0 * (y)) + \
  (xdim0 * ydim0 * (z)) + (d) * xdim0 * ydim0 * zdim0)
#define OPS_ACC_MD1(d, x, y, z)                  \
  ((x) + (xdim1 * (y)) + \
  (xdim1 * ydim1 * (z)) + (d) * xdim1 * ydim1 * zdim1)
#define OPS_ACC_MD2(d, x, y, z)                  \
  ((x) + (xdim2 * (y)) + \
  (xdim2 * ydim2 * (z)) + (d) * xdim2 * ydim2 * zdim2)
#define OPS_ACC_MD3(d, x, y, z)                  \
  ((x) + (xdim3 * (y)) + \
  (xdim3 * ydim3 * (z)) + (d) * xdim3 * ydim3 * zdim3)
#define OPS_ACC_MD4(d, x, y, z)                  \
  ((x) + (xdim4 * (y)) + \
  (xdim4 * ydim4 * (z)) + (d) * xdim4 * ydim4 * zdim4)
#define OPS_ACC_MD5(d, x, y, z)                  \
  ((x) + (xdim5 * (y)) + \
  (xdim5 * ydim5 * (z)) + (d) * xdim5 * ydim5 * zdim5)
#define OPS_ACC_MD6(d, x, y, z)                  \
  ((x) + (xdim6 * (y)) + \
  (xdim6 * ydim6 * (z)) + (d) * xdim6 * ydim6 * zdim6)
#define OPS_ACC_MD7(d, x, y, z)                  \
  ((x) + (xdim7 * (y)) + \
  (xdim7 * ydim7 * (z)) + (d) * xdim7 * ydim7 * zdim7)
#define OPS_ACC_MD8(d, x, y, z)                  \
  ((x) + (xdim8 * (y)) + \
  (xdim8 * ydim8 * (z)) + (d) * xdim8 * ydim8 * zdim8)
#define OPS_ACC_MD9(d, x, y, z)                  \
  ((x) + (xdim9 * (y)) + \
  (xdim9 * ydim9 * (z)) + (d) * xdim9 * ydim9 * zdim9)
#define OPS_ACC_MD10(d, x, y, z)                    \
  ((x) + (xdim10 * (y)) + \
   (xdim10 * ydim10 * (z)) + (d) * xdim10 * ydim10 * zdim10)
#define OPS_ACC_MD11(d, x, y, z)                    \
  ((x) + (xdim11 * (y)) + \
   (xdim11 * ydim11 * (z)) + (d) * xdim11 * ydim11 * zdim11)
#define OPS_ACC_MD12(d, x, y, z)                    \
  ((x) + (xdim12 * (y)) + \
   (xdim12 * ydim12 * (z)) + (d) * xdim12 * ydim12 * zdim12)
#define OPS_ACC_MD13(d, x, y, z)                    \
  ((x) + (xdim13 * (y)) + \
   (xdim13 * ydim13 * (z)) + (d) * xdim13 * ydim13 * zdim13)
#define OPS_ACC_MD14(d, x, y, z)                    \
  ((x) + (xdim14 * (y)) + \
   (xdim14 * ydim14 * (z)) + (d) * xdim14 * ydim14 * zdim14)
#define OPS_ACC_MD15(d, x, y, z)                    \
  ((x) + (xdim15 * (y)) + \
   (xdim15 * ydim15 * (z)) + (d) * xdim15 * ydim15 * zdim15)
#define OPS_ACC_MD16(d, x, y, z)                    \
  ((x) + (xdim16 * (y)) + \
   (xdim16 * ydim16 * (z)) + (d) * xdim16 * ydim16 * zdim16)
#define OPS_ACC_MD17(d, x, y, z)                    \
  ((x) + (xdim17 * (y)) + \
   (xdim17 * ydim17 * (z)) + (d) * xdim17 * ydim17 * zdim17)
#define OPS_ACC_MD18(d, x, y, z)                    \
  ((x) + (xdim18 * (y)) + \
   (xdim18 * ydim18 * (z)) + (d) * xdim18 * ydim18 * zdim18)
#define OPS_ACC_MD19(d, x, y, z)                    \
  ((x) + (xdim19 * (y)) + \
   (xdim19 * ydim19 * (z)) + (d) * xdim19 * ydim19 * zdim19)
#define OPS_ACC_MD20(d, x, y, z)                    \
  ((x) + (xdim20 * (y)) + \
   (xdim20 * ydim20 * (z)) + (d) * xdim20 * ydim20 * zdim20)
#define OPS_ACC_MD21(d, x, y, z)                    \
  ((x) + (xdim21 * (y)) + \
   (xdim21 * ydim21 * (z)) + (d) * xdim21 * ydim21 * zdim21)
#define OPS_ACC_MD22(d, x, y, z)                    \
  ((x) + (xdim22 * (y)) + \
   (xdim22 * ydim22 * (z)) + (d) * xdim22 * ydim22 * zdim22)
#define OPS_ACC_MD23(d, x, y, z)                    \
  ((x) + (xdim23 * (y)) + \
   (xdim23 * ydim23 * (z)) + (d) * xdim23 * ydim23 * zdim23)
#define OPS_ACC_MD24(d, x, y, z)                    \
  ((x) + (xdim24 * (y)) + \
   (xdim24 * ydim24 * (z)) + (d) * xdim24 * ydim24 * zdim24)
#define OPS_ACC_MD25(d, x, y, z)                    \
  ((x) + (xdim25 * (y)) + \
   (xdim25 * ydim25 * (z)) + (d) * xdim25 * ydim25 * zdim25)
#define OPS_ACC_MD26(d, x, y, z)                    \
  ((x) + (xdim26 * (y)) + \
   (xdim26 * ydim26 * (z)) + (d) * xdim26 * ydim26 * zdim26)
#define OPS_ACC_MD27(d, x, y, z)                    \
  ((x) + (xdim27 * (y)) + \
   (xdim27 * ydim27 * (z)) + (d) * xdim27 * ydim27 * zdim27)
#define OPS_ACC_MD28(d, x, y, z)                    \
  ((x) + (xdim28 * (y)) + \
   (xdim28 * ydim28 * (z)) + (d) * xdim28 * ydim28 * zdim28)
#define OPS_ACC_MD29(d, x, y, z)                    \
  ((x) + (xdim29 * (y)) + \
   (xdim29 * ydim29 * (z)) + (d) * xdim29 * ydim29 * zdim29)
#define OPS_ACC_MD30(d, x, y, z)                    \
  ((x) + (xdim30 * (y)) + \
   (xdim30 * ydim30 * (z)) + (d) * xdim30 * ydim30 * zdim30)
#define OPS_ACC_MD31(d, x, y, z)                    \
  ((x) + (xdim31 * (y)) + \
   (xdim31 * ydim31 * (z)) + (d) * xdim31 * ydim31 * zdim31)
#define OPS_ACC_MD32(d, x, y, z)                    \
  ((x) + (xdim32 * (y)) + \
   (xdim32 * ydim32 * (z)) + (d) * xdim32 * ydim32 * zdim32)
#define OPS_ACC_MD33(d, x, y, z)                    \
  ((x) + (xdim33 * (y)) + \
   (xdim33 * ydim33 * (z)) + (d) * xdim33 * ydim33 * zdim33)
#define OPS_ACC_MD34(d, x, y, z)                    \
  ((x) + (xdim34 * (y)) + \
   (xdim34 * ydim34 * (z)) + (d) * xdim34 * ydim34 * zdim34)
#define OPS_ACC_MD35(d, x, y, z)                    \
  ((x) + (xdim35 * (y)) + \
   (xdim35 * ydim35 * (z)) + (d) * xdim35 * ydim35 * zdim35)
#define OPS_ACC_MD36(d, x, y, z)                    \
  ((x) + (xdim36 * (y)) + \
   (xdim36 * ydim36 * (z)) + (d) * xdim36 * ydim36 * zdim36)
#define OPS_ACC_MD37(d, x, y, z)                    \
  ((x) + (xdim37 * (y)) + \
   (xdim37 * ydim37 * (z)) + (d) * xdim37 * ydim37 * zdim37)
#define OPS_ACC_MD38(d, x, y, z)                    \
  ((x) + (xdim38 * (y)) + \
   (xdim38 * ydim38 * (z)) + (d) * xdim38 * ydim38 * zdim38)
#define OPS_ACC_MD39(d, x, y, z)                    \
  ((x) + (xdim39 * (y)) + \
   (xdim39 * ydim39 * (z)) + (d) * xdim39 * ydim39 * zdim39)
#define OPS_ACC_MD40(d, x, y, z)                    \
  ((x) + (xdim40 * (y)) + \
   (xdim40 * ydim40 * (z)) + (d) * xdim40 * ydim40 * zdim40)
#define OPS_ACC_MD41(d, x, y, z)                    \
  ((x) + (xdim41 * (y)) + \
   (xdim41 * ydim41 * (z)) + (d) * xdim41 * ydim41 * zdim41)
#define OPS_ACC_MD42(d, x, y, z)                    \
  ((x) + (xdim42 * (y)) + \
   (xdim42 * ydim42 * (z)) + (d) * xdim42 * ydim42 * zdim42)
#define OPS_ACC_MD43(d, x, y, z)                    \
  ((x) + (xdim43 * (y)) + \
   (xdim43 * ydim43 * (z)) + (d) * xdim43 * ydim43 * zdim43)
#define OPS_ACC_MD44(d, x, y, z)                    \
  ((x) + (xdim44 * (y)) + \
   (xdim44 * ydim44 * (z)) + (d) * xdim44 * ydim44 * zdim44)
#define OPS_ACC_MD45(d, x, y, z)                    \
  ((x) + (xdim45 * (y)) + \
   (xdim45 * ydim45 * (z)) + (d) * xdim45 * ydim45 * zdim45)
#define OPS_ACC_MD46(d, x, y, z)                    \
  ((x) + (xdim46 * (y)) + \
   (xdim46 * ydim46 * (z)) + (d) * xdim46 * ydim46 * zdim46)
#define OPS_ACC_MD47(d, x, y, z)                    \
  ((x) + (xdim47 * (y)) + \
   (xdim47 * ydim47 * (z)) + (d) * xdim47 * ydim47 * zdim47)
#define OPS_ACC_MD48(d, x, y, z)                    \
  ((x) + (xdim48 * (y)) + \
   (xdim48 * ydim48 * (z)) + (d) * xdim48 * ydim48 * zdim48)
#define OPS_ACC_MD49(d, x, y, z)                    \
  ((x) + (xdim49 * (y)) + \
   (xdim49 * ydim49 * (z)) + (d) * xdim49 * ydim49 * zdim49)
#define OPS_ACC_MD50(d, x, y, z)                    \
  ((x) + (xdim50 * (y)) + \
   (xdim50 * ydim50 * (z)) + (d) * xdim50 * ydim50 * zdim50)
#define OPS_ACC_MD51(d, x, y, z)                    \
  ((x) + (xdim51 * (y)) + \
   (xdim51 * ydim51 * (z)) + (d) * xdim51 * ydim51 * zdim51)
#define OPS_ACC_MD52(d, x, y, z)                    \
  ((x) + (xdim52 * (y)) + \
   (xdim52 * ydim52 * (z)) + (d) * xdim52 * ydim52 * zdim52)
#define OPS_ACC_MD53(d, x, y, z)                    \
  ((x) + (xdim53 * (y)) + \
   (xdim53 * ydim53 * (z)) + (d) * xdim53 * ydim53 * zdim53)
#define OPS_ACC_MD54(d, x, y, z)                    \
  ((x) + (xdim54 * (y)) + \
   (xdim54 * ydim54 * (z)) + (d) * xdim54 * ydim54 * zdim54)
#define OPS_ACC_MD55(d, x, y, z)                    \
  ((x) + (xdim55 * (y)) + \
   (xdim55 * ydim55 * (z)) + (d) * xdim55 * ydim55 * zdim55)
#define OPS_ACC_MD56(d, x, y, z)                    \
  ((x) + (xdim56 * (y)) + \
   (xdim56 * ydim56 * (z)) + (d) * xdim56 * ydim56 * zdim56)
#define OPS_ACC_MD57(d, x, y, z)                    \
  ((x) + (xdim57 * (y)) + \
   (xdim57 * ydim57 * (z)) + (d) * xdim57 * ydim57 * zdim57)
#define OPS_ACC_MD58(d, x, y, z)                    \
  ((x) + (xdim58 * (y)) + \
   (xdim58 * ydim58 * (z)) + (d) * xdim58 * ydim58 * zdim58)
#define OPS_ACC_MD59(d, x, y, z)                    \
  ((x) + (xdim59 * (y)) + \
   (xdim59 * ydim59 * (z)) + (d) * xdim59 * ydim59 * zdim59)
#define OPS_ACC_MD60(d, x, y, z)                    \
  ((x) + (xdim60 * (y)) + \
   (xdim60 * ydim60 * (z)) + (d) * xdim60 * ydim60 * zdim60)
#define OPS_ACC_MD61(d, x, y, z)                    \
  ((x) + (xdim61 * (y)) + \
   (xdim61 * ydim61 * (z)) + (d) * xdim61 * ydim61 * zdim61)
#define OPS_ACC_MD62(d, x, y, z)                    \
  ((x) + (xdim62 * (y)) + \
   (xdim62 * ydim62 * (z)) + (d) * xdim62 * ydim62 * zdim62)
#define OPS_ACC_MD63(d, x, y, z)                    \
  ((x) + (xdim63 * (y)) + \
   (xdim63 * ydim63 * (z)) + (d) * xdim63 * ydim63 * zdim63)
#define OPS_ACC_MD64(d, x, y, z)                    \
  ((x) + (xdim64 * (y)) + \
   (xdim64 * ydim64 * (z)) + (d) * xdim64 * ydim64 * zdim64)
#define OPS_ACC_MD65(d, x, y, z)                    \
  ((x) + (xdim65 * (y)) + \
   (xdim65 * ydim65 * (z)) + (d) * xdim65 * ydim65 * zdim65)
#define OPS_ACC_MD66(d, x, y, z)                    \
  ((x) + (xdim66 * (y)) + \
   (xdim66 * ydim66 * (z)) + (d) * xdim66 * ydim66 * zdim66)
#define OPS_ACC_MD67(d, x, y, z)                    \
  ((x) + (xdim67 * (y)) + \
   (xdim67 * ydim67 * (z)) + (d) * xdim67 * ydim67 * zdim67)
#define OPS_ACC_MD68(d, x, y, z)                    \
  ((x) + (xdim68 * (y)) + \
   (xdim68 * ydim68 * (z)) + (d) * xdim68 * ydim68 * zdim68)
#define OPS_ACC_MD69(d, x, y, z)                    \
  ((x) + (xdim69 * (y)) + \
   (xdim69 * ydim69 * (z)) + (d) * xdim69 * ydim69 * zdim69)
#define OPS_ACC_MD70(d, x, y, z)                    \
  ((x) + (xdim70 * (y)) + \
   (xdim70 * ydim70 * (z)) + (d) * xdim70 * ydim70 * zdim70)
#define OPS_ACC_MD71(d, x, y, z)                    \
  ((x) + (xdim71 * (y)) + \
   (xdim71 * ydim71 * (z)) + (d) * xdim71 * ydim71 * zdim71)
#define OPS_ACC_MD72(d, x, y, z)                    \
  ((x) + (xdim72 * (y)) + \
   (xdim72 * ydim72 * (z)) + (d) * xdim72 * ydim72 * zdim72)
#define OPS_ACC_MD73(d, x, y, z)                    \
  ((x) + (xdim73 * (y)) + \
   (xdim73 * ydim73 * (z)) + (d) * xdim73 * ydim73 * zdim73)
#define OPS_ACC_MD74(d, x, y, z)                    \
  ((x) + (xdim74 * (y)) + \
   (xdim74 * ydim74 * (z)) + (d) * xdim74 * ydim74 * zdim74)
#define OPS_ACC_MD75(d, x, y, z)                    \
  ((x) + (xdim75 * (y)) + \
   (xdim75 * ydim75 * (z)) + (d) * xdim75 * ydim75 * zdim75)
#define OPS_ACC_MD76(d, x, y, z)                    \
  ((x) + (xdim76 * (y)) + \
   (xdim76 * ydim76 * (z)) + (d) * xdim76 * ydim76 * zdim76)
#define OPS_ACC_MD77(d, x, y, z)                    \
  ((x) + (xdim77 * (y)) + \
   (xdim77 * ydim77 * (z)) + (d) * xdim77 * ydim77 * zdim77)
#define OPS_ACC_MD78(d, x, y, z)                    \
  ((x) + (xdim78 * (y)) + \
   (xdim78 * ydim78 * (z)) + (d) * xdim78 * ydim78 * zdim78)
#define OPS_ACC_MD79(d, x, y, z)                    \
  ((x) + (xdim79 * (y)) + \
   (xdim79 * ydim79 * (z)) + (d) * xdim79 * ydim79 * zdim79)
#define OPS_ACC_MD80(d, x, y, z)                    \
  ((x) + (xdim80 * (y)) + \
   (xdim80 * ydim80 * (z)) + (d) * xdim80 * ydim80 * zdim80)
#define OPS_ACC_MD81(d, x, y, z)                    \
  ((x) + (xdim81 * (y)) + \
   (xdim81 * ydim81 * (z)) + (d) * xdim81 * ydim81 * zdim81)
#define OPS_ACC_MD82(d, x, y, z)                    \
  ((x) + (xdim82 * (y)) + \
   (xdim82 * ydim82 * (z)) + (d) * xdim82 * ydim82 * zdim82)
#define OPS_ACC_MD83(d, x, y, z)                    \
  ((x) + (xdim83 * (y)) + \
   (xdim83 * ydim83 * (z)) + (d) * xdim83 * ydim83 * zdim83)
#define OPS_ACC_MD84(d, x, y, z)                    \
  ((x) + (xdim84 * (y)) + \
   (xdim84 * ydim84 * (z)) + (d) * xdim84 * ydim84 * zdim84)
#define OPS_ACC_MD85(d, x, y, z)                    \
  ((x) + (xdim85 * (y)) + \
   (xdim85 * ydim85 * (z)) + (d) * xdim85 * ydim85 * zdim85)
#define OPS_ACC_MD86(d, x, y, z)                    \
  ((x) + (xdim86 * (y)) + \
   (xdim86 * ydim86 * (z)) + (d) * xdim86 * ydim86 * zdim86)
#define OPS_ACC_MD87(d, x, y, z)                    \
  ((x) + (xdim87 * (y)) + \
   (xdim87 * ydim87 * (z)) + (d) * xdim87 * ydim87 * zdim87)
#define OPS_ACC_MD88(d, x, y, z)                    \
  ((x) + (xdim88 * (y)) + \
   (xdim88 * ydim88 * (z)) + (d) * xdim88 * ydim88 * zdim88)
#define OPS_ACC_MD89(d, x, y, z)                    \
  ((x) + (xdim89 * (y)) + \
   (xdim89 * ydim89 * (z)) + (d) * xdim89 * ydim89 * zdim89)
#define OPS_ACC_MD90(d, x, y, z)                    \
  ((x) + (xdim90 * (y)) + \
   (xdim90 * ydim90 * (z)) + (d) * xdim90 * ydim90 * zdim90)
#define OPS_ACC_MD91(d, x, y, z)                    \
  ((x) + (xdim91 * (y)) + \
   (xdim91 * ydim91 * (z)) + (d) * xdim91 * ydim91 * zdim91)
#define OPS_ACC_MD92(d, x, y, z)                    \
  ((x) + (xdim92 * (y)) + \
   (xdim92 * ydim92 * (z)) + (d) * xdim92 * ydim92 * zdim92)
#define OPS_ACC_MD93(d, x, y, z)                    \
  ((x) + (xdim93 * (y)) + \
   (xdim93 * ydim93 * (z)) + (d) * xdim93 * ydim93 * zdim93)
#define OPS_ACC_MD94(d, x, y, z)                    \
  ((x) + (xdim94 * (y)) + \
   (xdim94 * ydim94 * (z)) + (d) * xdim94 * ydim94 * zdim94)
#define OPS_ACC_MD95(d, x, y, z)                    \
  ((x) + (xdim95 * (y)) + \
   (xdim95 * ydim95 * (z)) + (d) * xdim95 * ydim95 * zdim95)
#define OPS_ACC_MD96(d, x, y, z)                    \
  ((x) + (xdim96 * (y)) + \
   (xdim96 * ydim96 * (z)) + (d) * xdim96 * ydim96 * zdim96)
#define OPS_ACC_MD97(d, x, y, z)                    \
  ((x) + (xdim97 * (y)) + \
   (xdim97 * ydim97 * (z)) + (d) * xdim97 * ydim97 * zdim97)
#define OPS_ACC_MD98(d, x, y, z)                    \
  ((x) + (xdim98 * (y)) + \
   (xdim98 * ydim98 * (z)) + (d) * xdim98 * ydim98 * zdim98)
#define OPS_ACC_MD99(d, x, y, z)                    \
  ((x) + (xdim99 * (y)) + \
   (xdim99 * ydim99 * (z)) + (d) * xdim99 * ydim99 * zdim99)
#endif
#else
/// TODO #define OPS_ACC_MD0(x,y,z,d) (ops_stencil_check_3d_md(0, x, y, z,
/// xdim0, ydim0))
/// --> int ops_stencil_check_3d_md(int arg_idx, int idx0, int idx1, int idx2,
/// int dim0, int dim1, int mult_d, int d);
#endif
#elif defined OPS_2D // macros for 2D application
#ifndef OPS_DEBUG
#ifndef OPS_SOA
#define OPS_ACC_MD0(d, x, y) ((x)*multi_d0 + (d) + (xdim0 * (y)*multi_d0))
#define OPS_ACC_MD1(d, x, y) ((x)*multi_d1 + (d) + (xdim1 * (y)*multi_d1))
#define OPS_ACC_MD2(d, x, y) ((x)*multi_d2 + (d) + (xdim2 * (y)*multi_d2))
#define OPS_ACC_MD3(d, x, y) ((x)*multi_d3 + (d) + (xdim3 * (y)*multi_d3))
#define OPS_ACC_MD4(d, x, y) ((x)*multi_d4 + (d) + (xdim4 * (y)*multi_d4))
#define OPS_ACC_MD5(d, x, y) ((x)*multi_d5 + (d) + (xdim5 * (y)*multi_d5))
#define OPS_ACC_MD6(d, x, y) ((x)*multi_d6 + (d) + (xdim6 * (y)*multi_d6))
#define OPS_ACC_MD7(d, x, y) ((x)*multi_d7 + (d) + (xdim7 * (y)*multi_d7))
#define OPS_ACC_MD8(d, x, y) ((x)*multi_d8 + (d) + (xdim8 * (y)*multi_d8))
#define OPS_ACC_MD9(d, x, y) ((x)*multi_d9 + (d) + (xdim9 * (y)*multi_d9))
#define OPS_ACC_MD10(d, x, y) ((x)*multi_d10 + (d) + (xdim10 * (y)*multi_d10))
#define OPS_ACC_MD11(d, x, y) ((x)*multi_d11 + (d) + (xdim11 * (y)*multi_d11))
#define OPS_ACC_MD12(d, x, y) ((x)*multi_d12 + (d) + (xdim12 * (y)*multi_d12))
#define OPS_ACC_MD13(d, x, y) ((x)*multi_d13 + (d) + (xdim13 * (y)*multi_d13))
#define OPS_ACC_MD14(d, x, y) ((x)*multi_d14 + (d) + (xdim14 * (y)*multi_d14))
#define OPS_ACC_MD15(d, x, y) ((x)*multi_d15 + (d) + (xdim15 * (y)*multi_d15))
#define OPS_ACC_MD16(d, x, y) ((x)*multi_d16 + (d) + (xdim16 * (y)*multi_d16))
#define OPS_ACC_MD17(d, x, y) ((x)*multi_d17 + (d) + (xdim17 * (y)*multi_d17))
#define OPS_ACC_MD18(d, x, y) ((x)*multi_d18 + (d) + (xdim18 * (y)*multi_d18))
#define OPS_ACC_MD19(d, x, y) ((x)*multi_d19 + (d) + (xdim19 * (y)*multi_d19))
#define OPS_ACC_MD20(d, x, y) ((x)*multi_d20 + (d) + (xdim20 * (y)*multi_d20))
#define OPS_ACC_MD21(d, x, y) ((x)*multi_d21 + (d) + (xdim21 * (y)*multi_d21))
#define OPS_ACC_MD22(d, x, y) ((x)*multi_d22 + (d) + (xdim22 * (y)*multi_d22))
#define OPS_ACC_MD23(d, x, y) ((x)*multi_d23 + (d) + (xdim23 * (y)*multi_d23))
#define OPS_ACC_MD24(d, x, y) ((x)*multi_d24 + (d) + (xdim24 * (y)*multi_d24))
#define OPS_ACC_MD25(d, x, y) ((x)*multi_d25 + (d) + (xdim25 * (y)*multi_d25))
#define OPS_ACC_MD26(d, x, y) ((x)*multi_d26 + (d) + (xdim26 * (y)*multi_d26))
#define OPS_ACC_MD27(d, x, y) ((x)*multi_d27 + (d) + (xdim27 * (y)*multi_d27))
#define OPS_ACC_MD28(d, x, y) ((x)*multi_d28 + (d) + (xdim28 * (y)*multi_d28))
#define OPS_ACC_MD29(d, x, y) ((x)*multi_d29 + (d) + (xdim29 * (y)*multi_d29))
#define OPS_ACC_MD30(d, x, y) ((x)*multi_d30 + (d) + (xdim30 * (y)*multi_d30))
#define OPS_ACC_MD31(d, x, y) ((x)*multi_d31 + (d) + (xdim31 * (y)*multi_d31))
#define OPS_ACC_MD32(d, x, y) ((x)*multi_d32 + (d) + (xdim32 * (y)*multi_d32))
#define OPS_ACC_MD33(d, x, y) ((x)*multi_d33 + (d) + (xdim33 * (y)*multi_d33))
#define OPS_ACC_MD34(d, x, y) ((x)*multi_d34 + (d) + (xdim34 * (y)*multi_d34))
#define OPS_ACC_MD35(d, x, y) ((x)*multi_d35 + (d) + (xdim35 * (y)*multi_d35))
#define OPS_ACC_MD36(d, x, y) ((x)*multi_d36 + (d) + (xdim36 * (y)*multi_d36))
#define OPS_ACC_MD37(d, x, y) ((x)*multi_d37 + (d) + (xdim37 * (y)*multi_d37))
#define OPS_ACC_MD38(d, x, y) ((x)*multi_d38 + (d) + (xdim38 * (y)*multi_d38))
#define OPS_ACC_MD39(d, x, y) ((x)*multi_d39 + (d) + (xdim39 * (y)*multi_d39))
#define OPS_ACC_MD40(d, x, y) ((x)*multi_d40 + (d) + (xdim40 * (y)*multi_d40))
#define OPS_ACC_MD41(d, x, y) ((x)*multi_d41 + (d) + (xdim41 * (y)*multi_d41))
#define OPS_ACC_MD42(d, x, y) ((x)*multi_d42 + (d) + (xdim42 * (y)*multi_d42))
#define OPS_ACC_MD43(d, x, y) ((x)*multi_d43 + (d) + (xdim43 * (y)*multi_d43))
#define OPS_ACC_MD44(d, x, y) ((x)*multi_d44 + (d) + (xdim44 * (y)*multi_d44))
#define OPS_ACC_MD45(d, x, y) ((x)*multi_d45 + (d) + (xdim45 * (y)*multi_d45))
#define OPS_ACC_MD46(d, x, y) ((x)*multi_d46 + (d) + (xdim46 * (y)*multi_d46))
#define OPS_ACC_MD47(d, x, y) ((x)*multi_d47 + (d) + (xdim47 * (y)*multi_d47))
#define OPS_ACC_MD48(d, x, y) ((x)*multi_d48 + (d) + (xdim48 * (y)*multi_d48))
#define OPS_ACC_MD49(d, x, y) ((x)*multi_d49 + (d) + (xdim49 * (y)*multi_d49))
#define OPS_ACC_MD50(d, x, y) ((x)*multi_d50 + (d) + (xdim50 * (y)*multi_d50))
#define OPS_ACC_MD51(d, x, y) ((x)*multi_d51 + (d) + (xdim51 * (y)*multi_d51))
#define OPS_ACC_MD52(d, x, y) ((x)*multi_d52 + (d) + (xdim52 * (y)*multi_d52))
#define OPS_ACC_MD53(d, x, y) ((x)*multi_d53 + (d) + (xdim53 * (y)*multi_d53))
#define OPS_ACC_MD54(d, x, y) ((x)*multi_d54 + (d) + (xdim54 * (y)*multi_d54))
#define OPS_ACC_MD55(d, x, y) ((x)*multi_d55 + (d) + (xdim55 * (y)*multi_d55))
#define OPS_ACC_MD56(d, x, y) ((x)*multi_d56 + (d) + (xdim56 * (y)*multi_d56))
#define OPS_ACC_MD57(d, x, y) ((x)*multi_d57 + (d) + (xdim57 * (y)*multi_d57))
#define OPS_ACC_MD58(d, x, y) ((x)*multi_d58 + (d) + (xdim58 * (y)*multi_d58))
#define OPS_ACC_MD59(d, x, y) ((x)*multi_d59 + (d) + (xdim59 * (y)*multi_d59))
#define OPS_ACC_MD60(d, x, y) ((x)*multi_d60 + (d) + (xdim60 * (y)*multi_d60))
#define OPS_ACC_MD61(d, x, y) ((x)*multi_d61 + (d) + (xdim61 * (y)*multi_d61))
#define OPS_ACC_MD62(d, x, y) ((x)*multi_d62 + (d) + (xdim62 * (y)*multi_d62))
#define OPS_ACC_MD63(d, x, y) ((x)*multi_d63 + (d) + (xdim63 * (y)*multi_d63))
#define OPS_ACC_MD64(d, x, y) ((x)*multi_d64 + (d) + (xdim64 * (y)*multi_d64))
#define OPS_ACC_MD65(d, x, y) ((x)*multi_d65 + (d) + (xdim65 * (y)*multi_d65))
#define OPS_ACC_MD66(d, x, y) ((x)*multi_d66 + (d) + (xdim66 * (y)*multi_d66))
#define OPS_ACC_MD67(d, x, y) ((x)*multi_d67 + (d) + (xdim67 * (y)*multi_d67))
#define OPS_ACC_MD68(d, x, y) ((x)*multi_d68 + (d) + (xdim68 * (y)*multi_d68))
#define OPS_ACC_MD69(d, x, y) ((x)*multi_d69 + (d) + (xdim69 * (y)*multi_d69))
#define OPS_ACC_MD70(d, x, y) ((x)*multi_d70 + (d) + (xdim70 * (y)*multi_d70))
#define OPS_ACC_MD71(d, x, y) ((x)*multi_d71 + (d) + (xdim71 * (y)*multi_d71))
#define OPS_ACC_MD72(d, x, y) ((x)*multi_d72 + (d) + (xdim72 * (y)*multi_d72))
#define OPS_ACC_MD73(d, x, y) ((x)*multi_d73 + (d) + (xdim73 * (y)*multi_d73))
#define OPS_ACC_MD74(d, x, y) ((x)*multi_d74 + (d) + (xdim74 * (y)*multi_d74))
#define OPS_ACC_MD75(d, x, y) ((x)*multi_d75 + (d) + (xdim75 * (y)*multi_d75))
#define OPS_ACC_MD76(d, x, y) ((x)*multi_d76 + (d) + (xdim76 * (y)*multi_d76))
#define OPS_ACC_MD77(d, x, y) ((x)*multi_d77 + (d) + (xdim77 * (y)*multi_d77))
#define OPS_ACC_MD78(d, x, y) ((x)*multi_d78 + (d) + (xdim78 * (y)*multi_d78))
#define OPS_ACC_MD79(d, x, y) ((x)*multi_d79 + (d) + (xdim79 * (y)*multi_d79))
#define OPS_ACC_MD80(d, x, y) ((x)*multi_d80 + (d) + (xdim80 * (y)*multi_d80))
#define OPS_ACC_MD81(d, x, y) ((x)*multi_d81 + (d) + (xdim81 * (y)*multi_d81))
#define OPS_ACC_MD82(d, x, y) ((x)*multi_d82 + (d) + (xdim82 * (y)*multi_d82))
#define OPS_ACC_MD83(d, x, y) ((x)*multi_d83 + (d) + (xdim83 * (y)*multi_d83))
#define OPS_ACC_MD84(d, x, y) ((x)*multi_d84 + (d) + (xdim84 * (y)*multi_d84))
#define OPS_ACC_MD85(d, x, y) ((x)*multi_d85 + (d) + (xdim85 * (y)*multi_d85))
#define OPS_ACC_MD86(d, x, y) ((x)*multi_d86 + (d) + (xdim86 * (y)*multi_d86))
#define OPS_ACC_MD87(d, x, y) ((x)*multi_d87 + (d) + (xdim87 * (y)*multi_d87))
#define OPS_ACC_MD88(d, x, y) ((x)*multi_d88 + (d) + (xdim88 * (y)*multi_d88))
#define OPS_ACC_MD89(d, x, y) ((x)*multi_d89 + (d) + (xdim89 * (y)*multi_d89))
#define OPS_ACC_MD90(d, x, y) ((x)*multi_d90 + (d) + (xdim90 * (y)*multi_d90))
#define OPS_ACC_MD91(d, x, y) ((x)*multi_d91 + (d) + (xdim91 * (y)*multi_d91))
#define OPS_ACC_MD92(d, x, y) ((x)*multi_d92 + (d) + (xdim92 * (y)*multi_d92))
#define OPS_ACC_MD93(d, x, y) ((x)*multi_d93 + (d) + (xdim93 * (y)*multi_d93))
#define OPS_ACC_MD94(d, x, y) ((x)*multi_d94 + (d) + (xdim94 * (y)*multi_d94))
#define OPS_ACC_MD95(d, x, y) ((x)*multi_d95 + (d) + (xdim95 * (y)*multi_d95))
#define OPS_ACC_MD96(d, x, y) ((x)*multi_d96 + (d) + (xdim96 * (y)*multi_d96))
#define OPS_ACC_MD97(d, x, y) ((x)*multi_d97 + (d) + (xdim97 * (y)*multi_d97))
#define OPS_ACC_MD98(d, x, y) ((x)*multi_d98 + (d) + (xdim98 * (y)*multi_d98))
#define OPS_ACC_MD99(d, x, y) ((x)*multi_d99 + (d) + (xdim99 * (y)*multi_d99))
#else
#define OPS_ACC_MD0(d, x, y)  ((x) + (xdim0  * (y)) + (d) * xdim0  * ydim0 )
#define OPS_ACC_MD1(d, x, y)  ((x) + (xdim1  * (y)) + (d) * xdim1  * ydim1 )
#define OPS_ACC_MD2(d, x, y)  ((x) + (xdim2  * (y)) + (d) * xdim2  * ydim2 )
#define OPS_ACC_MD3(d, x, y)  ((x) + (xdim3  * (y)) + (d) * xdim3  * ydim3 )
#define OPS_ACC_MD4(d, x, y)  ((x) + (xdim4  * (y)) + (d) * xdim4  * ydim4 )
#define OPS_ACC_MD5(d, x, y)  ((x) + (xdim5  * (y)) + (d) * xdim5  * ydim5 )
#define OPS_ACC_MD6(d, x, y)  ((x) + (xdim6  * (y)) + (d) * xdim6  * ydim6 )
#define OPS_ACC_MD7(d, x, y)  ((x) + (xdim7  * (y)) + (d) * xdim7  * ydim7 )
#define OPS_ACC_MD8(d, x, y)  ((x) + (xdim8  * (y)) + (d) * xdim8  * ydim8 )
#define OPS_ACC_MD9(d, x, y)  ((x) + (xdim9  * (y)) + (d) * xdim9  * ydim9 )
#define OPS_ACC_MD10(d, x, y) ((x) + (xdim10 * (y)) + (d) * xdim10 * ydim10)
#define OPS_ACC_MD11(d, x, y) ((x) + (xdim11 * (y)) + (d) * xdim11 * ydim11)
#define OPS_ACC_MD12(d, x, y) ((x) + (xdim12 * (y)) + (d) * xdim12 * ydim12)
#define OPS_ACC_MD13(d, x, y) ((x) + (xdim13 * (y)) + (d) * xdim13 * ydim13)
#define OPS_ACC_MD14(d, x, y) ((x) + (xdim14 * (y)) + (d) * xdim14 * ydim14)
#define OPS_ACC_MD15(d, x, y) ((x) + (xdim15 * (y)) + (d) * xdim15 * ydim15)
#define OPS_ACC_MD16(d, x, y) ((x) + (xdim16 * (y)) + (d) * xdim16 * ydim16)
#define OPS_ACC_MD17(d, x, y) ((x) + (xdim17 * (y)) + (d) * xdim17 * ydim17)
#define OPS_ACC_MD18(d, x, y) ((x) + (xdim18 * (y)) + (d) * xdim18 * ydim18)
#define OPS_ACC_MD19(d, x, y) ((x) + (xdim19 * (y)) + (d) * xdim19 * ydim19)
#define OPS_ACC_MD20(d, x, y) ((x) + (xdim20 * (y)) + (d) * xdim20 * ydim20)
#define OPS_ACC_MD21(d, x, y) ((x) + (xdim21 * (y)) + (d) * xdim21 * ydim21)
#define OPS_ACC_MD22(d, x, y) ((x) + (xdim22 * (y)) + (d) * xdim22 * ydim22)
#define OPS_ACC_MD23(d, x, y) ((x) + (xdim23 * (y)) + (d) * xdim23 * ydim23)
#define OPS_ACC_MD24(d, x, y) ((x) + (xdim24 * (y)) + (d) * xdim24 * ydim24)
#define OPS_ACC_MD25(d, x, y) ((x) + (xdim25 * (y)) + (d) * xdim25 * ydim25)
#define OPS_ACC_MD26(d, x, y) ((x) + (xdim26 * (y)) + (d) * xdim26 * ydim26)
#define OPS_ACC_MD27(d, x, y) ((x) + (xdim27 * (y)) + (d) * xdim27 * ydim27)
#define OPS_ACC_MD28(d, x, y) ((x) + (xdim28 * (y)) + (d) * xdim28 * ydim28)
#define OPS_ACC_MD29(d, x, y) ((x) + (xdim29 * (y)) + (d) * xdim29 * ydim29)
#define OPS_ACC_MD30(d, x, y) ((x) + (xdim30 * (y)) + (d) * xdim30 * ydim30)
#define OPS_ACC_MD31(d, x, y) ((x) + (xdim31 * (y)) + (d) * xdim31 * ydim31)
#define OPS_ACC_MD32(d, x, y) ((x) + (xdim32 * (y)) + (d) * xdim32 * ydim32)
#define OPS_ACC_MD33(d, x, y) ((x) + (xdim33 * (y)) + (d) * xdim33 * ydim33)
#define OPS_ACC_MD34(d, x, y) ((x) + (xdim34 * (y)) + (d) * xdim34 * ydim34)
#define OPS_ACC_MD35(d, x, y) ((x) + (xdim35 * (y)) + (d) * xdim35 * ydim35)
#define OPS_ACC_MD36(d, x, y) ((x) + (xdim36 * (y)) + (d) * xdim36 * ydim36)
#define OPS_ACC_MD37(d, x, y) ((x) + (xdim37 * (y)) + (d) * xdim37 * ydim37)
#define OPS_ACC_MD38(d, x, y) ((x) + (xdim38 * (y)) + (d) * xdim38 * ydim38)
#define OPS_ACC_MD39(d, x, y) ((x) + (xdim39 * (y)) + (d) * xdim39 * ydim39)
#define OPS_ACC_MD40(d, x, y) ((x) + (xdim40 * (y)) + (d) * xdim40 * ydim40)
#define OPS_ACC_MD41(d, x, y) ((x) + (xdim41 * (y)) + (d) * xdim41 * ydim41)
#define OPS_ACC_MD42(d, x, y) ((x) + (xdim42 * (y)) + (d) * xdim42 * ydim42)
#define OPS_ACC_MD43(d, x, y) ((x) + (xdim43 * (y)) + (d) * xdim43 * ydim43)
#define OPS_ACC_MD44(d, x, y) ((x) + (xdim44 * (y)) + (d) * xdim44 * ydim44)
#define OPS_ACC_MD45(d, x, y) ((x) + (xdim45 * (y)) + (d) * xdim45 * ydim45)
#define OPS_ACC_MD46(d, x, y) ((x) + (xdim46 * (y)) + (d) * xdim46 * ydim46)
#define OPS_ACC_MD47(d, x, y) ((x) + (xdim47 * (y)) + (d) * xdim47 * ydim47)
#define OPS_ACC_MD48(d, x, y) ((x) + (xdim48 * (y)) + (d) * xdim48 * ydim48)
#define OPS_ACC_MD49(d, x, y) ((x) + (xdim49 * (y)) + (d) * xdim49 * ydim49)
#define OPS_ACC_MD50(d, x, y) ((x) + (xdim50 * (y)) + (d) * xdim50 * ydim50)
#define OPS_ACC_MD51(d, x, y) ((x) + (xdim51 * (y)) + (d) * xdim51 * ydim51)
#define OPS_ACC_MD52(d, x, y) ((x) + (xdim52 * (y)) + (d) * xdim52 * ydim52)
#define OPS_ACC_MD53(d, x, y) ((x) + (xdim53 * (y)) + (d) * xdim53 * ydim53)
#define OPS_ACC_MD54(d, x, y) ((x) + (xdim54 * (y)) + (d) * xdim54 * ydim54)
#define OPS_ACC_MD55(d, x, y) ((x) + (xdim55 * (y)) + (d) * xdim55 * ydim55)
#define OPS_ACC_MD56(d, x, y) ((x) + (xdim56 * (y)) + (d) * xdim56 * ydim56)
#define OPS_ACC_MD57(d, x, y) ((x) + (xdim57 * (y)) + (d) * xdim57 * ydim57)
#define OPS_ACC_MD58(d, x, y) ((x) + (xdim58 * (y)) + (d) * xdim58 * ydim58)
#define OPS_ACC_MD59(d, x, y) ((x) + (xdim59 * (y)) + (d) * xdim59 * ydim59)
#define OPS_ACC_MD60(d, x, y) ((x) + (xdim60 * (y)) + (d) * xdim60 * ydim60)
#define OPS_ACC_MD61(d, x, y) ((x) + (xdim61 * (y)) + (d) * xdim61 * ydim61)
#define OPS_ACC_MD62(d, x, y) ((x) + (xdim62 * (y)) + (d) * xdim62 * ydim62)
#define OPS_ACC_MD63(d, x, y) ((x) + (xdim63 * (y)) + (d) * xdim63 * ydim63)
#define OPS_ACC_MD64(d, x, y) ((x) + (xdim64 * (y)) + (d) * xdim64 * ydim64)
#define OPS_ACC_MD65(d, x, y) ((x) + (xdim65 * (y)) + (d) * xdim65 * ydim65)
#define OPS_ACC_MD66(d, x, y) ((x) + (xdim66 * (y)) + (d) * xdim66 * ydim66)
#define OPS_ACC_MD67(d, x, y) ((x) + (xdim67 * (y)) + (d) * xdim67 * ydim67)
#define OPS_ACC_MD68(d, x, y) ((x) + (xdim68 * (y)) + (d) * xdim68 * ydim68)
#define OPS_ACC_MD69(d, x, y) ((x) + (xdim69 * (y)) + (d) * xdim69 * ydim69)
#define OPS_ACC_MD70(d, x, y) ((x) + (xdim70 * (y)) + (d) * xdim70 * ydim70)
#define OPS_ACC_MD71(d, x, y) ((x) + (xdim71 * (y)) + (d) * xdim71 * ydim71)
#define OPS_ACC_MD72(d, x, y) ((x) + (xdim72 * (y)) + (d) * xdim72 * ydim72)
#define OPS_ACC_MD73(d, x, y) ((x) + (xdim73 * (y)) + (d) * xdim73 * ydim73)
#define OPS_ACC_MD74(d, x, y) ((x) + (xdim74 * (y)) + (d) * xdim74 * ydim74)
#define OPS_ACC_MD75(d, x, y) ((x) + (xdim75 * (y)) + (d) * xdim75 * ydim75)
#define OPS_ACC_MD76(d, x, y) ((x) + (xdim76 * (y)) + (d) * xdim76 * ydim76)
#define OPS_ACC_MD77(d, x, y) ((x) + (xdim77 * (y)) + (d) * xdim77 * ydim77)
#define OPS_ACC_MD78(d, x, y) ((x) + (xdim78 * (y)) + (d) * xdim78 * ydim78)
#define OPS_ACC_MD79(d, x, y) ((x) + (xdim79 * (y)) + (d) * xdim79 * ydim79)
#define OPS_ACC_MD80(d, x, y) ((x) + (xdim80 * (y)) + (d) * xdim80 * ydim80)
#define OPS_ACC_MD81(d, x, y) ((x) + (xdim81 * (y)) + (d) * xdim81 * ydim81)
#define OPS_ACC_MD82(d, x, y) ((x) + (xdim82 * (y)) + (d) * xdim82 * ydim82)
#define OPS_ACC_MD83(d, x, y) ((x) + (xdim83 * (y)) + (d) * xdim83 * ydim83)
#define OPS_ACC_MD84(d, x, y) ((x) + (xdim84 * (y)) + (d) * xdim84 * ydim84)
#define OPS_ACC_MD85(d, x, y) ((x) + (xdim85 * (y)) + (d) * xdim85 * ydim85)
#define OPS_ACC_MD86(d, x, y) ((x) + (xdim86 * (y)) + (d) * xdim86 * ydim86)
#define OPS_ACC_MD87(d, x, y) ((x) + (xdim87 * (y)) + (d) * xdim87 * ydim87)
#define OPS_ACC_MD88(d, x, y) ((x) + (xdim88 * (y)) + (d) * xdim88 * ydim88)
#define OPS_ACC_MD89(d, x, y) ((x) + (xdim89 * (y)) + (d) * xdim89 * ydim89)
#define OPS_ACC_MD90(d, x, y) ((x) + (xdim90 * (y)) + (d) * xdim90 * ydim90)
#define OPS_ACC_MD91(d, x, y) ((x) + (xdim91 * (y)) + (d) * xdim91 * ydim91)
#define OPS_ACC_MD92(d, x, y) ((x) + (xdim92 * (y)) + (d) * xdim92 * ydim92)
#define OPS_ACC_MD93(d, x, y) ((x) + (xdim93 * (y)) + (d) * xdim93 * ydim93)
#define OPS_ACC_MD94(d, x, y) ((x) + (xdim94 * (y)) + (d) * xdim94 * ydim94)
#define OPS_ACC_MD95(d, x, y) ((x) + (xdim95 * (y)) + (d) * xdim95 * ydim95)
#define OPS_ACC_MD96(d, x, y) ((x) + (xdim96 * (y)) + (d) * xdim96 * ydim96)
#define OPS_ACC_MD97(d, x, y) ((x) + (xdim97 * (y)) + (d) * xdim97 * ydim97)
#define OPS_ACC_MD98(d, x, y) ((x) + (xdim98 * (y)) + (d) * xdim98 * ydim98)
#define OPS_ACC_MD99(d, x, y) ((x) + (xdim99 * (y)) + (d) * xdim99 * ydim99)
#endif
#else
/// TODO #define OPS_ACC0(x,y) (ops_stencil_check_2d_md(0, x, -1, -1))
///--> ops_stencil_check_2d_md(int arg_idx, int idx0, int idx1, int dim0, int
/// dim1, int mult_d, int d);
#endif
#else             // macros for 1D application
#ifndef OPS_DEBUG // no debug checks
#define OPS_ACC_MD0(d, x) ((x)*multi_d0 + d)
#define OPS_ACC_MD1(d, x) ((x)*multi_d1 + d)
#define OPS_ACC_MD2(d, x) ((x)*multi_d2 + d)
#define OPS_ACC_MD3(d, x) ((x)*multi_d3 + d)
#define OPS_ACC_MD4(d, x) ((x)*multi_d4 + d)
#define OPS_ACC_MD5(d, x) ((x)*multi_d5 + d)
#define OPS_ACC_MD6(d, x) ((x)*multi_d6 + d)
#define OPS_ACC_MD7(d, x) ((x)*multi_d7 + d)
#define OPS_ACC_MD8(d, x) ((x)*multi_d8 + d)
#define OPS_ACC_MD9(d, x) ((x)*multi_d9 + d)
#define OPS_ACC_MD10(d, x) ((x)*multi_d10 + d)
#define OPS_ACC_MD11(d, x) ((x)*multi_d11 + d)
#define OPS_ACC_MD12(d, x) ((x)*multi_d12 + d)
#define OPS_ACC_MD13(d, x) ((x)*multi_d13 + d)
#define OPS_ACC_MD14(d, x) ((x)*multi_d14 + d)
#define OPS_ACC_MD15(d, x) ((x)*multi_d15 + d)
#define OPS_ACC_MD16(d, x) ((x)*multi_d16 + d)
#define OPS_ACC_MD17(d, x) ((x)*multi_d17 + d)
#define OPS_ACC_MD18(d, x) ((x)*multi_d18 + d)
#define OPS_ACC_MD19(d, x) ((x)*multi_d19 + d)
#define OPS_ACC_MD20(d, x) ((x)*multi_d20 + d)
#define OPS_ACC_MD21(d, x) ((x)*multi_d21 + d)
#define OPS_ACC_MD22(d, x) ((x)*multi_d22 + d)
#define OPS_ACC_MD23(d, x) ((x)*multi_d23 + d)
#define OPS_ACC_MD24(d, x) ((x)*multi_d24 + d)
#define OPS_ACC_MD25(d, x) ((x)*multi_d25 + d)
#define OPS_ACC_MD26(d, x) ((x)*multi_d26 + d)
#define OPS_ACC_MD27(d, x) ((x)*multi_d27 + d)
#define OPS_ACC_MD28(d, x) ((x)*multi_d28 + d)
#define OPS_ACC_MD29(d, x) ((x)*multi_d29 + d)
#define OPS_ACC_MD30(d, x) ((x)*multi_d30 + d)
#define OPS_ACC_MD31(d, x) ((x)*multi_d31 + d)
#define OPS_ACC_MD32(d, x) ((x)*multi_d32 + d)
#define OPS_ACC_MD33(d, x) ((x)*multi_d33 + d)
#define OPS_ACC_MD34(d, x) ((x)*multi_d34 + d)
#define OPS_ACC_MD35(d, x) ((x)*multi_d35 + d)
#define OPS_ACC_MD36(d, x) ((x)*multi_d36 + d)
#define OPS_ACC_MD37(d, x) ((x)*multi_d37 + d)
#define OPS_ACC_MD38(d, x) ((x)*multi_d38 + d)
#define OPS_ACC_MD39(d, x) ((x)*multi_d39 + d)
#define OPS_ACC_MD40(d, x) ((x)*multi_d40 + d)
#define OPS_ACC_MD41(d, x) ((x)*multi_d41 + d)
#define OPS_ACC_MD42(d, x) ((x)*multi_d42 + d)
#define OPS_ACC_MD43(d, x) ((x)*multi_d43 + d)
#define OPS_ACC_MD44(d, x) ((x)*multi_d44 + d)
#define OPS_ACC_MD45(d, x) ((x)*multi_d45 + d)
#define OPS_ACC_MD46(d, x) ((x)*multi_d46 + d)
#define OPS_ACC_MD47(d, x) ((x)*multi_d47 + d)
#define OPS_ACC_MD48(d, x) ((x)*multi_d48 + d)
#define OPS_ACC_MD49(d, x) ((x)*multi_d49 + d)
#define OPS_ACC_MD50(d, x) ((x)*multi_d50 + d)
#define OPS_ACC_MD51(d, x) ((x)*multi_d51 + d)
#define OPS_ACC_MD52(d, x) ((x)*multi_d52 + d)
#define OPS_ACC_MD53(d, x) ((x)*multi_d53 + d)
#define OPS_ACC_MD54(d, x) ((x)*multi_d54 + d)
#define OPS_ACC_MD55(d, x) ((x)*multi_d55 + d)
#define OPS_ACC_MD56(d, x) ((x)*multi_d56 + d)
#define OPS_ACC_MD57(d, x) ((x)*multi_d57 + d)
#define OPS_ACC_MD58(d, x) ((x)*multi_d58 + d)
#define OPS_ACC_MD59(d, x) ((x)*multi_d59 + d)
#define OPS_ACC_MD60(d, x) ((x)*multi_d60 + d)
#define OPS_ACC_MD61(d, x) ((x)*multi_d61 + d)
#define OPS_ACC_MD62(d, x) ((x)*multi_d62 + d)
#define OPS_ACC_MD63(d, x) ((x)*multi_d63 + d)
#define OPS_ACC_MD64(d, x) ((x)*multi_d64 + d)
#define OPS_ACC_MD65(d, x) ((x)*multi_d65 + d)
#define OPS_ACC_MD66(d, x) ((x)*multi_d66 + d)
#define OPS_ACC_MD67(d, x) ((x)*multi_d67 + d)
#define OPS_ACC_MD68(d, x) ((x)*multi_d68 + d)
#define OPS_ACC_MD69(d, x) ((x)*multi_d69 + d)
#define OPS_ACC_MD70(d, x) ((x)*multi_d70 + d)
#define OPS_ACC_MD71(d, x) ((x)*multi_d71 + d)
#define OPS_ACC_MD72(d, x) ((x)*multi_d72 + d)
#define OPS_ACC_MD73(d, x) ((x)*multi_d73 + d)
#define OPS_ACC_MD74(d, x) ((x)*multi_d74 + d)
#define OPS_ACC_MD75(d, x) ((x)*multi_d75 + d)
#define OPS_ACC_MD76(d, x) ((x)*multi_d76 + d)
#define OPS_ACC_MD77(d, x) ((x)*multi_d77 + d)
#define OPS_ACC_MD78(d, x) ((x)*multi_d78 + d)
#define OPS_ACC_MD79(d, x) ((x)*multi_d79 + d)
#define OPS_ACC_MD80(d, x) ((x)*multi_d80 + d)
#define OPS_ACC_MD81(d, x) ((x)*multi_d81 + d)
#define OPS_ACC_MD82(d, x) ((x)*multi_d82 + d)
#define OPS_ACC_MD83(d, x) ((x)*multi_d83 + d)
#define OPS_ACC_MD84(d, x) ((x)*multi_d84 + d)
#define OPS_ACC_MD85(d, x) ((x)*multi_d85 + d)
#define OPS_ACC_MD86(d, x) ((x)*multi_d86 + d)
#define OPS_ACC_MD87(d, x) ((x)*multi_d87 + d)
#define OPS_ACC_MD88(d, x) ((x)*multi_d88 + d)
#define OPS_ACC_MD89(d, x) ((x)*multi_d89 + d)
#define OPS_ACC_MD90(d, x) ((x)*multi_d90 + d)
#define OPS_ACC_MD91(d, x) ((x)*multi_d91 + d)
#define OPS_ACC_MD92(d, x) ((x)*multi_d92 + d)
#define OPS_ACC_MD93(d, x) ((x)*multi_d93 + d)
#define OPS_ACC_MD94(d, x) ((x)*multi_d94 + d)
#define OPS_ACC_MD95(d, x) ((x)*multi_d95 + d)
#define OPS_ACC_MD96(d, x) ((x)*multi_d96 + d)
#define OPS_ACC_MD97(d, x) ((x)*multi_d97 + d)
#define OPS_ACC_MD98(d, x) ((x)*multi_d98 + d)
#define OPS_ACC_MD99(d, x) ((x)*multi_d99 + d)
#else
#define OPS_ACC_MD0(d, x) (ops_stencil_check_1d_md(0, x, multi_d0, d))
#define OPS_ACC_MD1(d, x) (ops_stencil_check_1d_md(1, x, multi_d1, d))
#define OPS_ACC_MD2(d, x) (ops_stencil_check_1d_md(2, x, multi_d2, d))
#define OPS_ACC_MD3(d, x) (ops_stencil_check_1d_md(3, x, multi_d3, d))
#define OPS_ACC_MD4(d, x) (ops_stencil_check_1d_md(4, x, multi_d4, d))
#define OPS_ACC_MD5(d, x) (ops_stencil_check_1d_md(5, x, multi_d5, d))
#define OPS_ACC_MD6(d, x) (ops_stencil_check_1d_md(6, x, multi_d6, d))
#define OPS_ACC_MD7(d, x) (ops_stencil_check_1d_md(7, x, multi_d7, d))
#define OPS_ACC_MD8(d, x) (ops_stencil_check_1d_md(8, x, multi_d8, d))
#define OPS_ACC_MD9(d, x) (ops_stencil_check_1d_md(9, x, multi_d9, d))
#define OPS_ACC_MD10(d, x) (ops_stencil_check_1d_md(10, x, multi_d10, d))
#define OPS_ACC_MD11(d, x) (ops_stencil_check_1d_md(11, x, multi_d11, d))
#define OPS_ACC_MD12(d, x) (ops_stencil_check_1d_md(12, x, multi_d12, d))
#define OPS_ACC_MD13(d, x) (ops_stencil_check_1d_md(13, x, multi_d13, d))
#define OPS_ACC_MD14(d, x) (ops_stencil_check_1d_md(14, x, multi_d14, d))
#define OPS_ACC_MD15(d, x) (ops_stencil_check_1d_md(15, x, multi_d15, d))
#define OPS_ACC_MD16(d, x) (ops_stencil_check_1d_md(16, x, multi_d16, d))
#define OPS_ACC_MD17(d, x) (ops_stencil_check_1d_md(17, x, multi_d17, d))
#define OPS_ACC_MD18(d, x) (ops_stencil_check_1d_md(18, x, multi_d18, d))
#define OPS_ACC_MD19(d, x) (ops_stencil_check_1d_md(19, x, multi_d19, d))
#define OPS_ACC_MD20(d, x) (ops_stencil_check_1d_md(20, x, multi_d20, d))
#define OPS_ACC_MD21(d, x) (ops_stencil_check_1d_md(21, x, multi_d21, d))
#define OPS_ACC_MD22(d, x) (ops_stencil_check_1d_md(22, x, multi_d22, d))
#define OPS_ACC_MD23(d, x) (ops_stencil_check_1d_md(23, x, multi_d23, d))
#define OPS_ACC_MD24(d, x) (ops_stencil_check_1d_md(24, x, multi_d24, d))
#define OPS_ACC_MD25(d, x) (ops_stencil_check_1d_md(25, x, multi_d25, d))
#define OPS_ACC_MD26(d, x) (ops_stencil_check_1d_md(26, x, multi_d26, d))
#define OPS_ACC_MD27(d, x) (ops_stencil_check_1d_md(27, x, multi_d27, d))
#define OPS_ACC_MD28(d, x) (ops_stencil_check_1d_md(28, x, multi_d28, d))
#define OPS_ACC_MD29(d, x) (ops_stencil_check_1d_md(29, x, multi_d29, d))
#define OPS_ACC_MD30(d, x) (ops_stencil_check_1d_md(30, x, multi_d30, d))
#define OPS_ACC_MD31(d, x) (ops_stencil_check_1d_md(31, x, multi_d31, d))
#define OPS_ACC_MD32(d, x) (ops_stencil_check_1d_md(32, x, multi_d32, d))
#define OPS_ACC_MD33(d, x) (ops_stencil_check_1d_md(33, x, multi_d33, d))
#define OPS_ACC_MD34(d, x) (ops_stencil_check_1d_md(34, x, multi_d34, d))
#define OPS_ACC_MD35(d, x) (ops_stencil_check_1d_md(35, x, multi_d35, d))
#define OPS_ACC_MD36(d, x) (ops_stencil_check_1d_md(36, x, multi_d36, d))
#define OPS_ACC_MD37(d, x) (ops_stencil_check_1d_md(37, x, multi_d37, d))
#define OPS_ACC_MD38(d, x) (ops_stencil_check_1d_md(38, x, multi_d38, d))
#define OPS_ACC_MD39(d, x) (ops_stencil_check_1d_md(39, x, multi_d39, d))
#define OPS_ACC_MD40(d, x) (ops_stencil_check_1d_md(40, x, multi_d40, d))
#define OPS_ACC_MD41(d, x) (ops_stencil_check_1d_md(41, x, multi_d41, d))
#define OPS_ACC_MD42(d, x) (ops_stencil_check_1d_md(42, x, multi_d42, d))
#define OPS_ACC_MD43(d, x) (ops_stencil_check_1d_md(43, x, multi_d43, d))
#define OPS_ACC_MD44(d, x) (ops_stencil_check_1d_md(44, x, multi_d44, d))
#define OPS_ACC_MD45(d, x) (ops_stencil_check_1d_md(45, x, multi_d45, d))
#define OPS_ACC_MD46(d, x) (ops_stencil_check_1d_md(46, x, multi_d46, d))
#define OPS_ACC_MD47(d, x) (ops_stencil_check_1d_md(47, x, multi_d47, d))
#define OPS_ACC_MD48(d, x) (ops_stencil_check_1d_md(48, x, multi_d48, d))
#define OPS_ACC_MD49(d, x) (ops_stencil_check_1d_md(49, x, multi_d49, d))
#define OPS_ACC_MD50(d, x) (ops_stencil_check_1d_md(50, x, multi_d50, d))
#define OPS_ACC_MD51(d, x) (ops_stencil_check_1d_md(51, x, multi_d51, d))
#define OPS_ACC_MD52(d, x) (ops_stencil_check_1d_md(52, x, multi_d52, d))
#define OPS_ACC_MD53(d, x) (ops_stencil_check_1d_md(53, x, multi_d53, d))
#define OPS_ACC_MD54(d, x) (ops_stencil_check_1d_md(54, x, multi_d54, d))
#define OPS_ACC_MD55(d, x) (ops_stencil_check_1d_md(55, x, multi_d55, d))
#define OPS_ACC_MD56(d, x) (ops_stencil_check_1d_md(56, x, multi_d56, d))
#define OPS_ACC_MD57(d, x) (ops_stencil_check_1d_md(57, x, multi_d57, d))
#define OPS_ACC_MD58(d, x) (ops_stencil_check_1d_md(58, x, multi_d58, d))
#define OPS_ACC_MD59(d, x) (ops_stencil_check_1d_md(59, x, multi_d59, d))
#define OPS_ACC_MD60(d, x) (ops_stencil_check_1d_md(60, x, multi_d60, d))
#define OPS_ACC_MD61(d, x) (ops_stencil_check_1d_md(61, x, multi_d61, d))
#define OPS_ACC_MD62(d, x) (ops_stencil_check_1d_md(62, x, multi_d62, d))
#define OPS_ACC_MD63(d, x) (ops_stencil_check_1d_md(63, x, multi_d63, d))
#define OPS_ACC_MD64(d, x) (ops_stencil_check_1d_md(64, x, multi_d64, d))
#define OPS_ACC_MD65(d, x) (ops_stencil_check_1d_md(65, x, multi_d65, d))
#define OPS_ACC_MD66(d, x) (ops_stencil_check_1d_md(66, x, multi_d66, d))
#define OPS_ACC_MD67(d, x) (ops_stencil_check_1d_md(67, x, multi_d67, d))
#define OPS_ACC_MD68(d, x) (ops_stencil_check_1d_md(68, x, multi_d68, d))
#define OPS_ACC_MD69(d, x) (ops_stencil_check_1d_md(69, x, multi_d69, d))
#define OPS_ACC_MD70(d, x) (ops_stencil_check_1d_md(70, x, multi_d70, d))
#define OPS_ACC_MD71(d, x) (ops_stencil_check_1d_md(71, x, multi_d71, d))
#define OPS_ACC_MD72(d, x) (ops_stencil_check_1d_md(72, x, multi_d72, d))
#define OPS_ACC_MD73(d, x) (ops_stencil_check_1d_md(73, x, multi_d73, d))
#define OPS_ACC_MD74(d, x) (ops_stencil_check_1d_md(74, x, multi_d74, d))
#define OPS_ACC_MD75(d, x) (ops_stencil_check_1d_md(75, x, multi_d75, d))
#define OPS_ACC_MD76(d, x) (ops_stencil_check_1d_md(76, x, multi_d76, d))
#define OPS_ACC_MD77(d, x) (ops_stencil_check_1d_md(77, x, multi_d77, d))
#define OPS_ACC_MD78(d, x) (ops_stencil_check_1d_md(78, x, multi_d78, d))
#define OPS_ACC_MD79(d, x) (ops_stencil_check_1d_md(79, x, multi_d79, d))
#define OPS_ACC_MD80(d, x) (ops_stencil_check_1d_md(80, x, multi_d80, d))
#define OPS_ACC_MD81(d, x) (ops_stencil_check_1d_md(81, x, multi_d81, d))
#define OPS_ACC_MD82(d, x) (ops_stencil_check_1d_md(82, x, multi_d82, d))
#define OPS_ACC_MD83(d, x) (ops_stencil_check_1d_md(83, x, multi_d83, d))
#define OPS_ACC_MD84(d, x) (ops_stencil_check_1d_md(84, x, multi_d84, d))
#define OPS_ACC_MD85(d, x) (ops_stencil_check_1d_md(85, x, multi_d85, d))
#define OPS_ACC_MD86(d, x) (ops_stencil_check_1d_md(86, x, multi_d86, d))
#define OPS_ACC_MD87(d, x) (ops_stencil_check_1d_md(87, x, multi_d87, d))
#define OPS_ACC_MD88(d, x) (ops_stencil_check_1d_md(88, x, multi_d88, d))
#define OPS_ACC_MD89(d, x) (ops_stencil_check_1d_md(89, x, multi_d89, d))
#define OPS_ACC_MD90(d, x) (ops_stencil_check_1d_md(90, x, multi_d90, d))
#define OPS_ACC_MD91(d, x) (ops_stencil_check_1d_md(91, x, multi_d91, d))
#define OPS_ACC_MD92(d, x) (ops_stencil_check_1d_md(92, x, multi_d92, d))
#define OPS_ACC_MD93(d, x) (ops_stencil_check_1d_md(93, x, multi_d93, d))
#define OPS_ACC_MD94(d, x) (ops_stencil_check_1d_md(94, x, multi_d94, d))
#define OPS_ACC_MD95(d, x) (ops_stencil_check_1d_md(95, x, multi_d95, d))
#define OPS_ACC_MD96(d, x) (ops_stencil_check_1d_md(96, x, multi_d96, d))
#define OPS_ACC_MD97(d, x) (ops_stencil_check_1d_md(97, x, multi_d97, d))
#define OPS_ACC_MD98(d, x) (ops_stencil_check_1d_md(98, x, multi_d98, d))
#define OPS_ACC_MD99(d, x) (ops_stencil_check_1d_md(99, x, multi_d99, d))
#endif
#endif
#endif

/**--------------------------Set SIMD Vector lenght--------------------------**/
#ifndef SIMD_VEC
#define SIMD_VEC 4
#endif

#ifndef ROUND_DOWN
#define ROUND_DOWN(N, step) (((N) / (step)) * (step))
#endif

extern int xdim0, xdim1, xdim2, xdim3, xdim4, xdim5, xdim6, xdim7, xdim8, xdim9,
    xdim10, xdim11, xdim12, xdim13, xdim14, xdim15, xdim16, xdim17, xdim18,
    xdim19, xdim20, xdim21, xdim22, xdim23, xdim24, xdim25, xdim26, xdim27,
    xdim28, xdim29, xdim30, xdim31, xdim32, xdim33, xdim34, xdim35, xdim36,
    xdim37, xdim38, xdim39, xdim40, xdim41, xdim42, xdim43, xdim44, xdim45,
    xdim46, xdim47, xdim48, xdim49, xdim50, xdim51, xdim52, xdim53, xdim54,
    xdim55, xdim56, xdim57, xdim58, xdim59, xdim60, xdim61, xdim62, xdim63,
    xdim64, xdim65, xdim66, xdim67, xdim68, xdim69, xdim70, xdim71, xdim72,
    xdim73, xdim74, xdim75, xdim76, xdim77, xdim78, xdim79, xdim80, xdim81,
    xdim82, xdim83, xdim84, xdim85, xdim86, xdim87, xdim88, xdim89, xdim90,
    xdim91, xdim92, xdim93, xdim94, xdim95, xdim96, xdim97, xdim98, xdim99;

#if defined OPS_3D || defined OPS_SOA
extern int ydim0, ydim1, ydim2, ydim3, ydim4, ydim5, ydim6, ydim7, ydim8, ydim9,
    ydim10, ydim11, ydim12, ydim13, ydim14, ydim15, ydim16, ydim17, ydim18,
    ydim19, ydim20, ydim21, ydim22, ydim23, ydim24, ydim25, ydim26, ydim27,
    ydim28, ydim29, ydim30, ydim31, ydim32, ydim33, ydim34, ydim35, ydim36,
    ydim37, ydim38, ydim39, ydim40, ydim41, ydim42, ydim43, ydim44, ydim45,
    ydim46, ydim47, ydim48, ydim49, ydim50, ydim51, ydim52, ydim53, ydim54,
    ydim55, ydim56, ydim57, ydim58, ydim59, ydim60, ydim61, ydim62, ydim63,
    ydim64, ydim65, ydim66, ydim67, ydim68, ydim69, ydim70, ydim71, ydim72,
    ydim73, ydim74, ydim75, ydim76, ydim77, ydim78, ydim79, ydim80, ydim81,
    ydim82, ydim83, ydim84, ydim85, ydim86, ydim87, ydim88, ydim89, ydim90,
    ydim91, ydim92, ydim93, ydim94, ydim95, ydim96, ydim97, ydim98, ydim99;

#endif

#if defined OPS_3D && defined OPS_SOA
extern int zdim0, zdim1, zdim2, zdim3, zdim4, zdim5, zdim6, zdim7, zdim8, zdim9,
    zdim10, zdim11, zdim12, zdim13, zdim14, zdim15, zdim16, zdim17, zdim18,
    zdim19, zdim20, zdim21, zdim22, zdim23, zdim24, zdim25, zdim26, zdim27,
    zdim28, zdim29, zdim30, zdim31, zdim32, zdim33, zdim34, zdim35, zdim36,
    zdim37, zdim38, zdim39, zdim40, zdim41, zdim42, zdim43, zdim44, zdim45,
    zdim46, zdim47, zdim48, zdim49, zdim50, zdim51, zdim52, zdim53, zdim54,
    zdim55, zdim56, zdim57, zdim58, zdim59, zdim60, zdim61, zdim62, zdim63,
    zdim64, zdim65, zdim66, zdim67, zdim68, zdim69, zdim70, zdim71, zdim72,
    zdim73, zdim74, zdim75, zdim76, zdim77, zdim78, zdim79, zdim80, zdim81,
    zdim82, zdim83, zdim84, zdim85, zdim86, zdim87, zdim88, zdim89, zdim90,
    zdim91, zdim92, zdim93, zdim94, zdim95, zdim96, zdim97, zdim98, zdim99;

#endif

#ifndef OPS_SOA
extern int multi_d0, multi_d1, multi_d2, multi_d3, multi_d4, multi_d5, multi_d6,
    multi_d7, multi_d8, multi_d9, multi_d10, multi_d11, multi_d12, multi_d13,
    multi_d14, multi_d15, multi_d16, multi_d17, multi_d18, multi_d19, multi_d20,
    multi_d21, multi_d22, multi_d23, multi_d24, multi_d25, multi_d26, multi_d27,
    multi_d28, multi_d29, multi_d30, multi_d31, multi_d32, multi_d33, multi_d34,
    multi_d35, multi_d36, multi_d37, multi_d38, multi_d39, multi_d40, multi_d41,
    multi_d42, multi_d43, multi_d44, multi_d45, multi_d46, multi_d47, multi_d48,
    multi_d49, multi_d50, multi_d51, multi_d52, multi_d53, multi_d54, multi_d55,
    multi_d56, multi_d57, multi_d58, multi_d59, multi_d60, multi_d61, multi_d62,
    multi_d63, multi_d64, multi_d65, multi_d66, multi_d67, multi_d68, multi_d69,
    multi_d70, multi_d71, multi_d72, multi_d73, multi_d74, multi_d75, multi_d76,
    multi_d77, multi_d78, multi_d79, multi_d80, multi_d81, multi_d82, multi_d83,
    multi_d84, multi_d85, multi_d86, multi_d87, multi_d88, multi_d89, multi_d90,
    multi_d91, multi_d92, multi_d93, multi_d94, multi_d95, multi_d96, multi_d97,
    multi_d98, multi_d99;
#endif
#endif // OPS_MACROS_H

