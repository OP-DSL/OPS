#ifndef OPS_MACROS_H
#define OPS_MACROS_H
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

/** @file
  * @brief header file declaring the functions for the ops mpi backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS macros
  */
#if OPS_API > 1

#ifdef OPS_2D 
#define OPS_ACCS(dat, x, y) (*(dat.ptr + (x) + (y)*dat.xdim))
#ifdef OPS_SOA
#define OPS_ACCM(dat, d, x, y) (*(dat.ptr + (x) + (y)*dat.xdim + (d) * dat.xdim * dat.ydim))
#else
#define OPS_ACCM(dat, d, x, y) (*(dat.ptr + (x) * dat.dim + (y)*dat.dim*dat.xdim))
#endif
#ifndef __OPENCL_VERSION__
#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME
#define OPS_ACC(...) GET_MACRO(__VA_ARGS__, OPS_ACCM, OPS_ACCS)(__VA_ARGS__)
#endif
#endif

#ifdef OPS_3D 
#define OPS_ACCS(dat, x, y, z) (*(dat.ptr + (x) + (y)*dat.xdim + (z)*dat.xdim*dat.ydim))
#ifdef OPS_SOA
#define OPS_ACCM(dat, d, x, y, z) (*(dat.ptr + (x) + (y)*dat.xdim + (z)*dat.xdim*dat.ydim + (d) * dat.xdim * dat.ydim * dat. zdim))
#else
#define OPS_ACCM(dat, d, x, y, z) (*(dat.ptr + (x) * dat.dim + (y)*dat.dim*dat.xdim + (z)*dat.dim*dat.xdim*dat.ydim))
#endif
#ifndef __OPENCL_VERSION__
#define GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME
#define OPS_ACC(...) GET_MACRO(__VA_ARGS__, OPS_ACCM, OPS_ACCS)(__VA_ARGS__)
#endif
#endif

#ifdef OPS_4D 
#define OPS_ACCS(dat, x, y, z, u) (*(dat.ptr + (x) + (y)*dat.xdim + (z)*dat.xdim*dat.ydim + (u)*dat.xdim*dat.ydim*dat.zdim))
#ifdef OPS_SOA
#define OPS_ACCM(dat, d, x, y, z, u) (*(dat.ptr + (x) + (y)*dat.xdim + (z)*dat.xdim*dat.ydim + (u)*dat.xdim*dat.ydim*dat.zdim + (d) * dat.xdim * dat.ydim * dat.zdim * dat.udim))
#else
#define OPS_ACCM(dat, d, x, y, z, u) (*(dat.ptr + (x) * dat.dim + (y)*dat.dim*dat.xdim + (z)*dat.dim*dat.xdim*dat.ydim + (u)*dat.dim+dat.xdim*dat.ydim*dat.zdim))
#endif
#ifndef __OPENCL_VERSION__
#define GET_MACRO(_1,_2,_3,_4,_5,_6,NAME,...) NAME
#define OPS_ACC(...) GET_MACRO(__VA_ARGS__, OPS_ACCM, OPS_ACCS)(__VA_ARGS__)
#endif
#endif

#ifdef OPS_5D 
#define OPS_ACCS(dat, x, y, z, u, v) (*(dat.ptr + (x) + (y)*dat.xdim + (z)*dat.xdim*dat.ydim + (u)*dat.xdim*dat.ydim*dat.zdim + (v)*dat.xdim*dat.ydim*dat.zdim*dat.udim))
#ifdef OPS_SOA
#define OPS_ACCM(dat, d, x, y, z, u, v) (*(dat.ptr + (x) + (y)*dat.xdim + (z)*dat.xdim*dat.ydim + (u)*dat.xdim*dat.ydim*dat.zdim + (v)*dat.xdim*dat.ydim*dat.zdim*dat.udim + (d) * dat.xdim * dat.ydim * dat.zdim * dat.udim * dat.vdim))
#else
#define OPS_ACCM(dat, d, x, y, z, u, v) (*(dat.ptr + (x) * dat.dim + (y)*dat.dim*dat.xdim + (z)*dat.dim*dat.xdim*dat.ydim + (u)*dat.dim+dat.xdim*dat.ydim*dat.zdim + (v)*dat.dim*dat.xdim*dat.ydim*dat.zdim*dat.udim))
#endif
#ifndef __OPENCL_VERSION__
#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,NAME,...) NAME
#define OPS_ACC(...) GET_MACRO(__VA_ARGS__, OPS_ACCM, OPS_ACCS)(__VA_ARGS__)
#endif
#endif

#ifndef __cplusplus
typedef struct ptr_double {
#ifdef __OPENCL_VERSION__
  __global
#endif
  double *restrict ptr;
  int xdim;
#if defined(OPS_3D) || defined(OPS_4D) || defined(OPS_5D)
  int ydim;
#endif
#if defined(OPS_4D) || defined(OPS_5D)
  int zdim;
#endif
#if defined(OPS_5D)
  int udim;
#endif
} ptr_double;

typedef struct ptr_int {
#ifdef __OPENCL_VERSION__
  __global
#endif
  int *restrict ptr;
  int xdim;
#if defined(OPS_3D) || defined(OPS_4D) || defined(OPS_5D)
  int ydim;
#endif
#if defined(OPS_4D) || defined(OPS_5D)
  int zdim;
#endif
#if defined(OPS_5D)
  int udim;
#endif
} ptr_int;

typedef struct ptrm_double {
#ifdef __OPENCL_VERSION__
  __global
#endif
  double *restrict ptr;
  int xdim;
#if (defined(OPS_2D) && defined(OPS_SOA)) || defined(OPS_3D) || defined(OPS_4D) || defined(OPS_5D)
  int ydim;
#endif
#if (defined(OPS_3D) && defined(OPS_SOA)) || defined(OPS_4D) || defined(OPS_5D)
  int zdim;
#endif
#if (defined(OPS_4D) && defined(OPS_SOA)) || defined(OPS_5D)
  int udim;
#endif
#ifndef OPS_SOA
  int dim;
#endif
} ptrm_double;

typedef struct ptrm_int {
#ifdef __OPENCL_VERSION__
  __global
#endif
  int *restrict ptr;
  int xdim;
#if (defined(OPS_2D) && defined(OPS_SOA)) || defined(OPS_3D) || defined(OPS_4D) || defined(OPS_5D)
  int ydim;
#endif
#if (defined(OPS_3D) && defined(OPS_SOA)) || defined(OPS_4D) || defined(OPS_5D)
  int zdim;
#endif
#if (defined(OPS_4D) && defined(OPS_SOA)) || defined(OPS_5D)
  int udim;
#endif
#ifndef OPS_SOA
  int dim;
#endif
} ptrm_int;

#endif

#else /* OPS_API */

#if defined OPS_5D
#ifndef OPS_DEBUG // no debug checks
#define OPS_ACC0(x, y, z, u, v)  (x + xdim0  * (y) + ydim0  * xdim0  * (z) + zdim0  * ydim0  * xdim0  * (u) + udim0  * zdim0  * ydim0  * xdim0  * (v))
#define OPS_ACC1(x, y, z, u, v)  (x + xdim1  * (y) + ydim1  * xdim1  * (z) + zdim1  * ydim1  * xdim1  * (u) + udim1  * zdim1  * ydim1  * xdim1  * (v))
#define OPS_ACC2(x, y, z, u, v)  (x + xdim2  * (y) + ydim2  * xdim2  * (z) + zdim2  * ydim2  * xdim2  * (u) + udim2  * zdim2  * ydim2  * xdim2  * (v))
#define OPS_ACC3(x, y, z, u, v)  (x + xdim3  * (y) + ydim3  * xdim3  * (z) + zdim3  * ydim3  * xdim3  * (u) + udim3  * zdim3  * ydim3  * xdim3  * (v))
#define OPS_ACC4(x, y, z, u, v)  (x + xdim4  * (y) + ydim4  * xdim4  * (z) + zdim4  * ydim4  * xdim4  * (u) + udim4  * zdim4  * ydim4  * xdim4  * (v))
#define OPS_ACC5(x, y, z, u, v)  (x + xdim5  * (y) + ydim5  * xdim5  * (z) + zdim5  * ydim5  * xdim5  * (u) + udim5  * zdim5  * ydim5  * xdim5  * (v))
#define OPS_ACC6(x, y, z, u, v)  (x + xdim6  * (y) + ydim6  * xdim6  * (z) + zdim6  * ydim6  * xdim6  * (u) + udim6  * zdim6  * ydim6  * xdim6  * (v))
#define OPS_ACC7(x, y, z, u, v)  (x + xdim7  * (y) + ydim7  * xdim7  * (z) + zdim7  * ydim7  * xdim7  * (u) + udim7  * zdim7  * ydim7  * xdim7  * (v))
#define OPS_ACC8(x, y, z, u, v)  (x + xdim8  * (y) + ydim8  * xdim8  * (z) + zdim8  * ydim8  * xdim8  * (u) + udim8  * zdim8  * ydim8  * xdim8  * (v))
#define OPS_ACC9(x, y, z, u, v)  (x + xdim9  * (y) + ydim9  * xdim9  * (z) + zdim9  * ydim9  * xdim9  * (u) + udim9  * zdim9  * ydim9  * xdim9  * (v))
#define OPS_ACC10(x, y, z, u, v) (x + xdim10 * (y) + ydim10 * xdim10 * (z) + zdim10 * ydim10 * xdim10 * (u) + udim10 * zdim10 * ydim10 * xdim10 * (v))
#define OPS_ACC11(x, y, z, u, v) (x + xdim11 * (y) + ydim11 * xdim11 * (z) + zdim11 * ydim11 * xdim11 * (u) + udim11 * zdim11 * ydim11 * xdim11 * (v))
#define OPS_ACC12(x, y, z, u, v) (x + xdim12 * (y) + ydim12 * xdim12 * (z) + zdim12 * ydim12 * xdim12 * (u) + udim12 * zdim12 * ydim12 * xdim12 * (v))
#define OPS_ACC13(x, y, z, u, v) (x + xdim13 * (y) + ydim13 * xdim13 * (z) + zdim13 * ydim13 * xdim13 * (u) + udim13 * zdim13 * ydim13 * xdim13 * (v))
#define OPS_ACC14(x, y, z, u, v) (x + xdim14 * (y) + ydim14 * xdim14 * (z) + zdim14 * ydim14 * xdim14 * (u) + udim14 * zdim14 * ydim14 * xdim14 * (v))
#define OPS_ACC15(x, y, z, u, v) (x + xdim15 * (y) + ydim15 * xdim15 * (z) + zdim15 * ydim15 * xdim15 * (u) + udim15 * zdim15 * ydim15 * xdim15 * (v))
#define OPS_ACC16(x, y, z, u, v) (x + xdim16 * (y) + ydim16 * xdim16 * (z) + zdim16 * ydim16 * xdim16 * (u) + udim16 * zdim16 * ydim16 * xdim16 * (v))
#define OPS_ACC17(x, y, z, u, v) (x + xdim17 * (y) + ydim17 * xdim17 * (z) + zdim17 * ydim17 * xdim17 * (u) + udim17 * zdim17 * ydim17 * xdim17 * (v))
#define OPS_ACC18(x, y, z, u, v) (x + xdim18 * (y) + ydim18 * xdim18 * (z) + zdim18 * ydim18 * xdim18 * (u) + udim18 * zdim18 * ydim18 * xdim18 * (v))
#define OPS_ACC19(x, y, z, u, v) (x + xdim19 * (y) + ydim19 * xdim19 * (z) + zdim19 * ydim19 * xdim19 * (u) + udim19 * zdim19 * ydim19 * xdim19 * (v))
#define OPS_ACC20(x, y, z, u, v) (x + xdim20 * (y) + ydim20 * xdim20 * (z) + zdim20 * ydim20 * xdim20 * (u) + udim20 * zdim20 * ydim20 * xdim20 * (v))
#define OPS_ACC21(x, y, z, u, v) (x + xdim21 * (y) + ydim21 * xdim21 * (z) + zdim21 * ydim21 * xdim21 * (u) + udim21 * zdim21 * ydim21 * xdim21 * (v))
#define OPS_ACC22(x, y, z, u, v) (x + xdim22 * (y) + ydim22 * xdim22 * (z) + zdim22 * ydim22 * xdim22 * (u) + udim22 * zdim22 * ydim22 * xdim22 * (v))
#define OPS_ACC23(x, y, z, u, v) (x + xdim23 * (y) + ydim23 * xdim23 * (z) + zdim23 * ydim23 * xdim23 * (u) + udim23 * zdim23 * ydim23 * xdim23 * (v))
#define OPS_ACC24(x, y, z, u, v) (x + xdim24 * (y) + ydim24 * xdim24 * (z) + zdim24 * ydim24 * xdim24 * (u) + udim24 * zdim24 * ydim24 * xdim24 * (v))
#define OPS_ACC25(x, y, z, u, v) (x + xdim25 * (y) + ydim25 * xdim25 * (z) + zdim25 * ydim25 * xdim25 * (u) + udim25 * zdim25 * ydim25 * xdim25 * (v))
#define OPS_ACC26(x, y, z, u, v) (x + xdim26 * (y) + ydim26 * xdim26 * (z) + zdim26 * ydim26 * xdim26 * (u) + udim26 * zdim26 * ydim26 * xdim26 * (v))
#define OPS_ACC27(x, y, z, u, v) (x + xdim27 * (y) + ydim27 * xdim27 * (z) + zdim27 * ydim27 * xdim27 * (u) + udim27 * zdim27 * ydim27 * xdim27 * (v))
#define OPS_ACC28(x, y, z, u, v) (x + xdim28 * (y) + ydim28 * xdim28 * (z) + zdim28 * ydim28 * xdim28 * (u) + udim28 * zdim28 * ydim28 * xdim28 * (v))
#define OPS_ACC29(x, y, z, u, v) (x + xdim29 * (y) + ydim29 * xdim29 * (z) + zdim29 * ydim29 * xdim29 * (u) + udim29 * zdim29 * ydim29 * xdim29 * (v))
#define OPS_ACC30(x, y, z, u, v) (x + xdim30 * (y) + ydim30 * xdim30 * (z) + zdim30 * ydim30 * xdim30 * (u) + udim30 * zdim30 * ydim30 * xdim30 * (v))
#define OPS_ACC31(x, y, z, u, v) (x + xdim31 * (y) + ydim31 * xdim31 * (z) + zdim31 * ydim31 * xdim31 * (u) + udim31 * zdim31 * ydim31 * xdim31 * (v))
#define OPS_ACC32(x, y, z, u, v) (x + xdim32 * (y) + ydim32 * xdim32 * (z) + zdim32 * ydim32 * xdim32 * (u) + udim32 * zdim32 * ydim32 * xdim32 * (v))
#define OPS_ACC33(x, y, z, u, v) (x + xdim33 * (y) + ydim33 * xdim33 * (z) + zdim33 * ydim33 * xdim33 * (u) + udim33 * zdim33 * ydim33 * xdim33 * (v))
#define OPS_ACC34(x, y, z, u, v) (x + xdim34 * (y) + ydim34 * xdim34 * (z) + zdim34 * ydim34 * xdim34 * (u) + udim34 * zdim34 * ydim34 * xdim34 * (v))
#define OPS_ACC35(x, y, z, u, v) (x + xdim35 * (y) + ydim35 * xdim35 * (z) + zdim35 * ydim35 * xdim35 * (u) + udim35 * zdim35 * ydim35 * xdim35 * (v))
#define OPS_ACC36(x, y, z, u, v) (x + xdim36 * (y) + ydim36 * xdim36 * (z) + zdim36 * ydim36 * xdim36 * (u) + udim36 * zdim36 * ydim36 * xdim36 * (v))
#define OPS_ACC37(x, y, z, u, v) (x + xdim37 * (y) + ydim37 * xdim37 * (z) + zdim37 * ydim37 * xdim37 * (u) + udim37 * zdim37 * ydim37 * xdim37 * (v))
#define OPS_ACC38(x, y, z, u, v) (x + xdim38 * (y) + ydim38 * xdim38 * (z) + zdim38 * ydim38 * xdim38 * (u) + udim38 * zdim38 * ydim38 * xdim38 * (v))
#define OPS_ACC39(x, y, z, u, v) (x + xdim39 * (y) + ydim39 * xdim39 * (z) + zdim39 * ydim39 * xdim39 * (u) + udim39 * zdim39 * ydim39 * xdim39 * (v))
#define OPS_ACC40(x, y, z, u, v) (x + xdim40 * (y) + ydim40 * xdim40 * (z) + zdim40 * ydim40 * xdim40 * (u) + udim40 * zdim40 * ydim40 * xdim40 * (v))
#define OPS_ACC41(x, y, z, u, v) (x + xdim41 * (y) + ydim41 * xdim41 * (z) + zdim41 * ydim41 * xdim41 * (u) + udim41 * zdim41 * ydim41 * xdim41 * (v))
#define OPS_ACC42(x, y, z, u, v) (x + xdim42 * (y) + ydim42 * xdim42 * (z) + zdim42 * ydim42 * xdim42 * (u) + udim42 * zdim42 * ydim42 * xdim42 * (v))
#define OPS_ACC43(x, y, z, u, v) (x + xdim43 * (y) + ydim43 * xdim43 * (z) + zdim43 * ydim43 * xdim43 * (u) + udim43 * zdim43 * ydim43 * xdim43 * (v))
#define OPS_ACC44(x, y, z, u, v) (x + xdim44 * (y) + ydim44 * xdim44 * (z) + zdim44 * ydim44 * xdim44 * (u) + udim44 * zdim44 * ydim44 * xdim44 * (v))
#define OPS_ACC45(x, y, z, u, v) (x + xdim45 * (y) + ydim45 * xdim45 * (z) + zdim45 * ydim45 * xdim45 * (u) + udim45 * zdim45 * ydim45 * xdim45 * (v))
#define OPS_ACC46(x, y, z, u, v) (x + xdim46 * (y) + ydim46 * xdim46 * (z) + zdim46 * ydim46 * xdim46 * (u) + udim46 * zdim46 * ydim46 * xdim46 * (v))
#define OPS_ACC47(x, y, z, u, v) (x + xdim47 * (y) + ydim47 * xdim47 * (z) + zdim47 * ydim47 * xdim47 * (u) + udim47 * zdim47 * ydim47 * xdim47 * (v))
#define OPS_ACC48(x, y, z, u, v) (x + xdim48 * (y) + ydim48 * xdim48 * (z) + zdim48 * ydim48 * xdim48 * (u) + udim48 * zdim48 * ydim48 * xdim48 * (v))
#define OPS_ACC49(x, y, z, u, v) (x + xdim49 * (y) + ydim49 * xdim49 * (z) + zdim49 * ydim49 * xdim49 * (u) + udim49 * zdim49 * ydim49 * xdim49 * (v))
#define OPS_ACC50(x, y, z, u, v) (x + xdim50 * (y) + ydim50 * xdim50 * (z) + zdim50 * ydim50 * xdim50 * (u) + udim50 * zdim50 * ydim50 * xdim50 * (v))
#define OPS_ACC51(x, y, z, u, v) (x + xdim51 * (y) + ydim51 * xdim51 * (z) + zdim51 * ydim51 * xdim51 * (u) + udim51 * zdim51 * ydim51 * xdim51 * (v))
#define OPS_ACC52(x, y, z, u, v) (x + xdim52 * (y) + ydim52 * xdim52 * (z) + zdim52 * ydim52 * xdim52 * (u) + udim52 * zdim52 * ydim52 * xdim52 * (v))
#define OPS_ACC53(x, y, z, u, v) (x + xdim53 * (y) + ydim53 * xdim53 * (z) + zdim53 * ydim53 * xdim53 * (u) + udim53 * zdim53 * ydim53 * xdim53 * (v))
#define OPS_ACC54(x, y, z, u, v) (x + xdim54 * (y) + ydim54 * xdim54 * (z) + zdim54 * ydim54 * xdim54 * (u) + udim54 * zdim54 * ydim54 * xdim54 * (v))
#define OPS_ACC55(x, y, z, u, v) (x + xdim55 * (y) + ydim55 * xdim55 * (z) + zdim55 * ydim55 * xdim55 * (u) + udim55 * zdim55 * ydim55 * xdim55 * (v))
#define OPS_ACC56(x, y, z, u, v) (x + xdim56 * (y) + ydim56 * xdim56 * (z) + zdim56 * ydim56 * xdim56 * (u) + udim56 * zdim56 * ydim56 * xdim56 * (v))
#define OPS_ACC57(x, y, z, u, v) (x + xdim57 * (y) + ydim57 * xdim57 * (z) + zdim57 * ydim57 * xdim57 * (u) + udim57 * zdim57 * ydim57 * xdim57 * (v))
#define OPS_ACC58(x, y, z, u, v) (x + xdim58 * (y) + ydim58 * xdim58 * (z) + zdim58 * ydim58 * xdim58 * (u) + udim58 * zdim58 * ydim58 * xdim58 * (v))
#define OPS_ACC59(x, y, z, u, v) (x + xdim59 * (y) + ydim59 * xdim59 * (z) + zdim59 * ydim59 * xdim59 * (u) + udim59 * zdim59 * ydim59 * xdim59 * (v))
#define OPS_ACC60(x, y, z, u, v) (x + xdim60 * (y) + ydim60 * xdim60 * (z) + zdim60 * ydim60 * xdim60 * (u) + udim60 * zdim60 * ydim60 * xdim60 * (v))
#define OPS_ACC61(x, y, z, u, v) (x + xdim61 * (y) + ydim61 * xdim61 * (z) + zdim61 * ydim61 * xdim61 * (u) + udim61 * zdim61 * ydim61 * xdim61 * (v))
#define OPS_ACC62(x, y, z, u, v) (x + xdim62 * (y) + ydim62 * xdim62 * (z) + zdim62 * ydim62 * xdim62 * (u) + udim62 * zdim62 * ydim62 * xdim62 * (v))
#define OPS_ACC63(x, y, z, u, v) (x + xdim63 * (y) + ydim63 * xdim63 * (z) + zdim63 * ydim63 * xdim63 * (u) + udim63 * zdim63 * ydim63 * xdim63 * (v))
#define OPS_ACC64(x, y, z, u, v) (x + xdim64 * (y) + ydim64 * xdim64 * (z) + zdim64 * ydim64 * xdim64 * (u) + udim64 * zdim64 * ydim64 * xdim64 * (v))
#define OPS_ACC65(x, y, z, u, v) (x + xdim65 * (y) + ydim65 * xdim65 * (z) + zdim65 * ydim65 * xdim65 * (u) + udim65 * zdim65 * ydim65 * xdim65 * (v))
#define OPS_ACC66(x, y, z, u, v) (x + xdim66 * (y) + ydim66 * xdim66 * (z) + zdim66 * ydim66 * xdim66 * (u) + udim66 * zdim66 * ydim66 * xdim66 * (v))
#define OPS_ACC67(x, y, z, u, v) (x + xdim67 * (y) + ydim67 * xdim67 * (z) + zdim67 * ydim67 * xdim67 * (u) + udim67 * zdim67 * ydim67 * xdim67 * (v))
#define OPS_ACC68(x, y, z, u, v) (x + xdim68 * (y) + ydim68 * xdim68 * (z) + zdim68 * ydim68 * xdim68 * (u) + udim68 * zdim68 * ydim68 * xdim68 * (v))
#define OPS_ACC69(x, y, z, u, v) (x + xdim69 * (y) + ydim69 * xdim69 * (z) + zdim69 * ydim69 * xdim69 * (u) + udim69 * zdim69 * ydim69 * xdim69 * (v))
#define OPS_ACC70(x, y, z, u, v) (x + xdim70 * (y) + ydim70 * xdim70 * (z) + zdim70 * ydim70 * xdim70 * (u) + udim70 * zdim70 * ydim70 * xdim70 * (v))
#define OPS_ACC71(x, y, z, u, v) (x + xdim71 * (y) + ydim71 * xdim71 * (z) + zdim71 * ydim71 * xdim71 * (u) + udim71 * zdim71 * ydim71 * xdim71 * (v))
#define OPS_ACC72(x, y, z, u, v) (x + xdim72 * (y) + ydim72 * xdim72 * (z) + zdim72 * ydim72 * xdim72 * (u) + udim72 * zdim72 * ydim72 * xdim72 * (v))
#define OPS_ACC73(x, y, z, u, v) (x + xdim73 * (y) + ydim73 * xdim73 * (z) + zdim73 * ydim73 * xdim73 * (u) + udim73 * zdim73 * ydim73 * xdim73 * (v))
#define OPS_ACC74(x, y, z, u, v) (x + xdim74 * (y) + ydim74 * xdim74 * (z) + zdim74 * ydim74 * xdim74 * (u) + udim74 * zdim74 * ydim74 * xdim74 * (v))
#define OPS_ACC75(x, y, z, u, v) (x + xdim75 * (y) + ydim75 * xdim75 * (z) + zdim75 * ydim75 * xdim75 * (u) + udim75 * zdim75 * ydim75 * xdim75 * (v))
#define OPS_ACC76(x, y, z, u, v) (x + xdim76 * (y) + ydim76 * xdim76 * (z) + zdim76 * ydim76 * xdim76 * (u) + udim76 * zdim76 * ydim76 * xdim76 * (v))
#define OPS_ACC77(x, y, z, u, v) (x + xdim77 * (y) + xdim77 * ydim77 * (z) + zdim77 * ydim77 * xdim77 * (u) + udim77 * zdim77 * ydim77 * xdim77 * (v))
#define OPS_ACC78(x, y, z, u, v) (x + xdim78 * (y) + xdim78 * ydim78 * (z) + zdim78 * ydim78 * xdim78 * (u) + udim78 * zdim78 * ydim78 * xdim78 * (v))
#define OPS_ACC79(x, y, z, u, v) (x + xdim79 * (y) + xdim79 * ydim79 * (z) + zdim79 * ydim79 * xdim79 * (u) + udim79 * zdim79 * ydim79 * xdim79 * (v))
#define OPS_ACC80(x, y, z, u, v) (x + xdim80 * (y) + xdim80 * ydim80 * (z) + zdim80 * ydim80 * xdim80 * (u) + udim80 * zdim80 * ydim80 * xdim80 * (v))
#define OPS_ACC81(x, y, z, u, v) (x + xdim81 * (y) + xdim81 * ydim81 * (z) + zdim81 * ydim81 * xdim81 * (u) + udim81 * zdim81 * ydim81 * xdim81 * (v))
#define OPS_ACC82(x, y, z, u, v) (x + xdim82 * (y) + xdim82 * ydim82 * (z) + zdim82 * ydim82 * xdim82 * (u) + udim82 * zdim82 * ydim82 * xdim82 * (v))
#define OPS_ACC83(x, y, z, u, v) (x + xdim83 * (y) + xdim83 * ydim83 * (z) + zdim83 * ydim83 * xdim83 * (u) + udim83 * zdim83 * ydim83 * xdim83 * (v))
#define OPS_ACC84(x, y, z, u, v) (x + xdim84 * (y) + xdim84 * ydim84 * (z) + zdim84 * ydim84 * xdim84 * (u) + udim84 * zdim84 * ydim84 * xdim84 * (v))
#define OPS_ACC85(x, y, z, u, v) (x + xdim85 * (y) + xdim85 * ydim85 * (z) + zdim85 * ydim85 * xdim85 * (u) + udim85 * zdim85 * ydim85 * xdim85 * (v))
#define OPS_ACC86(x, y, z, u, v) (x + xdim86 * (y) + xdim86 * ydim86 * (z) + zdim86 * ydim86 * xdim86 * (u) + udim86 * zdim86 * ydim86 * xdim86 * (v))
#define OPS_ACC87(x, y, z, u, v) (x + xdim87 * (y) + xdim87 * ydim87 * (z) + zdim87 * ydim87 * xdim87 * (u) + udim87 * zdim87 * ydim87 * xdim87 * (v))
#define OPS_ACC88(x, y, z, u, v) (x + xdim88 * (y) + xdim88 * ydim88 * (z) + zdim88 * ydim88 * xdim88 * (u) + udim88 * zdim88 * ydim88 * xdim88 * (v))
#define OPS_ACC89(x, y, z, u, v) (x + xdim89 * (y) + xdim89 * ydim89 * (z) + zdim89 * ydim89 * xdim89 * (u) + udim89 * zdim89 * ydim89 * xdim89 * (v))
#define OPS_ACC90(x, y, z, u, v) (x + xdim90 * (y) + xdim90 * ydim90 * (z) + zdim90 * ydim90 * xdim90 * (u) + udim90 * zdim90 * ydim90 * xdim90 * (v))
#define OPS_ACC91(x, y, z, u, v) (x + xdim91 * (y) + xdim91 * ydim91 * (z) + zdim91 * ydim91 * xdim91 * (u) + udim91 * zdim91 * ydim91 * xdim91 * (v))
#define OPS_ACC92(x, y, z, u, v) (x + xdim92 * (y) + xdim92 * ydim92 * (z) + zdim92 * ydim92 * xdim92 * (u) + udim92 * zdim92 * ydim92 * xdim92 * (v))
#define OPS_ACC93(x, y, z, u, v) (x + xdim93 * (y) + xdim93 * ydim93 * (z) + zdim93 * ydim93 * xdim93 * (u) + udim93 * zdim93 * ydim93 * xdim93 * (v))
#define OPS_ACC94(x, y, z, u, v) (x + xdim94 * (y) + xdim94 * ydim94 * (z) + zdim94 * ydim94 * xdim94 * (u) + udim94 * zdim94 * ydim94 * xdim94 * (v))
#define OPS_ACC95(x, y, z, u, v) (x + xdim95 * (y) + xdim95 * ydim95 * (z) + zdim95 * ydim95 * xdim95 * (u) + udim95 * zdim95 * ydim95 * xdim95 * (v))
#define OPS_ACC96(x, y, z, u, v) (x + xdim96 * (y) + xdim96 * ydim96 * (z) + zdim96 * ydim96 * xdim96 * (u) + udim96 * zdim96 * ydim96 * xdim96 * (v))
#define OPS_ACC97(x, y, z, u, v) (x + xdim97 * (y) + xdim97 * ydim97 * (z) + zdim97 * ydim97 * xdim97 * (u) + udim97 * zdim97 * ydim97 * xdim97 * (v))
#define OPS_ACC98(x, y, z, u, v) (x + xdim98 * (y) + xdim98 * ydim98 * (z) + zdim98 * ydim98 * xdim98 * (u) + udim98 * zdim98 * ydim98 * xdim98 * (v))
#define OPS_ACC99(x, y, z, u, v) (x + xdim99 * (y) + xdim99 * ydim99 * (z) + zdim99 * ydim99 * xdim99 * (u) + udim99 * zdim99 * ydim99 * xdim99 * (v))
#else // do debug checks
#define OPS_ACC0(x, y, z, u, v)  (ops_stencil_check_5d(0 , x, y, z, u, v, xdim0,  ydim0 , zdim0 , udim0 ))
#define OPS_ACC1(x, y, z, u, v)  (ops_stencil_check_5d(1 , x, y, z, u, v, xdim1,  ydim1 , zdim1 , udim1 ))
#define OPS_ACC2(x, y, z, u, v)  (ops_stencil_check_5d(2 , x, y, z, u, v, xdim2,  ydim2 , zdim2 , udim2 ))
#define OPS_ACC3(x, y, z, u, v)  (ops_stencil_check_5d(3 , x, y, z, u, v, xdim3,  ydim3 , zdim3 , udim3 ))
#define OPS_ACC4(x, y, z, u, v)  (ops_stencil_check_5d(4 , x, y, z, u, v, xdim4,  ydim4 , zdim4 , udim4 ))
#define OPS_ACC5(x, y, z, u, v)  (ops_stencil_check_5d(5 , x, y, z, u, v, xdim5,  ydim5 , zdim5 , udim5 ))
#define OPS_ACC6(x, y, z, u, v)  (ops_stencil_check_5d(6 , x, y, z, u, v, xdim6,  ydim6 , zdim6 , udim6 ))
#define OPS_ACC7(x, y, z, u, v)  (ops_stencil_check_5d(7 , x, y, z, u, v, xdim7,  ydim7 , zdim7 , udim7 ))
#define OPS_ACC8(x, y, z, u, v)  (ops_stencil_check_5d(8 , x, y, z, u, v, xdim8,  ydim8 , zdim8 , udim8 ))
#define OPS_ACC9(x, y, z, u, v)  (ops_stencil_check_5d(9 , x, y, z, u, v, xdim9,  ydim9 , zdim9 , udim9 ))
#define OPS_ACC10(x, y, z, u, v) (ops_stencil_check_5d(10, x, y, z, u, v, xdim10, ydim10, zdim10, udim10))
#define OPS_ACC11(x, y, z, u, v) (ops_stencil_check_5d(11, x, y, z, u, v, xdim11, ydim11, zdim11, udim11))
#define OPS_ACC12(x, y, z, u, v) (ops_stencil_check_5d(12, x, y, z, u, v, xdim12, ydim12, zdim12, udim12))
#define OPS_ACC13(x, y, z, u, v) (ops_stencil_check_5d(13, x, y, z, u, v, xdim13, ydim13, zdim13, udim13))
#define OPS_ACC14(x, y, z, u, v) (ops_stencil_check_5d(14, x, y, z, u, v, xdim14, ydim14, zdim14, udim14))
#define OPS_ACC15(x, y, z, u, v) (ops_stencil_check_5d(15, x, y, z, u, v, xdim15, ydim15, zdim15, udim15))
#define OPS_ACC16(x, y, z, u, v) (ops_stencil_check_5d(16, x, y, z, u, v, xdim16, ydim16, zdim16, udim16))
#define OPS_ACC17(x, y, z, u, v) (ops_stencil_check_5d(17, x, y, z, u, v, xdim17, ydim17, zdim17, udim17))
#define OPS_ACC18(x, y, z, u, v) (ops_stencil_check_5d(18, x, y, z, u, v, xdim18, ydim18, zdim18, udim18))
#define OPS_ACC19(x, y, z, u, v) (ops_stencil_check_5d(19, x, y, z, u, v, xdim19, ydim19, zdim19, udim19))
#define OPS_ACC20(x, y, z, u, v) (ops_stencil_check_5d(20, x, y, z, u, v, xdim20, ydim20, zdim20, udim20))
#define OPS_ACC21(x, y, z, u, v) (ops_stencil_check_5d(21, x, y, z, u, v, xdim21, ydim21, zdim21, udim21))
#define OPS_ACC22(x, y, z, u, v) (ops_stencil_check_5d(22, x, y, z, u, v, xdim22, ydim22, zdim22, udim22))
#define OPS_ACC23(x, y, z, u, v) (ops_stencil_check_5d(23, x, y, z, u, v, xdim23, ydim23, zdim23, udim23))
#define OPS_ACC24(x, y, z, u, v) (ops_stencil_check_5d(24, x, y, z, u, v, xdim24, ydim24, zdim24, udim24))
#define OPS_ACC25(x, y, z, u, v) (ops_stencil_check_5d(25, x, y, z, u, v, xdim25, ydim25, zdim25, udim25))
#define OPS_ACC26(x, y, z, u, v) (ops_stencil_check_5d(26, x, y, z, u, v, xdim26, ydim26, zdim26, udim26))
#define OPS_ACC27(x, y, z, u, v) (ops_stencil_check_5d(27, x, y, z, u, v, xdim27, ydim27, zdim27, udim27))
#define OPS_ACC28(x, y, z, u, v) (ops_stencil_check_5d(28, x, y, z, u, v, xdim28, ydim28, zdim28, udim28))
#define OPS_ACC29(x, y, z, u, v) (ops_stencil_check_5d(29, x, y, z, u, v, xdim29, ydim29, zdim29, udim29))
#define OPS_ACC30(x, y, z, u, v) (ops_stencil_check_5d(30, x, y, z, u, v, xdim30, ydim30, zdim30, udim30))
#define OPS_ACC31(x, y, z, u, v) (ops_stencil_check_5d(31, x, y, z, u, v, xdim31, ydim31, zdim31, udim31))
#define OPS_ACC32(x, y, z, u, v) (ops_stencil_check_5d(32, x, y, z, u, v, xdim32, ydim32, zdim32, udim32))
#define OPS_ACC33(x, y, z, u, v) (ops_stencil_check_5d(33, x, y, z, u, v, xdim33, ydim33, zdim33, udim33))
#define OPS_ACC34(x, y, z, u, v) (ops_stencil_check_5d(34, x, y, z, u, v, xdim34, ydim34, zdim34, udim34))
#define OPS_ACC35(x, y, z, u, v) (ops_stencil_check_5d(35, x, y, z, u, v, xdim35, ydim35, zdim35, udim35))
#define OPS_ACC36(x, y, z, u, v) (ops_stencil_check_5d(36, x, y, z, u, v, xdim36, ydim36, zdim36, udim36))
#define OPS_ACC37(x, y, z, u, v) (ops_stencil_check_5d(37, x, y, z, u, v, xdim37, ydim37, zdim37, udim37))
#define OPS_ACC38(x, y, z, u, v) (ops_stencil_check_5d(38, x, y, z, u, v, xdim38, ydim38, zdim38, udim38))
#define OPS_ACC39(x, y, z, u, v) (ops_stencil_check_5d(39, x, y, z, u, v, xdim39, ydim39, zdim39, udim39))
#define OPS_ACC40(x, y, z, u, v) (ops_stencil_check_5d(40, x, y, z, u, v, xdim40, ydim40, zdim40, udim40))
#define OPS_ACC41(x, y, z, u, v) (ops_stencil_check_5d(41, x, y, z, u, v, xdim41, ydim41, zdim41, udim41))
#define OPS_ACC42(x, y, z, u, v) (ops_stencil_check_5d(42, x, y, z, u, v, xdim42, ydim42, zdim42, udim42))
#define OPS_ACC43(x, y, z, u, v) (ops_stencil_check_5d(43, x, y, z, u, v, xdim43, ydim43, zdim43, udim43))
#define OPS_ACC44(x, y, z, u, v) (ops_stencil_check_5d(44, x, y, z, u, v, xdim44, ydim44, zdim44, udim44))
#define OPS_ACC45(x, y, z, u, v) (ops_stencil_check_5d(45, x, y, z, u, v, xdim45, ydim45, zdim45, udim45))
#define OPS_ACC46(x, y, z, u, v) (ops_stencil_check_5d(46, x, y, z, u, v, xdim46, ydim46, zdim46, udim46))
#define OPS_ACC47(x, y, z, u, v) (ops_stencil_check_5d(47, x, y, z, u, v, xdim47, ydim47, zdim47, udim47))
#define OPS_ACC48(x, y, z, u, v) (ops_stencil_check_5d(48, x, y, z, u, v, xdim48, ydim48, zdim48, udim48))
#define OPS_ACC49(x, y, z, u, v) (ops_stencil_check_5d(49, x, y, z, u, v, xdim49, ydim49, zdim49, udim49))
#define OPS_ACC50(x, y, z, u, v) (ops_stencil_check_5d(50, x, y, z, u, v, xdim50, ydim50, zdim50, udim50))
#define OPS_ACC51(x, y, z, u, v) (ops_stencil_check_5d(51, x, y, z, u, v, xdim51, ydim51, zdim51, udim51))
#define OPS_ACC52(x, y, z, u, v) (ops_stencil_check_5d(52, x, y, z, u, v, xdim52, ydim52, zdim52, udim52))
#define OPS_ACC53(x, y, z, u, v) (ops_stencil_check_5d(53, x, y, z, u, v, xdim53, ydim53, zdim53, udim53))
#define OPS_ACC54(x, y, z, u, v) (ops_stencil_check_5d(54, x, y, z, u, v, xdim54, ydim54, zdim54, udim54))
#define OPS_ACC55(x, y, z, u, v) (ops_stencil_check_5d(55, x, y, z, u, v, xdim55, ydim55, zdim55, udim55))
#define OPS_ACC56(x, y, z, u, v) (ops_stencil_check_5d(56, x, y, z, u, v, xdim56, ydim56, zdim56, udim56))
#define OPS_ACC57(x, y, z, u, v) (ops_stencil_check_5d(57, x, y, z, u, v, xdim57, ydim57, zdim57, udim57))
#define OPS_ACC58(x, y, z, u, v) (ops_stencil_check_5d(58, x, y, z, u, v, xdim58, ydim58, zdim58, udim58))
#define OPS_ACC59(x, y, z, u, v) (ops_stencil_check_5d(59, x, y, z, u, v, xdim59, ydim59, zdim59, udim59))
#define OPS_ACC60(x, y, z, u, v) (ops_stencil_check_5d(60, x, y, z, u, v, xdim60, ydim60, zdim60, udim60))
#define OPS_ACC61(x, y, z, u, v) (ops_stencil_check_5d(61, x, y, z, u, v, xdim61, ydim61, zdim61, udim61))
#define OPS_ACC62(x, y, z, u, v) (ops_stencil_check_5d(62, x, y, z, u, v, xdim62, ydim62, zdim62, udim62))
#define OPS_ACC63(x, y, z, u, v) (ops_stencil_check_5d(63, x, y, z, u, v, xdim63, ydim63, zdim63, udim63))
#define OPS_ACC64(x, y, z, u, v) (ops_stencil_check_5d(64, x, y, z, u, v, xdim64, ydim64, zdim64, udim64))
#define OPS_ACC65(x, y, z, u, v) (ops_stencil_check_5d(65, x, y, z, u, v, xdim65, ydim65, zdim65, udim65))
#define OPS_ACC66(x, y, z, u, v) (ops_stencil_check_5d(66, x, y, z, u, v, xdim66, ydim66, zdim66, udim66))
#define OPS_ACC67(x, y, z, u, v) (ops_stencil_check_5d(67, x, y, z, u, v, xdim67, ydim67, zdim67, udim67))
#define OPS_ACC68(x, y, z, u, v) (ops_stencil_check_5d(68, x, y, z, u, v, xdim68, ydim68, zdim68, udim68))
#define OPS_ACC69(x, y, z, u, v) (ops_stencil_check_5d(69, x, y, z, u, v, xdim69, ydim69, zdim69, udim69))
#define OPS_ACC70(x, y, z, u, v) (ops_stencil_check_5d(70, x, y, z, u, v, xdim70, ydim70, zdim70, udim70))
#define OPS_ACC71(x, y, z, u, v) (ops_stencil_check_5d(71, x, y, z, u, v, xdim71, ydim71, zdim71, udim71))
#define OPS_ACC72(x, y, z, u, v) (ops_stencil_check_5d(72, x, y, z, u, v, xdim72, ydim72, zdim72, udim72))
#define OPS_ACC73(x, y, z, u, v) (ops_stencil_check_5d(73, x, y, z, u, v, xdim73, ydim73, zdim73, udim73))
#define OPS_ACC74(x, y, z, u, v) (ops_stencil_check_5d(74, x, y, z, u, v, xdim74, ydim74, zdim74, udim74))
#define OPS_ACC75(x, y, z, u, v) (ops_stencil_check_5d(75, x, y, z, u, v, xdim75, ydim75, zdim75, udim75))
#define OPS_ACC76(x, y, z, u, v) (ops_stencil_check_5d(76, x, y, z, u, v, xdim76, ydim76, zdim76, udim76))
#define OPS_ACC77(x, y, z, u, v) (ops_stencil_check_5d(77, x, y, z, u, v, xdim77, ydim77, zdim77, udim77))
#define OPS_ACC78(x, y, z, u, v) (ops_stencil_check_5d(78, x, y, z, u, v, xdim78, ydim78, zdim78, udim78))
#define OPS_ACC79(x, y, z, u, v) (ops_stencil_check_5d(79, x, y, z, u, v, xdim79, ydim79, zdim79, udim79))
#define OPS_ACC80(x, y, z, u, v) (ops_stencil_check_5d(80, x, y, z, u, v, xdim80, ydim80, zdim80, udim80))
#define OPS_ACC81(x, y, z, u, v) (ops_stencil_check_5d(81, x, y, z, u, v, xdim81, ydim81, zdim81, udim81))
#define OPS_ACC82(x, y, z, u, v) (ops_stencil_check_5d(82, x, y, z, u, v, xdim82, ydim82, zdim82, udim82))
#define OPS_ACC83(x, y, z, u, v) (ops_stencil_check_5d(83, x, y, z, u, v, xdim83, ydim83, zdim83, udim83))
#define OPS_ACC84(x, y, z, u, v) (ops_stencil_check_5d(84, x, y, z, u, v, xdim84, ydim84, zdim84, udim84))
#define OPS_ACC85(x, y, z, u, v) (ops_stencil_check_5d(85, x, y, z, u, v, xdim85, ydim85, zdim85, udim85))
#define OPS_ACC86(x, y, z, u, v) (ops_stencil_check_5d(86, x, y, z, u, v, xdim86, ydim86, zdim86, udim86))
#define OPS_ACC87(x, y, z, u, v) (ops_stencil_check_5d(87, x, y, z, u, v, xdim87, ydim87, zdim87, udim87))
#define OPS_ACC88(x, y, z, u, v) (ops_stencil_check_5d(88, x, y, z, u, v, xdim88, ydim88, zdim88, udim88))
#define OPS_ACC89(x, y, z, u, v) (ops_stencil_check_5d(89, x, y, z, u, v, xdim89, ydim89, zdim89, udim89))
#define OPS_ACC90(x, y, z, u, v) (ops_stencil_check_5d(90, x, y, z, u, v, xdim90, ydim90, zdim90, udim90))
#define OPS_ACC91(x, y, z, u, v) (ops_stencil_check_5d(91, x, y, z, u, v, xdim91, ydim91, zdim91, udim91))
#define OPS_ACC92(x, y, z, u, v) (ops_stencil_check_5d(92, x, y, z, u, v, xdim92, ydim92, zdim92, udim92))
#define OPS_ACC93(x, y, z, u, v) (ops_stencil_check_5d(93, x, y, z, u, v, xdim93, ydim93, zdim93, udim93))
#define OPS_ACC94(x, y, z, u, v) (ops_stencil_check_5d(94, x, y, z, u, v, xdim94, ydim94, zdim94, udim94))
#define OPS_ACC95(x, y, z, u, v) (ops_stencil_check_5d(95, x, y, z, u, v, xdim95, ydim95, zdim95, udim95))
#define OPS_ACC96(x, y, z, u, v) (ops_stencil_check_5d(96, x, y, z, u, v, xdim96, ydim96, zdim96, udim96))
#define OPS_ACC97(x, y, z, u, v) (ops_stencil_check_5d(97, x, y, z, u, v, xdim97, ydim97, zdim97, udim97))
#define OPS_ACC98(x, y, z, u, v) (ops_stencil_check_5d(98, x, y, z, u, v, xdim98, ydim98, zdim98, udim98))
#define OPS_ACC99(x, y, z, u, v) (ops_stencil_check_5d(99, x, y, z, u, v, xdim99, ydim99, zdim99, udim99))
#endif               // end debug checks

#elif defined OPS_4D     // macros for 4D application
#ifndef OPS_DEBUG // no debug checks
#define OPS_ACC0(x, y, z, u)  (x + xdim0  * (y) + ydim0  * xdim0  * (z) + zdim0  * ydim0  * xdim0  * (u))
#define OPS_ACC1(x, y, z, u)  (x + xdim1  * (y) + ydim1  * xdim1  * (z) + zdim1  * ydim1  * xdim1  * (u))
#define OPS_ACC2(x, y, z, u)  (x + xdim2  * (y) + ydim2  * xdim2  * (z) + zdim2  * ydim2  * xdim2  * (u))
#define OPS_ACC3(x, y, z, u)  (x + xdim3  * (y) + ydim3  * xdim3  * (z) + zdim3  * ydim3  * xdim3  * (u))
#define OPS_ACC4(x, y, z, u)  (x + xdim4  * (y) + ydim4  * xdim4  * (z) + zdim4  * ydim4  * xdim4  * (u))
#define OPS_ACC5(x, y, z, u)  (x + xdim5  * (y) + ydim5  * xdim5  * (z) + zdim5  * ydim5  * xdim5  * (u))
#define OPS_ACC6(x, y, z, u)  (x + xdim6  * (y) + ydim6  * xdim6  * (z) + zdim6  * ydim6  * xdim6  * (u))
#define OPS_ACC7(x, y, z, u)  (x + xdim7  * (y) + ydim7  * xdim7  * (z) + zdim7  * ydim7  * xdim7  * (u))
#define OPS_ACC8(x, y, z, u)  (x + xdim8  * (y) + ydim8  * xdim8  * (z) + zdim8  * ydim8  * xdim8  * (u))
#define OPS_ACC9(x, y, z, u)  (x + xdim9  * (y) + ydim9  * xdim9  * (z) + zdim9  * ydim9  * xdim9  * (u))
#define OPS_ACC10(x, y, z, u) (x + xdim10 * (y) + ydim10 * xdim10 * (z) + zdim10 * ydim10 * xdim10 * (u))
#define OPS_ACC11(x, y, z, u) (x + xdim11 * (y) + ydim11 * xdim11 * (z) + zdim11 * ydim11 * xdim11 * (u))
#define OPS_ACC12(x, y, z, u) (x + xdim12 * (y) + ydim12 * xdim12 * (z) + zdim12 * ydim12 * xdim12 * (u))
#define OPS_ACC13(x, y, z, u) (x + xdim13 * (y) + ydim13 * xdim13 * (z) + zdim13 * ydim13 * xdim13 * (u))
#define OPS_ACC14(x, y, z, u) (x + xdim14 * (y) + ydim14 * xdim14 * (z) + zdim14 * ydim14 * xdim14 * (u))
#define OPS_ACC15(x, y, z, u) (x + xdim15 * (y) + ydim15 * xdim15 * (z) + zdim15 * ydim15 * xdim15 * (u))
#define OPS_ACC16(x, y, z, u) (x + xdim16 * (y) + ydim16 * xdim16 * (z) + zdim16 * ydim16 * xdim16 * (u))
#define OPS_ACC17(x, y, z, u) (x + xdim17 * (y) + ydim17 * xdim17 * (z) + zdim17 * ydim17 * xdim17 * (u))
#define OPS_ACC18(x, y, z, u) (x + xdim18 * (y) + ydim18 * xdim18 * (z) + zdim18 * ydim18 * xdim18 * (u))
#define OPS_ACC19(x, y, z, u) (x + xdim19 * (y) + ydim19 * xdim19 * (z) + zdim19 * ydim19 * xdim19 * (u))
#define OPS_ACC20(x, y, z, u) (x + xdim20 * (y) + ydim20 * xdim20 * (z) + zdim20 * ydim20 * xdim20 * (u))
#define OPS_ACC21(x, y, z, u) (x + xdim21 * (y) + ydim21 * xdim21 * (z) + zdim21 * ydim21 * xdim21 * (u))
#define OPS_ACC22(x, y, z, u) (x + xdim22 * (y) + ydim22 * xdim22 * (z) + zdim22 * ydim22 * xdim22 * (u))
#define OPS_ACC23(x, y, z, u) (x + xdim23 * (y) + ydim23 * xdim23 * (z) + zdim23 * ydim23 * xdim23 * (u))
#define OPS_ACC24(x, y, z, u) (x + xdim24 * (y) + ydim24 * xdim24 * (z) + zdim24 * ydim24 * xdim24 * (u))
#define OPS_ACC25(x, y, z, u) (x + xdim25 * (y) + ydim25 * xdim25 * (z) + zdim25 * ydim25 * xdim25 * (u))
#define OPS_ACC26(x, y, z, u) (x + xdim26 * (y) + ydim26 * xdim26 * (z) + zdim26 * ydim26 * xdim26 * (u))
#define OPS_ACC27(x, y, z, u) (x + xdim27 * (y) + ydim27 * xdim27 * (z) + zdim27 * ydim27 * xdim27 * (u))
#define OPS_ACC28(x, y, z, u) (x + xdim28 * (y) + ydim28 * xdim28 * (z) + zdim28 * ydim28 * xdim28 * (u))
#define OPS_ACC29(x, y, z, u) (x + xdim29 * (y) + ydim29 * xdim29 * (z) + zdim29 * ydim29 * xdim29 * (u))
#define OPS_ACC30(x, y, z, u) (x + xdim30 * (y) + ydim30 * xdim30 * (z) + zdim30 * ydim30 * xdim30 * (u))
#define OPS_ACC31(x, y, z, u) (x + xdim31 * (y) + ydim31 * xdim31 * (z) + zdim31 * ydim31 * xdim31 * (u))
#define OPS_ACC32(x, y, z, u) (x + xdim32 * (y) + ydim32 * xdim32 * (z) + zdim32 * ydim32 * xdim32 * (u))
#define OPS_ACC33(x, y, z, u) (x + xdim33 * (y) + ydim33 * xdim33 * (z) + zdim33 * ydim33 * xdim33 * (u))
#define OPS_ACC34(x, y, z, u) (x + xdim34 * (y) + ydim34 * xdim34 * (z) + zdim34 * ydim34 * xdim34 * (u))
#define OPS_ACC35(x, y, z, u) (x + xdim35 * (y) + ydim35 * xdim35 * (z) + zdim35 * ydim35 * xdim35 * (u))
#define OPS_ACC36(x, y, z, u) (x + xdim36 * (y) + ydim36 * xdim36 * (z) + zdim36 * ydim36 * xdim36 * (u))
#define OPS_ACC37(x, y, z, u) (x + xdim37 * (y) + ydim37 * xdim37 * (z) + zdim37 * ydim37 * xdim37 * (u))
#define OPS_ACC38(x, y, z, u) (x + xdim38 * (y) + ydim38 * xdim38 * (z) + zdim38 * ydim38 * xdim38 * (u))
#define OPS_ACC39(x, y, z, u) (x + xdim39 * (y) + ydim39 * xdim39 * (z) + zdim39 * ydim39 * xdim39 * (u))
#define OPS_ACC40(x, y, z, u) (x + xdim40 * (y) + ydim40 * xdim40 * (z) + zdim40 * ydim40 * xdim40 * (u))
#define OPS_ACC41(x, y, z, u) (x + xdim41 * (y) + ydim41 * xdim41 * (z) + zdim41 * ydim41 * xdim41 * (u))
#define OPS_ACC42(x, y, z, u) (x + xdim42 * (y) + ydim42 * xdim42 * (z) + zdim42 * ydim42 * xdim42 * (u))
#define OPS_ACC43(x, y, z, u) (x + xdim43 * (y) + ydim43 * xdim43 * (z) + zdim43 * ydim43 * xdim43 * (u))
#define OPS_ACC44(x, y, z, u) (x + xdim44 * (y) + ydim44 * xdim44 * (z) + zdim44 * ydim44 * xdim44 * (u))
#define OPS_ACC45(x, y, z, u) (x + xdim45 * (y) + ydim45 * xdim45 * (z) + zdim45 * ydim45 * xdim45 * (u))
#define OPS_ACC46(x, y, z, u) (x + xdim46 * (y) + ydim46 * xdim46 * (z) + zdim46 * ydim46 * xdim46 * (u))
#define OPS_ACC47(x, y, z, u) (x + xdim47 * (y) + ydim47 * xdim47 * (z) + zdim47 * ydim47 * xdim47 * (u))
#define OPS_ACC48(x, y, z, u) (x + xdim48 * (y) + ydim48 * xdim48 * (z) + zdim48 * ydim48 * xdim48 * (u))
#define OPS_ACC49(x, y, z, u) (x + xdim49 * (y) + ydim49 * xdim49 * (z) + zdim49 * ydim49 * xdim49 * (u))
#define OPS_ACC50(x, y, z, u) (x + xdim50 * (y) + ydim50 * xdim50 * (z) + zdim50 * ydim50 * xdim50 * (u))
#define OPS_ACC51(x, y, z, u) (x + xdim51 * (y) + ydim51 * xdim51 * (z) + zdim51 * ydim51 * xdim51 * (u))
#define OPS_ACC52(x, y, z, u) (x + xdim52 * (y) + ydim52 * xdim52 * (z) + zdim52 * ydim52 * xdim52 * (u))
#define OPS_ACC53(x, y, z, u) (x + xdim53 * (y) + ydim53 * xdim53 * (z) + zdim53 * ydim53 * xdim53 * (u))
#define OPS_ACC54(x, y, z, u) (x + xdim54 * (y) + ydim54 * xdim54 * (z) + zdim54 * ydim54 * xdim54 * (u))
#define OPS_ACC55(x, y, z, u) (x + xdim55 * (y) + ydim55 * xdim55 * (z) + zdim55 * ydim55 * xdim55 * (u))
#define OPS_ACC56(x, y, z, u) (x + xdim56 * (y) + ydim56 * xdim56 * (z) + zdim56 * ydim56 * xdim56 * (u))
#define OPS_ACC57(x, y, z, u) (x + xdim57 * (y) + ydim57 * xdim57 * (z) + zdim57 * ydim57 * xdim57 * (u))
#define OPS_ACC58(x, y, z, u) (x + xdim58 * (y) + ydim58 * xdim58 * (z) + zdim58 * ydim58 * xdim58 * (u))
#define OPS_ACC59(x, y, z, u) (x + xdim59 * (y) + ydim59 * xdim59 * (z) + zdim59 * ydim59 * xdim59 * (u))
#define OPS_ACC60(x, y, z, u) (x + xdim60 * (y) + ydim60 * xdim60 * (z) + zdim60 * ydim60 * xdim60 * (u))
#define OPS_ACC61(x, y, z, u) (x + xdim61 * (y) + ydim61 * xdim61 * (z) + zdim61 * ydim61 * xdim61 * (u))
#define OPS_ACC62(x, y, z, u) (x + xdim62 * (y) + ydim62 * xdim62 * (z) + zdim62 * ydim62 * xdim62 * (u))
#define OPS_ACC63(x, y, z, u) (x + xdim63 * (y) + ydim63 * xdim63 * (z) + zdim63 * ydim63 * xdim63 * (u))
#define OPS_ACC64(x, y, z, u) (x + xdim64 * (y) + ydim64 * xdim64 * (z) + zdim64 * ydim64 * xdim64 * (u))
#define OPS_ACC65(x, y, z, u) (x + xdim65 * (y) + ydim65 * xdim65 * (z) + zdim65 * ydim65 * xdim65 * (u))
#define OPS_ACC66(x, y, z, u) (x + xdim66 * (y) + ydim66 * xdim66 * (z) + zdim66 * ydim66 * xdim66 * (u))
#define OPS_ACC67(x, y, z, u) (x + xdim67 * (y) + ydim67 * xdim67 * (z) + zdim67 * ydim67 * xdim67 * (u))
#define OPS_ACC68(x, y, z, u) (x + xdim68 * (y) + ydim68 * xdim68 * (z) + zdim68 * ydim68 * xdim68 * (u))
#define OPS_ACC69(x, y, z, u) (x + xdim69 * (y) + ydim69 * xdim69 * (z) + zdim69 * ydim69 * xdim69 * (u))
#define OPS_ACC70(x, y, z, u) (x + xdim70 * (y) + ydim70 * xdim70 * (z) + zdim70 * ydim70 * xdim70 * (u))
#define OPS_ACC71(x, y, z, u) (x + xdim71 * (y) + ydim71 * xdim71 * (z) + zdim71 * ydim71 * xdim71 * (u))
#define OPS_ACC72(x, y, z, u) (x + xdim72 * (y) + ydim72 * xdim72 * (z) + zdim72 * ydim72 * xdim72 * (u))
#define OPS_ACC73(x, y, z, u) (x + xdim73 * (y) + ydim73 * xdim73 * (z) + zdim73 * ydim73 * xdim73 * (u))
#define OPS_ACC74(x, y, z, u) (x + xdim74 * (y) + ydim74 * xdim74 * (z) + zdim74 * ydim74 * xdim74 * (u))
#define OPS_ACC75(x, y, z, u) (x + xdim75 * (y) + ydim75 * xdim75 * (z) + zdim75 * ydim75 * xdim75 * (u))
#define OPS_ACC76(x, y, z, u) (x + xdim76 * (y) + ydim76 * xdim76 * (z) + zdim76 * ydim76 * xdim76 * (u))
#define OPS_ACC77(x, y, z, u) (x + xdim77 * (y) + xdim77 * ydim77 * (z) + zdim77 * ydim77 * xdim77 * (u))
#define OPS_ACC78(x, y, z, u) (x + xdim78 * (y) + xdim78 * ydim78 * (z) + zdim78 * ydim78 * xdim78 * (u))
#define OPS_ACC79(x, y, z, u) (x + xdim79 * (y) + xdim79 * ydim79 * (z) + zdim79 * ydim79 * xdim79 * (u))
#define OPS_ACC80(x, y, z, u) (x + xdim80 * (y) + xdim80 * ydim80 * (z) + zdim80 * ydim80 * xdim80 * (u))
#define OPS_ACC81(x, y, z, u) (x + xdim81 * (y) + xdim81 * ydim81 * (z) + zdim81 * ydim81 * xdim81 * (u))
#define OPS_ACC82(x, y, z, u) (x + xdim82 * (y) + xdim82 * ydim82 * (z) + zdim82 * ydim82 * xdim82 * (u))
#define OPS_ACC83(x, y, z, u) (x + xdim83 * (y) + xdim83 * ydim83 * (z) + zdim83 * ydim83 * xdim83 * (u))
#define OPS_ACC84(x, y, z, u) (x + xdim84 * (y) + xdim84 * ydim84 * (z) + zdim84 * ydim84 * xdim84 * (u))
#define OPS_ACC85(x, y, z, u) (x + xdim85 * (y) + xdim85 * ydim85 * (z) + zdim85 * ydim85 * xdim85 * (u))
#define OPS_ACC86(x, y, z, u) (x + xdim86 * (y) + xdim86 * ydim86 * (z) + zdim86 * ydim86 * xdim86 * (u))
#define OPS_ACC87(x, y, z, u) (x + xdim87 * (y) + xdim87 * ydim87 * (z) + zdim87 * ydim87 * xdim87 * (u))
#define OPS_ACC88(x, y, z, u) (x + xdim88 * (y) + xdim88 * ydim88 * (z) + zdim88 * ydim88 * xdim88 * (u))
#define OPS_ACC89(x, y, z, u) (x + xdim89 * (y) + xdim89 * ydim89 * (z) + zdim89 * ydim89 * xdim89 * (u))
#define OPS_ACC90(x, y, z, u) (x + xdim90 * (y) + xdim90 * ydim90 * (z) + zdim90 * ydim90 * xdim90 * (u))
#define OPS_ACC91(x, y, z, u) (x + xdim91 * (y) + xdim91 * ydim91 * (z) + zdim91 * ydim91 * xdim91 * (u))
#define OPS_ACC92(x, y, z, u) (x + xdim92 * (y) + xdim92 * ydim92 * (z) + zdim92 * ydim92 * xdim92 * (u))
#define OPS_ACC93(x, y, z, u) (x + xdim93 * (y) + xdim93 * ydim93 * (z) + zdim93 * ydim93 * xdim93 * (u))
#define OPS_ACC94(x, y, z, u) (x + xdim94 * (y) + xdim94 * ydim94 * (z) + zdim94 * ydim94 * xdim94 * (u))
#define OPS_ACC95(x, y, z, u) (x + xdim95 * (y) + xdim95 * ydim95 * (z) + zdim95 * ydim95 * xdim95 * (u))
#define OPS_ACC96(x, y, z, u) (x + xdim96 * (y) + xdim96 * ydim96 * (z) + zdim96 * ydim96 * xdim96 * (u))
#define OPS_ACC97(x, y, z, u) (x + xdim97 * (y) + xdim97 * ydim97 * (z) + zdim97 * ydim97 * xdim97 * (u))
#define OPS_ACC98(x, y, z, u) (x + xdim98 * (y) + xdim98 * ydim98 * (z) + zdim98 * ydim98 * xdim98 * (u))
#define OPS_ACC99(x, y, z, u) (x + xdim99 * (y) + xdim99 * ydim99 * (z) + zdim99 * ydim99 * xdim99 * (u))
#else // do debug checks
#define OPS_ACC0(x, y, z, u)  (ops_stencil_check_4d(0 , x, y, z, u, xdim0,  ydim0 , zdim0 ))
#define OPS_ACC1(x, y, z, u)  (ops_stencil_check_4d(1 , x, y, z, u, xdim1,  ydim1 , zdim1 ))
#define OPS_ACC2(x, y, z, u)  (ops_stencil_check_4d(2 , x, y, z, u, xdim2,  ydim2 , zdim2 ))
#define OPS_ACC3(x, y, z, u)  (ops_stencil_check_4d(3 , x, y, z, u, xdim3,  ydim3 , zdim3 ))
#define OPS_ACC4(x, y, z, u)  (ops_stencil_check_4d(4 , x, y, z, u, xdim4,  ydim4 , zdim4 ))
#define OPS_ACC5(x, y, z, u)  (ops_stencil_check_4d(5 , x, y, z, u, xdim5,  ydim5 , zdim5 ))
#define OPS_ACC6(x, y, z, u)  (ops_stencil_check_4d(6 , x, y, z, u, xdim6,  ydim6 , zdim6 ))
#define OPS_ACC7(x, y, z, u)  (ops_stencil_check_4d(7 , x, y, z, u, xdim7,  ydim7 , zdim7 ))
#define OPS_ACC8(x, y, z, u)  (ops_stencil_check_4d(8 , x, y, z, u, xdim8,  ydim8 , zdim8 ))
#define OPS_ACC9(x, y, z, u)  (ops_stencil_check_4d(9 , x, y, z, u, xdim9,  ydim9 , zdim9 ))
#define OPS_ACC10(x, y, z, u) (ops_stencil_check_4d(10, x, y, z, u, xdim10, ydim10, zdim10))
#define OPS_ACC11(x, y, z, u) (ops_stencil_check_4d(11, x, y, z, u, xdim11, ydim11, zdim11))
#define OPS_ACC12(x, y, z, u) (ops_stencil_check_4d(12, x, y, z, u, xdim12, ydim12, zdim12))
#define OPS_ACC13(x, y, z, u) (ops_stencil_check_4d(13, x, y, z, u, xdim13, ydim13, zdim13))
#define OPS_ACC14(x, y, z, u) (ops_stencil_check_4d(14, x, y, z, u, xdim14, ydim14, zdim14))
#define OPS_ACC15(x, y, z, u) (ops_stencil_check_4d(15, x, y, z, u, xdim15, ydim15, zdim15))
#define OPS_ACC16(x, y, z, u) (ops_stencil_check_4d(16, x, y, z, u, xdim16, ydim16, zdim16))
#define OPS_ACC17(x, y, z, u) (ops_stencil_check_4d(17, x, y, z, u, xdim17, ydim17, zdim17))
#define OPS_ACC18(x, y, z, u) (ops_stencil_check_4d(18, x, y, z, u, xdim18, ydim18, zdim18))
#define OPS_ACC19(x, y, z, u) (ops_stencil_check_4d(19, x, y, z, u, xdim19, ydim19, zdim19))
#define OPS_ACC20(x, y, z, u) (ops_stencil_check_4d(20, x, y, z, u, xdim20, ydim20, zdim20))
#define OPS_ACC21(x, y, z, u) (ops_stencil_check_4d(21, x, y, z, u, xdim21, ydim21, zdim21))
#define OPS_ACC22(x, y, z, u) (ops_stencil_check_4d(22, x, y, z, u, xdim22, ydim22, zdim22))
#define OPS_ACC23(x, y, z, u) (ops_stencil_check_4d(23, x, y, z, u, xdim23, ydim23, zdim23))
#define OPS_ACC24(x, y, z, u) (ops_stencil_check_4d(24, x, y, z, u, xdim24, ydim24, zdim24))
#define OPS_ACC25(x, y, z, u) (ops_stencil_check_4d(25, x, y, z, u, xdim25, ydim25, zdim25))
#define OPS_ACC26(x, y, z, u) (ops_stencil_check_4d(26, x, y, z, u, xdim26, ydim26, zdim26))
#define OPS_ACC27(x, y, z, u) (ops_stencil_check_4d(27, x, y, z, u, xdim27, ydim27, zdim27))
#define OPS_ACC28(x, y, z, u) (ops_stencil_check_4d(28, x, y, z, u, xdim28, ydim28, zdim28))
#define OPS_ACC29(x, y, z, u) (ops_stencil_check_4d(29, x, y, z, u, xdim29, ydim29, zdim29))
#define OPS_ACC30(x, y, z, u) (ops_stencil_check_4d(30, x, y, z, u, xdim30, ydim30, zdim30))
#define OPS_ACC31(x, y, z, u) (ops_stencil_check_4d(31, x, y, z, u, xdim31, ydim31, zdim31))
#define OPS_ACC32(x, y, z, u) (ops_stencil_check_4d(32, x, y, z, u, xdim32, ydim32, zdim32))
#define OPS_ACC33(x, y, z, u) (ops_stencil_check_4d(33, x, y, z, u, xdim33, ydim33, zdim33))
#define OPS_ACC34(x, y, z, u) (ops_stencil_check_4d(34, x, y, z, u, xdim34, ydim34, zdim34))
#define OPS_ACC35(x, y, z, u) (ops_stencil_check_4d(35, x, y, z, u, xdim35, ydim35, zdim35))
#define OPS_ACC36(x, y, z, u) (ops_stencil_check_4d(36, x, y, z, u, xdim36, ydim36, zdim36))
#define OPS_ACC37(x, y, z, u) (ops_stencil_check_4d(37, x, y, z, u, xdim37, ydim37, zdim37))
#define OPS_ACC38(x, y, z, u) (ops_stencil_check_4d(38, x, y, z, u, xdim38, ydim38, zdim38))
#define OPS_ACC39(x, y, z, u) (ops_stencil_check_4d(39, x, y, z, u, xdim39, ydim39, zdim39))
#define OPS_ACC40(x, y, z, u) (ops_stencil_check_4d(40, x, y, z, u, xdim40, ydim40, zdim40))
#define OPS_ACC41(x, y, z, u) (ops_stencil_check_4d(41, x, y, z, u, xdim41, ydim41, zdim41))
#define OPS_ACC42(x, y, z, u) (ops_stencil_check_4d(42, x, y, z, u, xdim42, ydim42, zdim42))
#define OPS_ACC43(x, y, z, u) (ops_stencil_check_4d(43, x, y, z, u, xdim43, ydim43, zdim43))
#define OPS_ACC44(x, y, z, u) (ops_stencil_check_4d(44, x, y, z, u, xdim44, ydim44, zdim44))
#define OPS_ACC45(x, y, z, u) (ops_stencil_check_4d(45, x, y, z, u, xdim45, ydim45, zdim45))
#define OPS_ACC46(x, y, z, u) (ops_stencil_check_4d(46, x, y, z, u, xdim46, ydim46, zdim46))
#define OPS_ACC47(x, y, z, u) (ops_stencil_check_4d(47, x, y, z, u, xdim47, ydim47, zdim47))
#define OPS_ACC48(x, y, z, u) (ops_stencil_check_4d(48, x, y, z, u, xdim48, ydim48, zdim48))
#define OPS_ACC49(x, y, z, u) (ops_stencil_check_4d(49, x, y, z, u, xdim49, ydim49, zdim49))
#define OPS_ACC50(x, y, z, u) (ops_stencil_check_4d(50, x, y, z, u, xdim50, ydim50, zdim50))
#define OPS_ACC51(x, y, z, u) (ops_stencil_check_4d(51, x, y, z, u, xdim51, ydim51, zdim51))
#define OPS_ACC52(x, y, z, u) (ops_stencil_check_4d(52, x, y, z, u, xdim52, ydim52, zdim52))
#define OPS_ACC53(x, y, z, u) (ops_stencil_check_4d(53, x, y, z, u, xdim53, ydim53, zdim53))
#define OPS_ACC54(x, y, z, u) (ops_stencil_check_4d(54, x, y, z, u, xdim54, ydim54, zdim54))
#define OPS_ACC55(x, y, z, u) (ops_stencil_check_4d(55, x, y, z, u, xdim55, ydim55, zdim55))
#define OPS_ACC56(x, y, z, u) (ops_stencil_check_4d(56, x, y, z, u, xdim56, ydim56, zdim56))
#define OPS_ACC57(x, y, z, u) (ops_stencil_check_4d(57, x, y, z, u, xdim57, ydim57, zdim57))
#define OPS_ACC58(x, y, z, u) (ops_stencil_check_4d(58, x, y, z, u, xdim58, ydim58, zdim58))
#define OPS_ACC59(x, y, z, u) (ops_stencil_check_4d(59, x, y, z, u, xdim59, ydim59, zdim59))
#define OPS_ACC60(x, y, z, u) (ops_stencil_check_4d(60, x, y, z, u, xdim60, ydim60, zdim60))
#define OPS_ACC61(x, y, z, u) (ops_stencil_check_4d(61, x, y, z, u, xdim61, ydim61, zdim61))
#define OPS_ACC62(x, y, z, u) (ops_stencil_check_4d(62, x, y, z, u, xdim62, ydim62, zdim62))
#define OPS_ACC63(x, y, z, u) (ops_stencil_check_4d(63, x, y, z, u, xdim63, ydim63, zdim63))
#define OPS_ACC64(x, y, z, u) (ops_stencil_check_4d(64, x, y, z, u, xdim64, ydim64, zdim64))
#define OPS_ACC65(x, y, z, u) (ops_stencil_check_4d(65, x, y, z, u, xdim65, ydim65, zdim65))
#define OPS_ACC66(x, y, z, u) (ops_stencil_check_4d(66, x, y, z, u, xdim66, ydim66, zdim66))
#define OPS_ACC67(x, y, z, u) (ops_stencil_check_4d(67, x, y, z, u, xdim67, ydim67, zdim67))
#define OPS_ACC68(x, y, z, u) (ops_stencil_check_4d(68, x, y, z, u, xdim68, ydim68, zdim68))
#define OPS_ACC69(x, y, z, u) (ops_stencil_check_4d(69, x, y, z, u, xdim69, ydim69, zdim69))
#define OPS_ACC70(x, y, z, u) (ops_stencil_check_4d(70, x, y, z, u, xdim70, ydim70, zdim70))
#define OPS_ACC71(x, y, z, u) (ops_stencil_check_4d(71, x, y, z, u, xdim71, ydim71, zdim71))
#define OPS_ACC72(x, y, z, u) (ops_stencil_check_4d(72, x, y, z, u, xdim72, ydim72, zdim72))
#define OPS_ACC73(x, y, z, u) (ops_stencil_check_4d(73, x, y, z, u, xdim73, ydim73, zdim73))
#define OPS_ACC74(x, y, z, u) (ops_stencil_check_4d(74, x, y, z, u, xdim74, ydim74, zdim74))
#define OPS_ACC75(x, y, z, u) (ops_stencil_check_4d(75, x, y, z, u, xdim75, ydim75, zdim75))
#define OPS_ACC76(x, y, z, u) (ops_stencil_check_4d(76, x, y, z, u, xdim76, ydim76, zdim76))
#define OPS_ACC77(x, y, z, u) (ops_stencil_check_4d(77, x, y, z, u, xdim77, ydim77, zdim77))
#define OPS_ACC78(x, y, z, u) (ops_stencil_check_4d(78, x, y, z, u, xdim78, ydim78, zdim78))
#define OPS_ACC79(x, y, z, u) (ops_stencil_check_4d(79, x, y, z, u, xdim79, ydim79, zdim79))
#define OPS_ACC80(x, y, z, u) (ops_stencil_check_4d(80, x, y, z, u, xdim80, ydim80, zdim80))
#define OPS_ACC81(x, y, z, u) (ops_stencil_check_4d(81, x, y, z, u, xdim81, ydim81, zdim81))
#define OPS_ACC82(x, y, z, u) (ops_stencil_check_4d(82, x, y, z, u, xdim82, ydim82, zdim82))
#define OPS_ACC83(x, y, z, u) (ops_stencil_check_4d(83, x, y, z, u, xdim83, ydim83, zdim83))
#define OPS_ACC84(x, y, z, u) (ops_stencil_check_4d(84, x, y, z, u, xdim84, ydim84, zdim84))
#define OPS_ACC85(x, y, z, u) (ops_stencil_check_4d(85, x, y, z, u, xdim85, ydim85, zdim85))
#define OPS_ACC86(x, y, z, u) (ops_stencil_check_4d(86, x, y, z, u, xdim86, ydim86, zdim86))
#define OPS_ACC87(x, y, z, u) (ops_stencil_check_4d(87, x, y, z, u, xdim87, ydim87, zdim87))
#define OPS_ACC88(x, y, z, u) (ops_stencil_check_4d(88, x, y, z, u, xdim88, ydim88, zdim88))
#define OPS_ACC89(x, y, z, u) (ops_stencil_check_4d(89, x, y, z, u, xdim89, ydim89, zdim89))
#define OPS_ACC90(x, y, z, u) (ops_stencil_check_4d(90, x, y, z, u, xdim90, ydim90, zdim90))
#define OPS_ACC91(x, y, z, u) (ops_stencil_check_4d(91, x, y, z, u, xdim91, ydim91, zdim91))
#define OPS_ACC92(x, y, z, u) (ops_stencil_check_4d(92, x, y, z, u, xdim92, ydim92, zdim92))
#define OPS_ACC93(x, y, z, u) (ops_stencil_check_4d(93, x, y, z, u, xdim93, ydim93, zdim93))
#define OPS_ACC94(x, y, z, u) (ops_stencil_check_4d(94, x, y, z, u, xdim94, ydim94, zdim94))
#define OPS_ACC95(x, y, z, u) (ops_stencil_check_4d(95, x, y, z, u, xdim95, ydim95, zdim95))
#define OPS_ACC96(x, y, z, u) (ops_stencil_check_4d(96, x, y, z, u, xdim96, ydim96, zdim96))
#define OPS_ACC97(x, y, z, u) (ops_stencil_check_4d(97, x, y, z, u, xdim97, ydim97, zdim97))
#define OPS_ACC98(x, y, z, u) (ops_stencil_check_4d(98, x, y, z, u, xdim98, ydim98, zdim98))
#define OPS_ACC99(x, y, z, u) (ops_stencil_check_4d(99, x, y, z, u, xdim99, ydim99, zdim99))
#endif               // end debug checks

#elif defined OPS_3D     // macros for 3D application
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

/**---------Multi-D ops_dats macros (multiple elements per grid point)-------**/
#ifndef OPS_ACC_MD_MACROS
#define OPS_ACC_MD_MACROS
#ifdef OPS_5D
#ifndef OPS_DEBUG
#ifndef OPS_SOA
#define OPS_ACC_MD0(d, x, y, z, u, v)  ((d) + multi_d0  * (x) + xdim0  * multi_d0  * (y) + ydim0  * xdim0  * multi_d0  * (z) + zdim0  * ydim0  * xdim0  * multi_d0  * (u) + udim0  * zdim0  * ydim0  * xdim0  * multi_d0  * (v))
#define OPS_ACC_MD1(d, x, y, z, u, v)  ((d) + multi_d1  * (x) + xdim1  * multi_d1  * (y) + ydim1  * xdim1  * multi_d1  * (z) + zdim1  * ydim1  * xdim1  * multi_d1  * (u) + udim1  * zdim1  * ydim1  * xdim1  * multi_d1  * (v))
#define OPS_ACC_MD2(d, x, y, z, u, v)  ((d) + multi_d2  * (x) + xdim2  * multi_d2  * (y) + ydim2  * xdim2  * multi_d2  * (z) + zdim2  * ydim2  * xdim2  * multi_d2  * (u) + udim2  * zdim2  * ydim2  * xdim2  * multi_d2  * (v))
#define OPS_ACC_MD3(d, x, y, z, u, v)  ((d) + multi_d3  * (x) + xdim3  * multi_d3  * (y) + ydim3  * xdim3  * multi_d3  * (z) + zdim3  * ydim3  * xdim3  * multi_d3  * (u) + udim3  * zdim3  * ydim3  * xdim3  * multi_d3  * (v))
#define OPS_ACC_MD4(d, x, y, z, u, v)  ((d) + multi_d4  * (x) + xdim4  * multi_d4  * (y) + ydim4  * xdim4  * multi_d4  * (z) + zdim4  * ydim4  * xdim4  * multi_d4  * (u) + udim4  * zdim4  * ydim4  * xdim4  * multi_d4  * (v))
#define OPS_ACC_MD5(d, x, y, z, u, v)  ((d) + multi_d5  * (x) + xdim5  * multi_d5  * (y) + ydim5  * xdim5  * multi_d5  * (z) + zdim5  * ydim5  * xdim5  * multi_d5  * (u) + udim5  * zdim5  * ydim5  * xdim5  * multi_d5  * (v))
#define OPS_ACC_MD6(d, x, y, z, u, v)  ((d) + multi_d6  * (x) + xdim6  * multi_d6  * (y) + ydim6  * xdim6  * multi_d6  * (z) + zdim6  * ydim6  * xdim6  * multi_d6  * (u) + udim6  * zdim6  * ydim6  * xdim6  * multi_d6  * (v))
#define OPS_ACC_MD7(d, x, y, z, u, v)  ((d) + multi_d7  * (x) + xdim7  * multi_d7  * (y) + ydim7  * xdim7  * multi_d7  * (z) + zdim7  * ydim7  * xdim7  * multi_d7  * (u) + udim7  * zdim7  * ydim7  * xdim7  * multi_d7  * (v))
#define OPS_ACC_MD8(d, x, y, z, u, v)  ((d) + multi_d8  * (x) + xdim8  * multi_d8  * (y) + ydim8  * xdim8  * multi_d8  * (z) + zdim8  * ydim8  * xdim8  * multi_d8  * (u) + udim8  * zdim8  * ydim8  * xdim8  * multi_d8  * (v))
#define OPS_ACC_MD9(d, x, y, z, u, v)  ((d) + multi_d9  * (x) + xdim9  * multi_d9  * (y) + ydim9  * xdim9  * multi_d9  * (z) + zdim9  * ydim9  * xdim9  * multi_d9  * (u) + udim9  * zdim9  * ydim9  * xdim9  * multi_d9  * (v))
#define OPS_ACC_MD10(d, x, y, z, u, v) ((d) + multi_d10 * (x) + xdim10 * multi_d10 * (y) + ydim10 * xdim10 * multi_d10 * (z) + zdim10 * ydim10 * xdim10 * multi_d10 * (u) + udim10 * zdim10 * ydim10 * xdim10 * multi_d10 * (v))
#define OPS_ACC_MD11(d, x, y, z, u, v) ((d) + multi_d11 * (x) + xdim11 * multi_d11 * (y) + ydim11 * xdim11 * multi_d11 * (z) + zdim11 * ydim11 * xdim11 * multi_d11 * (u) + udim11 * zdim11 * ydim11 * xdim11 * multi_d11 * (v))
#define OPS_ACC_MD12(d, x, y, z, u, v) ((d) + multi_d12 * (x) + xdim12 * multi_d12 * (y) + ydim12 * xdim12 * multi_d12 * (z) + zdim12 * ydim12 * xdim12 * multi_d12 * (u) + udim12 * zdim12 * ydim12 * xdim12 * multi_d12 * (v))
#define OPS_ACC_MD13(d, x, y, z, u, v) ((d) + multi_d13 * (x) + xdim13 * multi_d13 * (y) + ydim13 * xdim13 * multi_d13 * (z) + zdim13 * ydim13 * xdim13 * multi_d13 * (u) + udim13 * zdim13 * ydim13 * xdim13 * multi_d13 * (v))
#define OPS_ACC_MD14(d, x, y, z, u, v) ((d) + multi_d14 * (x) + xdim14 * multi_d14 * (y) + ydim14 * xdim14 * multi_d14 * (z) + zdim14 * ydim14 * xdim14 * multi_d14 * (u) + udim14 * zdim14 * ydim14 * xdim14 * multi_d14 * (v))
#define OPS_ACC_MD15(d, x, y, z, u, v) ((d) + multi_d15 * (x) + xdim15 * multi_d15 * (y) + ydim15 * xdim15 * multi_d15 * (z) + zdim15 * ydim15 * xdim15 * multi_d15 * (u) + udim15 * zdim15 * ydim15 * xdim15 * multi_d15 * (v))
#define OPS_ACC_MD16(d, x, y, z, u, v) ((d) + multi_d16 * (x) + xdim16 * multi_d16 * (y) + ydim16 * xdim16 * multi_d16 * (z) + zdim16 * ydim16 * xdim16 * multi_d16 * (u) + udim16 * zdim16 * ydim16 * xdim16 * multi_d16 * (v))
#define OPS_ACC_MD17(d, x, y, z, u, v) ((d) + multi_d17 * (x) + xdim17 * multi_d17 * (y) + ydim17 * xdim17 * multi_d17 * (z) + zdim17 * ydim17 * xdim17 * multi_d17 * (u) + udim17 * zdim17 * ydim17 * xdim17 * multi_d17 * (v))
#define OPS_ACC_MD18(d, x, y, z, u, v) ((d) + multi_d18 * (x) + xdim18 * multi_d18 * (y) + ydim18 * xdim18 * multi_d18 * (z) + zdim18 * ydim18 * xdim18 * multi_d18 * (u) + udim18 * zdim18 * ydim18 * xdim18 * multi_d18 * (v))
#define OPS_ACC_MD19(d, x, y, z, u, v) ((d) + multi_d19 * (x) + xdim19 * multi_d19 * (y) + ydim19 * xdim19 * multi_d19 * (z) + zdim19 * ydim19 * xdim19 * multi_d19 * (u) + udim19 * zdim19 * ydim19 * xdim19 * multi_d19 * (v))
#define OPS_ACC_MD20(d, x, y, z, u, v) ((d) + multi_d20 * (x) + xdim20 * multi_d20 * (y) + ydim20 * xdim20 * multi_d20 * (z) + zdim20 * ydim20 * xdim20 * multi_d20 * (u) + udim20 * zdim20 * ydim20 * xdim20 * multi_d20 * (v))
#define OPS_ACC_MD21(d, x, y, z, u, v) ((d) + multi_d21 * (x) + xdim21 * multi_d21 * (y) + ydim21 * xdim21 * multi_d21 * (z) + zdim21 * ydim21 * xdim21 * multi_d21 * (u) + udim21 * zdim21 * ydim21 * xdim21 * multi_d21 * (v))
#define OPS_ACC_MD22(d, x, y, z, u, v) ((d) + multi_d22 * (x) + xdim22 * multi_d22 * (y) + ydim22 * xdim22 * multi_d22 * (z) + zdim22 * ydim22 * xdim22 * multi_d22 * (u) + udim22 * zdim22 * ydim22 * xdim22 * multi_d22 * (v))
#define OPS_ACC_MD23(d, x, y, z, u, v) ((d) + multi_d23 * (x) + xdim23 * multi_d23 * (y) + ydim23 * xdim23 * multi_d23 * (z) + zdim23 * ydim23 * xdim23 * multi_d23 * (u) + udim23 * zdim23 * ydim23 * xdim23 * multi_d23 * (v))
#define OPS_ACC_MD24(d, x, y, z, u, v) ((d) + multi_d24 * (x) + xdim24 * multi_d24 * (y) + ydim24 * xdim24 * multi_d24 * (z) + zdim24 * ydim24 * xdim24 * multi_d24 * (u) + udim24 * zdim24 * ydim24 * xdim24 * multi_d24 * (v))
#define OPS_ACC_MD25(d, x, y, z, u, v) ((d) + multi_d25 * (x) + xdim25 * multi_d25 * (y) + ydim25 * xdim25 * multi_d25 * (z) + zdim25 * ydim25 * xdim25 * multi_d25 * (u) + udim25 * zdim25 * ydim25 * xdim25 * multi_d25 * (v))
#define OPS_ACC_MD26(d, x, y, z, u, v) ((d) + multi_d26 * (x) + xdim26 * multi_d26 * (y) + ydim26 * xdim26 * multi_d26 * (z) + zdim26 * ydim26 * xdim26 * multi_d26 * (u) + udim26 * zdim26 * ydim26 * xdim26 * multi_d26 * (v))
#define OPS_ACC_MD27(d, x, y, z, u, v) ((d) + multi_d27 * (x) + xdim27 * multi_d27 * (y) + ydim27 * xdim27 * multi_d27 * (z) + zdim27 * ydim27 * xdim27 * multi_d27 * (u) + udim27 * zdim27 * ydim27 * xdim27 * multi_d27 * (v))
#define OPS_ACC_MD28(d, x, y, z, u, v) ((d) + multi_d28 * (x) + xdim28 * multi_d28 * (y) + ydim28 * xdim28 * multi_d28 * (z) + zdim28 * ydim28 * xdim28 * multi_d28 * (u) + udim28 * zdim28 * ydim28 * xdim28 * multi_d28 * (v))
#define OPS_ACC_MD29(d, x, y, z, u, v) ((d) + multi_d29 * (x) + xdim29 * multi_d29 * (y) + ydim29 * xdim29 * multi_d29 * (z) + zdim29 * ydim29 * xdim29 * multi_d29 * (u) + udim29 * zdim29 * ydim29 * xdim29 * multi_d29 * (v))
#define OPS_ACC_MD30(d, x, y, z, u, v) ((d) + multi_d30 * (x) + xdim30 * multi_d30 * (y) + ydim30 * xdim30 * multi_d30 * (z) + zdim30 * ydim30 * xdim30 * multi_d30 * (u) + udim30 * zdim30 * ydim30 * xdim30 * multi_d30 * (v))
#define OPS_ACC_MD31(d, x, y, z, u, v) ((d) + multi_d31 * (x) + xdim31 * multi_d31 * (y) + ydim31 * xdim31 * multi_d31 * (z) + zdim31 * ydim31 * xdim31 * multi_d31 * (u) + udim31 * zdim31 * ydim31 * xdim31 * multi_d31 * (v))
#define OPS_ACC_MD32(d, x, y, z, u, v) ((d) + multi_d32 * (x) + xdim32 * multi_d32 * (y) + ydim32 * xdim32 * multi_d32 * (z) + zdim32 * ydim32 * xdim32 * multi_d32 * (u) + udim32 * zdim32 * ydim32 * xdim32 * multi_d32 * (v))
#define OPS_ACC_MD33(d, x, y, z, u, v) ((d) + multi_d33 * (x) + xdim33 * multi_d33 * (y) + ydim33 * xdim33 * multi_d33 * (z) + zdim33 * ydim33 * xdim33 * multi_d33 * (u) + udim33 * zdim33 * ydim33 * xdim33 * multi_d33 * (v))
#define OPS_ACC_MD34(d, x, y, z, u, v) ((d) + multi_d34 * (x) + xdim34 * multi_d34 * (y) + ydim34 * xdim34 * multi_d34 * (z) + zdim34 * ydim34 * xdim34 * multi_d34 * (u) + udim34 * zdim34 * ydim34 * xdim34 * multi_d34 * (v))
#define OPS_ACC_MD35(d, x, y, z, u, v) ((d) + multi_d35 * (x) + xdim35 * multi_d35 * (y) + ydim35 * xdim35 * multi_d35 * (z) + zdim35 * ydim35 * xdim35 * multi_d35 * (u) + udim35 * zdim35 * ydim35 * xdim35 * multi_d35 * (v))
#define OPS_ACC_MD36(d, x, y, z, u, v) ((d) + multi_d36 * (x) + xdim36 * multi_d36 * (y) + ydim36 * xdim36 * multi_d36 * (z) + zdim36 * ydim36 * xdim36 * multi_d36 * (u) + udim36 * zdim36 * ydim36 * xdim36 * multi_d36 * (v))
#define OPS_ACC_MD37(d, x, y, z, u, v) ((d) + multi_d37 * (x) + xdim37 * multi_d37 * (y) + ydim37 * xdim37 * multi_d37 * (z) + zdim37 * ydim37 * xdim37 * multi_d37 * (u) + udim37 * zdim37 * ydim37 * xdim37 * multi_d37 * (v))
#define OPS_ACC_MD38(d, x, y, z, u, v) ((d) + multi_d38 * (x) + xdim38 * multi_d38 * (y) + ydim38 * xdim38 * multi_d38 * (z) + zdim38 * ydim38 * xdim38 * multi_d38 * (u) + udim38 * zdim38 * ydim38 * xdim38 * multi_d38 * (v))
#define OPS_ACC_MD39(d, x, y, z, u, v) ((d) + multi_d39 * (x) + xdim39 * multi_d39 * (y) + ydim39 * xdim39 * multi_d39 * (z) + zdim39 * ydim39 * xdim39 * multi_d39 * (u) + udim39 * zdim39 * ydim39 * xdim39 * multi_d39 * (v))
#define OPS_ACC_MD40(d, x, y, z, u, v) ((d) + multi_d40 * (x) + xdim40 * multi_d40 * (y) + ydim40 * xdim40 * multi_d40 * (z) + zdim40 * ydim40 * xdim40 * multi_d40 * (u) + udim40 * zdim40 * ydim40 * xdim40 * multi_d40 * (v))
#define OPS_ACC_MD41(d, x, y, z, u, v) ((d) + multi_d41 * (x) + xdim41 * multi_d41 * (y) + ydim41 * xdim41 * multi_d41 * (z) + zdim41 * ydim41 * xdim41 * multi_d41 * (u) + udim41 * zdim41 * ydim41 * xdim41 * multi_d41 * (v))
#define OPS_ACC_MD42(d, x, y, z, u, v) ((d) + multi_d42 * (x) + xdim42 * multi_d42 * (y) + ydim42 * xdim42 * multi_d42 * (z) + zdim42 * ydim42 * xdim42 * multi_d42 * (u) + udim42 * zdim42 * ydim42 * xdim42 * multi_d42 * (v))
#define OPS_ACC_MD43(d, x, y, z, u, v) ((d) + multi_d43 * (x) + xdim43 * multi_d43 * (y) + ydim43 * xdim43 * multi_d43 * (z) + zdim43 * ydim43 * xdim43 * multi_d43 * (u) + udim43 * zdim43 * ydim43 * xdim43 * multi_d43 * (v))
#define OPS_ACC_MD44(d, x, y, z, u, v) ((d) + multi_d44 * (x) + xdim44 * multi_d44 * (y) + ydim44 * xdim44 * multi_d44 * (z) + zdim44 * ydim44 * xdim44 * multi_d44 * (u) + udim44 * zdim44 * ydim44 * xdim44 * multi_d44 * (v))
#define OPS_ACC_MD45(d, x, y, z, u, v) ((d) + multi_d45 * (x) + xdim45 * multi_d45 * (y) + ydim45 * xdim45 * multi_d45 * (z) + zdim45 * ydim45 * xdim45 * multi_d45 * (u) + udim45 * zdim45 * ydim45 * xdim45 * multi_d45 * (v))
#define OPS_ACC_MD46(d, x, y, z, u, v) ((d) + multi_d46 * (x) + xdim46 * multi_d46 * (y) + ydim46 * xdim46 * multi_d46 * (z) + zdim46 * ydim46 * xdim46 * multi_d46 * (u) + udim46 * zdim46 * ydim46 * xdim46 * multi_d46 * (v))
#define OPS_ACC_MD47(d, x, y, z, u, v) ((d) + multi_d47 * (x) + xdim47 * multi_d47 * (y) + ydim47 * xdim47 * multi_d47 * (z) + zdim47 * ydim47 * xdim47 * multi_d47 * (u) + udim47 * zdim47 * ydim47 * xdim47 * multi_d47 * (v))
#define OPS_ACC_MD48(d, x, y, z, u, v) ((d) + multi_d48 * (x) + xdim48 * multi_d48 * (y) + ydim48 * xdim48 * multi_d48 * (z) + zdim48 * ydim48 * xdim48 * multi_d48 * (u) + udim48 * zdim48 * ydim48 * xdim48 * multi_d48 * (v))
#define OPS_ACC_MD49(d, x, y, z, u, v) ((d) + multi_d49 * (x) + xdim49 * multi_d49 * (y) + ydim49 * xdim49 * multi_d49 * (z) + zdim49 * ydim49 * xdim49 * multi_d49 * (u) + udim49 * zdim49 * ydim49 * xdim49 * multi_d49 * (v))
#define OPS_ACC_MD50(d, x, y, z, u, v) ((d) + multi_d50 * (x) + xdim50 * multi_d50 * (y) + ydim50 * xdim50 * multi_d50 * (z) + zdim50 * ydim50 * xdim50 * multi_d50 * (u) + udim50 * zdim50 * ydim50 * xdim50 * multi_d50 * (v))
#define OPS_ACC_MD51(d, x, y, z, u, v) ((d) + multi_d51 * (x) + xdim51 * multi_d51 * (y) + ydim51 * xdim51 * multi_d51 * (z) + zdim51 * ydim51 * xdim51 * multi_d51 * (u) + udim51 * zdim51 * ydim51 * xdim51 * multi_d51 * (v))
#define OPS_ACC_MD52(d, x, y, z, u, v) ((d) + multi_d52 * (x) + xdim52 * multi_d52 * (y) + ydim52 * xdim52 * multi_d52 * (z) + zdim52 * ydim52 * xdim52 * multi_d52 * (u) + udim52 * zdim52 * ydim52 * xdim52 * multi_d52 * (v))
#define OPS_ACC_MD53(d, x, y, z, u, v) ((d) + multi_d53 * (x) + xdim53 * multi_d53 * (y) + ydim53 * xdim53 * multi_d53 * (z) + zdim53 * ydim53 * xdim53 * multi_d53 * (u) + udim53 * zdim53 * ydim53 * xdim53 * multi_d53 * (v))
#define OPS_ACC_MD54(d, x, y, z, u, v) ((d) + multi_d54 * (x) + xdim54 * multi_d54 * (y) + ydim54 * xdim54 * multi_d54 * (z) + zdim54 * ydim54 * xdim54 * multi_d54 * (u) + udim54 * zdim54 * ydim54 * xdim54 * multi_d54 * (v))
#define OPS_ACC_MD55(d, x, y, z, u, v) ((d) + multi_d55 * (x) + xdim55 * multi_d55 * (y) + ydim55 * xdim55 * multi_d55 * (z) + zdim55 * ydim55 * xdim55 * multi_d55 * (u) + udim55 * zdim55 * ydim55 * xdim55 * multi_d55 * (v))
#define OPS_ACC_MD56(d, x, y, z, u, v) ((d) + multi_d56 * (x) + xdim56 * multi_d56 * (y) + ydim56 * xdim56 * multi_d56 * (z) + zdim56 * ydim56 * xdim56 * multi_d56 * (u) + udim56 * zdim56 * ydim56 * xdim56 * multi_d56 * (v))
#define OPS_ACC_MD57(d, x, y, z, u, v) ((d) + multi_d57 * (x) + xdim57 * multi_d57 * (y) + ydim57 * xdim57 * multi_d57 * (z) + zdim57 * ydim57 * xdim57 * multi_d57 * (u) + udim57 * zdim57 * ydim57 * xdim57 * multi_d57 * (v))
#define OPS_ACC_MD58(d, x, y, z, u, v) ((d) + multi_d58 * (x) + xdim58 * multi_d58 * (y) + ydim58 * xdim58 * multi_d58 * (z) + zdim58 * ydim58 * xdim58 * multi_d58 * (u) + udim58 * zdim58 * ydim58 * xdim58 * multi_d58 * (v))
#define OPS_ACC_MD59(d, x, y, z, u, v) ((d) + multi_d59 * (x) + xdim59 * multi_d59 * (y) + ydim59 * xdim59 * multi_d59 * (z) + zdim59 * ydim59 * xdim59 * multi_d59 * (u) + udim59 * zdim59 * ydim59 * xdim59 * multi_d59 * (v))
#define OPS_ACC_MD60(d, x, y, z, u, v) ((d) + multi_d60 * (x) + xdim60 * multi_d60 * (y) + ydim60 * xdim60 * multi_d60 * (z) + zdim60 * ydim60 * xdim60 * multi_d60 * (u) + udim60 * zdim60 * ydim60 * xdim60 * multi_d60 * (v))
#define OPS_ACC_MD61(d, x, y, z, u, v) ((d) + multi_d61 * (x) + xdim61 * multi_d61 * (y) + ydim61 * xdim61 * multi_d61 * (z) + zdim61 * ydim61 * xdim61 * multi_d61 * (u) + udim61 * zdim61 * ydim61 * xdim61 * multi_d61 * (v))
#define OPS_ACC_MD62(d, x, y, z, u, v) ((d) + multi_d62 * (x) + xdim62 * multi_d62 * (y) + ydim62 * xdim62 * multi_d62 * (z) + zdim62 * ydim62 * xdim62 * multi_d62 * (u) + udim62 * zdim62 * ydim62 * xdim62 * multi_d62 * (v))
#define OPS_ACC_MD63(d, x, y, z, u, v) ((d) + multi_d63 * (x) + xdim63 * multi_d63 * (y) + ydim63 * xdim63 * multi_d63 * (z) + zdim63 * ydim63 * xdim63 * multi_d63 * (u) + udim63 * zdim63 * ydim63 * xdim63 * multi_d63 * (v))
#define OPS_ACC_MD64(d, x, y, z, u, v) ((d) + multi_d64 * (x) + xdim64 * multi_d64 * (y) + ydim64 * xdim64 * multi_d64 * (z) + zdim64 * ydim64 * xdim64 * multi_d64 * (u) + udim64 * zdim64 * ydim64 * xdim64 * multi_d64 * (v))
#define OPS_ACC_MD65(d, x, y, z, u, v) ((d) + multi_d65 * (x) + xdim65 * multi_d65 * (y) + ydim65 * xdim65 * multi_d65 * (z) + zdim65 * ydim65 * xdim65 * multi_d65 * (u) + udim65 * zdim65 * ydim65 * xdim65 * multi_d65 * (v))
#define OPS_ACC_MD66(d, x, y, z, u, v) ((d) + multi_d66 * (x) + xdim66 * multi_d66 * (y) + ydim66 * xdim66 * multi_d66 * (z) + zdim66 * ydim66 * xdim66 * multi_d66 * (u) + udim66 * zdim66 * ydim66 * xdim66 * multi_d66 * (v))
#define OPS_ACC_MD67(d, x, y, z, u, v) ((d) + multi_d67 * (x) + xdim67 * multi_d67 * (y) + ydim67 * xdim67 * multi_d67 * (z) + zdim67 * ydim67 * xdim67 * multi_d67 * (u) + udim67 * zdim67 * ydim67 * xdim67 * multi_d67 * (v))
#define OPS_ACC_MD68(d, x, y, z, u, v) ((d) + multi_d68 * (x) + xdim68 * multi_d68 * (y) + ydim68 * xdim68 * multi_d68 * (z) + zdim68 * ydim68 * xdim68 * multi_d68 * (u) + udim68 * zdim68 * ydim68 * xdim68 * multi_d68 * (v))
#define OPS_ACC_MD69(d, x, y, z, u, v) ((d) + multi_d69 * (x) + xdim69 * multi_d69 * (y) + ydim69 * xdim69 * multi_d69 * (z) + zdim69 * ydim69 * xdim69 * multi_d69 * (u) + udim69 * zdim69 * ydim69 * xdim69 * multi_d69 * (v))
#define OPS_ACC_MD70(d, x, y, z, u, v) ((d) + multi_d70 * (x) + xdim70 * multi_d70 * (y) + ydim70 * xdim70 * multi_d70 * (z) + zdim70 * ydim70 * xdim70 * multi_d70 * (u) + udim70 * zdim70 * ydim70 * xdim70 * multi_d70 * (v))
#define OPS_ACC_MD71(d, x, y, z, u, v) ((d) + multi_d71 * (x) + xdim71 * multi_d71 * (y) + ydim71 * xdim71 * multi_d71 * (z) + zdim71 * ydim71 * xdim71 * multi_d71 * (u) + udim71 * zdim71 * ydim71 * xdim71 * multi_d71 * (v))
#define OPS_ACC_MD72(d, x, y, z, u, v) ((d) + multi_d72 * (x) + xdim72 * multi_d72 * (y) + ydim72 * xdim72 * multi_d72 * (z) + zdim72 * ydim72 * xdim72 * multi_d72 * (u) + udim72 * zdim72 * ydim72 * xdim72 * multi_d72 * (v))
#define OPS_ACC_MD73(d, x, y, z, u, v) ((d) + multi_d73 * (x) + xdim73 * multi_d73 * (y) + ydim73 * xdim73 * multi_d73 * (z) + zdim73 * ydim73 * xdim73 * multi_d73 * (u) + udim73 * zdim73 * ydim73 * xdim73 * multi_d73 * (v))
#define OPS_ACC_MD74(d, x, y, z, u, v) ((d) + multi_d74 * (x) + xdim74 * multi_d74 * (y) + ydim74 * xdim74 * multi_d74 * (z) + zdim74 * ydim74 * xdim74 * multi_d74 * (u) + udim74 * zdim74 * ydim74 * xdim74 * multi_d74 * (v))
#define OPS_ACC_MD75(d, x, y, z, u, v) ((d) + multi_d75 * (x) + xdim75 * multi_d75 * (y) + ydim75 * xdim75 * multi_d75 * (z) + zdim75 * ydim75 * xdim75 * multi_d75 * (u) + udim75 * zdim75 * ydim75 * xdim75 * multi_d75 * (v))
#define OPS_ACC_MD76(d, x, y, z, u, v) ((d) + multi_d76 * (x) + xdim76 * multi_d76 * (y) + ydim76 * xdim76 * multi_d76 * (z) + zdim76 * ydim76 * xdim76 * multi_d76 * (u) + udim76 * zdim76 * ydim76 * xdim76 * multi_d76 * (v))
#define OPS_ACC_MD77(d, x, y, z, u, v) ((d) + multi_d77 * (x) + xdim77 * multi_d77 * (y) + ydim77 * xdim77 * multi_d77 * (z) + zdim77 * ydim77 * xdim77 * multi_d77 * (u) + udim77 * zdim77 * ydim77 * xdim77 * multi_d77 * (v))
#define OPS_ACC_MD78(d, x, y, z, u, v) ((d) + multi_d78 * (x) + xdim78 * multi_d78 * (y) + ydim78 * xdim78 * multi_d78 * (z) + zdim78 * ydim78 * xdim78 * multi_d78 * (u) + udim78 * zdim78 * ydim78 * xdim78 * multi_d78 * (v))
#define OPS_ACC_MD79(d, x, y, z, u, v) ((d) + multi_d79 * (x) + xdim79 * multi_d79 * (y) + ydim79 * xdim79 * multi_d79 * (z) + zdim79 * ydim79 * xdim79 * multi_d79 * (u) + udim79 * zdim79 * ydim79 * xdim79 * multi_d79 * (v))
#define OPS_ACC_MD80(d, x, y, z, u, v) ((d) + multi_d80 * (x) + xdim80 * multi_d80 * (y) + ydim80 * xdim80 * multi_d80 * (z) + zdim80 * ydim80 * xdim80 * multi_d80 * (u) + udim80 * zdim80 * ydim80 * xdim80 * multi_d80 * (v))
#define OPS_ACC_MD81(d, x, y, z, u, v) ((d) + multi_d81 * (x) + xdim81 * multi_d81 * (y) + ydim81 * xdim81 * multi_d81 * (z) + zdim81 * ydim81 * xdim81 * multi_d81 * (u) + udim81 * zdim81 * ydim81 * xdim81 * multi_d81 * (v))
#define OPS_ACC_MD82(d, x, y, z, u, v) ((d) + multi_d82 * (x) + xdim82 * multi_d82 * (y) + ydim82 * xdim82 * multi_d82 * (z) + zdim82 * ydim82 * xdim82 * multi_d82 * (u) + udim82 * zdim82 * ydim82 * xdim82 * multi_d82 * (v))
#define OPS_ACC_MD83(d, x, y, z, u, v) ((d) + multi_d83 * (x) + xdim83 * multi_d83 * (y) + ydim83 * xdim83 * multi_d83 * (z) + zdim83 * ydim83 * xdim83 * multi_d83 * (u) + udim83 * zdim83 * ydim83 * xdim83 * multi_d83 * (v))
#define OPS_ACC_MD84(d, x, y, z, u, v) ((d) + multi_d84 * (x) + xdim84 * multi_d84 * (y) + ydim84 * xdim84 * multi_d84 * (z) + zdim84 * ydim84 * xdim84 * multi_d84 * (u) + udim84 * zdim84 * ydim84 * xdim84 * multi_d84 * (v))
#define OPS_ACC_MD85(d, x, y, z, u, v) ((d) + multi_d85 * (x) + xdim85 * multi_d85 * (y) + ydim85 * xdim85 * multi_d85 * (z) + zdim85 * ydim85 * xdim85 * multi_d85 * (u) + udim85 * zdim85 * ydim85 * xdim85 * multi_d85 * (v))
#define OPS_ACC_MD86(d, x, y, z, u, v) ((d) + multi_d86 * (x) + xdim86 * multi_d86 * (y) + ydim86 * xdim86 * multi_d86 * (z) + zdim86 * ydim86 * xdim86 * multi_d86 * (u) + udim86 * zdim86 * ydim86 * xdim86 * multi_d86 * (v))
#define OPS_ACC_MD87(d, x, y, z, u, v) ((d) + multi_d87 * (x) + xdim87 * multi_d87 * (y) + ydim87 * xdim87 * multi_d87 * (z) + zdim87 * ydim87 * xdim87 * multi_d87 * (u) + udim87 * zdim87 * ydim87 * xdim87 * multi_d87 * (v))
#define OPS_ACC_MD88(d, x, y, z, u, v) ((d) + multi_d88 * (x) + xdim88 * multi_d88 * (y) + ydim88 * xdim88 * multi_d88 * (z) + zdim88 * ydim88 * xdim88 * multi_d88 * (u) + udim88 * zdim88 * ydim88 * xdim88 * multi_d88 * (v))
#define OPS_ACC_MD89(d, x, y, z, u, v) ((d) + multi_d89 * (x) + xdim89 * multi_d89 * (y) + ydim89 * xdim89 * multi_d89 * (z) + zdim89 * ydim89 * xdim89 * multi_d89 * (u) + udim89 * zdim89 * ydim89 * xdim89 * multi_d89 * (v))
#define OPS_ACC_MD90(d, x, y, z, u, v) ((d) + multi_d90 * (x) + xdim90 * multi_d90 * (y) + ydim90 * xdim90 * multi_d90 * (z) + zdim90 * ydim90 * xdim90 * multi_d90 * (u) + udim90 * zdim90 * ydim90 * xdim90 * multi_d90 * (v))
#define OPS_ACC_MD91(d, x, y, z, u, v) ((d) + multi_d91 * (x) + xdim91 * multi_d91 * (y) + ydim91 * xdim91 * multi_d91 * (z) + zdim91 * ydim91 * xdim91 * multi_d91 * (u) + udim91 * zdim91 * ydim91 * xdim91 * multi_d91 * (v))
#define OPS_ACC_MD92(d, x, y, z, u, v) ((d) + multi_d92 * (x) + xdim92 * multi_d92 * (y) + ydim92 * xdim92 * multi_d92 * (z) + zdim92 * ydim92 * xdim92 * multi_d92 * (u) + udim92 * zdim92 * ydim92 * xdim92 * multi_d92 * (v))
#define OPS_ACC_MD93(d, x, y, z, u, v) ((d) + multi_d93 * (x) + xdim93 * multi_d93 * (y) + ydim93 * xdim93 * multi_d93 * (z) + zdim93 * ydim93 * xdim93 * multi_d93 * (u) + udim93 * zdim93 * ydim93 * xdim93 * multi_d93 * (v))
#define OPS_ACC_MD94(d, x, y, z, u, v) ((d) + multi_d94 * (x) + xdim94 * multi_d94 * (y) + ydim94 * xdim94 * multi_d94 * (z) + zdim94 * ydim94 * xdim94 * multi_d94 * (u) + udim94 * zdim94 * ydim94 * xdim94 * multi_d94 * (v))
#define OPS_ACC_MD95(d, x, y, z, u, v) ((d) + multi_d95 * (x) + xdim95 * multi_d95 * (y) + ydim95 * xdim95 * multi_d95 * (z) + zdim95 * ydim95 * xdim95 * multi_d95 * (u) + udim95 * zdim95 * ydim95 * xdim95 * multi_d95 * (v))
#define OPS_ACC_MD96(d, x, y, z, u, v) ((d) + multi_d96 * (x) + xdim96 * multi_d96 * (y) + ydim96 * xdim96 * multi_d96 * (z) + zdim96 * ydim96 * xdim96 * multi_d96 * (u) + udim96 * zdim96 * ydim96 * xdim96 * multi_d96 * (v))
#define OPS_ACC_MD97(d, x, y, z, u, v) ((d) + multi_d97 * (x) + xdim97 * multi_d97 * (y) + ydim97 * xdim97 * multi_d97 * (z) + zdim97 * ydim97 * xdim97 * multi_d97 * (u) + udim97 * zdim97 * ydim97 * xdim97 * multi_d97 * (v))
#define OPS_ACC_MD98(d, x, y, z, u, v) ((d) + multi_d98 * (x) + xdim98 * multi_d98 * (y) + ydim98 * xdim98 * multi_d98 * (z) + zdim98 * ydim98 * xdim98 * multi_d98 * (u) + udim98 * zdim98 * ydim98 * xdim98 * multi_d98 * (v))
#define OPS_ACC_MD99(d, x, y, z, u, v) ((d) + multi_d99 * (x) + xdim99 * multi_d99 * (y) + ydim99 * xdim99 * multi_d99 * (z) + zdim99 * ydim99 * xdim99 * multi_d99 * (u) + udim99 * zdim99 * ydim99 * xdim99 * multi_d99 * (v))
#else
#define OPS_ACC_MD0(d, x, y, z, u, v)  ((x) + xdim0  * (y) + ydim0  * xdim0  * (z) + zdim0  * ydim0  * xdim0  * (u) + udim0  * zdim0  * ydim0  * xdim0  * (v) + (d) * vdim0  * udim0  * zdim0  * ydim0  * xdim0 )
#define OPS_ACC_MD1(d, x, y, z, u, v)  ((x) + xdim1  * (y) + ydim1  * xdim1  * (z) + zdim1  * ydim1  * xdim1  * (u) + udim1  * zdim1  * ydim1  * xdim1  * (v) + (d) * vdim1  * udim1  * zdim1  * ydim1  * xdim1 )
#define OPS_ACC_MD2(d, x, y, z, u, v)  ((x) + xdim2  * (y) + ydim2  * xdim2  * (z) + zdim2  * ydim2  * xdim2  * (u) + udim2  * zdim2  * ydim2  * xdim2  * (v) + (d) * vdim2  * udim2  * zdim2  * ydim2  * xdim2 )
#define OPS_ACC_MD3(d, x, y, z, u, v)  ((x) + xdim3  * (y) + ydim3  * xdim3  * (z) + zdim3  * ydim3  * xdim3  * (u) + udim3  * zdim3  * ydim3  * xdim3  * (v) + (d) * vdim3  * udim3  * zdim3  * ydim3  * xdim3 )
#define OPS_ACC_MD4(d, x, y, z, u, v)  ((x) + xdim4  * (y) + ydim4  * xdim4  * (z) + zdim4  * ydim4  * xdim4  * (u) + udim4  * zdim4  * ydim4  * xdim4  * (v) + (d) * vdim4  * udim4  * zdim4  * ydim4  * xdim4 )
#define OPS_ACC_MD5(d, x, y, z, u, v)  ((x) + xdim5  * (y) + ydim5  * xdim5  * (z) + zdim5  * ydim5  * xdim5  * (u) + udim5  * zdim5  * ydim5  * xdim5  * (v) + (d) * vdim5  * udim5  * zdim5  * ydim5  * xdim5 )
#define OPS_ACC_MD6(d, x, y, z, u, v)  ((x) + xdim6  * (y) + ydim6  * xdim6  * (z) + zdim6  * ydim6  * xdim6  * (u) + udim6  * zdim6  * ydim6  * xdim6  * (v) + (d) * vdim6  * udim6  * zdim6  * ydim6  * xdim6 )
#define OPS_ACC_MD7(d, x, y, z, u, v)  ((x) + xdim7  * (y) + ydim7  * xdim7  * (z) + zdim7  * ydim7  * xdim7  * (u) + udim7  * zdim7  * ydim7  * xdim7  * (v) + (d) * vdim7  * udim7  * zdim7  * ydim7  * xdim7 )
#define OPS_ACC_MD8(d, x, y, z, u, v)  ((x) + xdim8  * (y) + ydim8  * xdim8  * (z) + zdim8  * ydim8  * xdim8  * (u) + udim8  * zdim8  * ydim8  * xdim8  * (v) + (d) * vdim8  * udim8  * zdim8  * ydim8  * xdim8 )
#define OPS_ACC_MD9(d, x, y, z, u, v)  ((x) + xdim9  * (y) + ydim9  * xdim9  * (z) + zdim9  * ydim9  * xdim9  * (u) + udim9  * zdim9  * ydim9  * xdim9  * (v) + (d) * vdim9  * udim9  * zdim9  * ydim9  * xdim9 )
#define OPS_ACC_MD10(d, x, y, z, u, v) ((x) + xdim10 * (y) + ydim10 * xdim10 * (z) + zdim10 * ydim10 * xdim10 * (u) + udim10 * zdim10 * ydim10 * xdim10 * (v) + (d) * vdim10 * udim10 * zdim10 * ydim10 * xdim10)
#define OPS_ACC_MD11(d, x, y, z, u, v) ((x) + xdim11 * (y) + ydim11 * xdim11 * (z) + zdim11 * ydim11 * xdim11 * (u) + udim11 * zdim11 * ydim11 * xdim11 * (v) + (d) * vdim11 * udim11 * zdim11 * ydim11 * xdim11)
#define OPS_ACC_MD12(d, x, y, z, u, v) ((x) + xdim12 * (y) + ydim12 * xdim12 * (z) + zdim12 * ydim12 * xdim12 * (u) + udim12 * zdim12 * ydim12 * xdim12 * (v) + (d) * vdim12 * udim12 * zdim12 * ydim12 * xdim12)
#define OPS_ACC_MD13(d, x, y, z, u, v) ((x) + xdim13 * (y) + ydim13 * xdim13 * (z) + zdim13 * ydim13 * xdim13 * (u) + udim13 * zdim13 * ydim13 * xdim13 * (v) + (d) * vdim13 * udim13 * zdim13 * ydim13 * xdim13)
#define OPS_ACC_MD14(d, x, y, z, u, v) ((x) + xdim14 * (y) + ydim14 * xdim14 * (z) + zdim14 * ydim14 * xdim14 * (u) + udim14 * zdim14 * ydim14 * xdim14 * (v) + (d) * vdim14 * udim14 * zdim14 * ydim14 * xdim14)
#define OPS_ACC_MD15(d, x, y, z, u, v) ((x) + xdim15 * (y) + ydim15 * xdim15 * (z) + zdim15 * ydim15 * xdim15 * (u) + udim15 * zdim15 * ydim15 * xdim15 * (v) + (d) * vdim15 * udim15 * zdim15 * ydim15 * xdim15)
#define OPS_ACC_MD16(d, x, y, z, u, v) ((x) + xdim16 * (y) + ydim16 * xdim16 * (z) + zdim16 * ydim16 * xdim16 * (u) + udim16 * zdim16 * ydim16 * xdim16 * (v) + (d) * vdim16 * udim16 * zdim16 * ydim16 * xdim16)
#define OPS_ACC_MD17(d, x, y, z, u, v) ((x) + xdim17 * (y) + ydim17 * xdim17 * (z) + zdim17 * ydim17 * xdim17 * (u) + udim17 * zdim17 * ydim17 * xdim17 * (v) + (d) * vdim17 * udim17 * zdim17 * ydim17 * xdim17)
#define OPS_ACC_MD18(d, x, y, z, u, v) ((x) + xdim18 * (y) + ydim18 * xdim18 * (z) + zdim18 * ydim18 * xdim18 * (u) + udim18 * zdim18 * ydim18 * xdim18 * (v) + (d) * vdim18 * udim18 * zdim18 * ydim18 * xdim18)
#define OPS_ACC_MD19(d, x, y, z, u, v) ((x) + xdim19 * (y) + ydim19 * xdim19 * (z) + zdim19 * ydim19 * xdim19 * (u) + udim19 * zdim19 * ydim19 * xdim19 * (v) + (d) * vdim19 * udim19 * zdim19 * ydim19 * xdim19)
#define OPS_ACC_MD20(d, x, y, z, u, v) ((x) + xdim20 * (y) + ydim20 * xdim20 * (z) + zdim20 * ydim20 * xdim20 * (u) + udim20 * zdim20 * ydim20 * xdim20 * (v) + (d) * vdim20 * udim20 * zdim20 * ydim20 * xdim20)
#define OPS_ACC_MD21(d, x, y, z, u, v) ((x) + xdim21 * (y) + ydim21 * xdim21 * (z) + zdim21 * ydim21 * xdim21 * (u) + udim21 * zdim21 * ydim21 * xdim21 * (v) + (d) * vdim21 * udim21 * zdim21 * ydim21 * xdim21)
#define OPS_ACC_MD22(d, x, y, z, u, v) ((x) + xdim22 * (y) + ydim22 * xdim22 * (z) + zdim22 * ydim22 * xdim22 * (u) + udim22 * zdim22 * ydim22 * xdim22 * (v) + (d) * vdim22 * udim22 * zdim22 * ydim22 * xdim22)
#define OPS_ACC_MD23(d, x, y, z, u, v) ((x) + xdim23 * (y) + ydim23 * xdim23 * (z) + zdim23 * ydim23 * xdim23 * (u) + udim23 * zdim23 * ydim23 * xdim23 * (v) + (d) * vdim23 * udim23 * zdim23 * ydim23 * xdim23)
#define OPS_ACC_MD24(d, x, y, z, u, v) ((x) + xdim24 * (y) + ydim24 * xdim24 * (z) + zdim24 * ydim24 * xdim24 * (u) + udim24 * zdim24 * ydim24 * xdim24 * (v) + (d) * vdim24 * udim24 * zdim24 * ydim24 * xdim24)
#define OPS_ACC_MD25(d, x, y, z, u, v) ((x) + xdim25 * (y) + ydim25 * xdim25 * (z) + zdim25 * ydim25 * xdim25 * (u) + udim25 * zdim25 * ydim25 * xdim25 * (v) + (d) * vdim25 * udim25 * zdim25 * ydim25 * xdim25)
#define OPS_ACC_MD26(d, x, y, z, u, v) ((x) + xdim26 * (y) + ydim26 * xdim26 * (z) + zdim26 * ydim26 * xdim26 * (u) + udim26 * zdim26 * ydim26 * xdim26 * (v) + (d) * vdim26 * udim26 * zdim26 * ydim26 * xdim26)
#define OPS_ACC_MD27(d, x, y, z, u, v) ((x) + xdim27 * (y) + ydim27 * xdim27 * (z) + zdim27 * ydim27 * xdim27 * (u) + udim27 * zdim27 * ydim27 * xdim27 * (v) + (d) * vdim27 * udim27 * zdim27 * ydim27 * xdim27)
#define OPS_ACC_MD28(d, x, y, z, u, v) ((x) + xdim28 * (y) + ydim28 * xdim28 * (z) + zdim28 * ydim28 * xdim28 * (u) + udim28 * zdim28 * ydim28 * xdim28 * (v) + (d) * vdim28 * udim28 * zdim28 * ydim28 * xdim28)
#define OPS_ACC_MD29(d, x, y, z, u, v) ((x) + xdim29 * (y) + ydim29 * xdim29 * (z) + zdim29 * ydim29 * xdim29 * (u) + udim29 * zdim29 * ydim29 * xdim29 * (v) + (d) * vdim29 * udim29 * zdim29 * ydim29 * xdim29)
#define OPS_ACC_MD30(d, x, y, z, u, v) ((x) + xdim30 * (y) + ydim30 * xdim30 * (z) + zdim30 * ydim30 * xdim30 * (u) + udim30 * zdim30 * ydim30 * xdim30 * (v) + (d) * vdim30 * udim30 * zdim30 * ydim30 * xdim30)
#define OPS_ACC_MD31(d, x, y, z, u, v) ((x) + xdim31 * (y) + ydim31 * xdim31 * (z) + zdim31 * ydim31 * xdim31 * (u) + udim31 * zdim31 * ydim31 * xdim31 * (v) + (d) * vdim31 * udim31 * zdim31 * ydim31 * xdim31)
#define OPS_ACC_MD32(d, x, y, z, u, v) ((x) + xdim32 * (y) + ydim32 * xdim32 * (z) + zdim32 * ydim32 * xdim32 * (u) + udim32 * zdim32 * ydim32 * xdim32 * (v) + (d) * vdim32 * udim32 * zdim32 * ydim32 * xdim32)
#define OPS_ACC_MD33(d, x, y, z, u, v) ((x) + xdim33 * (y) + ydim33 * xdim33 * (z) + zdim33 * ydim33 * xdim33 * (u) + udim33 * zdim33 * ydim33 * xdim33 * (v) + (d) * vdim33 * udim33 * zdim33 * ydim33 * xdim33)
#define OPS_ACC_MD34(d, x, y, z, u, v) ((x) + xdim34 * (y) + ydim34 * xdim34 * (z) + zdim34 * ydim34 * xdim34 * (u) + udim34 * zdim34 * ydim34 * xdim34 * (v) + (d) * vdim34 * udim34 * zdim34 * ydim34 * xdim34)
#define OPS_ACC_MD35(d, x, y, z, u, v) ((x) + xdim35 * (y) + ydim35 * xdim35 * (z) + zdim35 * ydim35 * xdim35 * (u) + udim35 * zdim35 * ydim35 * xdim35 * (v) + (d) * vdim35 * udim35 * zdim35 * ydim35 * xdim35)
#define OPS_ACC_MD36(d, x, y, z, u, v) ((x) + xdim36 * (y) + ydim36 * xdim36 * (z) + zdim36 * ydim36 * xdim36 * (u) + udim36 * zdim36 * ydim36 * xdim36 * (v) + (d) * vdim36 * udim36 * zdim36 * ydim36 * xdim36)
#define OPS_ACC_MD37(d, x, y, z, u, v) ((x) + xdim37 * (y) + ydim37 * xdim37 * (z) + zdim37 * ydim37 * xdim37 * (u) + udim37 * zdim37 * ydim37 * xdim37 * (v) + (d) * vdim37 * udim37 * zdim37 * ydim37 * xdim37)
#define OPS_ACC_MD38(d, x, y, z, u, v) ((x) + xdim38 * (y) + ydim38 * xdim38 * (z) + zdim38 * ydim38 * xdim38 * (u) + udim38 * zdim38 * ydim38 * xdim38 * (v) + (d) * vdim38 * udim38 * zdim38 * ydim38 * xdim38)
#define OPS_ACC_MD39(d, x, y, z, u, v) ((x) + xdim39 * (y) + ydim39 * xdim39 * (z) + zdim39 * ydim39 * xdim39 * (u) + udim39 * zdim39 * ydim39 * xdim39 * (v) + (d) * vdim39 * udim39 * zdim39 * ydim39 * xdim39)
#define OPS_ACC_MD40(d, x, y, z, u, v) ((x) + xdim40 * (y) + ydim40 * xdim40 * (z) + zdim40 * ydim40 * xdim40 * (u) + udim40 * zdim40 * ydim40 * xdim40 * (v) + (d) * vdim40 * udim40 * zdim40 * ydim40 * xdim40)
#define OPS_ACC_MD41(d, x, y, z, u, v) ((x) + xdim41 * (y) + ydim41 * xdim41 * (z) + zdim41 * ydim41 * xdim41 * (u) + udim41 * zdim41 * ydim41 * xdim41 * (v) + (d) * vdim41 * udim41 * zdim41 * ydim41 * xdim41)
#define OPS_ACC_MD42(d, x, y, z, u, v) ((x) + xdim42 * (y) + ydim42 * xdim42 * (z) + zdim42 * ydim42 * xdim42 * (u) + udim42 * zdim42 * ydim42 * xdim42 * (v) + (d) * vdim42 * udim42 * zdim42 * ydim42 * xdim42)
#define OPS_ACC_MD43(d, x, y, z, u, v) ((x) + xdim43 * (y) + ydim43 * xdim43 * (z) + zdim43 * ydim43 * xdim43 * (u) + udim43 * zdim43 * ydim43 * xdim43 * (v) + (d) * vdim43 * udim43 * zdim43 * ydim43 * xdim43)
#define OPS_ACC_MD44(d, x, y, z, u, v) ((x) + xdim44 * (y) + ydim44 * xdim44 * (z) + zdim44 * ydim44 * xdim44 * (u) + udim44 * zdim44 * ydim44 * xdim44 * (v) + (d) * vdim44 * udim44 * zdim44 * ydim44 * xdim44)
#define OPS_ACC_MD45(d, x, y, z, u, v) ((x) + xdim45 * (y) + ydim45 * xdim45 * (z) + zdim45 * ydim45 * xdim45 * (u) + udim45 * zdim45 * ydim45 * xdim45 * (v) + (d) * vdim45 * udim45 * zdim45 * ydim45 * xdim45)
#define OPS_ACC_MD46(d, x, y, z, u, v) ((x) + xdim46 * (y) + ydim46 * xdim46 * (z) + zdim46 * ydim46 * xdim46 * (u) + udim46 * zdim46 * ydim46 * xdim46 * (v) + (d) * vdim46 * udim46 * zdim46 * ydim46 * xdim46)
#define OPS_ACC_MD47(d, x, y, z, u, v) ((x) + xdim47 * (y) + ydim47 * xdim47 * (z) + zdim47 * ydim47 * xdim47 * (u) + udim47 * zdim47 * ydim47 * xdim47 * (v) + (d) * vdim47 * udim47 * zdim47 * ydim47 * xdim47)
#define OPS_ACC_MD48(d, x, y, z, u, v) ((x) + xdim48 * (y) + ydim48 * xdim48 * (z) + zdim48 * ydim48 * xdim48 * (u) + udim48 * zdim48 * ydim48 * xdim48 * (v) + (d) * vdim48 * udim48 * zdim48 * ydim48 * xdim48)
#define OPS_ACC_MD49(d, x, y, z, u, v) ((x) + xdim49 * (y) + ydim49 * xdim49 * (z) + zdim49 * ydim49 * xdim49 * (u) + udim49 * zdim49 * ydim49 * xdim49 * (v) + (d) * vdim49 * udim49 * zdim49 * ydim49 * xdim49)
#define OPS_ACC_MD50(d, x, y, z, u, v) ((x) + xdim50 * (y) + ydim50 * xdim50 * (z) + zdim50 * ydim50 * xdim50 * (u) + udim50 * zdim50 * ydim50 * xdim50 * (v) + (d) * vdim50 * udim50 * zdim50 * ydim50 * xdim50)
#define OPS_ACC_MD51(d, x, y, z, u, v) ((x) + xdim51 * (y) + ydim51 * xdim51 * (z) + zdim51 * ydim51 * xdim51 * (u) + udim51 * zdim51 * ydim51 * xdim51 * (v) + (d) * vdim51 * udim51 * zdim51 * ydim51 * xdim51)
#define OPS_ACC_MD52(d, x, y, z, u, v) ((x) + xdim52 * (y) + ydim52 * xdim52 * (z) + zdim52 * ydim52 * xdim52 * (u) + udim52 * zdim52 * ydim52 * xdim52 * (v) + (d) * vdim52 * udim52 * zdim52 * ydim52 * xdim52)
#define OPS_ACC_MD53(d, x, y, z, u, v) ((x) + xdim53 * (y) + ydim53 * xdim53 * (z) + zdim53 * ydim53 * xdim53 * (u) + udim53 * zdim53 * ydim53 * xdim53 * (v) + (d) * vdim53 * udim53 * zdim53 * ydim53 * xdim53)
#define OPS_ACC_MD54(d, x, y, z, u, v) ((x) + xdim54 * (y) + ydim54 * xdim54 * (z) + zdim54 * ydim54 * xdim54 * (u) + udim54 * zdim54 * ydim54 * xdim54 * (v) + (d) * vdim54 * udim54 * zdim54 * ydim54 * xdim54)
#define OPS_ACC_MD55(d, x, y, z, u, v) ((x) + xdim55 * (y) + ydim55 * xdim55 * (z) + zdim55 * ydim55 * xdim55 * (u) + udim55 * zdim55 * ydim55 * xdim55 * (v) + (d) * vdim55 * udim55 * zdim55 * ydim55 * xdim55)
#define OPS_ACC_MD56(d, x, y, z, u, v) ((x) + xdim56 * (y) + ydim56 * xdim56 * (z) + zdim56 * ydim56 * xdim56 * (u) + udim56 * zdim56 * ydim56 * xdim56 * (v) + (d) * vdim56 * udim56 * zdim56 * ydim56 * xdim56)
#define OPS_ACC_MD57(d, x, y, z, u, v) ((x) + xdim57 * (y) + ydim57 * xdim57 * (z) + zdim57 * ydim57 * xdim57 * (u) + udim57 * zdim57 * ydim57 * xdim57 * (v) + (d) * vdim57 * udim57 * zdim57 * ydim57 * xdim57)
#define OPS_ACC_MD58(d, x, y, z, u, v) ((x) + xdim58 * (y) + ydim58 * xdim58 * (z) + zdim58 * ydim58 * xdim58 * (u) + udim58 * zdim58 * ydim58 * xdim58 * (v) + (d) * vdim58 * udim58 * zdim58 * ydim58 * xdim58)
#define OPS_ACC_MD59(d, x, y, z, u, v) ((x) + xdim59 * (y) + ydim59 * xdim59 * (z) + zdim59 * ydim59 * xdim59 * (u) + udim59 * zdim59 * ydim59 * xdim59 * (v) + (d) * vdim59 * udim59 * zdim59 * ydim59 * xdim59)
#define OPS_ACC_MD60(d, x, y, z, u, v) ((x) + xdim60 * (y) + ydim60 * xdim60 * (z) + zdim60 * ydim60 * xdim60 * (u) + udim60 * zdim60 * ydim60 * xdim60 * (v) + (d) * vdim60 * udim60 * zdim60 * ydim60 * xdim60)
#define OPS_ACC_MD61(d, x, y, z, u, v) ((x) + xdim61 * (y) + ydim61 * xdim61 * (z) + zdim61 * ydim61 * xdim61 * (u) + udim61 * zdim61 * ydim61 * xdim61 * (v) + (d) * vdim61 * udim61 * zdim61 * ydim61 * xdim61)
#define OPS_ACC_MD62(d, x, y, z, u, v) ((x) + xdim62 * (y) + ydim62 * xdim62 * (z) + zdim62 * ydim62 * xdim62 * (u) + udim62 * zdim62 * ydim62 * xdim62 * (v) + (d) * vdim62 * udim62 * zdim62 * ydim62 * xdim62)
#define OPS_ACC_MD63(d, x, y, z, u, v) ((x) + xdim63 * (y) + ydim63 * xdim63 * (z) + zdim63 * ydim63 * xdim63 * (u) + udim63 * zdim63 * ydim63 * xdim63 * (v) + (d) * vdim63 * udim63 * zdim63 * ydim63 * xdim63)
#define OPS_ACC_MD64(d, x, y, z, u, v) ((x) + xdim64 * (y) + ydim64 * xdim64 * (z) + zdim64 * ydim64 * xdim64 * (u) + udim64 * zdim64 * ydim64 * xdim64 * (v) + (d) * vdim64 * udim64 * zdim64 * ydim64 * xdim64)
#define OPS_ACC_MD65(d, x, y, z, u, v) ((x) + xdim65 * (y) + ydim65 * xdim65 * (z) + zdim65 * ydim65 * xdim65 * (u) + udim65 * zdim65 * ydim65 * xdim65 * (v) + (d) * vdim65 * udim65 * zdim65 * ydim65 * xdim65)
#define OPS_ACC_MD66(d, x, y, z, u, v) ((x) + xdim66 * (y) + ydim66 * xdim66 * (z) + zdim66 * ydim66 * xdim66 * (u) + udim66 * zdim66 * ydim66 * xdim66 * (v) + (d) * vdim66 * udim66 * zdim66 * ydim66 * xdim66)
#define OPS_ACC_MD67(d, x, y, z, u, v) ((x) + xdim67 * (y) + ydim67 * xdim67 * (z) + zdim67 * ydim67 * xdim67 * (u) + udim67 * zdim67 * ydim67 * xdim67 * (v) + (d) * vdim67 * udim67 * zdim67 * ydim67 * xdim67)
#define OPS_ACC_MD68(d, x, y, z, u, v) ((x) + xdim68 * (y) + ydim68 * xdim68 * (z) + zdim68 * ydim68 * xdim68 * (u) + udim68 * zdim68 * ydim68 * xdim68 * (v) + (d) * vdim68 * udim68 * zdim68 * ydim68 * xdim68)
#define OPS_ACC_MD69(d, x, y, z, u, v) ((x) + xdim69 * (y) + ydim69 * xdim69 * (z) + zdim69 * ydim69 * xdim69 * (u) + udim69 * zdim69 * ydim69 * xdim69 * (v) + (d) * vdim69 * udim69 * zdim69 * ydim69 * xdim69)
#define OPS_ACC_MD70(d, x, y, z, u, v) ((x) + xdim70 * (y) + ydim70 * xdim70 * (z) + zdim70 * ydim70 * xdim70 * (u) + udim70 * zdim70 * ydim70 * xdim70 * (v) + (d) * vdim70 * udim70 * zdim70 * ydim70 * xdim70)
#define OPS_ACC_MD71(d, x, y, z, u, v) ((x) + xdim71 * (y) + ydim71 * xdim71 * (z) + zdim71 * ydim71 * xdim71 * (u) + udim71 * zdim71 * ydim71 * xdim71 * (v) + (d) * vdim71 * udim71 * zdim71 * ydim71 * xdim71)
#define OPS_ACC_MD72(d, x, y, z, u, v) ((x) + xdim72 * (y) + ydim72 * xdim72 * (z) + zdim72 * ydim72 * xdim72 * (u) + udim72 * zdim72 * ydim72 * xdim72 * (v) + (d) * vdim72 * udim72 * zdim72 * ydim72 * xdim72)
#define OPS_ACC_MD73(d, x, y, z, u, v) ((x) + xdim73 * (y) + ydim73 * xdim73 * (z) + zdim73 * ydim73 * xdim73 * (u) + udim73 * zdim73 * ydim73 * xdim73 * (v) + (d) * vdim73 * udim73 * zdim73 * ydim73 * xdim73)
#define OPS_ACC_MD74(d, x, y, z, u, v) ((x) + xdim74 * (y) + ydim74 * xdim74 * (z) + zdim74 * ydim74 * xdim74 * (u) + udim74 * zdim74 * ydim74 * xdim74 * (v) + (d) * vdim74 * udim74 * zdim74 * ydim74 * xdim74)
#define OPS_ACC_MD75(d, x, y, z, u, v) ((x) + xdim75 * (y) + ydim75 * xdim75 * (z) + zdim75 * ydim75 * xdim75 * (u) + udim75 * zdim75 * ydim75 * xdim75 * (v) + (d) * vdim75 * udim75 * zdim75 * ydim75 * xdim75)
#define OPS_ACC_MD76(d, x, y, z, u, v) ((x) + xdim76 * (y) + ydim76 * xdim76 * (z) + zdim76 * ydim76 * xdim76 * (u) + udim76 * zdim76 * ydim76 * xdim76 * (v) + (d) * vdim76 * udim76 * zdim76 * ydim76 * xdim76)
#define OPS_ACC_MD77(d, x, y, z, u, v) ((x) + xdim77 * (y) + ydim77 * xdim77 * (z) + zdim77 * ydim77 * xdim77 * (u) + udim77 * zdim77 * ydim77 * xdim77 * (v) + (d) * vdim77 * udim77 * zdim77 * ydim77 * xdim77)
#define OPS_ACC_MD78(d, x, y, z, u, v) ((x) + xdim78 * (y) + ydim78 * xdim78 * (z) + zdim78 * ydim78 * xdim78 * (u) + udim78 * zdim78 * ydim78 * xdim78 * (v) + (d) * vdim78 * udim78 * zdim78 * ydim78 * xdim78)
#define OPS_ACC_MD79(d, x, y, z, u, v) ((x) + xdim79 * (y) + ydim79 * xdim79 * (z) + zdim79 * ydim79 * xdim79 * (u) + udim79 * zdim79 * ydim79 * xdim79 * (v) + (d) * vdim79 * udim79 * zdim79 * ydim79 * xdim79)
#define OPS_ACC_MD80(d, x, y, z, u, v) ((x) + xdim80 * (y) + ydim80 * xdim80 * (z) + zdim80 * ydim80 * xdim80 * (u) + udim80 * zdim80 * ydim80 * xdim80 * (v) + (d) * vdim80 * udim80 * zdim80 * ydim80 * xdim80)
#define OPS_ACC_MD81(d, x, y, z, u, v) ((x) + xdim81 * (y) + ydim81 * xdim81 * (z) + zdim81 * ydim81 * xdim81 * (u) + udim81 * zdim81 * ydim81 * xdim81 * (v) + (d) * vdim81 * udim81 * zdim81 * ydim81 * xdim81)
#define OPS_ACC_MD82(d, x, y, z, u, v) ((x) + xdim82 * (y) + ydim82 * xdim82 * (z) + zdim82 * ydim82 * xdim82 * (u) + udim82 * zdim82 * ydim82 * xdim82 * (v) + (d) * vdim82 * udim82 * zdim82 * ydim82 * xdim82)
#define OPS_ACC_MD83(d, x, y, z, u, v) ((x) + xdim83 * (y) + ydim83 * xdim83 * (z) + zdim83 * ydim83 * xdim83 * (u) + udim83 * zdim83 * ydim83 * xdim83 * (v) + (d) * vdim83 * udim83 * zdim83 * ydim83 * xdim83)
#define OPS_ACC_MD84(d, x, y, z, u, v) ((x) + xdim84 * (y) + ydim84 * xdim84 * (z) + zdim84 * ydim84 * xdim84 * (u) + udim84 * zdim84 * ydim84 * xdim84 * (v) + (d) * vdim84 * udim84 * zdim84 * ydim84 * xdim84)
#define OPS_ACC_MD85(d, x, y, z, u, v) ((x) + xdim85 * (y) + ydim85 * xdim85 * (z) + zdim85 * ydim85 * xdim85 * (u) + udim85 * zdim85 * ydim85 * xdim85 * (v) + (d) * vdim85 * udim85 * zdim85 * ydim85 * xdim85)
#define OPS_ACC_MD86(d, x, y, z, u, v) ((x) + xdim86 * (y) + ydim86 * xdim86 * (z) + zdim86 * ydim86 * xdim86 * (u) + udim86 * zdim86 * ydim86 * xdim86 * (v) + (d) * vdim86 * udim86 * zdim86 * ydim86 * xdim86)
#define OPS_ACC_MD87(d, x, y, z, u, v) ((x) + xdim87 * (y) + ydim87 * xdim87 * (z) + zdim87 * ydim87 * xdim87 * (u) + udim87 * zdim87 * ydim87 * xdim87 * (v) + (d) * vdim87 * udim87 * zdim87 * ydim87 * xdim87)
#define OPS_ACC_MD88(d, x, y, z, u, v) ((x) + xdim88 * (y) + ydim88 * xdim88 * (z) + zdim88 * ydim88 * xdim88 * (u) + udim88 * zdim88 * ydim88 * xdim88 * (v) + (d) * vdim88 * udim88 * zdim88 * ydim88 * xdim88)
#define OPS_ACC_MD89(d, x, y, z, u, v) ((x) + xdim89 * (y) + ydim89 * xdim89 * (z) + zdim89 * ydim89 * xdim89 * (u) + udim89 * zdim89 * ydim89 * xdim89 * (v) + (d) * vdim89 * udim89 * zdim89 * ydim89 * xdim89)
#define OPS_ACC_MD90(d, x, y, z, u, v) ((x) + xdim90 * (y) + ydim90 * xdim90 * (z) + zdim90 * ydim90 * xdim90 * (u) + udim90 * zdim90 * ydim90 * xdim90 * (v) + (d) * vdim90 * udim90 * zdim90 * ydim90 * xdim90)
#define OPS_ACC_MD91(d, x, y, z, u, v) ((x) + xdim91 * (y) + ydim91 * xdim91 * (z) + zdim91 * ydim91 * xdim91 * (u) + udim91 * zdim91 * ydim91 * xdim91 * (v) + (d) * vdim91 * udim91 * zdim91 * ydim91 * xdim91)
#define OPS_ACC_MD92(d, x, y, z, u, v) ((x) + xdim92 * (y) + ydim92 * xdim92 * (z) + zdim92 * ydim92 * xdim92 * (u) + udim92 * zdim92 * ydim92 * xdim92 * (v) + (d) * vdim92 * udim92 * zdim92 * ydim92 * xdim92)
#define OPS_ACC_MD93(d, x, y, z, u, v) ((x) + xdim93 * (y) + ydim93 * xdim93 * (z) + zdim93 * ydim93 * xdim93 * (u) + udim93 * zdim93 * ydim93 * xdim93 * (v) + (d) * vdim93 * udim93 * zdim93 * ydim93 * xdim93)
#define OPS_ACC_MD94(d, x, y, z, u, v) ((x) + xdim94 * (y) + ydim94 * xdim94 * (z) + zdim94 * ydim94 * xdim94 * (u) + udim94 * zdim94 * ydim94 * xdim94 * (v) + (d) * vdim94 * udim94 * zdim94 * ydim94 * xdim94)
#define OPS_ACC_MD95(d, x, y, z, u, v) ((x) + xdim95 * (y) + ydim95 * xdim95 * (z) + zdim95 * ydim95 * xdim95 * (u) + udim95 * zdim95 * ydim95 * xdim95 * (v) + (d) * vdim95 * udim95 * zdim95 * ydim95 * xdim95)
#define OPS_ACC_MD96(d, x, y, z, u, v) ((x) + xdim96 * (y) + ydim96 * xdim96 * (z) + zdim96 * ydim96 * xdim96 * (u) + udim96 * zdim96 * ydim96 * xdim96 * (v) + (d) * vdim96 * udim96 * zdim96 * ydim96 * xdim96)
#define OPS_ACC_MD97(d, x, y, z, u, v) ((x) + xdim97 * (y) + ydim97 * xdim97 * (z) + zdim97 * ydim97 * xdim97 * (u) + udim97 * zdim97 * ydim97 * xdim97 * (v) + (d) * vdim97 * udim97 * zdim97 * ydim97 * xdim97)
#define OPS_ACC_MD98(d, x, y, z, u, v) ((x) + xdim98 * (y) + ydim98 * xdim98 * (z) + zdim98 * ydim98 * xdim98 * (u) + udim98 * zdim98 * ydim98 * xdim98 * (v) + (d) * vdim98 * udim98 * zdim98 * ydim98 * xdim98)
#define OPS_ACC_MD99(d, x, y, z, u, v) ((x) + xdim99 * (y) + ydim99 * xdim99 * (z) + zdim99 * ydim99 * xdim99 * (u) + udim99 * zdim99 * ydim99 * xdim99 * (v) + (d) * vdim99 * udim99 * zdim99 * ydim99 * xdim99)
#endif
#else
/// TODO #define OPS_ACC0(x,y) (ops_stencil_check_2d_md(0, x, -1, -1))
///--> ops_stencil_check_2d_md(int arg_idx, int idx0, int idx1, int dim0, int
/// dim1, int mult_d, int d);
#endif
#elif defined OPS_4D
#ifndef OPS_DEBUG
#ifndef OPS_SOA
#define OPS_ACC_MD0(d, x, y, z, u)  ((d) + multi_d0  * (x) + xdim0  * multi_d0  * (y) + ydim0  * xdim0  * multi_d0  * (z) + zdim0  * ydim0  * xdim0  * multi_d0  * (u))
#define OPS_ACC_MD1(d, x, y, z, u)  ((d) + multi_d1  * (x) + xdim1  * multi_d1  * (y) + ydim1  * xdim1  * multi_d1  * (z) + zdim1  * ydim1  * xdim1  * multi_d1  * (u))
#define OPS_ACC_MD2(d, x, y, z, u)  ((d) + multi_d2  * (x) + xdim2  * multi_d2  * (y) + ydim2  * xdim2  * multi_d2  * (z) + zdim2  * ydim2  * xdim2  * multi_d2  * (u))
#define OPS_ACC_MD3(d, x, y, z, u)  ((d) + multi_d3  * (x) + xdim3  * multi_d3  * (y) + ydim3  * xdim3  * multi_d3  * (z) + zdim3  * ydim3  * xdim3  * multi_d3  * (u))
#define OPS_ACC_MD4(d, x, y, z, u)  ((d) + multi_d4  * (x) + xdim4  * multi_d4  * (y) + ydim4  * xdim4  * multi_d4  * (z) + zdim4  * ydim4  * xdim4  * multi_d4  * (u))
#define OPS_ACC_MD5(d, x, y, z, u)  ((d) + multi_d5  * (x) + xdim5  * multi_d5  * (y) + ydim5  * xdim5  * multi_d5  * (z) + zdim5  * ydim5  * xdim5  * multi_d5  * (u))
#define OPS_ACC_MD6(d, x, y, z, u)  ((d) + multi_d6  * (x) + xdim6  * multi_d6  * (y) + ydim6  * xdim6  * multi_d6  * (z) + zdim6  * ydim6  * xdim6  * multi_d6  * (u))
#define OPS_ACC_MD7(d, x, y, z, u)  ((d) + multi_d7  * (x) + xdim7  * multi_d7  * (y) + ydim7  * xdim7  * multi_d7  * (z) + zdim7  * ydim7  * xdim7  * multi_d7  * (u))
#define OPS_ACC_MD8(d, x, y, z, u)  ((d) + multi_d8  * (x) + xdim8  * multi_d8  * (y) + ydim8  * xdim8  * multi_d8  * (z) + zdim8  * ydim8  * xdim8  * multi_d8  * (u))
#define OPS_ACC_MD9(d, x, y, z, u)  ((d) + multi_d9  * (x) + xdim9  * multi_d9  * (y) + ydim9  * xdim9  * multi_d9  * (z) + zdim9  * ydim9  * xdim9  * multi_d9  * (u))
#define OPS_ACC_MD10(d, x, y, z, u) ((d) + multi_d10 * (x) + xdim10 * multi_d10 * (y) + ydim10 * xdim10 * multi_d10 * (z) + zdim10 * ydim10 * xdim10 * multi_d10 * (u))
#define OPS_ACC_MD11(d, x, y, z, u) ((d) + multi_d11 * (x) + xdim11 * multi_d11 * (y) + ydim11 * xdim11 * multi_d11 * (z) + zdim11 * ydim11 * xdim11 * multi_d11 * (u))
#define OPS_ACC_MD12(d, x, y, z, u) ((d) + multi_d12 * (x) + xdim12 * multi_d12 * (y) + ydim12 * xdim12 * multi_d12 * (z) + zdim12 * ydim12 * xdim12 * multi_d12 * (u))
#define OPS_ACC_MD13(d, x, y, z, u) ((d) + multi_d13 * (x) + xdim13 * multi_d13 * (y) + ydim13 * xdim13 * multi_d13 * (z) + zdim13 * ydim13 * xdim13 * multi_d13 * (u))
#define OPS_ACC_MD14(d, x, y, z, u) ((d) + multi_d14 * (x) + xdim14 * multi_d14 * (y) + ydim14 * xdim14 * multi_d14 * (z) + zdim14 * ydim14 * xdim14 * multi_d14 * (u))
#define OPS_ACC_MD15(d, x, y, z, u) ((d) + multi_d15 * (x) + xdim15 * multi_d15 * (y) + ydim15 * xdim15 * multi_d15 * (z) + zdim15 * ydim15 * xdim15 * multi_d15 * (u))
#define OPS_ACC_MD16(d, x, y, z, u) ((d) + multi_d16 * (x) + xdim16 * multi_d16 * (y) + ydim16 * xdim16 * multi_d16 * (z) + zdim16 * ydim16 * xdim16 * multi_d16 * (u))
#define OPS_ACC_MD17(d, x, y, z, u) ((d) + multi_d17 * (x) + xdim17 * multi_d17 * (y) + ydim17 * xdim17 * multi_d17 * (z) + zdim17 * ydim17 * xdim17 * multi_d17 * (u))
#define OPS_ACC_MD18(d, x, y, z, u) ((d) + multi_d18 * (x) + xdim18 * multi_d18 * (y) + ydim18 * xdim18 * multi_d18 * (z) + zdim18 * ydim18 * xdim18 * multi_d18 * (u))
#define OPS_ACC_MD19(d, x, y, z, u) ((d) + multi_d19 * (x) + xdim19 * multi_d19 * (y) + ydim19 * xdim19 * multi_d19 * (z) + zdim19 * ydim19 * xdim19 * multi_d19 * (u))
#define OPS_ACC_MD20(d, x, y, z, u) ((d) + multi_d20 * (x) + xdim20 * multi_d20 * (y) + ydim20 * xdim20 * multi_d20 * (z) + zdim20 * ydim20 * xdim20 * multi_d20 * (u))
#define OPS_ACC_MD21(d, x, y, z, u) ((d) + multi_d21 * (x) + xdim21 * multi_d21 * (y) + ydim21 * xdim21 * multi_d21 * (z) + zdim21 * ydim21 * xdim21 * multi_d21 * (u))
#define OPS_ACC_MD22(d, x, y, z, u) ((d) + multi_d22 * (x) + xdim22 * multi_d22 * (y) + ydim22 * xdim22 * multi_d22 * (z) + zdim22 * ydim22 * xdim22 * multi_d22 * (u))
#define OPS_ACC_MD23(d, x, y, z, u) ((d) + multi_d23 * (x) + xdim23 * multi_d23 * (y) + ydim23 * xdim23 * multi_d23 * (z) + zdim23 * ydim23 * xdim23 * multi_d23 * (u))
#define OPS_ACC_MD24(d, x, y, z, u) ((d) + multi_d24 * (x) + xdim24 * multi_d24 * (y) + ydim24 * xdim24 * multi_d24 * (z) + zdim24 * ydim24 * xdim24 * multi_d24 * (u))
#define OPS_ACC_MD25(d, x, y, z, u) ((d) + multi_d25 * (x) + xdim25 * multi_d25 * (y) + ydim25 * xdim25 * multi_d25 * (z) + zdim25 * ydim25 * xdim25 * multi_d25 * (u))
#define OPS_ACC_MD26(d, x, y, z, u) ((d) + multi_d26 * (x) + xdim26 * multi_d26 * (y) + ydim26 * xdim26 * multi_d26 * (z) + zdim26 * ydim26 * xdim26 * multi_d26 * (u))
#define OPS_ACC_MD27(d, x, y, z, u) ((d) + multi_d27 * (x) + xdim27 * multi_d27 * (y) + ydim27 * xdim27 * multi_d27 * (z) + zdim27 * ydim27 * xdim27 * multi_d27 * (u))
#define OPS_ACC_MD28(d, x, y, z, u) ((d) + multi_d28 * (x) + xdim28 * multi_d28 * (y) + ydim28 * xdim28 * multi_d28 * (z) + zdim28 * ydim28 * xdim28 * multi_d28 * (u))
#define OPS_ACC_MD29(d, x, y, z, u) ((d) + multi_d29 * (x) + xdim29 * multi_d29 * (y) + ydim29 * xdim29 * multi_d29 * (z) + zdim29 * ydim29 * xdim29 * multi_d29 * (u))
#define OPS_ACC_MD30(d, x, y, z, u) ((d) + multi_d30 * (x) + xdim30 * multi_d30 * (y) + ydim30 * xdim30 * multi_d30 * (z) + zdim30 * ydim30 * xdim30 * multi_d30 * (u))
#define OPS_ACC_MD31(d, x, y, z, u) ((d) + multi_d31 * (x) + xdim31 * multi_d31 * (y) + ydim31 * xdim31 * multi_d31 * (z) + zdim31 * ydim31 * xdim31 * multi_d31 * (u))
#define OPS_ACC_MD32(d, x, y, z, u) ((d) + multi_d32 * (x) + xdim32 * multi_d32 * (y) + ydim32 * xdim32 * multi_d32 * (z) + zdim32 * ydim32 * xdim32 * multi_d32 * (u))
#define OPS_ACC_MD33(d, x, y, z, u) ((d) + multi_d33 * (x) + xdim33 * multi_d33 * (y) + ydim33 * xdim33 * multi_d33 * (z) + zdim33 * ydim33 * xdim33 * multi_d33 * (u))
#define OPS_ACC_MD34(d, x, y, z, u) ((d) + multi_d34 * (x) + xdim34 * multi_d34 * (y) + ydim34 * xdim34 * multi_d34 * (z) + zdim34 * ydim34 * xdim34 * multi_d34 * (u))
#define OPS_ACC_MD35(d, x, y, z, u) ((d) + multi_d35 * (x) + xdim35 * multi_d35 * (y) + ydim35 * xdim35 * multi_d35 * (z) + zdim35 * ydim35 * xdim35 * multi_d35 * (u))
#define OPS_ACC_MD36(d, x, y, z, u) ((d) + multi_d36 * (x) + xdim36 * multi_d36 * (y) + ydim36 * xdim36 * multi_d36 * (z) + zdim36 * ydim36 * xdim36 * multi_d36 * (u))
#define OPS_ACC_MD37(d, x, y, z, u) ((d) + multi_d37 * (x) + xdim37 * multi_d37 * (y) + ydim37 * xdim37 * multi_d37 * (z) + zdim37 * ydim37 * xdim37 * multi_d37 * (u))
#define OPS_ACC_MD38(d, x, y, z, u) ((d) + multi_d38 * (x) + xdim38 * multi_d38 * (y) + ydim38 * xdim38 * multi_d38 * (z) + zdim38 * ydim38 * xdim38 * multi_d38 * (u))
#define OPS_ACC_MD39(d, x, y, z, u) ((d) + multi_d39 * (x) + xdim39 * multi_d39 * (y) + ydim39 * xdim39 * multi_d39 * (z) + zdim39 * ydim39 * xdim39 * multi_d39 * (u))
#define OPS_ACC_MD40(d, x, y, z, u) ((d) + multi_d40 * (x) + xdim40 * multi_d40 * (y) + ydim40 * xdim40 * multi_d40 * (z) + zdim40 * ydim40 * xdim40 * multi_d40 * (u))
#define OPS_ACC_MD41(d, x, y, z, u) ((d) + multi_d41 * (x) + xdim41 * multi_d41 * (y) + ydim41 * xdim41 * multi_d41 * (z) + zdim41 * ydim41 * xdim41 * multi_d41 * (u))
#define OPS_ACC_MD42(d, x, y, z, u) ((d) + multi_d42 * (x) + xdim42 * multi_d42 * (y) + ydim42 * xdim42 * multi_d42 * (z) + zdim42 * ydim42 * xdim42 * multi_d42 * (u))
#define OPS_ACC_MD43(d, x, y, z, u) ((d) + multi_d43 * (x) + xdim43 * multi_d43 * (y) + ydim43 * xdim43 * multi_d43 * (z) + zdim43 * ydim43 * xdim43 * multi_d43 * (u))
#define OPS_ACC_MD44(d, x, y, z, u) ((d) + multi_d44 * (x) + xdim44 * multi_d44 * (y) + ydim44 * xdim44 * multi_d44 * (z) + zdim44 * ydim44 * xdim44 * multi_d44 * (u))
#define OPS_ACC_MD45(d, x, y, z, u) ((d) + multi_d45 * (x) + xdim45 * multi_d45 * (y) + ydim45 * xdim45 * multi_d45 * (z) + zdim45 * ydim45 * xdim45 * multi_d45 * (u))
#define OPS_ACC_MD46(d, x, y, z, u) ((d) + multi_d46 * (x) + xdim46 * multi_d46 * (y) + ydim46 * xdim46 * multi_d46 * (z) + zdim46 * ydim46 * xdim46 * multi_d46 * (u))
#define OPS_ACC_MD47(d, x, y, z, u) ((d) + multi_d47 * (x) + xdim47 * multi_d47 * (y) + ydim47 * xdim47 * multi_d47 * (z) + zdim47 * ydim47 * xdim47 * multi_d47 * (u))
#define OPS_ACC_MD48(d, x, y, z, u) ((d) + multi_d48 * (x) + xdim48 * multi_d48 * (y) + ydim48 * xdim48 * multi_d48 * (z) + zdim48 * ydim48 * xdim48 * multi_d48 * (u))
#define OPS_ACC_MD49(d, x, y, z, u) ((d) + multi_d49 * (x) + xdim49 * multi_d49 * (y) + ydim49 * xdim49 * multi_d49 * (z) + zdim49 * ydim49 * xdim49 * multi_d49 * (u))
#define OPS_ACC_MD50(d, x, y, z, u) ((d) + multi_d50 * (x) + xdim50 * multi_d50 * (y) + ydim50 * xdim50 * multi_d50 * (z) + zdim50 * ydim50 * xdim50 * multi_d50 * (u))
#define OPS_ACC_MD51(d, x, y, z, u) ((d) + multi_d51 * (x) + xdim51 * multi_d51 * (y) + ydim51 * xdim51 * multi_d51 * (z) + zdim51 * ydim51 * xdim51 * multi_d51 * (u))
#define OPS_ACC_MD52(d, x, y, z, u) ((d) + multi_d52 * (x) + xdim52 * multi_d52 * (y) + ydim52 * xdim52 * multi_d52 * (z) + zdim52 * ydim52 * xdim52 * multi_d52 * (u))
#define OPS_ACC_MD53(d, x, y, z, u) ((d) + multi_d53 * (x) + xdim53 * multi_d53 * (y) + ydim53 * xdim53 * multi_d53 * (z) + zdim53 * ydim53 * xdim53 * multi_d53 * (u))
#define OPS_ACC_MD54(d, x, y, z, u) ((d) + multi_d54 * (x) + xdim54 * multi_d54 * (y) + ydim54 * xdim54 * multi_d54 * (z) + zdim54 * ydim54 * xdim54 * multi_d54 * (u))
#define OPS_ACC_MD55(d, x, y, z, u) ((d) + multi_d55 * (x) + xdim55 * multi_d55 * (y) + ydim55 * xdim55 * multi_d55 * (z) + zdim55 * ydim55 * xdim55 * multi_d55 * (u))
#define OPS_ACC_MD56(d, x, y, z, u) ((d) + multi_d56 * (x) + xdim56 * multi_d56 * (y) + ydim56 * xdim56 * multi_d56 * (z) + zdim56 * ydim56 * xdim56 * multi_d56 * (u))
#define OPS_ACC_MD57(d, x, y, z, u) ((d) + multi_d57 * (x) + xdim57 * multi_d57 * (y) + ydim57 * xdim57 * multi_d57 * (z) + zdim57 * ydim57 * xdim57 * multi_d57 * (u))
#define OPS_ACC_MD58(d, x, y, z, u) ((d) + multi_d58 * (x) + xdim58 * multi_d58 * (y) + ydim58 * xdim58 * multi_d58 * (z) + zdim58 * ydim58 * xdim58 * multi_d58 * (u))
#define OPS_ACC_MD59(d, x, y, z, u) ((d) + multi_d59 * (x) + xdim59 * multi_d59 * (y) + ydim59 * xdim59 * multi_d59 * (z) + zdim59 * ydim59 * xdim59 * multi_d59 * (u))
#define OPS_ACC_MD60(d, x, y, z, u) ((d) + multi_d60 * (x) + xdim60 * multi_d60 * (y) + ydim60 * xdim60 * multi_d60 * (z) + zdim60 * ydim60 * xdim60 * multi_d60 * (u))
#define OPS_ACC_MD61(d, x, y, z, u) ((d) + multi_d61 * (x) + xdim61 * multi_d61 * (y) + ydim61 * xdim61 * multi_d61 * (z) + zdim61 * ydim61 * xdim61 * multi_d61 * (u))
#define OPS_ACC_MD62(d, x, y, z, u) ((d) + multi_d62 * (x) + xdim62 * multi_d62 * (y) + ydim62 * xdim62 * multi_d62 * (z) + zdim62 * ydim62 * xdim62 * multi_d62 * (u))
#define OPS_ACC_MD63(d, x, y, z, u) ((d) + multi_d63 * (x) + xdim63 * multi_d63 * (y) + ydim63 * xdim63 * multi_d63 * (z) + zdim63 * ydim63 * xdim63 * multi_d63 * (u))
#define OPS_ACC_MD64(d, x, y, z, u) ((d) + multi_d64 * (x) + xdim64 * multi_d64 * (y) + ydim64 * xdim64 * multi_d64 * (z) + zdim64 * ydim64 * xdim64 * multi_d64 * (u))
#define OPS_ACC_MD65(d, x, y, z, u) ((d) + multi_d65 * (x) + xdim65 * multi_d65 * (y) + ydim65 * xdim65 * multi_d65 * (z) + zdim65 * ydim65 * xdim65 * multi_d65 * (u))
#define OPS_ACC_MD66(d, x, y, z, u) ((d) + multi_d66 * (x) + xdim66 * multi_d66 * (y) + ydim66 * xdim66 * multi_d66 * (z) + zdim66 * ydim66 * xdim66 * multi_d66 * (u))
#define OPS_ACC_MD67(d, x, y, z, u) ((d) + multi_d67 * (x) + xdim67 * multi_d67 * (y) + ydim67 * xdim67 * multi_d67 * (z) + zdim67 * ydim67 * xdim67 * multi_d67 * (u))
#define OPS_ACC_MD68(d, x, y, z, u) ((d) + multi_d68 * (x) + xdim68 * multi_d68 * (y) + ydim68 * xdim68 * multi_d68 * (z) + zdim68 * ydim68 * xdim68 * multi_d68 * (u))
#define OPS_ACC_MD69(d, x, y, z, u) ((d) + multi_d69 * (x) + xdim69 * multi_d69 * (y) + ydim69 * xdim69 * multi_d69 * (z) + zdim69 * ydim69 * xdim69 * multi_d69 * (u))
#define OPS_ACC_MD70(d, x, y, z, u) ((d) + multi_d70 * (x) + xdim70 * multi_d70 * (y) + ydim70 * xdim70 * multi_d70 * (z) + zdim70 * ydim70 * xdim70 * multi_d70 * (u))
#define OPS_ACC_MD71(d, x, y, z, u) ((d) + multi_d71 * (x) + xdim71 * multi_d71 * (y) + ydim71 * xdim71 * multi_d71 * (z) + zdim71 * ydim71 * xdim71 * multi_d71 * (u))
#define OPS_ACC_MD72(d, x, y, z, u) ((d) + multi_d72 * (x) + xdim72 * multi_d72 * (y) + ydim72 * xdim72 * multi_d72 * (z) + zdim72 * ydim72 * xdim72 * multi_d72 * (u))
#define OPS_ACC_MD73(d, x, y, z, u) ((d) + multi_d73 * (x) + xdim73 * multi_d73 * (y) + ydim73 * xdim73 * multi_d73 * (z) + zdim73 * ydim73 * xdim73 * multi_d73 * (u))
#define OPS_ACC_MD74(d, x, y, z, u) ((d) + multi_d74 * (x) + xdim74 * multi_d74 * (y) + ydim74 * xdim74 * multi_d74 * (z) + zdim74 * ydim74 * xdim74 * multi_d74 * (u))
#define OPS_ACC_MD75(d, x, y, z, u) ((d) + multi_d75 * (x) + xdim75 * multi_d75 * (y) + ydim75 * xdim75 * multi_d75 * (z) + zdim75 * ydim75 * xdim75 * multi_d75 * (u))
#define OPS_ACC_MD76(d, x, y, z, u) ((d) + multi_d76 * (x) + xdim76 * multi_d76 * (y) + ydim76 * xdim76 * multi_d76 * (z) + zdim76 * ydim76 * xdim76 * multi_d76 * (u))
#define OPS_ACC_MD77(d, x, y, z, u) ((d) + multi_d77 * (x) + xdim77 * multi_d77 * (y) + ydim77 * xdim77 * multi_d77 * (z) + zdim77 * ydim77 * xdim77 * multi_d77 * (u))
#define OPS_ACC_MD78(d, x, y, z, u) ((d) + multi_d78 * (x) + xdim78 * multi_d78 * (y) + ydim78 * xdim78 * multi_d78 * (z) + zdim78 * ydim78 * xdim78 * multi_d78 * (u))
#define OPS_ACC_MD79(d, x, y, z, u) ((d) + multi_d79 * (x) + xdim79 * multi_d79 * (y) + ydim79 * xdim79 * multi_d79 * (z) + zdim79 * ydim79 * xdim79 * multi_d79 * (u))
#define OPS_ACC_MD80(d, x, y, z, u) ((d) + multi_d80 * (x) + xdim80 * multi_d80 * (y) + ydim80 * xdim80 * multi_d80 * (z) + zdim80 * ydim80 * xdim80 * multi_d80 * (u))
#define OPS_ACC_MD81(d, x, y, z, u) ((d) + multi_d81 * (x) + xdim81 * multi_d81 * (y) + ydim81 * xdim81 * multi_d81 * (z) + zdim81 * ydim81 * xdim81 * multi_d81 * (u))
#define OPS_ACC_MD82(d, x, y, z, u) ((d) + multi_d82 * (x) + xdim82 * multi_d82 * (y) + ydim82 * xdim82 * multi_d82 * (z) + zdim82 * ydim82 * xdim82 * multi_d82 * (u))
#define OPS_ACC_MD83(d, x, y, z, u) ((d) + multi_d83 * (x) + xdim83 * multi_d83 * (y) + ydim83 * xdim83 * multi_d83 * (z) + zdim83 * ydim83 * xdim83 * multi_d83 * (u))
#define OPS_ACC_MD84(d, x, y, z, u) ((d) + multi_d84 * (x) + xdim84 * multi_d84 * (y) + ydim84 * xdim84 * multi_d84 * (z) + zdim84 * ydim84 * xdim84 * multi_d84 * (u))
#define OPS_ACC_MD85(d, x, y, z, u) ((d) + multi_d85 * (x) + xdim85 * multi_d85 * (y) + ydim85 * xdim85 * multi_d85 * (z) + zdim85 * ydim85 * xdim85 * multi_d85 * (u))
#define OPS_ACC_MD86(d, x, y, z, u) ((d) + multi_d86 * (x) + xdim86 * multi_d86 * (y) + ydim86 * xdim86 * multi_d86 * (z) + zdim86 * ydim86 * xdim86 * multi_d86 * (u))
#define OPS_ACC_MD87(d, x, y, z, u) ((d) + multi_d87 * (x) + xdim87 * multi_d87 * (y) + ydim87 * xdim87 * multi_d87 * (z) + zdim87 * ydim87 * xdim87 * multi_d87 * (u))
#define OPS_ACC_MD88(d, x, y, z, u) ((d) + multi_d88 * (x) + xdim88 * multi_d88 * (y) + ydim88 * xdim88 * multi_d88 * (z) + zdim88 * ydim88 * xdim88 * multi_d88 * (u))
#define OPS_ACC_MD89(d, x, y, z, u) ((d) + multi_d89 * (x) + xdim89 * multi_d89 * (y) + ydim89 * xdim89 * multi_d89 * (z) + zdim89 * ydim89 * xdim89 * multi_d89 * (u))
#define OPS_ACC_MD90(d, x, y, z, u) ((d) + multi_d90 * (x) + xdim90 * multi_d90 * (y) + ydim90 * xdim90 * multi_d90 * (z) + zdim90 * ydim90 * xdim90 * multi_d90 * (u))
#define OPS_ACC_MD91(d, x, y, z, u) ((d) + multi_d91 * (x) + xdim91 * multi_d91 * (y) + ydim91 * xdim91 * multi_d91 * (z) + zdim91 * ydim91 * xdim91 * multi_d91 * (u))
#define OPS_ACC_MD92(d, x, y, z, u) ((d) + multi_d92 * (x) + xdim92 * multi_d92 * (y) + ydim92 * xdim92 * multi_d92 * (z) + zdim92 * ydim92 * xdim92 * multi_d92 * (u))
#define OPS_ACC_MD93(d, x, y, z, u) ((d) + multi_d93 * (x) + xdim93 * multi_d93 * (y) + ydim93 * xdim93 * multi_d93 * (z) + zdim93 * ydim93 * xdim93 * multi_d93 * (u))
#define OPS_ACC_MD94(d, x, y, z, u) ((d) + multi_d94 * (x) + xdim94 * multi_d94 * (y) + ydim94 * xdim94 * multi_d94 * (z) + zdim94 * ydim94 * xdim94 * multi_d94 * (u))
#define OPS_ACC_MD95(d, x, y, z, u) ((d) + multi_d95 * (x) + xdim95 * multi_d95 * (y) + ydim95 * xdim95 * multi_d95 * (z) + zdim95 * ydim95 * xdim95 * multi_d95 * (u))
#define OPS_ACC_MD96(d, x, y, z, u) ((d) + multi_d96 * (x) + xdim96 * multi_d96 * (y) + ydim96 * xdim96 * multi_d96 * (z) + zdim96 * ydim96 * xdim96 * multi_d96 * (u))
#define OPS_ACC_MD97(d, x, y, z, u) ((d) + multi_d97 * (x) + xdim97 * multi_d97 * (y) + ydim97 * xdim97 * multi_d97 * (z) + zdim97 * ydim97 * xdim97 * multi_d97 * (u))
#define OPS_ACC_MD98(d, x, y, z, u) ((d) + multi_d98 * (x) + xdim98 * multi_d98 * (y) + ydim98 * xdim98 * multi_d98 * (z) + zdim98 * ydim98 * xdim98 * multi_d98 * (u))
#define OPS_ACC_MD99(d, x, y, z, u) ((d) + multi_d99 * (x) + xdim99 * multi_d99 * (y) + ydim99 * xdim99 * multi_d99 * (z) + zdim99 * ydim99 * xdim99 * multi_d99 * (u))
#else
#define OPS_ACC_MD0(d, x, y, z, u)  ((x) + xdim0  * (y) + ydim0  * xdim0  * (z) + zdim0  * ydim0  * xdim0  * (u) + (d) * udim0  * zdim0  * ydim0  * xdim0 )
#define OPS_ACC_MD1(d, x, y, z, u)  ((x) + xdim1  * (y) + ydim1  * xdim1  * (z) + zdim1  * ydim1  * xdim1  * (u) + (d) * udim1  * zdim1  * ydim1  * xdim1 )
#define OPS_ACC_MD2(d, x, y, z, u)  ((x) + xdim2  * (y) + ydim2  * xdim2  * (z) + zdim2  * ydim2  * xdim2  * (u) + (d) * udim2  * zdim2  * ydim2  * xdim2 )
#define OPS_ACC_MD3(d, x, y, z, u)  ((x) + xdim3  * (y) + ydim3  * xdim3  * (z) + zdim3  * ydim3  * xdim3  * (u) + (d) * udim3  * zdim3  * ydim3  * xdim3 )
#define OPS_ACC_MD4(d, x, y, z, u)  ((x) + xdim4  * (y) + ydim4  * xdim4  * (z) + zdim4  * ydim4  * xdim4  * (u) + (d) * udim4  * zdim4  * ydim4  * xdim4 )
#define OPS_ACC_MD5(d, x, y, z, u)  ((x) + xdim5  * (y) + ydim5  * xdim5  * (z) + zdim5  * ydim5  * xdim5  * (u) + (d) * udim5  * zdim5  * ydim5  * xdim5 )
#define OPS_ACC_MD6(d, x, y, z, u)  ((x) + xdim6  * (y) + ydim6  * xdim6  * (z) + zdim6  * ydim6  * xdim6  * (u) + (d) * udim6  * zdim6  * ydim6  * xdim6 )
#define OPS_ACC_MD7(d, x, y, z, u)  ((x) + xdim7  * (y) + ydim7  * xdim7  * (z) + zdim7  * ydim7  * xdim7  * (u) + (d) * udim7  * zdim7  * ydim7  * xdim7 )
#define OPS_ACC_MD8(d, x, y, z, u)  ((x) + xdim8  * (y) + ydim8  * xdim8  * (z) + zdim8  * ydim8  * xdim8  * (u) + (d) * udim8  * zdim8  * ydim8  * xdim8 )
#define OPS_ACC_MD9(d, x, y, z, u)  ((x) + xdim9  * (y) + ydim9  * xdim9  * (z) + zdim9  * ydim9  * xdim9  * (u) + (d) * udim9  * zdim9  * ydim9  * xdim9 )
#define OPS_ACC_MD10(d, x, y, z, u) ((x) + xdim10 * (y) + ydim10 * xdim10 * (z) + zdim10 * ydim10 * xdim10 * (u) + (d) * udim10 * zdim10 * ydim10 * xdim10)
#define OPS_ACC_MD11(d, x, y, z, u) ((x) + xdim11 * (y) + ydim11 * xdim11 * (z) + zdim11 * ydim11 * xdim11 * (u) + (d) * udim11 * zdim11 * ydim11 * xdim11)
#define OPS_ACC_MD12(d, x, y, z, u) ((x) + xdim12 * (y) + ydim12 * xdim12 * (z) + zdim12 * ydim12 * xdim12 * (u) + (d) * udim12 * zdim12 * ydim12 * xdim12)
#define OPS_ACC_MD13(d, x, y, z, u) ((x) + xdim13 * (y) + ydim13 * xdim13 * (z) + zdim13 * ydim13 * xdim13 * (u) + (d) * udim13 * zdim13 * ydim13 * xdim13)
#define OPS_ACC_MD14(d, x, y, z, u) ((x) + xdim14 * (y) + ydim14 * xdim14 * (z) + zdim14 * ydim14 * xdim14 * (u) + (d) * udim14 * zdim14 * ydim14 * xdim14)
#define OPS_ACC_MD15(d, x, y, z, u) ((x) + xdim15 * (y) + ydim15 * xdim15 * (z) + zdim15 * ydim15 * xdim15 * (u) + (d) * udim15 * zdim15 * ydim15 * xdim15)
#define OPS_ACC_MD16(d, x, y, z, u) ((x) + xdim16 * (y) + ydim16 * xdim16 * (z) + zdim16 * ydim16 * xdim16 * (u) + (d) * udim16 * zdim16 * ydim16 * xdim16)
#define OPS_ACC_MD17(d, x, y, z, u) ((x) + xdim17 * (y) + ydim17 * xdim17 * (z) + zdim17 * ydim17 * xdim17 * (u) + (d) * udim17 * zdim17 * ydim17 * xdim17)
#define OPS_ACC_MD18(d, x, y, z, u) ((x) + xdim18 * (y) + ydim18 * xdim18 * (z) + zdim18 * ydim18 * xdim18 * (u) + (d) * udim18 * zdim18 * ydim18 * xdim18)
#define OPS_ACC_MD19(d, x, y, z, u) ((x) + xdim19 * (y) + ydim19 * xdim19 * (z) + zdim19 * ydim19 * xdim19 * (u) + (d) * udim19 * zdim19 * ydim19 * xdim19)
#define OPS_ACC_MD20(d, x, y, z, u) ((x) + xdim20 * (y) + ydim20 * xdim20 * (z) + zdim20 * ydim20 * xdim20 * (u) + (d) * udim20 * zdim20 * ydim20 * xdim20)
#define OPS_ACC_MD21(d, x, y, z, u) ((x) + xdim21 * (y) + ydim21 * xdim21 * (z) + zdim21 * ydim21 * xdim21 * (u) + (d) * udim21 * zdim21 * ydim21 * xdim21)
#define OPS_ACC_MD22(d, x, y, z, u) ((x) + xdim22 * (y) + ydim22 * xdim22 * (z) + zdim22 * ydim22 * xdim22 * (u) + (d) * udim22 * zdim22 * ydim22 * xdim22)
#define OPS_ACC_MD23(d, x, y, z, u) ((x) + xdim23 * (y) + ydim23 * xdim23 * (z) + zdim23 * ydim23 * xdim23 * (u) + (d) * udim23 * zdim23 * ydim23 * xdim23)
#define OPS_ACC_MD24(d, x, y, z, u) ((x) + xdim24 * (y) + ydim24 * xdim24 * (z) + zdim24 * ydim24 * xdim24 * (u) + (d) * udim24 * zdim24 * ydim24 * xdim24)
#define OPS_ACC_MD25(d, x, y, z, u) ((x) + xdim25 * (y) + ydim25 * xdim25 * (z) + zdim25 * ydim25 * xdim25 * (u) + (d) * udim25 * zdim25 * ydim25 * xdim25)
#define OPS_ACC_MD26(d, x, y, z, u) ((x) + xdim26 * (y) + ydim26 * xdim26 * (z) + zdim26 * ydim26 * xdim26 * (u) + (d) * udim26 * zdim26 * ydim26 * xdim26)
#define OPS_ACC_MD27(d, x, y, z, u) ((x) + xdim27 * (y) + ydim27 * xdim27 * (z) + zdim27 * ydim27 * xdim27 * (u) + (d) * udim27 * zdim27 * ydim27 * xdim27)
#define OPS_ACC_MD28(d, x, y, z, u) ((x) + xdim28 * (y) + ydim28 * xdim28 * (z) + zdim28 * ydim28 * xdim28 * (u) + (d) * udim28 * zdim28 * ydim28 * xdim28)
#define OPS_ACC_MD29(d, x, y, z, u) ((x) + xdim29 * (y) + ydim29 * xdim29 * (z) + zdim29 * ydim29 * xdim29 * (u) + (d) * udim29 * zdim29 * ydim29 * xdim29)
#define OPS_ACC_MD30(d, x, y, z, u) ((x) + xdim30 * (y) + ydim30 * xdim30 * (z) + zdim30 * ydim30 * xdim30 * (u) + (d) * udim30 * zdim30 * ydim30 * xdim30)
#define OPS_ACC_MD31(d, x, y, z, u) ((x) + xdim31 * (y) + ydim31 * xdim31 * (z) + zdim31 * ydim31 * xdim31 * (u) + (d) * udim31 * zdim31 * ydim31 * xdim31)
#define OPS_ACC_MD32(d, x, y, z, u) ((x) + xdim32 * (y) + ydim32 * xdim32 * (z) + zdim32 * ydim32 * xdim32 * (u) + (d) * udim32 * zdim32 * ydim32 * xdim32)
#define OPS_ACC_MD33(d, x, y, z, u) ((x) + xdim33 * (y) + ydim33 * xdim33 * (z) + zdim33 * ydim33 * xdim33 * (u) + (d) * udim33 * zdim33 * ydim33 * xdim33)
#define OPS_ACC_MD34(d, x, y, z, u) ((x) + xdim34 * (y) + ydim34 * xdim34 * (z) + zdim34 * ydim34 * xdim34 * (u) + (d) * udim34 * zdim34 * ydim34 * xdim34)
#define OPS_ACC_MD35(d, x, y, z, u) ((x) + xdim35 * (y) + ydim35 * xdim35 * (z) + zdim35 * ydim35 * xdim35 * (u) + (d) * udim35 * zdim35 * ydim35 * xdim35)
#define OPS_ACC_MD36(d, x, y, z, u) ((x) + xdim36 * (y) + ydim36 * xdim36 * (z) + zdim36 * ydim36 * xdim36 * (u) + (d) * udim36 * zdim36 * ydim36 * xdim36)
#define OPS_ACC_MD37(d, x, y, z, u) ((x) + xdim37 * (y) + ydim37 * xdim37 * (z) + zdim37 * ydim37 * xdim37 * (u) + (d) * udim37 * zdim37 * ydim37 * xdim37)
#define OPS_ACC_MD38(d, x, y, z, u) ((x) + xdim38 * (y) + ydim38 * xdim38 * (z) + zdim38 * ydim38 * xdim38 * (u) + (d) * udim38 * zdim38 * ydim38 * xdim38)
#define OPS_ACC_MD39(d, x, y, z, u) ((x) + xdim39 * (y) + ydim39 * xdim39 * (z) + zdim39 * ydim39 * xdim39 * (u) + (d) * udim39 * zdim39 * ydim39 * xdim39)
#define OPS_ACC_MD40(d, x, y, z, u) ((x) + xdim40 * (y) + ydim40 * xdim40 * (z) + zdim40 * ydim40 * xdim40 * (u) + (d) * udim40 * zdim40 * ydim40 * xdim40)
#define OPS_ACC_MD41(d, x, y, z, u) ((x) + xdim41 * (y) + ydim41 * xdim41 * (z) + zdim41 * ydim41 * xdim41 * (u) + (d) * udim41 * zdim41 * ydim41 * xdim41)
#define OPS_ACC_MD42(d, x, y, z, u) ((x) + xdim42 * (y) + ydim42 * xdim42 * (z) + zdim42 * ydim42 * xdim42 * (u) + (d) * udim42 * zdim42 * ydim42 * xdim42)
#define OPS_ACC_MD43(d, x, y, z, u) ((x) + xdim43 * (y) + ydim43 * xdim43 * (z) + zdim43 * ydim43 * xdim43 * (u) + (d) * udim43 * zdim43 * ydim43 * xdim43)
#define OPS_ACC_MD44(d, x, y, z, u) ((x) + xdim44 * (y) + ydim44 * xdim44 * (z) + zdim44 * ydim44 * xdim44 * (u) + (d) * udim44 * zdim44 * ydim44 * xdim44)
#define OPS_ACC_MD45(d, x, y, z, u) ((x) + xdim45 * (y) + ydim45 * xdim45 * (z) + zdim45 * ydim45 * xdim45 * (u) + (d) * udim45 * zdim45 * ydim45 * xdim45)
#define OPS_ACC_MD46(d, x, y, z, u) ((x) + xdim46 * (y) + ydim46 * xdim46 * (z) + zdim46 * ydim46 * xdim46 * (u) + (d) * udim46 * zdim46 * ydim46 * xdim46)
#define OPS_ACC_MD47(d, x, y, z, u) ((x) + xdim47 * (y) + ydim47 * xdim47 * (z) + zdim47 * ydim47 * xdim47 * (u) + (d) * udim47 * zdim47 * ydim47 * xdim47)
#define OPS_ACC_MD48(d, x, y, z, u) ((x) + xdim48 * (y) + ydim48 * xdim48 * (z) + zdim48 * ydim48 * xdim48 * (u) + (d) * udim48 * zdim48 * ydim48 * xdim48)
#define OPS_ACC_MD49(d, x, y, z, u) ((x) + xdim49 * (y) + ydim49 * xdim49 * (z) + zdim49 * ydim49 * xdim49 * (u) + (d) * udim49 * zdim49 * ydim49 * xdim49)
#define OPS_ACC_MD50(d, x, y, z, u) ((x) + xdim50 * (y) + ydim50 * xdim50 * (z) + zdim50 * ydim50 * xdim50 * (u) + (d) * udim50 * zdim50 * ydim50 * xdim50)
#define OPS_ACC_MD51(d, x, y, z, u) ((x) + xdim51 * (y) + ydim51 * xdim51 * (z) + zdim51 * ydim51 * xdim51 * (u) + (d) * udim51 * zdim51 * ydim51 * xdim51)
#define OPS_ACC_MD52(d, x, y, z, u) ((x) + xdim52 * (y) + ydim52 * xdim52 * (z) + zdim52 * ydim52 * xdim52 * (u) + (d) * udim52 * zdim52 * ydim52 * xdim52)
#define OPS_ACC_MD53(d, x, y, z, u) ((x) + xdim53 * (y) + ydim53 * xdim53 * (z) + zdim53 * ydim53 * xdim53 * (u) + (d) * udim53 * zdim53 * ydim53 * xdim53)
#define OPS_ACC_MD54(d, x, y, z, u) ((x) + xdim54 * (y) + ydim54 * xdim54 * (z) + zdim54 * ydim54 * xdim54 * (u) + (d) * udim54 * zdim54 * ydim54 * xdim54)
#define OPS_ACC_MD55(d, x, y, z, u) ((x) + xdim55 * (y) + ydim55 * xdim55 * (z) + zdim55 * ydim55 * xdim55 * (u) + (d) * udim55 * zdim55 * ydim55 * xdim55)
#define OPS_ACC_MD56(d, x, y, z, u) ((x) + xdim56 * (y) + ydim56 * xdim56 * (z) + zdim56 * ydim56 * xdim56 * (u) + (d) * udim56 * zdim56 * ydim56 * xdim56)
#define OPS_ACC_MD57(d, x, y, z, u) ((x) + xdim57 * (y) + ydim57 * xdim57 * (z) + zdim57 * ydim57 * xdim57 * (u) + (d) * udim57 * zdim57 * ydim57 * xdim57)
#define OPS_ACC_MD58(d, x, y, z, u) ((x) + xdim58 * (y) + ydim58 * xdim58 * (z) + zdim58 * ydim58 * xdim58 * (u) + (d) * udim58 * zdim58 * ydim58 * xdim58)
#define OPS_ACC_MD59(d, x, y, z, u) ((x) + xdim59 * (y) + ydim59 * xdim59 * (z) + zdim59 * ydim59 * xdim59 * (u) + (d) * udim59 * zdim59 * ydim59 * xdim59)
#define OPS_ACC_MD60(d, x, y, z, u) ((x) + xdim60 * (y) + ydim60 * xdim60 * (z) + zdim60 * ydim60 * xdim60 * (u) + (d) * udim60 * zdim60 * ydim60 * xdim60)
#define OPS_ACC_MD61(d, x, y, z, u) ((x) + xdim61 * (y) + ydim61 * xdim61 * (z) + zdim61 * ydim61 * xdim61 * (u) + (d) * udim61 * zdim61 * ydim61 * xdim61)
#define OPS_ACC_MD62(d, x, y, z, u) ((x) + xdim62 * (y) + ydim62 * xdim62 * (z) + zdim62 * ydim62 * xdim62 * (u) + (d) * udim62 * zdim62 * ydim62 * xdim62)
#define OPS_ACC_MD63(d, x, y, z, u) ((x) + xdim63 * (y) + ydim63 * xdim63 * (z) + zdim63 * ydim63 * xdim63 * (u) + (d) * udim63 * zdim63 * ydim63 * xdim63)
#define OPS_ACC_MD64(d, x, y, z, u) ((x) + xdim64 * (y) + ydim64 * xdim64 * (z) + zdim64 * ydim64 * xdim64 * (u) + (d) * udim64 * zdim64 * ydim64 * xdim64)
#define OPS_ACC_MD65(d, x, y, z, u) ((x) + xdim65 * (y) + ydim65 * xdim65 * (z) + zdim65 * ydim65 * xdim65 * (u) + (d) * udim65 * zdim65 * ydim65 * xdim65)
#define OPS_ACC_MD66(d, x, y, z, u) ((x) + xdim66 * (y) + ydim66 * xdim66 * (z) + zdim66 * ydim66 * xdim66 * (u) + (d) * udim66 * zdim66 * ydim66 * xdim66)
#define OPS_ACC_MD67(d, x, y, z, u) ((x) + xdim67 * (y) + ydim67 * xdim67 * (z) + zdim67 * ydim67 * xdim67 * (u) + (d) * udim67 * zdim67 * ydim67 * xdim67)
#define OPS_ACC_MD68(d, x, y, z, u) ((x) + xdim68 * (y) + ydim68 * xdim68 * (z) + zdim68 * ydim68 * xdim68 * (u) + (d) * udim68 * zdim68 * ydim68 * xdim68)
#define OPS_ACC_MD69(d, x, y, z, u) ((x) + xdim69 * (y) + ydim69 * xdim69 * (z) + zdim69 * ydim69 * xdim69 * (u) + (d) * udim69 * zdim69 * ydim69 * xdim69)
#define OPS_ACC_MD70(d, x, y, z, u) ((x) + xdim70 * (y) + ydim70 * xdim70 * (z) + zdim70 * ydim70 * xdim70 * (u) + (d) * udim70 * zdim70 * ydim70 * xdim70)
#define OPS_ACC_MD71(d, x, y, z, u) ((x) + xdim71 * (y) + ydim71 * xdim71 * (z) + zdim71 * ydim71 * xdim71 * (u) + (d) * udim71 * zdim71 * ydim71 * xdim71)
#define OPS_ACC_MD72(d, x, y, z, u) ((x) + xdim72 * (y) + ydim72 * xdim72 * (z) + zdim72 * ydim72 * xdim72 * (u) + (d) * udim72 * zdim72 * ydim72 * xdim72)
#define OPS_ACC_MD73(d, x, y, z, u) ((x) + xdim73 * (y) + ydim73 * xdim73 * (z) + zdim73 * ydim73 * xdim73 * (u) + (d) * udim73 * zdim73 * ydim73 * xdim73)
#define OPS_ACC_MD74(d, x, y, z, u) ((x) + xdim74 * (y) + ydim74 * xdim74 * (z) + zdim74 * ydim74 * xdim74 * (u) + (d) * udim74 * zdim74 * ydim74 * xdim74)
#define OPS_ACC_MD75(d, x, y, z, u) ((x) + xdim75 * (y) + ydim75 * xdim75 * (z) + zdim75 * ydim75 * xdim75 * (u) + (d) * udim75 * zdim75 * ydim75 * xdim75)
#define OPS_ACC_MD76(d, x, y, z, u) ((x) + xdim76 * (y) + ydim76 * xdim76 * (z) + zdim76 * ydim76 * xdim76 * (u) + (d) * udim76 * zdim76 * ydim76 * xdim76)
#define OPS_ACC_MD77(d, x, y, z, u) ((x) + xdim77 * (y) + ydim77 * xdim77 * (z) + zdim77 * ydim77 * xdim77 * (u) + (d) * udim77 * zdim77 * ydim77 * xdim77)
#define OPS_ACC_MD78(d, x, y, z, u) ((x) + xdim78 * (y) + ydim78 * xdim78 * (z) + zdim78 * ydim78 * xdim78 * (u) + (d) * udim78 * zdim78 * ydim78 * xdim78)
#define OPS_ACC_MD79(d, x, y, z, u) ((x) + xdim79 * (y) + ydim79 * xdim79 * (z) + zdim79 * ydim79 * xdim79 * (u) + (d) * udim79 * zdim79 * ydim79 * xdim79)
#define OPS_ACC_MD80(d, x, y, z, u) ((x) + xdim80 * (y) + ydim80 * xdim80 * (z) + zdim80 * ydim80 * xdim80 * (u) + (d) * udim80 * zdim80 * ydim80 * xdim80)
#define OPS_ACC_MD81(d, x, y, z, u) ((x) + xdim81 * (y) + ydim81 * xdim81 * (z) + zdim81 * ydim81 * xdim81 * (u) + (d) * udim81 * zdim81 * ydim81 * xdim81)
#define OPS_ACC_MD82(d, x, y, z, u) ((x) + xdim82 * (y) + ydim82 * xdim82 * (z) + zdim82 * ydim82 * xdim82 * (u) + (d) * udim82 * zdim82 * ydim82 * xdim82)
#define OPS_ACC_MD83(d, x, y, z, u) ((x) + xdim83 * (y) + ydim83 * xdim83 * (z) + zdim83 * ydim83 * xdim83 * (u) + (d) * udim83 * zdim83 * ydim83 * xdim83)
#define OPS_ACC_MD84(d, x, y, z, u) ((x) + xdim84 * (y) + ydim84 * xdim84 * (z) + zdim84 * ydim84 * xdim84 * (u) + (d) * udim84 * zdim84 * ydim84 * xdim84)
#define OPS_ACC_MD85(d, x, y, z, u) ((x) + xdim85 * (y) + ydim85 * xdim85 * (z) + zdim85 * ydim85 * xdim85 * (u) + (d) * udim85 * zdim85 * ydim85 * xdim85)
#define OPS_ACC_MD86(d, x, y, z, u) ((x) + xdim86 * (y) + ydim86 * xdim86 * (z) + zdim86 * ydim86 * xdim86 * (u) + (d) * udim86 * zdim86 * ydim86 * xdim86)
#define OPS_ACC_MD87(d, x, y, z, u) ((x) + xdim87 * (y) + ydim87 * xdim87 * (z) + zdim87 * ydim87 * xdim87 * (u) + (d) * udim87 * zdim87 * ydim87 * xdim87)
#define OPS_ACC_MD88(d, x, y, z, u) ((x) + xdim88 * (y) + ydim88 * xdim88 * (z) + zdim88 * ydim88 * xdim88 * (u) + (d) * udim88 * zdim88 * ydim88 * xdim88)
#define OPS_ACC_MD89(d, x, y, z, u) ((x) + xdim89 * (y) + ydim89 * xdim89 * (z) + zdim89 * ydim89 * xdim89 * (u) + (d) * udim89 * zdim89 * ydim89 * xdim89)
#define OPS_ACC_MD90(d, x, y, z, u) ((x) + xdim90 * (y) + ydim90 * xdim90 * (z) + zdim90 * ydim90 * xdim90 * (u) + (d) * udim90 * zdim90 * ydim90 * xdim90)
#define OPS_ACC_MD91(d, x, y, z, u) ((x) + xdim91 * (y) + ydim91 * xdim91 * (z) + zdim91 * ydim91 * xdim91 * (u) + (d) * udim91 * zdim91 * ydim91 * xdim91)
#define OPS_ACC_MD92(d, x, y, z, u) ((x) + xdim92 * (y) + ydim92 * xdim92 * (z) + zdim92 * ydim92 * xdim92 * (u) + (d) * udim92 * zdim92 * ydim92 * xdim92)
#define OPS_ACC_MD93(d, x, y, z, u) ((x) + xdim93 * (y) + ydim93 * xdim93 * (z) + zdim93 * ydim93 * xdim93 * (u) + (d) * udim93 * zdim93 * ydim93 * xdim93)
#define OPS_ACC_MD94(d, x, y, z, u) ((x) + xdim94 * (y) + ydim94 * xdim94 * (z) + zdim94 * ydim94 * xdim94 * (u) + (d) * udim94 * zdim94 * ydim94 * xdim94)
#define OPS_ACC_MD95(d, x, y, z, u) ((x) + xdim95 * (y) + ydim95 * xdim95 * (z) + zdim95 * ydim95 * xdim95 * (u) + (d) * udim95 * zdim95 * ydim95 * xdim95)
#define OPS_ACC_MD96(d, x, y, z, u) ((x) + xdim96 * (y) + ydim96 * xdim96 * (z) + zdim96 * ydim96 * xdim96 * (u) + (d) * udim96 * zdim96 * ydim96 * xdim96)
#define OPS_ACC_MD97(d, x, y, z, u) ((x) + xdim97 * (y) + ydim97 * xdim97 * (z) + zdim97 * ydim97 * xdim97 * (u) + (d) * udim97 * zdim97 * ydim97 * xdim97)
#define OPS_ACC_MD98(d, x, y, z, u) ((x) + xdim98 * (y) + ydim98 * xdim98 * (z) + zdim98 * ydim98 * xdim98 * (u) + (d) * udim98 * zdim98 * ydim98 * xdim98)
#define OPS_ACC_MD99(d, x, y, z, u) ((x) + xdim99 * (y) + ydim99 * xdim99 * (z) + zdim99 * ydim99 * xdim99 * (u) + (d) * udim99 * zdim99 * ydim99 * xdim99)
#endif
#else
/// TODO #define OPS_ACC0(x,y) (ops_stencil_check_2d_md(0, x, -1, -1))
///--> ops_stencil_check_2d_md(int arg_idx, int idx0, int idx1, int dim0, int
/// dim1, int mult_d, int d);
#endif
#elif defined OPS_3D     // macros for 3D application
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

#ifndef OPS_NO_GLOBALS

/**--------------------------Set SIMD Vector length--------------------------**/
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

#if defined OPS_3D || defined OPS_4D || defined OPS_5D || defined OPS_SOA
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

#if (defined OPS_3D && defined OPS_SOA) || defined OPS_4D || defined OPS_5D
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

#if (defined OPS_4D && defined OPS_SOA) || defined OPS_5D
extern int udim0, udim1, udim2, udim3, udim4, udim5, udim6, udim7, udim8, udim9,
    udim10, udim11, udim12, udim13, udim14, udim15, udim16, udim17, udim18,
    udim19, udim20, udim21, udim22, udim23, udim24, udim25, udim26, udim27,
    udim28, udim29, udim30, udim31, udim32, udim33, udim34, udim35, udim36,
    udim37, udim38, udim39, udim40, udim41, udim42, udim43, udim44, udim45,
    udim46, udim47, udim48, udim49, udim50, udim51, udim52, udim53, udim54,
    udim55, udim56, udim57, udim58, udim59, udim60, udim61, udim62, udim63,
    udim64, udim65, udim66, udim67, udim68, udim69, udim70, udim71, udim72,
    udim73, udim74, udim75, udim76, udim77, udim78, udim79, udim80, udim81,
    udim82, udim83, udim84, udim85, udim86, udim87, udim88, udim89, udim90,
    udim91, udim92, udim93, udim94, udim95, udim96, udim97, udim98, udim99;

#endif

#if defined OPS_5D && defined OPS_SOA
extern int vdim0, vdim1, vdim2, vdim3, vdim4, vdim5, vdim6, vdim7, vdim8, vdim9,
    vdim10, vdim11, vdim12, vdim13, vdim14, vdim15, vdim16, vdim17, vdim18,
    vdim19, vdim20, vdim21, vdim22, vdim23, vdim24, vdim25, vdim26, vdim27,
    vdim28, vdim29, vdim30, vdim31, vdim32, vdim33, vdim34, vdim35, vdim36,
    vdim37, vdim38, vdim39, vdim40, vdim41, vdim42, vdim43, vdim44, vdim45,
    vdim46, vdim47, vdim48, vdim49, vdim50, vdim51, vdim52, vdim53, vdim54,
    vdim55, vdim56, vdim57, vdim58, vdim59, vdim60, vdim61, vdim62, vdim63,
    vdim64, vdim65, vdim66, vdim67, vdim68, vdim69, vdim70, vdim71, vdim72,
    vdim73, vdim74, vdim75, vdim76, vdim77, vdim78, vdim79, vdim80, vdim81,
    vdim82, vdim83, vdim84, vdim85, vdim86, vdim87, vdim88, vdim89, vdim90,
    vdim91, vdim92, vdim93, vdim94, vdim95, vdim96, vdim97, vdim98, vdim99;

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
#endif /* OPS_SOA */
#endif /* OPS_NO_GLOBALS */
#endif // end OPS_ACC_MD_MACROS
#endif /* OPS_API */
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif // OPS_MACROS_H

