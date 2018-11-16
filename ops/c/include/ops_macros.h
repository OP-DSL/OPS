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
#endif
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif // OPS_MACROS_H

