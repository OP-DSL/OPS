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
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
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
 * @brief OPS internal types and function declarations
 * @author Istvan Reguly
 * @details this header contains type and function declarations needed by ops_lib_core.h
 */

#ifndef __OPS_INTERNAL1
#define __OPS_INTERNAL1

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/queue.h> //contains double linked list implementation
#include <complex.h>

#include "ops_macros.h"
#include "ops_util.h"
#include "ops_exceptions.h"

#define OPS_FTN_INTEROP extern "C"

class OPS_instance;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef __cuda_cuda_h__
#define __host__
#define __device__
#endif

#ifndef __PGI
typedef unsigned int uint;
#endif
typedef long long ll;
typedef unsigned long long ull;



OPS_FTN_INTEROP
ops_arg ops_arg_gbl_char(char *data, int dim, int size, ops_access acc);

OPS_FTN_INTEROP
ops_dat_core* ops_decl_dat_char(ops_block_core *, int, int *, int *, int *, int *, int *, char *,
                          int, char const *, char const *);
void ops_decl_const_char(int, char const *, int, char *, char const *);
OPS_FTN_INTEROP
void ops_reduction_result_char(ops_reduction_core *handle, int type_size, char *ptr);
/*
* run-time type-checking routines
*/
inline int type_error(const double *a, const char *type) {
  (void)a;
  return (strcmp(type, "double") && strcmp(type, "double:soa"));
}
inline int type_error(const float *a, const char *type) {
  (void)a;
  return (strcmp(type, "float") && strcmp(type, "float:soa"));
}
inline int type_error(const int *a, const char *type) {
  (void)a;
  return (strcmp(type, "int") && strcmp(type, "int:soa"));
}
inline int type_error(const uint *a, const char *type) {
  (void)a;
  return (strcmp(type, "uint") && strcmp(type, "uint:soa"));
}
inline int type_error(const ll *a, const char *type) {
  (void)a;
  return (strcmp(type, "ll") && strcmp(type, "ll:soa"));
}
inline int type_error(const ull *a, const char *type) {
  (void)a;
  return (strcmp(type, "ull") && strcmp(type, "ull:soa"));
}
inline int type_error(const bool *a, const char *type) {
  (void)a;
  return (strcmp(type, "bool") && strcmp(type, "bool:soa"));
}
inline int type_error(const std::complex<float> *a, const char *type) {
  (void)a;
  return strcmp(type, "complexf");
}
inline int type_error(const std::complex<double> *a, const char *type) {
  (void)a;
  return strcmp(type, "complexd");
}
#endif


#endif
