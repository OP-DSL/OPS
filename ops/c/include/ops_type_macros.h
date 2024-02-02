#ifndef OPS_TYPE_MACROS_H
#define OPS_TYPE_MACROS_H
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
  * @brief header file declaring macros for functions and zero elements
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS macros
  */



#ifndef MIN
#define MIN(a, b) ((a < b) ? (a) : (b))
#endif
#ifndef MIN3
#define MIN3(a, b, c) MIN(a,MIN(b,c))
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? (a) : (b))
#endif
#ifndef MAX3
#define MAX3(a, b, c) MAX(a,MAX(b,c))
#endif
#ifndef SIGN
#define SIGN(a, b) ((b < 0.0) ? (a * (-1)) : (a))
#endif

#ifndef ZERO_INF
#define ZERO_INF
#define ZERO_double 0.0;
#define INFINITY_double DBL_MAX;

#define ZERO_float 0.0f;
#define INFINITY_float FLT_MAX;

#define ZERO_int 0;
#define INFINITY_int INT_MAX;

#define ZERO_long 0;
#define INFINITY_long LONG_MAX;

#define ZERO_ll 0;
#define INFINITY_ll LLONG_MAX;

#define ZERO_uint 0;
#define INFINITY_uint UINT_MAX;

#define ZERO_ulong 0;
#define INFINITY_ulong ULONG_MAX;

#define ZERO_ull 0;
#define INFINITY_ull ULLONG_MAX;

#define ZERO_bool 0;

#ifndef __cplusplus
#define ZERO_complexd complexd(0,0)
#define INFINITY_complexd complexd(DBL_MAX,DBL_MAX)

#define ZERO_complexf complexf(0,0)
#define INFINITY_complexf complexf(FLT_MAX,FLT_MAX)
#endif

#endif

#endif
#endif
