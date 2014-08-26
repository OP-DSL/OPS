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

/** @brief dummy function for CPU backend
  * @author Gihan Mudalige
  * @details Implements dummy functions from the MPI backend for the sequential
  * cpu backend (OpenMP and Sequential)
  */


#include "ops_lib_core.h"

void ops_set_dirtybit_host(ops_arg *args, int nargs)
{
  for (int n=0; n<nargs; n++) {
    if((args[n].argtype == OPS_ARG_DAT) &&
       (args[n].acc == OPS_INC || args[n].acc == OPS_WRITE || args[n].acc == OPS_RW) ) {
      //printf("setting dirty bit on host\n");
      args[n].dat->dirty_hd = 1;
    }
  }
}

int ops_is_root()
{
  return 1;
}

void ops_set_halo_dirtybit(ops_arg *arg)
{
  (void)arg;
}

void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range)
{
  (void)arg;
  (void)iter_range;
}

void ops_exchange_halo(ops_arg* arg, int d)
{
  (void)arg;
}

void ops_exchange_halo2(ops_arg* arg, int* d_pos, int* d_neg /*depth*/)
{
  (void)arg;
  (void)d_pos;
  (void)d_neg;
}

void ops_exchange_halo3(ops_arg* arg, int* d_pos, int* d_neg /*depth*/, int *iter_range)
{
  (void)arg;
  (void)d_pos;
  (void)d_neg;
  (void)iter_range;
}

void ops_halo_exchanges(ops_arg* args, int nargs, int *range) {
  (void)args;
  (void)range;
}

void ops_mpi_reduce_float(ops_arg* args, float* data)
{
  (void)args;
  (void)data;
}

void ops_mpi_reduce_double(ops_arg* args, double* data)
{
  (void)args;
  (void)data;
}

void ops_mpi_reduce_int(ops_arg* args, int* data)
{
  (void)args;
  (void)data;
}

void ops_compute_moment(double t, double *first, double *second) {
  *first = t;
  *second = t*t;
}

void ops_printf(const char* format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void ops_fprintf(FILE *stream, const char *format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stream, format, argptr);
  va_end(argptr);
}

