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

void ops_H_D_exchanges(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_H_D_exchanges_cuda(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit_cuda(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

int ops_is_root()
{
  return 1;
}

void ops_set_halo_dirtybit(ops_arg *arg)
{
  (void)arg;
}

void ops_exchange_halo(ops_arg* arg, int d)
{
  (void)arg;
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
