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

/** @brief  Header file for the parallel I/O functions
  * @author Gihan Mudalige (started 28-08-2015)
  * @details
  */

#ifndef __OPS_HDF5_H
#define __OPS_HDF5_H

#ifdef __cplusplus
extern "C" {
#endif
ops_dat ops_decl_dat_hdf5(ops_block block, int dat_size, char const *type,
                          char const *dat_name, char const *file_name);
ops_block ops_decl_block_hdf5(int dims, const char *block_name,
                              char const *file_name);

ops_stencil ops_decl_stencil_hdf5(int dims, int points,
                                  const char *stencil_name,
                                  char const *file_name);

ops_stencil ops_decl_strided_stencil_hdf5(int dims, int points,
                                          const char *stencil_name,
                                          char const *file_name);

ops_halo ops_decl_halo_hdf5(ops_dat from, ops_dat to, char const *file_name);

void ops_fetch_dat_hdf5_file(ops_dat dat, char const *file_name);
void ops_fetch_block_hdf5_file(ops_block block, char const *file_name);
void ops_fetch_stencil_hdf5_file(ops_stencil stencil, char const *file_name);
void ops_fetch_halo_hdf5_file(ops_halo halo, char const *file_name);
void ops_read_dat_hdf5(ops_dat dat);
void ops_dump_to_hdf5(char const *file_name);

#ifdef __cplusplus
}
#endif
#endif
/* __OPS_HDF5_H */
