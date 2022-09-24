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

/** @file
 * @brief  Header file for the parallel I/O functions
 * @author Gihan Mudalige (started 28-08-2015)
 * @details
 */

#ifndef __OPS_HDF5_H
#define __OPS_HDF5_H
#include <string>
#include <tuple>
#include <vector>

/**
 * This routine defines a dataset to be read in from a named hdf5 file.
 *
 *
 * @param block      structured block
 * @param dat_size   dimension of dataset (number of items per grid element)
 * @param type       the name of type used for output diagnostics
 *                   (e.g. "double", "float")
 * @param dat_name   name of the dat used for output diagnostics
 * @param file_name  HDF5 file to read and obtain the data from
 * @return
 */
OPS_FTN_INTEROP
ops_dat ops_decl_dat_hdf5(ops_block block, int dat_size, char const *type,
                          char const *dat_name, char const *file_name);

/**
 * This routine reads the details of a structured grid block from a named
 * HDF5 file.
 *
 * Although this routine does not read in any extra information about the block
 * from the named HDF5 file than what is already specified in the arguments, it
 * is included here for error checking(e.g. check if blocks defined in an HDF5
 * file is matching with the declared arguments in an application) and
 * completeness.
 *
 * @param dims        dimension of the block
 * @param block_name  a name used for output diagnostics
 * @param file_name   HDF5 file to read and obtain the block information from
 * @return
 */
OPS_FTN_INTEROP
ops_block ops_decl_block_hdf5(int dims, const char *block_name,
                              char const *file_name);

/**
 *
 * @param dims          dimension of loop iteration
 * @param points        number of points in the stencil
 * @param stencil_name  string representing the name of the stencil
 * @param file_name     HDF5 file to read from
 * @return
 */
OPS_FTN_INTEROP
ops_stencil ops_decl_stencil_hdf5(int dims, int points,
                                  const char *stencil_name,
                                  char const *file_name);

/**
 *
 * @param dims          dimension of loop iteration
 * @param points        number of points in the stencil
 * @param stencil_name  string representing the name of the stencil
 * @param file_name     HDF5 file to read from
 * @return
 */
OPS_FTN_INTEROP
ops_stencil ops_decl_strided_stencil_hdf5(int dims, int points,
                                          const char *stencil_name,
                                          char const *file_name);

/**
 * This routine reads in a halo relationship between two datasets defined on
 * two different blocks from a named HDF5 file.
 *
 * @param from       origin dataset
 * @param to         destination dataset
 * @param file_name  HDF5 file to read and obtain the data from
 * @return
 */
OPS_FTN_INTEROP
ops_halo ops_decl_halo_hdf5(ops_dat from, ops_dat to, char const *file_name);

/**
 * Write the details of an ::ops_dat to a named HDF5 file.
 *
 * Can be used over MPI (puts the data in an ::ops_dat into an HDF5 file
 * using MPI I/O)
 * @param dat
 * @param file_name
 */
OPS_FTN_INTEROP
void ops_fetch_dat_hdf5_file(ops_dat dat, char const *file_name);

/**
 * Write the details of an ::ops_block to a named HDF5 file.
 *
 * Can be used over MPI (puts the data in an ::ops_block into an HDF5 file
 * using MPI I/O)
 *
 * @param block      ops block to be written
 * @param file_name  HDF5 file to write to
 */
OPS_FTN_INTEROP
void ops_fetch_block_hdf5_file(ops_block block, char const *file_name);

/**
 * Write the details of an ::ops_stencil to a named HDF5 file.
 *
 * Can be used over MPI (puts the data in an ::ops_stencil into an HDF5 file
 * using MPI I/O)
 *
 * @param stencil    ::ops_stencil to be written
 * @param file_name  HDF5 file to write to
 */
OPS_FTN_INTEROP
void ops_fetch_stencil_hdf5_file(ops_stencil stencil, char const *file_name);

/**
 * Write the details of an ::ops_halo to a named HDF5 file.
 *
 * Can be used over MPI (puts the data in an ::ops_halo into an HDF5 file
 * using MPI I/O)
 *
 * @param halo    ::ops_halo to be written
 * @param file_name  HDF5 file to write to
 */
OPS_FTN_INTEROP
void ops_fetch_halo_hdf5_file(ops_halo halo, char const *file_name);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
OPS_FTN_INTEROP
void ops_read_dat_hdf5(ops_dat dat);
#endif /* DOXYGEN_SHOULD_SKIP_THIS*/

/**
 * Write all state (blocks, datasets, stencils) to a named HDF5 file.
 *
 * @param file_name  HDF5 file to write to
 */
OPS_FTN_INTEROP
void ops_dump_to_hdf5(char const *file_name);

void ops_write_const_hdf5(char const *name, int dim, char const *type,
                          char *const_data, char const *file_name);
void ops_get_const_hdf5(char const *name, int dim, char const *type,
                        char *const_data, char const *file_name);

/**
 * Write data on a plane to HDF5 file.
* @param dat       the ops_dat holds the plane
 * @param cross_section_dir the plane direction (0:I, 1:J and 2:K)
 * @param pos the plane postion(e.g., 16 for I=16)
 * @param file_name  HDF5 file name ".h5"
 * @param data_name  data_name
 */

void ops_write_plane_hdf5(const ops_dat dat, const int cross_section_dir,
                          const int pos, char const *file_name,
                          const char *data_name);

/**
 * Write a group of slices/planes of a group of ops_dats into a series of HDF5 files. Each HDF5 will named by I(J,K)Pos.h5, where all ops_dats for this
 plane will be wrote into this file. The slices will be named as blockname_datname_key

 * @param planes  plane groups specified as {{ cross_section_dir,pos},{...},{...},...}
 * @param key  a decoration to a dat name for distinguishing, e.g., time steps.
 * @param data       the ops_dat to be sliced
 * @param  ops_dats to be written, which specified as {{ops_dats at plane 0},{ops_dats at plane 1}...}
 */
void ops_write_plane_group_hdf5(
    const std::vector<std::pair<int, int>> &planes, const std::string &key,
    const std::vector<std::vector<ops_dat>> &data_list);

/**
 * Write a group of slices/planes of a group of ops_dats into a series of HDF5 files. Each HDF5 will be named as specified, where all ops_dats for this
 plane will be wrote into this file. The slices will be named as blockname_datname_key

 * @param planes  plane groups specified as {{ cross_section_dir,pos},{...},{...},...}
 * @param plane_names plane names
 * @param key  a decoration to a dat name for distinguishing, e.g., time steps.
 * @param data       the ops_dat to be sliced
 * @param  ops_dats to be written, which specified as {{ops_dats at plane 0},{ops_dats at plane 1}...}
 */
void ops_write_plane_group_hdf5(
    const std::vector<std::pair<int, int>> &planes,
    std::vector<std::string> &plane_names, const std::string &key,
    const std::vector<std::vector<ops_dat>> &data_list);
#endif
/* __OPS_HDF5_H */
