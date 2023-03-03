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
 * @brief HDF5 file I/O backend implementation for none-MPI parallelisations
 * @author Jianping Meng (started 03-Mar-2023)
 * @details Implements the OPS API calls for the HDF5 file I/O functionality
 */
#include <string>
#include <vector>

// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

// hdf5 header
#include <hdf5.h>
#include <hdf5_hl.h>
#include <ops_exceptions.h>

hid_t h5_type(const char *type) {
  hid_t h5t{0};
  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0) {
    h5t = H5T_NATIVE_DOUBLE;
  } else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real") == 0) {
    h5t = H5T_NATIVE_FLOAT;
  } else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 ||
             strcmp(type, "integer(4)") == 0) {
    h5t = H5T_NATIVE_INT;
  } else if (strcmp(type, "long") == 0) {
    h5t = H5T_NATIVE_LONG;
  } else if ((strcmp(type, "long long") == 0) || (strcmp(type, "ll") == 0)) {
    h5t = H5T_NATIVE_LLONG;
  } else if (strcmp(type, "short") == 0) {
    h5t = H5T_NATIVE_SHORT;
  } else if (strcmp(type, "char") == 0) {
    h5t = H5T_NATIVE_CHAR;

  } else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: Unknown data type for converting to hdf5 recognised types";
    throw ex;
  }
  return h5t;
}

void split_h5_name(const char *data_name,
                       std::vector<std::string> &h5_name_list) {
  std::stringstream name_stream(data_name);
  std::string segment;
  while (std::getline(name_stream, segment, '/')) {
    h5_name_list.push_back(segment);
  }
}