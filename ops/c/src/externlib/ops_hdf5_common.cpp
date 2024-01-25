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
#include "ops_hdf5_common.h"
#include <ops_exceptions.h>

int half_type_init = 0;
hid_t H5T_IEEE_FP16;

hid_t create_float16_type() {
  hid_t new_type = H5Tcopy(H5T_IEEE_F32LE);
  size_t spos, epos, esize, mpos, msize;
  H5Tget_fields(new_type, &spos, &epos, &esize, &mpos, &msize);
  // for float16
  mpos = 0;
  msize = 10;
  esize = 5;
  epos = 10;
  spos = 15;
  // for single precision
  //   mpos = 0;
  // msize = 23;
  // esize = 8;
  // epos = 23;
  // spos = 31;
  H5Tset_fields(new_type, spos, epos, esize, mpos, msize);
  H5Tset_precision(new_type, 16);
  H5Tset_size(new_type, 2);
  H5Tset_ebias(new_type, 15);
  return new_type;
}

const char *ops_hdf5_type_to_string(hid_t t) {
  if (half_type_init == 0) {H5T_IEEE_FP16 = create_float16_type(); half_type_init=1;}
  char *text = NULL;
  if (H5Tequal(t, H5T_NATIVE_INT)) {
    text = (char *)malloc(4 * sizeof(char));
    strcpy(text, "int");
  } else if (H5Tequal(t, H5T_NATIVE_LONG)) {
    text = (char *)malloc(5 * sizeof(char));
    strcpy(text, "long");
  } else if (H5Tequal(t, H5T_NATIVE_LLONG)) {
    text = (char *)malloc(20 * sizeof(char));
    strcpy(text, "long long");
  } else if (H5Tequal(t, H5T_NATIVE_FLOAT)) {
    text = (char *)malloc(6 * sizeof(char));
    strcpy(text, "float");
  } else if (H5Tequal(t, H5T_NATIVE_DOUBLE)) {
    text = (char *)malloc(7 * sizeof(char));
    strcpy(text, "double");
  } else if (H5Tequal(t, H5T_NATIVE_CHAR)) {
    text = (char *)malloc(5 * sizeof(char));
    strcpy(text, "char");
  } else if (H5Tequal(t, H5T_IEEE_FP16)) {
    text = (char *)malloc(5 * sizeof(char));
    strcpy(text, "half");
  } else {
    text = (char *)malloc(13 * sizeof(char));
    strcpy(text, "UNRECOGNISED");
  }

  return (const char *)text;
}


hid_t h5_type(const char *type) {
    hid_t h5t{0};
    if (half_type_init == 0) { H5T_IEEE_FP16 = create_float16_type(); half_type_init=1; }

    if (strcmp(type, "double") == 0 ||
        strcmp(type, "real(8)") == 0 ||
        strcmp(type, "real(kind=8)") == 0 ||
        strcmp(type, "double precision") == 0)
    {
        h5t = H5T_NATIVE_DOUBLE;
    }
    else if (strcmp(type, "float") == 0 ||
             strcmp(type, "real") == 0 ||
             strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0)
    {
        h5t = H5T_NATIVE_FLOAT;
    }
    else if (strcmp(type, "half") == 0)
    {
        h5t = H5T_IEEE_FP16;
    }
    else if (strcmp(type, "int") == 0 ||
             strcmp(type, "int(4)") == 0 ||
             strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0 ||
             strcmp(type, "integer(kind=4)") == 0)
    {
        h5t = H5T_NATIVE_INT;
    }
    else if (strcmp(type, "long") == 0)
    {
        h5t = H5T_NATIVE_LONG;
    }
    else if ((strcmp(type, "long long") == 0) || (strcmp(type, "ll") == 0))
    {
        h5t = H5T_NATIVE_LLONG;
    }
    else if (strcmp(type, "short") == 0)
    {
        h5t = H5T_NATIVE_SHORT;
    }
    else if (strcmp(type, "char") == 0)
    {
        h5t = H5T_NATIVE_CHAR;
    }
    else
    {
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

  // create the dataset or open the dataset if existing
void H5_dataset_space(const hid_t file_id, const int data_dims,
                      const hsize_t *global_data_size,
                      const std::vector<std::string> &h5_name_list,
                      const char *data_type, REAL_PRECISION real_precision,
                      std::vector<hid_t> &groupid_list, hid_t &dataset_id,
                      hid_t &file_space) {

  hid_t parent_group{file_id};
  const char *data_name = h5_name_list.back().c_str();
  for (int grp = 0; grp < (h5_name_list.size() - 1); grp++) {
    if (H5Lexists(parent_group, h5_name_list[grp].c_str(), H5P_DEFAULT) == 0) {
      parent_group = H5Gcreate(parent_group, h5_name_list[grp].c_str(),
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    } else {
      parent_group =
          H5Gopen(parent_group, h5_name_list[grp].c_str(), H5P_DEFAULT);
    }
    groupid_list[grp] = parent_group;
  }

  if (H5Lexists(parent_group, data_name, H5P_DEFAULT) == 0) {
    hid_t data_plist_id{H5Pcreate(H5P_DATASET_CREATE)};
    file_space = H5Screate_simple(data_dims, global_data_size, NULL);

    hid_t type = h5_type(data_type);
    if (type == H5T_NATIVE_FLOAT || type == H5T_NATIVE_DOUBLE || type == H5T_IEEE_FP16) {
      hid_t real_type;
      if (type == H5T_NATIVE_DOUBLE) {
        if (real_precision == REAL_PRECISION::Double) {
          real_type = H5T_NATIVE_DOUBLE;
        }
        if (real_precision == REAL_PRECISION::Single) {
          real_type = H5T_NATIVE_FLOAT;
        }
        if (real_precision == REAL_PRECISION::Half) {
          real_type = H5T_IEEE_FP16;
        }
      }

      if (type == H5T_NATIVE_FLOAT) {
        if (real_precision == REAL_PRECISION::Single) {
          real_type = H5T_NATIVE_FLOAT;
        }
        if (real_precision == REAL_PRECISION::Half) {
          real_type = H5T_IEEE_FP16;
        }
      }

      if (type == H5T_IEEE_FP16) {
        real_type = H5T_IEEE_FP16;
      }

      dataset_id = H5Dcreate(parent_group, data_name, real_type, file_space,
                             H5P_DEFAULT, data_plist_id, H5P_DEFAULT);
    } else {
      dataset_id =
          H5Dcreate(parent_group, data_name, h5_type(data_type), file_space,
                    H5P_DEFAULT, data_plist_id, H5P_DEFAULT);
    }
    H5Pclose(data_plist_id);
  } else {
    dataset_id = H5Dopen(parent_group, data_name, H5P_DEFAULT);
    file_space = H5Dget_space(dataset_id);
    int ndims{H5Sget_simple_extent_ndims(file_space)};
    bool dims_consistent{ndims == data_dims};
    bool size_consistent{true};
    if (dims_consistent) {
      hsize_t *size{new hsize_t(ndims)};
      H5Sget_simple_extent_dims(file_space, size, NULL);
      for (int d = 0; d < ndims; d++) {
        size_consistent = size_consistent && (size[d] == global_data_size[d]);
      }
      delete size;
    }
    if ((not dims_consistent) || (not size_consistent)) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: inconstent data size in storage detected for  "
         << data_name;
      throw ex;
    }
  }
}

// create the dataset or open the dataset if existing
void H5_dataset_space(const hid_t file_id, const int data_dims,
                      const hsize_t *global_data_size,
                      const std::vector<std::string> &h5_name_list,
                      const char *data_type, std::vector<hid_t> &groupid_list,
                      hid_t &dataset_id, hid_t &file_space) {

  hid_t parent_group{file_id};
  const char *data_name = h5_name_list.back().c_str();
  for (int grp = 0; grp < (h5_name_list.size() - 1); grp++) {
    if (H5Lexists(parent_group, h5_name_list[grp].c_str(), H5P_DEFAULT) == 0) {
      parent_group = H5Gcreate(parent_group, h5_name_list[grp].c_str(),
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    } else {
      parent_group =
          H5Gopen(parent_group, h5_name_list[grp].c_str(), H5P_DEFAULT);
    }
    groupid_list[grp] = parent_group;
  }

  if (H5Lexists(parent_group, data_name, H5P_DEFAULT) == 0) {
    hid_t data_plist_id{H5Pcreate(H5P_DATASET_CREATE)};
    file_space = H5Screate_simple(data_dims, global_data_size, NULL);
    dataset_id = H5Dcreate(parent_group, data_name, h5_type(data_type),
                           file_space, H5P_DEFAULT, data_plist_id, H5P_DEFAULT);
    H5Pclose(data_plist_id);
  } else {
    dataset_id = H5Dopen(parent_group, data_name, H5P_DEFAULT);
    file_space = H5Dget_space(dataset_id);
    int ndims{H5Sget_simple_extent_ndims(file_space)};
    bool dims_consistent{ndims == data_dims};
    bool size_consistent{true};
    if (dims_consistent) {
      hsize_t *size{new hsize_t(ndims)};
      H5Sget_simple_extent_dims(file_space, size, NULL);
      for (int d = 0; d < ndims; d++) {
        size_consistent = size_consistent && (size[d] == global_data_size[d]);
      }
      delete size;
    }
    if ((not dims_consistent) || (not size_consistent)) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: inconstent data size in storage detected for  "
         << data_name;
      throw ex;
    }
  }
}
