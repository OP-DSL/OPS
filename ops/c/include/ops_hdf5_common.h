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
 * @brief  Header file for common routines used in HDF5 output
 * @author Jianping (started 03-mar-2023)
 * @details
 */

#ifndef __OPS_HDF5_COMMON_H
#define __OPS_HDF5_COMMON_H
#include "hdf5.h"
#include "hdf5_hl.h"
#include<string>
#include<vector>
#include "ops_exceptions.h"

hid_t h5_type(const char *type);

void split_h5_name(const char *data_name,
                   std::vector<std::string> &h5_name_list);

void H5_dataset_space(const hid_t file_id, const int data_dims,
                      const hsize_t *global_data_size,
                      const std::vector<std::string> &h5_name_list,
                      const char *data_type, std::vector<hid_t> &groupid_list,
                      hid_t &dataset_id, hid_t &file_space);

void H5_dataset_space(const hid_t file_id, const int data_dims,
                      const hsize_t *global_data_size,
                      const std::vector<std::string> &h5_name_list,
                      const char *data_type, const int real_precision,
                      std::vector<hid_t> &groupid_list, hid_t &dataset_id,
                      hid_t &file_space);

#endif
/* __OPS_HDF5_COMMON_H */
