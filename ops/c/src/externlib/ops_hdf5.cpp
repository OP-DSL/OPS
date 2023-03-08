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
 * @author Gihan Mudalige (started 28-08-2015)
 * @details Implements the OPS API calls for the HDF5 file I/O functionality
 */
#include <math.h>
#include <string>
#include <vector>
#include <sstream>

// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

// hdf5 header
#include <hdf5.h>
#include <hdf5_hl.h>
#include <ops_lib_core.h>
#include <ops_util.h>
#include <ops_exceptions.h>
#include "ops_hdf5_common.h"

/*******************************************************************************
 * Routine to write an ops_block to a named hdf5 file,
 * if file does not exist, creates it
 * if the block does not exists in file creates block as HDF5 group
 *******************************************************************************/

static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)ops_calloc(len + 16, sizeof(char));
  return strncpy(dest, src, len);
}

void ops_fetch_block_hdf5_file(ops_block block, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t group_id; // group identifier
  hid_t plist_id; // property list identifier

  // Set up file access property list for I/O
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (file_exist(file_name) == 0) {
    if (block->instance->is_root())
      block->instance->ostream()
          << "File " << file_name << " does not exist .... creating file\n";
    FILE *fp;
    fp = fopen(file_name, "w");
    fclose(fp);
    // Create a new file
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  if (H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      if (block->instance->is_root())
        block->instance->ostream()
            << "ops_block " << block->name << " does not exists in file "
            << file_name << " ... creating ops_block\n";
    // create group - ops_block
    group_id =
        H5Gcreate(file_id, block->name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(group_id);
  }

  // open existing group -- an ops_block is a group
  group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);

  // attach attributes to block
  H5LTset_attribute_string(file_id, block->name, "ops_type",
                           "ops_block"); // ops type
  H5LTset_attribute_int(file_id, block->name, "dims", &(block->dims),
                        1); // dim
  H5LTset_attribute_int(file_id, block->name, "index", &(block->index),
                        1); // index

  H5Gclose(group_id);
  H5Pclose(plist_id);
  H5Fclose(file_id);
}

/*******************************************************************************
 * Routine to write an ops_stencil to a named hdf5 file,
 * if file does not exist, creates it
 *******************************************************************************/

void ops_fetch_stencil_hdf5_file(ops_stencil stencil, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t dset_id;  // dataset identifier
  hid_t plist_id; // property list identifier

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (file_exist(file_name) == 0) {
    //    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
    //      if (stencil->instance->is_root()) block->instance->ostream() <<
    //      "File "<<file_name<<" does not exist .... creating file\n";
    FILE *fp;
    fp = fopen(file_name, "w");
    fclose(fp);
    // Create a new file
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  hsize_t rank = 1;
  hsize_t elems = stencil->dims * stencil->points;

  /* create and write the dataset */
  if (H5Lexists(file_id, stencil->name, H5P_DEFAULT) == 0) {
    //    if (block->instance->is_root()) block->instance->ostream() <<
    //    "ops_stencil "<<stencil->name<<" does not exists in the file ...
    //    creating data\n";
    H5LTmake_dataset(file_id, stencil->name, rank, &elems, H5T_NATIVE_INT,
                     stencil->stencil);
  } else {
    dset_id = H5Dopen2(file_id, stencil->name, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             stencil->stencil);
    H5Dclose(dset_id);
  }

  // attach attributes to stencil
  H5LTset_attribute_string(file_id, stencil->name, "ops_type",
                           "ops_stencil"); // ops type
  H5LTset_attribute_int(file_id, stencil->name, "dims", &(stencil->dims),
                        1); // dim
  H5LTset_attribute_int(file_id, stencil->name, "index", &(stencil->index),
                        1); // index
  H5LTset_attribute_int(file_id, stencil->name, "points", &(stencil->points),
                        1); // number of points
  H5LTset_attribute_int(file_id, stencil->name, "stride", stencil->stride,
                        stencil->dims); // strides

  H5Pclose(plist_id);
  H5Fclose(file_id);
}

/*******************************************************************************
 * Routine to write an ops_halo to a named hdf5 file,
 * if file does not exist, creates it
 *******************************************************************************/
void ops_fetch_halo_hdf5_file(ops_halo halo, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t group_id; // group identifier
  hid_t plist_id; // property list identifier

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (file_exist(file_name) == 0) {
    if (halo->from->block->instance->OPS_diags > 2)
      if (halo->from->block->instance->is_root())
        halo->from->block->instance->ostream()
            << "File " << file_name << " does not exist .... creating file\n";
    FILE *fp;
    fp = fopen(file_name, "w");
    fclose(fp);

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  char halo_name[100]; // strlen(halo->from->name)+strlen(halo->to->name)];
  sprintf(halo_name, "from_%s_to_%s", halo->from->name, halo->to->name);

  /* create and write the a group that holds the halo information */
  if (H5Lexists(file_id, halo_name, H5P_DEFAULT) == 0) {
    if (halo->from->block->instance->OPS_diags > 2)
      if (halo->from->block->instance->is_root())
        halo->from->block->instance->ostream()
            << "ops_halo " << halo_name
            << " does not exists in the file ... creating group to hold halo\n";
    group_id =
        H5Gcreate(file_id, halo_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(group_id);
  }
  group_id = H5Gopen2(file_id, halo_name, H5P_DEFAULT);

  // take the maximum dimension of the two blocks connected via this halo
  int dim = MAX(halo->from->block->dims, halo->to->block->dims);

  // printf("halo name %s, from name %s, to name %s\n",halo_name,
  // halo->from->name, halo->to->name);
  // attach attributes to halo
  H5LTset_attribute_string(file_id, halo_name, "ops_type",
                           "ops_halo"); // ops type
  H5LTset_attribute_string(file_id, halo_name, "from_dat_name",
                           halo->from->name); // from ops_dat (name)
  H5LTset_attribute_string(file_id, halo_name, "to_dat_name",
                           halo->to->name); // to ops_dat (name)
  H5LTset_attribute_int(file_id, halo_name, "from_dat_index",
                        &(halo->from->index), 1); // from_dat_index
  H5LTset_attribute_int(file_id, halo_name, "to_dat_index", &(halo->to->index),
                        1); // from_dat_index

  H5LTset_attribute_int(file_id, halo_name, "iter_size", halo->iter_size,
                        dim); // iteration size
  H5LTset_attribute_int(file_id, halo_name, "from_base", halo->from_base,
                        dim); // base from ops_dat
  H5LTset_attribute_int(file_id, halo_name, "to_base", halo->to_base,
                        dim); // base of to ops_dat
  H5LTset_attribute_int(file_id, halo_name, "from_dir", halo->from_dir,
                        dim); // copy from direction
  H5LTset_attribute_int(file_id, halo_name, "to_dir", halo->to_dir,
                        dim); // copy to direction
  H5LTset_attribute_int(file_id, halo_name, "index", &(halo->index),
                        1); // index

  H5Gclose(group_id);
  H5Pclose(plist_id);
  H5Fclose(file_id);
}

/*******************************************************************************
 * Routine to remove x-dimension padding introduced when creating dats to be
 * x-dimension memory aligned
 * this needs to be removed when writing to HDF5 files  - dimension of block is
 *1
 ********************************************************************************/
void remove_padding1D(ops_dat dat, hsize_t *size, char *data) {
  hsize_t index = 0;
  hsize_t count = 0;
  for (hsize_t i = 0; i < size[0]; i++) {
    index = i;
    memcpy(&data[count * dat->elem_size], &dat->data[index * dat->elem_size],
           dat->elem_size);
    count++;
  }
  return;
}

/*******************************************************************************
 * Routine to remove x-dimension padding introduced when creating dats to be
 * x-dimension memory aligned
 * this needs to be removed when writing to HDF5 files  - dimension of block is
 *2
 ********************************************************************************/
void remove_padding2D(ops_dat dat, hsize_t *size, char *data) {
  hsize_t index = 0;
  hsize_t count = 0;
  for (hsize_t j = 0; j < size[1]; j++) {
    for (hsize_t i = 0; i < size[0]; i++) {
      index = i + j * dat->size[0];
      memcpy(&data[count * dat->elem_size], &dat->data[index * dat->elem_size],
             dat->elem_size);
      count++;
    }
  }
  return;
}

/*******************************************************************************
 * Routine to remove x-dimension padding introduced when creating dats to be
 * x-dimension memory aligned
 * this needs to be removed when writing to HDF5 files  - dimension of block is
 *3
 ********************************************************************************/
void remove_padding3D(ops_dat dat, hsize_t *size, char *data) {
  hsize_t index = 0;
  hsize_t count = 0;

  for (hsize_t k = 0; k < size[2]; k++) {
    for (hsize_t j = 0; j < size[1]; j++) {
      for (hsize_t i = 0; i < size[0]; i++) {
        index = i + j * dat->size[0] + // need to stride in dat->size as data
                                       // block includes intra-block halos
                k * dat->size[0] * dat->size[1]; // +

        memcpy(&data[count * dat->elem_size],
               &dat->data[index * dat->elem_size], dat->elem_size);
        count++;
      }
    }
  }
  return;
}

/*******************************************************************************
 * Routine to write an ops_dat to a named hdf5 file,
 * if file does not exist, creates it
 * if the data set does not exists in file creates data set
 *******************************************************************************/

void ops_fetch_dat_hdf5_file(ops_dat dat, char const *file_name) {

  ops_block block = dat->block;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t group_id; // group identifier
  hid_t plist_id; // property list identifier

  hsize_t g_size[OPS_MAX_DIM] = {0};
  int gbl_size[OPS_MAX_DIM] = {0};
  for (int d = 0; d < block->dims; d++) {
    // pure data size (i.e. without block halos) to be noted as an attribute
    gbl_size[d] = dat->size[d] + dat->d_m[d] - dat->d_p[d];
    // the number of elements thats actually written
    g_size[d] = dat->size[d];
  }

  /* Need to strip out the padding from the x-dimension*/
  g_size[0] = g_size[0] - dat->x_pad;
  hsize_t t_size = 1;
  for (int d = 0; d < block->dims; d++)
    t_size *= g_size[d];
  char *data = (char *)ops_malloc(t_size * dat->elem_size);

  int *range{new int(2 * block->dims)};
  for (int d = 0; d < block->dims; d++) {
    range[2 * d] = dat->d_m[d];
    range[2 * d + 1] = dat->size[d]-dat->d_p[d];
    // ops_printf("range[%d]=%d range[%d]=%d size=%d d_p=%d d_m=%d\n", 2 * d,
    //            range[2 * d], 2 * d + 1, range[2 * d + 1], dat->size[d],
    //            dat->d_p[d], dat->d_m[d]);
  }
  ops_dat_fetch_data_slab_host(dat, 0, data, range);
  delete range;
  // make sure we multiply by the number of data values per element (i.e.
  // dat->dim)
  // g_size[1] = g_size[1]*dat->dim;
  if (block->dims == 1)
    g_size[0] = g_size[0] * dat->dim; // -- this needs to be tested for 1D
  else if (block->dims == 2)
    // Jianping Meng: it looks that growing the zero index is better
    g_size[0] =
        g_size[0] * dat->dim; //**note we are using [1] instead of [0] here !!
  else if (block->dims == 3) {
    g_size[0] =
        g_size[0] * dat->dim; //**note that for 3D we are using [0] here !!
  }

  hsize_t G_SIZE[OPS_MAX_DIM];
  if (block->dims == 1) {
    G_SIZE[0] = g_size[0];
  } else if (block->dims == 2) {
    G_SIZE[0] = g_size[1];
    G_SIZE[1] = g_size[0];
  } else if (block->dims == 3) {
    G_SIZE[0] = g_size[2];
    G_SIZE[1] = g_size[1];
    G_SIZE[2] = g_size[0];
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (file_exist(file_name) == 0) {
    if (dat->block->instance->OPS_diags > 3)
      if (dat->block->instance->is_root())
        dat->block->instance->ostream()
            << "File " << file_name << "does not exist .... creating file\n";
    FILE *fp;
    fp = fopen(file_name, "w");
    fclose(fp);

    // Create a new file
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  if (H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_fetch_dat_hdf5_file: ops_block on which this ops_dat "
       << dat->name << " is declared does not exist in the file";
    throw ex;

  } else {
    // open existing group -- an ops_block is a group
    group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);

    if (H5Lexists(group_id, dat->name, H5P_DEFAULT) ==
        0) { // dat does not exisits .. create
      if (dat->block->instance->OPS_diags > 2)
        if (dat->block->instance->is_root())
          dat->block->instance->ostream()
              << "ops_fetch_dat_hdf5_file: ops_dat " << dat->name
              << " does not exists in the ops_block " << block->name
              << " ... creating ops_dat\n";

      H5LTmake_dataset(group_id, dat->name, block->dims, G_SIZE,
                       h5_type(dat->type), data);

      // attach attributes to dat
      H5LTset_attribute_string(group_id, dat->name, "ops_type",
                               "ops_dat"); // ops type
      H5LTset_attribute_string(group_id, dat->name, "block",
                               block->name); // block
      H5LTset_attribute_int(group_id, dat->name, "block_index", &(block->index),
                            1); // block index
      H5LTset_attribute_int(group_id, dat->name, "dim", &(dat->dim), 1); // dim
      H5LTset_attribute_int(group_id, dat->name, "size", gbl_size,
                            block->dims); // size
      H5LTset_attribute_int(group_id, dat->name, "d_m", dat->d_m,
                            block->dims); // d_m

      // need to substract x_pad from d_p before writing attribute to file
      int orig_d_p[OPS_MAX_DIM];
      for (int d = 0; d < block->dims; d++)
        orig_d_p[d] = dat->d_p[d];
      orig_d_p[0] = dat->d_p[0] - dat->x_pad;

      H5LTset_attribute_int(group_id, dat->name, "d_p", orig_d_p,
                            block->dims); // d_p
      H5LTset_attribute_int(group_id, dat->name, "base", dat->base,
                            block->dims);                               // base
      H5LTset_attribute_string(group_id, dat->name, "type", dat->type); // type
    } else { // dat exisits .. check attributes and if matching .. ovewrite

      dat->block->instance->ostream()
          << "Dataset '" << dat->name << "' already found in file " << file_name
          << " ... ";

      char ops_type[10], type[10];
      char blk[40];
      int bindex, dim, gsize[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM],
          base[OPS_MAX_DIM];

      if (H5LTget_attribute_string(group_id, dat->name, "ops_type", ops_type) <
          0) { // ops type
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"ops_type\" not found in dat" << dat->name;
        throw ex;
      } else {
        if (strcmp("ops_dat", ops_type) != 0) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_type: " << ops_type << " of dat: " << dat->name
             << " is not ops_dat";
          throw ex;
        }
      }
      if (H5LTget_attribute_string(group_id, dat->name, "block", blk) <
          0) { // block
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"block\" not found in dat" << dat->name;
        throw ex;
      } else {
        if (strcmp(block->name, blk) != 0) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_block name: " << block->name
             << "of dat: " << dat->name << " does not match block name :" << blk
             << " in file";
          throw ex;
        }
      }
      if (H5LTget_attribute_int(group_id, dat->name, "block_index", &bindex) <
          0) { // block index
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"block_index\" not found in dat" << dat->name;
        throw ex;
      } else {
        if (block->index != bindex) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_block index: " << block->index
             << "of dat: " << dat->name
             << " does not match block index :" << bindex << " in file";
          throw ex;
        }
      }
      if (H5LTget_attribute_int(group_id, dat->name, "dim", &dim) < 0) { // dim
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"dim\" not found in dat" << dat->name;
        throw ex;
      } else {
        if (dat->dim != dim) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_dat dim: " << dat->dim << "of dat: " << dat->name
             << " does not match dat dim :" << dim << "in file";
          throw ex;
        }
      }

      if (H5LTget_attribute_int(group_id, dat->name, "size", gsize) <
          0) { // size
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"size\" not found in dat" << dat->name;
        throw ex;
      } else {
        for (int i = 0; i < dat->dim; i++) {
          if (gbl_size[i] !=
              gsize[i]) { // remember checking global size as computed above
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_dat size: " << dat->size[i]
               << " of dimension: " << i << " of dat: " << dat->name
               << " does not match dat size: " << gsize[i] << " in file ";
            throw ex;
          }
        }
      }

      if (H5LTget_attribute_int(group_id, dat->name, "d_m", d_m) < 0) { // d_m
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"d_m\" not found in dat" << dat->name;
        throw ex;
      } else {
        for (int i = 0; i < dat->dim; i++) {
          if (dat->d_m[i] != d_m[i]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_dat d_m: " << dat->d_m[i] << "of dimension: " << i
               << " of dat: " << dat->name
               << " does not match dat d_m: " << d_m[i] << "in file ";
            throw ex;
          }
        }
      }
      if (H5LTget_attribute_int(group_id, dat->name, "d_p", d_p) < 0) { // d_p
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"d_p\" not found in dat" << dat->name;
        throw ex;
      } else {
        // need to substract x_pad from d_p before checking attribute on file
        int orig_d_p[OPS_MAX_DIM];
        for (int d = 0; d < block->dims; d++)
          orig_d_p[d] = dat->d_p[d];
        orig_d_p[0] = dat->d_p[0] - dat->x_pad;

        for (int i = 0; i < dat->dim; i++) {
          if (orig_d_p[i] != d_p[i]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_dat d_p: " << orig_d_p[i]
               << " of dimension: " << i << " of dat: " << dat->name
               << " does not match dat d_p: " << d_p[i] << " in file ";
            throw ex;
          }
        }
      }
      if (H5LTget_attribute_int(group_id, dat->name, "base", base) <
          0) { // base
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"base\" not found in dat" << dat->name;
        throw ex;
      } else {
        for (int i = 0; i < dat->dim; i++) {
          if (dat->base[i] != base[i]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_dat base: " << dat->base[i]
               << "of dimension: " << i << "of dat: " << dat->name
               << " does not match dat base: " << base[i] << "in file ";
            throw ex;
          }
        }
      }
      if (H5LTget_attribute_string(group_id, dat->name, "type", type) <
          0) { // type
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Attribute \"type\" not found in dat" << dat->name;
        throw ex;
      } else {
        if (strcmp(dat->type, type) != 0) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_dat type: " << dat->type << " of dat: " << dat->name
             << " does not match dat type :" << type << "in file";
          throw ex;
        }
      }

      // all good , overwrite the existing dataset
      dat->block->instance->ostream() << " overwriting"
                                      << "\n";

      hid_t dset_id = H5Dopen(group_id, dat->name, H5P_DEFAULT);
      hid_t dataspace = H5Dget_space(dset_id);

      if (strcmp(dat->type, "double") == 0 || strcmp(dat->type, "real(8)") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else if (strcmp(dat->type, "float") == 0 ||
               strcmp(dat->type, "real(4)") == 0 ||
               strcmp(dat->type, "real") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else if (strcmp(dat->type, "int") == 0 ||
               strcmp(dat->type, "int(4)") == 0 ||
               strcmp(dat->type, "integer(4)") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else if (strcmp(dat->type, "long") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else if ((strcmp(dat->type, "long long") == 0) ||
               (strcmp(dat->type, "ll") == 0))
        H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else if (strcmp(dat->type, "short") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_SHORT, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else if (strcmp(dat->type, "char") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_CHAR, H5S_ALL, dataspace, H5P_DEFAULT,
                 data);
      else {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Unknown type in ops_fetch_dat_hdf5_file(): " << dat->type;
        throw ex;
      }
    }

    H5Gclose(group_id);
    H5Fclose(file_id);
  }
}

/*******************************************************************************
 * Routine to read an ops_block from an hdf5 file
 *******************************************************************************/
ops_block ops_decl_block_hdf5(int dims, const char *block_name,
                              char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_block_hdf5: file " << file_name << " does not exist";
    throw ex;
  }

  // Set up file access property list for I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  // check if ops_block exists
  if (H5Lexists(file_id, block_name, H5P_DEFAULT) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_block_hdf5: ops_block " << block_name
       << " does not exist in the file";
    throw ex;
  }

  // ops_block exists .. now check ops_type and dims
  char read_ops_type[20];
  if (H5LTget_attribute_string(file_id, block_name, "ops_type", read_ops_type) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_block_hdf5: Attribute \"ops_type\" not found in "
          "block "
       << block_name;
    throw ex;
  } else {
    if (strcmp("ops_block", read_ops_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_block_hdf5: ops_type of block " << block_name
         << " is not ops_block";
      throw ex;
    }
  }
  int read_dims;
  if (H5LTget_attribute_int(file_id, block_name, "dims", &read_dims) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_block_hdf5: Attribute \"dims\" not found in block "
       << block_name;
    throw ex;
  } else {
    if (dims != read_dims) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_block_hdf5: Unequal dims of block " << block_name
         << ": dims on file " << read_dims << "dims specified " << dims;
      throw ex;
    }
  }
  int read_index;
  if (H5LTget_attribute_int(file_id, block_name, "index", &read_index) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_block_hdf5: Attribute \"index\" not found in block "
       << block_name;
    throw ex;
  }

  // checks passed ..

  H5Pclose(plist_id);
  H5Fclose(file_id);

  return ops_decl_block(read_dims, block_name);
}

/*******************************************************************************
 * Routine to read an ops_stencil from an hdf5 file
 *******************************************************************************/
ops_stencil ops_decl_stencil_hdf5(int dims, int points,
                                  const char *stencil_name,
                                  char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: file " << file_name
       << " does not exist";
    throw ex;
  }

  // Set up file access property list for I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  // check if ops_stencil exists
  if (H5Lexists(file_id, stencil_name, H5P_DEFAULT) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: ops_stencil " << stencil_name
       << " not found in file";
    throw ex;
  }

  // ops_stencil exists .. now check ops_type and dims
  char read_ops_type[20];
  if (H5LTget_attribute_string(file_id, stencil_name, "ops_type",
                               read_ops_type) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"ops_type\" not found in "
          "stencil"
       << stencil_name;
    throw ex;

  } else {
    if (strcmp("ops_stencil", read_ops_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_stencil_hdf5: ops_type of stencil" << stencil_name
         << " is not ops_stencil";
      throw ex;
    }
  }
  int read_dims;
  if (H5LTget_attribute_int(file_id, stencil_name, "dims", &read_dims) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"dims\" not found in "
          "stencil"
       << stencil_name;
    throw ex;

  } else {
    if (dims != read_dims) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_stencil_hdf5: Unequal dims of stencil "
         << stencil_name << " dims on file " << read_dims << ", dims specified "
         << dims;
      throw ex;
    }
  }
  int read_points;
  if (H5LTget_attribute_int(file_id, stencil_name, "points", &read_points) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"points\" not found in "
          "stencil"
       << stencil_name;
    throw ex;

  } else {
    if (points != read_points) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_stencil_hdf5: Unequal points of stencil "
         << stencil_name << " points on file " << read_points
         << ", points specified " << points;
      throw ex;
    }
  }
  // checks passed ..

  // get the strides
  int read_stride[OPS_MAX_DIM];
  if (H5LTget_attribute_int(file_id, stencil_name, "stride", read_stride) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"stride\" not found in "
          "stencil"
       << stencil_name;
    throw ex;
  }

  std::vector<int> read_sten(read_dims * read_points);
  H5LTread_dataset_int(file_id, stencil_name, read_sten.data());
  H5Pclose(plist_id);
  H5Fclose(file_id);
  // use decl_strided stencil for both normal and strided stencils
  return ops_decl_strided_stencil(read_dims, read_points, read_sten.data(),
                                  read_stride, stencil_name);
}

/*******************************************************************************
 * Routine to read an ops_halo from an hdf5 file
 *******************************************************************************/
ops_halo ops_decl_halo_hdf5(ops_dat from, ops_dat to, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_halo_hdf5: file " << file_name << " does not exist";
    throw ex;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  // check if ops_halo exists
  char halo_name[100]; // strlen(halo->from->name)+strlen(halo->to->name)];
  sprintf(halo_name, "from_%s_to_%s", from->name, to->name);
  if (H5Lexists(file_id, halo_name, H5P_DEFAULT) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: ops_halo " << halo_name
       << " does not exist in the file";
    throw ex;
  }

  // ops_stencil exists .. now check ops_type
  char read_ops_type[20];
  if (H5LTget_attribute_string(file_id, halo_name, "ops_type", read_ops_type) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"ops_type\" not found in "
          "halo"
       << halo_name;
    throw ex;
  } else {
    if (strcmp("ops_halo", read_ops_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_stencil_hdf5: ops_type of halo " << halo_name
         << " not equal to ops_halo";
      throw ex;
    }
  }

  // check whether dimensions are equal
  if (from->block->dims != to->block->dims) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: dimensions of ops_dats connected by "
          "halo"
       << halo_name << " are not equal to each other";
    throw ex;
  }

  // checks passed ..

  // get the iter_size
  int read_iter_size[OPS_MAX_DIM];
  if (H5LTget_attribute_int(file_id, halo_name, "iter_size", read_iter_size) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"iter_size\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the from_base
  int read_from_base[OPS_MAX_DIM];
  if (H5LTget_attribute_int(file_id, halo_name, "from_base", read_from_base) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"from_base\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the to_base
  int read_to_base[OPS_MAX_DIM];
  if (H5LTget_attribute_int(file_id, halo_name, "to_base", read_to_base) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"to_base\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the from_dir
  int read_from_dir[OPS_MAX_DIM];
  if (H5LTget_attribute_int(file_id, halo_name, "from_dir", read_from_dir) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"from_dir\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the to_dir
  int read_to_dir[OPS_MAX_DIM];
  if (H5LTget_attribute_int(file_id, halo_name, "to_dir", read_to_dir) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"to_dir\" not found in halo"
       << halo_name;
    throw ex;
  }

  H5Pclose(plist_id);
  H5Fclose(file_id);

  return ops_decl_halo(from, to, read_iter_size, read_from_base, read_to_base,
                       read_from_dir, read_to_dir);
}

/*******************************************************************************
 * Routine to read an ops_dat from an hdf5 file
 *******************************************************************************/
ops_dat ops_decl_dat_hdf5(ops_block block, int dat_dim, char const *type,
                          char const *dat_name, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t group_id; // group identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: file " << file_name << " does not exist";
    throw ex;
  }

  // Set up file access property list for I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  if (H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: Error: ops_decl_dat_hdf5: ops_block on which this ops_dat "
       << dat_name << " is declared does not exist in the file";
    throw ex;
  }

  // open existing group -- an ops_block is a group
  group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);

  // check if ops_dat exists
  if (H5Lexists(group_id, dat_name, H5P_DEFAULT) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: Error: ops_decl_dat_hdf5: ops_dat " << dat_name
       << " does not exist in block " << block->name;
    throw ex;
  }

  // ops_dat exists .. now check ops_type, block_index, type and dim
  char read_ops_type[20];
  if (H5LTget_attribute_string(group_id, dat_name, "ops_type", read_ops_type) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"ops_type\" not found in data "
          "set"
       << dat_name;
    throw ex;
  } else {
    if (strcmp("ops_dat", read_ops_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_dat_hdf5: ops_type of dat " << dat_name
         << " is not ops_dat";
      throw ex;
    }
  }
  int read_block_index;
  if (H5LTget_attribute_int(group_id, dat_name, "block_index",
                            &read_block_index) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"block_index\" not found in "
          "data set "
       << dat_name;
    throw ex;
  } else {
    if (block->index != read_block_index) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_dat_hdf5: Attribute \"block_index\" mismatch for "
            "data set "
         << dat_name << " read " << read_block_index
         << " versus provided: " << block->index;
      throw ex;
    }
  }
  int read_dim;
  if (H5LTget_attribute_int(group_id, dat_name, "dim", &read_dim) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"dim\" not found in data set "
       << dat_name;
    throw ex;
  } else {
    if (dat_dim != read_dim) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_dat_hdf5: Attribute \"dim\" mismatch for data set "
         << dat_name << " read " << read_dim << " versus provided: " << dat_dim;
      throw ex;
    }
  }
  char read_type[20];
  if (H5LTget_attribute_string(group_id, dat_name, "type", read_type) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"type\" not found in data set "
       << dat_name;
    throw ex;
  } else {
    if (strcmp(type, read_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_dat_hdf5: Attribute \"type\" mismatch for data "
            "set "
         << dat_name << " read " << read_type << " versus provided: " << type;
      throw ex;
    }
  }

  // checks passed .. now read in all other details of ops_dat from file

  int read_size[OPS_MAX_DIM];
  if (H5LTget_attribute_int(group_id, dat_name, "size", read_size) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"size\" not found in data set "
       << dat_name;
    throw ex;
  }

  int read_d_m[OPS_MAX_DIM];
  if (H5LTget_attribute_int(group_id, dat_name, "d_m", read_d_m) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"d_m\" not found in data set "
       << dat_name;
    throw ex;
  }

  int read_d_p[OPS_MAX_DIM];
  if (H5LTget_attribute_int(group_id, dat_name, "d_p", read_d_p) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"d_p\" not found in data set "
       << dat_name;
    throw ex;
  }

  int read_base[OPS_MAX_DIM];
  if (H5LTget_attribute_int(group_id, dat_name, "base", read_base) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"base\" not found in data set "
       << dat_name;
    throw ex;
  }

  // set type size
  int type_size;
  if (strcmp(read_type, "double") == 0)
    type_size = sizeof(double);
  else if (strcmp(read_type, "float") == 0)
    type_size = sizeof(float);
  else if (strcmp(read_type, "int") == 0)
    type_size = sizeof(int);
  else if (strcmp(read_type, "long") == 0)
    type_size = sizeof(long);
  else if ((strcmp(read_type, "long long") == 0) ||
           (strcmp(read_type, "ll") == 0))
    type_size = sizeof(long long);
  else if (strcmp(read_type, "char") == 0)
    type_size = sizeof(char);
  else if (strcmp(read_type, "short") == 0)
    type_size = sizeof(short);
  else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: Unknown type in ops_decl_dat_hdf5(): " << type;
    throw ex;
  }

  // read in the actual data
  hsize_t t_size = 1;
  for (int d = 0; d < block->dims; d++)
    t_size *= read_size[d] - read_d_m[d] + read_d_p[d];
  char *data = (char *)ops_malloc(t_size * dat_dim * type_size);

  if (strcmp(read_type, "double") == 0)
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_DOUBLE, data);
  else if (strcmp(read_type, "float") == 0)
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_FLOAT, data);
  else if (strcmp(read_type, "int") == 0)
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_INT, data);
  else if (strcmp(read_type, "long") == 0)
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_LONG, data);
  else if ((strcmp(read_type, "long long") == 0) ||
           (strcmp(read_type, "ll") == 0))
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_LLONG, data);
  else if (strcmp(read_type, "char") == 0)
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_CHAR, data);
  else if (strcmp(read_type, "short") == 0)
    H5LTread_dataset(group_id, dat_name, H5T_NATIVE_SHORT, data);
  else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: Unknown type in ops_decl_dat_hdf5(): " << type;
    throw ex;
  }

  int stride[] = {1, 1, 1, 1, 1};
  ops_dat created_dat = ops_decl_dat_char(
      block, dat_dim, read_size /*global dat size in each dimension*/,
      read_base, read_d_m, read_d_p, stride, data, type_size /*size of(type)*/,
      type,
      dat_name); // TODO: multigrid stride support

  created_dat->is_hdf5 = 1;
  created_dat->hdf5_file = copy_str(file_name);
  created_dat->user_managed = 0;
  created_dat->mem = t_size * dat_dim * type_size;

  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

  return created_dat;
}

/*******************************************************************************
 * Routine to dump all ops_blocks, ops_dats etc to a named
 * HDF5 file
 *******************************************************************************/
// --- This routine is identical to the sequential routine in ops_hdf5.c
void ops_dump_to_hdf5(char const *file_name) {
  ops_dat_entry *item;
  for (int n = 0; n < OPS_instance::getOPSInstance()->OPS_block_index; n++) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      if (OPS_instance::getOPSInstance()->is_root())
        OPS_instance::getOPSInstance()->ostream()
            << "Dumping block "
            << OPS_instance::getOPSInstance()->OPS_block_list[n].block->name
            << " to HDF5 file " << file_name << "\n";
    ops_fetch_block_hdf5_file(
        OPS_instance::getOPSInstance()->OPS_block_list[n].block, file_name);
  }

  TAILQ_FOREACH(item, &OPS_instance::getOPSInstance()->OPS_dat_list, entries) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      if (OPS_instance::getOPSInstance()->is_root())
        OPS_instance::getOPSInstance()->ostream()
            << "Dumping dat " << (item->dat)->name << " to HDF5 file "
            << file_name << "\n";
    if (item->dat->e_dat !=
        1) // currently cannot write edge dats .. need to fix this
      ops_fetch_dat_hdf5_file(item->dat, file_name);
  }

  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_stencil_index; i++) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      if (OPS_instance::getOPSInstance()->is_root())
        OPS_instance::getOPSInstance()->ostream()
            << "Dumping stencil "
            << OPS_instance::getOPSInstance()->OPS_stencil_list[i]->name
            << " to HDF5 file " << file_name << "\n";
    ops_fetch_stencil_hdf5_file(
        OPS_instance::getOPSInstance()->OPS_stencil_list[i], file_name);
  }

  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_halo_index; i++) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      if (OPS_instance::getOPSInstance()->is_root())
        OPS_instance::getOPSInstance()->ostream()
            << "Dumping halo "
            << OPS_instance::getOPSInstance()->OPS_halo_list[i]->from->name
            << "--"
            << OPS_instance::getOPSInstance()->OPS_halo_list[i]->to->name
            << " to HDF5 file " << file_name << "\n";
    ops_fetch_halo_hdf5_file(OPS_instance::getOPSInstance()->OPS_halo_list[i],
                             file_name);
  }
}

/*******************************************************************************
 * Routine to copy over an ops_dat to a user specified memory pointer
 *******************************************************************************/
extern "C" char *ops_fetch_dat_char(ops_dat dat, char *u_dat) {
  // fetch data onto the host ( if needed ) based on the backend
  ops_get_data(dat);
  hsize_t t_size = 1;
  for (int d = 0; d < dat->block->dims; d++)
    t_size *= dat->size[d];
  u_dat = (char *)ops_malloc(t_size * dat->elem_size);
  memcpy(u_dat, dat->data, t_size * dat->elem_size);
  return (u_dat);
}

/*******************************************************************************
 * Routine to read in a constant from a named hdf5 file
 *******************************************************************************/

void ops_get_const_hdf5(char const *name, int dim, char const *type,
                        char *const_data, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t attr;      // attribute identifier

  if (file_exist(file_name) == 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "File " << file_name
       << " does not exist .... aborting ops_get_const_hdf5()\n";
    throw ex;
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);

  /// find dimension of this constant with available attributes
  int const_dim = 0;

  // open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    OPS_instance::getOPSInstance()->ostream()
        << "dataset '" << name << "' not found in file " << file_name << "\n";
    H5Fclose(file_id);
    const_data = NULL;
    return;
  }

  // get OID of the dim attribute
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &const_dim);
  H5Aclose(attr);
  H5Dclose(dset_id);
  if (const_dim != dim) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "dim of constant " << const_dim << " in file " << file_name
       << " and requested dim " << dim << "do not match\n";
    throw ex;
  }

  // find type with available attributes
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);

  int attlen = H5Aget_storage_size(attr);
  H5Tset_size(atype, attlen + 1);

  // read attribute
  char *typ = (char *)ops_malloc((attlen + 1) * sizeof(char));
  H5Aread(attr, atype, typ);
  H5Aclose(attr);
  H5Sclose(dataspace);
  H5Dclose(dset_id);
  if (strcmp(typ, type) != 0) {
    OPS_instance::getOPSInstance()->ostream()
        << "type of constant " << typ << " in file " << file_name
        << " and requested type " << type
        << " do not match, performing automatic type conversion\n";
    strcpy(typ, type);
  }

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  dataspace = H5Dget_space(dset_id);

  char *data;
  // initialize data buffer and read data
  if (strcmp(typ, "int") == 0 || strcmp(typ, "int(4)") == 0 ||
      strcmp(typ, "integer") == 0 || strcmp(typ, "integer(4)") == 0) {
    data = (char *)xmalloc(sizeof(int) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(int) * const_dim);
  } else if (strcmp(typ, "long") == 0) {
    data = (char *)xmalloc(sizeof(long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(long) * const_dim);
  } else if (strcmp(typ, "long long") == 0) {
    data = (char *)xmalloc(sizeof(long long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(long long) * const_dim);
  } else if (strcmp(typ, "float") == 0 || strcmp(typ, "real(4)") == 0 ||
             strcmp(typ, "real") == 0) {
    data = (char *)xmalloc(sizeof(float) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(float) * const_dim);
  } else if (strcmp(typ, "double") == 0 ||
             strcmp(typ, "double precision") == 0 ||
             strcmp(typ, "real(8)") == 0) {
    data = (char *)xmalloc(sizeof(double) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(double) * const_dim);
  } else if (strcmp(typ, "char") == 0) {
    data = (char *)xmalloc(sizeof(char) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(char) * const_dim);
  } else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Unknown type in file " << file_name << " for constant " << name
       << "\n";
    throw ex;
  }
  ops_free(typ);
  free(data);

  H5Dclose(dset_id);
  H5Fclose(file_id);
}

/*******************************************************************************
 * Routine to write a constant to a named hdf5 file
 *******************************************************************************/

void ops_write_const_hdf5(char const *name, int dim, char const *type,
                          char *const_data, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  htri_t status;   // status for checking return values
  hid_t attr;      // attribute identifier

  if (file_exist(file_name) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 3) {
      OPS_instance::getOPSInstance()->ostream()
          << "File " << file_name << " does not exist .... creating file\n";
    }
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(file_id);
  }

  /* Open the existing file. */
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);

  // Check if const already exists in data set
  status = H5Lexists(file_id, name, H5P_DEFAULT);
  if (status > 0) {
    OPS_instance::getOPSInstance()->ostream()
        << "dataset '" << name << "' already found in file " << file_name
        << " ...";

    // open existing data set
    dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

    // get OID of the dim attribute
    int const_dim = 0;
    attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_INT, &const_dim);
    H5Aclose(attr);
    if (const_dim != dim) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "dim of constant " << const_dim << " in file " << file_name
         << " and requested dim " << dim << "do not match\n";
      throw ex;
    }
    // find type with available attributes
    dataspace = H5Screate(H5S_SCALAR);
    hid_t atype = H5Tcopy(H5T_C_S1);
    attr = H5Aopen(dset_id, "type", H5P_DEFAULT);

    int attlen = H5Aget_storage_size(attr);
    H5Tset_size(atype, attlen + 1);

    // read attribute
    char *typ = (char *)ops_malloc((attlen + 1) * sizeof(char));
    H5Aread(attr, atype, typ);
    H5Aclose(attr);
    H5Sclose(dataspace);
    if (strcmp(typ, type) != 0) {
      OPS_instance::getOPSInstance()->ostream()
          << "type of constant " << typ << " in file " << file_name
          << " and requested type " << type
          << " do not match, performing automatic type conversion\n";
      strcpy(typ, type);
    }

    // existing const attributes matches with const to be written .. overwrite
    OPS_instance::getOPSInstance()->ostream() << " overwriting"
                                              << "\n";
    dataspace = H5Dget_space(dset_id);

    // write to the existing dataset with default properties
    if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
        strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
               const_data);
    } else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
               strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
               const_data);
    } else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
               strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
               strcmp(type, "integer(4)") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
               const_data);
    } else if ((strcmp(type, "long") == 0) || (strcmp(type, "long:soa") == 0)) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
               const_data);
    } else if ((strcmp(type, "long long") == 0) ||
               (strcmp(type, "long long:soa") == 0)) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
               const_data);
    } else if (strcmp(type, "char") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_CHAR, H5S_ALL, dataspace, H5P_DEFAULT,
               const_data);
    }
    ops_free(typ);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    return;
  }

  OPS_instance::getOPSInstance()->ostream()
      << "Const dataset " << name << " not found in file " << file_name
      << " ... creating const\n";

  // Create the dataspace for the dataset.
  hsize_t dims_of_const = (hsize_t)dim;
  dataspace = H5Screate_simple(1, &dims_of_const, NULL);

  // Create the dataset with default properties
  if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
      strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_DOUBLE, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
             strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
             strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_INT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long") == 0) || (strcmp(type, "long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long long") == 0) ||
             (strcmp(type, "long long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LLONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "char") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_CHAR, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_CHAR, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Unknown type for write_const for constant " << name << "\n";
    throw ex;
  }

  H5Sclose(dataspace);

  /*attach attributes to constant*/

  // open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  // create the data space for the attribute
  dims_of_const = 1;
  dataspace = H5Screate_simple(1, &dims_of_const, NULL);

  // Create an int attribute - dimension
  hid_t attribute = H5Acreate(dset_id, "dim", H5T_NATIVE_INT, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT);
  // Write the attribute data.
  H5Awrite(attribute, H5T_NATIVE_INT, &dim);
  // Close the attribute.
  H5Aclose(attribute);
  H5Sclose(dataspace);

  // Create a string attribute - type
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);

  int attlen = strlen(type);
  H5Tset_size(atype, attlen);
  attribute =
      H5Acreate(dset_id, "type", atype, dataspace, H5P_DEFAULT, H5P_DEFAULT);

  if (strcmp(type, "double") == 0 || strcmp(type, "double precision") == 0 ||
      strcmp(type, "real(8)") == 0)
    H5Awrite(attribute, atype, "double");
  else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 ||
           strcmp(type, "integer") == 0 || strcmp(type, "integer(4)") == 0)
    H5Awrite(attribute, atype, "int");
  else if (strcmp(type, "long") == 0)
    H5Awrite(attribute, atype, "long");
  else if (strcmp(type, "long long") == 0)
    H5Awrite(attribute, atype, "long long");
  else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0 ||
           strcmp(type, "real") == 0)
    H5Awrite(attribute, atype, "float");
  else if (strcmp(type, "char") == 0)
    H5Awrite(attribute, atype, "char");
  else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Unknown type in type " << type << " for constant " << name
       << ": cannot write constant to file\n";
    throw ex;
  }

  H5Aclose(attribute);
  H5Sclose(dataspace);
  H5Dclose(dset_id);

  H5Fclose(file_id);
}

void determin_plane_buf_size(const ops_dat &data, const int buf_dims,
                       const int cross_section_dir, int *buf_size) {
  const int space_dim{data->block->dims};
  int *size{new int(space_dim)};
  for (int d = 0; d < space_dim; d++) {
    size[d] = data->size[d] - (data->d_p[d] - data->d_m[d]);
  }

  int reduced_index{0};
  for (int d = 0; d < space_dim; d++) {
    if (d != cross_section_dir) {
      buf_size[reduced_index] = size[d];

      reduced_index++;
    }
  }
  delete size;
}

void determine_plane_range(const ops_dat &data, const int cross_section_dir,
                     const int pos, int *range) {
  const int space_dim{data->block->dims};
  int *size{new int(space_dim)};
  for (int d = 0; d < space_dim; d++) {
    size[d] = data->size[d] - (data->d_p[d] - data->d_m[d]);
  }

  for (int d = 0; d < data->block->dims; d++) {
    range[2 * d] = 0;
    range[2 * d + 1] = size[d];
  }
  range[2 * cross_section_dir + 1] = pos + 1;
  range[2 * cross_section_dir] = pos;
  delete size;
}

hid_t H5_file_handle(const char *file_name) {

  hid_t file_plist_id{H5Pcreate(H5P_FILE_ACCESS)};

  hid_t file_id;
  if (file_exist(file_name) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 3)
      ops_printf("File %s does not exist .... creating file\n", file_name);

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, file_plist_id);
  } else {
    file_id = H5Fopen(file_name, H5F_ACC_RDWR, file_plist_id);
  };
  H5Pclose(file_plist_id);
  return file_id;
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

void write_buf_hdf5(char const *file_name, const char *data_name,
                          const ops_dat &dat, const int dims, const int *size,
                          char *buf) {
  // HDF5 APIs definitions
  hid_t file_id{H5_file_handle(file_name)};
  hsize_t *size_f{new hsize_t(dims)};
  for (int d = 0; d < dims; d++) {
    size_f[d] = size[dims - d - 1];
  }
  hid_t file_space;
  hid_t dataset_id;
  std::vector<std::string> h5_name_list;
  split_h5_name(data_name, h5_name_list);
  std::vector<hid_t> groupid_list;
  groupid_list.resize(h5_name_list.size() - 1);

  H5_dataset_space(file_id, dims, size_f, h5_name_list, dat->type, groupid_list,
                   dataset_id, file_space);
  H5Dwrite(dataset_id, h5_type(dat->type), H5S_ALL, file_space, H5P_DEFAULT,
           buf);
  H5Sclose(file_space);
  H5Dclose(dataset_id);
  for (int grp = groupid_list.size() - 1; grp >= 0; grp--) {
    H5Gclose(groupid_list[grp]);
  }
  H5Fclose(file_id);
  delete size_f;
}

void ops_write_plane_hdf5(const ops_dat dat, const int cross_section_dir,
                          const int pos, char const *file_name,
                          const char *data_name) {
  if ((cross_section_dir >= 0) && (cross_section_dir <= dat->block->dims)) {
    if ((pos >= dat->base[cross_section_dir]) &&
        (pos <= dat->size[cross_section_dir])) {

      int dims{dat->block->dims - 1};
      int *range{new int(2 * dat->block->dims)};
      int *size{new int(dims)};
      determin_plane_buf_size(dat, dims, cross_section_dir, size);
      hsize_t buf_element_size{1};
      for (int i = 0; i < dims; i++) {
        buf_element_size *= size[i];
      }
      // Consider multi-dim data
      buf_element_size *= dat->elem_size;
      char *write_buf = (char *)ops_malloc(buf_element_size);
      determine_plane_range(dat, cross_section_dir, pos, range);
      ops_dat_fetch_data_slab_host(dat, 0, write_buf, range);
      // Consider the multi-dim data
      size[0] *= (dat->dim);
      write_buf_hdf5(file_name, data_name, dat, dims, size, write_buf);
      delete range;
      free(write_buf);
      delete size;
    } else {
      ops_printf("The dat %s doesn't have the specified plane = %d \n",
                 dat->name, pos);
    }
  } else {
    ops_printf(
        "The block %s doesn't have the specified cross section direction "
        "%d\n",
        dat->block->name, cross_section_dir);
  }
}

void ops_write_data_slab_hdf5(const ops_dat dat, const int *range,
                              const char *file_name, const char *data_name) {
  const int dims{dat->block->dims};
  int *size{new int(dims)};
  size_t total_size{1};
  for (int d = 0; d < dims; d++) {
    size[d] = range[2 * d + 1] - range[2 * d];
    total_size *= size[d];
  }
  size[0] *= (dat->dim);
  total_size *= (dat->elem_size);

  char *write_buf = (char *)ops_malloc(total_size);
  ops_dat_fetch_data_slab_host(dat, 0, write_buf, (int *)range);
  write_buf_hdf5(file_name, data_name, dat, dims, size, write_buf);
  free(write_buf);
  delete size;
}

void ops_write_plane_group_hdf5(
    const std::vector<std::pair<int, int>> &planes,
    std::vector<std::string> &plane_names, const std::string &key,
    const std::vector<std::vector<ops_dat>> &data_list) {
  const size_t plane_num{planes.size()};
  std::vector<std::string> plane_name_base{"I", "J", "K"};
  if (plane_names.size() < plane_num) {
    size_t current_size(plane_names.size());
    plane_names.resize(planes.size());

    for (size_t p = current_size; p < planes.size(); p++) {
      plane_names[p] = plane_name_base[planes[p].first % OPS_MAX_DIM] +
                       std::to_string(planes[p].second);
    }
  }

  for (size_t p = 0; p < plane_num; p++) {
    for (const auto &data_plane : data_list) {
      for (const auto &data : data_plane) {
        const int cross_section_dir{planes[p].first};
        const int pos{planes[p].second};
        std::string block_name{data->block->name};
        std::string data_name{data->name};
        std::string file_name{plane_names[p] + ".h5"};
        std::string dataset_name{block_name + "/" + key + "/" + data_name};
        ops_write_plane_hdf5(data, cross_section_dir, pos, file_name.c_str(),
                             dataset_name.c_str());
      }
    }
  }
}

void ops_write_plane_group_hdf5(
    const std::vector<std::pair<int, int>> &planes, const std::string &key,
    const std::vector<std::vector<ops_dat>> &data_list) {
  const size_t plane_num{planes.size()};
  std::vector<std::string> plane_names;
  plane_names.resize(plane_num);
  std::vector<std::string> plane_name_base{"I", "J", "K"};
  for (size_t p = 0; p < plane_num; p++) {
    plane_names[p] = plane_name_base[planes[p].first % OPS_MAX_DIM] +
                     std::to_string(planes[p].second);
  }
  ops_write_plane_group_hdf5(planes, plane_names, key, data_list);
}
