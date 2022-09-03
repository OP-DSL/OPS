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
 * @brief HDF5 file I/O backend implementation for MPI
 * @author Gihan Mudalige (started 28-08-2015)
 * @details Implements the OPS API calls for the HDF5 file I/O functionality
 */

#include <math.h>
#include <mpi.h>
#include <ops_exceptions.h>
#include <ops_mpi_core.h>
#include <ops_util.h>
#include <vector>
#include <tuple>

// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

// hdf5 header
#include <hdf5.h>
#include <hdf5_hl.h>

static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)calloc(len + 16, sizeof(char));
  return strncpy(dest, src, len);
}

//
// MPI Communicator for parallel I/O
//

MPI_Comm OPS_MPI_HDF5_WORLD;

extern sub_block_list *OPS_sub_block_list; // pointer to list holding sub-block
// geometries

extern sub_dat_list *OPS_sub_dat_list; // pointer to list holding sub-dat
// details

extern void (*ops_read_dat_hdf5_dynamic)(ops_dat dat);
/*******************************************************************************
 * Routine to remove the intra-block (i.e. MPI) halos from the flattened 1D dat
 * before writing to HDF5 files - Maximum dimension of block is 1
 *******************************************************************************/
void remove_mpi_halos1D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
  int index = 0;
  int count = 0;
  // for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  //  for(int k = disp[2]; k < disp[2]+size[2]; k++) {
  // for(int j = disp[1]; j < disp[1]+size[1]; j++) {
  for (int i = disp[0]; i < disp[0] + size[0]; i++) {
    index = i; //+
    // j * dat->size[0]; //+ // need to stride in dat->size as data block
    // includes intra-block halos
    // k * dat->size[0] * dat->size[1];// +
    // l * dat->size[0] * dat->size[1] * dat->size[2] +
    // m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
    memcpy(&data[count * dat->elem_size], &dat->data[index * dat->elem_size],
           dat->elem_size);
    count++;
  }
  // }
  //}
  //}
  //}
  return;
  /*
     x = disp[0] to size[0]
     y = disp[1] to size[1]
     z = disp[2] to size[2]
     t = disp[3] to size[4]
     u = disp[4] to size[4]
     index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3 + u * D1 * D2 * D3 *
     D4;


     D1 - dat->size[0]
     D2 - dat->size[1]
     D3 - dat->size[2]
     D4 - dat->size[3]
   */
}

/*******************************************************************************
 * Routine to remove the intra-block (i.e. MPI) halos from the flattened 1D dat
 * before writing to HDF5 files - Maximum dimension of block is 2
 *******************************************************************************/
void remove_mpi_halos2D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
  int index = 0;
  int count = 0;
  // for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  //  for(int k = disp[2]; k < disp[2]+size[2]; k++) {
  for (int j = disp[1]; j < disp[1] + size[1]; j++) {
    for (int i = disp[0]; i < disp[0] + size[0]; i++) {
      index = i + j * dat->size[0]; //+ // need to stride in dat->size as data
      // block includes intra-block halos
      // k * dat->size[0] * dat->size[1];// +
      // l * dat->size[0] * dat->size[1] * dat->size[2] +
      // m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
      memcpy(&data[count * dat->elem_size], &dat->data[index * dat->elem_size],
             dat->elem_size);
      count++;
    }
  }
  //}
  //}
  //}
  return;
  /*
     x = disp[0] to size[0]
     y = disp[1] to size[1]
     z = disp[2] to size[2]
     t = disp[3] to size[4]
     u = disp[4] to size[4]
     index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3 + u * D1 * D2 * D3 *
     D4;


     D1 - dat->size[0]
     D2 - dat->size[1]
     D3 - dat->size[2]
     D4 - dat->size[3]
   */
}

/*******************************************************************************
 * Routine to remove the intra-block (i.e. MPI) halos from the flattened 1D dat
 * before writing to HDF5 files - Maximum dimension of block is 3
 *******************************************************************************/
void remove_mpi_halos3D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
  int index = 0;
  int count = 0;
  // for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  for (int k = disp[2]; k < disp[2] + size[2]; k++) {
    for (int j = disp[1]; j < disp[1] + size[1]; j++) {
      for (int i = disp[0]; i < disp[0] + size[0]; i++) {
        index = i + j * dat->size[0] + // need to stride in dat->size as data
                                       // block includes intra-block halos
                k * dat->size[0] * dat->size[1]; // +
        // l * dat->size[0] * dat->size[1] * dat->size[2] +
        // m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
        memcpy(&data[count * dat->elem_size],
               &dat->data[index * dat->elem_size], dat->elem_size);
        count++;
      }
    }
  }
  //}
  //}
  return;
}

/*******************************************************************************
 * Routine to remove the intra-block (i.e. MPI) halos from the flattened 1D dat
 * before writing to HDF5 files - Maximum dimension of block is 4
 *******************************************************************************/
void remove_mpi_halos4D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
}

/*******************************************************************************
 * Routine to remove the intra-block (i.e. MPI) halos from the flattened 1D dat
 * before writing to HDF5 files - Maximum dimension of block is 5
 *******************************************************************************/
void remove_mpi_halos5D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
}

/*******************************************************************************
 * Routine to add the intra-block halos (i.e. MPI) from the flattened 1D data
 * after reading from an HDF5 file - Maximum dimension of block is 2
 *******************************************************************************/
void add_mpi_halos2D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
  int index = 0;
  int count = 0;
  // for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  //  for(int k = disp[2]; k < disp[2]+size[2]; k++) {
  for (int j = disp[1]; j < disp[1] + size[1]; j++) {
    for (int i = disp[0]; i < disp[0] + size[0]; i++) {
      index = i + j * dat->size[0]; //+ // need to stride in dat->size as data
      // block includes intra-block halos
      // k * dat->size[0] * dat->size[1];// +
      // l * dat->size[0] * dat->size[1] * dat->size[2] +
      // m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
      memcpy(&dat->data[index * dat->elem_size], &data[count * dat->elem_size],
             dat->elem_size);
      count++;
    }
  }
  //}
  //}
  //}
  return;
}

void add_mpi_halos3D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data) {
  int index = 0;
  int count = 0;
  // for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  for (int k = disp[2]; k < disp[2] + size[2]; k++) {
    for (int j = disp[1]; j < disp[1] + size[1]; j++) {
      for (int i = disp[0]; i < disp[0] + size[0]; i++) {
        index = i + j * dat->size[0] + // need to stride in dat->size as data
                                       // block includes intra-block halos
                k * dat->size[0] * dat->size[1]; // +
        // l * dat->size[0] * dat->size[1] * dat->size[2] +
        // m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
        memcpy(&dat->data[index * dat->elem_size],
               &data[count * dat->elem_size], dat->elem_size);
        count++;
      }
    }
  }
  //}
  //}
  return;
};
void add_mpi_halos4D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data){};
void add_mpi_halos5D(ops_dat dat, hsize_t *size, hsize_t *disp, char *data){};

/*******************************************************************************
 * Routine to write an ops_block to a named hdf5 file,
 * if file does not exist, creates it
 * if the block does not exists in file creates block as a hdf5 group
 *******************************************************************************/
void ops_fetch_block_hdf5_file(ops_block block, char const *file_name) {
  sub_block *sb = OPS_sub_block_list[block->index];

  if (sb->owned == 1) {
    // HDF5 APIs definitions
    hid_t file_id;  // file identifier
    hid_t group_id; // group identifier
    hid_t plist_id; // property list identifier

    // create new communicator
    int my_rank, comm_size;
    MPI_Comm OPS_MPI_HDF5_BLOCK_WORLD;
    // use the communicator for MPI procs holding this block
    MPI_Comm_dup(sb->comm1, &OPS_MPI_HDF5_BLOCK_WORLD);
    MPI_Comm_rank(OPS_MPI_HDF5_BLOCK_WORLD, &my_rank);
    MPI_Comm_size(OPS_MPI_HDF5_BLOCK_WORLD, &comm_size);

    // MPI variables
    MPI_Info info = MPI_INFO_NULL;

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_BLOCK_WORLD, info);

    if (file_exist(file_name) == 0) {
      if (OPS_instance::getOPSInstance()->OPS_diags > 2)
        ops_printf("File %s does not exist .... creating file\n", file_name);
      if (ops_is_root()) {
        FILE *fp;
        fp = fopen(file_name, "w");
        fclose(fp);
      }
      // Create a new file collectively and release property list identifier.
      file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
      H5Fclose(file_id);
    }

    file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

    if (H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
      if (OPS_instance::getOPSInstance()->OPS_diags > 2)
        ops_printf(
            "ops_block %s does not exists in file %s ... creating ops_block\n",
            block->name, file_name);
      // create group - ops_block
      group_id = H5Gcreate(file_id, block->name, H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT);
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
    MPI_Comm_free(&OPS_MPI_HDF5_BLOCK_WORLD);
  }
  MPI_Barrier(OPS_MPI_GLOBAL); // wait for every rank to finish their I/O
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

  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    MPI_Barrier(OPS_MPI_GLOBAL);
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      ops_printf("File %s does not exist .... creating file\n", file_name);
    MPI_Barrier(OPS_MPI_HDF5_WORLD);
    if (ops_is_root()) {
      FILE *fp;
      fp = fopen(file_name, "w");
      fclose(fp);
    }
    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  hsize_t rank = 1;
  hsize_t elems = stencil->dims * stencil->points;

  /* create and write the dataset */
  if (H5Lexists(file_id, stencil->name, H5P_DEFAULT) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      ops_printf(
          "ops_stencil %s does not exists in the file ... creating data\n",
          stencil->name);
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
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
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

  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    MPI_Barrier(OPS_MPI_GLOBAL);
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      ops_printf("File %s does not exist .... creating file\n", file_name);
    MPI_Barrier(OPS_MPI_HDF5_WORLD);
    if (ops_is_root()) {
      FILE *fp;
      fp = fopen(file_name, "w");
      fclose(fp);
    }
    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  char halo_name[100]; // strlen(halo->from->name)+strlen(halo->to->name)];
  sprintf(halo_name, "from_%s_to_%s", halo->from->name, halo->to->name);

  /* create and write the a group that holds the halo information */
  if (H5Lexists(file_id, halo_name, H5P_DEFAULT) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      ops_printf(
          "ops_halo %s does not exists in the file ... creating group to "
          "hold halo\n",
          halo_name);
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
 * Routine to write an ops_dat to a named hdf5 file,
 * if file does not exist, creates it
 * if the data set does not exists in file creates data set
 *******************************************************************************/
void ops_fetch_dat_hdf5_file(ops_dat dat, char const *file_name) {
  sub_block *sb = OPS_sub_block_list[dat->block->index];
  if (sb->owned == 1) {
    // fetch data onto the host ( if needed ) based on the backend
    ops_get_data(dat);

    // compute the number of elements that this process will write to the final
    // file
    // also compute the correct offsets on the final file that this process
    // should begin from to write
    sub_dat *sd = OPS_sub_dat_list[dat->index];
    ops_block block = dat->block;

    hsize_t disp[block->dims]; // global disps to compute the chunk data set
    // dimensions
    hsize_t l_disp[block->dims]; // local disps to remove MPI halos
    hsize_t size[block->dims];   // local size to compute the chunk data set
    // dimensions
    hsize_t gbl_size[block->dims]; // global size to compute the chunk data set
    // dimensions

    int g_size[block->dims]; // global size of the dat attribute to write to
    // hdf5 file
    int g_d_m[block->dims]; // global size of the block halo (-) depth
                            // attribute
    // to write to hdf5 file
    int g_d_p[block->dims]; // global size of the block halo (+) depth
                            // attribute
    // to write to hdf5 file

    hsize_t count[block->dims];  // parameters for for hdf5 file chuck writing
    hsize_t stride[block->dims]; // parameters for for hdf5 file chuck writing

    for (int d = 0; d < block->dims; d++) {
      // remove left MPI halo to get start disp from beginning of dat
      // include left block halo
      disp[d] = sd->decomp_disp[d] -
                sd->gbl_d_m[d];    // global displacements of the data set
      l_disp[d] = 0 - sd->d_im[d]; // local displacements of the data set (i.e.
      // per MPI proc)
      size[d] = sd->decomp_size[d]; // local size to compute the chunk data set
      // dimensions
      gbl_size[d] = sd->gbl_size[d]; // global size to compute the chunk data
      // set dimensions

      g_d_m[d] = sd->gbl_d_m[d]; // global halo depth(-) attribute to be written
      // to hdf5 file
      g_d_p[d] = sd->gbl_d_p[d]; // global halo depth(+) attribute to be written
      // to hdf5 file
      g_size[d] = sd->gbl_size[d] + g_d_m[d] -
                  g_d_p[d]; // global size attribute to be written to hdf5 file

      count[d] = 1;
      stride[d] = 1;

      // printf("l_disp[%d] = %d ",d,l_disp[d]);
      // printf("disp[%d] = %d ",d,disp[d]);
      // printf("size[%d] = %d ",d,size[d]);
      // printf("dat->size[%d] = %d ",d,dat->size[d]);
      // printf("gbl_size[%d] = %d \n",d,sd->gbl_size[d]);
      // printf("g_size[%d] = %d ",d,g_size[d]);
      // printf("dat->d_m[%d] = %d ",d,g_d_m[d]);
      // printf("dat->d_p[%d] = %d ",d,g_d_p[d]);
    }

    hsize_t t_size = 1;
    for (int d = 0; d < dat->block->dims; d++)
      t_size *= size[d];
    // printf("t_size = %d ",t_size);
    char *data = (char *)ops_malloc(t_size * dat->elem_size);

    // create new communicator
    int my_rank, comm_size;
    MPI_Comm OPS_MPI_HDF5_BLOCK_WORLD;
    // use the communicator for MPI procs holding this block
    MPI_Comm_dup(sb->comm1, &OPS_MPI_HDF5_BLOCK_WORLD);
    MPI_Comm_rank(OPS_MPI_HDF5_BLOCK_WORLD, &my_rank);
    MPI_Comm_size(OPS_MPI_HDF5_BLOCK_WORLD, &comm_size);

    if (block->dims == 1)
      remove_mpi_halos1D(dat, size, l_disp, data);
    else if (block->dims == 2)
      remove_mpi_halos2D(dat, size, l_disp, data);
    else if (block->dims == 3)
      remove_mpi_halos3D(dat, size, l_disp, data);
    else if (block->dims == 4)
      remove_mpi_halos4D(dat, size, l_disp, data);
    else if (block->dims == 5)
      remove_mpi_halos5D(dat, size, l_disp, data);

    // make sure we multiply by the number of data values per
    // element (i.e. dat->dim) to get full size of the data
    size[0] = size[0] * dat->dim;
    disp[0] = disp[0] * dat->dim;
    if (block->dims == 1) {
      gbl_size[0] = gbl_size[0] * dat->dim; //-- this needs to be tested for 1D
    } else if (block->dims == 2) {
      // Jianping Meng: I found that growing the dim 0 rather than dim 1 can
      // lead to a more consistent post-processing procedure for multi-dim data
      //**note we are using [1] instead of [0] here !!
      gbl_size[0] = gbl_size[0] * dat->dim;
    } else if (block->dims == 3) {
      gbl_size[0] =
          gbl_size[0] * dat->dim; //**note that for 3D we are using [0] here !!
    }

    // MPI variables
    MPI_Info info = MPI_INFO_NULL;

    // HDF5 APIs definitions
    hid_t file_id;   // file identifier
    hid_t group_id;  // group identifier
    hid_t dset_id;   // dataset identifier
    hid_t filespace; // data space identifier
    hid_t plist_id;  // property list identifier
    hid_t memspace;  // memory space identifier

    // hsize_t CHUNK_SIZE[block->dims];

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_BLOCK_WORLD, info);

    if (file_exist(file_name) == 0) {
      // MPI_Barrier(OPS_MPI_GLOBAL);
      if (OPS_instance::getOPSInstance()->OPS_diags > 2)
        ops_printf("File %s does not exist .... creating file\n", file_name);
      // MPI_Barrier(OPS_MPI_HDF5_BLOCK_WORLD);
      if (ops_is_root()) {
        FILE *fp;
        fp = fopen(file_name, "w");
        fclose(fp);
      }
      // Create a new file collectively and release property list identifier.
      file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
      H5Fclose(file_id);
    }

    file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);

    if (H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: Error: ops_fetch_dat_hdf5_file: ops_block on which this "
            "ops_dat "
         << dat->name << " is declared does not exist in the file";
      throw ex;
    } else {
      // open existing group -- an ops_block is a group
      group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);
      int overwriting = 0;

      if (H5Lexists(group_id, dat->name, H5P_DEFAULT) == 0) {
        if (OPS_instance::getOPSInstance()->OPS_diags > 2)
          ops_printf(
              "ops_fetch_dat_hdf5_file: ops_dat %s does not exists in the "
              "ops_block %s ... creating ops_dat\n",
              dat->name, block->name);

        // transpose global size as on hdf5 file the dims are written transposed
        hsize_t GBL_SIZE[block->dims];
        if (block->dims == 1) {
          GBL_SIZE[0] = gbl_size[0];
        } else if (block->dims == 2) {
          GBL_SIZE[0] = gbl_size[1];
          GBL_SIZE[1] = gbl_size[0];
        } else if (block->dims == 3) {
          GBL_SIZE[0] = gbl_size[2];
          GBL_SIZE[1] = gbl_size[1];
          GBL_SIZE[2] = gbl_size[0];
        }

        // Create the dataspace for the dataset
        filespace =
            H5Screate_simple(block->dims, GBL_SIZE, NULL); // space in file

        // Create a reasonable chunk size
        /*int cart_dims[OPS_MAX_DIM], cart_periods[OPS_MAX_DIM],
            cart_coords[OPS_MAX_DIM];
        MPI_Cart_get(sb->comm, block->dims, cart_dims, cart_periods,
                     cart_coords);

        for (int i = 0; i < block->dims; i++) {
          CHUNK_SIZE[i] = MAX((GBL_SIZE[i] / 10), 1); // need to have a constant
        chinksize
                                                      // regardless of the
        number of mpi procs

          printf("%s GBL_SIZE[%d] = %d, size[%d] = %d, cart_dims[%d] = %d, "
                 "CHUNK_SIZE[%d] = %d, value[%d] = %lf\n",
                 dat->name, i, GBL_SIZE[i], i, size[i], i, cart_dims[i], i,
                 CHUNK_SIZE[i], i, GBL_SIZE[i] / (double)cart_dims[i]);
        }*/

        // Create chunked dataset
        plist_id = H5Pcreate(H5P_DATASET_CREATE);
        // H5Pset_chunk(plist_id, block->dims, GBL_SIZE); // chunk data set need
        //  to be the same size
        //  on each proc

        // Create the dataset with default properties and close filespace.
        if (strcmp(dat->type, "double") == 0 ||
            strcmp(dat->type, "double precision") == 0 ||
            strcmp(dat->type, "real(8)") == 0)
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_DOUBLE, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        else if (strcmp(dat->type, "float") == 0 ||
                 strcmp(dat->type, "real(4)") == 0 ||
                 strcmp(dat->type, "real") == 0)
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_FLOAT, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        else if (strcmp(dat->type, "int") == 0 ||
                 strcmp(dat->type, "int(4)") == 0 ||
                 strcmp(dat->type, "integer") == 0 ||
                 strcmp(dat->type, "integer(4)") == 0) {
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_INT, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        } else if (strcmp(dat->type, "long") == 0)
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_LONG, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        else if ((strcmp(dat->type, "long long") == 0) ||
                 (strcmp(dat->type, "ll") == 0))
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_LLONG, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        else if (strcmp(dat->type, "short") == 0)
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_SHORT, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        else if (strcmp(dat->type, "char") == 0)
          dset_id = H5Dcreate(group_id, dat->name, H5T_NATIVE_CHAR, filespace,
                              H5P_DEFAULT, plist_id, H5P_DEFAULT);
        else {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: Unknown type in ops_fetch_dat_hdf5_file(): "
             << dat->type;
          throw ex;
        }
        H5Pclose(plist_id);
        H5Sclose(filespace);
        H5Dclose(dset_id);

        // attach attributes to dat
        H5LTset_attribute_string(group_id, dat->name, "ops_type",
                                 "ops_dat"); // ops type
        H5LTset_attribute_string(group_id, dat->name, "block",
                                 block->name); // block
        H5LTset_attribute_int(group_id, dat->name, "block_index",
                              &(block->index), 1); // block index
        H5LTset_attribute_int(group_id, dat->name, "dim", &(dat->dim),
                              1); // dim
        H5LTset_attribute_int(group_id, dat->name, "size", g_size,
                              block->dims); // size
        H5LTset_attribute_int(group_id, dat->name, "d_m", g_d_m,
                              block->dims); // d_m
        H5LTset_attribute_int(group_id, dat->name, "d_p", g_d_p,
                              block->dims); // d_p
        H5LTset_attribute_int(group_id, dat->name, "base", dat->base,
                              block->dims); // base
        H5LTset_attribute_string(group_id, dat->name, "type",
                                 dat->type); // type
      } else {
        ops_printf("Dataset %s already found in file %s ... ", dat->name,
                   file_name);
        overwriting = 1;
      }

      //
      // check attributes .. error if not equal
      //
      char read_ops_type[20];
      if (H5LTget_attribute_string(group_id, dat->name, "ops_type",
                                   read_ops_type) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_fetch_dat_hdf5_file: Attribute \"ops_type\" not "
              "found in data set"
           << dat->name;
        throw ex;
      } else {
        if (strcmp("ops_dat", read_ops_type) != 0) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_fetch_dat_hdf5_file: ops_type of dat " << dat->name
             << " is not ops_dat";
          throw ex;
        }
      }

      char read_block_name[30];
      if (H5LTget_attribute_string(group_id, dat->name, "block",
                                   read_block_name) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"block\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        if (strcmp(block->name, read_block_name) != 0) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_decl_block_hdf5: Attribute \"block\" mismatch "
                "for data set "
             << dat->name << "block name: " << block->name
             << " read block name: " << read_block_name;
          throw ex;
        }
      }

      int read_block_index;
      if (H5LTget_attribute_int(group_id, dat->name, "block_index",
                                &read_block_index) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"block_index\" not "
              "found in data set "
           << dat->name;
        throw ex;
      } else {
        if (block->index != read_block_index) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_decl_block_hdf5: Attribute \"block_index\" "
                "mismatch for data set "
             << dat->name << " read " << read_block_index
             << " versus provided: " << block->index;
          throw ex;
        }
      }

      int read_dim;
      if (H5LTget_attribute_int(group_id, dat->name, "dim", &read_dim) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"dim\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        if (dat->dim != read_dim) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_decl_block_hdf5: Attribute \"dim\" mismatch for "
                "data set "
             << dat->name << " read " << read_dim
             << " versus provided: " << dat->dim;
          throw ex;
        }
      }

      int read_size[block->dims];
      if (H5LTget_attribute_int(group_id, dat->name, "size", read_size) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"size\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        for (int d = 0; d < block->dims; d++) {
          if (g_size[d] != read_size[d]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_decl_block_hdf5: Attribute \"size\" mismatch "
                  "for data set "
               << dat->name << " read " << read_size[d]
               << " versus provided: " << g_size[d] << " in dim " << d;
            throw ex;
          }
        }
      }

      int read_d_m[block->dims];
      if (H5LTget_attribute_int(group_id, dat->name, "d_m", read_d_m) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"d_m\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        for (int d = 0; d < block->dims; d++) {
          if (g_d_m[d] != read_d_m[d]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_decl_block_hdf5: Attribute \"d_m\" mismatch "
                  "for data set "
               << dat->name << " read " << read_d_m[d]
               << " versus provided: " << g_d_m[d] << " in dim " << d;
            throw ex;
          }
        }
      }

      int read_d_p[block->dims];
      if (H5LTget_attribute_int(group_id, dat->name, "d_p", read_d_p) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"d_p\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        for (int d = 0; d < block->dims; d++) {
          if (g_d_p[d] != read_d_p[d]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_decl_block_hdf5: Attribute \"d_p\" mismatch "
                  "for data set "
               << dat->name << " read " << read_d_p[d]
               << " versus provided: " << g_d_p[d] << " in dim " << d;
            throw ex;
          }
        }
      }

      int read_base[block->dims];
      if (H5LTget_attribute_int(group_id, dat->name, "base", read_base) < 0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"base\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        for (int d = 0; d < block->dims; d++) {
          if (dat->base[d] != read_base[d]) {
            OPSException ex(OPS_HDF5_ERROR);
            ex << "Error: ops_decl_block_hdf5: Attribute \"base\" mismatch "
                  "for data set "
               << dat->name << " read " << read_base[d]
               << " versus provided: " << dat->base[d] << " in dim " << d;
            throw ex;
          }
        }
      }

      char read_type[15];
      if (H5LTget_attribute_string(group_id, dat->name, "type", read_type) <
          0) {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: ops_decl_block_hdf5: Attribute \"type\" not found in "
              "data set "
           << dat->name;
        throw ex;
      } else {
        if (strcmp(dat->type, read_type) != 0) {
          OPSException ex(OPS_HDF5_ERROR);
          ex << "Error: ops_decl_block_hdf5: Attribute \"type\" mismatch for "
                "data set "
             << dat->name << " read " << read_type
             << " versus provided: " << dat->type;
          throw ex;
        }
      }

      // all good , overwrite the existing dataset
      if (overwriting)
        ops_printf("overwriting\n");

      // open existing dat
      dset_id = H5Dopen(group_id, dat->name, H5P_DEFAULT);

      // Need to flip the dimensions to accurately write to HDF5 chunk
      // decomposition
      hsize_t DISP[block->dims];
      hsize_t SIZE[block->dims];
      if (block->dims == 1) {
        DISP[0] = disp[0];
        SIZE[0] = size[0];
      } else if (block->dims == 2) {
        DISP[0] = disp[1];
        DISP[1] = disp[0];
        SIZE[0] = size[1];
        SIZE[1] = size[0];
      } else if (block->dims == 3) {
        DISP[0] = disp[2];
        DISP[1] = disp[1]; // note how dimension 1 remains the same !!
        DISP[2] = disp[0];
        SIZE[0] = size[2];
        SIZE[1] = size[1]; // note how dimension 1 remains the same !!
        SIZE[2] = size[0];
      }

      memspace = H5Screate_simple(
          block->dims, SIZE,
          NULL); // block of memory to write to file by each proc

      // Select hyperslab
      filespace = H5Dget_space(dset_id);
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, DISP, stride, count, SIZE);

      // Create property list for collective dataset write.
      plist_id = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

      // write data
      if (strcmp(dat->type, "double") == 0 ||
          strcmp(dat->type, "double precision") == 0 ||
          strcmp(dat->type, "real(8)") == 0) {
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id,
                 data);
      } else if (strcmp(dat->type, "float") == 0 ||
                 strcmp(dat->type, "real(4)") == 0 ||
                 strcmp(dat->type, "real") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id,
                 data);
      else if (strcmp(dat->type, "int") == 0 ||
               strcmp(dat->type, "int(4)") == 0 ||
               strcmp(dat->type, "integer") == 0 ||
               strcmp(dat->type, "integer(4)") == 0) {
        H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, data);
      } else if (strcmp(dat->type, "long") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, filespace, plist_id, data);
      else if ((strcmp(dat->type, "long long") == 0) ||
               (strcmp(dat->type, "ll") == 0))
        H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, filespace, plist_id,
                 data);
      else if (strcmp(dat->type, "short") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_SHORT, memspace, filespace, plist_id,
                 data);
      else if (strcmp(dat->type, "char") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_CHAR, memspace, filespace, plist_id, data);
      else {
        OPSException ex(OPS_HDF5_ERROR);
        ex << "Error: Unknown type in ops_fetch_dat_hdf5_file(): " << dat->type;
        throw ex;
      }

      MPI_Barrier(OPS_MPI_HDF5_BLOCK_WORLD);
      free(data);

      H5Sclose(filespace);
      H5Pclose(plist_id);
      H5Dclose(dset_id);
      H5Sclose(memspace);

      H5Gclose(group_id);
      H5Fclose(file_id);
      MPI_Comm_free(&OPS_MPI_HDF5_BLOCK_WORLD);
    }
  }
  MPI_Barrier(OPS_MPI_GLOBAL); // wait for every rank to finish their I/O
  return;
}

/*******************************************************************************
 * Routine to read an ops_block from an hdf5 file
 *******************************************************************************/
ops_block ops_decl_block_hdf5(int dims, const char *block_name,
                              char const *file_name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    MPI_Barrier(OPS_MPI_GLOBAL);
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_block_hdf5: file " << file_name << " does not exist";
    throw ex;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
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
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    MPI_Barrier(OPS_MPI_GLOBAL);
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: file " << file_name
       << " does not exist";
    throw ex;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
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
          "stencil "
       << stencil_name;
    throw ex;
  } else {
    if (strcmp("ops_stencil", read_ops_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_stencil_hdf5: ops_type of stencil " << stencil_name
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
  int read_stride[read_dims];
  if (H5LTget_attribute_int(file_id, stencil_name, "stride", read_stride) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"stride\" not found in "
          "stencil"
       << stencil_name;
    throw ex;
  }

  int read_sten[read_dims * read_points];
  H5LTread_dataset_int(file_id, stencil_name, read_sten);
  H5Pclose(plist_id);
  H5Fclose(file_id);

  // use decl_strided stencil for both normal and strided stencils
  return ops_decl_strided_stencil(read_dims, read_points, read_sten,
                                  read_stride, stencil_name);
}

/*******************************************************************************
 * Routine to read an ops_halo from an hdf5 file
 *******************************************************************************/
ops_halo ops_decl_halo_hdf5(ops_dat from, ops_dat to, char const *file_name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    MPI_Barrier(OPS_MPI_GLOBAL);
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_halo_hdf5: file " << file_name << " does not exist";
    throw ex;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
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
    ex << "Error: ops_decl_halo_hdf5: Attribute \"ops_type\" not found in halo "
       << halo_name;
    throw ex;
  } else {
    if (strcmp("ops_halo", read_ops_type) != 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_decl_halo_hdf5: ops_type of halo " << halo_name
         << " is not ops_halo";
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
  int dim = from->block->dims;

  // checks passed ..

  // get the iter_size
  int read_iter_size[dim];
  if (H5LTget_attribute_int(file_id, halo_name, "iter_size", read_iter_size) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"iter_size\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the from_base
  int read_from_base[dim];
  if (H5LTget_attribute_int(file_id, halo_name, "from_base", read_from_base) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"from_base\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the to_base
  int read_to_base[dim];
  if (H5LTget_attribute_int(file_id, halo_name, "to_base", read_to_base) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"to_base\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the from_dir
  int read_from_dir[dim];
  if (H5LTget_attribute_int(file_id, halo_name, "from_dir", read_from_dir) <
      0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_stencil_hdf5: Attribute \"from_dir\" not found in "
          "halo"
       << halo_name;
    throw ex;
  }
  // get the to_dir
  int read_to_dir[dim];
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
 * Routine to read an ops_dat from an hdf5 file - only reads the meta data of
 * the ops_dat the actual data is read later from within ops_partition()
 *******************************************************************************/
ops_dat ops_decl_dat_hdf5(ops_block block, int dat_dim, char const *type,
                          char const *dat_name, char const *file_name) {
  ops_read_dat_hdf5_dynamic = ops_read_dat_hdf5;

  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t group_id; // group identifier
  hid_t plist_id; // property list identifier

  // open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    MPI_Barrier(OPS_MPI_GLOBAL);
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: file " << file_name << " does not exist";
    throw ex;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
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
          "set "
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
  char read_type[15];
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

  int read_size[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "size", read_size) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"size\" not found in data set "
       << dat_name;
    throw ex;
  }

  int read_d_m[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "d_m", read_d_m) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"d_m\" not found in data set "
       << dat_name;
    throw ex;
  }

  int read_d_p[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "d_p", read_d_p) < 0) {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: ops_decl_dat_hdf5: Attribute \"d_p\" not found in data set "
       << dat_name;
    throw ex;
  }

  int read_base[block->dims];
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
  else if (strcmp(read_type, "short") == 0)
    type_size = sizeof(short);
  else if (strcmp(read_type, "char") == 0)
    type_size = sizeof(char);
  else {
    OPSException ex(OPS_HDF5_ERROR);
    ex << "Error: Unknown type in ops_decl_dat_hdf5(): " << read_type;
    throw ex;
  }

  char *data = NULL;

  int stride[] = {1, 1, 1, 1, 1};
  ops_dat created_dat = ops_decl_dat_char(
      block, dat_dim, read_size /*global dat size in each dimension*/,
      read_base, read_d_m, read_d_p, stride, data /*null for now*/,
      type_size /*size of(type)*/, type, dat_name); // TODO: multigrid stride

  created_dat->is_hdf5 = 1;
  created_dat->hdf5_file = copy_str(file_name);
  created_dat->user_managed = 0;

  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);

  return created_dat;
  /**
    When ops_decomp_dats are encountered read the correct hyperslab chunk form
    hdf5 file and pad the data with the correct mpi-halo depths
    then attache it to the ops_dat->data of this ops_dat
   **/
}

/*******************************************************************************
 * Routine to do delayed read of data within ops_partition() from an hdf5 file
 * only used with the MPI backends
 *******************************************************************************/
void ops_read_dat_hdf5(ops_dat dat) {
  sub_block *sb = OPS_sub_block_list[dat->block->index];
  if (sb->owned == 1) {
    // compute the number of elements that this process will read from file
    // also compute the correct offsets on the file that this process should
    // begin from to read
    sub_dat *sd = OPS_sub_dat_list[dat->index];
    ops_block block = dat->block;
    sub_block *sb = OPS_sub_block_list[dat->block->index];

    hsize_t disp[block->dims]; // global disps to compute the chunk data set
    // dimensions
    hsize_t l_disp[block->dims]; // local disps to remove MPI halos
    hsize_t size[block->dims];   // local size to compute the chunk data set
    // dimensions
    hsize_t size2[block->dims];    // local size - stored for later use
    hsize_t gbl_size[block->dims]; // global size to compute the chunk data set
    // dimensions

    // int g_d_m[block->dims]; // global size of the block halo (-) depth
    // attribute
    // to read from hdf5 file
    // int g_d_p[block->dims]; // global size of the block halo (+) depth
    // attribute
    // to read from hdf5 file

    hsize_t count[block->dims];  // parameters for for hdf5 file chuck reading
    hsize_t stride[block->dims]; // parameters for for hdf5 file chuck reading

    for (int d = 0; d < block->dims; d++) {
      // remove left MPI halo to get start disp from beginning of dat
      // include left block halo
      disp[d] = sd->decomp_disp[d] -
                sd->gbl_d_m[d];    // global displacements of the data set
      l_disp[d] = 0 - sd->d_im[d]; // local displacements of the data set (i.e.
      // per MPI proc)
      size[d] = sd->decomp_size[d]; // local size to compute the chunk data set
      // dimensions
      size2[d] = sd->decomp_size[d]; // local size - stored for later use
      gbl_size[d] = sd->gbl_size[d]; // global size to compute the chunk data
      // set dimensions

      /*g_d_m[d] =
        sd->gbl_d_m[d]; // global halo depth(-) to be read from hdf5 file
      g_d_p[d] =
        sd->gbl_d_p[d]; // global halo depth(+) to be read from hdf5 file
      */

      count[d] = 1;
      stride[d] = 1;

      // printf("l_disp[%d] = %d ",d,l_disp[d]);
      // printf("disp[%d] = %d ",d,disp[d]);
      // printf("size[%d] = %d ",d,size[d]);
      // printf("dat->size[%d] = %d ",d,dat->size[d]);
      // printf("gbl_size[%d] = %d ",d,gbl_size[d]);
      // printf("dat->d_m[%d] = %d ",d,g_d_m[d]);
      // printf("dat->d_p[%d] = %d ",d,g_d_p[d]);
    }

    hsize_t t_size = 1;
    for (int d = 0; d < dat->block->dims; d++)
      t_size *= size[d];
    char *data = (char *)ops_malloc(t_size * dat->elem_size);
    dat->mem = t_size * dat->elem_size;

    // make sure we multiply by the number of
    // data values per element (i.e. dat->dim) to get full size of the data
    size[0] = size[0] * dat->dim;
    disp[0] = disp[0] * dat->dim;

    if (block->dims == 1)
      gbl_size[0] = gbl_size[0] * dat->dim; //-- this needs to be tested for 1D
    else if (block->dims == 2)
      // Jianping Meng: It looks that growing the zeroth index  is better
      gbl_size[0] = gbl_size[0] *
                    dat->dim; //**note we are using [1] instead of [0] here !!
    else if (block->dims == 3)
      gbl_size[0] =
          gbl_size[0] * dat->dim; //**note that for 3D we are using [0] here !!

    // create new communicator
    int my_rank, comm_size;
    // use the communicator for MPI procs holding this block
    MPI_Comm_dup(sb->comm1, &OPS_MPI_HDF5_WORLD);
    MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
    MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

    // MPI variables
    MPI_Info info = MPI_INFO_NULL;

    // HDF5 APIs definitions
    hid_t file_id;   // file identifier
    hid_t group_id;  // file identifier
    hid_t dset_id;   // dataset identifier
    hid_t filespace; // data space identifier
    hid_t plist_id;  // property list identifier
    hid_t memspace;  // memory space identifier

    // open given hdf5 file .. if it exists
    if (file_exist(dat->hdf5_file) == 0) {
      MPI_Barrier(OPS_MPI_GLOBAL);
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: ops_read_dat_hdf5: file " << dat->hdf5_file
         << " does not exist";
      throw ex;
    }

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
    file_id = H5Fopen(dat->hdf5_file, H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);

    if (H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: Error: ops_decl_dat_hdf5: ops_block on which this ops_dat "
         << dat->name << " is declared does not exist in the file";
      throw ex;
    }

    // open existing group -- an ops_block is a group
    group_id = H5Gopen(file_id, block->name, H5P_DEFAULT);

    // check if ops_dat exists
    if (H5Lexists(group_id, dat->name, H5P_DEFAULT) == 0) {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: Error: ops_decl_dat_hdf5: ops_dat " << dat->name
         << " does not exist in block " << block->name;
      throw ex;
    }

    dset_id = H5Dopen(group_id, dat->name, H5P_DEFAULT);

    /*
    hsize_t GBL_SIZE[block->dims];
    if (block->dims == 1) {
      GBL_SIZE[0] = gbl_size[0];
    } else if (block->dims == 2) {
      GBL_SIZE[0] = gbl_size[1];
      GBL_SIZE[1] = gbl_size[0];
    } else if (block->dims == 3) {
      GBL_SIZE[0] = gbl_size[2];
      GBL_SIZE[1] = gbl_size[1];
      GBL_SIZE[2] = gbl_size[0];
    }
    */

    // Need to flip the dimensions to accurately read from HDF5 chunk
    // decomposition
    hsize_t DISP[block->dims];
    hsize_t SIZE[block->dims];
    if (block->dims == 2) {
      DISP[0] = disp[1];
      DISP[1] = disp[0];
      SIZE[0] = size[1];
      SIZE[1] = size[0];
    } else if (block->dims == 3) {
      DISP[0] = disp[2];
      DISP[1] = disp[1]; // note how dimension 1 remains the same !!
      DISP[2] = disp[0];
      SIZE[0] = size[2];
      SIZE[1] = size[1]; // note how dimension 1 remains the same !!
      SIZE[2] = size[0];
    }

    memspace = H5Screate_simple(
        block->dims, size,
        NULL); // block of memory to read from file by each proc

    // Select hyperslab
    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, DISP, stride, count, SIZE);

    // Create property list for collective dataset read.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // read data
    if (strcmp(dat->type, "double") == 0)
      H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data);
    else if (strcmp(dat->type, "float") == 0)
      H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data);
    else if (strcmp(dat->type, "int") == 0)
      H5Dread(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, data);
    else if (strcmp(dat->type, "long") == 0)
      H5Dread(dset_id, H5T_NATIVE_LONG, memspace, filespace, plist_id, data);
    else if ((strcmp(dat->type, "long long") == 0) ||
             (strcmp(dat->type, "ll") == 0))
      H5Dread(dset_id, H5T_NATIVE_LLONG, memspace, filespace, plist_id, data);
    else if (strcmp(dat->type, "char") == 0)
      H5Dread(dset_id, H5T_NATIVE_CHAR, memspace, filespace, plist_id, data);
    else if (strcmp(dat->type, "short") == 0)
      H5Dread(dset_id, H5T_NATIVE_SHORT, memspace, filespace, plist_id, data);
    else {
      OPSException ex(OPS_HDF5_ERROR);
      ex << "Error: Unknown type in ops_read_dat_hdf5(): " << dat->type;
      throw ex;
    }

    // add MPI halos
    if (block->dims == 2)
      add_mpi_halos2D(dat, size2, l_disp, data);
    else if (block->dims == 3)
      add_mpi_halos3D(dat, size2, l_disp, data);
    else if (block->dims == 4)
      add_mpi_halos4D(dat, size2, l_disp, data);
    else if (block->dims == 5)
      add_mpi_halos5D(dat, size2, l_disp, data);

    free(data);
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Dclose(dset_id);
    H5Sclose(memspace);
    H5Gclose(group_id);
    H5Fclose(file_id);

    MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
  }
  return;
}

/*******************************************************************************
 * Routine to dump all ops_blocks, ops_dats etc to a named
 * HDF5 file
 *******************************************************************************/
// --- This routine is identical to the sequential routine in ops_hdf5.c
void ops_dump_to_hdf5(char const *file_name) {
  ops_dat_entry *item;
  for (int n = 0; n < OPS_instance::getOPSInstance()->OPS_block_index; n++) {
    printf("Dumping block %15s to HDF5 file %s\n",
           OPS_instance::getOPSInstance()->OPS_block_list[n].block->name,
           file_name);
    ops_fetch_block_hdf5_file(
        OPS_instance::getOPSInstance()->OPS_block_list[n].block, file_name);
  }

  TAILQ_FOREACH(item, &OPS_instance::getOPSInstance()->OPS_dat_list, entries) {
    printf("Dumping dat %15s to HDF5 file %s\n", (item->dat)->name, file_name);
    if (item->dat->e_dat !=
        1) // currently cannot write edge dats .. need to fix this
      ops_fetch_dat_hdf5_file(item->dat, file_name);
  }

  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_stencil_index; i++) {
    printf("Dumping stencil %15s to HDF5 file %s\n",
           OPS_instance::getOPSInstance()->OPS_stencil_list[i]->name,
           file_name);
    ops_fetch_stencil_hdf5_file(
        OPS_instance::getOPSInstance()->OPS_stencil_list[i], file_name);
  }

  printf("halo index = %d \n", OPS_instance::getOPSInstance()->OPS_halo_index);
  for (int i = 0; i < OPS_instance::getOPSInstance()->OPS_halo_index; i++) {
    printf("Dumping halo %15s--%15s to HDF5 file %s\n",
           OPS_instance::getOPSInstance()->OPS_halo_list[i]->from->name,
           OPS_instance::getOPSInstance()->OPS_halo_list[i]->to->name,
           file_name);
    ops_fetch_halo_hdf5_file(OPS_instance::getOPSInstance()->OPS_halo_list[i],
                             file_name);
  }
}

/*******************************************************************************
 * Routine to copy over an ops_dat to a user specified memory pointer
 *******************************************************************************/
extern "C" char *ops_fetch_dat_char(ops_dat dat, char *u_dat) {
  sub_block *sb = OPS_sub_block_list[dat->block->index];
  if (sb->owned == 1) {
    // fetch data onto the host ( if needed ) based on the backend
    ops_get_data(dat);

    // compute the number of elements that this process will copy over to the
    // user space
    sub_dat *sd = OPS_sub_dat_list[dat->index];
    ops_block block = dat->block;

    hsize_t l_disp[block->dims]; // local disps to remove MPI halos
    hsize_t size[block->dims];   // local size to compute the chunk data set
    // dimensions

    for (int d = 0; d < block->dims; d++) {
      l_disp[d] = 0 - sd->d_im[d]; // local displacements of the data set (i.e.
      // per MPI proc)
      size[d] = sd->decomp_size[d]; // local size to compute the chunk data set
      // dimensions
    }

    hsize_t t_size = 1;
    for (int d = 0; d < dat->block->dims; d++)
      t_size *= size[d];
    u_dat = (char *)ops_malloc(t_size * dat->elem_size);

    // create new communicator
    int my_rank, comm_size;
    // use the communicator for MPI procs holding this block
    MPI_Comm_dup(sb->comm1, &OPS_MPI_HDF5_WORLD);
    MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
    MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

    if (block->dims == 1)
      remove_mpi_halos1D(dat, size, l_disp, u_dat);
    else if (block->dims == 2)
      remove_mpi_halos2D(dat, size, l_disp, u_dat);
    else if (block->dims == 3)
      remove_mpi_halos3D(dat, size, l_disp, u_dat);
    else if (block->dims == 4)
      remove_mpi_halos4D(dat, size, l_disp, u_dat);
    else if (block->dims == 5)
      remove_mpi_halos5D(dat, size, l_disp, u_dat);
  }
  return u_dat;
}

typedef struct {
  const char *type_str; // dataset type as string
  hsize_t size;         // dataset size (first dimension)
  hsize_t dim;          // element size (second dimension)
  size_t elem_bytes;    // element byte-size
} ops_hdf5_dataset_properties;

const char *ops_hdf5_type_to_string(hid_t t) {
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
  } else {
    text = (char *)malloc(13 * sizeof(char));
    strcpy(text, "UNRECOGNISED");
  }

  return (const char *)text;
}

herr_t get_dataset_properties(hid_t dset_id,
                              ops_hdf5_dataset_properties *dset_props) {
  hid_t status;

  if (dset_props == NULL) {
    return -1;
  }

  // Get dimension and size:
  hid_t dataspace = H5Dget_space(dset_id);
  if (dataspace < 0) {
    return -1;
  }
  int ndims = H5Sget_simple_extent_ndims(dataspace);
  if (ndims == 0) {
    dset_props->size = 0;
    dset_props->dim = 0;
    H5Sclose(dataspace);
  } else {
    hsize_t dims[ndims];
    hsize_t maxdims[ndims];
    status = H5Sget_simple_extent_dims(dataspace, dims, maxdims);
    H5Sclose(dataspace);
    if (status < 0) {
      return -1;
    }
    dset_props->size = dims[0];
    dset_props->dim = (ndims > 1) ? dims[1] : 1;
  }

  // Get type information:
  hid_t t = H5Dget_type(dset_id);
  if (t < 0) {
    return -1;
  }
  dset_props->type_str = ops_hdf5_type_to_string(t);
  if (H5Tequal(t, H5T_NATIVE_INT)) {
    dset_props->elem_bytes = sizeof(int);
  } else if (H5Tequal(t, H5T_NATIVE_LONG)) {
    dset_props->elem_bytes = sizeof(long);
  } else if (H5Tequal(t, H5T_NATIVE_LLONG)) {
    dset_props->elem_bytes = sizeof(long long);
  } else if (H5Tequal(t, H5T_NATIVE_FLOAT)) {
    dset_props->elem_bytes = sizeof(float);
  } else if (H5Tequal(t, H5T_NATIVE_DOUBLE)) {
    dset_props->elem_bytes = sizeof(double);
  } else if (H5Tequal(t, H5T_NATIVE_CHAR)) {
    dset_props->elem_bytes = sizeof(char);
  } else {
    size_t name_len = H5Iget_name(dset_id, NULL, 0);
    char name[name_len];
    H5Iget_name(dset_id, name, name_len + 1);
    ops_printf("Error: Do not recognise type of dataset '%s'\n", name);
    exit(2);
  }
  dset_props->elem_bytes *= dset_props->dim;

  return 0;
}

/*******************************************************************************
 * Routine to read in a constant from a named hdf5 file
 *******************************************************************************/
void ops_get_const_hdf5(char const *name, int dim, char const *type,
                        char *const_data, char const *file_name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier
  hid_t dset_id;  // dataset identifier
  hid_t status;

  if (file_exist(file_name) == 0) {
    ops_printf("File %s does not exist .... aborting ops_get_const_hdf5()\n",
               file_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  file_id = H5Fopen(file_name, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  // find dimension of this constant with available attributes
  int const_dim = 0;
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    ops_printf("dataset with '%s' not found in file '%s' \n", name, file_name);
    H5Fclose(file_id);
    const_data = NULL;
    return;
  }

  // Create property list for collective dataset read.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  ops_hdf5_dataset_properties dset_props;
  status = get_dataset_properties(dset_id, &dset_props);
  if (status < 0) {
    ops_printf("Could not get properties of dataset '%s' in file '%s'\n", name,
               file_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  const_dim = dset_props.size;
  if (const_dim != dim) {
    ops_printf(
        "dim of constant %d in file %s and requested dim %d do not match\n",
        const_dim, file_name, dim);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  const char *typ = dset_props.type_str;
  if (strcmp(typ, type) != 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 1)
      ops_printf(
          "type of constant %s in file %s and requested type %s do not match, "
          "performing automatic type conversion\n",
          typ, file_name, type);
    typ = type;
  }
  H5Dclose(dset_id);

  // open the dataset with default properties
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

  char *data;
  // initialize data buffer and read data
  if (strcmp(typ, "int") == 0 || strcmp(typ, "int(4)") == 0 ||
      strcmp(typ, "integer") == 0 || strcmp(typ, "integer(4)") == 0) {
    data = (char *)xmalloc(sizeof(int) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(int) * const_dim);
  } else if (strcmp(typ, "long") == 0) {
    data = (char *)xmalloc(sizeof(long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(long) * const_dim);
  } else if (strcmp(typ, "long long") == 0) {
    data = (char *)xmalloc(sizeof(long long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(long long) * const_dim);
  } else if (strcmp(typ, "float") == 0 || strcmp(typ, "real(4)") == 0 ||
             strcmp(typ, "real") == 0) {
    data = (char *)xmalloc(sizeof(float) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(float) * const_dim);
  } else if (strcmp(typ, "double") == 0 ||
             strcmp(typ, "double precision") == 0 ||
             strcmp(typ, "real(8)") == 0) {
    data = (char *)xmalloc(sizeof(double) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(double) * const_dim);
  } else if (strcmp(typ, "char") == 0) {
    data = (char *)xmalloc(sizeof(char) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(char) * const_dim);
  } else {
    ops_printf("Unknown type in file %s for constant %s\n", file_name, name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  free(data);

  free((char *)dset_props.type_str);

  H5Pclose(plist_id);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
}

/*******************************************************************************
 * Routine to write a constant to a named hdf5 file
 *******************************************************************************/
void ops_write_const_hdf5(char const *name, int dim, char const *type,
                          char *const_data, char const *file_name) {
  // letting know that writing is happening ...
  // ops_printf("Writing '%s' to file '%s'\n", name, file_name);

  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OPS_MPI_GLOBAL, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t plist_id;  // property list identifier
  hid_t dataspace; // data space identifier
  htri_t status;   // status for checking return values

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 3) {
      ops_printf("File %s does not exist .... creating file\n", file_name);
    }
    file_id = H5Fcreate(file_name, H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  /* Open the existing file. */
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  // Check if const already exists in data set
  status = H5Lexists(file_id, name, H5P_DEFAULT);
  if (status > 0) {
    ops_printf("Const dataset %s already found in file %s ... ", name,
               file_name);

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

    // open existing data set
    dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

    // find dimension of this constant with available attributes
    int const_dim = 0;

    // Create property list for collective dataset read/write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    ops_hdf5_dataset_properties dset_props;
    status = get_dataset_properties(dset_id, &dset_props);
    if (status < 0) {
      ops_printf("Could not get properties of dataset '%s' in file '%s'\n",
                 name, file_name);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }

    const_dim = dset_props.size;
    if (const_dim != dim) {
      ops_printf(
          "dim of constant %d in file %s and requested dim %d do not match\n",
          const_dim, file_name, dim);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }

    const char *typ = dset_props.type_str;
    if (strcmp(typ, type) != 0) {
      if (OPS_instance::getOPSInstance()->OPS_diags > 1)
        ops_printf("type of constant %s in file %s and requested type %s do "
                   "not match, performing automatic type conversion\n",
                   typ, file_name, type);
      typ = type;
    }

    // existing const attributes matches with const to be written .. overwriting
    ops_printf(" overwriting\n");

    dataspace = H5Dget_space(dset_id);

    // Write to the exisiting dataset with default properties
    if (strcmp(type, "double") == 0 || strcmp(type, "double precision") == 0 ||
        strcmp(type, "real(8)") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, plist_id,
               const_data);
    } else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0 ||
               strcmp(type, "real") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, plist_id,
               const_data);
    } else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 ||
               strcmp(type, "integer") == 0 ||
               strcmp(type, "integer(4)") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, plist_id,
               const_data);
    } else if ((strcmp(type, "long") == 0)) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, plist_id,
               const_data);
    } else if ((strcmp(type, "long long") == 0)) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, plist_id,
               const_data);
    } else if (strcmp(type, "char") == 0) {
      // write data
      H5Dwrite(dset_id, H5T_NATIVE_CHAR, H5S_ALL, dataspace, plist_id,
               const_data);
    }

    H5Pclose(plist_id);
    H5Sclose(dataspace);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
    return;
  }

  if (OPS_instance::getOPSInstance()->OPS_diags > 2)
    ops_printf(
        "Const dataset '%s' not found in file '%s ... creating const' \n", name,
        file_name);

  // Create the dataspace for the dataset.
  hsize_t dims_of_const = {dim};
  dataspace = H5Screate_simple(1, &dims_of_const, NULL);

  // Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // Create the dataset with default properties
  if (strcmp(type, "double") == 0 || strcmp(type, "double precision") == 0 ||
      strcmp(type, "real(8)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_DOUBLE, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 ||
             strcmp(type, "integer") == 0 || strcmp(type, "integer(4)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_INT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, plist_id, const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long long") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LLONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "char") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_CHAR, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_CHAR, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else {
    ops_printf(
        "Unknown type %s for constant %s: cannot write constant to file\n",
        type, name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  H5Pclose(plist_id);
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
    ops_printf(
        "Unknown type %s for constant %s: cannot write constant to file\n",
        type, name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  H5Aclose(attribute);
  H5Sclose(dataspace);
  H5Dclose(dset_id);

  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
}

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

void determine_global_range(const ops_dat &dat, const int cross_section_dir,
                            const int pos, int *range) {
  const int space_dim{dat->block->dims};
  const sub_dat *sd = OPS_sub_dat_list[dat->index];
  int *size{new int(space_dim)};
  for (int d = 0; d < space_dim; d++) {
    size[d] = sd->gbl_size[d] - (sd->gbl_d_p[d] - sd->gbl_d_m[d]);
  }

  for (int d = 0; d < dat->block->dims; d++) {
    range[2 * d] = 0;
    range[2 * d + 1] = size[d];
  }
  range[2 * cross_section_dir + 1] = pos + 1;
  range[2 * cross_section_dir] = pos;
  delete size;
}

void determine_local_range(const ops_dat &dat, int *global_range,
                           int *local_range) {
  ops_arg *dat_arg;
  const int space_dim{dat->block->dims};
  if (space_dim == 3) {
    int s3D_000[]{0, 0, 0};
    ops_stencil S3D_000{ops_decl_stencil(3, 1, s3D_000, "000")};
    ops_arg arg = ops_arg_dat(dat, dat->dim, S3D_000, dat->type, OPS_READ);
    dat_arg = &arg;
  }

  if (space_dim == 2) {
    int s2D_000[]{0, 0, 0};
    ops_stencil S2D_000{ops_decl_stencil(2, 1, s2D_000, "000")};

    ops_arg arg = ops_arg_dat(dat, dat->dim, S2D_000, dat->type, OPS_READ);
    dat_arg = &arg;
  }

  int *arg_idx{new int(space_dim)};

  int *local_start{new int(space_dim)};
  int *local_end{new int(space_dim)};
  if (compute_ranges(dat_arg, 1, dat->block, global_range, local_start,
                     local_end, arg_idx) < 0) {
    return;
  }
  for (int i = 0; i < space_dim; i++) {
    local_range[2 * i] = local_start[i];
    local_range[2 * i + 1] = local_end[i];
  }
  delete arg_idx;
  delete local_start;
  delete local_end;
}

template <int dir>
void copy_loop_slab(char *dest, char *src, const int *dest_size,
                    const int *src_size, const int *d_m, const int elem_size,
                    const int *range_max_dim) {
  // TODO: add OpenMP here if needed
#if OPS_MAX_DIM > 4
  for (int m = 0; m < dest_size[4]; m++) {
    size_t moff_dest =
        m * dest_size[0] * dest_size[1] * dest_size[2] * dest_size[3];
    size_t moff_src = (range_max_dim[2 * 4] + m - d_m[4]) * src_size[3] *
                      src_size[2] * src_size[1] * src_size[0];
#else
  size_t moff_dest = 0;
  size_t moff_src = 0;
#endif
#if OPS_MAX_DIM > 3
    for (int l = 0; l < dest_size[3]; l++) {
      size_t loff_dest = l * dest_size[0] * dest_size[1] * dest_size[2];
      size_t loff_src = (range_max_dim[2 * 3] + l - d_m[3]) * src_size[2] *
                        src_size[1] * src_size[0];
#else
  size_t loff_dest = 0;
  size_t loff_src = 0;
#endif
      for (int k = 0; k < dest_size[2]; k++)
        for (int j = 0; j < dest_size[1]; j++)
          if (dir == 0) {
            memcpy(&dest[moff_dest + loff_dest +
                         k * dest_size[0] * dest_size[1] + j * dest_size[0]],
                   &src[(moff_src + loff_src +
                         (range_max_dim[2 * 2] + k - d_m[2]) * src_size[1] *
                             src_size[0] +
                         (range_max_dim[2 * 1] + j - d_m[1]) * src_size[0] +
                         range_max_dim[2 * 0] - d_m[0]) *
                        elem_size],
                   dest_size[0]);
          } else {
            memcpy(&src[(moff_src + loff_src +
                         (range_max_dim[2 * 2] + k - d_m[2]) * src_size[1] *
                             src_size[0] +
                         (range_max_dim[2 * 1] + j - d_m[1]) * src_size[0] +
                         range_max_dim[2 * 0] - d_m[0]) *
                        elem_size],
                   &dest[moff_dest + loff_dest +
                         k * dest_size[0] * dest_size[1] + j * dest_size[0]],
                   dest_size[0]);
          }

#if OPS_MAX_DIM > 3
    }
#endif
#if OPS_MAX_DIM > 4
  }
#endif
}

// create a h5 file or open a h5 file if existing
hid_t H5_file_handle(const MPI_Comm &mpi_comm, const char *file_name) {
  int my_rank;
  MPI_Comm_rank(mpi_comm, &my_rank);
  MPI_Info info = MPI_INFO_NULL;
  hid_t file_plist_id{H5Pcreate(H5P_FILE_ACCESS)};
  H5Pset_fapl_mpio(file_plist_id, mpi_comm, info);
  if (file_exist(file_name) == 0) {
    if (OPS_instance::getOPSInstance()->OPS_diags > 3)
      ops_printf("File %s does not exist .... creating file\n", file_name);
    if (my_rank == 0) {
      FILE *fp;
      fp = fopen(file_name, "w");
      fclose(fp);
    }
    // Create a new file collectively and release property list identifier.
    hid_t file_id{
        H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, file_plist_id)};
    H5Fclose(file_id);
  }

  hid_t file_id{H5Fopen(file_name, H5F_ACC_RDWR, file_plist_id)};
  H5Pclose(file_plist_id);
  return file_id;
}

// create the dataset or open the dataset if existing
void H5_dataset_space(const hid_t file_id, const int data_dims,
                      const hsize_t *global_data_size, const char *data_name,
                      const char *data_type, hid_t &dataset_id,
                      hid_t &file_space) {

  if (H5Lexists(file_id, data_name, H5P_DEFAULT) == 0) {
    hid_t data_plist_id{H5Pcreate(H5P_DATASET_CREATE)};
    file_space = H5Screate_simple(data_dims, global_data_size, NULL);
    dataset_id = H5Dcreate(file_id, data_name, h5_type(data_type), file_space,
                           H5P_DEFAULT, data_plist_id, H5P_DEFAULT);
    H5Pclose(data_plist_id);
  } else {
    dataset_id = H5Dopen(file_id, data_name, H5P_DEFAULT);
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

void copy_data_buf(const ops_dat &dat, const int *local_range,
                   char *local_buf) {
  const sub_dat *sd = OPS_sub_dat_list[dat->index];
  ops_execute(dat->block->instance);
  ops_get_data(dat);
  int local_buf_size[OPS_MAX_DIM] = {1};
  int range_max_dim[2 * OPS_MAX_DIM] = {0};
  int d_m[OPS_MAX_DIM]{0};
  for (int d = 0; d < dat->block->dims; d++) {
    local_buf_size[d] = local_range[2 * d + 1] - local_range[2 * d + 0];
    range_max_dim[2 * d] = local_range[2 * d];
    range_max_dim[2 * d + 1] = local_range[2 * d + 1];
  }
  for (int d = dat->block->dims; d < OPS_MAX_DIM; d++) {
    local_buf_size[d] = 1;
    range_max_dim[2 * d] = 0;
    range_max_dim[2 * d + 1] = 1;
  }

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    d_m[d] = sd->d_im[d] + dat->d_m[d];
    // printf("At rank %d d_im[%d]=%d d_m[%d]=%d\n", ops_my_global_rank, d,
    //        sd->d_im[d], d, dat->d_m[d]);
  }
  local_buf_size[0] *= dat->elem_size;

  if (dat->block->dims > 5)
    throw OPSException(OPS_NOT_IMPLEMENTED,
                       "Error, missing OPS implementation: ops_dat_fetch_data "
                       "not implemented for dims>5");
  if (dat->block->instance->OPS_soa && dat->dim > 1)
    throw OPSException(OPS_NOT_IMPLEMENTED,
                       "Error, missing OPS implementation: ops_dat_fetch_data "
                       "not implemented for SoA");
  copy_loop_slab<0>(local_buf, dat->data, local_buf_size, dat->size, d_m,
                    dat->elem_size, range_max_dim);
  dat->dirty_hd = 1;
}

void write_buf_hdf5(const char *file_name, const char *data_name,
                    const ops_dat &dat, const int cross_section_dir,
                    const int *local_range, const int *global_range,
                    const char *buf) {
  const sub_block *sb{OPS_sub_block_list[dat->block->index]};
  int my_block_rank;
  MPI_Comm PLANE_WORLD;
  MPI_Comm BLOCK_WORLD{sb->comm1};
  MPI_Comm_rank(BLOCK_WORLD, &my_block_rank);
  int color{0};
  if ((local_range[2 * cross_section_dir + 1] -
       local_range[2 * cross_section_dir]) == 0) {
    color = MPI_UNDEFINED;
  }
  MPI_Comm_split(BLOCK_WORLD, color, my_block_rank, &PLANE_WORLD);
  if (color == 0) {
    int my_plane_rank;
    MPI_Comm_rank(PLANE_WORLD, &my_plane_rank);
    MPI_Barrier(PLANE_WORLD);
    const int space_dim{dat->block->dims};
    const int data_dims{space_dim - 1};
    const sub_dat *sd = OPS_sub_dat_list[dat->index];
    hsize_t *local_data_size_c{new hsize_t(data_dims)};
    hsize_t *local_data_size_f{new hsize_t(data_dims)};
    hsize_t *global_data_size_c{new hsize_t(data_dims)};
    hsize_t *global_data_size_f{new hsize_t(data_dims)};
    hsize_t *global_data_disp_c{new hsize_t(data_dims)};
    hsize_t *global_data_disp_f{new hsize_t(data_dims)};

    {
      int reduced_index{0};
      for (int d = 0; d < space_dim; d++) {
        if (d != cross_section_dir) {
          local_data_size_c[reduced_index] =
              local_range[2 * d + 1] - local_range[2 * d];
          global_data_size_c[reduced_index] =
              global_range[2 * d + 1] - global_range[2 * d];
          global_data_disp_c[reduced_index] =
              sd->decomp_disp[d] < 0 ? 0 : sd->decomp_disp[d];
          reduced_index++;
        }
      }
    }

    for (int d = 0; d < data_dims; d++) {
      local_data_size_f[d] = local_data_size_c[data_dims - d - 1];
      global_data_size_f[d] = global_data_size_c[data_dims - d - 1];
      global_data_disp_f[d] = global_data_disp_c[data_dims - d - 1];
    }

    hid_t file_id{H5_file_handle(PLANE_WORLD, file_name)};

    // space in file

    hid_t file_space;

    hid_t dataset_id;

    H5_dataset_space(file_id, data_dims, global_data_size_f, data_name,
                     dat->type, dataset_id, file_space);

    // block of memory to write to file by each proc
    hid_t memspace{H5Screate_simple(data_dims, local_data_size_f, NULL)};
    hsize_t *stride{new hsize_t(data_dims)};
    hsize_t *count{new hsize_t(data_dims)};
    for (int d = 0; d < data_dims; d++) {
      stride[d] = 1;
      count[d] = 1;
    }
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, global_data_disp_f, stride,
                        count, local_data_size_f);
    // Create property list for collective dataset write.
    hid_t xfer_data_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_data_plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dataset_id, h5_type(dat->type), memspace, file_space,
             xfer_data_plist_id, buf);
    H5Pclose(xfer_data_plist_id);
    H5Sclose(file_space);
    H5Sclose(memspace);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    delete count;
    delete stride;

    MPI_Barrier(PLANE_WORLD);
    delete local_data_size_c;
    delete local_data_size_f;
    delete global_data_size_c;
    delete global_data_size_f;
    delete global_data_disp_c;
    delete global_data_disp_f;
    MPI_Comm_free(&PLANE_WORLD);
  }
}

void ops_write_dataslice_hdf5(char const *file_name, const char *data_name,
                              const ops_dat &dat, const int cross_section_dir,
                              const int pos) {
  sub_block *sb = OPS_sub_block_list[dat->block->index];
  if (sb->owned == 1) {
    const int space_dim{dat->block->dims};
    int *global_range{new int(space_dim)};
    determine_global_range(dat, cross_section_dir, pos, global_range);
    int *local_range{new int(2 * space_dim)};
    // TODO if the plane is out of global range, computer range will generate
    // error
    determine_local_range(dat, global_range, local_range);
    size_t local_buf_size{dat->elem_size};
    for (int i = 0; i < space_dim; i++) {
      local_buf_size *= (local_range[2 * i + 1] - local_range[2 * i]);
    }
    // if (local_buf_size > 0) {
    char *local_buf = (char *)ops_malloc(local_buf_size);
    copy_data_buf(dat, local_range, local_buf);
    // if (local_buf_size>1){
    //   double *data_p{(double *)local_buf};
    //   printf("At rank %d data= %f %f %f %f\n", ops_my_global_rank, data_p[0],
    //          data_p[1], data_p[2], data_p[3]);
    // }
    write_buf_hdf5(file_name, data_name, dat, cross_section_dir, local_range,
                   global_range, local_buf);
    free(local_buf);
    //}
    delete global_range;
    delete local_range;
  }
}

void ops_write_slice_group_hdf5(
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
        if ((cross_section_dir >= 0) &&
            (cross_section_dir <= data->block->dims)) {
          const sub_dat *sd{OPS_sub_dat_list[data->index]};
          if ((pos >= sd->gbl_base[cross_section_dir]) &&
              (pos <= sd->gbl_size[cross_section_dir])) {
            std::string block_name{data->block->name};
            std::string data_name{data->name};
            std::string file_name{plane_names[p] + ".h5"};
            std::string dataset_name{block_name + "_" + data_name + "_" + key};
            ops_write_dataslice_hdf5(file_name.c_str(), dataset_name.c_str(),
                                     data, cross_section_dir, pos);
          } else {
            ops_printf("The dat %s doesn't have the specified plane %s = %d \n",
                       data->name, plane_name_base[cross_section_dir], pos);
          }
        } else {
          ops_printf(
              "The block %s doesn't have the specified cross section direction "
              "%d\n",
              data->block->name, cross_section_dir);
        }
      }
    }
  }
}

void ops_write_slice_group_hdf5(
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
  ops_write_slice_group_hdf5(planes, plane_names, key, data_list);
}
