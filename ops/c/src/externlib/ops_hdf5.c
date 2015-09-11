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

/** @brief HDF5 file I/O backend implementation for none-MPI parallelisations
  * @author Gihan Mudalige (started 28-08-2015)
  * @details Implements the OPS API calls for the HDF5 file I/O functionality
  */

#include <math.h>


// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

//hdf5 header
#include <hdf5.h>
#include <hdf5_hl.h>

#include <ops_lib_core.h>
#include <ops_util.h>

/*******************************************************************************
* Routine to write an ops_block to a named hdf5 file,
* if file does not exist, creates it
* if the block does not exists in file creates block
*******************************************************************************/

void ops_fetch_block_hdf5_file(ops_block block, char const *file_name) {

	//HDF5 APIs definitions
	hid_t file_id;      //file identifier
	hid_t group_id;      //group identifier
	hid_t dset_id;      //dataset identifier
	hid_t filespace;    //data space identifier
	hid_t plist_id;     //property list identifier
	hid_t memspace;     //memory space identifier
	hid_t attr;         //attribute identifier
	herr_t err;         //error code

	//Set up file access property list for I/O
	plist_id = H5Pcreate(H5P_FILE_ACCESS);

	if (file_exist(file_name) == 0) {
	  ops_printf("File %s does not exist .... creating file\n", file_name);
	  FILE *fp; fp = fopen(file_name, "w");
	  fclose(fp);
	  //Create a new file
	  file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	  H5Fclose(file_id);
	}

	file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

	if(H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
	  ops_printf("ops_block %s does not exists in file %s ... creating ops_block\n",
	    block->name, file_name);
	    //create group - ops_block
	    group_id = H5Gcreate(file_id, block->name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	    H5Gclose(group_id);
	}

	//open existing group -- an ops_block is a group
	group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);

	//attach attributes to block
	H5LTset_attribute_string(file_id, block->name, "ops_type", "ops_block"); //ops type
	H5LTset_attribute_int(file_id, block->name, "dims", &(block->dims), 1); //dim
	H5LTset_attribute_int(file_id, block->name, "index", &(block->index), 1); //index

	H5Gclose(group_id);
	H5Pclose(plist_id);
	H5Fclose(file_id);
}

/*******************************************************************************
* Routine to write an ops_block to a named hdf5 file,
* if file does not exist, creates it
* if the block does not exists in file creates block
*******************************************************************************/

void ops_fetch_stencil_hdf5_file(ops_stencil stencil, char const *file_name) {
  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t dset_id;      //dataset identifier
  hid_t filespace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (file_exist(file_name) == 0) {
    ops_printf("File %s does not exist .... creating file\n", file_name);
    FILE *fp; fp = fopen(file_name, "w");
    fclose(fp);
    //Create a new file
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  hsize_t rank = 1;
  hsize_t elems =  stencil->dims*stencil->points;

  /* create and write the dataset */
  if(H5Lexists(file_id, stencil->name, H5P_DEFAULT) == 0) {
    ops_printf("ops_stencil %s does not exists in the file ... creating data\n", stencil->name);
    H5LTmake_dataset(file_id,stencil->name,rank,&elems,H5T_NATIVE_INT,stencil->stencil);
  }
  else {
    dset_id = H5Dopen2(file_id, stencil->name, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, stencil->stencil);
    H5Dclose(dset_id);
  }

  //attach attributes to stencil
  H5LTset_attribute_string(file_id, stencil->name, "ops_type", "ops_stencil"); //ops type
  H5LTset_attribute_int(file_id, stencil->name, "dims", &(stencil->dims), 1); //dim
  H5LTset_attribute_int(file_id, stencil->name, "index", &(stencil->index), 1); //index
  H5LTset_attribute_int(file_id, stencil->name, "points", &(stencil->points), 1); //number of points
  H5LTset_attribute_int(file_id, stencil->name, "stride", stencil->stride, stencil->dims); //strides

  H5Pclose(plist_id);
  H5Fclose (file_id);
}

/*******************************************************************************
* Routine to write an ops_dat to a named hdf5 file,
* if file does not exist, creates it
* if the data set does not exists in file creates data set
*******************************************************************************/

void ops_fetch_dat_hdf5_file(ops_dat dat, char const *file_name) {

  //fetch data onto the host ( if needed ) based on the backend
  ops_get_data(dat);

  ops_block block = dat->block;

  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t group_id;      //group identifier
  hid_t dset_id;      //dataset identifier
  hid_t filespace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  hsize_t g_size[block->dims];
  int gbl_size[block->dims];
  for (int d = 0; d < block->dims; d++) {
	//pure data size (i.e. without block halos) to be noted as an attribute
	gbl_size[d] = dat->size[d] + dat->d_m[d] - dat->d_p[d];
	//the number of elements thats actually written
	g_size[d] = dat->size[d];
  }

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (file_exist(file_name) == 0) {
	ops_printf("File %s does not exist .... creating file\n", file_name);
    FILE *fp; fp = fopen(file_name, "w");
    fclose(fp);

    //Create a new file
	file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  if(H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
	ops_printf("ops_fetch_dat_hdf5_file: ops_block on which this ops_dat %s is declared does not exists in the file ... Aborting\n", dat->name);
  }
  else {
	//open existing group -- an ops_block is a group
	group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);

	if(H5Lexists(group_id, dat->name, H5P_DEFAULT) == 0) {
      ops_printf("ops_fetch_dat_hdf5_file: ops_dat %s does not exists in the ops_block %s ... creating ops_dat\n",
        dat->name, block->name);

	  if(strcmp(dat->type,"double") == 0)
        H5LTmake_dataset(group_id,dat->name,block->dims,g_size,H5T_NATIVE_DOUBLE,dat->data);
      else if(strcmp(dat->type,"float") == 0)
        H5LTmake_dataset(group_id,dat->name,block->dims,g_size,H5T_NATIVE_FLOAT,dat->data);
      else if(strcmp(dat->type,"int") == 0)
         H5LTmake_dataset(group_id,dat->name,block->dims,g_size,H5T_NATIVE_INT,dat->data);
      else if(strcmp(dat->type,"long") == 0)
         H5LTmake_dataset(group_id,dat->name,block->dims,g_size,H5T_NATIVE_LONG,dat->data);
      else if(strcmp(dat->type,"long long") == 0)
         H5LTmake_dataset(group_id,dat->name,block->dims,g_size,H5T_NATIVE_LLONG,dat->data);
      else {
		printf("Unknown type in ops_fetch_dat_hdf5_file()\n");
		exit(-2);
      }

	  //attach attributes to dat
	  H5LTset_attribute_string(group_id, dat->name, "ops_type", "ops_dat"); //ops type
	  H5LTset_attribute_string(group_id, dat->name, "block", block->name); //block
	  H5LTset_attribute_int(group_id, dat->name, "block_index", &(block->index), 1); //block index
	  H5LTset_attribute_int(group_id, dat->name, "dim", &(dat->dim), 1); //dim
	  H5LTset_attribute_int(group_id, dat->name, "size", gbl_size, block->dims); //size
	  H5LTset_attribute_int(group_id, dat->name, "d_m", dat->d_m, block->dims); //d_m
	  H5LTset_attribute_int(group_id, dat->name, "d_p", dat->d_p, block->dims); //d_p
	  H5LTset_attribute_int(group_id, dat->name, "base", dat->base, block->dims); //base
      H5LTset_attribute_string(group_id, dat->name, "type", dat->type); //type
    }

	H5Gclose(group_id);
	H5Fclose(file_id);
  }
}