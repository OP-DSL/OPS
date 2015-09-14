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
* if the block does not exists in file creates block as HDF5 group
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
* Routine to write an ops_stencil to a named hdf5 file,
* if file does not exist, creates it
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

/*******************************************************************************
* Routine to read an ops_block from an hdf5 file
*******************************************************************************/
ops_block ops_decl_block_hdf5(int dims, char *block_name,
                      char const *file_name) {

  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t plist_id;     //property list identifier
  herr_t err;         //error code

  //open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    ops_printf("ops_decl_block_hdf5: File %s does not exist .... aborting\n", file_name);
	exit(-2);
  }

  //Set up file access property list for I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  //check if ops_block exists
  if(H5Lexists(file_id, block_name, H5P_DEFAULT) == 0)
    ops_printf("ops_decl_block_hdf5: ops_block %s does not exists in the file ... aborting\n", block_name);

  //ops_block exists .. now check ops_type and dims
  char read_ops_type[10];
  if (H5LTget_attribute_string(file_id, block_name, "ops_type", read_ops_type) < 0){
    ops_printf("ops_decl_block_hdf5: Attribute \"ops_type\" not found in block %s .. Aborting\n",block_name);
    exit(-2);
  } else {
    if (strcmp("ops_block",read_ops_type) != 0) {
      ops_printf("ops_decl_block_hdf5: ops_type of block %s is defined are not equal to ops_block.. Aborting\n",block_name);
      exit(-2);
    }
  }
  int read_dims;
  if (H5LTget_attribute_int(file_id, block_name, "dims", &read_dims) < 0) {
    ops_printf("ops_decl_block_hdf5: Attribute \"dims\" not found in block %s .. Aborting\n",block_name);
    exit(-2);
  }
  else {
    if (dims != read_dims) {
      ops_printf("ops_decl_block_hdf5: Unequal dims of block %s: dims on file %d, dims specified %d .. Aborting\n",
         block_name,read_dims, dims);
      exit(-2);
    }
  }
  int read_index;
  if (H5LTget_attribute_int(file_id, block_name, "index", &read_index) < 0) {
    ops_printf("ops_decl_block_hdf5: Attribute \"index\" not found in block %s .. Aborting\n",block_name);
    exit(-2);
  }

  //checks passed ..

  H5Pclose(plist_id);
  H5Fclose (file_id);

  return ops_decl_block(read_dims, block_name );

}

/*******************************************************************************
* Routine to read an ops_stencil from an hdf5 file
*******************************************************************************/
ops_stencil ops_decl_stencil_hdf5(int dims, int points, char *stencil_name,
                      char const *file_name) {
  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t plist_id;     //property list identifier
  herr_t err;         //error code

  //open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    ops_printf("ops_decl_stencil_hdf5: File %s does not exist .... aborting\n", file_name);
    exit(-2);
  }

  //Set up file access property list for I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  //check if ops_stencil exists
  if(H5Lexists(file_id, stencil_name, H5P_DEFAULT) == 0)
    ops_printf("ops_decl_stencil_hdf5: ops_stencil %s does not exists in the file ... aborting\n", stencil_name);

  //ops_stencil exists .. now check ops_type and dims
  char read_ops_type[10];
  if (H5LTget_attribute_string(file_id, stencil_name, "ops_type", read_ops_type) < 0){
    ops_printf("ops_decl_stencil_hdf5: Attribute \"ops_type\" not found in stencil %s .. Aborting\n",stencil_name);
    exit(-2);
  } else {
    if (strcmp("ops_stencil",read_ops_type) != 0) {
      ops_printf("ops_decl_stencil_hdf5: ops_type of stencil %s is defined are not equal to ops_stencil.. Aborting\n",stencil_name);
      exit(-2);
    }
  }
  int read_dims;
  if (H5LTget_attribute_int(file_id, stencil_name, "dims", &read_dims) < 0) {
    ops_printf("ops_decl_stencil_hdf5: Attribute \"dims\" not found in stencil %s .. Aborting\n",stencil_name);
    exit(-2);
  }
  else {
    if (dims != read_dims) {
      ops_printf("ops_decl_stencil_hdf5: Unequal dims of stencil %s: dims on file %d, dims specified %d .. Aborting\n",
         stencil_name,read_dims, dims);
      exit(-2);
    }
  }
  int read_points;
  if (H5LTget_attribute_int(file_id, stencil_name, "points", &read_points) < 0) {
    ops_printf("ops_decl_stencil_hdf5: Attribute \"points\" not found in stencil %s .. Aborting\n",stencil_name);
    exit(-2);
  }
  else {
    if (points != read_points) {
      ops_printf("ops_decl_stencil_hdf5: Unequal points of stencil %s: points on file %d, points specified %d .. Aborting\n",
         stencil_name,read_points, points);
      exit(-2);
    }
  }
  //checks passed ..

  //get the strides
  int read_stride[read_dims];
  if (H5LTget_attribute_int(file_id, stencil_name, "stride", read_stride) < 0) {
    ops_printf("ops_decl_stencil_hdf5: Attribute \"stride\" not found in stencil %s .. Aborting\n",stencil_name);
    exit(-2);
  }

  int read_sten[read_dims*read_points];
  H5LTread_dataset_int(file_id,stencil_name,read_sten);
  H5Pclose(plist_id);
  H5Fclose (file_id);
  //use decl_strided stencil for both normal and strided stencils
  return ops_decl_strided_stencil (read_dims, read_points, read_sten, read_stride, stencil_name);

}

/*******************************************************************************
* Routine to read an ops_dat from an hdf5 file
*******************************************************************************/
ops_dat ops_decl_dat_hdf5(ops_block block, int dat_dim,
                      char const *type,
                      char const *dat_name,
                      char const *file_name) {

  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t group_id;      //group identifier
  hid_t dset_id;      //dataset identifier
  hid_t filespace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  //open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    ops_printf("ops_decl_dat_hdf5: File %s does not exist .... aborting\n", file_name);
    exit(-2);
  }

  //Set up file access property list for I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  if(H5Lexists(file_id, block->name, H5P_DEFAULT) == 0) {
      ops_printf("ops_decl_dat_hdf5: ops_block on which this ops_dat %s is declared does not exists in the file ... Aborting\n", dat_name);
      exit(-2);
  }

  //open existing group -- an ops_block is a group
  group_id = H5Gopen2(file_id, block->name, H5P_DEFAULT);

  //check if ops_dat exists
  if(H5Lexists(group_id, dat_name, H5P_DEFAULT) == 0) {
    ops_printf("ops_decl_dat_hdf5: ops_dat %s does not exists in the block %s ... aborting\n", dat_name, block->name);
    exit(-2);
  }

  //ops_dat exists .. now check ops_type, block_index, type and dim
  char read_ops_type[10];
  if (H5LTget_attribute_string(group_id, dat_name, "ops_type", read_ops_type) < 0){
    ops_printf("ops_decl_dat_hdf5: Attribute \"ops_type\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  } else {
    if (strcmp("ops_dat",read_ops_type) != 0) {
      ops_printf("ops_decl_dat_hdf5: ops_type of dat %s is defined are not equal to ops_dat.. Aborting\n",dat_name);
      exit(-2);
    }
  }
  int read_block_index;
  if (H5LTget_attribute_int(group_id, dat_name, "block_index", &read_block_index) < 0) {
    ops_printf("ops_decl_dat_hdf5: Attribute \"block_index\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  }
  else {
    if (block->index != read_block_index) {
      ops_printf("ops_decl_dat_hdf5: Unequal dims of data set %s: block index on file %d, block index specified for this dat %d .. Aborting\n",
         dat_name,read_block_index, block->index);
      exit(-2);
    }
  }
  int read_dim;
  if (H5LTget_attribute_int(group_id, dat_name, "dim", &read_dim) < 0) {
    ops_printf("ops_decl_dat_hdf5: Attribute \"dim\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  }
  else {
    if (dat_dim != read_dim) {
      ops_printf("ops_decl_dat_hdf5: Unequal dims of data set %s: dim on file %d, dim specified %d .. Aborting\n",
         dat_name,read_dim, dat_dim);
      exit(-2);
    }
  }
  char read_type[15];
  if (H5LTget_attribute_string(group_id, dat_name, "type", read_type) < 0){
    ops_printf("ops_decl_dat_hdf5: Attribute \"type\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  } else {
    if (strcmp(type,read_type) != 0) {
      ops_printf("ops_decl_dat_hdf5: Type of data of data set %s is not equal: type on file %s, type specified %s .. Aborting\n",
        dat_name, read_type, type);
      exit(-2);
    }
  }

  //checks passed .. now read in all other details of ops_dat from file

  int read_size[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "size", read_size) < 0) {
    ops_printf("ops_decl_dat_hdf5: Attribute \"size\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  }

  int read_d_m[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "d_m", read_d_m) < 0) {
    ops_printf("ops_decl_dat_hdf5: Attribute \"d_m\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  }

  int read_d_p[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "d_p", read_d_p) < 0) {
    ops_printf("ops_decl_dat_hdf5: Attribute \"d_p\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  }

  int read_base[block->dims];
  if (H5LTget_attribute_int(group_id, dat_name, "base", read_base) < 0) {
    ops_printf("ops_decl_dat_hdf5: Attribute \"base\" not found in data set %s .. Aborting\n",dat_name);
    exit(-2);
  }

  //set type size
  int type_size;
  if(strcmp(read_type,"double") == 0)
    type_size = sizeof(double);
  else if(strcmp(read_type,"float") == 0)
    type_size = sizeof(float);
  else if(strcmp(read_type,"int") == 0)
    type_size = sizeof(int);
  else if(strcmp(read_type,"long") == 0)
    type_size = sizeof(long);
  else if(strcmp(read_type,"long long") == 0)
    type_size = sizeof(long long);
  else{
    printf("Unknown type %s in ops_decl_dat_hdf5()\n", read_type);
    exit(2);
  }

  //read in the actual data
  int t_size = 1;
  for (int d = 0; d < block->dims; d++)
	t_size *= read_size[d] - read_d_m[d] + read_d_p[d];
  char* data = (char *)malloc(t_size*dat_dim*type_size);

  if(strcmp(read_type,"double") == 0)
	  H5LTread_dataset(group_id, dat_name, H5T_NATIVE_DOUBLE, data);
    else if(strcmp(read_type,"float") == 0)
      H5LTread_dataset(group_id, dat_name, H5T_NATIVE_FLOAT, data);
    else if(strcmp(read_type,"int") == 0)
      H5LTread_dataset(group_id, dat_name, H5T_NATIVE_INT, data);
    else if(strcmp(read_type,"long") == 0)
      H5LTread_dataset(group_id, dat_name, H5T_NATIVE_LONG, data);
    else if(strcmp(read_type,"long long") == 0)
      H5LTread_dataset(group_id, dat_name, H5T_NATIVE_LLONG, data);
    else {
      printf("Unknown type in ops_decl_dat_hdf5()\n");
      exit(-2);
  }

  ops_dat created_dat = ops_decl_dat_char(block, dat_dim,
      read_size/*global dat size in each dimension*/,
      read_base, read_d_m, read_d_p, data,
      type_size/*size of(type)*/, type, dat_name );

  created_dat->is_hdf5 = 1;
  created_dat->hdf5_file = file_name;
  created_dat->user_managed = 0;

  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

  return created_dat;

 }