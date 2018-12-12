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
  * @brief OPS c++ header file
  * @author Gihan Mudalige
  * @details This header file should be included by all C++ OPS applications
  */

#ifndef __OPS_LIB_CPP_H
#define __OPS_LIB_CPP_H

#include <ops_lib_core.h>

/*
* run-time type-checking routines
*/
#ifndef DOXYGEN_SHOULD_SKIP_THIS
inline int type_error(const double *a, const char *type) {
  (void)a;
  return (strcmp(type, "double") && strcmp(type, "double:soa"));
}
inline int type_error(const float *a, const char *type) {
  (void)a;
  return (strcmp(type, "float") && strcmp(type, "float:soa"));
}
inline int type_error(const int *a, const char *type) {
  (void)a;
  return (strcmp(type, "int") && strcmp(type, "int:soa"));
}
inline int type_error(const uint *a, const char *type) {
  (void)a;
  return (strcmp(type, "uint") && strcmp(type, "uint:soa"));
}
inline int type_error(const ll *a, const char *type) {
  (void)a;
  return (strcmp(type, "ll") && strcmp(type, "ll:soa"));
}
inline int type_error(const ull *a, const char *type) {
  (void)a;
  return (strcmp(type, "ull") && strcmp(type, "ull:soa"));
}
inline int type_error(const bool *a, const char *type) {
  (void)a;
  return (strcmp(type, "bool") && strcmp(type, "bool:soa"));
}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 * Passes a scalar or small array that is invariant of the iteration space.
 *
 * (not to be confused with ::ops_decl_const, which facilitates
 * global scope variables).
 *
 * @tparam T
 * @param data  data array
 * @param dim   array dimension
 * @param type  string representing the type of data held in data
 * @param acc   access type
 * @return
 */
template <class T>
ops_arg ops_arg_gbl(T *data, int dim, char const *type, ops_access acc) {
  return ops_arg_gbl_char((char *)data, dim, sizeof(T), acc);
}

template <class T>
void ops_decl_const2(char const *name, int dim, char const *type, T *data) {
  if (type_error(data, type)) {
    printf("Error: incorrect type specified for constant \"%s\" \n", name);
    exit(1);
  }

  ops_decl_const_char(dim, type, sizeof(T), (char *)data, name);
}

/**
 * This routine returns the reduced value held by a reduction handle.
 *
 * @tparam T
 * @param handle  the ::ops_reduction handle
 * @param ptr     a pointer to write the results to, memory size has to match
 *                the declared
 */
template <class T> void ops_reduction_result(ops_reduction handle, T *ptr) {
  if (type_error(ptr, handle->type)) {
    printf(
        "Error: incorrect type specified for constant in ops_reduction_result");
    exit(1);
  }
  ops_reduction_result_char(handle, sizeof(T), (char *)ptr);
}

/**
 * This routine updates/changes the value of a constant.
 *
 * @tparam T
 * @param name  a name used to identify the constant
 * @param dim   dimension of dataset (number of items per element)
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param data  pointer to new values for constant of type @p T
 */
template <class T>
void ops_update_const(char const *name, int dim, char const *type, T *data) {
  (void)dim;
  if (type_error(data, type)) {
    printf("Error: incorrect type specified for constant in ops_update_const");
    exit(1);
  }
  ops_execute();
  ops_decl_const_char(dim, type, sizeof(T), (char *)data, name);
}

/**
 * This routine defines a global constant: a variable in global scope.
 *
 * Global constants need to be declared upfront so that they can be correctly
 * handled for different parallelizations. For e.g. CUDA on GPUs.
 * Once defined they remain unchanged throughout the program, unless changed
 * by a call to ops_update_const().
 * @tparam T
 * @param name  a name used to identify the constant
 * @param dim   dimension of dataset (number of items per element)
 * @param type  the name of type used for output diagnostics
 *              (e.g. "double", "float")
 * @param data  pointer to input data of type @p T
 */
template <class T>
void ops_decl_const(char const *name, int dim, char const *type, T *data) {
  (void)dim;
  if (type_error(data, type)) {
    printf("Error: incorrect type specified for constant in op_decl_const");
    exit(1);
  }
}

/*template < class T >
ops_dat ops_decl_dat ( ops_block block, int data_size,
                      int *block_size, int* offset, T *data,
                      char const * type,
                      char const * name )
{

  if ( type_error ( data, type ) ) {
    printf ( "incorrect type specified for dataset \"%s\" \n", name );
    exit ( 1 );
  }

  return ops_decl_dat_char(block, data_size, block_size, offset, (char *)data,
sizeof(T), type, name );

}*/

/**
 * This routine defines a dataset.
 *
 * The @p size allows to declare different sized data arrays on a given block.
 * @p d_m and @p d_p are depth of the "block halos" that are used to indicate
 * the offset from the edge of a block (in both the negative and positive
 * directions of each dimension).
 *
 * @tparam T
 * @param block       structured block
 * @param data_size   dimension of dataset (number of items per grid element)
 * @param block_size  size in each dimension of the block
 * @param base        base indices in each dimension of the block
 * @param d_m         padding from the face in the negative direction for each
 *                    dimension (used for block halo)
 * @param d_p         padding from the face in the positive direction for each
 *                    dimension (used for block halo)
 * @param stride
 * @param data        input data of type @p T
 * @param type        the name of type used for output diagnostics
 *                    (e.g. "double", "float")
 * @param name        a name used for output diagnostics
 * @return
 */
template <class T>
ops_dat ops_decl_dat(ops_block block, int data_size, int *block_size, int *base,
                     int *d_m, int *d_p, int *stride, T *data, char const *type,
                     char const *name) {

  if (type_error(data, type)) {
    printf("Error: incorrect type specified for dataset \"%s\" \n", name);
    exit(1);
  }

  return ops_decl_dat_char(block, data_size, block_size, base, d_m, d_p,
                           stride, (char *)data, sizeof(T), type, name);
}

/**
 * This routine defines a dataset.
 *
 * The @p size allows to declare different sized data arrays on a given block.
 * @p d_m and @p d_p are depth of the "block halos" that are used to indicate
 * the offset from the edge of a block (in both the negative and positive
 * directions of each dimension).
 *
 * @tparam T
 * @param block       structured block
 * @param data_size   dimension of dataset (number of items per grid element)
 * @param block_size  size in each dimension of the block
 * @param base        base indices in each dimension of the block
 * @param d_m         padding from the face in the negative direction for each
 *                    dimension (used for block halo)
 * @param d_p         padding from the face in the positive direction for each
 *                    dimension (used for block halo)
 * @param data        input data of type @p T
 * @param type        the name of type used for output diagnostics
 *                    (e.g. "double", "float")
 * @param name        a name used for output diagnostics
 * @return
 */
template <class T>
ops_dat ops_decl_dat(ops_block block, int data_size, int *block_size, int *base,
                     int *d_m, int *d_p, T *data, char const *type,
                     char const *name) {

  int stride[OPS_MAX_DIM];
  for (int i = 0; i < OPS_MAX_DIM; i++) stride[i] = 1;
  return ops_decl_dat_char(block, data_size, block_size, base, d_m, d_p,
                           stride, (char *)data, sizeof(T), type, name);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
//
// wrapper functions to handle MPI global reductions
//

inline void ops_mpi_reduce(ops_arg *args, float *data) {
  ops_mpi_reduce_float(args, data);
}

inline void ops_mpi_reduce(ops_arg *args, double *data) {
  ops_mpi_reduce_double(args, data);
}

inline void ops_mpi_reduce(ops_arg *args, int *data) {
  ops_mpi_reduce_int(args, data);
}

// needed as a dummy, "do nothing" routine for the non-mpi backends
template <class T> void ops_mpi_reduce(ops_arg *args, T *data) {
  // printf("should not be here\n");
}

#ifndef __CUDACC__
#define __host__ 
#define __device__ 
#endif

template<typename T>
class ACC {
public:
  //////////////////////////////////////////////////
  // 2D
  /////////////////////////////////////////////////
#if defined(OPS_2D)
  __host__ __device__
  ACC(int _sizex, T *_ptr) : sizex(_sizex), ptr(_ptr) {}
  __host__ __device__
  ACC(int _mdim, int _sizey, int _sizex, T *_ptr) : sizex(_sizex),
#ifdef OPS_SOA
    sizey(_sizey),
#else
    mdim(_mdim),
#endif
    ptr(_ptr)
  {}
  __host__ __device__
  const T& operator()(int xoff, int yoff) const {return *(ptr + xoff + yoff*sizex);}
  __host__ __device__
  T& operator()(int xoff, int yoff) {return *(ptr + xoff + yoff*sizex);}
  __host__ __device__
  const T& operator()(int d, int xoff, int yoff) const {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + d * sizex*sizey);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim );
#endif
  }
  __host__ __device__
  T& operator()(int d, int xoff, int yoff) {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + d * sizex*sizey);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim );
#endif
  }
  __host__ __device__
  void next(int offset) {
    ptr += offset;
  }
#endif
  //////////////////////////////////////////////////
  // 3D
  /////////////////////////////////////////////////
#if defined(OPS_3D)
  __host__ __device__
  ACC(int _sizex, int _sizey, T *_ptr) : sizex(_sizex), sizey(_sizey), ptr(_ptr) {}
  __host__ __device__
  ACC(int _mdim, int _sizez, int _sizey, int _sizex, T *_ptr) : sizex(_sizex), sizey(_sizey),
#ifdef OPS_SOA
    sizez(_sizez),
#else
    mdim(_mdim),
#endif
    ptr(_ptr)
  {}
  __host__ __device__
  const T& operator()(int xoff, int yoff, int zoff) const {return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey);}
  __host__ __device__
  T& operator()(int xoff, int yoff, int zoff) {return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey);}
  __host__ __device__
  const T& operator()(int d, int xoff, int yoff, int zoff) const {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + d * sizex*sizey*sizez);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim + zoff*sizex*sizey*mdim);
#endif
  }
  __host__ __device__
  T& operator()(int d, int xoff, int yoff, int zoff) {
#ifdef OPS_SOA
    return *(ptr + xoff + yoff*sizex + zoff*sizex*sizey + d * sizex*sizey*sizez);
#else
    return *(ptr + d + xoff*mdim + yoff*sizex*mdim + zoff*sizex*sizey*mdim);
#endif
  }
#endif

private:
#ifndef OPS_SOA
  int mdim;
#endif
  int sizex;
#if defined(OPS_3D) || defined(OPS_4D) || defined (OPS_5D) || (defined(OPS_2D) && defined(OPS_SOA))
  int sizey;
#endif
#if defined(OPS_4D) || defined (OPS_5D) || (defined(OPS_3D) && defined(OPS_SOA))
  int sizez;
#endif
#if defined (OPS_5D) || (defined(OPS_4D) && defined(OPS_SOA))
  int sizeu;
#endif
#if defined(OPS_5D) && defined(OPS_SOA)
  int sizev;
#endif
  T *__restrict__ ptr;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* __OPS_LIB_CPP_H */
