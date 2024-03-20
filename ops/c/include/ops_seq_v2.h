#ifndef __OPS_SEQ_V2_H
#define __OPS_SEQ_V2_H

#ifndef OPS_API
#define OPS_API 2
#endif

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS

inline int mult(int* size, int dim)
{
  int result = 1;
  if(dim > 0) {
    for(int i = 0; i<dim;i++) result *= size[i];
  }
  return result;
}

inline int add(int* coords, int* size, int dim)
{
  int result = coords[0];
  for(int i = 1; i<=dim;i++) result += coords[i]*mult(size,i);
  return result;
}


inline int off(int ndim, int dim , int* start, int* end, int* size, int* stride)
{

  int i = 0;
  int c1[OPS_MAX_DIM];
  int c2[OPS_MAX_DIM];

  for(i=0; i<=dim; i++) c1[i] = start[i]+1;
  for(i=dim+1; i<ndim; i++) c1[i] = start[i];

  for(i = 0; i<dim; i++) c2[i] = end[i];
  for(i=dim; i<ndim; i++) c2[i] = start[i];

  for (i = 0; i < ndim; i++) {
    c1[i] *= stride[i];
    c2[i] *= stride[i];
  }
  int off =  add(c1, size, dim) - add(c2, size, dim);

  return off;
}

inline int address(int ndim, int dat_size, int* start, int* size, int* stride, int* base_off, int *d_m)
{
  int base = 0;
  for(int i=0; i<ndim; i++) {
    base = base + dat_size * mult(size, i) * (start[i] * stride[i] - base_off[i] - d_m[i]);
  }
  return base;
}

inline void stencil_depth(ops_stencil sten, int* d_pos, int* d_neg)
{
  for(int dim = 0;dim<sten->dims;dim++){
    d_pos[dim] = 0; d_neg[dim] = 0;
  }
  for(int p=0;p<sten->points; p++) {
    for(int dim = 0;dim<sten->dims;dim++){
    d_pos[dim] = MAX(d_pos[dim],sten->stencil[sten->dims*p + dim]);
    d_neg[dim] = MIN(d_neg[dim],sten->stencil[sten->dims*p + dim]);
    }
  }
}

#if __cplusplus >= 201103L
// ops_par_loop implementation with variadic template arguments
#include <utility>
#if __cplusplus >= 201402L
// after c++14 use built in types.
template <size_t... I> using indices = std::index_sequence<I...>;
template <size_t Num> using build_indices = std::make_index_sequence<Num>;

#else
// pre c++14 we need our own helper structs
template <size_t... Is> struct indices {};
template <size_t N, size_t... Is>
struct build_indices : public build_indices<N - 1, N - 1, Is...> {};
template <size_t... Is> struct build_indices<0, Is...> : indices<Is...> {};
#endif

// helper struct to get the underlying type of the kernel parameters
// e.g. const ACC<double> & to ACC<double>
template <typename T> struct param_remove_cvref {
  using type =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

// helper struct to get the underlying type of pointer kernel parameters
// e.g. const int * to int *
template <typename T> struct param_remove_cvref<T *> {
  using type = typename std::add_pointer<typename std::remove_cv<
      typename std::remove_reference<T>::type>::type>::type; // remove_reference may be wrong here..
};

template <typename T>
using param_remove_cvref_t = typename param_remove_cvref<T>::type;

// helper struct to create, pass and free parameters to the kernel
template <typename ParamT> struct param_handler {
  static char *construct(const ops_arg &arg, int dim, int ndim, int start[], ops_block block) { 
    if (arg.argtype == OPS_ARG_GBL) {
      if (arg.acc == OPS_READ) return arg.data;
      else
  #ifdef OPS_MPI
        return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * block->index;
  #else //OPS_MPI
        return ((ops_reduction)arg.data)->data;
  #endif //OPS_MPI
    } else if (arg.argtype == OPS_ARG_IDX) {
  #ifdef OPS_MPI
      sub_block_list sb = OPS_sub_block_list[block->index]; //TODO: Multigrid
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = start[d];
  #endif //OPS_MPI
      return (char *)block->instance->arg_idx;
    }
    //assert(false && "Arg should be one of OPS_ARG_GBL or OPS_ARG_IDX");
    return nullptr;
  }
  static ParamT get(char *data) { return (ParamT)data; } 

#ifdef OPS_MPI 
static void shift_arg(const ops_arg &arg, char *p, int m, const int* start,
                      const int offs[], const sub_block_list &sb, OPS_instance *instance)
#else //OPS_MPI
static void shift_arg(const ops_arg &arg, char *p, int m, const int* start, 
                      const int offs[], OPS_instance *instance)
#endif
{
  if (arg.argtype == OPS_ARG_IDX && m < OPS_MAX_DIM) {
    instance->arg_idx[m]++;
#ifdef OPS_MPI
    for (int d = 0; d < m && d < OPS_MAX_DIM; d++) instance->arg_idx[d] = sb->decomp_disp[d] + start[d];
#else //OPS_MPI
    for (int d = 0; d < m && d < OPS_MAX_DIM; d++) instance->arg_idx[d] = start[d];
#endif //OPS_MPI
  }
}

  static void free(char *) {}
};

template <typename T> struct param_handler<ACC<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, int start[], ops_block block) { 
    if (arg.argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM] = {};
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d];
  #endif //OPS_MPI
#ifdef OPS_1D
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], (T*)(arg.data //base of 2D array
#elif defined(OPS_2D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], (T*)(arg.data //base of 2D array
#elif defined(OPS_3D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2], (T*)(arg.data //base of 3D array
#elif defined(OPS_4D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2], arg.dat->size[3], (T*)(arg.data //base of 3D array
#else
      return (char *) ((arg.dat->data //TODO
#endif
      + address(ndim, arg.dat->block->instance->OPS_soa ? arg.dat->type_size : arg.dat->elem_size, &start[0], 
        arg.dat->size, arg.stencil->stride, arg.dat->base,
        d_m))); //TODO
    } 
    //assert(false && "Arg must be OPS_ARG_DAT if accessed by ACC");
    return nullptr;
  }
  static ACC<T>& get(char *data) { return *((ACC<T> *)data); }

#ifdef OPS_MPI 
static void shift_arg(const ops_arg &arg, char *p, int m, const int* start,
                      const int offs[], const sub_block_list &sb, OPS_instance *instance)
#else //OPS_MPI
static void shift_arg(const ops_arg &arg, char *p, int m, const int* start, 
                      const int offs[], OPS_instance *instance)
#endif
{
  if (arg.argtype == OPS_ARG_DAT) {
    int offset = (arg.dat->block->instance->OPS_soa ? 1 : arg.dat->dim) * offs[m];
    //p = p + ((OPS_soa ? arg.dat->type_size : arg.dat->elem_size) * offs[i][m]);
    ((ACC<T>*)p)->next(offset); // T must be ACC<type> we need to set to the next element
  } 
}

  static void free(char *data) { delete (ACC<T> *)data;}
};

static void initoffs(const ops_arg &arg, int *offs, const int &ndim, int *start, int *end) {
  if(arg.stencil != nullptr) {
    offs[0] = arg.stencil->stride[0]*1;
    for(int n=1; n<ndim; ++n){
      offs[n] = off(ndim, n, start, end, arg.dat->size, arg.stencil->stride);
    }
  }
}

template <typename... ParamType, typename... OPSARG, size_t... J>
void ops_par_loop_impl(indices<J...>, void (*kernel)(ParamType...),
                      char const *name, ops_block block, int dim, int *range,
                      OPSARG... arguments) {
  constexpr int N = sizeof...(OPSARG);

  int  count[OPS_MAX_DIM] = {0};
  ops_arg args[N] = {arguments...};
  
  #ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,N,range,name)) return;
  #endif

  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block 
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(block->instance, args, name);
  #endif

  char *p_a[N] = 
    {param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim, ndim, start, block)...};
  //Offs decl
  int offs[N][OPS_MAX_DIM] = {};
  (void) std::initializer_list<int>{(initoffs(arguments, offs[J], ndim, start, end), 0)...};

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  ops_H_D_exchanges_host(args, N);
  ops_halo_exchanges(args,N,range);
  ops_H_D_exchanges_host(args, N);

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel((param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))... );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    // shift pointers to data
  #ifdef OPS_MPI
    (void) std::initializer_list<int>{(
      param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], m, start, offs[J], sb, block->instance),0)...};
  #else //OPS_MPI
    (void) std::initializer_list<int>{(
      param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], m, start, offs[J], block->instance),0)...};
  #endif //OPS_MPI
  }

  #ifdef OPS_DEBUG_DUMP
  (void) std::initializer_list<int>{(
    arguments.argtype == OPS_ARG_DAT && arguments.acc != OPS_READ? ops_dump3(arguments.dat,name),0:0)...};
  #endif
  (void) std::initializer_list<int>{(
  (arguments.argtype == OPS_ARG_DAT && arguments.acc != OPS_READ)?  ops_set_halo_dirtybit3(&arguments,range),0:0)...};
  ops_set_dirtybit_host(args, N);

  (void) std::initializer_list<int>{
    (param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]),0)...};
}


template <typename... ParamType, typename... OPSARG, size_t... J>
ops_iter_par_loop_desc<ParamType...> ops_par_loop_impl_V2(indices<J...>, void (*kernel)(ParamType...),
                      char const *name, ops_block block, int dim, int *range,
                      OPSARG... arguments) {
  constexpr int N = sizeof...(OPSARG);

  ops_iter_par_loop_desc<ParamType...> loop_obj;
  ops_arg args[N] = {arguments...};
  loop_obj.args.insert(loop_obj.args.end(), args, args + N);
  loop_obj.block = block;
  loop_obj.dim = dim;
  loop_obj.argType = OPS_ITER_PAR_ARG_TYPE::OPS_PAR_LOOP;
//   loop_opj->kernel_func = std::function<void(ParamType...)>(kernel);
  loop_obj.range.insert(loop_obj.range.end(), range, range + (dim * 2));
  loop_obj.name = std::string(name);
  return (loop_obj);
  }


template<typename... PramType>
void ops_par_loop_sigle_executer(ops_iter_par_loop_desc<PramType...>& desc)
{
  std::cout << "desk kernel name: " << desc.name << std::endl;
  
//   int N = desc.args.size();
//   int  count[OPS_MAX_DIM] = {0};

//   #ifdef CHECKPOINTING
//   if (!ops_checkpointing_name_before(desc.args.data(), N, desc.range.data(), desc.name.c_str())) return;
//   #endif

//   int start[OPS_MAX_DIM];
//   int end[OPS_MAX_DIM];

//   #ifdef OPS_MPI
//   sub_block_list sb = OPS_sub_block_list[desc.block->index];
//   if (!sb->owned) return;
//   //compute locally allocated range for the sub-block 
//   int ndim = sb->ndim;
//   for (int n=0; n<ndim; n++) {
//     start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
//     if (start[n] >= desc.range[2*n]) start[n] = 0;
//     else start[n] = desc.range[2*n] - start[n];
//     if (sb->id_m[n]==MPI_PROC_NULL && desc.range[2*n] < 0) start[n] = desc.range[2*n];
//     if (end[n] >= desc.range[2*n+1]) end[n] = desc.range[2*n+1] - sb->decomp_disp[n];
//     else end[n] = sb->decomp_size[n];
//     if (sb->id_p[n]==MPI_PROC_NULL && (desc.range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
//       end[n] += (desc.range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
//   }
//   #else //!OPS_MPI
//   int ndim = block->dims;
//   for (int n=0; n<ndim; n++) {
//     start[n] = desc.range[2*n];end[n] = desc.range[2*n+1];
//   }
//   #endif //OPS_MPI

//   #ifdef OPS_DEBUG
//   ops_register_args(desc.block->instance, desc.args.data(), desc.name.c_str());
//   #endif

//   char *p_a[N];

//   for (unsigned int i = 0; i < N; i++)
//     p_a[i] = param_handler<param_remove_cvref_t<ParamType>>::construct(desc.args[i], desc.dim, ndim, start, desc.block);
//   //Offs decl
//   int offs[N][OPS_MAX_DIM] = {};
//   (void) std::initializer_list<int>{(initoffs(desc.args[J], offs[J], ndim, start, end), 0)...};

//   int total_range = 1;
//   for (int n=0; n<ndim; n++) {
//     count[n] = end[n]-start[n];  // number in each dimension
//     total_range *= count[n];
//     total_range *= (count[n]<0?0:1);
//   }
//   count[dim-1]++;     // extra in last to ensure correct termination

//   ops_H_D_exchanges_host(desc.args.data(), N);
//   ops_halo_exchanges(desc.args.data(),N,desc.range.data());
//   ops_H_D_exchanges_host(desc.args.data(), N);

//   for (int nt=0; nt<total_range; nt++) {
//     // call kernel function, passing in pointers to data

//     kernel((param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))... );

//     count[0]--;   // decrement counter
//     int m = 0;    // max dimension with changed index
//     while (count[m]==0) {
//       count[m] =  end[m]-start[m];// reset counter
//       m++;                        // next dimension
//       count[m]--;                 // decrement counter
//     }

//     // shift pointers to data
//   #ifdef OPS_MPI
//     (void) std::initializer_list<int>{(
//       param_handler<param_remove_cvref_t<ParamType>>::shift_arg(desc.args[J], p_a[J], m, start, offs[J], sb, desc.block->instance),0)...};
//   #else //OPS_MPI
//     (void) std::initializer_list<int>{(
//       param_handler<param_remove_cvref_t<ParamType>>::shift_arg(desc.args[J], p_a[J], m, start, offs[J], desc.block->instance),0)...};
//   #endif //OPS_MPI
//   }

//   #ifdef OPS_DEBUG_DUMP
//   (void) std::initializer_list<int>{(
//     desc.args[J].argtype == OPS_ARG_DAT && desc.args[J].acc != OPS_READ? ops_dump3(desc.args[J].dat,desc.name.c_str()),0:0)...};
//   #endif
//   (void) std::initializer_list<int>{(
//   (desc.args[J].argtype == OPS_ARG_DAT && desc.args[J].acc != OPS_READ)?  ops_set_halo_dirtybit3(&desc.args[J],desc.range),0:0)...};
//   ops_set_dirtybit_host(args, N);

//   (void) std::initializer_list<int>{
//     (param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]),0)...};
}

template <typename... DESC_ARGS>
void ops_iter_par_loop(const std::string unique_name, unsigned int& iter, DESC_ARGS&&... descs)
{
    (ops_par_loop_sigle_executer(descs), ...);
}

#endif /*DOXYGEN_SHOULD_SKIP_THIS*/

/**
 * Performs a parallel loop, executing the user-defined function kernel,
 * passing data as specified by the list arguments. The iteration space
 * is dim dimensional, and the bounds are specified by range
 *
 * Arguments to kernel are passed through as ACC<datatype>& references
 * for ops_dats, and need to be accessed with the overloaded () operator.
 * For other types of arguments, pointers are passed that can be
 * dereferenced directly 
 *
 * @param kernel   user kernel, #of arguments must match the ops_arg parameters
 * @param name     a name for the parallel loop
 * @param block    the ops_block to iterate on
 * @param dim      dimensionality of the block
 * @param range    loop bounds in the following order: {dim0_lower,
 *                 dim0_upper_exclusive, dim1_lower, ...}
 * @param arguments a list of ops_arg arguments
 */
template <typename... ParamType, typename... OPSARG>
ops_iter_par_loop_desc<ParamType...> ops_par_loop(void (*kernel)(ParamType...), char const *name,
                  ops_block block, int dim, int *range,
                  OPSARG... arguments) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSARG), 
      "number of parameters of the kernel shoud match the number of ops_arg");
  ops_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                    block, dim, range, arguments...);
  ops_iter_par_loop_desc<ParamType...> desc = ops_par_loop_impl_V2(build_indices<sizeof...(ParamType)>{}, kernel, name,
                    block, dim, range, arguments...);
//   ops_par_loop_executer(desc);
//   ops_iter_par_loop(desc);
  return (desc);
}

 //TODO: implementation
template<typename T> 
ops_iter_par_loop_desc<ACC<T>, ACC<T>> ops_par_copy(ops_dat arg_target, ops_dat arg_origin)
{
    ops_iter_par_loop_desc<ACC<T>, ACC<T>> desc;
    desc.argType = OPS_ITER_PAR_ARG_TYPE::OPS_DAT_COPY;
    return (desc);
}

#endif /* C++11 */

#endif /* ifndef __OPS_SEQ_V2_H */
