#ifndef __OPS_SEQ_V2_H
#define __OPS_SEQ_V2_H

#ifndef OPS_API
#define OPS_API 2
#endif

#include "ops_lib_cpp.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS

static int arg_idx[OPS_MAX_DIM];

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
#ifdef OPS_BATCHED
    c1[i] *= (i==OPS_BATCHED?1:stride[i-(OPS_BATCHED<i)]);
    c2[i] *= (i==OPS_BATCHED?1:stride[i-(OPS_BATCHED<i)]);
#else
    c1[i] *= stride[i];
    c2[i] *= stride[i];
#endif
  }
  int off =  add(c1, size, dim) - add(c2, size, dim);

  return off;
}

inline int address(int ndim, int dat_size, int* start, int* size, int* stride, int* base_off, int *d_m)
{
  int base = 0;
  for(int i=0; i<ndim; i++) {
#ifdef OPS_BATCHED
    base = base + dat_size * mult(size, i) * (start[i] * (i==OPS_BATCHED?1:stride[i-(OPS_BATCHED<i)]) - base_off[i] - d_m[i]);
#else
    base = base + dat_size * mult(size, i) * (start[i] * (stride[i]) - base_off[i] - d_m[i]);
#endif
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
  static char *construct(const ops_arg &arg, int ndim, int start[], ops_block block) { 
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
      for (int d = 0; d < ndim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
  #else //OPS_MPI
#ifdef OPS_BATCHED
      for (int d = 0; d < block->batchdim; d++) arg_idx[d] = start[d];
      for (int d = block->batchdim; d < ndim-1; d++) arg_idx[d] = start[d+1];
      if (block->count > 1) arg_idx[ndim-1] = start[block->batchdim];
#else
      for (int d = 0; d < ndim; d++) arg_idx[d] = start[d];
#endif

  #endif //OPS_MPI
      return (char *)arg_idx;
    }
    //assert(false && "Arg should be one of OPS_ARG_GBL or OPS_ARG_IDX");
    return nullptr;
  }
  static ParamT get(char *data) { return (ParamT)data; } 

#ifdef OPS_MPI 
static void shift_arg(const ops_arg &arg, char **p, int m, const int* start,
                      const int offs[], const sub_block_list &sb)
#else //OPS_MPI
static void shift_arg(const ops_arg &arg, char **p, int m, const int* start, 
                      const int offs[], int ndim)
#endif
{
  if (arg.argtype == OPS_ARG_IDX) {
#ifdef OPS_BATCHED
    if (m == OPS_BATCHED) arg_idx[ndim-1]++;
    else arg_idx[m-(OPS_BATCHED<m)]++;
#else
    arg_idx[m]++;
#endif
#ifdef OPS_MPI
    for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];
#else //OPS_MPI
    for (int d = 0; d < m; d++) {
#ifdef OPS_BATCHED
      if (d == OPS_BATCHED) arg_idx[ndim-1] = start[d];
      else arg_idx[d-(OPS_BATCHED<d)] = start[d];
#else
      arg_idx[d] = start[d];
#endif
    }
#endif //OPS_MPI
  }
  else if (arg.argtype == OPS_ARG_GBL && arg.acc != OPS_READ) {
#ifdef OPS_BATCHED
    if (m == OPS_BATCHED) *p += ((ops_reduction)arg.data)->size;
    else if (m > OPS_BATCHED) *p = ((ops_reduction)arg.data)->data;
#endif
  }
}

  static void free(char *) {}
};

template <typename T> struct param_handler<ACC<T>> {
  static char *construct(const ops_arg &arg, int ndim, int start[], ops_block block) { 
    if (arg.argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM];
  #ifdef OPS_MPI
      for (int d = 0; d < ndim; d++) d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < ndim; d++) d_m[d] = arg.dat->d_m[d];
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
      + address(ndim, OPS_instance::getOPSInstance()->OPS_soa ? arg.dat->type_size : arg.dat->elem_size, &start[0], 
        arg.dat->size, arg.stencil->stride, arg.dat->base,
        d_m))); //TODO
    } 
    //assert(false && "Arg must be OPS_ARG_DAT if accessed by ACC");
    return nullptr;
  }
  static ACC<T>& get(char *data) { return *((ACC<T> *)data); }

#ifdef OPS_MPI 
static void shift_arg(const ops_arg &arg, char **p, int m, const int* start,
                      const int offs[], int ndim, const sub_block_list &sb)
#else //OPS_MPI
static void shift_arg(const ops_arg &arg, char **p, int m, const int* start, 
                      const int offs[], int ndim)
#endif
{
  if (arg.argtype == OPS_ARG_DAT) {
    int offset = (OPS_instance::getOPSInstance()->OPS_soa ? 1 : arg.dat->dim) * offs[m];
    //p = p + ((OPS_soa ? arg.dat->type_size : arg.dat->elem_size) * offs[i][m]);
    ((ACC<T>*)*p)->next(offset); // T must be ACC<type> we need to set to the next element
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

template <typename... ParamType, typename... OPSARG, size_t... I>
void ops_par_loop_impl(indices<I...>, void (*kernel)(ParamType...), 
                      char const *name, ops_block block, int dim, int *range,
                      OPSARG... arguments) {
  constexpr int N = sizeof...(OPSARG);

  int  count[dim+1];
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
#ifdef OPS_BATCHED
  int ndim = block->dims + (block->count>1 || OPS_BATCHED < block->dims);
#else
  int ndim = block->dims;
#endif
  for (int n=0; n<ndim; n++) {
    if (n==block->batchdim) {start[n] = 0; end[n]=block->count;}
    else if (n<block->batchdim) {start[n] = range[2*n]; end[n] = range[2*n+1];}
    else {start[n] = range[2*(n-1)];end[n] = range[2*(n-1)+1];}
  }
  #endif //OPS_MPI

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  ops_H_D_exchanges_host(args, N);
  ops_halo_exchanges(args,N,range);
  ops_H_D_exchanges_host(args, N);

  char *p_a[N] = 
    {param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, ndim, start, block)...};
  //Offs decl
  int offs[N][OPS_MAX_DIM];
  (void) std::initializer_list<int>{(initoffs(arguments, offs[I], ndim, start, end), 0)...};

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }
  count[ndim-1]++;     // extra in last to ensure correct termination

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel((param_handler<param_remove_cvref_t<ParamType>>::get(p_a[I]))... );

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
      //shift_arg<param_remove_cvref_t<ParamType>>(arguments, p_a[I], m, start, offs[I], sb),0)...};
      param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, &p_a[I], m, start, offs[I], ndim, sb),0)...};
  #else //OPS_MPI
    (void) std::initializer_list<int>{(
      param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, &p_a[I], m, start, offs[I], ndim),0)...};
  #endif //OPS_MPI
  }

  (void) std::initializer_list<int>{
    (param_handler<param_remove_cvref_t<ParamType>>::free(p_a[I]),0)...};

  #ifdef OPS_DEBUG_DUMP
  (void) std::initializer_list<int>{(
    arguments.argtype == OPS_ARG_DAT && arguments.acc != OPS_READ? ops_dump3(arguments.dat,name),0:0)...};
  #endif
  (void) std::initializer_list<int>{(
  (arguments.argtype == OPS_ARG_DAT && arguments.acc != OPS_READ)?  ops_set_halo_dirtybit3(&arguments,range),0:0)...};
  ops_set_dirtybit_host(args, N);

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
void ops_par_loop(void (*kernel)(ParamType...), char const *name,
                  ops_block block, int dim, int *range,
                  OPSARG... arguments) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSARG), 
      "number of parameters of the kernel shoud match the number of ops_arg");
  ops_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                    block, dim, range, arguments...);
}


#endif

#endif /* ifndef __OPS_SEQ_V2_H */
