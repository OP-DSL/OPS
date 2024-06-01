// ops_cuda_atomic.h
#ifndef OPS_CUDA_ATOMIC_H
#define OPS_CUDA_ATOMIC_H

#if defined(__CUDA_ARCH__) && defined(__CUDACC__)
__device__ half atomicMax(half* address, const half val)
{
    unsigned short* address_as_us = (unsigned short*)address;
    unsigned short old = *address_as_us, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __half_as_ushort(::fmaxf(__ushort_as_half(assumed), val)));
    } while (assumed != old);
    return __ushort_as_half(old);
}

__device__ float atomicMax(float* address, const float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ double atomicMax(double* address, const double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ half atomicMin(half* address, const half val)
{
    unsigned short* address_as_us = (unsigned short*)address;
    unsigned short old = *address_as_us, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __half_as_ushort(::fminf(__ushort_as_half(assumed), val)));
    } while (assumed != old);
    return __ushort_as_half(old);
}

__device__ float atomicMin(float* address, const float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ double atomicMin(double* address, const double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif // OPS_CUDA_ATOMIC_H