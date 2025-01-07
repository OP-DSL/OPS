#pragma once

#if defined(__CUDACC__) && !defined(__CUDACC_RTC__)
#include <cassert>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DEVICE __device__

#define H_MIN ::min
#define H_MAX ::max
#else
#include <cmath>
#include <cassert>

#define DEVICE

#define H_MIN std::min
#define H_MAX std::max
#endif

#define m_pi (4.0*atan(1.0))

namespace ops::f2c {

constexpr int round32(int x) { return (x + 31) & ~31; }
constexpr size_t round32(size_t x) { return (x + 31) & ~31; }

DEVICE inline void trap() {
#ifdef __HIPCC__
    __builtin_trap();
#else
    assert(false);
#endif
}

/* Fortran intrinsics */
/*
template<typename T>
inline constexpr T pow(T x, int e) {
    if (e < 0)  return 0;
    if (e == 0) return 1;

    T r = x;
    for (int i = 1; i < e; ++i)
        r *= x;

    return r;
}*/

inline constexpr float pow(float x, float e) { return std::pow(x, e); }
inline constexpr float pow(int x, float e) { return std::pow((float) x, e); }

inline constexpr double pow(double x, double e) { return std::pow(x, e); }
inline constexpr double pow(int x, double e) { return std::pow((double) x, e); }

inline constexpr double pow(float x, double e) { return std::pow((double) x, e); }
inline constexpr double pow(double x, float e) { return std::pow(x, (double) e); }

inline constexpr int pow(int x, int e) { return std::pow(x, e); }

DEVICE inline int abs(int x) { return ::abs(x); }
DEVICE inline int64_t abs(int64_t x) { return ::abs(x); }
inline constexpr float abs(float x) { return fabsf(x); }
inline constexpr double abs(double x) { return fabs(x); }

inline constexpr double dble(int x) { return (double)x; }
inline constexpr double dble(int64_t x) { return (double)x; }
inline constexpr double dble(float x) { return (double)x; }
inline constexpr double dble(double x) { return x; }

inline constexpr int int_(int x) { return x; }
inline constexpr int int_(int64_t x) { return (int)x; }
inline constexpr int int_(float x) { return (int)x; }
inline constexpr int int_(double x) { return (int)x; }

DEVICE inline int min(int x0, int x1) { return H_MIN(x0, x1); }
DEVICE inline int min(int x0, int x1, int x2) { return H_MIN(H_MIN(x0, x1), x2); }
DEVICE inline int min(int x0, int x1, int x2, int x3) { return H_MIN(H_MIN(x0, x1), H_MIN(x2, x3)); }

DEVICE inline int64_t min(int64_t x0, int64_t x1) { return H_MIN(x0, x1); }
DEVICE inline int64_t min(int64_t x0, int64_t x1, int64_t x2) { return H_MIN(H_MIN(x0, x1), x2); }
DEVICE inline int64_t min(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return H_MIN(H_MIN(x0, x1), H_MIN(x2, x3)); }

inline constexpr float min(float x0, float x1) { return fminf(x0, x1); }
inline constexpr float min(float x0, float x1, float x2) { return fminf(fminf(x0, x1), x2); }
inline constexpr float min(float x0, float x1, float x2, float x3) { return fminf(fminf(x0, x1), fminf(x2, x3)); }

inline constexpr double min(double x0, double x1) { return fmin(x0, x1); }
inline constexpr double min(double x0, double x1, double x2) { return fmin(fmin(x0, x1), x2); }
inline constexpr double min(double x0, double x1, double x2, double x3) { return fmin(fmin(x0, x1), fmin(x2, x3)); }

DEVICE inline int max(int x0, int x1) { return H_MAX(x0, x1); }
DEVICE inline int max(int x0, int x1, int x2) { return H_MAX(H_MAX(x0, x1), x2); }
DEVICE inline int max(int x0, int x1, int x2, int x3) { return H_MAX(H_MAX(x0, x1), H_MAX(x2, x3)); }

DEVICE inline int64_t max(int64_t x0, int64_t x1) { return H_MAX(x0, x1); }
DEVICE inline int64_t max(int64_t x0, int64_t x1, int64_t x2) { return H_MAX(H_MAX(x0, x1), x2); }
DEVICE inline int64_t max(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return H_MAX(H_MAX(x0, x1), H_MAX(x2, x3)); }

inline constexpr float max(float x0, float x1) { return fmaxf(x0, x1); }
inline constexpr float max(float x0, float x1, float x2) { return fmaxf(fmaxf(x0, x1), x2); }
inline constexpr float max(float x0, float x1, float x2, float x3) { return fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)); }

inline constexpr double max(double x0, double x1) { return fmax(x0, x1); }
inline constexpr double max(double x0, double x1, double x2) { return fmax(fmax(x0, x1), x2); }
inline constexpr double max(double x0, double x1, double x2, double x3) { return fmax(fmax(x0, x1), fmax(x2, x3)); }

inline constexpr int mod(int a, int p) { return a % p; }
inline constexpr int64_t mod(int64_t a, int64_t p) { return a % p; }
inline constexpr float mod(float a, float p) { return fmodf(a, p); }
inline constexpr double mod(double a, double p) { return fmod(a, p); }

inline constexpr int nint(float x) { return lroundf(x); }
inline constexpr int nint(double x) { return lround(x); }

DEVICE inline int copysign(int x, int y) { return y >= 0 ? abs(x) : -abs(x); }
DEVICE inline int64_t copysign(int64_t x, int64_t y) { return y >= 0 ? abs(x) : -abs(x); }
inline constexpr float copysign(float x, float y) { return copysignf(x, y); }
inline constexpr double copysign(double x, double y) { return ::copysign(x, y); }

// ----------

inline constexpr float acos(float x) { return acosf(x); }
inline constexpr double acos(double x) { return ::acos(x); }

inline constexpr float asin(float x) { return asinf(x); }
inline constexpr double asin(double x) { return ::asin(x); }

inline constexpr float atan(float x) { return atanf(x); }
inline constexpr double atan(double x) { return ::atan(x); }

inline constexpr float atan2(float x, float y) { return atan2f(x, y); }
inline constexpr double atan2(double x, double y) { return ::atan2(x, y); }

inline constexpr float cos(float x) { return cosf(x); }
inline constexpr double cos(double x) { return ::cos(x); }

inline constexpr float cosh(float x) { return coshf(x); }
inline constexpr double cosh(double x) { return ::cosh(x); }

inline constexpr float exp(float x) { return expf(x); }
inline constexpr double exp(double x) { return ::exp(x); }

inline constexpr float log(float x) { return logf(x); }
inline constexpr double log(double x) { return ::log(x); }

inline constexpr float log10(float x) { return log10f(x); }
inline constexpr double log10(double x) { return ::log10(x); }

inline constexpr float sin(float x) { return sinf(x); }
inline constexpr double sin(double x) { return ::sin(x); }

inline constexpr float sinh(float x) { return sinhf(x); }
inline constexpr double sinh(double x) { return ::sinh(x); }

inline constexpr float sqrt(float x) { return sqrtf(x); }
inline constexpr double sqrt(double x) { return ::sqrt(x); }

inline constexpr float tan(float x) { return tanf(x); }
inline constexpr double tan(double x) { return ::tan(x); }

inline constexpr float tanh(float x) { return tanhf(x); }
inline constexpr double tanh(double x) { return ::tanh(x); }

} // namespace ops::prelude
