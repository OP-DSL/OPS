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

//#define m_pi (4.0*atan(1.0))

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

//------------------------ SIN ---------------------
extern "C" {
    inline constexpr float fpow_float(float *x, float *e);
    inline constexpr double fpow_dble(double *x, double *e);
    inline constexpr int fpow_int(int *x, int *e);
}

inline constexpr float pow(float x, float e) { return fpow_float(&x, &e); }
inline constexpr double pow(double x, double e) { return fpow_dble(&x, &e); }
inline constexpr int pow(int x, int e) { return fpow_int(&x, &e); }

inline constexpr float pow(int x, float e) { return std::pow((float) x, e); }
inline constexpr double pow(int x, double e) { return std::pow((double) x, e); }

inline constexpr double pow(float x, double e) { return std::pow((double) x, e); }
inline constexpr double pow(double x, float e) { return std::pow(x, (double) e); }

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

//------------------------ SIN ---------------------
extern "C" {
    inline constexpr float fsin_float(float *x);
    inline constexpr double fsin_dble(double *x);
}

inline constexpr float sin(float x) { return fsin_float(&x); }
inline constexpr double sin(double x) { return fsin_dble(&x); }

//------------------------ SINH ---------------------
extern "C" {
    inline constexpr float fsinh_float(float *x);
    inline constexpr double fsinh_dble(double *x);
}

inline constexpr float sinh(float x) { return fsinh_float(&x); }
inline constexpr double sinh(double x) { return fsinh_dble(&x); }

//------------------------ COS ---------------------
extern "C" {
    inline constexpr float fcos_float(float *x);
    inline constexpr double fcos_dble(double *x);
}

inline constexpr float cos(float x) { return fcos_float(&x); }
inline constexpr double cos(double x) { return fcos_dble(&x); }

//------------------------ COSH ---------------------
extern "C" {
    inline constexpr float fcosh_float(float *x);
    inline constexpr double fcosh_dble(double *x);
}

inline constexpr float cosh(float x) { return fcosh_float(&x); }
inline constexpr double cosh(double x) { return fcosh_dble(&x); }

//------------------------ TAN ---------------------
extern "C" {
    inline constexpr float ftan_float(float *x);
    inline constexpr double ftan_dble(double *x);
}

inline constexpr float tan(float x) { return ftan_float(&x); }
inline constexpr double tan(double x) { return ftan_dble(&x); }

//------------------------ TANH ---------------------
extern "C" {
    inline constexpr float ftanh_float(float *x);
    inline constexpr double ftanh_dble(double *x);
}

inline constexpr float tanh(float x) { return ftanh_float(&x); }
inline constexpr double tanh(double x) { return ftanh_dble(&x); }

//------------------------ ASIN ---------------------
extern "C" {
    inline constexpr float fasin_float(float *x);
    inline constexpr double fasin_dble(double *x);
}

inline constexpr float asin(float x) { return fasin_float(&x); }
inline constexpr double asin(double x) { return fasin_dble(&x); }

//------------------------ ACOS ---------------------
extern "C" {
    inline constexpr float facos_float(float *x);
    inline constexpr double facos_dble(double *x);
}

inline constexpr float acos(float x) { return facos_float(&x); }
inline constexpr double acos(double x) { return facos_dble(&x); }

//------------------------ ATAN ---------------------
extern "C" {
    inline constexpr float fatan_float(float *x);
    inline constexpr double fatan_dble(double *x);
}

inline constexpr float atan(float x) { return fatan_float(&x); }
inline constexpr double atan(double x) { return fatan_dble(&x); }

//------------------------ ATAN2 ---------------------
extern "C" {
    inline constexpr float fatan2_float(float *x, float *y);
    inline constexpr double fatan2_dble(double *x, double *y);
}

inline constexpr float atan2(float x, float y) { return fatan2_float(&x, &y); }
inline constexpr double atan2(double x, double y) { return fatan2_dble(&x, &y); }

//------------------------ SQRT ---------------------
extern "C" {
    inline constexpr float fsqrt_float(float *x);
    inline constexpr double fsqrt_dble(double *x);
}
inline constexpr float sqrt(float x) { return fsqrt_float(&x); }
inline constexpr double sqrt(double x) { return fsqrt_dble(&x); }

//------------------------ EXP ---------------------
extern "C" {
    inline constexpr float fexp_float(float *x);
    inline constexpr double fexp_dble(double *x);
}
inline constexpr float exp(float x) { return fexp_float(&x); }
inline constexpr double exp(double x) { return fexp_dble(&x); }

//------------------------ LOG ---------------------
extern "C" {
    inline constexpr float flog_float(float *x);
    inline constexpr double flog_dble(double *x);
}

inline constexpr float log(float x) { return flog_float(&x); }
inline constexpr double log(double x) { return flog_dble(&x); }

//------------------------ LOG10 ---------------------
extern "C" {
    inline constexpr float flog10_float(float *x);
    inline constexpr double flog10_dble(double *x);
}

inline constexpr float log10(float x) { return flog10_float(&x); }
inline constexpr double log10(double x) { return flog10_dble(&x); }

} // namespace ops::prelude
