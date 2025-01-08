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
    inline constexpr float ftn_pow_float(float *x, float *e);
    inline constexpr double ftn_pow_double(double *x, double *e);
    inline constexpr int ftn_pow_int(int *x, int *e);
}

inline constexpr float pow(float x, float e) { return ftn_pow_float(&x, &e); }
inline constexpr double pow(double x, double e) { return ftn_pow_double(&x, &e); }
inline constexpr int pow(int x, int e) { return ftn_pow_int(&x, &e); }

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
    inline constexpr float ftn_sin_float(float *x);
    inline constexpr double ftn_sin_double(double *x);
}

inline constexpr float sin(float x) { return ftn_sin_float(&x); }
inline constexpr double sin(double x) { return ftn_sin_double(&x); }

//------------------------ SINH ---------------------
extern "C" {
    inline constexpr float ftn_sinh_float(float *x);
    inline constexpr double ftn_sinh_double(double *x);
}

inline constexpr float sinh(float x) { return ftn_sinh_float(&x); }
inline constexpr double sinh(double x) { return ftn_sinh_double(&x); }

//------------------------ COS ---------------------
extern "C" {
    inline constexpr float ftn_cos_float(float *x);
    inline constexpr double ftn_cos_double(double *x);
}

inline constexpr float cos(float x) { return ftn_cos_float(&x); }
inline constexpr double cos(double x) { return ftn_cos_double(&x); }

//------------------------ COSH ---------------------
extern "C" {
    inline constexpr float ftn_cosh_float(float *x);
    inline constexpr double ftn_cosh_double(double *x);
}

inline constexpr float cosh(float x) { return ftn_cosh_float(&x); }
inline constexpr double cosh(double x) { return ftn_cosh_double(&x); }

//------------------------ TAN ---------------------
extern "C" {
    inline constexpr float ftn_tan_float(float *x);
    inline constexpr double ftn_tan_double(double *x);
}

inline constexpr float tan(float x) { return ftn_tan_float(&x); }
inline constexpr double tan(double x) { return ftn_tan_double(&x); }

//------------------------ TANH ---------------------
extern "C" {
    inline constexpr float ftn_tanh_float(float *x);
    inline constexpr double ftn_tanh_double(double *x);
}

inline constexpr float tanh(float x) { return ftn_tanh_float(&x); }
inline constexpr double tanh(double x) { return ftn_tanh_double(&x); }

//------------------------ ASIN ---------------------
extern "C" {
    inline constexpr float ftn_asin_float(float *x);
    inline constexpr double ftn_asin_double(double *x);
}

inline constexpr float asin(float x) { return ftn_asin_float(&x); }
inline constexpr double asin(double x) { return ftn_asin_double(&x); }

//------------------------ ACOS ---------------------
extern "C" {
    inline constexpr float ftn_acos_float(float *x);
    inline constexpr double ftn_acos_double(double *x);
}

inline constexpr float acos(float x) { return ftn_acos_float(&x); }
inline constexpr double acos(double x) { return ftn_acos_double(&x); }

//------------------------ ATAN ---------------------
extern "C" {
    inline constexpr float ftn_atan_float(float *x);
    inline constexpr double ftn_atan_double(double *x);
}

inline constexpr float atan(float x) { return ftn_atan_float(&x); }
inline constexpr double atan(double x) { return ftn_atan_double(&x); }

//------------------------ ATAN2 ---------------------
extern "C" {
    inline constexpr float ftn_atan2_float(float *x, float *y);
    inline constexpr double ftn_atan2_double(double *x, double *y);
}

inline constexpr float atan2(float x, float y) { return ftn_atan2_float(&x, &y); }
inline constexpr double atan2(double x, double y) { return ftn_atan2_double(&x, &y); }

//------------------------ SQRT ---------------------
extern "C" {
    inline constexpr float ftn_sqrt_float(float *x);
    inline constexpr double ftn_sqrt_double(double *x);
}
inline constexpr float sqrt(float x) { return ftn_sqrt_float(&x); }
inline constexpr double sqrt(double x) { return ftn_sqrt_double(&x); }

//------------------------ EXP ---------------------
extern "C" {
    inline constexpr float ftn_exp_float(float *x);
    inline constexpr double ftn_exp_double(double *x);
}
inline constexpr float exp(float x) { return ftn_exp_float(&x); }
inline constexpr double exp(double x) { return ftn_exp_double(&x); }

//------------------------ LOG ---------------------
extern "C" {
    inline constexpr float ftn_log_float(float *x);
    inline constexpr double ftn_log_double(double *x);
}
inline constexpr float log(float x) { return ftn_log_float(&x); }
inline constexpr double log(double x) { return ftn_log_double(&x); }

//------------------------ LOG10 ---------------------
extern "C" {
    inline constexpr float ftn_log10_float(float *x);
    inline constexpr double ftn_log10_double(double *x);
}
inline constexpr float log10(float x) { return ftn_log10_float(&x); }
inline constexpr double log10(double x) { return ftn_log10_double(&x); }

} // namespace ops::prelude
