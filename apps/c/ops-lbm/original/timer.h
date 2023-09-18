#if __cplusplus>=201103L
#include <chrono>
#else
#ifdef __unix__
#include <sys/time.h>
#elif defined (_WIN32) || defined(WIN32)
#include <windows.h>
#endif
#endif

void timer(double *cpu, double *et) {
#if __cplusplus>=201103L
  (void)cpu;
  *et = (double)std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count()/1000000.0;
#else
#ifdef __unix__
  (void)cpu;
  struct timeval t;

  gettimeofday(&t, (struct timezone *)0);
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
#elif defined(_WIN32) || defined(WIN32)
  (void)cpu;
  DWORD time = GetTickCount();
  *et = ((double)time)/1000.0;
#endif
#endif
}