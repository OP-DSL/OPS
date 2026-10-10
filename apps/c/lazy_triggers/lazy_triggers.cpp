// lazy_triggers: regression test for lazy execution (OPS_TILING, OPS_LAZY_QUEUE, chain executors).
//
// With lazy execution, ops_par_loop only queues a loop. Every OPS call that lets the host observe
// or modify data therefore has to execute the queue first. For each such call this test
//   1. enqueues producing loops, then calls it immediately (the queue is non-empty), and
//   2. enqueues the same loops, flushes explicitly with ops_execute(), then calls it again.
// Both must give the same answer, and the answer must match the analytic value where one is known.
// Prints "This run is considered PASSED" if everything agrees.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#define OPS_2D
#include <ops_seq_v2.h>
#include <ops_lazy_executor.h>
#include "lazy_triggers_kernels.h"

static int nx = 24, ny = 16;
static double gval = 1.0;
static double cval = 1.0; // a declared constant (not read by any kernel; only its registry entry is tested)

struct Ctx {
  ops_block block;
  ops_dat A, B, C, D, X;
  ops_reduction h;
  ops_stencil S00, Sm10;
  ops_halo_group halos;
};

static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) {
    failures++;
    printf("FAIL: %s\n", what);
  }
}

// Loops that produce the data the triggers read. The global is changed right after it is passed:
// a queued loop must run with the value it was enqueued with.
static bool with_sum = false; // the reduction accumulates until its result is read: queue it only when it will be

static void produce(Ctx &c, double g) {
  gval = g;
  int full[] = {-1, nx + 1, -1, ny + 1};
  int inner[] = {0, nx, 0, ny};
  ops_par_loop(k_init, "init", c.block, 2, full,
               ops_arg_dat(c.A, 1, c.S00, "double", OPS_WRITE), ops_arg_idx());
  ops_par_loop(k_addg, "addg", c.block, 2, full,
               ops_arg_dat(c.A, 1, c.S00, "double", OPS_READ),
               ops_arg_dat(c.B, 1, c.S00, "double", OPS_WRITE),
               ops_arg_gbl(&gval, 1, "double", OPS_READ));
  gval = -1000.0;
  ops_par_loop(k_scale, "scale", c.block, 2, full,
               ops_arg_dat(c.B, 1, c.S00, "double", OPS_READ),
               ops_arg_dat(c.C, 1, c.S00, "double", OPS_WRITE));
  if (with_sum)
    ops_par_loop(k_sum, "sum", c.block, 2, inner,
                 ops_arg_dat(c.C, 1, c.S00, "double", OPS_READ),
                 ops_arg_reduce(c.h, 1, "double", OPS_INC));
}

static double expected_c(int i, int j, double g) { return 2.0 * (i + 100.0 * j + g); }

// ---- triggers: each returns a text fingerprint of what the host saw --------------------------

static std::string fingerprint(const double *v, size_t n) {
  double s = 0;
  for (size_t k = 0; k < n; k++) s += v[k] * (double)(1 + k % 97);
  char buf[96];
  snprintf(buf, sizeof buf, "n=%zu s=%.17g", n, s);
  return buf;
}

static std::string trig_fetch(Ctx &c, double g) {
  int disp[2], size[2];
  ops_dat_get_extents(c.C, 0, disp, size);
  std::vector<double> buf((size_t)size[0] * size[1]);
  ops_dat_fetch_data(c.C, 0, (char *)buf.data());
  for (int j = 0; j < size[1]; j++)
    for (int i = 0; i < size[0]; i++)
      check(buf[i + j * size[0]] == expected_c(disp[0] + i, disp[1] + j, g), "fetch_data value");
  return fingerprint(buf.data(), buf.size());
}

static std::string trig_reduction(Ctx &c, double g) {
  double s = -1;
  ops_reduction_result(c.h, &s);
  double sx = 0, sy = 0;
  for (int i = 0; i < nx; i++) sx += i;
  for (int j = 0; j < ny; j++) sy += j;
  double expected = 2.0 * (ny * sx + 100.0 * nx * sy + (double)nx * ny * g);
  check(fabs(s - expected) < 1e-9 * fabs(expected), "reduction value");
  char buf[64];
  snprintf(buf, sizeof buf, "%.17g", s);
  return buf;
}

static std::string trig_raw(Ctx &c, double g) {
  int disp[2], size[2], stride[2];
  ops_dat_get_raw_metadata(c.C, 0, disp, size, stride, NULL, NULL);
  ops_memspace memspace = OPS_HOST;
  double *p = (double *)ops_dat_get_raw_pointer(c.C, 0, c.S00, &memspace);
  std::vector<double> v;
  for (int j = 0; j < size[1]; j++)
    for (int i = 0; i < size[0]; i++) {
      double x = p[i + j * stride[0]];
      check(x == expected_c(disp[0] + i, disp[1] + j, g), "raw pointer value");
      v.push_back(x);
    }
  ops_dat_release_raw_data(c.C, 0, OPS_READ);
  return fingerprint(v.data(), v.size());
}

#ifndef OPS_MPI
static std::string trig_print(Ctx &c, double g) {
  (void)g;
  const char *file = "lazy_triggers_C.txt";
  remove(file);
  ops_print_dat_to_txtfile(c.C, file);
  std::ifstream in(file);
  std::stringstream ss;
  ss << in.rdbuf();
  check(ss.str().size() > 100, "print_dat_to_txtfile wrote the file");
  return ss.str();
}
#endif

// ops_halo_transfer copies column x=0 of C into the halo column x=-1 of B; k_shift reads it.
static std::string trig_halo(Ctx &c, double g) {
  ops_halo_transfer(c.halos);
  int col[] = {0, 1, 0, ny};
  ops_par_loop(k_shift, "shift", c.block, 2, col,
               ops_arg_dat(c.B, 1, c.Sm10, "double", OPS_READ),
               ops_arg_dat(c.D, 1, c.S00, "double", OPS_WRITE));
  int disp[2], size[2];
  ops_dat_get_extents(c.D, 0, disp, size);
  std::vector<double> buf((size_t)size[0] * size[1]);
  ops_dat_fetch_data(c.D, 0, (char *)buf.data());
  std::vector<double> v;
  for (int j = 0; j < size[1]; j++)
    for (int i = 0; i < size[0]; i++)
      if (disp[0] + i == 0) {
        check(buf[i + j * size[0]] == expected_c(0, disp[1] + j, g), "halo transfer value");
        v.push_back(buf[i + j * size[0]]);
      }
  return fingerprint(v.data(), v.size());
}

typedef std::string (*Trigger)(Ctx &, double);

static void run_trigger(Ctx &c, const char *name, Trigger t, double g) {
  with_sum = (t == trig_reduction);
  produce(c, g);
  std::string immediate = t(c, g); // the queue is not empty here
  produce(c, g);
  ops_execute(); // explicit flush
  std::string flushed = t(c, g);
  with_sum = false;
  bool same = immediate == flushed;
  printf("%-24s %s\n", name, same ? "ok" : "DIFFERS");
  check(same, name);
}

int main(int argc, const char **argv) {
  bool check_exit = true;
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "NO_EXIT_CHECK") == 0) check_exit = false;

  ops_init(argc, argv, 1);

  Ctx c;
  c.block = ops_decl_block(2, "grid");
  int size[] = {nx, ny}, base[] = {0, 0}, d_m[] = {-1, -1}, d_p[] = {1, 1};
  double *xdata = NULL;
#ifndef OPS_MPI
  // user-managed memory: only a loop that really ran can have written it by the time ops_exit returns
  xdata = (double *)calloc((size_t)(nx + 2) * (ny + 2), sizeof(double));
#endif
  c.A = ops_decl_dat(c.block, 1, size, base, d_m, d_p, (double *)NULL, "double", "A");
  c.B = ops_decl_dat(c.block, 1, size, base, d_m, d_p, (double *)NULL, "double", "B");
  c.C = ops_decl_dat(c.block, 1, size, base, d_m, d_p, (double *)NULL, "double", "C");
  c.D = ops_decl_dat(c.block, 1, size, base, d_m, d_p, (double *)NULL, "double", "D");
  c.X = ops_decl_dat(c.block, 1, size, base, d_m, d_p, xdata, "double", "X");
  int s00[] = {0, 0};
  c.S00 = ops_decl_stencil(2, 1, s00, "0,0");
  int sm10[] = {-1, 0};
  c.Sm10 = ops_decl_stencil(2, 1, sm10, "-1,0");
  c.h = ops_decl_reduction_handle(sizeof(double), "double", "sum");

  // halo: column x=0 of C -> halo column x=-1 of B
  int iter[] = {1, ny}, from_base[] = {0, 0}, to_base[] = {-1, 0}, dir[] = {1, 2};
  ops_halo hl = ops_decl_halo(c.C, c.B, iter, from_base, to_base, dir, dir);
  c.halos = ops_decl_halo_group(1, &hl);

  ops_decl_const("cval", 1, "double", &cval);

  ops_partition("");

  run_trigger(c, "ops_dat_fetch_data", trig_fetch, 1.0);
  run_trigger(c, "ops_reduction_result", trig_reduction, 2.0);
  run_trigger(c, "ops_dat_get_raw_pointer", trig_raw, 3.0);
#ifndef OPS_MPI
  run_trigger(c, "ops_print_dat_to_txtfile", trig_print, 4.0);
#endif
  run_trigger(c, "ops_halo_transfer", trig_halo, 5.0);

  int inner[] = {0, nx, 0, ny};

  // Constants: the registry keeps a snapshot, and a changed value executes the loops queued so far.
  {
    OPS_instance *inst = OPS_instance::getOPSInstance();
    const ops_const_entry *e = ops_find_const(inst, "cval");
    check(e && e->dim == 1 && e->size == (int)sizeof(double) && *(const double *)e->snapshot == 1.0 &&
              e->version == 1,
          "constant is registered by ops_decl_const");
    ops_update_const("cval", 1, "double", &cval); // same value: nothing changes
    e = ops_find_const(inst, "cval");
    check(e && e->version == 1, "updating a constant to the same value keeps its version");
    check(ops_find_const(inst, "no_such_constant") == NULL, "unknown constant");

    produce(c, 8.0);
    ops_par_loop(k_copy, "copy to X", c.block, 2, inner,
                 ops_arg_dat(c.C, 1, c.S00, "double", OPS_READ),
                 ops_arg_dat(c.X, 1, c.S00, "double", OPS_WRITE));
    cval = 2.0;
    ops_update_const("cval", 1, "double", &cval); // must run the queued loops first
    e = ops_find_const(inst, "cval");
    check(e && *(const double *)e->snapshot == 2.0 && e->version == 2, "constant snapshot after the update");
#ifndef OPS_MPI
    if (check_exit) {
      bool ok = true;
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          if (xdata[(i + 1) + (j + 1) * (nx + 2)] != expected_c(i, j, 8.0)) ok = false;
      check(ok, "loops queued before ops_update_const were executed by it");
      printf("%-24s %s\n", "ops_update_const", ok ? "ok" : "DIFFERS");
    }
#endif
  }

  // Loops still queued at ops_exit must be executed, not dropped.
  produce(c, 6.0);
  ops_par_loop(k_copy, "copy to X", c.block, 2, inner,
               ops_arg_dat(c.C, 1, c.S00, "double", OPS_READ),
               ops_arg_dat(c.X, 1, c.S00, "double", OPS_WRITE));
  ops_timing_output(std::cout); // must not lose queued loops from its statistics either
  produce(c, 7.0);
  ops_par_loop(k_copy, "copy to X", c.block, 2, inner,
               ops_arg_dat(c.C, 1, c.S00, "double", OPS_READ),
               ops_arg_dat(c.X, 1, c.S00, "double", OPS_WRITE));
  ops_exit();

#ifndef OPS_MPI
  if (check_exit) {
    bool ok = true;
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        if (xdata[(i + 1) + (j + 1) * (nx + 2)] != expected_c(i, j, 7.0)) ok = false;
    check(ok, "loops queued at ops_exit were executed");
    printf("%-24s %s\n", "ops_exit", ok ? "ok" : "DIFFERS");
  }
  free(xdata);
#else
  (void)check_exit;
#endif

  if (failures == 0)
    printf("This run is considered PASSED\n");
  else
    printf("This test is considered FAILED (%d checks)\n", failures);
  return failures != 0;
}
