// CPU-only runtime regression: link against libops_seq.a and invoke once per mode.
// Example: ./block_selection_smoke 2 -OPS_AUTOTUNE_MODE=2
// Mode 2 expects OPS_BLOCKSIZE_CSV with kernel 42 -> (8,4,1).
#include <ops_lib_core.h>
#include <ops_internal2.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

static bool equals(ops::dim3 block, unsigned x, unsigned y, unsigned z) {
  return block.x == x && block.y == y && block.z == z;
}

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  const int mode = std::atoi(argv[1]);
  const bool retired = mode >= 3 && mode <= 7;
  try {
    try {
      ops_init(argc, argv, 2);
    } catch (const OPSException &error) {
      if (retired) {
        check(std::string(error.what()).find("no longer supported") != std::string::npos,
              "Unexpected mode validation error");
        std::cout << "PASS: retired mode rejected\n";
        return 0;
      }
      throw;
    }
    check(!retired, "Retired mode was accepted");
    auto *instance = OPS_instance::getOPSInstance();
    check(instance->OPS_autotune_mode == mode, "Requested mode changed");
    instance->OPS_block_size_x = 32;
    instance->OPS_block_size_y = 4;
    instance->OPS_block_size_z = 1;
    int range[] = {0, 64, 0, 64, 0, 64};
    if (mode == 1) {
      // Synthetic timings exercise candidate exploration and convergence.
      for (int i = 0; i < 2000; ++i) {
        auto block = ops_get_kernel_block_size(42, 3, range, 0, nullptr, 1024, 32);
        check(block.x * block.y * block.z <= 1024, "Invalid candidate");
        ops_record_kernel_performance(42, equals(block, 8, 4, 1) ? 1.0 : 10.0);
      }
      check(equals(ops_get_kernel_block_size(42, 3, range, 0, nullptr, 1024, 32), 8, 4, 1),
            "Autotuning did not retain the fastest candidate");
    } else if (mode == 2) {
      check(equals(ops_get_kernel_block_size(42, 3, range, 0, nullptr, 1024, 32), 8, 4, 1),
            "Precomputed block was not loaded");
      check(equals(ops_get_kernel_block_size(43, 3, range, 0, nullptr, 1024, 32), 32, 4, 1),
            "Missing-kernel default fallback changed");
    } else {
      check(equals(ops_get_kernel_block_size(42, 3, range, 0, nullptr, 1024, 32), 32, 4, 1),
            "Default block selection changed");
    }
    ops_exit();
    std::cout << "PASS: mode " << mode << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
