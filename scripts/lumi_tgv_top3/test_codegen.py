"""Real TGV translation plus CPU compilation of every block-selection fragment.

Requires translator dependencies and a C++ compiler. HIP API declarations below
model the signatures involved in the regression; this is not a GPU build/run.
"""
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import launcher as L


class TgvCodegenTests(unittest.TestCase):
    def test_every_tgv_hip_and_cuda_block_selection(self):
        python = os.environ.get('OPS_TEST_TRANSLATOR_PYTHON', sys.executable)
        compiler = shutil.which('c++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('A C++ compiler is required')
        with tempfile.TemporaryDirectory(prefix='top3_codegen_') as temp:
            root = Path(temp)
            ops = Path(__file__).resolve().parents[2]
            L.snapshot_ops(ops, root)
            snapshot = root / 'build/OPS'
            app = snapshot / 'apps/c/TGV_StoreNone'
            env = dict(os.environ)
            L.configure_libclang(python, env, io.StringIO())
            result = subprocess.run(
                [python, str(snapshot / 'ops_translator/ops-translator'),
                 '-DOPS_ACC_IGNORE', '-I', str(snapshot / 'ops/c/include'),
                 '--file_paths', 'opensbli.cpp'], cwd=str(app), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            pieces = ['''#include "ops_lib_core.h"
struct dim3 {
  unsigned int x, y, z;
  dim3(unsigned int a, unsigned int b, unsigned int c) : x(a), y(b), z(c) {}
};
struct hipFuncAttributes { int maxThreadsPerBlock; int numRegs; };
int hipFuncGetAttributes(hipFuncAttributes *, const void *);
void hipSafeCall(std::ostream &, int);
''']
            for backend, extension in [('hip', 'cpp'), ('cuda', 'cu')]:
                kernels = sorted((app / backend).glob('*_kernel.' + extension))
                self.assertEqual(len(kernels), 26, backend)
                for path in kernels:
                    source = path.read_text()
                    self.assertIn('dim3 tblock(selected_block.x, selected_block.y, selected_block.z);', source)
                    if backend == 'cuda':
                        self.assertIn('cudaFuncGetAttributes', source)
                        self.assertNotIn('hipFuncGetAttributes', source)
                        continue
                    self.assertIsNone(re.search(r'\b(?:cuda[A-Z]\w*|cutilSafeCall)\b', source), path.name)
                    fragment = re.search(r'    hipFuncAttributes attributes;.*?dim3 tblock\([^;]+;', source, re.S)
                    self.assertIsNotNone(fragment, path.name)
                    kernel = re.search(r'reinterpret_cast<const void \*>\((\w+)\)', fragment.group()).group(1)
                    pieces.append('void ' + kernel + '();\nvoid check_' + kernel +
                                  '(ops_block block, ops_arg *args, int *start_indx, int *end_indx) {\n' +
                                  fragment.group() + '\n}\n')
            probe = root / 'all_tgv_block_selections.cpp'
            probe.write_text('\n'.join(pieces))
            result = subprocess.run([compiler, '-std=c++11', '-fsyntax-only',
                '-I' + str(snapshot / 'ops/c/include'), str(probe)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            # Check the actual make target enables the lazy wrappers. The empty
            # archive is only a prerequisite placeholder for make -n, not a link.
            archive = snapshot/'ops/c/lib/hip/libops_mpi_hip.a'
            archive.parent.mkdir(parents=True, exist_ok=True); archive.touch()
            (app/'.generated').touch()
            plan = subprocess.run(['make', '-n', '-C', str(app),
                'OPS_INSTALL_PATH=' + str(snapshot/'ops'), 'OPS_COMPILER=hip',
                'GPU_VENDOR=AMD', 'HIP_INSTALL_PATH=/unused/rocm',
                'HIPMPICXX=hipcc', 'TARGETS=mpi_hip_tiled', L.build_target({'tiled': True})],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(plan.returncode, 0, plan.stdout)
            self.assertIn('-DOPS_LAZY', plan.stdout)
            self.assertIn('-o opensbli_mpi_hip_tiled', plan.stdout)


if __name__ == '__main__':
    unittest.main()
