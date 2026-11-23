"""Snapshot, toolchain, make commands and the build driver with fake compilers and a fake translator.
The real OPS makefiles and the real apps/c/TGV_StoreNone/Makefile are used; no real compiler, network
or GPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import launcher as L
import sources as S
import testlib as T
from common import read_json, sha

FAKE_COMPILER = r'''#!%s
"""Fake compiler/linker for tests: writes outputs, compiles nothing."""
import os, sys
from pathlib import Path
args = sys.argv[1:]
if '--version' in args:
    print('fake compiler 1.0'); sys.exit(0)
fail = os.environ.get('FAKE_FAIL', '')
if fail and any(fail in a for a in args):
    print('fake compiler: error in', fail); sys.exit(1)
with open(os.environ['FAKE_COMPILE_LOG'], 'a') as f:
    f.write(Path(sys.argv[0]).name + ' ' + ' '.join(args) + '\n')
out = args[args.index('-o') + 1] if '-o' in args else None
if '-c' in args:
    source = Path(args[args.index('-c') + 1])
    out = Path(out or source.stem + '.o')
    out.write_text('OBJ %%s\n' %% source.name)
else:
    inputs = [a for a in args if a.endswith(('.cpp', '.o', '.a'))]
    Path(out).write_text('#!/bin/sh\n# fake executable\n' + ''.join('# %%s\n' %% a for a in inputs))
    Path(out).chmod(0o755)
''' % sys.executable
FAKE_TRANSLATOR = r'''#!%s
"""Fake OPS translator interpreter: answers probes and writes stand-in generated sources."""
import os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:1] == ['-c']:
    sys.exit(0)
assert Path(args[0]).name == 'ops-translator' and '-DOPS_ACC_IGNORE' in args and '-t' not in args, args
files = args[args.index('--file_paths') + 1:]
Path('hip').mkdir(exist_ok=True)
hooks = '' if os.environ.get('FAKE_TRANSLATOR_NO_HOOKS') else 'ops_get_kernel_block_size(1); ops_record_kernel_performance(1, t);'
for name in files:
    stem = name[:-4]
    Path(stem + '_ops.cpp').write_text('// generated from %%s\n' %% name)
    Path('hip/%%s_kernel.cpp' %% stem).write_text('// kernel %%s\n%%s\n' %% (stem, hooks))
Path('hip/hip_kernels.cpp').write_text('// master\n')
with open(os.environ['FAKE_COMPILE_LOG'], 'a') as f:
    f.write('translator ' + ' '.join(args) + '\n')
''' % sys.executable
OPS_FILES = ['opensbli.cpp']          # the whole application is one OpenSBLI-generated file


def fake_toolchain(tmp):
    """Fake ROCm, MPI and compilers; returns the environment for the builder and the translator python."""
    tmp = Path(tmp)
    rocm, mpi, bindir = tmp / 'rocm', tmp / 'mpi', tmp / 'toolbin'
    for folder in (rocm / 'include', rocm / 'llvm/bin', rocm / 'bin', mpi / 'include', mpi / 'lib', mpi / 'gtl/lib', bindir):
        folder.mkdir(parents=True)
    (mpi / 'include/mpi.h').write_text('/* fake */\n')
    (mpi / 'gtl/lib/libmpi_gtl_hsa.so').write_text('fake\n')
    for path in (rocm / 'llvm/bin/clang', rocm / 'llvm/bin/clang++', rocm / 'bin/hipcc', bindir / 'other-cxx'):
        path.write_text(FAKE_COMPILER)
        path.chmod(0o755)
    python = bindir / 'translator-python'
    python.write_text(FAKE_TRANSLATOR)
    python.chmod(0o755)
    env = dict(os.environ, PATH=os.pathsep.join([str(bindir), str(rocm / 'bin'), os.environ['PATH']]),
               HIP_INSTALL_PATH=str(rocm), CRAY_MPICH_DIR=str(mpi), TGV_REQUIRE_GTL='0', TGV_BUILD_JOBS='4',
               FAKE_COMPILE_LOG=str(tmp / 'compile.log'), LOADEDMODULES='PrgEnv-gnu/fake:rocm/fake')
    for key in list(env):
        if (key.startswith('TGV_') and key not in ('TGV_REQUIRE_GTL', 'TGV_BUILD_JOBS')) \
                or key in ('HIP_CLANG_PATH', 'ROCM_PATH', 'PE_MPICH_GTL_DIR_amd_gfx90a', 'MAKEFLAGS', 'MFLAGS'):
            del env[key]
    return env, python


class Temp(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tgv_sources_')
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class PrepareTests(Temp):
    def test_campaign_holds_frozen_copies_and_the_checkout_is_untouched(self):
        ops = Path(L.HERE).parents[1]
        before = subprocess.check_output(['git', '-C', str(ops), 'status', '--porcelain', '-uno'])
        campaign = T.make_campaign(self.tmp)
        self.assertEqual(subprocess.check_output(['git', '-C', str(ops), 'status', '--porcelain', '-uno']), before)
        snap = read_json(campaign / 'source_snapshot.json')
        files = snap['ops']['files_sha256']
        for rel in ('ops/c/Makefile', 'ops/c/src/externlib/ops_util.cpp', 'makefiles/Makefile.c_app', 'makefiles/Makefile.hip',
                    'ops_translator/ops-translator/__main__.py', 'apps/c/TGV_StoreNone/Makefile',
                    'apps/c/TGV_StoreNone/opensbli.cpp', 'apps/c/TGV_StoreNone/opensbliblock00_kernels.h',
                    'apps/c/TGV_StoreNone/defdec_data_set.h'):
            self.assertEqual(sha(campaign / 'source/OPS' / rel), files[rel])
        self.assertTrue(all(k.startswith(S.OPS_PREFIXES) for k in files))
        self.assertFalse(any(k.startswith(('ops/fortran/', 'apps/c/SENGA2', 'apps/c/CloverLeaf', 'apps/c/TGV_StoreAll')) for k in files))
        self.assertFalse(any(k.endswith('_ops.cpp') or '/hip/' in k for k in files if k.startswith(S.APP_DIR)))  # untracked
        self.assertRegex(snap['ops']['commit'], '^[0-9a-f]{40}$')
        config = read_json(campaign / 'campaign.json')
        self.assertEqual((config['account'], config['application'], config['builds']),
                         (T.ACCOUNT, 'apps/c/TGV_StoreNone', {'small': {'grid': 20}, 'full': {'grid': 20}}))
        self.assertEqual(config['consistency_tolerance'], 0.0)
        self.assertEqual(snap['ops']['app_makefile_edit'], 'none')    # the committed Makefile already uses Makefile.c_app
        self.assertEqual(sorted(p.name for p in (campaign / 'launcher').iterdir()), sorted(L.PACKAGE_FILES))
        self.assertEqual(config['package']['files_sha256']['engine.py'], sha(Path(L.HERE) / 'engine.py'))
        S.verify_snapshot(campaign)
        (campaign / 'source/OPS/apps/c/TGV_StoreNone/opensbli.cpp').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed since prepare: source/OPS/apps/c/TGV_StoreNone/opensbli.cpp'):
            S.verify_snapshot(campaign)
        with self.assertRaisesRegex(ValueError, 'Campaign already exists'):
            T.quiet(L.main, ['prepare', '--account', 'a', '--campaign', str(campaign)])

    def test_account_is_required_and_never_assumed(self):
        with patch.dict(os.environ):
            os.environ.pop('SLURM_ACCOUNT', None)
            with self.assertRaisesRegex(ValueError, 'No Slurm account: pass --account'):
                L.main(['prepare', '--campaign', str(self.tmp / 'c')])
            self.assertFalse((self.tmp / 'c').exists())
            os.environ['SLURM_ACCOUNT'] = 'from_env'
            (self.tmp / 'env.sh').write_text('true\n')
            T.quiet(L.main, ['prepare', '--campaign', str(self.tmp / 'c'), '--env-file', str(self.tmp / 'env.sh')])
            self.assertEqual(read_json(self.tmp / 'c/campaign.json')['account'], 'from_env')
        status = subprocess.run([str(Path(L.HERE) / 'run.sh'), 'prepare', '--campaign', str(self.tmp / 'd')],
                                env={k: v for k, v in os.environ.items() if k != 'SLURM_ACCOUNT'},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(status.returncode, 2)
        self.assertIn('ERROR: No Slurm account', status.stderr)

    def test_application_makefile_checks(self):
        text = (Path(L.HERE).parents[1] / S.APP_DIR / 'Makefile').read_text()
        ops_files, other = S.check_app_makefile(text)
        self.assertEqual(ops_files, OPS_FILES)
        self.assertEqual(other, [])
        self.assertEqual((S.APP, S.TARGET, S.EXECUTABLE), ('opensbli', 'opensbli_mpi_hip', 'tgvstorenone_mpi_hip'))
        neither = '\n'.join(l for l in text.split('\n') if 'Makefile.c_app' not in l)
        with self.assertRaisesRegex(ValueError, 'does not include makefiles/Makefile.c_app'):
            S.check_app_makefile(neither)
        with self.assertRaisesRegex(ValueError, 'does not include makefiles/Makefile.c_app'):
            S.check_app_makefile(text.replace('Makefile.c_app', 'Makefile.c_app_legacy'))
        with self.assertRaisesRegex(ValueError, 'APP=opensbli'):
            S.check_app_makefile(text.replace('APP=opensbli', 'APP=other'))

    def test_the_application_must_take_the_step_count_as_fourth_argument(self):
        source = (Path(L.HERE).parents[1] / S.APP_DIR / 'opensbli.cpp').read_text()
        self.assertIn(S.NITER_MARKER, source)
        self.assertIn('strtol(argv[4]', source)                       # NX NY NZ NITER
        self.assertIn('Total Wall time %lf', source)                   # the line the launcher times a run with
        self.assertIn('ops_timing_output', source)                     # the kernel table (call counts, totals)

    def test_no_application_source_is_shipped_in_the_package(self):
        here = Path(L.HERE)
        tracked = subprocess.check_output(['git', '-C', str(here), 'ls-files', '--cached', '--others', '--exclude-standard', '.']).decode().split()
        self.assertTrue(tracked)
        self.assertEqual([p for p in tracked if p.lower().endswith(('.cpp', '.h', '.in', '.o', '.a'))], [])
        self.assertEqual(sorted(p for p in tracked if p.startswith('fixtures/')),
                         ['fixtures/fake_srun.py', 'fixtures/lumi_mi250x_tgvstorenone_best_logmod.csv', 'fixtures/mock_tgv.py'])


class ToolchainTests(Temp):
    def test_defaults_and_overrides(self):
        env, _ = fake_toolchain(self.tmp)
        tc = S.toolchain(env)
        self.assertEqual((tc['cc'], tc['cxx'], tc['mpicxx']), (str(self.tmp / 'rocm/llvm/bin/clang'),
                                                               str(self.tmp / 'rocm/llvm/bin/clang++'),
                                                               str(self.tmp / 'rocm/llvm/bin/clang++')))
        self.assertEqual(tc['hipcc'], str(self.tmp / 'rocm/bin/hipcc'))
        self.assertEqual(tc['mpi_hip_link'], '-L%s/lib -L%s/gtl/lib -lmpi -lmpi_gtl_hsa' % (self.tmp / 'mpi', self.tmp / 'mpi'))
        self.assertEqual(tc['hip_link'], '-L%s/lib -L%s/lib64 -lamdhip64 -lrocm_smi64' % (self.tmp / 'rocm', self.tmp / 'rocm'))
        self.assertEqual((tc['cxxflags'], tc['hipflags']), ('-O3 -fPIC -std=c++11',
                                                            '-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__'))
        custom = S.toolchain(dict(env, TGV_CXXFLAGS='-O2', TGV_HIPFLAGS='-x hip -std=c++17', TGV_LINK_EXTRA='-lfoo',
                                  TGV_HIP_LINK='-lamdhip64', TGV_HIPRAND_LINK='-lhiprand', TGV_MPI_HIP_LINK='-lmpi',
                                  TGV_CC=str(self.tmp / 'toolbin/other-cxx'), TGV_CXX=str(self.tmp / 'toolbin/other-cxx'),
                                  TGV_HIPCC=str(self.tmp / 'toolbin/other-cxx')))
        self.assertEqual((custom['cxxflags'], custom['hipflags'], custom['mpi_hip_link'], custom['hip_link']),
                         ('-O2', '-x hip -std=c++17', '-lmpi -lfoo', '-lamdhip64'))
        self.assertEqual(custom['cxx'], str(self.tmp / 'toolbin/other-cxx'))
        app = S.app_make_args(tc, self.tmp)
        self.assertIn('HIPFLAGS=' + tc['hipflags'], app)             # as in lumi_tgv_top3: one flag set for both
        self.assertIn('HIPFLAGS=' + tc['hipflags'], S.library_make_args(tc, self.tmp))
        for token in ('MPI_LINK=-lmpi', 'TARGETS=mpi_hip', 'CFLAGS=-O3 -fPIC'):
            self.assertIn(token, app)
        self.assertIn('HDF5_INSTALL_PATH=', app)
        with self.assertRaisesRegex(ValueError, 'TGV_HIPCC=no-such-hipcc: command not found'):
            S.toolchain(dict(env, TGV_HIPCC='no-such-hipcc'))
        with self.assertRaisesRegex(ValueError, 'MPI installation not found'):
            S.toolchain({k: v for k, v in env.items() if k != 'CRAY_MPICH_DIR'})
        with self.assertRaisesRegex(ValueError, 'Cannot locate ROCm clang'):
            S.toolchain(dict(env, HIP_CLANG_PATH=str(self.tmp / 'missing')))
        with self.assertRaisesRegex(ValueError, 'Cannot locate libmpi_gtl_hsa'):
            S.toolchain(dict({k: v for k, v in env.items() if k != 'TGV_REQUIRE_GTL'}, TGV_GTL_DIR=str(self.tmp / 'no')))

    def test_make_plans_of_the_library_and_the_application(self):
        campaign = T.make_campaign(self.tmp)
        env, _ = fake_toolchain(self.tmp)
        tc = S.toolchain(env)
        ops = campaign / 'source/OPS'

        def dry(directory, args, target):
            r = subprocess.run(['make', '-n', '-C', str(directory)] + args + [target], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout)
            return r.stdout
        text = dry(ops / 'ops/c', S.library_make_args(tc, ops), 'mpi_hip')
        self.assertNotIn('hdf5', text.lower())
        compiles = [line for line in text.splitlines() if ' -c ' in line]
        self.assertEqual(len(compiles), 12)
        hip = [line for line in compiles if line.startswith(tc['hipcc'] + ' ')]
        self.assertEqual(len(hip), 2)                                         # ops_hip_common, ops_mpi_rt_support_hip
        self.assertTrue(all('--offload-arch=gfx90a' in line for line in hip))
        self.assertTrue(all(line.startswith((tc['cxx'] + ' ', tc['mpicxx'] + ' ', tc['hipcc'] + ' ')) for line in compiles))
        self.assertIn('ar -r %s/ops/c/lib/hip/libops_mpi_hip.a' % ops, text)
        # the application rule, with stand-ins for the generated files and the library
        app = ops / S.APP_DIR
        for name in OPS_FILES:
            (app / (name[:-4] + '_ops.cpp')).write_text('')
        (app / 'hip').mkdir()
        (app / 'hip/hip_kernels.cpp').write_text('')
        (ops / 'ops/c/lib/hip').mkdir(parents=True)
        (ops / 'ops/c/lib/hip/libops_mpi_hip.a').write_text('')
        (app / '.generated').write_text('')
        text = dry(app, ['-o', '.generated'] + S.app_make_args(tc, ops), S.TARGET)
        link = [line for line in text.splitlines() if line.strip().startswith(tc['hipcc'])]
        self.assertEqual(len(link), 1, text)
        line = link[0]
        for token in ('-DOPS_MPI', '-std=c++11', '--offload-arch=gfx90a', './hip/hip_kernels.cpp', 'opensbli_ops.cpp',
                      '-lops_mpi_hip', '-lmpi_gtl_hsa', '-lamdhip64', '-lhiprand', '-o opensbli_mpi_hip',
                      '-I%s/include' % tc['mpi']):
            self.assertIn(token, line)
        self.assertNotIn('ops-translator', text)                             # -o .generated: never regenerated by make

    def test_hook_check(self):
        folder = self.tmp / 'hip'
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, 'No generated kernel wrappers'):
            S.check_hooks(folder)
        (folder / 'a_kernel.cpp').write_text('ops_get_kernel_block_size(); ops_record_kernel_performance();')
        (folder / 'hip_kernels.cpp').write_text('#include "a_kernel.cpp"')
        self.assertEqual(S.check_hooks(folder), 1)
        (folder / 'b_kernel.cpp').write_text('ops_get_kernel_block_size();')
        with self.assertRaisesRegex(ValueError, r'1 of 2 generated HIP kernels lack .*b_kernel.cpp lacks ops_record_kernel_performance'):
            S.check_hooks(folder)

    def test_the_real_generated_kernels_of_the_checkout_carry_the_hooks(self):
        template = (Path(L.HERE).parents[1] / S.HOOK_TEMPLATE).read_text()
        self.assertTrue(all(h in template for h in S.HOOKS))
        self.assertIn('hipFuncAttributes', template)

    def test_gpu_aware_mpi_preflight(self):
        bindir = self.tmp / 'lddbin'
        bindir.mkdir()
        ldd = bindir / 'ldd'
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'])

        def with_output(text):
            ldd.write_text('#!/bin/sh\ncat <<EOF\n%s\nEOF\n' % text)
            ldd.chmod(0o755)
            with patch.dict(os.environ, env):
                return S.gpu_preflight('/x/exe', env)
        self.assertIn('libmpi_gtl_hsa', with_output('libmpi_gtl_hsa.so.0 => /opt/cray/libmpi_gtl_hsa.so.0\nlibc.so.6 => /lib64/libc.so.6'))
        with self.assertRaisesRegex(ValueError, 'not linked against libmpi_gtl_hsa'):
            with_output('libmpi.so => /opt/cray/libmpi.so')
        with self.assertRaisesRegex(ValueError, 'unresolved libraries'):
            with_output('libmpi_gtl_hsa.so.0 => not found')


class BuildTests(Temp):
    def setUp(self):
        Temp.setUp(self)
        self.env, python = fake_toolchain(self.tmp)
        self.campaign = T.make_campaign(self.tmp, grid=40, small_grid=20, extra=['--translator-python', str(python)])

    def build(self, **env):
        return T.quiet(S.Builder(self.campaign, env=dict(self.env, **env)).build)

    def log(self):
        return (self.tmp / 'compile.log').read_text()

    def test_one_build_with_the_real_makefiles(self):
        manifest, text = self.build()
        self.assertIn('Autotuning hooks present in all 1 generated HIP kernels', text)
        self.assertIn('BUILD OK', text)
        root = self.campaign / 'build/hip'
        self.assertEqual(manifest['sha256'], sha(root / 'tgvstorenone_mpi_hip'))
        self.assertEqual((manifest['generated_hip_kernels_with_hooks'], manifest['application']), (1, S.APP_DIR))   # fake translator: 1 file
        self.assertEqual(manifest['modules'], 'PrgEnv-gnu/fake:rocm/fake')
        self.assertTrue((root / 'OPS/ops/c/lib/hip/libops_mpi_hip.a').is_file())
        log = (root / 'build.log').read_text()
        self.assertIn('ops-translator -DOPS_ACC_IGNORE -I', log)
        self.assertIn('make -C %s/OPS/ops/c' % root, log)
        self.assertIn('make -C %s/OPS/apps/c/TGV_StoreNone -o .generated' % root, log)
        compiles = self.log()
        self.assertEqual(sum(l.startswith('translator ') for l in compiles.splitlines()), 1)
        hip_link = [l for l in compiles.splitlines() if l.startswith('hipcc ') and '-o opensbli_mpi_hip' in l]
        self.assertEqual(len(hip_link), 1)
        self.assertIn('./hip/hip_kernels.cpp', hip_link[0])
        self.assertIn('-std=c++11', hip_link[0])
        self.assertEqual(manifest['executable'], 'build/hip/tgvstorenone_mpi_hip')    # make target frozen under the TGV name
        self.assertIn('./hip/hip_kernels.cpp', (root / 'tgvstorenone_mpi_hip').read_text())
        # the frozen snapshot and the user's checkout were not touched
        self.assertFalse((self.campaign / 'source/OPS' / S.APP_DIR / 'hip').exists())
        S.verify_snapshot(self.campaign)
        again, text = self.build()
        self.assertEqual(again['sha256'], manifest['sha256'])
        self.assertIn('already complete', text)
        (root / 'tgvstorenone_mpi_hip').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'frozen executable changed'):
            self.build()
        self.assertEqual(__import__('engine').Campaign(self.campaign).build('small')['sha256'], manifest['sha256'])

    def test_a_library_failure_names_the_step_and_a_rerun_resumes(self):
        with self.assertRaises(ValueError) as caught:
            self.build(FAKE_FAIL='ops_util.cpp')
        message = str(caught.exception)
        self.assertIn('BUILD FAILED at step: OPS library (make mpi_hip)', message)
        self.assertIn(str(self.campaign / 'build/hip/build.log'), message)
        self.assertIn('fake compiler: error in ops_util.cpp', (self.campaign / 'build/hip/build.log').read_text())
        self.assertFalse((self.campaign / 'build/hip/build_manifest.json').exists())
        with self.assertRaisesRegex(ValueError, 'The build is not complete'):
            __import__('engine').Campaign(self.campaign).build('full')
        before = self.log().count('ops_lib_core.cpp')
        self.assertGreaterEqual(before, 1)
        manifest, _ = self.build()
        self.assertEqual(self.log().count('ops_lib_core.cpp'), before)       # finished objects reused
        self.assertEqual(sum(l.startswith('translator ') for l in self.log().splitlines()), 1)                 # code generation not repeated
        self.assertEqual(manifest['sha256'], sha(self.campaign / 'build/hip/tgvstorenone_mpi_hip'))

    def test_application_failure(self):
        with self.assertRaisesRegex(ValueError, r'BUILD FAILED at step: application \(make opensbli_mpi_hip\)'):
            self.build(FAKE_FAIL='hip_kernels.cpp')

    def test_build_fails_when_generated_kernels_lack_the_hooks(self):
        with self.assertRaisesRegex(ValueError, '1 of 1 generated HIP kernels lack the block-size autotuning hooks'):
            self.build(FAKE_TRANSLATOR_NO_HOOKS='1')
        self.assertFalse((self.campaign / 'build/hip/build_manifest.json').exists())
        self.assertFalse((self.campaign / 'build/hip/tgvstorenone_mpi_hip').exists())

    def test_changed_flags_discard_earlier_objects(self):
        with self.assertRaises(ValueError):
            self.build(FAKE_FAIL='hip_kernels.cpp')
        before = self.log().count('ops_lib_core.cpp')
        _, text = self.build(TGV_CXXFLAGS='-O1 -fPIC')
        self.assertIn('Toolchain variables changed', text)
        self.assertEqual(self.log().count('ops_lib_core.cpp'), before + 1)
        self.assertIn(' -O1 -fPIC ', [l for l in self.log().splitlines() if 'ops_lib_core.cpp' in l][-1])

    def test_print_toolchain_lists_every_variable(self):
        status, text = T.quiet(S.print_toolchain, self.env)
        self.assertEqual(status, 0)
        for name, _, _ in S.TOOLCHAIN_HELP:
            self.assertIn(name, text)
        self.assertIn(str(self.tmp / 'rocm/bin/hipcc'), text)
        status, text = T.quiet(S.print_toolchain, dict(self.env, TGV_HIPCC='missing-hipcc'))
        self.assertEqual(status, 1)
        self.assertIn('NOT RESOLVABLE HERE', text)


if __name__ == '__main__':
    unittest.main()
