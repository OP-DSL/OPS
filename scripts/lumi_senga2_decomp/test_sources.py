"""Snapshot, SENGA2 acquisition, source preparation and the build driver with fake compilers
(criteria 19, 20 and the build part of the design). No real compiler, network or GPU."""
import hashlib
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
import os, re, sys
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
    text = source.read_text(errors='replace')
    out.write_text('OBJ %%s\n%%s' %% (source.name, text))
    for flag in args:
        if flag.startswith('-J'):
            for name in re.findall(r'^\s*module\s+(\w+)\s*$', text, re.I | re.M):
                Path(flag[2:], name.lower() + '.mod').write_text('mod\n')
else:
    parts = []
    for a in args:
        if a.startswith('@'):
            parts += [Path(o).read_text() for o in Path(a[1:]).read_text().split()]
        elif a.endswith('.o'):
            parts.append(Path(a).read_text())
    Path(out).write_text('#!/bin/sh\n# fake executable\n' + ''.join(parts))
    Path(out).chmod(0o755)
''' % sys.executable
FAKE_TRANSLATOR = r'''#!%s
"""Fake OPS translator interpreter: answers probes and writes stand-in generated sources."""
import os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:1] == ['-c']:
    sys.exit(0)
files = args[args.index('--file_paths') + 1:]
assert '-t' in args and args[args.index('-t') + 1] == 'f2c_hip' and Path(args[0]).name == 'ops-translator', args
Path('f2c_hip').mkdir(exist_ok=True)
hooks = '' if os.environ.get('FAKE_TRANSLATOR_NO_HOOKS') else 'ops_get_kernel_block_size(1); ops_record_kernel_performance(1, t);'
for name in files:
    stem = name[:-4]
    Path(stem + '_ops.F90').write_text(Path(name).read_text() + '! generated\n')
    Path('f2c_hip/%%s_f2c_kernel.cpp' %% stem).write_text('// kernel %%s\n%%s\n' %% (stem, hooks))
    Path('f2c_hip/%%s_f2c_kernel.F90' %% stem).write_text('MODULE %%s_MODULE\nEND MODULE\n' %% stem.upper())
Path('f2c_hip/f2c_hip_kernels.cpp').write_text('// master\n')
''' % sys.executable


def fake_toolchain(tmp):
    """Fake ROCm, MPI and compilers; returns the environment for the builder."""
    tmp = Path(tmp)
    rocm, mpi, bindir = tmp / 'rocm', tmp / 'mpi', tmp / 'toolbin'
    for folder in (rocm / 'include', rocm / 'llvm/bin', rocm / 'bin', mpi / 'include', mpi / 'lib', mpi / 'gtl/lib', bindir):
        folder.mkdir(parents=True)
    (mpi / 'include/mpi.h').write_text('/* fake */\n')
    (mpi / 'gtl/lib/libmpi_gtl_hsa.so').write_text('fake\n')
    for path in (rocm / 'llvm/bin/clang', rocm / 'llvm/bin/clang++', rocm / 'bin/hipcc', bindir / 'ftn'):
        path.write_text(FAKE_COMPILER)
        path.chmod(0o755)
    python = bindir / 'translator-python'
    python.write_text(FAKE_TRANSLATOR)
    python.chmod(0o755)
    env = dict(os.environ, PATH=os.pathsep.join([str(bindir), str(rocm / 'bin'), os.environ['PATH']]),
               HIP_INSTALL_PATH=str(rocm), CRAY_MPICH_DIR=str(mpi), SENGA_REQUIRE_GTL='0', SENGA_BUILD_JOBS='4',
               FAKE_COMPILE_LOG=str(tmp / 'compile.log'), LOADEDMODULES='PrgEnv-gnu/fake:rocm/fake')
    for key in list(env):
        if (key.startswith('SENGA_') and key not in ('SENGA_REQUIRE_GTL', 'SENGA_BUILD_JOBS')) or key.startswith('HDF5') \
                or key in ('HIP_CLANG_PATH', 'ROCM_PATH'):
            del env[key]
    return env, python


def tree_digest(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and '.git' not in path.parts:
            h.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    return h.hexdigest()


class Temp(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='senga_sources_')
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class PrepareTests(Temp):
    def test_campaign_holds_frozen_copies_and_the_checkouts_are_untouched(self):
        ops = Path(L.HERE).parents[1]
        before = subprocess.check_output(['git', '-C', str(ops), 'status', '--porcelain', '-uno'])
        campaign = T.make_campaign(self.tmp)
        self.assertEqual(subprocess.check_output(['git', '-C', str(ops), 'status', '--porcelain', '-uno']), before)
        snap = read_json(campaign / 'source_snapshot.json')
        for rel in ('ops/fortran/Makefile', 'ops/fortran/src/ops_for_declarations.F90', 'ops/c/src/externlib/ops_util.cpp',
                    'makefiles/Makefile.f_app', 'makefiles/Makefile.hip', 'ops_translator/ops-translator/__main__.py'):
            self.assertEqual(sha(campaign / 'source/OPS' / rel), snap['ops']['files_sha256'][rel])
        self.assertTrue(all(k.startswith(S.OPS_PREFIXES) for k in snap['ops']['files_sha256']))
        self.assertFalse(any('SENGA2' in k for k in snap['ops']['files_sha256']))
        self.assertRegex(snap['ops']['commit'], '^[0-9a-f]{40}$')
        config = read_json(campaign / 'campaign.json')
        self.assertEqual((config['account'], config['hdf5'], config['builds']), (T.ACCOUNT, 'strip', {'small': {'grid': 20}, 'full': {'grid': 20}}))
        self.assertEqual(sorted(p.name for p in (campaign / 'launcher').iterdir()), sorted(L.PACKAGE_FILES))
        self.assertEqual(config['package']['files_sha256']['engine.py'], sha(Path(L.HERE) / 'engine.py'))
        self.assertEqual(tree_digest(self.tmp / 'app_tree'), tree_digest(T.synthetic_senga(self.tmp / 'again')))
        S.verify_snapshot(campaign)
        (campaign / 'source/SENGA2/src/finish.F90').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed since prepare: source/SENGA2/src/finish.F90'):
            S.verify_snapshot(campaign)
        with self.assertRaisesRegex(ValueError, 'Campaign already exists'):
            T.quiet(L.main, ['prepare', '--account', 'a', '--campaign', str(campaign), '--senga-root', str(self.tmp / 'app_tree')])

    def test_account_is_required_and_never_assumed(self):
        with patch.dict(os.environ):
            os.environ.pop('SLURM_ACCOUNT', None)
            with self.assertRaisesRegex(ValueError, 'No Slurm account: pass --account'):
                L.main(['prepare', '--campaign', str(self.tmp / 'c'), '--senga-root', str(self.tmp)])
            self.assertFalse((self.tmp / 'c').exists())
            os.environ['SLURM_ACCOUNT'] = 'from_env'
            tree = T.synthetic_senga(self.tmp / 'tree')
            (self.tmp / 'env.sh').write_text('true\n')
            T.quiet(L.main, ['prepare', '--campaign', str(self.tmp / 'c'), '--senga-root', str(tree), '--env-file', str(self.tmp / 'env.sh')])
            self.assertEqual(read_json(self.tmp / 'c/campaign.json')['account'], 'from_env')
        status = subprocess.run([str(Path(L.HERE) / 'run.sh'), 'prepare', '--campaign', str(self.tmp / 'd')],
                                env={k: v for k, v in os.environ.items() if k != 'SLURM_ACCOUNT'},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(status.returncode, 2)
        self.assertIn('ERROR: No Slurm account', status.stderr)

    def git_tree(self):
        tree = T.synthetic_senga(self.tmp / 'upstream')
        (tree / 'src/stale.o').write_text('junk')
        (tree / 'src/docs').mkdir()
        (tree / 'src/docs/manual.txt').write_text('not needed')
        (tree / '.gitignore').write_text('*.o\n')
        run = lambda *a: subprocess.check_output(['git', '-C', str(tree), '-c', 'user.name=t', '-c', 'user.email=t@example.org'] + list(a))
        run('init', '-q')
        run('add', '-A')
        run('commit', '-q', '-m', 'synthetic')
        return tree, run('rev-parse', 'HEAD').decode().strip()

    def test_clone_at_a_pinned_commit_without_vendoring(self):
        tree, commit = self.git_tree()
        (tree / 'src/finish.F90').write_text('later change\n')
        subprocess.check_output(['git', '-C', str(tree), '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '-am', 'later'])
        (self.tmp / 'env.sh').write_text('true\n')
        argv = ['prepare', '--account', T.ACCOUNT, '--env-file', str(self.tmp / 'env.sh'), '--senga-repo', 'file://' + str(tree)]
        status, text = T.quiet(L.main, argv + ['--campaign', str(self.tmp / 'c1'), '--senga-commit', commit])
        self.assertEqual(status, 0, text)
        config = read_json(self.tmp / 'c1/campaign.json')
        self.assertEqual((config['senga_commit'], config['senga_origin']), (commit, 'file://' + str(tree)))
        copied = self.tmp / 'c1/source/SENGA2'
        self.assertEqual((copied / 'src/finish.F90').read_text(), 'SUBROUTINE finish\nEND SUBROUTINE finish\n')   # the pinned version
        self.assertFalse((copied / 'src/docs').exists() or (copied / 'src/stale.o').exists() or (copied / '.git').exists())
        self.assertFalse((self.tmp / 'c1/source/.senga_clone').exists())
        with self.assertRaisesRegex(ValueError, 'Commit 0123456789abcdef not found'):
            T.quiet(L.main, argv + ['--campaign', str(self.tmp / 'c2'), '--senga-commit', '0123456789abcdef'])
        with self.assertRaisesRegex(ValueError, r'(?s)Cannot clone .*pass its path with --senga-root'):
            T.quiet(L.main, ['prepare', '--account', T.ACCOUNT, '--env-file', str(self.tmp / 'env.sh'), '--campaign',
                             str(self.tmp / 'c3'), '--senga-repo', 'file://' + str(self.tmp / 'nowhere')])

    def test_senga_root_copy_reports_local_modifications(self):
        tree, commit = self.git_tree()
        (tree / 'input/cont.dat').write_text(T.CONT_DAT.replace('1   10   1', '1   50   1'))
        digest = tree_digest(tree)
        info = S.copy_senga(tree, self.tmp / 'copy')
        self.assertEqual((info['commit'], info['modified_files']), (commit, ['input/cont.dat']))
        self.assertEqual(tree_digest(tree), digest)                          # the checkout is never modified
        self.assertFalse((self.tmp / 'copy/src/stale.o').exists())
        with self.assertRaisesRegex(ValueError, 'is not a SENGA2 tree'):
            S.copy_senga(self.tmp, self.tmp / 'copy2')

    def test_defaults_pin_the_repository_and_commit(self):
        self.assertEqual(S.SENGA_REPO, 'https://github.com/vishnu-ncl/SENGA2')
        self.assertTrue(S.SENGA_COMMIT.startswith('a5c76d35') and len(S.SENGA_COMMIT) == 40)
        args = L.parser().parse_args(['prepare'])
        self.assertEqual((args.senga_repo, args.senga_commit, args.senga_root, args.hdf5), (S.SENGA_REPO, S.SENGA_COMMIT, None, 'strip'))
        other = L.parser().parse_args(['prepare', '--senga-root', '/x', '--senga-repo', 'u', '--senga-commit', 'c', '--hdf5', 'keep'])
        self.assertEqual((str(other.senga_root), other.senga_repo, other.senga_commit, other.hdf5), ('/x', 'u', 'c', 'keep'))

    def test_no_application_source_is_shipped_in_the_package(self):
        here = Path(L.HERE)
        tracked = subprocess.check_output(['git', '-C', str(here), 'ls-files', '--cached', '--others', '--exclude-standard', '.']).decode().split()
        self.assertTrue(tracked)
        self.assertEqual([p for p in tracked if p.lower().endswith(('.f90', '.f', '.inc', '.dat', '.h5', '.o', '.a', '.mod'))], [])
        self.assertEqual(sorted(p for p in tracked if p.startswith('fixtures/')),
                         ['fixtures/fake_srun.py', 'fixtures/hpec_mi300a_senga2_best_logmod.csv', 'fixtures/mock_senga2.py'])


class PreparationTests(Temp):
    def test_grid_edit_is_exact_and_fails_on_an_unknown_layout(self):
        path = self.tmp / 'com_senga2.F90'
        path.write_text(T.COM_SENGA)
        S.set_grid(path, 1440)
        text = path.read_text()
        self.assertIn('parameter :: nxglbl=1440, nyglbl=1440, nzglbl=1440\n', text)
        self.assertIn('parameter :: nxsize=1440, nysize=1440, nzsize=1440\n', text)
        self.assertEqual(text.replace('1440', '256'), T.COM_SENGA)           # nothing else changed
        path.write_text(T.COM_SENGA.replace('nxsize=256, nysize=256, nzsize=256', 'nxsize=256'))
        with self.assertRaisesRegex(ValueError, 'Cannot set the grid: expected exactly one "nxsize='):
            S.set_grid(path, 64)
        path.write_text(T.COM_SENGA.replace('ngzmax=nxglbl', 'ngzmax=512'))
        with self.assertRaisesRegex(ValueError, 'cubic grid assumption'):
            S.set_grid(path, 64)

    def test_hdf5_strip_comments_calls_and_stubs_dump_routines(self):
        tree = T.synthetic_senga(self.tmp / 'tree')
        report = S.strip_hdf5_sources(tree / 'src')
        self.assertEqual(sorted(report['stubbed']), ['print_alldats.F90', 'print_dats.F90', 'print_output.F90'])
        self.assertEqual(report['commented'], {'output.F90': 2, 'senga2.F90': 4})
        main = (tree / 'src/senga2.F90').read_text()
        self.assertIn(S.MARK + 'call ops_decl_dat_hdf5(d_drun, senga_grid, 1, "real(kind=8)", &\n' + S.MARK + '"DRUN", \'x.h5\', status)\n', main)
        self.assertIn('    call ops_init(6)\n', main)
        self.assertIn('    call output\n', main)
        self.assertEqual((tree / 'src/print_output.F90').read_text().splitlines()[0], 'SUBROUTINE print_output()')
        (tree / 'src/extra.F90').write_text('  x = ops_is_hdf5_ready()\n')
        with self.assertRaisesRegex(ValueError, 'HDF5 strip left OPS HDF5 references behind'):
            S.strip_hdf5_sources(tree / 'src')

    def test_library_rule_and_make_plan_without_hdf5(self):
        campaign = T.make_campaign(self.tmp)
        env, _ = fake_toolchain(self.tmp)
        ops = campaign / 'source/OPS'
        original = (ops / 'ops/fortran/Makefile').read_text()
        self.assertIn('ops_mpi_hdf5_mpi.o', original[original.index('\nf2c_mpi_hip:'):].split('\n\n')[0])
        tc = S.toolchain(env, 'strip')
        self.assertEqual((tc['f90'], tc['mpif90']), (str(self.tmp / 'toolbin/ftn'),) * 2)
        self.assertEqual(tc['cxx'], str(self.tmp / 'rocm/llvm/bin/clang++'))
        self.assertEqual(tc['mpi_hip_link'], '-L%s/lib -L%s/gtl/lib -lmpi -lmpi_gtl_hsa' % (self.tmp / 'mpi', self.tmp / 'mpi'))

        def plan():
            r = subprocess.run(['make', '-n', '-C', str(ops / 'ops/fortran')] + S.library_make_args(tc, ops) + ['f2c_mpi_hip'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stdout)
            return r.stdout
        self.assertIn('ops_mpi_hdf5.cpp', plan())
        S.strip_hdf5_ops_library(ops)
        text = plan()
        self.assertNotIn('hdf5', text.lower())
        compiles = [line for line in text.splitlines() if ' -c ' in line]
        self.assertEqual(len(compiles), 15)                                   # 12 C++ and 3 Fortran sources
        self.assertTrue(all('-DOPS_FTN' in line for line in compiles if '.cpp' in line))
        hip = [line for line in compiles if 'ops_mpi_rt_support_hip.cpp' in line or 'ops_hip_common.cpp' in line]
        self.assertEqual(len(hip), 2)
        self.assertTrue(all(line.startswith(tc['hipcc'] + ' ') and '--offload-arch=gfx90a' in line for line in hip))
        fortran = [line for line in compiles if '.F90' in line]
        self.assertTrue(all(line.startswith(tc['f90'] + ' -O3 -g -ffree-form -ffree-line-length-none ') and '-fopenmp' not in line
                            and '/mod/hip/f2c_hip' in line for line in fortran))
        self.assertIn('ar -r %s/ops/fortran/lib/hip/libops_f2c_mpi_hip.a' % ops, text)
        with self.assertRaisesRegex(ValueError, 'No f2c_mpi_hip rule'):
            (ops / 'ops/fortran/Makefile').write_text('all:\n')
            S.strip_hdf5_ops_library(ops)

    def test_keep_hdf5_needs_a_parallel_hdf5(self):
        env, _ = fake_toolchain(self.tmp)
        with self.assertRaisesRegex(ValueError, '--hdf5 keep needs a parallel HDF5'):
            S.toolchain(env, 'keep')
        (self.tmp / 'hdf5/include').mkdir(parents=True)
        (self.tmp / 'hdf5/include/hdf5.h').write_text('')
        tc = S.toolchain(dict(env, HDF5_DIR=str(self.tmp / 'hdf5')), 'keep')
        self.assertEqual(tc['hdf5_link'], '-L%s/lib -lhdf5_hl -lhdf5 -lz' % (self.tmp / 'hdf5'))
        args = S.library_make_args(tc, self.tmp)
        self.assertIn('HDF5_INSTALL_PATH=%s' % (self.tmp / 'hdf5'), args)
        self.assertIn('HDF5_INSTALL_PATH=', S.library_make_args(S.toolchain(env, 'strip'), self.tmp))

    def test_every_tool_and_flag_can_be_overridden(self):
        env, _ = fake_toolchain(self.tmp)
        custom = dict(env, SENGA_F90='ftn', SENGA_MPIF90='ftn', SENGA_FFLAGS='-g -fopenmp -M 878 -f free',
                      SENGA_APP_FLAGS='-O3 -N 1023', SENGA_APP_FLAGS_LARGE='-O1 -N 1023', SENGA_CXXFLAGS='-O2',
                      SENGA_HIPFLAGS='-x hip -std=c++17', SENGA_LINK_EXTRA='-lfoo', SENGA_HIP_LINK='-lamdhip64',
                      SENGA_HIPRAND_LINK='-lhiprand', SENGA_MPI_HIP_LINK='-lmpi_gtl_hsa', SENGA_FMOD_FLAG='-module ',
                      SENGA_CC=str(self.tmp / 'toolbin/ftn'), SENGA_CXX=str(self.tmp / 'toolbin/ftn'),
                      SENGA_HIPCC=str(self.tmp / 'toolbin/ftn'), SENGA_LARGE_FILES='b_ops.F90')
        tc = S.toolchain(custom)
        src = self.tmp / 'src'
        (src / 'f2c_hip').mkdir(parents=True)
        (src / 'f2c_hip/k_f2c_kernel.F90').write_text('')
        steps, objects, link = S.app_commands(tc, self.tmp / 'OPS', src, ['o.F90'], ['a_ops.F90', 'b_ops.F90'])
        by = {s['source']: ' '.join(s['command']) for s in steps}
        self.assertEqual([s['source'] for s in steps], ['f2c_hip/f2c_hip_kernels.cpp', 'o.F90', 'f2c_hip/k_f2c_kernel.F90', 'a_ops.F90', 'b_ops.F90'])
        self.assertIn(' -g -fopenmp -M 878 -f free -O3 -N 1023 ', by['a_ops.F90'])
        self.assertIn(' -g -fopenmp -M 878 -f free -O1 -N 1023 ', by['b_ops.F90'])          # reduced optimisation for a large file
        self.assertIn(' -module %s/OPS/ops/fortran/mod/hip/f2c_hip ' % self.tmp, by['o.F90'].replace('-module  ', '-module '))
        self.assertTrue(by['f2c_hip/f2c_hip_kernels.cpp'].startswith('%s -O2 -x hip -std=c++17 -DOPS_FTN -DOPS_MPI ' % (self.tmp / 'toolbin/ftn')))
        self.assertEqual(link[-6:], ['-lmpi_gtl_hsa', '-lamdhip64', '-lhiprand', '-lfoo', '-o', 'senga2_f2c_mpi_hip'])
        default = S.toolchain(env)
        self.assertIn('-std=c++14', ' '.join(S.app_commands(default, self.tmp / 'OPS', src, [], ['a_ops.F90'])[0][0]['command']))
        self.assertEqual(len(default['large_files']), 16)
        self.assertEqual({name for name, _, _ in S.TOOLCHAIN_HELP if name.startswith('SENGA_')} - set(
            'SENGA_CC / SENGA_CXX,SENGA_F90 / SENGA_MPIF90'.split(',')) <= {
                k for k in ('SENGA_HIPCC SENGA_ROCM_ROOT SENGA_MPICXX SENGA_MPI_ROOT SENGA_GTL_DIR SENGA_CXXFLAGS SENGA_HIPFLAGS '
                            'SENGA_FFLAGS SENGA_APP_FLAGS SENGA_APP_FLAGS_LARGE SENGA_LARGE_FILES SENGA_FMOD_FLAG '
                            'SENGA_MPI_HIP_LINK SENGA_HIP_LINK SENGA_HIPRAND_LINK SENGA_LINK_EXTRA SENGA_HDF5_ROOT '
                            'SENGA_HDF5_LINK SENGA_REQUIRE_GTL SENGA_BUILD_JOBS').split()}, True)
        with self.assertRaisesRegex(ValueError, 'SENGA_F90=no-such-compiler: command not found'):
            S.toolchain(dict(env, SENGA_F90='no-such-compiler'))
        with self.assertRaisesRegex(ValueError, 'MPI installation not found'):
            S.toolchain({k: v for k, v in env.items() if k != 'CRAY_MPICH_DIR'})
        with self.assertRaisesRegex(ValueError, 'Cannot locate ROCm clang'):
            S.toolchain(dict(env, HIP_CLANG_PATH=str(self.tmp / 'missing')))

    def test_hook_check(self):
        folder = self.tmp / 'f2c_hip'
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, 'No generated kernel wrappers'):
            S.check_hooks(folder)
        (folder / 'a_f2c_kernel.cpp').write_text('ops_get_kernel_block_size(); ops_record_kernel_performance();')
        (folder / 'f2c_hip_kernels.cpp').write_text('#include "a_f2c_kernel.cpp"')
        self.assertEqual(S.check_hooks(folder), 1)
        (folder / 'b_f2c_kernel.cpp').write_text('ops_get_kernel_block_size();')
        with self.assertRaisesRegex(ValueError, r'1 of 2 generated HIP kernels lack .*b_f2c_kernel.cpp lacks ops_record_kernel_performance'):
            S.check_hooks(folder)

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

    def build(self, name, **env):
        return T.quiet(S.Builder(self.campaign, name, env=dict(self.env, **env)).build)

    def test_two_separate_builds_with_their_own_hashes(self):
        small, text = self.build('small')
        self.assertIn('BUILD OK (small, 20^3)', text)
        self.assertIn('Autotuning hooks present in all 5 generated HIP kernels', text)
        full, _ = self.build('full')
        self.assertNotEqual(small['sha256'], full['sha256'])
        for name, manifest, grid in (('small', small, 20), ('full', full, 40)):
            root = self.campaign / 'build' / name
            self.assertEqual(manifest['sha256'], sha(root / 'senga2_f2c_mpi_hip'))
            self.assertEqual((manifest['grid'], manifest['prepared']['hdf5'], manifest['generated_hip_kernels_with_hooks']), (grid, 'strip', 5))
            self.assertIn('nxglbl=%d, nyglbl=%d, nzglbl=%d' % (grid, grid, grid), (root / 'SENGA2/src/com_senga2.F90').read_text())
            self.assertIn('nxglbl=%d' % grid, (root / 'senga2_f2c_mpi_hip').read_text())      # the grid is compiled in
            self.assertNotIn('hdf5', (root / 'OPS/ops/fortran/Makefile').read_text().split('\nf2c_mpi_hip:')[1].split('\n\n')[0])
            self.assertTrue((root / 'OPS/ops/fortran/lib/hip/libops_f2c_mpi_hip.a').is_file())
            self.assertEqual(manifest['toolchain']['fflags'], S.GNU_FFLAGS)
            self.assertEqual(manifest['modules'], 'PrgEnv-gnu/fake:rocm/fake')
            log = (root / 'build.log').read_text()
            self.assertIn('ops-translator -DOPS_ACC_IGNORE -I', log)
            self.assertIn('make -C %s/OPS/ops/fortran' % root, log)
            self.assertIn('-c senga2_ops.F90 -o senga2_ops_f2c_mpi_hip.o', log)
            self.assertIn('@f2c_mpi_hip_objects.rsp', log)
        # the frozen snapshot and the user's tree were not edited
        self.assertIn('nxglbl=256', (self.campaign / 'source/SENGA2/src/com_senga2.F90').read_text())
        self.assertIn('nxglbl=256', (self.tmp / 'app_tree/src/com_senga2.F90').read_text())
        self.assertIn('ops_mpi_hdf5_mpi.o', (self.campaign / 'source/OPS/ops/fortran/Makefile').read_text())
        again, text = self.build('small')
        self.assertEqual(again['sha256'], small['sha256'])
        self.assertIn('already complete', text)
        (self.campaign / 'build/small/senga2_f2c_mpi_hip').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'frozen small executable changed'):
            self.build('small')

    def test_a_compile_failure_names_the_step_and_a_rerun_resumes(self):
        with self.assertRaises(ValueError) as caught:
            self.build('small', FAKE_FAIL='output_ops.F90')
        message = str(caught.exception)
        self.assertIn('BUILD FAILED (small build) at step: compile output_ops.F90', message)
        self.assertIn('exit status: 1', message)
        self.assertIn(str(self.campaign / 'build/small/build.log'), message)
        self.assertIn('fake compiler: error in output_ops.F90', (self.campaign / 'build/small/build.log').read_text())
        self.assertFalse((self.campaign / 'build/small/build_manifest.json').exists())
        with self.assertRaisesRegex(ValueError, 'The small build is not complete'):
            __import__('engine').Campaign(self.campaign).build('small')
        before = (self.tmp / 'compile.log').read_text().count('-c com_senga2.F90')
        manifest, _ = self.build('small')
        self.assertEqual((self.tmp / 'compile.log').read_text().count('-c com_senga2.F90'), before)   # finished objects reused
        self.assertEqual(manifest['sha256'], sha(self.campaign / 'build/small/senga2_f2c_mpi_hip'))

    def test_failures_in_kernels_and_library_are_reported(self):
        with self.assertRaisesRegex(ValueError, r'BUILD FAILED \(small build\) at step: compile f2c_hip/senga2_f2c_kernel.F90'):
            self.build('small', FAKE_FAIL='f2c_hip/senga2_f2c_kernel.F90')
        with self.assertRaisesRegex(ValueError, r'BUILD FAILED \(full build\) at step: OPS library \(make f2c_mpi_hip\)'):
            self.build('full', FAKE_FAIL='ops_for_rt_support.F90')
        with self.assertRaisesRegex(ValueError, r'BUILD FAILED \(full build\) at step: SENGA2 fftlib \(Makeftl with F90='):
            self.build('full', FAKE_FAIL='b.f')

    def test_build_fails_when_generated_kernels_lack_the_hooks(self):
        with self.assertRaisesRegex(ValueError, '5 of 5 generated HIP kernels lack the block-size autotuning hooks'):
            self.build('small', FAKE_TRANSLATOR_NO_HOOKS='1')
        self.assertFalse((self.campaign / 'build/small/build_manifest.json').exists())
        self.assertFalse((self.campaign / 'build/small/senga2_f2c_mpi_hip').exists())

    def test_changed_flags_discard_earlier_objects(self):
        with self.assertRaises(ValueError):
            self.build('small', FAKE_FAIL='output_ops.F90')
        before = (self.tmp / 'compile.log').read_text().count('-c com_senga2.F90')
        _, text = self.build('small', SENGA_APP_FLAGS='-O1')
        self.assertIn('Toolchain variables changed', text)
        log = (self.tmp / 'compile.log').read_text()
        self.assertEqual(log.count('-c com_senga2.F90'), before + 1)
        self.assertIn(' -O1 ', [l for l in log.splitlines() if '-c com_senga2.F90' in l][-1])

    def test_keep_hdf5_leaves_sources_and_links_hdf5(self):
        (self.tmp / 'hdf5/include').mkdir(parents=True)
        (self.tmp / 'hdf5/include/hdf5.h').write_text('')
        config = self.campaign / 'campaign.json'
        config.write_text(config.read_text().replace('"hdf5": "strip"', '"hdf5": "keep"'))
        manifest, _ = self.build('small', HDF5_DIR=str(self.tmp / 'hdf5'))
        src = self.campaign / 'build/small/SENGA2/src'
        self.assertIn('use OPS_Fortran_hdf5_Declarations', (src / 'senga2.F90').read_text())
        self.assertNotIn(S.MARK, (src / 'senga2.F90').read_text() + (src / 'print_output.F90').read_text())
        self.assertIn('-lhdf5_hl', ' '.join(manifest['link_command']))
        self.assertIn('ops_mpi_hdf5.cpp', (self.tmp / 'compile.log').read_text())

    def test_print_toolchain_lists_every_variable(self):
        status, text = T.quiet(S.print_toolchain, self.env, 'strip')
        self.assertEqual(status, 0)
        for name, _, _ in S.TOOLCHAIN_HELP:
            self.assertIn(name, text)
        self.assertIn(str(self.tmp / 'toolbin/ftn'), text)
        status, text = T.quiet(S.print_toolchain, dict(self.env, SENGA_HIPCC='missing-hipcc'), 'strip')
        self.assertEqual(status, 1)
        self.assertIn('NOT RESOLVABLE HERE', text)


if __name__ == '__main__':
    unittest.main()
