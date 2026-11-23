"""Source snapshot and the build of TGV_StoreNone (OPS C, MPI + HIP).

TGV_StoreNone lives in the OPS repository (apps/c/TGV_StoreNone): no external application
repository is cloned. Nothing here touches the user's OPS checkout: every build happens in a
copy under the campaign directory. The grid and the number of time steps are command-line
arguments of the application (NX NY NZ NITER), so one executable serves every grid and every
run and there is no input file."""
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from common import (require, sha, atomic, read_json, now, shell_join, query, lock)

APP_DIR = 'apps/c/TGV_StoreNone'
APP = 'opensbli'                       # APP of the application Makefile (OpenSBLI-generated code)
TARGET = 'opensbli_mpi_hip'            # $(APP)_mpi_hip of makefiles/Makefile.c_app
EXECUTABLE = 'tgvstorenone_mpi_hip'    # name of the frozen executable (as in the earlier TGV campaigns)
BUILD_NAME = 'hip'                     # $C/build/hip/: the single build of a campaign
OPS_PREFIXES = ('ops/c/', 'makefiles/', 'ops_translator/', APP_DIR + '/')
OPS_REQUIRED = ('ops/c/Makefile', 'ops/c/src/externlib/ops_util.cpp', 'ops/c/src/core/ops_lib_core.cpp',
                'makefiles/Makefile.common', 'makefiles/Makefile.hip', 'makefiles/Makefile.mpi',
                'makefiles/Makefile.c_app', 'ops_translator/ops-translator/__main__.py',
                'ops_translator/requirements.txt', APP_DIR + '/Makefile', APP_DIR + '/opensbli.cpp',
                APP_DIR + '/opensbliblock00_kernels.h', APP_DIR + '/defdec_data_set.h')
# The C/C++ translator renders the HIP kernels from the cuda template (target.name == "hip").
HOOK_TEMPLATE = 'ops_translator/resources/templates/cpp/cuda/loop_host.cpp.j2'
HOOKS = ('ops_get_kernel_block_size', 'ops_record_kernel_performance')
TRANSLATOR_MODULES = 'clang.cindex, fparser, jinja2, pcpp, sympy, numpy'
# opensbli.cpp must take the number of time steps as its fourth argument (NX NY NZ NITER).
NITER_MARKER = 'requested_niter'


def git(root, *args, **kw):
    """Read-only git query with a timeout (a status of a large tree can hang)."""
    r = subprocess.run(['git', '-C', str(root)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=kw.get('timeout', 120))
    require(r.returncode == 0, 'git %s failed in %s: %s' % (' '.join(args), root, r.stderr.decode(errors='replace').strip()))
    return r.stdout


# --------------------------------------------------------------------------- snapshot
def makefile_list(text, name):
    """Value of a (possibly continued) make variable as a list of words."""
    joined = text.replace('\\\n', ' ')
    found = re.findall(r'^%s[ \t]*=[ \t]*(.*)$' % re.escape(name), joined, re.M)
    require(len(found) == 1, 'Expected exactly one definition of %s in the application Makefile, found %d' % (name, len(found)))
    return found[0].split('#', 1)[0].split()


def check_app_makefile(text):
    """(OPS_FILES, OTHER_FILES) of the application Makefile, which must use the modern translator."""
    require(makefile_list(text, 'APP') == [APP], 'The application Makefile does not define APP=%s' % APP)
    require(re.search(r'^include \$\(OPS_INSTALL_PATH\)/\.\./makefiles/Makefile\.c_app[ \t]*$', text, re.M),
            'The application Makefile does not include makefiles/Makefile.c_app (the modern OPS translator)')
    return makefile_list(text, 'OPS_FILES'), makefile_list(text, 'OTHER_FILES')


def snapshot_ops(ops, campaign):
    """Copy the tracked OPS files the build needs (library, makefiles, translator, application);
    the checkout itself is never modified."""
    ops = Path(ops)
    target = campaign / 'source/OPS'
    require(not target.exists(), 'OPS snapshot already exists in %s' % campaign)
    require((ops / '.git').exists(), '%s is not a Git checkout of OPS (use --ops-root)' % ops)
    listing = [p for p in git(ops, 'ls-files', '-z').decode().split('\0') if p]
    for rel in OPS_REQUIRED + (HOOK_TEMPLATE,):
        require(rel in listing and (ops / rel).is_file(),
                'Required OPS file is missing or not tracked by Git: %s. Is this the feature/gpu_block_size branch?' % rel)
    template = (ops / HOOK_TEMPLATE).read_text()
    require(all(h in template for h in HOOKS),
            'The OPS translator in %s does not emit the block-size autotuning hooks for C/C++ HIP kernels '
            '(%s lacks %s). Update the checkout to feature/gpu_block_size.' % (ops, HOOK_TEMPLATE, ' / '.join(HOOKS)))
    util = (ops / 'ops/c/src/externlib/ops_util.cpp').read_text()
    require('OPS_DISABLE_CSV_LOGS' in util and 'OPS_CSV_LOG_LEVEL' in util and 'OPS_BLOCKSIZE_CSV' in util,
            'The OPS runtime lacks the rank-selective CSV controls (OPS_DISABLE_CSV_LOGS / OPS_CSV_LOG_LEVEL)')
    check_app_makefile((ops / APP_DIR / 'Makefile').read_text())
    require(NITER_MARKER in (ops / APP_DIR / 'opensbli.cpp').read_text(),
            '%s/opensbli.cpp does not accept the number of time steps as its fourth argument (NX NY NZ NITER); '
            'the launcher sets the run length with it. Update the checkout to feature/gpu_block_size.' % APP_DIR)
    copied = {}
    for rel in listing:
        if not rel.startswith(OPS_PREFIXES):
            continue
        source = ops / rel
        require(source.is_file() and not source.is_symlink(), 'Unsupported or missing tracked file: %s' % source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(dest))
        copied[rel] = sha(dest)
    patch = git(ops, 'diff', 'HEAD', '--', *[p.rstrip('/') for p in OPS_PREFIXES])
    (campaign / 'source/ops_local_changes.patch').write_bytes(patch)
    return dict(root=str(ops), commit=git(ops, 'rev-parse', 'HEAD').decode().strip(),
                uncommitted_changes_in_snapshot=bool(patch.strip()), files_sha256=copied,
                application=APP_DIR, app_makefile_edit='none')


def verify_snapshot(campaign):
    snap = read_json(campaign / 'source_snapshot.json')
    for rel, digest in snap['ops']['files_sha256'].items():
        require(sha(campaign / 'source/OPS' / rel) == digest, 'Frozen source changed since prepare: source/OPS/%s' % rel)
    return snap


def check_hooks(generated):
    """Every generated HIP kernel wrapper must call both autotuning hooks."""
    kernels = sorted(p for p in Path(generated).glob('*_kernel.cpp'))
    require(kernels, 'No generated kernel wrappers (*_kernel.cpp) under %s: code generation did not run' % generated)
    missing = {}
    for path in kernels:
        text = path.read_text(errors='replace')
        lacking = [h for h in HOOKS if h not in text]
        if lacking:
            missing[path.name] = lacking
    require(not missing,
            '%d of %d generated HIP kernels lack the block-size autotuning hooks (first: %s). The executable would '
            'ignore the tuned blocks; no tuning catalog will be fabricated. Use an OPS checkout whose translator '
            'emits ops_get_kernel_block_size and ops_record_kernel_performance (feature/gpu_block_size).'
            % (len(missing), len(kernels), ', '.join('%s lacks %s' % (k, '+'.join(v)) for k, v in sorted(missing.items())[:5])))
    return len(kernels)


# --------------------------------------------------------------------------- toolchain
TOOLCHAIN_HELP = [
    ('TGV_HIPCC', 'HIP compiler driver: OPS HIP sources and the application (all kernels in one hipcc call)',
     'hipcc found on PATH'),
    ('TGV_ROCM_ROOT', 'ROCm installation root', '$HIP_INSTALL_PATH, $ROCM_PATH or the parent of hipcc'),
    ('TGV_CC / TGV_CXX', 'host C/C++ compilers for the OPS library', 'ROCm clang / clang++ ($HIP_CLANG_PATH or <rocm>/llvm/bin)'),
    ('TGV_MPICXX', 'C++ compiler for the MPI sources of OPS', '$TGV_CXX'),
    ('TGV_MPI_ROOT', 'MPI installation (include/mpi.h, lib/)', '$CRAY_MPICH_DIR'),
    ('TGV_GTL_DIR', 'directory of libmpi_gtl_hsa', '$PE_MPICH_GTL_DIR_amd_gfx90a or <mpi>/gtl/lib'),
    ('TGV_CXXFLAGS', 'host C++ flags (OPS library and application)', '-O3 -fPIC -std=c++11'),
    ('TGV_CFLAGS', 'host C flags (OPS library)', '-O3 -fPIC'),
    ('TGV_HIPFLAGS', 'HIP flags (OPS HIP sources, the application and its kernels)',
     '-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__'),
    ('TGV_MPI_HIP_LINK', 'MPI + GPU transport libraries', '-L<mpi>/lib -L<gtl> -lmpi -lmpi_gtl_hsa'),
    ('TGV_HIP_LINK', 'HIP runtime libraries', '-L<rocm>/lib -L<rocm>/lib64 -lamdhip64 -lrocm_smi64'),
    ('TGV_HIPRAND_LINK', 'hipRAND library', '-L<rocm>/lib -lhiprand'),
    ('TGV_LINK_EXTRA', 'extra link flags appended to the MPI link flags', '(empty)'),
    ('TGV_REQUIRE_GTL', 'set to 0 to skip the libmpi_gtl_hsa check of the executable (not for LUMI runs)', '1'),
    ('TGV_BUILD_JOBS', 'parallel make jobs for the OPS library', '8'),
]


def locate_clang(env, rocm):
    override = env.get('HIP_CLANG_PATH')
    candidates = [Path(override)] if override else [rocm / 'llvm/bin', rocm / 'lib/llvm/bin', rocm / 'bin']
    for directory in candidates:
        cc, cxx = directory / 'clang', directory / 'clang++'
        if all(p.is_file() and os.access(str(p), os.X_OK) for p in (cc, cxx)):
            return str(cc), str(cxx)   # not resolved: the executable name selects C++
    raise ValueError('Cannot locate ROCm clang/clang++ (looked in %s). Set HIP_CLANG_PATH to the ROCm llvm/bin '
                     'directory, or set TGV_CC and TGV_CXX.' % ', '.join(str(c) for c in candidates))


def toolchain(env):
    """Resolve every compiler and flag from the environment (see TOOLCHAIN_HELP)."""
    def tool(name, default):
        value = env.get(name) or default
        require(value, '%s is not set and no default could be found' % name)
        path = shutil.which(value, path=env.get('PATH'))
        require(path, '%s=%s: command not found. Load the toolchain in the environment file or set %s.' % (name, value, name))
        return str(Path(path).absolute())

    hipcc = tool('TGV_HIPCC', 'hipcc')
    rocm = Path(env.get('TGV_ROCM_ROOT') or env.get('HIP_INSTALL_PATH') or env.get('ROCM_PATH')
                or str(Path(hipcc).resolve().parent.parent))
    require((rocm / 'include').is_dir(), 'ROCm root %s has no include/ directory; set TGV_ROCM_ROOT' % rocm)
    if env.get('TGV_CC') and env.get('TGV_CXX'):
        cc, cxx = tool('TGV_CC', ''), tool('TGV_CXX', '')
    else:
        cc, cxx = locate_clang(env, rocm)
    mpi = Path(env.get('TGV_MPI_ROOT') or env.get('CRAY_MPICH_DIR') or '')
    require(str(mpi) != '.' and (mpi / 'include/mpi.h').is_file(),
            'MPI installation not found: set TGV_MPI_ROOT (or load cray-mpich so that CRAY_MPICH_DIR/include/mpi.h exists)')
    gtl = env.get('TGV_GTL_DIR') or env.get('PE_MPICH_GTL_DIR_amd_gfx90a', '')
    gtl = gtl[2:] if gtl.startswith('-L') else gtl
    if not gtl:
        found = sorted(mpi.glob('gtl/lib/libmpi_gtl_hsa.so*'))
        gtl = str(found[0].parent) if found else ''
    require_gtl = env.get('TGV_REQUIRE_GTL', '1') != '0'
    require(not require_gtl or (gtl and Path(gtl).is_dir()),
            'Cannot locate libmpi_gtl_hsa (GPU-aware Cray MPICH). Load craype-accel-amd-gfx90a or set TGV_GTL_DIR.')
    mpi_hip_link = env.get('TGV_MPI_HIP_LINK') or ('-L%s/lib %s-lmpi%s' % (
        mpi, ('-L%s ' % gtl) if gtl else '', ' -lmpi_gtl_hsa' if gtl else ''))
    extra = env.get('TGV_LINK_EXTRA', '')
    return dict(
        ops_compiler='hip', hipcc=hipcc, rocm=str(rocm), cc=cc, cxx=cxx, mpicxx=tool('TGV_MPICXX', cxx),
        mpi=str(mpi), gtl=gtl, require_gtl=require_gtl,
        cxxflags=env.get('TGV_CXXFLAGS', '-O3 -fPIC -std=c++11'), cflags=env.get('TGV_CFLAGS', '-O3 -fPIC'),
        hipflags=env.get('TGV_HIPFLAGS', '-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__'),
        mpi_hip_link=(mpi_hip_link + ' ' + extra).strip(),
        hip_link=env.get('TGV_HIP_LINK') or '-L%s/lib -L%s/lib64 -lamdhip64 -lrocm_smi64' % (rocm, rocm),
        hiprand_link=env.get('TGV_HIPRAND_LINK') or '-L%s/lib -lhiprand' % rocm,
        jobs=int(env.get('TGV_BUILD_JOBS', '8')))


def common_make_args(tc, ops):
    """Variables of both make calls. OPS_COMPILER=hip selects makefiles/Makefile.hip, which sets no
    host compiler or flags itself, so every compiler and flag is given on the command line."""
    return ['OPS_COMPILER=' + tc['ops_compiler'], 'OPS_INSTALL_PATH=%s' % (ops / 'ops'),
            'CC=' + tc['cc'], 'CXX=' + tc['cxx'], 'MPICC=' + tc['cc'], 'MPICXX=' + tc['mpicxx'],
            'MPICPP=' + tc['mpicxx'], 'HIPCC=' + tc['hipcc'], 'HIPMPICXX=' + tc['hipcc'],
            'HIP_INSTALL_PATH=' + tc['rocm'], 'ROCM_PATH=' + tc['rocm'], 'MPI_INSTALL_PATH=' + tc['mpi'],
            'MPI_INC=%s/include' % tc['mpi'], 'MPI_LIB=%s/lib' % tc['mpi'], 'I_MPI_INC=-I%s/include' % tc['mpi'],
            'MPI_LINK=-lmpi', 'AMD_ARCH=gfx90a', 'GPU_VENDOR=AMD', 'HDF5_INSTALL_PATH=', 'TARGETS=mpi_hip',
            'CXXFLAGS=' + tc['cxxflags'], 'CFLAGS=' + tc['cflags'], 'HIPFLAGS=' + tc['hipflags']]


def library_make_args(tc, ops):
    """`make -C ops/c mpi_hip` -> ops/c/lib/hip/libops_mpi_hip.a"""
    return common_make_args(tc, ops)


def app_make_args(tc, ops):
    """`make -C apps/c/TGV_StoreNone opensbli_mpi_hip` (rule $(APP)_mpi_hip of Makefile.c_app)."""
    return common_make_args(tc, ops) + [
        'MPI_HIP_LINK=' + tc['mpi_hip_link'], 'HIP_LINK=' + tc['hip_link'], 'HIPRAND=' + tc['hiprand_link']]


def translator_interpreter(explicit=None):
    candidates = [explicit] if explicit else [sys.executable, 'python3'] + ['python3.%d' % m for m in range(14, 7, -1)]
    for candidate in candidates:
        executable = shutil.which(str(candidate))
        if not executable:
            continue
        r = query([executable, '-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'])
        if r['status'] == 'OK':
            return str(Path(executable).absolute())   # keep virtualenv paths unresolved
    raise ValueError('The OPS translator needs Python >= 3.8 (the launcher itself runs on 3.6). Load a newer Python '
                     'in the environment file (on LUMI: module load cray-python), or pass --translator-python '
                     '/path/to/venv/bin/python3 that already has ops_translator/requirements.txt installed.')


def gpu_preflight(exe, env=None):
    r = subprocess.run(['ldd', str(exe)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    require(r.returncode == 0, 'ldd failed on %s:\n%s' % (exe, r.stdout))
    missing = [line.strip() for line in r.stdout.splitlines() if 'not found' in line]
    require(not missing, 'Executable %s has unresolved libraries:\n%s' % (exe, '\n'.join(missing)))
    require('libmpi_gtl_hsa' in r.stdout,
            'Executable %s is not linked against libmpi_gtl_hsa, so -gpudirect cannot work with Cray MPICH. Check '
            'TGV_MPI_HIP_LINK / TGV_GTL_DIR and that craype-accel-amd-gfx90a is loaded.' % exe)
    return r.stdout


# --------------------------------------------------------------------------- build
class Builder(object):
    """The one build of a campaign: $C/build/hip/{OPS/, build.log, build_manifest.json, tgvstorenone_mpi_hip}.
    make produces apps/c/TGV_StoreNone/opensbli_mpi_hip; it is frozen under the name tgvstorenone_mpi_hip."""

    def __init__(self, campaign, env=None):
        self.campaign = Path(campaign)
        self.name = BUILD_NAME
        self.config = read_json(self.campaign / 'campaign.json')
        self.root = self.campaign / 'build' / BUILD_NAME
        self.ops = self.root / 'OPS'
        self.app = self.ops / APP_DIR
        self.env = dict(os.environ if env is None else env)
        self.log_path = self.root / 'build.log'
        self.out = None

    def say(self, message):
        print(message, flush=True)
        if self.out:
            self.out.write(message + '\n')
            self.out.flush()

    def run(self, command, cwd=None, what=None, env=None):
        command = [str(c) for c in command]
        self.out.write('\n$ ' + (('cd %s && ' % cwd) if cwd else '') + shell_join(command) + '\n')
        self.out.flush()
        r = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env or self.env, stdout=self.out,
                           stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise ValueError('BUILD FAILED at step: %s\n  command: %s\n  directory: %s\n  exit status: %d\n'
                             '  full output: %s (the failing command is the last "$ " line)\n'
                             'Fix the toolchain variables (run.sh build --print-toolchain lists them) and rerun the '
                             'same build command; finished objects are reused.'
                             % (what or command[0], shell_join(command), cwd or os.getcwd(), r.returncode, self.log_path))

    def stage_sources(self):
        marker = self.root / 'prepared.json'
        if marker.is_file():
            return read_json(marker)
        snap = verify_snapshot(self.campaign)
        if self.ops.exists():
            shutil.rmtree(str(self.ops))
        shutil.copytree(str(self.campaign / 'source/OPS'), str(self.ops))
        report = dict(prepared_utc=now(), ops_commit=snap['ops']['commit'], application=APP_DIR,
                      app_makefile_sha256=sha(self.app / 'Makefile'), source_edits='none')
        atomic(marker, report)
        self.say('Copied the frozen OPS sources (library, makefiles, translator, %s) into %s' % (APP_DIR, self.ops))
        return report

    def translator(self):
        python = self.config.get('translator_python') or ''
        if not python:
            venv = self.campaign / 'build/translator-env'
            if not (venv / 'bin/python3').exists():
                self.run([translator_interpreter(), '-m', 'venv', str(venv)], what='create the translator virtualenv')
                self.run([str(venv / 'bin/python3'), '-m', 'pip', 'install', '-r',
                          str(self.ops / 'ops_translator/requirements.txt')],
                         what='install the OPS translator requirements (needs network access to PyPI; '
                              'otherwise pass --translator-python at prepare time)')
            python = str(venv / 'bin/python3')
        python = translator_interpreter(python)
        probe = subprocess.run([python, '-c', 'import ' + TRANSLATOR_MODULES], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
        require(probe.returncode == 0, 'The translator Python %s lacks the OPS translator requirements (%s):\n%s'
                % (python, TRANSLATOR_MODULES, probe.stdout.strip()))
        return python

    def generate(self, python):
        """The .generated rule of makefiles/Makefile.c_app, run with the translator Python."""
        ops_files, other = check_app_makefile((self.app / 'Makefile').read_text())
        for name in ops_files + other:
            require((self.app / name).is_file(), 'The application Makefile lists %s but the file is missing' % name)
        generated = [f[:-4] + '_ops.cpp' for f in ops_files]
        marker = self.app / '.generated'
        if not (self.root / 'generated.json').is_file():
            if marker.exists():
                marker.unlink()
            self.say('Generating OPS sources for %d files (all targets, as makefiles/Makefile.c_app does)' % len(ops_files))
            self.run([python, str(self.ops / 'ops_translator/ops-translator'), '-DOPS_ACC_IGNORE', '-I',
                      str(self.ops / 'ops/c/include') + '/', '--file_paths'] + ops_files,
                     cwd=self.app, what='OPS code generation (ops-translator)')
            for name in generated + ['hip/hip_kernels.cpp']:
                require((self.app / name).is_file(), 'Code generation did not produce %s; see %s' % (name, self.log_path))
            atomic(self.root / 'generated.json', dict(utc=now(), files=generated))
        marker.write_text(now() + '\n')        # newer than every source: make never regenerates
        count = check_hooks(self.app / 'hip')
        self.say('Autotuning hooks present in all %d generated HIP kernels' % count)
        return count

    def reset_if_toolchain_changed(self, tc):
        """Objects built with other compilers or flags are never mixed into the executable."""
        signature = hashlib.sha256(repr(sorted((k, str(v)) for k, v in tc.items() if k != 'jobs')).encode()).hexdigest()
        flag_file = self.root / '.toolchain_signature'
        if flag_file.is_file() and flag_file.read_text().strip() == signature:
            return
        if flag_file.is_file():
            self.say('Toolchain variables changed since the last attempt: discarding the earlier OPS library and objects')
        for sub in ('obj', 'lib'):
            path = self.ops / 'ops/c' / sub
            if path.exists():
                shutil.rmtree(str(path))
        exe = self.app / TARGET
        if exe.exists():
            exe.unlink()
        flag_file.write_text(signature + '\n')

    def library(self, tc):
        lib = self.ops / 'ops/c/lib' / tc['ops_compiler'] / 'libops_mpi_hip.a'
        if not lib.is_file():
            self.say('Building the OPS C MPI+HIP library (make mpi_hip)')
            self.run(['make', '-C', str(self.ops / 'ops/c'), '-j', str(tc['jobs'])] + library_make_args(tc, self.ops) +
                     ['mpi_hip'], what='OPS library (make mpi_hip)')
        require(lib.is_file(), 'OPS library build finished but %s is missing; see %s' % (lib, self.log_path))
        return lib

    def application(self, tc):
        command = ['make', '-C', str(self.app), '-o', '.generated'] + app_make_args(tc, self.ops) + [TARGET]
        exe = self.app / TARGET
        if not exe.is_file():
            self.say('Compiling and linking %s (one hipcc call with every generated kernel; several minutes)' % TARGET)
            self.run(command, what='application (make %s)' % TARGET)
        require(exe.is_file(), 'make %s finished but %s is missing; see %s' % (TARGET, exe, self.log_path))
        return command

    def build(self):
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / 'build_manifest.json'
        with lock(self.root / '.lock'):
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                require(sha(self.root / EXECUTABLE) == manifest['sha256'],
                        'The frozen executable changed after it was built (%s). Start a new campaign.' % (self.root / EXECUTABLE))
                print('Build already complete: %s sha256=%s' % (self.root / EXECUTABLE, manifest['sha256']), flush=True)
                return manifest
            with self.log_path.open('a') as out:
                self.out = out
                out.write('\n==== build started %s ====\n' % now())
                prepared = self.stage_sources()
                tc = toolchain(self.env)
                out.write('Toolchain:\n' + ''.join('  %s = %s\n' % (k, v) for k, v in sorted(tc.items())))
                self.env.update(OPS_INSTALL_PATH=str(self.ops / 'ops'), OPS_COMPILER=tc['ops_compiler'],
                                HIP_INSTALL_PATH=tc['rocm'], ROCM_PATH=tc['rocm'], MPICH_GPU_SUPPORT_ENABLED='1')
                python = self.translator()
                kernels = self.generate(python)
                self.reset_if_toolchain_changed(tc)
                self.library(tc)
                command = self.application(tc)
                exe = self.app / TARGET
                ldd = gpu_preflight(exe, self.env) if tc['require_gtl'] else 'SKIPPED (TGV_REQUIRE_GTL=0)'
                frozen = self.root / EXECUTABLE
                shutil.copy2(str(exe), str(frozen))
                versions = {k: query([tc[k], '--version'])['stdout'].strip().splitlines()[:2] for k in ('hipcc', 'cxx')}
                manifest = dict(build=self.name, application=APP_DIR, executable=str(frozen.relative_to(self.campaign)),
                                sha256=sha(frozen), built_utc=now(), toolchain=tc, compiler_versions=versions,
                                prepared=prepared, generated_hip_kernels_with_hooks=kernels,
                                translator_python=python, make_command=command, gtl_linked=tc['require_gtl'], ldd=ldd,
                                modules=self.env.get('LOADEDMODULES', ''),
                                source_snapshot_sha256=sha(self.campaign / 'source_snapshot.json'))
                atomic(manifest_path, manifest)
                self.say('BUILD OK: %s sha256=%s' % (frozen, manifest['sha256']))
                self.out = None
                return manifest


def print_toolchain(env):
    print('Build variables (set them in the environment file given to prepare with --env-file, or export them):\n')
    for name, meaning, default in TOOLCHAIN_HELP:
        print('  %-24s %s\n  %-24s default: %s' % (name, meaning, '', default))
    print('\nResolved in the current environment:')
    try:
        for key, value in sorted(toolchain(env).items()):
            print('  %-16s %s' % (key, value))
    except ValueError as error:
        print('  NOT RESOLVABLE HERE: %s' % error)
        return 1
    return 0
