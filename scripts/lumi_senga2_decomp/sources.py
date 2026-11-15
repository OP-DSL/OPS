"""Source snapshot, SENGA2 acquisition, source preparation and the two builds.

Nothing here touches the user's OPS or SENGA2 checkouts: every edit happens in
copies under the campaign directory."""
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from common import (require, sha, atomic, read_json, now, shell_join, query, lock)

SENGA_REPO = 'https://github.com/vishnu-ncl/SENGA2'
SENGA_COMMIT = 'a5c76d3597117a514da93ce2e8a434674cd5bf04'
OPS_PREFIXES = ('ops/c/', 'ops/fortran/', 'makefiles/', 'ops_translator/')
OPS_REQUIRED = ('ops/fortran/Makefile', 'ops/fortran/src/ops_for_declarations.F90',
                'ops/c/src/externlib/ops_util.cpp', 'ops/c/src/core/ops_lib_core.cpp',
                'makefiles/Makefile.common', 'makefiles/Makefile.hip', 'makefiles/Makefile.mpi',
                'makefiles/Makefile.cuda', 'ops_translator/ops-translator/__main__.py',
                'ops_translator/requirements.txt')
HOOK_TEMPLATE = 'ops_translator/resources/templates/fortran/f2c_cuda/loop_f2c_host.cpp.j2'
HOOKS = ('ops_get_kernel_block_size', 'ops_record_kernel_performance')
SENGA_SUBDIRS = ('fftlib', 'rndlib', 'parlib')
SENGA_REQUIRED = ('src/com_senga2.F90', 'src/senga2.F90', 'src/Makefile', 'src/fftlib/Makeftl',
                  'src/rndlib/Makeran', 'src/parlib/Makepar', 'input/cont.dat', 'input/chem.dat',
                  'input/diff.dat', 'input/radn.dat', 'TGV_species.dat', 't_initial.dat')
RUN_INPUTS = ('input/chem.dat', 'input/diff.dat', 'input/radn.dat', 'TGV_species.dat', 't_initial.dat')
SENGA_JUNK = ('.o', '.mod', '.a', '.so', '.rsp')
# Generated files compiled at reduced optimisation in SENGA2 src/lumi_compile.sh (Cray ftn).
LARGE_FILES = ('ops_data_init_ops.F90 bounds_ops.F90 boundt_ops.F90 bountt_ops.F90 temper_ops.F90 '
               'chrate_ops.F90 indata_ops.F90 tempin_ops.F90 d2fdxy_ops.F90 d2fdxz_ops.F90 d2fdyz_ops.F90 '
               'rhscal_ops.F90 rhsvel_ops.F90 lincom_ops.F90 fincom_ops.F90 flamin_ops.F90')
EXECUTABLE = 'senga2_f2c_mpi_hip'
# gfortran defaults, as in makefiles/Makefile.gnu (the OPS Fortran sources need free-form long lines).
GNU_FFLAGS = '-O3 -g -ffree-form -ffree-line-length-none -fopenmp'
HDF5_STUBS = {'print_dats.F90': 'print_dats', 'print_alldats.F90': 'print_alldats',
              'print_output.F90': 'print_output'}
HDF5_LIB_OBJECTS = ('$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o ',
                    '$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_hip.o ')
MARK = '! lumi_senga2_decomp --hdf5 strip: '


def git(root, *args, **kw):
    """Read-only git query with a timeout (a status of a large tree can hang)."""
    r = subprocess.run(['git', '-C', str(root)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=kw.get('timeout', 120))
    require(r.returncode == 0, 'git %s failed in %s: %s' % (' '.join(args), root, r.stderr.decode(errors='replace').strip()))
    return r.stdout


def blob_sha(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


# --------------------------------------------------------------------------- snapshots
def snapshot_ops(ops, campaign):
    """Copy the tracked OPS files the build needs; the checkout itself is never modified."""
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
            'The OPS translator in %s does not emit the block-size autotuning hooks for Fortran f2c HIP kernels '
            '(%s lacks %s). Update the checkout to feature/gpu_block_size (commit b5fcd54ed or later).'
            % (ops, HOOK_TEMPLATE, ' / '.join(HOOKS)))
    util = (ops / 'ops/c/src/externlib/ops_util.cpp').read_text()
    require('OPS_DISABLE_CSV_LOGS' in util and 'OPS_CSV_LOG_LEVEL' in util and 'OPS_BLOCKSIZE_CSV' in util,
            'The OPS runtime lacks the rank-selective CSV controls (OPS_DISABLE_CSV_LOGS / OPS_CSV_LOG_LEVEL)')
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
                uncommitted_changes_in_snapshot=bool(patch.strip()), files_sha256=copied)


def senga_wanted(rel):
    parts = rel.split('/')
    if parts[0] == 'src':
        return len(parts) == 2 or parts[1] in SENGA_SUBDIRS
    return parts[0] == 'input' or rel in ('TGV_species.dat', 't_initial.dat')


def copy_senga(root, target):
    """Copy the build and run inputs of a SENGA2 tree. With Git: tracked files only."""
    root = Path(root)
    require((root / 'src/com_senga2.F90').is_file(),
            '%s is not a SENGA2 tree (src/com_senga2.F90 not found). Pass the repository root to --senga-root.' % root)
    require(not target.exists(), 'SENGA2 copy already exists: %s' % target)
    info = dict(root=str(root), commit='', modified_files=[], untracked_generated_skipped=0)
    if (root / '.git').exists():
        info['commit'] = git(root, 'rev-parse', 'HEAD').decode().strip()
        listing = [p for p in git(root, 'ls-files', '-z').decode().split('\0') if p]
        head = {}
        for line in git(root, 'ls-tree', '-r', '-z', 'HEAD').decode().split('\0'):
            if line:
                meta, name = line.split('\t', 1)
                head[name] = meta.split()[2]
        names = [p for p in listing if senga_wanted(p)]
    else:
        names, head = [], None
        for path in sorted(root.rglob('*')):
            rel = path.relative_to(root).as_posix()
            if not path.is_file() or not senga_wanted(rel):
                continue
            if path.suffix in SENGA_JUNK or path.name.endswith('_ops.F90') or path.name in ('.generated', 'main.cpp'):
                info['untracked_generated_skipped'] += 1
                continue
            names.append(rel)
    copied = {}
    for rel in names:
        source = root / rel
        require(source.is_file() and not source.is_symlink(), 'Missing SENGA2 file: %s' % source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(dest))
        copied[rel] = sha(dest)
        if head is not None and head.get(rel) != blob_sha(dest):
            info['modified_files'].append(rel)
    for rel in SENGA_REQUIRED:
        require(rel in copied, 'SENGA2 tree lacks %s (looked in %s)' % (rel, root))
    info['files_sha256'] = copied
    return info


def fetch_senga(campaign, senga_root, repo, commit, log=print):
    """--senga-root: copy that tree. Otherwise clone `repo` and check out the pinned commit."""
    target = campaign / 'source/SENGA2'
    if senga_root:
        info = copy_senga(Path(senga_root).expanduser().resolve(), target)
        info.update(origin='--senga-root', requested_commit=commit)
        if info['commit'] and not info['commit'].startswith(commit) and not commit.startswith(info['commit']):
            log('WARNING: --senga-root is at commit %s, not the pinned %s. Recorded; pass --senga-commit to silence.'
                % (info['commit'], commit))
        if info['modified_files']:
            log('WARNING: %d SENGA2 files differ from their commit: %s' % (len(info['modified_files']),
                                                                          ', '.join(info['modified_files'][:8])))
        return info
    require(shutil.which('git'), 'git is required to clone SENGA2; alternatively pass --senga-root')
    clone = campaign / 'source/.senga_clone'
    if clone.exists():
        shutil.rmtree(str(clone))
    log('Cloning %s (commit %s)' % (repo, commit))
    r = subprocess.run(['git', 'clone', '--quiet', repo, str(clone)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    require(r.returncode == 0,
            'Cannot clone %s:\n%s\nIf this node has no network access, clone the repository elsewhere, check out '
            'commit %s and pass its path with --senga-root.' % (repo, r.stdout.strip(), commit))
    r = subprocess.run(['git', '-C', str(clone), 'checkout', '--quiet', '--detach', commit], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    require(r.returncode == 0, 'Commit %s not found in %s:\n%s' % (commit, repo, r.stdout.strip()))
    actual = git(clone, 'rev-parse', 'HEAD').decode().strip()
    require(actual.startswith(commit) or commit.startswith(actual), 'Checked out %s instead of %s' % (actual, commit))
    info = copy_senga(clone, target)
    info.update(origin=repo, requested_commit=commit, root=repo)
    shutil.rmtree(str(clone))
    return info


def verify_snapshot(campaign):
    snap = read_json(campaign / 'source_snapshot.json')
    for name, key in (('OPS', 'ops'), ('SENGA2', 'senga')):
        for rel, digest in snap[key]['files_sha256'].items():
            require(sha(campaign / 'source' / name / rel) == digest,
                    'Frozen source changed since prepare: source/%s/%s' % (name, rel))
    return snap


# --------------------------------------------------------------------------- preparation
def makefile_list(text, name):
    """Value of a (possibly continued) make variable as a list of words."""
    joined = text.replace('\\\n', ' ')
    found = re.findall(r'^%s[ \t]*=[ \t]*(.*)$' % re.escape(name), joined, re.M)
    require(len(found) == 1, 'Expected exactly one definition of %s in the SENGA2 Makefile, found %d' % (name, len(found)))
    return found[0].split('#', 1)[0].split()


def set_grid(path, n):
    """Global and local sizes in com_senga2.F90 (compile-time parameters)."""
    text = Path(path).read_text()
    for names in (('nxglbl', 'nyglbl', 'nzglbl'), ('nxsize', 'nysize', 'nzsize')):
        pattern = r'(integer\(kind=4\),[ \t]*parameter[ \t]*::[ \t]*)%s=\d+,[ \t]*%s=\d+,[ \t]*%s=\d+' % names
        text, count = re.subn(pattern, lambda m: m.group(1) + ', '.join('%s=%d' % (k, n) for k in names), text)
        require(count == 1, 'Cannot set the grid: expected exactly one "%s=..., %s=..., %s=..." parameter line in %s, '
                'found %d. The SENGA2 source layout differs from commit %s.' % (names + (path, count, SENGA_COMMIT[:8])))
    for derived in ('ngzmax=nxglbl', 'nszmax=nxsize'):
        require(derived in text.replace(' ', ''), 'Expected "%s" in %s (cubic grid assumption)' % (derived, path))
    Path(path).write_text(text)
    check = Path(path).read_text()
    require(len(re.findall(r'=%d\b' % n, check)) >= 6, 'Grid edit of %s did not take effect' % path)


CONT_GRID = 'Global domain size (nx,ny,nz)'
CONT_PROC = 'No. of processors (x,y,z)'
CONT_STEP = 'Time step; start step'
CONT_DUMP = 'Intervals between dumps'


def cont_dat(template, grid, steps):
    """Run-control file for one run: grid, step count, and no dump/report inside the steps.

    Step line:   tstep ntime1 ntime nstpsw -> '<tstep> 1 <steps> <nstpsw>' (tstep and switch kept).
    Dump line:   ntdump ntrept ntstat fmt  -> all three intervals = steps + 1, so that
                 MOD(itime, interval) is never 0 for itime = 1..steps. The initial
                 `call output` at itime = 0 cannot be suppressed by any interval."""
    lines = template.splitlines()
    seen = set()
    interval = steps + 1

    def after(index, label):
        require(index + 1 < len(lines) and lines[index + 1].split(), 'cont.dat: no value line after "%s"' % label)
        return lines[index + 1].split()

    for i, line in enumerate(lines):
        text = line.strip()
        if text == CONT_GRID:
            require(len(after(i, text)) == 3, 'cont.dat: malformed grid line')
            lines[i + 1] = ' %d    %d    %d' % (grid, grid, grid)
            seen.add('grid')
        elif text == CONT_PROC:
            require(after(i, text) == ['1', '1', '1'], 'cont.dat: SENGA2 processor grid must stay 1 1 1 (OPS partitions)')
            seen.add('proc')
        elif text.startswith(CONT_STEP):
            parts = after(i, text)
            require(len(parts) == 4, 'cont.dat: expected "tstep ntime1 ntime nstpsw", got %r' % lines[i + 1])
            lines[i + 1] = ' %s    1   %d   %s' % (parts[0], steps, parts[3])
            seen.add('step')
        elif text.startswith(CONT_DUMP):
            parts = after(i, text)
            require(len(parts) == 4, 'cont.dat: expected "ntdump ntrept ntstat fmt", got %r' % lines[i + 1])
            lines[i + 1] = ' %d    %d    %d      %s' % (interval, interval, interval, parts[3])
            seen.add('dump')
    missing = {'grid', 'proc', 'step', 'dump'} - seen
    require(not missing, 'cont.dat template lacks the expected section(s): %s' % ', '.join(sorted(missing)))
    return '\n'.join(lines) + '\n'


def cont_values(text):
    """Parsed (grid, ntime1, steps, ntdump, ntrept, ntstat) of a cont.dat, for checks and tests."""
    lines = text.splitlines()
    out = {}
    for i, line in enumerate(lines):
        t = line.strip()
        if t == CONT_GRID:
            out['grid'] = [int(v) for v in lines[i + 1].split()]
        elif t.startswith(CONT_STEP):
            p = lines[i + 1].split()
            out.update(tstep=p[0], ntime1=int(p[1]), steps=int(p[2]), nstpsw=int(p[3]), step_line=lines[i + 1])
        elif t.startswith(CONT_DUMP):
            p = lines[i + 1].split()
            out.update(ntdump=int(p[0]), ntrept=int(p[1]), ntstat=int(p[2]), dump_line=lines[i + 1])
    return out


def strip_hdf5_sources(src):
    """Reproduce the HPEC no-HDF5 preparation on a copy: stub the three dump
    routines, comment the OPS HDF5 module use and every `call ops_*hdf5*`."""
    report = dict(stubbed=[], commented={})
    for name, routine in HDF5_STUBS.items():
        path = src / name
        require(path.is_file(), 'Expected SENGA2 file %s for the HDF5 strip' % path)
        path.write_text('SUBROUTINE %s()\n  %sHDF5 output disabled\nEND SUBROUTINE %s\n' % (routine, MARK, routine))
        report['stubbed'].append(name)
    use_re = re.compile(r'^\s*use\s+OPS_Fortran_hdf5_Declarations\b', re.I)
    call_re = re.compile(r'^\s*call\s+ops_\w*hdf5\w*', re.I)
    for path in sorted(src.glob('*.F90')):
        lines = path.read_text().split('\n')
        count, continuing = 0, False
        for i, line in enumerate(lines):
            code = line.split('!', 1)[0].rstrip()
            if continuing or use_re.match(line) or call_re.match(line):
                lines[i] = MARK + line.strip()
                count += 1
                continuing = code.endswith('&')
        if count:
            path.write_text('\n'.join(lines))
            report['commented'][path.name] = count
    left = []
    for path in sorted(src.glob('*.F90')):
        for number, line in enumerate(path.read_text().split('\n'), 1):
            code = line.split('!', 1)[0]
            if re.search(r'OPS_Fortran_hdf5_Declarations|\bops_\w*hdf5\w*', code, re.I):
                left.append('%s:%d: %s' % (path.name, number, line.strip()))
    require(not left, 'HDF5 strip left OPS HDF5 references behind:\n' + '\n'.join(left[:20]))
    return report


def strip_hdf5_ops_library(ops):
    """Drop the three HDF5 objects from the f2c_mpi_hip library rule (campaign copy only)."""
    makefile = ops / 'ops/fortran/Makefile'
    text = makefile.read_text()
    start = text.find('\nf2c_mpi_hip:')
    require(start >= 0, 'No f2c_mpi_hip rule in %s' % makefile)
    end = text.find('\n\n', start + 1)
    end = len(text) if end < 0 else end
    rule = text[start:end]
    for needle in HDF5_LIB_OBJECTS + (HDF5_LIB_OBJECTS[1].strip(),):
        rule = rule.replace(needle, '')
    require('hdf5' not in rule.lower(), 'HDF5 objects are still listed in the f2c_mpi_hip rule of %s' % makefile)
    require(rule.count('ops_for_declarations_f2c_hip.o') == 2 and rule.count('ops_for_math_support_f2c_hip.o') == 2,
            'Unexpected f2c_mpi_hip rule in %s after the HDF5 strip' % makefile)
    makefile.write_text(text[:start] + rule + text[end:])


def prepare_tree(senga, ops, grid, hdf5):
    """All source edits of one build, on copies. Returns a report for the build manifest."""
    require(hdf5 in ('strip', 'keep'), '--hdf5 must be strip or keep')
    src = senga / 'src'
    set_grid(src / 'com_senga2.F90', grid)
    report = dict(grid=grid, hdf5=hdf5)
    if hdf5 == 'strip':
        report['hdf5_sources'] = strip_hdf5_sources(src)
        strip_hdf5_ops_library(ops)
    cont_dat((senga / 'input/cont.dat').read_text(), grid, 1)   # fail early on an unexpected template
    return report


def check_hooks(generated):
    """Every generated f2c HIP kernel wrapper must call both autotuning hooks."""
    kernels = sorted(p for p in generated.glob('*_kernel.cpp'))
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
    ('SENGA_HIPCC', 'HIP compiler driver for OPS HIP sources and the generated kernels', 'hipcc found on PATH'),
    ('SENGA_ROCM_ROOT', 'ROCm installation root', '$HIP_INSTALL_PATH, $ROCM_PATH or the parent of hipcc'),
    ('SENGA_CC / SENGA_CXX', 'host C/C++ compilers for the OPS library', 'ROCm clang / clang++ ($HIP_CLANG_PATH or <rocm>/llvm/bin)'),
    ('SENGA_MPICXX', 'C++ compiler for the MPI sources of OPS', '$SENGA_CXX'),
    ('SENGA_F90 / SENGA_MPIF90', 'Fortran compiler / MPI Fortran compiler and linker', 'ftn'),
    ('SENGA_MPI_ROOT', 'MPI installation (include/mpi.h, lib/)', '$CRAY_MPICH_DIR'),
    ('SENGA_GTL_DIR', 'directory of libmpi_gtl_hsa', '$PE_MPICH_GTL_DIR_amd_gfx90a or <mpi>/gtl/lib'),
    ('SENGA_CXXFLAGS', 'host C++ flags', '-O3 -fPIC -std=c++11'),
    ('SENGA_HIPFLAGS', 'HIP flags (c++11 becomes c++14 for the generated kernels, as in Makefile.f_app)',
     '-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__'),
    ('SENGA_FFLAGS', 'Fortran flags for the OPS Fortran library and the application (as makefiles/Makefile.gnu)',
     '-O3 -g -ffree-form -ffree-line-length-none -fopenmp'),
    ('SENGA_APP_FLAGS', 'extra Fortran flags of the application, including the optimisation level',
     '-g -cpp -O3 -fallow-argument-mismatch'),
    ('SENGA_APP_FLAGS_LARGE', 'if set, replaces SENGA_APP_FLAGS for the files in SENGA_LARGE_FILES', '(unset)'),
    ('SENGA_LARGE_FILES', 'large generated files that may need a lower optimisation level', 'the 16 files of src/lumi_compile.sh'),
    ('SENGA_FMOD_FLAG', 'Fortran option that sets the module output directory', '-J'),
    ('SENGA_MPI_HIP_LINK', 'MPI + GPU transport libraries', '-L<mpi>/lib -L<gtl> -lmpi -lmpi_gtl_hsa'),
    ('SENGA_HIP_LINK', 'HIP runtime libraries', '-L<rocm>/lib -L<rocm>/lib64 -lamdhip64 -lrocm_smi64'),
    ('SENGA_HIPRAND_LINK', 'hipRAND library', '-L<rocm>/lib -lhiprand'),
    ('SENGA_LINK_EXTRA', 'extra link flags (OpenMP runtime for omp_get_wtime, C++ runtime)', '-fopenmp -lstdc++'),
    ('SENGA_HDF5_ROOT', 'parallel HDF5 root, only with --hdf5 keep', '$HDF5_INSTALL_PATH, $HDF5_DIR or $HDF5_ROOT'),
    ('SENGA_HDF5_LINK', 'HDF5 link flags, only with --hdf5 keep', '-L<hdf5>/lib -lhdf5_hl -lhdf5 -lz'),
    ('SENGA_REQUIRE_GTL', 'set to 0 to skip the libmpi_gtl_hsa check of the executable (not for LUMI runs)', '1'),
    ('SENGA_BUILD_JOBS', 'parallel compile jobs', '8'),
]


def locate_clang(env, rocm):
    override = env.get('HIP_CLANG_PATH')
    candidates = [Path(override)] if override else [rocm / 'llvm/bin', rocm / 'lib/llvm/bin', rocm / 'bin']
    for directory in candidates:
        cc, cxx = directory / 'clang', directory / 'clang++'
        if all(p.is_file() and os.access(str(p), os.X_OK) for p in (cc, cxx)):
            return str(cc), str(cxx)   # not resolved: the executable name selects C++
    raise ValueError('Cannot locate ROCm clang/clang++ (looked in %s). Set HIP_CLANG_PATH to the ROCm llvm/bin '
                     'directory, or set SENGA_CC and SENGA_CXX.' % ', '.join(str(c) for c in candidates))


def toolchain(env, hdf5='strip'):
    """Resolve every compiler and flag from the environment (see TOOLCHAIN_HELP)."""
    def tool(name, default):
        value = env.get(name) or default
        require(value, '%s is not set and no default could be found' % name)
        path = shutil.which(value, path=env.get('PATH'))
        require(path, '%s=%s: command not found. Load the toolchain in the environment file or set %s.' % (name, value, name))
        return str(Path(path).absolute())

    hipcc = tool('SENGA_HIPCC', 'hipcc')
    rocm = Path(env.get('SENGA_ROCM_ROOT') or env.get('HIP_INSTALL_PATH') or env.get('ROCM_PATH')
                or str(Path(hipcc).resolve().parent.parent))
    require((rocm / 'include').is_dir(), 'ROCm root %s has no include/ directory; set SENGA_ROCM_ROOT' % rocm)
    if env.get('SENGA_CC') and env.get('SENGA_CXX'):
        cc, cxx = tool('SENGA_CC', ''), tool('SENGA_CXX', '')
    else:
        cc, cxx = locate_clang(env, rocm)
    mpi = Path(env.get('SENGA_MPI_ROOT') or env.get('CRAY_MPICH_DIR') or '')
    require(str(mpi) != '.' and (mpi / 'include/mpi.h').is_file(),
            'MPI installation not found: set SENGA_MPI_ROOT (or load cray-mpich so that CRAY_MPICH_DIR/include/mpi.h exists)')
    gtl = env.get('SENGA_GTL_DIR') or env.get('PE_MPICH_GTL_DIR_amd_gfx90a', '')
    gtl = gtl[2:] if gtl.startswith('-L') else gtl
    if not gtl:
        found = sorted(mpi.glob('gtl/lib/libmpi_gtl_hsa.so*'))
        gtl = str(found[0].parent) if found else ''
    require_gtl = env.get('SENGA_REQUIRE_GTL', '1') != '0'
    require(not require_gtl or (gtl and Path(gtl).is_dir()),
            'Cannot locate libmpi_gtl_hsa (GPU-aware Cray MPICH). Load craype-accel-amd-gfx90a or set SENGA_GTL_DIR.')
    tc = dict(
        ops_compiler='hip', hipcc=hipcc, rocm=str(rocm), cc=cc, cxx=cxx,
        mpicxx=tool('SENGA_MPICXX', cxx), f90=tool('SENGA_F90', 'ftn'), mpif90=tool('SENGA_MPIF90', 'ftn'),
        mpi=str(mpi), gtl=gtl, require_gtl=require_gtl,
        cxxflags=env.get('SENGA_CXXFLAGS', '-O3 -fPIC -std=c++11'),
        hipflags=env.get('SENGA_HIPFLAGS', '-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__'),
        fflags=env.get('SENGA_FFLAGS', GNU_FFLAGS),
        app_flags=env.get('SENGA_APP_FLAGS', '-g -cpp -O3 -fallow-argument-mismatch'),
        app_flags_large=env.get('SENGA_APP_FLAGS_LARGE', ''),
        large_files=(env.get('SENGA_LARGE_FILES') or LARGE_FILES).split(),
        fmod_flag=env.get('SENGA_FMOD_FLAG', '-J'),
        mpi_hip_link=env.get('SENGA_MPI_HIP_LINK') or ('-L%s/lib %s-lmpi%s' % (
            mpi, ('-L%s ' % gtl) if gtl else '', ' -lmpi_gtl_hsa' if gtl else '')),
        hip_link=env.get('SENGA_HIP_LINK') or '-L%s/lib -L%s/lib64 -lamdhip64 -lrocm_smi64' % (rocm, rocm),
        hiprand_link=env.get('SENGA_HIPRAND_LINK') or '-L%s/lib -lhiprand' % rocm,
        link_extra=env.get('SENGA_LINK_EXTRA', '-fopenmp -lstdc++'),
        jobs=int(env.get('SENGA_BUILD_JOBS', '8')), hdf5=hdf5, hdf5_root='', hdf5_link='')
    if hdf5 == 'keep':
        root = env.get('SENGA_HDF5_ROOT') or env.get('HDF5_INSTALL_PATH') or env.get('HDF5_DIR') or env.get('HDF5_ROOT') or ''
        require(root and (Path(root) / 'include/hdf5.h').is_file(),
                '--hdf5 keep needs a parallel HDF5: load cray-hdf5-parallel or set SENGA_HDF5_ROOT (include/hdf5.h not found)')
        tc.update(hdf5_root=root, hdf5_link=env.get('SENGA_HDF5_LINK') or '-L%s/lib -lhdf5_hl -lhdf5 -lz' % root)
    return tc


def library_make_args(tc, ops):
    """Variables for `make -C ops/fortran f2c_mpi_hip` (OPS_COMPILER=hip sets no compiler itself)."""
    mod = ops / 'ops/fortran/mod' / tc['ops_compiler']
    j = tc['fmod_flag']
    args = ['OPS_COMPILER=' + tc['ops_compiler'], 'OPS_INSTALL_PATH=%s' % (ops / 'ops'),
            'CC=' + tc['cc'], 'CXX=' + tc['cxx'], 'MPICC=' + tc['cc'], 'MPICXX=' + tc['mpicxx'],
            'MPICPP=' + tc['mpicxx'], 'FC=' + tc['f90'], 'F90=' + tc['f90'], 'MPIF90=' + tc['mpif90'],
            'HIPCC=' + tc['hipcc'], 'HIPMPICXX=' + tc['hipcc'],
            'HIP_INSTALL_PATH=' + tc['rocm'], 'ROCM_PATH=' + tc['rocm'], 'MPI_INSTALL_PATH=' + tc['mpi'],
            'MPI_INC=%s/include' % tc['mpi'], 'MPI_LIB=%s/lib' % tc['mpi'], 'I_MPI_INC=-I%s/include' % tc['mpi'],
            'AMD_ARCH=gfx90a', 'GPU_VENDOR=AMD',
            # Command-line variables override the Makefile's own "CXXFLAGS += -DOPS_FTN".
            'CXXFLAGS=%s -DOPS_FTN' % tc['cxxflags'], 'HIPFLAGS=' + tc['hipflags'], 'FFLAGS=' + tc['fflags'],
            'FMODS=%s%s -I%s' % (j, mod, mod),
            'FMODS_F2C_HIP=%s%s/f2c_hip -I%s/f2c_hip -I%s' % (j, mod, mod, mod)]
    if tc['hdf5'] == 'keep':
        args += ['HDF5_INSTALL_PATH=' + tc['hdf5_root'], 'HDF5_INC=-I%s/include' % tc['hdf5_root']]
    else:
        args += ['HDF5_INSTALL_PATH=', 'HDF5_INC=']
    return args


def app_commands(tc, ops, src, other_files, ops_generated):
    """Explicit compile and link commands of senga2_f2c_mpi_hip, mirroring the
    f2c_mpi_hip rules of makefiles/Makefile.f_app (lines 559-575), one command
    per file so that a failure names the file and stops the build."""
    mod = ops / 'ops/fortran/mod' / tc['ops_compiler']
    inc = ops / 'ops/c/include'
    j = tc['fmod_flag']
    fmods = ['%s%s/f2c_hip' % (j, mod), '-I%s/f2c_hip' % mod, '-I%s' % mod]
    mpi_inc = '-I%s/include' % tc['mpi']
    cxxflags = [f for f in shlex.split(tc['cxxflags']) if f not in ('-fopenmp', '-qopenmp')]
    hipflags = shlex.split(tc['hipflags'].replace('-std=c++11', '-std=c++14'))
    kernels = sorted(p.name for p in (src / 'f2c_hip').glob('*.F90'))
    require(kernels, 'No generated Fortran kernels in %s/f2c_hip' % src)
    cxx = dict(source='f2c_hip/f2c_hip_kernels.cpp', object='f2c_hip/f2c_hip_kernels_f2c_mpi_hip.o', group='hip',
               command=[tc['hipcc']] + cxxflags + hipflags + ['-DOPS_FTN', '-DOPS_MPI', '-I%s' % inc, '-I.',
                        '-I%s/include' % tc['rocm'], '-DMPICH_IGNORE_CXX_SEEK', mpi_inc, '-c',
                        'f2c_hip/f2c_hip_kernels.cpp', '-o', 'f2c_hip/f2c_hip_kernels_f2c_mpi_hip.o'])

    def fortran(source, group):
        flags = tc['app_flags']
        if tc['app_flags_large'] and Path(source).name in tc['large_files']:
            flags = tc['app_flags_large']
        obj = source[:-4] + '_f2c_mpi_hip.o'
        return dict(source=source, object=obj, group=group,
                    command=[tc['mpif90']] + shlex.split(tc['fflags']) + shlex.split(flags) + fmods +
                            ['-DOPS_MPI', '-DOPS_F2C_INTEROP', '-c', source, '-o', obj])

    steps = [cxx] + [fortran(f, 'other') for f in other_files]
    steps += [fortran('f2c_hip/' + f, 'kernels') for f in kernels]
    steps += [fortran(f, 'generated') for f in ops_generated]
    objects = [s['object'] for s in steps]
    link = ([tc['mpif90']] + shlex.split(tc['fflags']) + shlex.split(tc['app_flags']) +
            ['-DOPS_MPI', '-I%s' % inc] + fmods + [mpi_inc, '@f2c_mpi_hip_objects.rsp',
             'fftlib/fftlib.a', 'rndlib/rndlib.a', 'parlib/parlib.a', '-lm',
             '-L%s' % (ops / 'ops/fortran/lib' / tc['ops_compiler']), '-lops_f2c_mpi_hip'] +
            shlex.split(tc['hdf5_link']) + shlex.split(tc['mpi_hip_link']) + shlex.split(tc['hip_link']) +
            shlex.split(tc['hiprand_link']) + shlex.split(tc['link_extra']) + ['-o', EXECUTABLE])
    return steps, objects, link


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
            'SENGA_MPI_HIP_LINK / SENGA_GTL_DIR and that craype-accel-amd-gfx90a is loaded.' % exe)
    return r.stdout


# --------------------------------------------------------------------------- build
class Builder(object):
    def __init__(self, campaign, name, env=None):
        self.campaign, self.name = Path(campaign), name
        self.config = read_json(self.campaign / 'campaign.json')
        require(name in self.config['builds'], 'Unknown build %r (choose from %s)' % (name, ', '.join(sorted(self.config['builds']))))
        self.grid = self.config['builds'][name]['grid']
        self.root = self.campaign / 'build' / name
        self.ops, self.senga = self.root / 'OPS', self.root / 'SENGA2'
        self.src = self.senga / 'src'
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
            raise ValueError('BUILD FAILED (%s build) at step: %s\n  command: %s\n  directory: %s\n  exit status: %d\n'
                             '  full output: %s (the failing command is the last "$ " line)\n'
                             'Fix the toolchain variables (run.sh build --print-toolchain lists them) and rerun the '
                             'same build command; finished objects are reused.'
                             % (self.name, what or command[0], shell_join(command), cwd or os.getcwd(), r.returncode, self.log_path))

    def stage_sources(self):
        marker = self.root / 'prepared.json'
        if marker.is_file():
            return read_json(marker)
        snap = verify_snapshot(self.campaign)
        for path in (self.ops, self.senga):
            if path.exists():
                shutil.rmtree(str(path))
        shutil.copytree(str(self.campaign / 'source/OPS'), str(self.ops))
        shutil.copytree(str(self.campaign / 'source/SENGA2'), str(self.senga))
        report = prepare_tree(self.senga, self.ops, self.grid, self.config['hdf5'])
        report.update(prepared_utc=now(), ops_commit=snap['ops']['commit'], senga_commit=snap['senga']['commit'],
                      com_senga2_sha256=sha(self.src / 'com_senga2.F90'))
        atomic(marker, report)
        self.say('Prepared sources for the %s build: grid %d^3, HDF5 %s' % (self.name, self.grid, self.config['hdf5']))
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
        probe = subprocess.run([python, '-c', 'import fparser, jinja2, pcpp, sympy, numpy'], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
        require(probe.returncode == 0, 'The translator Python %s lacks the OPS translator requirements:\n%s'
                % (python, probe.stdout.strip()))
        return python

    def generate(self, python):
        text = (self.src / 'Makefile').read_text()
        ops_files = makefile_list(text, 'OPS_FILES')
        other = makefile_list(text, 'OTHER_FILES')
        for name in ops_files + other:
            require((self.src / name).is_file(), 'SENGA2 Makefile lists %s but the file is missing' % name)
        generated = [f[:-4] + '_ops.F90' for f in ops_files]
        if not (self.src / '.generated').is_file():
            for stale in list((self.src / 'f2c_hip').glob('*')) if (self.src / 'f2c_hip').is_dir() else []:
                stale.unlink()
            self.say('Generating f2c HIP sources for %d OPS files (about one minute)' % len(ops_files))
            self.run([python, str(self.ops / 'ops_translator/ops-translator'), '-DOPS_ACC_IGNORE', '-I',
                      str(self.ops / 'ops/c/include') + '/', '-t', 'f2c_hip', '--file_paths'] + ops_files,
                     cwd=self.src, what='OPS code generation (ops-translator -t f2c_hip)')
            for name in generated + ['f2c_hip/f2c_hip_kernels.cpp']:
                require((self.src / name).is_file(), 'Code generation did not produce %s; see %s' % (name, self.log_path))
            (self.src / '.generated').write_text(now() + '\n')
        count = check_hooks(self.src / 'f2c_hip')
        self.say('Autotuning hooks present in all %d generated HIP kernels' % count)
        return other, generated, count

    def reset_if_toolchain_changed(self, tc):
        """Objects built with other compilers or flags are never mixed into the executable."""
        signature = hashlib.sha256(repr(sorted((k, str(v)) for k, v in tc.items() if k != 'jobs')).encode()).hexdigest()
        flag_file = self.root / '.toolchain_signature'
        if flag_file.is_file() and flag_file.read_text().strip() == signature:
            return
        if flag_file.is_file():
            self.say('Toolchain variables changed since the last attempt: discarding earlier objects and libraries')
        for sub in ('obj', 'lib', 'mod'):
            path = self.ops / 'ops/fortran' / sub
            if path.exists():
                shutil.rmtree(str(path))
        stale = list(self.src.glob('*_f2c_mpi_hip.o')) + list((self.src / 'f2c_hip').glob('*_f2c_mpi_hip.o'))
        stale += list(self.src.glob('*.mod')) + [self.src / s / (s + '.a') for s in SENGA_SUBDIRS]
        for path in stale:
            if path.exists():
                path.unlink()
        flag_file.write_text(signature + '\n')

    def libraries(self, tc):
        lib = self.ops / 'ops/fortran/lib' / tc['ops_compiler'] / 'libops_f2c_mpi_hip.a'
        if not lib.is_file():
            self.say('Building the OPS Fortran f2c MPI+HIP library')
            self.run(['make', '-C', str(self.ops / 'ops/fortran'), '-j', str(tc['jobs'])] +
                     library_make_args(tc, self.ops) + ['f2c_mpi_hip'], what='OPS library (make f2c_mpi_hip)')
        require(lib.is_file(), 'OPS library build finished but %s is missing; see %s' % (lib, self.log_path))
        mods = [p.name.lower() for p in (self.ops / 'ops/fortran/mod' / tc['ops_compiler'] / 'f2c_hip').glob('*.mod')]
        require('ops_fortran_declarations.mod' in mods,
                'OPS Fortran module ops_fortran_declarations.mod was not written to mod/%s/f2c_hip. Check that '
                'SENGA_FMOD_FLAG (%s) is the module-output option of %s.' % (tc['ops_compiler'], tc['fmod_flag'], tc['f90']))
        for sub, script, var, compiler in (('fftlib', 'Makeftl', 'F90', tc['f90']),
                                           ('rndlib', 'Makeran', 'F90', tc['f90']),
                                           ('parlib', 'Makepar', 'MPIF90', tc['mpif90'])):
            archive = self.src / sub / (sub + '.a')
            if archive.is_file():
                continue
            # The upstream scripts do not stop on a compile error: run with -e and count the members.
            members = len(re.findall(r'^\$\w+ .*-c \S+', (self.src / sub / script).read_text(), re.M))
            self.run(['sh', '-e', './' + script], cwd=self.src / sub, env=dict(self.env, **{var: compiler}),
                     what='SENGA2 %s (%s with %s=%s)' % (sub, script, var, compiler))
            listing = subprocess.run(['ar', 't', str(archive)], stdout=subprocess.PIPE, universal_newlines=True)
            require(archive.is_file() and len(listing.stdout.split()) == members,
                    'SENGA2 %s is incomplete: expected %d objects in %s. See %s.' % (sub, members, archive, self.log_path))

    def compile_app(self, tc, other, generated):
        steps, objects, link = app_commands(tc, self.ops, self.src, other, generated)

        def pending(step):
            obj, source = self.src / step['object'], self.src / step['source']
            return not obj.is_file() or obj.stat().st_mtime < source.stat().st_mtime

        self.say('Compiling %d sources (1 HIP translation unit with all kernels, %d Fortran files)'
                 % (len(steps), len(steps) - 1))
        hip = [s for s in steps if s['group'] == 'hip' and pending(s)]
        process = None
        if hip:   # the long HIP compile runs beside the serial Fortran module chain
            self.out.write('\n$ cd %s && %s   (background, output in hip_kernels.log)\n' % (self.src, shell_join(hip[0]['command'])))
            self.out.flush()
            hip_log = (self.root / 'hip_kernels.log').open('w')
            process = subprocess.Popen(hip[0]['command'], cwd=str(self.src), env=self.env, stdout=hip_log, stderr=subprocess.STDOUT)
        try:
            for step in [s for s in steps if s['group'] == 'other']:
                if pending(step):
                    self.run(step['command'], cwd=self.src, what='compile ' + step['source'])
            self.parallel([s for s in steps if s['group'] == 'kernels' and pending(s)], tc['jobs'])
            for step in [s for s in steps if s['group'] == 'generated']:
                if pending(step):
                    self.run(step['command'], cwd=self.src, what='compile ' + step['source'])
        except BaseException:
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait()
                hip_log.close()
            raise
        if process:
            self.say('Waiting for the HIP kernel translation unit (hipcc); this can take several minutes')
            rc = process.wait()
            hip_log.close()
            self.out.write((self.root / 'hip_kernels.log').read_text(errors='replace'))
            require(rc == 0, 'BUILD FAILED (%s build) at step: hipcc f2c_hip/f2c_hip_kernels.cpp (exit status %d)\n'
                    '  command: %s\n  output: %s' % (self.name, rc, shell_join(hip[0]['command']), self.root / 'hip_kernels.log'))
        (self.src / 'f2c_mpi_hip_objects.rsp').write_text('\n'.join(objects) + '\n')
        self.run(link, cwd=self.src, what='link ' + EXECUTABLE)
        return steps, link

    def parallel(self, steps, jobs):
        """The generated kernel modules only use OPS modules and OPS_CONSTANTS, so they can compile concurrently."""
        running, queue, failed = [], list(steps), None
        while (queue or running) and not failed:
            while queue and len(running) < max(1, jobs):
                step = queue.pop(0)
                log = (self.src / (step['object'] + '.log')).open('w')
                running.append((step, log, subprocess.Popen(step['command'], cwd=str(self.src), env=self.env,
                                                            stdout=log, stderr=subprocess.STDOUT)))
            time.sleep(0.02)
            for item in list(running):
                step, log, proc = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                log.close()
                text = (self.src / (step['object'] + '.log')).read_text(errors='replace')
                self.out.write('\n$ cd %s && %s\n%s' % (self.src, shell_join(step['command']), text))
                (self.src / (step['object'] + '.log')).unlink()
                if proc.returncode != 0:
                    failed = (step, proc.returncode, text)
        for step, log, proc in running:
            proc.wait()
            log.close()
        self.out.flush()
        if failed:
            step, rc, text = failed
            raise ValueError('BUILD FAILED (%s build) at step: compile %s (exit status %d)\n  command: %s\n%s\n'
                             '  full output: %s' % (self.name, step['source'], rc, shell_join(step['command']),
                                                   '\n'.join('    ' + l for l in text.splitlines()[-15:]), self.log_path))

    def build(self):
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / 'build_manifest.json'
        with lock(self.root / '.lock'):
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                require(sha(self.root / EXECUTABLE) == manifest['sha256'],
                        'The frozen %s executable changed after it was built (%s). Start a new campaign.' % (self.name, self.root / EXECUTABLE))
                print('Build %s already complete: %s sha256=%s' % (self.name, self.root / EXECUTABLE, manifest['sha256']), flush=True)
                return manifest
            with self.log_path.open('a') as out:
                self.out = out
                out.write('\n==== %s build started %s ====\n' % (self.name, now()))
                prepared = self.stage_sources()
                tc = toolchain(self.env, self.config['hdf5'])
                out.write('Toolchain:\n' + ''.join('  %s = %s\n' % (k, v) for k, v in sorted(tc.items())))
                self.env.update(OPS_INSTALL_PATH=str(self.ops / 'ops'), OPS_COMPILER=tc['ops_compiler'],
                                HIP_INSTALL_PATH=tc['rocm'], ROCM_PATH=tc['rocm'], MPICH_GPU_SUPPORT_ENABLED='1')
                python = self.translator()
                other, generated, kernels = self.generate(python)
                self.reset_if_toolchain_changed(tc)
                self.libraries(tc)
                steps, link = self.compile_app(tc, other, generated)
                exe = self.src / EXECUTABLE
                require(exe.is_file(), 'Link finished but %s is missing' % exe)
                ldd = gpu_preflight(exe, self.env) if tc['require_gtl'] else 'SKIPPED (SENGA_REQUIRE_GTL=0)'
                frozen = self.root / EXECUTABLE
                shutil.copy2(str(exe), str(frozen))
                versions = {k: query([tc[k], '--version'])['stdout'].strip().splitlines()[:2]
                            for k in ('hipcc', 'cxx', 'f90', 'mpif90')}
                manifest = dict(build=self.name, grid=self.grid, executable=str(frozen.relative_to(self.campaign)),
                                sha256=sha(frozen), built_utc=now(), toolchain=tc, compiler_versions=versions,
                                prepared=prepared, generated_hip_kernels_with_hooks=kernels,
                                translator_python=python, link_command=link, compile_steps=len(steps),
                                gtl_linked=tc['require_gtl'], ldd=ldd,
                                modules=self.env.get('LOADEDMODULES', ''),
                                source_snapshot_sha256=sha(self.campaign / 'source_snapshot.json'))
                atomic(manifest_path, manifest)
                self.say('BUILD OK (%s, %d^3): %s sha256=%s' % (self.name, self.grid, frozen, manifest['sha256']))
                self.out = None
                return manifest


def print_toolchain(env, hdf5):
    print('Build variables (set them in the environment file given to prepare with --env-file, or export them):\n')
    for name, meaning, default in TOOLCHAIN_HELP:
        print('  %-24s %s\n  %-24s default: %s' % (name, meaning, '', default))
    print('\nResolved in the current environment:')
    try:
        for key, value in sorted(toolchain(env, hdf5).items()):
            print('  %-16s %s' % (key, value))
    except ValueError as error:
        print('  NOT RESOLVABLE HERE: %s' % error)
        return 1
    return 0
