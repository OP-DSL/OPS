#!/usr/bin/env python3
"""Portable LUMI campaign launcher for the frozen TGV StoreNone Top 3 dataset."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import runtime
from runtime import require, sha, atomic, shape_text
HERE = Path(__file__).resolve().parent
NODES = (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40)
CPU_BIND = 'map_cpu:49,57,17,25,1,9,33,41'
VERSION = 1


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def shell_join(command):
    return ' '.join(shlex.quote(arg) for arg in command)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process holds the lock: {path}')
        yield


def contained(root, rel):
    path = root / rel
    require(not Path(rel).is_absolute(), f'Path escapes package: {rel}')
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f'Path escapes package: {rel}')
    return path


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, data, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(data[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def verify_package(root, prefixes=None):
    m = read_json(root / 'dataset_manifest.json')
    require(m.get('application') == 'tgvstorenone', 'Wrong dataset application')
    for rel, spec in m['files'].items():
        if prefixes and not any(rel.startswith(p) for p in prefixes) and rel != 'selections.csv':
            continue
        path = contained(root, rel)
        require(path.is_file() and not path.is_symlink() and sha(path) == spec['sha256'], f'Invalid dataset file: {path}')
    require(sha(root / 'selections.csv') == m['selection_sha256'], 'Selection checksum mismatch')
    return m


def ensure_dataset(ops_root, explicit=None):
    if explicit:
        root = explicit.resolve()
        return root, verify_package(root)
    base = ops_root / 'dataset-lumi'
    archive = base / 'TOP3_DATASET.tar.gz'
    require(archive.is_file(), f'Missing {archive}; provide --dataset or the archive')
    checksum = (base / 'TOP3_DATASET.tar.gz.sha256').read_text().split()[0]
    require(re.fullmatch('[0-9a-fA-F]{64}', checksum) and sha(archive) == checksum.lower(), 'Dataset archive checksum mismatch')
    root = base / 'TOP3_DATASET'
    with lock(base / '.top3-extraction.lock'):
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            for entry in members:
                rel = Path(entry.name)
                require(rel.parts and rel.parts[0] == 'TOP3_DATASET' and '..' not in rel.parts and not rel.is_absolute(), 'Unsafe archive path')
                require(entry.isfile() or entry.isdir(), f'Archive links/devices are not allowed: {entry.name}')
            identity = tar.extractfile('TOP3_DATASET/dataset_manifest.json').read()
            replacement = json.loads(identity)
            if root.exists():
                if (root / 'dataset_manifest.json').read_bytes() == identity:
                    m = verify_package(root)
                    print(f'Reusing validated dataset: {root}', flush=True)
                    return root, m
                require(sha(root / 'dataset_manifest.json') in replacement.get('supersedes_manifest_sha256', []),
                        'Extracted dataset is not a recognized previous version; use --dataset explicitly or move the old folder aside')
                verify_package(root)
            stage = Path(tempfile.mkdtemp(prefix='.top3-extract-', dir=base))
            try:
                for entry in members:
                    dest = contained(stage, entry.name)
                    if entry.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(entry) as src, dest.open('wb') as out:
                            shutil.copyfileobj(src, out)
                m = verify_package(stage / 'TOP3_DATASET')
                backup = None
                if root.exists():
                    backup = Path(tempfile.mkdtemp(prefix='TOP3_DATASET.previous-', dir=base)) / 'TOP3_DATASET'
                    root.rename(backup)
                try:
                    (stage / 'TOP3_DATASET').rename(root)
                except OSError:
                    if backup is not None:
                        backup.rename(root)
                    raise
                if backup is not None:
                    print(f'Dataset updated; previous package preserved at: {backup}', flush=True)
            finally:
                shutil.rmtree(stage)
    print(f'Extracted and validated dataset: {root}', flush=True)
    return root, m


def catalog_check(folder, expected=None):
    m = read_json(folder / 'manifest.json')
    unsigned = dict(m)
    sid = unsigned.pop('set_id')
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() == sid, f'Changed manifest: {folder}')
    ctx = m['context']
    schema = m['schema_version']
    require(schema in (1, 2), 'Unknown catalog schema')
    names = {'blocks.csv'} if schema == 2 else {f'rank_{i}/blocks.csv' for i in range(int(ctx['ACTIVE_NUM_RANKS']))}
    require(set(m['files']) == names, 'Invalid rank coverage in manifest')
    if expected:
        require(ctx['APP_NAME'] == 'tgvstorenone' and ctx['GPU_ARCH'] == 'gfx90a', 'Wrong application/GPU')
        require(ctx['shape'] == shape_text(expected['shape']) and int(ctx['ACTIVE_NUM_RANKS']) == expected['ranks'], 'Wrong MPI context')
        require([int(ctx[f'GRID_{a}']) for a in 'XYZ'] == expected['grid'], 'Wrong grid')
        require(ctx['APP_ARGS'] == ' '.join(map(str, expected['grid'] + [500])), 'Wrong simulation iterations')
    missing = []
    for name, digest in m['files'].items():
        p = contained(folder, name)
        if not p.is_file():
            missing.append(name)
            continue
        require(sha(p) == digest, f'Changed block file: {p}')
        with p.open(newline='') as f:
            reader = csv.DictReader(f)
            require(reader.fieldnames == ['kernel_id', 'bx', 'by', 'bz'], 'Invalid block CSV header')
            values = list(reader)
        require(values and len({v['kernel_id'] for v in values}) == len(values), 'Empty/duplicate kernels')
        for v in values:
            dims = [int(v[k]) for k in ('bx', 'by', 'bz')]
            require(int(v['kernel_id']) >= 0 and min(dims) > 0 and product(dims) <= 1024, 'Invalid block dimensions')
    return m, missing


def make_plan(dataset, selected):
    selections = rows(dataset / 'selections.csv')
    result, coverage = [], []
    for n in selected:
        original = sorted((r for r in selections if int(r['Nodes']) == n), key=lambda r: int(r['Top rank']))
        require([r['Top rank'] for r in original] == ['1', '2', '3'], f'Missing Top 3: {n}')
        table = rows(dataset / f'n{n:03d}/cases.csv')
        require(len(table) == 4 and {r['role'] for r in table} == {'C', 'D_top1', 'D_top2', 'D_top3'}, f'Invalid roles: {n}')
        local = []
        for row in table:
            grid = list(map(int, row['global_grid'].split('x')))
            shape = list(map(int, row['mpi_decomposition'].split('x')))
            role = row['role']
            source = original[0] if role == 'C' else original[int(role[-1]) - 1]
            require(row['global_grid'] == source['Global grid'].replace('×', 'x'), 'Grid differs from selections')
            field = 'Fixed baseline MPI decomposition' if role == 'C' else 'Tested MPI decomposition'
            require(row['mpi_decomposition'] == source[field].replace('×', 'x'), 'MPI differs from selections')
            require(len(grid) == len(shape) == 3 and len(set(grid)) == 1 and product(shape) == n * 8, 'Invalid cube/MPI')
            require(all(g % p == 0 for g, p in zip(grid, shape)), 'Indivisible grid')
            require(int(row['simulation_iterations']) == 500 and int(row['mpi_ranks']) == n * 8, 'Wrong ranks/steps')
            case = dict(nodes=n, ranks=n*8, grid=grid, shape=shape, niter=500, mode=2,
                        roles=[role], catalog_path=row['catalog_path'], source_run_path=row['source_run_path'],
                        original_executable_sha256=row['historical_executable_sha256'], source_set_id=row['set_id'])
            m, missing = catalog_check(contained(dataset, row['catalog_path']), case)
            require(m['set_id'] == row['set_id'] and m['context']['EXECUTABLE_SHA256'] == case['original_executable_sha256'], 'Case/manifest mismatch')
            case['catalog_status'] = 'INCOMPLETE' if missing else 'COMPLETE'
            case['case_id'] = f'n{n:03d}_decomp{shape_text(shape)}_mode2'
            duplicate = next((c for c in local if c['case_id'] == case['case_id']), None)
            if duplicate:
                require(duplicate['source_set_id'] == case['source_set_id'], 'Different sets share same case')
                duplicate['roles'].append(role)
            else:
                local.append(case)
            coverage.append(dict(nodes=n, role=role, status=case['catalog_status'], missing_files=len(missing)))
        ref = next(c for c in local if 'C' in c['roles'])
        baseline = dict(ref, mode=0, roles=['A'], catalog_status='NOT_REQUIRED', catalog_path='',
                        source_set_id='', case_id=ref['case_id'].replace('_mode2', '_mode0'))
        result.extend([baseline] + local)
    return result, coverage


def query(command, timeout=15):
    if not shutil.which(command[0]):
        return dict(status='NOT_QUERIED', command=command, message=f'{command[0]} unavailable')
    try:
        r = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return dict(status='OK' if r.returncode == 0 else 'ERROR', command=command,
                    returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(status='UNKNOWN', command=command, message=str(e))


def resource_args(account, partition, nodes, walltime):
    return [f'--account={account}', f'--partition={partition}', f'--nodes={nodes}',
            '--ntasks-per-node=8', '--gpus-per-node=8', '--cpus-per-task=7',
            '--mem=0', f'--time={walltime}', '--exclusive']


def analyze(args, coverage):
    snapshot = dict(queried_utc=now(), note='Point-in-time observations; estimates are not reservations.',
        partitions=query(['scontrol', 'show', 'partition', '-o']),
        nodes=query(['sinfo', '-N', '-h', '-o', '%N|%P|%T|%G|%c|%m']),
        reservations=query(['scontrol', 'show', 'reservation', '-o']),
        associations=query(['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', f'user={os.environ.get("USER", "")}', 'format=Account,Partition,QOS']),
        queue=query(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%i|%j|%T|%R']), reports=[])
    partition_nodes = {}
    for line in snapshot['nodes'].get('stdout', '').splitlines():
        fields = line.split('|')
        if len(fields) != 6:
            continue
        name, partition, state, gres, cpus, memory = fields
        partition_nodes.setdefault(partition.rstrip('*'), {})[name] = state
    snapshot['partition_node_counts'] = {}
    for partition, nodes in partition_nodes.items():
        counts = {}
        for state in nodes.values():
            counts[state] = counts.get(state, 0) + 1
        snapshot['partition_node_counts'][partition] = dict(unique_nodes=len(nodes), states=counts,
            note='Do not add across partitions; idle does not imply allocatable for this account/reservation.')
    for n in args.nodes:
        partitions = [args.partition] if args.partition != 'auto' else (['small-g', 'standard-g'] if n <= 4 else ['standard-g'])
        for part in partitions:
            require(part not in ('dev-g', 'debug'), 'Development partitions are not used for this scientific campaign')
            command = ['sbatch', '--test-only'] + resource_args(args.account, part, n, args.time) + ['--job-name=tgv_top3_probe', '--wrap=true']
            probe = query(command) if args.account else dict(status='NOT_QUERIED', message='--account is required for an exact probe')
            state = 'ELIGIBLE' if probe['status'] == 'OK' else ('REJECTED' if probe['status'] == 'ERROR' else 'UNKNOWN')
            text = probe.get('stdout', '') + probe.get('stderr', '')
            match = re.search(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', text)
            report = dict(nodes=n, ranks=n*8, partition=part, dataset_status='COMPLETE' if all(c['status'] == 'COMPLETE' for c in coverage if c['nodes'] == n) else 'INCOMPLETE',
                          eligibility=state, estimated_start=match[0] if match else '', probe=probe)
            snapshot['reports'].append(report)
            print(f"{n:2} nodes | {part:10} | dataset {report['dataset_status']:10} | {state:11} | start {report['estimated_start'] or 'unknown'}", flush=True)
    return snapshot


def select_partitions(report, args):
    selected = {}
    for n in args.nodes:
        options = [r for r in report['reports'] if r['nodes'] == n and r['eligibility'] == 'ELIGIBLE']
        require(options, f'No successful Slurm eligibility probe for {n} nodes. Inspect resource_report.json.')
        options.sort(key=lambda r: (not bool(r['estimated_start']), r['estimated_start'], r['partition'] != 'standard-g'))
        selected[str(n)] = options[0]['partition']
    return selected


def snapshot_ops(ops, campaign):
    target = campaign / 'build/OPS'
    require(not target.exists(), 'Build snapshot already exists; use --resume')
    listing = subprocess.check_output(['git', '-C', str(ops), 'ls-files', '-z']).decode().split('\0')
    required = ('apps/c/TGV_StoreNone/Makefile', 'apps/c/TGV_StoreNone/opensbli.cpp',
                'apps/c/TGV_StoreNone/opensbliblock00_kernels.h', 'apps/c/TGV_StoreNone/defdec_data_set.h',
                'makefiles/Makefile.c_app', 'makefiles/Makefile.hip', 'ops/c/Makefile',
                'ops_translator/ops-translator/__main__.py', 'ops_translator/requirements.txt')
    for rel in required:
        require(rel in listing and (ops / rel).is_file(), f'Required build input is missing or not tracked by Git: {rel}. Update the OPS checkout before submitting.')
    prefixes = ('ops/c/', 'makefiles/', 'ops_translator/', 'apps/c/TGV_StoreNone/')
    copied = {}
    for rel in filter(None, listing):
        if not rel.startswith(prefixes):
            continue
        source = ops / rel
        require(source.is_file() and not source.is_symlink(), f'Unsupported/missing source: {source}')
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied[rel] = sha(dest)
    util = (target / 'ops/c/src/externlib/ops_util.cpp').read_text()
    app = (target / 'apps/c/TGV_StoreNone/opensbli.cpp').read_text()
    require('OPS_DISABLE_CSV_LOGS' in util and 'OPS_CSV_LOG_LEVEL' in util, 'OPS runtime lacks rank-selective CSV controls; use the prepared OPS checkout')
    require('requested_niter' in app, 'TGV must accept the fourth argument (iteration count) for compatibility smoke tests')
    data = dict(commit=subprocess.check_output(['git', '-C', str(ops), 'rev-parse', 'HEAD'], universal_newlines=True).strip(),
                source_files_sha256=copied, source_root=str(ops))
    atomic(campaign / 'source_snapshot.json', data)
    (campaign / 'source_changes.patch').write_bytes(subprocess.check_output(['git', '-C', str(ops), 'diff', 'HEAD', '--', *prefixes]))


def translator_interpreter(explicit=None):
    candidates = [explicit] if explicit else [sys.executable, 'python3'] + [f'python3.{minor}' for minor in range(14, 7, -1)]
    for candidate in candidates:
        executable = shutil.which(str(candidate))
        if not executable:
            continue
        result = query([executable, '-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'])
        if result['status'] == 'OK':
            # Preserve virtualenv paths; resolving symlinks would discard that environment.
            return str(Path(executable).absolute())
    raise ValueError('OPS translation requires Python >= 3.8. Load a newer Python in --env-file, '
                     'or pass --translator-python /path/to/venv/bin/python3 with the OPS translator dependencies. '
                     'The launcher itself supports Python >= 3.6.')


def configure_libclang(python, env, out):
    # The clang and libclang wheels share a Python package. Depending on pip's
    # installation order, cindex may search for libclang-16.so even though the
    # matching bundled library is named clang/native/libclang.so.
    code = '''import os
from pathlib import Path
import clang.cindex as ci
import jinja2, fparser, pcpp, numpy, sympy, h5py
library = os.environ.get('LIBCLANG_PATH')
if not library:
    bundled = Path(ci.__file__).resolve().parent / 'native/libclang.so'
    if bundled.is_file():
        library = str(bundled)
if library:
    ci.Config.set_library_file(library)
index = ci.Index.create()
unit = index.parse('ops_probe.cpp', args=['-xc++', '-std=c++11'],
                   unsaved_files=[('ops_probe.cpp',
                       'int bits(float x) { return __builtin_bit_cast(int, x); }')])
errors = [str(d) for d in unit.diagnostics if d.severity >= ci.Diagnostic.Error]
if errors:
    raise RuntimeError('C++ parser probe failed: ' + '; '.join(errors))
# Parsing alone succeeds with the conflicting clang wheel. Traversal exposes
# missing Python cursor enums, including BUILTIN_BIT_CAST_EXPR (280).
kinds = [node.kind for node in unit.cursor.walk_preorder()]
if not any(kind.value == 280 for kind in kinds):
    raise RuntimeError('C++ parser probe did not produce the bit-cast expression')
print(ci.conf.get_filename())
'''
    result = subprocess.run([python, '-c', code], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    out.write('Checking translator dependencies and native libclang C++ parsing\n')
    out.write(result.stdout + result.stderr)
    out.flush()
    require(result.returncode == 0,
            'Translator/libclang preflight failed; inspect build/build.log. '
            'Use the libclang package without the conflicting clang package. '
            'LIBCLANG_PATH, if set, must name a compatible shared-library file (not a directory).')
    library = result.stdout.strip().splitlines()[-1]
    env['LIBCLANG_PATH'] = library
    return library


def rocm_compiler_flags(hipcc, hip_root):
    # Link with hipcc and compile host objects with its ROCm toolchain too.
    # Cray CC can inject compiler-runtime calls (__cray_*set_detect) that a
    # direct hipcc link does not supply. MPI uses explicit Cray headers/libs.
    override = os.environ.get('HIP_CLANG_PATH')
    candidates = [Path(override)] if override else [
        hip_root / 'llvm/bin', hip_root / 'lib/llvm/bin', hip_root / 'bin']
    for directory in candidates:
        cc, cxx = directory / 'clang', directory / 'clang++'
        if all(p.is_file() and os.access(str(p), os.X_OK) for p in (cc, cxx)):
            # Do not resolve clang++ symlinks: the executable name selects C++.
            return [f'CC={cc}', f'CXX={cxx}', f'MPICC={cc}', f'MPICXX={cxx}',
                    f'MPICPP={cxx}', f'HIPCC={hipcc}', f'HIPMPICXX={hipcc}']
    raise ValueError('Cannot locate ROCm clang/clang++ for host compilation. '
                     'Set HIP_CLANG_PATH to the compiler directory used by hipcc.')


def build_binary(campaign):
    config = read_json(campaign / 'campaign.json')
    target = campaign / 'build/OPS'
    with lock(campaign / 'build/.lock'):
        if (campaign / 'build/build_manifest.json').is_file():
            manifest = read_json(campaign / 'build/build_manifest.json')
            require(sha(campaign / manifest['executable']) == manifest['sha256'], 'Frozen executable changed')
            return
        source = read_json(campaign / 'source_snapshot.json')
        for rel, digest in source['source_files_sha256'].items():
            require(sha(target / rel) == digest, f'Snapshot changed: {rel}')
        hipcc = shutil.which('hipcc')
        mpi_root = Path(os.environ.get('CRAY_MPICH_DIR', ''))
        require(hipcc and (mpi_root / 'include/mpi.h').is_file(), 'Load ROCm and Cray MPICH via --env-file; hipcc or mpi.h missing')
        hip_root = Path(os.environ.get('HIP_INSTALL_PATH') or os.environ.get('ROCM_PATH') or str(Path(hipcc).resolve().parent.parent))
        compiler_flags = rocm_compiler_flags(hipcc, hip_root)
        mpi_lib = mpi_root / 'lib'
        gtl = os.environ.get('PE_MPICH_GTL_DIR_amd_gfx90a', '')
        if gtl.startswith('-L'):
            gtl = gtl[2:]
        if not gtl:
            found = list(mpi_root.glob('gtl/lib/libmpi_gtl_hsa.so*'))
            if found:
                gtl = str(found[0].parent)
        require(gtl and Path(gtl).is_dir(), 'Cannot locate libmpi_gtl_hsa; set PE_MPICH_GTL_DIR_amd_gfx90a in the environment profile')
        translator = config.get('translator_python')
        python = translator_interpreter(translator)
        log = campaign / 'build/build.log'
        env = dict(os.environ)
        with log.open('a') as out:
            def run(cmd):
                out.write(shell_join(cmd) + '\n'); out.flush()
                subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
            if not translator:
                venv = campaign / 'build/translator-env'
                if not (venv / 'bin/python3').exists():
                    run([python, '-m', 'venv', str(venv)])
                # Repair only the campaign-owned environment on a retry. Both
                # distributions own clang/cindex.py, so remove both before
                # reinstalling the one coherent package from requirements.
                if query([str(venv / 'bin/python3'), '-m', 'pip', 'show', 'clang'])['status'] == 'OK':
                    run([str(venv / 'bin/python3'), '-m', 'pip', 'uninstall', '-y', 'clang', 'libclang'])
                run([str(venv / 'bin/python3'), '-m', 'pip', 'install', '-r', str(target / 'ops_translator/requirements.txt')])
                translator = str(venv / 'bin/python3')
            translator = translator_interpreter(translator)
            # Wrapper fixes the exact interpreter used by Makefile.c_app's python3 command.
            shim = campaign / 'build/python-bin'; shim.mkdir(exist_ok=True)
            wrapper = shim / 'python3'
            wrapper.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(translator) + ' "$@"\n'); wrapper.chmod(0o755)
            env['PATH'] = str(shim) + os.pathsep + env['PATH']
            env.update(OPS_INSTALL_PATH=str(target / 'ops'), OPS_COMPILER='hip', AMD_ARCH='gfx90a',
                       HIP_INSTALL_PATH=str(hip_root), ROCM_PATH=str(hip_root), MPI_INSTALL_PATH=str(mpi_root),
                       MPICH_GPU_SUPPORT_ENABLED='1')
            configure_libclang(translator, env, out)
            flags = ['OPS_COMPILER=hip', f'OPS_INSTALL_PATH={target / "ops"}', *compiler_flags,
                     f'HIP_INSTALL_PATH={hip_root}', f'MPI_INSTALL_PATH={mpi_root}', f'MPI_INC={mpi_root / "include"}',
                     f'MPI_LIB={mpi_lib}', f'I_MPI_INC=-I{mpi_root / "include"}', 'MPI_LINK=-lmpi',
                     f'MPI_HIP_LINK=-L{mpi_lib} -L{gtl} -lmpi -lmpi_gtl_hsa',
                     'AMD_ARCH=gfx90a', 'GPU_VENDOR=AMD', 'CXXFLAGS=-O3 -fPIC -std=c++11',
                     'CFLAGS=-O3 -fPIC', 'HIPFLAGS=-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__',
                     f'HIP_LINK=-L{hip_root / "lib"} -L{hip_root / "lib64"} -lamdhip64 -lrocm_smi64', 'TARGETS=mpi_hip']
            run(['make', '-C', str(target / 'ops/c'), '-j', str(config['build_jobs']), *flags, 'mpi_hip'])
            # Generate sources first: Makefile.c_app lists generated .cpp files as
            # link prerequisites but does not provide individual creation rules.
            run(['make', '-C', str(target / 'apps/c/TGV_StoreNone'), *flags, '.generated'])
            run(['make', '-C', str(target / 'apps/c/TGV_StoreNone'), '-j', str(config['build_jobs']), *flags, 'opensbli_mpi_hip'])
        exe = target / 'apps/c/TGV_StoreNone/opensbli_mpi_hip'
        runtime.gpu_preflight(exe)
        frozen = campaign / 'build/tgvstorenone_mpi_hip'
        shutil.copy2(exe, frozen)
        environment = {k: v for k, v in env.items() if k.startswith(('OPS_', 'HIP', 'ROCM', 'CRAY_', 'PE_', 'MPICH_', 'LOADEDMODULES', 'AMD_ARCH'))}
        compiler_commands = dict(flag.split('=', 1) for flag in compiler_flags)
        versions = {tool: query([tool, '--version']) for tool in sorted(set(compiler_commands.values()))}
        atomic(campaign / 'build/build_manifest.json', dict(executable=str(frozen.relative_to(campaign)),
            sha256=sha(frozen), built_utc=now(), compiler_versions=versions,
            compiler_commands=compiler_commands, environment=environment,
            translator_python=translator, libclang_path=env['LIBCLANG_PATH'],
            source_snapshot_sha256=sha(campaign / 'source_snapshot.json'),
            transfer_policy='Runtime metadata smoke checks required per decomposition; not historical-binary replay'))
        print(f'Built once: {frozen}', flush=True)


def first_kernel_rows(path):
    out = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            kid = (row.get('kernel_id') or '').strip()
            if kid.isdigit() and kid not in out:
                out[kid] = row
    require(out, f'No measured kernel metadata: {path}')
    return out


def validate_transfer(historical_log, current_log, blocks):
    old, new = first_kernel_rows(historical_log), first_kernel_rows(current_log)
    require(old.keys() == new.keys(), 'Smoke kernel IDs do not match historical IDs')
    fields = ('stencil_sig', 'nstencil_args', 'dim', 'elem_size', 'size_x', 'size_y', 'size_z',
              'd_m_x', 'd_m_y', 'd_m_z', 'd_p_x', 'd_p_y', 'd_p_z',
              'num_read', 'num_write', 'num_rw', 'grid_x', 'grid_y', 'grid_z',
              'n_args_x_offset', 'n_args_y_offset', 'n_args_z_offset', 'max_radius_x', 'max_radius_y', 'max_radius_z',
              'total_offset_points')
    for kid in old:
        for field in fields:
            require(field in old[kid] and field in new[kid], f'Missing metadata field: {field}')
            a, b = old[kid][field], new[kid][field]
            equal = a == b if field == 'stencil_sig' else math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-8)
            require(equal, f'Kernel {kid} changed {field}: {a} -> {b}')
    for block in rows(blocks):
        kid = block['kernel_id']
        require(kid in new, f'Block references an unobserved kernel: {kid}')
        require(product(int(block[k]) for k in ('bx', 'by', 'bz')) <= float(new[kid]['max_threads']), f'Block exceeds current kernel limit: {kid}')
    return dict(status='PASS', kernels=len(new), fields_checked=list(fields),
                missing_catalog_kernels=sorted(set(new) - {r['kernel_id'] for r in rows(blocks)}, key=int),
                historical_log_sha256=sha(historical_log), smoke_log_sha256=sha(current_log),
                limitation='Checks runtime IDs, stencil/array metadata, local geometry and block limits; does not prove identical kernel arithmetic or binary equivalence.')


def execute_case(campaign, case, exe, catalog, run, smoke=False):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'ranks').mkdir(exist_ok=True)
    spec = dict(case=case, executable=str(exe), workdir=str(run), catalog=str(catalog) if catalog else '')
    atomic(run / 'execution.json', spec)
    command = ['srun', f'--nodes={case["nodes"]}', f'--ntasks={case["ranks"]}', '--ntasks-per-node=8',
               '--kill-on-bad-exit=1', '--cpu-bind=' + CPU_BIND, 'bash', str(campaign / 'launcher/run.sh'),
               '--rank-spec', str(run / 'execution.json'), '--rank-output', str(run)]
    start = time.monotonic()
    with (run / 'srun.log').open('w') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, MPICH_GPU_SUPPORT_ENABLED='1')).returncode
    metrics = [read_json(p) for p in (run / 'ranks').glob('status_*.json')]
    metrics.sort(key=lambda r: r['rank_id'])
    good = rc == 0 and len(metrics) == case['ranks'] and {r['rank_id'] for r in metrics} == set(range(case['ranks']))
    good = good and all(r['rank_status'] == 'OK' and math.isfinite(r['rank_wall_time_s']) and r['rank_wall_time_s'] > 0 for r in metrics)
    reported = [r['app_reported_wall_time_s'] for r in metrics if r['app_reported_wall_time_s'] is not None]
    good = good and bool(reported) and all(math.isfinite(x) and x > 0 for x in reported)
    result = dict(case=case, status='FAILED', ranks=metrics, app_wall_time_s=max(reported) if reported else 0,
                  job_wall_time_s=time.monotonic()-start, srun_exit_code=rc)
    if good:
        try:
            result.update(runtime.ops_totals(run / 'ranks/rank_0.log'))
        except ValueError as e:
            good = False
            result['error'] = str(e)
    result['status'] = 'SMOKE_PASS' if good and smoke else ('COMMITTED' if good else 'FAILED')
    atomic(run / 'result.json', result)
    c = case
    fields = dict(APP_NAME='tgvstorenone', APP_ARGS=' '.join(map(str, c['grid'] + [c['niter']])),
        CAMPAIGN_ID=campaign.name, CASE_ID=c['case_id'], RUN_CLASS='SMOKE' if smoke else 'SCIENTIFIC',
        SCIENTIFIC_RESULT='NO' if smoke else 'YES', NITER=c['niter'], GPU_ARCH='gfx90a',
        EXECUTABLE_SHA256=sha(exe), DECOMPOSITION_MODE=1, GRID_X=c['grid'][0], GRID_Y=c['grid'][1], GRID_Z=c['grid'][2],
        GRID_CONFIG='cfg1', TOTAL_CELLS=product(c['grid']), NUM_NODES=c['nodes'], NUM_MPI_RANKS=c['ranks'],
        GPUS_PER_NODE=8, NUM_GPUS_USED=c['ranks'], RANKS_PER_GPU=1, DECOMP_PX=c['shape'][0],
        DECOMP_PY=c['shape'][1], DECOMP_PZ=c['shape'][2], OPS_AUTOTUNE_MODE=c['mode'], GPU_DIRECT='YES',
        MPICH_GPU_SUPPORT_ENABLED=1, RUN_ITERATION=1, TOP3_ROLES=','.join(c['roles']), CSV_LOG_RANKS='0',
        PRECOMPUTED_CATALOG=str(catalog) if catalog else '', PRECOMPUTED_SET_ID=read_json(catalog / 'manifest.json')['set_id'] if catalog else '',
        APP_WALL_TIME=result['app_wall_time_s'], JOB_WALL_TIME=result['job_wall_time_s'], PROCESS_EXIT_CODE=rc,
        RUN_EXIT_CODE=0 if good else (rc or 1), RUN_STATUS='OK' if good else 'FAILED',
        VALIDATION_STATUS='PASS' if good else 'FAIL', COMMIT_STATUS='COMMITTED' if good and not smoke else 'UNCOMMITTED',
        KERNEL_TOTAL_TIME=result.get('kernel_time_s', 'NA'), HALO_TOTAL_TIME=result.get('halo_time_s', 'NA'))
    (run / 'config.txt').write_text(''.join(f'{k}={v}\n' for k, v in fields.items()))
    write_csv(run / 'mpi_rank_metrics.csv', metrics,
              ['rank_id', 'local_rank', 'gpu_id', 'hostname', 'rank_wall_time_s', 'app_reported_wall_time_s', 'rank_status', 'precomputed_loaded'])
    shutil.copyfile(run / 'ranks/rank_0.log', run / 'run.log') if (run / 'ranks/rank_0.log').exists() else None
    if catalog:
        shutil.copyfile(catalog / 'manifest.json', run / 'precomputed_manifest.json')
    require(good, f'Execution failed: inspect {run}')
    return result


def run_node(campaign, nodes):
    config = read_json(campaign / 'campaign.json')
    require(int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) == nodes, 'Allocation node count differs from plan')
    dataset = Path(config['dataset'])
    require(sha(dataset / 'dataset_manifest.json') == config['dataset_manifest_sha256'], 'Dataset identity changed')
    verify_package(dataset, [f'n{nodes:03d}/'])
    build = read_json(campaign / 'build/build_manifest.json')
    exe = campaign / build['executable']
    require(sha(exe) == build['sha256'], 'Executable hash mismatch')
    runtime.gpu_preflight(exe)
    root = campaign / f'OPS_LOGS_{config["stamp"]}_n{nodes:03d}'
    root.mkdir(exist_ok=True)
    with lock(root / '.node.lock'):
        planned = [dict(c, executable_sha256=build['sha256']) for c in config['cases'] if c['nodes'] == nodes]
        # A single, separate one-step metadata smoke per unique precomputed decomposition.
        for case in (c for c in planned if c['mode'] == 2):
            original = contained(dataset, case['catalog_path'])
            m, missing = catalog_check(original, case)
            require(not missing, 'Incomplete catalog before smoke')
            smoke = root / '_compatibility' / case['case_id']
            report_path = smoke / 'compatibility.json'
            if report_path.exists():
                report = read_json(report_path)
                require(report['executable_sha256'] == build['sha256'] and report['source_set_id'] == m['set_id'] and report['status'] == 'PASS', 'Stale compatibility report')
                require(sha(contained(root, report['smoke_log'])) == report['smoke_log_sha256'], 'Smoke metadata changed')
            else:
                smoke.mkdir(parents=True, exist_ok=True)
                attempt = Path(tempfile.mkdtemp(prefix='attempt_', dir=smoke))
                execute_case(campaign, dict(case, niter=1), exe, original, attempt, smoke=True)
                historical = contained(dataset, case['source_run_path']) / 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv'
                candidates = list((attempt / 'rank_0').glob('*/ops_blocksize_tuning_logmod.csv'))
                require(len(candidates) == 1, 'Smoke did not emit kernel metadata')
                blockfile = original / ('blocks.csv' if m['schema_version'] == 2 else 'rank_0/blocks.csv')
                report = validate_transfer(historical, candidates[0], blockfile)
                report.update(executable_sha256=build['sha256'], historical_executable_sha256=m['context']['EXECUTABLE_SHA256'],
                              source_set_id=m['set_id'], checked_utc=now(), smoke_log=str(candidates[0].relative_to(root)))
                atomic(report_path, report)
            # New campaign manifest links to, but never modifies, the historical manifest.
            replay = root / 'precomputed_blocks' / f'decomp{shape_text(case["shape"])}'
            if not replay.exists():
                stage = Path(tempfile.mkdtemp(prefix='.replay-', dir=root))
                try:
                    for name in m['files']:
                        dest = stage / name; dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original / name, dest)
                    shutil.copyfile(original / 'manifest.json', stage / 'original_manifest.json')
                    derived = dict(m, context=dict(m['context'], EXECUTABLE_SHA256=build['sha256']),
                                   source_set_id=m['set_id'], transfer_validation_sha256=sha(report_path))
                    derived.pop('set_id')
                    derived['set_id'] = hashlib.sha256(json.dumps(derived, sort_keys=True).encode()).hexdigest()
                    atomic(stage / 'manifest.json', derived)
                    replay.parent.mkdir(parents=True, exist_ok=True)
                    stage.rename(replay)
                finally:
                    if stage.exists(): shutil.rmtree(stage)
            checked, missing = catalog_check(replay, case)
            require(not missing and checked['context']['EXECUTABLE_SHA256'] == build['sha256'] and checked['source_set_id'] == m['set_id'], 'Invalid replay catalog')
        results = []
        for case in planned:
            final = runtime.target_path(root, case)
            if final.exists():
                require(config['resume'] and runtime.complete(final, case), f'Existing result requires --resume and successful validation: {final}')
                results.append(dict(read_json(final / 'result.json'), result_directory=str(final)))
                print('SKIP committed:', case['case_id'], flush=True)
                continue
            attempts = root / '_attempts' / case['case_id']; attempts.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix='attempt_', dir=attempts))
            replay = root / 'precomputed_blocks' / f'decomp{shape_text(case["shape"])}' if case['mode'] == 2 else None
            result = execute_case(campaign, case, exe, replay, run)
            final.parent.mkdir(parents=True, exist_ok=True)
            run.rename(final)
            results.append(dict(result, result_directory=str(final)))
            print('COMMITTED:', case['case_id'], flush=True)
        summary = []
        for r in results:
            c = r['case']
            summary.append(dict(nodes=nodes, roles=','.join(c['roles']), global_grid=shape_text(c['grid']),
                mpi=shape_text(c['shape']), blocks='default' if c['mode'] == 0 else 'precomputed',
                wall_time_s=r['app_wall_time_s'], kernel_time_s=r['kernel_time_s'], halo_time_s=r['halo_time_s'],
                result_directory=r['result_directory']))
        write_csv(root / 'summary.csv', summary)
        comparisons = []
        for metric in ('app_wall_time_s', 'kernel_time_s', 'halo_time_s'):
            values = {role: r[metric] for r in results for role in r['case']['roles']}
            for role in ('D_top1', 'D_top2', 'D_top3'):
                comparisons.append(dict(nodes=nodes, metric=metric, role=role, C=values['C'], D=values[role],
                    C_over_D=values['C']/values[role] if values[role] else '',
                    A_over_D=values['A']/values[role] if values[role] else '', repetitions=1))
        write_csv(root / 'comparisons.csv', comparisons)
        atomic(root / 'comparisons.json', dict(comparisons=comparisons,
            note='One measured execution per unique case. Smoke tests excluded. Kernel/halo are OPS rank means, not an additive wall-time breakdown.'))


def job_script(campaign, action, nodes=None):
    command = ['bash', str(campaign / 'launcher/run.sh'), action, '--campaign', str(campaign)]
    if nodes is not None:
        command += ['--node', str(nodes)]
    return '#!/usr/bin/env bash\nset -euo pipefail\nsource ' + shlex.quote(str(campaign / 'environment.sh')) + '\n' + shell_join(command) + '\n'


def freeze_launcher(campaign, env_file):
    dest = campaign / 'launcher'; dest.mkdir(parents=True)
    for name in ('run.sh', 'launcher.py', 'runtime.py'):
        shutil.copy2(HERE / name, dest / name)
    shutil.copy2(env_file, campaign / 'environment.sh')


def scheduler_state(job):
    result = query(['sacct', '-n', '-X', '-j', job, '--format=JobIDRaw,State', '-P'])
    require(result['status'] == 'OK', f'Cannot safely inspect prior job {job}; no duplicate will be sent')
    for line in result.get('stdout', '').splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[0].strip() == job:
            return fields[1].split()[0].rstrip('+')
    raise ValueError(f'State of prior job {job} is unknown; no duplicate will be sent')


def submit(campaign, args, config):
    with lock(campaign / '.submission.lock'):
        if (campaign / 'build/build_manifest.json').is_file():
            built = read_json(campaign / 'build/build_manifest.json')
            require(sha(campaign / built['executable']) == built['sha256'], 'Frozen binary changed before submission')
        state_path = campaign / 'submissions.json'
        state = read_json(state_path) if state_path.exists() else dict(build_job='', node_jobs={})
        failures = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
        if args.resume:
            if state['build_job'] and not (campaign / 'build/build_manifest.json').is_file():
                previous = scheduler_state(state['build_job'])
                if previous in failures:
                    state.setdefault('history', []).append(dict(role='build', job_id=state['build_job'], state=previous))
                    state['build_job'] = ''
            for node, job in list(state['node_jobs'].items()):
                previous = scheduler_state(job)
                if previous in failures:
                    state.setdefault('history', []).append(dict(role=node, job_id=job, state=previous))
                    del state['node_jobs'][node]
            atomic(state_path, state)
        for rel, digest in config['launcher_sha256'].items():
            require(sha(campaign / rel) == digest, f'Frozen launcher/profile changed: {rel}')
        def sbatch(command):
            r = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            require(r.returncode == 0, f'sbatch failed: {r.stderr.strip()}')
            job = r.stdout.strip().split(';')[0]
            require(re.fullmatch('[0-9]+', job), f'Unexpected sbatch result: {r.stdout}')
            return job
        if not (campaign / 'build/build_manifest.json').is_file():
            if not state['build_job']:
                path = campaign / 'slurm/build.sh'; path.parent.mkdir(exist_ok=True)
                path.write_text(job_script(campaign, '--internal-build'))
                command = ['sbatch', '--parsable', f'--account={config["account"]}', f'--partition={config["build_partition"]}',
                           '--nodes=1', '--ntasks=1', f'--cpus-per-task={config["build_jobs"]}', '--gpus-per-node=1',
                           f'--time={config["build_time"]}', '--job-name=tgv_top3_build',
                           f'--output={campaign}/slurm/build-%j.out', str(path)]
                state['build_job'] = sbatch(command)
                atomic(state_path, state)
                print('BUILD submitted:', state['build_job'], flush=True)
            else:
                print('Existing build job:', state['build_job'], '(not resubmitted)', flush=True)
        for n in config['nodes']:
            if str(n) in state['node_jobs']:
                print(f'{n} nodes already submitted: {state["node_jobs"][str(n)]}; inspect --status before any manual retry.', flush=True)
                continue
            path = campaign / 'slurm' / f'n{n:03d}.sh'; path.parent.mkdir(exist_ok=True)
            path.write_text(job_script(campaign, '--internal-node', n))
            command = ['sbatch', '--parsable'] + resource_args(config['account'], config['partitions'][str(n)], n, config['time'])
            if state['build_job']:
                command += ['--dependency=afterok:' + state['build_job']]
            command += ['--job-name=tgv_top3_' + str(n), f'--output={campaign}/slurm/n{n:03d}-%j.out', str(path)]
            state['node_jobs'][str(n)] = sbatch(command)
            atomic(state_path, state)
            print(f'{n} nodes submitted: {state["node_jobs"][str(n)]}', flush=True)


def collect(campaign):
    config = read_json(campaign / 'campaign.json')
    summaries, comparisons, absent = [], [], []
    for n in config['nodes']:
        root = campaign / f'OPS_LOGS_{config["stamp"]}_n{n:03d}'
        if (root / 'summary.csv').is_file() and (root / 'comparisons.csv').is_file():
            summaries.extend(rows(root / 'summary.csv'))
            comparisons.extend(rows(root / 'comparisons.csv'))
        else:
            absent.append(n)
    if summaries:
        write_csv(campaign / 'summary_all_nodes.csv', summaries)
        write_csv(campaign / 'comparisons_all_nodes.csv', comparisons)
    atomic(campaign / 'collection_status.json', dict(collected_utc=now(), missing_nodes=absent, complete=not absent))
    print(f'Collected {len(summaries)} cases; nodes without summaries: {absent}', flush=True)


def campaign_directory(args):
    if args.campaign:
        return args.campaign.expanduser().resolve()
    output = args.output_root
    if output is None:
        require(args.account, 'Specify --account (or SLURM_ACCOUNT) to use /scratch/<account>/ops-decomposition/campaigns, or provide --output-root explicitly.')
        output = Path('/scratch') / args.account / 'ops-decomposition' / 'campaigns'
    return output.expanduser().resolve() / f'TGV_TOP3_{args.date_stamp}'


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog='No action prints help. --submit sends jobs; --dry-run and --analyze-resources never send jobs. Incomplete selected catalogs block execution.')
    p.add_argument('--ops-root', type=Path, default=HERE.parent.parent)
    p.add_argument('--dataset', type=Path, help='Extracted package; default: OPS_ROOT/dataset-lumi/TOP3_DATASET.tar.gz')
    p.add_argument('--output-root', type=Path, help='Campaign parent directory; default: /scratch/<account>/ops-decomposition/campaigns')
    p.add_argument('--account', default=os.environ.get('SLURM_ACCOUNT', ''), help='Slurm project account; defaults to SLURM_ACCOUNT; also determines the default output path')
    p.add_argument('--partition', default='standard-g', help='Production partition or auto (probe small-g for <=4 nodes and standard-g)')
    p.add_argument('--time', default='01:00:00', help='Allocation time for all cases AND compatibility smoke tests in each node job')
    p.add_argument('--nodes', default=','.join(map(str, NODES)))
    p.add_argument('--date-stamp', default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--env-file', type=Path, default=HERE / 'environment.sh')
    p.add_argument('--translator-python', type=Path, help='Existing Python with OPS translator requirements; otherwise build creates a venv')
    p.add_argument('--build-partition', default='small-g')
    p.add_argument('--build-time', default='00:30:00')
    p.add_argument('--build-jobs', type=int, default=4)
    p.add_argument('--resume', action='store_true', help='Reuse an existing identical campaign; never duplicate submitted job IDs')
    p.add_argument('--campaign', type=Path, help='Existing campaign directory for status/collect/internal steps')
    actions = p.add_mutually_exclusive_group()
    for name in ('validate', 'analyze-resources', 'dry-run', 'build', 'submit', 'status', 'collect'):
        actions.add_argument('--' + name, action='store_true')
    actions.add_argument('--internal-build', action='store_true', help=argparse.SUPPRESS)
    actions.add_argument('--internal-node', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--node', type=int, choices=NODES, help=argparse.SUPPRESS)
    p.add_argument('--rank-spec', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--rank-output', type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    require(sys.version_info >= (3, 6), 'Python 3.6 or newer is required for the launcher')
    if args.rank_spec:
        return runtime.rank_worker(args.rank_spec, args.rank_output)
    if args.internal_build:
        build_binary(args.campaign.resolve()); return 0
    if args.internal_node:
        require(args.node is not None, '--node required')
        run_node(args.campaign.resolve(), args.node); return 0
    if args.status or args.collect:
        require(args.campaign, '--campaign required')
        if args.collect:
            collect(args.campaign.resolve())
        else:
            state = read_json(args.campaign / 'submissions.json')
            ids = ','.join(filter(None, [state['build_job'], *state['node_jobs'].values()]))
            print(json.dumps(dict(submissions=state, queue=query(['squeue', '-j', ids]),
                accounting=query(['sacct', '-j', ids, '--format=JobID,JobName,State,ExitCode,Elapsed,Reason', '-P'])), indent=2))
        return 0
    if not any((args.validate, args.analyze_resources, args.dry_run, args.build, args.submit)):
        p.print_help(); return 0
    args.ops_root = args.ops_root.expanduser().resolve()
    args.nodes = [int(n) for n in args.nodes.split(',')]
    require(args.nodes and len(set(args.nodes)) == len(args.nodes) and set(args.nodes) <= set(NODES), 'Invalid/duplicate --nodes')
    args.nodes.sort()
    require(re.fullmatch('[A-Za-z0-9_-]+', args.date_stamp), 'Invalid date stamp')
    for name in ('partition', 'build_partition'):
        require(re.fullmatch('[A-Za-z0-9_-]+', getattr(args, name)), f'Invalid {name}')
    require(not args.account or re.fullmatch('[A-Za-z0-9_-]+', args.account), 'Invalid account')
    for value in (args.time, args.build_time):
        require(re.fullmatch(r'(?:\d+-)?\d+:\d{2}:\d{2}', value), 'Use Slurm time [days-]HH:MM:SS')
    require(1 <= args.build_jobs <= 56, 'Invalid build concurrency')
    campaign = campaign_directory(args) if not args.validate else None
    dataset, dataset_manifest = ensure_dataset(args.ops_root, args.dataset)
    cases, coverage = make_plan(dataset, args.nodes)
    incomplete = [c for c in coverage if c['status'] != 'COMPLETE']
    print(f'Plan: {len(args.nodes)} node sections; {len(cases)} unique measured cases; {len(incomplete)} incomplete roles.', flush=True)
    if args.validate:
        require(not incomplete, f'Incomplete selected catalogs: {sorted({c["nodes"] for c in incomplete})}; see dataset missing_blocks.csv')
        print('PASS: selected dataset/catalogs. Target executable is not validated until build + runtime metadata smoke checks.')
        return 0
    # Read-only analysis/dry run write their plan to a separate report directory.
    if args.analyze_resources or args.dry_run:
        report_dir = campaign / 'preflight'; report_dir.mkdir(parents=True, exist_ok=True)
        atomic(report_dir / 'plan.json', dict(cases=cases, coverage=coverage, dataset=str(dataset),
            measured_repetitions=1, simulation_steps=500, compatibility_smoke_steps=1,
            output_campaign=str(campaign), would_submit=False))
        write_csv(report_dir / 'coverage.csv', coverage)
        if args.analyze_resources:
            resource = analyze(args, coverage)
            atomic(report_dir / 'resource_report.json', resource)
            write_csv(report_dir / 'submission_plan.csv', [{k: v for k, v in r.items() if k != 'probe'} for r in resource['reports']])
        else:
            for n in args.nodes:
                count = sum(c['nodes'] == n for c in cases)
                blocked = any(c['nodes'] == n for c in incomplete)
                print(f'{n} nodes / {8*n} ranks: {count} measured cases; {"BLOCKED: missing blocks" if blocked else "dataset ready; GPU/MPI not tested"}', flush=True)
            print('Dry run: no modules, build, scheduler probe or submission performed.')
        print('Report:', report_dir)
        return 0
    require(not incomplete, f'Cannot build/submit selected campaign: missing blocks for nodes {sorted({c["nodes"] for c in incomplete})}. Update/repair the dataset; no selected nodes will be skipped.')
    require(args.env_file.is_file(), 'Missing environment profile')
    require(not re.search(r'\s', str(campaign)), 'OPS Makefiles require a campaign/build path without whitespace; choose another --output-root')
    resource = analyze(args, coverage) if args.submit else None
    if args.submit:
        campaign.mkdir(parents=True, exist_ok=True)
        atomic(campaign / 'resource_report.json', resource)
        require(args.account, '--account required')
        partitions = select_partitions(resource, args)
        build_probe = query(['sbatch', '--test-only', f'--account={args.account}', f'--partition={args.build_partition}',
            '--nodes=1', '--ntasks=1', f'--cpus-per-task={args.build_jobs}', '--gpus-per-node=1', f'--time={args.build_time}', '--wrap=true'])
        atomic(campaign / 'build_resource_probe.json', build_probe)
        require(build_probe['status'] == 'OK', f'Build allocation rejected/unverified: {build_probe}')
    else:
        partitions = {str(n): args.partition for n in args.nodes}
    campaign.mkdir(parents=True, exist_ok=True)
    with lock(campaign / '.prepare.lock'):
        config_path = campaign / 'campaign.json'
        if config_path.exists():
            require(args.resume, 'Campaign already exists; use --resume with the same inputs')
            config = read_json(config_path)
            require(config['nodes'] == args.nodes and config['cases'] == cases and config['dataset_manifest_sha256'] == sha(dataset / 'dataset_manifest.json'), 'Resume plan differs')
            require(config['account'] == args.account and config['time'] == args.time, 'Resume account/time differ')
            require(sha(campaign / 'environment.sh') == sha(args.env_file), 'Resume environment profile differs')
            if resource:
                require(all(any(r['nodes'] == n and r['partition'] == config['partitions'][str(n)] and r['eligibility'] == 'ELIGIBLE' for r in resource['reports']) for n in args.nodes), 'Previously selected partition is no longer eligible')
            config['resume'] = True
        else:
            freeze_launcher(campaign, args.env_file.resolve())
            snapshot_ops(args.ops_root, campaign)
            config = dict(version=VERSION, stamp=args.date_stamp, nodes=args.nodes, cases=cases,
                dataset=str(dataset), dataset_manifest_sha256=sha(dataset / 'dataset_manifest.json'),
                ops_root=str(args.ops_root), account=args.account, partitions=partitions, time=args.time,
                build_partition=args.build_partition, build_time=args.build_time, build_jobs=args.build_jobs,
                translator_python=str(args.translator_python.absolute()) if args.translator_python else '',
                resume=False, created_utc=now(), launcher_sha256={str(f.relative_to(campaign)): sha(f)
                    for f in [campaign / 'environment.sh', *(campaign / 'launcher').glob('*')]})
        atomic(config_path, config)
        if resource:
            atomic(campaign / 'resource_report.json', resource)
    if args.build:
        command = ['bash', '-c', 'set -euo pipefail; source "$1"; shift; exec "$@"', 'tgv-build',
            str(campaign / 'environment.sh'), 'bash', str(campaign / 'launcher/run.sh'), '--internal-build', '--campaign', str(campaign)]
        subprocess.run(command, check=True)
    else:
        submit(campaign, args, config)
    print('Campaign:', campaign, flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
