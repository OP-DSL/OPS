#!/usr/bin/env bash
#SBATCH --job-name=tgv-yz-n016
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
# Compare MPI shapes with one fixed precomputed catalog; never tune or build.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPS_YZ_SOURCE_SCRIPT="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
# Slurm copies this script into its spool: use the submission directory there.
export OPS_YZ_DEFAULT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$script_dir/../.." && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
exec "${OPS_LAUNCHER_PYTHON:-python3}" - "$@" <<'PY'
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser(description='TGV StoreNone: 16 nodes, 2016^3, MPI 8x4x4 vs 2x8x8; fixed blocks, GPU Direct, no tuning.')
parser.add_argument('--executable', required=True, type=Path, help='Existing TGV StoreNone MPI+HIP executable supporting the fourth argument (steps)')
parser.add_argument('--blocks', required=True, type=Path, help='Shared blocks.csv from the 16-node 8x4x4 precomputed catalog')
parser.add_argument('--output', required=True, type=Path, help='New output directory; existing directories are rejected')
parser.add_argument('--ops-root', type=Path, default=Path(os.environ['OPS_YZ_DEFAULT_ROOT']), help='OPS checkout (default: checkout containing script, or Slurm submission directory)')
parser.add_argument('--env-file', type=Path, help='Site module profile; defaults to lumi_tgv_top3/environment.sh')
parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print plan; no files, GPU execution or submission')
args = parser.parse_args()

def require(ok, message):
    if not ok:
        raise SystemExit('ERROR: ' + message)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def write_csv(path, rows):
    require(bool(rows), 'No rows for ' + str(path))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

root = args.ops_root.resolve()
helpers = root / 'scripts/lumi_tgv_top3'
exe, blocks, output = args.executable.resolve(), args.blocks.resolve(), args.output.resolve()
profile = (args.env_file or helpers / 'environment.sh').resolve()
require(exe.is_file() and os.access(str(exe), os.X_OK), 'Executable missing or not executable: ' + str(exe))
require(blocks.is_file(), 'Missing blocks.csv: ' + str(blocks))
require(profile.is_file() and (helpers / 'runtime.py').is_file(), 'Missing environment/runtime; supply --ops-root and optionally --env-file')
require(not output.exists(), 'Output already exists: ' + str(output))
with blocks.open() as f:
    catalog = list(csv.DictReader(f))
try:
    ids = [int(row['kernel_id']) for row in catalog]
    require(len(ids) == 25 and set(ids) == set(range(2, 27)), 'Expected exactly TGV tuned kernel IDs 2..26; kernel 1 keeps default fallback')
    for row in catalog:
        b = [int(row[k]) for k in ('bx', 'by', 'bz')]
        require(all(v > 0 and v & (v - 1) == 0 for v in b) and 32 <= b[0]*b[1]*b[2] <= 1024,
                'Invalid block: ' + str(row))
except (KeyError, ValueError):
    raise SystemExit('ERROR: Expected columns kernel_id,bx,by,bz with integer values')
shapes = {'8x4x4': [8, 4, 4], '2x8x8': [2, 8, 8]}
order = [(rep, shape) for rep in (1, 2, 3)
         for shape in (['8x4x4', '2x8x8'] if rep % 2 else ['2x8x8', '8x4x4'])]
plan = dict(grid=[2016]*3, nodes=16, ranks=128, steps=500, repetitions=3,
            shapes=shapes, order=order, smoke_steps=1, gpu_direct=True,
            executable_source=str(exe), executable_sha256=digest(exe),
            blocks_source=str(blocks), blocks_sha256=digest(blocks),
            comparison='Same shared 8x4x4 catalog for both shapes; transfer, not retuning.',
            wall_metric='Application-reported Total Wall time, maximum across ranks',
            kernel_halo_metric='OPS totals printed by rank 0 (rank means)',
            grid_note='First logged range per kernel, rank 0 only; not all ranks or calls.')
print(json.dumps(plan, indent=2), flush=True)
if args.dry_run:
    sys.exit(0)
require(os.environ.get('SLURM_JOB_ID'), 'Run through sbatch, or use --dry-run')
require(int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0'))) == 16,
        'Expected an allocation of exactly 16 nodes')
require(shutil.which('srun'), 'srun not found')
# Load the same HIP/Cray MPI environment used by the existing launcher.
env_result = subprocess.run(['bash', '-c', 'set -e; source "$1" >&2; env -0', 'bash', str(profile)],
                            stdout=subprocess.PIPE, check=True)
env = dict(item.decode().split('=', 1) for item in env_result.stdout.split(b'\0') if item)
env['MPICH_GPU_SUPPORT_ENABLED'] = '1'
env['OMP_NUM_THREADS'] = '1'
output.mkdir(parents=True, exist_ok=False)
for name in ('support', 'precomputed', 'build'):
    (output / name).mkdir()
shutil.copy2(helpers / 'runtime.py', output / 'support/runtime.py')
shutil.copy2(profile, output / 'support/environment.sh')
shutil.copy2(os.environ['OPS_YZ_SOURCE_SCRIPT'], output / 'support/compare_yz_n016.sh')
shutil.copy2(exe, output / 'build/tgvstorenone_mpi_hip')
shutil.copy2(blocks, output / 'precomputed/blocks.csv')
require(digest(output / 'build/tgvstorenone_mpi_hip') == plan['executable_sha256'] and
        digest(output / 'precomputed/blocks.csv') == plan['blocks_sha256'], 'Inputs changed during copying')
# Runtime needs this dispatch field; this is not a historical compatibility certificate.
write_json(output / 'precomputed/manifest.json', dict(schema_version=2, purpose='fixed-catalog geometry transfer',
           source=str(blocks), sha256=plan['blocks_sha256']))
plan.update(slurm_job_id=os.environ['SLURM_JOB_ID'],
            script_sha256=digest(output / 'support/compare_yz_n016.sh'),
            runtime_sha256=digest(output / 'support/runtime.py'),
            profile_sha256=digest(output / 'support/environment.sh'),
            loaded_modules=env.get('LOADEDMODULES', ''))
write_json(output / 'plan.json', plan)
spec = importlib.util.spec_from_file_location('yz_runtime', str(output / 'support/runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
# Linkage preflight uses the loaded module environment.
link = subprocess.run(['ldd', str(output / 'build/tgvstorenone_mpi_hip')], env=env,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
(output / 'build/ldd.log').write_text(link.stdout)
require(link.returncode == 0 and 'libmpi_gtl_hsa' in link.stdout and 'not found' not in link.stdout,
        'GPU-aware MPI linkage unavailable; see build/ldd.log')
worker = 'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import runtime; sys.exit(runtime.rank_worker(Path(sys.argv[2]),Path(sys.argv[3])))'
results, geometry = [], []

def execute(shape, rep, steps, smoke=False):
    run = output / ('smoke' if smoke else 'measurements') / ('decomp' + shape) / ('iter_' + str(rep))
    (run / 'ranks').mkdir(parents=True)
    (run / 'rank_0').mkdir()
    case = dict(nodes=16, ranks=128, grid=[2016]*3, niter=steps, shape=shapes[shape], mode=2)
    write_json(run / 'execution.json', dict(case=case, executable=str(output / 'build/tgvstorenone_mpi_hip'),
               catalog=str(output / 'precomputed'), workdir=str(run)))
    command = ['srun', '--nodes=16', '--ntasks=128', '--ntasks-per-node=8', '--cpus-per-task=7',
               '--kill-on-bad-exit=1', '--cpu-bind=map_cpu:49,57,17,25,1,9,33,41',
               sys.executable, '-c', worker, str(output / 'support'), str(run / 'execution.json'), str(run)]
    print(('SMOKE' if smoke else 'MEASURE') + ' ' + shape + ' repetition ' + str(rep), flush=True)
    write_json(run / 'command.json', command)
    with (run / 'srun.log').open('w') as log:
        rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    statuses = [json.loads(p.read_text()) for p in (run / 'ranks').glob('status_*.json')]
    require(rc == 0 and len(statuses) == 128 and {s['rank_id'] for s in statuses} == set(range(128))
            and all(s['rank_status'] == 'OK' for s in statuses), 'Run failed: ' + str(run))
    times = [s['app_reported_wall_time_s'] for s in statuses if s['app_reported_wall_time_s'] is not None]
    require(times and all(math.isfinite(t) and t > 0 for t in times), 'Missing/invalid application wall time: ' + str(run))
    metrics = runtime.ops_totals(run / 'ranks/rank_0.log')
    record = dict(shape=shape, repetition=rep, steps=steps, wall_time_s=max(times), **metrics)
    write_json(run / 'result.json', dict(status='OK', metrics=record, ranks=statuses))
    # Require diagnostic CSVs and retain their actual geometry, not derived global sizes.
    csv_path = run / 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv'
    seen = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            kid = row.get('kernel_id', '')
            if not kid.isdigit() or kid in seen:
                continue
            seen.add(kid)
            if not smoke:
                geometry.append(dict(shape=shape, repetition=rep, rank=0, kernel_id=kid,
                    **{k: row[k] for k in ('grid_x', 'grid_y', 'grid_z', 'bx', 'by', 'bz')}))
    require({'2', '23', '24'} <= seen, 'Missing expected TGV kernels in rank-0 CSV: ' + str(csv_path))
    if not smoke:
        results.append(record)
        write_csv(output / 'runs.csv', results)
        write_csv(output / 'rank0_grids.csv', geometry)

# Launch/catalog checks only; these do not certify numerical equivalence.
for shape in shapes:
    execute(shape, 1, 1, smoke=True)
for rep, shape in order:
    execute(shape, rep, 500)
summary = []
for shape in shapes:
    group = [r for r in results if r['shape'] == shape]
    summary.append(dict(shape=shape, repetitions=len(group),
        **{key: statistics.median(r[key] for r in group) for key in ('wall_time_s', 'kernel_time_s', 'halo_time_s')}))
for row in summary:
    row['wall_speedup_vs_8x4x4'] = summary[0]['wall_time_s'] / row['wall_time_s']
write_csv(output / 'summary.csv', summary)
print(json.dumps(summary, indent=2))
print('Completed: ' + str(output / 'summary.csv'))
PY
