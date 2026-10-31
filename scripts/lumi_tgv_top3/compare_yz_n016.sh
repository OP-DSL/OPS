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
# TGV StoreNone: unchanged 2016^3 input, new MPI, existing 8x4x4 reference.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPS_YZ_SOURCE_SCRIPT="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
export OPS_YZ_DEFAULT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$script_dir/../.." && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
exec "${OPS_LAUNCHER_PYTHON:-python3}" - "$@" <<'PY'
import argparse
import csv
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise SystemExit('ERROR: ' + message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_config(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def config_text(fields):
    return ''.join(str(k) + '=' + str(v) + '\n' for k, v in fields.items())


def write_config(path, fields):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(config_text(fields))
    os.replace(str(temp), str(path))


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def positive(value):
    return math.isfinite(float(value)) and float(value) > 0


def rank_worker(config_path):
    c = read_config(config_path)
    run = config_path.parent
    rank, local = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_LOCALID'])
    require(0 <= rank < 128 and 0 <= local < 8, 'Expected 128 ranks, eight local GCDs')
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(local), MPICH_GPU_SUPPORT_ENABLED='1',
               OMP_NUM_THREADS='1', OPS_AUTOTUNE_MODE='2', OPS_BLOCKSIZE_CSV=c['PRECOMPUTED_BLOCKS_FILE'],
               OPS_AUTOTUNE_LOGS='1' if rank == 0 else '0',
               OPS_DISABLE_CSV_LOGS='0' if rank == 0 else '1',
               OPS_CSV_LOG_LEVEL='0' if rank == 0 else '2')
    for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env['OPS_LOGS_DIR'] = str(run / ('rank_0' if rank == 0 else '.disabled_csv/' + str(rank)))
    command = [c['EXECUTABLE']] + c['APP_ARGS'].split()
    command += ['OPS_FORCE_DECOMP_' + a + '=' + c['DECOMP_P' + a] for a in 'XYZ']
    command += ['-gpudirect', '-OPS_DIAGS=2', '-OPS_AUTOTUNE_MODE=2']
    log = run / 'ranks' / ('rank_' + str(rank) + '.log')
    start = time.monotonic()
    with log.open('w') as f:
        rc = subprocess.run(command, env=env, cwd=str(run), stdin=subprocess.DEVNULL,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.monotonic() - start
    text = log.read_text(errors='replace')
    loaded = any(int(n) == 25 and p.strip() == c['PRECOMPUTED_BLOCKS_FILE']
                 for n, p in re.findall(r'Loaded (\d+) precomputed block sizes from ([^\r\n]+)', text))
    reported = re.findall(r'Total\s+Wall\s+time\s*:?\s*([0-9.Ee+-]+)', text, re.I)
    good = rc == 0 and loaded and not re.search(r'considered FAILED|validation.*failed|nan detected', text, re.I)
    if rank == 0:
        good = good and 'GPU Direct enabled' in text and 'Tiling enabled' not in text
    status = dict(rank_id=rank, local_rank=local, gpu_id=local, hostname=os.uname().nodename,
                  rank_wall_time_s=elapsed, app_reported_wall_time_s=reported[-1] if reported else '',
                  rank_status='OK' if good else 'FAILED', returncode=rc, precomputed_loaded=loaded)
    write_csv(run / 'ranks' / ('status_' + str(rank) + '.csv'), [status])
    return 0 if good else 1


parser = argparse.ArgumentParser(description='TGV StoreNone, 16 nodes: new MPI with the existing 8x4x4 blocks and historical reference; unchanged 2016^3 input.')
parser.add_argument('--account', default=os.environ.get('SLURM_JOB_ACCOUNT') or os.environ.get('SLURM_ACCOUNT') or 'project_465003260')
parser.add_argument('--campaign', type=Path, help='Reference TGV_TOP3_20261022_083420 campaign (automatic project scratch lookup)')
parser.add_argument('--executable', type=Path, help='Override reference executable path; hash must match the reference')
parser.add_argument('--blocks', type=Path, help='Override reference blocks.csv path; contents must match the reference')
parser.add_argument('--output', type=Path, help='Override automatic project scratch TGV_YZ_N016_<timestamp> directory')
parser.add_argument('--ops-root', type=Path, default=Path(os.environ['OPS_YZ_DEFAULT_ROOT']))
parser.add_argument('--env-file', type=Path)
parser.add_argument('--decompositions', default='2x8x8', help='Comma-separated new MPI shapes; default 2x8x8. The existing 8x4x4 reference is never rerun.')
parser.add_argument('--repetitions', type=int, default=1, help='Measurements per new MPI shape; default one, as in the Top3 campaign')
parser.add_argument('--rank-config', type=Path, help=argparse.SUPPRESS)
action = parser.add_mutually_exclusive_group()
action.add_argument('--dry-run', action='store_true')
action.add_argument('--submit', action='store_true', help='Validate inputs before submitting the 16-node job')
args = parser.parse_args()
if args.rank_config:
    sys.exit(rank_worker(args.rank_config.resolve()))

require(args.repetitions > 0, 'Repetitions must be positive')
shapes = {}
for name in args.decompositions.split(','):
    require(re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*', name), 'Invalid MPI shape: ' + name)
    dims = [int(v) for v in name.split('x')]
    require(dims[0]*dims[1]*dims[2] == 128 and all(2016 % v == 0 for v in dims),
            'MPI shape must give 128 equal subdomains of 2016^3: ' + name)
    require(name != '8x4x4', '8x4x4 is the existing historical reference; it will not be rerun')
    require(name not in shapes, 'Duplicate MPI shape: ' + name)
    shapes[name] = dims
require(args.account and '/' not in args.account and not any(c.isspace() for c in args.account), 'Invalid Slurm account')
root = args.ops_root.resolve()
campaigns = Path('/scratch') / args.account / 'ops-decomposition/campaigns'
reference_name = 'TGV_TOP3_20261022_083420'
logs_relative = Path('OPS_LOGS_20261022_083420_n016')
blocks_relative = logs_relative / 'precomputed_blocks/decomp8x4x4/blocks.csv'
candidates = [campaigns / reference_name] + [Path(m) / args.account / 'ops-decomposition/campaigns' / reference_name
              for m in ('/pfs/lustrep1/scratch', '/pfs/lustrep2/scratch')]
campaign = (args.campaign or next((p for p in candidates if (p / 'build/tgvstorenone_mpi_hip').is_file()
                                  and (p / blocks_relative).is_file()), candidates[0])).resolve()
exe = (args.executable or campaign / 'build/tgvstorenone_mpi_hip').resolve()
blocks = (args.blocks or campaign / blocks_relative).resolve()
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
output = (args.output or campaigns / ('TGV_YZ_N016_' + stamp)).resolve()
# Keep the submission timestamp when the job starts later in the queue.
output_stamp = re.fullmatch(r'TGV_YZ_N016_(\d{8}_\d{6}_\d{6})', output.name)
if output_stamp:
    stamp = output_stamp.group(1)
profile = (args.env_file or root / 'scripts/lumi_tgv_top3/environment.sh').resolve()
require(exe.is_file() and os.access(str(exe), os.X_OK), 'Executable missing or not executable: ' + str(exe) + '; set --campaign if moved')
require(blocks.is_file(), 'Missing blocks.csv: ' + str(blocks))
require(profile.is_file(), 'Missing environment profile: ' + str(profile))
require(not output.exists(), 'Output already exists: ' + str(output))
with blocks.open() as f:
    catalog = list(csv.DictReader(f))
try:
    ids = [int(r['kernel_id']) for r in catalog]
    require(len(ids) == 25 and set(ids) == set(range(2, 27)), 'Expected exactly TGV tuned kernel IDs 2..26')
    for row in catalog:
        b = [int(row[k]) for k in ('bx', 'by', 'bz')]
        require(all(v > 0 and v & (v-1) == 0 for v in b) and 32 <= b[0]*b[1]*b[2] <= 1024, 'Invalid block: ' + str(row))
except (KeyError, ValueError):
    raise SystemExit('ERROR: Expected integer kernel_id,bx,by,bz columns')
exe_hash, blocks_hash = digest(exe), digest(blocks)
require((campaign / blocks_relative).is_file() and digest(campaign / blocks_relative) == blocks_hash,
        'Blocks must match the historical 8x4x4 catalog')
references, reference_paths = {}, {}
for role, variant, mode in [('A', 'autotune_off', '0'), ('C', 'autotune_precomputed', '2')]:
    found = list((campaign / logs_relative).glob('tgvstorenone_hip_2016x2016x2016_cfg1_comparison_*/mode_1/decomp8x4x4/iter_1/' + variant + '/config.txt'))
    require(len(found) == 1, 'Expected one historical ' + variant + '/config.txt under ' + str(campaign))
    c = read_config(found[0])
    expected = dict(APP_NAME='tgvstorenone', GRID_X='2016', GRID_Y='2016', GRID_Z='2016', NITER='500',
                    NUM_NODES='16', NUM_MPI_RANKS='128', DECOMP_PX='8', DECOMP_PY='4', DECOMP_PZ='4',
                    OPS_AUTOTUNE_MODE=mode, GPU_DIRECT='YES', RUN_STATUS='OK', RUN_EXIT_CODE='0', EXECUTABLE_SHA256=exe_hash)
    require(all(c.get(k) == v for k, v in expected.items()), 'Historical context/executable differs: ' + str(found[0]))
    try:
        require(all(positive(c[k]) for k in ('APP_WALL_TIME', 'KERNEL_TOTAL_TIME', 'HALO_TOTAL_TIME')), 'Invalid historical timing')
    except (KeyError, ValueError):
        raise SystemExit('ERROR: Missing/invalid timings in ' + str(found[0]))
    references[role], reference_paths[role] = c, found[0]
plan = dict(APP_NAME='tgvstorenone', APP_ARGS='2016 2016 2016 500', GRID_X=2016, GRID_Y=2016, GRID_Z=2016,
            NUM_NODES=16, NUM_MPI_RANKS=128, NITER=500, DECOMPOSITIONS=','.join(shapes), REPETITIONS=args.repetitions,
            ACCOUNT=args.account, REFERENCE_CAMPAIGN=campaign, OUTPUT_DIRECTORY=output, GPU_DIRECT='YES', OPS_TILING='NO',
            EXECUTABLE_SOURCE=exe, EXECUTABLE_SHA256=exe_hash, BLOCKS_SOURCE=blocks, BLOCKS_SHA256=blocks_hash,
            REFERENCE_KIND='HISTORICAL', REFERENCE_MPI='8x4x4', REFERENCE_RERUN='NO',
            NO_AUTOTUNE_CONFIG=reference_paths['A'], DEFAULT_MPI_PRECOMPUTED_CONFIG=reference_paths['C'],
            WALL_TIME_NO_AUTOTUNE=references['A']['APP_WALL_TIME'], WALL_TIME_DEFAULT_MPI_PRECOMPUTED=references['C']['APP_WALL_TIME'])
print(config_text(plan), flush=True)
if args.dry_run:
    sys.exit(0)
if args.submit:
    require(not os.environ.get('SLURM_JOB_ID'), 'Use --submit from a login node')
    require(shutil.which('sbatch'), 'sbatch not found')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['sbatch', '--account=' + args.account, '--output=' + str(output.parent / (output.name + '-%j.out')),
               os.environ['OPS_YZ_SOURCE_SCRIPT'], '--account', args.account, '--ops-root', str(root),
               '--campaign', str(campaign), '--executable', str(exe), '--blocks', str(blocks), '--output', str(output),
               '--env-file', str(profile), '--decompositions', ','.join(shapes), '--repetitions', str(args.repetitions)]
    subprocess.run(command, check=True)
    sys.exit(0)
if not os.environ.get('SLURM_JOB_ID'):
    print('Inputs validated. Add --submit to send this experiment to Slurm.')
    sys.exit(0)
require(int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0'))) == 16, 'Expected an allocation of exactly 16 nodes')
require(shutil.which('srun'), 'srun not found')
profile_result = subprocess.run(['bash', '-c', 'set -e; source "$1" >&2; env -0', 'bash', str(profile)], stdout=subprocess.PIPE, check=True)
env = dict(item.decode().split('=', 1) for item in profile_result.stdout.split(b'\0') if item)
env.update(MPICH_GPU_SUPPORT_ENABLED='1', OMP_NUM_THREADS='1', OPS_LAUNCHER_PYTHON=sys.executable)
output.mkdir(parents=True, exist_ok=False)
logs = output / ('OPS_LOGS_' + stamp + '_n016')
for p in (output / 'launcher', output / 'build', logs / 'references', logs / 'precomputed_blocks'):
    p.mkdir(parents=True, exist_ok=True)
shutil.copy2(os.environ['OPS_YZ_SOURCE_SCRIPT'], output / 'launcher/compare_yz_n016.sh')
shutil.copy2(profile, output / 'environment.sh')
shutil.copy2(exe, output / 'build/tgvstorenone_mpi_hip')
frozen_exe = output / 'build/tgvstorenone_mpi_hip'
require(digest(frozen_exe) == exe_hash, 'Executable changed while copying')
for role in references:
    shutil.copy2(reference_paths[role], logs / 'references' / (role + '_config.txt'))
    require(read_config(logs / 'references' / (role + '_config.txt')) == references[role], 'Reference changed while copying')
for shape in shapes:
    dest = logs / 'precomputed_blocks' / ('decomp' + shape)
    dest.mkdir()
    shutil.copy2(blocks, dest / 'blocks.csv')
    require(digest(dest / 'blocks.csv') == blocks_hash, 'Blocks changed while copying')
    write_config(dest / 'config.txt', dict(SOURCE_MPI='8x4x4', TARGET_MPI=shape, BLOCKS_SOURCE=blocks,
                 BLOCKS_SHA256=blocks_hash, POLICY='FIXED_CATALOG_TRANSFER', RETUNED='NO'))
plan.update(SLURM_JOB_ID=os.environ['SLURM_JOB_ID'], SCRIPT_SHA256=digest(output / 'launcher/compare_yz_n016.sh'),
            ENVIRONMENT_SHA256=digest(output / 'environment.sh'), LOADEDMODULES=env.get('LOADEDMODULES', ''),
            RUN_STATUS='RUNNING', OPS_LOGS_DIRECTORY=logs)
write_config(output / 'config.txt', plan)
link = subprocess.run(['ldd', str(frozen_exe)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
(output / 'build/ldd.log').write_text(link.stdout)
if not (link.returncode == 0 and 'libmpi_gtl_hsa' in link.stdout and 'not found' not in link.stdout):
    plan.update(RUN_STATUS='FAILED', ERROR='GPU-aware MPI linkage unavailable; see build/ldd.log')
    write_config(output / 'config.txt', plan)
    raise SystemExit('ERROR: ' + plan['ERROR'])
summary, comparisons, geometry = [], [], []


def execute(shape, rep, smoke=False):
    steps = 1 if smoke else 500
    label = 'tgvstorenone_hip_2016x2016x2016_cfg1_comparison_' + stamp + '_n016'
    base = logs / '_compatibility' if smoke else logs
    run = base / label / 'mode_1' / ('decomp' + shape) / ('iter_' + str(rep)) / 'autotune_precomputed'
    (run / 'ranks').mkdir(parents=True)
    (run / 'rank_0').mkdir()
    c = dict(APP_NAME='tgvstorenone', APP_ARGS='2016 2016 2016 ' + str(steps), CAMPAIGN_ID=output.name,
             CASE_ID='n016_decomp' + shape + '_mode2', RUN_CLASS='SMOKE' if smoke else 'SCIENTIFIC',
             SCIENTIFIC_RESULT='NO' if smoke else 'YES', NITER=steps, GPU_ARCH='gfx90a', EXECUTABLE=frozen_exe,
             EXECUTABLE_SHA256=exe_hash, DECOMPOSITION_MODE=1, GRID_X=2016, GRID_Y=2016, GRID_Z=2016,
             GRID_CONFIG='cfg1', TOTAL_CELLS=2016**3, NUM_NODES=16, NUM_MPI_RANKS=128, GPUS_PER_NODE=8,
             NUM_GPUS_USED=128, RANKS_PER_GPU=1, DECOMP_PX=shapes[shape][0], DECOMP_PY=shapes[shape][1], DECOMP_PZ=shapes[shape][2],
             OPS_AUTOTUNE_MODE=2, GPU_DIRECT='UNVERIFIED', MPICH_GPU_SUPPORT_ENABLED=1, OPS_TILING='NO',
             RUN_ITERATION=rep, TOP3_ROLES='D_new', CSV_LOG_RANKS=0,
             PRECOMPUTED_CATALOG=logs / 'precomputed_blocks' / ('decomp' + shape),
             PRECOMPUTED_BLOCKS_FILE=logs / 'precomputed_blocks' / ('decomp' + shape) / 'blocks.csv',
             PRECOMPUTED_SET_ID=blocks_hash, PRECOMPUTED_SET_ID_KIND='BLOCKS_SHA256', BLOCKS_SOURCE_MPI='8x4x4',
             REFERENCE_KIND='HISTORICAL', REFERENCE_RERUN='NO', NO_AUTOTUNE_REFERENCE='DEFAULT_MPI_DEFAULT_BLOCKS',
             NO_AUTOTUNE_MPI='8x4x4', PRECOMPUTED_MPI=shape,
             NO_AUTOTUNE_CONFIG=reference_paths['A'], DEFAULT_MPI_PRECOMPUTED_CONFIG=reference_paths['C'],
             PRECOMPUTED_CONFIG=run / 'config.txt', WALL_TIME_NO_AUTOTUNE=references['A']['APP_WALL_TIME'],
             WALL_TIME_DEFAULT_MPI_PRECOMPUTED=references['C']['APP_WALL_TIME'], WALL_TIME_PRECOMPUTED='NA',
             SPEEDUP_PRECOMPUTED='NA', SPEEDUP_DEFAULT_MPI_PRECOMPUTED='NA', APP_WALL_TIME='NA', JOB_WALL_TIME='NA',
             PROCESS_EXIT_CODE='NA', RUN_EXIT_CODE='NA', RUN_STATUS='RUNNING', VALIDATION_STATUS='PENDING',
             COMMIT_STATUS='UNCOMMITTED', KERNEL_TOTAL_TIME='NA', HALO_TOTAL_TIME='NA')
    write_config(run / 'config.txt', c)
    command = ['srun', '--nodes=16', '--ntasks=128', '--ntasks-per-node=8', '--cpus-per-task=7',
               '--kill-on-bad-exit=1', '--cpu-bind=map_cpu:49,57,17,25,1,9,33,41',
               'bash', str(output / 'launcher/compare_yz_n016.sh'), '--rank-config', str(run / 'config.txt')]
    (run / 'command.txt').write_text(' '.join(shlex.quote(s) for s in command) + '\n')
    print(('SMOKE ' if smoke else 'MEASURE ') + shape + ' iteration ' + str(rep), flush=True)
    start = time.monotonic()
    try:
        with (run / 'srun.log').open('w') as f:
            rc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT).returncode
    except OSError as e:
        c.update(RUN_STATUS='FAILED', RUN_EXIT_CODE=1, VALIDATION_STATUS='FAIL', ERROR=str(e).replace('\n', ' '))
        write_config(run / 'config.txt', c)
        raise
    c.update(PROCESS_EXIT_CODE=rc, JOB_WALL_TIME=time.monotonic()-start)
    rank_log = run / 'ranks/rank_0.log'
    if rank_log.exists():
        shutil.copyfile(rank_log, run / 'run.log')
    try:
        statuses = []
        for p in (run / 'ranks').glob('status_*.csv'):
            with p.open() as f:
                statuses.extend(csv.DictReader(f))
        statuses.sort(key=lambda s: int(s['rank_id']))
        write_csv(run / 'mpi_rank_metrics.csv', statuses)
        require(rc == 0 and len(statuses) == 128 and {int(s['rank_id']) for s in statuses} == set(range(128))
                and all(s['rank_status'] == 'OK' and s['precomputed_loaded'] == 'True' for s in statuses), 'MPI rank failure or missing status')
        times = [float(s['app_reported_wall_time_s']) for s in statuses if s['app_reported_wall_time_s']]
        require(times and all(positive(v) for v in times), 'Missing/invalid application wall time')
        text = rank_log.read_text(errors='replace')
        require('GPU Direct enabled' in text and 'Tiling enabled' not in text, 'GPU Direct/tiling check failed')
        metrics = {}
        for key, pattern in [('KERNEL_TOTAL_TIME', 'Total kernel time'), ('HALO_TOTAL_TIME', 'Total (?:tiled )?halo exchange time')]:
            values = re.findall(r'^\s*' + pattern + r':\s*([0-9.Ee+-]+)\s*$', text, re.M)
            require(len(values) == 1 and math.isfinite(float(values[0])) and float(values[0]) >= 0, 'Invalid ' + key)
            metrics[key] = float(values[0])
        found, ranges = set(), []
        with (run / 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv').open() as f:
            for r in csv.DictReader(f):
                kid = r.get('kernel_id', '')
                if not kid.isdigit() or kid in found:
                    continue
                found.add(kid)
                ranges.append(dict(mpi=shape, repetition=rep, rank=0, kernel_id=kid,
                    **{k: r[k] for k in ('grid_x', 'grid_y', 'grid_z', 'bx', 'by', 'bz')}))
        require({'2', '23', '24'} <= found, 'Missing expected TGV kernel grids')
        c.update(metrics)
        c.update(APP_WALL_TIME=max(times), WALL_TIME_PRECOMPUTED=max(times), GPU_DIRECT='YES', RUN_EXIT_CODE=0,
                 RUN_STATUS='OK', VALIDATION_STATUS='PASS', COMMIT_STATUS='UNCOMMITTED' if smoke else 'COMMITTED')
        if not smoke:
            c['SPEEDUP_PRECOMPUTED'] = float(references['A']['APP_WALL_TIME']) / max(times)
            c['SPEEDUP_DEFAULT_MPI_PRECOMPUTED'] = float(references['C']['APP_WALL_TIME']) / max(times)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        c.update(RUN_STATUS='FAILED', RUN_EXIT_CODE=rc or 1, VALIDATION_STATUS='FAIL', ERROR=str(e).replace('\n', ' '))
        write_config(run / 'config.txt', c)
        raise SystemExit('ERROR: Run failed: ' + str(run) + ': ' + str(e))
    write_config(run / 'config.txt', c)
    if not smoke:
        summary.append(dict(nodes=16, roles='D_new', global_grid='2016x2016x2016', mpi=shape, blocks='precomputed',
                       wall_time_s=c['APP_WALL_TIME'], kernel_time_s=c['KERNEL_TOTAL_TIME'], halo_time_s=c['HALO_TOTAL_TIME'],
                       result_directory=str(run), iteration=rep, reference_kind='HISTORICAL'))
        for metric, key in [('app_wall_time_s', 'APP_WALL_TIME'), ('kernel_time_s', 'KERNEL_TOTAL_TIME'), ('halo_time_s', 'HALO_TOTAL_TIME')]:
            a, ref, measured = float(references['A'][key]), float(references['C'][key]), c[key]
            comparisons.append(dict(nodes=16, metric=metric, role='D_new', A=a, C=ref, D=measured,
                               C_over_D=ref/measured if measured > 0 else 'NA', A_over_D=a/measured if measured > 0 else 'NA',
                               repetitions=1, mpi=shape, iteration=rep, reference_kind='HISTORICAL'))
        geometry.extend(ranges)
        write_csv(logs / 'summary.csv', summary)
        write_csv(logs / 'comparisons.csv', comparisons)
        write_csv(logs / 'rank0_grids.csv', geometry)

try:
    for shape in shapes:
        execute(shape, 1, smoke=True)
    for rep in range(1, args.repetitions + 1):
        for shape in shapes:
            execute(shape, rep)
except (SystemExit, OSError, ValueError) as e:
    plan.update(RUN_STATUS='FAILED', ERROR=str(e).replace('\n', ' '))
    write_config(output / 'config.txt', plan)
    raise
plan['RUN_STATUS'] = 'OK'
write_config(output / 'config.txt', plan)
print('Completed: ' + str(logs / 'summary.csv'))
PY
