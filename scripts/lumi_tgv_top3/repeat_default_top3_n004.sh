#!/usr/bin/env bash
#SBATCH --job-name=tgv-repeat-n004
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
# Rank-zero tuning once per MPI shape, then interleaved replay in one allocation.
# Catalog/rank handling follows compare_yz_n016.sh; historical runs are not controls.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPS_REPEAT_SOURCE_SCRIPT="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
export OPS_REPEAT_DEFAULT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$script_dir/../.." && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
exec "${OPS_LAUNCHER_PYTHON:-python3}" - "$@" <<'PY'
import argparse
import csv
from datetime import datetime
import hashlib
import io
import math
import statistics
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


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
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(config_text(fields))
    os.replace(str(tmp), str(path))


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, values):
    if values:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]))
            w.writeheader()
            w.writerows(values)


# Export rank 0's best measured block for each kernel, without cross-rank voting.
def rank0_catalog(run):
    source = run / 'rank_0/autotune_on'
    best = {}
    for row in rows(source / 'ops_blocksize_best_logmod.csv'):
        kid = (row.get('kernel_id') or '').strip()
        if not kid or kid.startswith(('Autotune --', 'total of rows --', 'app name --',
                                      'default total time --', 'best total time --')):
            continue
        k = int(kid)
        if k == 1:  # One-time initialization retains the established fallback.
            continue
        b = tuple(int(row[a]) for a in ('bx', 'by', 'bz'))
        t = float(row['best_time'])
        require(k in range(2, 27) and all(v > 0 for v in b) and b[0]*b[1]*b[2] <= 1024
                and math.isfinite(t) and 0 < t < 1e299, 'Invalid/unmeasured rank-0 block in ' + str(source))
        if k not in best or t < best[k][0]:
            best[k] = (t, b)
    require(set(best) == set(range(2, 27)), 'Missing tuned kernel IDs 2..26 in ' + str(source))
    metadata = {}
    with (source / 'ops_blocksize_tuning_logmod.csv').open() as f:
        for row in csv.DictReader(f):
            if (row.get('kernel_id') or '').isdigit():
                metadata.setdefault(int(row['kernel_id']), row)
            if set(range(2, 27)) <= set(metadata):
                break
    require(set(range(2, 27)) <= set(metadata), 'Missing rank-0 tuning metadata in ' + str(source))
    selected = []
    for k in range(2, 27):
        b = best[k][1]
        require(b[0]*b[1]*b[2] <= float(metadata[k]['max_threads']), 'Block exceeds kernel thread limit')
        selected.append(dict(kernel_id=k, bx=b[0], by=b[1], bz=b[2]))
    return selected


def catalog_bytes(selected):
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=['kernel_id', 'bx', 'by', 'bz'], lineterminator='\n')
    w.writeheader()
    w.writerows(selected)
    return f.getvalue().encode()


# Rank statuses are appended to existing rank logs, not written as extra files.
RANK_CODE = r'''
import csv, io, os, re, subprocess, sys, time
from pathlib import Path
path = Path(sys.argv[1]); run = path.parent
c = dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
rank, local = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_LOCALID'])
if not (0 <= rank < int(c['NUM_MPI_RANKS']) and 0 <= local < 8): raise SystemExit('Invalid MPI rank/GCD')
phase = int(c['OPS_AUTOTUNE_MODE'])
mode = 1 if phase == 1 and rank == 0 else (0 if phase == 1 else 2)
logging = rank == 0
env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(local), MPICH_GPU_SUPPORT_ENABLED='1',
           OMP_NUM_THREADS='1', OPS_AUTOTUNE_MODE=str(mode), OPS_AUTOTUNE_LOGS='1' if logging else '0',
           OPS_DISABLE_CSV_LOGS='0' if logging else '1', OPS_CSV_LOG_LEVEL='0' if logging else '2')
for key in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','OPS_BLOCKSIZE_CSV',
            'OPS_TILING','OPS_TILING_MAXDEPTH','OPS_TILESIZE_X','OPS_TILESIZE_Y','OPS_TILESIZE_Z'):
    env.pop(key, None)
env['OPS_LOGS_DIR'] = str(run / ('rank_' + str(rank))) if logging else str(run / 'ranks')
if phase == 2: env['OPS_BLOCKSIZE_CSV'] = c['PRECOMPUTED_BLOCKS_FILE']
command = [c['EXECUTABLE']] + c['APP_ARGS'].split()
command += ['OPS_FORCE_DECOMP_'+a+'='+c['DECOMP_P'+a] for a in 'XYZ']
command += ['-gpudirect','-OPS_DIAGS=2','-OPS_AUTOTUNE_MODE='+str(mode)]
if mode == 0: command += ['OPS_BLOCK_SIZE_X=32','OPS_BLOCK_SIZE_Y=4','OPS_BLOCK_SIZE_Z=1']
log = run / 'ranks' / ('rank_'+str(rank)+'.log')
start = time.monotonic()
with log.open('w') as f:
    try:
        rc = subprocess.run(command, env=env, cwd=str(run), stdin=subprocess.DEVNULL,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    except OSError as e:
        f.write(str(e)+'\n'); rc = 1
text = log.read_text(errors='replace')
loaded = phase == 1 or any(int(n) == 25 and p.strip() == c['PRECOMPUTED_BLOCKS_FILE']
    for n,p in re.findall(r'Loaded (\d+) precomputed block sizes from ([^\r\n]+)', text))
reported = re.findall(r'Total\s+Wall\s+time\s*:?\s*([0-9.Ee+-]+)', text, re.I)
good = rc == 0 and loaded and not re.search(r'considered FAILED|validation.*failed|nan detected', text, re.I)
if rank == 0: good = good and 'GPU Direct enabled' in text and 'Tiling enabled' not in text
s = dict(rank_id=rank, local_rank=local, gpu_id=local, hostname=os.uname().nodename,
         rank_wall_time_s=time.monotonic()-start, app_reported_wall_time_s=reported[-1] if reported else '',
         rank_status='OK' if good else 'FAILED', returncode=rc, autotune_mode=mode,
         precomputed_loaded=str(loaded) if phase == 2 else 'NA',
         cpu_affinity=','.join(map(str, sorted(os.sched_getaffinity(0)))))
buf=io.StringIO(); csv.writer(buf, lineterminator='').writerow(s.values())
with log.open('a') as f: f.write('\nOPS_RANK_RESULT\t'+buf.getvalue()+'\n')
sys.exit(0 if good else 1)
'''
RANK_FIELDS = ['rank_id','local_rank','gpu_id','hostname','rank_wall_time_s',
               'app_reported_wall_time_s','rank_status','returncode','autotune_mode','precomputed_loaded','cpu_affinity']


def collect_ranks(run, phase, nranks):
    values = []
    for path in (run / 'ranks').glob('rank_*.log'):
        status = [line.split('\t',1)[1] for line in path.read_text(errors='replace').splitlines()
                  if line.startswith('OPS_RANK_RESULT\t')]
        require(len(status) == 1, 'Missing/duplicate rank status in ' + str(path))
        value = next(csv.reader(status))
        require(len(value) == len(RANK_FIELDS), 'Malformed rank status')
        row = dict(zip(RANK_FIELDS, value))
        require(path.name == 'rank_' + row['rank_id'] + '.log', 'Rank status/file mismatch')
        values.append(row)
    values.sort(key=lambda s: int(s['rank_id']))
    write_csv(run / 'mpi_rank_metrics.csv', values)
    require(len(values) == nranks and {int(s['rank_id']) for s in values} == set(range(nranks)), 'Missing MPI rank status')
    for s in values:
        expected = 1 if phase == 1 and int(s['rank_id']) == 0 else (0 if phase == 1 else 2)
        require(s['rank_status'] == 'OK' and s['returncode'] == '0' and int(s['autotune_mode']) == expected
                and (phase == 1 or s['precomputed_loaded'] == 'True'), 'Rank failed or ran the wrong mode')
    return values


SHAPES = ('4x4x2', '2x16x1', '4x8x1', '2x8x2')
ROLES = dict(zip(SHAPES, ('DEFAULT', 'TOP1', 'TOP2', 'TOP3')))
CPU_BIND = 'map_cpu:49,57,17,25,1,9,33,41'
METRICS = (
    ('wall_time_s', 'APP_WALL_TIME', None),
    ('kernel_time_s', 'KERNEL_TOTAL_TIME', r'Total kernel time:\s*([0-9.Ee+-]+)'),
    ('halo_time_s', 'HALO_TOTAL_TIME', r'Total (?:tiled )?halo exchange time:\s*([0-9.Ee+-]+)'),
    ('sent_mb_per_rank', 'SENT_MB_PER_RANK', r'Average volume of MPI communications per process:\s*([0-9.Ee+-]+)\s*MB'),
    ('messages_per_rank', 'MESSAGES_PER_RANK', r'Average number of MPI messages per process:\s*([0-9.Ee+-]+)'),
)


def placement(statuses):
    mapping = [(s['hostname'], s['local_rank'], s['gpu_id'], s['cpu_affinity']) for s in statuses]
    hosts = {s[0] for s in mapping}
    require(len(hosts) == 4, 'Expected four distinct hosts')
    for host in hosts:
        local = [s for s in mapping if s[0] == host]
        require(len(local) == 8 and {s[1] for s in local} == {str(i) for i in range(8)},
                'Expected eight unique local ranks per host')
    require(all(s[1] == s[2] and s[3] for s in mapping), 'Missing/invalid CPU or GPU binding')
    return mapping


def check_replay_blocks(run, data):
    expected = {int(r['kernel_id']): tuple(int(r[a]) for a in ('bx', 'by', 'bz'))
                for r in csv.DictReader(io.StringIO(data.decode()))}
    seen = set()
    with (run / 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv').open() as f:
        for row in csv.DictReader(f):
            kid = (row.get('kernel_id') or '').strip()
            if not kid.isdigit() or int(kid) not in expected:
                continue
            k = int(kid)
            require(tuple(int(row[a]) for a in ('bx', 'by', 'bz')) == expected[k],
                    'Replay diagnostics differ from the frozen blocks for kernel ' + kid)
            seen.add(k)
    require(seen == set(expected), 'Replay missing tuned kernel diagnostics')


def aggregate(logs, summary, comparisons):
    results = []
    for shape in SHAPES:
        measurements = [r for r in summary if r['mpi'] == shape and r['blocks'] == 'precomputed']
        for metric, _, _ in METRICS:
            values = [r[metric] for r in measurements]
            results.append(dict(mpi=shape, role=ROLES[shape], metric=metric, repetitions=len(values),
                                median=statistics.median(values), minimum=min(values), maximum=max(values)))
    write_csv(logs / 'statistics.csv', results)
    paired = []
    for shape in SHAPES[1:]:
        for metric, _, _ in METRICS:
            pairs = [r for r in comparisons if r['mpi'] == shape and r['metric'] == metric]
            ratios = [r['baseline_over_measured'] for r in pairs]
            changes = [r['change_pct'] for r in pairs]
            paired.append(dict(mpi=shape, role=ROLES[shape], metric=metric, repetitions=len(pairs),
                               median_baseline_over_measured=statistics.median(ratios),
                               minimum_baseline_over_measured=min(ratios), maximum_baseline_over_measured=max(ratios),
                               median_change_pct=statistics.median(changes),
                               minimum_change_pct=min(changes), maximum_change_pct=max(changes)))
    write_csv(logs / 'comparison_statistics.csv', paired)


def main():
    parser = argparse.ArgumentParser(description='TGV StoreNone, 4 nodes: tune rank 0 once for each MPI shape, then repeat default/Top3 replay in one job.')
    parser.add_argument('--account', default=os.environ.get('SLURM_JOB_ACCOUNT') or os.environ.get('SLURM_ACCOUNT') or 'project_465003260')
    parser.add_argument('--campaign', type=Path, help='Original TGV_TOP3_20261026_101612 campaign, used to locate/verify the binary only')
    parser.add_argument('--executable', type=Path, help='Existing binary; must match the original default-MPI context/hash')
    parser.add_argument('--ops-root', type=Path, default=Path(os.environ['OPS_REPEAT_DEFAULT_ROOT']))
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--output', type=Path, help='New campaign directory; existing output is rejected')
    parser.add_argument('--repetitions', type=int, default=3, help='Replay rounds; default 3, each default -> Top1 -> Top2 -> Top3')
    parser.add_argument('--time', default='01:30:00', help='Slurm time limit (HH:MM:SS); default 01:30:00')
    parser.add_argument('--expected-executable-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--expected-environment-sha256', help=argparse.SUPPRESS)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--submit', action='store_true')
    action.add_argument('--run', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.repetitions > 0, 'Repetitions must be positive')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.account or ''), 'Invalid Slurm account')
    require(re.fullmatch(r'\d{2,3}:[0-5]\d:[0-5]\d', args.time) and any(int(v) for v in args.time.split(':')),
            'Invalid --time; expected positive HH:MM:SS')
    campaigns = Path('/scratch') / args.account / 'ops-decomposition/campaigns'
    name = 'TGV_TOP3_20261026_101612'
    candidates = [campaigns / name] + [Path(m) / args.account / 'ops-decomposition/campaigns' / name
                  for m in ('/pfs/lustrep1/scratch', '/pfs/lustrep2/scratch')]
    campaign = (args.campaign or next((p for p in candidates if (p / 'build/tgvstorenone_mpi_hip').is_file()), candidates[0])).resolve()
    exe = (args.executable or campaign / 'build/tgvstorenone_mpi_hip').resolve()
    profile = (args.env_file or args.ops_root / 'scripts/lumi_tgv_top3/environment.sh').resolve()
    require(exe.is_file() and os.access(str(exe), os.X_OK), 'Missing/nonexecutable program: ' + str(exe))
    require(profile.is_file(), 'Missing environment profile: ' + str(profile))
    exe_hash, profile_hash = digest(exe), digest(profile)
    if args.expected_executable_sha256:
        require(exe_hash == args.expected_executable_sha256, 'Executable changed after submission')
    if args.expected_environment_sha256:
        require(profile_hash == args.expected_environment_sha256, 'Environment profile changed after submission')
    source_script = Path(os.environ['OPS_REPEAT_SOURCE_SCRIPT'])
    script_hash = digest(source_script)
    refs = list((campaign / 'OPS_LOGS_20261026_101612_n004').glob(
        'tgvstorenone_hip_1280x1280x1280_cfg1_comparison_*/mode_1/decomp4x4x2/iter_1/autotune_precomputed/config.txt'))
    require(len(refs) == 1, 'Expected one original default-MPI config to verify binary/context')
    reference = read_config(refs[0])
    expected = dict(APP_NAME='tgvstorenone', GRID_X='1280', GRID_Y='1280', GRID_Z='1280', NITER='500',
                    NUM_NODES='4', NUM_MPI_RANKS='32', DECOMP_PX='4', DECOMP_PY='4', DECOMP_PZ='2',
                    OPS_AUTOTUNE_MODE='2', GPU_DIRECT='YES', RUN_STATUS='OK', RUN_EXIT_CODE='0', EXECUTABLE_SHA256=exe_hash)
    require(all(reference.get(k) == v for k, v in expected.items()), 'Original context/executable mismatch: ' + str(refs[0]))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or campaigns / ('TGV_REPEAT_N004_' + stamp)).resolve()
    match = re.fullmatch(r'TGV_REPEAT_N004_(\d{8}_\d{6}_\d{6})', output.name)
    if match:
        stamp = match.group(1)
    require(not output.exists(), 'Output already exists: ' + str(output))
    plan = dict(APP_NAME='tgvstorenone', APP_ARGS='1280 1280 1280 500', NUM_NODES=4, NUM_MPI_RANKS=32,
                DECOMPOSITIONS=','.join(SHAPES), TRAINING_ORDER=','.join(SHAPES), REPLAY_ORDER=','.join(SHAPES),
                AUTOTUNE_SAMPLE_RANKS=0, AUTOTUNE_REPETITIONS=1, TRAINING_EXECUTIONS=4,
                REPLAY_REPETITIONS=args.repetitions, REPLAY_EXECUTIONS=4*args.repetitions,
                MAIN_EXECUTIONS=4+4*args.repetitions, REFERENCE_KIND='SAME_JOB_SAME_ROUND', REFERENCE_RERUN='YES',
                DEFAULT_MPI=SHAPES[0], EXECUTABLE=exe, EXECUTABLE_SHA256=exe_hash, GPU_DIRECT='YES',
                CPU_BIND=CPU_BIND, TIME_LIMIT=args.time, OUTPUT_DIRECTORY=output)
    print(config_text(plan), flush=True)
    if args.submit:
        require(not os.environ.get('SLURM_JOB_ID'), 'Use --submit from a login node')
        output.parent.mkdir(parents=True, exist_ok=True)
        command = ['sbatch', '--account='+args.account, '--time='+args.time,
                   '--output='+str(output.parent / (output.name+'-%j.out')), str(source_script), '--run',
                   '--account', args.account, '--ops-root', str(args.ops_root.resolve()), '--campaign', str(campaign),
                   '--executable', str(exe), '--env-file', str(profile), '--output', str(output),
                   '--repetitions', str(args.repetitions), '--time', args.time,
                   '--expected-executable-sha256', exe_hash, '--expected-environment-sha256', profile_hash]
        subprocess.run(command, check=True)
        return
    if not args.run:
        print('Inputs validated. Use --submit to launch one job; no job submitted.')
        return
    require(os.environ.get('SLURM_JOB_ID'), '--run requires a Slurm allocation')
    require(int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0'))) == 4,
            'Expected exactly four allocated nodes')
    profile_result = subprocess.run(['bash', '-c', 'set -e; source "$1" >&2; env -0', 'bash', str(profile)],
                                    stdout=subprocess.PIPE, check=True)
    env = dict(item.decode().split('=', 1) for item in profile_result.stdout.split(b'\0') if item)
    env.update(MPICH_GPU_SUPPORT_ENABLED='1', OMP_NUM_THREADS='1')
    link = subprocess.run(['ldd', str(exe)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    require(link.returncode == 0 and 'libmpi_gtl_hsa' in link.stdout and 'not found' not in link.stdout,
            'GPU-aware MPI linkage unavailable: ' + link.stdout)
    output.mkdir(parents=True, exist_ok=False)
    logs = output / ('OPS_LOGS_'+stamp+'_n004')
    logs.mkdir()
    label = 'tgvstorenone_hip_1280x1280x1280_cfg1_comparison_'+stamp+'_n004'
    summary, comparisons, mapping, catalogs = [], [], [], {}

    def execute(shape, phase, rep, baseline=None):
        variant = 'autotune_on' if phase == 1 else 'autotune_precomputed'
        run = logs / label / 'mode_1' / ('decomp'+shape) / ('iter_'+str(rep)) / variant
        (run / 'ranks').mkdir(parents=True)
        (run / 'rank_0').mkdir()
        cat = logs / 'precomputed_blocks' / ('decomp'+shape) / 'blocks.csv'
        dims = shape.split('x')
        c = dict(APP_NAME='tgvstorenone', APP_ARGS='1280 1280 1280 500', CAMPAIGN_ID=output.name,
                 CASE_ID='n004_decomp'+shape+'_mode'+str(phase)+'_iter'+str(rep),
                 RUN_CLASS='TRAINING' if phase == 1 else 'SCIENTIFIC', SCIENTIFIC_RESULT='NO' if phase == 1 else 'YES',
                 NITER=500, GPU_ARCH='gfx90a', EXECUTABLE=exe, EXECUTABLE_SHA256=exe_hash, SCRIPT_SHA256=script_hash,
                 ENVIRONMENT_PROFILE=profile, ENVIRONMENT_SHA256=profile_hash, LOADEDMODULES=env.get('LOADEDMODULES', ''),
                 SLURM_JOB_ID=os.environ['SLURM_JOB_ID'], SLURM_JOB_NODELIST=os.environ.get('SLURM_JOB_NODELIST', ''),
                 CPU_BIND=CPU_BIND, DECOMPOSITION_MODE=1, GRID_X=1280, GRID_Y=1280, GRID_Z=1280,
                 GRID_CONFIG='cfg1', TOTAL_CELLS=1280**3, NUM_NODES=4, NUM_MPI_RANKS=32, GPUS_PER_NODE=8, NUM_GPUS_USED=32,
                 RANKS_PER_GPU=1, DECOMP_PX=dims[0], DECOMP_PY=dims[1], DECOMP_PZ=dims[2], OPS_AUTOTUNE_MODE=phase,
                 SAMPLED_AUTOTUNE='YES', AUTOTUNE_SAMPLE_RANKS=0, AUTOTUNE_REPETITIONS=1, GPU_DIRECT='UNVERIFIED',
                 MPICH_GPU_SUPPORT_ENABLED=1, OPS_TILING='NO', RUN_ITERATION=rep, TOP3_ROLES=ROLES[shape], CSV_LOG_RANKS=0,
                 PRECOMPUTED_CATALOG=cat.parent, PRECOMPUTED_BLOCKS_FILE=cat, PRECOMPUTED_SET_ID='',
                 PRECOMPUTED_SET_ID_KIND='BLOCKS_SHA256', PRECOMPUTED_SOURCE_CONFIG='', BLOCKS_SOURCE_MPI=shape,
                 PRECOMPUTED_SELECTION='rank 0 best measured block per kernel', INITIALIZATION_FALLBACK_KERNEL=1,
                 REFERENCE_KIND='NONE' if phase == 1 else 'SAME_JOB_SAME_ROUND', REFERENCE_RERUN='YES',
                 DEFAULT_MPI=SHAPES[0], PRECOMPUTED_MPI=shape, DEFAULT_MPI_PRECOMPUTED_CONFIG='',
                 ORIGINAL_BINARY_CONFIG=refs[0], WALL_TIME_DEFAULT_MPI_PRECOMPUTED='NA', SPEEDUP_DEFAULT_MPI_PRECOMPUTED='NA',
                 WALL_TIME_AUTOTUNE='NA', WALL_TIME_PRECOMPUTED='NA', JOB_WALL_TIME='NA', PROCESS_EXIT_CODE='NA',
                 RUN_EXIT_CODE='NA', RUN_STATUS='RUNNING', VALIDATION_STATUS='PENDING', COMMIT_STATUS='UNCOMMITTED',
                 COMMUNICATION_COUNTERS_STABLE='NA')
        c.update({key: 'NA' for _, key, _ in METRICS})
        write_config(run / 'config.txt', c)
        rc = 1
        try:
            require(digest(exe) == exe_hash and digest(profile) == profile_hash, 'Executable/environment changed during experiment')
            if phase == 2:
                require(shape in catalogs, 'Missing same-configuration autotuning')
                source, data = catalogs[shape]
                require(cat.read_bytes() == data, 'Wrong/mutated per-configuration catalog')
                c.update(PRECOMPUTED_SOURCE_CONFIG=source, PRECOMPUTED_SET_ID=hashlib.sha256(data).hexdigest(),
                         WALL_TIME_AUTOTUNE=read_config(source)['APP_WALL_TIME'])
                write_config(run / 'config.txt', c)
            command = ['srun', '--nodes=4', '--ntasks=32', '--ntasks-per-node=8', '--cpus-per-task=7',
                       '--distribution=block:block', '--kill-on-bad-exit=1', '--cpu-bind='+CPU_BIND,
                       sys.executable, '-c', RANK_CODE, str(run / 'config.txt')]
            print('RUN '+ROLES[shape]+' '+shape+' '+variant+' iteration '+str(rep), flush=True)
            start = time.monotonic()
            with (run / 'run.log').open('w') as f:
                rc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT).returncode
            c.update(PROCESS_EXIT_CODE=rc, JOB_WALL_TIME=time.monotonic()-start)
            launch_text = (run / 'run.log').read_text(errors='replace')
            rank0 = run / 'ranks/rank_0.log'
            text = rank0.read_text(errors='replace') if rank0.exists() else ''
            (run / 'run.log').write_text(text + ('\nSLURM LAUNCH OUTPUT:\n'+launch_text if launch_text else ''))
            statuses = collect_ranks(run, phase, 32)
            require(rc == 0, 'srun failed')
            current_mapping = placement(statuses)
            if not mapping:
                mapping.extend(current_mapping)
            require(current_mapping == mapping, 'Rank-to-host/CPU/GPU mapping changed between executions')
            c['RANK_PLACEMENT_SHA256'] = hashlib.sha256(repr(mapping).encode()).hexdigest()
            times = [float(s['app_reported_wall_time_s']) for s in statuses if s['app_reported_wall_time_s']]
            require(times and all(math.isfinite(t) and t > 0 for t in times), 'Missing/invalid application wall time')
            require('GPU Direct enabled' in text and 'Tiling enabled' not in text, 'GPU Direct/tiling check failed')
            for _, key, pattern in METRICS:
                if pattern is None:
                    c[key] = max(times)
                    continue
                found = re.findall(r'^\s*'+pattern+r'\s*$', text, re.M)
                require(len(found) == 1 and math.isfinite(float(found[0])) and float(found[0]) > 0, 'Missing/invalid '+key)
                c[key] = float(found[0])
            if phase == 1:
                data = catalog_bytes(rank0_catalog(run))
                cat.parent.mkdir(parents=True)
                cat.write_bytes(data)
                c.update(PRECOMPUTED_SOURCE_CONFIG=run / 'config.txt', PRECOMPUTED_SET_ID=hashlib.sha256(data).hexdigest(),
                         AUTOTUNE_RANK_0_BEST_SHA256=digest(run / 'rank_0/autotune_on/ops_blocksize_best_logmod.csv'))
            else:
                require(cat.read_bytes() == data, 'Catalog changed during replay')
                check_replay_blocks(run, data)
                earlier = [r for r in summary if r['mpi'] == shape and r['blocks'] == 'precomputed']
                stable = all(r['sent_mb_per_rank'] == c['SENT_MB_PER_RANK'] and r['messages_per_rank'] == c['MESSAGES_PER_RANK']
                             for r in earlier)
                c['COMMUNICATION_COUNTERS_STABLE'] = 'YES' if stable else 'NO'
                if not stable:
                    print('WARNING: reported communication counters changed for '+shape, flush=True)
                if baseline is None:
                    require(shape == SHAPES[0], 'Alternative lacks a same-round baseline')
                    baseline_config, baseline_values = run / 'config.txt', c
                else:
                    baseline_config, baseline_values = baseline
                    require(baseline_values['RUN_ITERATION'] == rep, 'Baseline is from another round')
                c.update(DEFAULT_MPI_PRECOMPUTED_CONFIG=baseline_config,
                         WALL_TIME_DEFAULT_MPI_PRECOMPUTED=baseline_values['APP_WALL_TIME'],
                         SPEEDUP_DEFAULT_MPI_PRECOMPUTED=baseline_values['APP_WALL_TIME']/c['APP_WALL_TIME'])
            c['WALL_TIME_AUTOTUNE' if phase == 1 else 'WALL_TIME_PRECOMPUTED'] = c['APP_WALL_TIME']
            c.update(GPU_DIRECT='YES', RUN_EXIT_CODE=0, RUN_STATUS='OK', VALIDATION_STATUS='PASS', COMMIT_STATUS='COMMITTED')
        except (OSError, ValueError, KeyError) as e:
            c.update(RUN_STATUS='FAILED', RUN_EXIT_CODE=rc or 1, VALIDATION_STATUS='FAIL',
                     COMMIT_STATUS='UNCOMMITTED', ERROR=str(e).replace('\n', ' '))
            write_config(run / 'config.txt', c)
            raise ValueError('Run failed: '+str(run)+': '+str(e))
        write_config(run / 'config.txt', c)
        row = dict(nodes=4, roles=ROLES[shape], global_grid='1280x1280x1280', mpi=shape,
                   blocks='dynamic' if phase == 1 else 'precomputed', iteration=rep, scientific_result=c['SCIENTIFIC_RESULT'],
                   reference_kind=c['REFERENCE_KIND'], result_directory=str(run),
                   baseline_config=c['DEFAULT_MPI_PRECOMPUTED_CONFIG'], catalog_sha256=c['PRECOMPUTED_SET_ID'],
                   rank_placement_sha256=c['RANK_PLACEMENT_SHA256'], communication_counters_stable=c['COMMUNICATION_COUNTERS_STABLE'])
        row.update({metric: c[key] for metric, key, _ in METRICS})
        summary.append(row)
        if phase == 2 and shape != SHAPES[0]:
            for metric, key, _ in METRICS:
                ref, measured = baseline_values[key], c[key]
                comparisons.append(dict(nodes=4, role=ROLES[shape], mpi=shape, baseline_mpi=SHAPES[0], iteration=rep,
                    metric=metric, baseline=ref, measured=measured, baseline_over_measured=ref/measured,
                    change_pct=100*(measured/ref-1), baseline_config=str(baseline_config), config=str(run / 'config.txt'),
                    reference_kind='SAME_JOB_SAME_ROUND'))
        write_csv(logs / 'summary.csv', summary)
        write_csv(logs / 'comparisons.csv', comparisons)
        return run / 'config.txt', c

    # Finish all training before measuring any replay; keep each catalog immutable.
    for shape in SHAPES:
        source, _ = execute(shape, 1, 1)
        catalogs[shape] = (source, (logs / 'precomputed_blocks' / ('decomp'+shape) / 'blocks.csv').read_bytes())
    for rep in range(1, args.repetitions+1):
        baseline = execute(SHAPES[0], 2, rep)
        for shape in SHAPES[1:]:
            execute(shape, 2, rep, baseline)
    aggregate(logs, summary, comparisons)
    print('Completed: '+str(logs / 'summary.csv'))


try:
    main()
except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
    raise SystemExit('ERROR: '+str(e))
PY
