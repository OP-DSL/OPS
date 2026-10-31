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
# Each new MPI shape trains its own catalog before replay; 8x4x4 is historical.
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
import io
import math
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
if not (0 <= rank < 128 and 0 <= local < 8): raise SystemExit('Invalid MPI rank/GCD')
phase = int(c['OPS_AUTOTUNE_MODE'])
mode = 1 if phase == 1 and rank == 0 else (0 if phase == 1 else 2)
logging = rank == 0
env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(local), MPICH_GPU_SUPPORT_ENABLED='1',
           OMP_NUM_THREADS='1', OPS_AUTOTUNE_MODE=str(mode), OPS_AUTOTUNE_LOGS='1' if logging else '0',
           OPS_DISABLE_CSV_LOGS='0' if logging else '1', OPS_CSV_LOG_LEVEL='0' if logging else '2')
for key in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','OPS_BLOCKSIZE_CSV'):
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
         precomputed_loaded=str(loaded) if phase == 2 else 'NA')
buf=io.StringIO(); csv.writer(buf, lineterminator='').writerow(s.values())
with log.open('a') as f: f.write('\nOPS_RANK_RESULT\t'+buf.getvalue()+'\n')
sys.exit(0 if good else 1)
'''
RANK_FIELDS = ['rank_id','local_rank','gpu_id','hostname','rank_wall_time_s',
               'app_reported_wall_time_s','rank_status','returncode','autotune_mode','precomputed_loaded']


def collect_ranks(run, phase):
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
    require(len(values) == 128 and {int(s['rank_id']) for s in values} == set(range(128)), 'Missing MPI rank status')
    for s in values:
        expected = 1 if phase == 1 and int(s['rank_id']) == 0 else (0 if phase == 1 else 2)
        require(s['rank_status'] == 'OK' and s['returncode'] == '0' and int(s['autotune_mode']) == expected
                and (phase == 1 or s['precomputed_loaded'] == 'True'), 'Rank failed or ran the wrong mode')
    return values


parser = argparse.ArgumentParser(description='TGV StoreNone 2016^3: autotune and replay a separate catalog for each new MPI decomposition, 16 nodes.')
parser.add_argument('--account', default=os.environ.get('SLURM_JOB_ACCOUNT') or os.environ.get('SLURM_ACCOUNT') or 'project_465003260')
parser.add_argument('--campaign', type=Path, help='Historical TGV_TOP3_20261022_083420 reference (automatic project scratch lookup)')
parser.add_argument('--executable', type=Path, help='Existing executable; hash must match historical reference')
parser.add_argument('--output', type=Path, help='Override automatic project scratch TGV_YZ_N016_<timestamp> directory')
parser.add_argument('--ops-root', type=Path, default=Path(os.environ['OPS_YZ_DEFAULT_ROOT']))
parser.add_argument('--env-file', type=Path)
parser.add_argument('--decompositions', default='2x8x8,4x4x8,2x16x4', help='New shapes; default all three. Never reruns 8x4x4.')
parser.add_argument('--repetitions', type=int, default=1, help='Replay measurements per shape; one fresh tuning run per shape')
parser.add_argument('--precomputed-source', type=Path, help='Optional OPS_LOGS directory with matching per-shape tuning logs/catalogs; incompatible catalogs are rejected')
action = parser.add_mutually_exclusive_group()
action.add_argument('--dry-run', action='store_true')
action.add_argument('--submit', action='store_true')
args = parser.parse_args()


def main():
    require(args.repetitions > 0, 'Repetitions must be positive')
    shapes = {}
    for shape in args.decompositions.split(','):
        require(re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*', shape), 'Invalid MPI shape: ' + shape)
        dims = [int(v) for v in shape.split('x')]
        require(dims[0]*dims[1]*dims[2] == 128 and all(2016 % v == 0 for v in dims), 'Shape must divide 2016^3 into 128 equal parts')
        require(shape != '8x4x4', '8x4x4 is historical and will not be rerun')
        require(shape not in shapes, 'Duplicate decomposition')
        shapes[shape] = dims
    require(args.account and '/' not in args.account and not any(c.isspace() for c in args.account), 'Invalid Slurm account')
    campaigns = Path('/scratch') / args.account / 'ops-decomposition/campaigns'
    candidates = [campaigns / 'TGV_TOP3_20261022_083420'] + [Path(m) / args.account / 'ops-decomposition/campaigns/TGV_TOP3_20261022_083420'
                  for m in ('/pfs/lustrep1/scratch', '/pfs/lustrep2/scratch')]
    campaign = (args.campaign or next((p for p in candidates if (p / 'build/tgvstorenone_mpi_hip').is_file()), candidates[0])).resolve()
    exe = (args.executable or campaign / 'build/tgvstorenone_mpi_hip').resolve()
    profile = (args.env_file or args.ops_root / 'scripts/lumi_tgv_top3/environment.sh').resolve()
    require(exe.is_file() and os.access(str(exe), os.X_OK), 'Missing/nonexecutable program: ' + str(exe) + '; use --campaign if moved')
    require(profile.is_file(), 'Missing environment profile: ' + str(profile))
    exe_hash, profile_hash = digest(exe), digest(profile)
    source_script = Path(os.environ['OPS_YZ_SOURCE_SCRIPT'])
    script_hash = digest(source_script)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or campaigns / ('TGV_YZ_N016_' + stamp)).resolve()
    output_stamp = re.fullmatch(r'TGV_YZ_N016_(\d{8}_\d{6}_\d{6})', output.name)
    if output_stamp: stamp = output_stamp.group(1)
    require(not output.exists(), 'Output already exists: ' + str(output))
    refs, ref_paths = {}, {}
    common = dict(APP_NAME='tgvstorenone', GRID_X='2016', GRID_Y='2016', GRID_Z='2016', NITER='500',
                  NUM_NODES='16', NUM_MPI_RANKS='128', EXECUTABLE_SHA256=exe_hash, GPU_DIRECT='YES', RUN_STATUS='OK', RUN_EXIT_CODE='0')
    for role, variant, mode in [('A','autotune_off','0'), ('C','autotune_precomputed','2')]:
        found = list((campaign / 'OPS_LOGS_20261022_083420_n016').glob(
            'tgvstorenone_hip_2016x2016x2016_cfg1_comparison_*/mode_1/decomp8x4x4/iter_1/' + variant + '/config.txt'))
        require(len(found) == 1, 'Expected one historical ' + variant + '/config.txt')
        c = read_config(found[0])
        expected = dict(common, DECOMP_PX='8', DECOMP_PY='4', DECOMP_PZ='4', OPS_AUTOTUNE_MODE=mode)
        require(all(c.get(k) == v for k,v in expected.items()), 'Historical context/executable mismatch: ' + str(found[0]))
        require(all(math.isfinite(float(c[k])) and float(c[k]) > 0 for k in ('APP_WALL_TIME','KERNEL_TOTAL_TIME','HALO_TOTAL_TIME')), 'Invalid reference timing')
        refs[role], ref_paths[role] = c, found[0]

    def validate_source(config_path, shape, expected_hash=None):
        c = read_config(config_path)
        expected = dict(common, DECOMP_PX=str(shapes[shape][0]), DECOMP_PY=str(shapes[shape][1]), DECOMP_PZ=str(shapes[shape][2]),
                        OPS_AUTOTUNE_MODE='1', SAMPLED_AUTOTUNE='YES', AUTOTUNE_SAMPLE_RANKS='0', COMMIT_STATUS='COMMITTED')
        require(all(c.get(k) == v for k,v in expected.items()), 'Catalog tuning context mismatch: ' + str(config_path))
        metrics = rows(config_path.parent / 'mpi_rank_metrics.csv')
        require(len(metrics) == 128 and {r['rank_id'] for r in metrics} == {str(r) for r in range(128)}
                and all(r['rank_status'] == 'OK' for r in metrics), 'Incomplete tuning rank metrics')
        data = catalog_bytes(rank0_catalog(config_path.parent))
        if expected_hash:
            require(hashlib.sha256(data).hexdigest() == expected_hash, 'Catalog differs from the measured tuning results')
        return c, data

    reused = {}
    if args.precomputed_source:
        source = args.precomputed_source.resolve()
        require(source.is_dir(), 'Missing --precomputed-source directory')
        for shape in shapes:
            cat = source / 'precomputed_blocks' / ('decomp' + shape) / 'blocks.csv'
            if not cat.exists():
                continue  # Missing shape trains afresh; incompatible existing shape fails.
            found = list(source.glob('tgvstorenone_hip_2016x2016x2016_cfg1_comparison_*/mode_1/decomp' + shape + '/iter_1/autotune_on/config.txt'))
            require(len(found) == 1, 'Catalog lacks unique same-shape tuning provenance: ' + str(cat))
            c, data = validate_source(found[0], shape, digest(cat))
            require(cat.read_bytes() == data, 'Noncanonical/tampered catalog: ' + str(cat))
            reused[shape] = (found[0], data)
    plan = dict(APP_NAME='tgvstorenone', APP_ARGS='2016 2016 2016 500', NUM_NODES=16, NUM_MPI_RANKS=128,
                DECOMPOSITIONS=','.join(shapes), AUTOTUNE_MODES='1 2', AUTOTUNE_SAMPLE_RANKS='0',
                AUTOTUNE_REPETITIONS=1, REPLAY_REPETITIONS=args.repetitions, OUTPUT_DIRECTORY=output,
                EXECUTABLE=exe, EXECUTABLE_SHA256=exe_hash, GPU_DIRECT='YES', REFERENCE_RERUN='NO',
                REFERENCE_KIND='HISTORICAL', REUSED_CATALOGS=','.join(reused) or 'NONE',
                MAIN_EXECUTIONS=len(shapes)-len(reused)+len(shapes)*args.repetitions)
    print(config_text(plan), flush=True)
    if args.dry_run: return
    if args.submit:
        require(not os.environ.get('SLURM_JOB_ID'), 'Use --submit from a login node')
        output.parent.mkdir(parents=True, exist_ok=True)
        command = ['sbatch', '--account='+args.account, '--output='+str(output.parent / (output.name+'-%j.out')),
                   str(source_script), '--account', args.account, '--ops-root', str(args.ops_root.resolve()),
                   '--campaign', str(campaign), '--executable', str(exe), '--output', str(output), '--env-file', str(profile),
                   '--decompositions', ','.join(shapes), '--repetitions', str(args.repetitions)]
        if args.precomputed_source: command += ['--precomputed-source', str(args.precomputed_source.resolve())]
        subprocess.run(command, check=True)
        return
    if not os.environ.get('SLURM_JOB_ID'):
        print('Inputs validated. Add --submit to send the job.')
        return
    require(int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES','0'))) == 16, 'Expected exactly 16 nodes')
    profile_result = subprocess.run(['bash','-c','set -e; source "$1" >&2; env -0','bash',str(profile)], stdout=subprocess.PIPE, check=True)
    env = dict(item.decode().split('=',1) for item in profile_result.stdout.split(b'\0') if item)
    env.update(MPICH_GPU_SUPPORT_ENABLED='1', OMP_NUM_THREADS='1')
    link = subprocess.run(['ldd',str(exe)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    require(link.returncode == 0 and 'libmpi_gtl_hsa' in link.stdout and 'not found' not in link.stdout, 'GPU-aware MPI linkage unavailable: '+link.stdout)
    output.mkdir(parents=True, exist_ok=False)
    logs = output / ('OPS_LOGS_'+stamp+'_n016')
    logs.mkdir()
    label = 'tgvstorenone_hip_2016x2016x2016_cfg1_comparison_'+stamp+'_n016'
    summary, comparisons = [], []

    def execute(shape, phase, rep, trained=None):
        require(digest(exe) == exe_hash and digest(profile) == profile_hash, 'Executable/environment changed during experiment')
        variant = 'autotune_on' if phase == 1 else 'autotune_precomputed'
        run = logs / label / 'mode_1' / ('decomp'+shape) / ('iter_'+str(rep)) / variant
        (run / 'ranks').mkdir(parents=True)
        (run / 'rank_0').mkdir()
        cat = logs / 'precomputed_blocks' / ('decomp'+shape) / 'blocks.csv'
        c = dict(APP_NAME='tgvstorenone', APP_ARGS='2016 2016 2016 500', CAMPAIGN_ID=output.name,
                 CASE_ID='n016_decomp'+shape+'_mode'+str(phase), RUN_CLASS='SCIENTIFIC', SCIENTIFIC_RESULT='YES',
                 NITER=500, GPU_ARCH='gfx90a', EXECUTABLE=exe, EXECUTABLE_SHA256=exe_hash, SCRIPT_SHA256=script_hash,
                 ENVIRONMENT_PROFILE=profile, ENVIRONMENT_SHA256=profile_hash, LOADEDMODULES=env.get('LOADEDMODULES',''),
                 SLURM_JOB_ID=os.environ['SLURM_JOB_ID'], DECOMPOSITION_MODE=1, GRID_X=2016, GRID_Y=2016, GRID_Z=2016,
                 GRID_CONFIG='cfg1', TOTAL_CELLS=2016**3, NUM_NODES=16, NUM_MPI_RANKS=128, GPUS_PER_NODE=8, NUM_GPUS_USED=128,
                 RANKS_PER_GPU=1, DECOMP_PX=shapes[shape][0], DECOMP_PY=shapes[shape][1], DECOMP_PZ=shapes[shape][2],
                 OPS_AUTOTUNE_MODE=phase, SAMPLED_AUTOTUNE='YES', AUTOTUNE_SAMPLE_RANKS='0', AUTOTUNE_REPETITIONS=1,
                 GPU_DIRECT='UNVERIFIED', MPICH_GPU_SUPPORT_ENABLED=1, OPS_TILING='NO', RUN_ITERATION=rep,
                 TOP3_ROLES='TUNING' if phase == 1 else 'D_new', CSV_LOG_RANKS='0',
                 PRECOMPUTED_CATALOG=cat.parent, PRECOMPUTED_BLOCKS_FILE=cat, PRECOMPUTED_SET_ID='',
                 PRECOMPUTED_SET_ID_KIND='BLOCKS_SHA256', PRECOMPUTED_SOURCE_CONFIG='', BLOCKS_SOURCE_MPI=shape,
                 PRECOMPUTED_SELECTION='rank 0 best measured block per kernel',
                 REFERENCE_KIND='HISTORICAL', REFERENCE_RERUN='NO', NO_AUTOTUNE_REFERENCE='DEFAULT_MPI_DEFAULT_BLOCKS',
                 NO_AUTOTUNE_MPI='8x4x4', PRECOMPUTED_MPI=shape, NO_AUTOTUNE_CONFIG=ref_paths['A'],
                 DEFAULT_MPI_PRECOMPUTED_CONFIG=ref_paths['C'], PRECOMPUTED_CONFIG=run/'config.txt' if phase == 2 else '',
                 WALL_TIME_NO_AUTOTUNE=refs['A']['APP_WALL_TIME'], WALL_TIME_DEFAULT_MPI_PRECOMPUTED=refs['C']['APP_WALL_TIME'],
                 WALL_TIME_AUTOTUNE='NA', WALL_TIME_PRECOMPUTED='NA', SPEEDUP_PRECOMPUTED='NA',
                 SPEEDUP_DEFAULT_MPI_PRECOMPUTED='NA', APP_WALL_TIME='NA', JOB_WALL_TIME='NA', PROCESS_EXIT_CODE='NA',
                 RUN_EXIT_CODE='NA', RUN_STATUS='RUNNING', VALIDATION_STATUS='PENDING', COMMIT_STATUS='UNCOMMITTED',
                 KERNEL_TOTAL_TIME='NA', HALO_TOTAL_TIME='NA')
        if trained:
            source, data = trained
            c.update(PRECOMPUTED_SOURCE_CONFIG=source, PRECOMPUTED_SET_ID=hashlib.sha256(data).hexdigest(),
                     WALL_TIME_AUTOTUNE=read_config(source)['APP_WALL_TIME'])
        write_config(run/'config.txt', c)
        rc = 1
        try:
            if phase == 2:
                require(trained is not None, 'Missing same-configuration autotuning')
                validate_source(trained[0], shape, c['PRECOMPUTED_SET_ID'])
                require(cat.read_bytes() == trained[1], 'Wrong/mutated per-configuration catalog')
            command = ['srun','--nodes=16','--ntasks=128','--ntasks-per-node=8','--cpus-per-task=7',
                       '--kill-on-bad-exit=1','--cpu-bind=map_cpu:49,57,17,25,1,9,33,41',
                       sys.executable,'-c',RANK_CODE,str(run/'config.txt')]
            print('RUN '+shape+' '+variant+' iteration '+str(rep), flush=True)
            start = time.monotonic()
            with (run/'run.log').open('w') as f:
                rc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT).returncode
            c.update(PROCESS_EXIT_CODE=rc, JOB_WALL_TIME=time.monotonic()-start)
            launch_text = (run/'run.log').read_text(errors='replace')
            rank0 = run/'ranks/rank_0.log'
            if rank0.exists():
                text = rank0.read_text(errors='replace')
                (run/'run.log').write_text(text + ('\nSLURM LAUNCH OUTPUT:\n'+launch_text if launch_text else ''))
            else:
                text = ''
            statuses = collect_ranks(run, phase)
            require(rc == 0, 'srun failed')
            times = [float(s['app_reported_wall_time_s']) for s in statuses if s['app_reported_wall_time_s']]
            require(times and all(math.isfinite(t) and t > 0 for t in times), 'Missing/invalid application wall time')
            require('GPU Direct enabled' in text and 'Tiling enabled' not in text, 'GPU Direct/tiling check failed')
            metrics = {}
            for key, pattern in [('KERNEL_TOTAL_TIME','Total kernel time'),('HALO_TOTAL_TIME','Total (?:tiled )?halo exchange time')]:
                found = re.findall(r'^\s*'+pattern+r':\s*([0-9.Ee+-]+)\s*$', text, re.M)
                require(len(found) == 1 and math.isfinite(float(found[0])) and float(found[0]) >= 0, 'Invalid '+key)
                metrics[key] = float(found[0])
            if phase == 1:
                data = catalog_bytes(rank0_catalog(run))
                cat.parent.mkdir(parents=True)
                cat.write_bytes(data)
                c.update(PRECOMPUTED_SOURCE_CONFIG=run/'config.txt', PRECOMPUTED_SET_ID=hashlib.sha256(data).hexdigest())
                c['AUTOTUNE_RANK_0_BEST_SHA256'] = digest(run/'rank_0/autotune_on/ops_blocksize_best_logmod.csv')
            else:
                ids = set()
                with (run/'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv').open() as f:
                    for row in csv.DictReader(f):
                        if (row.get('kernel_id') or '').isdigit(): ids.add(int(row['kernel_id']))
                        if set(range(2,27)) <= ids: break
                require(set(range(2,27)) <= ids, 'Replay missing expected kernel diagnostics')
            c.update(metrics)
            c.update(APP_WALL_TIME=max(times), GPU_DIRECT='YES', RUN_EXIT_CODE=0, RUN_STATUS='OK',
                     VALIDATION_STATUS='PASS', COMMIT_STATUS='COMMITTED')
            c['WALL_TIME_AUTOTUNE' if phase == 1 else 'WALL_TIME_PRECOMPUTED'] = max(times)
            if phase == 2:
                c['SPEEDUP_PRECOMPUTED'] = float(refs['A']['APP_WALL_TIME'])/max(times)
                c['SPEEDUP_DEFAULT_MPI_PRECOMPUTED'] = float(refs['C']['APP_WALL_TIME'])/max(times)
        except (OSError, ValueError, KeyError) as e:
            c.update(RUN_STATUS='FAILED', RUN_EXIT_CODE=rc or 1, VALIDATION_STATUS='FAIL', ERROR=str(e).replace('\n',' '))
            write_config(run/'config.txt', c)
            raise ValueError('Run failed: '+str(run)+': '+str(e))
        write_config(run/'config.txt', c)
        summary.append(dict(nodes=16, roles=c['TOP3_ROLES'], global_grid='2016x2016x2016', mpi=shape,
                            blocks='dynamic' if phase == 1 else 'precomputed', wall_time_s=c['APP_WALL_TIME'],
                            kernel_time_s=c['KERNEL_TOTAL_TIME'], halo_time_s=c['HALO_TOTAL_TIME'],
                            result_directory=str(run), iteration=rep, reference_kind='HISTORICAL'))
        if phase == 2:
            for metric, key in [('app_wall_time_s','APP_WALL_TIME'),('kernel_time_s','KERNEL_TOTAL_TIME'),('halo_time_s','HALO_TOTAL_TIME')]:
                a, ref, measured = float(refs['A'][key]), float(refs['C'][key]), c[key]
                comparisons.append(dict(nodes=16, metric=metric, role='D_new', A=a, C=ref, D=measured,
                    C_over_D=ref/measured if measured > 0 else 'NA', A_over_D=a/measured if measured > 0 else 'NA',
                    repetitions=1, mpi=shape, iteration=rep, reference_kind='HISTORICAL'))
        write_csv(logs/'summary.csv', summary)
        write_csv(logs/'comparisons.csv', comparisons)
        return run/'config.txt'

    for shape in shapes:
        if shape in reused:
            source, data = reused[shape]
            validate_source(source, shape, hashlib.sha256(data).hexdigest())
            cat = logs/'precomputed_blocks'/('decomp'+shape)/'blocks.csv'
            cat.parent.mkdir(parents=True)
            cat.write_bytes(data)
            trained = (source, data)
        else:
            source = execute(shape, 1, 1)
            trained = (source, (logs/'precomputed_blocks'/('decomp'+shape)/'blocks.csv').read_bytes())
        for rep in range(1, args.repetitions+1):
            execute(shape, 2, rep, trained)
    print('Completed: '+str(logs/'summary.csv'))


try:
    main()
except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
    raise SystemExit('ERROR: '+str(e))
PY
