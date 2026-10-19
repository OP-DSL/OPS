"""OPS rank execution and metrics, adapted from the existing TGV Top 3 driver."""
import csv
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
import hashlib


def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        tmp = Path(f.name)
    tmp.replace(path)

def shape_text(dims):
    return 'x'.join(map(str, dims))

def gpu_preflight(exe):
    result = subprocess.run(['ldd', str(exe)], capture_output=True, text=True)
    require(result.returncode == 0 and 'libmpi_gtl_hsa' in result.stdout and 'not found' not in result.stdout,
            'GPU-aware Cray MPI linkage unavailable. Load the matching ROCm/Cray MPI environment; check ldd for libmpi_gtl_hsa.')

def rank_worker(spec_path, run):
    spec = json.loads(spec_path.read_text())
    case = spec['case']
    rank = int(os.environ['SLURM_PROCID'])
    local = int(os.environ['SLURM_LOCALID'])
    require(0 <= local < 8, 'Expected one of eight local MPI ranks per LUMI GPU node')
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(local), MPICH_GPU_SUPPORT_ENABLED='1',
               OPS_AUTOTUNE_LOGS='1' if rank == 0 else '0', OPS_AUTOTUNE_MODE=str(case['mode']),
               OPS_DISABLE_CSV_LOGS='0' if rank == 0 else '1',
               OPS_CSV_LOG_LEVEL='0' if rank == 0 else '2')
    for key in ('OPS_BLOCKSIZE_CSV', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env['OMP_NUM_THREADS'] = '1'
    if rank == 0:
        logs = run / 'rank_0'
        logs.mkdir(exist_ok=True)
        env['OPS_LOGS_DIR'] = str(logs)
    else:
        # No shared writable logging directory for non-root ranks.
        env['OPS_LOGS_DIR'] = str(run / '.disabled_csv' / str(rank))
    blocks = None
    if case['mode'] == 2:
        catalog = Path(spec['catalog'])
        manifest = json.loads((catalog / 'manifest.json').read_text())
        blocks = catalog / ('blocks.csv' if manifest['schema_version'] == 2 else f'rank_{rank}/blocks.csv')
        env['OPS_BLOCKSIZE_CSV'] = str(blocks)
    args = [spec['executable']] + list(map(str, case['grid'] + [case['niter']]))
    args += [f'OPS_FORCE_DECOMP_{a}={v}' for a, v in zip('XYZ', case['shape'])]
    args += ['-gpudirect', '-OPS_DIAGS=2', f"-OPS_AUTOTUNE_MODE={case['mode']}"]
    if case['mode'] == 0:
        args += ['OPS_BLOCK_SIZE_X=32', 'OPS_BLOCK_SIZE_Y=4', 'OPS_BLOCK_SIZE_Z=1']
    start = time.monotonic()
    log = run / 'ranks' / f'rank_{rank}.log'
    with log.open('w') as f:
        rc = subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=spec['workdir']).returncode
    elapsed = time.monotonic() - start
    text = log.read_text(errors='replace')
    loaded = True
    if blocks:
        with blocks.open() as f:
            expected = sum(1 for _ in csv.DictReader(f))
        loaded = any(int(n) == expected and p.strip() == str(blocks)
                     for n, p in re.findall(r'Loaded (\d+) precomputed block sizes from ([^\r\n]+)', text))
    reported = re.findall(r'Total\s+Wall\s+time\s*:?\s*([0-9.Ee+-]+)', text, re.I)
    good = rc == 0 and loaded and not re.search(r'considered FAILED|validation.*failed|nan detected', text, re.I)
    if rank == 0:
        good = good and 'GPU Direct enabled' in text
    atomic(run / 'ranks' / f'status_{rank}.json', dict(rank_id=rank, local_rank=local, gpu_id=local,
           hostname=os.uname().nodename, rank_wall_time_s=elapsed,
           app_reported_wall_time_s=float(reported[-1]) if reported else None,
           rank_status='OK' if good else 'FAILED', returncode=rc, precomputed_loaded=loaded))
    return 0 if good else 1

def target_path(root, case):
    label = f"tgvstorenone_hip_{shape_text(case['grid'])}_cfg1_comparison_{root.name.removeprefix('OPS_LOGS_')}"
    return root / label / 'mode_1' / ('decomp' + shape_text(case['shape'])) / 'iter_1' / ('autotune_off' if case['mode'] == 0 else 'autotune_precomputed')

def complete(run, case):
    try:
        result = json.loads((run / 'result.json').read_text())
        config = dict(line.split('=', 1) for line in (run / 'config.txt').read_text().splitlines() if '=' in line)
        return (result['case'] == case and result['status'] == 'COMMITTED'
                and (run / 'mpi_rank_metrics.csv').is_file()
                and all(config.get(k) == v for k, v in {'RUN_STATUS': 'OK', 'RUN_EXIT_CODE': '0',
                    'VALIDATION_STATUS': 'PASS', 'COMMIT_STATUS': 'COMMITTED', 'GPU_DIRECT': 'YES',
                    'EXECUTABLE_SHA256': case['executable_sha256']}.items())
                and math.isfinite(result['app_wall_time_s']) and result['app_wall_time_s'] > 0
                and len(result['ranks']) == case['ranks']
                and {r['rank_id'] for r in result['ranks']} == set(range(case['ranks']))
                and all(r['rank_status'] == 'OK' and (run / 'ranks' / f"rank_{r['rank_id']}.log").is_file() for r in result['ranks']))
    except (OSError, ValueError, KeyError, TypeError):
        return False

def ops_totals(log):
    text = log.read_text(errors='replace')
    result = {}
    for key, label in [('kernel_time_s', 'Total kernel time'), ('halo_time_s', 'Total halo exchange time')]:
        found = re.findall(r'^\s*' + label + r':\s*([0-9.Ee+-]+)\s*$', text, re.M)
        require(len(found) == 1, f'Expected exactly one {label} in {log}')
        value = float(found[0])
        require(math.isfinite(value) and value >= 0, f'Invalid {label} in {log}')
        result[key] = value
    return result
