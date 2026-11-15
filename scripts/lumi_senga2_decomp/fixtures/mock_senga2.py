#!/usr/bin/env python3
"""Test double of senga2_f2c_mpi_hip: prints the lines the launcher parses.

It is not SENGA2 and computes nothing. Behaviour is steered by the JSON file named in
MOCK_SENGA_CONFIG: unforced_grid, executed_ids, best_csv, fail (rules applied by rank 0)."""
import json
import os
from pathlib import Path
import sys
import time


def value(args, key, default=None):
    for a in args:
        if a.startswith(key):
            return a[len(key):]
    return default


def dims_create(n):
    dims, factors, p = [1, 1, 1], [], 2
    while n > 1:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    for f in sorted(factors, reverse=True):
        dims[dims.index(min(dims))] *= f
    return sorted(dims, reverse=True)


def after(lines, start):
    for i, line in enumerate(lines):
        if line.strip().startswith(start):
            return lines[i + 1].split()
    sys.exit('mock: cont.dat lacks "%s"' % start)


args = sys.argv[1:]
rank, nranks = int(os.environ.get('SLURM_PROCID', '0')), int(os.environ.get('SLURM_NTASKS', '1'))
root = rank == 0
cfg = {}
if os.environ.get('MOCK_SENGA_CONFIG') and Path(os.environ['MOCK_SENGA_CONFIG']).is_file():
    cfg = json.loads(Path(os.environ['MOCK_SENGA_CONFIG']).read_text())
for needed in ('input/cont.dat', 'input/chem.dat', 'input/diff.dat', 'input/radn.dat', 'TGV_species.dat', 't_initial.dat', 'output'):
    if not Path(needed).exists():
        print('Error: mock cannot open %s' % needed)
        sys.exit(2)
cont = Path('input/cont.dat').read_text().splitlines()
grid = int(after(cont, 'Global domain size (nx,ny,nz)')[0])
steps = int(after(cont, 'Time step; start step')[2])
mode = int(value(args, '-OPS_AUTOTUNE_MODE=', '1'))
forced = [value(args, 'OPS_FORCE_DECOMP_%s=' % a) for a in 'XYZ']
pgrid = [int(v) for v in forced] if all(forced) else cfg.get('unforced_grid') or dims_create(nranks)
text = 'x'.join(map(str, pgrid)) if all(forced) else 'unforced'
executed = cfg.get('executed_ids') or list(range(2, 30))
kinds = set()
for rule in cfg.get('fail', []):
    if rule.get('decomposition', text) == text and rule.get('mode', mode) == mode and root:
        if rule.get('once_file'):
            if Path(rule['once_file']).exists():
                continue
            Path(rule['once_file']).write_text('fired\n')
        kinds.add(rule['kind'])
if root:
    print('\n OPS_diags = %s' % value(args, '-OPS_DIAGS=', '0'))
    print('\n OPS_autotune_mode = %d' % mode)
    if '-gpudirect' in args and 'no_gpudirect' not in kinds:
        print('\n GPU Direct enabled')
if any(grid % p or grid // p < 5 for p in pgrid):
    print('Error: overpartitioning! Trying to exchange a 5-deep halo for DRUN, but dataset is only %d wide on this process.'
          % min(grid // p for p in pgrid))
    sys.exit(1)
if 'overpartition' in kinds:
    print('Error: overpartitioning! Trying to exchange a 5-deep halo for DRUN, but dataset is only 4 wide on this process.')
    sys.exit(1)
if 'oom' in kinds:
    print('hipErrorOutOfMemory: out of memory while allocating device buffers')
    sys.exit(1)
if 'exit' in kinds:
    sys.exit(3)
if 'sleep' in kinds:
    time.sleep(600)
shown = [pgrid[0] + 1] + pgrid[1:] if 'wrong_grid' in kinds else pgrid
if root:
    print('block "SENGA_GRID" decomposed on to a processor grid of %s  ' % ' x '.join(map(str, shown)))
    print('Finished block decomposition')
if mode == 2:
    path = os.environ.get('OPS_BLOCKSIZE_CSV', '')
    ids = [int(line.split(',')[0]) for line in Path(path).read_text().splitlines()[1:] if line.split(',')[0].isdigit()]
    print('[OPS] Loaded %d precomputed block sizes from %s' % (len(ids), path))
    for kid in executed:
        if kid not in ids:
            print('[OPS] WARNING: kernel %d not found in CSV, using default (32,4,1)' % kid)
last = steps // 2 if 'few_steps' in kinds else steps
shift = 1e-3 if 'fingerprint' in kinds else 0.0
if root:
    with open('output/tgv_stat.dat', 'a') as f:
        for i in range(last + 1):
            if i:
                print('%7d%12.4E%5d' % (i, 1e-12 * 1.01 ** i, -3))
                print('MPI_MAX error: %12.4E' % (5e-10 + 1e-12 * i))
            else:
                print('%7d%12.4E%5d' % (0, 1e-12, 0))
            tke = float('nan') if 'nan' in kinds and i == last else 0.125 * (1 - 1e-4 * i) * (1 + shift)
            f.write('%20.9e%20.9e%20.9e\n%20.9e\n' % (1e-12 * i, tke, 3.0e6 + grid + i, 300.0 + 0.01 * i))
if mode == 1 and os.environ.get('OPS_DISABLE_CSV_LOGS') != '1':
    folder = Path(os.environ['OPS_LOGS_DIR']) / 'autotune_on'
    level = int(os.environ.get('OPS_CSV_LOG_LEVEL', '0'))
    best = Path(cfg['best_csv']).read_text() if cfg.get('best_csv') else 'kernel_id,bx,by,bz,best_time,default_time,max_threads_per_block\n'
    if 'empty_best' in kinds:
        best = best.splitlines()[0] + '\n'
    if level < 2 and 'no_best' not in kinds:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'ops_blocksize_best_logmod.csv').write_text(best)
    if level < 1:
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'ops_blocksize_tuning_logmod.csv').open('w') as f:
            f.write('kernel_id,bx,by,bz,execution_time,label\n')
            for kid in executed:
                f.write('%d,32,4,1,0.001,0\n%d,16,8,1,0.0009,1\n' % (kid, kid))
            f.write('\nAutotune -- 1\ntotal of rows -- %d\n' % (2 * len(executed)))
seconds = {0: 100.0, 1: 300.0, 2: 90.0}[mode] * (1 + 0.001 * (sum(pgrid) % 7)) * cfg.get('time_scale', 1.0)
if root:
    print('\nName                            Count Time     MPI-time     Bandwidth(GB/s)')
    for kid in executed:
        print('mock_kernel_%-20d %-5d %-5d %f (0.000000) 0.010000 (0.000000)  100.00' % (kid, kid, 10 * steps, 0.5 + 0.01 * kid))
    print('Total kernel time: %g' % (0.7 * seconds))
    print('Total halo exchange time: %g' % (0.02 * seconds))
    print('Total user halo exchange time: %g' % (0.1 * seconds))
    if 'no_ttime' not in kinds:
        print('IPROC: %5d total_ttime   =%16.7f seconds' % (0, seconds))
