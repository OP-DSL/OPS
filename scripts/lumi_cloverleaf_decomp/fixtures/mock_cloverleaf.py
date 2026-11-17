#!/usr/bin/env python3
"""Test double of cloverleaf_mpi_hip: prints the lines the launcher parses.

It is not CloverLeaf and computes nothing. It reads clover.in from the working directory (as
CloverLeaf does), prints the stdout lines of rank 0 in the format of the real application and
OPS, and writes clover.out with field summaries. Behaviour is steered by the JSON file named in
MOCK_CLOVER_CONFIG: unforced_grid, executed_ids, best_csv, time_scale, fail (rules applied by rank 0)."""
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


args = sys.argv[1:]
rank, nranks = int(os.environ.get('SLURM_PROCID', '0')), int(os.environ.get('SLURM_NTASKS', '1'))
root = rank == 0
cfg = {}
if os.environ.get('MOCK_CLOVER_CONFIG') and Path(os.environ['MOCK_CLOVER_CONFIG']).is_file():
    cfg = json.loads(Path(os.environ['MOCK_CLOVER_CONFIG']).read_text())
if not Path('clover.in').is_file():
    print('Error: mock cannot open clover.in')
    sys.exit(2)
inputs = {}
for line in Path('clover.in').read_text().splitlines():
    parts = line.replace('=', ' ').split()
    if len(parts) == 2:
        inputs[parts[0]] = parts[1]
grid = [int(inputs[k]) for k in ('x_cells', 'y_cells', 'z_cells')]
steps = int(inputs['end_step'])
mode = int(value(args, '-OPS_AUTOTUNE_MODE=', '1'))
forced = [value(args, 'OPS_FORCE_DECOMP_%s=' % a) for a in 'XYZ']
pgrid = [int(v) for v in forced] if all(forced) else cfg.get('unforced_grid') or dims_create(nranks)
text = 'x'.join(map(str, pgrid)) if all(forced) else 'unforced'
executed = cfg.get('executed_ids') or list(range(12, 40))
kinds = set()
for rule in cfg.get('fail', []):
    if rule.get('decomposition', text) == text and rule.get('mode', mode) == mode and root:
        if rule.get('once_file'):
            if Path(rule['once_file']).exists():
                continue
            Path(rule['once_file']).write_text('fired\n')
        kinds.add(rule['kind'])
if root:
    print('\n OPS_diags = %s' % value(args, '-OPS_DIAGS=', '1'))
    print('\n OPS_autotune_mode = %d' % mode)
    if '-gpudirect' in args and 'no_gpudirect' not in kinds:
        print('\n GPU Direct enabled')
    print('\n Using HIP device: 0 AMD Instinct MI250X')
    print(' Clover version 1.000000')
    print('Output file clover.out opened. All output will go there')
    out = open('clover.out', 'w')
    out.write('\nClover version 1.000000\n\n Clover will run from the following input:-\n\n')
    out.write(Path('clover.in').read_text())
if any(g % p or g // p < 2 for g, p in zip(grid, pgrid)):
    print('Error: overpartitioning! Trying to exchange a 2-deep halo for density0, but dataset is only %d wide on this '
          'process.' % min(g // p for g, p in zip(grid, pgrid)))
    sys.exit(1)
if 'overpartition' in kinds:
    print('Error: overpartitioning! Trying to exchange a 2-deep halo for density0, but dataset is only 1 wide on this process.')
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
    print('block "clover grid" decomposed on to a processor grid of %s  ' % ' x '.join(map(str, shown)))
    print('Global x_min = 2, y_min = 2, z_min = 2')
if mode == 2:
    path = os.environ.get('OPS_BLOCKSIZE_CSV', '')
    ids = [int(line.split(',')[0]) for line in Path(path).read_text().splitlines()[1:] if line.split(',')[0].isdigit()]
    print('[OPS] Loaded %d precomputed block sizes from %s' % (len(ids), path))
    for kid in executed:
        if kid not in ids:
            print('[OPS] WARNING: kernel %d not found in CSV, using default (32,4,1)' % kid)
last = steps // 2 if 'few_steps' in kinds else steps
shift = 1e-2 if 'energy_shift' in kinds else 0.0


def summary(step, final=False):
    vol = 1000.0
    mass = 0.2 * 1000.0 + 0.8 * 20.0
    ie = 1.25e3 - 1e-1 * step
    ke = (0.0 if step == 0 else 1.0e-2 * step) * (1 + shift)
    if final and 'nan' in kinds:
        ke = float('nan')
    out.write('\n\n Time %lf\n' % (0.004 * step))
    out.write('              %-10s  %-10s  %-10s  %-10s  %-15s  %-15s  %-s\n' % (
        ' Volume', ' Mass', ' Density', ' Pressure', ' Internal Energy', 'Kinetic Energy', 'Total Energy'))
    out.write(' step:   %3d   %-10.3E  %-10.3E  %-10.3E  %-10.3E  %-15.3E  %-15.3E  %-.3E' % (
        step, vol, mass, mass / vol, 0.4 * ie / vol, ie, ke, ie + ke))
    out.flush()


if root:
    summary(0)
    for i in range(1, last + 1):
        print(' Step %d time %11.7f control sound timestep  %3.2E  1, 1 x  1.250000E-02  y 6.250000E-03'
              % (i, 0.004 * (i - 1) * (1 + shift), 4e-3))
        if i % 10 == 0 and i != last:
            summary(i)
    if last == steps:
        summary(last, final=True)
        out.write('\n\n Calculation complete\n\n Clover is finishing\n')
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
    print('\n\nName                                    Count Time     MPI-time     Bandwidth(GB/s)')
    print('Name                                  ---------------------------------------------------------------------')
    for kid in executed:
        print('mock_kernel_%-28d %-5d %-5d %f (0.000000) 0.010000 (0.000000)  100.00' % (kid, kid, 10 * steps, 0.5 + 0.01 * kid))
    print('Total kernel time: %g' % (0.7 * seconds))
    print('Total halo exchange time: %g' % (0.02 * seconds))
    if nranks > 1:
        print('Total user halo exchange time: %g' % (0.1 * seconds))
        print('Average number of MPI messages per process: %g' % (26.0 * steps))
        print('Average volume of MPI communications per process: %g MB' % (12.5 * steps))
    print('\n\nProfiler Output       :      Time(sec)   Percentage   Bandwidth(GB/s)')
    print('Timestep              :         %6.2f         0.50              0.00' % (0.1 * seconds))
    print('The Rest              :           0.00         0.00              -nan\n')
    print('Compute Time          :         %6.2f       100.00   ' % (0.7 * seconds))
    print('Communication Time    :           0.00         0.00   ')
    print('Total Time            :         %6.2f       100.00   ' % (0.7 * seconds))
    if 'no_wall' not in kinds:
        print('\nTotal Wall time %f' % seconds)
    out.close()
