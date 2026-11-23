#!/usr/bin/env python3
"""Test double of tgvstorenone_mpi_hip: prints the lines the launcher parses.

It is not the TGV application and computes nothing. Like the real executable it takes
NX NY NZ NITER followed by the OPS options, reads no input file and writes no output file; rank 0
prints the lines of the real application and of OPS (format copied from a LUMI run.log of
tgvstorenone on 8 ranks). Behaviour is steered by the JSON file named in MOCK_TGV_CONFIG:
unforced_grid, executed_ids, best_csv, time_scale, fail (rules applied by rank 0)."""
import json
import os
from pathlib import Path
import sys
import time

# Kernel table of the real application, in the order OPS prints it: (name, calls per time step).
# opensbliblock00Kernel032 is the initialisation (one call in total); Kernel033 ("Save equations")
# is called once per time step, the six boundary loops four times, the Runge-Kutta loops three times.
KERNELS = ([('opensbliblock00Kernel032', 0)] + [('opensbliblock00Kernel%03d' % k, 4) for k in range(26, 32)] +
           [('opensbliblock00Kernel033', 1)] +
           [('opensbliblock00Kernel%03d' % k, 3) for k in (1, 5, 9, 17, 18, 0, 2, 3, 4, 6, 7, 8, 10, 11, 24, 25, 35, 34)])


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
if os.environ.get('MOCK_TGV_CONFIG') and Path(os.environ['MOCK_TGV_CONFIG']).is_file():
    cfg = json.loads(Path(os.environ['MOCK_TGV_CONFIG']).read_text())
if len(args) < 3 or not all(a.isdigit() for a in args[:3]):
    print('Error, requires 3 arguments for the mesh size')
    sys.exit(255)
grid = [int(a) for a in args[:3]]
steps = int(args[3]) if len(args) > 3 and args[3].isdigit() else 500
mode = int(value(args, '-OPS_AUTOTUNE_MODE=', '1'))
forced = [value(args, 'OPS_FORCE_DECOMP_%s=' % a) for a in 'XYZ']
pgrid = [int(v) for v in forced] if all(forced) else cfg.get('unforced_grid') or dims_create(nranks)
text = 'x'.join(map(str, pgrid)) if all(forced) else 'unforced'
executed = cfg.get('executed_ids') or list(range(1, len(KERNELS) + 1))
kinds = set()
for rule in cfg.get('fail', []):
    if rule.get('decomposition', text) == text and rule.get('mode', mode) == mode and root:
        if rule.get('once_file'):
            if Path(rule['once_file']).exists():
                continue
            Path(rule['once_file']).write_text('fired\n')
        kinds.add(rule['kind'])
if root:
    if all(forced):
        for axis, parts in zip('xyz', pgrid):
            print('\n Forced decomposition in %s direction = %d' % (axis, parts))
    print('\n OPS_diags = %s' % value(args, '-OPS_DIAGS=', '1'))
    print('\n OPS_autotune_mode = %d' % mode)
    if '-gpudirect' in args and 'no_gpudirect' not in kinds:
        print('\n GPU Direct enabled')
    print('\n Using HIP device: 0 AMD Instinct MI250X')
else:
    print('\n Using HIP device: 0 AMD Instinct MI250X')
if any(g % p or g // p < 2 for g, p in zip(grid, pgrid)):
    print('Error: overpartitioning! Trying to exchange a 2-deep halo for rho_B0, but dataset is only %d wide on this '
          'process.' % min(g // p for g, p in zip(grid, pgrid)))
    sys.exit(1)
if 'overpartition' in kinds:
    print('Error: overpartitioning! Trying to exchange a 2-deep halo for rho_B0, but dataset is only 1 wide on this process.')
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
    print('block "opensbliblock00" decomposed on to a processor grid of %s  ' % ' x '.join(map(str, shown)))
    print('Finished block decomposition')
    print('OpenSBLI TGV case SN version size %dx%dx%d' % tuple(grid))
if mode == 2:
    path = os.environ.get('OPS_BLOCKSIZE_CSV', '')
    ids = [int(line.split(',')[0]) for line in Path(path).read_text().splitlines()[1:] if line.split(',')[0].isdigit()]
    print('[OPS] Loaded %d precomputed block sizes from %s' % (len(ids), path))
    for kid in executed:
        if kid not in ids:
            print('[OPS] WARNING: kernel %d not found in CSV, using default (32,4,1)' % kid)
done = steps // 2 if 'few_steps' in kinds else steps          # time steps actually executed
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
    print('\n\nName                      Count Time     MPI-time     Bandwidth(GB/s)')
    print('Name                    -------------------------------------------------------')
    for kid, (name, per_step) in enumerate(KERNELS, start=1):
        calls = per_step * done if per_step else 1
        if 'extra_calls' in kinds and name == 'opensbliblock00Kernel034':
            calls += 1
        seconds_text = 'nan' if 'nan' in kinds and name == 'opensbliblock00Kernel024' else '%f' % (0.5 + 0.01 * kid)
        print('%-25s %-5d %-5d %s (0.000000) 0.010000 (0.000000)  %-13.2f' % (name, kid, calls, seconds_text, 100.0))
    print('Total kernel time: %g' % (0.7 * seconds))
    print('Total halo exchange time: %g' % (0.02 * seconds))
    if nranks > 1:
        print('Average number of MPI messages per process: %g' % (37.5 * steps))
        print('Average volume of MPI communications per process: %g MB' % (12.5 * steps))
    print('\nTimings are:')
    print('-----------------------------------------')
    if 'no_wall' not in kinds:
        print('Total Wall time %f' % seconds)
