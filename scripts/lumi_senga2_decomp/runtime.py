"""One application run: per-rank launch, validation, tuning catalog and run record."""
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import plan as P
from common import (require, sha, sha_bytes, atomic, atomic_bytes, read_json, read_json_or, now,
                    shape_text, parse_shape, query)
import sources

CPU_BIND = '49,57,17,25,1,9,33,41'       # one core next to each GCD of a LUMI-G node
RE_TTIME = re.compile(r'total_ttime\s*=\s*([0-9.Ee+-]+)')
RE_GRID = re.compile(r'decomposed on to a processor grid of\s+(\d+)\s+x\s+(\d+)\s+x\s+(\d+)')
RE_LOADED = re.compile(r'\[OPS\] Loaded (\d+) precomputed block sizes from ([^\r\n]+)')
RE_NOT_FOUND = re.compile(r'\[OPS\] WARNING: kernel (\d+) not found in CSV')
RE_STEP = re.compile(r'^[ \t]*(\d+)[ \t]+\d\.\d+E[+-]\d+[ \t]+-?\d+[ \t]*$', re.M)
RE_MODE = re.compile(r'OPS_autotune_mode = (\d+)')
RE_KERNEL = re.compile(r'^(\S+)\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+) \(', re.M)
RE_KERNEL_LINE = re.compile(r'^\S+[ \t]+\d+[ \t]+\d+[ \t]+\S+ \(.*$', re.M)
TOTALS = (('kernel_time_s', 'Total kernel time'), ('halo_time_s', 'Total halo exchange time'),
          ('user_halo_time_s', 'Total user halo exchange time'))
# Per-rank halo phase timers printed by OPS (ops_mpi_halo_timers_output, OPS_diags > 1), one line per rank:
# OPS_HALO_TIMERS version=1 rank=N gather=.. sendrecv=.. scatter=.. sum=.. calls=.. gather_x=.. ... bytes_z=..
RE_HALO_TIMERS = re.compile(r'^OPS_HALO_TIMERS version=(\d+) ([^\r\n]*)$', re.M)
HALO_PHASES = ('gather', 'sendrecv', 'scatter')
FAILURE_PATTERNS = (('OOM', re.compile(r'out of memory|OutOfMemory', re.I)),
                    ('OVERPARTITION', re.compile(r'overpartitioning', re.I)),
                    ('OTHER', re.compile(r'(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])', re.I)),
                    ('OTHER', re.compile(r'\bError\b|\bERROR\b')))
CATEGORIES = ('OOM', 'OVERPARTITION', 'TIMEOUT', 'FINGERPRINT_MISMATCH', 'OTHER')
FOOTERS = ('Autotune --', 'total of rows --', 'app name --', 'default total time --', 'best total time --')
BEST_CSV = 'rank_0/autotune_on/ops_blocksize_best_logmod.csv'
TUNING_CSV = 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv'
ENV_KEYS = ('MPICH_GPU_SUPPORT_ENABLED', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
            'GPU_DEVICE_ORDINAL', 'OMP_NUM_THREADS', 'SLURM_CPU_BIND', 'SLURM_CPU_BIND_TYPE', 'SLURM_CPU_BIND_LIST')
# Every field of design item 2.12 (a test asserts this list against a real record).
RECORD_FIELDS = (
    'campaign_id', 'plan_hash', 'seed', 'plan_position', 'decomposition', 'mode', 'repetition', 'pair_index',
    'order_in_pair', 'stage', 'slurm_job_id', 'slurm_step_id', 'node_list', 'start_utc', 'end_utc', 'argv',
    'argv_other_ranks', 'environment', 'executable_sha256', 'catalog_sha256', 'catalog_rows', 'package_commit',
    'ops_commit', 'senga_commit', 'module_list', 'printed_grid', 'total_ttime_s', 'kernel_time_s', 'halo_time_s',
    'user_halo_time_s', 'halo_timers', 'ranks', 'verdict', 'reason')


class CatalogError(ValueError):
    pass


def parse_halo_timers(logs, nranks):
    """Halo phase timers of every rank (the last OPS_HALO_TIMERS line of each rank log).

    status: OK (all ranks), PARTIAL (some) or MISSING (none, e.g. an OPS build without the timers).
    Missing timers never invalidate a run. per_rank maps the rank (as a string, for JSON) to its numeric fields."""
    per_rank, versions = {}, set()
    for rank, text in sorted(logs.items()):
        found = RE_HALO_TIMERS.findall(text or '')
        if not found:
            continue
        version, body = found[-1]
        fields = {}
        for token in body.split():
            key, sep, value = token.partition('=')
            if sep:
                try:
                    fields[key] = float(value)
                except ValueError:
                    pass
        if all(p in fields for p in HALO_PHASES):
            per_rank[str(rank)] = fields
            versions.add(int(version))
    status = 'MISSING' if not per_rank else ('OK' if len(per_rank) == nranks else 'PARTIAL')
    return dict(status=status, version=max(versions) if versions else None, ranks_reported=len(per_rank),
                per_rank=per_rank)


# --------------------------------------------------------------------------- per-rank worker
def rank_environment(run, rank, local, attempt, catalog_path, base):
    """Environment of one rank. D and P runs have identical logging (none);
    only rank 0 of an autotune run writes the tuning CSVs."""
    tuning = run['mode'] == 'A' and rank == 0
    env = dict(base, ROCR_VISIBLE_DEVICES=str(local), MPICH_GPU_SUPPORT_ENABLED='1', OMP_NUM_THREADS='1',
               OPS_AUTOTUNE_MODE=str(P.rank_mode(run['mode'], rank)),
               OPS_AUTOTUNE_LOGS='1' if tuning else '0',
               OPS_DISABLE_CSV_LOGS='0' if tuning else '1',
               OPS_CSV_LOG_LEVEL='0' if tuning else '2',
               OPS_LOGS_DIR=str(Path(attempt) / ('rank_0' if tuning else 'ops_logs_unused')))
    for key in ('OPS_BLOCKSIZE_CSV', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    if run['mode'] == 'P':
        env['OPS_BLOCKSIZE_CSV'] = str(catalog_path)
    return env


def rank_worker(spec_path):
    """Executed by srun once per MPI rank (run.sh rank --spec ...)."""
    spec = read_json(spec_path)
    run, attempt = spec['run'], Path(spec['attempt'])
    rank, local = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_LOCALID'])
    require(0 <= rank < run['ranks'], 'Rank %d outside the planned %d ranks' % (rank, run['ranks']))
    require(0 <= local < 8, 'Expected at most eight ranks per LUMI-G node (one per GCD); SLURM_LOCALID=%d' % local)
    catalog = (spec.get('catalog') or {}).get('path', '')
    env = rank_environment(run, rank, local, attempt, catalog, os.environ)
    argv = P.check_argv(run['argv'] if rank == 0 else run['argv_other_ranks'])
    log = attempt / 'ranks' / ('rank_%d.log' % rank)
    start = time.time()
    with log.open('w') as f:
        try:
            rc = subprocess.run([spec['executable']] + argv, env=env, cwd=spec['workdir'], stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        except OSError as error:
            f.write('LAUNCH ERROR: %s\n' % error)
            rc = 127
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    recorded = {k: env[k] for k in sorted(env) if k.startswith('OPS_') or k in ENV_KEYS}
    atomic(attempt / 'ranks' / ('status_%d.json' % rank), dict(
        rank=rank, local_rank=local, gpu_id=local, host=os.environ.get('SLURMD_NODENAME') or os.uname().nodename,
        returncode=rc, elapsed_s=time.time() - start, autotune_mode=P.rank_mode(run['mode'], rank), argv=argv,
        environment=recorded, cpu_affinity=affinity, slurm_job_id=os.environ.get('SLURM_JOB_ID', ''),
        slurm_step_id=os.environ.get('SLURM_STEP_ID', '')))
    return 0 if rc == 0 else 1


# --------------------------------------------------------------------------- tuning catalog
def read_best_csv(path):
    """Rows of rank 0's ops_blocksize_best_logmod.csv: (rows, footer dict)."""
    path = Path(path)
    if not path.is_file():
        raise CatalogError('Best-block CSV is missing: %s (did rank 0 run in autotune mode with CSV logs on?)' % path)
    lines = path.read_text(errors='replace').splitlines()
    if not lines or not lines[0].strip():
        raise CatalogError('Best-block CSV is empty: %s' % path)
    header = next(csv.reader([lines[0]]))
    header = [h.strip() for h in header]
    if header[:4] != ['kernel_id', 'bx', 'by', 'bz'] or 'best_time' not in header or 'max_threads_per_block' not in header:
        raise CatalogError('Unexpected header in %s: %s' % (path, lines[0][:120]))
    out, footer = [], {}
    for number, line in enumerate(lines[1:], start=2):
        text = line.strip()
        if not text:
            continue
        label = next((f for f in FOOTERS if text.startswith(f)), None)
        if label:
            footer[label.rstrip(' -')] = text[len(label):].strip()
            continue
        values = next(csv.reader([text]))
        if len(values) != len(header):
            raise CatalogError('Malformed row at %s:%d (%d fields, expected %d)' % (path, number, len(values), len(header)))
        row = dict(zip(header, values))
        try:
            kid = int(row['kernel_id'])
            block = tuple(int(row[k]) for k in ('bx', 'by', 'bz'))
            best = float(row['best_time'])
            limit = int(float(row['max_threads_per_block']))
        except ValueError:
            raise CatalogError('Non-numeric value at %s:%d: %s' % (path, number, text[:100]))
        if kid < 0 or min(block) <= 0 or not math.isfinite(best) or best <= 0 or limit <= 0:
            raise CatalogError('Invalid block or time for kernel %d at %s:%d: block=%s best_time=%r max_threads=%d'
                               % (kid, path, number, block, best, limit))
        if block[0] * block[1] * block[2] > min(1024, limit):
            raise CatalogError('Block %s of kernel %d exceeds the thread limit min(1024, %d) at %s:%d'
                               % ('x'.join(map(str, block)), kid, limit, path, number))
        out.append(dict(kernel_id=kid, block=block, best_time=best, line=number))
    return out, footer


def reduce_best_rows(entries, path=''):
    """One row per kernel id: the lowest best_time. A tie between different blocks is ambiguous and fails."""
    best, reduced = {}, 0
    for entry in entries:
        kid = entry['kernel_id']
        if kid not in best:
            best[kid] = entry
            continue
        reduced += 1
        old = best[kid]
        if entry['best_time'] < old['best_time']:
            best[kid] = entry
        elif entry['best_time'] == old['best_time'] and entry['block'] != old['block']:
            raise CatalogError('Duplicate kernel id %d with equal best_time and different blocks (%s lines %d and %d): '
                               'cannot choose one block' % (kid, path, old['line'], entry['line']))
    return [best[k] for k in sorted(best)], reduced


def executed_ids(tuning_csv):
    """Kernel ids rank 0 executed in the autotune run (first column of the tuning log, streamed)."""
    path = Path(tuning_csv)
    if not path.is_file():
        raise CatalogError('Tuning log is missing: %s. Without it the set of executed kernels is unknown.' % path)
    ids = set()
    with path.open(errors='replace') as f:
        for line in f:
            field = line.split(',', 1)[0]
            if field.isdigit():
                ids.add(int(field))
    if not ids:
        raise CatalogError('Tuning log has no kernel rows: %s' % path)
    return sorted(ids)


def catalog_bytes(selected):
    f = io.StringIO()
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['kernel_id', 'bx', 'by', 'bz'])
    for entry in selected:
        writer.writerow([entry['kernel_id']] + list(entry['block']))
    return f.getvalue().encode()


def build_catalog(attempt, folder, meta):
    """Freeze the catalog of one decomposition from the autotune run in `attempt`.

    This is the only function that writes a catalog, and its only input is rank 0's
    best-block CSV of that run. On any defect it raises CatalogError and writes nothing."""
    attempt, folder = Path(attempt), Path(folder)
    if folder.exists():
        raise CatalogError('Catalog folder already exists: %s (a frozen catalog is never rebuilt)' % folder)
    best = attempt / BEST_CSV
    entries, footer = read_best_csv(best)
    if not entries:
        raise CatalogError('Best-block CSV has no kernel rows (only header/footers): %s. The run did not finish the '
                           'candidate search of any kernel.' % best)
    declared = footer.get('total of rows')
    if declared is not None and (not declared.isdigit() or int(declared) != len(entries)):
        raise CatalogError('Best-block CSV footer declares %s rows but %d were read: %s' % (declared, len(entries), best))
    selected, reduced = reduce_best_rows(entries, str(best))
    ids = [e['kernel_id'] for e in selected]
    if len(set(ids)) != len(ids):
        raise CatalogError('Duplicate kernel ids after reduction in %s' % best)
    executed = executed_ids(attempt / TUNING_CSV)
    stray = sorted(set(ids) - set(executed))
    if stray:
        raise CatalogError('Best-block CSV lists kernels absent from the tuning log (%s): %s' % (stray[:10], best))
    data = catalog_bytes(selected)
    folder.mkdir(parents=True)
    blocks = folder / 'blocks.csv'
    atomic_bytes(blocks, data)
    os.chmod(str(blocks), 0o444)
    source_run = meta.get('source_run') or str(attempt)    # where the autotune run is kept (final folder)
    manifest = dict(meta, schema_version=2, rows=len(selected), sha256=sha_bytes(data), files={'blocks.csv': sha_bytes(data)},
                    kernel_ids=ids, executed_kernel_ids=executed,
                    fallback_kernel_ids=sorted(set(executed) - set(ids)),
                    kernel_coverage=len(ids) / float(len(executed)), duplicates_reduced=reduced,
                    footer=footer, footer_present=bool(footer), source_run=source_run,
                    source_best_csv=str(Path(source_run) / BEST_CSV),
                    source_best_csv_sha256=sha(best), source_attempt=str(attempt), frozen_utc=now(),
                    selection='rank 0, one row per kernel id, lowest best_time; product <= min(1024, max_threads)')
    # set_id as in the TGV campaigns: sha256 of the manifest without set_id (json, sorted keys).
    manifest['set_id'] = manifest_set_id(manifest)
    atomic(folder / 'manifest.json', manifest)
    return manifest


def manifest_set_id(manifest):
    unsigned = {k: v for k, v in manifest.items() if k != 'set_id'}
    return sha_bytes(json.dumps(unsigned, sort_keys=True).encode())


def load_catalog(folder, decomposition, executable_sha256):
    """A frozen catalog is reusable only if it is intact and belongs to this decomposition and executable."""
    folder = Path(folder)
    manifest = read_json_or(folder / 'manifest.json')
    require(manifest is not None and (folder / 'blocks.csv').is_file(), 'No frozen catalog in %s' % folder)
    require(manifest['decomposition'] == decomposition,
            'Catalog %s belongs to decomposition %s, not %s: a run never replays the blocks of another decomposition'
            % (folder, manifest['decomposition'], decomposition))
    require(sha(folder / 'blocks.csv') == manifest['sha256'],
            'Frozen catalog changed on disk (sha256 differs from its manifest): %s' % (folder / 'blocks.csv'))
    require(manifest['executable_sha256'] == executable_sha256,
            'Catalog %s was tuned with another executable (%s)' % (folder, manifest['executable_sha256'][:12]))
    require('set_id' not in manifest or manifest['set_id'] == manifest_set_id(manifest),
            'Catalog manifest changed on disk (set_id does not match its content): %s' % (folder / 'manifest.json'))
    return dict(path=str(folder / 'blocks.csv'), folder=str(folder), decomposition=decomposition,
                rows=manifest['rows'], sha256=manifest['sha256'], kernel_ids=manifest['kernel_ids'],
                executed_kernel_ids=manifest['executed_kernel_ids'], set_id=manifest.get('set_id', ''))


def verify_catalog_for(run, catalog, executable_sha256):
    """Identity check right before a replay run is launched."""
    require(run['mode'] == 'P', 'Only precomputed runs take a catalog')
    require(catalog is not None, 'Precomputed run %s has no catalog' % run['run_id'])
    wanted = shape_text(run['triple'])
    require(catalog.get('decomposition') == wanted,
            'Refusing to launch %s (decomposition %s) with the catalog of %s'
            % (run['run_id'], wanted, catalog.get('decomposition')))
    fresh = load_catalog(catalog['folder'], wanted, executable_sha256)
    require(fresh['sha256'] == catalog['sha256'] and fresh['path'] == catalog['path'], 'Catalog identity changed')
    return fresh


# --------------------------------------------------------------------------- validation
def read_fingerprint(path):
    """Last record of output/tgv_stat.dat: etime, tke, enstrophy, tmax.

    SENGA2 writes four values with format (3e20.9), i.e. two lines per record."""
    path = Path(path)
    if not path.is_file():
        return None, 0, 'output/tgv_stat.dat was not written'
    tokens = path.read_text(errors='replace').split()
    if not tokens or len(tokens) % 4:
        return None, 0, 'output/tgv_stat.dat has %d values, not a multiple of four' % len(tokens)
    try:
        values = [float(t.replace('D', 'E').replace('d', 'e')) for t in tokens[-4:]]
    except ValueError:
        return None, len(tokens) // 4, 'output/tgv_stat.dat ends with a non-numeric value: %s' % ' '.join(tokens[-4:])
    if not all(math.isfinite(v) for v in values):
        return None, len(tokens) // 4, 'output/tgv_stat.dat ends with a non-finite value: %s' % ' '.join(tokens[-4:])
    return dict(zip(('etime', 'tke', 'enstrophy', 'tmax'), values)), len(tokens) // 4, ''


def fingerprint_difference(a, b):
    worst = 0.0
    for key in ('tke', 'enstrophy', 'tmax'):
        scale = max(abs(a[key]), abs(b[key]))
        worst = max(worst, abs(a[key] - b[key]) / scale if scale > 0 else 0.0)
    return worst


def kernel_table(text):
    return {int(k): float(t) for _, k, _, t in RE_KERNEL.findall(text)}


def validate_run(attempt, run, expect):
    """All validity rules of one run (design item 2.10) against the files in `attempt`.

    expect: srun_returncode, timed_out, catalog (P runs: path/rows/kernel_ids/executed_kernel_ids),
    reference (fingerprint dict or None), tolerance."""
    attempt = Path(attempt)
    reasons, categories, m = [], [], {}

    def fail(category, message):
        categories.append(category)
        reasons.append(message)

    if expect.get('timed_out'):
        fail('TIMEOUT', 'run exceeded its wall-clock limit and was terminated')
    if expect.get('srun_returncode', 0) != 0:
        fail('OTHER', 'srun exit status %s' % expect.get('srun_returncode'))
    # --- per-rank status records
    statuses = {}
    for path in (attempt / 'ranks').glob('status_*.json'):
        s = read_json_or(path)
        if s is not None:
            statuses[s['rank']] = s
    m['ranks'] = [dict(rank=s['rank'], host=s['host'], gpu_id=s['gpu_id'], local_rank=s['local_rank'],
                       returncode=s['returncode'], elapsed_s=s['elapsed_s'], autotune_mode=s['autotune_mode'])
                  for _, s in sorted(statuses.items())]
    if set(statuses) != set(range(run['ranks'])):
        fail('OTHER', '%d of %d per-rank status records present' % (len(statuses), run['ranks']))
    bad = sorted(r for r, s in statuses.items() if s['returncode'] != 0)
    if bad:
        fail('OTHER', 'nonzero exit status on rank(s) %s' % ','.join(map(str, bad[:10])))
    pairs = [(s['host'], s['gpu_id']) for s in statuses.values()]
    if len(set(pairs)) != len(pairs):
        dup = sorted(p for p in set(pairs) if pairs.count(p) > 1)
        fail('OTHER', 'ranks share a GPU: duplicated (host, GPU id) %s' % (dup[:4],))
    wrong_mode = sorted(r for r, s in statuses.items() if s['autotune_mode'] != P.rank_mode(run['mode'], r))
    if wrong_mode:
        fail('OTHER', 'rank(s) %s ran the wrong autotune mode' % wrong_mode[:10])
    # --- logs
    logs = {}
    for rank in range(run['ranks']):
        path = attempt / 'ranks' / ('rank_%d.log' % rank)
        logs[rank] = path.read_text(errors='replace') if path.is_file() else ''
    srun_log = attempt / 'srun.log'
    everything = list(logs.items()) + [('srun', srun_log.read_text(errors='replace') if srun_log.is_file() else '')]
    for index, (category, pattern) in enumerate(FAILURE_PATTERNS):
        for rank, text in everything:
            if index == 2:   # "nan": ignore the OPS kernel table, whose derived columns are not physics
                text = RE_KERNEL_LINE.sub('', text)
            found = pattern.search(text)
            if found:
                line = text[text.rfind('\n', 0, found.start()) + 1:text.find('\n', found.end()) % (len(text) + 1)]
                fail(category, 'failure text in the log of rank %s: %s' % (rank, line.strip()[:160]))
                break
    root = logs.get(0, '')
    times = RE_TTIME.findall(root)
    try:
        values = [float(t) for t in times]
    except ValueError:
        values = []
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        fail('OTHER', 'expected exactly one positive total_ttime in the root log, found %d' % len(times))
    else:
        m['total_ttime_s'] = values[0]
        m['total_ttime_text'] = times[0]          # the printed digits, kept for config.txt
    for key, label in TOTALS:
        found = re.findall(r'^[ \t]*' + label + r':[ \t]*([0-9.eE+-]+)[ \t]*$', root, re.M)
        m[key] = float(found[-1]) if found else None
        m[key[:-2] + '_text'] = found[-1] if found else None      # kernel_time_text, ... (printed digits)
    if any(m[key] is None for key, _ in TOTALS):
        fail('OTHER', 'OPS timing totals missing from the root log (is -OPS_DIAGS=2 in effect?)')
    m['kernel_times'] = kernel_table(root)
    m['halo_timers'] = parse_halo_timers(logs, run['ranks'])
    enabled = 'GPU Direct enabled' in root
    m['gpu_direct'] = enabled
    if run['gpudirect'] and not enabled:
        fail('OTHER', '"GPU Direct enabled" not printed by the root rank')
    if not run['gpudirect'] and enabled:
        fail('OTHER', '"GPU Direct enabled" printed in a run launched without -gpudirect')
    echoed = RE_MODE.findall(root)
    if not echoed or int(echoed[-1]) != P.rank_mode(run['mode'], 0):
        fail('OTHER', 'root rank did not report OPS_autotune_mode = %d' % P.rank_mode(run['mode'], 0))
    grids = RE_GRID.findall(root)
    m['printed_grid'] = [int(v) for v in grids[-1]] if grids else None
    if len(grids) != 1:
        fail('OTHER', 'expected one "decomposed on to a processor grid" line in the root log, found %d' % len(grids))
    elif run.get('triple') is not None and m['printed_grid'] != [int(v) for v in run['triple']]:
        fail('OTHER', 'OPS used processor grid %s but %s was forced' % (shape_text(m['printed_grid']), shape_text(run['triple'])))
    elif m['printed_grid'][0] * m['printed_grid'][1] * m['printed_grid'][2] != run['ranks']:
        fail('OTHER', 'printed processor grid %s does not use %d ranks' % (shape_text(m['printed_grid']), run['ranks']))
    steps = {int(s) for s in RE_STEP.findall(root)}
    m['steps_seen'] = len(steps & set(range(1, run['steps'] + 1)))
    if m['steps_seen'] != run['steps'] or (steps and max(steps) != run['steps']):
        fail('OTHER', 'step history has %d of %d steps (last step printed: %s)'
             % (m['steps_seen'], run['steps'], max(steps) if steps else 'none'))
    # --- physical fingerprint
    fingerprint, records, problem = read_fingerprint(attempt / 'work/output/tgv_stat.dat')
    m['fingerprint'], m['fingerprint_records'] = fingerprint, records
    if fingerprint is None:
        fail('OTHER', problem)
    elif expect.get('reference'):
        m['fingerprint_difference'] = fingerprint_difference(fingerprint, expect['reference'])
        if m['fingerprint_difference'] > expect['tolerance']:
            fail('FINGERPRINT_MISMATCH', 'final tke/enstrophy/tmax differ from the reference by %.3e (tolerance %.1e)'
                 % (m['fingerprint_difference'], expect['tolerance']))
    # --- block replay
    loaded = {rank: RE_LOADED.findall(text) for rank, text in logs.items()}
    not_found = {rank: sorted({int(k) for k in RE_NOT_FOUND.findall(text)}) for rank, text in logs.items()}
    if run['mode'] == 'P':
        catalog = expect['catalog']
        m['catalog_sha256'], m['catalog_rows'] = catalog['sha256'], catalog['rows']
        wrong = []
        for rank in range(run['ranks']):
            lines = loaded.get(rank) or []
            if not lines or any(int(n) != catalog['rows'] or path.strip() != catalog['path'] for n, path in lines):
                wrong.append(rank)
        if wrong:
            first = (loaded.get(wrong[0]) or [('none', 'no "Loaded" line')])[0]
            fail('OTHER', 'rank(s) %s did not load exactly the %d catalog rows from %s (rank %d: %s from %s)'
                 % (','.join(map(str, wrong[:10])), catalog['rows'], catalog['path'], wrong[0], first[0], first[1].strip()))
        ids = set(catalog['kernel_ids'])
        expected_fallback = sorted(set(catalog['executed_kernel_ids']) - ids)
        m['fallback_kernel_ids_rank0'] = not_found.get(0, [])
        if not_found.get(0, []) != expected_fallback:
            fail('OTHER', 'rank 0 fell back to default blocks for kernels %s; expected exactly the kernels executed in '
                 'the autotune run but absent from the catalog: %s' % (not_found.get(0, [])[:12], expected_fallback[:12]))
        others = set()
        for rank, found in not_found.items():
            if rank:
                others.update(found)
        m['fallback_kernel_ids_other_ranks'] = sorted(others)
        if others & ids:
            fail('OTHER', 'a non-root rank reported catalog kernels as missing: %s' % sorted(others & ids)[:12])
    else:
        m['catalog_sha256'], m['catalog_rows'] = '', 0
        offenders = sorted(rank for rank, lines in loaded.items() if lines)
        if offenders:
            fail('OTHER', 'rank(s) %s loaded precomputed blocks in a %s run' % (offenders[:10], P.MODE_NAME[run['mode']]))
    category = next((c for c in CATEGORIES if c in categories), '') if categories else ''
    return dict(valid=not reasons, category=category, reasons=reasons, metrics=m)


# --------------------------------------------------------------------------- execution
def prepare_workdir(work, senga_inputs, grid, steps):
    """Fresh working directory with every input SENGA2 reads and an empty output/."""
    work = Path(work)
    require(not work.exists(), 'Working directory already exists: %s' % work)
    (work / 'input').mkdir(parents=True)
    (work / 'output').mkdir()
    for rel in sources.RUN_INPUTS:
        source = Path(senga_inputs) / rel
        require(source.is_file(), 'SENGA2 input file missing: %s' % source)
        shutil.copyfile(str(source), str(work / rel))
    template = (Path(senga_inputs) / 'input/cont.dat').read_text()
    (work / 'input/cont.dat').write_text(sources.cont_dat(template, grid, steps))


def srun_command(run, launcher_dir, spec_path, cpu_bind):
    per_node = min(8, run['ranks'])
    cores = cpu_bind.split(',')[:per_node]
    # A step only owns the cores it requests, and Slurm hands them out from the lowest
    # core upwards. With fewer than 8 tasks per node and 7 cores each, the step would not
    # own the cores of the map (49 for GCD 0), and srun refuses the binding ("CPU binding
    # outside of job step allocation"). Each task therefore requests a share of all 56
    # usable cores, so the step owns the whole node and the map stays GCD-local.
    cpus_per_task = 56 // per_node
    return ['srun', '--nodes=%d' % run['nodes'], '--ntasks=%d' % run['ranks'], '--ntasks-per-node=%d' % per_node,
            '--cpus-per-task=%d' % cpus_per_task, '--kill-on-bad-exit=1', '--cpu-bind=map_cpu:' + ','.join(cores),
            'bash', str(Path(launcher_dir) / 'run.sh'), 'rank', '--spec', str(spec_path)]


def launch(command, log_path, timeout, env):
    """Run srun with a wall-clock limit. Returns (returncode, timed_out)."""
    with Path(log_path).open('w') as log:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
        except OSError as error:
            log.write('Cannot start %s: %s\n' % (command[0], error))
            return 127, False
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            log.write('\nLAUNCHER: wall-clock limit of %d s exceeded; terminating the step\n' % timeout)
            for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
                try:
                    os.killpg(process.pid, sig)
                except OSError:
                    pass
                try:
                    return process.wait(timeout=grace), True
                except subprocess.TimeoutExpired:
                    continue
            return -9, True


def remaining_seconds(env=None):
    """Seconds left in the Slurm allocation, or None when it cannot be determined."""
    env = os.environ if env is None else env
    end = env.get('SLURM_JOB_END_TIME', '')
    if end.isdigit():
        return int(end) - time.time()
    job = env.get('SLURM_JOB_ID', '')
    if job:
        r = query(['squeue', '-h', '-j', job, '-o', '%L'], timeout=60)
        text = r.get('stdout', '').strip()
        if r['status'] == 'OK' and re.fullmatch(r'(\d+-)?\d+(:\d+){0,2}', text):
            from common import slurm_seconds
            return float(slurm_seconds(text))
    start, limit = env.get('SENGA_JOB_START_EPOCH', ''), env.get('SENGA_JOB_TIME_LIMIT_S', '')
    if start.isdigit() and limit.isdigit():
        return int(start) + int(limit) - time.time()
    return None
