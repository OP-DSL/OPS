"""Dataset layout of the sweep, written directly by the job loop.

It is the structure of the earlier TGV decomposition campaigns (OPS_LOGS_<stamp>_n<NNN>,
see README.md section 6.1): one grid folder, mode_1/decomp<PxPyPz>/iter_<r>/<variant>/ run
folders with config.txt, run.log, mpi_rank_metrics.csv and ranks/, the frozen catalogs under
precomputed_blocks/, attempts under _attempts/, and campaign_manifest.json + plan.jsonl.

A run reaches its final folder only when it satisfies the commit rule; failed and
interrupted attempts stay in _attempts/. Qualification runs are not part of the dataset."""
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil

import plan as P
from common import require, sha, sha_bytes, atomic, atomic_bytes, read_json_or, now, shape_text

LAYOUT_VERSION = 1
APP = 'senga2'
GRID_CONFIG = 'cfg1'
GPUS_PER_NODE = 8
TIMING_PARSER_VERSION = 'senga2_total_ttime_v2'   # v2: + OPS_HALO_TIMERS (halo phase timers)
SWEEP_POLICY = 'all_ordered_v1'
VARIANT = {'D': 'autotune_off', 'A': 'autotune_on', 'P': 'autotune_precomputed'}
VARIANT_ORDER = ('autotune_off', 'autotune_on', 'autotune_precomputed')
VARIANT_TIME_KEY = {'autotune_off': 'NO_AUTOTUNE', 'autotune_on': 'AUTOTUNE', 'autotune_precomputed': 'PRECOMPUTED'}
BLOCK_SELECTION = {'D': ('default', '32', '4', '1'), 'A': ('dynamic', 'dynamic', 'dynamic', 'dynamic'),
                   'P': ('precomputed', 'per_kernel', 'per_kernel', 'per_kernel')}
COMMIT_RULE = 'RUN_STATUS=OK, RUN_EXIT_CODE=0, VALIDATION_STATUS=PASS, COMMIT_STATUS=COMMITTED'
COMMITTED = dict(RUN_STATUS='OK', RUN_EXIT_CODE='0', VALIDATION_STATUS='PASS', COMMIT_STATUS='COMMITTED')
# Halo phase timers (OPS_HALO_TIMERS, one line per rank; see halo_timer_values). Empty when OPS did not print
# them (HALO_TIMERS_STATUS=MISSING); never 0. Times in seconds, statistics over ranks.
HALO_PHASE_KEYS = (('gather', 'GATHER'), ('sendrecv', 'SENDRECV'), ('scatter', 'SCATTER'), ('sum', 'PHASES_SUM'))
HALO_AXES = ('x', 'y', 'z')
HALO_TIMER_KEYS = (('HALO_TIMERS_STATUS', 'HALO_TIMERS_VERSION', 'HALO_TIMERS_RANKS') +
                   tuple('HALO_%s_TIME_%s' % (k, s) for _, k in HALO_PHASE_KEYS for s in ('MIN', 'MEAN', 'MAX')) +
                   ('HALO_UNACCOUNTED_TIME_MEAN', 'HALO_SLOWEST_RANK_SENDRECV', 'HALO_SLOWEST_RANK_SCATTER',
                    'HALO_EXCHANGE_CALLS') +
                   tuple('HALO_%s_TIME_%s_MEAN' % (k, a.upper()) for _, k in HALO_PHASE_KEYS[:3] for a in HALO_AXES) +
                   tuple('HALO_BYTES_SENT_%s_MEAN' % a.upper() for a in HALO_AXES))
RANK_HALO_FIELDS = (('rank_gather_time_s', 'gather'), ('rank_sendrecv_time_s', 'sendrecv'),
                    ('rank_scatter_time_s', 'scatter'), ('rank_halo_phase_sum_s', 'sum'), ('rank_halo_calls', 'calls'))
# Keys of a run config.txt, in the order of the TGV campaign files (OPS_LOGS_20260928_161749_n006).
# The two PRECOMPUTED_* keys are written only in precomputed runs. Extra keys follow COMMIT_STATUS.
CONFIG_KEYS = (
    'APP_NAME', 'APP_ARGS', 'CAMPAIGN_ID', 'CASE_ID', 'ATTEMPT_ID', 'RUN_CLASS', 'SCIENTIFIC_RESULT', 'NITER',
    'GPU_ARCH', 'EXECUTABLE_SHA256', 'DECOMPOSITION_MODE', 'GRID_X', 'GRID_Y', 'GRID_Z', 'GRID_CONFIG',
    'TOTAL_CELLS', 'NUM_NODES', 'GPUS_PER_NODE', 'NUM_GPUS_USED', 'NUM_MPI_RANKS', 'DECOMPOSITION_POLICY',
    'RANKS_PER_GPU', 'DECOMP_PX', 'DECOMP_PY', 'DECOMP_PZ', 'COORD_SOURCE', 'OPS_AUTOTUNE_MODE', 'AUTOTUNE_VARIANT',
    'SAMPLED_AUTOTUNE', 'AUTOTUNE_SAMPLE_RANKS', 'BLOCK_SELECTION_MODE', 'BLOCK_X', 'BLOCK_Y', 'BLOCK_Z',
    'RUN_ITERATION', 'TIMING_PARSER_VERSION', 'PRECOMPUTED_CATALOG', 'PRECOMPUTED_SET_ID', 'PROCESS_EXIT_CODE',
    'JOB_WALL_TIME', 'APP_WALL_TIME', 'WALL_TIME_DECOMPOSITION_S', 'RANK_WALL_TIME_MIN', 'RANK_WALL_TIME_MAX',
    'RANK_WALL_TIME_MEAN', 'SLOWEST_RANK', 'RANK_TIME_IMBALANCE', 'RUN_EXIT_CODE', 'VALIDATION_STATUS', 'RUN_STATUS',
    'COMMIT_STATUS')
EXTRA_KEYS = (
    'DATASET_LAYOUT_VERSION', 'LAUNCHER_CAMPAIGN_ID', 'RUN_ID', 'STAGE', 'SENTINEL_KIND', 'PLAN_SHA256',
    'PLAN_POSITION', 'DECOMP_POSITION', 'SEED', 'PAIR_INDEX', 'ORDER_IN_PAIR', 'DECOMPOSITION_FORCED', 'PRINTED_GRID',
    'HALO60_RISK', 'GPU_DIRECT', 'GPU_DIRECT_REQUESTED', 'GPU_DIRECT_REPORTED', 'MPICH_GPU_SUPPORT_ENABLED',
    'KERNEL_TOTAL_TIME', 'HALO_TOTAL_TIME', 'USER_HALO_TOTAL_TIME') + HALO_TIMER_KEYS + (
    'AUTOTUNE_CATALOG_STATUS', 'CATALOG_ROWS',
    'CATALOG_SHA256', 'EXECUTED_KERNELS', 'KERNEL_COVERAGE', 'FALLBACK_KERNEL_IDS_RANK0',
    'FALLBACK_KERNEL_IDS_OTHER_RANKS', 'FINGERPRINT_TKE', 'FINGERPRINT_ENSTROPHY', 'FINGERPRINT_TMAX',
    'FINGERPRINT_DIFFERENCE', 'FINGERPRINT_REFERENCE_RUN', 'SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_NODELIST',
    'NUM_HOSTS', 'START_UTC', 'END_UTC', 'CPU_BIND', 'OPS_ARGS_RANK0', 'OPS_ARGS_OTHER_RANKS', 'PACKAGE_COMMIT',
    'OPS_COMMIT', 'SENGA2_COMMIT', 'VERDICT', 'FAILURE_CATEGORY', 'FAILURE_REASON', 'RECORD_FILE')
RANK_FIELDS = ['run_id', 'rank_id', 'local_rank', 'gpu_id', 'hostname', 'coord_x', 'coord_y', 'coord_z', 'coord_source',
               'local_nx', 'local_ny', 'local_nz', 'rank_wall_time_s', 'app_reported_wall_time_s',
               'rank_compute_time_s', 'rank_mpi_time_s', 'rank_halo_time_s', 'rank_wait_time_s', 'rank_status'] + \
             [name for name, _ in RANK_HALO_FIELDS]   # halo phase timers of the rank (empty if OPS did not print them)
COORD_SOURCE = 'not_measured'      # OPS does not print the rank -> coordinate map; nothing is assumed
RE_TTIME = re.compile(r'total_ttime\s*=\s*([0-9.Ee+-]+)')
RE_HALO = re.compile(r'^[ \t]*Total halo exchange time:[ \t]*([0-9.eE+-]+)[ \t]*$', re.M)
SWEEP_A = re.compile(r'^sweep_p(\d{3,})_(\d+x\d+x\d+)_A$')
SWEEP_PAIR = re.compile(r'^sweep_p(\d{3,})_(\d+x\d+x\d+)_pair(\d+)_([DP])$')
BASELINE = re.compile(r'^baseline_unforced_r(\d+)$')
SENTINEL = re.compile(r'^sentinel_[A-Za-z0-9_]+$')
ATTEMPT = re.compile(r'^attempt_(\d+)$')


def date_stamp(config):
    """<stamp> of the dataset name: the campaign's creation stamp (YYYYMMDD_HHMMSS)."""
    if config.get('date_stamp'):
        return config['date_stamp']
    found = re.search(r'(\d{8}_\d{6})$', config['campaign_id'])
    if found:
        return found.group(1)
    digits = re.sub(r'\D', '', config['created_utc'])
    require(len(digits) >= 14, 'Cannot derive the dataset date stamp from campaign.json')
    return digits[:8] + '_' + digits[8:14]


def parse_config(path):
    """KEY=value lines (the readers of the TGV datasets use the same rule)."""
    return dict(line.split('=', 1) for line in Path(path).read_text().splitlines() if '=' in line)


def config_text(pairs):
    out = []
    for key, value in pairs:
        text = '' if value is None else str(value)
        out.append('%s=%s' % (key, ' '.join(text.split()) if '\n' in text or '\r' in text else text))
    return '\n'.join(out) + '\n'


def committed(values):
    return all(values.get(k) == v for k, v in COMMITTED.items())


def fmt(value):
    return '' if value is None else '%.6f' % value


def halo_timer_values(record):
    """config.txt values of the halo phase timers (record['halo_timers'], parsed by runtime.parse_halo_timers).

    HALO_UNACCOUNTED_TIME_MEAN = HALO_TOTAL_TIME - HALO_PHASES_SUM_TIME_MEAN: both are means over ranks (OPS prints
    "Total halo exchange time" as the mean of the per-kernel MPI times); the difference is the halo-related time
    outside gather/sendrecv/scatter (host-device exchanges, dirty-bit bookkeeping)."""
    timers = record.get('halo_timers') or {}
    per = timers.get('per_rank') or {}
    values = dict((key, '') for key in HALO_TIMER_KEYS)
    values.update(HALO_TIMERS_STATUS=timers.get('status') or 'MISSING', HALO_TIMERS_RANKS=len(per),
                  HALO_TIMERS_VERSION=timers.get('version') if timers.get('version') is not None else '')
    if not per:
        return values
    ranks = sorted(per, key=int)

    def column(name):
        return [(per[r][name], r) for r in ranks if name in per[r]]

    for field, key in HALO_PHASE_KEYS:
        col = column(field)
        if col:
            numbers = [v for v, _ in col]
            values.update({'HALO_%s_TIME_MIN' % key: fmt(min(numbers)), 'HALO_%s_TIME_MAX' % key: fmt(max(numbers)),
                           'HALO_%s_TIME_MEAN' % key: fmt(sum(numbers) / len(numbers))})
    for field in ('sendrecv', 'scatter'):
        col = column(field)
        if col:
            values['HALO_SLOWEST_RANK_%s' % field.upper()] = max(col, key=lambda c: (c[0], int(c[1])))[1]
    calls = column('calls')
    if calls:
        values['HALO_EXCHANGE_CALLS'] = '%d' % max(v for v, _ in calls)
    sums = [v for v, _ in column('sum')]
    if sums and record.get('halo_time_s') is not None:
        values['HALO_UNACCOUNTED_TIME_MEAN'] = fmt(record['halo_time_s'] - sum(sums) / len(sums))
    for field, key in HALO_PHASE_KEYS[:3]:
        for axis in HALO_AXES:
            col = [v for v, _ in column('%s_%s' % (field, axis))]
            if col:
                values['HALO_%s_TIME_%s_MEAN' % (key, axis.upper())] = fmt(sum(col) / len(col))
    for axis in HALO_AXES:
        col = [v for v, _ in column('bytes_%s' % axis)]
        if col:
            values['HALO_BYTES_SENT_%s_MEAN' % axis.upper()] = '%.0f' % (sum(col) / len(col))
    return values


def attempt_number(path):
    found = ATTEMPT.match(Path(path).name)
    return int(found.group(1)) if found else 0


class Layout(object):
    """Paths and files of the dataset of one campaign (sweep scope only)."""

    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.stamp = date_stamp(config)
        self.nodes, self.ranks, g = int(config['nodes']), int(config['ranks']), int(config['grid'])
        self.grid = [g, g, g]
        self.suffix = '%s_n%03d' % (self.stamp, self.nodes)
        self.campaign_id = self.suffix
        self.dataset = self.root / ('OPS_LOGS_' + self.suffix)
        self.prefix = '%s_hip_%s_%s' % (APP, shape_text(self.grid), GRID_CONFIG)
        self.grid_folder = self.dataset / ('%s_comparison_%s' % (self.prefix, self.suffix))
        self.attempts_root = self.dataset / '_attempts'

    # ----- names and places
    def catalog_folder(self, decomposition):
        return self.dataset / 'precomputed_blocks' / self.prefix / 'mode_1' / ('decomp' + decomposition)

    def case_id(self, label, iteration, variant):
        return '%s_g%d_%s_%s_iter_%d_%s' % (self.prefix, self.ranks, label, self.suffix, iteration, variant)

    def locate(self, run_id):
        """Where a sweep-scope run lives: its CASE_ID, final folder and attempts folder."""
        found = SWEEP_A.match(run_id)
        pair = SWEEP_PAIR.match(run_id)
        if found or pair:
            decomposition = (found or pair).group(2)
            iteration = int(pair.group(3)) if pair else 1
            mode = pair.group(4) if pair else 'A'
            kind, label = 'sweep', 'decomp' + decomposition
            final = self.grid_folder / 'mode_1' / label / ('iter_%d' % iteration) / VARIANT[mode]
        elif BASELINE.match(run_id):
            decomposition, mode, iteration = None, 'D', int(BASELINE.match(run_id).group(1))
            kind, label = 'baseline', 'unforced'
            final = self.grid_folder / 'baseline_unforced' / ('iter_%d' % iteration) / VARIANT[mode]
        elif SENTINEL.match(run_id):
            decomposition, mode, iteration = None, 'D', 1
            kind, label = 'sentinel', run_id
            final = self.grid_folder / 'sentinels' / run_id / VARIANT[mode]
        else:
            raise ValueError('Run %r has no place in the dataset layout' % run_id)
        variant = VARIANT[mode]
        case = self.case_id(label, iteration, variant)
        return dict(run_id=run_id, kind=kind, decomposition=decomposition, mode=mode, iteration=iteration,
                    variant=variant, case_id=case, final=final,
                    attempts=self.attempts_root / case / ('iter_%d' % iteration) / variant)

    def attempts(self, loc):
        folder = loc['attempts']
        found = [p for p in folder.glob('attempt_*') if p.is_dir()] if folder.is_dir() else []
        return sorted(found, key=attempt_number)

    def new_attempt(self, loc):
        numbers = [attempt_number(p) for p in self.attempts(loc)]
        record = read_json_or(loc['final'] / 'record.json')
        if record:
            numbers.append(attempt_number(record['attempt']))
        return loc['attempts'] / ('attempt_%03d' % (max(numbers + [0]) + 1))

    # ----- state
    def state(self, run_id):
        """('MISSING' | 'INTERRUPTED' | 'VALID' | 'FAILED' | 'PENDING_COMMIT', record, folder).

        VALID means committed (in the final folder). PENDING_COMMIT is a valid attempt whose move
        into the final folder did not happen (the job ended in between); the job loop commits it."""
        loc = self.locate(run_id)
        record = read_json_or(loc['final'] / 'record.json')
        if record is not None:
            return 'VALID', record, loc['final']
        live = [a for a in self.attempts(loc) if not (a / 'SUPERSEDED.json').exists()]
        if not live:
            return 'MISSING', None, None
        attempt = live[-1]
        record = read_json_or(attempt / 'record.json')
        if record is None:
            return 'INTERRUPTED', None, attempt
        values = parse_config(attempt / 'config.txt') if (attempt / 'config.txt').is_file() else {}
        return ('PENDING_COMMIT' if committed(values) else 'FAILED'), record, attempt

    def commit(self, loc, attempt):
        final = loc['final']
        require(not final.exists(), 'Cannot commit %s: %s already exists' % (attempt, final))
        values = parse_config(attempt / 'config.txt')
        require(committed(values), 'Refusing to commit %s: config.txt does not satisfy %s' % (attempt, COMMIT_RULE))
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(str(attempt), str(final))
        return final

    def supersede(self, run_id, reason, job=''):
        """Set every current attempt of a run aside. A committed run leaves the final tree:
        it goes back to _attempts/ with COMMIT_STATUS=SUPERSEDED, so it never satisfies the commit rule."""
        loc = self.locate(run_id)
        moved = []
        record = read_json_or(loc['final'] / 'record.json')
        if record is not None:
            target = loc['attempts'] / record['attempt']
            if target.exists():
                target = self.new_attempt(loc)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(loc['final']), str(target))
            moved.append((str(loc['final']), str(target)))
        for attempt in self.attempts(loc):
            if (attempt / 'SUPERSEDED.json').exists():
                continue
            config = attempt / 'config.txt'
            if config.is_file():
                text = re.sub(r'(?m)^COMMIT_STATUS=COMMITTED$', 'COMMIT_STATUS=SUPERSEDED', config.read_text())
                atomic_bytes(config, text.encode())
            origin = next((a for a, b in moved if b == str(attempt)), '')
            atomic(attempt / 'SUPERSEDED.json', dict(reason=reason, utc=now(), slurm_job_id=job,
                                                     moved_from_final=origin))
        if moved:
            self.refresh()

    def pending(self):
        """Valid attempts whose commit was interrupted."""
        out = []
        for path in sorted(self.attempts_root.glob('*/iter_*/*/attempt_*/record.json')) if self.attempts_root.is_dir() else []:
            record = read_json_or(path)
            if record is None or (path.parent / 'SUPERSEDED.json').exists():
                continue
            state, _, folder = self.state(record['run_id'])
            if state == 'PENDING_COMMIT' and folder == path.parent:
                out.append((self.locate(record['run_id']), path.parent))
        return out

    def record_paths(self):
        """(record.json path, committed) of every sweep record in the dataset."""
        if not self.dataset.is_dir():
            return []
        final = []
        for pattern in ('mode_1/decomp*/iter_*/autotune_*/record.json', 'baseline_unforced/iter_*/autotune_*/record.json',
                        'sentinels/*/autotune_*/record.json'):
            final += sorted(self.grid_folder.glob(pattern))
        attempts = sorted(self.attempts_root.glob('*/iter_*/*/attempt_*/record.json')) if self.attempts_root.is_dir() else []
        return [(p, True) for p in final] + [(p, False) for p in attempts]

    def sweep_runs_exist(self):
        return bool(self.record_paths()) or (self.attempts_root.is_dir() and any(self.attempts_root.iterdir()))

    # ----- campaign level files
    def plan_cases(self, plan):
        cases = []
        for run in plan['runs']:
            if run['stage'] != 'sweep':
                continue
            loc = self.locate(run['run_id'])
            identity = dict(app=APP, nodes=self.nodes, ranks=self.ranks, grid=list(self.grid), grid_config=GRID_CONFIG,
                            decomposition_mode=1, decomposition=list(run['triple']),
                            autotune_mode=P.MODE_NUMBER[run['mode']], repetition=loc['iteration'], app_args='')
            encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
            case = dict(schema_version=1, case_id=loc['case_id'], identity_sha256=sha_bytes(encoded), **identity)
            case.update(target=str(loc['final'].relative_to(self.dataset)), state='PLANNED')
            cases.append(case)
        require(len({c['case_id'] for c in cases}) == len(cases), 'The sweep plan has duplicate case identities')
        return cases

    def ensure(self, plan, plan_sha):
        """Create the dataset skeleton once; later calls verify that it belongs to the same plan."""
        cases = self.plan_cases(plan)
        data = ''.join(json.dumps(c, sort_keys=True, separators=(',', ':')) + '\n' for c in cases).encode()
        jsonl = self.dataset / 'plan.jsonl'
        if jsonl.is_file():
            require(jsonl.read_bytes() == data, '%s differs from the frozen sweep plan; the dataset belongs to another '
                    'plan. Nothing was changed.' % jsonl)
        else:
            atomic_bytes(jsonl, data)
        reps = plan['repetitions']
        manifest = dict(
            apps=[APP], attempts_root=str(self.attempts_root), autotune_modes=[0, 1, 2], autotune_repetitions=1,
            autotune_sample_ranks=[0], backend='hip', campaign_id=self.campaign_id, case_count=len(cases),
            commit_rule=COMMIT_RULE, dataset_root=str(self.dataset), decomposition_policy=SWEEP_POLICY,
            decompositions=[d['triple'] for d in plan['decompositions']], effective_grid_configs=[GRID_CONFIG],
            effective_grid_count=1, effective_grid_shape_count=1,
            effective_grids=[dict(grid=list(self.grid), grid_config=GRID_CONFIG)], gpu_type=self.gpu_arch(),
            gpus_per_node=GPUS_PER_NODE, grid=list(self.grid), grid_policy='cubic_single', mpi_ranks=self.ranks,
            nodes=self.nodes, plan_sha256=sha_bytes(data), qualification_level='', repetitions=reps,
            requested_grid_configs=[GRID_CONFIG], run_class='SCIENTIFIC', sampled_autotune=True, schema_version=1,
            scientific_result=True, dataset_layout_version=LAYOUT_VERSION, launcher_campaign_id=self.config['campaign_id'],
            launcher_plan_sha256=plan_sha, timing_parser_version=TIMING_PARSER_VERSION,
            app_wall_time='total_ttime printed by the root rank of SENGA2 (includes initialisation)',
            control_runs=dict(
                baseline_unforced='%s/baseline_unforced/iter_<r>/autotune_off: %d default-block runs without '
                                  'OPS_FORCE_DECOMP_*; RUN_CLASS=BASELINE_UNFORCED' % (self.grid_folder.name, reps),
                sentinels='%s/sentinels/<label>/autotune_off: forced d0, default blocks; RUN_CLASS=SENTINEL'
                          % self.grid_folder.name,
                note='not in plan.jsonl and outside mode_1/, so mode_*/decomp*/iter_*/autotune_* never matches them'))
        path = self.dataset / 'campaign_manifest.json'
        old = read_json_or(path)
        if old is None:
            atomic(path, manifest)
        else:
            require(old.get('plan_sha256') == manifest['plan_sha256'] and old.get('case_count') == len(cases),
                    '%s belongs to another plan; nothing was changed.' % path)
        grid_config = self.grid_folder / 'config.txt'
        if not grid_config.is_file():
            atomic_bytes(grid_config, config_text([
                ('APP_NAME', APP), ('GRID_CONFIG', GRID_CONFIG), ('GRID', shape_text(self.grid)),
                ('NUM_DECOMPOSITION_GPUS', self.ranks), ('RUN_ITERATIONS', reps), ('AUTOTUNE_MODES', '0 1 2'),
                ('SAMPLED_AUTOTUNE', 1), ('AUTOTUNE_SAMPLE_RANKS', '0'), ('PRECOMPUTED_SOURCE', ''),
                ('TIMING_PARSER_VERSION', TIMING_PARSER_VERSION)]).encode())
        self.attempts_root.mkdir(parents=True, exist_ok=True)
        self.refresh()

    def gpu_arch(self):
        manifest = read_json_or(self.root / 'build' / 'full' / 'build_manifest.json') or {}
        return gpu_arch(manifest)

    # ----- aggregates of the grid folder (rebuilt from the committed runs, so they are never stale)
    def refresh(self):
        if not self.grid_folder.is_dir():
            return
        found = []
        for path in self.grid_folder.glob('mode_1/decomp*/iter_*/autotune_*/config.txt'):
            values = parse_config(path)
            if committed(values):
                found.append((values.get('END_UTC', ''), str(path), path, values))
        found.sort()
        for variant in VARIANT_ORDER:
            lines = ['WALL_TIME_DECOMPOSITION_%s_ITER_%s_S=%s' % (v['DECOMP_PX'] + 'x' + v['DECOMP_PY'] + 'x' + v['DECOMP_PZ'],
                                                                 v['RUN_ITERATION'], v['APP_WALL_TIME'])
                     for _, _, p, v in found if p.parent.name == variant]
            atomic_bytes(self.grid_folder / ('%s_summary.txt' % variant), ''.join(l + '\n' for l in lines).encode())
        for iteration in sorted(self.grid_folder.glob('mode_1/decomp*/iter_*')):
            if not iteration.is_dir():
                continue
            pairs = [('AUTOTUNE_MODES', '0 1 2'), ('RUN_ITERATION', iteration.name.split('_', 1)[1])]
            halo = []   # after the TGV keys, so the TGV key order is kept as a prefix
            for variant in VARIANT_ORDER:
                path = iteration / variant / 'config.txt'
                if path.is_file():
                    values = parse_config(path)
                    if committed(values):
                        pairs += [('WALL_TIME_' + VARIANT_TIME_KEY[variant], values['APP_WALL_TIME']),
                                  ('JOB_WALL_TIME_' + VARIANT_TIME_KEY[variant], values['JOB_WALL_TIME'])]
                        if values.get('HALO_TIMERS_STATUS'):   # runs written before the halo timers have no key
                            halo += [('HALO_TIMERS_STATUS_' + VARIANT_TIME_KEY[variant], values['HALO_TIMERS_STATUS'])]
                            halo += [('HALO_%s_TIME_MEAN_%s' % (key, VARIANT_TIME_KEY[variant]),
                                      values.get('HALO_%s_TIME_MEAN' % key, '')) for _, key in HALO_PHASE_KEYS]
            atomic_bytes(iteration / 'config.txt', config_text(pairs + halo).encode())

    # ----- per-run files
    def run_config(self, loc, record, good, catalog_manifest=None):
        mode, kind = record['mode'], loc['kind']
        triple = record['triple'] if record.get('triple') else record.get('printed_grid')
        triple = triple or [None, None, None]
        elapsed = [r['elapsed_s'] for r in record.get('ranks') or []]
        mean = sum(elapsed) / len(elapsed) if elapsed else None
        slowest = max(record['ranks'], key=lambda r: r['elapsed_s'])['rank'] if elapsed else ''
        selection = BLOCK_SELECTION[mode]
        rc = record['srun_returncode']
        autotune = record.get('autotune') or {}
        env0 = (record.get('environment') or {}).get('rank_0') or {}
        reported = bool(record.get('gpu_direct_reported'))
        mpich = env0.get('MPICH_GPU_SUPPORT_ENABLED', '')
        fingerprint = record.get('fingerprint') or {}
        values = dict(
            APP_NAME=APP, APP_ARGS='', CAMPAIGN_ID=self.campaign_id, CASE_ID=loc['case_id'],
            ATTEMPT_ID=attempt_number(record['attempt']),
            RUN_CLASS={'sweep': 'SCIENTIFIC', 'baseline': 'BASELINE_UNFORCED', 'sentinel': 'SENTINEL'}[kind],
            SCIENTIFIC_RESULT='YES', NITER=record['steps'], GPU_ARCH=record.get('gpu_arch', ''),
            EXECUTABLE_SHA256=record['executable_sha256'], DECOMPOSITION_MODE=1, GRID_X=self.grid[0],
            GRID_Y=self.grid[1], GRID_Z=self.grid[2], GRID_CONFIG=GRID_CONFIG,
            TOTAL_CELLS=self.grid[0] * self.grid[1] * self.grid[2], NUM_NODES=record['nodes'],
            GPUS_PER_NODE=GPUS_PER_NODE, NUM_GPUS_USED=record['nranks'], NUM_MPI_RANKS=record['nranks'],
            DECOMPOSITION_POLICY={'sweep': SWEEP_POLICY, 'baseline': 'unforced', 'sentinel': 'forced_d0'}[kind],
            RANKS_PER_GPU=1, DECOMP_PX=triple[0], DECOMP_PY=triple[1], DECOMP_PZ=triple[2], COORD_SOURCE=COORD_SOURCE,
            OPS_AUTOTUNE_MODE=P.MODE_NUMBER[mode], AUTOTUNE_VARIANT=loc['variant'], SAMPLED_AUTOTUNE=1,
            AUTOTUNE_SAMPLE_RANKS='0', BLOCK_SELECTION_MODE=selection[0], BLOCK_X=selection[1], BLOCK_Y=selection[2],
            BLOCK_Z=selection[3], RUN_ITERATION=loc['iteration'], TIMING_PARSER_VERSION=TIMING_PARSER_VERSION,
            PRECOMPUTED_CATALOG=str(Path(record['catalog_path']).parent) if mode == 'P' else None,
            PRECOMPUTED_SET_ID=(catalog_manifest or {}).get('set_id', '') if mode == 'P' else None,
            PROCESS_EXIT_CODE=rc, JOB_WALL_TIME=fmt(record['launcher_elapsed_s']),
            APP_WALL_TIME=record.get('total_ttime_text') or '', WALL_TIME_DECOMPOSITION_S=record.get('total_ttime_text') or '',
            RANK_WALL_TIME_MIN=fmt(min(elapsed)) if elapsed else '', RANK_WALL_TIME_MAX=fmt(max(elapsed)) if elapsed else '',
            RANK_WALL_TIME_MEAN=fmt(mean), SLOWEST_RANK=slowest,
            RANK_TIME_IMBALANCE=fmt(max(elapsed) / mean) if elapsed and mean else '',
            RUN_EXIT_CODE=0 if good else (rc if rc else 1),
            VALIDATION_STATUS='PASS' if good else 'FAIL', RUN_STATUS='OK' if good else 'FAILED',
            COMMIT_STATUS='COMMITTED' if good else 'UNCOMMITTED',
            DATASET_LAYOUT_VERSION=LAYOUT_VERSION, LAUNCHER_CAMPAIGN_ID=record['campaign_id'], RUN_ID=record['run_id'],
            STAGE=record['stage'], SENTINEL_KIND=record.get('sentinel_kind', ''), PLAN_SHA256=record['plan_hash'],
            PLAN_POSITION=record['plan_position'], DECOMP_POSITION=record.get('decomp_position'), SEED=record['seed'],
            PAIR_INDEX=record.get('pair_index'), ORDER_IN_PAIR=record.get('order_in_pair'),
            DECOMPOSITION_FORCED='YES' if record.get('triple') else 'NO',
            PRINTED_GRID=shape_text(record['printed_grid']) if record.get('printed_grid') else '',
            HALO60_RISK=('YES' if P.halo60_risk(triple, self.grid[0]) else 'NO') if triple[0] else '',
            GPU_DIRECT='YES' if record['gpudirect'] and reported and mpich == '1' else 'NO',
            GPU_DIRECT_REQUESTED='YES' if record['gpudirect'] else 'NO', GPU_DIRECT_REPORTED='YES' if reported else 'NO',
            MPICH_GPU_SUPPORT_ENABLED=mpich, KERNEL_TOTAL_TIME=record.get('kernel_time_text') or '',
            HALO_TOTAL_TIME=record.get('halo_time_text') or '', USER_HALO_TOTAL_TIME=record.get('user_halo_time_text') or '',
            AUTOTUNE_CATALOG_STATUS=autotune.get('status', '') if mode == 'A' else '',
            CATALOG_ROWS=(autotune.get('catalog_rows', '') if mode == 'A' else record.get('catalog_rows') if mode == 'P' else ''),
            CATALOG_SHA256=(autotune.get('catalog_sha256', '') if mode == 'A' else record.get('catalog_sha256') if mode == 'P' else ''),
            EXECUTED_KERNELS=autotune.get('executed_kernels', '') if mode == 'A' else '',
            KERNEL_COVERAGE=('%.6f' % autotune['kernel_coverage']) if mode == 'A' and autotune.get('kernel_coverage') is not None else '',
            FALLBACK_KERNEL_IDS_RANK0=' '.join(map(str, record.get('fallback_kernel_ids_rank0') or [])),
            FALLBACK_KERNEL_IDS_OTHER_RANKS=' '.join(map(str, record.get('fallback_kernel_ids_other_ranks') or [])),
            FINGERPRINT_TKE=repr(fingerprint['tke']) if 'tke' in fingerprint else '',
            FINGERPRINT_ENSTROPHY=repr(fingerprint['enstrophy']) if 'enstrophy' in fingerprint else '',
            FINGERPRINT_TMAX=repr(fingerprint['tmax']) if 'tmax' in fingerprint else '',
            FINGERPRINT_DIFFERENCE=repr(record['fingerprint_difference']) if record.get('fingerprint_difference') is not None else '',
            FINGERPRINT_REFERENCE_RUN=record.get('fingerprint_reference') or '',
            SLURM_JOB_ID=record['slurm_job_id'], SLURM_STEP_ID=record['slurm_step_id'],
            SLURM_JOB_NODELIST=record['node_list'], NUM_HOSTS=len(record.get('hosts') or []),
            START_UTC=record['start_utc'], END_UTC=record['end_utc'],
            CPU_BIND=(record.get('environment') or {}).get('cpu_bind', ''),
            OPS_ARGS_RANK0=' '.join(record['argv']), OPS_ARGS_OTHER_RANKS=' '.join(record['argv_other_ranks']),
            PACKAGE_COMMIT=record['package_commit'], OPS_COMMIT=record['ops_commit'], SENGA2_COMMIT=record['senga_commit'],
            VERDICT=record['verdict'], FAILURE_CATEGORY=record.get('category', ''),
            FAILURE_REASON=record.get('reason', '') or (autotune.get('reason', '') if mode == 'A' else ''),
            RECORD_FILE='record.json')
        values.update(halo_timer_values(record))
        pairs = []
        for key in CONFIG_KEYS + EXTRA_KEYS:
            if key in ('PRECOMPUTED_CATALOG', 'PRECOMPUTED_SET_ID') and mode != 'P':
                continue
            pairs.append((key, '' if values[key] is None else values[key]))
        return pairs

    def rank_rows(self, attempt, loc, record):
        triple = record['triple'] if record.get('triple') else record.get('printed_grid')
        local = ['', '', '']
        if triple and all(self.grid[i] % int(triple[i]) == 0 for i in range(3)):
            local = [self.grid[i] // int(triple[i]) for i in range(3)]
        statuses = {r['rank']: r for r in record.get('ranks') or []}
        halo_timers = (record.get('halo_timers') or {}).get('per_rank') or {}
        rows = []
        for rank in range(record['nranks']):
            log = attempt / 'ranks' / ('rank_%d.log' % rank)
            text = log.read_text(errors='replace') if log.is_file() else ''
            ttime, halo = RE_TTIME.findall(text), RE_HALO.findall(text)
            s = statuses.get(rank)
            row = dict(run_id=loc['case_id'], rank_id=rank, coord_x='', coord_y='', coord_z='', coord_source=COORD_SOURCE,
                       local_nx=local[0], local_ny=local[1], local_nz=local[2],
                       app_reported_wall_time_s=ttime[0] if len(ttime) == 1 else '',
                       rank_compute_time_s='', rank_mpi_time_s='', rank_halo_time_s=halo[-1] if halo else '',
                       rank_wait_time_s='')
            timers = halo_timers.get(str(rank)) or {}
            for name, field in RANK_HALO_FIELDS:
                value = timers.get(field)
                row[name] = '' if value is None else ('%d' % value if field == 'calls' else fmt(value))
            if s is None:
                row.update(local_rank='', gpu_id='', hostname='', rank_wall_time_s='', rank_status='MISSING')
            else:
                row.update(local_rank=s['local_rank'], gpu_id=s['gpu_id'], hostname=s['host'],
                           rank_wall_time_s=fmt(s['elapsed_s']), rank_status='OK' if s['returncode'] == 0 else 'FAILED')
            rows.append(row)
        return rows

    def write_run_files(self, attempt, record, loc, good):
        """config.txt, run.log, mpi_rank_metrics.csv, ranks/metric_<r>.csv; precomputed runs also get
        a copy of their catalog (blocks.csv) and of its manifest (precomputed_manifest.json)."""
        attempt = Path(attempt)
        manifest = None
        if record['mode'] == 'P' and record.get('catalog_path'):
            folder = Path(record['catalog_path']).parent
            manifest = read_json_or(folder / 'manifest.json')
            shutil.copyfile(str(folder / 'blocks.csv'), str(attempt / 'blocks.csv'))
            shutil.copyfile(str(folder / 'manifest.json'), str(attempt / 'precomputed_manifest.json'))
        rows = self.rank_rows(attempt, loc, record)
        for row in rows:
            f = io.StringIO()
            csv.DictWriter(f, fieldnames=RANK_FIELDS, lineterminator='\n').writerow(row)
            atomic_bytes(attempt / 'ranks' / ('metric_%d.csv' % row['rank_id']), f.getvalue().encode())
        f = io.StringIO()
        writer = csv.DictWriter(f, fieldnames=RANK_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        atomic_bytes(attempt / 'mpi_rank_metrics.csv', f.getvalue().encode())
        with (attempt / 'run.log').open('w') as out:
            for log in sorted((attempt / 'ranks').glob('rank_*.log'), key=lambda p: p.name):
                out.write('===== %s =====\n' % log.name)
                with log.open(errors='replace') as source:
                    shutil.copyfileobj(source, out)
        atomic_bytes(attempt / 'config.txt', config_text(self.run_config(loc, record, good, manifest)).encode())


def gpu_arch(build_manifest):
    """GPU architecture the executable was compiled for (from its recorded HIP flags)."""
    flags = ((build_manifest or {}).get('toolchain') or {}).get('hipflags', '')
    found = re.findall(r'--offload-arch=(\S+)', flags)
    return found[-1] if found else ''
