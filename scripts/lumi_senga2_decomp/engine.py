"""Campaign state and the job loops (qualification and sweep) that run inside a Slurm allocation."""
import hashlib
import json
import os
from pathlib import Path
import time

import dataset as D
import plan as P
import runtime as R
from common import (require, sha, atomic, read_json, read_json_or, now, shape_text, shell_join, StopJob,
                    EXIT_OK, EXIT_TIME_GUARD, EXIT_ABORT_FAILURES, EXIT_RUN_FAILURES)

DEFAULT_ESTIMATES = {('small', 'A'): 1500, ('small', 'D'): 600, ('small', 'P'): 600,
                     ('full', 'A'): 3600, ('full', 'D'): 1200, ('full', 'P'): 1200}
RUNTIME_INVALID = ('OOM', 'OVERPARTITION')
MAX_CONSECUTIVE_FAILED = 3


class Campaign(object):
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        require((self.root / 'campaign.json').is_file(),
                '%s is not a campaign directory (campaign.json missing). Create one with: run.sh prepare ...' % self.root)
        self.config = read_json(self.root / 'campaign.json')
        self.id = self.config['campaign_id']
        self.layout = D.Layout(self.root, self.config)    # the sweep is stored in the dataset layout

    @property
    def package(self):
        """Identity of the launcher in $C/launcher: the last upgrade, or the one copied by prepare."""
        upgrades = (read_json_or(self.root / 'launcher_upgrades.json') or {}).get('upgrades') or []
        return upgrades[-1]['package'] if upgrades else self.config['package']

    def build(self, name):
        path = self.root / 'build' / name / 'build_manifest.json'
        require(path.is_file(), 'The %s build is not complete (%s missing). Run: bash %s build --campaign %s --build %s'
                % (name, path, self.root / 'launcher/run.sh', self.root, name))
        manifest = read_json(path)
        exe = self.root / manifest['executable']
        require(exe.is_file() and os.access(str(exe), os.X_OK), 'Executable missing: %s' % exe)
        return manifest

    def verify_executable(self, name):
        manifest = self.build(name)
        require(sha(self.root / manifest['executable']) == manifest['sha256'],
                'The %s executable changed since it was built (sha256 differs from build_manifest.json). '
                'Results of different executables are never mixed: start a new campaign.' % name)
        return manifest

    def plan(self, kind):
        name = 'plan.json' if kind == 'sweep' else 'smoke_plan.json'
        path = self.root / name
        require(path.is_file(), 'No frozen %s in %s. Create it with: run.sh %s --campaign %s'
                % (name, self.root, 'plan' if kind == 'sweep' else 'smoke --dry-run', self.root))
        data = path.read_bytes()
        digest = (self.root / (name + '.sha256')).read_text().split()[0]
        require(hashlib.sha256(data).hexdigest() == digest,
                '%s changed after it was frozen (sha256 mismatch). The plan is never regenerated or edited.' % path)
        return json.loads(data.decode()), digest

    # ----- run records. Qualification: runs/qualification/<run_id>/attempt_NNN/. Sweep: the dataset
    # layout (dataset.py): final folders for committed runs, _attempts/ for everything else.
    def run_dir(self, scope, run_id):
        require(scope != 'sweep', 'Sweep runs are stored in the dataset layout (Campaign.layout)')
        return self.root / 'runs' / scope / run_id

    def attempts(self, scope, run_id):
        if scope == 'sweep':
            loc = self.layout.locate(run_id)
            final = [loc['final']] if (loc['final'] / 'record.json').is_file() else []
            return self.layout.attempts(loc) + final
        folder = self.run_dir(scope, run_id)
        return sorted(p for p in folder.glob('attempt_*') if p.is_dir()) if folder.is_dir() else []

    def state(self, scope, run_id):
        """('MISSING' | 'INTERRUPTED' | 'VALID' | 'FAILED', record, attempt) of the current attempt.
        Sweep runs may also be 'PENDING_COMMIT' (see dataset.Layout.state)."""
        if scope == 'sweep':
            return self.layout.state(run_id)
        live = [a for a in self.attempts(scope, run_id) if not (a / 'SUPERSEDED.json').exists()]
        if not live:
            return 'MISSING', None, None
        attempt = live[-1]
        record = read_json_or(attempt / 'record.json')
        if record is None:
            return 'INTERRUPTED', None, attempt
        good = record['verdict'] == 'VALID' and record.get('autotune', {}).get('status', 'FROZEN') == 'FROZEN'
        return ('VALID' if good else 'FAILED'), record, attempt

    def supersede(self, scope, run_id, reason):
        if scope == 'sweep':
            return self.layout.supersede(run_id, reason, os.environ.get('SLURM_JOB_ID', ''))
        for attempt in self.attempts(scope, run_id):
            if not (attempt / 'SUPERSEDED.json').exists():
                atomic(attempt / 'SUPERSEDED.json', dict(reason=reason, utc=now(),
                                                         slurm_job_id=os.environ.get('SLURM_JOB_ID', '')))

    def all_records(self):
        """Every record on disk, superseded ones included (flagged)."""
        out = []
        base = self.root / 'runs'
        paths = [(p, True) for p in sorted(base.glob('*/*/attempt_*/record.json'))] if base.is_dir() else []
        for path, committed in paths + self.layout.record_paths():
            record = read_json_or(path)
            if record is None:
                continue
            marker = read_json_or(path.parent / 'SUPERSEDED.json')
            record['superseded'] = marker is not None
            record['superseded_reason'] = marker['reason'] if marker else ''
            record['committed'] = committed       # sweep: in the final dataset tree
            record['record_path'] = str(path)
            out.append(record)
        return out

    def catalog_folder(self, scope, build, decomposition):
        if scope == 'sweep':
            require(build == 'full', 'The sweep only uses the full build')
            return self.layout.catalog_folder(decomposition)
        return self.root / 'catalogs' / scope / ('%s_%s' % (build, decomposition))


class Engine(object):
    def __init__(self, campaign, scope, plan, plan_sha, retry_failed=False, no_time_guard=False, estimates=None,
                 env=None):
        self.c, self.scope, self.plan, self.plan_sha = campaign, scope, plan, plan_sha
        self.retry_failed, self.no_time_guard = retry_failed, no_time_guard
        self.env = dict(os.environ if env is None else env)
        cfg = campaign.config
        self.tolerance = cfg['fingerprint_tolerance']
        self.estimates = dict(DEFAULT_ESTIMATES)
        measured = read_json_or(campaign.root / 'measured_times.json') or {}
        for mode, seconds in (measured.get('elapsed_s') or {}).items():
            self.estimates[('full', mode)] = 1.5 * seconds + 60
        for key, value in (cfg.get('estimates') or {}).items():
            self.estimates[('full', key)] = value
        for key, value in (estimates or {}).items():
            if value:
                self.estimates[('full', key)] = value
        self.timeout_factor = cfg.get('timeout_factor', 3.0)
        self.margin = cfg.get('time_guard_margin', 1.2)
        self.job = self.env.get('SLURM_JOB_ID', 'nojob')
        self.failures, self.launched = [], 0

    # ----- helpers
    def log(self, message):
        line = '%s [job %s] %s' % (now(), self.job, message)
        print(line, flush=True)
        path = self.c.root / 'state/events.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as f:
            f.write(line + '\n')

    def estimate(self, run):
        return float(self.estimates[(run['build'], run['mode'])])

    def guard(self, runs):
        if self.no_time_guard:
            return
        left = R.remaining_seconds(self.env)
        require(left is not None,
                'Cannot determine the remaining allocation time (no SLURM_JOB_END_TIME, no squeue answer, no '
                'SENGA_JOB_START_EPOCH/SENGA_JOB_TIME_LIMIT_S). Run through the job script written by submit, or '
                'pass --no-time-guard to run without the guard.')
        need = sum(self.estimate(r) for r in runs) * self.margin + 60
        if left < need:
            names = ' + '.join(r['run_id'] for r in runs)
            raise StopJob(EXIT_TIME_GUARD,
                          'TIME GUARD: %d s left in this allocation, but %s needs about %d s (estimate x %.1f + 60 s). '
                          'Stopped cleanly before launching it; nothing is lost. Continue with: bash %s submit '
                          '--campaign %s --resume' % (left, names, need, self.margin, self.c.root / 'launcher/run.sh', self.c.root))

    def reference_path(self, run):
        return self.c.root / 'references' / ('%s_steps%d.json' % (run['build'], run['steps']))

    def check_identity(self, builds):
        """Resume is refused when the executable or the plan differs from what earlier records used."""
        manifests = {name: self.c.verify_executable(name) for name in builds}
        for record in self.c.all_records():
            if record['scope'] != self.scope or record['superseded']:
                continue
            want = manifests.get(record['build'])
            require(want is None or record['executable_sha256'] == want['sha256'],
                    'Executable hash changed: run %s used %s, the %s build is now %s. Resume refused; start a new campaign.'
                    % (record['run_id'], record['executable_sha256'][:12], record['build'], want and want['sha256'][:12]))
            require(record['plan_hash'] == self.plan_sha,
                    'Plan hash changed: run %s belongs to plan %s, the stored plan is %s. Resume refused.'
                    % (record['run_id'], record['plan_hash'][:12], self.plan_sha[:12]))
        nodes = self.env.get('SLURM_JOB_NUM_NODES') or self.env.get('SLURM_NNODES')
        need = max(r['nodes'] for r in self.plan['runs'] if r['build'] in builds)
        require(nodes is None or int(nodes) >= need,
                'This allocation has %s node(s) but the planned runs need %d' % (nodes, need))
        return manifests

    # ----- one run
    def execute(self, run, catalog=None):
        c = self.c
        manifest = c.build(run['build'])
        exe = c.root / manifest['executable']
        require(run.get('argv_resolved', True), 'Run %s still has an unresolved d0 placeholder' % run['run_id'])
        P.check_argv(run['argv'])
        P.check_argv(run['argv_other_ranks'])
        if run['mode'] == 'P':
            catalog = R.verify_catalog_for(run, catalog, manifest['sha256'])
        else:
            require(catalog is None, 'Only precomputed runs take a catalog')
        loc = c.layout.locate(run['run_id']) if self.scope == 'sweep' else None
        if loc:
            attempt = c.layout.new_attempt(loc)
        else:
            folder = c.run_dir(self.scope, run['run_id'])
            folder.mkdir(parents=True, exist_ok=True)
            attempt = folder / ('attempt_%03d' % (len(c.attempts(self.scope, run['run_id'])) + 1))
        (attempt / 'ranks').mkdir(parents=True)
        if run['mode'] == 'A':
            (attempt / 'rank_0').mkdir()
        grid = c.config['builds'][run['build']]['grid']
        R.prepare_workdir(attempt / 'work', c.root / 'source/SENGA2', grid, run['steps'])
        spec = dict(run=run, attempt=str(attempt), workdir=str(attempt / 'work'), executable=str(exe), catalog=catalog)
        atomic(attempt / 'execution.json', spec)
        command = R.srun_command(run, c.root / 'launcher', attempt / 'execution.json', c.config['cpu_bind'])
        timeout = max(300.0, self.timeout_factor * self.estimate(run))
        self.log('RUN %s  stage=%s mode=%s decomposition=%s ranks=%d steps=%d  (limit %d s)'
                 % (run['run_id'], run['stage'], run['mode'], shape_text(run['triple']) if run['triple'] else 'unforced',
                    run['ranks'], run['steps'], timeout))
        start_utc, start = now(), time.time()
        rc, timed_out = R.launch(command, attempt / 'srun.log', timeout,
                                 dict(self.env, MPICH_GPU_SUPPORT_ENABLED='1', OMP_NUM_THREADS='1'))
        elapsed = time.time() - start
        self.launched += 1
        reference = read_json_or(self.reference_path(run))
        result = R.validate_run(attempt, run, dict(srun_returncode=rc, timed_out=timed_out, catalog=catalog,
                                                   reference=reference and reference['values'], tolerance=self.tolerance))
        m = result['metrics']
        if result['valid'] and reference is None:
            reference = dict(values=m['fingerprint'], run_id=run['run_id'], scope=self.scope, attempt=attempt.name,
                             set_utc=now(), note='first valid run of this build and step count')
            atomic(self.reference_path(run), reference)
        statuses = {s['rank']: s for s in (read_json_or(p) for p in (attempt / 'ranks').glob('status_*.json')) if s}
        first, other = statuses.get(0, {}), statuses.get(1, {})
        cfg = c.config
        autotune = self.freeze(run, attempt, result, manifest, loc) if run['mode'] == 'A' else None
        good = result['valid'] and (autotune is None or autotune['status'] == 'FROZEN')
        base = loc['final'] if loc and good else attempt     # where the run is kept from now on
        record = dict(
            record_version=1, campaign_id=c.id, plan_hash=self.plan_sha, seed=run['seed'], plan_position=run['position'],
            run_id=run['run_id'], scope=self.scope, attempt=attempt.name, stage=run['stage'],
            decomposition=shape_text(run['triple']) if run['triple'] else 'unforced',
            decomposition_planned=run['decomposition'], triple=run['triple'], mode=run['mode'],
            mode_name=run['mode_name'], repetition=run['repetition'], pair_index=run['pair_index'],
            order_in_pair=run['order_in_pair'], decomp_position=run['decomp_position'], build=run['build'],
            grid=grid, nodes=run['nodes'], nranks=run['ranks'], steps=run['steps'], gpudirect=run['gpudirect'],
            sentinel_kind=run.get('sentinel_kind', ''),
            slurm_job_id=self.env.get('SLURM_JOB_ID', ''), slurm_step_id=first.get('slurm_step_id', ''),
            node_list=self.env.get('SLURM_JOB_NODELIST', ''), hosts=sorted({r['host'] for r in m['ranks']}),
            start_utc=start_utc, end_utc=now(), launcher_elapsed_s=elapsed, srun_command=shell_join(command),
            srun_returncode=rc, timed_out=timed_out, argv=run['argv'], argv_other_ranks=run['argv_other_ranks'],
            environment=dict(rank_0=first.get('environment', {}), other_ranks=other.get('environment', {}),
                             cpu_bind='map_cpu:' + cfg['cpu_bind'], rank_0_cpu_affinity=first.get('cpu_affinity', [])),
            executable_sha256=manifest['sha256'], catalog_sha256=m['catalog_sha256'], catalog_rows=m['catalog_rows'],
            catalog_path=catalog['path'] if catalog else '',
            package_commit=c.package['commit'], ops_commit=cfg['ops_commit'], senga_commit=cfg['senga_commit'],
            module_list=self.env.get('LOADEDMODULES', ''), printed_grid=m['printed_grid'],
            total_ttime_s=m.get('total_ttime_s'), kernel_time_s=m['kernel_time_s'], halo_time_s=m['halo_time_s'],
            user_halo_time_s=m['user_halo_time_s'], halo_timers=m.get('halo_timers'),
            kernel_times=m['kernel_times'], ranks=m['ranks'],
            gpu_direct_reported=m['gpu_direct'], steps_seen=m['steps_seen'], fingerprint=m['fingerprint'],
            fingerprint_records=m['fingerprint_records'], fingerprint_difference=m.get('fingerprint_difference'),
            fingerprint_reference=reference and reference.get('run_id'), fingerprint_tolerance=self.tolerance,
            fallback_kernel_ids_rank0=m.get('fallback_kernel_ids_rank0'),
            fallback_kernel_ids_other_ranks=m.get('fallback_kernel_ids_other_ranks'),
            total_ttime_text=m.get('total_ttime_text'), kernel_time_text=m.get('kernel_time_text'),
            halo_time_text=m.get('halo_time_text'), user_halo_time_text=m.get('user_halo_time_text'),
            gpu_arch=D.gpu_arch(manifest), attempt_dir=str(attempt), run_dir=str(base),
            case_id=loc['case_id'] if loc else '',
            cont_dat=str(base / 'work/input/cont.dat'), verdict='VALID' if result['valid'] else 'INVALID',
            category=result['category'], reason='; '.join(result['reasons']), reasons=result['reasons'])
        if autotune is not None:
            record['autotune'] = autotune
        if loc:
            # Dataset files first, the record last (it marks the attempt as finished), then the
            # commit: a run enters its final folder only if it satisfies the commit rule.
            c.layout.write_run_files(attempt, record, loc, good)
        atomic(attempt / 'record.json', record)
        if loc and good:
            c.layout.commit(loc, attempt)
            c.layout.refresh()
        if result['valid']:
            self.log('  VALID  total_ttime=%.3f s  elapsed=%.0f s%s' % (
                m['total_ttime_s'], elapsed,
                ('  catalog=%s' % record['autotune']['status']) if run['mode'] == 'A' else ''))
        else:
            self.failures.append(run['run_id'])
            self.log('  INVALID [%s] %s\n         logs: %s' % (result['category'], record['reason'], attempt))
        if loc:
            self.log('  %s %s' % ('COMMITTED' if good else 'NOT COMMITTED (kept in _attempts)', base))
        if run['mode'] == 'A' and record['autotune']['status'] != 'FROZEN':
            if result['valid']:
                self.failures.append(run['run_id'])
            self.log('  AUTOTUNE_FAILED for %s: %s. No catalog was written and no precomputed run will be launched.'
                     % (record['decomposition'], record['autotune']['reason']))
        return record

    def freeze(self, run, attempt, result, manifest, loc=None):
        if not result['valid']:
            return dict(status='FAILED', reason='autotune run invalid: ' + '; '.join(result['reasons']))
        decomposition = shape_text(run['triple'])
        folder = self.c.catalog_folder(self.scope, run['build'], decomposition)
        grid = self.c.config['builds'][run['build']]['grid']
        context = dict(APP_NAME=D.APP, GRID_X=str(grid), GRID_Y=str(grid), GRID_Z=str(grid), GRID_CONFIG=D.GRID_CONFIG,
                       DECOMPOSITION_MODE='1', ACTIVE_NUM_RANKS=str(run['ranks']), shape=decomposition, APP_ARGS='',
                       GPU_ARCH=D.gpu_arch(manifest), EXECUTABLE_SHA256=manifest['sha256'], SAMPLED_AUTOTUNE='YES',
                       AUTOTUNE_SAMPLE_RANKS='0')
        try:
            made = R.build_catalog(attempt, folder, dict(
                decomposition=decomposition, scope=self.scope, build=run['build'], campaign_id=self.c.id,
                executable_sha256=manifest['sha256'], source_run_id=run['run_id'], plan_hash=self.plan_sha,
                slurm_job_id=self.env.get('SLURM_JOB_ID', ''), context=context, repetition=1,
                app_wall_time_s=result['metrics'].get('total_ttime_s'),
                source_run=str(loc['final']) if loc else str(attempt)))
        except R.CatalogError as error:
            return dict(status='FAILED', reason=str(error))
        return dict(status='FROZEN', reason='', catalog_sha256=made['sha256'], catalog_rows=made['rows'],
                    executed_kernels=len(made['executed_kernel_ids']), kernel_coverage=made['kernel_coverage'],
                    folder=str(folder), set_id=made['set_id'])

    def current(self, run_id):
        """State of a run; attempts cut short by the end of an allocation are set aside."""
        state, record, attempt = self.c.state(self.scope, run_id)
        if state == 'PENDING_COMMIT':
            self.commit_pending(self.c.layout.locate(run_id), attempt)
            state, record, attempt = self.c.state(self.scope, run_id)
        if state == 'INTERRUPTED':
            self.log('%s: attempt %s has no record (the allocation ended during the run); it is kept and marked superseded'
                     % (run_id, attempt.name))
            self.c.supersede(self.scope, run_id, 'interrupted: no record written')
            return 'MISSING', None
        if state == 'FAILED' and self.retry_failed:
            self.log('%s: failed earlier (%s); --retry-failed set, the failed attempt is kept and marked superseded'
                     % (run_id, record['category'] or 'AUTOTUNE_FAILED'))
            self.c.supersede(self.scope, run_id, 'explicit --retry-failed')
            return 'MISSING', None
        return state, record

    def commit_pending(self, loc, attempt):
        self.log('%s: valid attempt %s was not yet moved to its final folder (the job ended in between); committing it'
                 % (loc['run_id'], attempt.name))
        self.c.layout.commit(loc, attempt)
        self.c.layout.refresh()

    def catalog_of(self, a_run, a_record):
        """Catalog frozen by the autotune run of the same decomposition, or (None, reason)."""
        if a_record is None or a_record['verdict'] != 'VALID' or a_record.get('autotune', {}).get('status') != 'FROZEN':
            return None, 'AUTOTUNE_FAILED'
        try:
            manifest = self.c.build(a_run['build'])
            catalog = R.load_catalog(self.c.catalog_folder(self.scope, a_run['build'], a_record['decomposition']),
                                     a_record['decomposition'], manifest['sha256'])
            require(catalog['sha256'] == a_record['autotune']['catalog_sha256'],
                    'catalog sha256 differs from the one recorded by its autotune run')
        except ValueError as error:
            self.log('CATALOG REJECTED for %s: %s. No precomputed run will be launched.' % (a_record['decomposition'], error))
            return None, 'CATALOG_MISMATCH'
        return catalog, ''

    def single(self, run, catalog=None):
        """Run once unless a valid record exists; a failed run is never retried automatically."""
        state, record = self.current(run['run_id'])
        if state == 'VALID':
            self.log('SKIP %s: valid record exists' % run['run_id'])
            return record
        if state == 'FAILED':
            self.failures.append(run['run_id'])
            self.log('SKIP %s: failed earlier (%s) and is not retried automatically (use --retry-failed)'
                     % (run['run_id'], record['category'] or 'AUTOTUNE_FAILED'))
            return record
        self.guard([run])
        return self.execute(run, catalog)

    # ----- qualification
    def run_smoke(self, builds):
        self.check_identity(builds)
        d0, catalogs, a_records = None, {}, {}
        for planned in self.plan['runs']:
            if planned['build'] not in builds:
                continue
            run = planned
            if planned['decomposition'] == 'd0':
                if d0 is None:
                    state, record, _ = self.c.state(self.scope, 'q4_unforced_D')
                    if state != 'VALID':
                        self.failures.append(planned['run_id'])
                        self.log('SKIP %s: d0 unknown because q4_unforced_D has no valid record' % planned['run_id'])
                        continue
                    d0 = record['printed_grid']
                    atomic(self.c.root / 'state/qualification_d0.json', dict(d0=d0, source='q4_unforced_D'))
                run = P.resolve_d0(planned, d0)
            key = (run['stage'], run['build'], shape_text(run['triple']) if run['triple'] else 'unforced')
            catalog = None
            if run['mode'] == 'P':
                a_run, a_record = a_records.get(key, (None, None))
                catalog, why = self.catalog_of(a_run, a_record) if a_run else (None, 'AUTOTUNE_FAILED')
                if catalog is None:
                    self.failures.append(run['run_id'])
                    self.log('SKIP %s: %s for its decomposition; precomputed run not launched' % (run['run_id'], why))
                    continue
            record = self.single(run, catalog)
            if run['mode'] == 'A':
                a_records[key] = (run, record)
        if self.failures:
            self.log('Qualification pass finished with problems in: %s' % ', '.join(sorted(set(self.failures))))
            return EXIT_RUN_FAILURES
        self.log('Qualification pass finished: every run in this job is valid')
        return EXIT_OK

    # ----- sweep
    def decomposition_state_path(self, text):
        return self.c.root / 'state/decompositions' / (text + '.json')

    def run_sweep(self):
        c = self.c
        self.check_identity(['full'])
        c.layout.ensure(self.plan, self.plan_sha)
        self.log('Dataset: %s' % c.layout.dataset)
        for loc, attempt in c.layout.pending():
            self.commit_pending(loc, attempt)
        abort = c.root / 'state/ABORTED.json'
        if abort.exists():
            if not self.retry_failed:
                raise StopJob(EXIT_ABORT_FAILURES,
                              'An earlier job aborted this sweep (%s). Inspect state/events.log, then resubmit with '
                              '--retry-failed to continue.' % read_json(abort).get('reason', ''))
            abort.rename(abort.with_name('ABORTED.cleared_%s.json' % self.job))
        by_id = {r['run_id']: r for r in self.plan['runs']}
        # 1. unforced baseline: defines d0 (never a constant)
        base = [self.single(r) for r in self.plan['runs'] if r['stage'] == 'baseline']
        grids = sorted({tuple(r['printed_grid']) for r in base if r and r['verdict'] == 'VALID'})
        if len(grids) != 1:
            raise StopJob(EXIT_RUN_FAILURES,
                          'Cannot identify d0: the unforced baseline has %s. Inspect %s/baseline_unforced/ and %s, and '
                          'resubmit with --retry-failed.' % ('no valid run' if not grids else 'printed different grids %s' % grids,
                                                             c.layout.grid_folder, c.layout.attempts_root))
        d0 = list(grids[0])
        atomic(c.root / 'state/d0.json', dict(d0=d0, source='processor grid printed by the unforced baseline runs',
                                              runs=[r['run_id'] for r in base if r['verdict'] == 'VALID']))
        self.log('d0 (grid chosen by OPS without OPS_FORCE_DECOMP) = %s' % shape_text(d0))
        # 2. sentinel at the start of this allocation
        state, _, _ = c.state(self.scope, 'sentinel_start')
        if state in ('MISSING', 'INTERRUPTED'):
            self.single(P.resolve_d0(by_id['sentinel_start'], d0))
        else:
            mine = [r for r in c.all_records() if r['scope'] == self.scope and r['stage'] == 'sentinel'
                    and r['slurm_job_id'] == self.env.get('SLURM_JOB_ID', '') and not r['superseded']]
            if not mine:
                template = dict(by_id['sentinel_start'], run_id='sentinel_alloc_%s' % self.job,
                                sentinel_kind='allocation_start')
                self.single(P.resolve_d0(template, d0))
        # 3. decompositions in the frozen order
        consecutive = 0
        for dec in self.plan['decompositions']:
            before = self.launched
            status = self.decomposition(dec, by_id)
            if self.launched > before:   # only decompositions that ran in this job count
                consecutive = 0 if status == 'COMPLETE' else consecutive + 1
            if status != 'COMPLETE':
                self.failures.append('decomposition ' + dec['decomposition'])
                self.log('DECOMPOSITION %s (plan position %d): %s' % (dec['decomposition'], dec['position'], status))
            if consecutive >= MAX_CONSECUTIVE_FAILED:
                reason = '%d consecutive failed decompositions, last %s' % (consecutive, dec['decomposition'])
                atomic(abort, dict(reason=reason, utc=now(), slurm_job_id=self.job))
                raise StopJob(EXIT_ABORT_FAILURES, 'ABORT: %s. Nothing else is launched. Inspect state/events.log and the '
                              'run folders; continue with submit --resume --retry-failed once the cause is fixed.' % reason)
            sentinel = by_id.get('sentinel_after_p%03d' % dec['position'])
            if sentinel:
                self.single(P.resolve_d0(sentinel, d0))
        if self.failures:
            self.log('Sweep pass finished; runs with problems: %s' % ', '.join(sorted(set(self.failures))))
            return EXIT_RUN_FAILURES
        self.log('Sweep complete: every planned run has a valid record')
        return EXIT_OK

    def decomposition(self, dec, by_id):
        """A first, then the interleaved D/P pairs. Returns COMPLETE or the reason it is not."""
        text = dec['decomposition']
        prefix = 'sweep_p%03d_%s' % (dec['position'], text)
        marker = self.decomposition_state_path(text)
        if marker.exists():
            if not self.retry_failed:
                self.failures.append(prefix)
                return read_json(marker)['status']
            marker.rename(marker.with_name('%s.cleared_%s.json' % (text, self.job)))
        a_run = by_id[prefix + '_A']
        a_record = self.single(a_run)
        catalog, why = self.catalog_of(a_run, a_record)
        complete = catalog is not None
        for pair in range(1, self.plan['repetitions'] + 1):
            members = sorted((r for r in self.plan['runs'] if r['run_id'].startswith('%s_pair%d_' % (prefix, pair))),
                             key=lambda r: r['order_in_pair'])
            if catalog is None:   # D runs still serve the decomposition comparison
                members = [r for r in members if r['mode'] == 'D']
            records = self.pair(members, catalog)
            for run, record in zip(members, records):
                if record is None or record['verdict'] != 'VALID':
                    complete = False
                if (run['mode'] == 'D' and record is not None and record['verdict'] != 'VALID'
                        and record['category'] in RUNTIME_INVALID):
                    atomic(marker, dict(status='INVALID_AT_RUNTIME', decomposition=text, run_id=run['run_id'],
                                        category=record['category'], reason=record['reason'], utc=now()))
                    self.log('%s marked INVALID_AT_RUNTIME (%s in %s); its remaining runs are skipped'
                             % (text, record['category'], run['run_id']))
                    return 'INVALID_AT_RUNTIME'
        if catalog is None:
            return why
        return 'COMPLETE' if complete else 'INCOMPLETE'

    def pair(self, members, catalog):
        """A D/P pair is atomic within one allocation: an incomplete pair is rerun whole."""
        states = [self.current(r['run_id']) for r in members]
        names = [s for s, _ in states]
        if len(members) == 2:
            jobs = {rec['slurm_job_id'] for s, rec in states if s == 'VALID'}
            if names == ['VALID', 'VALID'] and len(jobs) == 1:
                return [rec for _, rec in states]
            if 'FAILED' in names:
                for run, (s, rec) in zip(members, states):
                    if s == 'FAILED':
                        self.failures.append(run['run_id'])
                self.log('SKIP pair %s: a member failed earlier; not retried automatically (use --retry-failed)'
                         % ' / '.join(r['run_id'] for r in members))
                return [rec for _, rec in states]
            if 'VALID' in names:
                reason = 'incomplete pair: partner missing or run in another allocation'
                for run, (s, rec) in zip(members, states):
                    if s == 'VALID':
                        self.log('%s: %s; the partial run is kept, marked superseded, and the pair is rerun whole'
                                 % (run['run_id'], reason))
                        self.c.supersede(self.scope, run['run_id'], reason)
            self.guard(members)
        else:
            if names and names[0] in ('VALID', 'FAILED'):
                return [self.single(members[0])]
            self.guard(members)
        out = []
        for run in members:
            record = self.execute(run, catalog if run['mode'] == 'P' else None)
            out.append(record)
            if run['mode'] == 'D' and record['verdict'] != 'VALID' and record['category'] in RUNTIME_INVALID:
                break
        return out + [None] * (len(members) - len(out))
