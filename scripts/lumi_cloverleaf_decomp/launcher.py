#!/usr/bin/env python3
"""CloverLeaf_3D MPI-decomposition sweep on LUMI-G: default, autotuned and replayed GPU block sizes.

Typical order (every command is described in README.md):
  plan (dry run) -> prepare -> build -> smoke -> gonogo -> plan --campaign --steps N -> submit [--resume] -> collect
A campaign prepared with an older package is brought up to date with `upgrade` (run from the checkout).
No command submits a job unless --submit (or the submit command) is given."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import collect as C
import engine as E
import plan as P
import runtime as R
import sources as S
from common import (require, sha, atomic, atomic_bytes, read_json, read_json_or, now, shell_join, query, lock,
                    write_csv, slurm_seconds, StopJob, EXIT_OK, EXIT_ERROR)

HERE = Path(__file__).resolve().parent
PACKAGE_FILES = ('run.sh', 'launcher.py', 'common.py', 'plan.py', 'sources.py', 'runtime.py', 'engine.py', 'collect.py',
                 'dataset.py')
DEFAULT_STEPS = 100
NAME = re.compile(r'[A-Za-z0-9_.-]+')
TIME = re.compile(r'(?:\d+-)?\d+:\d{2}:\d{2}')


def plan_config(args):
    return dict(ranks=args.nodes * 8, nodes=args.nodes, grid=args.grid, steps=args.steps,
                repetitions=args.repetitions, seed=args.seed, fixed_order=args.fixed_order,
                sentinel_every=args.sentinel_every, small_grid=args.small_grid, smoke_steps=args.smoke_steps,
                decompositions=P.decomposition_option(args.decompositions))


def add_plan_options(p, steps_default=DEFAULT_STEPS):
    p.add_argument('--nodes', type=int, default=20, help='LUMI-G nodes; 8 MPI ranks (one per GCD) each (default 20)')
    p.add_argument('--grid', type=int, default=1440, help='cubic global grid N of the sweep (default 1440)')
    p.add_argument('--small-grid', type=int, default=256, help='cubic grid of the one-node qualification runs (default 256)')
    p.add_argument('--steps', type=int, default=steps_default,
                   help='time steps (end_step) of the qualification runs and default of the sweep (default 100, a '
                        'placeholder: gonogo suggests the value for runs of about 5 minutes)')
    p.add_argument('--smoke-steps', type=int, default=10, help='time steps of the Q4 memory checks (default 10)')
    p.add_argument('--repetitions', type=int, default=1, help='default/precomputed pairs per decomposition (default 1)')
    p.add_argument('--decompositions', default='default', metavar='default|all|PxxPyxPz,...',
                   help='default: the 16 fixed decompositions (%s); all: every valid ordered triple; or an explicit '
                        'comma-separated list' % ' '.join(P.DEFAULT_DECOMPOSITIONS))
    p.add_argument('--seed', type=int, default=P.DEFAULT_SEED, help='seed of the decomposition order (default %(default)s)')
    p.add_argument('--fixed-order', action='store_true', help='lexicographic decomposition order instead of the seeded one')
    p.add_argument('--sentinel-every', type=int, default=8, help='sentinel run after every N decompositions (default 8)')


def write_plan(folder, plan, name='plan'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = P.plan_bytes(plan)
    atomic_bytes(folder / (name + '.json'), data)
    (folder / (name + '.json.sha256')).write_text('%s  %s.json\n' % (P.plan_hash(plan), name))
    write_csv(folder / (name + '.csv'), P.plan_rows(plan), P.PLAN_CSV_FIELDS)
    return P.plan_hash(plan)


def print_plan(plan, digest, where):
    if plan['kind'] == 'sweep':
        print('Sweep plan: %d ranks on %d nodes, %d^3, %d steps, seed %d, order: %s'
              % (plan['ranks'], plan['nodes'], plan['grid'], plan['steps'], plan['seed'], plan['order']))
        for item in plan['skipped']:
            print('  NOTICE skipped %s: %s' % (item['decomposition'], item['reason']))
        for item in plan['excluded']:
            print('  NOTICE excluded %s: %s' % (item['decomposition'], item['reason']))
        print('  %d decompositions (%s; %d skipped as invalid, %d excluded): %s'
              % (len(plan['decompositions']), plan['decomposition_set'], len(plan['skipped']), len(plan['excluded']),
                 ' '.join(sorted(d['decomposition'] for d in plan['decompositions']))))
    else:
        print('Qualification plan: small grid %d^3, full grid %d^3 on %d ranks' % (plan['small_grid'], plan['grid'], plan['ranks']))
        for stage in sorted(plan['stages']):
            print('  %s: %s' % (stage, plan['stages'][stage]))
        for item in plan['skipped']:
            print('  NOTICE not planned %s: %s' % (item['run_id'], item['reason']))
    counts = P.summary(plan)
    for stage, mode in sorted(counts):
        print('  %-9s %-11s %4d runs' % (stage, P.MODE_NAME[mode], counts[(stage, mode)]))
    print('  total %d runs; plan sha256 %s' % (len(plan['runs']), digest))
    print('  written: %s/{%s.json,%s.csv,%s.json.sha256}' % ((where,) + (('plan',) * 3 if plan['kind'] == 'sweep' else ('smoke_plan',) * 3)))


def command_plan(args):
    args.steps_given = args.steps is not None
    if not args.steps_given:
        args.steps = DEFAULT_STEPS
    if args.campaign:
        c = E.Campaign(args.campaign)
        if (c.root / 'plan.json').exists():
            plan, digest = c.plan('sweep')
            print('The sweep plan of this campaign is already frozen and is never regenerated:')
            print_plan(plan, digest, c.root)
            return EXIT_OK
        cfg = dict(c.config['plan'], exclude=args.exclude)
        if args.steps_given:
            cfg['steps'] = args.steps                 # the sweep step count, chosen after qualification
        plan = P.make_sweep_plan(cfg)
        digest = write_plan(c.root, plan)
        print_plan(plan, digest, c.root)
        print('Plan frozen in the campaign with %d steps per run. Resume always reads this file.' % plan['steps'])
        go = read_json_or(c.root / 'gonogo.json') or {}
        if go.get('projected_steps') != plan['steps']:
            print('Next: bash %s gonogo --campaign %s --steps %d ... (the budget must be projected with %d steps '
                  'before submit)' % (c.root / 'launcher/run.sh', c.root, plan['steps'], plan['steps']))
        return EXIT_OK
    require(args.output_dir, 'Give --output-dir DIR for a dry run, or --campaign DIR to freeze the plan of a campaign')
    cfg = dict(plan_config(args), exclude=args.exclude)
    plan = P.make_sweep_plan(cfg)
    print_plan(plan, write_plan(args.output_dir, plan), args.output_dir)
    smoke = P.make_smoke_plan(cfg)
    print_plan(smoke, write_plan(args.output_dir, smoke, 'smoke_plan'), args.output_dir)
    print('Dry run only: nothing was built or submitted.')
    return EXIT_OK


# --------------------------------------------------------------------------- prepare
def package_identity(ops_root):
    info = dict(commit='', dirty=None, files_sha256={name: sha(HERE / name) for name in PACKAGE_FILES})
    r = query(['git', '-C', str(ops_root), 'rev-parse', 'HEAD'])
    if r['status'] == 'OK':
        info['commit'] = r['stdout'].strip()
        d = query(['git', '-C', str(ops_root), 'status', '--porcelain', '-uall', '--', str(HERE)], timeout=120)
        info['dirty'] = bool(d['stdout'].strip()) if d['status'] == 'OK' else None
    return info


def command_prepare(args):
    account = args.account or os.environ.get('SLURM_ACCOUNT', '')
    require(account, 'No Slurm account: pass --account <your LUMI project> or export SLURM_ACCOUNT. '
                     'The package never assumes a project.')
    require(NAME.fullmatch(account), 'Invalid account name %r' % account)
    for value in (args.partition, args.smoke_partition, args.build_partition, args.date_stamp):
        require(NAME.fullmatch(value), 'Invalid name %r' % value)
    require(args.consistency_tolerance >= 0, '--consistency-tolerance must not be negative')
    cfg = plan_config(args)
    P.make_sweep_plan(dict(cfg, exclude=[]))   # parameter check before anything is written
    smoke = P.make_smoke_plan(cfg)
    if args.campaign:
        campaign = args.campaign.expanduser().resolve()
    else:
        root = args.output_root or Path('/scratch') / account / 'ops-cloverleaf-decomp' / 'campaigns'
        campaign = root.expanduser().resolve() / ('CLOVERLEAF_DECOMP_' + args.date_stamp)
    require(not re.search(r'\s', str(campaign)), 'The campaign path must not contain whitespace (make variables): %s' % campaign)
    require(not (campaign / 'campaign.json').exists(), 'Campaign already exists: %s' % campaign)
    env_file = args.env_file.expanduser().resolve()
    require(env_file.is_file(), 'Environment file not found: %s' % env_file)
    ops_root = args.ops_root.expanduser().resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    with lock(campaign / '.prepare.lock'):
        (campaign / 'launcher').mkdir(exist_ok=True)
        for name in PACKAGE_FILES:
            shutil.copy2(str(HERE / name), str(campaign / 'launcher' / name))
        shutil.copy2(str(env_file), str(campaign / 'environment.sh'))
        print('Snapshot of OPS sources (library, makefiles, translator, %s) from %s' % (S.APP_DIR, ops_root), flush=True)
        ops = S.snapshot_ops(ops_root, campaign)
        atomic(campaign / 'source_snapshot.json', dict(ops=ops, created_utc=now()))
        package = package_identity(ops_root)
        config = dict(
            version=1, campaign_id=campaign.name, created_utc=now(), date_stamp=args.date_stamp, account=account,
            application=S.APP_DIR, partition=args.partition,
            smoke_partition=args.smoke_partition, build_partition=args.build_partition,
            plan=cfg, seed=cfg['seed'], nodes=cfg['nodes'], ranks=cfg['ranks'], grid=cfg['grid'], steps=cfg['steps'],
            repetitions=cfg['repetitions'], sentinel_every=cfg['sentinel_every'],
            builds=dict(small=dict(grid=cfg['small_grid']), full=dict(grid=cfg['grid'])),
            consistency_tolerance=args.consistency_tolerance, cpu_bind=args.cpu_bind, timeout_factor=3.0,
            time_guard_margin=1.2, translator_python=str(args.translator_python.expanduser().absolute()) if args.translator_python else '',
            ops_root=str(ops_root), ops_commit=ops['commit'], ops_uncommitted_changes=ops['uncommitted_changes_in_snapshot'],
            package=package, environment_sha256=sha(campaign / 'environment.sh'))
        atomic(campaign / 'campaign.json', config)
        digest = write_plan(campaign, smoke, 'smoke_plan')
    print('OPS commit %s%s' % (ops['commit'], ' (with uncommitted changes, saved as source/ops_local_changes.patch)'
                               if ops['uncommitted_changes_in_snapshot'] else ''))
    print('Qualification plan frozen (sha256 %s)' % digest)
    print('Campaign: %s' % campaign)
    print('Next: bash %s build --campaign %s' % (campaign / 'launcher/run.sh', campaign))
    return EXIT_OK


# --------------------------------------------------------------------------- Slurm helpers
def resource_args(c, partition, nodes, walltime):
    return ['--account=' + c.config['account'], '--partition=' + partition, '--nodes=%d' % nodes,
            '--ntasks-per-node=8', '--gpus-per-node=8', '--cpus-per-task=7', '--mem=0', '--time=' + walltime, '--exclusive']


def job_script(c, walltime, job_args):
    command = ['bash', str(c.root / 'launcher/run.sh'), 'job', '--campaign', str(c.root)] + job_args
    return ('#!/usr/bin/env bash\n'
            '# Written by lumi_cloverleaf_decomp; reproduces one job of campaign %s\n'
            'set -o pipefail\n'
            'export CLOVER_JOB_START_EPOCH=$(date +%%s)\n'
            'export CLOVER_JOB_TIME_LIMIT_S=%d\n'
            '# The environment file is sourced before "set -u": module functions may read unset variables.\n'
            'if ! source %s; then\n'
            '  echo "ERROR: the environment file failed: %s" >&2\n'
            '  exit 2\n'
            'fi\n'
            'set -u\n'
            '%s\n'
            'status=$?\n'
            'echo "lumi_cloverleaf_decomp job finished with exit status $status (see README.md, Exit statuses)"\n'
            'exit $status\n') % (c.id, slurm_seconds(walltime), shlex.quote(str(c.root / 'environment.sh')),
                                 c.root / 'environment.sh', shell_join(command))


def sbatch(c, kind, command, script_text, script_name, dry_run):
    path = c.root / 'slurm' / script_name
    path.parent.mkdir(exist_ok=True)
    path.write_text(script_text)
    command = command + [str(path)]
    if dry_run:
        print('DRY RUN, not submitted: ' + shell_join(command))
        return ''
    require(shutil.which('sbatch'), 'sbatch not found: submit from a LUMI login node')
    r = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(r.returncode == 0, 'sbatch failed: %s\n  command: %s' % (r.stderr.strip(), shell_join(command)))
    job = r.stdout.strip().split(';')[0]
    require(re.fullmatch('[0-9]+', job), 'Unexpected sbatch output: %r' % r.stdout)
    state = read_json_or(c.root / 'submissions.json') or dict(jobs=[])
    state['jobs'].append(dict(kind=kind, job_id=job, submitted_utc=now(), command=shell_join(command)))
    atomic(c.root / 'submissions.json', state)
    print('Submitted %s job %s' % (kind, job))
    return job


def active_jobs(c, kinds):
    """Job ids of earlier submissions of these kinds that Slurm still lists as pending or running."""
    state = read_json_or(c.root / 'submissions.json') or dict(jobs=[])
    ids = [j['job_id'] for j in state['jobs'] if j['kind'] in kinds]
    if not ids:
        return []
    r = query(['squeue', '-h', '-j', ','.join(ids), '-o', '%i'])
    if r['status'] == 'NOT_QUERIED':
        raise ValueError('squeue not found: cannot verify that earlier jobs (%s) have ended' % ','.join(ids))
    return [i for i in r.get('stdout', '').split() if i in ids]


def check_time(value):
    require(TIME.fullmatch(value), 'Use the Slurm time format [days-]HH:MM:SS, got %r' % value)
    return value


# --------------------------------------------------------------------------- build
def command_build(args):
    if args.print_toolchain:
        return S.print_toolchain(dict(os.environ))
    require(args.campaign, '--campaign is required')
    c = E.Campaign(args.campaign)
    if args.submit or args.dry_run:
        text = job_script(c, check_time(args.time), ['--kind', 'build'])
        command = ['sbatch', '--parsable', '--account=' + c.config['account'], '--partition=' + c.config['build_partition'],
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=%d' % args.cpus, '--gpus-per-node=1', '--time=' + args.time,
                   '--job-name=clover_build', '--output=%s/slurm/build-%%j.out' % c.root]
        sbatch(c, 'build', command, text, 'build.sh', args.dry_run)
        return EXIT_OK
    # Local build (login node or interactive allocation): source the frozen environment first.
    command = ['bash', '-c', 'source "$1" || { echo "ERROR: environment file failed: $1" >&2; exit 2; }; shift; exec "$@"',
               'clover-build', str(c.root / 'environment.sh'), 'bash', str(c.root / 'launcher/run.sh'), 'job',
               '--campaign', str(c.root), '--kind', 'build']
    return subprocess.run(command).returncode


# --------------------------------------------------------------------------- smoke / submit
def command_smoke(args):
    c = E.Campaign(args.campaign)
    plan, digest = c.plan('qualification')
    if not args.submit:
        print_plan(plan, digest, c.root)
        for run in plan['runs']:
            print('  %2d %-4s %-26s %-5s %-8s %3d ranks %3d steps  %s'
                  % (run['position'], run['stage'], run['run_id'], run['build'], run['decomposition'], run['ranks'],
                     run['steps'], ' '.join(run['argv'])))
        print('Dry run: nothing submitted. Add --submit to send the qualification jobs.')
        return EXIT_OK
    stages = ['small', 'full'] if args.stage == 'all' else [args.stage]
    c.verify_executable()
    busy = active_jobs(c, ['smoke-small', 'smoke-full'])
    require(not busy, 'Qualification job(s) %s are still pending or running; wait for them or cancel them first' % ','.join(busy))
    previous = ''
    for name in stages:
        nodes = 1 if name == 'small' else c.config['nodes']
        walltime = check_time(args.time_small if name == 'small' else args.time_full)
        if name == 'small':
            partition = args.smoke_partition or c.config['smoke_partition']
        else:
            partition = c.config['partition']
        job_args = ['--kind', 'smoke', '--builds', name] + (['--retry-failed'] if args.retry_failed else [])
        job_args += ['--no-time-guard'] if args.no_time_guard else []
        command = ['sbatch', '--parsable'] + resource_args(c, partition, nodes, walltime)
        if previous:
            command.append('--dependency=afterok:' + previous)
        command += ['--job-name=clover_q_' + name, '--output=%s/slurm/smoke-%s-%%j.out' % (c.root, name)]
        previous = sbatch(c, 'smoke-' + name, command, job_script(c, walltime, job_args), 'smoke_%s.sh' % name, args.dry_run)
    print('When the jobs have ended: bash %s gonogo --campaign %s ...' % (c.root / 'launcher/run.sh', c.root))
    return EXIT_OK


def command_submit(args):
    c = E.Campaign(args.campaign)
    plan, digest = c.plan('sweep')
    c.verify_executable()
    go = read_json_or(c.root / 'gonogo.json')
    if args.override_gonogo:
        atomic(c.root / 'gonogo_override.json', dict(reason=args.override_gonogo, utc=now(), gonogo=go))
        print('WARNING: go/no-go overridden by the user: %s' % args.override_gonogo)
    else:
        require(go is not None and go['verdict'] == 'GO',
                'The go/no-go verdict is %s. Run: bash %s gonogo --campaign %s ... (or, knowingly, pass '
                '--override-gonogo "reason", which is recorded).' % (go['verdict'] if go else 'missing',
                                                                    c.root / 'launcher/run.sh', c.root))
        require(go.get('projected_steps') == plan['steps'],
                'The go/no-go budget was projected for %s steps per run, but the frozen plan has %d. Run: bash %s gonogo '
                '--campaign %s --steps %d ... again (or, knowingly, pass --override-gonogo "reason").'
                % (go.get('projected_steps'), plan['steps'], c.root / 'launcher/run.sh', c.root, plan['steps']))
    records = [r for r in c.all_records() if r['scope'] == 'sweep']
    earlier = [j for j in (read_json_or(c.root / 'submissions.json') or dict(jobs=[]))['jobs'] if j['kind'] == 'sweep']
    require(args.resume or not (records or earlier),
            'This sweep was already submitted (%d job(s), %d run records). Use --resume to continue it.' % (len(earlier), len(records)))
    if not args.dry_run:
        busy = active_jobs(c, ['sweep'])
        require(not busy, 'Sweep job(s) %s are still pending or running. Wait for them (or scancel) before resubmitting, '
                'so that no run is duplicated.' % ','.join(busy))
    walltime = check_time(args.time)
    job_args = ['--kind', 'sweep']
    job_args += ['--retry-failed'] if args.retry_failed else []
    job_args += ['--no-time-guard'] if args.no_time_guard else []
    for mode, value in (('A', args.est_autotune), ('D', args.est_default), ('P', args.est_precomputed)):
        if value:
            job_args += ['--estimate', '%s=%d' % (mode, value)]
    require(args.jobs >= 1, '--jobs must be at least 1')
    previous = ''
    for index in range(args.jobs):
        command = ['sbatch', '--parsable'] + resource_args(c, c.config['partition'], c.config['nodes'], walltime)
        if previous:
            command.append('--dependency=afterany:' + previous)
        command += ['--job-name=clover_sweep', '--output=%s/slurm/sweep-%%j.out' % c.root]
        # --retry-failed applies to the first job of a chain only.
        text = job_script(c, walltime, job_args if index == 0 else [a for a in job_args if a != '--retry-failed'])
        previous = sbatch(c, 'sweep', command, text, 'sweep_%d.sh' % index, args.dry_run)
    print('Plan %s; %d job(s) of %s on %d nodes. A job that finds nothing left to do ends at once.'
          % (digest[:12], args.jobs, walltime, c.config['nodes']))
    print('Follow with: bash %s status --campaign %s' % (c.root / 'launcher/run.sh', c.root))
    return EXIT_OK


# --------------------------------------------------------------------------- inside a job
def command_job(args):
    c = E.Campaign(args.campaign)
    builds = [b for b in (args.builds or '').split(',') if b]
    if args.kind == 'build':
        S.Builder(c.root).build()
        return EXIT_OK
    estimates = {}
    for item in args.estimate or []:
        mode, value = item.split('=')
        estimates[mode] = float(value)
    with lock(c.root / 'state' / ('.%s.lock' % args.kind)):
        try:
            if args.kind == 'smoke':
                plan, digest = c.plan('qualification')
                engine = E.Engine(c, 'qualification', plan, digest, args.retry_failed, args.no_time_guard, estimates)
                return engine.run_smoke(builds)
            plan, digest = c.plan('sweep')
            engine = E.Engine(c, 'sweep', plan, digest, args.retry_failed, args.no_time_guard, estimates)
            return engine.run_sweep()
        except StopJob as stop:
            print('%s [job %s] %s' % (now(), os.environ.get('SLURM_JOB_ID', 'nojob'), stop), flush=True)
            with (c.root / 'state/events.log').open('a') as f:
                f.write('%s [job %s] %s\n' % (now(), os.environ.get('SLURM_JOB_ID', 'nojob'), stop))
            return stop.status


def command_gonogo(args):
    require(args.steps is None or args.steps >= 1, '--steps must be at least 1')
    require(args.target_seconds > 0, '--target-seconds must be positive')
    return C.gonogo(E.Campaign(args.campaign), args.gpu_hours_per_node_hour, args.remaining_gpu_hours,
                    args.disk_available_gb, args.steps, args.target_seconds, args.launch_overhead_s,
                    args.extra_allocations)


def command_collect(args):
    C.collect(E.Campaign(args.campaign))
    return EXIT_OK


def command_status(args):
    c = E.Campaign(args.campaign)
    print('Campaign %s  (account %s, %d nodes, %d^3, %d qualification steps)'
          % (c.id, c.config['account'], c.config['nodes'], c.config['grid'], c.config['steps']))
    manifest = read_json_or(c.root / 'build' / S.BUILD_NAME / 'build_manifest.json')
    print('  build %s' % (('sha256 ' + manifest['sha256']) if manifest else 'not built'))
    measured = read_json_or(c.root / 'measured_times.json')
    if measured and measured.get('per_step_s', {}).get('D'):
        print('  time per step (Q5, default blocks): %.4f s; suggested --steps for %.0f s: %s'
              % (measured['per_step_s']['D'], measured.get('target_seconds', 300), measured.get('suggested_steps')))
    for kind, scope in (('qualification', 'qualification'), ('sweep', 'sweep')):
        name = 'plan.json' if kind == 'sweep' else 'smoke_plan.json'
        if not (c.root / name).exists():
            print('  %s plan: not frozen' % kind)
            continue
        plan, digest = c.plan(kind)
        counts = {}
        for run in plan['runs']:
            state = c.state(scope, run['run_id'])[0]
            counts[state] = counts.get(state, 0) + 1
        print('  %s plan %s (%d steps): %s' % (kind, digest[:12], plan['steps'],
                                             ', '.join('%d %s' % (v, k) for k, v in sorted(counts.items()))))
    print('  dataset: %s%s' % (c.layout.dataset, '' if c.layout.dataset.is_dir() else ' (created by the first sweep job)'))
    go = read_json_or(c.root / 'gonogo.json')
    print('  go/no-go: %s' % (go['verdict'] if go else 'not evaluated'))
    d0 = read_json_or(c.root / 'state/d0.json')
    print('  d0: %s' % (P.shape_text(d0['d0']) if d0 else 'not yet known'))
    marks = sorted((c.root / 'state/decompositions').glob('*x*x*.json')) if (c.root / 'state/decompositions').is_dir() else []
    for mark in marks:
        if '.cleared_' not in mark.name:
            data = read_json(mark)
            print('  %s: %s (%s)' % (data['decomposition'], data['status'], data['category']))
    if (c.root / 'state/ABORTED.json').exists():
        print('  ABORTED: %s' % read_json(c.root / 'state/ABORTED.json')['reason'])
    state = read_json_or(c.root / 'submissions.json') or dict(jobs=[])
    ids = ','.join(j['job_id'] for j in state['jobs'])
    for job in state['jobs']:
        print('  job %s (%s) submitted %s' % (job['job_id'], job['kind'], job['submitted_utc']))
    if ids:
        r = query(['sacct', '-n', '-X', '-j', ids, '--format=JobID,JobName,State,ExitCode,Elapsed', '-P'])
        print('  sacct: ' + (r['stdout'].strip().replace('\n', '\n         ') if r['status'] == 'OK' else r.get('message', r.get('stderr', ''))))
    log = c.root / 'state/events.log'
    if log.is_file():
        print('  last events:')
        for line in log.read_text().splitlines()[-args.events:]:
            print('    ' + line)
    return EXIT_OK


# --------------------------------------------------------------------------- upgrade
def launcher_hashes(folder):
    folder = Path(folder)
    return {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()} if folder.is_dir() else {}


def command_upgrade(args):
    """Replace $C/launcher with this package. Refused while a job of the campaign may run or once the
    sweep has started; executables, plans, catalogs and qualification results are never touched."""
    c = E.Campaign(args.campaign)
    frozen = (c.root / 'launcher').resolve()
    require(HERE != frozen, 'Run upgrade with the run.sh of your OPS checkout (bash ~/OPS/scripts/lumi_cloverleaf_decomp/run.sh '
                            'upgrade --campaign %s), not with the campaign copy it replaces.' % c.root)
    busy = active_jobs(c, ['build', 'smoke-small', 'smoke-full', 'sweep'])
    require(not busy, 'Job(s) %s of this campaign are still pending or running (squeue). Wait for them to end, then '
                      'run upgrade again. Nothing was changed.' % ','.join(busy))
    sweep_jobs = [j['job_id'] for j in (read_json_or(c.root / 'submissions.json') or dict(jobs=[]))['jobs']
                  if j['kind'] == 'sweep']
    old_runs = sorted((c.root / 'runs/sweep').glob('*/attempt_*')) if (c.root / 'runs/sweep').is_dir() else []
    started = bool(sweep_jobs or old_runs or c.layout.sweep_runs_exist())
    require(args.restart_sweep or not started,
            'The sweep of this campaign has already started (%d sweep job(s) submitted, %d run attempt(s) on disk). '
            'A running sweep is never switched to another launcher. To set the sweep aside and start it again, '
            'add --restart-sweep (nothing is deleted). Nothing was changed.'
            % (len(sweep_jobs), len(old_runs) + len(c.layout.record_paths())))
    frozen_plan = (c.root / 'plan.json').exists() and not args.restart_sweep
    old_reps = c.config['repetitions']
    new_reps = old_reps if args.repetitions is None else args.repetitions
    old_set = c.config['plan'].get('decompositions', list(P.DEFAULT_DECOMPOSITIONS))
    new_set = old_set if args.decompositions is None else P.decomposition_option(args.decompositions)
    if new_reps != old_reps or new_set != old_set:
        require(not frozen_plan,
                'The sweep plan is already frozen (plan.json: %d repetition(s), decompositions %s). Add --restart-sweep '
                'to set the frozen sweep aside and plan it again. Nothing was changed.' % (old_reps, P.describe_set(old_set)))
    P.make_sweep_plan(dict(c.config['plan'], repetitions=new_reps, decompositions=new_set,
                           exclude=[]))   # parameter check before anything is changed
    with lock(c.root / 'state' / '.sweep.lock'), lock(c.root / 'state' / '.smoke.lock'):
        old = launcher_hashes(c.root / 'launcher')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archived = archive_sweep(c, stamp) if args.restart_sweep else []
        staging = c.root / ('launcher.new_' + stamp)
        staging.mkdir()
        for name in PACKAGE_FILES:
            shutil.copy2(str(HERE / name), str(staging / name))
        keep = c.root / 'launcher_previous' / stamp
        keep.parent.mkdir(exist_ok=True)
        os.rename(str(c.root / 'launcher'), str(keep))         # the old copy is kept, not deleted
        os.rename(str(staging), str(c.root / 'launcher'))
        new = launcher_hashes(c.root / 'launcher')
        if new_reps != old_reps or new_set != old_set:
            config = read_json(c.root / 'campaign.json')
            config['repetitions'] = new_reps
            config['plan'] = dict(config['plan'], repetitions=new_reps, decompositions=new_set)
            atomic(c.root / 'campaign.json', config)
        history = read_json_or(c.root / 'launcher_upgrades.json') or dict(upgrades=[])
        history['upgrades'].append(dict(
            utc=now(), previous_launcher=str(keep), old_files_sha256=old, new_files_sha256=new,
            package=package_identity(args.ops_root.expanduser().resolve()), repetitions_old=old_reps,
            repetitions_new=new_reps, decompositions_old=old_set, decompositions_new=new_set,
            sweep_restarted=bool(args.restart_sweep), archived=archived, dataset=str(c.layout.dataset)))
        atomic(c.root / 'launcher_upgrades.json', history)
    changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    print('Launcher of %s replaced by this package (%d file(s) changed: %s).' % (c.id, len(changed), ' '.join(changed) or 'none'))
    print('Previous launcher kept in %s; hashes recorded in %s' % (keep, c.root / 'launcher_upgrades.json'))
    if new_reps != old_reps:
        print('Repetitions per decomposition: %d -> %d (campaign.json)' % (old_reps, new_reps))
    else:
        print('Repetitions per decomposition: %d (unchanged)' % new_reps)
    print('Decompositions: %s%s' % (P.describe_set(new_set), '' if new_set == old_set else ' (was %s)' % P.describe_set(old_set)))
    if args.restart_sweep:
        print('Previous sweep set aside in %s (%d item(s); nothing deleted).'
              % (c.root / 'archive' / ('sweep_' + stamp), len(archived)))
    print('The sweep will write its dataset to %s' % c.layout.dataset)
    print('From now on use: bash %s <command> --campaign %s' % (c.root / 'launcher/run.sh', c.root))
    if args.restart_sweep or not (c.root / 'plan.json').exists():
        print('Next: bash %s plan --campaign %s' % (c.root / 'launcher/run.sh', c.root))
    return EXIT_OK


# Files and folders that belong to the sweep only. Qualification records, executables, the smoke
# plan, references, measured times and the qualification d0 are never moved.
SWEEP_ITEMS = ('plan.json', 'plan.json.sha256', 'plan.csv', 'runs/sweep', 'catalogs/sweep', 'state/decompositions',
               'state/d0.json', 'state/ABORTED.json', 'gonogo.json', 'gonogo_override.json', 'results')


def archive_sweep(c, stamp):
    """Move every sweep artefact into $C/archive/sweep_<stamp>/ (never deletes). Returns the moved paths."""
    target = c.root / 'archive' / ('sweep_' + stamp)
    target.mkdir(parents=True)
    moved = []
    for rel in SWEEP_ITEMS + (c.layout.dataset.name,):
        source = c.root / rel
        if source.exists():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(source), str(target / rel))
            moved.append(rel)
    submissions = c.root / 'submissions.json'
    record = read_json_or(submissions)
    if record and any(j['kind'] == 'sweep' for j in record['jobs']):
        atomic(target / 'submissions.json', record)
        atomic(submissions, dict(record, jobs=[j for j in record['jobs'] if j['kind'] != 'sweep']))
        moved.append('submissions.json (sweep entries)')
    atomic(target / 'ARCHIVED.json', dict(utc=now(), moved=moved, reason='upgrade --restart-sweep'))
    return moved


def command_rank(args):
    return R.rank_worker(args.spec)


def parser():
    p = argparse.ArgumentParser(prog='run.sh', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', metavar='COMMAND')

    s = sub.add_parser('plan', help='list the decompositions and write the ordered run list (dry run, no Slurm), '
                                    'or freeze the sweep plan of a campaign')
    add_plan_options(s, steps_default=None)
    s.add_argument('--output-dir', type=Path, help='write plan.json / plan.csv / smoke_plan.* here (dry run)')
    s.add_argument('--campaign', type=Path, help='freeze the sweep plan inside this campaign (uses its parameters; '
                   '--steps sets the time steps of the sweep runs)')
    s.add_argument('--exclude', action='append', default=[], metavar='PxxPyxPz', help='leave one decomposition out (repeatable)')
    s.set_defaults(func=command_plan)

    s = sub.add_parser('prepare', help='create a campaign: frozen launcher, environment and OPS snapshot (with CloverLeaf_3D)')
    add_plan_options(s)
    s.add_argument('--account', default='', help='Slurm account (LUMI project); default: $SLURM_ACCOUNT. Required.')
    s.add_argument('--ops-root', type=Path, default=HERE.parent.parent, help='OPS checkout (default: the one containing this script)')
    s.add_argument('--output-root', type=Path, help='parent of the campaign directory (default /scratch/<account>/ops-cloverleaf-decomp/campaigns)')
    s.add_argument('--campaign', type=Path, help='explicit campaign directory (must not exist yet)')
    s.add_argument('--date-stamp', default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    s.add_argument('--env-file', type=Path, default=HERE / 'environment.sh', help='shell file sourced before every build and job')
    s.add_argument('--translator-python', type=Path, help='Python >= 3.8 with ops_translator/requirements.txt; default: a venv is created at build time')
    s.add_argument('--partition', default='standard-g')
    s.add_argument('--smoke-partition', default='standard-g',
                   help='partition of the one-node qualification job (it uses a whole node, so standard-g)')
    s.add_argument('--build-partition', default='small-g')
    s.add_argument('--cpu-bind', default=R.CPU_BIND, help='cores for srun --cpu-bind=map_cpu (default %(default)s)')
    s.add_argument('--consistency-tolerance', type=float, default=1e-3,
                   help='relative tolerance of the cross-decomposition check of the final kinetic and internal energy and '
                        'simulated time (default 1e-3: the energies are printed with 4 significant digits); reported only')
    s.set_defaults(func=command_prepare)

    s = sub.add_parser('build', help='build the executable of the campaign (OPS C MPI+HIP library and cloverleaf_mpi_hip)')
    s.add_argument('--campaign', type=Path)
    s.add_argument('--submit', action='store_true', help='build in a Slurm job instead of this shell')
    s.add_argument('--dry-run', action='store_true', help='with the Slurm job: write the script, print sbatch, do not submit')
    s.add_argument('--time', default='02:00:00')
    s.add_argument('--cpus', type=int, default=16)
    s.add_argument('--print-toolchain', action='store_true', help='list every build variable and its resolved value')
    s.set_defaults(func=command_build)

    s = sub.add_parser('smoke', help='qualification stage Q1-Q5 (list it, or --submit the two jobs)')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--dry-run', action='store_true', help='list the Q1-Q5 runs (default when --submit is absent)')
    s.add_argument('--submit', action='store_true')
    s.add_argument('--stage', default='all', choices=('small', 'full', 'all'), help='small = Q1-Q3 (1 node), full = Q4-Q5')
    s.add_argument('--time-small', default='01:00:00')
    s.add_argument('--time-full', default='02:00:00')
    s.add_argument('--retry-failed', action='store_true', help='rerun qualification runs that failed (old attempts are kept)')
    s.add_argument('--no-time-guard', action='store_true', help='do not check the remaining allocation time before each run')
    s.add_argument('--smoke-partition', help='partition of the one-node job for this submission only '
                   '(default: the one stored in the campaign). Scheduling only: the runs are the same')
    s.set_defaults(func=command_smoke)

    s = sub.add_parser('gonogo', help='evaluate the qualification results, print the time per step; exit status 6 means NO-GO')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--steps', type=int, help='time steps per sweep run for the projection (default: the frozen plan, '
                   'else the campaign value)')
    s.add_argument('--target-seconds', type=float, default=300.0,
                   help='duration of the time-step loop the suggested --steps aims at (default 300)')
    s.add_argument('--gpu-hours-per-node-hour', type=float, help='billing rule of the project (check with lumi-allocations)')
    s.add_argument('--remaining-gpu-hours', type=float, help='GPU-hours left in the project')
    s.add_argument('--disk-available-gb', type=float, help='free quota where the campaign lives')
    s.add_argument('--launch-overhead-s', type=float, default=30.0, help='added per run to the projection (default 30)')
    s.add_argument('--extra-allocations', type=int, default=3, help='expected extra allocations (one sentinel each, default 3)')
    s.set_defaults(func=command_gonogo)

    s = sub.add_parser('submit', help='submit the sweep (first time) or continue it (--resume)')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--resume', action='store_true', help='continue: valid runs are skipped, the stored plan is reused')
    s.add_argument('--retry-failed', action='store_true', help='also rerun failed runs and clear an abort (old attempts are kept)')
    s.add_argument('--jobs', type=int, default=1, help='number of chained allocations to submit (default 1)')
    s.add_argument('--time', default='24:00:00', help='wall time of each allocation (default 24:00:00)')
    s.add_argument('--est-autotune', type=int, help='seconds per autotune run for the time guard')
    s.add_argument('--est-default', type=int, help='seconds per default run for the time guard')
    s.add_argument('--est-precomputed', type=int, help='seconds per precomputed run for the time guard')
    s.add_argument('--no-time-guard', action='store_true')
    s.add_argument('--override-gonogo', metavar='REASON', help='submit although the go/no-go verdict is not GO (recorded)')
    s.add_argument('--dry-run', action='store_true', help='write the job script and print sbatch without submitting')
    s.set_defaults(func=command_submit)

    s = sub.add_parser('status', help='progress of build, qualification and sweep')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--events', type=int, default=15)
    s.set_defaults(func=command_status)

    s = sub.add_parser('collect', help='write results/runs.csv and results/decompositions.csv')
    s.add_argument('--campaign', type=Path, required=True)
    s.set_defaults(func=command_collect)

    s = sub.add_parser('upgrade', help='replace the frozen launcher of a campaign whose sweep has not started')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--repetitions', type=int, help='new default/precomputed pairs per decomposition (only before the '
                   'sweep plan is frozen, or with --restart-sweep)')
    s.add_argument('--decompositions', metavar='default|all|PxxPyxPz,...', help='new decomposition set (only before the '
                   'sweep plan is frozen, or with --restart-sweep)')
    s.add_argument('--restart-sweep', action='store_true', help='move the frozen sweep plan, sweep runs, sweep dataset '
                   'and sweep job records into $C/archive/sweep_<utc>/ and start the sweep again (nothing is deleted; '
                   'qualification and the executable are kept)')
    s.add_argument('--ops-root', type=Path, default=HERE.parent.parent, help='OPS checkout of this package (recorded)')
    s.set_defaults(func=command_upgrade)

    s = sub.add_parser('job')     # internal: body of a Slurm job
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--kind', required=True, choices=('build', 'smoke', 'sweep'))
    s.add_argument('--builds', default='')
    s.add_argument('--retry-failed', action='store_true')
    s.add_argument('--no-time-guard', action='store_true')
    s.add_argument('--estimate', action='append')
    s.set_defaults(func=command_job)

    s = sub.add_parser('rank')    # internal: one MPI rank under srun
    s.add_argument('--spec', type=Path, required=True)
    s.set_defaults(func=command_rank)
    return p


def main(argv=None):
    require(sys.version_info >= (3, 6), 'Python 3.6 or newer is required')
    p = parser()
    args = p.parse_args(argv)
    if not getattr(args, 'func', None):
        p.print_help()
        return EXIT_OK
    return args.func(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except StopJob as stop:
        print(stop, file=sys.stderr)
        sys.exit(stop.status)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print('ERROR: %s' % error, file=sys.stderr)
        sys.exit(EXIT_ERROR)
