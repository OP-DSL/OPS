#!/usr/bin/env python3
"""SENGA2 MPI-decomposition sweep on LUMI-G: default, autotuned and replayed GPU block sizes.

Typical order (every command is described in README.md):
  plan (dry run) -> prepare -> build -> smoke -> gonogo -> plan --campaign -> submit [--resume] -> collect
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
PACKAGE_FILES = ('run.sh', 'launcher.py', 'common.py', 'plan.py', 'sources.py', 'runtime.py', 'engine.py', 'collect.py')
NAME = re.compile(r'[A-Za-z0-9_.-]+')
TIME = re.compile(r'(?:\d+-)?\d+:\d{2}:\d{2}')


def plan_config(args):
    return dict(ranks=args.nodes * 8, nodes=args.nodes, grid=args.grid, steps=args.steps,
                repetitions=args.repetitions, seed=args.seed, fixed_order=args.fixed_order,
                sentinel_every=args.sentinel_every, small_grid=args.small_grid, smoke_steps=args.smoke_steps)


def add_plan_options(p):
    p.add_argument('--nodes', type=int, default=20, help='LUMI-G nodes; 8 MPI ranks (one per GCD) each (default 20)')
    p.add_argument('--grid', type=int, default=1440, help='cubic global grid N of the sweep (default 1440)')
    p.add_argument('--small-grid', type=int, default=256, help='cubic grid of the qualification build (default 256)')
    p.add_argument('--steps', type=int, default=100, help='time steps per run (default 100)')
    p.add_argument('--smoke-steps', type=int, default=10, help='time steps of the Q4 memory checks (default 10)')
    p.add_argument('--repetitions', type=int, default=3, help='default/precomputed pairs per decomposition (default 3)')
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
        risky = [d['decomposition'] for d in plan['decompositions'] if d['halo60_risk']]
        print('  %d decompositions (%d skipped as invalid, %d excluded); %d flagged halo60_risk%s'
              % (len(plan['decompositions']), len(plan['skipped']), len(plan['excluded']), len(risky),
                 (': ' + ' '.join(risky)) if risky else ''))
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
    if args.campaign:
        c = E.Campaign(args.campaign)
        if (c.root / 'plan.json').exists():
            plan, digest = c.plan('sweep')
            print('The sweep plan of this campaign is already frozen and is never regenerated:')
            print_plan(plan, digest, c.root)
            return EXIT_OK
        cfg = dict(c.config['plan'], exclude=args.exclude, exclude_halo60_risk=args.exclude_halo60_risk)
        go = read_json_or(c.root / 'gonogo.json') or {}
        require(not go.get('required_plan_filter') or args.exclude_halo60_risk,
                'The go/no-go evaluation excluded the halo60-risk triples (they failed in qualification): freeze the '
                'plan with --exclude-halo60-risk.')
        plan = P.make_sweep_plan(cfg)
        digest = write_plan(c.root, plan)
        print_plan(plan, digest, c.root)
        print('Plan frozen in the campaign. Resume always reads this file.')
        return EXIT_OK
    require(args.output_dir, 'Give --output-dir DIR for a dry run, or --campaign DIR to freeze the plan of a campaign')
    cfg = dict(plan_config(args), exclude=args.exclude, exclude_halo60_risk=args.exclude_halo60_risk)
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
    require(args.hdf5 in ('strip', 'keep'), '--hdf5 must be strip or keep')
    cfg = plan_config(args)
    P.make_sweep_plan(dict(cfg, exclude=[], exclude_halo60_risk=False))   # parameter check before anything is written
    smoke = P.make_smoke_plan(cfg)
    if args.campaign:
        campaign = args.campaign.expanduser().resolve()
    else:
        root = args.output_root or Path('/scratch') / account / 'ops-senga2-decomp' / 'campaigns'
        campaign = root.expanduser().resolve() / ('SENGA2_DECOMP_' + args.date_stamp)
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
        print('Snapshot of OPS sources from %s' % ops_root, flush=True)
        ops = S.snapshot_ops(ops_root, campaign)
        senga = S.fetch_senga(campaign, args.senga_root, args.senga_repo, args.senga_commit)
        atomic(campaign / 'source_snapshot.json', dict(ops=ops, senga=senga, created_utc=now()))
        package = package_identity(ops_root)
        config = dict(
            version=1, campaign_id=campaign.name, created_utc=now(), account=account, partition=args.partition,
            smoke_partition=args.smoke_partition, build_partition=args.build_partition, hdf5=args.hdf5,
            plan=cfg, seed=cfg['seed'], nodes=cfg['nodes'], ranks=cfg['ranks'], grid=cfg['grid'], steps=cfg['steps'],
            repetitions=cfg['repetitions'], sentinel_every=cfg['sentinel_every'],
            builds=dict(small=dict(grid=cfg['small_grid']), full=dict(grid=cfg['grid'])),
            fingerprint_tolerance=args.fingerprint_tolerance, cpu_bind=args.cpu_bind, timeout_factor=3.0,
            time_guard_margin=1.2, translator_python=str(args.translator_python.expanduser().absolute()) if args.translator_python else '',
            ops_root=str(ops_root), ops_commit=ops['commit'], ops_uncommitted_changes=ops['uncommitted_changes_in_snapshot'],
            senga_origin=senga['origin'], senga_commit=senga['commit'] or 'unknown (not a Git checkout)',
            senga_requested_commit=args.senga_commit, senga_modified_files=senga['modified_files'], package=package,
            environment_sha256=sha(campaign / 'environment.sh'))
        atomic(campaign / 'campaign.json', config)
        digest = write_plan(campaign, smoke, 'smoke_plan')
    print('OPS commit %s%s' % (ops['commit'], ' (with uncommitted changes, saved as source/ops_local_changes.patch)'
                               if ops['uncommitted_changes_in_snapshot'] else ''))
    print('SENGA2 from %s, commit %s%s' % (senga['origin'], config['senga_commit'],
                                            (', %d files modified locally' % len(senga['modified_files'])) if senga['modified_files'] else ''))
    print('Qualification plan frozen (sha256 %s)' % digest)
    print('Campaign: %s' % campaign)
    print('Next: bash %s build --campaign %s --build all' % (campaign / 'launcher/run.sh', campaign))
    return EXIT_OK


# --------------------------------------------------------------------------- Slurm helpers
def resource_args(c, partition, nodes, walltime):
    return ['--account=' + c.config['account'], '--partition=' + partition, '--nodes=%d' % nodes,
            '--ntasks-per-node=8', '--gpus-per-node=8', '--cpus-per-task=7', '--mem=0', '--time=' + walltime, '--exclusive']


def job_script(c, walltime, job_args):
    command = ['bash', str(c.root / 'launcher/run.sh'), 'job', '--campaign', str(c.root)] + job_args
    return ('#!/usr/bin/env bash\n'
            '# Written by lumi_senga2_decomp; reproduces one job of campaign %s\n'
            'set -o pipefail\n'
            'export SENGA_JOB_START_EPOCH=$(date +%%s)\n'
            'export SENGA_JOB_TIME_LIMIT_S=%d\n'
            '# The environment file is sourced before "set -u": module functions may read unset variables.\n'
            'if ! source %s; then\n'
            '  echo "ERROR: the environment file failed: %s" >&2\n'
            '  exit 2\n'
            'fi\n'
            'set -u\n'
            '%s\n'
            'status=$?\n'
            'echo "lumi_senga2_decomp job finished with exit status $status (see README.md, Exit statuses)"\n'
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
        hdf5 = E.Campaign(args.campaign).config['hdf5'] if args.campaign else 'strip'
        return S.print_toolchain(dict(os.environ), hdf5)
    require(args.campaign, '--campaign is required')
    c = E.Campaign(args.campaign)
    builds = ['small', 'full'] if args.build == 'all' else [args.build]
    if args.submit or args.dry_run:
        text = job_script(c, check_time(args.time), ['--kind', 'build', '--builds', ','.join(builds)])
        command = ['sbatch', '--parsable', '--account=' + c.config['account'], '--partition=' + c.config['build_partition'],
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=%d' % args.cpus, '--gpus-per-node=1', '--time=' + args.time,
                   '--job-name=senga2_build', '--output=%s/slurm/build-%%j.out' % c.root]
        sbatch(c, 'build', command, text, 'build.sh', args.dry_run)
        return EXIT_OK
    # Local build (login node or interactive allocation): source the frozen environment first.
    command = ['bash', '-c', 'source "$1" || { echo "ERROR: environment file failed: $1" >&2; exit 2; }; shift; exec "$@"',
               'senga2-build', str(c.root / 'environment.sh'), 'bash', str(c.root / 'launcher/run.sh'), 'job',
               '--campaign', str(c.root), '--kind', 'build', '--builds', ','.join(builds)]
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
    for name in stages:
        c.verify_executable(name)
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
        command += ['--job-name=senga2_q_' + name, '--output=%s/slurm/smoke-%s-%%j.out' % (c.root, name)]
        previous = sbatch(c, 'smoke-' + name, command, job_script(c, walltime, job_args), 'smoke_%s.sh' % name, args.dry_run)
    print('When the jobs have ended: bash %s gonogo --campaign %s ...' % (c.root / 'launcher/run.sh', c.root))
    return EXIT_OK


def command_submit(args):
    c = E.Campaign(args.campaign)
    plan, digest = c.plan('sweep')
    c.verify_executable('full')
    go = read_json_or(c.root / 'gonogo.json')
    if args.override_gonogo:
        atomic(c.root / 'gonogo_override.json', dict(reason=args.override_gonogo, utc=now(), gonogo=go))
        print('WARNING: go/no-go overridden by the user: %s' % args.override_gonogo)
    else:
        require(go is not None and go['verdict'] == 'GO',
                'The go/no-go verdict is %s. Run: bash %s gonogo --campaign %s ... (or, knowingly, pass '
                '--override-gonogo "reason", which is recorded).' % (go['verdict'] if go else 'missing',
                                                                    c.root / 'launcher/run.sh', c.root))
        excluded = {e['decomposition'] for e in plan['excluded']}
        risky = {P.shape_text(t) for t in P.enumerate_decompositions(plan['ranks'], plan['grid'])[0]
                 if P.halo60_risk(t, plan['grid'])}
        require(not go.get('required_plan_filter') or risky <= excluded,
                'The go/no-go evaluation requires a plan without the halo60-risk triples, but the frozen plan contains them')
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
        command += ['--job-name=senga2_sweep', '--output=%s/slurm/sweep-%%j.out' % c.root]
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
        for name in builds:
            S.Builder(c.root, name).build()
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
    return C.gonogo(E.Campaign(args.campaign), args.gpu_hours_per_node_hour, args.remaining_gpu_hours,
                    args.disk_available_gb, args.exclude_halo60_risk, args.launch_overhead_s, args.extra_allocations)


def command_collect(args):
    C.collect(E.Campaign(args.campaign))
    return EXIT_OK


def command_status(args):
    c = E.Campaign(args.campaign)
    print('Campaign %s  (account %s, %d nodes, %d^3, %d steps, HDF5 %s)'
          % (c.id, c.config['account'], c.config['nodes'], c.config['grid'], c.config['steps'], c.config['hdf5']))
    for name in ('small', 'full'):
        manifest = read_json_or(c.root / 'build' / name / 'build_manifest.json')
        print('  build %-5s %s' % (name, ('sha256 ' + manifest['sha256']) if manifest else 'not built'))
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
        print('  %s plan %s: %s' % (kind, digest[:12], ', '.join('%d %s' % (v, k) for k, v in sorted(counts.items()))))
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


def command_rank(args):
    return R.rank_worker(args.spec)


def parser():
    p = argparse.ArgumentParser(prog='run.sh', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', metavar='COMMAND')

    s = sub.add_parser('plan', help='enumerate decompositions and write the ordered run list (dry run, no Slurm)')
    add_plan_options(s)
    s.add_argument('--output-dir', type=Path, help='write plan.json / plan.csv / smoke_plan.* here (dry run)')
    s.add_argument('--campaign', type=Path, help='freeze the sweep plan inside this campaign (uses its parameters)')
    s.add_argument('--exclude', action='append', default=[], metavar='PxxPyxPz', help='leave one decomposition out (repeatable)')
    s.add_argument('--exclude-halo60-risk', action='store_true', help='leave out the triples whose local y or z width is below 60')
    s.set_defaults(func=command_plan)

    s = sub.add_parser('prepare', help='create a campaign: frozen launcher, environment, OPS snapshot and SENGA2 sources')
    add_plan_options(s)
    s.add_argument('--account', default='', help='Slurm account (LUMI project); default: $SLURM_ACCOUNT. Required.')
    s.add_argument('--ops-root', type=Path, default=HERE.parent.parent, help='OPS checkout (default: the one containing this script)')
    s.add_argument('--output-root', type=Path, help='parent of the campaign directory (default /scratch/<account>/ops-senga2-decomp/campaigns)')
    s.add_argument('--campaign', type=Path, help='explicit campaign directory (must not exist yet)')
    s.add_argument('--date-stamp', default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    s.add_argument('--env-file', type=Path, default=HERE / 'environment.sh', help='shell file sourced before every build and job')
    s.add_argument('--senga-root', type=Path, help='existing SENGA2 checkout to copy (never modified)')
    s.add_argument('--senga-repo', default=S.SENGA_REPO, help='repository cloned when --senga-root is not given (default %(default)s)')
    s.add_argument('--senga-commit', default=S.SENGA_COMMIT, help='pinned SENGA2 commit (default %(default)s)')
    s.add_argument('--hdf5', default='strip', choices=('strip', 'keep'), help='strip: build without HDF5 (default); keep: link parallel HDF5')
    s.add_argument('--translator-python', type=Path, help='Python >= 3.8 with ops_translator/requirements.txt; default: a venv is created at build time')
    s.add_argument('--partition', default='standard-g')
    s.add_argument('--smoke-partition', default='standard-g',
                   help='partition of the one-node qualification job (it uses a whole node, so standard-g)')
    s.add_argument('--build-partition', default='small-g')
    s.add_argument('--cpu-bind', default=R.CPU_BIND, help='cores for srun --cpu-bind=map_cpu (default %(default)s)')
    s.add_argument('--fingerprint-tolerance', type=float, default=1e-6, help='relative tolerance on final tke/enstrophy/tmax')
    s.set_defaults(func=command_prepare)

    s = sub.add_parser('build', help='build the qualification (small) and sweep (full) executables in the campaign')
    s.add_argument('--campaign', type=Path)
    s.add_argument('--build', default='all', choices=('small', 'full', 'all'))
    s.add_argument('--submit', action='store_true', help='build in a Slurm job instead of this shell')
    s.add_argument('--dry-run', action='store_true', help='with the Slurm job: write the script, print sbatch, do not submit')
    s.add_argument('--time', default='03:00:00')
    s.add_argument('--cpus', type=int, default=16)
    s.add_argument('--print-toolchain', action='store_true', help='list every build variable and its resolved value')
    s.set_defaults(func=command_build)

    s = sub.add_parser('smoke', help='qualification stage Q1-Q5 (list it, or --submit the two jobs)')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--dry-run', action='store_true', help='list the Q1-Q5 runs (default when --submit is absent)')
    s.add_argument('--submit', action='store_true')
    s.add_argument('--stage', default='all', choices=('small', 'full', 'all'), help='small = Q1-Q3 (1 node), full = Q4-Q5')
    s.add_argument('--time-small', default='02:00:00')
    s.add_argument('--time-full', default='03:00:00')
    s.add_argument('--retry-failed', action='store_true', help='rerun qualification runs that failed (old attempts are kept)')
    s.add_argument('--no-time-guard', action='store_true', help='do not check the remaining allocation time before each run')
    s.add_argument('--smoke-partition', help='partition of the one-node job for this submission only '
                   '(default: the one stored in the campaign). Scheduling only: the runs are the same')
    s.set_defaults(func=command_smoke)

    s = sub.add_parser('gonogo', help='evaluate the qualification results; exit status 6 means NO-GO')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--gpu-hours-per-node-hour', type=float, help='billing rule of the project (check with lumi-allocations)')
    s.add_argument('--remaining-gpu-hours', type=float, help='GPU-hours left in the project')
    s.add_argument('--disk-available-gb', type=float, help='free quota where the campaign lives')
    s.add_argument('--exclude-halo60-risk', action='store_true', help='accept failed thin triples by excluding all halo60-risk triples')
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

    s = sub.add_parser('status', help='progress of builds, qualification and sweep')
    s.add_argument('--campaign', type=Path, required=True)
    s.add_argument('--events', type=int, default=15)
    s.set_defaults(func=command_status)

    s = sub.add_parser('collect', help='write results/runs.csv and results/decompositions.csv')
    s.add_argument('--campaign', type=Path, required=True)
    s.set_defaults(func=command_collect)

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
