"""Result tables and the go/no-go evaluation. Tables only: no ranking, no "best" selection,
no imputation. Missing values stay empty and incomplete decompositions are listed."""
import os
from pathlib import Path
import statistics

import plan as P
import runtime as R
from common import (atomic, read_json_or, now, shape_text, write_csv, sha, EXIT_OK, EXIT_NO_GO)

RUN_FIELDS = ['scope', 'stage', 'run_id', 'attempt', 'plan_position', 'decomp_position', 'decomposition',
              'decomposition_planned', 'mode', 'mode_name', 'repetition', 'pair_index', 'order_in_pair', 'build',
              'grid', 'nodes', 'nranks', 'steps', 'gpudirect', 'verdict', 'category', 'superseded',
              'superseded_reason', 'used_in_statistics', 'wall_time_s', 'kernel_time_s', 'halo_time_s',
              'user_halo_time_s', 'wall_time_minus_kernel_time_s', 'launcher_elapsed_s', 'printed_grid',
              'mpi_comm_volume_mb', 'catalog_rows', 'catalog_sha256', 'field_summary_step', 'kinetic_energy',
              'internal_energy', 'total_energy', 'sim_time', 'consistency_reference', 'consistency_rel_diff',
              'consistency_ok', 'slurm_job_id', 'slurm_step_id', 'node_list', 'start_utc', 'end_utc',
              'executable_sha256', 'plan_hash', 'seed', 'reason', 'case_id', 'committed', 'run_dir', 'record_path']
ESTIMATORS = {
    'T(d,m,i)': '"Total Wall time" printed by the root rank of run i of mode m (D default, A autotune, P precomputed); '
                'the time-step loop only',
    'X_median/min/max_wall_time_s': 'median, minimum and maximum of T over the repetitions of mode X; given only '
                                      'when every planned repetition of that mode is valid; no confidence interval',
    'S_pair<i>': 'paired ratio T(d,D,i) / T(d,P,i) of the i-th interleaved pair (both valid, same Slurm job)',
    'S_median_of_paired_ratios': 'median over i of S_pair<i>; given only for complete decompositions',
    'S_ratio_of_medians': 'D_median_wall_time_s / P_median_wall_time_s; a different estimator, tabulated beside it',
    'A_wall_time_s': 'single autotune run; never used as a D or P sample',
    'search_cost_A_over_D_median': 'A_wall_time_s / D_median_wall_time_s; one sample, no uncertainty',
    'catalog_default_time_share': 'share of the per-kernel time of the first valid D run that belongs to kernels in '
                                  'the catalog; flagged below 0.90',
    'sentinel relative half-range': '(max - min) / (2 * median) of the default-block runs of forced d0',
    'complete': 'one valid autotune run with a frozen catalog, and all D and P repetitions valid in same-job pairs',
    'consistency_rel_diff': 'largest relative difference of the final kinetic energy, internal energy (clover.out, '
                            'printed with 4 significant digits) and simulated time (stdout, 7 decimals) from the first '
                            'valid unforced baseline run; reported, never a reason to exclude a run',
}


def median(values):
    return statistics.median(values) if values else None


def stats(values, expected):
    """median/min/max, only when all expected repetitions are present."""
    if len(values) != expected or not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def run_row(record, used, consistency):
    f = record.get('field_summary') or {}
    t, k = record.get('wall_time_s'), record.get('kernel_time_s')
    reference, diff, ok = consistency.get(record['record_path'], (None, None, None))
    row = dict(record, used_in_statistics=used, field_summary_step=f.get('step'), kinetic_energy=f.get('kinetic_energy'),
               internal_energy=f.get('internal_energy'), total_energy=f.get('total_energy'), sim_time=f.get('sim_time'),
               consistency_reference=reference, consistency_rel_diff=diff, consistency_ok=ok,
               printed_grid=shape_text(record['printed_grid']) if record.get('printed_grid') else '',
               wall_time_minus_kernel_time_s=(t - k) if t is not None and k is not None else None)
    return row


def consistency_against(reference, records, tolerance):
    """{record_path: (reference run id, relative difference, within tolerance)} for records with a summary."""
    out = {}
    if reference is None:
        return out
    for r in records:
        diff = R.relative_difference(r.get('field_summary'), reference.get('field_summary'))
        if diff is not None:
            out[r['record_path']] = (reference['run_id'], diff, diff <= tolerance)
    return out


def collect(campaign):
    c = campaign
    records = c.all_records()
    plan = read_json_or(c.root / 'plan.json')
    out = c.root / 'results'
    out.mkdir(exist_ok=True)
    # Sweep records count only when committed to the final dataset tree (commit rule satisfied).
    live = [r for r in records if not r['superseded'] and r['verdict'] == 'VALID' and r['committed']]
    sweep = [r for r in live if r['scope'] == 'sweep']
    used_ids = set()
    dec_rows, incomplete = [], []
    d0 = (read_json_or(c.root / 'state/d0.json') or {}).get('d0')
    reps = plan['repetitions'] if plan else c.config['repetitions']
    tolerance = c.config['consistency_tolerance']
    baselines = sorted((r for r in sweep if r['stage'] == 'baseline' and r.get('field_summary')),
                       key=lambda r: (r['repetition'], r['start_utc']))
    reference = baselines[0] if baselines else None
    consistency = consistency_against(reference, sweep, tolerance)
    for dec in (plan['decompositions'] if plan else []):
        text = dec['decomposition']
        mine = [r for r in sweep if r['stage'] == 'sweep' and r['decomposition'] == text]
        everything = [r for r in records if r['scope'] == 'sweep' and r['stage'] == 'sweep'
                      and r['decomposition'] == text and not r['superseded']]
        a = [r for r in mine if r['mode'] == 'A' and r.get('autotune', {}).get('status') == 'FROZEN']
        d = sorted((r for r in mine if r['mode'] == 'D'), key=lambda r: r['pair_index'])
        p = sorted((r for r in mine if r['mode'] == 'P'), key=lambda r: r['pair_index'])
        pairs = {}
        for rd in d:
            for rp in p:
                if rd['pair_index'] == rp['pair_index'] and rd['slurm_job_id'] == rp['slurm_job_id']:
                    pairs[rd['pair_index']] = rd['wall_time_s'] / rp['wall_time_s']
        marker = read_json_or(c.root / 'state/decompositions' / (text + '.json'))
        complete = bool(a) and len(d) == reps and len(p) == reps and len(pairs) == reps
        failed = [r for r in everything if r['verdict'] != 'VALID' or
                  (r['mode'] == 'A' and r.get('autotune', {}).get('status') != 'FROZEN')]
        if marker:
            status = marker['status']
        elif complete:
            status = 'COMPLETE'
        elif any(r['mode'] == 'A' for r in failed):
            status = 'AUTOTUNE_FAILED'
        elif failed:
            status = 'INCOMPLETE'
        else:
            status = 'PENDING'
        dmed, dmin, dmax = stats([r['wall_time_s'] for r in d], reps)
        pmed, pmin, pmax = stats([r['wall_time_s'] for r in p], reps)
        manifest = read_json_or(c.catalog_folder('sweep', 'full', text) / 'manifest.json') or {}
        diffs = [consistency[r['record_path']][1] for r in a + d + p if r['record_path'] in consistency]
        share = None
        if d and manifest.get('kernel_ids') and d[0].get('kernel_times'):
            times = {int(k): v for k, v in d[0]['kernel_times'].items()}
            total = sum(times.values())
            share = sum(v for k, v in times.items() if k in set(manifest['kernel_ids'])) / total if total > 0 else None
        row = dict(
            decomposition=text, plan_position=dec['position'], px=dec['triple'][0], py=dec['triple'][1],
            pz=dec['triple'][2], local_nx=dec['local_size'][0], local_ny=dec['local_size'][1],
            local_nz=dec['local_size'][2], permutation_class=dec['permutation_class'],
            first_pair_mode=dec['first_pair_mode'],
            is_d0=bool(d0) and dec['triple'] == d0, status=status, complete=complete,
            n_valid_A=len(a), n_valid_D=len(d), n_valid_P=len(p), n_valid_pairs=len(pairs),
            A_wall_time_s=a[0]['wall_time_s'] if a else None,
            D_median_wall_time_s=dmed, D_min_wall_time_s=dmin, D_max_wall_time_s=dmax,
            P_median_wall_time_s=pmed, P_min_wall_time_s=pmin, P_max_wall_time_s=pmax,
            S_median_of_paired_ratios=median(list(pairs.values())) if complete else None,
            S_ratio_of_medians=(dmed / pmed) if complete else None,
            search_cost_A_over_D_median=(a[0]['wall_time_s'] / dmed) if a and dmed else None,
            D_median_kernel_time_s=stats([r['kernel_time_s'] for r in d], reps)[0],
            P_median_kernel_time_s=stats([r['kernel_time_s'] for r in p], reps)[0],
            D_median_halo_time_s=stats([r['halo_time_s'] for r in d], reps)[0],
            P_median_halo_time_s=stats([r['halo_time_s'] for r in p], reps)[0],
            D_median_user_halo_time_s=stats([r['user_halo_time_s'] for r in d], reps)[0],
            P_median_user_halo_time_s=stats([r['user_halo_time_s'] for r in p], reps)[0],
            catalog_rows=manifest.get('rows'), executed_kernels=len(manifest.get('executed_kernel_ids', [])) or None,
            kernel_coverage=manifest.get('kernel_coverage'), catalog_default_time_share=share,
            catalog_share_below_0p90=(share < 0.90) if share is not None else None,
            catalog_sha256=manifest.get('sha256'),
            max_consistency_rel_diff=max(diffs) if diffs else None,
            consistency_ok=(max(diffs) <= tolerance) if diffs else None,
            slurm_job_ids=' '.join(sorted({r['slurm_job_id'] for r in a + d + p})),
            n_node_lists=len({r['node_list'] for r in a + d + p}))
        for i in range(1, reps + 1):
            row['S_pair%d' % i] = pairs.get(i) if complete else None
        dec_rows.append(row)
        if complete:
            used_ids.update(r['record_path'] for r in a + d + p)
        else:
            missing = [r['run_id'] for r in (plan['runs'] if plan else [])
                       if r['stage'] == 'sweep' and r['decomposition'] == text
                       and r['run_id'] not in {x['run_id'] for x in a + d + p}]
            incomplete.append(dict(decomposition=text, plan_position=dec['position'], status=status,
                                   n_valid_A=len(a), n_valid_D=len(d), n_valid_P=len(p), n_valid_pairs=len(pairs),
                                   runs_without_valid_record=' '.join(missing),
                                   failure_reasons=' | '.join('%s: [%s] %s' % (r['run_id'], r['category'], r['reason'])
                                                              for r in failed)[:2000]))
    dec_fields = (['decomposition', 'plan_position', 'px', 'py', 'pz', 'local_nx', 'local_ny', 'local_nz',
                   'permutation_class', 'first_pair_mode', 'is_d0', 'status', 'complete', 'n_valid_A',
                   'n_valid_D', 'n_valid_P', 'n_valid_pairs', 'A_wall_time_s', 'D_median_wall_time_s',
                   'D_min_wall_time_s', 'D_max_wall_time_s', 'P_median_wall_time_s', 'P_min_wall_time_s',
                   'P_max_wall_time_s'] + ['S_pair%d' % i for i in range(1, reps + 1)] +
                  ['S_median_of_paired_ratios', 'S_ratio_of_medians', 'search_cost_A_over_D_median',
                   'D_median_kernel_time_s', 'P_median_kernel_time_s', 'D_median_halo_time_s', 'P_median_halo_time_s',
                   'D_median_user_halo_time_s', 'P_median_user_halo_time_s', 'catalog_rows', 'executed_kernels',
                   'kernel_coverage', 'catalog_default_time_share', 'catalog_share_below_0p90', 'catalog_sha256',
                   'max_consistency_rel_diff', 'consistency_ok', 'slurm_job_ids', 'n_node_lists'])
    write_csv(out / 'decompositions.csv', dec_rows, dec_fields)
    write_csv(out / 'incomplete_decompositions.csv', incomplete,
              ['decomposition', 'plan_position', 'status', 'n_valid_A', 'n_valid_D', 'n_valid_P', 'n_valid_pairs',
               'runs_without_valid_record', 'failure_reasons'])
    # sentinel and baseline runs: noise floor and the unforced/forced A/A check
    sentinels = sorted((r for r in sweep if r['stage'] == 'sentinel'), key=lambda r: r['start_utc'])
    by_job = {}
    for r in sentinels:
        by_job.setdefault(r['slurm_job_id'], []).append(r['wall_time_s'])

    def half_range(values):
        return (max(values) - min(values)) / (2 * statistics.median(values)) if len(values) > 1 else None

    noise = [dict(group='job ' + job, n=len(v), min_wall_time_s=min(v), median_wall_time_s=statistics.median(v),
                  max_wall_time_s=max(v), relative_half_range=half_range(v)) for job, v in sorted(by_job.items())]
    every = [r['wall_time_s'] for r in sentinels]
    if every:
        noise.append(dict(group='all allocations', n=len(every), min_wall_time_s=min(every),
                          median_wall_time_s=statistics.median(every), max_wall_time_s=max(every),
                          relative_half_range=half_range(every)))
    write_csv(out / 'sentinels.csv', noise, ['group', 'n', 'min_wall_time_s', 'median_wall_time_s',
                                             'max_wall_time_s', 'relative_half_range'])
    baseline = [r['wall_time_s'] for r in sweep if r['stage'] == 'baseline']
    forced = next((row for row in dec_rows if row['is_d0']), None)
    used_ids.update(r['record_path'] for r in sweep if r['stage'] in ('sentinel', 'baseline'))
    run_rows = [run_row(r, r['record_path'] in used_ids, consistency) for r in
                sorted(records, key=lambda r: (r['scope'], r['plan_position'], r['run_id'], r['attempt']))]
    write_csv(out / 'runs.csv', run_rows, RUN_FIELDS)
    package = c.package
    launcher_files = sorted(p.name for p in (c.root / 'launcher').iterdir() if p.is_file()) \
        if (c.root / 'launcher').is_dir() else []
    summary = dict(
        collected_utc=now(), campaign_id=c.id, plan_hash=(c.root / 'plan.json.sha256').read_text().split()[0] if plan else None,
        seed=c.config['seed'], d0=shape_text(d0) if d0 else None,
        unforced_baseline=dict(n=len(baseline), median_wall_time_s=median(baseline),
                               min_wall_time_s=min(baseline) if baseline else None,
                               max_wall_time_s=max(baseline) if baseline else None,
                               forced_d0_D_median_wall_time_s=forced and forced['D_median_wall_time_s']),
        consistency=dict(
            reference_run=reference['run_id'] if reference else None, tolerance=tolerance,
            quantities=list(R.CONSISTENCY_KEYS), checked=len(consistency),
            outside_tolerance=sorted(r['run_id'] for r in sweep if not consistency.get(r['record_path'], (0, 0, True))[2]),
            note='reported only: a run outside the tolerance is not excluded. The energies are printed with 4 '
                 'significant digits, so differences below about 1e-3 are not resolved by them.'),
        decompositions_planned=len(dec_rows), decompositions_complete=sum(1 for r in dec_rows if r['complete']),
        incomplete_decompositions=[r['decomposition'] for r in incomplete],
        grouping_key='decomposition (ordered Px x Py x Pz) within scope "sweep", stage "sweep"',
        estimators=ESTIMATORS,
        exclusions=dict(
            superseded_records=sum(1 for r in records if r['superseded']),
            invalid_records=sum(1 for r in records if not r['superseded'] and r['verdict'] != 'VALID'),
            qualification_records=sum(1 for r in records if r['scope'] != 'sweep'),
            uncommitted_sweep_records=sum(1 for r in records if r['scope'] == 'sweep' and not r['committed']),
            rule='statistics use only valid, non-superseded sweep records committed to the dataset tree; paired '
                 'statistics only complete decompositions; qualification runs never enter; nothing is imputed'),
        input_records=[dict(path=r['record_path'], sha256=sha(r['record_path']), verdict=r['verdict'],
                            superseded=r['superseded'], committed=r['committed']) for r in records],
        dataset=str(c.layout.dataset),
        code_version=dict(package_commit=package['commit'], package_dirty=package['dirty'],
                          launcher_sha256=package['files_sha256'],
                          launcher_sha256_now={n: sha(c.root / 'launcher' / n) for n in launcher_files},
                          launcher_upgrades=(read_json_or(c.root / 'launcher_upgrades.json') or {}).get('upgrades', [])),
        outputs=['runs.csv', 'decompositions.csv', 'incomplete_decompositions.csv', 'sentinels.csv'],
        note='Tables only. The minimum over many three-sample medians is optimistically biased (no confirmation '
             'stage was run); catalog stability is unmeasured (one tuning run per decomposition).')
    atomic(out / 'collection_manifest.json', summary)
    print('Collected %d run records (%d valid sweep records).' % (len(records), len(sweep)))
    print('Decompositions: %d planned, %d complete, %d incomplete (listed in incomplete_decompositions.csv, never imputed).'
          % (len(dec_rows), summary['decompositions_complete'], len(incomplete)))
    outside = summary['consistency']['outside_tolerance']
    if reference is None:
        print('Consistency: not evaluated (no valid unforced baseline run with a field summary yet).')
    else:
        print('Consistency: %d valid sweep runs compared with %s (tolerance %.1e): %d outside%s'
              % (len(consistency), reference['run_id'], tolerance, len(outside),
                 ' (reported only; see consistency_rel_diff in runs.csv)' if outside else ''))
    print('Results: %s' % out)
    return summary


# --------------------------------------------------------------------------- go / no-go
def tree_bytes(path):
    total = 0
    for folder, _, files in os.walk(str(path)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(folder, name))
            except OSError:
                pass
    return total


def time_per_step(q5, target_seconds):
    """Per-mode time per step and start-up overhead from the valid Q5 runs, and the step count whose
    D-mode time-step loop ("Total Wall time") lasts about target_seconds."""
    out = dict(per_step_s={}, overhead_s={}, elapsed_s={}, wall_time_s={})
    for mode in 'ADP':
        records = [rec for rid, (s, rec, _, run) in sorted(q5.items()) if run['mode'] == mode and s == 'VALID']
        if not records:
            continue
        steps = records[0]['steps']
        wall = statistics.mean(r['wall_time_s'] for r in records)
        elapsed = statistics.mean(r['launcher_elapsed_s'] for r in records)
        out['per_step_s'][mode] = wall / steps
        out['overhead_s'][mode] = max(0.0, elapsed - wall)
        out['elapsed_s'][mode] = elapsed
        out['wall_time_s'][mode] = wall
        out['q5_steps'] = steps
    per = out['per_step_s'].get('D')
    out['target_seconds'] = target_seconds
    out['suggested_steps'] = max(1, int(round(target_seconds / per))) if per else None
    return out


def gonogo(campaign, gpu_rate=None, remaining_gpu_hours=None, disk_available_gb=None, steps=None,
           target_seconds=300.0, launch_overhead_s=30.0, extra_allocations=3):
    c = campaign
    smoke, _ = c.plan('qualification')
    cfg = c.config
    checks, lines = [], []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        lines.append('%-4s %s: %s' % ('PASS' if ok else 'FAIL', name, detail))

    state = {}
    for run in smoke['runs']:
        s, record, attempt = c.state('qualification', run['run_id'])
        state[run['run_id']] = (s, record, attempt, run)
    # build preflight (Q0)
    manifest = read_json_or(c.root / 'build' / 'hip' / 'build_manifest.json')
    check('Q0 build', manifest is not None and manifest.get('gtl_linked') and manifest.get('generated_hip_kernels_with_hooks', 0) > 0,
          'missing: run.sh build --campaign' if manifest is None else
          'sha256 %s, %d kernels with hooks, libmpi_gtl_hsa linked: %s' % (
              manifest['sha256'][:12], manifest['generated_hip_kernels_with_hooks'], manifest['gtl_linked']))
    bad = [rid for rid, (s, _, _, _) in state.items() if s != 'VALID']
    details = []
    for rid in bad:
        s, record, _, _ = state[rid]
        details.append('%s %s%s' % (rid, s, (' [%s] %s' % (record['category'], record['reason'])) if record else ''))
    check('1 every qualification run valid', not bad,
          'all %d valid' % len(state) if not bad else '; '.join(details)[:1500])
    grids = ['%s -> %s' % (rid, shape_text(rec['printed_grid'])) for rid, (s, rec, _, run) in sorted(state.items())
             if s == 'VALID' and run['decomposition'] == 'unforced']
    forced_ok = all(rec['printed_grid'] == rec['triple'] for rid, (s, rec, _, run) in state.items()
                    if s == 'VALID' and run['decomposition'] != 'unforced')
    check('2 printed grid equals the forced triple; unforced grids recorded', forced_ok and len(grids) >= 2,
          'unforced: ' + (', '.join(grids) or 'no valid unforced run'))
    q4 = sorted(rid for rid, v in state.items() if v[3]['stage'] == 'Q4')
    check('3 full grid fits on every GCD (Q4 runs valid)', q4 and all(state[rid][0] == 'VALID' for rid in q4),
          ', '.join('%s %s' % (rid, state[rid][0]) for rid in q4) or 'no Q4 run planned')
    # Q5
    q5 = {rid: v for rid, v in state.items() if v[3]['stage'] == 'Q5'}
    a = [rec for rid, (s, rec, _, run) in q5.items() if run['mode'] == 'A' and s == 'VALID']
    d = [rec for rid, (s, rec, _, run) in sorted(q5.items()) if run['mode'] == 'D' and s == 'VALID']
    p = [rec for rid, (s, rec, _, run) in sorted(q5.items()) if run['mode'] == 'P' and s == 'VALID']
    spread = abs(d[0]['wall_time_s'] - d[1]['wall_time_s']) / min(d[0]['wall_time_s'], d[1]['wall_time_s']) if len(d) == 2 else None
    ok4 = bool(a) and a[0]['autotune']['catalog_rows'] > 0 and len(p) == 2 and spread is not None and spread < 0.05
    check('4 Q5 catalog non-empty, replay complete, the two default runs within 5 %', ok4,
          'catalog rows %s of %s executed kernels; valid P runs %d/2; D spread %s'
          % (a[0]['autotune']['catalog_rows'] if a else 'n/a', a[0]['autotune']['executed_kernels'] if a else 'n/a',
             len(p), ('%.2f %%' % (100 * spread)) if spread is not None else 'n/a'))
    # time per step and the step count of the sweep
    plan = read_json_or(c.root / 'plan.json')
    measured, projection = None, {}
    timing = time_per_step(q5, target_seconds)
    if steps is None:
        steps = plan['steps'] if plan else cfg['steps']
    if timing['per_step_s'].get('D'):
        per = timing['per_step_s']
        lines.append('     Time per step (Q5, %d^3 on %d ranks, %d steps): D %.4f s, P %.4f s, A %.4f s; start-up and '
                     'teardown about %.0f s per run' % (cfg['grid'], cfg['ranks'], timing['q5_steps'], per.get('D', 0),
                                                        per.get('P', 0), per.get('A', 0), timing['overhead_s'].get('D', 0)))
        lines.append('     Suggested --steps for a time-step loop of about %.0f s (default blocks): %d'
                     % (target_seconds, timing['suggested_steps']))
    # budget and disk projections from measured values, at the step count of the sweep
    if a and d and p:
        if plan:
            n_dec, reps = len(plan['decompositions']), plan['repetitions']
            others = sum(1 for r in plan['runs'] if r['stage'] in ('baseline', 'sentinel')) + extra_allocations
        else:
            n_dec = len(P.sweep_triples(dict(cfg['plan'], exclude=[]))[0])
            reps = cfg['repetitions']
            others = reps + 1 + (n_dec - 1) // cfg['sentinel_every'] + extra_allocations
        t = {m: timing['overhead_s'][m] + timing['per_step_s'][m] * steps for m in 'ADP'}
        measured = dict(elapsed_s=timing['elapsed_s'], per_step_s=timing['per_step_s'], overhead_s=timing['overhead_s'],
                        wall_time_s=timing['wall_time_s'], q5_steps=timing['q5_steps'],
                        suggested_steps=timing['suggested_steps'], target_seconds=target_seconds,
                        source='Q5 runs of this campaign', utc=now())
        n_runs = n_dec * (1 + 2 * reps) + others
        seconds = n_dec * (t['A'] + reps * t['D'] + reps * t['P']) + others * t['D'] + n_runs * launch_overhead_s
        node_hours = seconds / 3600.0 * cfg['nodes']
        sizes = {}
        for mode in 'ADP':
            first = min(rid for rid, (s, rec, _, run) in q5.items() if run['mode'] == mode and s == 'VALID')
            # + the rank logs once more: a sweep run also keeps run.log (all rank logs concatenated)
            sizes[mode] = tree_bytes(q5[first][2]) + sum(f.stat().st_size for f in (q5[first][2] / 'ranks').glob('rank_*.log'))
        disk_gb = (n_dec * (sizes['A'] + reps * sizes['D'] + reps * sizes['P']) + others * sizes['D']) / 1e9
        projection = dict(decompositions=n_dec, runs=n_runs, steps=steps, seconds_per_run=t, wall_hours=seconds / 3600.0,
                          node_hours=node_hours, launch_overhead_s=launch_overhead_s, disk_gb=disk_gb, bytes_per_run=sizes,
                          gpu_hours=node_hours * gpu_rate if gpu_rate else None)
        lines.append('     Measured elapsed per run (s, %d steps): A %.0f, D %.0f, P %.0f'
                     % (timing['q5_steps'], timing['elapsed_s']['A'], timing['elapsed_s']['D'], timing['elapsed_s']['P']))
        lines.append('     Projection: %d decompositions, %d runs of %d steps, %.1f h wall time on %d nodes = %.0f node-hours'
                     % (n_dec, n_runs, steps, seconds / 3600.0, cfg['nodes'], node_hours))
        if gpu_rate and remaining_gpu_hours is not None:
            check('5 budget: projection at most 80 % of the remaining allocation',
                  projection['gpu_hours'] <= 0.8 * remaining_gpu_hours,
                  '%.0f GPU-hours projected (%.2f GPU-hours per node-hour) vs 80 %% of %.0f remaining = %.0f'
                  % (projection['gpu_hours'], gpu_rate, remaining_gpu_hours, 0.8 * remaining_gpu_hours))
        else:
            check('5 budget: projection at most 80 % of the remaining allocation', False,
                  'not evaluated: pass --gpu-hours-per-node-hour (billing rule of your project, see lumi-allocations) '
                  'and --remaining-gpu-hours')
        if disk_available_gb is not None:
            check('6 disk: projected log volume fits', disk_gb <= disk_available_gb,
                  '%.1f GB projected vs %.1f GB available' % (disk_gb, disk_available_gb))
        else:
            check('6 disk: projected log volume fits', False,
                  '%.1f GB projected; not evaluated: pass --disk-available-gb (free quota of the campaign file system)' % disk_gb)
    else:
        check('5 budget: projection at most 80 % of the remaining allocation', False, 'no measured Q5 times yet')
        check('6 disk: projected log volume fits', False, 'no measured Q5 run sizes yet')
    # field summaries of runs with the same grid and step count agree (first valid run is the reference)
    tolerance = cfg['consistency_tolerance']
    groups, worst = {}, []
    for rid, (s, rec, _, run) in sorted(state.items(), key=lambda item: item[1][3]['position']):
        if s == 'VALID' and rec.get('field_summary'):
            groups.setdefault((rec['grid'], rec['steps']), []).append(rec)
    for key, records in sorted(groups.items()):
        for rec in records[1:]:
            diff = R.relative_difference(rec['field_summary'], records[0]['field_summary'])
            if diff is not None:
                worst.append((diff, rec['run_id'], records[0]['run_id']))
    outside = [w for w in worst if w[0] > tolerance]
    check('7 field summaries agree across decompositions (tolerance %.1e)' % tolerance, worst and not outside,
          ('largest relative difference %.2e (%s vs %s)' % max(worst)) if worst and not outside else
          ('; '.join('%s differs from %s by %.2e' % (r, ref, diff) for diff, r, ref in outside)[:1500] if outside
           else 'nothing to compare'))
    go = all(item['passed'] for item in checks)
    result = dict(verdict='GO' if go else 'NO-GO', evaluated_utc=now(), checks=checks, projection=projection,
                  projected_steps=steps, time_per_step=timing,
                  inputs=dict(gpu_hours_per_node_hour=gpu_rate, remaining_gpu_hours=remaining_gpu_hours,
                              disk_available_gb=disk_available_gb, steps=steps, target_seconds=target_seconds))
    atomic(c.root / 'gonogo.json', result)
    if measured:
        atomic(c.root / 'measured_times.json', measured)
    print('\n'.join(lines))
    print('VERDICT: %s%s' % (result['verdict'], '' if go else '  (do not submit the sweep; exit status %d)' % EXIT_NO_GO))
    if timing.get('suggested_steps') and not plan:
        print('Next: freeze the sweep plan with the chosen step count, e.g. bash %s plan --campaign %s --steps %d'
              % (c.root / 'launcher/run.sh', c.root, timing['suggested_steps']))
    return EXIT_OK if go else EXIT_NO_GO
