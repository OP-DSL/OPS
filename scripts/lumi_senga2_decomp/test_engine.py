"""Job loops with a fake Slurm: failure policy, resume, time guard, baseline, qualification,
go/no-go and the collector (criteria 10, 11, 13-15, 17, 20, 21)."""
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import engine as E
import launcher as L
import plan as P
import runtime as R
import testlib as T
from common import read_json, atomic, sha


class Base(unittest.TestCase):
    grid, reps, sentinel_every, exclude = 20, 1, 4, ()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='senga_engine_')
        self.tmp = Path(self.temp.name)
        self.campaign = T.make_campaign(self.tmp, grid=self.grid, reps=self.reps, sentinel_every=self.sentinel_every)
        T.fake_build(self.campaign)
        argv = ['plan', '--campaign', str(self.campaign)]
        for item in self.exclude:
            argv += ['--exclude', item]
        T.quiet(L.main, argv)
        self.plan = read_json(self.campaign / 'plan.json')
        self.order = [d['decomposition'] for d in self.plan['decompositions']]

    def tearDown(self):
        self.temp.cleanup()

    def job(self, job='1001', kind='sweep', extra=(), remaining=10 ** 6, **settings):
        before = len(T.srun_calls(self.tmp))
        status, text = T.run_job(self.campaign, kind, T.job_env(self.tmp, job=job, remaining=remaining, **settings), *extra)
        return status, text, T.srun_calls(self.tmp)[before:]

    def record(self, run_id, attempt=1, scope='sweep'):
        return read_json(self.campaign / 'runs' / scope / run_id / ('attempt_%03d' % attempt) / 'record.json')

    def rid(self, position, suffix):
        return 'sweep_p%03d_%s_%s' % (position, self.order[position], suffix)

    def collect(self):
        T.quiet(L.main, ['collect', '--campaign', str(self.campaign)])
        with (self.campaign / 'results/decompositions.csv').open() as f:
            return {row['decomposition']: row for row in csv.DictReader(f)}


class FailurePolicyTests(Base):
    def test_autotune_failures_never_produce_a_catalog_or_a_precomputed_run(self):
        self.assertEqual(len(self.order), 7)
        kinds = {0: 'no_best', 2: 'empty_best', 4: 'exit', 6: 'nan'}
        rules = [dict(decomposition=self.order[p], mode=1, kind=k) for p, k in kinds.items()]
        status, text, launched = self.job(fail=rules)
        self.assertEqual(status, 5, text)                                   # run failures, nonzero
        for position, kind in kinds.items():
            d = self.order[position]
            self.assertNotIn(self.rid(position, 'pair1_P'), launched)       # no P run
            self.assertIn(self.rid(position, 'pair1_D'), launched)          # D still executed
            self.assertFalse((self.campaign / 'catalogs/sweep' / ('full_' + d)).exists())   # no catalog file
            record = self.record(self.rid(position, 'A'))
            self.assertEqual(record['autotune']['status'], 'FAILED')
            self.assertEqual(record['verdict'], 'VALID' if kind in ('no_best', 'empty_best') else 'INVALID')
            self.assertIn('AUTOTUNE_FAILED for %s' % d, text)
            self.assertEqual(self.record(self.rid(position, 'pair1_D'))['verdict'], 'VALID')
        self.assertIn('Best-block CSV is missing', self.record(self.rid(0, 'A'))['autotune']['reason'])
        self.assertIn('no kernel rows', self.record(self.rid(2, 'A'))['autotune']['reason'])
        self.assertIn('autotune run invalid', self.record(self.rid(4, 'A'))['autotune']['reason'])
        self.assertEqual(sorted(p.name for p in (self.campaign / 'catalogs/sweep').iterdir()),
                         sorted('full_' + self.order[p] for p in (1, 3, 5)))
        self.assertEqual(len(list(self.campaign.glob('catalogs/**/blocks.csv'))), 3)
        table = self.collect()
        for position in kinds:
            row = table[self.order[position]]
            self.assertEqual((row['status'], row['complete'], row['n_valid_P']), ('AUTOTUNE_FAILED', 'False', '0'))
            self.assertEqual(row['D_median_total_ttime_s'] != '', True)      # D still serves the decomposition comparison
            self.assertEqual((row['P_median_total_ttime_s'], row['S_median_of_paired_ratios'], row['catalog_rows']), ('', '', ''))
        # No automatic retry of the failed autotune runs on resume.
        status, text, launched = self.job(job='1002', fail=rules)
        self.assertEqual(status, 5)
        self.assertEqual(launched, ['sentinel_alloc_1002'])

    def test_oom_in_a_default_run_invalidates_the_decomposition_and_the_sweep_continues(self):
        bad = self.order[2]                                                 # even position: D runs first
        status, text, launched = self.job(fail=[dict(decomposition=bad, mode=0, kind='oom')])
        self.assertEqual(status, 5, text)
        self.assertIn(self.rid(2, 'A'), launched)
        self.assertIn(self.rid(2, 'pair1_D'), launched)
        self.assertNotIn(self.rid(2, 'pair1_P'), launched)                  # remaining runs skipped
        self.assertEqual(self.record(self.rid(2, 'pair1_D'))['category'], 'OOM')
        mark = read_json(self.campaign / 'state/decompositions' / (bad + '.json'))
        self.assertEqual((mark['status'], mark['category']), ('INVALID_AT_RUNTIME', 'OOM'))
        for position in (3, 4, 5, 6):                                       # the sweep continued
            self.assertIn(self.rid(position, 'pair1_P'), launched)
        self.assertEqual(self.collect()[bad]['status'], 'INVALID_AT_RUNTIME')
        status, text, launched = self.job(job='1002')
        self.assertEqual((status, launched), (5, ['sentinel_alloc_1002']))   # still not retried

    def test_overpartitioning_is_treated_like_oom(self):
        bad = self.order[1]                                                 # odd position: P runs first, then D
        status, text, launched = self.job(fail=[dict(decomposition=bad, mode=0, kind='overpartition')])
        self.assertEqual(status, 5)
        self.assertEqual(launched[launched.index(self.rid(1, 'A')):][:3],
                         [self.rid(1, 'A'), self.rid(1, 'pair1_P'), self.rid(1, 'pair1_D')])
        self.assertEqual(self.record(self.rid(1, 'pair1_D'))['category'], 'OVERPARTITION')
        mark = read_json(self.campaign / 'state/decompositions' / (bad + '.json'))
        self.assertEqual((mark['status'], mark['category']), ('INVALID_AT_RUNTIME', 'OVERPARTITION'))
        self.assertIn(self.rid(2, 'A'), launched)                           # the sweep continued

    def test_three_consecutive_failed_decompositions_abort(self):
        rules = [dict(decomposition=self.order[p], mode=0, kind='exit') for p in (1, 2, 3)]
        status, text, launched = self.job(fail=rules)
        self.assertEqual(status, 4, text)
        self.assertIn('ABORT: 3 consecutive failed decompositions', text)
        self.assertTrue(any(r.startswith('sweep_p003') for r in launched))
        self.assertFalse(any(r.startswith(('sweep_p004', 'sweep_p005', 'sweep_p006')) for r in launched))
        self.assertTrue((self.campaign / 'state/ABORTED.json').exists())
        status, text, launched = self.job(job='1002')                       # a chained job does nothing
        self.assertEqual((status, launched), (4, []))
        self.assertIn('An earlier job aborted this sweep', text)
        status, text, launched = self.job(job='1003', extra=['--retry-failed'])
        self.assertEqual(status, 0, text)
        self.assertFalse((self.campaign / 'state/ABORTED.json').exists())
        self.assertTrue(all(row['status'] == 'COMPLETE' for row in self.collect().values()))

    def test_two_failures_then_a_success_do_not_abort(self):
        rules = [dict(decomposition=self.order[p], mode=0, kind='exit') for p in (1, 2, 4, 5)]
        status, text, launched = self.job(fail=rules)
        self.assertEqual(status, 5)
        self.assertTrue(any(r.startswith('sweep_p006') for r in launched))

    def test_failed_runs_are_never_retried_automatically(self):
        marker = self.tmp / 'fired_once'
        bad = self.rid(3, 'pair1_P')
        status, text, launched = self.job(fail=[dict(decomposition=self.order[3], mode=2, kind='exit', once_file=str(marker))])
        self.assertEqual(status, 5)
        self.assertEqual(launched.count(bad), 1)
        self.assertEqual(self.record(bad)['verdict'], 'INVALID')
        status, text, launched = self.job(job='1002')                       # the mock would now succeed
        self.assertEqual(status, 5)
        self.assertNotIn(bad, launched)
        self.assertIn('not retried automatically', text)
        status, text, launched = self.job(job='1003', extra=['--retry-failed'])
        self.assertEqual(status, 0, text)
        self.assertEqual(sorted(launched), sorted(['sentinel_alloc_1003', bad, self.rid(3, 'pair1_D')]))   # whole pair
        self.assertTrue((self.campaign / 'runs/sweep' / bad / 'attempt_001/SUPERSEDED.json').exists())
        self.assertEqual(self.record(bad, 2)['verdict'], 'VALID')
        self.assertEqual(self.record(bad, 1)['verdict'], 'INVALID')          # the failed attempt is kept

    def test_run_timeout_is_a_recorded_failure(self):
        c = E.Campaign(self.campaign)
        plan, digest = c.plan('sweep')
        with patch.dict(os.environ, T.job_env(self.tmp, fail=[dict(decomposition='unforced', mode=0, kind='sleep')])):
            engine = E.Engine(c, 'sweep', plan, digest)
            with patch.object(R, 'launch', lambda command, log, timeout, env: R_launch(command, log, 2, env)):
                record, _ = T.quiet(engine.execute, plan['runs'][0])
        self.assertEqual((record['verdict'], record['category'], record['timed_out']), ('INVALID', 'TIMEOUT', True))


R_launch = R.launch


class ResumeTests(Base):
    def test_resume_skips_valid_runs_and_reruns_an_incomplete_pair_whole(self):
        status, text, first = self.job()
        self.assertEqual(status, 0, text)
        plan_sha = sha(self.campaign / 'plan.json')
        pair = [self.rid(4, 'pair1_D'), self.rid(4, 'pair1_P')]
        second = max(pair, key=lambda r: self.record(r)['plan_position'])
        partial = [r for r in pair if r != second][0]
        (self.campaign / 'runs/sweep' / second / 'attempt_001/record.json').unlink()   # allocation ended mid-run
        status, text, launched = self.job(job='1002')
        self.assertEqual(status, 0, text)
        self.assertEqual(launched, ['sentinel_alloc_1002'] + sorted(pair, key=lambda r: self.record(r, 2)['plan_position']))
        self.assertEqual(sha(self.campaign / 'plan.json'), plan_sha)           # stored plan reused, never regenerated
        self.assertTrue(all(self.record(r, 2)['plan_hash'] == self.record(first[0])['plan_hash'] for r in pair))
        old = self.campaign / 'runs/sweep' / partial / 'attempt_001'
        self.assertEqual(read_json(old / 'record.json')['verdict'], 'VALID')   # partial run kept ...
        self.assertIn('incomplete pair', read_json(old / 'SUPERSEDED.json')['reason'])   # ... but superseded
        self.assertEqual({self.record(r, 2)['slurm_job_id'] for r in pair}, {'1002'})
        table = self.collect()
        self.assertEqual(table[self.order[4]]['slurm_job_ids'], '1001 1002')    # job id kept as a covariate
        with (self.campaign / 'results/runs.csv').open() as f:
            runs = list(csv.DictReader(f))
        flagged = [r for r in runs if r['run_id'] == partial and r['attempt'] == 'attempt_001'][0]
        self.assertEqual((flagged['superseded'], flagged['used_in_statistics']), ('True', 'False'))
        status, text, launched = self.job(job='1003')                          # nothing left
        self.assertEqual((status, launched), (0, ['sentinel_alloc_1003']))

    def test_pair_split_over_two_allocations_is_rerun(self):
        self.job()
        target = self.rid(5, 'pair1_P')
        path = self.campaign / 'runs/sweep' / target / 'attempt_001/record.json'
        path.write_text(json.dumps(dict(read_json(path), slurm_job_id='999')))
        status, text, launched = self.job(job='1002')
        self.assertEqual(sorted(launched), sorted(['sentinel_alloc_1002', target, self.rid(5, 'pair1_D')]))

    def test_resume_refuses_a_changed_executable(self):
        self.job()
        exe = self.campaign / 'build/full/senga2_f2c_mpi_hip'
        exe.write_text(exe.read_text() + '# rebuilt\n')
        with self.assertRaisesRegex(ValueError, 'executable changed since it was built'):
            self.job(job='1002')
        manifest = self.campaign / 'build/full/build_manifest.json'
        atomic(manifest, dict(read_json(manifest), sha256=sha(exe)))            # a "new build" with a consistent manifest
        with self.assertRaisesRegex(ValueError, 'Executable hash changed: run .* Resume refused'):
            self.job(job='1002')
        self.assertEqual(len(T.srun_calls(self.tmp)), 24)                      # nothing launched after the first job

    def test_resume_refuses_an_edited_plan(self):
        self.job()
        path = self.campaign / 'plan.json'
        path.write_text(path.read_text().replace('"seed": %d' % P.DEFAULT_SEED, '"seed": 1'))
        with self.assertRaisesRegex(ValueError, 'changed after it was frozen'):
            self.job(job='1002')

    def test_frozen_catalog_is_reused_only_if_its_hash_matches(self):
        self.job()
        good, bad = self.order[4], self.order[5]
        for position in (4, 5):                                               # both pairs were cut off
            for suffix in ('pair1_D', 'pair1_P'):
                (self.campaign / 'runs/sweep' / self.rid(position, suffix) / 'attempt_001/record.json').unlink()
        blocks = self.campaign / 'catalogs/sweep' / ('full_' + bad) / 'blocks.csv'
        digest = sha(self.campaign / 'catalogs/sweep' / ('full_' + good) / 'blocks.csv')
        os.chmod(str(blocks), 0o644)
        blocks.write_text(blocks.read_text().replace(',1,32,8', ',32,4,1'))
        status, text, launched = self.job(job='1002')
        self.assertEqual(status, 5, text)
        self.assertIn('CATALOG REJECTED for %s' % bad, text)
        self.assertIn(self.rid(4, 'pair1_P'), launched)                        # intact catalog reused in a later allocation
        self.assertEqual(self.record(self.rid(4, 'pair1_P'), 2)['catalog_sha256'], digest)
        self.assertNotIn(self.rid(4, 'A'), launched)                           # ... without a new autotune run
        self.assertNotIn(self.rid(5, 'pair1_P'), launched)                     # changed catalog: no replay
        self.assertNotIn(self.rid(5, 'A'), launched)                           # and no silent re-tuning either
        self.assertIn(self.rid(5, 'pair1_D'), launched)

    def test_a_replay_cannot_use_the_catalog_of_another_decomposition(self):
        self.job()
        c = E.Campaign(self.campaign)
        plan, digest = c.plan('sweep')
        mine, other = self.order[0], self.order[1]
        run = next(r for r in plan['runs'] if r['run_id'] == self.rid(0, 'pair1_P'))
        exe_sha = c.build('full')['sha256']
        foreign = R.load_catalog(c.catalog_folder('sweep', 'full', other), other, exe_sha)
        before = len(T.srun_calls(self.tmp))
        with patch.dict(os.environ, T.job_env(self.tmp)):
            engine = E.Engine(c, 'sweep', plan, digest)
            with self.assertRaisesRegex(ValueError, 'Refusing to launch .* with the catalog of %s' % other):
                engine.execute(run, foreign)
            disguised = dict(foreign, decomposition=mine)                      # same file, relabelled
            with self.assertRaisesRegex(ValueError, 'belongs to decomposition %s, not %s' % (other, mine)):
                engine.execute(run, disguised)
            with self.assertRaisesRegex(ValueError, 'has no catalog'):
                engine.execute(run, None)
            swapped = dict(R.load_catalog(c.catalog_folder('sweep', 'full', mine), mine, exe_sha), path=foreign['path'])
            with self.assertRaisesRegex(ValueError, 'Catalog identity changed'):
                engine.execute(run, swapped)
        self.assertEqual(len(T.srun_calls(self.tmp)), before)                  # nothing was launched
        self.assertEqual(len(c.attempts('sweep', run['run_id'])), 1)


class TimeGuardTests(Base):
    def test_job_stops_before_launching_when_time_is_short(self):
        status, text, launched = self.job(remaining=30)
        self.assertEqual(status, 3, text)                                     # distinct, documented status
        self.assertEqual(launched, [])
        self.assertIn('TIME GUARD', text)
        self.assertIn('submit --campaign', text)
        self.assertIn('TIME GUARD', (self.campaign / 'state/events.log').read_text())

    def test_pair_is_not_started_when_only_one_member_fits(self):
        c = E.Campaign(self.campaign)
        plan, digest = c.plan('sweep')
        with patch.dict(os.environ, T.job_env(self.tmp)):
            engine = E.Engine(c, 'sweep', plan, digest, estimates=dict(A=100, D=100, P=100))
            self.assertEqual(engine.estimate(plan['runs'][0]), 100)
            pair = [r for r in plan['runs'] if r.get('pair_index') == 1][:2]
            with patch.object(R, 'remaining_seconds', lambda env=None: 250.0):
                engine.guard(pair[:1])                                       # 100 * 1.2 + 60 = 180 <= 250
                with self.assertRaises(L.StopJob) as caught:
                    engine.guard(pair)                                       # 200 * 1.2 + 60 = 300 > 250
            self.assertEqual(caught.exception.status, 3)
            with patch.object(R, 'remaining_seconds', lambda env=None: None):
                with self.assertRaisesRegex(ValueError, 'Cannot determine the remaining allocation time'):
                    engine.guard(pair)
                E.Engine(c, 'sweep', plan, digest, no_time_guard=True).guard(pair)

    def test_remaining_time_sources(self):
        self.assertAlmostEqual(R.remaining_seconds(dict(SLURM_JOB_END_TIME=str(int(time.time()) + 500))), 500, delta=3)
        bindir = self.tmp / 'squeue_bin'
        bindir.mkdir()
        (bindir / 'squeue').write_text('#!/bin/sh\necho "1-02:03:04"\n')
        (bindir / 'squeue').chmod(0o755)
        with patch.dict(os.environ, dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'])):
            self.assertEqual(R.remaining_seconds(dict(SLURM_JOB_ID='5')), 93784.0)
        fallback = dict(SENGA_JOB_START_EPOCH=str(int(time.time()) - 100), SENGA_JOB_TIME_LIMIT_S='1000')
        self.assertAlmostEqual(R.remaining_seconds(fallback), 900, delta=3)
        self.assertIsNone(R.remaining_seconds({}))


class BaselineTests(Base):
    def test_d0_comes_from_the_printed_grid_of_the_unforced_run(self):
        status, text, launched = self.job()
        self.assertEqual(read_json(self.campaign / 'state/d0.json')['d0'], [2, 2, 2])
        self.assertEqual(launched[:2], ['baseline_unforced_r1', 'sentinel_start'])
        self.assertEqual(self.record('sentinel_start')['triple'], [2, 2, 2])

    def test_another_printed_grid_changes_d0(self):
        status, text, launched = self.job(unforced_grid=[4, 2, 1])
        self.assertEqual(status, 0, text)
        self.assertEqual(read_json(self.campaign / 'state/d0.json')['d0'], [4, 2, 1])
        sentinel = self.record('sentinel_start')
        self.assertEqual((sentinel['decomposition'], sentinel['decomposition_planned']), ('4x2x1', 'd0'))
        self.assertTrue(all(t in sentinel['argv'] for t in ('OPS_FORCE_DECOMP_X=4', 'OPS_FORCE_DECOMP_Y=2', 'OPS_FORCE_DECOMP_Z=1')))
        periodic = self.record('sentinel_after_p003')
        self.assertEqual(periodic['triple'], [4, 2, 1])
        self.assertFalse(any('OPS_FORCE_DECOMP' in t for t in self.record('baseline_unforced_r1')['argv']))
        table = self.collect()
        self.assertEqual([d for d, row in table.items() if row['is_d0'] == 'True'], ['4x2x1'])
        for name in ('launcher.py', 'engine.py', 'plan.py', 'runtime.py', 'collect.py'):
            self.assertNotIn('8x5x4', (Path(L.HERE) / name).read_text())      # no hardcoded baseline grid

    def test_failed_baseline_stops_the_sweep(self):
        status, text, launched = self.job(fail=[dict(decomposition='unforced', mode=0, kind='exit')])
        self.assertEqual(status, 5)
        self.assertEqual(launched, ['baseline_unforced_r1'])
        self.assertIn('Cannot identify d0', text)


class QualificationTests(Base):
    def smoke(self, **settings):
        return self.job(kind='smoke', extra=['--builds', 'small,full'], **settings)

    def gonogo(self, *extra):
        return T.quiet(L.main, ['gonogo', '--campaign', str(self.campaign)] + list(extra))

    FULL = ('--gpu-hours-per-node-hour', '4', '--remaining-gpu-hours', '10000', '--disk-available-gb', '500')

    def test_one_command_lists_q1_to_q5_with_both_builds(self):
        status, text = T.quiet(L.main, ['smoke', '--campaign', str(self.campaign), '--dry-run'])
        self.assertEqual(status, 0)
        for stage in ('Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
            self.assertIn('  %s: ' % stage, text)
        lines = [l.split() for l in text.splitlines() if l.strip().split()[:1] and l.split()[0].isdigit()]
        self.assertEqual([l[1] for l in lines], [r['stage'] for r in read_json(self.campaign / 'smoke_plan.json')['runs']])
        self.assertEqual({l[3] for l in lines if l[1] in ('Q1', 'Q2', 'Q3')}, {'small'})
        self.assertEqual({l[3] for l in lines if l[1] in ('Q4', 'Q5')}, {'full'})
        self.assertEqual([l[2] for l in lines if l[1] == 'Q5'], ['q5_d0_A', 'q5_d0_1_D', 'q5_d0_2_P', 'q5_d0_3_P', 'q5_d0_4_D'])
        self.assertIn('nothing submitted', text)
        self.assertEqual(T.srun_calls(self.tmp), [])

    def test_go_after_a_clean_qualification(self):
        status, text, launched = self.smoke()
        self.assertEqual(status, 0, text)
        self.assertEqual(launched[-5:], ['q5_d0_A', 'q5_d0_1_D', 'q5_d0_2_P', 'q5_d0_3_P', 'q5_d0_4_D'])
        small = self.record('q1_1x1x1_D', scope='qualification')
        full = self.record('q5_d0_1_D', scope='qualification')
        self.assertNotEqual(small['executable_sha256'], full['executable_sha256'])     # two builds, two hashes
        self.assertEqual((small['build'], small['grid'], full['build']), ('small', 20, 'full'))
        self.assertEqual(self.record('q3_2x2x2_D_nogpudirect', scope='qualification')['gpu_direct_reported'], False)
        self.assertTrue((self.campaign / 'catalogs/qualification/small_1x1x1/blocks.csv').is_file())
        status, text = self.gonogo(*self.FULL)
        self.assertEqual(status, 0, text)
        self.assertIn('VERDICT: GO', text)
        self.assertRegex(text, r'Projection: 7 decompositions, \d+ runs, [\d.]+ h wall time on 1 nodes = \d+ node-hours')
        self.assertRegex(text, r'Measured elapsed per run \(s\): A \d+, D \d+, P \d+')
        self.assertIn('GPU-hours projected', text)
        saved = read_json(self.campaign / 'gonogo.json')
        self.assertEqual(saved['verdict'], 'GO')
        self.assertEqual(sorted(read_json(self.campaign / 'measured_times.json')['elapsed_s']), ['A', 'D', 'P'])
        # qualification records never enter the sweep statistics
        self.job()
        T.quiet(L.main, ['collect', '--campaign', str(self.campaign)])
        with (self.campaign / 'results/runs.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r['used_in_statistics'] == 'False' for r in rows if r['scope'] == 'qualification'))
        self.assertTrue(any(r['scope'] == 'qualification' for r in rows))

    def test_no_go_is_nonzero(self):
        status, text = self.gonogo(*self.FULL)                                # nothing has run yet
        self.assertEqual(status, 6)
        self.assertIn('VERDICT: NO-GO', text)
        self.smoke()
        status, text = self.gonogo()                                          # budget and disk inputs missing
        self.assertEqual(status, 6)
        self.assertIn('FAIL 5 budget', text)
        self.assertIn('Projection:', text)                                    # still printed from measured times
        status, text = self.gonogo('--gpu-hours-per-node-hour', '4', '--remaining-gpu-hours', '0.000001', '--disk-available-gb', '500')
        self.assertEqual(status, 6)
        self.assertRegex(text, 'FAIL 5 budget.*GPU-hours projected')
        status, text = self.gonogo('--gpu-hours-per-node-hour', '4', '--remaining-gpu-hours', '10000', '--disk-available-gb', '0.0000001')
        self.assertEqual(status, 6)
        self.assertIn('FAIL 6 disk', text)

    def test_failed_qualification_runs_give_no_go(self):
        status, text, launched = self.smoke(fail=[dict(decomposition='2x2x2', mode=2, kind='exit')])
        self.assertEqual(status, 5)
        status, text = self.gonogo(*self.FULL)
        self.assertEqual(status, 6)
        self.assertIn('FAIL 1 every qualification run valid', text)
        self.assertIn('FAIL 4 Q5', text)

    def test_fingerprint_mismatch_between_decompositions_gives_no_go(self):
        status, text, launched = self.smoke(fail=[dict(decomposition='unforced', mode=0, kind='fingerprint')])
        self.assertEqual(status, 5)
        self.assertEqual(self.record('q3_unforced_D', scope='qualification')['category'], 'FINGERPRINT_MISMATCH')
        self.assertEqual(self.gonogo(*self.FULL)[0], 6)

    def test_submit_requires_go_and_never_submits_in_a_dry_run(self):
        with self.assertRaisesRegex(ValueError, 'go/no-go verdict is missing'):
            L.main(['submit', '--campaign', str(self.campaign), '--dry-run'])
        self.smoke()
        self.gonogo(*self.FULL)
        status, text = T.quiet(L.main, ['submit', '--campaign', str(self.campaign), '--dry-run', '--jobs', '2', '--time', '12:00:00'])
        self.assertEqual(status, 0)
        self.assertEqual(text.count('DRY RUN, not submitted: sbatch'), 2)
        self.assertIn('--account=%s --partition=standard-g --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 '
                      '--cpus-per-task=7 --mem=0 --time=12:00:00 --exclusive' % T.ACCOUNT, text)
        script = (self.campaign / 'slurm/sweep_0.sh').read_text()
        self.assertIn('SENGA_JOB_TIME_LIMIT_S=43200', script)
        self.assertIn('job --campaign %s --kind sweep' % self.campaign, script)
        self.assertFalse((self.campaign / 'submissions.json').exists())

    def test_submit_and_resume_with_a_fake_sbatch(self):
        self.smoke()
        self.gonogo(*self.FULL)
        bindir = self.tmp / 'bin'
        counter = self.tmp / 'sbatch_counter'
        (bindir / 'sbatch').write_text('#!/bin/sh\nn=$(cat %s 2>/dev/null || echo 7000)\nn=$((n+1))\necho $n > %s\n'
                                       'echo "$@" >> %s\necho "$n;lumi"\n' % (counter, counter, self.tmp / 'sbatch_args'))
        (bindir / 'squeue').write_text('#!/bin/sh\ncat %s 2>/dev/null\nexit 0\n' % (self.tmp / 'queue'))
        for name in ('sbatch', 'squeue'):
            (bindir / name).chmod(0o755)
        env = T.job_env(self.tmp)
        with patch.dict(os.environ, dict(PATH=env['PATH'])):
            status, text = T.quiet(L.main, ['submit', '--campaign', str(self.campaign), '--jobs', '2'])
            self.assertEqual(status, 0, text)
            jobs = read_json(self.campaign / 'submissions.json')['jobs']
            self.assertEqual([(j['kind'], j['job_id']) for j in jobs], [('sweep', '7001'), ('sweep', '7002')])
            lines = (self.tmp / 'sbatch_args').read_text().splitlines()
            self.assertNotIn('--dependency', lines[0])
            self.assertIn('--dependency=afterany:7001', lines[1])
            with self.assertRaisesRegex(ValueError, 'already submitted .* Use --resume'):
                L.main(['submit', '--campaign', str(self.campaign)])
            (self.tmp / 'queue').write_text('7002\n')
            with self.assertRaisesRegex(ValueError, r'Sweep job\(s\) 7002 are still pending or running'):
                L.main(['submit', '--campaign', str(self.campaign), '--resume'])
            (self.tmp / 'queue').write_text('')
            status, text = T.quiet(L.main, ['submit', '--campaign', str(self.campaign), '--resume', '--retry-failed'])
            self.assertEqual(status, 0)
            self.assertIn('--retry-failed', (self.campaign / 'slurm/sweep_0.sh').read_text())


class CollectorTests(Base):
    reps, exclude = 3, ('1x4x2', '4x1x2', '1x2x4', '2x1x4')

    def set_time(self, run_id, seconds, attempt=1):
        path = self.campaign / 'runs/sweep' / run_id / ('attempt_%03d' % attempt) / 'record.json'
        atomic(path, dict(read_json(path), total_ttime_s=seconds))

    def test_tables_use_the_declared_estimators_and_never_fill_gaps(self):
        self.assertEqual(len(self.order), 3)
        broken = self.order[1]
        status, text, launched = self.job(fail=[dict(decomposition=broken, mode=2, kind='exit',
                                                     once_file=str(self.tmp / 'once'))])
        self.assertEqual(status, 5)
        d_times, p_times = {1: 100.0, 2: 130.0, 3: 90.0}, {1: 80.0, 2: 100.0, 3: 90.0}
        for pair in (1, 2, 3):
            self.set_time(self.rid(0, 'pair%d_D' % pair), d_times[pair])
            self.set_time(self.rid(0, 'pair%d_P' % pair), p_times[pair])
        self.set_time(self.rid(0, 'A'), 400.0)
        table = self.collect()
        row = table[self.order[0]]
        self.assertEqual((row['status'], row['complete'], row['n_valid_A'], row['n_valid_D'], row['n_valid_P'],
                          row['n_valid_pairs']), ('COMPLETE', 'True', '1', '3', '3', '3'))
        self.assertEqual([float(row[k]) for k in ('D_median_total_ttime_s', 'D_min_total_ttime_s', 'D_max_total_ttime_s')],
                         [100.0, 90.0, 130.0])
        self.assertEqual([float(row[k]) for k in ('P_median_total_ttime_s', 'P_min_total_ttime_s', 'P_max_total_ttime_s')],
                         [90.0, 80.0, 100.0])
        self.assertEqual([float(row['S_pair%d' % i]) for i in (1, 2, 3)], [1.25, 1.3, 1.0])
        self.assertEqual(float(row['S_median_of_paired_ratios']), 1.25)        # median of ratios ...
        self.assertAlmostEqual(float(row['S_ratio_of_medians']), 100.0 / 90.0)  # ... differs from the ratio of medians
        self.assertEqual(float(row['A_total_ttime_s']), 400.0)
        self.assertEqual(float(row['search_cost_A_over_D_median']), 4.0)
        self.assertEqual((row['catalog_rows'], row['executed_kernels']), ('81', '84'))
        self.assertEqual(row['catalog_share_below_0p90'], 'True')              # flagged, not failed
        bad = table[broken]
        self.assertEqual((bad['status'], bad['complete'], bad['n_valid_P'], bad['n_valid_pairs']), ('INCOMPLETE', 'False', '2', '2'))
        for key in ('P_median_total_ttime_s', 'P_min_total_ttime_s', 'P_max_total_ttime_s', 'S_pair1', 'S_pair2', 'S_pair3',
                    'S_median_of_paired_ratios', 'S_ratio_of_medians'):
            self.assertEqual(bad[key], '', key)                                # missing values stay empty
        self.assertNotEqual(bad['D_median_total_ttime_s'], '')
        with (self.campaign / 'results/incomplete_decompositions.csv').open() as f:
            incomplete = list(csv.DictReader(f))
        self.assertEqual([r['decomposition'] for r in incomplete], [broken])
        self.assertIn('pair1_P', incomplete[0]['runs_without_valid_record'])
        self.assertIn('srun exit status', incomplete[0]['failure_reasons'])
        header = (self.campaign / 'results/decompositions.csv').read_text().splitlines()[0]
        for name in ('D_median_total_ttime_s', 'D_min_total_ttime_s', 'D_max_total_ttime_s', 'S_pair1',
                     'S_median_of_paired_ratios', 'S_ratio_of_medians'):
            self.assertIn(name, header)
        self.assertFalse(any(word in header.lower() for word in ('best', 'rank', 'winner', 'fastest')))
        with (self.campaign / 'results/runs.csv').open() as f:
            runs = list(csv.DictReader(f))
        self.assertEqual(len(runs), len(launched))                             # one row per run record
        failed = [r for r in runs if r['verdict'] == 'INVALID']
        self.assertEqual([(r['run_id'], r['used_in_statistics']) for r in failed], [(self.rid(1, 'pair1_P'), 'False')])
        manifest = read_json(self.campaign / 'results/collection_manifest.json')
        self.assertEqual(manifest['incomplete_decompositions'], [broken])
        self.assertEqual(len(manifest['input_records']), len(launched))
        self.assertIn('S_median_of_paired_ratios', manifest['estimators'])
        self.assertEqual(manifest['exclusions']['invalid_records'], 1)
        self.assertTrue(manifest['code_version']['launcher_sha256'])
        with (self.campaign / 'results/sentinels.csv').open() as f:
            noise = list(csv.DictReader(f))
        self.assertEqual([r['group'] for r in noise], ['job 1001', 'all allocations'])
        self.assertEqual(noise[0]['n'], '1')


if __name__ == '__main__':
    unittest.main()
