"""Plan, enumeration, ordering, arguments and run-control file (criteria 3-7, 20, 24)."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import launcher as L
import plan as P
import sources as S
import testlib as T

CFG = dict(ranks=160, nodes=20, grid=1440, steps=100, repetitions=3, seed=P.DEFAULT_SEED, fixed_order=False,
           sentinel_every=8, small_grid=256, smoke_steps=10)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = P.make_sweep_plan(CFG)
        cls.sweep = [r for r in cls.plan['runs'] if r['stage'] == 'sweep']

    def test_enumeration_has_63_distinct_exact_triples(self):
        triples = P.ordered_triples(160)
        self.assertEqual(len(triples), 63)
        self.assertEqual(len(set(triples)), 63)
        self.assertTrue(all(a * b * c == 160 for a, b, c in triples))
        self.assertEqual(len(self.plan['decompositions']), 63)
        self.assertEqual(len({d['decomposition'] for d in self.plan['decompositions']}), 63)
        self.assertEqual(len({d['permutation_class'] for d in self.plan['decompositions']}), 12)

    def test_product_different_from_ranks_is_rejected(self):
        for bad in ((4, 5, 4), (8, 5, 8), (160, 1, 2)):
            with self.assertRaisesRegex(ValueError, 'has product'):
                P.check_triple(bad, 160)
        with self.assertRaisesRegex(ValueError, 'has product 80 but the job has 160'):
            P.make_run('x', 'sweep', 'D', '4x5x4', CFG, 'full', 20, 160, 100)

    def test_validity_rule_and_halo_constant(self):
        self.assertEqual(P.HALO_DEPTH, 5)          # SENGA2 com_senga2.F90:32, OPS ops_mpi_rt_support.cpp:204-208
        self.assertEqual(self.plan['skipped'], [])  # nothing skipped for 1440 / 160
        self.assertEqual(P.validity((160, 1, 1), 1440), (True, ''))            # local width 9
        self.assertFalse(P.validity((160, 1, 1), 640)[0])                      # local width 4 < 5
        self.assertFalse(P.validity((32, 5, 1), 1000)[0])                      # 1000 / 32 is not an integer
        self.assertTrue(P.validity((2, 2, 2), 10)[0])                          # exactly the halo depth
        valid, skipped = P.enumerate_decompositions(160, 640)
        self.assertEqual(len(valid) + len(skipped), 63)
        self.assertIn('8x1x20', [P.shape_text(t) for t in valid])
        self.assertTrue(all(640 % p == 0 and 640 // p >= 5 for t in valid for p in t))
        self.assertTrue(any(s['decomposition'] == '160x1x1' and 'below the halo depth 5' in s['reason'] for s in skipped))

    def test_skipped_triples_are_announced(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, text = T.quiet(L.main, ['plan', '--output-dir', tmp, '--grid', '640'])
            self.assertIn('NOTICE skipped 160x1x1: local x width 4 is below the halo depth 5', text)
            plan = json.loads((Path(tmp) / 'plan.json').read_text())
            self.assertNotIn('160x1x1', [d['decomposition'] for d in plan['decompositions']])
            self.assertEqual(len(plan['decompositions']) + len(plan['skipped']), 63)

    def test_exactly_16_halo60_risk_triples(self):
        risky = [d for d in self.plan['decompositions'] if d['halo60_risk']]
        self.assertEqual(len(risky), 16)
        self.assertTrue(all(min(d['local_size'][1:]) < 60 for d in risky))
        self.assertTrue(all(min(d['local_size'][1:]) >= 60 for d in self.plan['decompositions'] if not d['halo60_risk']))

    def test_dry_run_writes_the_complete_ordered_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text = T.quiet(L.main, ['plan', '--output-dir', tmp])
            self.assertEqual(status, 0)
            self.assertIn('Dry run only', text)
            plan = json.loads((Path(tmp) / 'plan.json').read_text())
            self.assertEqual(plan, json.loads(P.plan_bytes(self.plan).decode()))
            with (Path(tmp) / 'plan.csv').open() as f:
                rows = list(csv.DictReader(f))
            digest = (Path(tmp) / 'plan.json.sha256').read_text().split()[0]
            self.assertEqual(digest, P.plan_hash(plan))
        self.assertEqual([int(r['position']) for r in rows], list(range(len(rows))))
        for field in ('position', 'decomposition', 'mode', 'repetition', 'pair_index', 'stage', 'argv', 'seed'):
            self.assertTrue(all(field in r for r in rows), field)
        self.assertTrue(all(r['seed'] == str(P.DEFAULT_SEED) and r['argv'] for r in rows))
        count = lambda stage, mode: sum(1 for r in rows if r['stage'] == stage and r['mode'] == mode)
        self.assertEqual((count('sweep', 'A'), count('sweep', 'D'), count('sweep', 'P')), (63, 189, 189))
        self.assertEqual(count('baseline', 'D'), 3)
        self.assertEqual(count('sentinel', 'D'), 8)      # 1 at the start + after every 8th decomposition (7)
        self.assertEqual(len(rows), 63 + 189 + 189 + 3 + 8)

    def test_same_seed_is_byte_identical_and_other_seed_differs(self):
        self.assertEqual(P.plan_bytes(P.make_sweep_plan(dict(CFG))), P.plan_bytes(self.plan))
        other = P.make_sweep_plan(dict(CFG, seed=7))
        order = lambda plan: [d['decomposition'] for d in plan['decompositions']]
        self.assertNotEqual(order(other), order(self.plan))
        self.assertEqual(sorted(order(other)), sorted(order(self.plan)))
        self.assertNotEqual(order(self.plan), sorted(order(self.plan)))
        fixed = P.make_sweep_plan(dict(CFG, fixed_order=True))
        self.assertEqual([tuple(d['triple']) for d in fixed['decompositions']], P.ordered_triples(160))

    def test_autotune_first_and_interleaved_pairs(self):
        starts = {'D': 0, 'P': 0}
        for dec in self.plan['decompositions']:
            runs = [r for r in self.sweep if r['decomposition'] == dec['decomposition']]
            self.assertEqual([r['position'] for r in runs], list(range(runs[0]['position'], runs[0]['position'] + 7)))
            modes = ''.join(r['mode'] for r in runs)
            expected = 'ADPPDDP' if dec['position'] % 2 == 0 else 'APDDPPD'
            self.assertEqual(modes, expected)
            self.assertEqual([r['pair_index'] for r in runs[1:]], [1, 1, 2, 2, 3, 3])
            self.assertEqual([r['order_in_pair'] for r in runs[1:]], [1, 2, 1, 2, 1, 2])
            starts[modes[1]] += 1
            self.assertEqual(dec['first_pair_mode'], modes[1])
        self.assertEqual(starts, {'D': 32, 'P': 31})

    def test_sentinels_and_baseline_positions(self):
        runs = self.plan['runs']
        self.assertEqual([r['run_id'] for r in runs[:4]],
                         ['baseline_unforced_r1', 'baseline_unforced_r2', 'baseline_unforced_r3', 'sentinel_start'])
        periodic = [r for r in runs if r.get('sentinel_kind') == 'periodic']
        self.assertEqual([r['after_decomp_position'] for r in periodic], [7, 15, 23, 31, 39, 47, 55])
        for r in periodic:
            self.assertEqual(runs[r['position'] - 1]['decomp_position'], r['after_decomp_position'])
            self.assertFalse(r['argv_resolved'])
        resolved = P.resolve_d0(periodic[0], [8, 5, 4])
        self.assertIn('OPS_FORCE_DECOMP_X=8', resolved['argv'])
        self.assertIn('OPS_FORCE_DECOMP_Z=4', resolved['argv_other_ranks'])
        with self.assertRaisesRegex(ValueError, 'has product'):
            P.resolve_d0(periodic[0], [8, 5, 5])

    def test_arguments_fit_the_fortran_buffer(self):
        for run in self.plan['runs']:
            for argv in (run['argv'], run['argv_other_ranks']):
                self.assertTrue(all(len(token) <= 63 for token in argv))
                self.assertIn('-gpudirect', argv)
                self.assertIn('-OPS_DIAGS=2', argv)
        with self.assertRaisesRegex(ValueError, 'is 64 characters long'):
            P.check_argv(['OPS_FORCE_DECOMP_X=' + '1' * 45])
        P.check_argv(['OPS_FORCE_DECOMP_X=' + '1' * 44])

    def test_forced_and_unforced_arguments(self):
        for run in self.sweep:
            x, y, z = run['triple']
            for argv in (run['argv'], run['argv_other_ranks']):
                for token in ('OPS_FORCE_DECOMP_X=%d' % x, 'OPS_FORCE_DECOMP_Y=%d' % y, 'OPS_FORCE_DECOMP_Z=%d' % z):
                    self.assertEqual(argv.count(token), 1)
        for run in self.plan['runs']:
            if run['stage'] == 'baseline':
                self.assertFalse(any('OPS_FORCE_DECOMP' in t for t in run['argv'] + run['argv_other_ranks']))

    def test_mode_arguments(self):
        block = ['OPS_BLOCK_SIZE_X=32', 'OPS_BLOCK_SIZE_Y=4', 'OPS_BLOCK_SIZE_Z=1']
        for run in self.sweep:
            first, other = run['argv'], run['argv_other_ranks']
            want = {'D': (0, 0), 'A': (1, 0), 'P': (2, 2)}[run['mode']]
            self.assertIn('-OPS_AUTOTUNE_MODE=%d' % want[0], first)
            self.assertIn('-OPS_AUTOTUNE_MODE=%d' % want[1], other)
            self.assertTrue(all(b in first and b in other for b in block))

    def test_exclusion_filter_is_explicit(self):
        plan = P.make_sweep_plan(dict(CFG, exclude_halo60_risk=True, exclude=['4x5x8']))
        self.assertEqual(len(plan['decompositions']), 63 - 16 - 1)
        self.assertEqual(len(plan['excluded']), 17)
        with self.assertRaisesRegex(ValueError, 'has product'):
            P.make_sweep_plan(dict(CFG, exclude=['4x5x9']))

    def test_qualification_sequence(self):
        smoke = P.make_smoke_plan(CFG)
        got = [(r['stage'], r['run_id'], r['build'], r['nodes'], r['ranks'], r['steps']) for r in smoke['runs']]
        self.assertEqual(got, [
            ('Q1', 'q1_1x1x1_D', 'small', 1, 1, 100), ('Q2', 'q2_1x1x1_A', 'small', 1, 1, 100),
            ('Q2', 'q2_1x1x1_P', 'small', 1, 1, 100), ('Q3', 'q3_2x2x2_D', 'small', 1, 8, 100),
            ('Q3', 'q3_8x1x1_D', 'small', 1, 8, 100), ('Q3', 'q3_1x8x1_D', 'small', 1, 8, 100),
            ('Q3', 'q3_1x1x8_D', 'small', 1, 8, 100), ('Q3', 'q3_unforced_D', 'small', 1, 8, 100),
            ('Q3', 'q3_2x2x2_D_nogpudirect', 'small', 1, 8, 100), ('Q4', 'q4_unforced_D', 'full', 20, 160, 10),
            ('Q4', 'q4_4x5x8_D', 'full', 20, 160, 10), ('Q4', 'q4_160x1x1_D', 'full', 20, 160, 10),
            ('Q4', 'q4_1x160x1_D', 'full', 20, 160, 10), ('Q4', 'q4_1x1x160_D', 'full', 20, 160, 10),
            ('Q5', 'q5_d0_A', 'full', 20, 160, 100), ('Q5', 'q5_d0_1_D', 'full', 20, 160, 100),
            ('Q5', 'q5_d0_2_P', 'full', 20, 160, 100), ('Q5', 'q5_d0_3_P', 'full', 20, 160, 100),
            ('Q5', 'q5_d0_4_D', 'full', 20, 160, 100)])
        self.assertEqual(sorted(smoke['stages']), ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        nogpu = next(r for r in smoke['runs'] if r['run_id'].endswith('nogpudirect'))
        self.assertNotIn('-gpudirect', nogpu['argv'])
        self.assertEqual(sum('-gpudirect' not in r['argv'] for r in smoke['runs']), 1)

    def test_run_control_file_has_no_dump_inside_the_run(self):
        text = S.cont_dat(T.CONT_DAT, 1440, 100)
        v = S.cont_values(text)
        self.assertEqual(v['step_line'].split(), ['1.0D-12', '1', '100', '1'])
        self.assertEqual((v['ntime1'], v['steps']), (1, 100))
        self.assertEqual(v['grid'], [1440, 1440, 1440])
        for interval in (v['ntdump'], v['ntrept'], v['ntstat']):
            self.assertTrue(all(step % interval != 0 for step in range(1, 101)), interval)
        self.assertEqual(v['dump_line'].split()[3], '0')            # dump format switch untouched
        self.assertEqual(S.cont_values(S.cont_dat(T.CONT_DAT, 256, 10))['steps'], 10)
        self.assertEqual(text.count('\n'), T.CONT_DAT.count('\n'))
        with self.assertRaisesRegex(ValueError, 'lacks the expected section'):
            S.cont_dat(T.CONT_DAT.replace('Intervals between dumps', 'Something else'), 64, 5)
        with self.assertRaisesRegex(ValueError, 'must stay 1 1 1'):
            S.cont_dat(T.CONT_DAT.replace(' 1      1      1', ' 2      1      1'), 64, 5)


if __name__ == '__main__':
    unittest.main()
