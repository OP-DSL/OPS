"""Plan, decomposition list, ordering, arguments and the run input clover.in."""
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
           sentinel_every=8, small_grid=256, smoke_steps=10, decompositions=list(P.DEFAULT_DECOMPOSITIONS))
FIXED16 = ['5x8x4', '4x10x4', '5x16x2', '4x20x2', '8x5x4', '5x32x1', '8x4x5', '8x10x2', '4x40x1', '8x20x1', '10x4x4',
           '2x40x2', '10x8x2', '10x2x8', '10x16x1', '16x10x1']


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = P.make_sweep_plan(CFG)
        cls.sweep = [r for r in cls.plan['runs'] if r['stage'] == 'sweep']

    def test_the_fixed_list_of_16_decompositions_is_the_default(self):
        self.assertEqual(list(P.DEFAULT_DECOMPOSITIONS), FIXED16)
        self.assertEqual(P.decomposition_option('default'), FIXED16)
        self.assertTrue(all(a * b * c == 160 for a, b, c in (map(int, t.split('x')) for t in FIXED16)))
        self.assertEqual(sorted(d['decomposition'] for d in self.plan['decompositions']), sorted(FIXED16))
        self.assertEqual((self.plan['skipped'], self.plan['excluded']), ([], []))
        self.assertEqual(self.plan['decomposition_set'], 'default (16)')
        self.assertIn('8x5x4', FIXED16)                     # what MPI_Dims_create gives for 160 ranks
        self.assertEqual(T.quiet(L.main, ['plan', '--output-dir', tempfile.mkdtemp()])[0], 0)

    def test_explicit_and_all_options(self):
        self.assertEqual(P.decomposition_option('4x10x4, 8x5x4'), ['4x10x4', '8x5x4'])
        self.assertEqual(P.decomposition_option('all'), 'all')
        with self.assertRaisesRegex(ValueError, 'Repeated decomposition'):
            P.decomposition_option('8x5x4,8x5x4')
        with self.assertRaisesRegex(ValueError, 'Invalid decomposition'):
            P.decomposition_option('8x5')
        everything = P.make_sweep_plan(dict(CFG, repetitions=1, decompositions='all'))
        self.assertEqual(len(everything['decompositions']) + len(everything['skipped']), 63)
        two = P.make_sweep_plan(dict(CFG, repetitions=1, decompositions=['4x10x4', '8x5x4']))
        self.assertEqual(sorted(d['decomposition'] for d in two['decompositions']), ['4x10x4', '8x5x4'])
        with self.assertRaisesRegex(ValueError, 'has product 80 but the job has 160'):
            P.make_sweep_plan(dict(CFG, decompositions=['4x5x4']))

    def test_product_different_from_ranks_is_rejected(self):
        for bad in ((4, 5, 4), (8, 5, 8), (160, 1, 2)):
            with self.assertRaisesRegex(ValueError, 'has product'):
                P.check_triple(bad, 160)
        with self.assertRaisesRegex(ValueError, 'has product 80 but the job has 160'):
            P.make_run('x', 'sweep', 'D', '4x5x4', CFG, 'full', 20, 160, 100)

    def test_validity_rule_and_halo_constant(self):
        self.assertEqual(P.HALO_DEPTH, 2)          # CloverLeaf_3D build_field.cpp d_p/d_m, OPS ops_mpi_rt_support.cpp
        self.assertEqual(P.validity((160, 1, 1), 1440), (True, ''))
        self.assertFalse(P.validity((160, 1, 1), 160)[0])                      # local width 1 < 2
        self.assertFalse(P.validity((32, 5, 1), 1000)[0])                      # 1000 / 32 is not an integer
        self.assertTrue(P.validity((2, 2, 2), 4)[0])                           # exactly the halo depth
        plan = P.make_sweep_plan(dict(CFG, grid=120, repetitions=1, decompositions=['5x32x1', '8x5x4']))
        self.assertEqual([s['decomposition'] for s in plan['skipped']], ['5x32x1'])
        self.assertIn('not divisible by Py=32', plan['skipped'][0]['reason'])
        with tempfile.TemporaryDirectory() as tmp:
            _, text = T.quiet(L.main, ['plan', '--output-dir', tmp, '--grid', '120', '--decompositions', '5x32x1,8x5x4'])
            self.assertIn('NOTICE skipped 5x32x1: grid 120 is not divisible by Py=32', text)

    def test_dry_run_writes_the_complete_ordered_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text = T.quiet(L.main, ['plan', '--output-dir', tmp, '--repetitions', '3'])
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
        count = lambda stage, mode: sum(1 for r in rows if r['stage'] == stage and r['mode'] == mode)
        self.assertEqual((count('sweep', 'A'), count('sweep', 'D'), count('sweep', 'P')), (16, 48, 48))
        self.assertEqual(count('baseline', 'D'), 3)
        self.assertEqual(count('sentinel', 'D'), 2)      # 1 at the start + after the 8th decomposition
        self.assertEqual(len(rows), 16 + 48 + 48 + 3 + 2)

    def test_default_plan_has_one_repetition_and_three_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text = T.quiet(L.main, ['plan', '--output-dir', tmp])
            self.assertEqual(status, 0)
            plan = json.loads((Path(tmp) / 'plan.json').read_text())
        self.assertEqual(plan, json.loads(P.plan_bytes(P.make_sweep_plan(dict(CFG, repetitions=1))).decode()))
        self.assertEqual((plan['repetitions'], plan['grid'], plan['nodes'], plan['ranks']), (1, 1440, 20, 160))
        count = lambda stage, mode: sum(1 for r in plan['runs'] if r['stage'] == stage and r['mode'] == mode)
        self.assertEqual((count('sweep', 'A'), count('sweep', 'D'), count('sweep', 'P')), (16, 16, 16))
        self.assertEqual((count('baseline', 'D'), count('sentinel', 'D')), (1, 2))
        self.assertIn('  sweep     default       16 runs', text)
        starts = {'D': 0, 'P': 0}
        for dec in plan['decompositions']:
            runs = [r for r in plan['runs'] if r['stage'] == 'sweep' and r['decomposition'] == dec['decomposition']]
            modes = ''.join(r['mode'] for r in runs)
            self.assertEqual(modes, 'ADP' if dec['position'] % 2 == 0 else 'APD')     # one pair, still interleaved
            starts[modes[1]] += 1
        self.assertEqual(starts, {'D': 8, 'P': 8})
        self.assertEqual([r['run_id'] for r in plan['runs'][:2]], ['baseline_unforced_r1', 'sentinel_start'])

    def test_same_seed_is_byte_identical_and_other_seed_differs(self):
        self.assertEqual(P.plan_bytes(P.make_sweep_plan(dict(CFG))), P.plan_bytes(self.plan))
        other = P.make_sweep_plan(dict(CFG, seed=7))
        order = lambda plan: [d['decomposition'] for d in plan['decompositions']]
        self.assertNotEqual(order(other), order(self.plan))
        self.assertEqual(sorted(order(other)), sorted(order(self.plan)))
        fixed = P.make_sweep_plan(dict(CFG, fixed_order=True))
        self.assertEqual(order(fixed), sorted(order(fixed), key=lambda t: [int(v) for v in t.split('x')]))

    def test_autotune_first_and_interleaved_pairs(self):
        for dec in self.plan['decompositions']:
            runs = [r for r in self.sweep if r['decomposition'] == dec['decomposition']]
            self.assertEqual([r['position'] for r in runs], list(range(runs[0]['position'], runs[0]['position'] + 7)))
            modes = ''.join(r['mode'] for r in runs)
            self.assertEqual(modes, 'ADPPDDP' if dec['position'] % 2 == 0 else 'APDDPPD')
            self.assertEqual([r['pair_index'] for r in runs[1:]], [1, 1, 2, 2, 3, 3])
            self.assertEqual(dec['first_pair_mode'], modes[1])

    def test_sentinels_and_baseline_positions(self):
        runs = self.plan['runs']
        self.assertEqual([r['run_id'] for r in runs[:4]],
                         ['baseline_unforced_r1', 'baseline_unforced_r2', 'baseline_unforced_r3', 'sentinel_start'])
        periodic = [r for r in runs if r.get('sentinel_kind') == 'periodic']
        self.assertEqual([r['after_decomp_position'] for r in periodic], [7])
        resolved = P.resolve_d0(periodic[0], [8, 5, 4])
        self.assertIn('OPS_FORCE_DECOMP_X=8', resolved['argv'])
        self.assertIn('OPS_FORCE_DECOMP_Z=4', resolved['argv_other_ranks'])
        with self.assertRaisesRegex(ValueError, 'has product'):
            P.resolve_d0(periodic[0], [8, 5, 5])

    def test_arguments(self):
        for run in self.plan['runs']:
            for argv in (run['argv'], run['argv_other_ranks']):
                self.assertTrue(all(len(token) <= 63 for token in argv))
                self.assertIn('-gpudirect', argv)
                self.assertIn('-OPS_DIAGS=2', argv)
                self.assertTrue(all(b in argv for b in ('OPS_BLOCK_SIZE_X=32', 'OPS_BLOCK_SIZE_Y=4', 'OPS_BLOCK_SIZE_Z=1')))
        with self.assertRaisesRegex(ValueError, 'is 64 characters long'):
            P.check_argv(['OPS_FORCE_DECOMP_X=' + '1' * 45])
        for run in self.sweep:
            x, y, z = run['triple']
            want = {'D': (0, 0), 'A': (1, 0), 'P': (2, 2)}[run['mode']]
            self.assertIn('-OPS_AUTOTUNE_MODE=%d' % want[0], run['argv'])
            self.assertIn('-OPS_AUTOTUNE_MODE=%d' % want[1], run['argv_other_ranks'])
            for token in ('OPS_FORCE_DECOMP_X=%d' % x, 'OPS_FORCE_DECOMP_Y=%d' % y, 'OPS_FORCE_DECOMP_Z=%d' % z):
                self.assertEqual(run['argv'].count(token), 1)
        for run in self.plan['runs']:
            if run['stage'] == 'baseline':
                self.assertFalse(any('OPS_FORCE_DECOMP' in t for t in run['argv'] + run['argv_other_ranks']))

    def test_exclusion_filter_is_explicit(self):
        plan = P.make_sweep_plan(dict(CFG, exclude=['4x40x1']))
        self.assertEqual((len(plan['decompositions']), plan['excluded']),
                         (15, [dict(decomposition='4x40x1', reason='explicit --exclude')]))
        with self.assertRaisesRegex(ValueError, 'not in the planned set'):
            P.make_sweep_plan(dict(CFG, exclude=['1x1x160']))

    def test_qualification_sequence(self):
        smoke = P.make_smoke_plan(CFG)
        got = [(r['stage'], r['run_id'], r['build'], r['nodes'], r['ranks'], r['steps']) for r in smoke['runs']]
        self.assertEqual(got, [
            ('Q1', 'q1_1x1x1_D', 'small', 1, 1, 100), ('Q2', 'q2_1x1x1_A', 'small', 1, 1, 100),
            ('Q2', 'q2_1x1x1_P', 'small', 1, 1, 100), ('Q3', 'q3_2x2x2_D', 'small', 1, 8, 100),
            ('Q3', 'q3_8x1x1_D', 'small', 1, 8, 100), ('Q3', 'q3_1x8x1_D', 'small', 1, 8, 100),
            ('Q3', 'q3_1x1x8_D', 'small', 1, 8, 100), ('Q3', 'q3_unforced_D', 'small', 1, 8, 100),
            ('Q3', 'q3_2x2x2_D_nogpudirect', 'small', 1, 8, 100), ('Q4', 'q4_unforced_D', 'full', 20, 160, 10),
            ('Q4', 'q4_2x40x2_D', 'full', 20, 160, 10),      # thinnest planned triple (local width 36)
            ('Q5', 'q5_d0_A', 'full', 20, 160, 100), ('Q5', 'q5_d0_1_D', 'full', 20, 160, 100),
            ('Q5', 'q5_d0_2_P', 'full', 20, 160, 100), ('Q5', 'q5_d0_3_P', 'full', 20, 160, 100),
            ('Q5', 'q5_d0_4_D', 'full', 20, 160, 100)])
        self.assertEqual(sorted(smoke['stages']), ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        self.assertEqual(sum('-gpudirect' not in r['argv'] for r in smoke['runs']), 1)

    def test_run_input_file(self):
        text = S.clover_in(1440, 1234)
        v = S.clover_in_values(text)
        self.assertEqual((v['x_cells'], v['y_cells'], v['z_cells'], v['end_step']), ('1440', '1440', '1440', '1234'))
        self.assertEqual((v['xmax'], v['ymax'], v['zmax']), ('10.0', '10.0', '10.0'))      # cubic domain
        self.assertEqual((v['test_problem'], v['profiler_on']), ('0', '1'))
        self.assertGreater(float(v['end_time']), 0.04 * 100000)       # end_step alone ends the run
        self.assertTrue(text.startswith('*clover\n') and text.endswith('*endclover\n'))
        self.assertIn('geometry=cuboid xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0 zmin=0.0 zmax=2.0', text)
        with self.assertRaises(ValueError):
            S.clover_in(1440, 0)


if __name__ == '__main__':
    unittest.main()
