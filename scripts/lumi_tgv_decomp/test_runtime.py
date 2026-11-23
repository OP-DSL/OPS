"""Catalog construction, run validation, mode wiring and the run record (criteria 8-12, 16)."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import engine as E
import launcher as L
import plan as P
import runtime as R
import testlib as T
from common import read_json, sha

FOOTER = '\n\nAutotune -- 1\ntotal of rows -- %d\napp name -- tgvstorenone_mpi_hip\ndefault total time -- 1\nbest total time -- 1\n'
HEADER = 'kernel_id,bx,by,bz,best_time,default_time,points,max_threads_per_block\n'


def attempt_with(tmp, best_text, executed=None):
    """An autotune attempt folder holding rank 0's CSVs."""
    attempt = Path(tempfile.mkdtemp(dir=str(tmp)))
    folder = attempt / 'rank_0/autotune_on'
    folder.mkdir(parents=True)
    if best_text is not None:
        (folder / 'ops_blocksize_best_logmod.csv').write_text(best_text)
    ids = executed if executed is not None else sorted(T.fixture_ids() + T.EXTRA_IDS)
    (folder / 'ops_blocksize_tuning_logmod.csv').write_text(
        'kernel_id,bx,by,bz\n' + ''.join('%d,32,4,1\n%d,16,8,1\n' % (k, k) for k in ids) + '\nAutotune -- 1\n')
    return attempt


META = dict(decomposition='4x5x8', executable_sha256='e' * 64, source_run_id='x_A', scope='sweep', build='full')


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tgv_catalog_')
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, text, executed=None):
        return R.build_catalog(attempt_with(self.tmp, text, executed), self.tmp / 'catalog', dict(META))

    def test_real_lumi_best_csv_becomes_a_catalog(self):
        source = T.BEST_FIXTURE.read_text()
        self.assertEqual(sum(1 for line in source.splitlines() if line.startswith(tuple(R.FOOTERS))), 5)
        manifest = self.build(source)
        blocks = (self.tmp / 'catalog/blocks.csv').read_text().splitlines()
        self.assertEqual(blocks[0], 'kernel_id,bx,by,bz')
        self.assertEqual(len(blocks) - 1, T.FIXTURE_ROWS)
        ids = [int(line.split(',')[0]) for line in blocks[1:]]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(ids, sorted(T.fixture_ids()))
        self.assertTrue(all(len(line.split(',')) == 4 and all(v.isdigit() and int(v) > 0 for v in line.split(','))
                            for line in blocks[1:]))
        self.assertTrue(all(int(a) * int(b) * int(c) <= 1024 for _, a, b, c in (l.split(',') for l in blocks[1:])))
        self.assertIn('2,1,64,4', blocks)                              # first data row of the LUMI file
        self.assertIn('15,16,1,16', blocks)
        self.assertEqual((manifest['rows'], manifest['sha256']), (T.FIXTURE_ROWS, sha(self.tmp / 'catalog/blocks.csv')))
        self.assertEqual(manifest['fallback_kernel_ids'], T.EXTRA_IDS)
        self.assertEqual(manifest['footer']['total of rows'], str(T.FIXTURE_ROWS))
        self.assertEqual(manifest['decomposition'], '4x5x8')
        self.assertEqual(oct((self.tmp / 'catalog/blocks.csv').stat().st_mode & 0o777), '0o444')
        loaded = R.load_catalog(self.tmp / 'catalog', '4x5x8', 'e' * 64)
        self.assertEqual((loaded['rows'], loaded['sha256']), (T.FIXTURE_ROWS, manifest['sha256']))

    def test_one_row_per_kernel_lowest_best_time(self):
        text = HEADER + '7,32,4,1,0.5,0.5,10,1024\n7,16,8,1,0.2,0.5,10,1024\n7,8,8,2,0.9,0.5,10,1024\n9,32,1,1,0.1,0.1,5,256\n' + FOOTER % 4
        manifest = self.build(text, [7, 9, 11])
        self.assertEqual((self.tmp / 'catalog/blocks.csv').read_text(), 'kernel_id,bx,by,bz\n7,16,8,1\n9,32,1,1\n')
        self.assertEqual((manifest['duplicates_reduced'], manifest['fallback_kernel_ids']), (2, [11]))

    def failing(self, text, pattern, executed=None):
        with self.assertRaisesRegex(R.CatalogError, pattern):
            self.build(text, executed)
        self.assertFalse((self.tmp / 'catalog').exists())            # nothing is ever written on failure

    def test_missing_empty_and_footer_only_files_fail(self):
        self.failing(None, 'Best-block CSV is missing')
        self.failing('', 'Best-block CSV is empty')
        self.failing(HEADER + FOOTER % 0, 'no kernel rows')
        self.failing(HEADER, 'no kernel rows')

    def test_invalid_and_ambiguous_rows_fail(self):
        row = '7,32,4,1,0.5,0.5,10,1024\n'
        self.failing(HEADER + row + '7,16,8,1,0.5,0.5,10,1024\n', 'Duplicate kernel id 7', [7])
        self.failing(HEADER + '7,64,32,1,0.5,0.5,10,1024\n', 'exceeds the thread limit', [7])
        self.failing(HEADER + '7,32,16,1,0.5,0.5,10,256\n', r'exceeds the thread limit min\(1024, 256\)', [7])
        self.failing(HEADER + '7,0,4,1,0.5,0.5,10,1024\n', 'Invalid block', [7])
        self.failing(HEADER + '7,32,4,1,nan,0.5,10,1024\n', 'Invalid block or time', [7])
        self.failing(HEADER + '7,32,x,1,0.5,0.5,10,1024\n', 'Non-numeric', [7])
        self.failing(HEADER + '7,32,4\n', 'Malformed row', [7])
        self.failing(HEADER + row + FOOTER % 3, 'footer declares 3 rows but 1', [7])
        self.failing(HEADER + row, 'absent from the tuning log', [8])
        self.failing('id,x,y,z\n' + row, 'Unexpected header', [7])

    def test_missing_tuning_log_fails(self):
        attempt = attempt_with(self.tmp, HEADER + '7,32,4,1,0.5,0.5,10,1024\n', [7])
        (attempt / R.TUNING_CSV).unlink()
        with self.assertRaisesRegex(R.CatalogError, 'Tuning log is missing'):
            R.build_catalog(attempt, self.tmp / 'catalog', dict(META))
        self.assertFalse((self.tmp / 'catalog').exists())

    def test_frozen_catalog_is_never_rebuilt_or_reused_when_changed(self):
        self.build(T.BEST_FIXTURE.read_text())
        with self.assertRaisesRegex(R.CatalogError, 'never rebuilt'):
            self.build(T.BEST_FIXTURE.read_text())
        with self.assertRaisesRegex(ValueError, 'belongs to decomposition 4x5x8, not 5x4x8'):
            R.load_catalog(self.tmp / 'catalog', '5x4x8', 'e' * 64)
        with self.assertRaisesRegex(ValueError, 'another executable'):
            R.load_catalog(self.tmp / 'catalog', '4x5x8', 'f' * 64)
        blocks = self.tmp / 'catalog/blocks.csv'
        os.chmod(str(blocks), 0o644)
        blocks.write_text(blocks.read_text().replace('15,16,1,16', '15,32,4,1'))
        with self.assertRaisesRegex(ValueError, 'changed on disk'):
            R.load_catalog(self.tmp / 'catalog', '4x5x8', 'e' * 64)

    def test_only_one_code_path_writes_a_catalog(self):
        here = Path(__file__).resolve().parent
        writers = []
        for name in L.PACKAGE_FILES:
            text = (here / name).read_text()
            if 'blocks.csv' in text and name.endswith('.py'):
                writers += [name for line in text.splitlines() if 'blocks' in line and ('atomic_bytes(' in line or 'write_' in line)]
        self.assertEqual(writers, ['runtime.py'])                    # the single atomic_bytes(blocks, ...) in build_catalog
        self.assertEqual(sum(text.count('build_catalog(') for text in
                             ((here / n).read_text() for n in L.PACKAGE_FILES if n.endswith('.py'))), 2)   # definition + Engine.freeze


class RunTests(unittest.TestCase):
    """One real pass of the job loop with the mock executable; tests then alter copies of its logs."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='tgv_runs_')
        cls.tmp = Path(cls.temp.name)
        cls.campaign = T.make_campaign(cls.tmp, grid=10, reps=1, decompositions='2x2x2')
        T.fake_build(cls.campaign)
        T.quiet(L.main, ['plan', '--campaign', str(cls.campaign)])
        cls.job_status, cls.text = T.run_job(cls.campaign, 'sweep', T.job_env(cls.tmp))
        cls.layout = T.layout(cls.campaign)
        cls.run_ids = [r['run_id'] for r in read_json(cls.campaign / 'plan.json')['runs']]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def attempt(self, suffix):
        found = [r for r in self.run_ids if r.endswith(suffix)]
        self.assertEqual(len(found), 1, suffix)
        folder = T.sweep_folder(self.campaign, found[0], 1)
        self.assertEqual(folder, self.layout.locate(found[0])['final'])       # committed
        return folder

    def dataset_csvs(self, folder):
        return {'mpi_rank_metrics.csv'} | {'ranks/metric_%d.csv' % r for r in range(read_json(folder / 'record.json')['nranks'])}

    def check(self, suffix, mutate=None, **overrides):
        source = self.attempt(suffix)
        spec = read_json(source / 'execution.json')
        with tempfile.TemporaryDirectory(dir=str(self.tmp)) as scratch:
            copy = T.copy_tree(source, Path(scratch) / 'attempt')
            if mutate:
                mutate(copy)
            expect = dict(srun_returncode=0, timed_out=False, catalog=spec['catalog'])
            expect.update(overrides)
            return R.validate_run(copy, spec['run'], expect)

    @staticmethod
    def edit(rank, old, new, count=1):
        def mutate(attempt):
            path = attempt / 'ranks' / ('rank_%d.log' % rank)
            text = path.read_text()
            assert old in text, old
            path.write_text(text.replace(old, new, count))
        return mutate

    @staticmethod
    def append(rank, line):
        def mutate(attempt):
            with (attempt / 'ranks' / ('rank_%d.log' % rank)).open('a') as f:
                f.write(line + '\n')
        return mutate

    @staticmethod
    def status(rank, **changes):
        def mutate(attempt):
            path = attempt / 'ranks' / ('status_%d.json' % rank)
            path.write_text(json.dumps(dict(read_json(path), **changes)))
        return mutate

    def rejected(self, result, category, pattern):
        self.assertFalse(result['valid'])
        self.assertEqual(result['category'], category, result['reasons'])
        self.assertRegex(' | '.join(result['reasons']), pattern)

    def test_the_mock_pass_itself_is_valid(self):
        self.assertEqual(self.job_status, 0, self.text)
        self.assertEqual(T.srun_calls(self.tmp), ['baseline_unforced_r1', 'sentinel_start', 'sweep_p000_2x2x2_A',
                                                  'sweep_p000_2x2x2_pair1_D', 'sweep_p000_2x2x2_pair1_P'])

    def test_valid_default_and_precomputed_runs_are_accepted(self):
        for suffix in ('pair1_D', 'pair1_P', '2x2x2_A', 'baseline_unforced_r1'):
            result = self.check(suffix)
            self.assertTrue(result['valid'], result['reasons'])
            self.assertEqual(result['category'], '')
        m = self.check('pair1_P')['metrics']
        self.assertEqual((m['catalog_rows'], m['fallback_kernel_ids_rank0']), (T.FIXTURE_ROWS, T.EXTRA_IDS))
        self.assertEqual(m['printed_grid'], [2, 2, 2])
        self.assertEqual(m['steps_seen'], 3)

    def test_missing_wall_time(self):
        self.rejected(self.check('pair1_D', self.edit(0, 'Total Wall time', 'Total Wall clock')), 'OTHER',
                      'exactly one positive "Total Wall time"')
        self.rejected(self.check('pair1_D', self.append(0, 'Total Wall time 5.000000')), 'OTHER', 'found 2')
        self.rejected(self.check('pair1_D', self.edit(0, 'Total kernel time', 'Kernel time')), 'OTHER', 'Total kernel time')

    def test_nonzero_exit(self):
        self.rejected(self.check('pair1_D', self.status(3, returncode=1), srun_returncode=1), 'OTHER',
                      r'srun exit status 1.*nonzero exit status on rank\(s\) 3')

    def test_missing_gpu_direct(self):
        self.rejected(self.check('pair1_D', self.edit(0, 'GPU Direct enabled', 'GPU Direct off')), 'OTHER', 'GPU Direct enabled')

    def test_printed_grid_differs_from_forced_triple(self):
        self.rejected(self.check('pair1_D', self.edit(0, 'grid of 2 x 2 x 2', 'grid of 4 x 2 x 1')), 'OTHER',
                      'OPS used processor grid 4x2x1 but 2x2x2 was forced')
        self.rejected(self.check('baseline_unforced_r1', self.edit(0, 'grid of 2 x 2 x 2', 'grid of 4 x 2 x 2')), 'OTHER',
                      'does not use 8 ranks')

    def test_fewer_rank_records(self):
        self.rejected(self.check('pair1_D', lambda a: (a / 'ranks/status_7.json').unlink()), 'OTHER', '7 of 8 per-rank status records')

    def test_duplicated_host_gpu_pair(self):
        first = read_json(self.attempt('pair1_D') / 'ranks/status_0.json')
        self.rejected(self.check('pair1_D', self.status(1, host=first['host'], gpu_id=first['gpu_id'])), 'OTHER', 'ranks share a GPU')

    def test_overpartitioning_text(self):
        self.rejected(self.check('pair1_D', self.append(5, 'Error: overpartitioning! Trying to exchange a 2-deep halo')),
                      'OVERPARTITION', 'failure text in the log of rank 5')

    def test_out_of_memory_text(self):
        self.rejected(self.check('pair1_D', self.append(2, 'hipErrorOutOfMemory: out of memory')), 'OOM', 'out of memory')

    def test_nan_and_error_text(self):
        self.rejected(self.check('pair1_D', self.append(0, ' residual = NaN')), 'OTHER', 'NaN')
        self.rejected(self.check('pair1_D', self.append(6, '[OPS] ERROR: failed to create directory')), 'OTHER', 'ERROR')
        ok = self.check('pair1_D', self.append(0, 'opensbliblock00Kernel099  99    10    0.000000 (nan) 0.0 (0.0)  nan'))
        self.assertTrue(ok['valid'], ok['reasons'])                  # derived kernel-table columns (deviation, bandwidth)
        self.rejected(self.check('pair1_D', self.append(0, 'opensbliblock00Kernel099  99    10    nan (0.000000) 0.0 (0.0)  1.0')),
                      'OTHER', 'non-finite time in the OPS kernel table: opensbliblock00Kernel099 nan')

    def test_precomputed_loaded_count_differs_from_catalog(self):
        self.rejected(self.check('pair1_P', self.edit(2, 'Loaded 25 precomputed', 'Loaded 24 precomputed')), 'OTHER',
                      r'rank\(s\) 2 did not load exactly the 25 catalog rows')
        self.rejected(self.check('pair1_P', self.edit(4, '[OPS] Loaded', '[OPS] Read')), 'OTHER', r'rank\(s\) 4 did not load')

    def test_precomputed_wrong_catalog_path(self):
        self.rejected(self.check('pair1_P', self.edit(3, 'decomp2x2x2/blocks.csv', 'decomp8x1x1/blocks.csv')), 'OTHER',
                      r'rank\(s\) 3 did not load.*decomp8x1x1')

    def test_precomputed_fallback_set_differs(self):
        template = '[OPS] WARNING: kernel %d not found in CSV, using default (32,4,1)'
        warning = template % 1 + '\n'                                 # kernel 1: the untuned initialisation loop
        self.rejected(self.check('pair1_P', self.edit(0, warning, '')), 'OTHER', 'rank 0 fell back to default blocks')
        self.rejected(self.check('pair1_P', self.append(0, template % 7777)), 'OTHER', 'rank 0 fell back')
        self.rejected(self.check('pair1_P', self.append(5, template % 18)), 'OTHER',
                      'non-root rank reported catalog kernels as missing')

    def test_default_run_containing_a_loaded_line(self):
        self.rejected(self.check('pair1_D', self.append(4, '[OPS] Loaded 25 precomputed block sizes from /x/blocks.csv')),
                      'OTHER', r'rank\(s\) \[4\] loaded precomputed blocks in a default run')
        self.rejected(self.check('2x2x2_A', self.append(1, '[OPS] Loaded 3 precomputed block sizes from /x/blocks.csv')),
                      'OTHER', 'loaded precomputed blocks in a autotune run')

    def test_diagnostics_from_the_kernel_table(self):
        # 3 time steps: 1 initialisation call + 6 boundary loops x 4 + 1 x 1 + 18 Runge-Kutta loops x 3 per step
        m = self.check('pair1_D')['metrics']
        self.assertEqual(m['diagnostics'], dict(step=3, kernels=26, kernel_calls=79 * 3 + 1))
        self.assertEqual((len(m['kernel_counts']), m['kernel_counts']['opensbliblock00Kernel033'],
                          m['kernel_counts']['opensbliblock00Kernel032'], m['kernel_counts']['opensbliblock00Kernel026'],
                          m['kernel_counts']['opensbliblock00Kernel024']), (26, 3, 1, 12, 9))
        self.assertEqual(len(m['kernel_times']), 26)

        def without_table(attempt):
            path = attempt / 'ranks/rank_0.log'
            path.write_text(R.RE_KERNEL_LINE.sub('', path.read_text()))
        self.rejected(self.check('pair1_D', without_table), 'OTHER', 'no OPS kernel table')
        self.rejected(self.check('pair1_D', self.edit(0, 'opensbliblock00Kernel033', 'opensbliblock00Kernel933')), 'OTHER',
                      'no opensbliblock00Kernel033 row')
        row = next(l for l in (self.attempt('pair1_D') / 'ranks/rank_0.log').read_text().splitlines()
                   if l.startswith('opensbliblock00Kernel024'))
        self.rejected(self.check('pair1_D', self.append(0, row)), 'OTHER', 'opensbliblock00Kernel024 appears twice')
        more = dict(m['diagnostics'], kernel_calls=m['diagnostics']['kernel_calls'] + 1)
        self.assertAlmostEqual(R.relative_difference(more, m['diagnostics']), 1.0 / 239, places=12)
        self.assertEqual(R.relative_difference(m['diagnostics'], m['diagnostics']), 0.0)
        self.assertIsNone(R.relative_difference({}, m['diagnostics']))

    def test_real_lumi_kernel_table(self):
        # rows copied from a LUMI run.log (tgvstorenone 1000x800x640, 2x1x4, 500 time steps)
        text = ('Name                      Count Time     MPI-time     Bandwidth(GB/s)\n'
                'Name                    -------------------------------------------------------\n'
                'opensbliblock00Kernel032  1     1     0.008138 (0.000207) 0.004667 (0.000490)  308.97       \n'
                'opensbliblock00Kernel026  2     2000  0.336723 (0.334750) 0.093096 (0.031815)  57.64        \n'
                'opensbliblock00Kernel033  8     500   2.837161 (0.041755) 0.025814 (0.017942)  840.30       \n'
                'opensbliblock00Kernel024  23    1500  66.800773 (0.617914) 30.581880 (5.537190)  182.01       \n'
                'Total kernel time: 212.331\n')
        diagnostics, counts, problem = R.read_diagnostics(text)
        self.assertEqual((diagnostics, problem), (dict(step=500, kernels=4, kernel_calls=4001), ''))
        self.assertEqual(counts['opensbliblock00Kernel026'], 2000)
        self.assertEqual(R.kernel_table(text), {1: 0.008138, 2: 0.336723, 8: 2.837161, 23: 66.800773})

    def test_fewer_steps(self):
        self.rejected(self.check('pair1_D', self.edit(0, 'opensbliblock00Kernel033  8     3 ', 'opensbliblock00Kernel033  8     2 ')),
                      'OTHER', 'the once-per-step loop opensbliblock00Kernel033 ran 2 times, not 3')

    def test_timeout_and_wrong_mode(self):
        self.rejected(self.check('pair1_D', timed_out=True, srun_returncode=-15), 'TIMEOUT', 'wall-clock limit')
        self.rejected(self.check('pair1_D', self.status(2, autotune_mode=1)), 'OTHER', 'wrong autotune mode')
        self.rejected(self.check('pair1_D', self.edit(0, 'OPS_autotune_mode = 0', 'OPS_autotune_mode = 1')), 'OTHER',
                      'did not report OPS_autotune_mode = 0')

    def test_record_has_every_field_of_the_design(self):
        for suffix in ('pair1_D', 'pair1_P', '2x2x2_A'):
            record = read_json(self.attempt(suffix) / 'record.json')
            missing = [f for f in R.RECORD_FIELDS if f not in record]
            self.assertEqual(missing, [], suffix)
        record = read_json(self.attempt('pair1_P') / 'record.json')
        config = read_json(self.campaign / 'campaign.json')
        plan_sha = (self.campaign / 'plan.json.sha256').read_text().split()[0]
        self.assertEqual((record['campaign_id'], record['plan_hash'], record['seed']), ('campaign', plan_sha, P.DEFAULT_SEED))
        self.assertEqual((record['decomposition'], record['mode'], record['repetition'], record['pair_index'],
                          record['order_in_pair'], record['stage']), ('2x2x2', 'P', 1, 1, 2, 'sweep'))
        self.assertEqual((record['slurm_job_id'], record['node_list']), ('1001', 'nid[005000-005000]'))
        self.assertRegex(record['slurm_step_id'], r'^\d+$')
        self.assertLess(record['start_utc'], record['end_utc'] + 'z')
        self.assertIn('-OPS_AUTOTUNE_MODE=2', record['argv'])
        env = record['environment']['rank_0']
        self.assertEqual((env['MPICH_GPU_SUPPORT_ENABLED'], env['ROCR_VISIBLE_DEVICES']), ('1', '0'))
        self.assertEqual(env['OPS_BLOCKSIZE_CSV'], str(self.layout.catalog_folder('2x2x2') / 'blocks.csv'))
        self.assertTrue(env['OPS_BLOCKSIZE_CSV'].endswith('precomputed_blocks/tgvstorenone_hip_10x10x10_cfg1/mode_1/decomp2x2x2/blocks.csv'))
        self.assertEqual(record['environment']['cpu_bind'], 'map_cpu:' + R.CPU_BIND)
        self.assertEqual(record['executable_sha256'], sha(self.campaign / 'build/hip/tgvstorenone_mpi_hip'))
        self.assertEqual((record['catalog_sha256'], record['catalog_rows']),
                         (sha(self.layout.catalog_folder('2x2x2') / 'blocks.csv'), T.FIXTURE_ROWS))
        self.assertEqual((record['package_commit'], record['ops_commit']), (config['package']['commit'], config['ops_commit']))
        self.assertRegex(record['ops_commit'], '^[0-9a-f]{40}$')
        self.assertEqual((record['app_args'], record['argv'][:4]), ('10 10 10 3', ['10', '10', '10', '3']))
        self.assertEqual(record['module_list'], 'LUMI/test:partition/G:rocm/test')
        self.assertEqual(record['printed_grid'], [2, 2, 2])
        self.assertGreater(record['wall_time_s'], 0)
        self.assertTrue(all(record[k] > 0 for k in ('kernel_time_s', 'halo_time_s', 'mpi_comm_volume_mb')))
        self.assertIsNone(record['user_halo_time_s'])                # optional OPS line: TGV has no user halo exchange
        self.assertEqual(record['diagnostics'], dict(step=3, kernels=26, kernel_calls=238))
        self.assertEqual(len(record['ranks']), 8)
        self.assertEqual(len({(r['host'], r['gpu_id']) for r in record['ranks']}), 8)
        self.assertEqual((record['verdict'], record['reason']), ('VALID', ''))

    def test_every_run_has_a_fresh_working_directory_and_no_input_file(self):
        folders = [p.parent for p, _ in self.layout.record_paths()]
        self.assertEqual(len(folders), 5)
        for attempt in folders:
            self.assertEqual(list((attempt / 'work').iterdir()), [])      # the application reads and writes no file
            spec = read_json(attempt / 'execution.json')
            self.assertEqual(spec['run']['argv'][:4], ['10', '10', '10', '3'])
            self.assertTrue(spec['workdir'].endswith('/work'))

    def test_logging_is_identical_for_default_and_precomputed(self):
        def logging(suffix):
            out = set()
            for path in (self.attempt(suffix) / 'ranks').glob('status_*.json'):
                env = read_json(path)['environment']
                out.add(tuple(env[k] for k in ('OPS_AUTOTUNE_LOGS', 'OPS_DISABLE_CSV_LOGS', 'OPS_CSV_LOG_LEVEL')))
                self.assertTrue(env['OPS_LOGS_DIR'].startswith(str(self.campaign)))
            return out
        self.assertEqual(logging('pair1_D'), {('0', '1', '2')})
        self.assertEqual(logging('pair1_P'), {('0', '1', '2')})
        def csvs(suffix):
            folder = self.attempt(suffix)
            return {p.relative_to(folder).as_posix() for p in folder.glob('**/*.csv')}, self.dataset_csvs(folder)
        found, data = csvs('pair1_D')
        self.assertEqual(found, data)                                # no OPS tuning CSV, only the dataset tables
        found, data = csvs('pair1_P')
        self.assertEqual(found, data | {'blocks.csv'})               # + the copy of its own catalog
        found, data = csvs('2x2x2_A')
        self.assertEqual(found, data | {R.BEST_CSV, R.TUNING_CSV})


class WiringTests(unittest.TestCase):
    """Criterion 8 at the real scale: 160 ranks on 20 fake nodes through the fake srun."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='tgv_wiring_')
        cls.tmp = Path(cls.temp.name)
        cls.campaign = T.make_campaign(cls.tmp, nodes=20, grid=1440, steps=2, reps=1, decompositions='default')
        T.fake_build(cls.campaign)
        T.quiet(L.main, ['plan', '--campaign', str(cls.campaign)])
        c = E.Campaign(cls.campaign)
        plan, digest = c.plan('sweep')
        cls.records = {}
        with patch.dict(os.environ, T.job_env(cls.tmp, nodes=20)):
            engine = E.Engine(c, 'sweep', plan, digest)
            text = '8x5x4'
            runs = {r['mode']: r for r in plan['runs'] if r['decomposition'] == text and r.get('pair_index') in (None, 1)}
            with patch('builtins.print'):
                cls.records['A'] = engine.execute(runs['A'])
                catalog, _ = engine.catalog_of(runs['A'], cls.records['A'])
                cls.records['D'] = engine.execute(runs['D'])
                cls.records['P'] = engine.execute(runs['P'], catalog)
        cls.catalog = catalog
        cls.campaign_layout = c.layout

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def statuses(self, mode):
        record = self.records[mode]
        self.assertEqual(record['run_dir'], str(self.campaign_layout.locate(record['run_id'])['final']))
        folder = Path(record['run_dir']) / 'ranks'
        out = [read_json(folder / ('status_%d.json' % rank)) for rank in range(160)]
        self.assertEqual(record['verdict'], 'VALID', record['reason'])
        return out

    def test_default_run(self):
        for s in self.statuses('D'):
            self.assertIn('-OPS_AUTOTUNE_MODE=0', s['argv'])
            self.assertTrue(all(t in s['argv'] for t in ('OPS_BLOCK_SIZE_X=32', 'OPS_BLOCK_SIZE_Y=4', 'OPS_BLOCK_SIZE_Z=1',
                                                         'OPS_FORCE_DECOMP_X=8', 'OPS_FORCE_DECOMP_Y=5', 'OPS_FORCE_DECOMP_Z=4',
                                                         '-gpudirect', '-OPS_DIAGS=2')))
            self.assertNotIn('OPS_BLOCKSIZE_CSV', s['environment'])
            self.assertEqual((s['environment']['OPS_DISABLE_CSV_LOGS'], s['environment']['OPS_CSV_LOG_LEVEL']), ('1', '2'))

    def test_autotune_run_tunes_rank_zero_only(self):
        statuses = self.statuses('A')
        self.assertIn('-OPS_AUTOTUNE_MODE=1', statuses[0]['argv'])
        env = statuses[0]['environment']
        self.assertEqual((env['OPS_AUTOTUNE_LOGS'], env['OPS_DISABLE_CSV_LOGS'], env['OPS_CSV_LOG_LEVEL']), ('1', '0', '0'))
        self.assertTrue(env['OPS_LOGS_DIR'].endswith('/rank_0'))
        for s in statuses[1:]:
            self.assertIn('-OPS_AUTOTUNE_MODE=0', s['argv'])
            self.assertTrue(all(t in s['argv'] for t in ('OPS_BLOCK_SIZE_X=32', 'OPS_BLOCK_SIZE_Y=4', 'OPS_BLOCK_SIZE_Z=1')))
            e = s['environment']
            self.assertEqual((e['OPS_AUTOTUNE_LOGS'], e['OPS_DISABLE_CSV_LOGS'], e['OPS_CSV_LOG_LEVEL']), ('0', '1', '2'))
        attempt = Path(self.records['A']['run_dir'])
        dataset = {'mpi_rank_metrics.csv'} | {'ranks/metric_%d.csv' % r for r in range(160)}
        self.assertEqual({p.relative_to(attempt).as_posix() for p in attempt.glob('**/*.csv')},
                         dataset | {R.BEST_CSV, R.TUNING_CSV})

    def test_precomputed_run_uses_its_own_catalog_on_every_rank(self):
        own = str(self.campaign_layout.catalog_folder('8x5x4') / 'blocks.csv')
        for s in self.statuses('P'):
            self.assertIn('-OPS_AUTOTUNE_MODE=2', s['argv'])
            self.assertEqual(s['environment']['OPS_BLOCKSIZE_CSV'], own)
        self.assertEqual(self.catalog['path'], own)

    def test_logs_stay_inside_the_campaign_and_gpus_are_distinct(self):
        for mode in 'ADP':
            statuses = self.statuses(mode)
            self.assertTrue(all(s['environment']['OPS_LOGS_DIR'].startswith(str(self.campaign) + '/') for s in statuses))
            self.assertEqual(len({(s['host'], s['gpu_id']) for s in statuses}), 160)
            self.assertEqual(len({s['host'] for s in statuses}), 20)
            self.assertTrue(all(s['environment']['ROCR_VISIBLE_DEVICES'] == str(s['local_rank']) for s in statuses))
            self.assertTrue(all(s['environment']['MPICH_GPU_SUPPORT_ENABLED'] == '1' for s in statuses))
        command = self.records['D']['srun_command']
        self.assertIn('--nodes=20 --ntasks=160 --ntasks-per-node=8 --cpus-per-task=7 --kill-on-bad-exit=1 '
                      '--cpu-bind=map_cpu:49,57,17,25,1,9,33,41', command)


class SrunCommandTests(unittest.TestCase):
    """The step must own the cores of the CPU map, also with fewer than 8 ranks per node."""

    def options(self, ranks, nodes):
        command = R.srun_command(dict(ranks=ranks, nodes=nodes), '/launcher', '/spec.json', R.CPU_BIND)
        return {t.split('=', 1)[0]: t.split('=', 1)[1] for t in command if t.startswith('--') and '=' in t}

    def test_full_nodes_keep_seven_cores_per_rank(self):
        o = self.options(160, 20)
        self.assertEqual(o['--cpus-per-task'], '7')
        self.assertEqual(o['--cpu-bind'], 'map_cpu:49,57,17,25,1,9,33,41')

    def test_single_rank_step_owns_the_whole_node(self):
        # LUMI refused map_cpu:49 for a 1-task step of 7 cores (cores 1-7 were allocated).
        o = self.options(1, 1)
        self.assertEqual(o['--cpus-per-task'], '56')
        self.assertEqual(o['--cpu-bind'], 'map_cpu:49')

    def test_partial_node_steps_cover_all_usable_cores(self):
        for per_node in (1, 2, 4, 8):
            o = self.options(per_node, 1)
            self.assertEqual(int(o['--cpus-per-task']) * per_node, 56)
            self.assertEqual(len(o['--cpu-bind'].split(':')[1].split(',')), per_node)


if __name__ == '__main__':
    unittest.main()
