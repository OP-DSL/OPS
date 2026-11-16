"""Dataset layout of the sweep (the TGV decomposition-campaign structure) and the upgrade of an
existing campaign. Brief criteria 1-7, 9 and 10 of run senga2_dataset_layout_20261116."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import dataset as D
import engine as E
import launcher as L
import plan as P
import runtime as R
import testlib as T
from common import read_json, sha

# Key order of a run config.txt, copied from the TGV campaign files (OPS_LOGS_20260928_161749_n006,
# .../decomp3x4x4/iter_1/autotune_precomputed/config.txt); the PRECOMPUTED_* pair only in precomputed runs.
TGV_RUN_KEYS = [
    'APP_NAME', 'APP_ARGS', 'CAMPAIGN_ID', 'CASE_ID', 'ATTEMPT_ID', 'RUN_CLASS', 'SCIENTIFIC_RESULT', 'NITER', 'GPU_ARCH',
    'EXECUTABLE_SHA256', 'DECOMPOSITION_MODE', 'GRID_X', 'GRID_Y', 'GRID_Z', 'GRID_CONFIG', 'TOTAL_CELLS', 'NUM_NODES',
    'GPUS_PER_NODE', 'NUM_GPUS_USED', 'NUM_MPI_RANKS', 'DECOMPOSITION_POLICY', 'RANKS_PER_GPU', 'DECOMP_PX', 'DECOMP_PY',
    'DECOMP_PZ', 'COORD_SOURCE', 'OPS_AUTOTUNE_MODE', 'AUTOTUNE_VARIANT', 'SAMPLED_AUTOTUNE', 'AUTOTUNE_SAMPLE_RANKS',
    'BLOCK_SELECTION_MODE', 'BLOCK_X', 'BLOCK_Y', 'BLOCK_Z', 'RUN_ITERATION', 'TIMING_PARSER_VERSION',
    'PRECOMPUTED_CATALOG', 'PRECOMPUTED_SET_ID', 'PROCESS_EXIT_CODE', 'JOB_WALL_TIME', 'APP_WALL_TIME',
    'WALL_TIME_DECOMPOSITION_S', 'RANK_WALL_TIME_MIN', 'RANK_WALL_TIME_MAX', 'RANK_WALL_TIME_MEAN', 'SLOWEST_RANK',
    'RANK_TIME_IMBALANCE', 'RUN_EXIT_CODE', 'VALIDATION_STATUS', 'RUN_STATUS', 'COMMIT_STATUS']
TGV_GRID_KEYS = ['APP_NAME', 'GRID_CONFIG', 'GRID', 'NUM_DECOMPOSITION_GPUS', 'RUN_ITERATIONS', 'AUTOTUNE_MODES',
                 'SAMPLED_AUTOTUNE', 'AUTOTUNE_SAMPLE_RANKS', 'PRECOMPUTED_SOURCE', 'TIMING_PARSER_VERSION']
TGV_ITER_KEYS = ['AUTOTUNE_MODES', 'RUN_ITERATION', 'WALL_TIME_NO_AUTOTUNE', 'JOB_WALL_TIME_NO_AUTOTUNE',
                 'WALL_TIME_AUTOTUNE', 'JOB_WALL_TIME_AUTOTUNE', 'WALL_TIME_PRECOMPUTED', 'JOB_WALL_TIME_PRECOMPUTED']
# methodology.md 4.3, verbatim
RANK_HEADER = ('run_id,rank_id,local_rank,gpu_id,hostname,coord_x,coord_y,coord_z,coord_source,local_nx,local_ny,local_nz,'
               'rank_wall_time_s,app_reported_wall_time_s,rank_compute_time_s,rank_mpi_time_s,rank_halo_time_s,'
               'rank_wait_time_s,rank_status')
SUMMARY_LINE = re.compile(r'^WALL_TIME_DECOMPOSITION_\d+x\d+x\d+_ITER_\d+_S=[0-9.]+$')


def keys(path):
    return [line.split('=', 1)[0] for line in Path(path).read_text().splitlines()]


def committed(values):
    return all(values.get(k) == v for k, v in D.COMMITTED.items())


class Sweep(unittest.TestCase):
    """A mocked sweep: 7 decompositions, one failed precomputed run, one run cut off by the end of
    an allocation, then a resume and an explicit retry."""
    reps = 1

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='senga_dataset_')
        cls.tmp = Path(cls.temp.name)
        cls.campaign = T.make_campaign(cls.tmp, grid=20, reps=cls.reps, sentinel_every=4,
                                       extra=['--date-stamp', '20261115_200600'])
        T.fake_build(cls.campaign)
        T.quiet(L.main, ['plan', '--campaign', str(cls.campaign)])
        cls.plan = read_json(cls.campaign / 'plan.json')
        cls.order = [d['decomposition'] for d in cls.plan['decompositions']]
        cls.layout = T.layout(cls.campaign)
        cls.failed = 'sweep_p002_%s_pair1_P' % cls.order[2]
        rule = dict(decomposition=cls.order[2], mode=2, kind='exit', once_file=str(cls.tmp / 'fired'))
        cls.first = T.run_job(cls.campaign, 'sweep', T.job_env(cls.tmp, job='1001', fail=[rule]))
        cls.cut = 'sweep_p005_%s_pair1_D' % cls.order[5]
        cls.cut_attempt = T.interrupt(cls.campaign, cls.cut)                # allocation ended during this run
        cls.second = T.run_job(cls.campaign, 'sweep', T.job_env(cls.tmp, job='1002'))
        cls.third = T.run_job(cls.campaign, 'sweep', T.job_env(cls.tmp, job='1003'), '--retry-failed')
        T.quiet(L.main, ['collect', '--campaign', str(cls.campaign)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def finals(self):
        return sorted(p.parent for p in self.layout.grid_folder.glob('**/config.txt')
                      if (p.parent / 'record.json').is_file())

    # ----- criterion 1: the tree
    def test_jobs_ran_as_intended(self):
        self.assertEqual(self.first[0], 5, self.first[1])            # one failed run
        self.assertEqual(self.second[0], 5, self.second[1])          # still failed, not retried automatically
        self.assertEqual(self.third[0], 0, self.third[1])
        self.assertIn('not retried automatically', self.second[1])

    def test_exact_tree(self):
        lay = self.layout
        self.assertEqual(lay.dataset.name, 'OPS_LOGS_20261115_200600_n001')
        self.assertEqual(lay.grid_folder.name, 'senga2_hip_20x20x20_cfg1_comparison_20261115_200600_n001')
        self.assertEqual(sorted(p.name for p in lay.dataset.iterdir()),
                         sorted(['_attempts', 'campaign_manifest.json', 'plan.jsonl', 'precomputed_blocks',
                                 lay.grid_folder.name]))
        self.assertEqual(sorted(p.name for p in lay.grid_folder.iterdir()),
                         sorted(['config.txt', 'autotune_off_summary.txt', 'autotune_on_summary.txt',
                                 'autotune_precomputed_summary.txt', 'mode_1', 'baseline_unforced', 'sentinels']))
        expected = set()
        for text in self.order:
            for r in range(1, self.reps + 1):
                expected |= {'mode_1/decomp%s/iter_%d/%s' % (text, r, v) for v in ('autotune_off', 'autotune_precomputed')}
            expected.add('mode_1/decomp%s/iter_1/autotune_on' % text)
        expected |= {'baseline_unforced/iter_%d/autotune_off' % r for r in range(1, self.reps + 1)}
        expected |= {'sentinels/%s/autotune_off' % s for s in
                     ('sentinel_start', 'sentinel_after_p003', 'sentinel_alloc_1002', 'sentinel_alloc_1003')}
        got = {p.relative_to(lay.grid_folder).as_posix() for p in self.finals()}
        self.assertEqual(got, expected)
        iters = {p.relative_to(lay.grid_folder).as_posix() for p in lay.grid_folder.glob('mode_1/decomp*/iter_*')}
        self.assertEqual(iters, {'mode_1/decomp%s/iter_%d' % (t, r) for t in self.order for r in range(1, self.reps + 1)})
        for text in self.order:
            for r in range(2, self.reps + 1):                        # the single autotune run is in iter_1 only
                self.assertFalse((lay.grid_folder / ('mode_1/decomp%s/iter_%d/autotune_on' % (text, r))).exists())
        for folder in self.finals():
            names = {p.name for p in folder.iterdir()}
            want = {'config.txt', 'run.log', 'mpi_rank_metrics.csv', 'ranks', 'record.json', 'srun.log', 'execution.json', 'work'}
            if folder.name == 'autotune_on':
                want.add('rank_0')
                self.assertEqual(sorted(p.name for p in (folder / 'rank_0/autotune_on').iterdir()),
                                 ['ops_blocksize_best_logmod.csv', 'ops_blocksize_tuning_logmod.csv'])
            if folder.name == 'autotune_precomputed':
                want |= {'blocks.csv', 'precomputed_manifest.json'}
            self.assertEqual(names, want, folder)
            ranks = {p.name for p in (folder / 'ranks').iterdir()}
            self.assertEqual(ranks, {'%s_%d.%s' % (k, r, e) for r in range(8)
                                     for k, e in (('rank', 'log'), ('metric', 'csv'), ('status', 'json'))})
        self.assertEqual(sorted(p.name for p in (lay.dataset / 'precomputed_blocks' / 'senga2_hip_20x20x20_cfg1' / 'mode_1').iterdir()),
                         sorted('decomp' + t for t in self.order))

    # ----- criterion 2: run config.txt
    def test_run_config_keys_and_values(self):
        for folder in self.finals():
            names = keys(folder / 'config.txt')
            record = read_json(folder / 'record.json')
            base = [k for k in TGV_RUN_KEYS if record['mode'] == 'P' or not k.startswith('PRECOMPUTED_')]
            self.assertEqual(names[:len(base)], base, folder)
            self.assertEqual(names[len(base):], list(D.EXTRA_KEYS))
            self.assertEqual(len(names), len(set(names)))
            c = D.parse_config(folder / 'config.txt')
            loc = self.layout.locate(record['run_id'])
            self.assertEqual(c['CASE_ID'], loc['case_id'])
            self.assertEqual((c['APP_NAME'], c['APP_ARGS'], c['CAMPAIGN_ID']), ('senga2', '', '20261115_200600_n001'))
            self.assertEqual(float(c['APP_WALL_TIME']), record['total_ttime_s'])
            self.assertEqual(c['APP_WALL_TIME'], c['WALL_TIME_DECOMPOSITION_S'])
            self.assertIn('total_ttime   =%16s' % c['APP_WALL_TIME'], (folder / 'ranks/rank_0.log').read_text())
            self.assertAlmostEqual(float(c['JOB_WALL_TIME']), record['launcher_elapsed_s'], places=5)
            self.assertEqual(float(c['KERNEL_TOTAL_TIME']), record['kernel_time_s'])
            self.assertEqual(float(c['HALO_TOTAL_TIME']), record['halo_time_s'])
            self.assertEqual(float(c['USER_HALO_TOTAL_TIME']), record['user_halo_time_s'])
            self.assertEqual(c['EXECUTABLE_SHA256'], record['executable_sha256'])
            self.assertEqual(int(c['NITER']), record['steps'])
            self.assertEqual([int(c['DECOMP_P' + a]) for a in 'XYZ'], record['triple'] or record['printed_grid'])
            self.assertEqual(int(c['OPS_AUTOTUNE_MODE']), {'D': 0, 'A': 1, 'P': 2}[record['mode']])
            iteration = 1 if record['stage'] == 'sentinel' else (record['pair_index'] or record['repetition'])
            self.assertEqual(int(c['RUN_ITERATION']), iteration)
            self.assertEqual(int(c['ATTEMPT_ID']), int(record['attempt'].split('_')[1]))
            self.assertEqual((c['RUN_ID'], c['SLURM_JOB_ID'], c['SLURM_STEP_ID'], c['PLAN_SHA256']),
                             (record['run_id'], record['slurm_job_id'], record['slurm_step_id'], record['plan_hash']))
            self.assertEqual((c['GPU_DIRECT'], c['MPICH_GPU_SUPPORT_ENABLED']), ('YES', '1'))
            self.assertEqual((int(c['NUM_NODES']), int(c['NUM_MPI_RANKS']), int(c['NUM_GPUS_USED'])), (1, 8, 8))
            self.assertEqual(c['TIMING_PARSER_VERSION'], D.TIMING_PARSER_VERSION)
            self.assertEqual(c['COORD_SOURCE'], 'not_measured')
            block = {'D': ['32', '4', '1'], 'A': ['dynamic'] * 3, 'P': ['per_kernel'] * 3}[record['mode']]
            self.assertEqual([c['BLOCK_' + a] for a in 'XYZ'], block)
            if record['mode'] == 'P':
                self.assertEqual((c['CATALOG_SHA256'], int(c['CATALOG_ROWS'])), (record['catalog_sha256'], record['catalog_rows']))
                self.assertEqual(c['FALLBACK_KERNEL_IDS_RANK0'], ' '.join(map(str, record['fallback_kernel_ids_rank0'])))
            if record['mode'] == 'A':
                self.assertEqual((c['AUTOTUNE_CATALOG_STATUS'], c['CATALOG_SHA256']),
                                 ('FROZEN', record['autotune']['catalog_sha256']))
            elapsed = [r['elapsed_s'] for r in record['ranks']]
            self.assertAlmostEqual(float(c['RANK_WALL_TIME_MAX']), max(elapsed), places=5)
            self.assertEqual(int(c['SLOWEST_RANK']), max(record['ranks'], key=lambda r: r['elapsed_s'])['rank'])

    # ----- criterion 3: iter, grid and summaries
    def test_iter_grid_and_summaries(self):
        lay = self.layout
        grid = lay.grid_folder / 'config.txt'
        self.assertEqual(keys(grid), TGV_GRID_KEYS)
        g = D.parse_config(grid)
        self.assertEqual((g['APP_NAME'], g['GRID'], g['NUM_DECOMPOSITION_GPUS'], g['RUN_ITERATIONS'], g['AUTOTUNE_MODES'],
                          g['SAMPLED_AUTOTUNE'], g['AUTOTUNE_SAMPLE_RANKS'], g['PRECOMPUTED_SOURCE']),
                         ('senga2', '20x20x20', '8', str(self.reps), '0 1 2', '1', '0', ''))
        for iteration in lay.grid_folder.glob('mode_1/decomp*/iter_*'):
            values = D.parse_config(iteration / 'config.txt')
            want = TGV_ITER_KEYS if iteration.name == 'iter_1' else \
                [k for k in TGV_ITER_KEYS if k not in ('WALL_TIME_AUTOTUNE', 'JOB_WALL_TIME_AUTOTUNE')]
            self.assertEqual(keys(iteration / 'config.txt'), want, iteration)
            self.assertEqual(values['RUN_ITERATION'], iteration.name.split('_')[1])
            for variant, key in D.VARIANT_TIME_KEY.items():
                if (iteration / variant).is_dir():
                    run = D.parse_config(iteration / variant / 'config.txt')
                    self.assertEqual((values['WALL_TIME_' + key], values['JOB_WALL_TIME_' + key]),
                                     (run['APP_WALL_TIME'], run['JOB_WALL_TIME']))
        for variant in D.VARIANT_ORDER:
            lines = (lay.grid_folder / ('%s_summary.txt' % variant)).read_text().splitlines()
            runs = sorted(lay.grid_folder.glob('mode_1/decomp*/iter_*/%s/config.txt' % variant))
            self.assertEqual(len(lines), len(runs))
            self.assertTrue(all(SUMMARY_LINE.match(l) for l in lines), lines[:3])
            for path in runs:
                c = D.parse_config(path)
                line = 'WALL_TIME_DECOMPOSITION_%sx%sx%s_ITER_%s_S=%s' % (c['DECOMP_PX'], c['DECOMP_PY'], c['DECOMP_PZ'],
                                                                         c['RUN_ITERATION'], c['APP_WALL_TIME'])
                self.assertEqual(lines.count(line), 1)

    # ----- criterion 4: per-rank table
    def test_mpi_rank_metrics(self):
        for folder in self.finals():
            text = (folder / 'mpi_rank_metrics.csv').read_text()
            self.assertEqual(text.splitlines()[0], RANK_HEADER)
            rows = list(csv.DictReader(text.splitlines()))
            record = read_json(folder / 'record.json')
            self.assertEqual([int(r['rank_id']) for r in rows], list(range(8)))
            case = D.parse_config(folder / 'config.txt')['CASE_ID']
            for row in rows:
                self.assertEqual(row['run_id'], case)
                for empty in ('coord_x', 'coord_y', 'coord_z', 'rank_compute_time_s', 'rank_mpi_time_s', 'rank_wait_time_s'):
                    self.assertEqual(row[empty], '', empty)                       # not measured: never filled
                self.assertEqual((row['coord_source'], row['rank_status']), ('not_measured', 'OK'))
                status = read_json(folder / 'ranks' / ('status_%s.json' % row['rank_id']))
                self.assertEqual((row['hostname'], int(row['gpu_id']), int(row['local_rank'])),
                                 (status['host'], status['gpu_id'], status['local_rank']))
                self.assertAlmostEqual(float(row['rank_wall_time_s']), status['elapsed_s'], places=5)
                triple = record['triple'] or record['printed_grid']
                self.assertEqual([int(row['local_n' + a]) for a in 'xyz'], [20 // p for p in triple])
                single = (folder / 'ranks' / ('metric_%s.csv' % row['rank_id'])).read_text().strip()
                self.assertEqual(single, text.splitlines()[int(row['rank_id']) + 1])
            self.assertEqual(float(rows[0]['app_reported_wall_time_s']), record['total_ttime_s'])
            self.assertEqual(float(rows[0]['rank_halo_time_s']), record['halo_time_s'])
            self.assertTrue(all(r['app_reported_wall_time_s'] == '' and r['rank_halo_time_s'] == '' for r in rows[1:]))
            log = (folder / 'run.log').read_text()
            self.assertEqual(re.findall(r'^===== (rank_\d+\.log) =====$', log, re.M),
                             sorted('rank_%d.log' % r for r in range(8)))

    # ----- criterion 5: commit rule, _attempts, resume
    def test_commit_rule_and_attempts(self):
        lay = self.layout
        for folder in self.finals():
            self.assertTrue(committed(D.parse_config(folder / 'config.txt')), folder)
        attempts = sorted(lay.attempts_root.glob('*/iter_*/*/attempt_*'))
        for attempt in attempts:
            if (attempt / 'config.txt').is_file():
                self.assertFalse(committed(D.parse_config(attempt / 'config.txt')), attempt)
        failed = lay.locate(self.failed)
        first = failed['attempts'] / 'attempt_001'
        c = D.parse_config(first / 'config.txt')
        self.assertEqual((c['RUN_STATUS'], c['VALIDATION_STATUS'], c['COMMIT_STATUS'], c['VERDICT']),
                         ('FAILED', 'FAIL', 'UNCOMMITTED', 'INVALID'))
        self.assertNotEqual(c['RUN_EXIT_CODE'], '0')
        self.assertTrue((first / 'SUPERSEDED.json').is_file())                  # set aside by --retry-failed
        self.assertEqual(read_json(failed['final'] / 'record.json')['attempt'], 'attempt_002')
        self.assertEqual(read_json(failed['final'] / 'record.json')['slurm_job_id'], '1003')
        self.assertEqual(first.parent, lay.attempts_root / failed['case_id'] / 'iter_1' / 'autotune_precomputed')
        # the run cut off by the end of the allocation stays in _attempts, superseded, and was rerun
        self.assertFalse((self.cut_attempt / 'record.json').exists())
        self.assertIn('interrupted', read_json(self.cut_attempt / 'SUPERSEDED.json')['reason'])
        self.assertEqual(read_json(lay.locate(self.cut)['final'] / 'record.json')['slurm_job_id'], '1002')
        # its partner was valid and committed, but a pair is atomic: it left the final tree, superseded, and was rerun
        partner = lay.locate(self.cut.replace('_D', '_P'))
        old = partner['attempts'] / 'attempt_001'
        self.assertEqual(read_json(old / 'record.json')['slurm_job_id'], '1001')
        self.assertIn('incomplete pair', read_json(old / 'SUPERSEDED.json')['reason'])
        self.assertEqual(read_json(old / 'SUPERSEDED.json')['moved_from_final'], str(partner['final']))
        self.assertEqual(D.parse_config(old / 'config.txt')['COMMIT_STATUS'], 'SUPERSEDED')
        self.assertEqual(read_json(partner['final'] / 'record.json')['slurm_job_id'], '1002')
        # no failed attempt appears anywhere in the final tree
        self.assertFalse([p for p in lay.grid_folder.glob('**/config.txt') if 'RUN_STATUS=FAILED' in p.read_text()])

    def test_pending_commit_is_recovered_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = T.copy_tree(self.campaign, Path(tmp) / 'campaign')
            lay = T.layout(campaign)
            run_id = 'sweep_p001_%s_pair1_D' % self.order[1]
            loc = lay.locate(run_id)
            name = read_json(loc['final'] / 'record.json')['attempt']
            os.rename(str(loc['final']), str(loc['attempts'] / name))     # job ended between record and move
            self.assertEqual(lay.state(run_id)[0], 'PENDING_COMMIT')
            env = T.job_env(Path(tmp), job='1004')
            status, text = T.run_job(campaign, 'sweep', env)
            self.assertEqual(status, 0, text)
            self.assertIn('was not yet moved to its final folder', text)
            self.assertEqual(T.srun_calls(Path(tmp)), ['sentinel_alloc_1004'])     # nothing else launched
            self.assertTrue((loc['final'] / 'record.json').is_file())

    # ----- criterion 6: catalogs
    def test_catalog_layout_and_set_id(self):
        lay = self.layout
        for text in self.order:
            folder = lay.catalog_folder(text)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ['blocks.csv', 'manifest.json'])
            manifest = read_json(folder / 'manifest.json')
            unsigned = {k: v for k, v in manifest.items() if k != 'set_id'}
            self.assertEqual(manifest['set_id'], hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest())
            self.assertEqual(manifest['files'], {'blocks.csv': sha(folder / 'blocks.csv')})
            self.assertEqual(manifest['sha256'], sha(folder / 'blocks.csv'))
            self.assertEqual((manifest['context']['shape'], manifest['context']['APP_NAME'], manifest['repetition']),
                             (text, 'senga2', 1))
            source = lay.grid_folder / ('mode_1/decomp%s/iter_1/autotune_on' % text)
            self.assertEqual(manifest['source_run'], str(source))
            self.assertEqual(sha(source / R.BEST_CSV), manifest['source_best_csv_sha256'])
            for p_run in lay.grid_folder.glob('mode_1/decomp%s/iter_*/autotune_precomputed' % text):
                c = D.parse_config(p_run / 'config.txt')
                self.assertEqual((c['PRECOMPUTED_CATALOG'], c['PRECOMPUTED_SET_ID']), (str(folder), manifest['set_id']))
                self.assertEqual((p_run / 'precomputed_manifest.json').read_bytes(), (folder / 'manifest.json').read_bytes())
                self.assertEqual(sha(p_run / 'blocks.csv'), manifest['files']['blocks.csv'])
                env = read_json(p_run / 'ranks/status_0.json')['environment']
                self.assertEqual(env['OPS_BLOCKSIZE_CSV'], str(folder / 'blocks.csv'))   # the replay read the frozen file
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(shutil.copytree(str(lay.catalog_folder(self.order[0])), str(Path(tmp) / 'cat')))
            os.chmod(str(copy / 'manifest.json'), 0o644)
            m = read_json(copy / 'manifest.json')
            exe = m['executable_sha256']
            R.load_catalog(copy, self.order[0], exe)
            (copy / 'manifest.json').write_text(json.dumps(dict(m, selection='edited'), indent=2))
            with self.assertRaisesRegex(ValueError, 'set_id does not match'):
                R.load_catalog(copy, self.order[0], exe)

    # ----- criterion 7: baseline and sentinels outside mode_1
    def test_reader_glob_returns_only_sweep_runs(self):
        lay = self.layout
        for root, pattern in ((lay.grid_folder, 'mode_*/decomp*/iter_*/autotune_*/config.txt'),
                              (lay.dataset, '*/mode_*/decomp*/iter_*/autotune_*/config.txt')):
            found = sorted(root.glob(pattern))
            self.assertEqual(len(found), len(self.order) * (1 + 2 * self.reps))
            self.assertEqual({D.parse_config(p)['RUN_CLASS'] for p in found}, {'SCIENTIFIC'})
            self.assertEqual({D.parse_config(p)['STAGE'] for p in found}, {'sweep'})
        classes = {D.parse_config(p)['RUN_CLASS'] for p in lay.grid_folder.glob('baseline_unforced/iter_*/autotune_off/config.txt')}
        self.assertEqual(classes, {'BASELINE_UNFORCED'})
        b = D.parse_config(next(lay.grid_folder.glob('baseline_unforced/iter_1/autotune_off/config.txt')))
        self.assertEqual((b['DECOMPOSITION_FORCED'], b['DECOMPOSITION_POLICY'], b['PRINTED_GRID']), ('NO', 'unforced', '2x2x2'))
        self.assertEqual({D.parse_config(p)['RUN_CLASS'] for p in lay.grid_folder.glob('sentinels/*/autotune_off/config.txt')},
                         {'SENTINEL'})

    def test_campaign_manifest_and_plan_jsonl(self):
        lay = self.layout
        manifest = read_json(lay.dataset / 'campaign_manifest.json')
        data = (lay.dataset / 'plan.jsonl').read_bytes()
        self.assertEqual(manifest['plan_sha256'], hashlib.sha256(data).hexdigest())
        cases = [json.loads(line) for line in data.decode().splitlines()]
        self.assertEqual(manifest['case_count'], len(cases))
        self.assertEqual(len(cases), len(self.order) * (1 + 2 * self.reps))
        self.assertEqual(len({c['case_id'] for c in cases}), len(cases))
        self.assertEqual((manifest['commit_rule'], manifest['campaign_id'], manifest['nodes'] * 8, manifest['mpi_ranks']),
                         (D.COMMIT_RULE, '20261115_200600_n001', 8, 8))
        for c in cases:
            self.assertEqual(sorted(c), sorted(['app', 'app_args', 'autotune_mode', 'case_id', 'decomposition',
                                                'decomposition_mode', 'grid', 'grid_config', 'identity_sha256', 'nodes',
                                                'ranks', 'repetition', 'schema_version', 'state', 'target']))
            identity = {k: c[k] for k in ('app', 'nodes', 'ranks', 'grid', 'grid_config', 'decomposition_mode',
                                          'decomposition', 'autotune_mode', 'repetition', 'app_args')}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(c['identity_sha256'], digest)
            config = D.parse_config(lay.dataset / c['target'] / 'config.txt')        # every case reached its target
            self.assertEqual((config['CASE_ID'], int(config['OPS_AUTOTUNE_MODE']), int(config['RUN_ITERATION'])),
                             (c['case_id'], c['autotune_mode'], c['repetition']))

    # ----- criterion 10: collect
    def test_collect_reads_the_dataset(self):
        out = self.campaign / 'results'
        for name in ('runs.csv', 'decompositions.csv', 'incomplete_decompositions.csv', 'sentinels.csv',
                     'collection_manifest.json'):
            self.assertTrue((out / name).is_file(), name)
        with (out / 'decompositions.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.order))
        self.assertTrue(all(r['status'] == 'COMPLETE' for r in rows), [r['status'] for r in rows])
        with (out / 'runs.csv').open() as f:
            runs = list(csv.DictReader(f))
        used = [r for r in runs if r['used_in_statistics'] == 'True']
        self.assertTrue(all(r['committed'] == 'True' for r in used))
        self.assertEqual(len([r for r in runs if r['committed'] == 'False']),
                         len(list(self.layout.attempts_root.glob('*/iter_*/*/attempt_*/record.json'))))
        manifest = read_json(out / 'collection_manifest.json')
        self.assertEqual(manifest['dataset'], str(self.layout.dataset))
        self.assertIn('launcher_sha256_now', manifest['code_version'])


class SweepThreeRepetitions(Sweep):
    """The general path: three D/P pairs, iter_1..3, the single autotune run in iter_1."""
    reps = 3

    def test_jobs_ran_as_intended(self):
        self.assertEqual(self.third[0], 0, self.third[1])


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='senga_upgrade_')
        self.tmp = Path(self.temp.name)
        self.campaign = T.make_campaign(self.tmp, grid=20, reps=3)
        T.fake_build(self.campaign)

    def age_launcher(self):
        """Make the frozen launcher look like the previous release: no dataset.py, another engine.py."""
        launcher = self.campaign / 'launcher'
        (launcher / 'dataset.py').unlink()
        (launcher / 'engine.py').write_text((launcher / 'engine.py').read_text() + '\n# previous release\n')
        self.old_engine = sha(launcher / 'engine.py')

    def tearDown(self):
        self.temp.cleanup()

    def upgrade(self, *extra):
        return T.quiet(L.main, ['upgrade', '--campaign', str(self.campaign)] + list(extra))

    def test_upgrade_replaces_the_launcher_and_records_hashes(self):
        self.age_launcher()
        status, text = self.upgrade('--repetitions', '1')
        self.assertEqual(status, 0, text)
        launcher = self.campaign / 'launcher'
        for name in L.PACKAGE_FILES:
            self.assertEqual(sha(launcher / name), sha(L.HERE / name), name)
        history = read_json(self.campaign / 'launcher_upgrades.json')['upgrades']
        self.assertEqual(len(history), 1)
        entry = history[0]
        self.assertEqual(entry['old_files_sha256']['engine.py'], self.old_engine)
        self.assertNotIn('dataset.py', entry['old_files_sha256'])
        self.assertEqual(entry['new_files_sha256']['dataset.py'], sha(L.HERE / 'dataset.py'))
        self.assertEqual((entry['repetitions_old'], entry['repetitions_new']), (3, 1))
        self.assertEqual(sha(Path(entry['previous_launcher']) / 'engine.py'), self.old_engine)   # old copy kept
        config = read_json(self.campaign / 'campaign.json')
        self.assertEqual((config['repetitions'], config['plan']['repetitions']), (1, 1))
        self.assertIn('Repetitions per decomposition: 3 -> 1', text)
        self.assertEqual(E.Campaign(self.campaign).package, entry['package'])
        # the plan frozen afterwards has one repetition, the grid config.txt says RUN_ITERATIONS=1
        T.quiet(L.main, ['plan', '--campaign', str(self.campaign)])
        self.assertEqual(read_json(self.campaign / 'plan.json')['repetitions'], 1)
        status, text = T.run_job(self.campaign, 'sweep', T.job_env(self.tmp))
        self.assertEqual(status, 0, text)
        lay = T.layout(self.campaign)
        self.assertEqual(D.parse_config(lay.grid_folder / 'config.txt')['RUN_ITERATIONS'], '1')
        self.assertEqual({p.name for p in lay.grid_folder.glob('mode_1/decomp*/iter_*')}, {'iter_1'})
        self.assertEqual(len(list(lay.grid_folder.glob('baseline_unforced/iter_*'))), 1)
        record = T.sweep_record(self.campaign, 'baseline_unforced_r1')
        self.assertEqual(record['package_commit'], entry['package']['commit'])

    def test_upgrade_keeps_qualification_and_gonogo_working(self):
        status, text = T.run_job(self.campaign, 'smoke', T.job_env(self.tmp), '--builds', 'small,full')
        self.assertEqual(status, 0, text)
        # strip what the previous release did not write, so the records look like the real ones
        for path in (self.campaign / 'runs/qualification').glob('*/attempt_*/record.json'):
            record = read_json(path)
            for key in ('total_ttime_text', 'kernel_time_text', 'halo_time_text', 'user_halo_time_text', 'gpu_arch',
                        'attempt_dir', 'run_dir', 'case_id'):
                record.pop(key, None)
            path.write_text(json.dumps(record))
        self.age_launcher()
        before = {p: sha(p) for p in sorted(self.campaign.glob('runs/qualification/**/*')) if p.is_file()}
        before.update({p: sha(p) for p in sorted(self.campaign.glob('catalogs/**/*')) if p.is_file()})
        before.update({p: sha(p) for p in sorted(self.campaign.glob('build/*/*')) if p.is_file()})
        status, text = self.upgrade('--repetitions', '1')
        self.assertEqual(status, 0, text)
        after = {p: sha(p) for p in before}
        self.assertEqual(before, after)                                     # nothing of qualification or builds changed
        status, text = T.quiet(L.main, ['gonogo', '--campaign', str(self.campaign), '--gpu-hours-per-node-hour', '4',
                                        '--remaining-gpu-hours', '10000', '--disk-available-gb', '500'])
        self.assertEqual(status, 0, text)
        self.assertIn('VERDICT: GO', text)
        status, text = T.quiet(L.main, ['status', '--campaign', str(self.campaign)])
        self.assertIn('dataset: %s' % T.layout(self.campaign).dataset, text)

    def test_refused_with_active_jobs(self):
        self.age_launcher()
        bindir = self.tmp / 'bin'
        bindir.mkdir()
        (bindir / 'squeue').write_text('#!/bin/sh\necho 4242\n')
        (bindir / 'squeue').chmod(0o755)
        (self.campaign / 'submissions.json').write_text(json.dumps(dict(jobs=[dict(kind='smoke-full', job_id='4242',
                                                                                  submitted_utc='', command='')])))
        with patch.dict(os.environ, dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'])):
            with self.assertRaisesRegex(ValueError, r'Job\(s\) 4242 of this campaign are still pending or running'):
                self.upgrade()
        self.assertEqual(sha(self.campaign / 'launcher/engine.py'), self.old_engine)
        self.assertFalse((self.campaign / 'launcher_upgrades.json').exists())

    def test_refused_once_the_sweep_has_started(self):
        self.upgrade()
        T.quiet(L.main, ['plan', '--campaign', str(self.campaign)])
        T.run_job(self.campaign, 'sweep', T.job_env(self.tmp, fail=[dict(decomposition='unforced', mode=0, kind='exit')]))
        engine = sha(self.campaign / 'launcher/engine.py')
        with self.assertRaisesRegex(ValueError, 'sweep of this campaign has already started'):
            self.upgrade()
        self.assertEqual(sha(self.campaign / 'launcher/engine.py'), engine)
        self.assertEqual(len(read_json(self.campaign / 'launcher_upgrades.json')['upgrades']), 1)

    def test_refused_after_a_sweep_submission(self):
        (self.campaign / 'submissions.json').write_text(json.dumps(dict(jobs=[dict(kind='sweep', job_id='77',
                                                                                  submitted_utc='', command='')])))
        bindir = self.tmp / 'bin'
        bindir.mkdir()
        (bindir / 'squeue').write_text('#!/bin/sh\nexit 0\n')
        (bindir / 'squeue').chmod(0o755)
        with patch.dict(os.environ, dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'])):
            with self.assertRaisesRegex(ValueError, '1 sweep job'):
                self.upgrade()

    def test_repetitions_cannot_change_after_the_plan_is_frozen(self):
        T.quiet(L.main, ['plan', '--campaign', str(self.campaign)])
        with self.assertRaisesRegex(ValueError, r'already frozen \(plan.json: 3 repetition'):
            self.upgrade('--repetitions', '1')
        self.assertFalse((self.campaign / 'launcher_upgrades.json').exists())
        status, text = self.upgrade()                                 # without a change it is allowed
        self.assertEqual(status, 0, text)
        self.assertEqual(read_json(self.campaign / 'campaign.json')['repetitions'], 3)

    def test_restart_sweep_archives_the_started_sweep_and_applies_mirror(self):
        self.upgrade('--decompositions', 'all')
        T.quiet(L.main, ['plan', '--campaign', str(self.campaign)])
        valid, _ = P.enumerate_decompositions(8, 20)
        self.assertEqual(len(read_json(self.campaign / 'plan.json')['decompositions']), len(valid))   # all triples
        T.run_job(self.campaign, 'sweep', T.job_env(self.tmp, fail=[dict(decomposition='unforced', mode=0, kind='exit')]))
        lay = T.layout(self.campaign)
        self.assertTrue(lay.dataset.is_dir())
        before = {p.relative_to(self.campaign): sha(p) for p in self.campaign.rglob('*') if p.is_file()
                  and not p.relative_to(self.campaign).parts[0] in ('launcher', 'launcher_previous', 'archive')}
        kept = {p for p in before if p.parts[0] in ('runs', 'build', 'references', 'catalogs') and 'sweep' not in p.parts}
        with self.assertRaisesRegex(ValueError, 'add --restart-sweep'):
            self.upgrade('--decompositions', 'mirror')
        status, text = self.upgrade('--decompositions', 'mirror', '--restart-sweep')
        self.assertEqual(status, 0, text)
        self.assertIn('Decompositions: mirror (was all)', text)
        archive = next((self.campaign / 'archive').glob('sweep_*'))
        moved = read_json(archive / 'ARCHIVED.json')['moved']
        self.assertIn('plan.json', moved)
        self.assertIn(lay.dataset.name, moved)
        self.assertFalse((self.campaign / 'plan.json').exists())
        self.assertFalse(lay.dataset.exists())
        # nothing deleted: every file that was there is either in place or under the archive with the same bytes
        for rel, digest in before.items():
            if str(rel) in ('campaign.json', 'launcher_upgrades.json', 'submissions.json'):
                continue                       # rewritten on purpose (submissions.json is copied into the archive)
            here, there = self.campaign / rel, archive / rel
            self.assertTrue(here.is_file() and sha(here) == digest or there.is_file() and sha(there) == digest, str(rel))
        for rel in kept:
            self.assertEqual(sha(self.campaign / rel), before[rel], str(rel))     # qualification, builds untouched
        self.assertEqual(read_json(self.campaign / 'campaign.json')['plan']['decompositions'], 'mirror')
        entry = read_json(self.campaign / 'launcher_upgrades.json')['upgrades'][-1]
        self.assertEqual((entry['decompositions_old'], entry['decompositions_new'], entry['sweep_restarted']),
                         ('all', 'mirror', True))
        # the sweep can be planned and run again with the mirror representatives (Px >= Pz)
        status, text = T.quiet(L.main, ['plan', '--campaign', str(self.campaign)])
        self.assertEqual(status, 0, text)
        decs = [d['decomposition'] for d in read_json(self.campaign / 'plan.json')['decompositions']]
        expected = sorted('%dx%dx%d' % t for t in valid if t[0] >= t[2])
        self.assertEqual(sorted(decs), expected)
        self.assertLess(len(decs), len(valid))
        status, text = T.run_job(self.campaign, 'sweep', T.job_env(self.tmp))
        self.assertEqual(status, 0, text)
        self.assertEqual({p.name for p in T.layout(self.campaign).grid_folder.glob('mode_1/decomp*')},
                         {'decomp' + d for d in decs})

    def test_decomposition_set_cannot_change_after_the_plan_is_frozen(self):
        T.quiet(L.main, ['plan', '--campaign', str(self.campaign)])
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            self.upgrade('--decompositions', 'all' if read_json(self.campaign / 'campaign.json')['plan'].get(
                'decompositions', 'all') == 'mirror' else 'mirror')
        self.assertFalse((self.campaign / 'launcher_upgrades.json').exists())

    def test_refused_from_the_frozen_copy(self):
        self.upgrade()
        r = subprocess.run(['bash', str(self.campaign / 'launcher/run.sh'), 'upgrade', '--campaign', str(self.campaign)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('Run upgrade with the run.sh of your OPS checkout', r.stderr)


class StampTests(unittest.TestCase):
    def test_dataset_stamp_of_an_existing_campaign(self):
        config = dict(campaign_id='SENGA2_DECOMP_20261115_200600', created_utc='2026-11-15T20:06:07Z', nodes=20, ranks=160,
                      grid=1440)
        lay = D.Layout('/c', config)
        self.assertEqual(lay.dataset, Path('/c/OPS_LOGS_20261115_200600_n020'))
        self.assertEqual(lay.grid_folder.name, 'senga2_hip_1440x1440x1440_cfg1_comparison_20261115_200600_n020')
        loc = lay.locate('sweep_p012_4x5x8_pair1_P')
        self.assertEqual(loc['case_id'], 'senga2_hip_1440x1440x1440_cfg1_g160_decomp4x5x8_20261115_200600_n020_iter_1_'
                                         'autotune_precomputed')
        self.assertEqual(loc['final'].relative_to(lay.dataset).as_posix(),
                         'senga2_hip_1440x1440x1440_cfg1_comparison_20261115_200600_n020/mode_1/decomp4x5x8/iter_1/'
                         'autotune_precomputed')
        self.assertEqual(loc['attempts'].relative_to(lay.dataset).as_posix(),
                         '_attempts/%s/iter_1/autotune_precomputed' % loc['case_id'])
        self.assertEqual(D.Layout('/c', dict(config, campaign_id='x')).stamp, '20261115_200607')   # from created_utc
        with self.assertRaisesRegex(ValueError, 'no place in the dataset'):
            lay.locate('q5_d0_A')


if __name__ == '__main__':
    unittest.main()
