"""Full-package CPU integration: every selected case and every planned MPI rank.

Real dataset and launcher; only Slurm, HIP binary execution and linkage are mocked.
Set OPS_TOP3_DATASET to an extracted version-2 package, or use the bundled copy.
"""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import launcher as L
import runtime as R
import prepare_dataset as P


class WholeDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Path(os.environ.get('OPS_TOP3_DATASET', str(L.HERE.parent.parent / 'dataset-lumi/TOP3_DATASET'))).resolve()
        if not (cls.dataset / 'dataset_manifest.json').exists():
            raise unittest.SkipTest('Extract the bundled dataset first, or set OPS_TOP3_DATASET')

    def test_entire_dataset_and_every_rank(self):
        P.validate(self.dataset)
        cases, coverage = L.make_plan(self.dataset, L.NODES)
        self.assertEqual(len(cases), 57)
        self.assertEqual(len(coverage), 48)
        self.assertTrue(all(c['status'] == 'COMPLETE' for c in coverage))
        observed, measured, smoke = [], [], []
        metadata = {}
        for c in cases:
            if c['mode'] == 2:
                source = self.dataset / c['source_run_path'] / 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv'
                metadata[c['case_id']] = list(P.metadata(source).values())
        with tempfile.TemporaryDirectory(prefix='top3_whole_dataset_') as tmp:
            campaign = Path(tmp); (campaign / 'build').mkdir()
            exe = campaign / 'build/fake-hip-app'; exe.write_text('CPU test only')
            L.atomic(campaign / 'build/build_manifest.json', dict(executable='build/fake-hip-app', sha256=L.sha(exe)))
            config = dict(dataset=str(self.dataset), dataset_manifest_sha256=L.sha(self.dataset / 'dataset_manifest.json'),
                          stamp='integration', nodes=list(L.NODES), cases=cases, resume=False)
            L.atomic(campaign / 'campaign.json', config)
            active = {}

            def execute(command, **kwargs):
                if command[0] == 'srun':
                    spec_path = Path(command[command.index('--rank-spec') + 1])
                    spec = L.read_json(spec_path); c = spec['case']; run = Path(spec['workdir'])
                    active.update(case=c, run=run, catalog=spec['catalog'])
                    (smoke if c['niter'] == 1 else measured).append(c['case_id'])
                    self.assertIn('--ntasks=%d' % c['ranks'], command)
                    for rank in range(c['ranks']):
                        with patch.dict(os.environ, SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank % 8)):
                            self.assertEqual(R.rank_worker(spec_path, run), 0)
                        if rank:
                            self.assertFalse((run / ('rank_%d' % rank)).exists())
                    return subprocess.CompletedProcess(command, 0)
                self.assertEqual(command[0], str(exe))
                c, run, env = active['case'], active['run'], kwargs['env']
                rank = int(os.environ['SLURM_PROCID'])
                self.assertIn('-gpudirect', command)
                self.assertEqual(env['MPICH_GPU_SUPPORT_ENABLED'], '1')
                self.assertEqual(env['ROCR_VISIBLE_DEVICES'], str(rank % 8))
                self.assertEqual(env['OPS_DISABLE_CSV_LOGS'], '0' if rank == 0 else '1')
                for axis, value in zip('XYZ', c['shape']):
                    self.assertIn('OPS_FORCE_DECOMP_%s=%d' % (axis, value), command)
                out = kwargs['stdout']
                if c['mode'] == 2:
                    blockfile = Path(env['OPS_BLOCKSIZE_CSV'])
                    self.assertEqual(blockfile, Path(active['catalog']) / 'blocks.csv')
                    self.assertEqual(len(L.rows(blockfile)), 25)
                    self.assertEqual(L.sha(blockfile), L.sha(self.dataset / c['catalog_path'] / 'blocks.csv'))
                    out.write('[OPS] Loaded 25 precomputed block sizes from %s\n' % blockfile)
                    if rank == 0:
                        L.write_csv(run / 'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv', metadata[c['case_id']])
                else:
                    self.assertNotIn('OPS_BLOCKSIZE_CSV', env)
                    self.assertIn('OPS_BLOCK_SIZE_X=32', command)
                out.write('GPU Direct enabled\nTotal Wall time 3.0\nTotal kernel time: 2.0\nTotal halo exchange time: 0.5\n')
                observed.append((c['case_id'], c['niter'], rank))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(R, 'gpu_preflight'), patch.object(L.subprocess, 'run', side_effect=execute):
                for n in L.NODES:
                    with patch.dict(os.environ, SLURM_JOB_NUM_NODES=str(n)), contextlib.redirect_stdout(io.StringIO()):
                        L.run_node(campaign, n)
                    root = campaign / ('OPS_LOGS_integration_n%03d' % n)
                    self.assertEqual(len(L.rows(root / 'summary.csv')), sum(c['nodes'] == n for c in cases))
                    self.assertEqual(len(L.rows(root / 'comparisons.csv')), 9)
                    for c in [c for c in cases if c['nodes'] == n]:
                        self.assertTrue(R.complete(R.target_path(root, c), dict(c, executable_sha256=L.sha(exe))))
                    print('PASS node section:', n, flush=True)
                self.assertEqual(len(smoke), 45)
                self.assertEqual(len(measured), 57)
                self.assertEqual(len(observed), sum(c['ranks'] * (2 if c['mode'] == 2 else 1) for c in cases))
                # Every prior result is resumed without another application launch.
                config['resume'] = True; L.atomic(campaign / 'campaign.json', config)
                launches = len(observed)
                for n in L.NODES:
                    with patch.dict(os.environ, SLURM_JOB_NUM_NODES=str(n)), contextlib.redirect_stdout(io.StringIO()):
                        L.run_node(campaign, n)
                self.assertEqual(len(observed), launches)
            L.collect(campaign)
            self.assertEqual(len(L.rows(campaign / 'summary_all_nodes.csv')), 57)
            self.assertEqual(len(L.rows(campaign / 'comparisons_all_nodes.csv')), 108)
            self.assertTrue(L.read_json(campaign / 'collection_status.json')['complete'])
            print('PASS:', launches, 'simulated rank executions; 57 measured cases, 45 smokes; all 12 node sections resumed.', flush=True)


if __name__ == '__main__':
    unittest.main()
