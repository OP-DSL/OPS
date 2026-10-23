"""CPU checks for the tiled experiment; no Slurm or GPU execution."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lumi_tgv_top3'))
import launcher as L
import runtime as R


class TiledTests(unittest.TestCase):
    def test_real_tiled_halo_output_and_invalid_totals(self):
        with tempfile.TemporaryDirectory(prefix='tgv_tiled_metrics_') as temp:
            log = Path(temp)/'run.log'
            for label in ('Total tiled halo exchange time', 'Total halo exchange time'):
                log.write_text('Total kernel time: 0.320326\n' + label +
                               ': 2.38419e-06\nTotal user halo exchange time: 0.12\n'
                               'Total Wall time 0.346961\n')
                self.assertEqual(R.ops_totals(log), dict(kernel_time_s=0.320326, halo_time_s=2.38419e-06))
            for halo_lines in ('', 'Total tiled halo exchange time: -1\n',
                               'Total tiled halo exchange time: 1e999\n',
                               'Total tiled halo exchange time: 0.2\nTotal halo exchange time: 0.3\n',
                               'Total tiled halo exchange time: 0.2\nTotal tiled halo exchange time: 0.3\n'):
                log.write_text('Total kernel time: 0.320326\n' + halo_lines)
                with self.assertRaises(ValueError):
                    R.ops_totals(log)

    def test_all_nodes_keep_default_and_precomputed_baselines(self):
        dataset = L.HERE.parents[1] / 'dataset-lumi/TOP3_DATASET'
        cases, coverage = L.make_plan(dataset, L.NODES)
        self.assertEqual(len(cases), 57)
        self.assertTrue(all(row['status'] == 'COMPLETE' for row in coverage))
        for n in L.NODES:
            local = [case for case in cases if case['nodes'] == n]
            roles = {role for case in local for role in case['roles']}
            self.assertEqual(roles, {'A', 'C', 'D_top1', 'D_top2', 'D_top3'})
            self.assertEqual(next(c for c in local if 'A' in c['roles'])['mode'], 0)
            self.assertEqual(next(c for c in local if 'C' in c['roles'])['mode'], 2)
        args = argparse.Namespace(campaign=None, output_root=Path('/tmp'), account='', date_stamp='test', tiled=True)
        self.assertEqual(L.campaign_directory(args).name, 'TGV_TOP3_TILED_test')
        self.assertEqual(L.build_target(vars(args)), 'opensbli_mpi_hip_tiled')
        args.tiled = False
        self.assertEqual(L.campaign_directory(args).name, 'TGV_TOP3_test')
        self.assertEqual(L.build_target(vars(args)), 'opensbli_mpi_hip')

    def test_tiled_worker_flags_and_confirmation(self):
        with tempfile.TemporaryDirectory(prefix='tgv_tiled_worker_') as temp:
            root = Path(temp)
            catalog = root/'catalog'; catalog.mkdir()
            (catalog/'blocks.csv').write_text('kernel_id,bx,by,bz\n2,128,1,2\n')
            R.atomic(catalog/'manifest.json', dict(schema_version=2))
            binary = root/'app'
            binary.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path('arguments.json').write_text(json.dumps(sys.argv[1:]))
Path('environment.json').write_text(json.dumps(dict(os.environ)))
print('GPU Direct enabled')
print('Total Wall time 1.25')
if 'OPS_BLOCKSIZE_CSV' in os.environ:
    print('Loaded 1 precomputed block sizes from ' + os.environ['OPS_BLOCKSIZE_CSV'])
if not Path('omit_tiling_confirmation').exists():
    print('Tiling enabled')
    print('Max tiling depth across processes = 6')
    for axis in 'XYZ': print('Tile size in ' + axis + ' = 8000')
''')
            binary.chmod(0o755)
            for mode in (0, 2):
                for steps in (1, 500):
                    run = root/('run_%d_%d' % (mode, steps)); (run/'ranks').mkdir(parents=True)
                    case = dict(grid=[1008]*3, shape=[4,2,2], niter=steps, mode=mode, tiling=L.TILING)
                    R.atomic(run/'spec.json', dict(case=case, executable=str(binary), workdir=str(run), catalog=str(catalog), compatibility_smoke=steps == 1))
                    with patch.dict(os.environ, SLURM_PROCID='0', SLURM_LOCALID='0', OPS_TILING_TRACE='1'):
                        self.assertEqual(R.rank_worker(run/'spec.json', run), 0)
                    args = json.loads((run/'arguments.json').read_text())
                    for arg in ('-gpudirect', '-OPS_DIAGS=2', 'OPS_TILING', 'OPS_TILING_MAXDEPTH=6',
                                'OPS_TILESIZE_X=8000', 'OPS_TILESIZE_Y=8000', 'OPS_TILESIZE_Z=8000'):
                        self.assertIn(arg, args)
                    env = json.loads((run/'environment.json').read_text())
                    self.assertEqual(env['MPICH_GPU_SUPPORT_ENABLED'], '1')
                    self.assertEqual(env['OPS_TILING_TRACE'], '1' if steps == 1 else '0')
                    self.assertEqual('OPS_BLOCKSIZE_CSV' in env, mode == 2)
                    self.assertEqual('OPS_BLOCK_SIZE_X=32' in args, mode == 0)
            (run/'omit_tiling_confirmation').touch()
            with patch.dict(os.environ, SLURM_PROCID='0', SLURM_LOCALID='0'):
                self.assertEqual(R.rank_worker(run/'spec.json', run), 1)

    def test_all_catalogs_with_empty_rank_zero_boundaries(self):
        dataset = L.HERE.parents[1] / 'dataset-lumi/TOP3_DATASET'
        cases, _ = L.make_plan(dataset, L.NODES)
        tested = 0
        with tempfile.TemporaryDirectory(prefix='tgv_empty_boundary_') as temp:
            current, trace = Path(temp)/'current.csv', Path(temp)/'rank_0.log'
            for case in cases:
                if case['mode'] != 2:
                    continue
                historical = dataset/case['source_run_path']/'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv'
                old = L.first_kernel_rows(historical)
                # TGV high-face kernels have empty rank-zero ranges on split axes.
                # This simulates planner evidence; production uses actual ranges.
                absent = {str(kid) for kid, parts in zip((3, 5, 7), case['shape']) if parts > 1}
                self.assertTrue(absent <= set(old))
                L.write_csv(current, [row for kid, row in old.items() if kid not in absent])
                trace.write_text(''.join('[OPS] Tiled range kernel_id=' + kid +
                                        ' dim=3 range=0,0,0,10,0,10\n' for kid in absent))
                blocks = dataset/case['catalog_path']/'blocks.csv'
                before = L.sha(blocks)
                report = L.validate_transfer(historical, current, blocks, L.TILING, trace)
                self.assertEqual(report['rank0_empty_kernel_ids'], sorted(absent, key=int))
                self.assertFalse(absent & set(report['block_limits_checked_kernel_ids']))
                self.assertEqual(L.sha(blocks), before)
                with self.assertRaisesRegex(ValueError, 'missing without empty-range evidence'):
                    L.validate_transfer(historical, current, blocks, L.TILING)
                tested += 1
            self.assertEqual(tested, 45)
            # A single executed tile invalidates an otherwise empty-kernel excuse.
            with trace.open('a') as f:
                f.write('[OPS] Tiled range kernel_id=3 dim=3 range=0,1,0,10,0,10\n')
            with self.assertRaisesRegex(ValueError, 'missing without empty-range evidence=3'):
                L.validate_transfer(historical, current, blocks, L.TILING, trace)
            trace.write_text('[OPS] Tiled range kernel_id=3 dim=3 range=1,0,0,10,0,10\n')
            with self.assertRaisesRegex(ValueError, 'Negative tiled range'):
                L.validate_transfer(historical, current, blocks, L.TILING, trace)
            trace.write_text('[OPS] Tiled range kernel_id=3 dim=3 range=0,0\n')
            with self.assertRaisesRegex(ValueError, 'Invalid tiled range dimensions'):
                L.validate_transfer(historical, current, blocks, L.TILING, trace)

    def test_transfer_records_geometry_but_rejects_changed_stencil(self):
        # Reuse the non-tiled regression fixture for all required metadata fields.
        from test_launcher import LauncherTests
        fixture = LauncherTests(); fixture.setUp()
        try:
            old, new = fixture.root/'old.csv', fixture.root/'new.csv'
            fixture.metadata(old); fixture.metadata(new, grid_x='30')
            catalog = fixture.catalog()
            with self.assertRaisesRegex(ValueError, 'changed grid_x'):
                L.validate_transfer(old, new, catalog/'blocks.csv')
            report = L.validate_transfer(old, new, catalog/'blocks.csv', L.TILING)
            self.assertEqual(report['geometry_changes'], [dict(kernel_id='2', field='grid_x', historical='24', tiled='30')])
            row = L.rows(new)[0]; row['stencil_sig'] = 'different-kernel'
            L.write_csv(new, [row])
            with self.assertRaisesRegex(ValueError, 'changed stencil_sig'):
                L.validate_transfer(old, new, catalog/'blocks.csv', L.TILING)
        finally:
            fixture.tearDown()

    def test_tiled_smokes_catalogs_measurements_collect_and_resume(self):
        from test_launcher import LauncherTests
        fixture = LauncherTests(); fixture.setUp()
        try:
            fixture.test_node_pipeline_smokes_results_transfer_and_resume(tiling=L.TILING)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
