"""Local orchestration tests; Slurm and GPU execution are simulated."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'compare_yz_n016.sh'


class CompareYZTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tgv_yz_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.exe = self.make_executable('tgv', '''#!/usr/bin/env python3
import csv, os, sys
from pathlib import Path
assert sys.argv[1:4] == ['2016'] * 3
assert '-gpudirect' in sys.argv and '-OPS_AUTOTUNE_MODE=2' in sys.argv
assert os.environ['MPICH_GPU_SUPPORT_ENABLED'] == '1'
assert os.environ['ROCR_VISIBLE_DEVICES'] == '0'
shape = [int(next(a.split('=')[1] for a in sys.argv if a.startswith('OPS_FORCE_DECOMP_'+axis+'='))) for axis in 'XYZ']
assert shape in [[8,4,4],[2,8,8]]
local = [2016 // v for v in shape]
logs = Path(os.environ['OPS_LOGS_DIR']) / 'autotune_on'
logs.mkdir(parents=True)
with (logs / 'ops_blocksize_tuning_logmod.csv').open('w') as f:
 w = csv.writer(f); w.writerow(['kernel_id','grid_x','grid_y','grid_z','bx','by','bz'])
 for k in [2,23,24]:
  g = [1,local[1]+2,local[2]+2] if k == 2 else local
  w.writerow([k] + g + [32,4,1])
print('Loaded 25 precomputed block sizes from ' + os.environ['OPS_BLOCKSIZE_CSV'])
print('GPU Direct enabled')
rep = int(Path.cwd().name.split('_')[-1])
wall = ([170,180,160] if shape == [8,4,4] else [150,140,160])[rep-1]
print('Total Wall time: ' + str(wall))
print('Total kernel time: 130.0')
print('Total halo exchange time: 20.0')
''')
        self.make_executable('ldd', '#!/bin/sh\necho "libmpi_gtl_hsa => /mock/libmpi_gtl_hsa.so"\n')
        self.make_executable('srun', '''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
assert '--nodes=16' in sys.argv and '--ntasks=128' in sys.argv
assert '--ntasks-per-node=8' in sys.argv and '--cpus-per-task=7' in sys.argv
assert '--cpu-bind=map_cpu:49,57,17,25,1,9,33,41' in sys.argv
start = sys.argv.index('-c') - 1
run = Path(sys.argv[-1])
with (run.parents[2] / 'mock_order.jsonl').open('a') as f:
 f.write(json.dumps(json.loads((run / 'execution.json').read_text())['case']) + '\\n')
env = dict(os.environ, SLURM_PROCID='0', SLURM_LOCALID='0')
rc = subprocess.run(sys.argv[start:], env=env).returncode
if rc: sys.exit(rc)
status = json.loads((run / 'ranks/status_0.json').read_text())
for rank in range(1,128):
 s = dict(status, rank_id=rank, local_rank=rank%8, gpu_id=rank%8)
 s['app_reported_wall_time_s'] += 2
 if os.environ.get('MOCK_BAD_RANK') and rank == 127:
  s['rank_status'] = 'FAILED'
 (run / ('ranks/status_%d.json' % rank)).write_text(json.dumps(s))
''')
        self.blocks = self.root / 'blocks.csv'
        self.blocks.write_text('kernel_id,bx,by,bz\n' + ''.join('%d,32,4,1\n' % k for k in range(2,27)))
        self.profile = self.root / 'profile.sh'
        self.profile.write_text('export PATH="' + str(self.bin) + ':$PATH"\n')
        self.output = self.root / 'output'
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
        self.env.update(SLURM_JOB_ID='123', SLURM_JOB_NUM_NODES='16')

    def make_executable(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)
        return path

    def run_script(self, *extra):
        return subprocess.run(['bash', str(SCRIPT), '--ops-root', str(HERE.parents[1]),
            '--executable', str(self.exe), '--blocks', str(self.blocks),
            '--env-file', str(self.profile), '--output', str(self.output)] + list(extra),
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def test_dry_run_and_invalid_catalog(self):
        result = self.run_script('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        self.blocks.write_text('kernel_id,bx,by,bz\n2,32,4,1\n')
        result = self.run_script('--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Expected exactly', result.stderr)
        self.assertFalse(self.output.exists())

    def test_full_comparison(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.output / 'summary.csv').open() as f:
            summary = list(csv.DictReader(f))
        self.assertEqual([r['shape'] for r in summary], ['8x4x4', '2x8x8'])
        self.assertEqual([float(r['wall_time_s']) for r in summary], [172, 152])
        self.assertAlmostEqual(float(summary[1]['wall_speedup_vs_8x4x4']), 172/152)
        order = [json.loads(line) for line in (self.output / 'mock_order.jsonl').read_text().splitlines()]
        self.assertEqual([r['niter'] for r in order], [1,1,500,500,500,500,500,500])
        self.assertEqual([r['shape'][0] for r in order], [8,2,8,2,2,8,8,2])
        with (self.output / 'rank0_grids.csv').open() as f:
            grids = list(csv.DictReader(f))
        self.assertEqual(len(grids), 18)
        self.assertTrue(any(r['shape'] == '2x8x8' and r['kernel_id'] == '23'
                            and r['grid_x'] == '1008' for r in grids))
        # No overwrite/resume that could mix experiments.
        again = self.run_script()
        self.assertNotEqual(again.returncode, 0)
        self.assertIn('Output already exists', again.stderr)

    def test_failed_rank_stops_before_measurements(self):
        self.env['MOCK_BAD_RANK'] = '1'
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Run failed', result.stderr)
        self.assertFalse((self.output / 'measurements').exists())
        self.assertFalse((self.output / 'summary.csv').exists())

    def test_requires_16_node_allocation(self):
        self.env['SLURM_JOB_NUM_NODES'] = '8'
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exactly 16 nodes', result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
