"""Check the existing Top3 output contract; Slurm/GPU execution is simulated."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'compare_yz_n016.sh'


def config(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


class CompareYZTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tgv_yz_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.campaign = self.root / 'TGV_TOP3_20261022_083420'
        self.exe = self.campaign / 'build/tgvstorenone_mpi_hip'
        self.exe.parent.mkdir(parents=True)
        self.make_executable(self.exe, '''#!/usr/bin/env python3
import csv, os, sys
from pathlib import Path
assert sys.argv[1:4] == ['2016'] * 3
assert sys.argv[4] in ['1','500']
assert '-gpudirect' in sys.argv and '-OPS_AUTOTUNE_MODE=2' in sys.argv
assert 'OPS_TILING' not in sys.argv
assert os.environ['MPICH_GPU_SUPPORT_ENABLED'] == '1'
assert os.environ['ROCR_VISIBLE_DEVICES'] == os.environ['SLURM_LOCALID']
assert not any(k in os.environ for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'])
shape = [int(next(a.split('=')[1] for a in sys.argv if a.startswith('OPS_FORCE_DECOMP_'+axis+'='))) for axis in 'XYZ']
assert shape != [8,4,4], 'Historical reference must NEVER run'
assert shape in [[2,8,8],[4,4,8],[2,16,4]]
local = [2016 // v for v in shape]
rank = int(os.environ['SLURM_PROCID'])
if rank == 0 and not (os.environ.get('MOCK_MISSING_CSV') and sys.argv[4] == '500'):
 logs = Path(os.environ['OPS_LOGS_DIR']) / 'autotune_on'
 logs.mkdir(parents=True)
 with (logs / 'ops_blocksize_tuning_logmod.csv').open('w') as f:
  w = csv.writer(f); w.writerow(['kernel_id','grid_x','grid_y','grid_z','bx','by','bz'])
  for k in [2,23,24]:
   g = [1,local[1]+2,local[2]+2] if k == 2 else local
   w.writerow([k] + g + [32,4,1])
if rank != 0:
 assert os.environ['OPS_DISABLE_CSV_LOGS'] == '1'
print('Loaded 25 precomputed block sizes from ' + os.environ['OPS_BLOCKSIZE_CSV'])
print('GPU Direct enabled')
rep = int(Path.cwd().parent.name.split('_')[-1])
print('Total Wall time: ' + str([150,140,160][rep-1]))
print('Total kernel time: 130.0')
print('Total halo exchange time: 20.0')
''')
        self.make_executable(self.bin / 'ldd', '#!/bin/sh\necho "libmpi_gtl_hsa => /mock/libmpi_gtl_hsa.so"\n')
        self.make_executable(self.bin / 'srun', '''#!/usr/bin/env python3
import csv, os, subprocess, sys
from pathlib import Path
assert '--nodes=16' in sys.argv and '--ntasks=128' in sys.argv
assert '--ntasks-per-node=8' in sys.argv and '--cpus-per-task=7' in sys.argv
assert '--cpu-bind=map_cpu:49,57,17,25,1,9,33,41' in sys.argv
start = sys.argv.index('bash')
conf = Path(sys.argv[-1]); run = conf.parent
c = dict(line.split('=',1) for line in conf.read_text().splitlines() if '=' in line)
with open(os.environ['MOCK_ORDER'], 'a') as f:
 f.write(c['PRECOMPUTED_MPI'] + ',' + c['NITER'] + ',' + c['RUN_ITERATION'] + '\\n')
for rank in [0,7]:
 env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank))
 rc = subprocess.run(sys.argv[start:], env=env).returncode
 if rc: sys.exit(rc)
with (run / 'ranks/status_0.csv').open() as f:
 status = next(csv.DictReader(f))
for rank in range(1,128):
 if rank == 7: continue
 s = dict(status, rank_id=rank, local_rank=rank%8, gpu_id=rank%8)
 s['app_reported_wall_time_s'] = float(s['app_reported_wall_time_s']) + 2
 if os.environ.get('MOCK_BAD_RANK') and rank == 127 and c['NITER'] == '500':
  s['rank_status'] = 'FAILED'
 with (run / ('ranks/status_%d.csv' % rank)).open('w') as f:
  w = csv.DictWriter(f, fieldnames=list(s)); w.writeheader(); w.writerow(s)
''')
        logs = self.campaign / 'OPS_LOGS_20261022_083420_n016'
        self.blocks = logs / 'precomputed_blocks/decomp8x4x4/blocks.csv'
        self.blocks.parent.mkdir(parents=True)
        self.blocks.write_text('kernel_id,bx,by,bz\n' + ''.join('%d,32,4,1\n' % k for k in range(2,27)))
        self.ref_paths = {}
        for mode, variant, wall, kernel, halo in [(0,'autotune_off',229.995363,198.507,31.0007),
                                                  (2,'autotune_precomputed',170.818726,144.144,26.1776)]:
            p = logs / 'tgvstorenone_hip_2016x2016x2016_cfg1_comparison_20261022_083420_n016/mode_1/decomp8x4x4/iter_1' / variant / 'config.txt'
            p.parent.mkdir(parents=True)
            c = dict(APP_NAME='tgvstorenone', GRID_X=2016, GRID_Y=2016, GRID_Z=2016, NITER=500,
                     NUM_NODES=16, NUM_MPI_RANKS=128, DECOMP_PX=8, DECOMP_PY=4, DECOMP_PZ=4,
                     OPS_AUTOTUNE_MODE=mode, GPU_DIRECT='YES', RUN_STATUS='OK', RUN_EXIT_CODE=0,
                     EXECUTABLE_SHA256=hashlib.sha256(self.exe.read_bytes()).hexdigest(),
                     APP_WALL_TIME=wall, KERNEL_TOTAL_TIME=kernel, HALO_TOTAL_TIME=halo)
            p.write_text(''.join(str(k)+'='+str(v)+'\n' for k,v in c.items()))
            self.ref_paths[mode] = p
        self.profile = self.root / 'profile.sh'
        self.profile.write_text('export PATH="' + str(self.bin) + ':$PATH"\n')
        self.output = self.root / 'output'
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
        self.env.update(SLURM_JOB_ID='123', SLURM_JOB_NUM_NODES='16', MOCK_ORDER=str(self.root / 'order.csv'))

    def make_executable(self, path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_script(self, *extra, auto_output=False):
        command = ['bash', str(SCRIPT), '--ops-root', str(HERE.parents[1]), '--campaign', str(self.campaign),
                   '--env-file', str(self.profile)]
        if not auto_output:
            command += ['--output', str(self.output)]
        return subprocess.run(command + list(extra), env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)

    def logroot(self):
        return next(self.output.glob('OPS_LOGS_*_n016'))

    def test_dry_run_resolves_paths_and_keeps_global_input(self):
        r = self.run_script('--dry-run', '--account', 'project_yz_test', auto_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        p = dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
        self.assertEqual(p['APP_ARGS'], '2016 2016 2016 500')
        self.assertEqual(p['DECOMPOSITIONS'], '2x8x8')
        self.assertEqual(p['REPETITIONS'], '1')
        self.assertEqual(p['REFERENCE_RERUN'], 'NO')
        self.assertEqual(p['EXECUTABLE_SOURCE'], str(self.exe))
        self.assertEqual(p['BLOCKS_SOURCE'], str(self.blocks))
        self.assertTrue(p['OUTPUT_DIRECTORY'].startswith('/scratch/project_yz_test/ops-decomposition/campaigns/TGV_YZ_N016_'))
        self.assertFalse(Path(p['OUTPUT_DIRECTORY']).exists())

    def test_top3_output_contract_and_no_json_or_baseline_rerun(self):
        self.output = self.root / 'TGV_YZ_N016_20261031_120000_000001'
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stderr)
        logs = self.logroot()
        self.assertEqual(logs.name, 'OPS_LOGS_20261031_120000_000001_n016')
        measured = list(logs.glob('tgv*/mode_1/decomp2x8x8/iter_1/autotune_precomputed/config.txt'))
        self.assertEqual(len(measured), 1)
        c = config(measured[0])
        expected = dict(APP_ARGS='2016 2016 2016 500', GRID_X='2016', GRID_Y='2016', GRID_Z='2016',
                        DECOMP_PX='2', DECOMP_PY='8', DECOMP_PZ='8', NUM_NODES='16', NUM_MPI_RANKS='128',
                        RUN_STATUS='OK', COMMIT_STATUS='COMMITTED', VALIDATION_STATUS='PASS', RUN_EXIT_CODE='0',
                        GPU_DIRECT='YES', OPS_AUTOTUNE_MODE='2', REFERENCE_KIND='HISTORICAL', REFERENCE_RERUN='NO')
        for k,v in expected.items():
            self.assertEqual(c[k], v, k)
        self.assertEqual(float(c['APP_WALL_TIME']), 152)
        self.assertAlmostEqual(float(c['SPEEDUP_DEFAULT_MPI_PRECOMPUTED']), 170.818726/152)
        self.assertEqual(c['DEFAULT_MPI_PRECOMPUTED_CONFIG'], str(self.ref_paths[2]))
        for name in ['run.log','srun.log','mpi_rank_metrics.csv','command.txt']:
            self.assertTrue((measured[0].parent / name).is_file(), name)
        self.assertEqual(len(table(measured[0].parent / 'mpi_rank_metrics.csv')), 128)
        summary = table(logs / 'summary.csv')
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]['mpi'], '2x8x8')
        self.assertEqual(summary[0]['global_grid'], '2016x2016x2016')
        self.assertEqual(len(table(logs / 'comparisons.csv')), 3)
        self.assertEqual(len(table(logs / 'rank0_grids.csv')), 3)
        self.assertEqual((self.root / 'order.csv').read_text(), '2x8x8,1,1\n2x8x8,500,1\n')
        self.assertEqual(list(self.output.rglob('*.json')), [])
        smoke = next(logs.glob('_compatibility/tgv*/mode_1/decomp*/iter_1/autotune_precomputed/config.txt'))
        self.assertEqual(config(smoke)['SCIENTIFIC_RESULT'], 'NO')
        self.assertEqual(config(smoke)['SPEEDUP_DEFAULT_MPI_PRECOMPUTED'], 'NA')
        again = self.run_script()
        self.assertNotEqual(again.returncode, 0)
        self.assertIn('Output already exists', again.stderr)

    def test_config_records_failure_and_excludes_failed_measurement(self):
        self.env['MOCK_BAD_RANK'] = '1'
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        logs = self.logroot()
        failed = next(logs.glob('tgv*/mode_1/decomp*/iter_1/autotune_precomputed/config.txt'))
        self.assertEqual(config(failed)['RUN_STATUS'], 'FAILED')
        self.assertEqual(config(failed)['COMMIT_STATUS'], 'UNCOMMITTED')
        self.assertEqual(config(self.output / 'config.txt')['RUN_STATUS'], 'FAILED')
        self.assertFalse((logs / 'summary.csv').exists())
        self.assertEqual(list(self.output.rglob('*.json')), [])

    def test_missing_grid_csv_cannot_be_reported_as_success(self):
        self.env['MOCK_MISSING_CSV'] = '1'
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        failed = next(self.logroot().glob('tgv*/mode_1/decomp*/iter_1/autotune_precomputed/config.txt'))
        self.assertEqual(config(failed)['VALIDATION_STATUS'], 'FAIL')

    def test_selected_shapes_keep_input_and_iteration_structure(self):
        r = self.run_script('--decompositions', '4x4x8,2x16x4', '--repetitions', '2')
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = table(self.logroot() / 'summary.csv')
        self.assertEqual(len(rows), 4)
        self.assertEqual({row['mpi'] for row in rows}, {'4x4x8','2x16x4'})
        self.assertEqual({row['iteration'] for row in rows}, {'1','2'})
        self.assertTrue(all(row['global_grid'] == '2016x2016x2016' for row in rows))

    def test_rejects_baseline_rerun_bad_decomposition_and_wrong_allocation(self):
        for shape in ['8x4x4','2x4x4','2x8x8,2x8x8']:
            r = self.run_script('--decompositions', shape, '--dry-run')
            self.assertNotEqual(r.returncode, 0)
        self.env['SLURM_JOB_NUM_NODES'] = '8'
        r = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('exactly 16 nodes', r.stderr)
        self.assertFalse(self.output.exists())

    def test_submit_validates_before_sbatch_and_pins_parameters(self):
        self.env.pop('SLURM_JOB_ID')
        self.env.pop('SLURM_JOB_NUM_NODES')
        self.env['PATH'] = str(self.bin) + os.pathsep + self.env['PATH']
        capture = self.root / 'sbatch_args.txt'
        self.env['MOCK_SBATCH_CAPTURE'] = str(capture)
        self.make_executable(self.bin / 'sbatch', '''#!/usr/bin/env python3
import os, sys
from pathlib import Path
Path(os.environ['MOCK_SBATCH_CAPTURE']).write_text('\\n'.join(sys.argv[1:]))
print('Submitted batch job 456')
''')
        r = self.run_script('--submit', '--account', 'project_yz_test')
        self.assertEqual(r.returncode, 0, r.stderr)
        sent = capture.read_text().splitlines()
        for key,value in [('--campaign',str(self.campaign)),('--executable',str(self.exe)),
                          ('--blocks',str(self.blocks)),('--output',str(self.output)),
                          ('--decompositions','2x8x8'),('--repetitions','1')]:
            self.assertEqual(sent[sent.index(key)+1], value)
        self.assertFalse(self.output.exists())
        capture.unlink()
        self.ref_paths[2].write_text(self.ref_paths[2].read_text().replace('GRID_X=2016','GRID_X=300'))
        r = self.run_script('--submit')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(capture.exists())

    def test_invalid_catalog_fails_preflight(self):
        self.blocks.write_text('kernel_id,bx,by,bz\n2,32,4,1\n')
        r = self.run_script('--dry-run')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Expected exactly', r.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
