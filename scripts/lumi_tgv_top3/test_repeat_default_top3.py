"""Exercise the real embedded rank worker with simulated Slurm and TGV output."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'repeat_default_top3_n004.sh'
SHAPES = ['4x4x2', '2x16x1', '4x8x1', '2x8x2']


def config(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


APP = r'''#!/usr/bin/env python3
import csv, os, sys
from pathlib import Path
assert sys.argv[1:5] == ['1280', '1280', '1280', '500']
assert '-gpudirect' in sys.argv and 'OPS_TILING' not in sys.argv
assert os.environ['MPICH_GPU_SUPPORT_ENABLED'] == '1'
assert os.environ['ROCR_VISIBLE_DEVICES'] == os.environ['SLURM_LOCALID']
assert not any(k in os.environ for k in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'OPS_TILING'])
run = Path.cwd()
c = dict(line.split('=', 1) for line in (run/'config.txt').read_text().splitlines() if '=' in line)
shape = 'x'.join(next(a.split('=')[1] for a in sys.argv if a.startswith('OPS_FORCE_DECOMP_'+axis+'=')) for axis in 'XYZ')
index = ['4x4x2', '2x16x1', '4x8x1', '2x8x2'].index(shape)
rank, phase, rep = int(os.environ['SLURM_PROCID']), int(c['OPS_AUTOTUNE_MODE']), int(c['RUN_ITERATION'])
mode = 1 if phase == 1 and rank == 0 else (0 if phase == 1 else 2)
assert '-OPS_AUTOTUNE_MODE='+str(mode) in sys.argv
assert os.environ['OPS_AUTOTUNE_MODE'] == str(mode)
blocks = (32, index+1, 1)
if phase == 1:
 assert 'OPS_BLOCKSIZE_CSV' not in os.environ
 if rank != 0: assert 'OPS_BLOCK_SIZE_X=32' in sys.argv
else:
 assert '/decomp'+shape+'/blocks.csv' in os.environ['OPS_BLOCKSIZE_CSV']
 with open(os.environ['OPS_BLOCKSIZE_CSV']) as f: saved = list(csv.DictReader(f))
 assert len(saved) == 25
 assert all(tuple(int(r[a]) for a in ['bx','by','bz']) == blocks for r in saved)
 count = 24 if os.environ.get('MOCK_BAD_LOAD') and rank == 1 else 25
 print('Loaded %d precomputed block sizes from %s' % (count, os.environ['OPS_BLOCKSIZE_CSV']))
if rank == 0:
 assert os.environ['OPS_DISABLE_CSV_LOGS'] == '0'
 logs = Path(os.environ['OPS_LOGS_DIR'])/'autotune_on'; logs.mkdir(parents=True)
 with (logs/'ops_blocksize_tuning_logmod.csv').open('w') as f:
  w=csv.writer(f); w.writerow(['kernel_id','bx','by','bz','max_threads'])
  for k in range(1,27):
   actual = (32,4,1) if phase == 2 and os.environ.get('MOCK_WRONG_BLOCKS') else blocks
   w.writerow([k]+list(actual)+[1024])
 if phase == 1:
  with (logs/'ops_blocksize_best_logmod.csv').open('w') as f:
   w=csv.writer(f); w.writerow(['kernel_id','bx','by','bz','best_time'])
   w.writerow([1,32,4,1,1e300])
   stop = 26 if os.environ.get('MOCK_MISSING_KERNEL') else 27
   for k in range(2,stop): w.writerow([k]+list(blocks)+[1.0])
   w.writerow([3,32,4,1,99.0])
   w.writerow(['Autotune -- 1','','','',''])
else: assert os.environ['OPS_DISABLE_CSV_LOGS'] == '1'
print('GPU Direct enabled')
if rank == 0:
 print('Total Wall time: '+str(300 if phase == 1 else 160 + 10*rep - 5*index))
 print('Total kernel time: '+str(250 if phase == 1 else 130 + 8*rep - 4*index))
 print('Total halo exchange time: '+str(40 if phase == 1 else 25 + rep - index))
 if not os.environ.get('MOCK_NO_VOLUME'):
  volume = [206474,480654,293501,276588][index]
  if os.environ.get('MOCK_COUNTER_DRIFT') and phase == 2 and rep == 2: volume += 100
  print('Average volume of MPI communications per process: %g MB' % volume)
 print('Average number of MPI messages per process: 36000')
'''

SRUN = r'''#!/usr/bin/env python3
import csv, io, os, subprocess, sys
from pathlib import Path
for arg in ['--nodes=4','--ntasks=32','--ntasks-per-node=8','--cpus-per-task=7',
            '--distribution=block:block','--cpu-bind=map_cpu:49,57,17,25,1,9,33,41']:
 assert arg in sys.argv
start = sys.argv.index('-c')-1
run = Path(sys.argv[-1]).parent
c = dict(line.split('=',1) for line in (run/'config.txt').read_text().splitlines() if '=' in line)
with open(os.environ['MOCK_ORDER'],'a') as f:
 f.write(c['PRECOMPUTED_MPI']+','+c['OPS_AUTOTUNE_MODE']+','+c['RUN_ITERATION']+'\n')
rcs=[]
for rank in [0,1,7]:
 env=dict(os.environ,SLURM_PROCID=str(rank),SLURM_LOCALID=str(rank%8))
 rcs.append(subprocess.run(sys.argv[start:],env=env).returncode)
if any(rcs): sys.exit(1)
def status(path):
 line=next(s.split('\t',1)[1] for s in path.read_text().splitlines() if s.startswith('OPS_RANK_RESULT\t'))
 return next(csv.reader([line]))
root = status(run/'ranks/rank_0.log')
for rank in range(32):
 path=run/('ranks/rank_%d.log'%rank)
 text=path.read_text() if path.exists() else ''
 s=status(path) if path.exists() else list(root)
 s[0]=str(rank); s[1]=s[2]=str(rank%8); s[3]='mock-node-'+str(rank//8)
 if rank not in [0,1,7]:
  s[5]=''
  s[8]='0' if c['OPS_AUTOTUNE_MODE']=='1' else '2'
 if os.environ.get('MOCK_BAD_RANK') and rank==31 and c['OPS_AUTOTUNE_MODE']=='2': s[6]='FAILED'
 if os.environ.get('MOCK_MAPPING_DRIFT') and rank==31 and c['OPS_AUTOTUNE_MODE']=='2': s[10]='9999'
 buf=io.StringIO(); csv.writer(buf,lineterminator='').writerow(s)
 text='\n'.join(line for line in text.splitlines() if not line.startswith('OPS_RANK_RESULT\t'))
 path.write_text(text+'\nOPS_RANK_RESULT\t'+buf.getvalue()+'\n')
if os.environ.get('MOCK_MUTATE_CATALOG') and c['PRECOMPUTED_MPI']=='2x8x2' and c['OPS_AUTOTUNE_MODE']=='1':
 cat=run.parents[4]/'precomputed_blocks/decomp4x4x2/blocks.csv'
 cat.write_bytes(cat.read_bytes()+b'\n')
'''


class RepeatDefaultTop3Test(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='tgv_repeat_test_')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bin = self.root/'bin'; self.bin.mkdir()
        self.campaign = self.root/'TGV_TOP3_20261026_101612'
        self.exe = self.campaign/'build/tgvstorenone_mpi_hip'
        self.exe.parent.mkdir(parents=True)
        self.executable(self.exe, APP)
        self.executable(self.bin/'srun', SRUN)
        self.executable(self.bin/'ldd', '#!/bin/sh\necho "libmpi_gtl_hsa => /mock/libmpi_gtl_hsa.so"\n')
        self.executable(self.bin/'sbatch', '#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["MOCK_SUBMISSION"],"w") as f: json.dump(sys.argv[1:],f)\n')
        self.ref = self.campaign/'OPS_LOGS_20261026_101612_n004/tgvstorenone_hip_1280x1280x1280_cfg1_comparison_20261026_101612_n004/mode_1/decomp4x4x2/iter_1/autotune_precomputed/config.txt'
        self.ref.parent.mkdir(parents=True)
        c = dict(APP_NAME='tgvstorenone',GRID_X=1280,GRID_Y=1280,GRID_Z=1280,NITER=500,NUM_NODES=4,NUM_MPI_RANKS=32,
                 DECOMP_PX=4,DECOMP_PY=4,DECOMP_PZ=2,OPS_AUTOTUNE_MODE=2,GPU_DIRECT='YES',RUN_STATUS='OK',RUN_EXIT_CODE=0,
                 EXECUTABLE_SHA256=hashlib.sha256(self.exe.read_bytes()).hexdigest(),APP_WALL_TIME=99999)
        self.ref.write_text(''.join(str(k)+'='+str(v)+'\n' for k,v in c.items()))
        self.profile=self.root/'profile.sh'; self.profile.write_text('export MPICH_GPU_SUPPORT_ENABLED=1\n')
        self.output=self.root/'TGV_REPEAT_N004_20261103_120000_000001'
        self.env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        self.env.update(SLURM_JOB_ID='123',SLURM_JOB_NUM_NODES='4',SLURM_JOB_NODELIST='mock-node-[0-3]',
                        PATH=str(self.bin)+os.pathsep+os.environ['PATH'],MOCK_ORDER=str(self.root/'order.csv'),
                        MOCK_SUBMISSION=str(self.root/'submission.json'), OPS_BLOCKSIZE_CSV='/must/not/be/inherited.csv',
                        HIP_VISIBLE_DEVICES='99',OPS_TILING='1')

    def executable(self, path, text):
        path.write_text(text); path.chmod(0o755)

    def run_script(self, *extra):
        command=['bash',str(SCRIPT),'--campaign',str(self.campaign),'--env-file',str(self.profile),'--output',str(self.output)]
        return subprocess.run(command+list(extra),env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)

    def logs(self):
        return next(self.output.glob('OPS_LOGS_*_n004'))

    def run_path(self, shape, phase=2, rep=1):
        return next(self.logs().glob('tgv*/mode_1/decomp'+shape+'/iter_'+str(rep)+'/'+('autotune_on' if phase==1 else 'autotune_precomputed')))

    def assert_success(self, result):
        if result.returncode:
            logs='\n'.join(str(p)+'\n'+p.read_text()[-2000:] for p in self.output.rglob('run.log'))
            self.fail(result.stderr+'\n'+logs)

    def test_dry_run_and_no_action_never_launch(self):
        for extra in [(),('--dry-run',)]:
            result=self.run_script(*extra); self.assert_success(result)
            self.assertIn('MAIN_EXECUTIONS=16',result.stdout)
            self.assertIn('REFERENCE_KIND=SAME_JOB_SAME_ROUND',result.stdout)
            self.assertIn('DECOMPOSITIONS='+','.join(SHAPES),result.stdout)
            self.assertFalse(self.output.exists())
            self.assertFalse((self.root/'order.csv').exists())

    def test_complete_protocol_and_same_round_comparisons(self):
        self.assert_success(self.run_script('--run'))
        order=''.join(s+',1,1\n' for s in SHAPES)
        order+=''.join(s+',2,'+str(rep)+'\n' for rep in range(1,4) for s in SHAPES)
        self.assertEqual((self.root/'order.csv').read_text(),order)
        summary=table(self.logs()/'summary.csv'); comparisons=table(self.logs()/'comparisons.csv')
        self.assertEqual(len(summary),16); self.assertEqual(len(comparisons),45)
        self.assertEqual(len(table(self.logs()/'statistics.csv')),20)
        self.assertEqual(len(table(self.logs()/'comparison_statistics.csv')),15)
        self.assertEqual(len(list(self.output.rglob('*.json'))),0)
        hashes=[]
        for shape in SHAPES:
            trained=config(self.run_path(shape,1)/'config.txt')
            self.assertEqual(trained['SCIENTIFIC_RESULT'],'NO')
            modes=table(self.run_path(shape,1)/'mpi_rank_metrics.csv')
            self.assertEqual([r['rank_id'] for r in modes if r['autotune_mode']=='1'],['0'])
            hashes.append(trained['PRECOMPUTED_SET_ID'])
            for rep in range(1,4):
                c=config(self.run_path(shape,2,rep)/'config.txt')
                self.assertEqual(c['PRECOMPUTED_SET_ID'],trained['PRECOMPUTED_SET_ID'])
                self.assertEqual(c['PRECOMPUTED_SOURCE_CONFIG'],str(self.run_path(shape,1)/'config.txt'))
                self.assertEqual(c['DEFAULT_MPI_PRECOMPUTED_CONFIG'],str(self.run_path(SHAPES[0],2,rep)/'config.txt'))
                self.assertEqual(float(c['WALL_TIME_DEFAULT_MPI_PRECOMPUTED']),160+10*rep)
                self.assertEqual(c['COMMUNICATION_COUNTERS_STABLE'],'YES')
                self.assertEqual(c['COMMIT_STATUS'],'COMMITTED')
                self.assertEqual(c['SLURM_JOB_ID'],'123')
        self.assertEqual(len(set(hashes)),4)
        self.assertEqual(len({r['rank_placement_sha256'] for r in summary}),1)
        for row in comparisons:
            rep=int(row['iteration']); index=SHAPES.index(row['mpi'])
            if row['metric']=='wall_time_s':
                self.assertEqual(float(row['baseline']),160+10*rep)
                self.assertAlmostEqual(float(row['baseline_over_measured']),(160+10*rep)/(160+10*rep-5*index))
            if row['metric']=='sent_mb_per_rank':
                self.assertEqual(float(row['baseline']),206474)
        self.assertNotEqual(self.run_script('--run').returncode,0)

    def test_invalid_training_stops_before_replay(self):
        self.env['MOCK_MISSING_KERNEL']='1'
        result=self.run_script('--run'); self.assertNotEqual(result.returncode,0)
        self.assertEqual((self.root/'order.csv').read_text(),'4x4x2,1,1\n')
        self.assertFalse((self.logs()/'precomputed_blocks').exists())
        self.assertEqual(config(self.run_path(SHAPES[0],1)/'config.txt')['RUN_STATUS'],'FAILED')

    def test_missing_volume_is_not_silently_ignored(self):
        self.env['MOCK_NO_VOLUME']='1'
        result=self.run_script('--run'); self.assertNotEqual(result.returncode,0)
        self.assertIn('SENT_MB_PER_RANK',result.stderr)

    def test_failed_rank_or_load_prevents_comparison(self):
        for flag in ['MOCK_BAD_RANK','MOCK_BAD_LOAD','MOCK_MAPPING_DRIFT','MOCK_WRONG_BLOCKS','MOCK_MUTATE_CATALOG']:
            with self.subTest(flag=flag):
                self.output=self.root/flag
                self.env[flag]='1'
                result=self.run_script('--run')
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(config(self.run_path(SHAPES[0])/ 'config.txt')['RUN_STATUS'],'FAILED')
                self.assertFalse((self.logs()/'comparisons.csv').exists())
                self.env.pop(flag)

    def test_counter_drift_is_reported_and_preserved(self):
        self.env['MOCK_COUNTER_DRIFT']='1'
        result=self.run_script('--run','--repetitions','2'); self.assert_success(result)
        self.assertIn('WARNING: reported communication counters changed',result.stdout)
        self.assertEqual(config(self.run_path(SHAPES[0],2,2)/'config.txt')['COMMUNICATION_COUNTERS_STABLE'],'NO')

    def test_submit_one_job_pins_inputs_and_does_not_run(self):
        self.env.pop('SLURM_JOB_ID')
        result=self.run_script('--submit'); self.assert_success(result)
        command=json.loads((self.root/'submission.json').read_text())
        self.assertIn('--time=01:30:00',command)
        self.assertEqual(command[command.index('--repetitions')+1],'3')
        self.assertIn('--run',command)
        self.assertEqual(command[command.index('--expected-executable-sha256')+1],hashlib.sha256(self.exe.read_bytes()).hexdigest())
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root/'order.csv').exists())

    def test_invalid_context_and_changed_queued_inputs(self):
        for extra in [('--repetitions','0'),('--time','00:00:00'),('--expected-executable-sha256','wrong'),
                      ('--expected-environment-sha256','wrong')]:
            with self.subTest(extra=extra): self.assertNotEqual(self.run_script('--dry-run',*extra).returncode,0)
        self.ref.write_text(self.ref.read_text().replace('GRID_X=1280','GRID_X=640'))
        self.assertNotEqual(self.run_script('--submit').returncode,0)
        self.assertFalse((self.root/'submission.json').exists())


if __name__ == '__main__':
    unittest.main()
