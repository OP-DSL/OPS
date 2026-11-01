"""Exercise the sampled tuning -> own-catalog replay and strict output contract.
Slurm and GPU execution are simulated; rank worker code itself is executed.
"""
import csv
import hashlib
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'compare_yz_n016.sh'
SHAPES = ['2x8x8','4x4x8','2x16x4']


def config(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


class CompareYZTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tgv_yz_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root/'bin'; self.bin.mkdir()
        self.campaign = self.root/'TGV_TOP3_20261022_083420'
        self.exe = self.campaign/'build/tgvstorenone_mpi_hip'
        self.exe.parent.mkdir(parents=True)
        self.executable(self.exe, '''#!/usr/bin/env python3
import csv, os, sys
from pathlib import Path
grid=int(os.environ.get('MOCK_GRID','2016'))
assert sys.argv[1:5] == [str(grid)]*3+['500'], 'Input/steps changed'
assert '-gpudirect' in sys.argv and 'OPS_TILING' not in sys.argv
assert os.environ['MPICH_GPU_SUPPORT_ENABLED'] == '1'
assert os.environ['ROCR_VISIBLE_DEVICES'] == os.environ['SLURM_LOCALID']
assert not any(k in os.environ for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'])
shape = 'x'.join(next(a.split('=')[1] for a in sys.argv if a.startswith('OPS_FORCE_DECOMP_'+axis+'=')) for axis in 'XYZ')
assert shape in os.environ.get('MOCK_SHAPES','2x8x8,4x4x8,2x16x4').split(','), 'Unexpected/default decomposition'
rank = int(os.environ['SLURM_PROCID'])
phase = 1 if Path.cwd().name == 'autotune_on' else 2
mode = 1 if phase == 1 and rank == 0 else (0 if phase == 1 else 2)
assert '-OPS_AUTOTUNE_MODE='+str(mode) in sys.argv
assert os.environ['OPS_AUTOTUNE_MODE'] == str(mode)
if phase == 1:
 assert 'OPS_BLOCKSIZE_CSV' not in os.environ, 'Training loaded a pre-existing catalog'
else:
 assert '/decomp'+shape+'/blocks.csv' in os.environ['OPS_BLOCKSIZE_CSV']
 with open(os.environ['OPS_BLOCKSIZE_CSV']) as f: blocks = list(csv.DictReader(f))
 assert len(blocks) == 25
 expected = {'2x8x8':(64,2,8),'4x4x8':(128,1,4),'2x16x4':(64,1,4),'2x8x2':(64,2,8),'1x8x4':(128,1,4),'6x4x2':(64,1,4),'2x16x8':(64,2,8),'8x16x2':(128,1,4),'2x16x10':(64,2,8),'2x10x16':(128,1,4),'2x8x20':(64,1,4)}[shape]
 for r in blocks:
  k = int(r['kernel_id'])
  assert tuple(int(r[a]) for a in ['bx','by','bz']) == ((1,128,2) if k == 2 else expected), 'Wrong shape-specific blocks'
 print('Loaded %d precomputed block sizes from %s' % (24 if os.environ.get('MOCK_BAD_LOAD') else 25, os.environ['OPS_BLOCKSIZE_CSV']))
logging = rank == 0
if logging:
 assert os.environ['OPS_DISABLE_CSV_LOGS'] == '0'
 logs = Path(os.environ['OPS_LOGS_DIR'])/'autotune_on'; logs.mkdir(parents=True)
 candidates = {'2x8x8':[(64,2,8,3.0),(32,4,4,2.0)],
               '4x4x8':[(128,1,4,1.0),(64,2,4,2.0)],
               '2x16x4':[(64,1,4,2.0),(32,2,4,2.0)],
               '2x8x2':[(64,2,8,3.0),(32,4,4,2.0)],
               '1x8x4':[(128,1,4,1.0),(64,2,4,2.0)],
               '6x4x2':[(64,1,4,2.0),(32,2,4,2.0)],
               '2x16x8':[(64,2,8,3.0),(32,4,4,2.0)],
               '8x16x2':[(128,1,4,1.0),(64,2,4,2.0)],
               '2x16x10':[(64,2,8,3.0),(32,4,4,2.0)],
               '2x10x16':[(128,1,4,1.0),(64,2,4,2.0)],
               '2x8x20':[(64,1,4,2.0),(32,2,4,2.0)]}[shape]
 b = candidates[rank] if phase == 1 else (32,4,1,1.0)
 with (logs/'ops_blocksize_tuning_logmod.csv').open('w') as f:
  w=csv.writer(f); w.writerow(['kernel_id','grid_x','grid_y','grid_z','bx','by','bz','stencil_sig','max_threads'])
  local=[grid//int(d) for d in shape.split('x')]
  for k in range(1,27):
   g=[1,local[1]+2,local[2]+2] if k == 2 else local
   w.writerow([k]+g+list(b[:3])+['stencil_'+str(k),1024])
 if phase == 1 and not (os.environ.get('MOCK_MISSING_SAMPLE') and rank == 0):
  with (logs/'ops_blocksize_best_logmod.csv').open('w') as f:
   w=csv.writer(f); w.writerow(['kernel_id','bx','by','bz','best_time'])
   w.writerow([1,32,4,1,1e300])
   for k in range(2,27):
    # Export the rank-0 best block without any cross-rank consensus.
    w.writerow([k]+([1,128,2,1.0] if k == 2 else list(b)))
   w.writerow([3,32,4,1,99.0])  # Worse duplicate observation must not win.
   w.writerow(['Autotune -- 1','','','',''])
else:
 assert os.environ['OPS_DISABLE_CSV_LOGS'] == '1'
print('GPU Direct enabled')
print('Total Wall time: '+str(300 if phase == 1 else 150))
print('Total kernel time: '+str(250 if phase == 1 else 130))
print('Total halo exchange time: 20.0')
''')
        self.executable(self.bin/'ldd', '#!/bin/sh\necho "libmpi_gtl_hsa => /mock/libmpi_gtl_hsa.so"\n')
        self.executable(self.bin/'srun', '''#!/usr/bin/env python3
import csv, io, os, subprocess, sys
from pathlib import Path
nodes=int(os.environ.get('MOCK_NODES','16')); nranks=nodes*8
assert '--nodes='+str(nodes) in sys.argv and '--ntasks='+str(nranks) in sys.argv
assert '--ntasks-per-node=8' in sys.argv and '--cpus-per-task=7' in sys.argv
assert '--cpu-bind=map_cpu:49,57,17,25,1,9,33,41' in sys.argv
start=sys.argv.index('-c')-1
path=Path(sys.argv[-1]); run=path.parent
c=dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
with open(os.environ['MOCK_ORDER'],'a') as f:
 f.write(c['PRECOMPUTED_MPI']+','+c['OPS_AUTOTUNE_MODE']+','+c['RUN_ITERATION']+'\\n')
rcs=[]
for rank in [0,1,7]:
 env=dict(os.environ,SLURM_PROCID=str(rank),SLURM_LOCALID=str(rank%8))
 rcs.append(subprocess.run(sys.argv[start:],env=env).returncode)
if any(rcs): sys.exit(1)
line=next(s.split('\\t',1)[1] for s in (run/'ranks/rank_0.log').read_text().splitlines() if s.startswith('OPS_RANK_RESULT\\t'))
status=next(csv.reader([line]))
for rank in range(nranks):
 if rank in [0,1,7]: continue
 s=list(status); s[0]=str(rank); s[1]=s[2]=str(rank%8); s[5]=str(float(s[5])+2)
 s[8]='0' if c['OPS_AUTOTUNE_MODE']=='1' else '2'
 if os.environ.get('MOCK_BAD_RANK') and rank==nranks-1 and c['OPS_AUTOTUNE_MODE']=='2': s[6]='FAILED'
 buf=io.StringIO(); csv.writer(buf,lineterminator='').writerow(s)
 (run/('ranks/rank_%d.log'%rank)).write_text('OPS_RANK_RESULT\\t'+buf.getvalue()+'\\n')
''')
        oldlogs=self.campaign/'OPS_LOGS_20261022_083420_n016'
        self.ref_paths={}
        for mode,variant,wall,kernel,halo in [(0,'autotune_off',229.995363,198.507,31.0007),
                                            (2,'autotune_precomputed',170.818726,144.144,26.1776)]:
            p=oldlogs/'tgvstorenone_hip_2016x2016x2016_cfg1_comparison_20261022_083420_n016/mode_1/decomp8x4x4/iter_1'/variant/'config.txt'
            p.parent.mkdir(parents=True)
            c=dict(APP_NAME='tgvstorenone',GRID_X=2016,GRID_Y=2016,GRID_Z=2016,NITER=500,NUM_NODES=16,NUM_MPI_RANKS=128,
                   DECOMP_PX=8,DECOMP_PY=4,DECOMP_PZ=4,OPS_AUTOTUNE_MODE=mode,GPU_DIRECT='YES',RUN_STATUS='OK',RUN_EXIT_CODE=0,
                   EXECUTABLE_SHA256=hashlib.sha256(self.exe.read_bytes()).hexdigest(),APP_WALL_TIME=wall,
                   KERNEL_TOTAL_TIME=kernel,HALO_TOTAL_TIME=halo)
            p.write_text(''.join(str(k)+'='+str(v)+'\n' for k,v in c.items())); self.ref_paths[mode]=p
        self.profile=self.root/'profile.sh'; self.profile.write_text('export PATH="'+str(self.bin)+':$PATH"\n')
        self.node_tag='n016'
        self.output=self.root/'TGV_YZ_N016_20261101_120000_000001'
        self.env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        self.env.update(SLURM_JOB_ID='123',SLURM_JOB_NUM_NODES='16',MOCK_ORDER=str(self.root/'order.csv'),
                        OPS_BLOCKSIZE_CSV='/must/not/be/inherited.csv',HIP_VISIBLE_DEVICES='99')

    def executable(self,path,text):
        path.write_text(text); path.chmod(0o755)

    def run_script(self,*extra,auto_output=False):
        cmd=['bash',str(SCRIPT),'--ops-root',str(HERE.parents[1]),'--campaign',str(self.campaign),'--env-file',str(self.profile)]
        if not auto_output: cmd+=['--output',str(self.output)]
        return subprocess.run(cmd+list(extra),env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)

    def logs(self):
        return next(self.output.glob('OPS_LOGS_*_'+self.node_tag))

    def scientific(self,shape,variant):
        return next(self.logs().glob('tgv*/mode_1/decomp'+shape+'/iter_1/'+variant))

    def assert_success(self,r):
        if r.returncode:
            diagnostics='\n'.join(str(p)+'\n'+p.read_text()[-4000:] for p in self.output.rglob('run.log'))
            self.fail(r.stderr+'\n'+diagnostics)

    def test_dry_run_has_three_shapes_and_six_main_executions(self):
        r=self.run_script('--dry-run','--account','project_yz_test',auto_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        p=dict(s.split('=',1) for s in r.stdout.splitlines() if '=' in s)
        self.assertEqual(p['APP_ARGS'],'2016 2016 2016 500')
        self.assertEqual(p['DECOMPOSITIONS'],','.join(SHAPES))
        self.assertEqual(p['AUTOTUNE_MODES'],'1 2')
        self.assertEqual(p['AUTOTUNE_SAMPLE_RANKS'],'0')
        self.assertEqual(p['MAIN_EXECUTIONS'],'6')
        self.assertEqual(p['REFERENCE_RERUN'],'NO')
        self.assertTrue(p['OUTPUT_DIRECTORY'].startswith('/scratch/project_yz_test/ops-decomposition/campaigns/TGV_YZ_N016_'))
        self.assertFalse(Path(p['OUTPUT_DIRECTORY']).exists())

    def test_each_shape_trains_own_catalog_and_only_agreed_files_exist(self):
        self.assert_success(self.run_script())
        logs=self.logs()
        self.assertEqual(logs.name,'OPS_LOGS_20261101_120000_000001_n016')
        self.assertEqual((self.root/'order.csv').read_text(),''.join(s+','+m+',1\n' for s in SHAPES for m in ['1','2']))
        self.assertEqual(len(table(logs/'summary.csv')),6)
        self.assertEqual(len(table(logs/'comparisons.csv')),9)
        self.assertEqual(len(list(self.output.iterdir())),1)
        self.assertEqual({p.name for p in logs.iterdir()}, {'precomputed_blocks','summary.csv','comparisons.csv',
                         'tgvstorenone_hip_2016x2016x2016_cfg1_comparison_20261101_120000_000001_n016'})
        hashes=[]
        for shape in SHAPES:
            cat=logs/'precomputed_blocks'/('decomp'+shape)/'blocks.csv'
            hashes.append(hashlib.sha256(cat.read_bytes()).hexdigest())
            self.assertEqual({p.name for p in cat.parent.iterdir()},{'blocks.csv'})
            tuned=self.scientific(shape,'autotune_on'); replay=self.scientific(shape,'autotune_precomputed')
            t,c=config(tuned/'config.txt'),config(replay/'config.txt')
            self.assertEqual(c['PRECOMPUTED_SOURCE_CONFIG'],str(tuned/'config.txt'))
            self.assertEqual(c['PRECOMPUTED_SET_ID'],hashes[-1])
            self.assertEqual(c['BLOCKS_SOURCE_MPI'],shape)
            self.assertEqual(float(c['WALL_TIME_AUTOTUNE']),302)
            self.assertEqual(float(c['WALL_TIME_PRECOMPUTED']),152)
            self.assertAlmostEqual(float(c['SPEEDUP_DEFAULT_MPI_PRECOMPUTED']),170.818726/152)
            self.assertEqual(t['WALL_TIME_PRECOMPUTED'],'NA')
            for run in (tuned,replay):
                fields=config(run/'config.txt')
                for k,v in dict(APP_ARGS='2016 2016 2016 500',RUN_STATUS='OK',COMMIT_STATUS='COMMITTED',
                                GRID_X='2016',GRID_Y='2016',GRID_Z='2016',REFERENCE_RERUN='NO',GPU_DIRECT='YES').items():
                    self.assertEqual(fields[k],v)
                allowed={'config.txt','run.log','mpi_rank_metrics.csv','ranks','rank_0'}
                self.assertEqual({p.name for p in run.iterdir()},allowed)
                self.assertEqual({p.suffix for p in (run/'ranks').iterdir()},{'.log'})
                self.assertEqual(len(table(run/'mpi_rank_metrics.csv')),128)
            modes=table(tuned/'mpi_rank_metrics.csv')
            self.assertEqual([r['rank_id'] for r in modes if r['autotune_mode']=='1'],['0'])
        self.assertEqual(len(set(hashes)),3,'Each shape must retain its independently generated catalog')
        self.assertEqual(list(self.output.rglob('*.json')),[])
        again=self.run_script(); self.assertNotEqual(again.returncode,0); self.assertIn('Output already exists',again.stderr)

    def test_missing_sample_prevents_export_and_replay(self):
        self.env['MOCK_MISSING_SAMPLE']='1'
        r=self.run_script('--decompositions','2x8x8')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual((self.root/'order.csv').read_text(),'2x8x8,1,1\n')
        self.assertEqual(config(self.scientific('2x8x8','autotune_on')/'config.txt')['RUN_STATUS'],'FAILED')
        self.assertFalse((self.logs()/'precomputed_blocks').exists())
        self.assertFalse((self.logs()/'summary.csv').exists())

    def test_failed_replay_is_not_in_comparisons(self):
        self.env['MOCK_BAD_RANK']='1'
        r=self.run_script('--decompositions','2x8x8')
        self.assertNotEqual(r.returncode,0)
        c=config(self.scientific('2x8x8','autotune_precomputed')/'config.txt')
        self.assertEqual(c['RUN_STATUS'],'FAILED'); self.assertEqual(c['COMMIT_STATUS'],'UNCOMMITTED')
        self.assertEqual(len(table(self.logs()/'summary.csv')),1)
        self.assertFalse((self.logs()/'comparisons.csv').exists())

    def test_catalog_load_failure_stops_job(self):
        self.env['MOCK_BAD_LOAD']='1'
        r=self.run_script('--decompositions','2x8x8')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(config(self.scientific('2x8x8','autotune_precomputed')/'config.txt')['VALIDATION_STATUS'],'FAIL')
        self.assertFalse((self.logs()/'comparisons.csv').exists())

    def test_reuse_requires_matching_tuning_and_rejects_swapped_catalog(self):
        self.assert_success(self.run_script())
        source=self.logs(); self.output=self.root/'reused'
        (self.root/'order.csv').write_text('')
        self.assert_success(self.run_script('--precomputed-source',str(source)))
        self.assertEqual((self.root/'order.csv').read_text(),''.join(s+',2,1\n' for s in SHAPES))
        self.assertEqual(len(table(self.logs()/'summary.csv')),3)
        for shape in SHAPES:
            self.assertFalse(list(self.logs().glob('tgv*/mode_1/decomp'+shape+'/iter_1/autotune_on')))
            c=config(self.scientific(shape,'autotune_precomputed')/'config.txt')
            self.assertTrue(c['PRECOMPUTED_SOURCE_CONFIG'].startswith(str(source)))
        bad=source/'precomputed_blocks/decomp4x4x8/blocks.csv'
        bad.write_bytes((source/'precomputed_blocks/decomp2x8x8/blocks.csv').read_bytes())
        self.output=self.root/'rejected'
        r=self.run_script('--precomputed-source',str(source),'--dry-run')
        self.assertNotEqual(r.returncode,0); self.assertIn('differs from',r.stderr)
        self.assertFalse(self.output.exists())

    def test_partial_catalog_source_trains_missing_configurations(self):
        self.assert_success(self.run_script('--decompositions','2x8x8'))
        source=self.logs(); self.output=self.root/'partial'
        (self.root/'order.csv').write_text('')
        self.assert_success(self.run_script('--precomputed-source',str(source)))
        self.assertEqual((self.root/'order.csv').read_text(),'2x8x8,2,1\n4x4x8,1,1\n4x4x8,2,1\n2x16x4,1,1\n2x16x4,2,1\n')

    def test_reject_default_bad_shape_wrong_allocation(self):
        for shape in ['8x4x4','2x4x4','2x8x8,2x8x8']:
            self.assertNotEqual(self.run_script('--decompositions',shape,'--dry-run').returncode,0)
        self.env['SLURM_JOB_NUM_NODES']='8'
        r=self.run_script(); self.assertNotEqual(r.returncode,0); self.assertIn('exactly 16 nodes',r.stderr)
        self.assertFalse(self.output.exists())

    def test_submit_pins_all_shapes_and_preflights_reference(self):
        self.env.pop('SLURM_JOB_ID'); self.env.pop('SLURM_JOB_NUM_NODES')
        self.env['PATH']=str(self.bin)+os.pathsep+self.env['PATH']
        capture=self.root/'submitted.txt'; self.env['MOCK_SUBMISSION']=str(capture)
        self.executable(self.bin/'sbatch','''#!/usr/bin/env python3
import os,sys
from pathlib import Path
Path(os.environ['MOCK_SUBMISSION']).write_text('\\n'.join(sys.argv[1:]))
print('Submitted batch job 456')
''')
        r=self.run_script('--submit'); self.assertEqual(r.returncode,0,r.stderr)
        sent=capture.read_text().splitlines()
        self.assertEqual(sent[sent.index('--decompositions')+1],','.join(SHAPES))
        self.assertNotIn('--blocks',sent)
        self.assertFalse(self.output.exists()); capture.unlink()
        self.ref_paths[2].write_text(self.ref_paths[2].read_text().replace('GRID_X=2016','GRID_X=300'))
        r=self.run_script('--submit'); self.assertNotEqual(r.returncode,0); self.assertFalse(capture.exists())

    def install_small_profiles(self, large=False, forty=False):
        stamp='20261026_095042' if forty else ('20261025_202825' if large else '20261026_101612')
        self.campaign=self.root/('TGV_TOP3_'+stamp)
        target=self.campaign/'build/tgvstorenone_mpi_hip'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target != self.exe:
            shutil.copy2(self.exe,target)
        self.exe=target
        self.small_top3={}
        profiles = [
            (4,1280,'4x4x2','1x32x1',207.746603,161.567888,166.525803),
            (6,1452,'4x4x3','4x3x4',249.07151,169.41902,169.240052)]
        if large:
            profiles=[(32,2544,'8x8x4','4x8x8',226.890253,168.163679,172.918445)]
        if forty:
            profiles=[(40,2720,'8x8x5','8x8x5',224.52848,176.107972,176.107972)]
        for n,grid,baseline,top3,a,c,t in profiles:
            tag='n%03d'%n
            root=self.campaign/('OPS_LOGS_'+stamp+'_'+tag)
            label='tgvstorenone_hip_{0}x{0}x{0}_cfg1_comparison_{2}_{1}'.format(grid,tag,stamp)
            for role,shape,variant,mode,wall in [('A',baseline,'autotune_off',0,a),
                ('C',baseline,'autotune_precomputed',2,c),('T',top3,'autotune_precomputed',2,t)]:
                path=root/label/'mode_1'/('decomp'+shape)/'iter_1'/variant/'config.txt'
                path.parent.mkdir(parents=True,exist_ok=True)
                dims=shape.split('x')
                fields=dict(APP_NAME='tgvstorenone',GRID_X=grid,GRID_Y=grid,GRID_Z=grid,NITER=500,
                    NUM_NODES=n,NUM_MPI_RANKS=n*8,DECOMP_PX=dims[0],DECOMP_PY=dims[1],DECOMP_PZ=dims[2],
                    OPS_AUTOTUNE_MODE=mode,GPU_DIRECT='YES',RUN_STATUS='OK',RUN_EXIT_CODE=0,
                    EXECUTABLE_SHA256=hashlib.sha256(self.exe.read_bytes()).hexdigest(),APP_WALL_TIME=wall,
                    KERNEL_TOTAL_TIME=130.0,HALO_TOTAL_TIME=20.0)
                path.write_text(''.join(str(k)+'='+str(v)+'\n' for k,v in fields.items()))
                if role=='T': self.small_top3[n]=path

    def test_four_and_six_node_workflows_keep_grids_and_compare_top3(self):
        self.install_small_profiles()
        for n,grid,shapes,top_time in [(4,1280,['2x8x2','1x8x4'],166.525803),(6,1452,['6x4x2'],169.240052)]:
            with self.subTest(nodes=n):
                self.node_tag='n%03d'%n
                self.output=self.root/('TGV_YZ_'+self.node_tag.upper()+'_20261101_120000_000001')
                self.env.update(SLURM_JOB_NUM_NODES=str(n),MOCK_NODES=str(n),MOCK_GRID=str(grid),MOCK_SHAPES=','.join(shapes))
                (self.root/'order.csv').write_text('')
                self.assert_success(self.run_script('--nodes',str(n)))
                self.assertEqual((self.root/'order.csv').read_text(),''.join(s+','+m+',1\n' for s in shapes for m in ['1','2']))
                summary=table(self.logs()/'summary.csv')
                self.assertEqual(len(summary),2*len(shapes))
                self.assertTrue(all(r['nodes']==str(n) and r['global_grid']=='x'.join([str(grid)]*3) for r in summary))
                for shape in shapes:
                    run=self.scientific(shape,'autotune_precomputed')
                    c=config(run/'config.txt')
                    self.assertEqual(c['APP_ARGS'],' '.join([str(grid)]*3+['500']))
                    self.assertEqual(c['NUM_MPI_RANKS'],str(n*8))
                    self.assertEqual(c['TOP3_REFERENCE_CONFIG'],str(self.small_top3[n]))
                    self.assertAlmostEqual(float(c['SPEEDUP_TOP3_PRECOMPUTED']),top_time/152)
                    self.assertEqual(len(table(run/'mpi_rank_metrics.csv')),n*8)
                comp=table(self.logs()/'comparisons.csv')
                wall=[r for r in comp if r['metric']=='app_wall_time_s']
                self.assertTrue(all(abs(float(r['T_over_D'])-top_time/152)<1e-10 for r in wall))
                self.assertEqual(list(self.output.rglob('*.json')),[])

    def test_pair_launcher_validates_both_before_submitting_two_jobs(self):
        self.install_small_profiles()
        self.env.pop('SLURM_JOB_ID'); self.env.pop('SLURM_JOB_NUM_NODES')
        self.env['PATH']=str(self.bin)+os.pathsep+self.env['PATH']
        capture=self.root/'pair_submissions.txt'; self.env['MOCK_SUBMISSION']=str(capture)
        self.executable(self.bin/'sbatch','''#!/usr/bin/env python3
import os,sys
with open(os.environ['MOCK_SUBMISSION'],'a') as f: f.write('\\n'.join(sys.argv[1:])+'\\n---\\n')
print('Submitted batch job 456')
''')
        command=['bash',str(HERE/'compare_yz_n004_n006.sh'),'--campaign',str(self.campaign),
                 '--env-file',str(self.profile),'--output-root',str(self.root/'pair_results'),'--submit']
        saved=self.small_top3[6].read_text(); self.small_top3[6].unlink()
        r=subprocess.run(command,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assertNotEqual(r.returncode,0); self.assertFalse(capture.exists())
        self.small_top3[6].write_text(saved)
        r=subprocess.run(command,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assertEqual(r.returncode,0,r.stderr)
        submissions=[p.splitlines() for p in capture.read_text().split('---\n') if p.strip()]
        self.assertEqual(len(submissions),2)
        for sent,n,shapes in zip(submissions,[4,6],['2x8x2,1x8x4','6x4x2']):
            self.assertIn('--nodes='+str(n),sent)
            self.assertEqual(sent[sent.index('--nodes')+1],str(n))
            self.assertEqual(sent[sent.index('--decompositions')+1],shapes)
            self.assertIn('TGV_YZ_N%03d_'%n,sent[sent.index('--output')+1])

    def test_32_nodes_tunes_once_and_compares_top2_and_top3(self):
        self.install_small_profiles(large=True)
        self.node_tag='n032'; self.output=self.root/'TGV_YZ_N032_20261101_120000_000001'
        self.env.update(SLURM_JOB_NUM_NODES='32',MOCK_NODES='32',MOCK_GRID='2544',MOCK_SHAPES='2x16x8,8x16x2')
        command=['bash',str(HERE/'compare_yz_n032.sh'),'--campaign',str(self.campaign),
                 '--env-file',str(self.profile),'--output',str(self.output)]
        r=subprocess.run(command,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assert_success(r)
        self.assertEqual((self.root/'order.csv').read_text(),'2x16x8,1,1\n2x16x8,2,1\n8x16x2,1,1\n8x16x2,2,1\n')
        for shape in ['2x16x8','8x16x2']:
            run=self.scientific(shape,'autotune_precomputed'); c=config(run/'config.txt')
            self.assertEqual(c['APP_ARGS'],'2544 2544 2544 500')
            self.assertEqual(c['NUM_NODES'],'32'); self.assertEqual(c['NUM_MPI_RANKS'],'256')
            self.assertEqual(c['TOP2_MPI'],'8x8x4'); self.assertEqual(c['TOP3_MPI'],'4x8x8')
            self.assertAlmostEqual(float(c['SPEEDUP_TOP2_PRECOMPUTED']),168.163679/152)
            self.assertAlmostEqual(float(c['SPEEDUP_TOP3_PRECOMPUTED']),172.918445/152)
            self.assertEqual(c['SPEEDUP_TOP2_PRECOMPUTED'],c['SPEEDUP_DEFAULT_MPI_PRECOMPUTED'])
            self.assertEqual(len(table(run/'mpi_rank_metrics.csv')),256)
            tuning=table(self.scientific(shape,'autotune_on')/'mpi_rank_metrics.csv')
            self.assertEqual([v['rank_id'] for v in tuning if v['autotune_mode']=='1'],['0'])
        self.assertEqual(len(table(self.logs()/'summary.csv')),4)
        self.assertEqual(len(table(self.logs()/'comparisons.csv')),6)
        self.assertEqual(list(self.output.rglob('*.json')),[])

    def test_32_node_entrypoint_submission_requests_32_nodes(self):
        self.install_small_profiles(large=True)
        self.env.pop('SLURM_JOB_ID'); self.env.pop('SLURM_JOB_NUM_NODES')
        self.env['PATH']=str(self.bin)+os.pathsep+self.env['PATH']
        capture=self.root/'submit32.txt'; self.env['MOCK_SUBMISSION']=str(capture)
        self.executable(self.bin/'sbatch','''#!/usr/bin/env python3
import os,sys
from pathlib import Path
Path(os.environ['MOCK_SUBMISSION']).write_text('\\n'.join(sys.argv[1:]))
''')
        cmd=['bash',str(HERE/'compare_yz_n032.sh'),'--campaign',str(self.campaign),
             '--env-file',str(self.profile),'--output',str(self.output),'--submit']
        r=subprocess.run(cmd,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assertEqual(r.returncode,0,r.stderr)
        sent=capture.read_text().splitlines()
        self.assertIn('--nodes=32',sent); self.assertIn('--job-name=tgv-yz-n032',sent)
        self.assertEqual(sent[sent.index('--nodes')+1],'32')
        self.assertEqual(sent[sent.index('--decompositions')+1],'2x16x8,8x16x2')
        self.assertFalse(self.output.exists())

    def test_40_node_entrypoint_submission_requests_40_nodes(self):
        self.install_small_profiles(forty=True)
        self.env.pop('SLURM_JOB_ID'); self.env.pop('SLURM_JOB_NUM_NODES')
        self.env['PATH']=str(self.bin)+os.pathsep+self.env['PATH']
        capture=self.root/'submit40.txt'; self.env['MOCK_SUBMISSION']=str(capture)
        self.executable(self.bin/'sbatch','''#!/usr/bin/env python3
import os,sys
from pathlib import Path
Path(os.environ['MOCK_SUBMISSION']).write_text('\\n'.join(sys.argv[1:]))
''')
        cmd=['bash',str(HERE/'compare_yz_n040.sh'),'--campaign',str(self.campaign),
             '--env-file',str(self.profile),'--output',str(self.output),'--submit']
        r=subprocess.run(cmd,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assertEqual(r.returncode,0,r.stderr)
        sent=capture.read_text().splitlines()
        self.assertIn('--nodes=40',sent); self.assertIn('--job-name=tgv-yz-n040',sent)
        self.assertEqual(sent[sent.index('--nodes')+1],'40')
        self.assertEqual(sent[sent.index('--decompositions')+1],'2x16x10,2x10x16,2x8x20')
        self.assertFalse(self.output.exists())



    def test_40_nodes_three_own_catalogs_rank_zero_only(self):
        self.install_small_profiles(forty=True)
        self.node_tag='n040'; self.output=self.root/'TGV_YZ_N040_20261101_120000_000001'
        shapes=['2x16x10','2x10x16','2x8x20']
        self.env.update(SLURM_JOB_NUM_NODES='40',MOCK_NODES='40',MOCK_GRID='2720',MOCK_SHAPES=','.join(shapes))
        command=['bash',str(HERE/'compare_yz_n040.sh'),'--campaign',str(self.campaign),
                 '--env-file',str(self.profile),'--output',str(self.output)]
        r=subprocess.run(command,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assert_success(r)
        self.assertEqual((self.root/'order.csv').read_text(),''.join(s+','+m+',1\n' for s in shapes for m in ['1','2']))
        for shape in shapes:
            run=self.scientific(shape,'autotune_precomputed'); c=config(run/'config.txt')
            self.assertEqual(c['APP_ARGS'],'2720 2720 2720 500')
            self.assertEqual(c['NUM_NODES'],'40'); self.assertEqual(c['NUM_MPI_RANKS'],'320')
            self.assertEqual(c['TOP3_MPI'],'8x8x5')
            self.assertAlmostEqual(float(c['SPEEDUP_TOP3_PRECOMPUTED']),176.107972/152)
            self.assertEqual(len(table(run/'mpi_rank_metrics.csv')),320)
            tuning=table(self.scientific(shape,'autotune_on')/'mpi_rank_metrics.csv')
            self.assertEqual([v['rank_id'] for v in tuning if v['autotune_mode']=='1'],['0'])
        self.assertEqual(len(table(self.logs()/'summary.csv')),6)
        self.assertEqual(len(table(self.logs()/'comparisons.csv')),9)
        self.assertEqual(list(self.output.rglob('*.json')),[])

if __name__=='__main__': unittest.main()
