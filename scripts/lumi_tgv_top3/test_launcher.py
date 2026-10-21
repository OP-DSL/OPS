"""CPU-only tests: no real Slurm submission, modules, compiler or GPU execution."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import launcher as L
import runtime as R


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tgv_top3_test_')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_clean_snapshot_contains_tgv_build_inputs(self):
        ops = Path(__file__).resolve().parents[2]
        L.snapshot_ops(ops, self.root)
        snapshot = self.root / 'build/OPS'
        recorded = L.read_json(self.root / 'source_snapshot.json')['source_files_sha256']
        for name in ('Makefile', 'opensbli.cpp', 'opensbliblock00_kernels.h', 'defdec_data_set.h'):
            rel = 'apps/c/TGV_StoreNone/' + name
            self.assertEqual(L.sha(snapshot / rel), recorded[rel])
        result = subprocess.run(
            ['make', '-n', '-C', str(snapshot / 'apps/c/TGV_StoreNone'),
             'OPS_INSTALL_PATH=' + str(snapshot / 'ops'), 'OPS_COMPILER=hip',
             'GPU_VENDOR=AMD', 'HIP_INSTALL_PATH=/unused/rocm', '.generated'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('--file_paths opensbli.cpp', result.stdout)

    def test_untracked_makefile_rejected_before_snapshot(self):
        ops = Path(__file__).resolve().parents[2]
        listing = subprocess.check_output(['git', '-C', str(ops), 'ls-files', '-z'])
        listing = listing.replace(b'apps/c/TGV_StoreNone/Makefile\0', b'')
        with patch.object(L.subprocess, 'check_output', return_value=listing):
            with self.assertRaisesRegex(ValueError, 'missing or not tracked.*Makefile'):
                L.snapshot_ops(ops, self.root)
        self.assertFalse((self.root / 'build/OPS').exists())

    def test_rocm_host_compilers_and_real_make_plan(self):
        ops = Path(__file__).resolve().parents[2]
        L.snapshot_ops(ops, self.root)
        snapshot = self.root / 'build/OPS'
        for layout in ('llvm/bin', 'lib/llvm/bin'):
            rocm = self.root / layout.replace('/', '_')
            compiler_dir = rocm / layout
            compiler_dir.mkdir(parents=True)
            for name in ('clang', 'clang++'):
                path = compiler_dir / name
                path.write_text('#!/bin/sh\nexit 99\n')
                path.chmod(0o755)
            with patch.dict(os.environ, {'HIP_CLANG_PATH': ''}):
                flags = L.rocm_compiler_flags('/rocm/hipcc', rocm)
            commands = dict(flag.split('=', 1) for flag in flags)
            self.assertEqual(commands['CXX'], str(compiler_dir / 'clang++'))
            self.assertEqual(commands['MPICXX'], commands['CXX'])
            result = subprocess.run(['make', '-n', '-C', str(snapshot / 'ops/c'),
                'OPS_COMPILER=hip', 'OPS_INSTALL_PATH=' + str(snapshot / 'ops'),
                'GPU_VENDOR=AMD', 'HIP_INSTALL_PATH=' + str(rocm),
                'MPI_INSTALL_PATH=/cray/mpi', 'I_MPI_INC=-I/cray/mpi/include',
                'CXXFLAGS=-O3 -fPIC -std=c++11', 'TARGETS=mpi_hip'] + flags + ['mpi_hip'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            for source in ('ops_lib_core.cpp', 'ops_lazy.cpp', 'ops_util.cpp', 'ops_mpi_core.cpp',
                           'ops_mpi_rt_support.cpp', 'ops_mpi_partition.cpp'):
                lines = [line for line in result.stdout.splitlines() if ' -c ' in line and '/' + source in line]
                self.assertEqual(len(lines), 1, result.stdout)
                self.assertTrue(lines[0].startswith(commands['CXX'] + ' '), lines[0])
            hip_lines = [line for line in result.stdout.splitlines() if ' -c ' in line and 'ops_mpi_rt_support_hip.cpp' in line]
            self.assertTrue(hip_lines[0].startswith('/rocm/hipcc '))
        with patch.dict(os.environ, {'HIP_CLANG_PATH': str(self.root / 'missing')}):
            with self.assertRaisesRegex(ValueError, 'Cannot locate ROCm'):
                L.rocm_compiler_flags('/rocm/hipcc', rocm)

    def test_libclang_bundled_library_parses_cpp_without_system_library(self):
        python = os.environ.get('OPS_TEST_TRANSLATOR_PYTHON', sys.executable)
        check = subprocess.run([python, '-c',
            'import clang.cindex, jinja2, fparser, pcpp, numpy, sympy, h5py'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check.returncode:
            self.skipTest('Translator dependencies unavailable; set OPS_TEST_TRANSLATOR_PYTHON')
        env = dict(os.environ)
        env.pop('LIBCLANG_PATH', None)
        out = io.StringIO()
        library = L.configure_libclang(python, env, out)
        self.assertEqual(env['LIBCLANG_PATH'], library)
        self.assertTrue(library)
        # An explicit configuration must be honored and validated, not silently
        # replaced by the bundled wheel when it is invalid.
        env['LIBCLANG_PATH'] = str(self.root / 'missing-libclang.so')
        with self.assertRaisesRegex(ValueError, 'libclang preflight failed'):
            L.configure_libclang(python, env, out)
        self.assertIn('missing-libclang.so', out.getvalue())

    def test_translator_requirements_have_one_clang_provider(self):
        requirements = (Path(__file__).resolve().parents[2] / 'ops_translator/requirements.txt').read_text()
        packages = [line.split('==')[0].strip().lower() for line in requirements.splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
        self.assertIn('libclang', packages)
        self.assertNotIn('clang', packages)

    def test_real_clang_wheel_collision_is_detected(self):
        wheel_dir = os.environ.get('OPS_TEST_CLANG_WHEELS')
        if not wheel_dir:
            self.skipTest('Set OPS_TEST_CLANG_WHEELS to downloaded clang/libclang 16.0.6 wheels')
        python = os.environ.get('OPS_TEST_TRANSLATOR_PYTHON', sys.executable)
        for label, packages in [('conflict', ['libclang', 'clang']), ('fixed', ['libclang'])]:
            dest = self.root / label
            for package in packages:
                wheel = next(Path(wheel_dir).glob(package + '-16.0.6-*.whl'))
                with zipfile.ZipFile(str(wheel)) as archive:
                    for entry in archive.namelist():
                        # Reproduce wheel installation into site-packages.
                        relative = entry.split('.data/platlib/', 1)[-1]
                        path = L.contained(dest, relative)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(archive.read(entry))
            env = dict(os.environ, PYTHONPATH=str(dest))
            env.pop('LIBCLANG_PATH', None)
            out = io.StringIO()
            if label == 'conflict':
                with self.assertRaisesRegex(ValueError, 'libclang preflight failed'):
                    L.configure_libclang(python, env, out)
                self.assertIn('Unknown template argument kind 280', out.getvalue())
            else:
                library = L.configure_libclang(python, env, out)
                self.assertEqual(library, str(dest / 'clang/native/libclang.so'))

    def package(self):
        package = self.root / 'source/TOP3_DATASET'
        package.mkdir(parents=True)
        (package / 'selections.csv').write_text('Nodes,Top rank\n6,1\n')
        m = dict(application='tgvstorenone', selection_sha256=L.sha(package/'selections.csv'),
                 files={'selections.csv':dict(sha256=L.sha(package/'selections.csv'))})
        L.atomic(package/'dataset_manifest.json',m)
        dest=self.root/'OPS/dataset-lumi';dest.mkdir(parents=True)
        archive=dest/'TOP3_DATASET.tar.gz'
        with tarfile.open(archive,'w:gz') as t:t.add(package,arcname='TOP3_DATASET')
        (dest/'TOP3_DATASET.tar.gz.sha256').write_text(L.sha(archive)+'  TOP3_DATASET.tar.gz\n')
        return dest.parent,package,archive

    def test_archive_extract_reuse_and_tamper(self):
        ops,source,archive=self.package()
        extracted,_=L.ensure_dataset(ops)
        self.assertEqual((extracted/'selections.csv').read_bytes(),(source/'selections.csv').read_bytes())
        L.ensure_dataset(ops)
        (extracted/'selections.csv').write_text('damaged')
        with self.assertRaisesRegex(ValueError,'Invalid dataset file'):L.ensure_dataset(ops)

    def test_archive_checksum(self):
        ops,_,archive=self.package();archive.write_bytes(archive.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):L.ensure_dataset(ops)

    def replace_package(self, source, archive, known):
        m = L.read_json(source/'dataset_manifest.json')
        m['supersedes_manifest_sha256'] = [known]
        (source/'selections.csv').write_text('Nodes,Top rank\n6,2\n')
        m['selection_sha256'] = L.sha(source/'selections.csv')
        m['files']['selections.csv']['sha256'] = m['selection_sha256']
        L.atomic(source/'dataset_manifest.json',m)
        with tarfile.open(archive,'w:gz') as t: t.add(source,arcname='TOP3_DATASET')
        (archive.parent/'TOP3_DATASET.tar.gz.sha256').write_text(L.sha(archive))

    def test_known_dataset_upgrade_preserves_previous(self):
        ops,source,archive=self.package()
        old,_=L.ensure_dataset(ops)
        old_bytes=(old/'selections.csv').read_bytes()
        self.replace_package(source,archive,L.sha(old/'dataset_manifest.json'))
        new,_=L.ensure_dataset(ops)
        backups=list(archive.parent.glob('TOP3_DATASET.previous-*/TOP3_DATASET'))
        self.assertEqual(len(backups),1)
        self.assertEqual((backups[0]/'selections.csv').read_bytes(),old_bytes)
        self.assertEqual((new/'selections.csv').read_bytes(),(source/'selections.csv').read_bytes())
        L.ensure_dataset(ops)

    def test_unknown_dataset_upgrade_is_rejected(self):
        ops,source,archive=self.package()
        old,_=L.ensure_dataset(ops)
        old_bytes=(old/'selections.csv').read_bytes()
        self.replace_package(source,archive,'0'*64)
        with self.assertRaisesRegex(ValueError,'recognized previous'): L.ensure_dataset(ops)
        self.assertEqual((old/'selections.csv').read_bytes(),old_bytes)

    def test_failed_upgrade_does_not_replace_old(self):
        ops,source,archive=self.package()
        old,_=L.ensure_dataset(ops)
        old_bytes=(old/'selections.csv').read_bytes()
        self.replace_package(source,archive,L.sha(old/'dataset_manifest.json'))
        (source/'selections.csv').write_text('corrupt new data')
        with tarfile.open(archive,'w:gz') as t: t.add(source,arcname='TOP3_DATASET')
        (archive.parent/'TOP3_DATASET.tar.gz.sha256').write_text(L.sha(archive))
        with self.assertRaisesRegex(ValueError,'Invalid dataset file'): L.ensure_dataset(ops)
        self.assertEqual((old/'selections.csv').read_bytes(),old_bytes)

    def test_archive_rejects_traversal_and_links(self):
        for filename,link in [('../escape',False),('TOP3_DATASET/link',True)]:
            base=self.root/('link' if link else 'traversal')/'dataset-lumi';base.mkdir(parents=True)
            archive=base/'TOP3_DATASET.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                entry=tarfile.TarInfo(filename)
                if link:entry.type=tarfile.SYMTYPE;entry.linkname='/etc/passwd'
                t.addfile(entry,io.BytesIO())
            (base/'TOP3_DATASET.tar.gz.sha256').write_text(L.sha(archive))
            with self.assertRaises(ValueError):L.ensure_dataset(base.parent)

    def test_escaping_manifest_path(self):
        with self.assertRaises(ValueError):L.contained(self.root,'../escape')

    def catalog(self):
        root=self.root/'catalog';root.mkdir()
        (root/'blocks.csv').write_text('kernel_id,bx,by,bz\n2,128,1,2\n')
        m=dict(schema_version=2,context=dict(ACTIVE_NUM_RANKS='48'),files={'blocks.csv':L.sha(root/'blocks.csv')})
        m['set_id']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
        L.atomic(root/'manifest.json',m)
        return root

    def test_missing_catalog_preserves_gap(self):
        root=self.catalog();(root/'blocks.csv').unlink()
        _,missing=L.catalog_check(root)
        self.assertEqual(missing,['blocks.csv'])

    def test_changed_blocks_rejected(self):
        root=self.catalog();(root/'blocks.csv').write_text('kernel_id,bx,by,bz\n2,64,1,2\n')
        with self.assertRaisesRegex(ValueError,'Changed block'):L.catalog_check(root)

    def test_probe_exact_resources_no_real_submission(self):
        args=argparse.Namespace(nodes=[6,40],partition='standard-g',account='project_test',time='02:30:00')
        seen=[]
        def fake(command,timeout=15):
            seen.append(command)
            return dict(status='OK',stdout='eligible start 2026-10-19T20:00:00',stderr='')
        with patch.object(L,'query',side_effect=fake):
            report=L.analyze(args,[dict(nodes=n,status='COMPLETE') for n in args.nodes])
        probes=[x for x in seen if x[0]=='sbatch']
        self.assertEqual(len(probes),2)
        for cmd in probes:
            self.assertIn('--test-only',cmd);self.assertIn('--time=02:30:00',cmd)
            self.assertIn('--ntasks-per-node=8',cmd);self.assertIn('--gpus-per-node=8',cmd)
        self.assertEqual(L.select_partitions(report,args),{'6':'standard-g','40':'standard-g'})

    def test_unknown_probe_not_eligible(self):
        args=argparse.Namespace(nodes=[6],partition='standard-g',account='',time='01:00:00')
        with patch.object(L,'query',return_value=dict(status='NOT_QUERIED')):
            report=L.analyze(args,[dict(nodes=6,status='COMPLETE')])
        with self.assertRaisesRegex(ValueError,'No successful'):L.select_partitions(report,args)

    def test_query_on_running_python(self):
        result = L.query([sys.executable, '-c', 'print("probe passed")'])
        self.assertEqual(result['status'], 'OK')
        self.assertEqual(result['stdout'].strip(), 'probe passed')

    def test_translator_selects_newer_python(self):
        def lookup(name):
            return '/available/python3.8' if name == 'python3.8' else '/available/old-python'
        def version(command, **kwargs):
            return dict(status='OK' if command[0].endswith('python3.8') else 'ERROR')
        with patch.object(L.shutil, 'which', side_effect=lookup), patch.object(L, 'query', side_effect=version):
            self.assertEqual(L.translator_interpreter(), '/available/python3.8')
            with self.assertRaisesRegex(ValueError, 'OPS translation requires Python >= 3.8'):
                L.translator_interpreter('/explicit/old-python')

    def test_translator_preserves_virtualenv_path(self):
        with patch.object(L.shutil, 'which', return_value='/venv/bin/python3'), patch.object(L, 'query', return_value=dict(status='OK')):
            self.assertEqual(L.translator_interpreter('/venv/bin/python3'), '/venv/bin/python3')

    def test_rank_worker_sets_gpu_and_correct_catalog(self):
        cat=self.catalog();run=self.root/'run';run.mkdir();(run/'ranks').mkdir()
        executable=self.root/'fake app.py'
        executable.write_text('''#!/usr/bin/env python3
import os,json
from pathlib import Path
Path('environment.json').write_text(json.dumps(dict(os.environ)))
print('Loaded 1 precomputed block sizes from '+os.environ['OPS_BLOCKSIZE_CSV'])
print('GPU Direct enabled')
print('Total Wall time 1.25')
''');executable.chmod(0o755)
        spec=dict(case=dict(grid=[64,64,64],niter=500,shape=[4,4,3],mode=2),executable=str(executable),workdir=str(run),catalog=str(cat))
        L.atomic(run/'spec.json',spec)
        with patch.dict(os.environ,dict(SLURM_PROCID='0',SLURM_LOCALID='0')):
            self.assertEqual(R.rank_worker(run/'spec.json',run),0)
        env=json.loads((run/'environment.json').read_text())
        self.assertEqual(env['MPICH_GPU_SUPPORT_ENABLED'],'1')
        self.assertEqual(env['OPS_BLOCKSIZE_CSV'],str(cat/'blocks.csv'))
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'],'0')
        with patch.dict(os.environ,dict(SLURM_PROCID='3',SLURM_LOCALID='3')):
            self.assertEqual(R.rank_worker(run/'spec.json',run),0)
        env=json.loads((run/'environment.json').read_text())
        self.assertEqual(env['OPS_DISABLE_CSV_LOGS'],'1')
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'],'3')

    def test_dependency_and_no_duplicate_submissions(self):
        campaign=self.root/'campaign';campaign.mkdir()
        (campaign/'campaign.json').write_text('{}')
        config=dict(nodes=[6,8],account='project_test',partitions={'6':'standard-g','8':'standard-g'},
                    time='02:00:00',build_jobs=4,build_partition='small-g',build_time='00:30:00',launcher_sha256={})
        sent=[]
        def fake(cmd,**kwargs):
            sent.append(cmd)
            return subprocess.CompletedProcess(cmd,0,stdout=str(100+len(sent))+'\n',stderr='')
        args=argparse.Namespace(resume=False)
        with patch.object(L.subprocess,'run',side_effect=fake):
            L.submit(campaign,args,config)
            L.submit(campaign,args,config)
        self.assertEqual(len(sent),3)
        for cmd in sent[1:]:self.assertIn('--dependency=afterok:101',cmd)
        self.assertEqual(L.read_json(campaign/'submissions.json')['node_jobs'],{'6':'102','8':'103'})

    def test_resume_only_failed_nodes(self):
        campaign=self.root/'resume';campaign.mkdir();(campaign/'build').mkdir()
        exe=campaign/'build/app';exe.write_text('binary')
        L.atomic(campaign/'build/build_manifest.json',dict(executable='build/app',sha256=L.sha(exe)))
        L.atomic(campaign/'submissions.json',dict(build_job='100',node_jobs={'6':'101','8':'102'}))
        config=dict(nodes=[6,8],account='project_test',partitions={'6':'standard-g','8':'standard-g'},time='01:00:00',launcher_sha256={})
        with patch.object(L,'scheduler_state',side_effect=lambda j:'FAILED' if j=='101' else 'RUNNING'),patch.object(L.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'103\n','')) as run:
            L.submit(campaign,argparse.Namespace(resume=True),config)
        self.assertEqual(run.call_count,1)
        self.assertEqual(L.read_json(campaign/'submissions.json')['node_jobs'],{'6':'103','8':'102'})

    def test_failed_rank_does_not_pass(self):
        run=self.root/'fail';run.mkdir();(run/'ranks').mkdir()
        exe=self.root/'fail.sh';exe.write_text('#!/bin/sh\nexit 3\n');exe.chmod(0o755)
        L.atomic(run/'spec.json',dict(case=dict(grid=[8,8,8],niter=500,shape=[1,1,1],mode=0),executable=str(exe),workdir=str(run),catalog=''))
        with patch.dict(os.environ,dict(SLURM_PROCID='0',SLURM_LOCALID='0')):
            self.assertEqual(R.rank_worker(run/'spec.json',run),1)
        self.assertEqual(L.read_json(run/'ranks/status_0.json')['rank_status'],'FAILED')

    def metadata(self, path, grid_x='24'):
        fields = ('nstencil_args dim elem_size size_x size_y size_z d_m_x d_m_y d_m_z '
                  'd_p_x d_p_y d_p_z num_read num_write num_rw grid_x grid_y grid_z '
                  'n_args_x_offset n_args_y_offset n_args_z_offset max_radius_x '
                  'max_radius_y max_radius_z total_offset_points').split()
        row = dict.fromkeys(fields, '1')
        row.update(kernel_id='2', stencil_sig='test-stencil', grid_x=grid_x, max_threads='1024')
        L.write_csv(path, [row])

    def test_transfer_rejects_geometry_and_block_limit_changes(self):
        catalog = self.catalog()
        old, new = self.root/'old.csv', self.root/'new.csv'
        self.metadata(old); self.metadata(new)
        self.assertEqual(L.validate_transfer(old,new,catalog/'blocks.csv')['status'],'PASS')
        self.metadata(new, grid_x='12')
        with self.assertRaisesRegex(ValueError,'changed grid_x'):
            L.validate_transfer(old,new,catalog/'blocks.csv')
        self.metadata(new)
        record = L.rows(new)[0]; record['max_threads'] = '128'; L.write_csv(new,[record])
        with self.assertRaisesRegex(ValueError,'current kernel limit'):
            L.validate_transfer(old,new,catalog/'blocks.csv')

    def test_node_pipeline_smokes_results_transfer_and_resume(self):
        dataset = self.root/'dataset'; dataset.mkdir()
        (dataset/'selections.csv').write_text('fixture\n')
        campaign = self.root/'campaign'; (campaign/'build').mkdir(parents=True)
        exe = campaign/'build/app'; exe.write_text('fake binary')
        L.atomic(campaign/'build/build_manifest.json',dict(executable='build/app',sha256=L.sha(exe)))
        cases = []
        for index, shape in enumerate(([4,4,3], [4,3,4], [3,4,4])):
            rel = f'n006/catalog{index}'
            folder = dataset/rel; folder.mkdir(parents=True)
            (folder/'blocks.csv').write_text('kernel_id,bx,by,bz\n2,128,1,2\n')
            ctx = dict(APP_NAME='tgvstorenone', GPU_ARCH='gfx90a', shape=L.shape_text(shape),
                       ACTIVE_NUM_RANKS='48', GRID_X='24', GRID_Y='24', GRID_Z='24',
                       APP_ARGS='24 24 24 500', EXECUTABLE_SHA256='historical-binary')
            manifest = dict(schema_version=2, context=ctx, files={'blocks.csv':L.sha(folder/'blocks.csv')})
            manifest['set_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            L.atomic(folder/'manifest.json',manifest)
            source = f'n006/source{index}'
            self.metadata(dataset/source/'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv')
            cases.append(dict(nodes=6,ranks=48,grid=[24]*3,shape=shape,niter=500,mode=2,
                roles=['C','D_top1'] if index == 0 else [f'D_top{index+1}'],
                catalog_path=rel,source_run_path=source,source_set_id=manifest['set_id'],
                original_executable_sha256='historical-binary',case_id=f'n006_case{index}'))
        cases.insert(0,dict(cases[0],mode=0,roles=['A'],catalog_path='',source_set_id='',case_id='n006_default'))
        files = {str(f.relative_to(dataset)):dict(sha256=L.sha(f)) for f in dataset.rglob('*') if f.is_file()}
        L.atomic(dataset/'dataset_manifest.json',dict(application='tgvstorenone',files=files,selection_sha256=L.sha(dataset/'selections.csv')))
        original_hashes = {name:spec['sha256'] for name,spec in files.items()}
        config = dict(dataset=str(dataset),dataset_manifest_sha256=L.sha(dataset/'dataset_manifest.json'),
                      stamp='test',cases=cases,nodes=[6],resume=False)
        L.atomic(campaign/'campaign.json',config)
        calls = []
        def fake_srun(command, **kwargs):
            self.assertEqual(command[0], 'srun')
            spec = L.read_json(Path(command[command.index('--rank-spec')+1]))
            case=spec['case']; run=Path(spec['workdir']); calls.append(case['niter'])
            for rank in range(case['ranks']):
                (run/'ranks'/f'rank_{rank}.log').write_text('Total kernel time: 2.0\nTotal halo exchange time: 0.5\n')
                L.atomic(run/'ranks'/f'status_{rank}.json',dict(rank_id=rank,rank_status='OK',
                    rank_wall_time_s=4.0,app_reported_wall_time_s=3.0,precomputed_loaded=True))
            self.metadata(run/'rank_0/autotune_on/ops_blocksize_tuning_logmod.csv')
            return subprocess.CompletedProcess(command,0)
        with patch.dict(os.environ,SLURM_JOB_NUM_NODES='6'), patch.object(R,'gpu_preflight'), patch.object(L.subprocess,'run',side_effect=fake_srun):
            L.run_node(campaign,6)
            self.assertEqual(calls,[1,1,1,500,500,500,500])
            root=campaign/'OPS_LOGS_test_n006'
            self.assertEqual(len(L.rows(root/'summary.csv')),4)
            self.assertEqual(len(L.rows(root/'comparisons.csv')),9)
            for case in cases:
                self.assertTrue(R.complete(R.target_path(root,case),dict(case,executable_sha256=L.sha(exe))))
            replay=L.read_json(root/'precomputed_blocks/decomp4x4x3/manifest.json')
            self.assertEqual(replay['context']['EXECUTABLE_SHA256'],L.sha(exe))
            self.assertEqual(replay['source_set_id'],cases[1]['source_set_id'])
            config['resume']=True; L.atomic(campaign/'campaign.json',config)
            L.run_node(campaign,6)
            self.assertEqual(len(calls),7)
        for name,digest in original_hashes.items(): self.assertEqual(L.sha(dataset/name),digest)
        L.collect(campaign)
        self.assertTrue(L.read_json(campaign/'collection_status.json')['complete'])


if __name__=='__main__':unittest.main()
