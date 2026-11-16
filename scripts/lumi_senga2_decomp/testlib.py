"""Shared test scaffolding: synthetic inputs, a fake Slurm and a mock executable.

No network, GPU, Slurm or LUMI is needed, and no SENGA2 source is used: the tree
written here only has the few lines the launcher edits or reads."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

import launcher as L
from common import sha, atomic, read_json

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
BEST_FIXTURE = FIXTURES / 'hpec_mi300a_senga2_best_logmod.csv'
ACCOUNT = 'acct_test'      # a placeholder, not a real project

CONT_DAT = '''Synthetic run control file (test scaffolding)

Global domain size (x,y,z) in metres
 6.28318531D-3   6.28318531D-3   6.28318531D-3

Global domain size (nx,ny,nz)
 256    256    256

No. of processors (x,y,z)
 1      1      1

Time step; start step, no of steps, step switch (0=fixed, 1=adaptive))
 1.0D-12    1   10   1

Intervals between dumps, reports, statistics, dump output (0/1=un/formatted)
 10    100    100      0

End of file
'''
COM_SENGA = '''MODULE com_senga
    integer(kind=4), parameter :: nxglbl=256, nyglbl=256, nzglbl=256
    integer(kind=4), parameter :: ngzmax=nxglbl
    integer(kind=4), parameter :: nxsize=256, nysize=256, nzsize=256
    integer(kind=4), parameter :: nszmax=nxsize
    integer(kind=4), parameter :: nhalox=5,nhaloy=5,nhaloz=5
END MODULE com_senga
'''
MAIN = '''PROGRAM senga2
    use OPS_Fortran_Reference
    use OPS_Fortran_hdf5_Declarations
    use com_senga
    call ops_init(6)
    call ops_fetch_block_hdf5_file(senga_grid, 'x.h5')
    call ops_decl_dat_hdf5(d_drun, senga_grid, 1, "real(kind=8)", &
                           "DRUN", 'x.h5', status)
    call output
    call ops_exit
END PROGRAM senga2
'''
MAKEFILE = '''USE_HDF5=1
OTHER_FILES = com_senga2.F90 \\
    finish.F90
OPS_FILES = senga2.F90 output.F90 print_dats.F90 print_alldats.F90 \\
    print_output.F90
APP_FLAGS = -g -cpp -O3
APP = senga2
MAIN_SRC=senga2
'''
LIB_SCRIPT = '#!/bin/sh\n$%s -O3 -c a.f\n$%s -O3 -c b.f\nar rcs %s.a a.o b.o\nrm *.o\nexit\n'


def synthetic_senga(root):
    """A stand-in tree with the layout the launcher expects (not SENGA2 code)."""
    root = Path(root)
    src = root / 'src'
    for sub in ('fftlib', 'rndlib', 'parlib'):
        (src / sub).mkdir(parents=True)
        var = 'MPIF90' if sub == 'parlib' else 'F90'
        script = {'fftlib': 'Makeftl', 'rndlib': 'Makeran', 'parlib': 'Makepar'}[sub]
        (src / sub / script).write_text(LIB_SCRIPT % (var, var, sub))
        for name in ('a.f', 'b.f'):
            (src / sub / name).write_text('      subroutine %s_%s\n      end\n' % (sub, name[0]))
    (root / 'input').mkdir()
    (src / 'com_senga2.F90').write_text(COM_SENGA)
    (src / 'senga2.F90').write_text(MAIN)
    (src / 'Makefile').write_text(MAKEFILE)
    (src / 'finish.F90').write_text('SUBROUTINE finish\nEND SUBROUTINE finish\n')
    (src / 'output.F90').write_text('SUBROUTINE output\n    use OPS_Fortran_hdf5_Declarations\n'
                                    '    call ops_fetch_dat_hdf5_file(d_uinf2, trim(fname))\nEND SUBROUTINE output\n')
    for name in ('print_dats', 'print_alldats', 'print_output'):
        (src / (name + '.F90')).write_text('SUBROUTINE %s()\n    use OPS_Fortran_hdf5_Declarations\n'
                                           '    call ops_fetch_block_hdf5_file(senga_grid, trim(fname))\n'
                                           'END SUBROUTINE %s\n' % (name, name))
    (root / 'input/cont.dat').write_text(CONT_DAT)
    for name in ('input/chem.dat', 'input/diff.dat', 'input/radn.dat', 'TGV_species.dat', 't_initial.dat'):
        (root / name).write_text('synthetic %s\n' % name)
    return root


def quiet(function, *args, **kwargs):
    """Call and return (result, printed text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = function(*args, **kwargs)
    return result, out.getvalue()


def make_campaign(tmp, nodes=1, grid=20, small_grid=20, steps=3, reps=1, sentinel_every=4, extra=()):
    """Run the real `prepare` against the real OPS checkout and a synthetic application tree."""
    tmp = Path(tmp)
    senga = synthetic_senga(tmp / 'app_tree')
    env_file = tmp / 'env.sh'
    env_file.write_text('export MPICH_GPU_SUPPORT_ENABLED=1\n')
    campaign = tmp / 'campaign'
    argv = ['prepare', '--account', ACCOUNT, '--campaign', str(campaign), '--senga-root', str(senga),
            '--env-file', str(env_file), '--nodes', str(nodes), '--grid', str(grid), '--small-grid', str(small_grid),
            '--steps', str(steps), '--smoke-steps', '2', '--repetitions', str(reps),
            '--sentinel-every', str(sentinel_every), '--decompositions', 'all'] + list(extra)
    status, text = quiet(L.main, argv)
    assert status == 0, text
    return campaign


def fake_build(campaign, names=('small', 'full')):
    """Stand-in for `build`: the mock executable with a recorded hash per build."""
    for name in names:
        folder = Path(campaign) / 'build' / name
        folder.mkdir(parents=True, exist_ok=True)
        exe = folder / 'senga2_f2c_mpi_hip'
        exe.write_text((FIXTURES / 'mock_senga2.py').read_text() + '\n# %s build\n' % name)
        exe.chmod(0o755)
        atomic(folder / 'build_manifest.json', dict(
            build=name, executable=str(exe.relative_to(campaign)), sha256=sha(exe), gtl_linked=True,
            generated_hip_kernels_with_hooks=581, grid=read_json(Path(campaign) / 'campaign.json')['builds'][name]['grid']))


def fixture_ids():
    return [int(line.split(',')[0]) for line in BEST_FIXTURE.read_text().splitlines()[1:] if line.split(',')[0].isdigit()]


EXTRA_IDS = [9001, 9002, 9003]       # executed but never fully tuned: fall back to default


def mock_config(tmp, **settings):
    """Write the mock's behaviour file; returns its path."""
    data = dict(best_csv=str(BEST_FIXTURE), executed_ids=sorted(fixture_ids() + EXTRA_IDS))
    data.update(settings)
    path = Path(tmp) / 'mock_config.json'
    path.write_text(json.dumps(data))
    return path


def job_env(tmp, job='1001', remaining=10 ** 6, nodes=1, **settings):
    """Environment of a fake Slurm allocation (fake srun first on PATH)."""
    tmp = Path(tmp)
    bindir = tmp / 'bin'
    bindir.mkdir(exist_ok=True)
    srun = bindir / 'srun'
    srun.write_text('#!/bin/sh\nexec %s %s "$@"\n' % (sys.executable, FIXTURES / 'fake_srun.py'))
    srun.chmod(0o755)
    config = mock_config(tmp, **settings)       # always rewritten: no behaviour leaks between jobs
    return dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'], SLURM_JOB_ID=str(job),
                SLURM_JOB_END_TIME=str(int(time.time() + remaining)), SLURM_JOB_NUM_NODES=str(nodes),
                SLURM_JOB_NODELIST='nid[005000-%06d]' % (5000 + nodes - 1), MOCK_SENGA_CONFIG=str(config),
                FAKE_SRUN_LOG=str(tmp / 'srun_calls.log'), OPS_LAUNCHER_PYTHON=sys.executable,
                LOADEDMODULES='LUMI/test:partition/G:rocm/test')


def srun_calls(tmp):
    """Run ids launched through the fake srun, in order."""
    log = Path(tmp) / 'srun_calls.log'
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
        out.append(json.loads(line)['run_id'])
    return out


def run_job(campaign, kind, env, *extra):
    """Run one job body in-process with the fake allocation; returns (status, printed text)."""
    from unittest.mock import patch
    with patch.dict(os.environ, env):
        return quiet(L.main, ['job', '--campaign', str(campaign), '--kind', kind] + list(extra))


def copy_tree(source, target):
    shutil.copytree(str(source), str(target), symlinks=True)
    return Path(target)


# ----- sweep runs live in the dataset layout (dataset.py)
def layout(campaign):
    import engine as E
    return E.Campaign(campaign).layout


def sweep_folder(campaign, run_id, attempt=1):
    """Folder of attempt N of a sweep run: its final folder if that attempt was committed, else _attempts/."""
    lay = layout(campaign)
    loc = lay.locate(run_id)
    record = loc['final'] / 'record.json'
    if record.is_file() and read_json(record)['attempt'] == 'attempt_%03d' % attempt:
        return loc['final']
    return loc['attempts'] / ('attempt_%03d' % attempt)


def sweep_record(campaign, run_id, attempt=1):
    return read_json(sweep_folder(campaign, run_id, attempt) / 'record.json')


def interrupt(campaign, run_id):
    """Turn a committed run into an attempt cut short by the end of an allocation:
    back in _attempts/, without record.json or dataset files (they are written at the end of a run)."""
    lay = layout(campaign)
    loc = lay.locate(run_id)
    name = read_json(loc['final'] / 'record.json')['attempt']
    target = loc['attempts'] / name
    os.rename(str(loc['final']), str(target))
    for leftover in ('record.json', 'config.txt', 'run.log', 'mpi_rank_metrics.csv'):
        (target / leftover).unlink()
    lay.refresh()
    return target
