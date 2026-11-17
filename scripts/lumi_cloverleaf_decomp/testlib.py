"""Shared test scaffolding: a fake Slurm and a mock executable.

No network, GPU, Slurm or LUMI is needed. The campaign snapshot is taken from the real OPS
checkout that contains this package (apps/c/CloverLeaf_3D included); the executable is a mock."""
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
BEST_FIXTURE = FIXTURES / 'hpec_mi250x_cloverleaf_best_logmod.csv'
ACCOUNT = 'acct_test'      # a placeholder, not a real project
FIXTURE_ROWS = 122         # kernel rows of the real best-block CSV (LUMI MI250X, cloverleaf 400x160x125)


def quiet(function, *args, **kwargs):
    """Call and return (result, printed text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = function(*args, **kwargs)
    return result, out.getvalue()


def make_campaign(tmp, nodes=1, grid=20, small_grid=20, steps=3, reps=1, sentinel_every=4, extra=(),
                  decompositions='all'):
    """Run the real `prepare` against the real OPS checkout."""
    tmp = Path(tmp)
    env_file = tmp / 'env.sh'
    env_file.write_text('export MPICH_GPU_SUPPORT_ENABLED=1\n')
    campaign = tmp / 'campaign'
    argv = ['prepare', '--account', ACCOUNT, '--campaign', str(campaign), '--env-file', str(env_file),
            '--nodes', str(nodes), '--grid', str(grid), '--small-grid', str(small_grid), '--steps', str(steps),
            '--smoke-steps', '2', '--repetitions', str(reps), '--sentinel-every', str(sentinel_every),
            '--decompositions', decompositions] + list(extra)
    status, text = quiet(L.main, argv)
    assert status == 0, text
    return campaign


def fake_build(campaign):
    """Stand-in for `build`: the mock executable with a recorded hash."""
    folder = Path(campaign) / 'build' / 'hip'
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / 'cloverleaf_mpi_hip'
    exe.write_text((FIXTURES / 'mock_cloverleaf.py').read_text())
    exe.chmod(0o755)
    atomic(folder / 'build_manifest.json', dict(
        build='hip', executable=str(exe.relative_to(campaign)), sha256=sha(exe), gtl_linked=True,
        generated_hip_kernels_with_hooks=141, toolchain=dict(hipflags='-O3 --offload-arch=gfx90a')))


def fixture_ids():
    return [int(line.split(',')[0]) for line in BEST_FIXTURE.read_text().splitlines()[1:] if line.split(',')[0].isdigit()]


EXTRA_IDS = [901, 902, 903]       # executed but never fully tuned: fall back to default (not in the fixture)


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
                SLURM_JOB_NODELIST='nid[005000-%06d]' % (5000 + nodes - 1), MOCK_CLOVER_CONFIG=str(config),
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
