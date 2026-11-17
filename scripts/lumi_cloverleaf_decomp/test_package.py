"""Package-level checks: syntax, no hardcoded identity, README commands, Slurm job scripts."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import launcher as L
import testlib as T
from common import read_json, slurm_seconds

HERE = Path(L.HERE)


class PackageTests(unittest.TestCase):
    def test_shell_and_python_syntax(self):
        shells = sorted(HERE.glob('*.sh'))
        self.assertEqual([p.name for p in shells], ['environment.sh', 'run.sh'])
        for path in shells:
            r = subprocess.run(['bash', '-n', str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(r.returncode, 0, r.stdout)
        sources = sorted(HERE.glob('*.py')) + sorted((HERE / 'fixtures').glob('*.py'))
        self.assertGreaterEqual(len(sources), 15)
        with tempfile.TemporaryDirectory() as tmp:
            for index, path in enumerate(sources):
                code = 'import py_compile, sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)'
                r = subprocess.run([sys.executable, '-c', code, str(path), os.path.join(tmp, '%d.pyc' % index)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(r.returncode, 0, r.stdout)

    def test_launcher_modules_parse_with_python_3_6_syntax(self):
        """The LUMI system python3 is 3.6: no walrus, no positional-only arguments, no text=/capture_output=."""
        for name in L.PACKAGE_FILES:
            if name.endswith('.py'):
                text = (HERE / name).read_text()
                for banned in (':=', 'capture_output=', 'text=True', 'from __future__ import annotations', 'shlex.join',
                               'math.prod', 'dirs_exist_ok', 'removeprefix', 'removesuffix'):
                    self.assertNotIn(banned, text, '%s uses %s' % (name, banned))

    def test_no_hardcoded_project_user_or_home(self):
        name = 'kev' + 'in'
        expression = '|'.join(['project_' + '[0-9]+', '/home/' + name, '/us' + 'ers/', name, 'm4rk' + '00v'])
        pattern = re.compile(expression)
        hits = []
        for path in sorted(HERE.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
                    if pattern.search(line):
                        hits.append('%s:%d: %s' % (path.relative_to(HERE), number, line.strip()[:80]))
        self.assertEqual(hits, [])
        r = subprocess.run(['grep', '-rnE', expression, '--exclude-dir=__pycache__', str(HERE)],
                           stdout=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(r.stdout, '')

    def test_job_scripts_and_sbatch_lines_take_the_account_from_the_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = T.make_campaign(tmp)
            T.fake_build(campaign)
            _, text = T.quiet(L.main, ['build', '--campaign', str(campaign), '--submit', '--dry-run'])
            self.assertIn('--account=%s --partition=small-g --nodes=1 --ntasks=1 --cpus-per-task=16' % T.ACCOUNT, text)
            script = (campaign / 'slurm/build.sh').read_text()
            self.assertIn('source %s/environment.sh' % campaign, script)
            self.assertIn('job --campaign %s --kind build' % campaign, script)
            self.assertIn('CLOVER_JOB_TIME_LIMIT_S=%d' % slurm_seconds('02:00:00'), script)
            bindir = Path(tmp) / 'bin'
            bindir.mkdir()
            (bindir / 'sbatch').write_text('#!/bin/sh\necho "$@" >> %s/sbatch_args\necho 4242\n' % tmp)
            (bindir / 'squeue').write_text('#!/bin/sh\nexit 0\n')
            (bindir / 'sacct').write_text('#!/bin/sh\necho "4242|clover_q_small|COMPLETED|0:0|00:10:00"\n')
            for name in ('sbatch', 'squeue', 'sacct'):
                (bindir / name).chmod(0o755)
            with patch.dict(os.environ, dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'])):
                status, text = T.quiet(L.main, ['smoke', '--campaign', str(campaign), '--submit'])
                self.assertEqual(status, 0, text)
                lines = (Path(tmp) / 'sbatch_args').read_text().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertIn('--account=%s --partition=standard-g --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 '
                              '--cpus-per-task=7 --mem=0 --time=01:00:00 --exclusive' % T.ACCOUNT, lines[0])
                self.assertIn('--partition=standard-g --nodes=1', lines[1])
                self.assertIn('--dependency=afterok:4242', lines[1])
                self.assertIn('--kind smoke --builds small', (campaign / 'slurm/smoke_small.sh').read_text())
                self.assertIn('--kind smoke --builds full', (campaign / 'slurm/smoke_full.sh').read_text())
                status, text = T.quiet(L.main, ['status', '--campaign', str(campaign)])
                self.assertEqual(status, 0)
                for expected in ('build sha256', 'qualification plan', 'sweep plan: not frozen', 'go/no-go: not evaluated',
                                 'job 4242 (smoke-small)', 'COMPLETED'):
                    self.assertIn(expected, text)
                # A campaign created with the old small-g default can send the one-node job elsewhere,
                # for one submission, without editing campaign.json.
                config = read_json(campaign / 'campaign.json')
                config['smoke_partition'] = 'small-g'
                (campaign / 'campaign.json').write_text(json.dumps(config))
                status, text = T.quiet(L.main, ['smoke', '--campaign', str(campaign), '--submit', '--stage', 'small'])
                self.assertEqual(status, 0, text)
                self.assertIn('--partition=small-g --nodes=1', (Path(tmp) / 'sbatch_args').read_text().splitlines()[-1])
                status, text = T.quiet(L.main, ['smoke', '--campaign', str(campaign), '--submit', '--stage', 'small',
                                                '--smoke-partition', 'dev-g'])
                self.assertEqual(status, 0, text)
                self.assertIn('--partition=dev-g --nodes=1', (Path(tmp) / 'sbatch_args').read_text().splitlines()[-1])
                self.assertEqual(read_json(campaign / 'campaign.json')['smoke_partition'], 'small-g')
            self.assertEqual([j['kind'] for j in read_json(campaign / 'submissions.json')['jobs']][:2], ['smoke-small', 'smoke-full'])

    def test_no_command_prints_help_and_errors_are_clear(self):
        r = subprocess.run(['bash', str(HERE / 'run.sh')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(r.returncode, 0)
        for command in ('plan', 'prepare', 'build', 'smoke', 'gonogo', 'submit', 'status', 'collect', 'upgrade'):
            self.assertIn(command, r.stdout)
        r = subprocess.run(['bash', str(HERE / 'run.sh'), 'status', '--campaign', '/nonexistent/campaign'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('ERROR: /nonexistent/campaign is not a campaign directory', r.stderr)


class ReadmeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (HERE / 'README.md').read_text()
        cls.blocks = re.findall(r'```bash\n(.*?)```', cls.text, re.S)
        parser = L.parser()
        actions = [a for a in parser._actions if a.__class__.__name__ == '_SubParsersAction'][0]
        cls.commands = {name: {opt for action in sub._actions for opt in action.option_strings}
                        for name, sub in actions.choices.items()}

    def invocations(self):
        out = []
        for block in self.blocks:
            for line in block.replace('\\\n', ' ').splitlines():
                line = line.split('#')[0].replace('<', '').replace('>', '').strip()
                if 'run.sh' not in line:
                    continue
                tokens = shlex.split(line)
                index = next(i for i, t in enumerate(tokens) if t.endswith('run.sh'))
                out.append((tokens[index + 1], [t.split('=')[0] for t in tokens[index + 2:] if t.startswith('--')], line))
        return out

    def test_every_command_and_option_shown_exists(self):
        calls = self.invocations()
        self.assertGreaterEqual(len(calls), 18)
        for command, options, line in calls:
            self.assertIn(command, self.commands, line)
            for option in options:
                self.assertIn(option, self.commands[command], line)
        shown = {command for command, _, _ in calls}
        self.assertEqual(shown, {'plan', 'prepare', 'build', 'smoke', 'gonogo', 'submit', 'status', 'collect', 'upgrade'})
        upgrade = [line for command, _, line in calls if command == 'upgrade']
        self.assertEqual(upgrade, ['bash ~/OPS/scripts/lumi_cloverleaf_decomp/run.sh upgrade --campaign $C',
                                   'bash ~/OPS/scripts/lumi_cloverleaf_decomp/run.sh upgrade --campaign $C '
                                   '--decompositions default --restart-sweep'])
        for option in re.findall(r'`(--[a-z0-9-]+)', self.text):          # options named in the prose
            self.assertTrue(any(option in opts for opts in self.commands.values()), option)

    def test_readme_examples_parse(self):
        parser = L.parser()
        for command, _, line in self.invocations():
            tokens = shlex.split(line)
            argv = tokens[next(i for i, t in enumerate(tokens) if t.endswith('run.sh')) + 1:]
            argv = [a.replace('$C', '/c') for a in argv]
            args = parser.parse_args([a if a not in ('rate', 'hours', 'GB', 'N') else '1' for a in argv])
            self.assertEqual(args.command, command)

    def test_readme_covers_the_whole_path_and_the_limits(self):
        order = ['run.sh plan --output-dir', 'run.sh prepare --account', 'run.sh build --campaign', 'run.sh smoke --campaign',
                 'run.sh gonogo --campaign', 'run.sh plan --campaign $C --steps', 'run.sh submit --campaign',
                 'run.sh collect --campaign']
        quick = self.text[self.text.index('## Quick path'):self.text.index('## 1. Steps')]
        positions = [quick.index(item) for item in order]
        self.assertEqual(positions, sorted(positions))
        for phrase in ('a kernel\n  that rank 0 does not execute is never tuned', 'each candidate', 'timed once',
                       'rank 0\n  is slower than its neighbours', 'no per-rank or per-node variation',
                       'What only LUMI can verify', 'libmpi_gtl_hsa', 'non-empty best-block CSV', 'numerically consistent',
                       'device memory per GCD', 'GPU direct is active', 'axis order', 'MPI_Dims_create',
                       'exact text of the log lines', 'wall times, budget consumption, disk', 'PrgEnv-gnu',
                       'selection-biased', 'Catalog stability is not measured', 'Exit statuses', 'never imputed',
                       'time per step', 'Suggested --steps', 'test_problem 0', 'Total Wall time', 'Total kernel time:',
                       'Total halo exchange time:', 'Average volume of MPI\ncommunications per process:',
                       '5x8x4 4x10x4 5x16x2 4x20x2 8x5x4 5x32x1 8x4x5 8x10x2 4x40x1 8x20x1 10x4x4 2x40x2 10x8x2 10x2x8 10x16x1 16x10x1'):
            self.assertIn(phrase, self.text, phrase)
        for name in ('environment.sh', 'run.sh'):
            self.assertTrue((HERE / name).is_file())
        variables = set(re.findall(r'`(CLOVER_[A-Z0-9_]+)`', self.text)) - {'CLOVER_IN_SHA256'}   # a config.txt key
        import sources as S
        known = set(re.findall(r'CLOVER_[A-Z0-9_]+', ' '.join(name for name, _, _ in S.TOOLCHAIN_HELP)))
        self.assertEqual(variables - known, set())
        self.assertEqual(known - variables, set())
        for status in (0, 2, 3, 4, 5, 6):
            self.assertRegex(self.text, r'\n\| %d \| ' % status)

    def test_environment_file_only_loads_modules_and_exports(self):
        text = (HERE / 'environment.sh').read_text()
        for line in ('module load LUMI', 'module load partition/G', 'module load PrgEnv-gnu', 'module load rocm',
                     'module load craype-accel-amd-gfx90a', 'export MPICH_GPU_SUPPORT_ENABLED=1', 'export CLOVER_CXXFLAGS=',
                     'export CLOVER_HIPFLAGS='):
            self.assertIn(line, text)
        self.assertNotIn('sbatch', text)
        self.assertNotIn('--account', text)
        self.assertNotIn('SENGA', text)
        # A ready PrgEnv-gnu environment is preserved: with PE_ENV=GNU, hipcc and CRAY_MPICH_DIR present
        # nothing is loaded and the defaults are set.
        with tempfile.TemporaryDirectory() as tmp:
            hipcc = Path(tmp) / 'hipcc'
            hipcc.write_text('#!/bin/sh\n')
            hipcc.chmod(0o755)
            command = 'source %s && echo "$CLOVER_CXXFLAGS|$CLOVER_HIPFLAGS|$AMD_ARCH|$MPICH_GPU_SUPPORT_ENABLED"' % (HERE / 'environment.sh')
            env = {k: v for k, v in os.environ.items() if not k.startswith('CLOVER_')}
            env.update(PATH=tmp + os.pathsep + os.environ['PATH'], CRAY_MPICH_DIR=tmp, PE_ENV='GNU')
            r = subprocess.run(['bash', '-c', command], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(r.stdout.strip(), '-O3 -fPIC -std=c++11|-O3 -std=c++11 --offload-arch=gfx90a '
                                               '-D__HIP_PLATFORM_AMD__|gfx90a|1', r.stderr)
            r = subprocess.run(['bash', '-c', command], env=dict(env, CLOVER_CXXFLAGS='-O1'),
                               stdout=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(r.stdout.strip().split('|')[0], '-O1')


if __name__ == '__main__':
    unittest.main()
