#!/usr/bin/env python3
"""Test double of srun: starts the command once per rank with the Slurm rank variables."""
import json
import os
import subprocess
import sys

options, command = {}, []
argv = sys.argv[1:]
while argv and argv[0].startswith('--'):
    key, _, val = argv.pop(0)[2:].partition('=')
    options[key] = val
command = argv
ntasks, per_node = int(options['ntasks']), int(options['ntasks-per-node'])
if os.environ.get('FAKE_SRUN_LOG'):
    with open(os.environ['FAKE_SRUN_LOG'], 'a') as f:
        f.write(json.dumps(dict(options=options, command=command)) + '\n')
counter = os.environ.get('FAKE_SRUN_LOG', '') + '.step'
step = 0
if os.environ.get('FAKE_SRUN_LOG'):
    step = int(open(counter).read()) + 1 if os.path.exists(counter) else 0
    open(counter, 'w').write(str(step))
procs = []
for rank in range(ntasks):
    env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank % per_node), SLURM_NTASKS=str(ntasks),
               SLURM_STEP_ID=str(step), SLURMD_NODENAME='nid%06d' % (5000 + rank // per_node),
               SLURM_CPU_BIND=options.get('cpu-bind', ''))
    procs.append(subprocess.Popen(command, env=env))
    if len(procs) % 40 == 0:
        for p in procs:
            p.wait()
codes = [p.wait() for p in procs]
sys.exit(max(codes) if codes else 1)
