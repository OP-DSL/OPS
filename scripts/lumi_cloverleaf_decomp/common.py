"""Small shared helpers. Python >= 3.6 (the LUMI system python3) is enough."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

# Exit statuses of the entry script (documented in README.md, "Exit statuses").
EXIT_OK = 0
EXIT_ERROR = 2            # usage, configuration or environment error
EXIT_TIME_GUARD = 3       # stopped cleanly: not enough allocation time for the next run
EXIT_ABORT_FAILURES = 4   # three consecutive failed decompositions
EXIT_RUN_FAILURES = 5     # pass finished, but at least one run failed (never retried automatically)
EXIT_NO_GO = 6            # go/no-go verdict is NO-GO


class StopJob(Exception):
    """Raised to leave the job loop with a specific exit status."""

    def __init__(self, status, message):
        Exception.__init__(self, message)
        self.status = status


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=str(path.parent), delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    os.replace(str(tmp), str(path))


def atomic(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + '\n').encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def read_json_or(path, default=None):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, data, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in data:
            writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in fields})
    os.replace(str(tmp), str(path))


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def shell_join(command):
    return ' '.join(shlex.quote(str(arg)) for arg in command)


def shape_text(dims):
    return 'x'.join(str(int(v)) for v in dims)


def parse_shape(text):
    parts = str(text).lower().split('x')
    require(len(parts) == 3 and all(p.isdigit() and int(p) > 0 for p in parts),
            'Invalid decomposition %r: expected PxxPyxPz with positive integers, for example 4x5x8' % (text,))
    return [int(p) for p in parts]


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            raise ValueError('Another process holds the lock %s. Is another job of this campaign still running? '
                             'Check with squeue before retrying.' % path)
        yield


def query(command, timeout=30, env=None, cwd=None):
    """Run an informational command; never raises."""
    if not shutil.which(str(command[0])):
        return dict(status='NOT_QUERIED', command=[str(c) for c in command], stdout='', stderr='',
                    message='%s unavailable' % command[0])
    try:
        r = subprocess.run([str(c) for c in command], universal_newlines=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout, env=env, cwd=cwd)
        return dict(status='OK' if r.returncode == 0 else 'ERROR', command=[str(c) for c in command],
                    returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(status='UNKNOWN', command=[str(c) for c in command], stdout='', stderr='', message=str(e))


def slurm_seconds(text):
    """[D-]HH:MM:SS, HH:MM:SS, MM:SS or MM -> seconds (Slurm time formats)."""
    text = text.strip()
    days = 0
    if '-' in text:
        d, text = text.split('-', 1)
        days = int(d)
        parts = [int(p) for p in text.split(':')]
        parts += [0] * (3 - len(parts))        # D-HH[:MM[:SS]]
        h, m, s = parts[:3]
    else:
        parts = [int(p) for p in text.split(':')]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = parts[:3]
    return ((days * 24 + h) * 60 + m) * 60 + s
