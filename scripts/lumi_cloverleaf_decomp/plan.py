"""Decomposition list and the frozen, ordered run plan (no Slurm needed)."""
import hashlib
import json

from common import require, shape_text, parse_shape

# The 16 decompositions of the sweep (fixed by the user on 2026-11-17: the 16 fastest of the SENGA2
# sweep at 160 ranks). Px x Py x Pz in OPS axis order (x is the contiguous dimension).
DEFAULT_DECOMPOSITIONS = ('5x8x4', '4x10x4', '5x16x2', '4x20x2', '8x5x4', '5x32x1', '8x4x5', '8x10x2',
                          '4x40x1', '8x20x1', '10x4x4', '2x40x2', '10x8x2', '10x2x8', '10x16x1', '16x10x1')
# Narrowest local subdomain OPS accepts: the halo depth of the CloverLeaf_3D dats.
# Source: apps/c/CloverLeaf_3D/build_field.cpp (d_p = {2,2,2}, d_m = {-2,-2,-2}) and the OPS check
# "Error: overpartitioning!" in ops/c/src/mpi/ops_mpi_rt_support.cpp.
HALO_DEPTH = 2
# OPS copies each argument through a 64-byte buffer (snprintf(temp, 64, ...) in
# ops/c/src/core/ops_lib_core.cpp): 63 characters at most.
ARG_MAX_CHARS = 63
DEFAULT_BLOCK = (32, 4, 1)
MODE_NUMBER = {'D': 0, 'A': 1, 'P': 2}   # default blocks, autotune, precomputed replay
MODE_NAME = {'D': 'default', 'A': 'autotune', 'P': 'precomputed'}
DEFAULT_SEED = 20261117
PLAN_VERSION = 1


def decomposition_option(text):
    """--decompositions: 'default' (the 16 of DEFAULT_DECOMPOSITIONS), 'all' (every valid ordered
    triple) or a comma-separated list PxxPyxPz,PxxPyxPz,..."""
    text = str(text).strip()
    if text in ('', 'default'):
        return list(DEFAULT_DECOMPOSITIONS)
    if text == 'all':
        return 'all'
    items = [t.strip() for t in text.split(',') if t.strip()]
    require(items, 'Empty --decompositions list')
    out = [shape_text(parse_shape(t)) for t in items]
    require(len(set(out)) == len(out), 'Repeated decomposition in --decompositions: %s' % text)
    return out


def describe_set(value):
    if value == 'all':
        return 'all'
    if list(value) == list(DEFAULT_DECOMPOSITIONS):
        return 'default (%d)' % len(DEFAULT_DECOMPOSITIONS)
    return 'list (%d)' % len(value)


def ordered_triples(ranks):
    """All ordered (Px, Py, Pz) with Px*Py*Pz == ranks, in lexicographic order."""
    require(isinstance(ranks, int) and ranks > 0, 'The rank count must be a positive integer')
    out = []
    for px in range(1, ranks + 1):
        if ranks % px:
            continue
        rest = ranks // px
        for py in range(1, rest + 1):
            if rest % py == 0:
                out.append((px, py, rest // py))
    return out


def check_triple(triple, ranks):
    triple = tuple(int(v) for v in triple)
    require(len(triple) == 3 and all(v > 0 for v in triple), 'Invalid decomposition %r' % (triple,))
    product = triple[0] * triple[1] * triple[2]
    require(product == ranks,
            'Decomposition %s has product %d but the job has %d MPI ranks. OPS only rejects a product '
            'larger than the rank count, so the launcher refuses anything but an exact match.'
            % (shape_text(triple), product, ranks))
    return triple


def validity(triple, grid):
    """(valid, reason). Valid iff every N/P is an integer >= HALO_DEPTH."""
    for axis, parts in zip('xyz', triple):
        if grid % parts:
            return False, 'grid %d is not divisible by P%s=%d' % (grid, axis, parts)
        if grid // parts < HALO_DEPTH:
            return False, ('local %s width %d is below the halo depth %d (OPS would abort with "overpartitioning")'
                           % (axis, grid // parts, HALO_DEPTH))
    return True, ''


def permutation_class(triple):
    return shape_text(sorted(triple))


def enumerate_decompositions(ranks, grid):
    valid, skipped = [], []
    for triple in ordered_triples(ranks):
        ok, reason = validity(triple, grid)
        if ok:
            valid.append(triple)
        else:
            skipped.append(dict(decomposition=shape_text(triple), reason=reason))
    return valid, skipped


def seeded_order(triples, seed):
    """Deterministic permutation that does not depend on the Python version."""
    def key(triple):
        return hashlib.sha256(('%d:%s' % (seed, shape_text(triple))).encode()).hexdigest()
    return sorted(triples, key=key)


def check_argv(argv):
    for token in argv:
        require(len(token) <= ARG_MAX_CHARS,
                'Application argument %r is %d characters long; OPS truncates arguments at %d '
                '(64-byte buffer in ops_lib_core.cpp). Shorten it.'
                % (token, len(token), ARG_MAX_CHARS))
        require(token and not any(c.isspace() for c in token), 'Invalid application argument %r' % (token,))
    return list(argv)


def app_argv(mode_number, triple, gpudirect=True):
    """Arguments of the CloverLeaf_3D executable (it reads clover.in; no positional arguments).

    The default block is passed in every mode so that a kernel missing from a
    precomputed catalog falls back to exactly 32,4,1."""
    argv = ['-OPS_AUTOTUNE_MODE=%d' % mode_number]
    argv += ['OPS_BLOCK_SIZE_%s=%d' % (a, v) for a, v in zip('XYZ', DEFAULT_BLOCK)]
    if triple is not None:
        argv += ['OPS_FORCE_DECOMP_%s=%s' % (a, v) for a, v in zip('XYZ', triple)]
    if gpudirect:
        argv.append('-gpudirect')
    argv.append('-OPS_DIAGS=2')
    return check_argv(argv)


def rank_mode(mode, rank):
    """Autotuning mode of one MPI rank: tuning happens on rank 0 only."""
    if mode == 'A':
        return 1 if rank == 0 else 0
    return MODE_NUMBER[mode]


def make_run(run_id, stage, mode, decomposition, cfg, build, nodes, ranks, steps, **extra):
    """decomposition: 'PxxPyxPz', 'unforced' or 'd0' (resolved from the unforced run)."""
    require(mode in MODE_NUMBER, 'Unknown mode %r' % (mode,))
    gpudirect = extra.pop('gpudirect', True)
    if decomposition == 'unforced':
        triple = None
    elif decomposition == 'd0':
        triple = ['{d0x}', '{d0y}', '{d0z}']
    else:
        triple = list(check_triple(parse_shape(decomposition), ranks))
    run = dict(run_id=run_id, stage=stage, mode=mode, mode_name=MODE_NAME[mode], decomposition=decomposition,
               triple=None if decomposition in ('unforced', 'd0') else triple,
               build=build, nodes=nodes, ranks=ranks, steps=steps, gpudirect=gpudirect,
               repetition=extra.pop('repetition', 1), pair_index=extra.pop('pair_index', None),
               order_in_pair=extra.pop('order_in_pair', None), decomp_position=extra.pop('decomp_position', None),
               seed=cfg['seed'], argv_resolved=decomposition != 'd0',
               argv=app_argv(rank_mode(mode, 0), triple, gpudirect),
               argv_other_ranks=app_argv(rank_mode(mode, 1), triple, gpudirect))
    run.update(extra)
    return run


def resolve_d0(run, d0):
    """Fill the placeholders of a 'd0' run with the grid printed by the unforced run."""
    if run['decomposition'] != 'd0':
        return dict(run)
    d0 = check_triple(d0, run['ranks'])
    values = dict(d0x=d0[0], d0y=d0[1], d0z=d0[2])
    out = dict(run, triple=list(d0), argv_resolved=True, resolved_decomposition=shape_text(d0))
    out['argv'] = check_argv([token.format(**values) for token in run['argv']])
    out['argv_other_ranks'] = check_argv([token.format(**values) for token in run['argv_other_ranks']])
    return out


def pair_order(decomp_position, pair_index):
    """D P / P D / D P for even plan positions, P D / D P / P D for odd ones."""
    first_is_d = (decomp_position + pair_index) % 2 == 1   # pair_index starts at 1
    return ('D', 'P') if first_is_d else ('P', 'D')


def sweep_triples(cfg):
    """(kept, skipped, excluded) of the sweep. cfg['decompositions'] is 'all' or a list of 'PxxPyxPz'."""
    ranks, grid = cfg['ranks'], cfg['grid']
    chosen = cfg.get('decompositions', list(DEFAULT_DECOMPOSITIONS))
    if chosen == 'all':
        valid, skipped = enumerate_decompositions(ranks, grid)
    else:
        valid, skipped = [], []
        for text in chosen:
            triple = check_triple(parse_shape(text), ranks)
            ok, reason = validity(triple, grid)
            if ok:
                valid.append(triple)
            else:
                skipped.append(dict(decomposition=shape_text(triple), reason=reason))
    excluded, requested = [], set()
    for text in cfg.get('exclude') or []:
        triple = check_triple(parse_shape(text), ranks)
        require(triple in valid, 'Excluded decomposition %s is not in the planned set' % shape_text(triple))
        requested.add(triple)
    kept = []
    for triple in valid:
        if triple in requested:
            excluded.append(dict(decomposition=shape_text(triple), reason='explicit --exclude'))
        else:
            kept.append(triple)
    return kept, skipped, excluded


def make_sweep_plan(cfg):
    """cfg: ranks, nodes, grid, steps, repetitions, seed, fixed_order, sentinel_every,
    exclude (list of 'PxxPyxPz'), decompositions ('all' or a list of 'PxxPyxPz'; default: the 16 of
    DEFAULT_DECOMPOSITIONS)."""
    ranks, nodes, grid, steps = cfg['ranks'], cfg['nodes'], cfg['grid'], cfg['steps']
    reps, every = cfg['repetitions'], cfg['sentinel_every']
    require(ranks == nodes * 8, 'This package runs one rank per GCD: ranks must equal 8 x nodes')
    require(reps >= 1 and every >= 1 and steps >= 1 and grid >= HALO_DEPTH, 'Invalid plan parameters')
    kept, skipped, excluded = sweep_triples(cfg)
    require(kept, 'No decomposition left to run (are the decompositions valid for %d ranks and grid %d?)' % (ranks, grid))
    order = sorted(kept) if cfg.get('fixed_order') else seeded_order(kept, cfg['seed'])
    common = dict(build='full', nodes=nodes, ranks=ranks, steps=steps)
    runs = []
    for rep in range(1, reps + 1):
        runs.append(make_run('baseline_unforced_r%d' % rep, 'baseline', 'D', 'unforced', cfg, repetition=rep, **common))
    runs.append(make_run('sentinel_start', 'sentinel', 'D', 'd0', cfg, sentinel_kind='allocation_start', **common))
    decomps = []
    for position, triple in enumerate(order):
        text = shape_text(triple)
        prefix = 'sweep_p%03d_%s' % (position, text)
        first = pair_order(position, 1)[0]
        decomps.append(dict(position=position, decomposition=text, triple=list(triple),
                            permutation_class=permutation_class(triple),
                            local_size=[grid // p for p in triple], first_pair_mode=first))
        runs.append(make_run(prefix + '_A', 'sweep', 'A', text, cfg, decomp_position=position, **common))
        for pair in range(1, reps + 1):
            for index, mode in enumerate(pair_order(position, pair), start=1):
                runs.append(make_run('%s_pair%d_%s' % (prefix, pair, mode), 'sweep', mode, text, cfg,
                                     repetition=pair, pair_index=pair, order_in_pair=index,
                                     decomp_position=position, **common))
        if (position + 1) % every == 0 and position + 1 < len(order):
            runs.append(make_run('sentinel_after_p%03d' % position, 'sentinel', 'D', 'd0', cfg,
                                 sentinel_kind='periodic', after_decomp_position=position, **common))
    for index, run in enumerate(runs):
        run['position'] = index
    chosen = cfg.get('decompositions', list(DEFAULT_DECOMPOSITIONS))
    return dict(plan_version=PLAN_VERSION, kind='sweep', seed=cfg['seed'],
                order='lexicographic (--fixed-order)' if cfg.get('fixed_order') else 'seeded sha256 permutation',
                ranks=ranks, nodes=nodes, grid=grid, steps=steps, repetitions=reps, sentinel_every=every,
                decomposition_set=describe_set(chosen), requested_decompositions=chosen,
                halo_depth=HALO_DEPTH, default_block=list(DEFAULT_BLOCK),
                skipped=skipped, excluded=excluded, decompositions=decomps, runs=runs,
                dynamic_runs='Every Slurm allocation after the first starts with one extra sentinel run '
                             '(sentinel_alloc_<jobid>, forced d0, default blocks). d0 is the processor grid '
                             'printed by OPS in the unforced baseline; {d0x},{d0y},{d0z} are filled at run time.')


def make_smoke_plan(cfg):
    """Qualification sequence Q1-Q5 (Q0 is the build and preflight). One executable serves every
    grid (CloverLeaf reads the grid from clover.in); 'small' and 'full' name the two problem sizes."""
    small, grid = cfg['small_grid'], cfg['grid']
    nodes, ranks, steps = cfg['nodes'], cfg['ranks'], cfg['steps']
    runs = []

    def add(run_id, stage, mode, decomposition, build, n, r, s, **extra):
        size = small if build == 'small' else grid
        if decomposition not in ('unforced', 'd0'):
            triple = check_triple(parse_shape(decomposition), r)
            ok, reason = validity(triple, size)
            if not ok:
                return skipped.append(dict(run_id=run_id, decomposition=decomposition, reason=reason))
        runs.append(make_run(run_id, stage, mode, decomposition, cfg, build, n, r, s, **extra))

    skipped = []
    one = dict(build='small', n=1, r=1, s=steps)
    add('q1_1x1x1_D', 'Q1', 'D', '1x1x1', **one)
    add('q2_1x1x1_A', 'Q2', 'A', '1x1x1', **one)
    add('q2_1x1x1_P', 'Q2', 'P', '1x1x1', **one)
    eight = dict(build='small', n=1, r=8, s=steps)
    for text in ('2x2x2', '8x1x1', '1x8x1', '1x1x8'):
        add('q3_%s_D' % text, 'Q3', 'D', text, **eight)
    add('q3_unforced_D', 'Q3', 'D', 'unforced', **eight)
    add('q3_2x2x2_D_nogpudirect', 'Q3', 'D', '2x2x2', gpudirect=False, **eight)
    full = dict(build='full', n=nodes, r=ranks, s=cfg['smoke_steps'])
    add('q4_unforced_D', 'Q4', 'D', 'unforced', **full)
    kept = sweep_triples(dict(cfg, exclude=[]))[0]
    if kept:   # the thinnest planned triple: the smallest local width of the sweep
        thin = min(kept, key=lambda t: (min(grid // p for p in t), t))
        add('q4_%s_D' % shape_text(thin), 'Q4', 'D', shape_text(thin), **full)
    five = dict(build='full', n=nodes, r=ranks, s=steps)
    add('q5_d0_A', 'Q5', 'A', 'd0', **five)
    for index, mode in enumerate(('D', 'P', 'P', 'D'), start=1):
        add('q5_d0_%d_%s' % (index, mode), 'Q5', mode, 'd0', repetition=(index + 1) // 2, **five)
    for index, run in enumerate(runs):
        run['position'] = index
    return dict(plan_version=PLAN_VERSION, kind='qualification', seed=cfg['seed'], ranks=ranks, nodes=nodes,
                grid=grid, small_grid=small, steps=steps, smoke_steps=cfg['smoke_steps'], skipped=skipped,
                stages=dict(Q0='build and preflight of the executable: generated HIP kernels carry the autotuning '
                               'hooks, the executable links libmpi_gtl_hsa, hash recorded (run.sh build)',
                            Q1='small grid, 1 rank, default blocks: field summary, device memory, time',
                            Q2='small grid, 1 rank: autotune then replay of its own catalog',
                            Q3='small grid, 8 ranks on 1 node, GPU direct: forced triples, unforced, and one run '
                               'without -gpudirect; field summaries agree across decompositions',
                            Q4='full grid, all ranks, %d steps: unforced (identifies d0) and the thinnest planned '
                               'triple' % cfg['smoke_steps'],
                            Q5='full grid, forced d0, %d steps: A then D P P D; measures the time per step' % steps),
                runs=runs)


def plan_bytes(plan):
    return (json.dumps(plan, indent=1, sort_keys=True) + '\n').encode()


def plan_hash(plan):
    return hashlib.sha256(plan_bytes(plan)).hexdigest()


PLAN_CSV_FIELDS = ['position', 'run_id', 'stage', 'decomposition', 'mode', 'mode_name', 'repetition', 'pair_index',
                   'order_in_pair', 'decomp_position', 'build', 'nodes', 'ranks', 'steps', 'gpudirect', 'seed',
                   'argv', 'argv_other_ranks']


def plan_rows(plan):
    out = []
    for run in plan['runs']:
        out.append(dict(run, argv=' '.join(run['argv']), argv_other_ranks=' '.join(run['argv_other_ranks'])))
    return out


def summary(plan):
    counts = {}
    for run in plan['runs']:
        key = (run['stage'], run['mode'])
        counts[key] = counts.get(key, 0) + 1
    return counts
