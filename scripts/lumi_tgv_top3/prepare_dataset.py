#!/usr/bin/env python3
"""Build and audit the complete portable Top 3 shared-block dataset (Python 3.6+)."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

NODES = (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40)
POLICY = 'shared sampled-rank consensus: frequency, measured best_time, block tuple'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_csv(path, values, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(values[0]))
        w.writeheader()
        w.writerows(values)


def inside(root, rel):
    p = root / rel
    require(not Path(rel).is_absolute(), 'Absolute package path: ' + rel)
    p.resolve().relative_to(root.resolve())
    return p


def helper(root):
    spec = importlib.util.spec_from_file_location('ops_original_catalog', root / 'tools/ops_precomputed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_files(root):
    m = read(root / 'dataset_manifest.json')
    for rel, item in m['files'].items():
        p = inside(root, rel)
        require(p.is_file() and not p.is_symlink() and sha(p) == item['sha256'], 'Changed/missing file: ' + str(p))
    require(sha(root / 'selections.csv') == m['selection_sha256'], 'Selections changed')
    return m


def metadata(path):
    # TGV-SN has 26 IDs, including initialization kernel 1. Read each first row.
    found = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            kid = (row.get('kernel_id') or '').strip()
            if kid.isdigit():
                found.setdefault(int(kid), row)
            if set(found) == set(range(1, 27)):
                break
    require(set(found) == set(range(1, 27)), 'Unexpected TGV kernel IDs: ' + str(path))
    return found


def shared_audit(root, case, ops):
    run = inside(root, case['source_run_path'])
    paths = {r: run / ('rank_%d/autotune_on/ops_blocksize_best_logmod.csv' % r) for r in (0, 1)}
    data, disagreements = ops.shared_blocks(paths)
    best = {r: ops.best_block_rows(p) for r, p in paths.items()}
    meta = {r: metadata(run / ('rank_%d/autotune_on/ops_blocksize_tuning_logmod.csv' % r)) for r in paths}
    values = list(csv.DictReader(io.StringIO(data.decode())))
    require({int(v['kernel_id']) for v in values} == set(range(2, 27)), 'Expected all 25 tuned kernels')
    audits = []
    for value in values:
        kid = int(value['kernel_id'])
        block = tuple(int(value[a]) for a in ('bx', 'by', 'bz'))
        candidates = [(best[r][kid][0], r) for r in paths if best[r][kid][1] == block]
        require(candidates, 'Unmeasured block selected')
        elapsed, source_rank = min(candidates)
        for r in paths:
            require(block[0] * block[1] * block[2] <= float(meta[r][kid]['max_threads']), 'Block exceeds kernel limit')
            require(meta[r][kid]['stencil_sig'] == meta[0][kid]['stencil_sig'], 'Rank stencil mismatch')
        audits.append(dict(nodes=case['nodes'], catalog_path=case['catalog_path'], kernel_id=kid,
            bx=block[0], by=block[1], bz=block[2], source_rank=source_rank,
            source_best_time_s=elapsed, source_best_log=str(paths[source_rank].relative_to(root)),
            source_best_log_sha256=sha(paths[source_rank]), policy=POLICY))
    return data, disagreements, audits


def validate(root):
    m = verify_files(root)
    require(m.get('dataset_version') == 2, 'Expected complete shared dataset version 2')
    ops = helper(root)
    selection = rows(root / 'selections.csv')
    require(len(selection) == 36, 'Expected 36 original selections')
    coverage = rows(root / 'coverage.csv')
    require(len(coverage) == 45 and len({r['catalog_path'] for r in coverage}) == 45, 'Expected 45 unique catalogs')
    all_cases = []
    for n in NODES:
        cases = rows(root / ('n%03d/cases.csv' % n))
        require(len(cases) == 4 and {c['role'] for c in cases} == {'C', 'D_top1', 'D_top2', 'D_top3'}, 'Missing/duplicate roles')
        for c in cases:
            rank = '1' if c['role'] == 'C' else c['role'][-1]
            src = [r for r in selection if int(r['Nodes']) == n and r['Top rank'] == rank]
            require(len(src) == 1, 'Missing/duplicate selected source')
            field = 'Fixed baseline MPI decomposition' if c['role'] == 'C' else 'Tested MPI decomposition'
            require(c['global_grid'] == src[0]['Global grid'].replace('×', 'x') and c['mpi_decomposition'] == src[0][field].replace('×', 'x'), 'Selection mapping changed')
            require(int(c['nodes']) == n and int(c['mpi_ranks']) == n * 8 and c['simulation_iterations'] == '500', 'Wrong node/rank/step count')
            grid = list(map(int, c['global_grid'].split('x')))
            shape = list(map(int, c['mpi_decomposition'].split('x')))
            require(len(grid) == len(shape) == 3 and len(set(grid)) == 1 and shape[0]*shape[1]*shape[2] == n*8 and all(g % s == 0 for g,s in zip(grid,shape)), 'Invalid grid/decomposition')
            all_cases.append(c)
    audits = []
    for cov in coverage:
        cases = [c for c in all_cases if c['catalog_path'] == cov['catalog_path']]
        require(cases, 'Unreferenced catalog')
        c = cases[0]
        folder = inside(root, cov['catalog_path'])
        cat = read(folder / 'manifest.json')
        require(cat['schema_version'] == 2 and set(cat['files']) == {'blocks.csv'}, 'Not a shared catalog')
        ops.validate_catalog(folder, cat['context'])
        expected = dict(APP_NAME='tgvstorenone', GRID_CONFIG='cfg1', GPU_ARCH='gfx90a',
            DECOMPOSITION_MODE='1', ACTIVE_NUM_RANKS=c['mpi_ranks'], shape=c['mpi_decomposition'],
            APP_ARGS=c['global_grid'].replace('x', ' ') + ' 500', EXECUTABLE_SHA256=c['historical_executable_sha256'])
        expected.update(dict(zip(('GRID_X','GRID_Y','GRID_Z'), c['global_grid'].split('x'))))
        require(all(cat['context'].get(k) == v for k,v in expected.items()), 'Catalog context mismatch')
        require(all(x['set_id'] == cat['set_id'] and x['catalog_status'] == 'COMPLETE' and x['schema_version'] == '2' for x in cases), 'Case/catalog mismatch')
        require(cov['catalog_status'] == 'COMPLETE' and cov['block_files_missing'] == '0' and cov['block_files_available'] == '1', 'Incomplete catalog')
        data, _, audit = shared_audit(root, c, ops)
        require((folder / 'blocks.csv').read_bytes() == data, 'Shared catalog differs from original best logs')
        audits.extend(audit)
    recorded = rows(root / 'block_provenance.csv')
    require(recorded == [{k:str(v) for k,v in row.items()} for row in audits], 'Per-kernel provenance differs')
    require(not rows(root / 'missing_blocks.csv'), 'Runtime files are missing')
    require(m['summary']['full_campaign_ready'] and m['summary']['complete_catalogs'] == 45 and m['summary']['incomplete_catalogs'] == 0, 'Summary is not complete')
    print('PASS: 12 node sections, 48 roles, 45 shared catalogs, 1125 measured block selections; no missing runtime files.', flush=True)
    return m


def build(source, output):
    original = verify_files(source)
    require(not output.exists(), 'Output already exists; build in a new directory')
    output.mkdir(parents=True)
    ops = helper(source)
    # Preserve original measured logs/configs byte-for-byte. Runtime catalogs are rebuilt separately.
    for rel in original['files']:
        if '/source_campaigns/' in rel or rel in ('selections.csv', 'tools/ops_precomputed.py'):
            dest = output / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, dest)
    shutil.copyfile(Path(__file__), output / 'prepare_dataset.py')
    cases = [c for n in NODES for c in rows(source / ('n%03d/cases.csv' % n))]
    coverage, provenance, converted = [], [], 0
    for old in rows(source / 'coverage.csv'):
        rel = old['catalog_path']; folder = output / rel; folder.mkdir(parents=True)
        c = next(c for c in cases if c['catalog_path'] == rel)
        previous = read(source / rel / 'manifest.json')
        data, disagreements, audit = shared_audit(source, c, ops)
        provenance.extend(audit)
        if previous['schema_version'] == 2:
            require(ops.digest(data) == previous['files']['blocks.csv'], 'Original shared catalog differs')
            shutil.copyfile(source / rel / 'manifest.json', folder / 'manifest.json')
            if (source / rel / 'historical_manifest.json').exists():
                shutil.copyfile(source / rel / 'historical_manifest.json', folder / 'historical_manifest.json')
            manifest = previous
            origin = c.get('block_origin', 'ORIGINAL_SHARED_CATALOG')
        else:
            converted += 1
            shutil.copyfile(source / rel / 'manifest.json', folder / 'historical_manifest.json')
            manifest = dict(schema_version=2, context=dict(previous['context'],
                SAMPLED_AUTOTUNE='YES', AUTOTUNE_SAMPLE_RANKS='0 1'), selection=POLICY,
                source_run=c['source_run_path'], repetition=previous['repetition'],
                files={'blocks.csv':ops.digest(data)}, historical_set_id=previous['set_id'],
                historical_manifest_sha256=sha(source / rel / 'manifest.json'),
                sampled_selection=dict(sample_ranks=[0,1], sample_count=2,
                    disagreement_count=len(disagreements), disagreements=disagreements),
                transfer_note='Derived from retained rank 0/1 best logs of the same selected tuning repetition. Not an exact recovery of historical per-rank blocks; historical timings do not measure this shared policy.')
            manifest['set_id'] = ops.digest(json.dumps(manifest, sort_keys=True).encode())
            write_json(folder / 'manifest.json', manifest)
            origin = 'DERIVED_SHARED_FROM_HISTORICAL_RANKS_0_1'
        (folder / 'blocks.csv').write_bytes(data)
        for c in cases:
            if c['catalog_path'] == rel:
                c.update(set_id=manifest['set_id'], schema_version='2', catalog_status='COMPLETE',
                    source_logs_status='COMPLETE_FOR_SAMPLE_RANKS', block_scope='ALL_MPI_RANKS',
                    block_origin=origin, target_executable_status='NOT_VALIDATED')
        coverage.append(dict(nodes=old['nodes'], roles=old['roles'], catalog_path=rel,
            set_id=manifest['set_id'], schema_version=2, catalog_status='COMPLETE',
            block_files_expected=1, block_files_available=1, block_files_missing=0,
            source_logs_status='COMPLETE_FOR_SAMPLE_RANKS', source_run_path=old['source_run_path'],
            sample_ranks='0 1', block_scope='ALL_MPI_RANKS', block_origin=origin,
            tuned_kernel_count=25, default_fallback_kernel_ids='1', target_executable_status='NOT_VALIDATED'))
        print('COMPLETE:', old['nodes'], manifest['context']['shape'], origin, flush=True)
    for n in NODES:
        write_csv(output / ('n%03d/cases.csv' % n), [c for c in cases if int(c['nodes']) == n])
    write_csv(output / 'coverage.csv', coverage)
    write_csv(output / 'block_provenance.csv', provenance)
    write_csv(output / 'missing_blocks.csv', [], ['nodes','roles','catalog_path','relative_file'])
    if (source / 'historical_missing_blocks.csv').is_file():
        shutil.copyfile(source / 'historical_missing_blocks.csv', output / 'historical_missing_blocks.csv')
    else:
        shutil.copyfile(source / 'missing_blocks.csv', output / 'historical_missing_blocks.csv')
    text = '''TGV STORE NONE TOP 3 — COMPLETE PORTABLE SHARED DATASET, VERSION 2

All 12 node sections (2,3,4,6,8,10,12,16,20,24,32,40), 48 roles and 45 unique
runtime catalogs are complete. Each catalog has one blocks.csv shared by all
MPI ranks. Columns: kernel_id,bx,by,bz. No autotuning is required.

The selected grids/MPI Top 3 and source tuning repetitions are unchanged.
For 6–40 nodes the historical shared catalog bytes are unchanged.
For 2–4 nodes the historical per-rank exports are incomplete. A new shared
catalog is derived from the retained rank 0/1 best logs of the same selected
repetition, using the original OPS_APP_lumi_original_tuning shared_blocks
policy: majority block, then lowest measured best_time, then block tuple.
Every chosen block is an observed best block. No missing rank measurements
are fabricated. Historical per-rank manifests and source logs are preserved;
historical_missing_blocks.csv records the old gaps, not runtime requirements.
These derived shared catalogs require NEW timing measurements; do not assign
historical timings to them. This is a transfer experiment, not exact replay.

All 25 tuned kernels (2–26) have blocks. Initialization kernel 1 keeps the
existing OPS default fallback; it has no tuned best entry in the originals.
block_provenance.csv traces all 1125 block selections to source rank/log/time.
coverage.csv and nNNN/cases.csv describe the executable runtime inputs.
source_campaigns/ contains original measured logs/configs without edits.
Original absolute paths are provenance; runtime catalog paths are relative.

Validate all files/catalogs/provenance (Python 3.6+):
  python3 prepare_dataset.py --validate-only --dataset /path/to/TOP3_DATASET
Rebuild from a verified source package, in a new directory:
  python3 prepare_dataset.py --source-package /path/to/source_package \\
    --output-root /path/to/new/TOP3_DATASET

Local validation covers package/runtime input compatibility. The newly built
HIP executable, MPI and GPU Direct still require the launcher smoke on LUMI.
'''
    (output / 'README.txt').write_text(text)
    registry = {str(p.relative_to(output)):dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted(output.rglob('*')) if p.is_file()}
    supersedes = list(dict.fromkeys([sha(source / 'dataset_manifest.json')] + original.get('supersedes_manifest_sha256', [])))
    summary = dict(node_sections=12, selected_top3_rows=36, roles=48, unique_catalogs=45,
        complete_catalogs=45, incomplete_catalogs=0, missing_block_files=0,
        derived_shared_catalogs=sum(c['block_origin'].startswith('DERIVED') for c in coverage),
        shared_block_files=45, traced_kernel_blocks=1125, full_campaign_ready=True,
        target_executable_status='NOT_VALIDATED')
    write_json(output / 'dataset_manifest.json', dict(schema_version=1, dataset_version=2,
        application='tgvstorenone', created_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(output / 'selections.csv'), supersedes_manifest_sha256=supersedes,
        summary=summary, block_policy=POLICY, files=registry))
    validate(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-package', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    if args.validate_only:
        validate(args.dataset.resolve())
    else:
        require(args.source_package and args.output_root, 'Provide --source-package and --output-root')
        build(args.source_package.resolve(), args.output_root.resolve())


if __name__ == '__main__':
    main()
