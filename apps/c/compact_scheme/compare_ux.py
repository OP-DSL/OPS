#!/usr/bin/env python3
"""compare_ux.py

Usage: compare_ux.py [ref.h5] [gpu.h5] [tol]

Compares /Compact3D/ux between two HDF5 files using absolute tolerance.
Exits with code 0 if max(abs(ref-gpu)) <= tol, otherwise exits 1.
"""

import h5py
import numpy as np
import sys

ref = "Compact3d_dev_seq.h5"
gpu = "Compact3d_cuda.h5"
tol = 1e-14
if len(sys.argv) >= 3:
    ref = sys.argv[1]
    gpu = sys.argv[2]
if len(sys.argv) >= 4:
    try:
        tol = float(sys.argv[3])
    except Exception:
        print("Invalid tolerance, using default", tol)

with h5py.File(ref, 'r') as fref, h5py.File(gpu, 'r') as fgpu:
    if '/Compact3D/ux' not in fref or '/Compact3D/ux' not in fgpu:
        print('Missing /Compact3D/ux dataset in one of the files'); sys.exit(2)
    dref = fref['/Compact3D/ux'][()]
    dgpu = fgpu['/Compact3D/ux'][()]

if dref.shape != dgpu.shape:
    print('Shape mismatch:', dref.shape, dgpu.shape)
    sys.exit(2)

diff = np.abs(dref - dgpu)
max_diff = float(diff.max())
mean_diff = float(diff.mean())
count = int(np.count_nonzero(diff > 0.0))

print(f"Total differing entries: {count}")
if count == 0:
    print('PASS: files identical')
    sys.exit(0)

# Show first 50 significant differences
idx = np.argwhere(diff > 0.0).ravel()
for k in range(min(50, idx.size)):
    i = idx[k]
    print(f"index {i}: ref={dref.flat[i]:.17g} gpu={dgpu.flat[i]:.17g} diff={diff.flat[i]:.17g}")

print(f"max diff: {max_diff:.17g}")
print(f"mean diff: {mean_diff:.17g}")

if max_diff <= tol:
    print(f'PASS: max_diff <= tol ({max_diff:.17g} <= {tol})')
    sys.exit(0)
else:
    print(f'FAIL: max_diff > tol ({max_diff:.17g} > {tol})')
    sys.exit(1)