#!/usr/bin/env python3
"""Compare multiple /Compact3D datasets between two HDF5 files.

Usage: compare_all.py file1.h5 file2.h5 [tol]
Checks datasets: ux, uy, uz and reports max/mean differences. Exits 0 if
all max diffs <= tol, else exits 1.
"""
import sys
import h5py
import numpy as np

datasets = ["/Compact3D/ux", "/Compact3D/uy", "/Compact3D/uz"]

if len(sys.argv) < 3:
    print("Usage: compare_all.py file1.h5 file2.h5 [tol]")
    sys.exit(2)

f1 = sys.argv[1]
f2 = sys.argv[2]
tol = 1e-14
if len(sys.argv) >= 4:
    tol = float(sys.argv[3])

fail = False
with h5py.File(f1, 'r') as A, h5py.File(f2, 'r') as B:
    for d in datasets:
        if d not in A or d not in B:
            print(f"Missing dataset {d} in one of the files")
            fail = True
            continue
        a = A[d][()]
        b = B[d][()]
        if a.shape != b.shape:
            print(f"Shape mismatch for {d}: {a.shape} vs {b.shape}")
            fail = True
            continue
        diff = np.abs(a - b)
        mx = float(diff.max())
        mn = float(diff.mean())
        nz = int(np.count_nonzero(diff > 0.0))
        print(f"{d}: max={mx:.17g} mean={mn:.17g} differing_entries={nz}")
        if mx > tol:
            fail = True

if fail:
    print(f"FAIL: one or more datasets exceed tol={tol}")
    sys.exit(1)
else:
    print(f"PASS: all datasets within tol={tol}")
    sys.exit(0)
