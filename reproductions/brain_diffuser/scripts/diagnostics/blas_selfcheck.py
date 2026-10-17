#!/usr/bin/env python
"""Refuse to run a ridge on a BLAS that computes the wrong matrix product.

Why: the OpenBLAS shipped with numpy 1.23 / scipy 1.9 predates Sapphire Rapids support (0.3.21) and returns
a wrong A @ A.T. sklearn.Ridge builds its gram matrix with that call when p > n (15724 > 8859 here); the
shared1000 reconstructions came out at chance. `OPENBLAS_CORETYPE=SkylakeX` selects a kernel that is correct.

Outputs: exit code 0 = this interpreter and these thread settings are safe; 1 = not, and nothing downstream counts.
Reads no data, writes nothing, no GPU, a few seconds. Run it in the same shell just before the regressions.

Usage:
    python diagnostics/blas_selfcheck.py && python scripts/vdvae_regression.py -sub 1
"""

import os
import sys
import time

import numpy as np
import sklearn
import sklearn.linear_model as skl

# p > n, as in the real fits; sklearn takes the dual (gram) path.
N, P, T = 2000, 15724, 8
TOL_PRODUCT = 1e-12   # relative; a correct BLAS lands around 1e-16
TOL_GRAD = 1e-10      # relative stationarity residual of the ridge solution

fail = []
t0 = time.time()

print('BLAS self-check  (numpy %s, sklearn %s)'
      % (np.__version__, sklearn.__version__))
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    print('   %-22s %s' % (v, os.environ.get(v, '<unset>')))
try:
    from threadpoolctl import threadpool_info
    for d in threadpool_info():
        print('   %-8s %-9s threads=%-4s arch=%s'
              % (d.get('user_api'), d.get('internal_api'), d.get('num_threads'),
                 d.get('architecture')))
except Exception:
    pass

# --- 1. is the matrix product self-consistent? ------------------------------
# Same A A^T three ways: syrk (numpy dispatches to it when both operands are the
# same object), gemm, and gemm on 500-row blocks. They must agree to rounding.
rng = np.random.default_rng(0)
A = rng.standard_normal((N, P))

K_syrk = A @ A.T
K_gemm = A @ A.T.copy()
K_blk = np.empty_like(K_syrk)
for i in range(0, N, 500):
    K_blk[i:i + 500] = A[i:i + 500] @ A.T

scale = np.abs(K_syrk).max()
for name, D in (('syrk vs blocked', K_syrk - K_blk),
                ('gemm vs blocked', K_gemm - K_blk),
                ('syrk vs gemm   ', K_syrk - K_gemm)):
    rel = float(np.abs(D).max() / scale)
    ok = rel < TOL_PRODUCT
    print('   %s   rel diff = %.3e   %s' % (name, rel, 'ok' if ok else 'BROKEN'))
    if not ok:
        fail.append('matrix product: ' + name.strip())
del K_syrk, K_gemm, K_blk

# --- 2. does Ridge return the actual minimiser? -----------------------------
# The objective is convex and separable per target, and w=0, b=mean(y) is always
# admissible at zero penalty; a correct solve has gradient ~0 and in-sample
# R2 >= 0: for any alpha, n and p. Either check failing means the solver did
# not minimise what it claims to minimise.
Y = A @ rng.standard_normal((P, T)) / np.sqrt(P) + rng.standard_normal((N, T))
alpha = 60000.0
m = skl.Ridge(alpha=alpha, fit_intercept=True).fit(A, Y)
W, b = m.coef_.T, m.intercept_

R = A @ W + b - Y
grad = float(np.abs(A.T @ R + alpha * W).max()
             / np.abs(A.T @ (Y - Y.mean(0))).max())
sse = ((Y - (A @ W + b)) ** 2).sum(0)
sst = ((Y - Y.mean(0)) ** 2).sum(0)
r2 = float(np.mean(1.0 - sse / sst))

print('   ridge |grad| / |X^T Yc| = %.3e   %s'
      % (grad, 'ok' if grad < TOL_GRAD else 'BROKEN'))
print('   ridge in-sample R2      = %+.4f   %s'
      % (r2, 'ok' if r2 >= 0.0 else 'BROKEN (negative is impossible)'))
if not grad < TOL_GRAD:
    fail.append('ridge is not at the minimum')
if r2 < 0.0:
    fail.append('ridge in-sample R2 < 0')

print('   %.1f s' % (time.time() - t0))
if fail:
    print('\nFAIL: ' + '; '.join(fail))
    print('Do NOT run the regressions in this environment: the features they')
    print('predict would be noise.')
    sys.exit(1)
print('\nPASS: this environment computes matrix products and ridge solves correctly.')
sys.exit(0)
