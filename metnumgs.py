"""
Gauss-Seidel Iteration - 4-node tire thermal network
Solves [A]{T} = {b}, where T is temperature rise above ambient (K).
  T1 = tread surface, T2 = inner tread/belt, T3 = carcass, T4 = inner gas
"""
import numpy as np

# ---------------- 1. Problem definition ----------------
A = np.array([[ 50, -30,   0,   0],
              [-30,  60, -25,   0],
              [  0, -25,  40, -10],
              [  0,   0, -10,  14]], dtype=float)   # thermal conductance (W/K)
b = np.array([1500, 500, 250, 0], dtype=float)       # heat generation (W)

# ---------------- 2. Settings ----------------
x0      = np.zeros(4)
tol     = 1e-4          # max |relative approx. error| (%)
max_it  = 200

def is_diagonally_dominant(A):
    d = np.abs(np.diag(A)); off = np.abs(A).sum(axis=1) - d
    return bool(np.all(d >= off) and np.any(d > off))

def gauss_seidel(A, b, x0, tol, max_it):
    n = len(b)
    x = x0.copy()
    history = []
    for k in range(1, max_it + 1):
        x_old = x.copy()
        for i in range(n):
            # Gauss-Seidel: use the MOST RECENT values (already updated in this sweep)
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.abs((x - x_old) / x) * 100
        err = np.nan_to_num(err, nan=0.0, posinf=100.0)
        history.append((k, x.copy(), err.copy()))
        if err.max() < tol:
            break
    return x, history

if __name__ == "__main__":
    print("Diagonally dominant:", is_diagonally_dominant(A))
    print(f"Initial guess = {x0}, tol = {tol} %, max iterations = {max_it}\n")
    x, hist = gauss_seidel(A, b, x0, tol, max_it)

    print(f"{'k':>3} {'T1':>10} {'T2':>10} {'T3':>10} {'T4':>10} "
          f"{'e1(%)':>10} {'e2(%)':>10} {'e3(%)':>10} {'e4(%)':>10}")
    for k, xs, er in hist:
        print(f"{k:>3} " + " ".join(f"{v:10.4f}" for v in xs) + " "
              + " ".join(f"{v:10.5f}" for v in er))

    exact = np.linalg.solve(A, b)
    print(f"\nConverged after {len(hist)} iterations")
    print("Gauss-Seidel solution (K):", np.round(x, 4))
    print("Direct solution       (K):", np.round(exact, 4))
    print("Residual ||Ax-b||        :", np.linalg.norm(A @ x - b))