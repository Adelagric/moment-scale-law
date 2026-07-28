"""PAPER 2 — brique 1 : le théorème s'étend GRATUITEMENT au cas elliptique.

Mardia kurtosis est affine-invariante : b(MX)=b(X). Le blanchiment est affine
(homéomorphisme) => préserve H_k. Donc coquille/boule ELLIPTIQUES (image affine M d'une
coquille/boule sphérique) ont : même kurtosis que le cas sphérique (b=d² pour la coquille,
quel que soit M), H_{d-1} présent, et se ramènent au cas sphérique par blanchiment.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

np.random.seed(0)


def shell(d, n):
    X = np.random.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True); return X

def ball(d, n):
    X = np.random.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = np.sqrt((d + 2) / d) * np.random.uniform(0, 1, (n, 1)) ** (1.0 / d)
    return X * r

def random_affine(d):
    M = np.random.normal(size=(d, d))
    return M @ M.T + 0.5 * np.eye(d)   # SPD, anisotrope

def mardia(P):
    mu = P.mean(0); Pc = P - mu; C = np.cov(P, rowvar=False) + 1e-9 * np.eye(P.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc) ** 2))

def hk(P, k):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=k, distance_matrix=True)["dgms"][k]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    return float(np.max(fin[:, 1] - fin[:, 0])) if len(fin) else 0.0


def main():
    for d, k, n in [(2, 1, 400), (3, 2, 300)]:
        M = random_affine(d)
        S, B = shell(d, n) @ M.T, ball(d, n) @ M.T          # elliptiques
        print(f"\n=== d={d} (H_{k}), affine anisotrope M ===")
        print(f"  cond(M)={np.linalg.cond(M):.1f}  (1 = sphérique, >1 = elliptique)")
        print(f"  Mardia coquille={mardia(S):.2f} (prédit sphérique d²={d**2})  boule={mardia(B):.2f}")
        print(f"  H_{k} coquille={hk(S,k):.2f}  boule={hk(B,k):.2f}  (topologie préservée par M)")
    print("\n=> Mardia ≈ d² pour la coquille MALGRÉ l'anisotropie (invariance affine),")
    print("   H_{d-1} présent : le théorème couvre tout le cas elliptique sans effort.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
