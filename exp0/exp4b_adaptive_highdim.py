"""PAPER 1 — blindage : l'attaque ADAPTATIVE (cov-préservée) à dimension croissante.

L'attaque conçue pour FAVORISER la TDA (collapse + correction affine préservant (μ,Σ),
donc B6/Maha aveugles par construction) reste-t-elle rattrapée par la kurtosis à haute
dimension ? Si oui à 384/768/1024, la redondance est inattaquable même contre l'attaque
la plus favorable à la persistance.

Usage : python exp4b_adaptive_highdim.py <cache.npz> <dim_réduite>
"""
from __future__ import annotations
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import squareform, pdist
from ripser import ripser

RNG = np.random.default_rng(0)
CATS = 4


def h1_land(X, grid):
    dm = squareform(pdist(X, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    lam = np.zeros_like(grid)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(grid - b, d - grid), 0, None))
    return lam

def knn_density(X, k=5):
    d = squareform(pdist(X, "euclidean")); d.sort(axis=1); return float(np.mean(d[:, k]))

def mardia(X):
    mu = X.mean(0); Pc = X - mu; C = np.cov(X, rowvar=False) + 1e-6*np.eye(X.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))

def collapse(X, alpha=0.5, rho=0.35):
    X = X.copy(); mu = X.mean(0); idx = RNG.choice(len(X), int(rho*len(X)), replace=False)
    X[idx] = mu + (1-alpha)*(X[idx]-mu); return X

def affine_match(X, mu_t, Sig_t):
    mu1 = X.mean(0); S1 = np.cov(X, rowvar=False) + 1e-8*np.eye(X.shape[1])
    A = np.real(sqrtm(Sig_t)) @ np.linalg.inv(np.real(sqrtm(S1)))
    return (X - mu1) @ A.T + mu_t


def main():
    cache, dim = sys.argv[1], int(sys.argv[2])
    z = np.load(cache); X, y = z["X"], z["y"]
    Xr = PCA(n_components=dim).fit_transform(X)
    grid = np.linspace(0, np.percentile(pdist(Xr[np.random.default_rng(0).choice(len(Xr),300,False)], "euclidean"), 95), 60)

    def win(W=250):
        per = W // CATS; idx = []
        for c in range(CATS): idx += list(RNG.choice(np.where(y == c)[0], per, replace=False))
        return Xr[np.array(idx)]

    train = [win() for _ in range(12)]
    ref = np.mean([h1_land(w, grid) for w in train], axis=0)
    Sig = np.mean([np.cov(w, rowvar=False) for w in train], axis=0); Si = np.linalg.pinv(Sig)
    mu = np.mean([w.mean(0) for w in train], axis=0)

    def sc(w):
        diff = w - mu
        return {"Δ_top": float(np.sum(np.abs(h1_land(w, grid) - ref))),
                "B6": float(np.linalg.norm(np.cov(w, rowvar=False) - Sig, "fro")),
                "Maha": float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, Si, diff)))),
                "kNN": knn_density(w), "Kurt": mardia(w)}

    N = 16
    H = [win() for _ in range(N)]
    A, dcov = [], []
    for _ in range(N):
        h = win(); mu_h, Sig_h = h.mean(0), np.cov(h, rowvar=False)
        a = affine_match(collapse(h), mu_h, Sig_h); A.append(a)
        dcov.append(np.linalg.norm(np.cov(a, rowvar=False) - Sig_h, "fro"))

    sh = [sc(w) for w in H]; sa = [sc(w) for w in A]
    print(f"\n=== {cache}  dim réduite={dim}  |Δcov|(attaque)={np.mean(dcov):.4f} ===")
    print("AUC bilatérale (attaque cov-préservée : B6/Maha doivent être aveugles) :")
    row = []
    for d in sh[0]:
        auc = roc_auc_score([0]*N+[1]*N, [s[d] for s in sh]+[s[d] for s in sa])
        row.append(f"{d}={max(auc,1-auc):.2f}")
    print("  " + "   ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
