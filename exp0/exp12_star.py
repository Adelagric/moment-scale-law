"""PAPER 2 — trancher : une boucle géométriquement complexe échappe-t-elle au 4ᵉ moment ?

Test du point faible de la loi N*=4·(features) : une étoile à k branches est UN seul trou
(β_1=1) mais géométriquement complexe (rayon variable). Pour k>=3 elle est isotrope
(covariance aveugle par symétrie). Si la kurtosis (4ᵉ moment) NE détecte PLUS le trou quand
k croît, alors N* dépend de la complexité GÉOMÉTRIQUE, pas seulement de Σβ_k.

étoile creuse (boucle, H_1=1) vs étoile pleine (H_1=0), r(θ)=1+a·cos(kθ). AUC creuse vs pleine.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import squareform, pdist
from ripser import ripser

RNG = np.random.default_rng(0)
GRID = np.linspace(0, 2.5, 60)


def h1_land(P):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(GRID-b, d-GRID), 0, None))
    return lam

def mardia(P):
    mu = P.mean(0); Pc = P-mu; C = np.cov(P, rowvar=False)+1e-9*np.eye(P.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))

def knn(P, k=5):
    d = squareform(pdist(P, "euclidean")); d.sort(axis=1); return float(np.mean(d[:, k]))

def star_loop(k, a=0.4, n=200, noise=0.02):
    t = RNG.uniform(0, 2*np.pi, n); r = 1 + a*np.cos(k*t)
    return np.c_[r*np.cos(t), r*np.sin(t)] + RNG.normal(0, noise, (n, 2))

def star_filled(k, a=0.4, n=200, noise=0.02):
    # rayon ×√2 => E[r²] apparié à l'étoile creuse (covariance matchée, B6 aveugle) — cf. cercle/disque
    t = RNG.uniform(0, 2*np.pi, n)
    r = np.sqrt(2)*(1 + a*np.cos(k*t))*np.sqrt(RNG.uniform(0, 1, n))
    return np.c_[r*np.cos(t), r*np.sin(t)] + RNG.normal(0, noise, (n, 2))


def auc(loop_fn, fill_fn, k, N=24):
    train = [loop_fn(k) for _ in range(15)]
    ref = np.mean([h1_land(w) for w in train], axis=0)
    Sig = np.mean([np.cov(w, rowvar=False) for w in train], axis=0)
    def sc(w):
        return {"Δ_top": float(np.sum(np.abs(h1_land(w)-ref))),
                "B6": float(np.linalg.norm(np.cov(w, rowvar=False)-Sig, "fro")),
                "kNN": knn(w), "Kurt": mardia(w)}
    H = [sc(loop_fn(k)) for _ in range(N)]; A = [sc(fill_fn(k)) for _ in range(N)]
    out = {}
    for det in H[0]:
        a_ = roc_auc_score([0]*N+[1]*N, [s[det] for s in H]+[s[det] for s in A])
        out[det] = max(a_, 1-a_)
    return out


def main():
    print(f"{'k branches':>11} {'Δ_top':>7} {'B6':>6} {'kNN':>6} {'Kurtosis':>9}")
    for k in (0, 3, 5, 8):
        o = auc(star_loop, star_filled, k)
        print(f"{k:>11} {o['Δ_top']:>7.2f} {o['B6']:>6.2f} {o['kNN']:>6.2f} {o['Kurt']:>9.2f}")
    print("\nLecture : si Kurtosis BAISSE quand k monte (Δ_top reste haut), alors une boucle")
    print("géométriquement complexe échappe au 4ᵉ moment => N* dépend de la géométrie, pas de Σβ_k")
    print("seul. Si Kurtosis reste haute => N*=4 même pour boucle complexe (loi topologique pure).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
