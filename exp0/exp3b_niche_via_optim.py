"""FRONT 1 (suite) — la conjecture étant FAUSSE, l'attaque optimisée est-elle une
vraie niche de détection ?

On génère un lot d'attaques : chacune casse H1 en matchant PAR OPTIMISATION la
covariance, la distribution kNN ET la kurtosis de Mardia d'une boucle saine. Puis panel
de détecteurs. Si Δ_top≈1 et B6/kNN/kurtosis≈0.5 => niche de détection RÉELLE, atteinte
par un attaquant qui optimise (là où le bricolage affine échouait).
"""
from __future__ import annotations
import numpy as np
import torch
from ripser import ripser
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import squareform, pdist

torch.manual_seed(1); np.random.seed(1)
N, K = 160, 8
GRID = np.linspace(0.0, 3.0, 60)


def h1_land(P):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(GRID - b, d - GRID), 0, None))
    return lam


def knn_sorted_t(P, k=K):
    D = torch.cdist(P, P) + torch.eye(len(P)) * 1e9
    vals, _ = torch.topk(D, k, dim=1, largest=False)
    return torch.sort(vals.reshape(-1)).values


def mardia_t(P):
    mu = P.mean(0); Pc = P - mu
    C = (Pc.T @ Pc) / (len(P) - 1) + 1e-6 * torch.eye(P.shape[1])
    m = torch.einsum("ij,jk,ik->i", Pc, torch.inverse(C), Pc)
    return (m ** 2).mean()


def knn_density_np(P, k=5):
    d = squareform(pdist(P, "euclidean")); d.sort(axis=1)
    return float(np.mean(d[:, k]))

def mardia_np(P):
    mu = P.mean(0); Pc = P - mu
    C = np.cov(P, rowvar=False) + 1e-6 * np.eye(P.shape[1])
    m = np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)
    return float(np.mean(m ** 2))


def circle(seed, n=N, r=1.0, noise=0.02):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    th = rng.uniform(0, 2*np.pi)
    P = np.c_[r*np.cos(t), r*np.sin(t)] @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (P + rng.normal(0, noise, (n, 2))).astype(np.float64)


def broken(seed, n=N, r=1.0, gap=0.9, noise=0.02):
    rng = np.random.default_rng(seed + 999)
    t = np.linspace(gap, 2*np.pi - gap, n)
    return (np.c_[r*np.cos(t), r*np.sin(t)] + rng.normal(0, noise, (n, 2))).astype(np.float64)


def optimize_attack(A, iters=1500):
    muA = torch.tensor(A.mean(0)); CA = torch.tensor(np.cov(A, rowvar=False))
    knnA = knn_sorted_t(torch.tensor(A)).detach()
    kurtA = mardia_t(torch.tensor(A)).detach()
    B = torch.tensor(broken(hash(A.tobytes()) % 10000), requires_grad=True)
    opt = torch.optim.Adam([B], lr=5e-3)
    for _ in range(iters):
        opt.zero_grad()
        mu = B.mean(0); Pc = B - mu; C = (Pc.T @ Pc) / (len(B) - 1)
        loss = ((C - CA)**2).sum() + ((mu - muA)**2).sum() \
            + 5*((knn_sorted_t(B) - knnA)**2).mean() + 0.1*(mardia_t(B) - kurtA)**2
        loss.backward(); opt.step()
    return B.detach().numpy()


def main():
    M = 16
    healthy = [circle(s) for s in range(M)]
    attacks = [optimize_attack(h) for h in healthy]

    # appariement moyen
    dC = np.mean([np.linalg.norm(np.cov(a, rowvar=False) - np.cov(h, rowvar=False), "fro")
                  for a, h in zip(attacks, healthy)])
    dK = np.mean([abs(mardia_np(a) - mardia_np(h)) for a, h in zip(attacks, healthy)])
    print(f"appariement moyen : |Δcov|={dC:.4f}  |Δkurtosis|={dK:.3f}")
    print(f"H1 sain moy={np.mean([np.max(h1_land(h)) for h in healthy]):.3f}  "
          f"H1 attaqué moy={np.mean([np.max(h1_land(a)) for a in attacks]):.3f}")

    ref_land = np.mean([h1_land(h) for h in healthy], axis=0)

    def score(P):
        return {
            "Δ_top": float(np.sum(np.abs(h1_land(P) - ref_land))),
            "B6_spec": float(np.linalg.norm(np.cov(P, rowvar=False) - np.cov(healthy[0], rowvar=False), "fro")),
            "B3_kNN": knn_density_np(P),
            "Kurt_Mardia": mardia_np(P),
        }

    sh = [score(h) for h in healthy]; sa = [score(a) for a in attacks]
    print(f"\n=== AUC sain vs attaque OPTIMISÉE (M={M}) ===")
    for det in sh[0]:
        yv = [0]*M + [1]*M
        sv = [s[det] for s in sh] + [s[det] for s in sa]
        auc = roc_auc_score(yv, sv)
        print(f"  {det:>12} : AUC={auc:.3f}  (bilatéral {max(auc,1-auc):.3f})")

    print("\nVERDICT : Δ_top≈1 & B6/kNN/kurtosis bilatéral≈0.5 => NICHE RÉELLE par optimisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
