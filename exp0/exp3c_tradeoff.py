"""FRONT 1 rebaptisé — la COURBE DE COMPROMIS.

Combien de topologie peut-on cacher en appariant un jeu croissant de statistiques ?
Protocole : init B = boucle CASSÉE (H1≈0). Pour chaque niveau de contrainte, on optimise
B pour matcher les stats de la cible A (un cercle, H1=h*), et on mesure le H1 FINAL de B.
  - H1(B) reste bas  => la topologie est cachable sous ces stats (attaque invisible).
  - H1(B) remonte vers h* => matcher ces stats FORCE la boucle : topologie non cachable.

L'emplacement du saut = le résultat (borne de compromis empirique).
Niveaux : {μ} ⊂ {μ,Σ} ⊂ {+kNN} ⊂ {+kurtosis} ⊂ {+distribution des distances}.
"""
from __future__ import annotations
import numpy as np
import torch
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

torch.manual_seed(0); np.random.seed(0)
N, K = 150, 8


def h1_max(P):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    return float(np.max(fin[:, 1] - fin[:, 0])) if len(fin) else 0.0


def circle(n=N, r=1.0, noise=0.02):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return (np.c_[r*np.cos(t), r*np.sin(t)] + np.random.normal(0, noise, (n, 2))).astype(np.float64)

def broken(n=N, r=1.0, gap=1.0, noise=0.02):
    t = np.linspace(gap, 2*np.pi - gap, n)
    return (np.c_[r*np.cos(t), r*np.sin(t)] + np.random.normal(0, noise, (n, 2))).astype(np.float64)


def knn_sorted(P, k=K):
    D = torch.cdist(P, P) + torch.eye(len(P)) * 1e9
    v, _ = torch.topk(D, k, dim=1, largest=False)
    return torch.sort(v.reshape(-1)).values

def mardia(P):
    mu = P.mean(0); Pc = P - mu
    C = (Pc.T @ Pc)/(len(P)-1) + 1e-6*torch.eye(P.shape[1])
    return (torch.einsum("ij,jk,ik->i", Pc, torch.inverse(C), Pc)**2).mean()

def dist_sorted(P):
    D = torch.cdist(P, P)
    iu = torch.triu_indices(len(P), len(P), 1)
    return torch.sort(D[iu[0], iu[1]]).values


def penalties(B, A):
    muA = torch.tensor(A.mean(0)); Ac = torch.tensor(A) - muA
    CA = (Ac.T @ Ac)/(len(A)-1)
    mu = B.mean(0); Bc = B - mu; C = (Bc.T @ Bc)/(len(B)-1)
    return {
        "mean": ((mu - muA)**2).sum(),
        "cov":  ((C - CA)**2).sum(),
        "knn":  ((knn_sorted(B) - knn_sorted(torch.tensor(A)).detach())**2).mean(),
        "kurt": (mardia(B) - mardia(torch.tensor(A)).detach())**2,
        "dist": ((dist_sorted(B) - dist_sorted(torch.tensor(A)).detach())**2).mean(),
    }

LEVELS = [
    ("μ",                    ["mean"]),
    ("μ,Σ",                  ["mean", "cov"]),
    ("+kNN",                 ["mean", "cov", "knn"]),
    ("+kurtosis",            ["mean", "cov", "knn", "kurt"]),
    ("+distances",           ["mean", "cov", "knn", "kurt", "dist"]),
]
W = {"mean": 1.0, "cov": 1.0, "knn": 5.0, "kurt": 0.1, "dist": 5.0}


def match(A, active, iters=2500):
    B = torch.tensor(broken(), requires_grad=True)
    opt = torch.optim.Adam([B], lr=5e-3)
    for _ in range(iters):
        opt.zero_grad()
        p = penalties(B, A)
        loss = sum(W[k]*p[k] for k in active)
        loss.backward(); opt.step()
    Bn = B.detach().numpy()
    res = {k: float(penalties(B, A)[k].detach()) for k in active}
    return Bn, res


def main():
    A = circle()
    hstar = h1_max(A)
    print(f"cible : cercle H1 = {hstar:.3f}   |   init cassé H1 = {h1_max(broken()):.3f}\n")
    print(f"{'niveau':>14} {'H1(B) final':>12} {'H1/h*':>7}  résidus")
    curve = []
    for name, active in LEVELS:
        Bn, res = match(A, active)
        h1 = h1_max(Bn)
        curve.append((name, h1 / hstar))
        rtxt = " ".join(f"{k}={v:.1e}" for k, v in res.items())
        print(f"{name:>14} {h1:>12.3f} {h1/hstar:>7.2f}  {rtxt}")

    print("\n=== COURBE DE COMPROMIS (topologie cachable = H1 bas) ===")
    for name, ratio in curve:
        bar = "█" * int(ratio * 40)
        print(f"  {name:>14} |{bar} {ratio:.2f}")
    print("\nLecture : le niveau où H1/h* saute vers 1 = la borne. En-deçà, la topologie")
    print("est cachable sous les stats appariées ; au-delà, l'appariement la force.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
