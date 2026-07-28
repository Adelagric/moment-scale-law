"""PAPER 2 — frontière : l'ordre de moment N* qui épingle la topologie croît-il avec la
complexité topologique (nombre de cycles) ?

Hypothèse : pour k boucles, il faut apparier des moments d'ordre croissant pour forcer la
topologie. Si N*(k) croît, alors la persistance = substitut à calcul borné d'un moment
d'ordre non borné, et son "avantage" croît avec les nombres de Betti.

Protocole : cible = k anneaux (H_1 = k) ; init = k disques pleins (H_1 = 0), densité locale
appariée. On apparie {μ ; +Σ ; +kurtosis (4) ; +moment 6 ; +distances} et on lit la
persistance H_1 totale récupérée.
"""
from __future__ import annotations
import numpy as np
import torch
from ripser import ripser
from scipy.spatial.distance import squareform, pdist
from exp3c_tradeoff import knn_sorted, dist_sorted

torch.manual_seed(0); np.random.seed(0)
DIM = 4
PTS_PER_LOOP = 90


def h1_total(P, k):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    life = np.sort(fin[:, 1] - fin[:, 0])[::-1] if len(fin) else np.array([0.0])
    return float(np.sum(life[:k]))  # somme des k plus longues barres H_1


def centers(k):
    return [np.array([3.0*i, 0, 0, 0]) for i in range(k)]

def rings(k, r=0.8, noise=0.04):
    out = []
    for c in centers(k):
        t = np.linspace(0, 2*np.pi, PTS_PER_LOOP, endpoint=False)
        P = np.zeros((PTS_PER_LOOP, DIM)); P[:, 0] = r*np.cos(t); P[:, 1] = r*np.sin(t)
        out.append(P + c + np.random.normal(0, noise, P.shape))
    return np.vstack(out).astype(np.float64)

def disks(k, r=0.8, noise=0.04, gap=1.0):
    # init = k ARCS cassés (H_1=0), init convergent (cf. exp3c) plutôt que disques pleins
    out = []
    for c in centers(k):
        t = np.linspace(gap, 2*np.pi - gap, PTS_PER_LOOP)
        P = np.zeros((PTS_PER_LOOP, DIM)); P[:, 0] = r*np.cos(t); P[:, 1] = r*np.sin(t)
        out.append(P + c + np.random.normal(0, noise, P.shape))
    return np.vstack(out).astype(np.float64)


def mardia_p(B, p):
    mu = B.mean(0); Bc = B - mu
    C = (Bc.T @ Bc)/(len(B)-1) + 1e-6*torch.eye(B.shape[1])
    m = torch.einsum("ij,jk,ik->i", Bc, torch.inverse(C), Bc)
    return (m ** p).mean()

def penalties(B, A):
    muA = torch.tensor(A.mean(0)); Ac = torch.tensor(A)-muA; CA = (Ac.T@Ac)/(len(A)-1)
    mu = B.mean(0); Bc = B-mu; C = (Bc.T@Bc)/(len(B)-1)
    At = torch.tensor(A)
    return {"mean": ((mu-muA)**2).sum(), "cov": ((C-CA)**2).sum(),
            "kurt": (mardia_p(B,2)-mardia_p(At,2).detach())**2,
            "m6": (mardia_p(B,3)-mardia_p(At,3).detach())**2,
            "dist": ((dist_sorted(B)-dist_sorted(At).detach())**2).mean()}

LEVELS = [("μ",["mean"]), ("μ,Σ",["mean","cov"]), ("+kurt(4)",["mean","cov","kurt"]),
          ("+m6",["mean","cov","kurt","m6"]), ("+dist",["mean","cov","kurt","m6","dist"])]
Wt = {"mean":1.,"cov":1.,"kurt":0.2,"m6":0.05,"dist":5.}


def curve(k, iters=2500):
    A = rings(k); hstar = h1_total(A, k); out = []
    for _, active in LEVELS:
        B = torch.tensor(disks(k).copy(), requires_grad=True)
        opt = torch.optim.Adam([B], lr=5e-3)
        for _ in range(iters):
            opt.zero_grad(); p = penalties(B, A)
            (sum(Wt[a]*p[a] for a in active)).backward(); opt.step()
        out.append(h1_total(B.detach().numpy(), k)/(hstar+1e-9))
    return out


def main():
    names = [n for n,_ in LEVELS]
    print(f"{'k boucles':>10} " + " ".join(f"{n:>9}" for n in names) + "   N* (seuil ≥0.5)")
    for k in (1, 2, 3):
        c = curve(k)
        star = next((n for n,r in zip(names,c) if r>=0.5), "aucun")
        print(f"{k:>10} " + " ".join(f"{r:>9.2f}" for r in c) + f"   -> {star}")
    print("\nSi N* monte avec k => la persistance capte en O(1) ce qui exigerait un moment")
    print("d'ordre croissant avec les nombres de Betti. C'est l'énoncé de frontière.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
