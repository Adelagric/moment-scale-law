"""FRONT 1 — éprouver la conjecture d'impossibilité (tenter de la RÉFUTER).

Conjecture (forme faible) : impossible de matcher covariance ET distribution de densité
locale (kNN) entre deux nuages sans matcher leur H1.

Test réfutatif : partir d'une boucle CASSÉE (H1 faible), optimiser ses positions par
gradient pour matcher (μ, Σ, distribution kNN complète) d'une boucle INTACTE, SANS aucun
terme sur H1. Puis mesurer :
  - qualité de l'appariement (μ, Σ, kNN)  -> peut-on matcher ?
  - H1 du résultat vs cible                -> H1 se reforme-t-il ?

Si on matche bien ET H1 reste cassé => CONTRE-EXEMPLE : conjecture forte FAUSSE.
Si matcher force H1 à se reformer        => conjecture SOUTENUE.
"""
from __future__ import annotations
import numpy as np
import torch
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
N, K = 160, 8


def h1_max(P):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    return float(np.max(fin[:, 1] - fin[:, 0])) if len(fin) else 0.0


def knn_sorted(P, k=K):
    """Signature de densité locale : distances aux 1..k voisins, triées globalement."""
    D = torch.cdist(P, P)
    D = D + torch.eye(len(P), device=P.device) * 1e9
    vals, _ = torch.topk(D, k, dim=1, largest=False)   # (n, k)
    return torch.sort(vals.reshape(-1)).values          # vecteur trié n*k


def cov_mu(P):
    mu = P.mean(0)
    Pc = P - mu
    C = (Pc.T @ Pc) / (len(P) - 1)
    return mu, C


def target_circle(n=N, r=1.0, noise=0.02):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.c_[r * np.cos(t), r * np.sin(t)] + np.random.normal(0, noise, (n, 2))
    return P.astype(np.float64)


def broken_init(n=N, r=1.0, gap=0.9, noise=0.02):
    """Boucle avec un secteur manquant -> arc (H1 cassé), points redistribués sur l'arc."""
    t = np.linspace(gap, 2 * np.pi - gap, n)
    P = np.c_[r * np.cos(t), r * np.sin(t)] + np.random.normal(0, noise, (n, 2))
    return P.astype(np.float64)


def main():
    A = target_circle()
    muA_np, CA_np = A.mean(0), np.cov(A, rowvar=False)
    knnA = knn_sorted(torch.tensor(A)).detach()
    h1A = h1_max(A)

    B = torch.tensor(broken_init(), requires_grad=True)
    muA = torch.tensor(muA_np); CA = torch.tensor(CA_np)
    opt = torch.optim.Adam([B], lr=5e-3)

    print(f"cible H1={h1A:.3f} | init cassé H1={h1_max(B.detach().numpy()):.3f}")
    for it in range(4000):
        opt.zero_grad()
        mu, C = cov_mu(B)
        knnB = knn_sorted(B)
        loss = ((C - CA) ** 2).sum() + ((mu - muA) ** 2).sum() + ((knnB - knnA) ** 2).mean() * 5
        loss.backward(); opt.step()
        if it % 1000 == 0 or it == 3999:
            Bn = B.detach().numpy()
            dC = np.linalg.norm(np.cov(Bn, rowvar=False) - CA_np, "fro")
            dknn = float(((knn_sorted(B).detach() - knnA) ** 2).mean().sqrt())
            print(f"it={it:4d} loss={loss.item():.4f} |Δcov|={dC:.4f} "
                  f"RMSE_kNN={dknn:.4f} H1(B)={h1_max(Bn):.3f}")

    Bn = B.detach().numpy()
    dC = np.linalg.norm(np.cov(Bn, rowvar=False) - CA_np, "fro")
    dknn = float(((knn_sorted(B).detach() - knnA) ** 2).mean().sqrt())
    h1B = h1_max(Bn)
    print("\n=== RÉSULTAT ===")
    print(f"appariement : |Δcov|={dC:.4f}  RMSE_kNN={dknn:.4f}  (petits = bien apparié)")
    print(f"H1 cible={h1A:.3f}   H1 obtenu={h1B:.3f}   ratio={h1B/h1A:.2f}")
    matched = dC < 0.05 and dknn < 0.05
    broke = h1B < 0.5 * h1A
    if matched and broke:
        print("=> CONTRE-EXEMPLE : matché (μ,Σ,kNN) mais H1 cassé. Conjecture forte FAUSSE.")
    elif matched and not broke:
        print("=> H1 s'est REFORMÉ en matchant. Conjecture SOUTENUE (forme faible).")
    else:
        print("=> appariement non atteint : test non concluant, ajuster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
