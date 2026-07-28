"""FRONT 1 — le seuil de compromis dépend-il de la dimension homologique ?

On mesure la courbe pour H0 (connexité : 2 clusters) ET H1 (boucle), sur les mêmes
niveaux de contrainte, et on trace la figure. Question : matcher quels moments force
la topologie à réapparaître, et est-ce le même seuil pour H0 et H1 ?
"""
from __future__ import annotations
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.spatial.distance import squareform, pdist
from exp3c_tradeoff import knn_sorted, mardia, dist_sorted, W, LEVELS

torch.manual_seed(0); np.random.seed(0)
N, K = 150, 8


def topo_signal(P, dim):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=max(1, dim), distance_matrix=True)["dgms"][dim]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    return float(np.max(fin[:, 1] - fin[:, 0])) if len(fin) else 0.0


# --- H1 : cercle (cible) vs arc cassé (init) ---
def circle(n=N, r=1.0, noise=0.02):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return (np.c_[r*np.cos(t), r*np.sin(t)] + np.random.normal(0, noise, (n, 2))).astype(np.float64)
def arc(n=N, r=1.0, gap=1.0, noise=0.02):
    t = np.linspace(gap, 2*np.pi - gap, n)
    return (np.c_[r*np.cos(t), r*np.sin(t)] + np.random.normal(0, noise, (n, 2))).astype(np.float64)

# --- H0 : deux clusters (cible) vs un blob (init) ---
def two_clusters(n=N, s=1.6, w=0.22):
    h = n//2
    a = np.random.normal(0, w, (h, 2)); a[:, 0] -= s
    b = np.random.normal(0, w, (n-h, 2)); b[:, 0] += s
    return np.vstack([a, b]).astype(np.float64)
def one_blob(n=N):
    # blob allongé, couvrant ~ la même étendue mais CONNEXE (H0=1)
    P = np.random.normal(0, 1.0, (n, 2)); P[:, 0] *= 1.6
    return P.astype(np.float64)


def penalties(B, A):
    muA = torch.tensor(A.mean(0)); Ac = torch.tensor(A) - muA
    CA = (Ac.T @ Ac)/(len(A)-1)
    mu = B.mean(0); Bc = B - mu; C = (Bc.T @ Bc)/(len(B)-1)
    return {
        "mean": ((mu - muA)**2).sum(), "cov": ((C - CA)**2).sum(),
        "knn": ((knn_sorted(B) - knn_sorted(torch.tensor(A)).detach())**2).mean(),
        "kurt": (mardia(B) - mardia(torch.tensor(A)).detach())**2,
        "dist": ((dist_sorted(B) - dist_sorted(torch.tensor(A)).detach())**2).mean(),
    }


def curve_for(A, init_fn, dim, iters=2500):
    hstar = topo_signal(A, dim)
    ratios = []
    for _, active in LEVELS:
        B = torch.tensor(init_fn(), requires_grad=True)
        opt = torch.optim.Adam([B], lr=5e-3)
        for _ in range(iters):
            opt.zero_grad()
            p = penalties(B, A)
            loss = sum(W[k]*p[k] for k in active)
            loss.backward(); opt.step()
        ratios.append(topo_signal(B.detach().numpy(), dim) / hstar)
    return ratios


def main():
    names = [n for n, _ in LEVELS]
    print("calcul courbe H1 (boucle)…")
    h1 = curve_for(circle(), arc, dim=1)
    print("calcul courbe H0 (connexité)…")
    h0 = curve_for(two_clusters(), one_blob, dim=0)

    print(f"\n{'niveau':>12} {'H0/h*':>8} {'H1/h*':>8}")
    for nm, a, b in zip(names, h0, h1):
        print(f"{nm:>12} {a:>8.2f} {b:>8.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(names))
    ax.plot(x, h0, "o-", lw=2, label="H₀ (connexité : 2 clusters)")
    ax.plot(x, h1, "s-", lw=2, label="H₁ (boucle)")
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("topologie récupérée  (signal / cible)")
    ax.set_xlabel("statistiques appariées (jeu croissant →)")
    ax.set_title("Borne de compromis : matcher quels moments force la topologie ?")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    out = "tradeoff_curve.png"
    fig.savefig(out, dpi=140)
    print(f"\nfigure -> exp0/{out}")
    print("Lecture : le niveau où chaque courbe franchit ~0.5 = seuil de cette dimension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
