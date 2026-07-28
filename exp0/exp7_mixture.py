"""PAPER 2 — brique 2 : généralisation au MÉLANGE de composantes elliptiques.

Modèle réaliste des embeddings (≈ mélange gaussien anisotrope). On mesure la courbe de
compromis pour deux features :
  (a) H_0 (connexité inter-composantes) : conjecture -> seuil au 2ᵉ moment (la séparation
      des moyennes ajoute de la variance inter-composantes, visible en covariance).
  (b) H_1 (trou DANS une composante) : conjecture -> seuil au 4ᵉ moment (théorème
      elliptique par composante).

Protocole : init = topologie détruite (composantes fusionnées / trou bouché), on apparie
un jeu croissant de stats de la cible, on lit le signal topologique final.
Loi attendue : le type de feature fixe l'ordre du moment, indépendamment du reste.
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
DIM = 4


def topo_signal(P, dim):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=max(1, dim), distance_matrix=True)["dgms"][dim]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    life = np.sort(fin[:, 1] - fin[:, 0])[::-1] if len(fin) else np.array([0.0])
    if dim == 0:  # signal "2 amas" = PROMINENCE de la fusion finale (gap du code-barres)
        return float(life[0] - life[1]) if len(life) > 1 else 0.0
    return float(life[0]) if len(life) else 0.0  # H1/H2 : plus longue barre (trou)


def spd(d, seed):
    r = np.random.default_rng(seed); M = r.normal(size=(d, d))
    return M @ M.T + 0.4 * np.eye(d)


# ---- (a) mélange H0 : 2 composantes elliptiques séparées vs 1 blob ----
M0 = spd(DIM, 1)
def mix_two(n=200, s=2.4):
    h = n // 2
    a = np.random.normal(size=(h, DIM)) * 0.35 @ M0
    b = np.random.normal(size=(n - h, DIM)) * 0.35 @ M0
    a[:, 0] -= s; b[:, 0] += s
    return a.astype(np.float64), b, np.vstack([a, b])
def mix_one(n=200):  # amas serré unique (faible variance, pas de gap) = init
    return (np.random.normal(size=(n, DIM)) * 0.35 @ M0).astype(np.float64)

# ---- (b) mélange H1 : une composante en anneau parmi des blobs vs anneau bouché ----
M1 = spd(DIM, 2)
def ring_mixture(n=240, filled=False):
    per = n // 6  # anneau robuste (2/3 des points), blobs plus légers
    blobs = [np.random.normal(size=(per, DIM)) * 0.3 + np.array([c, 0, 0, 0]) for c in (-3.5, 3.5)]
    t = np.random.uniform(0, 2*np.pi, n - 2*per)
    ring = np.zeros((len(t), DIM))
    if filled:
        rad = np.sqrt(np.random.uniform(0, 2, len(t)))
        ring[:, 0], ring[:, 1] = rad*np.cos(t), rad*np.sin(t)
    else:
        ring[:, 0], ring[:, 1] = np.cos(t), np.sin(t)
    ring += np.random.normal(0, 0.05, ring.shape)
    return np.vstack(blobs + [ring @ M1]).astype(np.float64)


def penalties(B, A):
    muA = torch.tensor(A.mean(0)); Ac = torch.tensor(A) - muA
    CA = (Ac.T @ Ac) / (len(A) - 1)
    mu = B.mean(0); Bc = B - mu; C = (Bc.T @ Bc) / (len(B) - 1)
    return {"mean": ((mu - muA)**2).sum(), "cov": ((C - CA)**2).sum(),
            "knn": ((knn_sorted(B) - knn_sorted(torch.tensor(A)).detach())**2).mean(),
            "kurt": (mardia(B) - mardia(torch.tensor(A)).detach())**2,
            "dist": ((dist_sorted(B) - dist_sorted(torch.tensor(A)).detach())**2).mean()}


def curve(A, init, dim, iters=2200):
    hstar = topo_signal(A, dim); out = []
    for _, active in LEVELS:
        B = torch.tensor(init.copy(), requires_grad=True)
        opt = torch.optim.Adam([B], lr=5e-3)
        for _ in range(iters):
            opt.zero_grad(); p = penalties(B, A)
            (sum(W[k]*p[k] for k in active)).backward(); opt.step()
        out.append(topo_signal(B.detach().numpy(), dim) / (hstar + 1e-9))
    return out


def main():
    names = [n for n, _ in LEVELS]
    _, _, A0 = mix_two(); h0 = curve(A0, mix_one(), dim=0)
    A1 = ring_mixture(filled=False); h1 = curve(A1, ring_mixture(filled=True), dim=1)

    print(f"{'niveau':>12} {'H0 mélange':>11} {'H1 dans compo':>14}")
    for nm, a, b in zip(names, h0, h1):
        print(f"{nm:>12} {a:>11.2f} {b:>14.2f}")
    def thr(c):
        for nm, r in zip(names, c):
            if r >= 0.5: return nm
        return "aucun"
    print(f"\nseuil H0 (connexité mélange) : {thr(h0)}   [conjecture : covariance]")
    print(f"seuil H1 (trou dans composante) : {thr(h1)}   [conjecture : kurtosis]")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = range(len(names))
    ax.plot(x, h0, "o-", lw=2, label=r"$H_0$ (mixture connectivity)")
    ax.plot(x, h1, "s-", lw=2, label=r"$H_1$ (hole within a component)")
    ax.axhline(0.5, ls=":", c="grey"); ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("recovered topology (signal / target)")
    ax.set_xlabel("matched statistics (growing set)")
    ax.set_title("Mixture of elliptical components: feature type sets the moment order")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("../docs/paper1/fig_mixture.png", dpi=150)
    print("\nfigure -> docs/paper1/fig_mixture.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
