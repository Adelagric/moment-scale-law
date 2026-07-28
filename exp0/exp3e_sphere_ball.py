"""FRONT 1 (point 1) — loi seuil vs dimension homologique, famille sphère/boule.

k-sphère (creuse, a H_k) vs (k+1)-boule (pleine, pas de H_k), pour k=0,1,2.
Init = version PLEINE (pas de trou). On apparie un jeu croissant de stats de la version
CREUSE, et on mesure si le trou H_k est forcé. Seuil = niveau où signal/cible franchit 0.5.

Descripteurs propres :
  H0 : 2e plus longue barre finie (échelle 2->1 composantes)
  H1 : plus longue barre H1
  H2 : plus longue barre H2
"""
from __future__ import annotations
import numpy as np
import torch
from ripser import ripser
from scipy.spatial.distance import squareform, pdist
from exp3c_tradeoff import knn_sorted, mardia, dist_sorted, W, LEVELS

torch.manual_seed(0); np.random.seed(0)


def topo_signal(P, dim):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=max(1, dim), distance_matrix=True)["dgms"][dim]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    life = np.sort(fin[:, 1] - fin[:, 0])[::-1] if len(fin) else np.array([0.0])
    # H0 : la plus longue barre finie = échelle de fusion des 2 amas (0-sphère creuse)
    return float(life[0]) if len(life) else 0.0


def sphere(k, n, noise=0.03):
    """Échantillon sur la k-sphère unité (dans R^{k+1})."""
    X = np.random.normal(size=(n, k + 1))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return (X + np.random.normal(0, noise, X.shape)).astype(np.float64)

def ball(k, n):
    """Échantillon plein dans la (k+1)-boule unité."""
    X = np.random.normal(size=(n, k + 1))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = np.random.uniform(0, 1, (n, 1)) ** (1.0 / (k + 1))
    return (X * r).astype(np.float64)

def two_clusters(n, s=1.3, w=0.12):     # 0-sphère épaissie (2 amas ±1.3) pour H0
    h = n // 2
    a = np.random.normal(0, w, (h, 1)); a -= s
    b = np.random.normal(0, w, (n - h, 1)); b += s
    return np.vstack([a, b]).astype(np.float64)

def one_blob(n, r=1.3):                  # 1-boule pleine = segment uniforme [-r, r]
    return (np.random.uniform(-r, r, (n, 1))).astype(np.float64)


def penalties(B, A):
    muA = torch.tensor(A.mean(0)); Ac = torch.tensor(A) - muA
    CA = (Ac.T @ Ac) / (len(A) - 1)
    mu = B.mean(0); Bc = B - mu; C = (Bc.T @ Bc) / (len(B) - 1)
    return {
        "mean": ((mu - muA) ** 2).sum(), "cov": ((C - CA) ** 2).sum(),
        "knn": ((knn_sorted(B) - knn_sorted(torch.tensor(A)).detach()) ** 2).mean(),
        "kurt": (mardia(B) - mardia(torch.tensor(A)).detach()) ** 2,
        "dist": ((dist_sorted(B) - dist_sorted(torch.tensor(A)).detach()) ** 2).mean(),
    }


def curve(A, init, dim, iters=2000):
    hstar = topo_signal(A, dim)
    out = []
    for _, active in LEVELS:
        B = torch.tensor(init.copy(), requires_grad=True)
        opt = torch.optim.Adam([B], lr=5e-3)
        for _ in range(iters):
            opt.zero_grad()
            p = penalties(B, A)
            loss = sum(W[k] * p[k] for k in active)
            loss.backward(); opt.step()
        out.append(topo_signal(B.detach().numpy(), dim) / (hstar + 1e-9))
    return out, hstar


def threshold(names, ratios):
    for nm, r in zip(names, ratios):
        if r >= 0.5:
            return nm
    return "aucun (<0.5 partout)"


def main():
    names = [n for n, _ in LEVELS]
    specs = [
        ("H0 amas écartés ±1.3",  two_clusters(150, s=1.3), one_blob(150), 0),
        ("H0 amas var-appariée",  two_clusters(150, s=0.75), one_blob(150), 0),
        ("H1 (cercle → disque)",  sphere(1, 150),    ball(1, 150),  1),
        ("H2 (sphère → boule)",   sphere(2, 110),    ball(2, 110),  2),
    ]
    print(f"{'':>22} " + " ".join(f"{n:>10}" for n in names))
    results = []
    for label, A, init, dim in specs:
        ratios, hstar = curve(A, init, dim)
        results.append((label, ratios))
        print(f"{label:>22} " + " ".join(f"{r:>10.2f}" for r in ratios) + f"   (h*={hstar:.2f})")

    print("\n=== SEUIL par dimension homologique (1er niveau où signal ≥ 0.5) ===")
    for label, ratios in results:
        print(f"  {label:>22} : {threshold(names, ratios)}")
    print("\nSi tous au même niveau (kurtosis) => seuil gouverné par la GÉOMÉTRIE")
    print("(creux/plein = variance du rayon² = 4ᵉ moment), pas par la dimension k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
