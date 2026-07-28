"""PAPER 2 — le lemme moments→Betti (borne SUP) est-il seulement VRAI ?

Attaque : la borne inf utilise des sphères portant une masse O(1). Mais une topologie peut
vivre à une ÉCHELLE MINUSCULE : un anneau de rayon ε portant une fraction f de la masse
perturbe les moments de O(f·ε^k) — arbitrairement petit. Si b features minuscules survivent à
BIEN PLUS que 4b moments appariés, alors N* dépend de l'échelle/masse, pas seulement de b,
et la borne sup N* ≤ C_d·b est FAUSSE telle qu'énoncée.

Test : nuage = gros blob + petit anneau (rayon ε, masse f). Comparé au MÊME blob + petit disque
plein (même ε, même f). Topologie diffère (H_1 = 1 vs 0 à l'échelle ε). On mesure de combien
les moments diffèrent quand ε -> 0.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

RNG = np.random.default_rng(0)


def cloud(eps, f, filled, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    k = int(f*n)
    blob = rng.normal(0, 1.0, (n-k, 2))
    t = rng.uniform(0, 2*np.pi, k)
    r = eps*np.sqrt(rng.uniform(0, 1, k)) if filled else np.full(k, eps)
    ring = np.c_[r*np.cos(t), r*np.sin(t)] + np.array([3.0, 0.0])  # loin du blob
    ring += rng.normal(0, eps*0.04, ring.shape)
    return np.vstack([blob, ring])


def moments_upto(P, N):
    """Tous les moments (mixtes) jusqu'à l'ordre total N."""
    out = []
    for tot in range(1, N+1):
        for i in range(tot+1):
            out.append(np.mean(P[:, 0]**i * P[:, 1]**(tot-i)))
    return np.array(out)


def h1_at_scale(P, eps):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True, thresh=4*eps)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    life = fin[:, 1]-fin[:, 0] if len(fin) else np.array([])
    return float(np.max(life)) if len(life) else 0.0


def main():
    f = 0.12
    print(f"{'ε':>7} {'max |Δmoment| (ordre≤12)':>26} {'H1 anneau':>10} {'H1 disque':>10}")
    for eps in (0.5, 0.2, 0.05, 0.01):
        A = cloud(eps, f, filled=False, seed=1)
        B = cloud(eps, f, filled=True, seed=1)
        mA, mB = moments_upto(A, 12), moments_upto(B, 12)
        dm_ = np.max(np.abs(mA-mB)/(np.abs(mB)+1e-12))  # écart RELATIF (moments non normalisés sinon)
        print(f"{eps:>7} {dm_:>26.2e} {h1_at_scale(A,eps):>10.3f} {h1_at_scale(B,eps):>10.3f}")
    print("\nLecture : si |Δmoment| -> 0 quand ε -> 0 ALORS QUE H1 reste tranché (anneau≫disque),")
    print("des features à petite échelle échappent à TOUS les moments d'ordre fixe :")
    print("=> N* dépend de l'ÉCHELLE, pas seulement de b. Borne sup N* ≤ C_d·b FAUSSE telle quelle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
