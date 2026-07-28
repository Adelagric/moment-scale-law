"""PAPER 2 — frontière, construction À LA MAIN (pas d'optimisation) via quadrature de Gauss.

Théorème visé : N*(m) = 4m. La quadrature de Gauss à m points d'une mesure radiale apparie
ses moments jusqu'à l'ordre 2m-1 (ordre nuage 4m-2). Les m nœuds => m coquilles concentriques
(topologie = m boucles) qui matchent une boule (topologie triviale) jusqu'au moment 4m-2.
=> l'ordre de moment pour épingler la topologie croît LINÉAIREMENT avec la complexité (m).

On vérifie, pour m=1,2,3 (d=2, cercles concentriques) :
  (a) les moments radiaux E[R^{2j}] coïncident pour j=1..2m-1 (=> moments nuage jusqu'à 4m-2) ;
  (b) la topologie diffère : m boucles H_1 pour les coquilles, ~0 pour la boule.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

np.random.seed(0)
NPTS = 900


def gauss01(m):
    """Nœuds/poids de Gauss-Legendre sur [0,1] (apparient les moments de Uniform(0,1)
    jusqu'à l'ordre 2m-1)."""
    x, w = leggauss(m)               # sur [-1,1]
    return 0.5*(x+1), 0.5*w          # -> [0,1], poids sommant à 1


def shells_cloud(m, noise=0.02):
    """m cercles concentriques aux rayons sqrt(nœuds de Gauss). Radial : m atomes."""
    u, w = gauss01(m)
    pts = []
    for ui, wi in zip(u, w):
        n = max(30, int(round(wi*NPTS)))
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        r = np.sqrt(ui)
        P = np.c_[r*np.cos(t), r*np.sin(t)] + np.random.normal(0, noise, (n, 2))
        pts.append(P)
    return np.vstack(pts)


def ball_cloud(noise=0.02):
    """Radial : R^2 ~ Uniform(0,1) (mesure hole-free dont les coquilles sont la quadrature)."""
    u = np.random.uniform(0, 1, NPTS); r = np.sqrt(u)
    t = np.random.uniform(0, 2*np.pi, NPTS)
    return np.c_[r*np.cos(t), r*np.sin(t)] + np.random.normal(0, noise, (NPTS, 2))


def radial_moments(P, K):
    R2 = (P**2).sum(1)
    return np.array([np.mean(R2**j) for j in range(1, K+1)])


def n_loops(P, thresh=0.15):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    life = fin[:, 1]-fin[:, 0] if len(fin) else np.array([])
    return int(np.sum(life > thresh)), (float(np.max(life)) if len(life) else 0.0)


def main():
    print(f"{'m':>3} {'ordre nuage apparié 4m-2':>24} {'max |Δ moment radial| j≤2m-1':>30} "
          f"{'H1 coquilles':>13} {'H1 boule':>9}")
    for m in (1, 2, 3):
        S = shells_cloud(m); B = ball_cloud()
        K = 2*m-1
        mS, mB = radial_moments(S, K), radial_moments(B, K)
        dmom = np.max(np.abs(mS-mB)/(np.abs(mB)+1e-9))   # écart relatif moments radiaux
        lS, _ = n_loops(S); lB, _ = n_loops(B)
        print(f"{m:>3} {4*m-2:>24} {dmom:>30.3f} {lS:>13} {lB:>9}")
    print("\nLecture : moments radiaux appariés (Δ petit) jusqu'à l'ordre nuage 4m-2, MAIS")
    print("H1 = m (coquilles) vs ~0 (boule). Donc N*(m) >= 4m-2, croissant linéairement en m :")
    print("la persistance capte en O(1) une topologie qui exige un moment d'ordre ~4m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
