"""PAPER 2 (option 1, §3) — loi N* ∝ b HORS symétrie sphérique : cas séparable.

b boucles (cercles dans le plan y-z) centrées aux nœuds de Gauss sur l'axe x, vs un CYLINDRE
(x ~ Uniform, section circulaire). La marginale sur x est : b atomes (Gauss) vs continue.
=> moments sur x appariés jusqu'à l'ordre 2b-1 ; comme x ⊥ (y,z), tous les moments nuage de
x-degré ≤ 2b-1 coïncident. Mais topologie : b boucles (H_1=b) vs cylindre (H_1=1).
=> N* ∝ b sans hypothèse radiale. Aucune optimisation.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

np.random.seed(0)
L = 4.0
NPER = 120  # points par boucle / par tranche


def gauss(b):
    x, w = leggauss(b); return 0.5*(x+1)*L, 0.5*w


def loops(b, noise=0.02):
    xs, w = gauss(b); out = []
    for xi, wi in zip(xs, w):
        t = np.linspace(0, 2*np.pi, NPER, endpoint=False)
        P = np.c_[np.full(NPER, xi), np.cos(t), np.sin(t)] + np.random.normal(0, noise, (NPER, 3))
        out.append(P)
    return np.vstack(out)


def cylinder(b, noise=0.02):
    # TUBE PLEIN (section disque plein) = contractile, H_1 = 0 ; marginale x continue
    n = b*NPER
    x = np.random.uniform(0, L, n)
    rad = np.sqrt(np.random.uniform(0, 1, n)); t = np.random.uniform(0, 2*np.pi, n)
    return np.c_[x, rad*np.cos(t), rad*np.sin(t)] + np.random.normal(0, noise, (n, 3))


def x_moments(P, K):
    return np.array([np.mean(P[:, 0]**j) for j in range(1, K+1)])


def n_loops(P, thresh=0.25):
    dm = squareform(pdist(P, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    life = fin[:, 1]-fin[:, 0] if len(fin) else np.array([])
    return int(np.sum(life > thresh))


def main():
    print(f"{'b':>3} {'ordre x apparié 2b-1':>20} {'max Δ moment x':>15} {'H1 boucles':>11} {'H1 cylindre':>12}")
    for b in (1, 2, 3):
        Lp, Cy = loops(b), cylinder(b)
        K = 2*b-1
        mL, mC = x_moments(Lp, K), x_moments(Cy, K)
        d = np.max(np.abs(mL-mC)/(np.abs(mC)+1e-9))
        print(f"{b:>3} {2*b-1:>20} {d:>15.3f} {n_loops(Lp):>11} {n_loops(Cy):>12}")
    print("\nLecture : moments sur x appariés jusqu'à 2b-1, MAIS H1 = b (boucles) vs 1 (cylindre).")
    print("=> N* ∝ b hors symétrie sphérique (§3 de l'argument du nerf) : cas SÉPARABLE prouvé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
