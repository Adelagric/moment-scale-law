"""§4 — le contre-exemple de l'anneau : b ne détermine PAS la topologie.

Produit les chiffres de la section « The converse is false » :
  - a* tel que l'anneau R~Unif[a,1] (densité planaire ∝ 1/r) ait exactement le ratio
    E[R^4]/E[R^2]^2 du disque plein, soit 4/3 ;
  - la vérification que TOUS les moments jusqu'à l'ordre 4 coïncident (en symétrie sphérique,
    E[R^4] est toute l'information d'ordre 4) ;
  - la persistance H1 des deux nuages, qui elle diffère.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist, squareform
from ripser import ripser

RNG = np.random.default_rng(0)


def ratio_ring(a):
    """E[R^4]/E[R^2]^2 pour R ~ Unif[a,1] en d=2."""
    return (9/5)*(1+a+a**2+a**3+a**4)/(1+a+a**2)**2


RATIO_DISK = 4/3          # disque plein uniforme en aire


def a_star():
    return brentq(lambda a: ratio_ring(a)-RATIO_DISK, 0.01, 0.99)


def cloud_ring(a, n=1200):
    r = RNG.uniform(a, 1, n); t = RNG.uniform(0, 2*np.pi, n)
    return np.c_[r*np.cos(t), r*np.sin(t)]


def cloud_disk(n=1200):
    r = np.sqrt(RNG.uniform(0, 1, n)); t = RNG.uniform(0, 2*np.pi, n)
    return np.c_[r*np.cos(t), r*np.sin(t)]


def mardia(P):
    mu = P.mean(0); Pc = P-mu
    C = np.cov(P, rowvar=False) + 1e-12*np.eye(2)
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))


def h1_max(P):
    d = ripser(squareform(pdist(P)), maxdim=1, distance_matrix=True)["dgms"][1]
    fin = d[np.isfinite(d[:, 1])] if len(d) else np.empty((0, 2))
    life = fin[:, 1]-fin[:, 0] if len(fin) else np.array([0.0])
    return float(life.max())


def main():
    a = a_star()
    print(f"a* = {a:.5f}   ratio(a*) = {ratio_ring(a):.6f}   ratio(disque) = {RATIO_DISK:.6f}")
    print(f"b_Mardia = d^2 * ratio : anneau {4*ratio_ring(a):.4f}  disque {4*RATIO_DISK:.4f}\n")

    # tous les moments jusqu'à l'ordre 4, sur grand échantillon, à E[R^2] normalisé
    big = 400000
    A = cloud_ring(a, big); B = cloud_disk(big)
    A = A/np.sqrt((A**2).sum(1).mean()); B = B/np.sqrt((B**2).sum(1).mean())
    print("moments jusqu'à l'ordre 4 (n=400000, E[R^2] normalisé) :")
    worst = 0.0
    for (i, j) in [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),
                   (4,0),(3,1),(2,2),(1,3),(0,4)]:
        ma = np.mean(A[:,0]**i*A[:,1]**j); mb = np.mean(B[:,0]**i*B[:,1]**j)
        worst = max(worst, abs(ma-mb))
        print(f"  E[x^{i} y^{j}] : anneau {ma:+.5f}   disque {mb:+.5f}   |Δ| {abs(ma-mb):.5f}")
    print(f"\n  écart maximal = {worst:.5f}  (niveau du bruit d'échantillonnage)\n")

    # topologie, sur les tailles citées dans le papier
    Ar, Dk = cloud_ring(a), cloud_disk()
    print(f"n=1200 par nuage :")
    print(f"  b_Mardia : anneau {mardia(Ar):.3f}   disque {mardia(Dk):.3f}")
    print(f"  |Δcov|_F : {np.linalg.norm(np.cov(Ar,rowvar=False)-np.cov(Dk,rowvar=False),'fro'):.4f}")
    print(f"  H1 max   : anneau {h1_max(Ar):.3f}   disque {h1_max(Dk):.3f}   <- la topologie diffère")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
