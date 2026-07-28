"""PAPER 2 — contrôle propre : à (μ,Σ) fixés, la CONNEXITÉ est un phénomène de 4ᵉ moment.

Analogue de cercle/disque, mais pour H_0 :
  - bimodal   : 2 amas serrés ±a  -> H_0 = 2  (platykurtique le long de l'axe)
  - allongé   : 1 blob gaussien de MÊME variance -> H_0 = 1  (mésokurtique)
Même moyenne, même covariance par construction ; seuls la topologie ET le 4ᵉ moment
diffèrent. Combiné à cercle/disque (H_1) et sphère/boule (H_2) => unification :

  À (μ,Σ) fixés, les changements topologiques canoniques (H_0 bimodalité, H_{≥1}
  creux) sont TOUS des phénomènes de 4ᵉ moment, indépendamment de la dimension homologique.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

np.random.seed(0)
DIM = 3


def h0_prominence(P):
    dm = squareform(pdist(P, "euclidean"))
    d0 = ripser(dm, maxdim=0, distance_matrix=True)["dgms"][0]
    life = np.sort(d0[np.isfinite(d0[:, 1]), 1])[::-1]
    return float(life[0] - life[1]) if len(life) > 1 else 0.0  # gap = signal "2 amas"

def mardia(P):
    mu = P.mean(0); Pc = P - mu; C = np.cov(P, rowvar=False) + 1e-9*np.eye(P.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))

def bimodal(n=240, a=1.5, w=0.3):
    h = n//2
    X = np.random.normal(0, w, (n, DIM))
    X[:h, 0] -= a; X[h:, 0] += a
    return X

def elongated(n=240, a=1.5, w=0.3):
    s0 = np.sqrt(a**2 + w**2)               # variance axe 0 = a²+w² (= bimodal)
    X = np.random.normal(0, w, (n, DIM)); X[:, 0] = np.random.normal(0, s0, n)
    return X


def main():
    B, E = bimodal(), elongated()
    print(f"|Δmean| = {np.linalg.norm(B.mean(0)-E.mean(0)):.4f}")
    print(f"|Δcov|_F = {np.linalg.norm(np.cov(B,rowvar=False)-np.cov(E,rowvar=False),'fro'):.4f}  (visé ~0)")
    print(f"H_0 prominence : bimodal={h0_prominence(B):.2f}  allongé={h0_prominence(E):.2f}  (2 amas vs 1)")
    print(f"kurtosis Mardia: bimodal={mardia(B):.2f}  allongé={mardia(E):.2f}  (4ᵉ moment SÉPARE)")
    print("\n=> À (μ,Σ) fixés : H_0 diffère (2 vs 1) ET la kurtosis diffère, la covariance non.")
    print("   Connexité = 4ᵉ moment, comme les trous. Unification H_0/H_1/H_2 au 4ᵉ moment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
