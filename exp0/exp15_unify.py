"""PAPER 2 — la loi est-elle N* ≍ b/ε (produit) ou N* ≍ 1/ε (échelle seule) ?

Christoffel-Darboux : un polynôme de degré N résout l'échelle 1/N UNIFORMÉMENT sur le domaine.
Donc degré ~1/ε devrait résoudre TOUS les features d'échelle ε, quel que soit leur nombre b.
=> prédiction : N* dépend de ε, PAS de b. Et dans la construction de Gauss, b coquilles dans la
boule unité sont espacées de ~1/b, donc ε~1/b et N*~1/ε~b : les deux lois n'en font qu'une.

Test : domaine fixe [-1,1]². b anneaux de rayon ε bien séparés. On fait varier b À ε FIXÉ.
  - si N* ~ constant en b  => loi unifiée N* ≍ 1/ε (le "b/ε" du papier double-compte)
  - si N* ~ b·(1/ε)        => loi produit confirmée
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import legendre as L

RNG = np.random.default_rng(0)


def basis(v, N):
    out = np.zeros((N+1, len(v)))
    for k in range(N+1):
        c = np.zeros(k+1); c[k] = 1
        out[k] = L.legval(v, c) * np.sqrt((2*k+1)/2)
    return out


def coeffs(P, N):
    Bx, By = basis(P[:, 0], N), basis(P[:, 1], N)
    return np.array([np.mean(Bx[i]*By[j]) for i in range(N+1) for j in range(N+1-i)])


def grid_centers(b, margin=0.75):
    """b centres bien séparés dans [-margin, margin]²."""
    k = int(np.ceil(np.sqrt(b)))
    xs = np.linspace(-margin, margin, k)
    pts = [(x, y) for y in xs for x in xs][:b]
    return np.array(pts)


def clouds(b, eps, filled, n_per=1500):
    C = grid_centers(b); out = []
    for c in C:
        t = RNG.uniform(0, 2*np.pi, n_per)
        r = eps*np.sqrt(RNG.uniform(0, 1, n_per)) if filled else np.full(n_per, eps)
        out.append(np.c_[r*np.cos(t), r*np.sin(t)] + c)
    return np.vstack(out)


def nstar(b, eps, thresh=0.05, Nmax=60):
    A, B = clouds(b, eps, False), clouds(b, eps, True)
    for N in range(2, Nmax+1, 2):
        if np.linalg.norm(coeffs(A, N) - coeffs(B, N)) > thresh:
            return N
    return None


def main():
    print("=== b varie, ε FIXÉ ===")
    print(f"{'ε':>6} {'b':>3} {'N*':>5}")
    for eps in (0.10, 0.05):
        for b in (1, 4, 9, 16):
            print(f"{eps:>6} {b:>3} {str(nstar(b, eps)):>5}")
    print("\n=== ε varie, b FIXÉ (contrôle) ===")
    print(f"{'b':>3} {'ε':>6} {'N*':>5} {'N*·ε':>6}")
    for b in (4,):
        for eps in (0.20, 0.10, 0.07, 0.05):
            ns = nstar(b, eps)
            print(f"{b:>3} {eps:>6} {str(ns):>5} {(ns*eps if ns else float('nan')):>6.2f}")
    print("\nLecture : N* plat en b (à ε fixé) => loi UNIFIÉE N* ≍ 1/ε ; le 'b/ε' double-compte.")
    print("          N*·ε constant => confirme la dépendance en 1/ε.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
