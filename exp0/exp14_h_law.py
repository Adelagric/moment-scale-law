"""PAPER 2 — déterminer h(ε) : à quel ordre N les moments voient-ils un feature d'échelle ε ?

Prédiction théorique : les moments d'ordre N résolvent l'échelle ~1/N (noyau de
Christoffel-Darboux), donc N*(ε) ~ 1/ε.

Mesure : anneau vs disque de rayon ε (même masse, même centre), on calcule la divergence des
coefficients dans une base ORTHONORMÉE (Legendre tensorisée sur [-1,1]², bien conditionnée —
les monômes bruts explosent). N*(ε) = plus petit N tel que la divergence dépasse un seuil.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import legendre as L

RNG = np.random.default_rng(0)
BOX = 1.0   # domaine normalisé [-1,1]^2


def ring(eps, n=4000):
    t = RNG.uniform(0, 2*np.pi, n)
    return np.c_[eps*np.cos(t), eps*np.sin(t)]

def disk(eps, n=4000):
    t = RNG.uniform(0, 2*np.pi, n); r = eps*np.sqrt(RNG.uniform(0, 1, n))
    return np.c_[r*np.cos(t), r*np.sin(t)]


def legendre_coeffs(P, N):
    """Coefficients <p, L_i(x)L_j(y)> pour i+j <= N, base orthonormée sur [-1,1]²."""
    x, y = P[:, 0]/BOX, P[:, 1]/BOX
    # évalue L_0..L_N normalisés
    def basis(v, N):
        out = np.zeros((N+1, len(v)))
        for k in range(N+1):
            c = np.zeros(k+1); c[k] = 1
            out[k] = L.legval(v, c) * np.sqrt((2*k+1)/2)   # normalisation L²([-1,1])
        return out
    Bx, By = basis(x, N), basis(y, N)
    coeffs = []
    for i in range(N+1):
        for j in range(N+1-i):
            coeffs.append(np.mean(Bx[i]*By[j]))
    return np.array(coeffs)


def divergence(eps, N):
    A, B = ring(eps), disk(eps)
    cA, cB = legendre_coeffs(A, N), legendre_coeffs(B, N)
    return float(np.linalg.norm(cA-cB))


def main():
    THRESH = 0.05
    print(f"{'ε':>7} {'N*(ε) (1er N où div>0.05)':>26} {'1/ε':>8} {'N*·ε':>7}")
    for eps in (0.8, 0.5, 0.3, 0.2, 0.1):
        star = None
        for N in range(2, 61, 2):
            if divergence(eps, N) > THRESH:
                star = N; break
        r = f"{star*eps:.2f}" if star else "-"
        print(f"{eps:>7} {str(star):>26} {1/eps:>8.1f} {r:>7}")
    print("\nLecture : si N*·ε ≈ constante => N*(ε) ~ C/ε, la loi h(ε) est LINÉAIRE en 1/ε :")
    print("l'ordre de moment requis explose comme l'inverse de l'échelle du feature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
