"""PAPER 2 — fermer l'écart : mesure DÉTERMINISTE du taux d'approximation de Jackson.

La borne sup N* ≤ C/(εf) vient de Jackson au 1er ordre : ‖φ − p_N‖ ≲ 1/(Nε) pour une bosse
Lipschitz. Avec une bosse LISSE (C^∞ à support compact, type Gevrey), la théorie prédit
‖φ − p_N‖ ≲ exp(−c(Nε)^{1/s}) : décroissance sur-algébrique. Alors exiger erreur < f donne
N ≳ (1/ε)·(log(1/f))^s au lieu de 1/(εf) — l'écart b↔b² devient b↔b·polylog(b).

Test déterministe (aucun échantillonnage) : erreur de meilleure approximation polynomiale de
degré N d'une bosse lisse de largeur ε, mesurée par projection sur base orthonormée (Legendre).
On regarde si log(erreur) décroît linéairement en N (exponentiel) ou en log N (algébrique).
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import legendre as L


def bump(x, eps, kind="smooth"):
    """Bosse de largeur eps centrée en 0, support [-eps, eps]."""
    u = np.clip(x/eps, -1, 1)
    inside = np.abs(x) < eps
    if kind == "smooth":       # C^infini à support compact (mollifier standard)
        out = np.zeros_like(x)
        out[inside] = np.exp(-1.0/(1.0 - u[inside]**2))
        return out/np.exp(-1.0)
    else:                      # Lipschitz (triangle) : cas Jackson 1er ordre
        return np.clip(1 - np.abs(x)/eps, 0, None)


def approx_error(eps, N, kind, M=20001):
    """Erreur L2 de la meilleure approximation de degré N (projection Legendre)."""
    x = np.linspace(-1, 1, M)
    f = bump(x, eps, kind)
    # coefficients de Legendre normalisés par quadrature
    err2 = np.trapezoid(f**2, x)
    for k in range(N+1):
        c = np.zeros(k+1); c[k] = 1
        Pk = L.legval(x, c)*np.sqrt((2*k+1)/2)
        ck = np.trapezoid(f*Pk, x)
        err2 -= ck**2
    return float(np.sqrt(max(err2, 1e-300)))


def main():
    eps = 0.15
    print(f"bosse de largeur ε={eps} — erreur d'approximation de degré N\n")
    print(f"{'N':>4} {'Nε':>6} {'err LISSE':>12} {'err LIPSCHITZ':>14}")
    Ns = [4, 8, 12, 16, 20, 24, 32, 40]
    es, el = [], []
    for N in Ns:
        a = approx_error(eps, N, "smooth"); b = approx_error(eps, N, "lip")
        es.append(a); el.append(b)
        print(f"{N:>4} {N*eps:>6.1f} {a:>12.3e} {b:>14.3e}")
    es, el = np.array(es), np.array(el); Ns = np.array(Ns, float)
    # exponentiel : log(err) ~ -c·N   |   algébrique : log(err) ~ -p·log(N)
    for name, e in (("LISSE", es), ("LIPSCHITZ", el)):
        m = e > 1e-14
        r_exp = np.corrcoef(Ns[m], np.log(e[m]))[0, 1]
        r_alg = np.corrcoef(np.log(Ns[m]), np.log(e[m]))[0, 1]
        p = np.polyfit(np.log(Ns[m]), np.log(e[m]), 1)[0]
        print(f"\n{name:>10} : R(log err vs N)={r_exp:.3f}   R(log err vs log N)={r_alg:.3f}"
              f"   pente algébrique={p:.2f}")
    print("\nLecture : bosse LISSE avec R(vs N) ≈ −1 et pente algébrique très raide")
    print("=> décroissance sur-algébrique/exponentielle => f entre en log(1/f), pas 1/f.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
