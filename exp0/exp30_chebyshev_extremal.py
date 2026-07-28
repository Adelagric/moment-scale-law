"""§3 — borne inférieure : contraste maximal d'un polynôme localisé à l'échelle ε.

Problème extrémal de Chebyshev : maximiser |p(0)| pour deg p <= N sous |p| <= 1 sur
ε <= |x| <= 1. La solution classique passe par t = (1+ε²-2x²)/(1-ε²) puis Chebyshev, d'où
    contraste = T_N((1+ε²)/(1-ε²)) ≈ exp(2Nε).
Distinguer une masse f de 0 par un test polynomial de degré <= N exige contraste > 1/f,
donc N >= log(1/f)/(2ε) — la borne inférieure du papier, valable pour les tests POLYNOMIAUX
(et non pour tous les tests : MMD n'est pas polynomial).
"""
from __future__ import annotations
import numpy as np


def contrast(N, eps):
    t = (1+eps**2)/(1-eps**2)
    return np.cosh(N*np.arccosh(t))          # T_N(t) pour t > 1


def main():
    print("contraste maximal d'un polynôme de degré N localisé à l'échelle ε\n")
    print(f"{'ε':>6} {'N':>5} {'Nε':>6} {'log(contraste)':>15} {'log(c)/(Nε)':>13}")
    for eps in (0.05, 0.1, 0.2):
        for N in (20, 50, 100):
            c = contrast(N, eps)
            print(f"{eps:>6} {N:>5} {N*eps:>6.1f} {np.log(c):>15.2f} {np.log(c)/(N*eps):>13.3f}")
    print("\n=> log(contraste) -> 2·N·ε : le contraste croît comme exp(2Nε).")
    print("   Pour un contraste > 1/f il faut N >= log(1/f)/(2ε).")
    print("   Portée : tests POLYNOMIAUX de degré <= N, pas tous les tests statistiques.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
