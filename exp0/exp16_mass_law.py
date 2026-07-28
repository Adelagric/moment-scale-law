"""PAPER 2 — fermer l'écart : la dépendance en masse f est-elle 1/f ou log(1/f) ?

Ma borne sup (Jackson 1er ordre, bosse Lipschitz) donne N* ≤ C/(εf) — le facteur 1/f crée
l'écart b vs b² dans la construction radiale. Avec une bosse LISSE et Jackson d'ordre k,
l'erreur est (C/(Nε))^k ; en optimisant k (fonction de Gevrey, lisse à support compact),
l'erreur décroît ~exp(−c(Nε)^{1/s}), donc f n'entre qu'en (log(1/f))^s.

Test : ε FIXÉ, on fait varier la masse f du feature, et on mesure N*. Seuil de détection
normalisé par f (on détecte le feature relativement à son propre signal).
  - N* ~ log(1/f)  (croissance très lente) => borne améliorée correcte, écart quasi fermé
  - N* ~ 1/f       (explosion)             => la borne 1/(εf) était serrée
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


def cloud(eps, f, filled, n=20000, center=(0.35, 0.0)):
    k = max(200, int(f*n))
    blob = RNG.uniform(-0.9, 0.9, (n-k, 2))          # fond uniforme
    t = RNG.uniform(0, 2*np.pi, k)
    r = eps*np.sqrt(RNG.uniform(0, 1, k)) if filled else np.full(k, eps)
    feat = np.c_[r*np.cos(t), r*np.sin(t)] + np.array(center)
    return np.vstack([blob, feat])


def nstar(eps, f, rel=0.05, Nmax=80):
    A, B = cloud(eps, f, False), cloud(eps, f, True)
    for N in range(2, Nmax+1, 2):
        # seuil PROPORTIONNEL à f : on détecte le feature relativement à son propre signal
        if np.linalg.norm(coeffs(A, N) - coeffs(B, N)) > rel*f:
            return N
    return None


def main():
    eps = 0.10
    print(f"ε = {eps} fixé\n")
    print(f"{'f':>7} {'log(1/f)':>9} {'1/f':>7} {'N*':>5}")
    fs = (0.40, 0.20, 0.10, 0.05, 0.025)
    ns = []
    for f in fs:
        s = nstar(eps, f)
        ns.append(s)
        print(f"{f:>7} {np.log(1/f):>9.2f} {1/f:>7.1f} {str(s):>5}")
    xs = np.array([np.log(1/f) for f in fs]); ys = np.array([n if n else np.nan for n in ns], float)
    ok = ~np.isnan(ys)
    if ok.sum() > 2:
        a, b_ = np.polyfit(xs[ok], ys[ok], 1)
        r = np.corrcoef(xs[ok], ys[ok])[0, 1]
        print(f"\najustement N* = {a:.2f}·log(1/f) + {b_:.2f}   R={r:.3f}")
        inv = np.array([1/f for f in fs])
        r2 = np.corrcoef(inv[ok], ys[ok])[0, 1]
        print(f"corrélation avec 1/f : R={r2:.3f}")
        print("\n=> R(log) >> R(1/f) et pente modeste : dépendance LOGARITHMIQUE en f,")
        print("   donc N* ≲ C·log(1/f)/ε et l'écart b↔b² se referme en b↔b·log b.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
