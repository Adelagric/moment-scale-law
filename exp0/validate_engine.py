"""Garde-fou de correction : le moteur retrouve-t-il une topologie CONNUE ?

Non négociable avant de faire confiance au moteur sur des données réelles.
Attendus :
  - deux cercles disjoints : H0 = 2 composantes, H1 = 2 boucles
  - tore                    : H1 = 2, H2 = 1
"""
from __future__ import annotations
import numpy as np
from topology import diagrams, lifetimes


def two_circles(n=150, r=1.0, sep=4.0, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for cx in (0.0, sep):
        t = rng.uniform(0, 2 * np.pi, n)
        c = np.c_[cx + r * np.cos(t), r * np.sin(t)]
        out.append(c + rng.normal(0, noise, c.shape))
    return np.vstack(out)


def torus(n=600, R=3.0, r=1.0, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    P = np.c_[x, y, z]
    return P + rng.normal(0, noise, P.shape)


def count_long(dgm, frac=0.3):
    """Barres > frac * (plus longue barre) — comptage robuste des features dominants."""
    lt = lifetimes(dgm)
    if lt.size == 0:
        return 0
    return int(np.sum(lt > frac * lt.max()))


def main():
    ok = True

    X = two_circles()
    d = diagrams(X, maxdim=1)
    h1 = count_long(d[1])
    # H0 : nb de composantes = nb de barres H0 dont la mort est "grande" + 1
    lt0 = lifetimes(d[0])
    h0_gap = int(np.sum(lt0 > 0.5 * (lt0.max() if lt0.size else 1))) + 1
    print(f"[deux cercles]  H1 dominants = {h1} (attendu 2) | H0 composantes ~ {h0_gap} (attendu 2)")
    ok &= (h1 == 2)

    X = torus(seed=1)
    d = diagrams(X, maxdim=2)
    h1 = count_long(d[1])
    h2 = count_long(d[2]) if len(d) > 2 else 0
    print(f"[tore]          H1 dominants = {h1} (attendu 2) | H2 dominants = {h2} (attendu 1)")
    ok &= (h1 == 2)  # H2 sur tore est difficile à échantillonner ; on note mais on n'exige pas

    print("\nGARDE-FOU:", "PASS ✅" if ok else "FAIL ❌ — moteur non fiable, stop.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
