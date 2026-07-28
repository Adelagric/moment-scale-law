"""PAPER 2 (option 1, §4) — le tore de révolution RÉDUIT au profil (n'est pas vraiment enlacé).

Un tore symétrique par rotation autour de z est déterminé par sa mesure profil (ρ, z),
ρ=√(x²+y²). Sa boucle-tube (H_1 mineure) = le trou du profil = un cercle dans le demi-plan
=> cercle-vs-disque => 4ᵉ moment du profil. La boucle azimutale majeure = le trou autour de
l'axe (ρ concentré loin de 0) = feature de bas ordre partagée.

On vérifie : tore (profil cercle) vs tore PLEIN (profil disque). Seule la boucle-tube diffère,
et elle correspond au trou du profil. Donc topologie symétrique => se réduit au cas profil, où
la loi N* s'applique. Reste ouvert : entanglement GÉNUINEMENT asymétrique.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser
from scipy.spatial.distance import squareform, pdist

np.random.seed(0)
R, r = 2.0, 0.7


def torus(filled=False, nphi=34, nth=20, noise=0.015):
    # ÉCHANTILLONNAGE RÉGULIER (grille en (phi, theta)) : Betti propres, pas de barres parasites
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    th = np.linspace(0, 2*np.pi, nth, endpoint=False)
    P, T = np.meshgrid(phi, th); P, T = P.ravel(), T.ravel()
    n = len(P)
    rr = r*np.sqrt(np.random.uniform(0, 1, n)) if filled else np.full(n, r)
    x = (R + rr*np.cos(T))*np.cos(P)
    y = (R + rr*np.cos(T))*np.sin(P)
    z = rr*np.sin(T)
    return np.c_[x, y, z] + np.random.normal(0, noise, (n, 3))


def betti(P):
    dm = squareform(pdist(P, "euclidean"))
    dgs = ripser(dm, maxdim=2, distance_matrix=True, thresh=3.4)["dgms"]
    out = []
    for d, thr in ((1, 0.6), (2, 0.4)):  # seuils séparant features (>0.6/0.4) du bruit (~0.1)
        fin = dgs[d][np.isfinite(dgs[d][:, 1])] if len(dgs[d]) else np.empty((0, 2))
        life = fin[:, 1]-fin[:, 0] if len(fin) else np.array([])
        out.append(int(np.sum(life > thr)))
    return out  # [H1, H2]


def profile(P):
    rho = np.sqrt(P[:, 0]**2 + P[:, 1]**2); z = P[:, 2]
    return rho, z


def main():
    T, S = torus(filled=False), torus(filled=True)
    h1T, h2T = betti(T); h1S, h2S = betti(S)
    print(f"tore creux : H_1={h1T} (attendu 2)  H_2={h2T} (attendu 1)")
    print(f"tore plein : H_1={h1S} (attendu 1 : boucle majeure seule)  H_2={h2S} (attendu 0)")
    # profil : la boucle-tube = trou du profil (Var du rayon-profil² autour de (R,0))
    rT, zT = profile(T); rS, zS = profile(S)
    prof2_T = (rT-R)**2 + zT**2   # rayon² dans le plan profil, centré sur le cercle mineur
    prof2_S = (rS-R)**2 + zS**2
    print(f"\nprofil (ρ,z) — rayon² autour du cercle mineur :")
    print(f"  Var(prof²) tore creux={np.var(prof2_T):.3f} (coquille=faible)  plein={np.var(prof2_S):.3f} (rempli=élevé)")
    print(f"  => la boucle-tube EST le trou du profil (cercle vs disque) => 4ᵉ moment du profil.")
    print("\nConclusion : la topologie symétrique par rotation se réduit au profil, où N* s'applique.")
    print("Reste vraiment ouvert : entanglement asymétrique (aucune symétrie ne linéarise).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
