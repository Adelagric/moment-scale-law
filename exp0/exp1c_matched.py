"""EXP 1c — test PROPRE : cercle vs disque à covariance EXACTEMENT égale.

Le test par blanchiment (exp1b) était pollué par le conditionnement numérique.
Ici, isolation sans artefact de « topologie au-delà du 2ᵈ ordre » :

  - cercle (rayon r)            : cov = (r²/2)·I₂ , H1 = 1 boucle
  - disque uniforme (rayon r√2) : cov = (r²/2)·I₂ , H1 = 0

=> MÊME moyenne, MÊME covariance, topologie DIFFÉRENTE. B6 (covariance) est aveugle
PAR CONSTRUCTION MATHÉMATIQUE, pas par bricolage. Bruit de fond identique en dim 2..7.

Question : Δ_top sépare-t-il, là où B6 ne PEUT pas ? Si oui, la niche existe en principe
(reste à savoir si elle se produit sur de vraies attaques — cf. Exp 1, où non).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
from ripser import ripser

RNG = np.random.default_rng(9)
DIM = 8
GRID = np.linspace(0.0, 2.2, 60)


def h1_landscape_euclid(X):
    dm = squareform(pdist(X, metric="euclidean"))
    dgm = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(GRID - b, d - GRID), 0, None))
    return lam


def embed(struct2d, bg_sd=0.10):
    n = len(struct2d)
    bg = RNG.normal(0, bg_sd, size=(n, DIM - 2))  # bruit de fond identique en loi
    return np.hstack([struct2d, bg])


def circle(n=250, r=1.0, noise=0.06):
    t = RNG.uniform(0, 2*np.pi, n)
    P = np.c_[r*np.cos(t), r*np.sin(t)] + RNG.normal(0, noise, (n, 2))
    return embed(P)


def disk(n=250, r=1.0, noise=0.06):
    R = r*np.sqrt(2)                      # cov disque uniforme = R²/4 = r²/2 = cov cercle
    rad = R*np.sqrt(RNG.uniform(0, 1, n))  # échantillonnage uniforme en AIRE
    t = RNG.uniform(0, 2*np.pi, n)
    P = np.c_[rad*np.cos(t), rad*np.sin(t)] + RNG.normal(0, noise, (n, 2))
    return embed(P)


def spectral(X, Sig_ref):
    return float(np.linalg.norm(np.cov(X, rowvar=False) - Sig_ref, "fro"))


def main():
    # confirmer l'égalité de covariance (les 2 premières dims)
    c = np.vstack([circle()[:, :2] for _ in range(30)])
    d = np.vstack([disk()[:, :2] for _ in range(30)])
    print("covariance (2 premières dims) :")
    print("  cercle :", np.round(np.cov(c, rowvar=False), 3).tolist())
    print("  disque :", np.round(np.cov(d, rowvar=False), 3).tolist())

    train = [circle() for _ in range(20)]
    Sig_ref = np.mean([np.cov(w, rowvar=False) for w in train], axis=0)
    land_ref = np.mean([h1_landscape_euclid(w) for w in train], axis=0)

    N = 24
    healthy = [circle() for _ in range(N)]
    attacked = [disk() for _ in range(N)]

    def sc(w):
        return {
            "Δ_top(H1)": float(np.sum(np.abs(h1_landscape_euclid(w) - land_ref))),
            "Spectral(B6)": spectral(w, Sig_ref),
        }
    sh = [sc(w) for w in healthy]; sa = [sc(w) for w in attacked]

    dcov = np.mean([np.linalg.norm(np.cov(a, rowvar=False)-np.cov(h, rowvar=False), "fro")
                    for a, h in zip(attacked, healthy)])
    print(f"\nΔcov cercle vs disque (doit être ~0) : {dcov:.4f}")

    print(f"\n=== AUC cercle(sain) vs disque(bouché), covariance égale (N={N}) ===")
    for det in sh[0]:
        yv = [0]*N + [1]*N
        sv = [s[det] for s in sh] + [s[det] for s in sa]
        auc = roc_auc_score(yv, sv)
        print(f"  {det:>13} : AUC={auc:.3f}  (bilatéral {max(auc,1-auc):.3f})")

    print("\nVERDICT : Δ_top≫0.5 & B6≈0.5 => la niche topologique existe EN PRINCIPE.")
    print("Rappel Exp 1 : sur de VRAIES attaques, elles perturbent le 2ᵈ ordre => B6 gagne.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
