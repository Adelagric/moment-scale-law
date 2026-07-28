"""EXP 2 — ATTAQUANT ADAPTATIF sur embeddings réels (question 2 + validation défense).

Construit une attaque qui (a) provoque un effondrement topologique, (b) PRÉSERVE
exactement moyenne+covariance de sa pré-image saine (pour évader B6/Mahalanobis).

Construction : bouche le trou (collapse central) PUIS correction affine ramenant
(mu, Sigma) à ceux de la pré-image. L'affine est un homéomorphisme => topologie
bouchée conservée, 2ᵈ ordre neutralisé. Métrique EUCLIDIENNE en espace PCA réduit
(cohérence avec les opérations linéaires de moment).

Contraste attendu :
  - collapse NAÏF          : change le 2ᵈ ordre => B6 gagne (comme Exp 1)
  - collapse ADAPTATIF     : 2ᵈ ordre préservé => B6 aveugle ; SEULE la TDA peut voir
Si Δ_top sépare l'adaptatif alors que B6≈0.5 => défense en profondeur prouvée SUR RÉEL.
Si Δ_top≈0.5 aussi => les fenêtres réelles n'ont pas de H1 assez fort : niche
seulement synthétique (honnête à rapporter).
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from exp0_gate import build_embeddings
from ripser import ripser

RNG = np.random.default_rng(555)
GRID = np.linspace(0.0, 3.0, 60)


def h1_land(X):
    dm = squareform(pdist(X, "euclidean"))
    dgm = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(GRID - b, d - GRID), 0, None))
    return lam


def collapse_raw(X, alpha=0.5, rho=0.35):
    """Effondrement : contracte une fraction rho des points vers le centroïde."""
    X = X.copy(); mu = X.mean(0)
    idx = RNG.choice(len(X), int(rho * len(X)), replace=False)
    X[idx] = mu + (1 - alpha) * (X[idx] - mu)
    return X


def affine_match(X, mu_t, Sig_t):
    """Application affine forçant (mu, Sigma)(X) -> (mu_t, Sig_t). Homéomorphisme."""
    mu1 = X.mean(0)
    Sig1 = np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1])
    A = np.real(sqrtm(Sig_t)) @ np.linalg.inv(np.real(sqrtm(Sig1)))
    return (X - mu1) @ A.T + mu_t


def main():
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)
    Xr = PCA(n_components=8).fit_transform(X)

    def healthy(W=250):
        per = W // len(categories); idx = []
        for c in range(len(categories)):
            idx += list(RNG.choice(np.where(y == c)[0], per, replace=False))
        return Xr[np.array(idx)]

    train = [healthy() for _ in range(20)]
    Sig_ref = np.mean([np.cov(w, rowvar=False) for w in train], axis=0)
    SigInv = np.linalg.pinv(Sig_ref)
    mu_ref = np.mean([w.mean(0) for w in train], axis=0)
    land_ref = np.mean([h1_land(w) for w in train], axis=0)

    def scores(w):
        diff = w - mu_ref
        maha = float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, SigInv, diff))))
        return {
            "Δ_top(H1)": float(np.sum(np.abs(h1_land(w) - land_ref))),
            "Mahalanobis": maha,
            "Spectral(B6)": float(np.linalg.norm(np.cov(w, rowvar=False) - Sig_ref, "fro")),
        }

    N = 24
    H = [healthy() for _ in range(N)]
    naive, adaptive = [], []
    dmoments = []
    for _ in range(N):
        h = healthy()
        mu_h, Sig_h = h.mean(0), np.cov(h, rowvar=False)
        naive.append(collapse_raw(h))
        a = affine_match(collapse_raw(h), mu_h, Sig_h)  # préserve (mu_h, Sig_h)
        adaptive.append(a)
        dmoments.append((np.linalg.norm(a.mean(0) - mu_h),
                         np.linalg.norm(np.cov(a, rowvar=False) - Sig_h, "fro")))
    dmoments = np.array(dmoments)
    print(f"attaque adaptative — écart aux moments de la pré-image :")
    print(f"  |Δmean|={dmoments[:,0].mean():.4f}   |Δcov|_F={dmoments[:,1].mean():.4f}  (visés ~0)")

    sh = [scores(w) for w in H]
    print(f"\n=== AUC sain vs attaqué (N={N}) ===")
    print(f"{'attaque':>10}" + "".join(f"{d:>14}" for d in sh[0]))
    for label, group in (("NAÏF", naive), ("ADAPTATIF", adaptive)):
        sa = [scores(w) for w in group]
        row = f"{label:>10}"
        for d in sh[0]:
            yv = [0]*N + [1]*N
            sv = [s[d] for s in sh] + [s[d] for s in sa]
            row += f"{roc_auc_score(yv, sv):>14.3f}"
        print(row)

    print("\nLecture :")
    print("  NAÏF: B6≈1.0 (comme Exp1).  ADAPTATIF: B6→0.5 (aveugle) ;")
    print("  Δ_top ADAPTATIF ≫0.5 => la TDA rattrape l'évasion => défense en profondeur ✅")
    print("  Δ_top ADAPTATIF ≈0.5 => H1 réel trop faible : niche seulement synthétique.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
