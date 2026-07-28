"""EXP 2c — contrôle d'artefact PROPRE et imparable.

M = Σ^{1/2} Q Σ^{-1/2} avec Q orthogonale aléatoire.
  - M Σ Mᵀ = Σ  => moyenne+covariance préservées EXACTEMENT (B6, Maha aveugles).
  - M inversible => homéomorphisme => topologie IDENTIQUE (aucun trou bouché).
  - Q ≠ I => M non-orthogonale => distances DÉFORMÉES.

Si Δ_top s'allume ici, il détecte une distorsion géométrique préservant à la fois les
moments ET la topologie : la preuve définitive que Δ_top (pipeline euclidien réduit)
ne mesure PAS la topologie mais la géométrie de base. Exp 2 s'effondre.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.stats import ortho_group
from scipy.spatial.distance import pdist, squareform
from exp0_gate import build_embeddings
from exp2b_scrutiny import h1_land, knn_density, mardia_kurtosis

RNG = np.random.default_rng(2718)


def moment_preserving_distortion(X):
    """x -> mu + M(x-mu), M = Σ^{1/2} Q Σ^{-1/2}. Préserve (mu,Σ) et la topologie."""
    mu = X.mean(0)
    S = np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1])
    S12 = np.real(sqrtm(S)); S12i = np.linalg.inv(S12)
    Q = ortho_group.rvs(X.shape[1], random_state=RNG)
    M = S12 @ Q @ S12i
    return (X - mu) @ M.T + mu


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
        return {
            "Δ_top": float(np.sum(np.abs(h1_land(w) - land_ref))),
            "B6_spec": float(np.linalg.norm(np.cov(w, rowvar=False) - Sig_ref, "fro")),
            "Maha": float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, SigInv, diff)))),
            "Kurt_Mardia": mardia_kurtosis(w),
        }

    N = 24
    H = [healthy() for _ in range(N)]
    distorted = [moment_preserving_distortion(healthy()) for _ in range(N)]

    # vérifier que (mu,Σ) sont bien préservés
    dmom = np.mean([np.linalg.norm(np.cov(d, rowvar=False) - np.cov(h, rowvar=False), "fro")
                    for d, h in zip(distorted, H)])
    print(f"|Δcov| distorsion vs sain (préservation par M) : {dmom:.4f}")

    sh = [scores(w) for w in H]; sd = [scores(w) for w in distorted]
    print(f"\n=== AUC sain vs distorsion préservant (μ,Σ) ET topologie (N={N}) ===")
    for det in sh[0]:
        yv = [0]*N + [1]*N
        sv = [s[det] for s in sh] + [s[det] for s in sd]
        auc = roc_auc_score(yv, sv)
        print(f"  {det:>12} : AUC={auc:.3f}  (bilatéral {max(auc,1-auc):.3f})")

    print("\nVERDICT :")
    print("  Δ_top ≫0.5 ici => il détecte une distorsion SANS topologie ni moments :")
    print("  le signal 'topologique' d'Exp 2 est un ARTEFACT géométrique. Exp 2 tombe.")
    print("  Δ_top ≈0.5 ici => Exp 2 mesurait bien de la topologie (peu probable vu 2b).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
