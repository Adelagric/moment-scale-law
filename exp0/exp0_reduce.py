"""EXP 0 — le négatif est-il un artefact de "mauvais espace" (384-d brut) ?

L'hypothèse de variété dit dim_intrinsèque << 384. Rips en haute dim est dominé par
la concentration. On (1) estime la dimension intrinsèque, (2) refait le test H1-vs-null
APRÈS réduction PCA à cette dimension. Si H1 apparaît après réduction, le négatif
précédent était un artefact de pipeline (il faut réduire avant TDA). Sinon, le négatif
est robuste : pas de variété exploitable pour ce couple (embedder, corpus).
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from exp0_gate import build_embeddings, window_pvalue
from exp0_sweep import sample_window, h1_frac_sig


def twonn_dim(X, rng):
    """Estimateur de dimension intrinsèque TwoNN (Facco et al. 2017)."""
    from scipy.spatial.distance import cdist
    n = min(len(X), 500)
    idx = rng.choice(len(X), n, replace=False)
    D = cdist(X[idx], X[idx])
    D.sort(axis=1)
    r1, r2 = D[:, 1], D[:, 2]
    mask = r1 > 1e-12
    mu = (r2[mask] / r1[mask])
    mu = mu[mu > 1 + 1e-9]
    F = np.arange(1, len(mu) + 1) / len(mu)
    x = np.log(np.sort(mu))
    ylog = -np.log(1 - F[:len(x)] + 1e-12)
    d = np.sum(x * ylog) / np.sum(x * x)  # régression sans intercept
    return float(d)


def pca_var_dim(X, frac=0.9):
    p = PCA().fit(X)
    c = np.cumsum(p.explained_variance_ratio_)
    return int(np.searchsorted(c, frac) + 1), p.explained_variance_ratio_[:5]


def main():
    rng = np.random.default_rng(3)
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)

    did = twonn_dim(X, rng)
    dpca, top5 = pca_var_dim(X, 0.9)
    print(f"dimension intrinsèque TwoNN ≈ {did:.1f}")
    print(f"PCA : {dpca} composantes pour 90% variance | top-5 ratios = {np.round(top5,3)}")

    print("\n=== H1 vs null gaussien, APRÈS réduction PCA ===")
    print(f"{'dim':>5} {'mode':>7} {'frac_sig':>9} {'med_p':>7}")
    for k in (max(3, int(round(did))), 15, 40):
        Xr = PCA(n_components=k).fit_transform(X)
        for mode in ("intra", "multi"):
            frac, medp = h1_frac_sig(Xr, y, W=150, mode=mode,
                                     categories=categories, rng=rng,
                                     n_windows=12, n_null=15)
            print(f"{k:>5} {mode:>7} {frac:9.0%} {medp:7.3f}")

    print("\nLecture : frac_sig qui monte nettement après réduction => réduire AVANT TDA.")
    print("frac_sig toujours ~0 => négatif robuste, pas de variété H1 exploitable ici.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
