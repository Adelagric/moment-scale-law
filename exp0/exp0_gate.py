"""EXPÉRIMENT 0 — GATE (protocole §2ter).

Question binaire et décisive : les fenêtres d'embeddings RÉELS portent-elles une
homologie H1 significative AU-DELÀ de ce qu'explique leur covariance ?

Null model = gaussienne de moyenne+covariance identiques à la fenêtre. C'est le null
qui répond DIRECTEMENT à la menace B6 (baseline spectrale) : si la persistance H1
réelle n'excède pas celle du null gaussien, la topologie ne capte rien que la
covariance ne capte déjà, et l'axe « topologique » se replie sur la connexité.

Métrique par fenêtre : p = P_null( maxpers_H1(null) >= maxpers_H1(réel) ).
Verdict : fraction de fenêtres avec p < 0.05.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ripser import ripser


def cosine_dm(X: np.ndarray) -> np.ndarray:
    """Matrice de distance cosinus (métrique honnête pour des embeddings)."""
    d = squareform(pdist(X, metric="cosine"))
    np.fill_diagonal(d, 0.0)
    return d


def max_h1_persistence(X: np.ndarray) -> float:
    dm = cosine_dm(X)
    dgm = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    if len(dgm) == 0:
        return 0.0
    fin = dgm[np.isfinite(dgm[:, 1])]
    if len(fin) == 0:
        return 0.0
    return float(np.max(fin[:, 1] - fin[:, 0]))


def matched_gaussian_null(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Points ~ N(mean, cov) de la fenêtre. Covariance rank-déficiente OK (SVD)."""
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    return rng.multivariate_normal(mu, cov, size=X.shape[0], method="svd")


def window_pvalue(Xwin: np.ndarray, n_null: int, rng: np.random.Generator):
    real = max_h1_persistence(Xwin)
    null = np.array([max_h1_persistence(matched_gaussian_null(Xwin, rng)) for _ in range(n_null)])
    p = float((np.sum(null >= real) + 1) / (n_null + 1))
    return real, null, p


def build_embeddings(categories, per_cat, seed=0, cache="emb_cache.npz"):
    import os
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"[cache] embeddings {z['X'].shape}")
        return z["X"], z["y"]
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    data = fetch_20newsgroups(subset="train", categories=categories,
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=seed)
    texts, labels = [], []
    rng = np.random.default_rng(seed)
    for cat_id in range(len(categories)):
        idx = [i for i, t in enumerate(data.target) if t == cat_id]
        idx = [i for i in idx if len(data.data[i].split()) >= 20][:per_cat]
        texts += [data.data[i] for i in idx]
        labels += [cat_id] * len(idx)
    print(f"[embed] {len(texts)} docs, model all-MiniLM-L6-v2 (384-d)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    np.savez(cache, X=X, y=y)
    print(f"[embed] done {X.shape}")
    return X, y


def main():
    rng = np.random.default_rng(42)
    # Corpus réel étiqueté par sujet (fournit aussi, plus tard, les shifts bénins N)
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)

    W = 80          # taille de fenêtre
    n_windows = 30  # fenêtres échantillonnées
    n_null = 20     # nulls gaussiens par fenêtre

    # Fenêtres INTRA-sujet (un seul sujet à la fois) : régime "sain" du protocole.
    results = []
    for _ in range(n_windows):
        cat = rng.integers(0, len(categories))
        pool = np.where(y == cat)[0]
        if len(pool) < W:
            continue
        idx = rng.choice(pool, size=W, replace=False)
        real, null, p = window_pvalue(X[idx], n_null, rng)
        results.append((cat, real, null.mean(), null.max(), p))

    print("\n=== EXP 0 : H1 réel vs null gaussien apparié (cov identique) ===")
    print(f"{'sujet':>6} {'H1_réel':>9} {'H1_null_moy':>12} {'H1_null_max':>12} {'p':>7}")
    ps = []
    for cat, real, nmean, nmax, p in results:
        print(f"{cat:>6} {real:9.4f} {nmean:12.4f} {nmax:12.4f} {p:7.3f}")
        ps.append(p)
    ps = np.array(ps)
    frac_sig = float(np.mean(ps < 0.05))
    print(f"\nfenêtres significatives (p<0.05) : {frac_sig:.0%}  ({np.sum(ps<0.05)}/{len(ps)})")
    print("médiane p =", round(float(np.median(ps)), 3))
    print("\nVERDICT EXP 0 :",
          "H1 réel > covariance ✅ (topologie non triviale)" if frac_sig >= 0.5
          else "H1 réel ≈ covariance ❌ → repli sur β0/systèmes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
