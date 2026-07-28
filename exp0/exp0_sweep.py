"""EXP 0 — stress du résultat négatif.

Un négatif à une seule config ne conclut rien. On varie :
  - taille de fenêtre W (une boucle a besoin d'assez de points) ;
  - régime intra-sujet vs multi-sujets (la structure peut être INTER-clusters) ;
et on teste séparément si beta_0 (le repli candidat) porte un signal sémantique.
"""
from __future__ import annotations
import numpy as np
from exp0_gate import build_embeddings, window_pvalue, cosine_dm
from ripser import ripser


def sample_window(X, y, W, mode, rng, categories):
    if mode == "intra":
        cat = rng.integers(0, len(categories))
        pool = np.where(y == cat)[0]
        if len(pool) < W:
            return None
        return rng.choice(pool, size=W, replace=False)
    else:  # multi : mélange équilibré de tous les sujets
        per = W // len(categories)
        idx = []
        for c in range(len(categories)):
            pool = np.where(y == c)[0]
            idx += list(rng.choice(pool, size=per, replace=False))
        return np.array(idx)


def h1_frac_sig(X, y, W, mode, categories, rng, n_windows=12, n_null=15):
    ps = []
    for _ in range(n_windows):
        idx = sample_window(X, y, W, mode, rng, categories)
        if idx is None:
            continue
        _, _, p = window_pvalue(X[idx], n_null, rng)
        ps.append(p)
    ps = np.array(ps)
    return float(np.mean(ps < 0.05)), float(np.median(ps))


def beta0_at_scale(X, scale):
    """Nb de composantes connexes du complexe de Rips au seuil `scale` (distance cosinus)."""
    dm = cosine_dm(X)
    d0 = ripser(dm, maxdim=0, distance_matrix=True)["dgms"][0]
    # une composante "vivante" à `scale` = barre H0 dont la mort > scale (+ la barre infinie)
    deaths = d0[:, 1]
    return int(np.sum(deaths > scale))  # inclut l'infinie (mort = inf > scale)


def main():
    rng = np.random.default_rng(7)
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)

    print("\n=== H1 : fraction significative (vs null gaussien apparié) ===")
    print(f"{'W':>5} {'mode':>7} {'frac_sig':>9} {'med_p':>7}")
    for W in (80, 150, 250):
        for mode in ("intra", "multi"):
            frac, medp = h1_frac_sig(X, y, W, mode, categories, rng)
            print(f"{W:>5} {mode:>7} {frac:9.0%} {medp:7.3f}")

    # ---- beta_0 porte-t-il un signal sémantique ? (validation du repli) ----
    print("\n=== beta_0 : intra-sujet vs multi-sujets (le repli est-il informatif ?) ===")
    W = 120
    scale = 0.9  # échelle cosinus intermédiaire
    for mode in ("intra", "multi"):
        b0 = []
        for _ in range(15):
            idx = sample_window(X, y, W, mode, rng, categories)
            if idx is None:
                continue
            b0.append(beta0_at_scale(X[idx], scale))
        b0 = np.array(b0)
        print(f"  {mode:>7} : beta_0 @ scale={scale}  moy={b0.mean():.2f}  (min={b0.min()}, max={b0.max()})")
    print("\n(attendu si le repli est valide : multi-sujets porte un beta_0 > intra-sujet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
