"""EXP 0 — beta_0 mesuré CORRECTEMENT (multi-échelle).

Correction du test précédent (une seule échelle => beta_0=1 trivial). Ici on regarde
la PERSISTANCE des barres H0 : à quelle échelle les clusters fusionnent. Signal du
repli : un window multi-sujets doit garder plusieurs composantes séparées longtemps
(barres H0 longues) là où un window mono-sujet fusionne tôt.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from exp0_gate import build_embeddings, cosine_dm
from exp0_sweep import sample_window
from ripser import ripser


def h0_bars(X):
    dm = cosine_dm(X)
    d0 = ripser(dm, maxdim=0, distance_matrix=True)["dgms"][0]
    deaths = np.sort(d0[np.isfinite(d0[:, 1]), 1])  # naissances=0 pour H0
    return deaths


def main():
    rng = np.random.default_rng(11)
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)
    W = 120

    # Concentration des distances (diagnostic malédiction de la dimension)
    samp = X[rng.choice(len(X), 300, replace=False)]
    dd = pdist(samp, metric="cosine")
    print(f"distances cosinus : moy={dd.mean():.3f} std={dd.std():.3f} "
          f"[p5={np.quantile(dd,.05):.3f}, p95={np.quantile(dd,.95):.3f}]")

    print("\n=== persistance H0 : intra vs multi (top-4 fusions les plus tardives) ===")
    print(f"{'mode':>7} {'k_sep':>6} {'top4 morts H0 (échelles de fusion)':>40}")
    stats = {}
    for mode in ("intra", "multi"):
        seps, tops = [], []
        for _ in range(20):
            idx = sample_window(X, y, W, mode, rng, categories)
            if idx is None:
                continue
            deaths = h0_bars(X[idx])
            top4 = deaths[-4:][::-1] if len(deaths) >= 4 else deaths[::-1]
            tops.append(top4)
            # k_sep = nb de clusters "réels" : barres H0 > seuil = moyenne+2σ des morts courtes
            thr = np.quantile(deaths, 0.5) + 2 * np.std(deaths[deaths < np.quantile(deaths, 0.9)] + 1e-9)
            seps.append(int(np.sum(deaths > thr)) + 1)  # +1 barre infinie
        seps = np.array(seps)
        tops = np.array([t for t in tops if len(t) == 4])
        stats[mode] = (seps, tops)
        print(f"{mode:>7} {seps.mean():6.2f}  {np.round(tops.mean(axis=0),3)}")

    si, sm = stats["intra"][0], stats["multi"][0]
    ti, tm = stats["intra"][1].mean(axis=0), stats["multi"][1].mean(axis=0)
    print(f"\nk_sep  : intra={si.mean():.2f}  multi={sm.mean():.2f}")
    print(f"fusion la plus tardive (mort H0 max) : intra={ti[0]:.3f}  multi={tm[0]:.3f}")
    verdict = tm[0] > ti[0] * 1.05 or sm.mean() > si.mean() + 0.5
    print("\nREPLI beta_0 :",
          "informatif ✅ (multi se sépare plus que intra)" if verdict
          else "NON informatif ❌ (β0 ne distingue pas les sujets à ces échelles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
