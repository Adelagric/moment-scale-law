"""EXP 0 — test corrigé, en espace réduit, avec agrégation de population.

Corrections vs runs précédents :
  - réduction PCA à la dimension intrinsèque AVANT TDA (Rips brut 384-d = concentration) ;
  - n_null=99 => p atteignable jusqu'à 0.01 (avant, 1/16=0.063 rendait p<0.05 impossible) ;
  - agrégation de POPULATION : la distribution des p par-fenêtre est-elle décalée
    sous l'uniforme ? (un signal faible mais réel se voit en population, pas par-fenêtre)
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from exp0_gate import build_embeddings, window_pvalue
from exp0_sweep import sample_window


def run(Xr, y, W, mode, categories, rng, n_windows=14, n_null=99):
    ps = []
    for _ in range(n_windows):
        idx = sample_window(Xr, y, W, mode, rng, categories)
        if idx is None:
            continue
        _, _, p = window_pvalue(Xr[idx], n_null, rng)
        ps.append(p)
    return np.array(ps)


def main():
    rng = np.random.default_rng(101)
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)

    for k in (8, 12):
        Xr = PCA(n_components=k).fit_transform(X)
        print(f"\n########## espace réduit PCA dim={k} ##########")
        for mode, W in (("intra", 150), ("multi", 250)):
            ps = run(Xr, y, W, mode, categories, rng)
            if len(ps) == 0:
                print(f"  {mode:>5} W={W}: (pas assez de données)")
                continue
            frac = np.mean(ps < 0.05)
            # test de population : H0 = p ~ Uniform(0,1). Si H1 réel > null, p décalé vers 0.
            # test du signe (p < 0.5) + KS contre uniforme, unilatéral via mediane.
            sign_p = stats.binomtest(int(np.sum(ps < 0.5)), len(ps), 0.5,
                                     alternative="greater").pvalue
            ks = stats.kstest(ps, "uniform").pvalue
            print(f"  {mode:>5} W={W}: med_p={np.median(ps):.3f}  frac(p<.05)={frac:.0%}  "
                  f"signe(p<.5)_pval={sign_p:.3f}  KS_vs_unif={ks:.3f}")

    print("\nInterprétation :")
    print("  signe_pval petit (<0.05)  => H1 réel systématiquement > null : signal de population réel.")
    print("  med_p ~0.5 & KS grand     => indistinguable du null : négatif robuste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
