"""EXP 2d — la niche topologique est-elle IRRÉDUCTIBLE, ou capturable par les moments ?

Test décisif : sur cercle vs disque (cov égale par construction, différence PUREMENT
topologique), la kurtosis de Mardia (ordre 4) sépare-t-elle aussi ?

  - Δ_top sépare, kurtosis PAS  => niche topologique IRRÉDUCTIBLE : la thèse tient.
  - kurtosis sépare AUSSI       => tout se réduit aux moments : la TDA n'a aucun
                                   avantage même en principe. L'axe détection tombe.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from exp1c_matched import circle, disk, h1_landscape_euclid
from exp2b_scrutiny import knn_density, mardia_kurtosis


def main():
    train = [circle() for _ in range(20)]
    land_ref = np.mean([h1_landscape_euclid(w) for w in train], axis=0)

    def scores(w):
        return {
            "Δ_top": float(np.sum(np.abs(h1_landscape_euclid(w) - land_ref))),
            "Kurt_Mardia": mardia_kurtosis(w),
            "B3_kNN": knn_density(w),
        }

    N = 24
    healthy = [circle() for _ in range(N)]
    attacked = [disk() for _ in range(N)]
    sh = [scores(w) for w in healthy]; sa = [scores(w) for w in attacked]

    print(f"=== cercle vs disque — qui sépare ? (N={N}) ===")
    for det in sh[0]:
        yv = [0]*N + [1]*N
        sv = [s[det] for s in sh] + [s[det] for s in sa]
        auc = roc_auc_score(yv, sv)
        print(f"  {det:>12} : AUC={auc:.3f}  (bilatéral {max(auc,1-auc):.3f})")

    print("\nVERDICT :")
    print("  Kurt_Mardia bilatéral ≈0.5 & Δ_top≈1 => niche IRRÉDUCTIBLE, thèse vivante.")
    print("  Kurt_Mardia bilatéral ≫0.5          => moments suffisent, détection TDA morte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
