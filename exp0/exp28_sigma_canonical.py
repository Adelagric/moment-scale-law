"""SPRINT 10 — mesure CANONIQUE de σ*/ε : un seul protocole, trois réglages, avec répétitions.

Motif : deux mesures antérieures du même réglage (bge/20NG) utilisaient des grilles de f
différentes et donnaient {0,97;1,08;1,31} puis {1,28;1,15;1,44} — incomparables, et la « médiane
sur les trois réglages » publiée excluait en fait le réglage d'origine.

Ici : protocole IDENTIQUE partout (même grille de f, même fenêtre d'acceptation, même grille de σ),
3 réglages × 3 échelles × R répétitions, pour donner la médiane, la plage ET la variance inter-runs.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import exp24_sigma_identified as S

RNG = S.RNG
F_GRID = (0.05, 0.035, 0.025, 0.018, 0.012)     # identique pour tous les réglages
AUC_WINDOW = (0.85, 0.995)
NGRID = 24
NWIN = 14


def one_ratio(Xr, y, eps):
    """Un ratio σ*/ε avec optimum intérieur, ou None."""
    for f in F_GRID:
        sig, a = S.auc_curve(Xr, y, eps, f, N=NWIN, ngrid=NGRID)
        i = int(np.argmax(a))
        if AUC_WINDOW[0] <= a.max() <= AUC_WINDOW[1] and 0 < i < len(sig)-1:
            return float(sig[i]/eps), f, float(a.max())
    return None


def main():
    setups = [("emb_bge1024.npz", 9, "bge-1024 / 20NG"),
              ("emb_cache.npz", 8, "MiniLM-384 / 20NG"),
              ("emb_agnews_bge1024.npz", 9, "bge-1024 / AG News")]
    FRACS = (0.25, 0.15, 0.08)
    REPS = 3
    print(f"protocole unique : f∈{F_GRID}, AUC∈{AUC_WINDOW}, {NGRID} valeurs de σ, "
          f"{NWIN} fenêtres, {REPS} répétitions\n")
    print(f"{'réglage':>20} {'ε/scale':>8} {'ratios (reps)':>26} {'médiane':>8} {'étendue':>8}")
    allr = []
    for cache, dim, lab in setups:
        try:
            z = np.load(cache); X, y = z["X"], z["y"]
        except FileNotFoundError:
            print(f"{lab:>20}  cache absent"); continue
        Xr = PCA(n_components=dim).fit_transform(X)
        scale = np.median(pdist(S.win(Xr, y), "euclidean"))
        for fr in FRACS:
            eps = fr*scale
            reps = []
            for _ in range(REPS):
                r = one_ratio(Xr, y, eps)
                if r: reps.append(r[0])
            if not reps:
                print(f"{lab:>20} {fr:>8.2f}   — pas d'optimum intérieur"); continue
            allr += reps
            rr = np.array(reps)
            print(f"{lab:>20} {fr:>8.2f} {str(np.round(rr,2)):>26} "
                  f"{np.median(rr):>8.2f} {rr.max()-rr.min():>8.2f}")
    r = np.array(allr)
    print(f"\n=== ENSEMBLE CANONIQUE (n={len(r)}) ===")
    print(f"  médiane = {np.median(r):.2f}")
    print(f"  plage   = [{r.min():.2f}, {r.max():.2f}]")
    print(f"  quartiles = [{np.percentile(r,25):.2f}, {np.percentile(r,75):.2f}]")
    print(f"  écart-type inter-runs (moyenne des étendues par cellule) : voir colonne 'étendue'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
