"""SPRINT 9 — la quatrième échelle : y a-t-il un PLANCHER DE RÉSOLUTION ?

Hypothèse (issue de la revue) : σ* ≈ max(ε, δ_n), où δ_n est l'espacement inter-points du nuage
réduit. Elle expliquerait la montée monotone de σ*/ε quand ε décroît (0,97 → 1,08 → 1,31) : à
petit ε la bande utile ne peut pas descendre sous la résolution de l'échantillon.

Test : mesurer δ_n, puis descendre ε jusqu'à ~δ_n et en dessous. On compare deux modèles :
    (A) σ* = c·ε                 (loi d'échelle pure)
    (B) σ* = max(c·ε, δ_n)       (loi avec plancher)
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
import exp24_sigma_identified as S

RNG = S.RNG


def spacing(w, k=1):
    """δ_n : distance médiane au k-ème plus proche voisin."""
    D = squareform(pdist(w, "euclidean")); D.sort(axis=1)
    return float(np.median(D[:, k]))


def find_optimum(Xr, y, eps, ngrid=26):
    """Cherche f donnant un optimum intérieur ; renvoie (f, σ*, AUC_max) ou None."""
    for f in (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
        sig, a = S.auc_curve(Xr, y, eps, f, N=14, ngrid=ngrid)
        i = int(np.argmax(a))
        if 0.85 <= a.max() <= 0.995 and 0 < i < len(sig)-1:
            return f, float(sig[i]), float(a.max())
    return None


def main():
    z = np.load("emb_bge1024.npz"); X, y = z["X"], z["y"]
    Xr = PCA(n_components=9).fit_transform(X)
    w0 = S.win(Xr, y, 200)
    scale = np.median(pdist(w0, "euclidean"))
    dn = spacing(w0)
    print(f"nuage réduit : dim 9, W=200")
    print(f"  échelle globale (distance médiane) = {scale:.4f}")
    print(f"  espacement inter-points δ_n (1er voisin, médiane) = {dn:.4f}")
    print(f"  δ_n / échelle = {dn/scale:.3f}\n")

    fracs = (0.25, 0.12, 0.08, 0.05, 0.035)     # la dernière descend sous δ_n
    print(f"{'ε/scale':>8} {'ε':>7} {'ε/δ_n':>7} {'f':>5} {'AUC':>5} {'σ*':>7} {'σ*/ε':>6} {'σ*/δ_n':>7}")
    rows = []
    for fr in fracs:
        eps = fr*scale
        r = find_optimum(Xr, y, eps)
        if r is None:
            print(f"{fr:>8.3f} {eps:>7.4f} {eps/dn:>7.2f}   — pas d'optimum intérieur")
            continue
        f, ss, am = r
        rows.append((eps, ss))
        print(f"{fr:>8.3f} {eps:>7.4f} {eps/dn:>7.2f} {f:>5.2f} {am:>5.2f} {ss:>7.4f} "
              f"{ss/eps:>6.2f} {ss/dn:>7.2f}")

    if len(rows) >= 3:
        e = np.array([r[0] for r in rows]); s_ = np.array([r[1] for r in rows])
        # (A) σ = c·ε
        cA = float(np.sum(s_*e)/np.sum(e*e))
        rA = float(np.sqrt(np.mean((s_-cA*e)**2)))
        # (B) σ = max(c·ε, δ_n) : balayage de c
        best = None
        for c in np.linspace(0.3, 3.0, 271):
            pred = np.maximum(c*e, dn)
            r_ = float(np.sqrt(np.mean((s_-pred)**2)))
            if best is None or r_ < best[1]: best = (float(c), r_)
        print(f"\n  modèle (A) σ*=c·ε        : c={cA:.2f}   RMSE={rA:.4f}")
        print(f"  modèle (B) σ*=max(c·ε,δ_n): c={best[0]:.2f}   RMSE={best[1]:.4f}")
        better = "B (plancher)" if best[1] < rA else "A (échelle pure)"
        print(f"  -> meilleur ajustement : {better}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
