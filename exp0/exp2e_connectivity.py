"""EXP 2e — dernier test équitable : connexité (β_0) à densité locale appariée.

Le vrai terrain de la persistance : une différence GLOBALE (connexe vs 2 composantes)
que les statistiques LOCALES ne peuvent voir.

  - sain  = 2 clumps + un PONT de points (même densité locale) => connexe, β_0=1
  - attaque = 2 clumps seuls (masse du pont réabsorbée), PUIS correction affine de
    covariance => déconnecté β_0=2, mais (μ,Σ) appariés au sain.

Si Δ_top(H0) sépare alors que kNN-densité ET kurtosis ET B6 sont au hasard => niche
IRRÉDUCTIBLE : la thèse détection ressuscite. Si kNN/kurtosis séparent aussi => le
motif tient jusqu'au bout, la persistance n'apporte rien.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
from exp2b_scrutiny import knn_density, mardia_kurtosis

RNG = np.random.default_rng(4242)
DIM = 8
GRID = np.linspace(0.0, 5.0, 80)


def landscape(dgm):
    fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(GRID - b, d - GRID), 0, None))
    return lam


def topo_score(X, ref0, ref1):
    dm = squareform(pdist(X, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"]
    s0 = np.sum(np.abs(landscape(dg[0]) - ref0))
    s1 = np.sum(np.abs(landscape(dg[1]) - ref1))
    return s0 + s1


# Construction "boucles" : PAS d'extrémités. Densité par arc identique partout =>
# chaque point voit un arc local identique => kNN vraiment aveugle. Seuls β_0 (1 vs 2)
# et le nombre de boucles diffèrent, globalement.
WIDTH, N = 0.05, 260


def loop(cx, r, n):
    t = RNG.uniform(0, 2 * np.pi, n)
    P = RNG.normal(0, WIDTH, size=(n, DIM))
    P[:, 0] += cx + r * np.cos(t)
    P[:, 1] += r * np.sin(t)
    return P


def healthy():
    # une grande boucle rayon 2 : arclength 4π, densité N/4π
    return loop(0.0, 2.0, N)


def attacked():
    # deux petites boucles rayon 1 (arclength 2·2π=4π, même densité N/4π) => β_0=2
    half = N // 2
    return np.vstack([loop(-1.6, 1.0, half), loop(+1.6, 1.0, N - half)])


def affine_match(X, mu_t, Sig_t):
    mu1 = X.mean(0)
    S1 = np.cov(X, rowvar=False) + 1e-8 * np.eye(DIM)
    A = np.real(sqrtm(Sig_t)) @ np.linalg.inv(np.real(sqrtm(S1)))
    return (X - mu1) @ A.T + mu_t


def main():
    train = [healthy() for _ in range(15)]
    dgs = [ripser(squareform(pdist(w, "euclidean")), maxdim=1, distance_matrix=True)["dgms"]
           for w in train]
    ref0 = np.mean([landscape(d[0]) for d in dgs], axis=0)
    ref1 = np.mean([landscape(d[1]) for d in dgs], axis=0)
    Sig_ref = np.mean([np.cov(w, rowvar=False) for w in train], axis=0)
    SigInv = np.linalg.pinv(Sig_ref); mu_ref = np.mean([w.mean(0) for w in train], axis=0)

    N = 24
    H = [healthy() for _ in range(N)]
    A = []
    for h in H:
        raw = attacked()
        A.append(affine_match(raw, h.mean(0), np.cov(h, rowvar=False)))  # apparie (μ,Σ)

    # appariement : covariance et densité locale
    dcov = np.mean([np.linalg.norm(np.cov(a, rowvar=False)-np.cov(h, rowvar=False), "fro")
                    for a, h in zip(A, H)])
    dknn = abs(np.mean([knn_density(a) for a in A]) - np.mean([knn_density(h) for h in H]))
    print(f"appariement : |Δcov|={dcov:.4f}   |Δ(kNN-densité moy)|={dknn:.4f}")

    def scores(w):
        diff = w - mu_ref
        return {
            "Δ_top": topo_score(w, ref0, ref1),
            "B6_spec": float(np.linalg.norm(np.cov(w, rowvar=False) - Sig_ref, "fro")),
            "Maha": float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, SigInv, diff)))),
            "B3_kNN": knn_density(w),
            "Kurt_Mardia": mardia_kurtosis(w),
        }

    sh = [scores(w) for w in H]; sa = [scores(w) for w in A]
    print(f"\n=== AUC connexe(sain) vs déconnecté(attaqué) (N={N}) ===")
    for det in sh[0]:
        yv = [0]*N + [1]*N
        sv = [s[det] for s in sh] + [s[det] for s in sa]
        auc = roc_auc_score(yv, sv)
        print(f"  {det:>12} : AUC={auc:.3f}  (bilatéral {max(auc,1-auc):.3f})")

    print("\nVERDICT :")
    print("  Δ_top≫0.5 & kNN/Kurt/B6 bilatéral≈0.5 => NICHE IRRÉDUCTIBLE, détection vivante.")
    print("  kNN ou Kurt bilatéral≫0.5           => motif confirmé, persistance redondante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
