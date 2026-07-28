"""EXP 2b — mise à l'épreuve d'Exp 2 sur ses 3 points faibles.

(1) DÉGÂT SÉMANTIQUE indépendant des détecteurs : taux de collision (paires quasi-
    dupliquées = perte d'information / de résolution retrieval). Invariant impossible
    à restaurer par une correction de covariance si le collapse a vraiment fusionné
    des points.
(2) BASELINES FORTS : kNN-densité (B3) + kurtosis multivariée de Mardia. L'attaque
    préserve les ordres 1-2 mais PAS 3-4 ni la densité locale. Si ces détecteurs
    attrapent l'adaptatif, la TDA est redondante.
(3) CONTRÔLE D'ARTEFACT AFFINE : on applique la MÊME matrice affine de l'attaque à une
    fenêtre saine INTACTE (trou non bouché). Si Δ_top s'y allume, il détecte la
    distorsion affine, pas la topologie.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from exp0_gate import build_embeddings
from ripser import ripser

RNG = np.random.default_rng(313)
GRID = np.linspace(0.0, 3.0, 60)


def h1_land(X):
    dm = squareform(pdist(X, "euclidean"))
    dgm = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(GRID - b, d - GRID), 0, None))
    return lam


def collapse_raw(X, alpha=0.5, rho=0.35):
    X = X.copy(); mu = X.mean(0)
    idx = RNG.choice(len(X), int(rho * len(X)), replace=False)
    X[idx] = mu + (1 - alpha) * (X[idx] - mu)
    return X


def affine_mat(Xraw, mu_t, Sig_t):
    mu1 = Xraw.mean(0)
    Sig1 = np.cov(Xraw, rowvar=False) + 1e-8 * np.eye(Xraw.shape[1])
    A = np.real(sqrtm(Sig_t)) @ np.linalg.inv(np.real(sqrtm(Sig1)))
    return A, mu1


def apply_affine(X, A, mu_src, mu_t):
    return (X - mu_src) @ A.T + mu_t


# ---- métrique de dégât (indépendante des détecteurs) ----
def collision_rate(X, eps):
    d = pdist(X, "euclidean")
    return float(np.mean(d < eps))


# ---- détecteurs ----
def knn_density(X, k=5):
    d = squareform(pdist(X, "euclidean"))
    d.sort(axis=1)
    return float(np.mean(d[:, k]))  # distance au k-ème voisin (petit = dense)

def mardia_kurtosis(X):
    mu = X.mean(0)
    C = np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1])
    Ci = np.linalg.inv(C); diff = X - mu
    m = np.einsum("ij,jk,ik->i", diff, Ci, diff)
    return float(np.mean(m ** 2))


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
    eps_dup = np.quantile(pdist(train[0], "euclidean"), 0.01)  # seuil quasi-duplicat

    def scores(w):
        diff = w - mu_ref
        return {
            "Δ_top": float(np.sum(np.abs(h1_land(w) - land_ref))),
            "B6_spec": float(np.linalg.norm(np.cov(w, rowvar=False) - Sig_ref, "fro")),
            "Maha": float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, SigInv, diff)))),
            "B3_kNN": knn_density(w),
            "Kurt_Mardia": mardia_kurtosis(w),
        }

    N = 24
    H = [healthy() for _ in range(N)]
    naive, adaptive, affine_ctrl = [], [], []
    dmg = {"healthy": [], "naive": [], "adaptive": [], "affine_ctrl": []}
    for h in H:
        mu_h, Sig_h = h.mean(0), np.cov(h, rowvar=False)
        nv = collapse_raw(h)
        A, mu_c = affine_mat(nv, mu_h, Sig_h)
        ad = apply_affine(nv, A, mu_c, mu_h)                 # attaque adaptative
        hp = healthy()                                       # fenêtre saine fraîche
        ac = apply_affine(hp, A, hp.mean(0), hp.mean(0))     # MÊME affine, trou intact
        naive.append(nv); adaptive.append(ad); affine_ctrl.append(ac)
        dmg["healthy"].append(collision_rate(h, eps_dup))
        dmg["naive"].append(collision_rate(nv, eps_dup))
        dmg["adaptive"].append(collision_rate(ad, eps_dup))
        dmg["affine_ctrl"].append(collision_rate(ac, eps_dup))

    print("=== (1) DÉGÂT SÉMANTIQUE : taux de collision (quasi-duplicats) ===")
    for k in dmg:
        print(f"  {k:>12} : {np.mean(dmg[k]):.4f}")
    print("  -> adaptatif ≫ healthy = dégât réel survit à la correction affine ;")
    print("     adaptatif ≈ healthy = attaque INERTE (Exp 2 sans objet).")

    sh = [scores(w) for w in H]
    dets = list(sh[0].keys())
    print(f"\n=== (2)+(3) AUC sain vs attaqué (N={N}) ===")
    print(f"{'attaque':>12}" + "".join(f"{d:>13}" for d in dets))
    for label, grp in (("naive", naive), ("adaptive", adaptive), ("affine_ctrl", affine_ctrl)):
        sa = [scores(w) for w in grp]
        row = f"{label:>12}"
        for d in dets:
            yv = [0]*N + [1]*N
            sv = [s[d] for s in sh] + [s[d] for s in sa]
            row += f"{roc_auc_score(yv, sv):>13.3f}"
        print(row)

    print("\nGrille de lecture :")
    print("  (3) affine_ctrl Δ_top ≈0.5  => 0.77 adaptatif = vraie topologie, pas artefact.")
    print("  (2) B3/Kurt adaptatif ≫0.5  => baseline classique suffit, TDA redondante.")
    print("  (1) dégât adaptatif ~healthy => attaque inerte, tout l'argument tombe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
