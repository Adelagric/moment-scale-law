"""SPRINT 1 — batterie ÉTENDUE : MMD-RBF, energy distance, KS radial.

Objection de revue : la théorie du papier (fonction test gaussienne = témoin RKHS d'un noyau
RBF) désigne MMD-RBF comme quasi-optimal, et il était absent. De plus, si tout se ramène à
Var(R²), la baseline honnête est un KS sur la loi de Mahalanobis, qui SUBSUME la kurtosis.

On rejoue les deux attaques (collapse ; cov-préservée) avec la batterie complète, IC bootstrap.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import ks_2samp
from ripser import ripser

RNG = np.random.default_rng(0)
CATS = 4


# ---------- détecteurs ----------
def h1_land(X, grid):
    dm = squareform(pdist(X, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    lam = np.zeros_like(grid)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(grid - b, d - grid), 0, None))
    return lam

def knn_density(X, k=5):
    d = squareform(pdist(X, "euclidean")); d.sort(axis=1); return float(np.mean(d[:, k]))

def mardia(X):
    mu = X.mean(0); Pc = X - mu; C = np.cov(X, rowvar=False) + 1e-6*np.eye(X.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))

def mmd_rbf(X, Y, gamma=None):
    """MMD² à noyau RBF (largeur = heuristique de la médiane)."""
    if gamma is None:
        med = np.median(pdist(np.vstack([X, Y])[:400], "euclidean"))
        gamma = 1.0/(2*med**2 + 1e-12)
    Kxx = np.exp(-gamma*cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma*cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma*cdist(X, Y, "sqeuclidean"))
    return float(Kxx.mean() + Kyy.mean() - 2*Kxy.mean())

def energy_dist(X, Y):
    return float(2*cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean())

def maha_radii(X, mu, Si):
    d = X - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, Si, d), 0))

def ks_radial(X, mu, Si, ref_radii):
    return float(ks_2samp(maha_radii(X, mu, Si), ref_radii).statistic)


# ---------- attaques ----------
def collapse(X, alpha=0.5, rho=0.35):
    X = X.copy(); mu = X.mean(0); idx = RNG.choice(len(X), int(rho*len(X)), replace=False)
    X[idx] = mu + (1-alpha)*(X[idx]-mu); return X

def affine_match(X, mu_t, Sig_t):
    mu1 = X.mean(0); S1 = np.cov(X, rowvar=False) + 1e-8*np.eye(X.shape[1])
    A = np.real(sqrtm(Sig_t)) @ np.linalg.inv(np.real(sqrtm(S1)))
    return (X - mu1) @ A.T + mu_t


def auc_ci(sh, sa, det, B=400):
    yv = np.array([0]*len(sh) + [1]*len(sa))
    sv = np.array([s[det] for s in sh] + [s[det] for s in sa])
    def bil(idx):
        a = roc_auc_score(yv[idx], sv[idx]); return max(a, 1-a)
    n = len(yv); boots = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if len(set(yv[idx].tolist())) > 1: boots.append(bil(idx))
    return bil(np.arange(n)), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    z = np.load("emb_bge1024.npz"); X, y = z["X"], z["y"]
    Xr = PCA(n_components=9).fit_transform(X)
    grid = np.linspace(0, np.percentile(pdist(Xr[RNG.choice(len(Xr),300,False)],"euclidean"),95), 60)

    def win(W=250):
        per = W//CATS; idx=[]
        for c in range(CATS): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]

    train = [win() for _ in range(12)]
    ref_pool = np.vstack(train)
    ref_land = np.mean([h1_land(w, grid) for w in train], axis=0)
    Sig = np.mean([np.cov(w, rowvar=False) for w in train], axis=0); Si = np.linalg.pinv(Sig)
    mu = np.mean([w.mean(0) for w in train], axis=0)
    ref_radii = maha_radii(ref_pool, mu, Si)
    ref_sample = ref_pool[RNG.choice(len(ref_pool), 250, replace=False)]

    def sc(w):
        return {
            "Δ_top": float(np.sum(np.abs(h1_land(w, grid) - ref_land))),
            "cov-drift": float(np.linalg.norm(np.cov(w, rowvar=False) - Sig, "fro")),
            "kNN": knn_density(w),
            "Kurt": mardia(w),
            "MMD-RBF": mmd_rbf(w, ref_sample),
            "Energy": energy_dist(w, ref_sample),
            "KS-radial": ks_radial(w, mu, Si, ref_radii),
        }

    N = 20
    H = [win() for _ in range(N)]
    coll = [collapse(win()) for _ in range(N)]
    adap = []
    for _ in range(N):
        h = win(); adap.append(affine_match(collapse(h), h.mean(0), np.cov(h, rowvar=False)))

    sh = [sc(w) for w in H]
    dets = list(sh[0].keys())
    print("AUC bilatérale [IC 95% bootstrap] — bge-1024, dim réduite 9, N=20\n")
    for name, grp in (("collapse", coll), ("cov-preserving", adap)):
        sa = [sc(w) for w in grp]
        print(f"  [{name}]")
        for d in dets:
            a, lo, hi = auc_ci(sh, sa, d)
            star = " <-- max" if a == max(auc_ci(sh, sa, dd)[0] for dd in dets) else ""
            print(f"     {d:>10} = {a:.2f} [{lo:.2f}, {hi:.2f}]{star}")
        print()
    print("Lecture : si MMD/Energy/KS-radial ≥ Δ_top, la dichotomie 'moments d'ordre bas vs")
    print("topologie' du papier est mal posée — MMD est d'ordre infini et n'est pas topologique.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
