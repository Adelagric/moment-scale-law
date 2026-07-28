"""PAPER 1 — réplique à dimension ambiante croissante (384 → 768 → 1024).

Question load-bearing : le négatif de détection + la redondance persistance↔moments
tiennent-ils à plus haute dimension, ou sont-ils un artefact du 384-d ?

Pour chaque embedder : (1) dim intrinsèque (TwoNN) vs dim ambiante ; (2) redondance en
espace réduit (attaque collapse + batterie B6/Maha/kNN/kurtosis vs Δ_top) ;
(3) même comparaison en espace BRUT (où tout le monde souffre de la concentration).
"""
from __future__ import annotations
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import squareform, pdist
from ripser import ripser

RNG = np.random.default_rng(0)
CATS = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]


def embed(model_name, cache, per_cat=200):
    import os
    if os.path.exists(cache):
        z = np.load(cache); print(f"[cache] {cache} {z['X'].shape}"); return z["X"], z["y"]
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    data = fetch_20newsgroups(subset="train", categories=CATS,
                              remove=("headers", "footers", "quotes"), shuffle=True, random_state=0)
    texts, labels = [], []
    for cid in range(len(CATS)):
        idx = [i for i, t in enumerate(data.target) if t == cid and len(data.data[i].split()) >= 20][:per_cat]
        texts += [data.data[i] for i in idx]; labels += [cid]*len(idx)
    print(f"[embed] {len(texts)} docs -> {model_name}")
    m = SentenceTransformer(model_name)
    X = np.asarray(m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    y = np.asarray(labels); np.savez(cache, X=X, y=y); print(f"[embed] {X.shape}")
    return X, y


def twonn(X, rng):
    from scipy.spatial.distance import cdist
    idx = rng.choice(len(X), min(len(X), 500), replace=False)
    D = cdist(X[idx], X[idx]); D.sort(axis=1)
    r1, r2 = D[:, 1], D[:, 2]; mask = r1 > 1e-12
    mu = (r2[mask]/r1[mask]); mu = mu[mu > 1+1e-9]
    F = np.arange(1, len(mu)+1)/len(mu); x = np.log(np.sort(mu)); ylog = -np.log(1-F[:len(x)]+1e-12)
    return float(np.sum(x*ylog)/np.sum(x*x))


def h1_land(X, grid):
    dm = squareform(pdist(X, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    lam = np.zeros_like(grid)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(grid-b, d-grid), 0, None))
    return lam

def knn_density(X, k=5):
    d = squareform(pdist(X, "euclidean")); d.sort(axis=1); return float(np.mean(d[:, k]))

def mardia(X):
    mu = X.mean(0); Pc = X-mu; C = np.cov(X, rowvar=False)+1e-6*np.eye(X.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))

def collapse(X, alpha=0.5, rho=0.35):
    X = X.copy(); mu = X.mean(0); idx = RNG.choice(len(X), int(rho*len(X)), replace=False)
    X[idx] = mu + (1-alpha)*(X[idx]-mu); return X


def battery(Xr, y, tag):
    grid = np.linspace(0, np.percentile(pdist(Xr[np.random.default_rng(0).choice(len(Xr),300,False)],"euclidean"),95), 60)
    def win(W=250):
        per = W//len(CATS); idx=[]
        for c in range(len(CATS)): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]
    train = [win() for _ in range(12)]
    ref = np.mean([h1_land(w, grid) for w in train], axis=0)
    Sig = np.mean([np.cov(w, rowvar=False) for w in train], axis=0); Si = np.linalg.pinv(Sig)
    mu = np.mean([w.mean(0) for w in train], axis=0)
    def sc(w):
        diff = w-mu
        return {"Δ_top": float(np.sum(np.abs(h1_land(w, grid)-ref))),
                "B6": float(np.linalg.norm(np.cov(w, rowvar=False)-Sig, "fro")),
                "Maha": float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, Si, diff)))),
                "kNN": knn_density(w), "Kurt": mardia(w)}
    N = 16; H = [win() for _ in range(N)]; A = [collapse(win()) for _ in range(N)]
    sh=[sc(w) for w in H]; sa=[sc(w) for w in A]
    print(f"  [{tag}] " + "  ".join(
        f"{d}={max(roc_auc_score([0]*N+[1]*N,[s[d] for s in sh]+[s[d] for s in sa]), 1-roc_auc_score([0]*N+[1]*N,[s[d] for s in sh]+[s[d] for s in sa])):.2f}"
        for d in sh[0]))


def main():
    model, dim, cache = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    X, y = embed(model, cache)
    rng = np.random.default_rng(1)
    did = twonn(X, rng)
    p = PCA().fit(X); dpca = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.9)+1)
    print(f"\n=== {model}  (ambiant={X.shape[1]}) ===")
    print(f"dim intrinsèque TwoNN ≈ {did:.1f}   |   PCA 90% var = {dpca} axes")
    k = max(4, int(round(did)))
    Xr = PCA(n_components=k).fit_transform(X)
    print(f"AUC bilatérale (redondance : kurtosis/kNN ≈ Δ_top => TDA redondante)")
    battery(Xr, y, f"réduit dim={k}")
    battery(X, y, "BRUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
