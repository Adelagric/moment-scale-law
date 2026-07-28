"""PAPER 1 — robustesse : 2ᵉ corpus (AG News, presse) + IC bootstrap sur les AUC.

Rejoue collapse ET adaptatif (cov-préservé) sur 20NG et AG News, en bge-1024, avec
intervalles de confiance bootstrap 95% sur chaque AUC. Objectif : montrer que la
domination kurtosis ≥ Δ_top n'est ni un artefact de corpus ni du bruit d'échantillonnage.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import squareform, pdist
from ripser import ripser

RNG = np.random.default_rng(0)
MODEL = "BAAI/bge-large-en-v1.5"


def build_agnews(cache="emb_agnews_bge1024.npz", per_cls=200):
    import os
    if os.path.exists(cache):
        z = np.load(cache); print(f"[cache] {cache} {z['X'].shape}"); return z["X"], z["y"]
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    ds = load_dataset("ag_news", split="train")
    texts, labels = [], []
    seen = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in ds:
        c = row["label"]
        if seen[c] < per_cls and len(row["text"].split()) >= 15:
            texts.append(row["text"]); labels.append(c); seen[c] += 1
        if all(v >= per_cls for v in seen.values()):
            break
    print(f"[embed] AG News {len(texts)} docs -> {MODEL}")
    m = SentenceTransformer(MODEL)
    X = np.asarray(m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    y = np.asarray(labels); np.savez(cache, X=X, y=y); return X, y


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
    mu = X.mean(0); Pc = X-mu; C = np.cov(X, rowvar=False)+1e-6*np.eye(X.shape[1])
    return float(np.mean(np.einsum("ij,jk,ik->i", Pc, np.linalg.inv(C), Pc)**2))

def collapse(X, alpha=0.5, rho=0.35):
    X = X.copy(); mu = X.mean(0); idx = RNG.choice(len(X), int(rho*len(X)), replace=False)
    X[idx] = mu + (1-alpha)*(X[idx]-mu); return X

def affine_match(X, mu_t, Sig_t):
    mu1 = X.mean(0); S1 = np.cov(X, rowvar=False)+1e-8*np.eye(X.shape[1])
    A = np.real(sqrtm(Sig_t)) @ np.linalg.inv(np.real(sqrtm(S1)))
    return (X-mu1) @ A.T + mu_t


def auc_ci(sh, sa, det, B=500):
    yv = np.array([0]*len(sh) + [1]*len(sa))
    sv = np.array([s[det] for s in sh] + [s[det] for s in sa])
    def bilat(idx): a = roc_auc_score(yv[idx], sv[idx]); return max(a, 1-a)
    boots = []
    n = len(yv)
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if len(set(yv[idx].tolist())) < 2: continue
        boots.append(bilat(idx))
    return bilat(np.arange(n)), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def run(X, y, tag, ncls, dim=9):
    Xr = PCA(n_components=dim).fit_transform(X)
    grid = np.linspace(0, np.percentile(pdist(Xr[np.random.default_rng(0).choice(len(Xr),300,False)], "euclidean"), 95), 60)
    def win(W=250):
        per = W//ncls; idx=[]
        for c in range(ncls): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]
    train = [win() for _ in range(12)]
    ref = np.mean([h1_land(w, grid) for w in train], axis=0)
    Sig = np.mean([np.cov(w, rowvar=False) for w in train], axis=0); Si = np.linalg.pinv(Sig)
    mu = np.mean([w.mean(0) for w in train], axis=0)
    def sc(w):
        diff = w-mu
        return {"Δ_top": float(np.sum(np.abs(h1_land(w, grid)-ref))),
                "B6": float(np.linalg.norm(np.cov(w, rowvar=False)-Sig, "fro")),
                "kNN": knn_density(w), "Kurt": mardia(w)}
    N = 20
    H = [win() for _ in range(N)]
    coll = [collapse(win()) for _ in range(N)]
    adap = [affine_match(collapse(win()), (h:=win()).mean(0), np.cov(h, rowvar=False)) for _ in range(N)]
    sh = [sc(w) for w in H]
    for atk_name, grp in (("collapse", coll), ("adaptatif", adap)):
        sa = [sc(w) for w in grp]
        parts = []
        for det in sh[0]:
            a, lo, hi = auc_ci(sh, sa, det)
            parts.append(f"{det}={a:.2f}[{lo:.2f},{hi:.2f}]")
        print(f"  [{tag} | {atk_name:>9}] " + "  ".join(parts))


def main():
    print("=== 20NG (bge-1024) ===")
    z = np.load("emb_bge1024.npz"); run(z["X"], z["y"], "20NG", 4)
    print("\n=== AG News (bge-1024) ===")
    X, y = build_agnews(); run(X, y, "AGNews", 4)
    print("\nLecture : sur les DEUX corpus, IC de Kurt au-dessus (ou chevauchant par le haut)")
    print("celui de Δ_top => domination robuste au corpus ET à l'échantillonnage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
