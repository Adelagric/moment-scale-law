"""SPRINT 8 — les trois écarts promesse/preuve de la revue finale.

(1) σ*/ε avec optimum IDENTIFIABLE sur d'AUTRES embedders/corpus (l'intro le promet ; seule la
    mesure retirée du plateau l'avait fait).
(2) ADVERSAIRE QUI VISE MMD : jusqu'ici il optimisait contre {μ,Σ,kNN,b} sans jamais toucher au
    détecteur vainqueur. On ajoute MMD (bande appariée) à l'objectif du gradient.
(3) coût de KS-radial (cellule au tiret en Table 2).
"""
from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist
import exp19_full_battery as E
import exp24_sigma_identified as S
import exp20_truly_adaptive as T

RNG = E.RNG


# ---------- (1) σ* identifiable, autres setups ----------
def sigma_star_multi(cache, dim, label):
    z = np.load(cache); X, y = z["X"], z["y"]
    Xr = PCA(n_components=dim).fit_transform(X)
    scale = np.median(pdist(S.win(Xr, y), "euclidean"))
    out = []
    for frac in (0.10, 0.15, 0.22):
        eps = frac*scale
        for f in (0.05, 0.035, 0.025, 0.018):
            sig, a = S.auc_curve(Xr, y, eps, f, N=14, ngrid=20)
            i = int(np.argmax(a))
            if 0.85 <= a.max() <= 0.995 and 0 < i < len(sig)-1:
                out.append((frac, eps, f, a.max(), sig[i], sig[i]/eps)); break
    print(f"\n  [{label}]")
    for frac, eps, f, am, s_, r in out:
        print(f"    ε/scale={frac:.2f}  ε={eps:.3f}  f={f:.3f}  AUC_max={am:.2f}  σ*={s_:.3f}  σ*/ε={r:.2f}")
    if out:
        r = np.array([o[5] for o in out])
        print(f"    -> médiane σ*/ε = {np.median(r):.2f}  (n={len(r)})")
    return [o[5] for o in out]


# ---------- (2) adversaire qui VISE MMD ----------
def mmd_t(Bt, Rt, sigma):
    g = 1.0/(2*sigma**2)
    def k(a, b): return torch.exp(-g*torch.cdist(a, b)**2)
    return k(Bt, Bt).mean() + k(Rt, Rt).mean() - 2*k(Bt, Rt).mean()


def mmd_aware_attack(h, ref, sigma, iters=2500):
    """Optimise contre {μ, Σ, kNN, b} ET MMD(bande appariée) — le détecteur vainqueur est visé."""
    A = torch.tensor(h); Rt = torch.tensor(ref)
    muA = A.mean(0); Ac = A-muA; CA = (Ac.T@Ac)/(len(A)-1)
    knnA = T.knn_sorted_t(A).detach(); kurtA = T.mardia_t(A).detach()
    B = torch.tensor(E.collapse(h).copy(), requires_grad=True)
    opt = torch.optim.Adam([B], lr=5e-3)
    for _ in range(iters):
        opt.zero_grad()
        mu = B.mean(0); Bc = B-mu; C = (Bc.T@Bc)/(len(B)-1)
        loss = (100*((mu-muA)**2).sum() + 200*((C-CA)**2).sum()
                + 5*((T.knn_sorted_t(B)-knnA)**2).mean()
                + 0.5*(T.mardia_t(B)-kurtA)**2
                + 300*mmd_t(B, Rt, sigma))          # <-- MMD dans l'objectif
        loss.backward(); opt.step()
    return B.detach().numpy()


def main():
    print("=== (1) σ*/ε identifiable sur d'autres embedders / corpus ===")
    allr = []
    for cache, dim, lab in (("emb_cache.npz", 8, "MiniLM-384 / 20NG"),
                            ("emb_agnews_bge1024.npz", 9, "bge-1024 / AG News")):
        try: allr += sigma_star_multi(cache, dim, lab)
        except FileNotFoundError: print(f"  [{lab}] cache absent")
    if allr:
        r = np.array(allr)
        print(f"\n  tous setups confondus : médiane={np.median(r):.2f}  min={r.min():.2f}  max={r.max():.2f}")

    print("\n=== (2) adversaire qui VISE MMD (bande appariée dans l'objectif) ===")
    z = np.load("emb_bge1024.npz"); X, y = z["X"], z["y"]
    Xr = PCA(n_components=9).fit_transform(X)
    def win(): return S.win(Xr, y, 200)
    ref = win(); med = np.median(pdist(ref, "euclidean")); sig = med/4
    N = 14
    H = [win() for _ in range(N)]
    A = [mmd_aware_attack(win(), ref, sig) for _ in range(N)]
    dets = {
        "MMD (visé)":   lambda w: E.mmd_rbf(w, ref, 1.0/(2*sig**2)),
        "kurtosis":     E.mardia,
        "kNN":          E.knn_density,
        "cov-drift":    lambda w: float(np.linalg.norm(np.cov(w,rowvar=False)-np.cov(ref,rowvar=False),"fro")),
        "total-pers":   lambda w: (lambda f: float((f[:,1]-f[:,0]).sum()) if len(f) else 0.0)(
                            __import__("exp25_propagate").diagram(w)),
    }
    print(f"  {'détecteur':>14} {'AUC':>6}")
    for k, fn in dets.items():
        sh = [fn(w) for w in H]; sa = [fn(w) for w in A]
        v = roc_auc_score([0]*N+[1]*N, sh+sa)
        print(f"  {k:>14} {max(v,1-v):>6.2f}")

    print("\n=== (3) coût de KS-radial ===")
    Sig = np.cov(ref, rowvar=False); Si = np.linalg.pinv(Sig); mu = ref.mean(0)
    rad = E.maha_radii(ref, mu, Si)
    ws = [win() for _ in range(20)]
    t0 = time.perf_counter()
    for w in ws: E.ks_radial(w, mu, Si, rad)
    ms = (time.perf_counter()-t0)/len(ws)*1000
    print(f"  KS-radial : {ms:.3f} ms/fenêtre  (kurtosis = 0.04 ms -> ×{ms/0.04:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
