"""SPRINT 2 — l'attaque VRAIMENT adaptative (objection 5 de la revue).

L'ancienne attaque préservait {μ, Σ} puis on « découvrait » que la kurtosis la voit : tautologique.
Ici l'adversaire s'optimise contre la statistique du DÉFENSEUR, kurtosis incluse :
on optimise le nuage attaqué pour apparier {μ, Σ, distribution kNN, kurtosis de Mardia} de sa
pré-image saine, SANS jamais toucher à la persistance.

Question : Δ_top gagne-t-il alors ? Si oui, la niche est réelle contre un adversaire adaptatif.
Si non (KS-radial / MMD à bonne largeur gagnent encore), le papier doit le dire.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist
import exp19_full_battery as E

torch.manual_seed(0)
RNG = E.RNG
K = 8


def knn_sorted_t(P, k=K):
    D = torch.cdist(P, P) + torch.eye(len(P))*1e9
    v, _ = torch.topk(D, k, dim=1, largest=False)
    return torch.sort(v.reshape(-1)).values

def mardia_t(P):
    mu = P.mean(0); Pc = P-mu
    C = (Pc.T@Pc)/(len(P)-1) + 1e-6*torch.eye(P.shape[1])
    return (torch.einsum("ij,jk,ik->i", Pc, torch.inverse(C), Pc)**2).mean()


def adaptive_attack(h, iters=1200, lr=5e-3):
    """Collapse puis optimisation pour apparier {μ, Σ, kNN, kurtosis} de h."""
    A = torch.tensor(h)
    muA = A.mean(0); Ac = A-muA; CA = (Ac.T@Ac)/(len(A)-1)
    knnA = knn_sorted_t(A).detach(); kurtA = mardia_t(A).detach()
    B = torch.tensor(E.collapse(h).copy(), requires_grad=True)
    opt = torch.optim.Adam([B], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        mu = B.mean(0); Bc = B-mu; C = (Bc.T@Bc)/(len(B)-1)
        loss = ((mu-muA)**2).sum() + ((C-CA)**2).sum() \
             + 5*((knn_sorted_t(B)-knnA)**2).mean() \
             + 0.5*(mardia_t(B)-kurtA)**2
        loss.backward(); opt.step()
    return B.detach().numpy()


def main():
    z = np.load("emb_bge1024.npz"); X, y = z["X"], z["y"]
    Xr = PCA(n_components=9).fit_transform(X)
    grid = np.linspace(0, np.percentile(pdist(Xr[RNG.choice(len(Xr),300,False)],"euclidean"),95), 60)

    def win(W=200):
        per = W//4; idx=[]
        for c in range(4): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]

    train = [win() for _ in range(12)]
    ref_pool = np.vstack(train); ref_s = ref_pool[RNG.choice(len(ref_pool),200,replace=False)]
    ref_land = np.mean([E.h1_land(w, grid) for w in train], axis=0)
    Sig = np.mean([np.cov(w,rowvar=False) for w in train],axis=0); Si = np.linalg.pinv(Sig)
    mu = np.mean([w.mean(0) for w in train],axis=0)
    ref_radii = E.maha_radii(ref_pool, mu, Si)
    med = np.median(pdist(ref_s,"euclidean"))

    def sc(w):
        return {
            "Δ_top": float(np.sum(np.abs(E.h1_land(w,grid)-ref_land))),
            "cov-drift": float(np.linalg.norm(np.cov(w,rowvar=False)-Sig,"fro")),
            "kNN": E.knn_density(w),
            "Kurt": E.mardia(w),
            "MMD(σ=med/4)": E.mmd_rbf(w, ref_s, 1.0/(2*(med/4)**2)),
            "KS-radial": E.ks_radial(w, mu, Si, ref_radii),
        }

    N = 16
    H = [win() for _ in range(N)]
    A = []
    resid = []
    for _ in range(N):
        h = win(); a = adaptive_attack(h); A.append(a)
        resid.append((np.linalg.norm(a.mean(0)-h.mean(0)),
                      np.linalg.norm(np.cov(a,rowvar=False)-np.cov(h,rowvar=False),"fro"),
                      abs(E.mardia(a)-E.mardia(h))))
    r = np.array(resid)
    print("attaque adaptative — résidus vs pré-image (visés ~0) :")
    print(f"  |Δμ|={r[:,0].mean():.4f}   |ΔΣ|_F={r[:,1].mean():.4f}   |Δb|={r[:,2].mean():.3f}\n")

    sh = [sc(w) for w in H]; sa = [sc(w) for w in A]
    print(f"AUC bilatérale [IC 95%] — adversaire optimisé contre {{μ,Σ,kNN,b}}, N={N}\n")
    best = None
    for d in sh[0]:
        a_, lo, hi = E.auc_ci(sh, sa, d)
        print(f"  {d:>13} = {a_:.2f} [{lo:.2f}, {hi:.2f}]")
        if best is None or a_ > best[1]: best = (d, a_)
    print(f"\n  meilleur détecteur : {best[0]} ({best[1]:.2f})")
    print("\n=> Δ_top en tête => niche réelle contre adversaire adaptatif.")
    print("   Un détecteur non topologique en tête => à rapporter tel quel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
