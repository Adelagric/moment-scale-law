"""SPRINT 4 — les deux derniers trous empiriques.

(A) DTM-Rips évalué (on le citait sans l'utiliser : la conclusion ne pouvait porter que sur le
    détecteur vanille). Rips pondéré p=infini (Anai et al. / Buchet et al.) :
        f(i)   = DTM_m(i)
        f(i,j) = max(f(i), f(j), d_ij/2)
(B) DÉLAI DE DÉTECTION à ARL fixé — la métrique d'un moniteur de flux (SPC/changepoint),
    pas l'AUC par fenêtre. Seuil calibré pour un ARL_0 cible sur flux sain, puis on mesure le
    nombre de fenêtres avant alarme après le point de changement.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform, pdist
from ripser import ripser
import exp19_full_battery as E

RNG = E.RNG


# ---------------- (A) DTM-Rips ----------------
def dtm(X, m=0.1):
    """Distance-to-measure : racine de la moyenne des carrés aux k plus proches voisins."""
    k = max(2, int(m*len(X)))
    D = squareform(pdist(X, "euclidean")); D.sort(axis=1)
    return np.sqrt((D[:, 1:k+1]**2).mean(axis=1))

def dtm_rips_matrix(X, m=0.1):
    f = dtm(X, m)
    D = squareform(pdist(X, "euclidean"))
    M = np.maximum(np.maximum.outer(f, f), D/2.0)
    np.fill_diagonal(M, 0.0)          # ripser exige une diagonale nulle (approximation)
    return M

def dtm_land(X, grid, m=0.1):
    M = dtm_rips_matrix(X, m)
    dg = ripser(M, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))
    lam = np.zeros_like(grid)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(grid-b, d-grid), 0, None))
    return lam


def setup(dim=9, W=200):
    z = np.load("emb_bge1024.npz"); X, y = z["X"], z["y"]
    Xr = PCA(n_components=dim).fit_transform(X)
    g = np.percentile(pdist(Xr[RNG.choice(len(Xr),300,False)], "euclidean"), 95)
    grid = np.linspace(0, g, 60); grid_d = np.linspace(0, g, 60)
    def win():
        per = W//4; idx=[]
        for c in range(4): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]
    train=[win() for _ in range(12)]; ref=np.vstack(train)
    ref_s=ref[RNG.choice(len(ref),200,replace=False)]
    land=np.mean([E.h1_land(w,grid) for w in train],axis=0)
    land_d=np.mean([dtm_land(w,grid_d) for w in train],axis=0)
    Sig=np.mean([np.cov(w,rowvar=False) for w in train],axis=0); Si=np.linalg.pinv(Sig)
    mu=np.mean([w.mean(0) for w in train],axis=0); rad=E.maha_radii(ref,mu,Si)
    med=np.median(pdist(ref_s,"euclidean"))
    def sc(w):
        return {"Δ_top (Rips)":float(np.sum(np.abs(E.h1_land(w,grid)-land))),
                "Δ_top (DTM)":float(np.sum(np.abs(dtm_land(w,grid_d)-land_d))),
                "Kurt":E.mardia(w),
                "MMD(med/4)":E.mmd_rbf(w,ref_s,1.0/(2*(med/4)**2)),
                "KS-radial":E.ks_radial(w,mu,Si,rad)}
    return win, sc


def attack(win):
    h = win(); return E.affine_match(E.collapse(h), h.mean(0), np.cov(h, rowvar=False))


def part_a():
    from sklearn.metrics import roc_auc_score
    print("=== (A) DTM-Rips vs Rips vanille (attaque cov-préservée) ===")
    win, sc = setup()
    N = 16
    H=[sc(win()) for _ in range(N)]; A=[sc(attack(win)) for _ in range(N)]
    print(f"  {'détecteur':>14} {'AUC':>6}  recall@FPR=5%")
    # recall : calibrage sur un lot sain séparé
    Hc=[sc(win()) for _ in range(60)]
    for d in H[0]:
        a=roc_auc_score([0]*N+[1]*N,[s[d] for s in H]+[s[d] for s in A]); a=max(a,1-a)
        hs=np.array([s[d] for s in Hc]); as_=np.array([s[d] for s in A])
        sign=1.0 if np.median(as_)>=np.median(hs) else -1.0
        thr=np.quantile(sign*hs,0.95); rec=float(np.mean(sign*as_>thr))
        print(f"  {d:>14} {a:>6.2f} {rec:>14.2f}")
    print()


def part_b():
    print("=== (B) Délai de détection à ARL_0 fixé (métrique streaming) ===")
    win, sc = setup()
    # flux sain long pour calibrer ARL_0 = 100 fenêtres (FPR = 1/100 par fenêtre)
    cal=[sc(win()) for _ in range(120)]
    dets=list(cal[0].keys())
    # flux : 20 fenêtres saines puis 20 attaquées ; on répète
    print(f"  ARL_0 cible = 100 fenêtres ; 25 flux ; délai médian après changement (max 20)")
    print(f"  {'détecteur':>14} {'délai médian':>13} {'% détectés':>11}")
    for d in dets:
        hs=np.array([s[d] for s in cal])
        sign=1.0 if True else 1.0
        # sens déterminé sur un lot attaqué
        probe=[sc(attack(win)) for _ in range(8)]
        sign=1.0 if np.median([p[d] for p in probe])>=np.median(hs) else -1.0
        thr=np.quantile(sign*hs, 1-1/100)
        delays=[]; det=0
        for _ in range(25):
            delay=None
            for t in range(20):
                s=sc(attack(win))
                if sign*s[d]>thr: delay=t+1; break
            if delay: delays.append(delay); det+=1
            else: delays.append(21)
        print(f"  {d:>14} {np.median(delays):>13.0f} {100*det/25:>10.0f}%")
    print("\n  (délai 21 = non détecté dans la fenêtre d'observation)")


def main():
    part_a(); part_b(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
