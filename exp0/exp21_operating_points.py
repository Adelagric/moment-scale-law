"""SPRINT 3 — combler les manques expérimentaux de la revue.

(1) POINTS DE FONCTIONNEMENT à FPR fixé : le pré-enregistrement promettait recall@FPR, on n'a
    rapporté que des AUC. On calibre le seuil sur des fenêtres SAINES (quantile), puis on mesure
    le recall des attaques. C'est la métrique d'un disjoncteur.
(2) TABLEAU DE COÛT : la moitié de l'argument est « PH est chère ». Aucune mesure jusqu'ici.
(3) BALAYAGE DE LA DIMENSION PCA : la réduction est un point d'attaque évident (160 axes pour
    90% de variance, on en garde ~9, méthode linéaire sur variété courbe).
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import exp19_full_battery as E

RNG = E.RNG


def build(dim):
    z = np.load("emb_bge1024.npz"); X, y = z["X"], z["y"]
    Xr = PCA(n_components=dim).fit_transform(X)
    grid = np.linspace(0, np.percentile(pdist(Xr[RNG.choice(len(Xr),300,False)],"euclidean"),95), 60)
    def win(W=200):
        per = W//4; idx=[]
        for c in range(4): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]
    train=[win() for _ in range(12)]; ref=np.vstack(train)
    ref_s=ref[RNG.choice(len(ref),200,replace=False)]
    land=np.mean([E.h1_land(w,grid) for w in train],axis=0)
    Sig=np.mean([np.cov(w,rowvar=False) for w in train],axis=0); Si=np.linalg.pinv(Sig)
    mu=np.mean([w.mean(0) for w in train],axis=0); rad=E.maha_radii(ref,mu,Si)
    med=np.median(pdist(ref_s,"euclidean"))
    def sc(w):
        return {"Δ_top":float(np.sum(np.abs(E.h1_land(w,grid)-land))),
                "cov-drift":float(np.linalg.norm(np.cov(w,rowvar=False)-Sig,"fro")),
                "kNN":E.knn_density(w),"Kurt":E.mardia(w),
                "MMD(med/4)":E.mmd_rbf(w,ref_s,1.0/(2*(med/4)**2)),
                "KS-radial":E.ks_radial(w,mu,Si,rad)}
    return win, sc


def part1_operating_points():
    print("=== (1) Recall @ FPR fixé (seuil calibré sur fenêtres saines) ===")
    win, sc = build(9)
    NH, NA = 150, 50
    H=[sc(win()) for _ in range(NH)]
    A=[sc(E.affine_match(E.collapse(win()), (h:=win()).mean(0), np.cov(h,rowvar=False)))
       for _ in range(NA)]
    dets=list(H[0].keys())
    print(f"  (attaque cov-préservée ; {NH} fenêtres saines pour calibrer, {NA} attaquées)")
    print(f"  {'détecteur':>12} {'recall@FPR=5%':>14} {'recall@FPR=1%':>14}")
    for d in dets:
        hs=np.array([s[d] for s in H]); as_=np.array([s[d] for s in A])
        # sens : on prend la direction qui sépare (score attaqué > ou < sain)
        sign = 1.0 if np.median(as_) >= np.median(hs) else -1.0
        hs, as_ = sign*hs, sign*as_
        r=[]
        for fpr in (0.05, 0.01):
            thr=np.quantile(hs, 1-fpr)
            r.append(float(np.mean(as_ > thr)))
        print(f"  {d:>12} {r[0]:>14.2f} {r[1]:>14.2f}")
    print("  NB : FPR=1% sur 150 fenêtres saines => queue estimée sur ~1,5 fenêtre ;")
    print("       chiffre indicatif, pas une estimation de queue fiable.\n")


def part2_cost():
    print("=== (2) Coût par fenêtre (200 points, dim 9), moyenne sur 20 ===")
    win, sc = build(9)
    ws=[win() for _ in range(20)]
    z=np.load("emb_bge1024.npz")
    import exp19_full_battery as EE
    # chronométrage individuel
    from scipy.spatial.distance import squareform, pdist as pd
    grid=np.linspace(0,3,60)
    ref_s=ws[0]
    timings={}
    def t(fn,n=20):
        t0=time.perf_counter()
        for w in ws[:n]: fn(w)
        return (time.perf_counter()-t0)/n*1000
    timings["Δ_top (Rips H1)"]=t(lambda w: E.h1_land(w,grid))
    timings["cov-drift"]=t(lambda w: np.linalg.norm(np.cov(w,rowvar=False)))
    timings["kNN"]=t(lambda w: E.knn_density(w))
    timings["Kurt"]=t(lambda w: E.mardia(w))
    timings["MMD-RBF"]=t(lambda w: E.mmd_rbf(w,ref_s))
    base=min(timings.values())
    print(f"  {'détecteur':>18} {'ms/fenêtre':>11} {'× le moins cher':>16}")
    for k,v in sorted(timings.items(), key=lambda kv:kv[1]):
        print(f"  {k:>18} {v:>11.2f} {v/base:>16.0f}")
    print()


def part3_pca_sweep():
    print("=== (3) Balayage de la dimension PCA (attaque cov-préservée, AUC bilatérale) ===")
    from sklearn.metrics import roc_auc_score
    print(f"  {'dim':>5} " + "".join(f"{d:>12}" for d in
          ["Δ_top","cov-drift","kNN","Kurt","MMD(med/4)","KS-radial"]))
    for dim in (5, 9, 15, 30, 50):
        win, sc = build(dim)
        N=14
        H=[sc(win()) for _ in range(N)]
        A=[sc(E.affine_match(E.collapse(win()), (h:=win()).mean(0), np.cov(h,rowvar=False)))
           for _ in range(N)]
        row=f"  {dim:>5} "
        for d in H[0]:
            a=roc_auc_score([0]*N+[1]*N,[s[d] for s in H]+[s[d] for s in A])
            row+=f"{max(a,1-a):>12.2f}"
        print(row)
    print("\n  => si Δ_top ne prend jamais la tête, la réduction n'est pas l'explication du négatif.")


def main():
    part1_operating_points()
    part2_cost()
    part3_pca_sweep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
