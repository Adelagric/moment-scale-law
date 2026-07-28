"""SPRINT 7 — propager la persistance totale dans TOUS les tableaux (revue #4, point A).

Le §6.7 montre que le résumé domine le choix de filtration ; il faut donc refaire avec la
persistance totale : (1) dimension ambiante, (2) balayage PCA, (3) délai/ARL, (4) IC vs kurtosis,
(5) DTM + persistance totale, (6) rappels manquants pour entropie/bottleneck.

Coût marginal ~nul : même appel à ripser, réduction différente de la sortie.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import exp19_full_battery as E
import exp22_dtm_arl as D

RNG = E.RNG


def diagram(X, dm=None):
    if dm is None:
        dm = squareform(pdist(X, "euclidean"))
    dg = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    return dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))


def summaries(fin, grid, ref_land):
    """(paysage L1, persistance totale, entropie) à partir d'UN diagramme."""
    life = fin[:, 1]-fin[:, 0] if len(fin) else np.array([0.0])
    lam = np.zeros_like(grid)
    for b, d in fin:
        lam = np.maximum(lam, np.clip(np.minimum(grid-b, d-grid), 0, None))
    p = life/max(life.sum(), 1e-12)
    return (float(np.sum(np.abs(lam-ref_land))), float(life.sum()),
            float(-(p*np.log(p+1e-12)).sum()))


def make(dim, cache="emb_bge1024.npz", W=200):
    z = np.load(cache); X, y = z["X"], z["y"]
    Xr = PCA(n_components=dim).fit_transform(X)
    def win():
        per = W//4; idx=[]
        for c in range(4): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
        return Xr[np.array(idx)]
    grid = np.linspace(0, np.percentile(pdist(Xr[RNG.choice(len(Xr),300,False)],"euclidean"),95), 60)
    tr=[win() for _ in range(12)]
    ref_land=np.mean([summaries(diagram(w),grid,np.zeros(60))[0]*0+
                      np.maximum.reduce([np.zeros(60)]) for w in tr],axis=0)  # placeholder
    # vrai paysage de référence
    lands=[]
    for w in tr:
        fin=diagram(w); lam=np.zeros_like(grid)
        for b,d in fin: lam=np.maximum(lam,np.clip(np.minimum(grid-b,d-grid),0,None))
        lands.append(lam)
    ref_land=np.mean(lands,axis=0)
    Sig=np.mean([np.cov(w,rowvar=False) for w in tr],axis=0)
    return win, grid, ref_land, Sig


def attack(win):
    h=win(); return E.affine_match(E.collapse(h), h.mean(0), np.cov(h,rowvar=False))


def auc(h, a):
    v=roc_auc_score([0]*len(h)+[1]*len(a), list(h)+list(a)); return max(v,1-v)


def main():
    print("=== (1) dimension AMBIANTE : paysage vs persistance totale ===")
    print(f"{'ambiant':>8} {'dim réd':>8} {'paysage':>9} {'total-pers':>11} {'kurtosis':>9}")
    for cache, amb, dr in (("emb_cache.npz",384,8),("emb_mpnet768.npz",768,7),("emb_bge1024.npz",1024,9)):
        try: win,grid,ref,Sig = make(dr, cache)
        except FileNotFoundError: continue
        N=16; H=[win() for _ in range(N)]; A=[attack(win) for _ in range(N)]
        sh=[summaries(diagram(w),grid,ref) for w in H]; sa=[summaries(diagram(w),grid,ref) for w in A]
        ku_h=[E.mardia(w) for w in H]; ku_a=[E.mardia(w) for w in A]
        print(f"{amb:>8} {dr:>8} {auc([x[0] for x in sh],[x[0] for x in sa]):>9.2f} "
              f"{auc([x[1] for x in sh],[x[1] for x in sa]):>11.2f} {auc(ku_h,ku_a):>9.2f}")

    print("\n=== (2) balayage PCA : le récit 'notre dim est le pire cas' tient-il ? ===")
    print(f"{'dim':>5} {'paysage':>9} {'total-pers':>11} {'kurtosis':>9}")
    for dim in (5,9,15,30,50):
        win,grid,ref,Sig=make(dim)
        N=14; H=[win() for _ in range(N)]; A=[attack(win) for _ in range(N)]
        sh=[summaries(diagram(w),grid,ref) for w in H]; sa=[summaries(diagram(w),grid,ref) for w in A]
        print(f"{dim:>5} {auc([x[0] for x in sh],[x[0] for x in sa]):>9.2f} "
              f"{auc([x[1] for x in sh],[x[1] for x in sa]):>11.2f} "
              f"{auc([E.mardia(w) for w in H],[E.mardia(w) for w in A]):>9.2f}")

    print("\n=== (3) IC bootstrap : la 'domination significative' tient-elle ? ===")
    win,grid,ref,Sig=make(9)
    N=20; H=[win() for _ in range(N)]; A=[attack(win) for _ in range(N)]
    sh=[summaries(diagram(w),grid,ref) for w in H]; sa=[summaries(diagram(w),grid,ref) for w in A]
    packs={"paysage":( [x[0] for x in sh],[x[0] for x in sa]),
           "total-pers":([x[1] for x in sh],[x[1] for x in sa]),
           "entropie":([x[2] for x in sh],[x[2] for x in sa]),
           "kurtosis":([E.mardia(w) for w in H],[E.mardia(w) for w in A])}
    for k,(h,a) in packs.items():
        boots=[]
        yv=np.array([0]*N+[1]*N); sv=np.array(list(h)+list(a))
        for _ in range(400):
            idx=RNG.integers(0,2*N,2*N)
            if len(set(yv[idx].tolist()))<2: continue
            v=roc_auc_score(yv[idx],sv[idx]); boots.append(max(v,1-v))
        print(f"  {k:>11} = {auc(h,a):.2f} [{np.percentile(boots,2.5):.2f}, {np.percentile(boots,97.5):.2f}]")

    print("\n=== (4) DTM + persistance totale (résumé × filtration) ===")
    win,grid,ref,Sig=make(9)
    N=16; H=[win() for _ in range(N)]; A=[attack(win) for _ in range(N)]
    for name, mat in (("Rips", lambda w: squareform(pdist(w,"euclidean"))),
                      ("DTM",  lambda w: D.dtm_rips_matrix(w))):
        th=[diagram(None,mat(w)) for w in H]; ta=[diagram(None,mat(w)) for w in A]
        tot_h=[float((f[:,1]-f[:,0]).sum()) if len(f) else 0.0 for f in th]
        tot_a=[float((f[:,1]-f[:,0]).sum()) if len(f) else 0.0 for f in ta]
        print(f"  {name:>5} + total-pers : AUC={auc(tot_h,tot_a):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
