"""SPRINT 6 — σ* identifiable + BOUCLAGE + résumés de persistance supplémentaires.

(D1) σ* n'était pas identifié : AUC=1,00 à l'optimum => plateau => argmax indéfini.
     Correctif : baisser f jusqu'à obtenir un maximum INTÉRIEUR strict, et rapporter
     l'intervalle {σ : AUC >= 0,95} et sa position relative à ε.
(D2) BOUCLAGE : la règle dit « estimez l'échelle de l'effet ». On ne l'estimait jamais.
     Ici on ESTIME ε à partir des seules données (profil de comptage de voisins), on prédit
     σ_pred = 0,48·ε̂, et on vérifie que ce σ atteint bien la puissance maximale.
(F)  Trois résumés de persistance de plus (persistance totale, entropie, bottleneck) pour
     fermer la réponse « vous n'avez testé qu'une vectorisation ».
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, cdist, squareform
from ripser import ripser

RNG = np.random.default_rng(0)


def mmd2(X, Y, s):
    g = 1.0/(2*s**2)
    return float(np.exp(-g*cdist(X,X,"sqeuclidean")).mean()
                 + np.exp(-g*cdist(Y,Y,"sqeuclidean")).mean()
                 - 2*np.exp(-g*cdist(X,Y,"sqeuclidean")).mean())


def win(Xr, y, W=200, ncls=4):
    per = W//ncls; idx=[]
    for c in range(ncls): idx += list(RNG.choice(np.where(y==c)[0], per, replace=False))
    return Xr[np.array(idx)]


def inject_blob(w, eps, f):
    w=w.copy(); n=len(w); k=max(3,int(f*n))
    centre=w[RNG.integers(0,n)]
    idx=RNG.choice(n,k,replace=False)
    d=RNG.normal(size=(k,w.shape[1])); d/=np.linalg.norm(d,axis=1,keepdims=True)
    r=eps*RNG.uniform(0,1,(k,1))**(1.0/w.shape[1])
    w[idx]=centre+d*r
    return w


# ---------- (D1) courbe AUC(σ) avec maximum intérieur ----------
def auc_curve(Xr, y, eps, f, N=16, ngrid=24):
    ref = win(Xr,y)
    H=[win(Xr,y) for _ in range(N)]
    A=[inject_blob(win(Xr,y),eps,f) for _ in range(N)]
    med=np.median(pdist(ref,"euclidean"))
    sig=np.geomspace(0.02*med,1.5*med,ngrid)
    aucs=[]
    for s in sig:
        sh=[mmd2(w,ref,s) for w in H]; sa=[mmd2(w,ref,s) for w in A]
        a=roc_auc_score([0]*N+[1]*N,sh+sa); aucs.append(max(a,1-a))
    return sig, np.array(aucs)


# ---------- (D2) estimateur de ε à partir des SEULES données ----------
def estimate_eps(healthy, attacked, med):
    """Profil de comptage de voisins : ε̂ = rayon où l'excès relatif de voisins est maximal."""
    radii = np.geomspace(0.05*med, 1.0*med, 30)
    def profile(ws):
        out=[]
        for w in ws:
            D=squareform(pdist(w,"euclidean"))
            out.append([(D<r).sum(axis=1).mean() for r in radii])
        return np.mean(out,axis=0)
    ph, pa = profile(healthy), profile(attacked)
    excess = (pa-ph)/np.maximum(ph,1e-9)
    return float(radii[int(np.argmax(excess))])


# ---------- (F) résumés de persistance supplémentaires ----------
def pers_summaries(X, ref_dgm=None):
    dm=squareform(pdist(X,"euclidean"))
    dg=ripser(dm,maxdim=1,distance_matrix=True)["dgms"][1]
    fin=dg[np.isfinite(dg[:,1])] if len(dg) else np.empty((0,2))
    life=fin[:,1]-fin[:,0] if len(fin) else np.array([0.0])
    total=float(life.sum())
    p=life/max(life.sum(),1e-12); ent=float(-(p*np.log(p+1e-12)).sum())
    return total, ent, fin


def bottleneck_to(ref_fin, fin):
    from persim import bottleneck
    try: return float(bottleneck(ref_fin if len(ref_fin) else np.empty((0,2)),
                                 fin if len(fin) else np.empty((0,2))))
    except Exception: return 0.0


def main():
    z=np.load("emb_bge1024.npz"); X,y=z["X"],z["y"]
    Xr=PCA(n_components=9).fit_transform(X)
    scale=np.median(pdist(win(Xr,y),"euclidean"))

    print("=== (D1) σ* avec maximum INTÉRIEUR (f abaissé jusqu'à désaturation) ===")
    print(f"{'f':>6} {'ε':>7} {'AUC_max':>8} {'σ*':>7} {'σ*/ε':>6} {'{σ:AUC≥.95}/ε':>16}")
    ratios=[]
    for f in (0.05, 0.04, 0.03):
        for frac in (0.15, 0.30):
            eps=frac*scale
            sig,a=auc_curve(Xr,y,eps,f)
            i=int(np.argmax(a))
            interior = 0 < i < len(sig)-1
            band=sig[a>=0.95]
            bs=f"[{band.min()/eps:.2f},{band.max()/eps:.2f}]" if len(band) else "vide"
            ratios.append(sig[i]/eps)
            print(f"{f:>6} {eps:>7.3f} {a.max():>8.2f} {sig[i]:>7.3f} {sig[i]/eps:>6.2f} {bs:>16}"
                  + ("" if interior else "  (bord!)"))
    r=np.array(ratios); print(f"  σ*/ε : moyenne={r.mean():.2f} ± {r.std():.2f}\n")

    print("=== (D2) BOUCLAGE : ε estimé des données -> σ prédit -> puissance atteinte ? ===")
    f=0.05
    for frac in (0.15, 0.30):
        eps_true=frac*scale
        H=[win(Xr,y) for _ in range(16)]
        A=[inject_blob(win(Xr,y),eps_true,f) for _ in range(16)]
        med=np.median(pdist(H[0],"euclidean"))
        eps_hat=estimate_eps(H,A,med)
        s_pred=0.48*eps_hat
        ref=win(Xr,y)
        sh=[mmd2(w,ref,s_pred) for w in H]; sa=[mmd2(w,ref,s_pred) for w in A]
        a_pred=roc_auc_score([0]*16+[1]*16,sh+sa); a_pred=max(a_pred,1-a_pred)
        sig,a=auc_curve(Xr,y,eps_true,f)
        print(f"  ε_true={eps_true:.3f}  ε̂={eps_hat:.3f} (ratio {eps_hat/eps_true:.2f})  "
              f"σ_pred={s_pred:.3f}  AUC(σ_pred)={a_pred:.2f}  vs AUC_max={a.max():.2f}")

    print("\n=== (F) résumés de persistance supplémentaires (attaque cov-préservée) ===")
    import exp19_full_battery as E
    H=[win(Xr,y) for _ in range(16)]
    A=[E.affine_match(E.collapse(win(Xr,y)),(h:=win(Xr,y)).mean(0),np.cov(h,rowvar=False))
       for _ in range(16)]
    _,_,ref_fin=pers_summaries(win(Xr,y))
    rows={"total persistence":[], "persistence entropy":[], "bottleneck to ref":[]}
    for grp in (H,A):
        tp,pe,bn=[],[],[]
        for w in grp:
            t,e,fin=pers_summaries(w); tp.append(t); pe.append(e); bn.append(bottleneck_to(ref_fin,fin))
        rows["total persistence"].append(tp); rows["persistence entropy"].append(pe)
        rows["bottleneck to ref"].append(bn)
    for k,(h_,a_) in rows.items():
        auc=roc_auc_score([0]*16+[1]*16,list(h_)+list(a_)); auc=max(auc,1-auc)
        print(f"  {k:>22} : AUC={auc:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
