"""EXP 1b — le test ÉQUITABLE : neutraliser B6 par construction.

Exp 1 était biaisé : même une fuite Δcov=0.019 suffit à B6 (covariance saine très
stable). Ici on BLANCHIT chaque fenêtre à covariance identité. Le blanchiment est
affine = homéomorphisme : il PRÉSERVE la topologie mais annule moyenne+covariance.
Donc B6 et Mahalanobis-au-centre sont aveugles PAR CONSTRUCTION, et il ne reste que
la question pure : Δ_top voit-il un trou bouché invisible au 2ᵈ ordre ?

Si Δ_top > hasard ici alors que B6 ≈ 0.5 => niche topologique réelle.
Si Δ_top ≈ 0.5 aussi => négatif robuste : rien à détecter au-delà du 2ᵈ ordre.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
from exp0_gate import build_embeddings
from exp1_detect import h1_landscape, spectral_score, mahalanobis_score

RNG = np.random.default_rng(77)


def whiten(X):
    """ZCA : centre + covariance -> identité. Affine => topologie préservée."""
    mu = X.mean(0)
    C = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    Wm = np.linalg.inv(np.real(sqrtm(C)))
    return (X - mu) @ Wm


def loopfill_raw(X, rho=0.2):
    """Bouche le centre en espace RÉDUIT brut. Le blanchiment (UNIQUE, appliqué
    ensuite comme pour les fenêtres saines) neutralisera le 2ᵈ ordre symétriquement."""
    X = X.copy(); n = len(X); k = int(rho * n)
    mu = X.mean(0)
    d2c = np.linalg.norm(X - mu, axis=1)
    inner = np.argsort(d2c)[: 3 * k]
    repl = RNG.choice(inner, k, replace=False)
    X[repl] = mu + RNG.normal(0, 0.12 * X.std(0), size=(k, X.shape[1]))
    return X


def main():
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)
    Xr = PCA(n_components=8).fit_transform(X)

    def healthy_raw(W=250):
        per = W // len(categories); idx = []
        for c in range(len(categories)):
            idx += list(RNG.choice(np.where(y == c)[0], per, replace=False))
        return Xr[np.array(idx)]

    # tout est blanchi -> cov=I partout
    train = [whiten(healthy_raw()) for _ in range(20)]
    Sig_ref = np.mean([np.cov(w, rowvar=False) for w in train], axis=0)
    SigInv = np.linalg.pinv(Sig_ref)
    mu = np.mean([w.mean(0) for w in train], axis=0)
    land_ref = np.mean([h1_landscape(w) for w in train], axis=0)

    # traitement SYMÉTRIQUE : sain et attaqué passent par EXACTEMENT un blanchiment
    N = 24
    healthy = [whiten(healthy_raw()) for _ in range(N)]
    attacked = [whiten(loopfill_raw(healthy_raw())) for _ in range(N)]

    def sc(w):
        return {
            "Δ_top(H1)": float(np.sum(np.abs(h1_landscape(w) - land_ref))),
            "Mahalanobis": mahalanobis_score(w, mu, SigInv),
            "Spectral(B6)": spectral_score(w, Sig_ref),
        }

    sh = [sc(w) for w in healthy]; sa = [sc(w) for w in attacked]
    dets = list(sh[0].keys())

    # vérif que B6 est bien neutralisé
    dcov = np.mean([np.linalg.norm(np.cov(a, rowvar=False) - np.cov(h, rowvar=False), "fro")
                    for a, h in zip(attacked, healthy)])
    print(f"Δcov sain vs attaqué (doit être ~0, B6 neutralisé) : {dcov:.4f}")

    print(f"\n=== AUC, B6 neutralisé par blanchiment (N={N}) ===")
    for d in dets:
        yv = [0]*len(sh) + [1]*len(sa)
        sv = [s[d] for s in sh] + [s[d] for s in sa]
        # AUC bilatérale : on prend max(AUC, 1-AUC) car un détecteur peut fonctionner
        # dans les deux sens ; on note le sens.
        auc = roc_auc_score(yv, sv)
        two = max(auc, 1 - auc)
        print(f"  {d:>13} : AUC={auc:.3f}  (bilatéral {two:.3f})")

    print("\nVERDICT :")
    print("  Δ_top bilatéral nettement >0.5 & B6≈0.5 => niche topologique réelle ✅")
    print("  Δ_top ≈0.5 => négatif ROBUSTE : la thèse détection tombe, repli systèmes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
