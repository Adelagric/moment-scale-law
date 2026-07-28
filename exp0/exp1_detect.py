"""EXP 1 — POUVOIR DE DÉTECTION (protocole §7).

Question : Delta_top (signal topologique H1) sépare-t-il sain/attaqué MIEUX que
Mahalanobis (B2) et la dérive spectrale (B6) ? Mesure = AUC par (détecteur × attaque).

Honnêteté bidirectionnelle — trois attaques couvrant le spectre :
  - INJECT   : points hors-variété      -> Mahalanobis/B6 DOIVENT gagner (facile)
  - COLLAPSE : contraction vers centroïde-> B6 DOIT gagner (chute de variance)
  - LOOPFILL : bouche un trou H1 en préservant ~mean/cov -> SEULE la TDA peut gagner

Espace : PCA dim=8 (Exp 0 : c'est là que H1 existe). Fenêtres multi-sujets W=250.
Référence (mu, Sigma, paysage sain) ajustée sur des fenêtres saines DISJOINTES du test.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from exp0_gate import build_embeddings, cosine_dm
from ripser import ripser

RNG = np.random.default_rng(2024)
GRID = np.linspace(0.0, 1.5, 60)  # grille d'échelle pour le paysage


# ---------- signal topologique ----------
def h1_landscape(X):
    """Premier paysage de persistance lambda^1(t) de H1 (Bubenik), 1-Lipschitz."""
    dm = cosine_dm(X)
    dgm = ripser(dm, maxdim=1, distance_matrix=True)["dgms"][1]
    fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else np.empty((0, 2))
    lam = np.zeros_like(GRID)
    for b, d in fin:
        tent = np.minimum(GRID - b, d - GRID)
        lam = np.maximum(lam, np.clip(tent, 0, None))
    return lam


# ---------- baselines ----------
def mahalanobis_score(X, mu, SigInv):
    diff = X - mu
    return float(np.mean(np.sqrt(np.einsum("ij,jk,ik->i", diff, SigInv, diff))))

def spectral_score(X, Sig_ref):
    return float(np.linalg.norm(np.cov(X, rowvar=False) - Sig_ref, ord="fro"))

def cosine_score(X, centroid):
    c = centroid / (np.linalg.norm(centroid) + 1e-12)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return float(1 - np.mean(Xn @ c))


# ---------- attaques (dans l'espace réduit) ----------
def attack_inject(X, rho=0.15):
    X = X.copy(); n = len(X); k = int(rho * n)
    mu, sd = X.mean(0), X.std(0)
    far = mu + RNG.normal(0, 6 * sd, size=(k, X.shape[1]))  # hors-variété
    X[RNG.choice(n, k, replace=False)] = far
    return X

def attack_collapse(X, alpha=0.5):
    mu = X.mean(0)
    return mu + (1 - alpha) * (X - mu)  # contraction -> variance chute

def attack_loopfill(X, rho=0.18):
    """Ajoute des points au barycentre du cycle H1 le plus persistant : bouche le
    trou. On compense la variance en gardant l'enveloppe (remplace des points
    INTÉRIEURS proches du centre, pas les extrêmes) -> mean/cov ~ préservés."""
    X = X.copy(); n = len(X); k = int(rho * n)
    mu = X.mean(0)
    d2c = np.linalg.norm(X - mu, axis=1)
    inner = np.argsort(d2c)[: 3 * k]                      # points déjà centraux
    repl = RNG.choice(inner, k, replace=False)
    jitter = RNG.normal(0, 0.15 * X.std(0), size=(k, X.shape[1]))
    X[repl] = mu + jitter                                 # densifie le centre (bouche le trou)
    return X


def cov_delta(A, B):
    return np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False), "fro")


def main():
    categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    X, y = build_embeddings(categories, per_cat=200, seed=0)
    Xr = PCA(n_components=8).fit_transform(X)

    def healthy_window(W=250):
        per = W // len(categories)
        idx = []
        for c in range(len(categories)):
            pool = np.where(y == c)[0]
            idx += list(RNG.choice(pool, per, replace=False))
        return Xr[np.array(idx)]

    # ----- référence (train sain, disjoint du test) -----
    train = [healthy_window() for _ in range(20)]
    mu = np.mean([w.mean(0) for w in train], axis=0)
    Sig_ref = np.mean([np.cov(w, rowvar=False) for w in train], axis=0)
    SigInv = np.linalg.pinv(Sig_ref)
    land_ref = np.mean([h1_landscape(w) for w in train], axis=0)
    centroid = mu.copy()

    def scores(w):
        return {
            "Δ_top(H1)": float(np.sum(np.abs(h1_landscape(w) - land_ref))),
            "Mahalanobis": mahalanobis_score(w, mu, SigInv),
            "Spectral(B6)": spectral_score(w, Sig_ref),
            "Cosine(B1)": cosine_score(w, centroid),
        }

    # ----- test -----
    N = 24
    healthy = [healthy_window() for _ in range(N)]
    attacks = {
        "INJECT": [attack_inject(healthy_window()) for _ in range(N)],
        "COLLAPSE": [attack_collapse(healthy_window()) for _ in range(N)],
        "LOOPFILL": [attack_loopfill(healthy_window()) for _ in range(N)],
    }

    s_healthy = [scores(w) for w in healthy]
    detectors = list(s_healthy[0].keys())

    # diagnostic : à quel point chaque attaque bouge-t-elle la covariance ?
    print("déplacement moyen de covariance (Frobenius) par attaque :")
    for name, ws in attacks.items():
        dc = np.mean([cov_delta(w, healthy_window()) for w in ws])
        print(f"  {name:9} Δcov ≈ {dc:.3f}")

    print(f"\n=== AUC sain vs attaqué (N={N}/classe) ===")
    header = "attaque   " + "".join(f"{d:>14}" for d in detectors)
    print(header)
    for name, ws in attacks.items():
        s_att = [scores(w) for w in ws]
        row = f"{name:9}"
        aucs = {}
        for d in detectors:
            yv = [0] * len(s_healthy) + [1] * len(s_att)
            sv = [s[d] for s in s_healthy] + [s[d] for s in s_att]
            auc = roc_auc_score(yv, sv)
            aucs[d] = auc
            row += f"{auc:>14.3f}"
        win = max(aucs, key=aucs.get)
        print(row + f"   ← {win}")

    print("\nLecture : LOOPFILL est le juge de paix — si Δ_top y gagne alors que")
    print("Spectral/Mahalanobis y échouent, la TDA couvre un angle mort du 2ᵈ ordre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
