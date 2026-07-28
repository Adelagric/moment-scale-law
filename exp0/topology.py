"""Utilitaires de persistance pour l'Expériment 0.

Spike exploratoire : le but est de DÉCIDER si les embeddings réels portent une
homologie supérieure stable au-dessus d'un modèle nul. Rien ici n'est optimisé.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser


def diagrams(X: np.ndarray, maxdim: int = 2, thresh: float | None = None):
    """Diagrammes de persistance H_0..H_maxdim d'un nuage de points."""
    kw = {}
    if thresh is not None:
        kw["thresh"] = thresh
    return ripser(X, maxdim=maxdim, **kw)["dgms"]


def lifetimes(dgm: np.ndarray) -> np.ndarray:
    """Persistances (mort - naissance), barres infinies exclues."""
    if len(dgm) == 0:
        return np.array([])
    d = dgm[np.isfinite(dgm[:, 1])]
    return d[:, 1] - d[:, 0]


def n_significant(dgm: np.ndarray, band: float) -> int:
    """Nombre de barres dont la persistance dépasse la demi-largeur `band`.

    `band` est la bande de confiance de Fasy et al. : une barre de longueur > 2*band
    n'est pas explicable par le bruit d'échantillonnage (voir confidence_band)."""
    lt = lifetimes(dgm)
    return int(np.sum(lt > 2.0 * band))


def confidence_band(
    X: np.ndarray,
    maxdim: int,
    n_boot: int = 30,
    subsample: float = 1.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Bande de confiance bootstrap (Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan,
    Singh 2014) sur la distance bottleneck au diagramme empirique.

    On rééchantillonne le nuage avec remise, on recalcule le diagramme, et on prend
    le quantile 0.95 de la distance bottleneck au diagramme de référence. Toute barre
    au-delà de cette bande est significative.
    """
    from persim import bottleneck

    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    ref = diagrams(X, maxdim=maxdim)
    k = max(2, int(subsample * n))
    dists = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=k)
        dboot = diagrams(X[idx], maxdim=maxdim)
        # bottleneck maximal sur les dimensions >=1 (celles qui nous intéressent)
        dmax = 0.0
        for dim in range(1, maxdim + 1):
            a = ref[dim] if dim < len(ref) else np.empty((0, 2))
            b = dboot[dim] if dim < len(dboot) else np.empty((0, 2))
            try:
                dmax = max(dmax, bottleneck(a, b))
            except Exception:
                pass
        dists.append(dmax)
    return float(np.quantile(dists, 0.95))
