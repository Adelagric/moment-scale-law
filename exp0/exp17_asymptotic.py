"""PAPER 2 — confirmer le RÉGIME ASYMPTOTIQUE de l'approximation (fermeture de l'écart).

exp16b plafonnait : projection de Legendre en float64, plage utile ~2 décades, régime
pré-asymptotique. Ici on passe par la FFT (stable jusqu'à ~1e-15, soit 14 décades) :
pour une bosse de Gevrey-s, les coefficients de Fourier décroissent en exp(−c·k^{1/s}),
alors qu'une bosse Lipschitz décroît en k^{-2}. L'erreur de meilleure approximation de degré N
est la queue des coefficients au-delà de N, donc même loi.

Test : log|ĉ_k| est-il linéaire en k^{1/2} (Gevrey-2, sur-algébrique) ou en log k (algébrique) ?
"""
from __future__ import annotations
import numpy as np

M = 1 << 16          # échantillons (FFT)


def sample(eps, kind):
    x = np.linspace(-1, 1, M, endpoint=False)
    u = np.clip(x/eps, -1, 1)
    inside = np.abs(x) < eps
    if kind == "smooth":                       # mollifier C^inf (Gevrey-2), support compact
        f = np.zeros_like(x)
        f[inside] = np.exp(-1.0/(1.0 - u[inside]**2))
        return f/f.max()
    return np.clip(1 - np.abs(x)/eps, 0, None)  # triangle Lipschitz


def spectrum(eps, kind):
    f = sample(eps, kind)
    c = np.abs(np.fft.rfft(f))/M
    return c


def fits(c, kmin, kmax, s=2.0):
    k = np.arange(kmin, kmax)
    y = c[kmin:kmax]
    m = y > 1e-15                      # au-dessus du plancher float64
    k, y = k[m], y[m]
    if len(k) < 8:
        return None
    ly = np.log(y)
    r_gev = np.corrcoef(k**(1/s), ly)[0, 1]     # Gevrey-s : log|c| ~ -c k^{1/s}
    r_alg = np.corrcoef(np.log(k), ly)[0, 1]    # algébrique : log|c| ~ -p log k
    p = np.polyfit(np.log(k), ly, 1)[0]
    slope_gev = np.polyfit(k**(1/s), ly, 1)[0]
    return r_gev, r_alg, p, slope_gev, len(k), y.min()


def main():
    for eps in (0.15, 0.30):
        print(f"\n===== ε = {eps} =====")
        for kind in ("smooth", "lip"):
            c = spectrum(eps, kind)
            res = fits(c, 5, 4000)
            if not res:
                print(f"  {kind}: pas assez de points"); continue
            r_gev, r_alg, p, sg, n, ymin = res
            print(f"  {kind:>6} | R(log|c| vs k^1/2) = {r_gev:+.4f}   "
                  f"R(log|c| vs log k) = {r_alg:+.4f}   pente alg = {p:.2f}   "
                  f"[{n} pts, min |c| = {ymin:.1e}]")
        # échelle : la décroissance dépend-elle de k·ε ?
        cs = spectrum(eps, "smooth")
        idx = [i for i in range(5, 3000) if cs[i] > 1e-14]
        if idx:
            k = np.array(idx); y = np.log(cs[k])
            sl = np.polyfit((k*eps)**0.5, y, 1)[0]
            print(f"  pente en (kε)^1/2 : {sl:.3f}  (loi d'échelle : dépend de kε)")
    print("\nLecture : pour la bosse LISSE, R(vs k^1/2) proche de -1 et bien meilleur que")
    print("R(vs log k) => décroissance sur-algébrique exp(-c·k^{1/2}) CONFIRMÉE (Gevrey-2).")
    print("La bosse LIPSCHITZ doit rester algébrique (pente ≈ -2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
