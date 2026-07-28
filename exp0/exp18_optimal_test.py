"""PAPER 2 — éliminer le polylog : quelle fonction test minimise N*(ε,f) ?

Le (log 1/f)² vient des bosses Gevrey (support compact mais régularité limitée). Une
GAUSSIENNE est analytique (coefficients qui chutent bien plus vite) mais a des QUEUES : elle
fuit hors de la boule de rayon ε, ce qui pollue le test de masse locale. Arbitrage
localisation ↔ régularité, tranché numériquement.

Critère : N*(ε,f) = plus petit N tel que  err_approx(N) + fuite  <  f
  - err_approx(N) = queue des coefficients de Chebyshev au-delà de N (via FFT, stable)
  - fuite = masse de la fonction test hors de la boule de rayon ε (0 si support compact)
On ajuste N* vs log(1/f) pour lire l'exposant du polylog.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfc

M = 1 << 15


def cheb_coeffs(f_vals):
    """Coefficients de Chebyshev via FFT (f échantillonné aux points de Chebyshev)."""
    n = len(f_vals)
    g = np.concatenate([f_vals, f_vals[-2:0:-1]])
    c = np.real(np.fft.rfft(g))/(n-1)
    c[0] /= 2; c[-1] /= 2
    return np.abs(c)


def cheb_nodes(n):
    return np.cos(np.pi*np.arange(n)/(n-1))


def test_function(kind, eps, sigma_mult=1.0):
    x = cheb_nodes(M)
    if kind == "gevrey":                      # mollifier C^inf, support compact [-eps, eps]
        u = np.clip(x/eps, -1, 1); f = np.zeros_like(x)
        ins = np.abs(x) < eps
        f[ins] = np.exp(-1.0/(1.0 - u[ins]**2))
        f /= f.max()
        leak = 0.0
    else:                                     # gaussienne, largeur sigma = eps*sigma_mult
        s = eps*sigma_mult
        f = np.exp(-x**2/(2*s**2))
        leak = float(erfc(eps/(np.sqrt(2)*s)))   # masse relative hors de [-eps, eps]
    return f, leak


def nstar(kind, eps, f_target, sigma_mult=1.0):
    vals, leak = test_function(kind, eps, sigma_mult)
    if leak >= f_target:
        return None                            # queues trop grosses : test inutilisable
    c = cheb_coeffs(vals)
    tail = np.cumsum(c[::-1])[::-1]            # tail[N] = somme des |c_k|, k>=N
    budget = f_target - leak
    idx = np.where(tail < budget)[0]
    return int(idx[0]) if len(idx) else None


def best_gaussian(eps, f_target):
    """Optimise la largeur de la gaussienne (arbitrage queues / régularité)."""
    best = None
    for sm in (0.05, 0.1, 0.15, 0.2, 0.3, 0.45):
        n = nstar("gauss", eps, f_target, sm)
        if n is not None and (best is None or n < best[0]):
            best = (n, sm)
    return best


def main():
    eps = 0.1
    print(f"ε = {eps}\n")
    print(f"{'f':>8} {'log(1/f)':>9} {'N* Gevrey':>10} {'N* Gauss':>9} {'σ/ε opt':>8}")
    fs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    ng, nga, xs = [], [], []
    for f in fs:
        a = nstar("gevrey", eps, f)
        bg = best_gaussian(eps, f)
        b, sm = (bg if bg else (None, None))
        ng.append(a); nga.append(b); xs.append(np.log(1/f))
        print(f"{f:>8.0e} {np.log(1/f):>9.2f} {str(a):>10} {str(b):>9} {str(sm):>8}")

    xs = np.array(xs)
    for name, ys in (("Gevrey", ng), ("Gauss", nga)):
        y = np.array([v if v else np.nan for v in ys], float)
        m = ~np.isnan(y)
        if m.sum() > 2:
            p = np.polyfit(np.log(xs[m]), np.log(y[m]), 1)[0]
            r = np.corrcoef(np.log(xs[m]), np.log(y[m]))[0, 1]
            print(f"\n{name:>7} : N* ~ (log 1/f)^{p:.2f}   R={r:.3f}")
    print("\nLecture : exposant ~2 => (log)² (Gevrey) ; exposant ~1 => (log)¹ (gain d'un log).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
