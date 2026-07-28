"""Figures du papier 1. Valeurs mesurées dans exp0/ (exp4b, exp1c)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = __import__("os").path.dirname(__file__)

# --- Figure 1 : AUC vs dimension ambiante (attaque adaptative, exp4b) ---
dims = [384, 768, 1024]
data = {
    "Kurtosis (Mardia)": [1.00, 1.00, 1.00],
    "k-NN density":      [0.91, 0.98, 0.98],
    r"$\Delta_{\mathrm{top}}$ (persistence)": [0.89, 0.77, 0.64],
    "Covariance drift (B6)": [0.66, 0.66, 0.54],
}
fig, ax = plt.subplots(figsize=(6.2, 4.0))
marks = ["o-", "^-", "s-", "d--"]
for (label, ys), m in zip(data.items(), marks):
    ax.plot(dims, ys, m, lw=2, ms=7, label=label)
ax.axhline(0.5, ls=":", c="grey", lw=1)
ax.set_xticks(dims); ax.set_ylim(0.45, 1.03)
ax.set_xlabel("ambient embedding dimension")
ax.set_ylabel("detection AUC (bilateral)")
ax.set_title("Adaptive covariance-preserving attack")
ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_auc_vs_dim.png", dpi=150)

# --- Figure 2 : coquille vs boule, covariance appariée (H1) ---
rng = np.random.default_rng(0)
n = 300
t = rng.uniform(0, 2*np.pi, n)
circle = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.05, (n, 2))
R = np.sqrt(2)
rad = R*np.sqrt(rng.uniform(0, 1, n))
u = rng.uniform(0, 2*np.pi, n)
disk = np.c_[rad*np.cos(u), rad*np.sin(u)]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3))
for a, P, ttl, hh in [(axs[0], circle, "shell — $H_1\\neq 0$", r"$\mathrm{Var}(R^2)=0$"),
                      (axs[1], disk, "ball — $H_1 = 0$", r"$\mathrm{Var}(R^2)>0$")]:
    a.scatter(P[:, 0], P[:, 1], s=6, alpha=0.6)
    a.set_title(ttl, fontsize=10); a.text(0.5, -1.9, hh, ha="center", fontsize=9)
    a.set_xlim(-2, 2); a.set_ylim(-2, 2.1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("Identical mean & covariance, different topology (and 4th moment)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_shell_ball.png", dpi=150)
print("figures écrites :", f"{OUT}/fig_auc_vs_dim.png", f"{OUT}/fig_shell_ball.png")
