# How fine a change can moments see?

Code and experiments for *How fine a change can moments see? A scale law for detecting
distribution shift, with a kernel calibration rule* (Adel Kaleche, independent researcher).

Every number in the paper is produced by one of the scripts in `exp0/`. The map below says which.

## Install

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Scripts that mix `torch` with `ripser` need `KMP_DUPLICATE_LIB_OK=TRUE` on macOS:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python exp0/exp28_sigma_canonical.py
```

Embedding caches (`exp0/emb_*.npz`, ~20 MB total) are **not** committed; the first script that
needs one downloads the corpus and the sentence-transformer and writes the cache. Building all
four takes a few minutes on CPU.

## Which script produces which number

| paper element | script(s) |
|---|---|
| §3 lower bound, Gauss construction (`N*≥4b−1`) | `exp9_gauss_frontier.py` |
| §3 separable case (`N*∝b` off spherical symmetry) | `exp10_separable.py` |
| §3 Chebyshev extremal (`contrast ≈ e^{2Nε}`) | `exp30_chebyshev_extremal.py` |
| §3 closed form `b=d²(1+Var(R²)/E[R²]²)`, shell/ball | `exp3f_theory_check.py` |
| §3 elliptical extension | `exp6_elliptical.py` |
| §3 Jackson/Gevrey rate over 14 decades | `exp17_asymptotic.py` |
| §3 Gaussian vs Gevrey test functions | `exp18_optimal_test.py` |
| §3 scale law `N*(ε) ∝ 1/ε` | `exp14_h_law.py` |
| §3 Observation 1, `H₀` bimodal vs elongated | `exp7b_connectivity_moment.py` |
| §4 annulus counterexample (`a*=0.295`, all moments to order 4) | `exp29_annulus_counterexample.py` |
| §4 optimisation counterexample (breaks `H₁` matching `μ,Σ,kNN`) | `exp3_impossibility.py` |
| §6.1 higher homology only after reduction | `exp0_gate.py`, `exp0_reduce.py`, `exp0_final.py` |
| §6.2 first pass, cheap statistics | `exp1_detect.py` |
| §6.3 circle-vs-disk idealised niche | `exp1c_matched.py`, `exp2d_niche_test.py` |
| §6.4 full battery, MMD bandwidth sweep | `exp19_full_battery.py` |
| §6.4 adversary optimised against `{μ,Σ,kNN,b}` | `exp20_truly_adaptive.py` |
| §6.4 MMD-aware adversary; KS-radial cost | `exp26_final_gaps.py` |
| §6.5 canonical `σ*/ε` (median 1.12, IQR, 3×3×3) | `exp28_sigma_canonical.py` |
| §6.5 identifiability and loop closure | `exp24_sigma_identified.py` |
| §6.5 resolution-floor test | `exp27_resolution_floor.py` |
| §6.6 operating points, cost table, PCA sweep | `exp21_operating_points.py` |
| §6.7 persistence summaries; propagation to all tables | `exp24_sigma_identified.py`, `exp25_propagate.py` |
| §6.8 DTM-Rips and streaming delay at fixed ARL | `exp22_dtm_arl.py` |
| §6.9 robustness, bootstrap CIs, two corpora | `exp5_robustness.py` |
| Table 1 (ambient dimension) | `exp4b_adaptive_highdim.py`, `exp25_propagate.py` |
| engine correctness gate (known topology) | `validate_engine.py` |

`topology.py` holds shared persistence helpers; `exp19_full_battery.py` holds the detector
definitions imported by later scripts.

## Superseded experiments, kept on purpose

Several scripts produced results that later versions of the paper **retracted**. They are kept so
the corrections are auditable:

- `exp1b_whiten.py` — self-referential whitening to blind the covariance baseline; abandoned
  because it polluted the baseline through numerical conditioning. Replaced by `exp1c_matched.py`.
- `exp23_sigma_star.py` — first `σ*/ε` measurement (ratio `0.48`); **retracted**: the AUC saturated
  at `1.00`, so the argmax was not identified. Superseded by `exp24` then `exp28`.
- `exp7_mixture.py` — trade-off curves on mixtures by optimisation; inconclusive, because matching
  global moments constrains per-component topology only weakly. The paper says so.
- `exp8_frontier.py` — attempt to measure `N*(b)` by optimisation; the optimiser converged while
  the topology stayed ambiguous, so the result was unusable. Replaced by the Gauss construction
  (`exp9`), which needs no optimisation.
- `exp16_mass_law.py` — Monte-Carlo test of the mass dependence; broken by sampling noise (the
  threshold scaled with `f`). Replaced by the deterministic `exp16b_jackson_rate.py` and `exp17`.

## Paper source

`docs/paper1/` contains `paper.tex`, `refs.bib`, the figures, and `verify.py`.

`verify.py` runs assertions against the **compiled PDF** rather than the LaTeX source, including
negative assertions ("this retracted claim is absent"). It exists because a silent `str.replace`
failure once produced a correction report for an edit that had not happened. Run it before
submitting:

```bash
cd docs/paper1 && tectonic paper.tex && python verify.py
```

## Caveats worth knowing before you re-run

- Single runs of the `σ*` sweep are unreliable: `AUC(σ)` is multimodal (§6.5), and the global
  argmax flips between the deformation scale and the cloud scale. Use repetitions.
- Detector AUCs differ between tables because each experiment re-draws its windows; the paper's
  "On repeated cells" paragraph documents the spread.
- Timings in the cost table are for `W=200`, reduction dimension 9, on CPU; Rips is superlinear in
  `W` and MMD quadratic in `n`, so the ratios are regime-dependent.
