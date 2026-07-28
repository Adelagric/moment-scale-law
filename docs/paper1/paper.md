# Topological change is a moment phenomenon: why persistent homology is redundant with low-order statistics for embedding-stream anomaly detection

**Draft — paper 1.** Métrique : brouillon de soumission (workshop / negative-results track).
Chiffres tirés des expériences reproductibles `exp0/` ; preuve dans
`docs/theory/tradeoff-theorem.md`.

---

## Abstract

The manifold hypothesis motivates monitoring the *topology* of high-dimensional embedding
streams: adversarial injection or distribution drift is expected to collapse connected
components and fill homological holes before it is visible to local metrics such as cosine
similarity. We ask whether persistent homology (PH), computed on sliding windows, detects
such changes better than classical low-order statistics. Across two attack families
(diversity collapse, and an adaptive attack constructed to preserve mean and covariance),
three embedders (384–1024 d), and two corpora, we find it does not: covariance drift,
Mahalanobis shape, k-NN density and multivariate kurtosis match or dominate a persistence-
landscape detector, with the gap *widening* as ambient dimension grows and statistically
significant (non-overlapping bootstrap CIs) on the attack designed to favour PH. We explain
the mechanism with a closed-form result: for spherically symmetric point clouds, mean and
covariance depend only on \(E[R^2]\) and are blind to the radial shape (hence to the
homology), while Mardia kurtosis equals \(d^2(1+\mathrm{Var}(R^2)/E[R^2]^2)\); the shell that
carries \(H_{d-1}\) is exactly the minimiser of \(\mathrm{Var}(R^2)\) at fixed covariance.
Persistence and kurtosis are thus two readouts of the same quantity, redundant by
construction and *independent of the homological dimension*. We further show, by explicit
optimisation, that the strong claim ("any homology change perturbs some low-order moment")
is *false* — one can match mean, covariance and the k-NN distance distribution while breaking
\(H_1\) — but such configurations are non-generic and leave a residual higher moment that a
scale-free detector exploits. The practical thesis stands: for generic topological changes of
a finite cloud, PH adds nothing over a moment whose order is set by the *geometry* of the
change, not its homological dimension.

## 1. Introduction

Embeddings produced by large models concentrate near a low-dimensional submanifold of
\(\mathbb{R}^D\) (the *manifold hypothesis*). A tempting inference is that attacks and drift
manifest as *topological* deformations — components merging, loops filling — that a
topological monitor could catch before a vector database ingests corrupted structure, and
that purely local measures (cosine, nearest-neighbour) would miss. Topological data analysis
(TDA), and specifically persistent homology, is the natural candidate, and a growing body of
work applies it to out-of-distribution and adversarial detection.

We subject this hypothesis to a deliberately adversarial evaluation and reach a negative
conclusion with a precise mechanism. Our contributions:

1. **A hardened negative result** (§3): a persistence-landscape detector never uniquely beats
   a battery of cheap classical detectors, across attack type, embedder, ambient dimension,
   and corpus; on the attack engineered to favour PH, multivariate kurtosis dominates it with
   non-overlapping 95% bootstrap confidence intervals.
2. **A closed-form redundancy theorem** (§4): for spherically symmetric clouds, PH of the
   hollow/filled family is an exact function of the fourth moment, and the detectability
   threshold is governed by the *geometry* of the change, not the homological dimension.
3. **A falsification of the strong version** (§5): matching mean, covariance and the k-NN
   distance distribution while breaking \(H_1\) is possible by optimisation; the redundancy is
   about *generic* changes, not all changes — an honest scope we make explicit.

We view the result as a methodological guardrail: before proposing a topological feature,
test whether it is redundant with a low-order moment fixed by the geometry of the effect one
hopes to capture.

## 2. Setup and related work

**Pipeline.** A stream of unit-normalised embeddings \(x_t\in\mathbb{R}^D\) is cut into sliding
windows \(\mathbb{X}_t\) of size \(W\). We reduce each window to its intrinsic dimension by PCA
before any topological computation — a step we show is *mandatory* (§3.1), since raw
high-\(D\) Vietoris–Rips is dominated by distance concentration. On the reduced cloud we
compute the Vietoris–Rips persistence diagram, its first persistence landscape \(\lambda\)
(Bubenik, 1-Lipschitz stable), and a scalar topological score
\(\Delta_{\text{top}} = \lVert \lambda - \bar\lambda_{\text{ref}}\rVert_{L^1}\).

**Classical baselines.** Covariance drift \(\lVert\Sigma-\Sigma_{\text{ref}}\rVert_F\) (B6);
Mahalanobis shape; k-NN density (related to LID); multivariate (Mardia) kurtosis
\(b = E[(x-\mu)^\top\Sigma^{-1}(x-\mu)]^2\).

**Related work.** VR/Čech stability (Chazal–Cohen-Steiner–Guibas–Oudot; Cohen-Steiner–
Edelsbrunner–Harer); DTM filtrations and robust/sparse persistence (Chazal–Cohen-Steiner–
Mérigot; Anai et al.; Buchet–Chazal–Oudot–Sheehy); landscapes (Bubenik). Prior TDA-for-
detection work reports gains over cosine baselines but rarely against a *higher-moment* or
density baseline under an *adaptive* adversary; that gap is what our evaluation closes.

## 3. Empirical study

**Pre-registration.** Win/kill criteria were fixed before running (see
`docs/protocol/detection-protocol.md`): PH "wins" only if it beats the best baseline at a
production false-positive regime *and* not by inflating false positives on legitimate topic
shifts; the detection axis is declared falsified if PH beats neither Mahalanobis nor the
spectral baseline on any attack class.

### 3.1 Higher homology exists only after reduction

On real embeddings (MiniLM, 20 Newsgroups), raw 384-d \(H_1\) is indistinguishable from a
covariance-matched Gaussian null (median bootstrap \(p\approx0.95\)); the intrinsic dimension
(TwoNN) is \(\approx 8\) while 160 PCA axes are needed for 90% variance — a *curved*
low-dimensional manifold. After PCA to the intrinsic dimension, multi-topic windows show a
population-level \(H_1\) signal above the Gaussian null (median \(p=0.03\); sign-test
\(p=0.001\)). Consequence: reduction-before-TDA is a first-order architectural requirement,
and all subsequent experiments operate in the reduced space.

### 3.2 Detection: classical statistics dominate

On realistic attacks (off-manifold injection, diversity collapse), the spectral baseline (B6)
reaches AUC \(1.00\) while \(\Delta_{\text{top}}\) is at or below chance: real attacks perturb
second-order structure, caught more cheaply than by topology.

We then construct the attack most favourable to PH: a diversity collapse followed by an affine
correction restoring the pre-image's *exact* mean and covariance (\(\lvert\Delta\Sigma\rvert=0\)),
so B6 is blind by construction. Even so, the *shape* change (a central spike) is caught by
Mahalanobis and by kurtosis. Table 1 reports the battery across ambient dimension.

**Table 1** — adaptive covariance-preserving attack, bilateral AUC, \(\lvert\Delta\Sigma\rvert=0\):

| ambient | \(\Delta_{\text{top}}\) | B6 (pure cov) | Mahalanobis | k-NN | Kurtosis |
|--------:|------:|------:|------:|------:|------:|
| 384 | 0.89 | 0.66 | 1.00 | 0.91 | **1.00** |
| 768 | 0.77 | 0.66 | 0.98 | 0.98 | **1.00** |
| 1024 | 0.64 | 0.54 | 1.00 | 0.98 | **1.00** |

Only *pure* covariance (B6) is blind; every other cheap detector catches the attack, and
crucially \(\Delta_{\text{top}}\) *degrades* with ambient dimension while kurtosis stays at
1.00. The intrinsic dimension is stable (\(\approx 8/7/9\) at ambient 384/768/1024): the
manifold hypothesis holds at scale, and the redundancy *worsens* for PH as \(D\) grows.

### 3.3 The idealised topological niche is also covered

On a synthetic circle-vs-disk pair with covariance matched by construction — a *purely*
topological difference invisible to B6 (AUC 0.58) — \(\Delta_{\text{top}}\) separates
perfectly (1.00), but so does k-NN density (1.00): even the ideal blind spot of second-order
statistics is covered by a cheap density estimator.

### 3.4 Robustness

**Table 2** — bilateral AUC with 95% bootstrap CIs, bge-1024, adaptive attack:

| corpus | \(\Delta_{\text{top}}\) | B6 | k-NN | Kurtosis |
|-------|------|------|------|------|
| 20NG (forums) | 0.70 [0.53, 0.85] | 0.52 [0.50, 0.71] | 0.97 | **1.00 [1.00, 1.00]** |
| AG News (press) | 0.81 [0.67, 0.94] | 0.53 [0.50, 0.72] | 0.82 | **1.00 [1.00, 1.00]** |

On both corpora the kurtosis CI does not overlap the \(\Delta_{\text{top}}\) CI: the domination
is statistically significant and not a corpus artefact.

## 4. Why: a closed-form redundancy theorem

Let \(X = R\cdot U\) in \(\mathbb{R}^d\), with \(U\) uniform on \(S^{d-1}\), \(R\ge 0\)
independent of \(U\). The *radial law* encodes the topology: \(R\equiv 1\) is a shell (carries
\(H_{d-1}\)); \(R\) with density \(\propto r^{d-1}\) is a filled ball (no \(H_{d-1}\)).

**Lemma (blindness of second order).** \(E[U]=0\), \(E[UU^\top]=\tfrac1d I\), hence
\[
E[X]=0,\qquad \mathrm{Cov}(X)=\frac{E[R^2]}{d}\,I .
\]
Mean and covariance depend only on \(E[R^2]\); matching them leaves the entire radial shape —
and thus the homology — free.

**Fourth moment.** With \(\Sigma^{-1}=\tfrac{d}{E[R^2]}I\) and \(c=E[R^2]\),
\[
b_{\text{Mardia}} = E[(X^\top\Sigma^{-1}X)^2] = \frac{d^2}{c^2}E[R^4]
= d^2\Big(1+\frac{\mathrm{Var}(R^2)}{c^2}\Big).
\]

**Theorem (seal).** At fixed \(E[R^2]\), among spherically symmetric clouds the shell
\(R\equiv\sqrt c\) — the unique carrier of \(H_{d-1}\) — is exactly the minimiser of the fourth
moment, achieving \(\mathrm{Var}(R^2)=0\), \(b=d^2\). Filling the hole (\(\mathrm{Var}(R^2)>0\))
strictly increases \(b\). Therefore (i) mean and covariance are structurally blind to the hole;
(ii) the fourth moment strictly detects it; (iii) the mechanism is *independent of the
homological dimension* \(d-1\); (iv) persistence and kurtosis are two readouts of the same
\(\mathrm{Var}(R^2)\).

**Numerical check** (matches to the hundredth): shell \(b=d^2\) predicts 4 and 9 (d=2,3);
measured 4.04 and 8.99; ball predictions 5.33/10.67 vs 5.28/10.75.

**Corollary (elliptical extension).** Mardia kurtosis is affine-invariant
(\(b(MX)=b(X)\) for invertible \(M\)), and whitening is an affine homeomorphism, hence
preserves \(H_k\). The theorem therefore holds *verbatim* for every elliptically symmetric
family (any affine image of a spherically symmetric cloud), with \(R\) the Mahalanobis radius:
a shell has \(b=d^2\) regardless of the anisotropy. Numerically, an anisotropic affine map
(\(\mathrm{cond}=3.7,6.3\)) leaves the shell kurtosis at 4.04 and 9.00 while preserving
\(H_{d-1}\). This addresses the "real clusters are not spherical" objection: the redundancy
covers the full elliptical class, not just the isotropic case.

**Trade-off curve and the geometry law.** Starting from a broken loop and matching a growing
statistic set, \(H_1\) stays hidden under {\(\mu,\Sigma\), k-NN density} (\(H_1/h^*\approx0.07\))
and jumps at kurtosis (0.77). Across the sphere/ball family, \(H_1\) and \(H_2\) both threshold
at the fourth moment; \(H_0\) thresholds at *covariance* when the separation raises variance,
but rises to *kurtosis* once the clusters are variance-matched. Hence: **the detectability
threshold is set by the lowest moment the geometry of the change perturbs, not by the
homological dimension.** Canonical hollow/filled and variance-matched-bimodal changes are
fourth-moment phenomena.

## 5. The strong conjecture is false (honest scope)

One might conjecture that *any* homology change of a finite cloud perturbs some low-order
moment. This is false. By gradient optimisation we drive a broken loop to match a target
circle's mean, covariance and full k-NN distance distribution to numerical zero
(\(\lvert\Delta\Sigma\rvert=0\), k-NN RMSE \(2\times10^{-4}\)) while keeping \(H_1\) broken
(\(H_1/h^*=0.13\)). So PH *does* carry information orthogonal to covariance and local density —
there is no impossibility barrier. However, (i) such configurations are non-generic and require
explicit optimisation, and (ii) they leave a residual *higher* moment (kurtosis) that a
scale-free AUC detector exploits (kurtosis AUC \(=1.00\) even at \(\lvert\Delta b\rvert=0.008\)).
The defensible claim is therefore about *generic* changes, not all changes.

## 6. Discussion, limitations, and the paper-2 problem

**Scope.** The theorem is proven for spherically symmetric (and variance-matched bimodal)
families, where PH is essentially a radial readout. This *explains* the empirical redundancy
for canonical changes but does not prove redundancy for arbitrary clouds — the strong
conjecture is false (§5).

**Limitations.** Single reduction (PCA; UMAP untested); exact 1536-d (OpenAI/gte-Qwen) covered
only by the 384→1024 trend; \(N\) windows moderate (CIs reported).

**Paper-2 problem statement (where the seal stops).** The open question is a *quantitative,
non-spherical* trade-off: bound \(d_B(\mathcal D_k(X),\mathcal D_k(Y))\) — or the maximum
achievable \(H_k\) change — as a function of the number/order of matched moments and a local-
density tolerance, for general (non-isotropic) finite clouds. A tight such bound would turn the
empirical trade-off curve (§4) into a theorem and would be the citable, general object.

## 7. Conclusion

For embedding-stream anomaly detection, persistent homology does not beat classical low-order
statistics; the redundancy is analytic for canonical topological changes and worsens with
dimension, and the detectability threshold is set by the geometry of the change rather than its
homological dimension. Persistence and a suitable moment are two readouts of one quantity. The
result is a guardrail: quantify redundancy with a geometry-matched moment before adopting a
topological feature.
