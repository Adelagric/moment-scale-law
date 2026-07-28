# Expériment 0 — le gate topologique

Décide si les embeddings réels portent une homologie supérieure stable **avant**
d'investir dans le moteur Rust. Protocole : [../docs/protocol/detection-protocol.md](../docs/protocol/detection-protocol.md) §2ter.

## Lancer

```bash
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install ripser persim
cd exp0
python validate_engine.py   # garde-fou : topologie connue (2 cercles, tore)
python exp0_gate.py         # H1 réel vs null gaussien, 384-d brut
python exp0_sweep.py        # balayage W × (intra/multi)
python exp0_beta0.py        # persistance H0 multi-échelle
python exp0_reduce.py       # dim intrinsèque + H1 après PCA
python exp0_final.py        # test de population corrigé (n_null=99), espace réduit
```

## Résultat (MiniLM 384-d, 20NG 4 sujets)

| étape | constat |
|-------|---------|
| garde-fou | moteur PASS (H1=2 sur deux cercles, tore) |
| 384-d brut | `H1` ≈ null gaussien (médiane p≈0,95) — concentration |
| dim intrinsèque | TwoNN ≈ **8**, PCA 160 axes/90 % var → variété **courbée** |
| PCA dim=8, multi W=250 | **médiane p=0,03 ; 64 % fenêtres p<0,05 ; signe p=0,001** |
| fragilité | signal intra disparaît à dim=12 → dim = hyperparamètre critique |

**Verdict : POSITIF CONDITIONNEL.** Il faut **réduire à la dim intrinsèque avant
la TDA** ; le moteur Rust doit donc porter un étage de PCA incrémentale.

**Réserves :** un embedder, un corpus, 14 fenêtres/config. À répliquer.
Bug corrigé en route : `n_null=15` rendait `p<0,05` inatteignable (min 1/16≈0,063) ;
`n_null=99` + test de population l'ont levé.

## Exp 1 — pouvoir de détection

```bash
python exp1_detect.py    # Δ_top vs Mahalanobis/B6 sur 3 attaques réelles
python exp1c_matched.py  # test propre : cercle vs disque, covariance égale
```

| test | constat |
|------|---------|
| Exp 1, attaques réelles (INJECT/COLLAPSE/LOOPFILL) | **B6 spectral domine (AUC 1,0 / 1,0 / 0,96)** ; Δ_top ≤ hasard |
| Exp 1c, cercle vs disque, **cov égale par preuve** | **Δ_top AUC=1,0 ; B6 AUC=0,58 (aveugle)** |

**Verdict honnête :** `Δ_top` n'est **pas** un meilleur détecteur général — sur des
attaques réelles qui perturbent le 2ᵈ ordre, la dérive spectrale (B6) gagne, moins cher.
MAIS `Δ_top` capte un **angle mort provable** de B6 : un changement de topologie **à
covariance constante**. La valeur de la thèse n'est donc pas « TDA détecte mieux »
mais **« TDA + B6 = défense en profondeur »** : les attaques naïves tombent sur B6,
les attaques *évasives contraintes à préserver le 2ᵈ ordre* tombent sur TDA. Ça relie
directement la question 2 (adversaire adaptatif / borne d'évasion).

Abandonné : `exp1b_whiten.py` — le blanchiment auto-référentiel pollue B6 par
conditionnement numérique (contradiction Δcov≈0 mais AUC=0,99). Remplacé par exp1c.

## Exp 2 — attaquant adaptatif sur embeddings réels

```bash
python exp2_adaptive.py  # effondrement préservant (mu, Sigma) via correction affine
```

Construction : collapse topologique PUIS correction affine ramenant `(mu, Sigma)` à
ceux de la pré-image saine (affine = homéomorphisme → topologie bouchée gardée,
2ᵈ ordre neutralisé, sans artefact numérique).

| attaque | Δ_top(H1) | Mahalanobis | Spectral B6 |
|---------|-----------|-------------|-------------|
| naïve | 0,97 | 0,00 | **1,00** |
| **adaptative** (cov préservée, `|Δcov|=0`) | **0,77** | 0,00 | **0,41 (aveugle)** |

**Défense en profondeur démontrée sur RÉEL.** L'attaquant qui préserve le 2ᵈ ordre
pour évader B6/Mahalanobis (0,41 / 0,00) reste attrapé par la TDA (0,77). `B6 ∪ Δ_top`
ne laisse aucune évasion par statistiques d'ordre 1–2. La correction affine coûte au
détecteur (0,97→0,77) : la contrainte de covariance atténue le signal sans l'annuler.
Réserves : 1 embedder 384-d, N=24, dim=8, euclidien.

## Exp 2b/2c/2d — mise à l'épreuve d'Exp 2 (l'axe détection tombe)

```bash
python exp2b_scrutiny.py      # dégât + baselines forts (kNN, kurtosis) + contrôle affine
python exp2c_clean_control.py # contrôle propre : distorsion préservant (μ,Σ) ET topologie
python exp2d_niche_test.py    # cercle/disque : kurtosis/kNN capturent-ils la niche ?
```

| test | constat |
|------|---------|
| dégât sémantique (collision) | adaptatif 0,015 ≈ sain 0,010 → attaque **quasi inerte** |
| **kurtosis Mardia** sur adaptatif | **AUC 1,0** → détecteur d'ordre 4 domine `Δ_top` (0,83) |
| contrôle propre `M=Σ^½QΣ^{-½}` | `Δ_top`=0,55 (hasard) → **pas** un artefact affine, juste redondant |
| **cercle/disque** (niche « pure ») | `Δ_top`=1,0 **mais kNN-densité=1,0 aussi** |

**Verdict détection : négatif robuste.** Aucun test ne montre la persistance détecter
ce que covariance (B6) / kurtosis / kNN-densité ne captent déjà — y compris la niche
topologique idéale. Correction d'une erreur en cours de route : `Δ_top` n'est PAS un
artefact affine (le contrôle propre le disculpe), il est **redondant**.

## Exp 2e — dernier test équitable (connexité, terrain propre de la PH)

```bash
python exp2e_connectivity.py   # une boucle vs deux, densité locale appariée
```

Cinq constructions (cercle/disque, clump+pont, barreaux, boucles) : dans **toutes**,
un détecteur bon marché (B6, kNN ou kurtosis) égale ou bat `Δ_top`.

**Mécanisme structurel (le vrai résultat) :** on ne peut pas changer la topologie d'un
nuage fini en gardant *à la fois* covariance ET densité locale fixes — les contraintes
se combattent. Sans appariement affine → B6 voit la covariance. Avec appariement affine
→ la distorsion anisotrope change la densité locale → kNN voit. La persistance ne reçoit
jamais une entrée que l'ordre 1–2 + densité ne savent pas déjà signaler.

## Front 1 — la conjecture d'impossibilité est RÉFUTÉE

```bash
KMP_DUPLICATE_LIB_OK=TRUE python exp3_impossibility.py    # contre-exemple par optimisation
KMP_DUPLICATE_LIB_OK=TRUE python exp3b_niche_via_optim.py # panel détecteurs sur attaque optimisée
```

- **exp3** : par descente de gradient, on casse `H1` (1,58→0,20, ratio 0,13) en appariant
  covariance ET distribution kNN **exactement** (RMSE 0,0002). → la conjecture forte
  (« topologie change ⇒ covariance ou densité locale change ») est **FAUSSE**. Aucune
  barrière de fond : la topologie porte de l'info orthogonale à covariance + densité locale.
- **exp3b** : mais ça ne ressuscite pas la détection. Même `|Δkurtosis|=0,008`, la kurtosis
  sépare à AUC 1,0 — l'AUC est *sans échelle*, un résidu systématique minuscule sépare
  quand la classe saine est homogène. Apparier à résidu-zéro ≠ apparier dans la variance
  intra-classe.

**Objet théorique correct : un compromis quantitatif, pas un binaire.** « Combien de
topologie peut-on cacher en appariant les N premiers moments + la densité locale ? »
Ni impossibilité, ni niche propre : une borne de compromis à caractériser.

## Front 1 rebaptisé — la courbe de compromis (résultat)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python exp3c_tradeoff.py   # courbe H1, niveaux emboîtés
KMP_DUPLICATE_LIB_OK=TRUE python exp3d_dimension.py  # H0 vs H1 + figure
```

Protocole : init = boucle CASSÉE ; on apparie un jeu croissant de stats de la cible
(cercle) et on mesure le `H1` final. Bas = topologie cachable ; haut = l'appariement force.

| niveau apparié | `H1`/h* |
|----------------|---------|
| μ | 0,00 |
| μ, Σ | 0,07 |
| + densité kNN | 0,07 |
| **+ kurtosis (4ᵉ moment)** | **0,77 ← saut** |
| + distribution des distances | 0,87 |

**Résultat `H1` : le seuil de compromis est le 4ᵉ moment.** La topologie de trou est
cachable sous {ordre ≤ 2 + densité locale} mais forcée par la kurtosis. Ça *explique*
pourquoi la persistance n'a jamais battu la kurtosis en détection de trous : pour un
trou, elles mesurent presque la même chose (un cercle-coquille a Mahalanobis²≈2 partout ;
boucher le trou change la distribution radiale → change la kurtosis).

Figure : `tradeoff_curve.png`.

## Front 1 — point 1 (loi du seuil) + point 2 (preuve)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python exp3e_sphere_ball.py   # H0/H1/H2, famille sphère/boule
KMP_DUPLICATE_LIB_OK=TRUE python exp3f_theory_check.py  # vérif du lemme + forme close
```

**Point 1 — la loi n'est PAS dimensionnelle.** Sur la famille `k`-sphère → `(k+1)`-boule :

| changement | seuil (1er moment forçant) |
|------------|----------------------------|
| H0, amas écartés (variance ↑) | **covariance** (2ᵉ) |
| H0, amas à variance appariée | **kurtosis** (4ᵉ) |
| H1 (cercle → disque) | **kurtosis** (4ᵉ) |
| H2 (sphère → boule) | **kurtosis** (4ᵉ) |

→ **Le seuil est fixé par le plus bas moment que la géométrie du changement perturbe,
pas par la dimension homologique.** Les changements creux/plein et bimodal-à-variance-
appariée sont des propriétés du 4ᵉ moment.

**Point 2 — SCELLÉ (preuve).** Pour `X = R·U` sphériquement symétrique : `Cov = (E[R²]/d)I`
(aveugle à la forme radiale) et `kurtosis Mardia = d²(1 + Var(R²)/E[R²]²)`. La coquille
(seule à porter `H_{d-1}`) minimise `Var(R²)=0` à covariance fixée → persistance et
kurtosis lisent la même `Var(R²)` : **redondance analytique, indépendante de la dimension**.
Forme close vérifiée au centième (coquille `b=d²` : prédit 4 et 9, mesuré 4,04 et 8,99).
Preuve complète : [docs/theory/tradeoff-theorem.md](../docs/theory/tradeoff-theorem.md).

## Paper 1 — réplique à dimension ambiante croissante (le négatif n'est pas un artefact 384-d)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python exp4_highdim.py "sentence-transformers/all-mpnet-base-v2" 768 emb_mpnet768.npz
KMP_DUPLICATE_LIB_OK=TRUE python exp4_highdim.py "BAAI/bge-large-en-v1.5" 1024 emb_bge1024.npz
```

| ambiant | dim intrinsèque (TwoNN) | PCA 90 % | redondance (réduit) |
|---------|------------------------|----------|---------------------|
| 384 (MiniLM) | ~8 | 160 | tient |
| 768 (mpnet) | ~7 | 189 | tient |
| 1024 (bge-large) | ~9,4 | 201 | tient (Δ_top 0,90 < baselines 1,0) |

**Dim intrinsèque stable ~7–9 alors que l'ambiante triple** → hypothèse de variété robuste.
**Redondance partout** (réduit ET brut) : kurtosis/kNN ≥ `Δ_top` ; l'écart se creuse même
avec la dimension. Le négatif de détection tient à l'échelle. *Réserve : l'attaque collapse
est « forte » (tous les détecteurs à 1,0), non discriminante — un balayage adaptatif
haute-dim (exp2-style à 1024-d) le renforcerait. 1536-d exact (gte-Qwen/OpenAI) = suite
directe. La forme close du théorème est indépendante de d par construction.*

### Blindage — attaque adaptative (cov-préservée) à dimension croissante

```bash
KMP_DUPLICATE_LIB_OK=TRUE python exp4b_adaptive_highdim.py emb_cache.npz 8
KMP_DUPLICATE_LIB_OK=TRUE python exp4b_adaptive_highdim.py emb_mpnet768.npz 7
KMP_DUPLICATE_LIB_OK=TRUE python exp4b_adaptive_highdim.py emb_bge1024.npz 9
```

Attaque collapse + correction affine préservant (μ,Σ) — conçue pour FAVORISER la TDA.
AUC bilatérale (`|Δcov|=0` partout) :

| ambiant | Δ_top | B6 (cov pure) | Maha | kNN | **Kurtosis** |
|---------|-------|---------------|------|-----|--------------|
| 384 | 0,89 | 0,66 | 1,00 | 0,91 | **1,00** |
| 768 | 0,77 | 0,66 | 0,98 | 0,98 | **1,00** |
| 1024 | 0,64 | 0,54 | 1,00 | 0,98 | **1,00** |

**Même l'attaque favorable à la TDA est dominée par la kurtosis (1,00) à toute dimension,
et `Δ_top` se DÉGRADE quand la dimension monte (0,89→0,64).** Seule B6 (covariance pure)
est aveugle par construction ; tout le reste de la batterie attrape l'attaque. La redondance
ne fait que s'aggraver pour la persistance à l'échelle. Blindage empirique du papier 1 :
2 types d'attaque × 3 embedders × 3 dimensions.

### Robustesse — 2ᵉ corpus + IC bootstrap

```bash
KMP_DUPLICATE_LIB_OK=TRUE python exp5_robustness.py   # 20NG + AG News, IC bootstrap 95%
```

AUC bilatérale [IC 95% bootstrap], bge-1024, dim réduite 9 :

| corpus | attaque | Δ_top | B6 | kNN | Kurtosis |
|--------|---------|-------|-----|-----|----------|
| 20NG | adaptatif | 0,70 [0,53–0,85] | 0,52 [0,50–0,71] | 0,97 | **1,00 [1,00–1,00]** |
| AG News | adaptatif | 0,81 [0,67–0,94] | 0,53 [0,50–0,72] | 0,82 | **1,00 [1,00–1,00]** |

**Sur l'attaque favorable à la TDA, l'IC de la kurtosis (1,00) ne chevauche PAS celui de
`Δ_top` (borne haute 0,85 / 0,94) — domination statistiquement significative, sur deux
corpus indépendants (forums vs presse).** Le négatif n'est ni un artefact de corpus ni du
bruit d'échantillonnage. Base empirique paper 1 = 2 attaques × 3 embedders × 3 dims × 2
corpora, avec IC. Réserve résiduelle : réduction PCA (vs UMAP), 1536-d exact — gold-plating.
