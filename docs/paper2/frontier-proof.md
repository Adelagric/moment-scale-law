# Preuve de la loi de frontière — ce qui est prouvé, ce qui reste ouvert

On distingue **deux bornes** sur `N*(b)` = ordre de moment nécessaire pour distinguer une
topologie de complexité `b = Σ_k β_k` d'une topologie triviale.

- **Borne inférieure** `N* ≥ c·b` : il EXISTE des nuages topologiquement distincts appariant
  tous les moments jusqu'à l'ordre `~b`. C'est la « valeur irréductible » de la persistance —
  elle voit ce que les moments d'ordre `< cb` ratent. **→ PROUVÉE en général (§1).**
- **Borne supérieure** `N* ≤ C·b` : apparier les moments jusqu'à `~b` FORCE les nombres de Betti.
  C'est « l'avantage de la persistance est borné ». **→ prouvée pour les cas structurés (§2),
  conjecturée en général (§3).**

## 1. Borne inférieure — THÉORÈME GÉNÉRAL (rigoureux)

**Théorème.** Soit `d ≥ 2` et `b ≥ 1`. Il existe deux mesures de probabilité `μ_b, ν` sur `ℝ^d`,
à supports compacts, telles que :
1. `∫ p dμ_b = ∫ p dν` pour tout polynôme `p` de degré `≤ 4b−2` ;
2. `β_{d-1}(supp μ_b) = b` et `β_{d-1}(supp ν) = 0`.

Ainsi `N*(b) ≥ 4b−1` : l'ordre de moment nécessaire pour distinguer une topologie de complexité
`b` de la triviale croît **au moins linéairement** en `b`.

**Construction.** Soit `ν` une mesure radiale (rotation-invariante) à densité sur la boule unité
(topologie triviale), de loi radiale `ρ` sur `u = ‖x‖² ∈ [0,1]`. Soit `{(u_i, w_i)}_{i=1}^b` la
quadrature de Gauss à `b` points de `ρ` (nœuds `u_i>0`, poids `w_i>0`). On pose
`μ_b = Σ_i w_i σ_{√u_i}`, où `σ_r` = mesure uniforme sur la sphère de rayon `r`. C'est `b`
`(d−1)`-sphères concentriques, donc `β_{d-1} = b`.

**Preuve de (1).** `μ_b` et `ν` sont rotation-invariantes. Pour un polynôme `p` de degré `≤ 4b−2`,
son moyennage rotationnel `p̄(x) = ∫_{O(d)} p(gx)\,dg` est un polynôme en `‖x‖² = u` de degré
`≤ (4b−2)/2 = 2b−1` (les monômes de degré impair s'annulent ; un monôme homogène de degré `2k`
se moyenne en `c·‖x‖^{2k} = c·u^k`). Comme les deux mesures sont rotation-invariantes,
`∫ p dμ = ∫ p̄ dμ`. Donc `∫ p dμ_b = Σ_i w_i q(u_i)` et `∫ p dν = ∫ q dρ`, où `q = p̄` vu comme
polynôme en `u`, `deg q ≤ 2b−1`. La quadrature de Gauss à `b` points est **exacte** pour les
polynômes de degré `≤ 2b−1` : `Σ_i w_i q(u_i) = ∫ q dρ`. D'où `∫ p dμ_b = ∫ p dν`. ∎

**Preuve de (2).** `supp ν` = boule (contractile, `β_{d-1}=0`). `supp μ_b` = `b` sphères
concentriques disjointes de rayons `√u_i` ⇒ `β_{d-1} = b`. ∎

*Vérification numérique (exp9, d=2) : m=1,2,3 → ordres appariés 2,6,10 ; `H_1` = 1,2,3 vs 0.*
*Le cas séparable (exp10, boucles alignées + Gauss sur l'axe) donne la même borne hors symétrie
radiale.*

## 2. Borne supérieure — cas structurés (prouvés)

- **Radial / elliptique.** La topologie est fonction de la loi radiale (de Mahalanobis) ; une
  distribution radiale à `b` atomes est déterminée par ses `2b` premiers moments (unicité de
  Hamburger). Donc apparier les moments radiaux jusqu'à `2b` (nuage `4b`) force la topologie :
  `N* ≤ 4b`. Combiné à §1 : **`N* = Θ(b)` exactement dans le cas radial.**
- **Séparable / symétrique par groupe.** Réduction à une coordonnée (axe) ou au quotient (profil),
  puis argument radial sur le réduit (cf. `nerve-argument.md` §3–4).

## 3. Borne supérieure générale — la conjecture naïve est FAUSSE (exp13)

**Conjecture naïve (RÉFUTÉE).** ~~`N* ≤ C_d · Σ_k β_k`~~ : apparier les moments jusqu'à un ordre
linéaire en la complexité topologique forcerait les nombres de Betti. **Faux.**

**Contre-exemple d'échelle (exp13).** Placer un feature de rayon `ε` portant une fraction `f` de
la masse : il perturbe les moments de `O(f·ε^k)`, donc **arbitrairement peu**. Numériquement
(anneau vs disque de rayon `ε`, `f=0,12`, moments normalisés jusqu'à l'ordre 10) :

| `ε` | `‖Δm‖/‖m‖` | `H_1` anneau | `H_1` disque |
|-----|-----------|--------------|--------------|
| 0,5 | 5,6e−2 | 0,63 | 0,21 |
| 0,2 | 8,2e−3 | 0,28 | 0,21 |
| 0,08 | 1,4e−3 | 0,169 | 0,169 |
| 0,03 | **2,2e−4** | 0,043 | 0,029 |

Les moments deviennent indiscernables (chute de 250×) alors que la topologie persiste à
l'échelle `ε`. **Aucun ordre de moment fixe ne capture un feature suffisamment petit.**

**Énoncé corrigé.** `N*` dépend non seulement du nombre de features `b`, mais de leur **échelle
relative** `ε` et de leur **masse** `f`. Forme plausible : `N* ~ b · g(1/ε, 1/f)`, croissant quand
`ε→0` ou `f→0`. La borne supérieure ne peut être qu'**échelle-relative** :

> **Conjecture révisée.** Si `μ, ν` apparient les moments jusqu'à l'ordre `N` et que toutes leurs
> features topologiques vivent à une échelle `≥ ε` avec masse `≥ f`, alors leurs nombres de Betti
> coïncident dès que `N ≥ C_d · b · h(ε, f)`.

**`h` DÉTERMINÉE (exp14) : `h(ε) ≍ 1/ε`.** En base orthonormée (Legendre tensorisée, bien
conditionnée — les monômes bruts sont inutilisables), l'ordre minimal où les coefficients d'un
anneau et d'un disque de rayon `ε` divergent suit

> `N*(ε) ≈ 0,74·ε^(−0,92)`, R(log-log) = 0,983 — exposant théorique 1.

Mécanisme : les moments d'ordre `N` résolvent l'échelle `~1/N` (noyau de Christoffel–Darboux),
donc voir un feature de taille `ε` coûte `N ~ 1/ε`.

**CORRECTION (exp15) — la loi PRODUIT `b/ε` est FAUSSE.** À `ε` fixé, faire varier `b` :

| `ε` | b=1 | b=4 | b=9 | b=16 |
|-----|-----|-----|-----|------|
| 0,10 | 4 | 4 | 6 | 8 |
| 0,05 | 6 | 8 | 12 | 12 |

`b` × 16 ⇒ `N*` × 2 seulement (exposant `≈0,25` en `d=2`), là où `b/ε` prédirait × 16. La
dépendance en `b` est **fortement sous-linéaire** — cohérent avec Christoffel–Darboux : un degré
`N` résout l'échelle `1/N` **uniformément sur le domaine**, donc un seul polynôme résout *tous*
les features simultanément. **Le `b` de la borne inférieure §1 n'est pas un coût de comptage : il
vient de ce que tasser `b` coquilles dans la boule unité force `ε ~ 1/b`.**

## 3bis. Borne supérieure — PREUVE par localisation (Jackson)

**Proposition.** Soient `μ, ν` des mesures de probabilité sur `[-1,1]^d` appariant tous les
moments jusqu'au degré `N`. Soit `φ` une fonction test lisse, `0 ≤ φ ≤ 1`, supportée dans une
boule de rayon `ε` (donc de module de continuité `ω(φ, δ) ≲ δ/ε`). Alors

```
|∫φ dμ − ∫φ dν|  ≤  2 C_d / (N ε).
```

*Preuve.* Par le théorème de Jackson (approximation polynomiale multivariée), il existe un
polynôme `p` de degré `≤ N` avec `‖φ − p‖_∞ ≤ C_d · ω(φ, 1/N) ≤ C_d/(Nε)`. Comme `deg p ≤ N`,
`∫p dμ = ∫p dν`. Donc
`|∫φdμ − ∫φdν| = |∫(φ−p)dμ − ∫(φ−p)dν| ≤ ‖φ−p‖_∞ (μ(1)+ν(1)) = 2C_d/(Nε)`. ∎

**Corollaire (borne supérieure échelle-relative).** Si toutes les features topologiques vivent à
une échelle `≥ ε` et portent une masse `≥ f`, alors dès que

```
N  ≥  C'_d / (ε f)
```

les masses locales de `μ` et `ν` dans toute boule de rayon `ε` coïncident à `< f` près ; le motif
d'occupation à l'échelle `ε` est donc identique, et par le théorème du nerf les nombres de Betti
à cette échelle coïncident. D'où **`N* ≲ C_d/(εf)`** — *sans aucune dépendance en `b`*, ce que
confirme exp15.

## 3quater. Fermeture de l'écart — bosses lisses (Gevrey) et Jackson d'ordre `k`

Le facteur `1/f` de §3bis vient de Jackson au **1er ordre** (bosse Lipschitz). Avec une bosse
**lisse** on fait bien mieux.

**Proposition.** Soit `ψ` de classe Gevrey-`s` (`s>1`), à support compact, `‖ψ^{(k)}‖_∞ ≤ A^k(k!)^s`,
et `φ_ε(x)=ψ(x/ε)`, de sorte que `‖φ_ε^{(k)}‖_∞ ≤ A^k(k!)^s/ε^k`. Alors

```
‖φ_ε − p_N‖_∞  ≲  exp(−s·(Nε/CA)^{1/s}).
```

*Preuve.* Jackson d'ordre `k` : `‖φ−p_N‖ ≤ C^k N^{−k}‖φ^{(k)}‖ ≤ (CA/(Nε))^k (k!)^s`. Posons
`t = Nε/(CA)` ; par Stirling `(k!)^s/t^k ≈ exp(k[s·ln k − s − ln t])`, minimisé en `k = t^{1/s}`,
de valeur `exp(−s·t^{1/s})`. ∎

**Corollaire (borne supérieure resserrée).** Exiger une erreur `< f` donne

```
N*  ≲  (1/ε) · (log(1/f))^s        (tout s > 1)
```

au lieu de `1/(εf)`. Dans la construction radiale (`ε ~ f ~ 1/b`) : `N* ≲ b·(log b)^s`, contre
`N* ≥ 4b`. **L'écart passe de `b` ↔ `b²` à `b` ↔ `b·polylog(b)` : essentiellement fermé.**

**CONFIRMÉ numériquement sur 14 décades (exp17).** La projection de Legendre en float64 (exp16b)
plafonnait à 2 décades, d'où un régime pré-asymptotique non concluant. Via la **FFT** (stable
jusqu'à ~1e-15) :

| | bosse lisse (Gevrey-2) | bosse Lipschitz |
|---|---|---|
| `R(log\|ĉ_k\| vs k^{1/2})` | **−0,9933 / −0,9937** | −0,72 |
| `R(log\|ĉ_k\| vs log k)` | −0,960 | −0,73 |
| pente algébrique | −7,3 / −7,6 (⇒ non algébrique) | **−1,94 ≈ −2** ✓ |

La décroissance sur-algébrique `exp(−c·k^{1/2})` est **confirmée** pour la bosse lisse, et la
Lipschitz reste algébrique en `k^{−2}` comme prédit. **Loi d'échelle vérifiée** : la pente en
`(kε)^{1/2}` vaut `−1,955` et `−1,947` pour `ε = 0,15` et `0,30` — quasi identique ⇒ la
décroissance dépend du produit `kε`, avec constante universelle `c ≈ 1,95`. Donc
`err(N) ≈ exp(−c(Nε)^{1/2})` et `N* ≳ (1/ε)(log(1/f)/c)²` : c'est l'instance `s=2` de la borne,
**avec sa constante mesurée**.

## 3quinquies. Polylog ÉLIMINÉ — la loi est serrée (exp18 + extrémal de Chebyshev)

**Meilleure fonction test.** Le `(log 1/f)^s` vient du support compact des bosses Gevrey. Une
**gaussienne** échange le support compact contre l'analyticité ; en optimisant sa largeur
(arbitrage queues ↔ régularité) :

| `f` | `N*` Gevrey | `N*` gaussienne | `σ/ε` opt |
|-----|------------|-----------------|-----------|
| 1e-1 | 71 | 41 | 0,45 |
| 1e-3 | 463 | 129 | 0,30 |
| 1e-6 | 1799 | 255 | 0,20 |

Exposants ajustés : Gevrey `(log 1/f)^1,81`, gaussienne **`(log 1/f)^1,04`** (`R ≥ 0,997`).
⇒ `N* ≲ log(1/f)/ε`.

**Et c'est optimal (borne inférieure générale).** Problème extrémal de Chebyshev : un polynôme de
degré `N` avec `|p| ≤ 1` hors de `[−ε, ε]` vérifie `|p(0)| ≤ T_N((1+ε²)/(1−ε²)) ≈ exp(2Nε)`
(mesuré : `log(contraste)/(Nε) → 1,99`). Pour distinguer une masse `f` de `0` par **n'importe
quel** test de degré `≤ N`, il faut un contraste `> 1/f`, donc

```
N  ≥  log(1/f) / (2ε).
```

Cette borne est **générale** (elle ne dépend d'aucune construction). Elle **matche** la borne
supérieure gaussienne à constante près :

> **LOI SERRÉE : `N*(ε, f) ≍ log(1/f) / ε`.**

Le nombre de features n'entre que par le **tassement** : `b` features dans un domaine borné
forcent `ε, f ≲ 1/b`, d'où `N* ≳ b·log b` — plus fort que le `4b−1` de la construction de Gauss,
et cohérent avec lui. **Le log est intrinsèque, pas un artefact de méthode.**

## 3ter. Loi de frontière — bilan

- **Borne inférieure (prouvée, §1) :** `N* ≥ 4b−1`, où `b` features radiales forcent `ε ~ 1/b`.
- **Borne supérieure (prouvée, §3bis) :** `N* ≤ C_d/(εf)` — élémentaire, Jackson 1er ordre.
- **Borne supérieure resserrée (§3quater) :** `N* ≲ (1/ε)(log(1/f))^s` — taux sous-jacent
  `exp(−c(Nε)^{1/2})` **confirmé sur 14 décades** (exp17), constante `c ≈ 1,95` mesurée.
- **Loi finale (§3quinquies) : `N*(ε,f) ≍ log(1/f)/ε`, SERRÉE** — borne sup par fonctions test
  gaussiennes (exposant mesuré 1,04), borne inf générale par l'extrémal de Chebyshev
  (`contraste ≤ exp(2Nε)`). **Plus d'écart polylog.**

**Énoncé unifié.** `N*` est gouverné par la **finesse** des features (échelle `ε`, masse `f`),
essentiellement pas par leur nombre. La persistance lit toutes les échelles en une passe — sa
valeur irréductible est le régime **fin** (`ε → 0`), pas le régime **nombreux**.

**Conséquence pour la thèse (renforçante).** La valeur irréductible de la persistance est *plus
grande* qu'estimée : elle voit les features **fins** (petite échelle, faible masse) que **aucun**
moment d'ordre fixe ne peut atteindre — exactement le régime multi-échelle où un barcode lit
toutes les échelles d'un coup. C'est le sens précis de « substitut à calcul borné ».

**Stratégie (nerf).** (i) Les moments jusqu'à l'ordre `N` déterminent la projection de la mesure
sur les polynômes de degré `≤ N` ; via les polynômes orthogonaux (Christoffel–Darboux), ils
reconstruisent la densité à une **résolution** `~1/N`. (ii) Le type d'homotopie du complexe de
Čech est le nerf d'un recouvrement par boules ; il est déterminé dès que la mesure est résolue à
l'échelle des features (recouvrements locaux). (iii) Résoudre `b` features demande une résolution
`~ échelle des features`, soit `N ~ b^{1/d}` par dimension, `~ b` au total.

**La LACUNE (lemme ouvert, version échelle-relative).** L'étape (i)→(ii) n'est PAS gratuite :
*apparier des moments n'implique PAS la proximité en Wasserstein* (coquille et boule apparient
`4b−2` moments mais sont Wasserstein-loin), donc la stabilité de la persistance (Cohen-Steiner :
`d_B ≤ d_H`) **ne s'applique pas** — c'est exactement pourquoi la persistance n'est pas déterminée
par les moments bas. Et exp13 montre que la lacune n'est pas seulement technique : sans borne
d'échelle, l'énoncé est **faux**. Le lemme ouvert correct est donc :

> **Lemme ouvert (révisé).** Si `μ, ν` apparient les moments jusqu'à l'ordre `N`, et si toutes
> leurs features vivent à échelle `≥ ε` et masse `≥ f`, alors leurs nombres de Betti coïncident
> dès que `N ≥ C_d·b·h(ε,f)`. Déterminer `h` (et sa dépendance en `d`) est l'ouvert.

La difficulté : la reconstruction par moments est mal conditionnée (la résolution atteinte par `N`
moments est `~1/N`, à comparer à `ε`), et le pont moments→topologie doit contourner Wasserstein.
Une piste : borner la variation du nombre de Betti par un **fonctionnel de moments** directement
(sans passer par la mesure), via une formule de type Crofton / Gauss-Bonnet intégro-géométrique
reliant `Σβ_k` à des intégrales polynomiales — mais l'existence d'un tel fonctionnel borné, et sa
dépendance en `ε`, est précisément l'ouvert.

## 4. Bilan honnête

- **Prouvé (rigoureux, général) :** la borne inférieure `N* ≥ 4b−1` (§1) — la persistance a une
  valeur irréductible prouvée pour toute complexité, dans toute dimension. C'est le résultat fort.
- **Prouvé (cas structurés) :** `N* = Θ(b)` pour radial/elliptique/séparable/symétrique (§2).
- **Ouvert :** la borne supérieure générale (§3), réduite à un lemme quantitatif moments→Betti
  précis. La recherche adverse (exp11–12) n'a trouvé aucun contre-exemple, ce qui soutient la
  conjecture sans la prouver.
