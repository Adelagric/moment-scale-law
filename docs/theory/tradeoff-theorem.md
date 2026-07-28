# Sceau théorique — pourquoi la persistance est redondante avec le 4ᵉ moment

> Résultat du Front 1. Remplace la « conjecture d'impossibilité » (réfutée, exp3) par un
> énoncé positif, prouvé, et vérifié numériquement au centième (exp3f).

## 1. Cadre

Soit `X ∈ ℝ^d` un vecteur aléatoire **sphériquement symétrique** :
`X = R · U`, où `U` est uniforme sur la sphère `S^{d-1}` et `R ≥ 0` est le rayon,
indépendant de `U`. La **forme radiale** (la loi de `R`) encode la topologie :

- **coquille** (`(d-1)`-sphère) : `R ≡ 1` → porte l'homologie `H_{d-1}` ;
- **boule pleine** : `R` de densité `∝ r^{d-1}` → `H_{d-1} = 0` (trou bouché).

Entre les deux, tout mélange radial interpole continûment le remplissage du trou.

## 2. Lemme (aveuglement du 2ᵈ ordre)

Pour `U` uniforme sur `S^{d-1}` : `E[U] = 0` et `E[U Uᵀ] = (1/d) I_d`. Donc, par
indépendance `R ⊥ U` :

```
E[X] = 0,        Cov(X) = E[R²] · E[U Uᵀ] = (E[R²]/d) · I_d.
```

**Moyenne et covariance ne dépendent que de `E[R²]`** — pas de la forme radiale.
Conséquence directe : deux distributions sphériquement symétriques de même `E[R²]` ont
**exactement les mêmes moments d'ordre 1 et 2**, quelle que soit leur topologie. Apparier
`(μ, Σ)` revient à apparier le seul scalaire `E[R²]`, laissant toute la forme radiale
— donc `H_{d-1}` — libre. *(Vérifié exp3f : coquille et boule à `E[R²]=1` ont `|Δcov|`
au niveau du bruit d'échantillonnage.)*

## 3. Le 4ᵉ moment lit exactement la variance du rayon²

La kurtosis multivariée de Mardia est `b = E[(Xᵀ Σ⁻¹ X)²]`. Avec `Σ⁻¹ = (d/E[R²]) I` :

```
Xᵀ Σ⁻¹ X = (d/E[R²]) ‖X‖² = (d/E[R²]) R².
```

D'où, en posant `c = E[R²]` :

```
b = (d²/c²) · E[R⁴] = (d²/c²) · (Var(R²) + c²) = d² · ( 1 + Var(R²)/c² ).
```

**Forme close.** À `E[R²] = c` fixé (donc à covariance fixée), `b` est une fonction
**strictement croissante de `Var(R²)`**, la variance du rayon au carré.

*Vérification exp3f — la théorie tombe au centième :*

| | prédit `b = d²(1+Var(R²)/c²)` | mesuré |
|---|---|---|
| coquille d=2 | `4·(1+0) = 4` | 4,04 |
| coquille d=3 | `9·(1+0) = 9` | 8,99 |
| boule d=2 | `4·(1+0,332) = 5,33` | 5,28 |
| boule d=3 | `9·(1+0,185) = 10,67` | 10,75 |

## 4. Théorème (sceau)

> **À `E[R²]` fixé, parmi les distributions sphériquement symétriques, la coquille
> `R ≡ √c` — l'unique porteuse de `H_{d-1}` — est exactement celle qui MINIMISE le 4ᵉ
> moment `b`, atteignant `Var(R²)=0`, `b = d²`. Tout remplissage du trou (`Var(R²)>0`)
> augmente STRICTEMENT `b`.**

Corollaires :

1. **`(μ, Σ)` sont structurellement aveugles** au trou (§2) : ils ne voient que `E[R²]`.
2. **Le 4ᵉ moment le détecte strictement** : il est minimal exactement sur la coquille.
3. **La dimension homologique `d-1` n'intervient pas dans le mécanisme** — il vaut pour
   tout `d`. C'est pourquoi `H_1` (d=2) et `H_2` (d=3) franchissent le seuil au *même*
   endroit (la kurtosis), empiriquement (exp3e).
4. **Persistance `H_{d-1}` et kurtosis sont deux lectures de la même quantité `Var(R²)`**
   (concentration radiale). Leur redondance n'est pas un accident empirique : elle est
   analytique. La persistance n'apporte, pour cette famille, rien au-delà du 4ᵉ moment.

## 5. Le cas `H_0` (connexité) — même mécanisme, moment fixé par la géométrie

Deux amas symétriques `±a` (variance intra `w²`, 1D) : `E[X²] = a² + w²` (apparié en
choisissant `a`), `E[X⁴] = a⁴ + 6a²w² + 3w⁴`. À variance fixée, la configuration bimodale
(`H_0 = 2`) a une kurtosis distincte (plus basse : platykurtique) de l'unimodale connexe.
Donc **la connexité à variance appariée est aussi un phénomène de 4ᵉ moment** (exp3e :
seuil `H_0` qui monte de la covariance à la kurtosis dès qu'on apparie la variance).

Si en revanche la séparation *augmente* la variance (amas plus écartés), elle est captée
dès le **2ᵈ moment**. D'où la loi générale :

> **Le seuil de compromis n'est pas fixé par la dimension homologique, mais par le plus
> bas moment que le changement géométrique perturbe.** Les changements canoniques
> (creux/plein, bimodal à variance appariée) sont des propriétés du 4ᵉ moment.

**Théorème d'unification (paper 2, exp7b).** À `(μ,Σ)` fixés, un mélange bimodal
symétrique (masses en `±a`, variance intra `w²`) est **platykurtique** le long de l'axe de
séparation : sa kurtosis de Mardia est *strictement sous* la valeur gaussienne `d(d+2)`,
alors qu'un nuage unimodal allongé de même `(μ,Σ)` (`H_0=1`) est mésokurtique. Numériquement
(d=3) : bimodal `13,6` vs allongé `15,0 = d(d+2)`, `|Δcov|` au plancher d'échantillonnage,
`H_0` prominence 1,3 vs 0,1. Donc la **connexité, comme le creux, est un phénomène de 4ᵉ
moment à 2ᵈ moment fixé**. Combiné à la coquille/boule :

> À `(μ,Σ)` fixés, les changements topologiques canoniques — bimodalité `H_0` ET creux
> `H_{≥1}` — sont **tous** gouvernés par le 4ᵉ moment, indépendamment de la dimension
> homologique. L'ordre du seuil = le plus bas moment perturbé (génériquement 4).

**Mélanges (limite honnête).** Apparier les moments *globaux* ne contraint que faiblement
la topologie *par composante* (la masse se redistribue) : l'approche par optimisation sur
mélanges (exp7) n'a PAS donné de courbes propres ; l'énoncé par composante tient via la
construction contrôlée (exp7b) + la décomposition de covariance totale.

## 7. Frontière — loi complexité↔moment (exp9, construction à la main via Gauss)

**Théorème (N\*(m) = 4m).** La quadrature de Gauss à `m` points d'une mesure radiale hole-free
donne `m` coquilles concentriques dont les moments nuage coïncident avec la boule triviale
jusqu'à l'ordre `4m−2` (exactitude de Gauss pour degré `2m−1` en `R²` ; nuages sphériques ⇒
tous les moments sont radiaux), alors que les coquilles portent `m` features homologiques et la
boule aucune. Donc **l'ordre de moment pour épingler une topologie à `m` échelles croît
LINÉAIREMENT** — aucun moment d'ordre fixe ne double la persistance pour une topologie riche.
Vérifié m=1,2,3 (ordres appariés 2,6,10 ; `H_1` = 1,2,3 vs 0). Sans optimisation.

**Interprétation.** `m=1` redonne `N*=4` (redondance kurtosis = le régime de la détection) ;
la vraie valeur de la persistance est le régime `m` grand (topologie multi-échelle), où l'ordre
de moment équivalent est prohibitif. Ça **délimite** la persistance au lieu de la disqualifier.

**Reste ouvert :** topologie globale non radiale (tore, cycles enlacés, variété courbée dim
intrinsèque ≪ ambiante = vrais embeddings) ; conjecture : même loi (ordre ~ nombres de Betti)
via argument local-à-global (nerf).

## 6. Portée honnête

- Ceci **explique** les négatifs de détection (Exp 1–2e) par un mécanisme exact, pour les
  familles canoniques sphériquement symétriques et bimodales.
- Ceci ne prouve **pas** que *toute* différence topologique se réduit aux moments : la
  conjecture forte est fausse (exp3 construit, par optimisation, des nuages appariant
  `(μ, Σ)` + densité kNN avec `H_1` cassé). Mais de telles configurations sont non
  génériques / coûteuses à instancier, et laissent un 4ᵉ moment résiduel qu'un détecteur
  AUC exploite (exp3b).
- La contribution défendable : **pour les changements topologiques *génériques* d'un nuage,
  la persistance est analytiquement redondante avec un moment d'ordre bas ; l'ordre est
  fixé par la géométrie du changement, pas par la dimension homologique.**
