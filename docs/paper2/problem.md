# Paper 2 — la borne de compromis générale (non sphérique)

Thread de recherche vivant, ouvert pendant la rédaction du papier 1. Objectif : le
*théorème quantitatif général* qui transforme la courbe empirique de compromis (paper 1 §4)
en énoncé, et qui est l'objet citable pour lui-même.

## Problème

Pour un nuage fini `X ⊂ ℝ^d`, borner la variation de persistance atteignable
`sup d_B(𝒟_k(X), 𝒟_k(Y))` (ou le max de changement de `H_k`) sous contraintes :
`Y` apparie les `N` premiers moments de `X` (à tolérance `ε_m`) et sa densité locale
(distribution kNN, à tolérance `ε_ρ`). Question : une borne de la forme

> pour changer `d_B(𝒟_k)` de `δ`, il faut perturber un moment d'ordre ≤ `f(k, géométrie)`
> d'au moins `g(δ)`.

Le paper 1 prouve le cas sphérique (`f = 4`, indépendant de `k`) et montre par contre-exemple
que la borne n'est PAS infinie (on peut casser `H_1` à moments d'ordre ≤ 3 + densité appariés).
Donc la borne générale est *quantitative*, pas binaire.

## Attaques, par difficulté croissante

1. **Elliptique — FAIT (exp6), replié dans paper 1.** Mardia affine-invariante + blanchiment
   homéomorphe → théorème verbatim pour toute famille elliptique. Vérifié : coquille
   anisotrope (cond 3,7/6,3) garde `b=d²` (4,04 / 9,00), `H_{d-1}` préservé. Le cas elliptique
   n'est donc plus un problème ouvert — paper 2 démarre au cas 2.
2. **Mélange de composantes elliptiques** (le cas « vrais embeddings » ≈ mélange gaussien) :
   `H_0` gouverné par la séparation des moyennes (2ᵉ moment inter-composantes) ; `H_{≥1}`
   par la forme radiale de chaque composante (4ᵉ). Conjecture : le seuil est le min sur les
   composantes du seuil géométrique local. À formaliser.
3. **Variété courbée** (embeddings réels, dim intrinsèque ≪ ambiante) : l'argument radial
   casse. Piste : argument local-à-global — la persistance `H_k` d'une variété échantillonnée
   se lit sur la *courbure/rayon de reach*, à relier à des moments locaux (moments de
   voisinage), pas globaux. C'est le vrai cœur dur.

## Première brique (exp6) : extension elliptique

Vérifier numériquement : shell/ball elliptiques (image affine `M` d'un shell/ball sphérique)
→ Mardia kurtosis identique au cas sphérique (`b = d²` pour la coquille, quel que soit `M`),
`H_{d-1}` présent, covariance appariée par blanchiment. Si oui, le théorème couvre toutes les
familles elliptiques sans effort. → point de départ du paper 2.

## Pistes de preuve pour le cas 2–3

- Interleaving contrôlé par un budget de moments (borne de type stabilité inverse).
- Couplage moment↔persistance via la fonction de distance-au-mesure (DTM) : la DTM est un
  moment local ; relier `d_B(DTM-Rips)` à un vecteur de moments locaux.
- Généricité : quantifier « à quel point » un contre-exemple (§5 paper 1) est non générique
  (mesure/volume dans l'espace des configurations appariées).
