# Frontière générale — l'argument local-à-global (option 1)

But : étendre la loi `N*(complexité) ∝ complexité topologique` au-delà du cas radial (exp9).
On distingue ce qui est **prouvé**, **argué** (cas séparable), et **conjecturé** (cas général).

## 1. Ce qui est PROUVÉ (radial, exp9)

Pour un nuage sphériquement symétrique, la topologie (m coquilles concentriques) est une
fonction de la loi radiale de `R²`, qui est une distribution à `m` atomes. Résoudre une
distribution à `m` atomes exige ses moments jusqu'à l'ordre `2m` (unicité de Hamburger / exactitude
de Gauss : une quadrature à `m` points apparie les moments jusqu'à `2m−1` et diffère au `2m`-ième).
Comme `R² = ‖x‖²`, l'ordre radial `2m` = ordre nuage `4m`. D'où **N\*(m) = 4m**, prouvé et
vérifié (m=1,2,3 : ordres appariés 2,6,10 ; `H_1` = 1,2,3 vs 0).

## 2. Le principe général (pont du nerf)

Deux faits en tension :

- **Les moments sont des moyennes polynomiales globales.** Connaître les moments jusqu'à l'ordre
  `N`, c'est connaître la projection de la mesure sur les polynômes de degré ≤ `N` (espace de
  dimension finie). Deux mesures égales sur ces moments sont indistinguables par toute statistique
  polynomiale de degré ≤ `N` — y compris invariantes à tout réarrangement de masse préservant ces
  moments.
- **La topologie est locale (théorème du nerf).** Le type d'homotopie du complexe de Čech/Rips est
  déterminé par le motif d'*intersections locales* des `ε`-boules — un datum combinatoire local,
  insensible aux moyennes globales.

Le pont : reconstruire le motif d'intersections (donc les nombres de Betti) à partir des moments
exige de *résoudre* la mesure à l'échelle des features individuelles. Chaque générateur d'homologie
est un « atome » à résoudre ; résoudre `b` atomes coûte des moments d'ordre `∝ b`. C'est
exactement le mécanisme radial, transposé.

## 3. Ce qui est ARGUÉ (topologie séparable)

Si les `b` features sont *séparables le long d'une coordonnée* (centres distincts sur un axe `e`),
la loi marginale sur `e` est une distribution à `b` amas. La quadrature de Gauss **sur cet axe**
apparie les moments marginaux jusqu'à l'ordre `2b−1` ; par le même argument qu'en §1, épingler les
`b` amas exige l'ordre `2b` sur `e`, donc un moment nuage d'ordre `∝ b`. La loi `N* ∝ b` s'étend
donc à toute topologie séparable par projection — sans hypothèse de symétrie sphérique.

**Confirmé (exp10).** `b` boucles (cercles y-z) aux nœuds de Gauss sur l'axe x, vs tube plein
contractile (`H_1=0`) : moments sur x appariés jusqu'à `2b−1`, `H_1 = b` vs `0` (vérifié b≤4).
Le cas séparable est donc prouvé numériquement, hors symétrie sphérique.

## 4. Ce qui reste CONJECTURÉ (topologie enlacée / variété courbée)

Pour une topologie **non séparable** (tore, cycles enlacés, variété courbée de dim intrinsèque ≪
ambiante — le régime des vrais embeddings), il n'existe pas d'axe ni de coordonnée radiale unique
portant toute la topologie. La généralisation de la quadrature (résoudre `b` features non alignés)
demande un argument de **Mayer–Vietoris / nerf** : décomposer le complexe en morceaux locaux,
borner le nombre de moments nécessaires pour certifier chaque recouvrement, et sommer.

**Réduction par symétrie (rétrécit l'ouvert — exp11).** Un nuage invariant sous un groupe de
symétrie `G` est déterminé par sa mesure sur le quotient (le profil). Sa topologie `G`-invariante
se lit sur le profil, de dimension inférieure, où les cas §1/§3 s'appliquent. Exemple : un tore de
révolution se réduit à son profil `(ρ,z)` ; sa boucle-tube = le trou du profil = cercle-vs-disque
= 4ᵉ moment du profil (vérifié sur grille régulière : tore creux `H_1=2, H_2=1`, tore plein
`H_1=1, H_2=0` ; `Var(rayon-profil²)` = 0,000 creux vs 0,020 plein — la boucle-tube est bien le
trou du profil). Donc **toute
topologie à symétrie de groupe obéit à la loi via réduction au quotient.** Le cas vraiment ouvert
se restreint à l'entanglement **génériquement asymétrique** — dont l'existence même comme
contre-exemple *visible par la persistance et échappant à la loi* est incertaine (les cycles
enlacés, p.ex., ont mêmes nombres de Betti et ne sont pas distingués par la persistance ordinaire).

**Conjecture (frontière générale).** Pour un nuage fini, l'ordre de moment nécessaire pour
distinguer sa classe d'homologie d'une classe triviale croît au moins linéairement avec la somme
des nombres de Betti `Σ_k β_k`. Le cas radial (§1) et le cas séparable (§3) en sont les instances
prouvées ; le cas enlacé est ouvert (preuve formelle).

**Recherche adverse de contre-exemple — AUCUN ne survit (exp11, exp12).** On a cherché
activement un contre-exemple (topologie visible-par-persistance échappant à la loi) :
- topologie *symétrique* (tore) → réduit au profil (§4 ci-dessus), loi tient ;
- *cycles enlacés* → mêmes nombres de Betti, invisibles à la persistance ordinaire → pas un
  contre-exemple ;
- perturbation *mesure-nulle* (1 point qui bouche) → poids `1/N → 0`, pas deux mesures distinctes ;
- *complexité géométrique* d'un feature isolé (étoile à `k` branches, `β_1=1`, rayon très
  variable) → à covariance appariée, la kurtosis détecte le trou à AUC 1,0 pour `k` jusqu'à 8
  (exp12) : `N*=4` tient même pour une boucle tordue. La complexité géométrique d'un *seul*
  feature ne gonfle PAS `N*`.

Verdict : `N*` est gouverné par le *nombre* de features (~4 moments chacun), pas par leur
arrangement ni leur forme. Aucun contre-exemple trouvé ⇒ forte évidence que la loi vaut
partout où la persistance a du contenu. (Verdict de recherche exhaustive, pas preuve formelle.)

**Corollaire pratique (si vraie).** La persistance lit `Σβ_k` en une passe (`O(1)` en ordre de
moment) ce qui exigerait sinon un moment d'ordre non borné avec la complexité. Elle n'a donc de
valeur irréductible que là où la topologie est *riche* — pas pour les changements canoniques mono-
échelle (`m=1`, redondants avec la kurtosis). C'est la délimitation honnête de l'outil.

## 5. Prochaine brique concrète

Tester le cas séparable (§3) numériquement — `b` boucles alignées sur un axe, quadrature de Gauss
sur cet axe — pour confirmer `N* ∝ b` hors symétrie sphérique. Puis attaquer un cas non séparable
minimal (deux tores emboîtés, ou tore vs sphère moment-appariés) pour sonder la conjecture §4.
