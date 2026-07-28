# Protocole expérimental — Pouvoir de détection du signal topologique

> **Statut : pré-enregistrement.** La condition de victoire (§7) et le critère de mort
> (§8) sont fixés *avant* toute exécution. Toute modification ultérieure doit être
> datée et justifiée dans le journal de fin de fichier — sinon le résultat n'est pas
> défendable.

## 0. Question et hypothèses

**Question (H1).** Un signal d'effondrement topologique — `Δ_top` calculé sur des
*persistence landscapes* d'une filtration **DTM–Sparse-Rips** en fenêtre glissante —
détecte-t-il les attaques/dégradations d'un flux d'embeddings **plus tôt** et/ou
**plus finement** que des détecteurs géométriques/statistiques bon marché, à un
taux de faux positifs viable en production ?

**Nulle (H0), falsifiable.** Le signal topologique n'apporte aucune amélioration
statistiquement significative (recall@FPR fixé **ou** lead-time) par rapport à la
*meilleure* baseline pauvre.

> On cherche à **réfuter H1**. Le protocole est conçu pour donner à la topologie
> toutes ses chances de perdre : si elle gagne quand même, le résultat est réel.

## 1. Le signal sous test (méthode)

- Filtration **DTM–Sparse-Rips** (Buchet–Chazal–Oudot–Sheehy 2016) sur la fenêtre
  glissante `𝕏_t` (landmarks `|L|=50`, masse DTM `m` à balayer).
- Homologie `H_0` **et** `H_1` (les deux barcodes).
- Vectorisation par **persistence landscapes** L¹ (Bubenik) ; score dual :
  - **rapide** : dérivée `‖λ(t) − λ(t−1)‖_{L¹}` (anti-faux-positif sur drift lent bénin) ;
  - **lent** : accumulateur **CUSUM** sur la dérivée (rattrape le poisoning basse fréquence).
- **Ablations obligatoires** (isolent *d'où vient* le gain, s'il existe) :
  - `A1` VR brut vs DTM-Sparse-Rips ;
  - `A2` landscapes vs bottleneck brut ;
  - `A3` **`H_0` seul vs `H_0+H_1`** — teste si l'homologie supérieure apporte
    quoi que ce soit *au-delà* de la connexité ;
  - `A4` dérivée seule vs dérivée+CUSUM.

## 2. Le gant de baselines (à écraser)

Ce sont les détecteurs qu'un rapporteur exigera. Ne pas les battre = mort de l'axe détection.

| # | Baseline | Ce qu'elle capte | Coût |
|---|----------|------------------|------|
| B1 | **Cosinus agrégé** (moy. au centroïde courant) | dérive de moyenne | trivial |
| B2 | **Mahalanobis** vs (μ, Σ) de référence | OOD elliptique | faible |
| B3 | **kNN / LID** (Ma et al. 2018) | densité locale, dim. intrinsèque | moyen |
| B4 | **MMD** two-sample (noyau RBF) fenêtre vs réf. | shift de distribution *principiel* | moyen |
| B5 | **Résidu PCA** (énergie hors sous-espace top-k réf.) | départ de variété linéaire | faible |
| B6 | **Dérive spectrale de covariance** (Δλ_max, Δ‖Σ‖_F) | **effondrement de variance** | faible |

> **B6 est l'adversaire le plus dangereux.** « Effondrement topologique » et
> « effondrement spectral » sont corrélés : un rapporteur dira que les nombres de
> Betti ne font que redétecter une chute de variance que B6 attrape gratuitement.
> **Le test décisif : trouver un régime où le signal topologique gagne *là où B6
> perd*** — typiquement une variation de `β_1` (fusion/ouverture de boucles) à
> variance quasi constante. Si la topologie ne bat B6 que là où B6 gagne déjà,
> elle est **dominée** et l'axe détection tombe.

## 2bis. Équité du banc (correctif M2)

**Toute** baseline B1–B6 est évaluée avec **le même wrapper streaming** que le signal
topologique : score instantané → dérivée pas-à-pas → accumulateur CUSUM, mêmes seuils
calibrés de la même façon. Interdit de comparer TDA-streaming à baseline-référence-statique :
un gain ainsi obtenu mesurerait le *formalisme streaming*, pas la *topologie*.

## 2ter. Expériment 0 — GATE (correctif M1)

**À exécuter en premier. Bloquant.** Question : le flux d'embeddings réels (3b)
contient-il une homologie supérieure **stable** (`H_1`/`H_2` à persistance
significative vs un modèle nul de points en position générale) ?

- Test : diagrammes de persistance sur fenêtres réelles vs shuffling/rotation
  aléatoire ; les barres `H_{≥1}` survivent-elles au-dessus du bruit (bandes de
  confiance de Fasy et al. / bootstrap) ?
- **Si NÉGATIF** : la contribution « topologique » se restreint honnêtement à un
  **monitoring de connexité (`β_0`) en streaming**, et l'on cesse de survendre les
  « boucles/trous ». Les ablations A3 et le cas de séparation B6 perdent leur objet
  sur données réelles → le rapporter comme fait, pas le masquer.
- **Si POSITIF** : poursuivre le protocole complet.

### Résultat Exp 0 (exécuté — MiniLM 384-d, 20NG 4 sujets)

**Verdict : POSITIF, CONDITIONNEL.** `H1` stable au-delà de la covariance existe —
mais seulement après réduction à la dimension intrinsèque.

- Dimension intrinsèque (TwoNN) ≈ **8** ; PCA = 160 axes pour 90 % var → **variété
  courbée** (bas rang intrinsèque, haut rang linéaire). L'hypothèse de variété tient.
- `H1` sur **384-d brut** : indistinguable du null gaussien apparié (médiane p≈0,95).
  → *artefact de concentration*, PAS un vrai négatif.
- `H1` après **PCA dim=8**, fenêtres multi-sujets W=250 : médiane p=0,03,
  64 % fenêtres p<0,05, test de signe p=0,001, KS p=0,000 → **signal de population réel**,
  au-delà du 2ᵈ ordre (donc au-delà de B6).
- **Fragilité** : à dim=12 le signal intra s'évanouit. La dim de réduction est un
  **hyperparamètre critique** à calibrer et à défendre.

**Conséquence architecturale (nouvelle) :** le pipeline — et donc le moteur Rust —
doit intégrer un **étage de réduction de dimension (PCA incrémentale) AVANT la
persistance**. Rips sur embeddings bruts est un cul-de-sac. Cet étage devient une
brique de premier ordre, pas un prétraitement optionnel.

**Réserves honnêtes :** un seul embedder, un seul corpus, 14 fenêtres/config.
« Existence d'homologie » ≠ « utilité pour la détection » (question du protocole
complet). À répliquer (seeds, corpora, embedders) avant toute généralisation.

## 3. Données

**Deux couches, deux rôles distincts — ne pas les confondre.**

### 3a. Couche validation (correction du moteur, PAS l'impact)
Nuages synthétiques à topologie connue : deux cercles (`β_0=2, β_1=2`), tore
(`β_0=1, β_1=2, β_2=1`), sphère. Effondrements contrôlés :
- fusion de deux cercles → `β_0 : 2→1` ;
- rebouchage d'un cercle → `β_1 : 1→0` à `β_0` constant (le cas « B6-aveugle »).

Sert **uniquement** à prouver que le moteur calcule la bonne topologie et à
fabriquer le cas de séparation B6. **Ne fonde aucune revendication d'impact.**

### 3b. Couche impact (embeddings réels, obligatoire)
Flux d'embeddings réels d'un modèle ouvert (sentence-transformers ou équivalent)
sur corpus **étiquetés par sujet** (arXiv par catégorie, 20NG, Wikipedia par
catégorie) → fournit les **onsets de shift bénin** (vérité terrain négative).

## 4. Taxonomie des événements (vérité terrain, onset `t*` connu)

| Classe | Nature | Rôle | Label |
|--------|--------|------|-------|
| **N — Shift bénin** | changement de sujet *légitime* | **négatif dur** (spécificité) | ne doit PAS déclencher |
| **B — Injection de prompts** | payloads d'injection insérés au flux | positif | onsets connus |
| **C — Effondrement sémantique** | diversité qui s'effondre (mode collapse) | positif | onset connu |

**Onsets nets ET graduels (correctif M4).** Chaque classe positive est déclinée en
variante *abrupte* (`t*` net) et *graduelle* (rampe sur `Δ` pas). Le **lead-time (§5.2)
n'a de pouvoir discriminant que sur les variantes graduelles** — sur un onset net,
tous les détecteurs convergent à `t*+latence`. V2 (§7) ne se teste que sur les rampes.

**Séparabilité N vs attaque, opérationnalisée (correctif M5).** Un shift bénin (N)
et une dérive malveillante doivent différer par une propriété *définie a priori*,
sinon aucun détecteur ne peut les séparer et la tâche est mal posée. Définition
retenue : un shift **N préserve la structure de variété** (les points restent sur
une sous-variété de dimension et topologie comparables — nouveau sujet = nouvelle
région de `ℳ`), tandis qu'une attaque **B/C/D altère la structure** (points
hors-variété, fusion de composantes, effondrement de dimension). Cette distinction
est ce que le protocole met à l'épreuve ; si elle est empiriquement fausse (N et
attaque topologiquement indistinguables sur le réel), c'est un **résultat négatif
publiable**, pas un échec du protocole.
| **D — Outliers adversariaux** | 1..k points hors-variété très loin | positif + test robustesse DTM | onsets connus |
| **E — Attaque topology-preserving** | perturbation *intra-classe d'homologie* | **test d'évasion** (→ question 2) | phase 2 |

La classe **N** est le cœur du protocole : c'est elle qui distingue un vrai
détecteur d'un détecteur de nouveauté trivial.

## 5. Métriques

1. **AUPRC / AUC** au niveau fenêtre (faible seul, mais standard).
2. **Lead-time** = pas entre `t*` réel et la détection. **C'est ici que vit la
   thèse « l'effondrement précède la base vectorielle »** : si la topologie détecte
   *plus tôt*, c'est le titre du papier.
3. **Recall @ FPR fixé** à un taux *production*. **Cible opérationnelle** FPR ≤ 0,1 %/jour,
   **mais** (correctif M3) estimer une queue à 0,1 % exige ~10⁴⁺ fenêtres bénignes
   indépendantes : soit on génère ce volume (flux N longs), soit on **rapporte à
   FPR=1 %** avec IC honnête et on *extrapole* la cible 0,1 % en le déclarant.
   Ne jamais affirmer un recall@0,1 % non soutenu par assez d'échantillons de queue.
4. **FPR spécifique sur classe N** (faux positifs sur shift *légitime*).
5. **Coût** : p50/p99, débit, vs chaque baseline. Comparaison **ajustée au coût** :
   la topologie doit justifier son surcoût par un gain de détection.

## 6. Rigueur statistique

- ≥ 10 flux/seeds indépendants ; IC bootstrap sur AUC/AUPRC et lead-time.
- Test de **DeLong** (diff d'AUC) ou bootstrap apparié sur les diffs.
- Correction multi-comparaisons (Holm) sur l'ensemble baselines × classes.

## 7. Condition de victoire (FIXÉE AVANT EXÉCUTION)

Le signal topologique « gagne » **si et seulement si**, sur au moins une classe
d'attaque parmi {B, C}, il satisfait **l'une** des deux, avec `p < 0,05` corrigé :

- **(V1)** recall @ FPR=0,1 % supérieur de **≥ 5 points** à la meilleure baseline ; **ou**
- **(V2)** lead-time médian réduit de **≥ 10 pas** vs la meilleure baseline,

**ET** dans les deux cas : FPR sur classe **N** ne dépassant pas celui de la
meilleure baseline (pas de gain acheté en faux positifs).

### Résultat Exp 1 (exécuté — même pipeline, espace réduit dim=8)

**La détection générale par TDA est réfutée ; une complémentarité provable émerge.**

- Attaques réelles (injection off-manifold, effondrement, bouche-trou) : **B6
  (dérive spectrale) domine** (AUC 1,0 / 1,0 / 0,96), `Δ_top` ≤ hasard. Les vraies
  attaques perturbent le 2ᵈ ordre → captées moins cher que par la topologie.
- Test propre cercle-vs-disque à **covariance égale par construction** : `Δ_top`
  AUC=1,0, **B6 aveugle (AUC=0,58)**. La topologie capte un **angle mort provable**
  de la covariance : changement de topologie *sans* changement de 2ᵈ ordre.

**Reformulation de la contribution (honnête et plus forte) :** la TDA n'est pas un
meilleur détecteur général. C'est un détecteur **complémentaire** qui ferme le canal
d'évasion laissé ouvert quand un attaquant contraint sa perturbation à préserver les
statistiques d'ordre 1–2 pour échapper à Mahalanobis/B6. Défense en profondeur :
`B6 ∪ Δ_top`. Ceci **motive directement la question 2** (borne d'évasion adaptative :
quel est le coût minimal d'une attaque efficace *et* préservant le 2ᵈ ordre ?).

**Exp 2 — confirmation sur réel (attaquant adaptatif).** Attaque = effondrement +
correction affine préservant *exactement* `(μ, Σ)` de la pré-image (`|Δcov|=0`,
homéomorphisme → topologie bouchée conservée). Résultat : **B6 aveugle (AUC 0,41),
Mahalanobis aveugle (0,00), `Δ_top` rattrape (AUC 0,77)**. La défense en profondeur
`B6 ∪ Δ_top` passe de « prouvée en synthétique » à **démontrée sur embeddings réels**.
Reste : la *courbe* d'évasion (dégât sémantique max sous seuil `Δ_top`) et la réplique 1536-d.

**Exp 2b/2c/2d — mise à l'épreuve : la complémentarité NE survit pas.** Trois tests
adverses ont démonté le résultat d'Exp 2 :
- **Dégât** : l'attaque adaptative est quasi inerte (la correction de covariance ré-étale
  les points effondrés) → peu de nuisance réelle à détecter.
- **Baselines forts** : la **kurtosis multivariée de Mardia attrape l'adaptatif à AUC 1,0**,
  dominant `Δ_top` (0,83). L'attaque préserve les ordres 1–2 mais pas l'ordre 4.
- **Niche « pure »** : sur cercle/disque (différence purement topologique), la
  **densité kNN sépare à 1,0** comme `Δ_top`. Même l'angle mort idéal du 2ᵈ ordre est
  couvert par un détecteur de densité classique.
- Rectification honnête : `Δ_top` n'est *pas* un artefact affine (contrôle propre
  `M=Σ^½QΣ^{-½}` → `Δ_top`≈0,55, hasard) ; il est **redondant**.

**Statut de l'axe détection : critère de mort §8 effectivement atteint.** Aucun test
construit ne montre la persistance capter ce que covariance/kurtosis/kNN-densité ne
captent déjà. Cause structurelle : un changement de topologie dans l'espace d'embedding
s'accompagne quasi toujours d'un changement de densité/moments qu'un détecteur moins
cher voit. La décision d'invoquer §8 (bascule vers l'axe systèmes / résultat négatif
publiable) appartient à l'utilisateur.

## 8. Critère de mort (pré-engagé)

Si, sur **aucune** classe {B, C, D}, le signal topologique ne bat **B2 (Mahalanobis)
et B6 (spectral)** au régime FPR=0,1 % ni en lead-time → **l'axe détection est
falsifié**. On ne rationalise pas *a posteriori* : on bascule le centre de gravité
sur la contribution **systèmes** (moteur streaming + éval adversaire adaptatif) et
on rapporte honnêtement l'échec de détection comme résultat négatif.

## 9. Ordre d'exécution

0. **Expériment 0 (GATE, §2ter)** — homologie supérieure stable sur réel ? Bloquant.
1. Moteur PH de référence + **couche 3a** (valider la correction sur topologie connue).
2. Baselines B1–B6 **avec wrapper streaming commun (§2bis)** — les avoir tôt cadre la barre.
3. Générateurs N/B/C/D (variantes nette + graduelle) + pipeline embeddings réels (3b).
4. Balayage `m`, `|L|`, fenêtre `W`, seuils ; ablations A1–A4.
5. Métriques §5, tests §6, verdict §7/§8.

> **Points de gate :** Exp 0 négatif → replier la thèse sur `β_0`/systèmes avant
> d'investir dans le moteur `H_{≥1}`. Étape 2 : si une baseline pauvre plafonne déjà
> le recall, la barre à battre est explicite dès le départ.

---
### Journal des modifications post-enregistrement
_(vide — toute entrée doit être datée et justifiée)_
