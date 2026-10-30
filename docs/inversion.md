# Problème inverse : formulation, gradient adjoint et optimisation

Ce document décrit la formulation du problème inverse du projet, la
dérivation du gradient par état adjoint, sa validation numérique, et les
deux formulations d'inversion implémentées.

**Avertissement liminaire.** Tout ce qui suit s'applique à un modèle
synthétique. L'anomalie de vitesse est un objet de test défini par le
projet. Aucun résultat présenté ici ne constitue l'observation d'un
aquifère, d'une ressource géothermique ou d'une structure géologique
réelle, au Bénin ou ailleurs.

---

## 1. Formulation

### 1.1 Fonction objectif

L'inversion minimise l'écart quadratique entre données simulées et
observées :

$$\Phi(m) = \tfrac{1}{2}\,\lVert F(m) - d_{obs}\rVert_2^2$$

où `F` est l'opérateur de modélisation directe, `d_obs` les données
synthétiques observées et `m` le paramètre inversé.

Le résidu est `r = F(m) - d_obs`. Avec cette convention de signe, le
gradient pointe dans la direction de croissance de `Φ`, et l'optimiseur
descend selon `-g`.

### 1.2 Choix du paramètre : m = v²

Le paramètre inversé est le **carré de la vitesse**, et non la vitesse
elle-même. La raison est structurelle : dans l'équation d'onde
discrétisée, `v²` intervient linéairement,

$$u_{tt} + 2\sigma u_t = v^2 \nabla^2 u + s$$

alors que `v` y intervient quadratiquement. Inverser `m = v²` rend la
dérivation de l'adjoint exacte et sans approximation. Le gradient par
rapport à la vitesse s'en déduit par dérivation composée :

$$\frac{\partial\Phi}{\partial v} = 2v \cdot \frac{\partial\Phi}{\partial m}$$

Cette conversion est implémentée dans
`seismic.inversion.fwi.gradient_with_respect_to_velocity`.

### 1.3 Taille du problème

Sur la grille de référence (151 × 301, pas de 2 m) :

| Quantité | Valeur |
|---|---:|
| Paramètres `N_m` | 45 451 |
| Récepteurs par tir | 26 |
| Échantillons temporels | 1 051 |
| Données par tir `N_d` | 27 326 |
| Jacobienne complète `N_d × N_m` | ≈ 1,24 · 10⁹ coefficients |
| Mémoire d'une Jacobienne dense | ≈ 9,9 Go |
| Coût de construction par différences finies | 45 451 simulations |

Ce tableau justifie à lui seul la méthode adjointe : la Jacobienne
complète n'est ni stockable ni calculable. C'est la motivation
historique de l'état adjoint en FWI.

---

## 2. Gradient par état adjoint

### 2.1 Principe

L'état adjoint calcule le gradient par rapport à **tous** les paramètres
au prix d'**une** simulation directe et d'**une** rétro-propagation,
indépendamment de `N_m`. Le gain par rapport aux différences finies est
donc d'un facteur `N_m`, soit plus de quarante-cinq mille ici.

### 2.2 Dérivation discrète

Un choix important a été fait : l'adjoint est dérivé du **schéma
discret**, et non obtenu en discrétisant l'adjoint continu. Les deux
approches diffèrent par des termes d'ordre du pas de discrétisation ;
seule la première donne le gradient exact de la fonction objectif
réellement calculée, ce qui permet au test de validation de passer à la
précision machine.

Le schéma direct s'écrit, sur les points intérieurs et en notation
matricielle :

$$D u^{n+1} = 2u^n - E u^{n-1} + \Delta t^2 \,\text{diag}(m)\, L u^n + \Delta t^2 s^n$$

avec `D = diag(1 + σΔt)`, `E = diag(1 - σΔt)` et `L` le laplacien à cinq
points sous conditions de Dirichlet homogènes.

En formant le lagrangien `Φ + Σₙ λⁿ⁺¹ᵀ (contrainte)` et en annulant la
dérivée par rapport à `uⁿ`, on obtient la récurrence adjointe :

$$D\lambda^n = 2\lambda^{n+1} - E\lambda^{n+2} + \Delta t^2 L\big(\text{diag}(m)\lambda^{n+1}\big) - P^T r^n$$

où `P` est l'opérateur d'échantillonnage aux récepteurs.

**Deux points méritent attention.** D'abord l'ordre des opérateurs
change : le direct applique `diag(m)·L(·)`, l'adjoint applique
`L(diag(m)·(·))`. C'est la transposée de `diag(m) @ L`, licite parce que
`L` est symétrique sous conditions de Dirichlet. Ensuite la récurrence se
parcourt en temps inverse, avec les conditions terminales
`λ^{N+1} = λ^{N+2} = 0`.

Le gradient s'accumule alors exactement comme :

$$\frac{\partial\Phi}{\partial m} = -\Delta t^2 \sum_n \lambda^{n+1} \odot \big(L u^n\big)$$

où `⊙` est le produit terme à terme.

### 2.3 Validation

Le test de dérivée directionnelle compare `⟨g, δ⟩` à la différence finie
centrée de `Φ` dans la direction `δ`. Résultats obtenus sur une grille de
10 m :

| ε | Différence finie centrée | Erreur relative |
|---:|---:|---:|
| 10⁻⁴ | 5,6881881457 · 10⁻² | 3,63 · 10⁻⁴ |
| 10⁻⁵ | 5,6861439997 · 10⁻² | 3,63 · 10⁻⁶ |
| 10⁻⁶ | 5,6861235585 · 10⁻² | 3,64 · 10⁻⁸ |
| 10⁻⁷ | 5,6861233656 · 10⁻² | 2,45 · 10⁻⁹ |
| 10⁻⁸ | 5,6861233777 · 10⁻² | 4,58 · 10⁻⁹ |

Valeur analytique : `⟨g, δ⟩ = 5,6861233516 · 10⁻²`.

L'erreur décroît en `ε²`, ce qui est la convergence attendue d'une
différence centrée, jusqu'au plancher d'arrondi vers 10⁻⁹. **Le gradient
est exact.** Ce test est rejoué automatiquement par
`tests/test_adjoint_gradient.py`.

Le propagateur d'inversion est par ailleurs vérifié contre le solveur de
référence `seismic.wave.acoustic_2d` : l'écart relatif est de l'ordre de
10⁻¹⁵, donc l'inversion inverse bien la même physique que le modèle
direct déjà validé du projet.

---

## 3. Deux conditions nécessaires au succès

Les premières inversions ont fait décroître `Φ` sans reconstruire
l'anomalie. Le diagnostic a identifié deux causes, toutes deux corrigées.

### 3.1 Préconditionnement d'illumination

Le gradient brut est dominé par les zones de forte amplitude du champ,
c'est-à-dire le voisinage immédiat de la source. Sans correction,
l'optimiseur dépense ses corrections en surface au lieu d'atteindre les
cibles profondes.

La diagonale du pseudo-hessien,

$$H_{\text{pseudo}}(x) = \sum_n \big(L u^n(x)\big)^2$$

mesure précisément l'illumination de chaque point. Son inverse régularisé
sert de matrice initiale `H₀` dans la récursion L-BFGS :

$$P(x) = \frac{1}{H_{\text{pseudo}}(x) + \varepsilon \max H_{\text{pseudo}}}$$

**Le préconditionneur entre par `H₀` et non par une modification du
gradient.** Ce point est important : préconditionner le gradient
directement rendrait incohérentes les paires de courbure `(s, y)` de
L-BFGS et invaliderait la dérivée directionnelle utilisée par la
condition d'Armijo. En passant par `H₀`, l'algorithme reste correct.

Référence : Shin, Jang et Min, *Improved amplitude preservation for
prestack depth migration by inverse scattering theory*, Geophysical
Prospecting, 2001.

### 3.2 Acquisition multi-tirs

Un tir unique n'éclaire qu'une ouverture angulaire limitée. Avec la
source d'origine du projet à `x = 60 m` et une cible à `x = 360 m`,
l'anomalie est vue sous un angle trop étroit pour être contrainte.

Cinq tirs répartis sur le profil (60, 180, 300, 420, 540 m) élargissent
la couverture. Les contributions s'additionnent dans `Φ` comme dans le
gradient, chaque tir donnant lieu à sa propre simulation directe et à sa
propre rétro-propagation.

### 3.3 Effet mesuré

Sur la grille de 10 m, 25 itérations, avec le modèle initial dépourvu
d'anomalie :

| Configuration | `Φ` final / initial | `E_cible` final | v moyenne anomalie |
|---|---:|---:|---:|
| 1 tir, sans préconditionneur | 1/2,5 | — | 1985 m/s |
| 5 tirs, avec préconditionneur | 1/15,8 | 0,086 | 1601 m/s |

Cible : 1500 m/s. Départ : 2000 m/s.

---

## 4. Optimisation

L'optimiseur est un L-BFGS à mémoire limitée dont la récursion à deux
boucles est implémentée explicitement à partir des noyaux BLAS du projet.
Voir `docs/linear_algebra_hpc.md` pour la cartographie des routines.

Éléments de l'implémentation :

- **recherche linéaire** par rebroussement avec condition d'Armijo, la
  dérivée directionnelle étant calculée avec le vrai gradient ;
- **condition de courbure** `yᵀs > 0` avant d'accepter une paire dans la
  mémoire, ce qui garantit que l'approximation de l'inverse de la
  hessienne reste définie positive ;
- **bornes physiques** sur la vitesse, imposées par projection ;
- **lissage gaussien** du gradient, avec rembourrage par réplication des
  bords. Sans ce rembourrage, le lissage atténuerait artificiellement le
  gradient sur les bords du modèle, y compris près de la surface ;
- **repli sur la plus forte pente** si la direction L-BFGS cesse d'être
  descendante, avec réinitialisation de la mémoire.

### 4.1 Métriques

Trois erreurs relatives sont suivies :

$$E_d = \frac{\lVert d_{pred} - d_{obs}\rVert_2}{\lVert d_{obs}\rVert_2}, \qquad E_m = \frac{\lVert v_{rec} - v_{vrai}\rVert_2}{\lVert v_{vrai}\rVert_2}, \qquad E_{cible} = \frac{\lVert (v_{rec} - v_{vrai})|_{\Omega}\rVert_2}{\lVert v_{vrai}|_{\Omega}\rVert_2}$$

`E_cible` est restreinte à la zone de l'anomalie `Ω`. Cette métrique est
nécessaire : `E_m` porte sur les 45 451 cellules dont seules 1 220
appartiennent à l'anomalie, de sorte qu'une reconstruction parfaite de la
cible ne ferait bouger `E_m` que de quelques millièmes. **Rapporter
uniquement `E_m` masquerait le résultat au lieu de le montrer.**

Noter que `Φ = ½‖r‖²` donne `‖r‖ = √(2Φ)` : `E_d` se déduit de `Φ` sans
simulation supplémentaire.

---

## 5. Seconde formulation : inversion paramétrique réduite

La FWI adjointe ne forme aucune matrice dense. C'est sa force, mais cela
signifie qu'elle n'utilise que du BLAS de niveau 1 (produits scalaires,
mises à jour vectorielles, normes) et aucune routine LAPACK.

Une seconde formulation, complémentaire, réduit le modèle à une
combinaison de quelques dizaines de fonctions gaussiennes couvrant une
zone d'intérêt :

$$m(x) = m_{ref}(x) + \sum_{k=1}^{N_k} c_k \varphi_k(x)$$

Avec `N_k ≈ 24`, la Jacobienne `∂r/∂c` redevient constructible — `N_k`
simulations par tir — et de taille raisonnable. Les équations normales
régularisées

$$(J^T J + \lambda I)\,\delta c = -J^T r$$

sont alors un vrai système symétrique défini positif, résolu par
factorisation de Cholesky (LAPACK DPOSV), avec repli sur une résolution
par SVD (LAPACK DGELSD) lorsque le conditionnement dépasse un seuil.

**Les deux formulations doivent être nommées distinctement.** La FWI
adjointe reconstruit un champ de vitesse sur toute la grille ; celle-ci
estime les coefficients d'une base réduite. Ce n'est pas la même chose,
et appeler la seconde « FWI » serait inexact.

---

## 6. Limites

Elles doivent accompagner tout résultat présenté.

**Crime inverse.** Les données observées sont générées par le même
solveur, la même grille, le même pas de temps et la même source que
l'inversion. C'est délibéré : cela isole le comportement de l'inversion
de toute erreur de modélisation. Mais cela rend le résultat plus
favorable qu'une situation réelle, où le solveur ne coïncide jamais avec
la physique du terrain. Toute présentation du résultat doit le dire.

**Physique simplifiée.** Le milieu est acoustique à densité constante. Il
ne modélise ni les ondes S, ni les conversions de mode, ni l'atténuation
intrinsèque, ni l'anisotropie.

**Conditions aux limites.** Les bords latéraux et le fond sont traités
par une zone absorbante de type éponge, pas par un PML. La surface
supérieure impose une pression nulle, ce qui n'est pas un modèle complet
de surface libre. Des réflexions parasites subsistent et contribuent au
résidu dans la fenêtre tardive.

**Signaux.** Les récepteurs enregistrent une valeur scalaire du champ. Ce
n'est pas un modèle de la réponse électromécanique d'un géophone réel, et
l'amplitude de la source n'est pas calibrée physiquement.

**Résolution.** La reconstruction ne récupère qu'une fraction du
contraste de l'anomalie, et ses bords sont lissés. C'est le comportement
attendu d'une FWI à ouverture limitée, bande passante finie et nombre
d'itérations borné. Aucune récupération ne doit être annoncée sans la
mesure chiffrée correspondante.

**Données synthétiques.** Il n'y a aucune donnée de terrain dans ce
projet.
