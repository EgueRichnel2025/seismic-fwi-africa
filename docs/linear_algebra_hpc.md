# Algèbre linéaire et calcul haute performance

Ce document répond à trois questions, dans l'ordre : quelles opérations
d'algèbre linéaire l'inversion contient-elle réellement, à quelles
routines BLAS et LAPACK correspondent-elles, et quelle bibliothèque les
exécute effectivement sur cette machine.

Le principe directeur du projet est qu'aucune opération n'est ajoutée
pour pouvoir afficher le nom d'une bibliothèque. Les routines listées ici
sont celles que le code appelle, aux tailles où il les appelle.

---

## 1. Pourquoi l'algèbre linéaire apparaît ici

La chaîne est : équation d'onde → discrétisation → fonction objectif →
gradient → mise à jour du modèle. L'algèbre linéaire n'apparaît qu'à
partir du gradient, et sa nature dépend de la formulation.

**FWI par état adjoint.** La Jacobienne n'est jamais formée — c'est tout
l'intérêt de la méthode. Il ne reste donc que des opérations sur des
vecteurs de taille `N_m = 45 451`, concentrées dans la récursion à deux
boucles de L-BFGS. Ce sont des opérations de **BLAS niveau 1**, et le
reconnaître est plus honnête que de fabriquer artificiellement des
matrices denses.

**Inversion paramétrique réduite.** En ramenant le modèle à `N_k ≈ 24`
coefficients, la Jacobienne `J` de taille `(N_d, N_k)` redevient
constructible. Apparaissent alors de vraies opérations de **BLAS niveau
2 et 3** et de **LAPACK** : produit matrice-vecteur, matrice normale,
factorisation.

C'est la raison d'être de cette seconde formulation : elle répond à un
besoin scientifique propre, et elle fait apparaître LAPACK légitimement.

---

## 2. Cartographie des opérations

| Opération | Routine | Où dans le code | Taille réelle | Rôle scientifique |
|---|---|---|---|---|
| produit scalaire | **DDOT** | `optimize._two_loop_recursion` | n = 45 451 | coefficients `ρ sᵀq`, `ρ yᵀq` de L-BFGS ; courbure `yᵀs` ; dérivée directionnelle `gᵀd` d'Armijo |
| mise à jour vectorielle | **DAXPY** | `optimize._two_loop_recursion`, mise à jour du modèle | n = 45 451 | `q ← q - α y` et `q ← q + (α-β) s` dans la récursion ; `m ← m + λ d` |
| norme euclidienne | **DNRM2** | `optimize.lbfgs_inversion` | n = 45 451 | norme du gradient, critères d'arrêt, test de courbure |
| matrice-vecteur | **DGEMV** | `parametric.gauss_newton_inversion` | (N_d, 24) | second membre `-Jᵀr` des équations normales |
| matrice-matrice | **DGEMM** | `parametric.gauss_newton_inversion` | (N_d, 24) | matrice normale `JᵀJ` |
| système symétrique | **DPOSV** | `parametric.gauss_newton_inversion` | n = 24 | résolution de `(JᵀJ + λI) δc = -Jᵀr` par Cholesky |
| moindres carrés | **DGELSD** | `parametric.gauss_newton_inversion` | (N_d, 24) | résolution par SVD si le conditionnement dépasse 10¹⁰ |

Les routines sont obtenues par `scipy.linalg.get_blas_funcs` et
`scipy.linalg.get_lapack_funcs`, qui renvoient les implémentations
réellement liées à l'installation. Le code est dans
`src/seismic/linear_algebra/kernels.py`.

### 2.1 Pourquoi DNRM2 plutôt qu'un calcul naïf

DNRM2 applique une mise à l'échelle interne qui évite les dépassements de
capacité. Le test `test_nrm2_avoids_overflow` le vérifie : sur un vecteur
de composantes valant 10²⁰⁰, la somme des carrés déborde vers l'infini
alors que DNRM2 retourne la bonne valeur. Ce n'est pas une préoccupation
théorique dans un code où les amplitudes de source ne sont pas calibrées.

### 2.2 Pourquoi DPOSV plutôt que DGESV

La matrice normale régularisée `JᵀJ + λI` est symétrique définie
positive par construction. La factorisation de Cholesky exploite cette
structure et coûte environ moitié moins qu'une factorisation LU. Le code
bascule sur DGELSD lorsque le conditionnement l'exige, ce qui est la
situation où Cholesky devient numériquement imprudente.

### 2.3 Ce qui a été écarté

Conformément au cadrage du projet, aucune QR ni SVD n'a été ajoutée pour
le principe. La seule SVD du projet est celle que DGELSD effectue en
interne, et elle n'est utilisée que lorsque le conditionnement la
justifie. Le solveur d'onde explicite n'a pas été transformé en un
système matriciel géant pour faire apparaître une factorisation : ce
serait absurde numériquement, un schéma explicite n'en ayant pas besoin.

---

## 3. Backend réellement utilisé

### 3.1 La question posée correctement

Un `import numpy` ne prouve rien. La seule affirmation défendable porte
sur la bibliothèque effectivement chargée dans le processus.

Le module `seismic.hpc.backend` croise trois sources :

1. `numpy.show_config` et `scipy.show_config` — dépendance déclarée à la
   compilation ;
2. `threadpoolctl.threadpool_info()` — bibliothèques réellement chargées
   en mémoire, source prioritaire ;
3. le chemin des bibliothèques partagées, lorsque la plateforme le
   permet.

Aucune n'est traitée comme suffisante seule. Si la déclaration de
compilation et le chargement à l'exécution divergent, c'est l'exécution
qui fait foi, et l'écart est signalé.

### 3.2 Résultat mesuré

Sur l'environnement de développement utilisé pour produire les résultats
du dépôt :

```
Fournisseur detecte : openblas
Intel oneMKL utilise : NON
numpy : scipy-openblas 0.3.34.106.0
scipy : scipy-openblas 0.3.31.dev
Charge a l'execution : blas / openblas v0.3.34.106.0
```

**Intel oneMKL n'est pas utilisé dans cet environnement.** C'est une
conclusion, pas un échec : elle est rapportée telle quelle, et le projet
ne prétend nulle part le contraire.

La raison est connue : un NumPy installé depuis PyPI avec `pip` est lié à
OpenBLAS. oneMKL n'est accessible qu'en passant par un environnement
conda-forge ou la distribution Intel :

```bash
conda install -c conda-forge numpy scipy "libblas=*=*mkl"
```

Reproduire le diagnostic :

```bash
python scripts/diagnose_blas_backend.py
```

Le script écrit `experiments/output/blas_backend_report.json`.

### 3.3 Comparer deux backends

Le protocole est le même dans les deux environnements : installer l'un
ou l'autre fournisseur, lancer `scripts/benchmark_linear_algebra.py`, et
comparer les deux fichiers JSON produits. Le script ne fait aucune
comparaison lui-même et n'annonce aucun gain : il mesure ce qui est, sur
le backend qui est.

---

## 4. Mesures de performance

### 4.1 Protocole

Les tailles mesurées sont celles que l'inversion manipule réellement.
Chaque mesure est précédée de deux appels d'échauffement, puis répétée
sept fois ; c'est la **médiane** qui est rapportée, pour limiter l'effet
des valeurs aberrantes. Le matériel, les versions et le nombre de threads
sont enregistrés avec les résultats.

### 4.2 Résultats sur OpenBLAS

Vecteurs de 45 451 composantes ; Jacobienne réduite (54 652, 24).

| Routine | Temps médian | Description |
|---|---:|---|
| DDOT | 9,89 · 10⁻⁶ s | produit scalaire, n = 45 451 |
| DAXPY | 1,13 · 10⁻⁵ s | mise à jour vectorielle, n = 45 451 |
| DNRM2 | 1,62 · 10⁻⁵ s | norme, n = 45 451 |
| DGEMV | 5,21 · 10⁻³ s | `Jᵀr`, (54 652, 24) |
| DGEMM | 1,13 · 10⁻² s | `JᵀJ`, (54 652, 24) |
| DPOSV | 1,11 · 10⁻⁵ s | Cholesky, n = 24 |
| DGELSD | 2,53 · 10⁻² s | moindres carrés par SVD |

### 4.3 Lecture honnête de ces chiffres

Ces mesures appellent une conclusion que le projet doit assumer :
**l'algèbre linéaire n'est pas le goulot d'étranglement de cette
inversion.** Une simulation directe en pleine résolution prend environ
0,93 s, soit cinq ordres de grandeur de plus qu'un DDOT sur le gradient
complet. Le coût dominant est la propagation d'ondes — une boucle
temporelle de 1 051 pas sur 45 451 points — pas les opérations BLAS.

Autrement dit, optimiser le backend BLAS ne changerait pratiquement pas
le temps total d'une FWI adjointe. Ce qui le changerait : vectoriser ou
paralléliser la boucle temporelle du solveur, ou distribuer les tirs, qui
sont indépendants les uns des autres et donc parallélisables sans
communication.

Dire l'inverse serait transformer le projet en concours de benchmarks
sans lien avec sa physique.

---

## 5. Validation croisée

Chaque routine est comparée à une référence NumPy indépendante dans
`tests/test_linear_algebra.py`, avec une tolérance déclarée de 10⁻¹².
Écarts relatifs observés aux tailles du projet :

| Routine | Écart relatif contre NumPy |
|---|---:|
| DPOSV | 3,07 · 10⁻¹⁶ |
| DGELSD | 2,30 · 10⁻¹⁵ |

Les tests couvrent aussi les cas limites : matrice non définie positive
rejetée par DPOSV avec un message explicite, matrice de rang déficient
correctement détectée par DGELSD (rang 3 sur 4 colonnes dont une
dupliquée), dimensions incompatibles refusées.

---

## 6. Coût et pistes de parallélisation

### 6.1 Coût mesuré

| Quantité | Valeur |
|---|---:|
| Simulation directe, grille 151 × 301, 1 051 pas | ≈ 0,93 s |
| Gradient adjoint (direct + rétro-propagation) | ≈ 2 × une simulation |
| Mémoire du champ direct stocké, float64 | ≈ 382 Mo |
| Mémoire du champ direct stocké, float32 | ≈ 191 Mo |

Le stockage du champ complet est le vrai coût mémoire de l'état adjoint :
la rétro-propagation a besoin du champ direct à chaque pas. Le paramètre
`wavefield_dtype` permet d'arbitrer entre précision et mémoire.

### 6.2 Pistes identifiées, non implémentées

Elles sont nommées comme perspectives, pas comme réalisations :

- **parallélisation sur les tirs** : les tirs sont indépendants, leurs
  contributions s'additionnent. C'est la parallélisation la plus
  rentable et la plus simple de toute la chaîne ;
- **vectorisation du noyau de propagation** : la boucle temporelle est le
  coût dominant ;
- **checkpointing** : recalculer une partie du champ direct plutôt que de
  le stocker, pour échanger du temps contre de la mémoire. Devient
  nécessaire en 3D ;
- **stratégie multi-échelle fréquentielle** : inverser d'abord les basses
  fréquences pour réduire le risque de cycle skipping.
