# seismic-fwi-africa

Prospection géophysique et imagerie sismique par modélisation de la
propagation des ondes et inversion de forme d'onde (*Full Waveform
Inversion*), avec calcul scientifique et perspectives d'application à
l'étude des ressources en eau souterraine et de la géothermie en Afrique.

---

## Avertissement scientifique

**Ce projet travaille exclusivement sur des données synthétiques.**

L'anomalie de vitesse du modèle est un objet de test défini par le
projet. Elle n'est pas un aquifère, elle n'a pas été découverte, et elle
ne correspond à aucune structure géologique réelle, au Bénin ou ailleurs.

Le projet ne comporte :

- aucune acquisition sismique de terrain ;
- aucune étude de site ;
- aucune inversion 3D ;
- aucune imagerie élastique ;
- aucune caractérisation confirmée de ressource.

Les applications mentionnées sont des **motivations et des perspectives**,
pas des résultats obtenus par ce logiciel.

---

## Ce que fait le projet

La chaîne numérique implémentée va de la physique à l'estimation d'un
modèle de vitesse du sous-sol :

```
modèle physique du sous-sol
   -> source sismique (ondelette de Ricker)
   -> propagation des ondes (modèle direct)
   -> signaux aux récepteurs
   -> comparaison observé / prédit
   -> résidu et fonction objectif
   -> gradient par état adjoint
   -> algèbre linéaire (BLAS / LAPACK)
   -> mise à jour du modèle
   -> modèle de vitesse reconstruit
   -> interprétation et limites
```

### Deux formulations d'inversion, nommées distinctement

| Formulation | Paramètres inversés | Gradient | Algèbre linéaire |
|---|---:|---|---|
| **FWI par état adjoint** | 45 451 cellules | état adjoint exact | BLAS niveau 1 (L-BFGS) |
| **Estimation paramétrique réduite** | ≈ 24 coefficients | Jacobienne par différences finies | BLAS 2/3 + LAPACK |

La distinction est importante et doit être respectée dans toute
présentation : la première reconstruit un champ de vitesse, la seconde
estime les coefficients d'une base réduite. Ce ne sont pas la même chose.

---

## Installation

Python 3.11 ou supérieur.

```bash
git clone https://github.com/EgueRichnel2025/seismic-fwi-africa.git
cd seismic-fwi-africa

python -m venv .venv

# Windows, Git Bash :
source .venv/Scripts/activate
# Linux / macOS :
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

Le paquet `threadpoolctl` est recommandé pour le diagnostic de backend :

```bash
python -m pip install threadpoolctl
```

---

## Utilisation

### Vérifier l'installation

```bash
python -m pytest -q
```

### Rejouer toute la chaîne

```bash
# Version grossière, moins d'une minute : vérifie que tout tourne
python scripts/reproduce_all.py --rapide

# Version complète
python scripts/reproduce_all.py
```

### Expériences individuelles

```bash
# Quelle bibliothèque BLAS/LAPACK est réellement chargée ?
python scripts/diagnose_blas_backend.py

# Temps des noyaux d'algèbre linéaire aux tailles réelles du projet
python scripts/benchmark_linear_algebra.py

# Inversion de forme d'onde par état adjoint
python scripts/run_fwi_adjoint.py --rapide      # grille 10 m, quelques secondes
python scripts/run_fwi_adjoint.py               # grille 2 m, pleine résolution

# Inversion paramétrique réduite (Gauss-Newton + LAPACK)
python scripts/run_parametric_inversion.py --rapide
python scripts/run_parametric_inversion.py
```

Les figures sont écrites dans `results/figures/`, les métriques et
modèles dans `experiments/output/`.

---

## Hypothèses physiques et limites

| Hypothèse | Conséquence |
|---|---|
| Milieu acoustique à densité constante | pas d'ondes S, pas de conversion de mode |
| Différences finies d'ordre 2 en espace et en temps | dispersion numérique aux hautes fréquences |
| Zone absorbante de type éponge (pas un PML) | réflexions de bord résiduelles |
| Surface supérieure à pression nulle | modèle incomplet de surface libre |
| Récepteurs = valeur scalaire du champ | pas de réponse électromécanique de géophone |
| Source non calibrée physiquement | amplitudes relatives, pas des pressions |
| Données générées par le solveur de l'inversion | « crime inverse » assumé, résultat plus favorable qu'en situation réelle |

Le point sur le crime inverse mérite d'être explicité à l'oral : les
données observées proviennent du même solveur, de la même grille et de la
même source que l'inversion. C'est délibéré — cela isole le comportement
de l'inversion de toute erreur de modélisation — mais cela rend le
résultat meilleur qu'il ne le serait sur des données réelles.

---

## Résultats vérifiés

### Gradient adjoint

Validé par test de dérivée directionnelle : l'erreur relative décroît en
`ε²` jusqu'au plancher d'arrondi à environ 2 · 10⁻⁹. Le gradient est
exact. Le test est rejoué automatiquement par la suite de tests.

Le propagateur d'inversion reproduit le solveur de référence du projet à
environ 10⁻¹⁵ en écart relatif.

### Backend BLAS/LAPACK

Sur l'environnement de développement : **OpenBLAS, pas Intel oneMKL**.
C'est rapporté tel quel. Un NumPy installé depuis PyPI avec `pip` est lié
à OpenBLAS ; oneMKL nécessite un environnement conda-forge ou la
distribution Intel.

### Coût de calcul

Une simulation directe en pleine résolution (151 × 301, 1 051 pas) prend
environ 0,93 s. L'algèbre linéaire est quatre à cinq ordres de grandeur
moins coûteuse : **le goulot d'étranglement est la propagation d'ondes,
pas BLAS**.

---

## Structure du dépôt

```
src/seismic/
├── model/            modèle de vitesse synthétique
├── wave/             ondelette de Ricker, solveur acoustique 2D
├── inversion/        propagateur, état adjoint, L-BFGS, paramétrique
├── linear_algebra/   appels explicites BLAS et LAPACK
├── hpc/              diagnostic du backend
└── visualization/    conventions graphiques

scripts/              expériences reproductibles
tests/                suite de tests
docs/                 documentation scientifique
results/figures/      figures
experiments/output/   métriques et modèles (non versionnés)
```

### Documentation

| Document | Contenu |
|---|---|
| `docs/inversion.md` | formulation, dérivation de l'adjoint, validation, limites |
| `docs/linear_algebra_hpc.md` | cartographie des routines, backend, mesures |
| `docs/etat_initial_gildas.md` | audit de reprise du dépôt |
| `docs/model_2d.md` | modèle de vitesse synthétique |
| `docs/wave_propagation.md` | solveur direct |
| `docs/decisions/` | décisions d'architecture (ADR) |
| `docs/journal_de_bord.md` | journal des choix et résultats |

---

## Organisation du travail

- **Richnel** : production audiovisuelle du documentaire (scénario,
  Blender, voix off, montage). Responsable de `media/video/` et
  `results/videos/`.
- **Gildas** : cœur scientifique et logiciel (modèle, solveur, données,
  inversion, BLAS/LAPACK, tests, documentation).

Les figures scientifiques de `results/figures/` sont reproductibles par
les commandes ci-dessus et peuvent servir au documentaire, à condition
que les limites ci-dessus soient respectées dans le commentaire.
