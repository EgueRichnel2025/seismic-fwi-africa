# État initial du dépôt — audit de reprise

Livrable de la phase 0 du plan de travail. Cette note consigne ce qui a
été **vérifié par exécution**, et non ce qui était supposé.

Date de l'audit : 30 octobre 2026.

---

## 1. État Git vérifié

| Élément | Valeur constatée |
|---|---|
| Dépôt distant | `https://github.com/EgueRichnel2025/seismic-fwi-africa` |
| Branche | `main` |
| Commit de départ | `d3ff3fa` — *feat: add scientific modeling and seismic validation core* |
| Historique | un seul commit |

**Point à retenir.** Le document de contexte craignait que le travail
scientifique soit resté local et non publié. Ce n'est pas le cas :
l'ensemble du socle scientifique est bien présent sur `origin/main`. Les
fichiers de production vidéo (`media/video/`, `results/videos/`) ne sont
en revanche pas dans le dépôt, ce qui est cohérent avec la séparation
voulue entre socle scientifique et montage.

---

## 2. Environnement

| Composant | Version d'audit |
|---|---|
| Python | 3.13.18 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| pytest | 8.x |
| Backend BLAS | scipy-openblas 0.3.34 |

Installation effectuée par `pip install -e ".[dev]"`, en suivant les
dépendances déclarées dans `pyproject.toml`. L'extra `dev` existe bien
et contient `pytest`.

---

## 3. Tests

```
python -m pytest -q
27 passed in 0.89s
```

Les 27 tests annoncés passent effectivement. Chiffre confirmé.

---

## 4. Audit du modèle direct

### 4.1 Ce qui a été vérifié dans le code

`src/seismic/wave/acoustic_2d.py` a été lu intégralement. Constats :

- l'amortissement est bien intégré **dans le schéma discrétisé**, et non
  appliqué par multiplication du champ après coup ;
- le noyau spatial de source est gaussien, de largeur physique fixe, et
  normalisé pour que `somme(noyau) · dx · dz = 1`, donc indépendant du
  pas de grille ;
- les bords du noyau de source sont mis à zéro, cohérent avec les
  conditions de Dirichlet ;
- la validation des entrées est soignée : finitude, positivité,
  régularité des axes, positions dans le domaine, type de
  `snapshot_stride` ;
- le coefficient `source_strength_numeric` est correctement nommé comme
  non calibré physiquement.

### 4.2 Condition CFL — correction d'une lecture antérieure

La quantité testée par le code est

```
cfl = v_max · dt · sqrt(1/dx² + 1/dz²)  ≤  1
```

C'est **la bonne normalisation** pour le schéma 2D d'ordre 2, et le test
est correct.

Valeur effective sur la configuration de référence (`dx = dz = 2 m`,
`dt = 4·10⁻⁴ s`, `v_max = 2800 m/s`) : **CFL = 0,7920**.

Une analyse préliminaire avait annoncé 0,56, en raisonnant en nombre de
Courant non normalisé (`v·dt/h`). Les deux chiffres décrivent la même
configuration — `0,56 · √2 = 0,792` — mais c'est le second qui se compare
à la limite du code. **La marge de stabilité est donc plus faible
qu'annoncé initialement** et ne doit pas être réduite sans recalcul.

### 4.3 Dispersion numérique

À `f₀ = 25 Hz` et `v = 1200 m/s`, la longueur d'onde dominante vaut 48 m,
soit 24 points de grille. Mais une ondelette de Ricker porte de l'énergie
jusqu'à environ `2,5 · f₀ ≈ 62 Hz`, soit environ 10 points par longueur
d'onde. C'est acceptable pour un schéma d'ordre 2, mais c'est la limite
basse : enrichir le contenu haute fréquence de la source exigerait de
raffiner la grille.

---

## 5. Audit des scripts d'inversion préexistants

Les trois scripts ont été lus ligne par ligne. Conclusion :

| Script | Ce qu'il fait réellement | Paramètres inversés |
|---|---|---:|
| `invert_anomaly_velocity.py` | balayage de vitesses candidates | 1 |
| `invert_anomaly_geometry.py` | Gauss-Newton, une itération | 3 |
| `invert_anomaly_geometry_iterative.py` | Gauss-Newton itératif | 3 |

Les deux derniers construisent une Jacobienne par différences finies
(une simulation par paramètre), mettent à l'échelle les colonnes,
diagnostiquent par SVD et retiennent un pas par recherche linéaire. Le
travail est propre.

**Mais ce ne sont pas des FWI.** Ils estiment le centre et la vitesse
d'une ellipse dont la forme est supposée connue a priori. Le document de
contexte avait raison de poser cet avertissement. Ces scripts sont
conservés : ils restent une vérification utile du pipeline, à condition
d'être nommés « estimation paramétrique ».

Par ailleurs, aucun de ces scripts n'importait `scipy.linalg` : avant
cette reprise, le projet n'avait **aucun point d'entrée BLAS ou LAPACK
explicite**, seulement `np.linalg.lstsq` et `np.linalg.svd` sur une
matrice à 3 colonnes — trop petite pour que l'algèbre linéaire ait un
sens en termes de performance.

---

## 6. Backend BLAS

```
Fournisseur detecte  : openblas
Intel oneMKL utilise : NON
```

Conforme à l'attendu : un NumPy installé depuis PyPI avec `pip` est lié à
OpenBLAS. Ce point est documenté dans `docs/linear_algebra_hpc.md` avec
la marche à suivre si l'on souhaite réellement oneMKL.

---

## 7. Coût de calcul mesuré

| Quantité | Mesure |
|---|---:|
| Une simulation directe, grille 151 × 301, 1 051 pas | **0,93 s** |

C'est le chiffre qui a orienté toute la stratégie de reprise. À ce coût,
une FWI par état adjoint revient à environ 2 s par itération et par tir,
ce qui rend le projet **non limité par le calcul**. Les deux formulations
d'inversion — adjointe sur toute la grille, et paramétrique réduite —
sont donc toutes deux réalisables.

---

## 8. Bugs trouvés pendant la reprise

| Bug | Détection | Correction |
|---|---|---|
| Lissage gaussien non conservatif : `np.convolve(mode="same")` suppose un champ nul hors des bords, ce qui atténuait artificiellement le gradient sur les bords du modèle | test `test_smoothing_preserves_a_constant_field` | rembourrage par réplication des bords |
| Appel `DGELSD` incomplet : tailles d'espace de travail manquantes | exécution | appel à la routine de requête `gelsd_lwork` |

---

## 9. Classement de l'existant

| Composant | État |
|---|---|
| Modèle de vitesse synthétique | **déjà fonctionnel** |
| Ondelette de Ricker | **déjà fonctionnel** |
| Solveur acoustique 2D | **déjà fonctionnel** |
| Validations directes (arrivée, convergence) | **déjà fonctionnel** |
| Données synthétiques observées | **déjà fonctionnel** |
| Scripts d'« inversion » | **partiellement fonctionnel** — corrects mais mal nommés |
| FWI réelle | **non commencé** → réalisé pendant cette reprise |
| BLAS / LAPACK explicites | **non commencé** → réalisé pendant cette reprise |
| Diagnostic de backend | **non commencé** → réalisé pendant cette reprise |

---

## 10. Blocages

Aucun blocage rencontré. Le dépôt était en état d'être repris
directement.
