# Journal de bord scientifique et technique

Ce document conserve la trace des décisions, validations, livrables et
limites rencontrées pendant le projet.

Les tests, les figures, les scripts et l'historique Git constituent ensemble
les éléments de preuve de la reproductibilité du travail.

## Historique

| Date | Étape | Livrables | État de validation | Référence |
|---|---|---|---|---|
| 2026-10-28 | Initialisation | `pyproject.toml`, README, structure Python | 2 tests réussis ; premier push confirmé par le terminal | `b861dbd` |
| 2026-10-28 | Modèle synthétique 2D | Modèle de vitesse, figure, tests et documentation | En attente de validation locale | Commit à retrouver dans l'historique Git |

## Règles de traçabilité

Pour chaque étape scientifique :

1. enregistrer les hypothèses et les unités ;
2. conserver le code qui produit le résultat ;
3. conserver les tests pertinents ;
4. enregistrer la commande utilisée pour reproduire le résultat ;
5. sauvegarder les figures utiles à l'analyse ;
6. noter les écarts et les limites ;
7. créer un commit Git identifiable après validation.

## Journal des expériences

### Expérience EXP-001 — Construction du modèle 2D

- Date de conception : 2026-10-28
- Objectif : créer une grille synthétique de vitesse sismique.
- Script : `scripts/plot_velocity_model.py`
- Tests : `tests/test_velocity_model.py`
- Figure attendue : `results/figures/model_2d_synthetic.png`
- Résultat d'exécution : à renseigner après validation locale.
- Écarts ou anomalies : à renseigner après validation locale.

## Protocole de présentation

Pour chaque figure présentée à la soutenance, expliquer :

- la grandeur affichée ;
- son unité ;
- la manière dont elle a été calculée ;
- les hypothèses utilisées ;
- le test qui vérifie sa cohérence ;
- ce que la figure permet de conclure ;
- ce qu'elle ne permet pas de conclure.

Une figure produite à partir d'un modèle synthétique ne doit jamais être
présentée comme une mesure réalisée sur un site réel.

### Expérience EXP-002 — Propagation acoustique 2D

- Date de conception : 2026-10-28
- Objectif : simuler la propagation d'une impulsion sismique dans le modèle
  de vitesse synthétique.
- Équation : équation acoustique 2D à densité constante.
- Méthode : différences finies d'ordre deux en espace et en temps.
- Source : impulsion de Ricker de fréquence centrale 25 Hz.
- Pas temporel prévu : 0,0004 s.
- Durée demandée : 0,24 s.
- Contrôle de stabilité : nombre CFL calculé par le solveur.
- Tests : `tests/test_acoustic_2d.py`
- Documentation : `docs/wave_propagation.md`
- Décision : `docs/decisions/ADR-002-propagation-acoustique-2d.md`
- Animation attendue : `results/figures/wave_propagation.gif`
- Figure statique attendue : `results/figures/wavefield_t_0.120s.png`
- État de validation : en attente d'exécution et d'inspection visuelle.
- Résultats, écarts et limites : à renseigner après validation locale.

---

## Reprise du cœur scientifique — 30 octobre 2026

Travail mené sur la branche `gildas/scientific-core`. Audit complet
consigné dans `docs/etat_initial_gildas.md`.

### Décisions prises

**D-01 — Inverser `m = v²` plutôt que `v`.** Dans l'équation d'onde
discrétisée, `v²` intervient linéairement. Ce choix rend la dérivation de
l'adjoint exacte. Le gradient en vitesse s'obtient par dérivation
composée.

**D-02 — Dériver l'adjoint du schéma discret, pas du continu.** Les deux
approches diffèrent par des termes d'ordre du pas de discrétisation.
Seule la première donne le gradient exact de la fonction objectif
réellement calculée, ce qui permet au test de dérivée directionnelle de
passer à la précision machine. C'est la décision la plus structurante de
cette reprise.

**D-03 — Conserver deux formulations d'inversion.** La FWI adjointe ne
forme aucune matrice dense et n'utilise donc que du BLAS niveau 1. Une
seconde formulation, à base réduite, fait apparaître légitimement
DGEMM, DGEMV, DPOSV et DGELSD. Les deux sont nommées distinctement :
appeler la seconde « FWI » serait inexact.

**D-04 — Préconditionner par `H₀`, pas par le gradient.** Préconditionner
le gradient directement rendrait incohérentes les paires de courbure de
L-BFGS et invaliderait la condition d'Armijo. Le pseudo-hessien entre
comme matrice initiale de la récursion.

**D-05 — Rapporter `E_cible` en plus de `E_m`.** L'anomalie occupe 1 220
cellules sur 45 451 : une reconstruction parfaite ne ferait bouger `E_m`
que de quelques millièmes. Rapporter uniquement `E_m` masquerait le
résultat.

**D-06 — Rapporter honnêtement l'absence d'Intel oneMKL.** Le backend
réellement chargé est OpenBLAS. Le projet ne prétend nulle part le
contraire, et la marche à suivre pour obtenir oneMKL est documentée.

### Problèmes rencontrés

**P-01 — Première inversion sans reconstruction.** `Φ` décroissait d'un
facteur 2,5 mais l'anomalie restait à 1985 m/s au lieu de 1500. Causes
identifiées : gradient dominé par le voisinage de la source, et tir
unique insuffisant pour éclairer une cible à 300 m de distance. Corrigé
par le préconditionneur d'illumination et une acquisition à cinq tirs.
Effet : `E_cible` passe de 0,333 à 0,086 sur la grille de 10 m.

**P-02 — Lissage gaussien non conservatif.** `np.convolve(mode="same")`
suppose un champ nul hors des bords et atténuait donc artificiellement le
gradient sur les bords du modèle, y compris près de la surface. Détecté
par le test `test_smoothing_preserves_a_constant_field`, qui vérifie
qu'un champ constant reste constant. Corrigé par rembourrage par
réplication.

**P-03 — Appel DGELSD incomplet.** La routine exige des tailles d'espace
de travail fournies par la routine de requête associée `gelsd_lwork`, et
un second membre de taille `max(m, n)`. Corrigé.

**P-04 — Lecture erronée du CFL.** Une analyse préliminaire annonçait
0,56 en raisonnant en nombre de Courant non normalisé. La valeur qui se
compare à la limite du code est 0,792. Même configuration, mais la marge
de stabilité est plus faible qu'annoncé.

### Résultats

Gradient adjoint validé : erreur relative décroissant en `ε²` jusqu'à
2,4 · 10⁻⁹. Propagateur d'inversion conforme au solveur de référence à
10⁻¹⁵.

Suite de tests portée de 27 à 54 tests.

### Limites qui restent à exposer

Crime inverse assumé, physique acoustique simplifiée, zone éponge au lieu
d'un PML, résolution limitée par l'ouverture et la bande passante. Toutes
sont détaillées dans `docs/inversion.md`, section 6.
