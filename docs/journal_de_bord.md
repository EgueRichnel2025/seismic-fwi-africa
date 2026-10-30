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
