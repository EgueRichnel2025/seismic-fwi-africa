# ADR-002 — Solveur acoustique 2D par différences finies

- Date : 2026-10-28
- Statut : proposé, validation locale en attente
- Domaine : modélisation physique et calcul numérique

## Contexte

Le modèle synthétique de vitesse 2D est disponible et ses propriétés
numériques sont vérifiées.

La prochaine étape consiste à simuler la propagation d'une onde dans
ce modèle avant de développer l'acquisition sismique et l'inversion.

## Décision

Utiliser un schéma explicite d'ordre deux en espace et en temps pour
l'équation acoustique 2D à densité constante.

Paramètres principaux :

- grille de 301 x 151 points ;
- dx = dz = 2 m ;
- vitesse maximale = 2800 m/s ;
- dt = 0,0004 s ;
- fréquence centrale de la source = 25 Hz ;
- position de la source = (60 m, 6 m) ;
- durée demandée = 0,24 s.

Le schéma est implémenté dans :

    src/seismic/wave/acoustic_2d.py

## Justification

1. Le modèle est directement relié à l'équation d'onde.
2. Le schéma numérique est explicite et explicable à la soutenance.
3. La stabilité temporelle est contrôlée à partir de la vitesse maximale.
4. Le résultat peut être visualisé et vérifié par des tests automatisés.
5. La simulation permet de préparer la future acquisition des sismogrammes.

## Conditions aux limites

La surface supérieure est fixée à une valeur nulle. Une atténuation
numérique est appliquée dans des zones proches des côtés et du fond.

Cette approche est une approximation. Elle ne doit pas être présentée
comme une PML parfaitement absorbante.

## Alternatives réservées aux phases ultérieures

- schémas spatiaux d'ordre supérieur ;
- formulation élastique ;
- conditions absorbantes plus avancées ;
- accélération par calcul parallèle ;
- optimisation et intégration de bibliothèques HPC.

Ces éléments ne seront considérés qu'après validation du premier solveur.

## Conditions d'acceptation

- tests automatisés réussis ;
- CFL inférieur ou égal à 1 ;
- champ calculé fini et non nul ;
- dimensions des résultats cohérentes ;
- animation et figure statique générées ;
- hypothèses et limites documentées.

## Conséquences

La simulation produit un champ d'onde synthétique, mais pas encore des
données sismiques mesurées. L'étape suivante consistera à enregistrer
le champ au niveau de géophones, puis à construire des sismogrammes.
