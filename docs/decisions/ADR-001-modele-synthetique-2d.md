# ADR-001 — Choix du modèle synthétique 2D

- Date : 2026-10-28
- Statut : accepté
- Domaine : modélisation géophysique

## Contexte

Le projet doit relier un problème géophysique à une modélisation
mathématique, à des calculs numériques et, à terme, à une inversion
de données sismiques.

Pour tester une chaîne directe puis inverse, il faut commencer par
un modèle de référence dont les paramètres sont entièrement connus.

## Décision

Construire une coupe synthétique 2D composée de trois couches horizontales
et d'une anomalie elliptique à vitesse réduite.

Paramètres retenus :

- étendue : 600 m x 300 m ;
- pas spatial : 2 m x 2 m ;
- vitesses de fond : 1200, 2000 et 2800 m/s ;
- anomalie : 1500 m/s ;
- centre de l'anomalie : (360 m, 125 m) ;
- demi-axes de l'anomalie : 55 m et 28 m.

Le code de référence est :

    src/seismic/model/velocity_model.py

## Justification

1. Le modèle est suffisamment simple pour être expliqué à l'oral.
2. Les différentes couches ont une interprétation physique explicite.
3. L'anomalie introduit un contraste local à retrouver.
4. La vérité synthétique est connue pour évaluer une future reconstruction.
5. Les résultats sont reproductibles à partir du code versionné.

## Alternatives écartées pour cette première version

### Utiliser immédiatement des données de terrain

Aucun jeu de données de terrain n'a été validé pour cette première étape.
Attribuer arbitrairement le modèle à une zone réelle serait trompeur.

### Modéliser directement un aquifère réel

Les paramètres actuels ne proviennent pas d'une caractérisation de site.
L'anomalie n'est donc pas identifiée comme une réserve d'eau souterraine.

### Commencer directement en 3D

La priorité est de valider le modèle, la propagation et l'acquisition en 2D
avant d'introduire les coûts supplémentaires d'un calcul en trois dimensions.

## Conséquences

- Le modèle sert de référence pour les expériences numériques.
- Les hypothèses doivent rester visibles dans les figures et les rapports.
- La prochaine étape devra simuler une propagation physiquement cohérente.
- Une inversion ultérieure devra être évaluée par comparaison au modèle connu.
- Aucun résultat ne devra être présenté comme une découverte géologique réelle.

## Conditions de réexamen

Cette décision pourra évoluer lorsque le solveur d'ondes sera validé,
lorsque les premières inversions seront disponibles ou lorsqu'un jeu de
données de terrain approprié aura été identifié et documenté.
