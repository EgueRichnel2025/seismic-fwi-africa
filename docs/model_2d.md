# Modèle synthétique 2D de vitesse sismique

## 1. Identification

- Projet : Seismic FWI Africa
- Version : 0.1 — Scientific Core / Proof of Concept
- Date de conception : 2026-10-28
- Type de modèle : synthétique
- Dimensions : 600 m horizontalement et 300 m en profondeur
- Implémentation : `src/seismic/model/velocity_model.py`
- Visualisation : `scripts/plot_velocity_model.py`
- Tests : `tests/test_velocity_model.py`

Ce modèle constitue une hypothèse numérique de travail. Il ne représente
pas une mesure géophysique effectuée au Bénin ni sur un autre site réel.

## 2. Objectif scientifique

Définir un milieu dans lequel nous pourrons ensuite simuler la propagation
d'ondes sismiques et produire des données synthétiques.

Le caractère synthétique est important : pour tester ultérieurement une
inversion, nous devons connaître le modèle de référence utilisé pour
générer les données.

Ce modèle de référence sera noté m_true.

## 3. Système de coordonnées

- x : distance horizontale, en mètres ;
- z : profondeur, en mètres, positive vers le bas ;
- v(x, z) : vitesse de propagation sismique, en mètres par seconde.

La matrice des vitesses est organisée comme suit :

    velocity_m_s.shape = (Nz, Nx)

Les lignes représentent la profondeur et les colonnes représentent
la position horizontale.

## 4. Discrétisation spatiale

Étendue horizontale : 600 m
Étendue verticale : 300 m

Pas spatial :

    dx = 2 m
    dz = 2 m

Nombre de points :

    Nx = 301
    Nz = 151

Nombre total de points :

    Nx * Nz = 45 451

Ces valeurs décrivent une grille régulière. Les points de grille ne
doivent pas être confondus avec des mesures expérimentales.

## 5. Modèle de vitesse de fond

Le modèle de fond est défini par :

    v_bg(z) = 1200 m/s, si 0 <= z < 60 m
    v_bg(z) = 2000 m/s, si 60 <= z < 180 m
    v_bg(z) = 2800 m/s, si 180 <= z <= 300 m

Les deux interfaces horizontales se situent à 60 m et 180 m.

Ces valeurs sont choisies pour construire une expérience numérique
reproductible. Elles ne sont pas calibrées sur une formation géologique
identifiée au Bénin.

## 6. Anomalie de vitesse

L'anomalie a une forme elliptique.

Centre :

    x_c = 360 m
    z_c = 125 m

Demi-axes :

    a = 55 m
    b = 28 m

Un point appartient à l'anomalie si :

    ((x - x_c) / a)^2 + ((z - z_c) / b)^2 <= 1

À l'intérieur de cette région :

    v(x, z) = 1500 m/s

À l'extérieur, la vitesse de fond est conservée.

L'anomalie constitue un contraste de vitesse destiné à tester les méthodes
de propagation et d'inversion. Elle ne doit pas être présentée comme la
preuve de la présence d'un aquifère.

## 7. Géométrie d'acquisition indicative

La figure montre des positions proposées pour les prochaines étapes :

- source sismique : x = 60 m, à la surface ;
- géophones : de x = 80 m à x = 580 m ;
- espacement nominal entre géophones : 20 m ;
- profondeur de la ligne d'acquisition : z = 0 m.

Cette géométrie n'est pas encore utilisée pour produire des données.
Elle sera implémentée dans le module d'acquisition.

## 8. Critères de validation

Le modèle est acceptable si :

1. les dimensions de la grille sont conformes aux paramètres ;
2. les pas dx et dz sont respectés ;
3. les vitesses sont finies et strictement positives ;
4. les trois couches ont les valeurs attendues ;
5. l'anomalie occupe la région elliptique définie ;
6. les points extérieurs à l'anomalie conservent la vitesse de fond ;
7. le script produit une figure exploitable ;
8. les tests automatisés passent.

## 9. Limites

Ce modèle ne prend pas encore en compte :

- des données mesurées sur un site réel ;
- une structure géologique complexe ;
- les variations de densité ;
- une formulation élastique complète ;
- les ondes de cisaillement ;
- les conditions de bord absorbantes ;
- la propagation d'ondes ;
- l'acquisition de sismogrammes ;
- l'inversion des données.

Ces éléments seront étudiés au fur et à mesure de la construction du projet.

## 10. Reproductibilité

Depuis la racine du dépôt :

    python scripts/plot_velocity_model.py
    python -m pytest -v

Figure produite :

    results/figures/model_2d_synthetic.png

## 11. Références méthodologiques

Les travaux de propagation d'ondes prévus dans la suite du projet s'appuieront
sur les formulations usuelles de l'équation acoustique 2D et sur des méthodes
numériques adaptées à cette équation.

1. Computational Seismology — Finite-Difference Method: Acoustic Waves in 2D.
   https://krischer.github.io/seismo_live_build/html/Computational%20Seismology/The%20Finite-Difference%20Method/fd_ac2d_homogeneous_solution.html

2. Adaptive finite difference for seismic wavefield modelling in acoustic media.
   Scientific Reports.
   https://www.nature.com/articles/srep30302
