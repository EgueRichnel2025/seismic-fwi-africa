# Propagation acoustique 2D par différences finies

## 1. Identification

- Projet : Seismic FWI Africa
- Version : 0.1 — Scientific Core / Proof of Concept
- Date : 2026-10-28
- Modèle de vitesse : `src/seismic/model/velocity_model.py`
- Solveur : `src/seismic/wave/acoustic_2d.py`
- Script de simulation : `scripts/simulate_wave_propagation.py`
- Tests : `tests/test_acoustic_2d.py`

Les données sont synthétiques. Cette simulation ne représente pas une
mesure effectuée sur un site géologique réel.

## 2. Objectif

Résoudre numériquement une équation d'onde dans le modèle de vitesse 2D
et produire des champs d'onde dont la cohérence numérique peut être testée.

## 3. Modèle physique

L'équation acoustique utilisée est :

    d²u/dt² = v²(x,z) [d²u/dx² + d²u/dz²] + s(x,z,t)

où :

- u est un champ scalaire ;
- v(x,z) est la vitesse du milieu, en m/s ;
- x et z sont les coordonnées spatiales, en mètres ;
- t est le temps, en secondes ;
- s est un terme source synthétique.

La formulation suppose un milieu acoustique à densité constante. Elle ne
représente pas la propagation élastique complète dans un milieu réel.

## 4. Discrétisation

Le champ est défini sur une grille régulière :

    dx = 2 m
    dz = 2 m

Le schéma utilise des différences centrées d'ordre deux en espace et un
schéma temporel explicite d'ordre deux.

Pour un point intérieur (i,j) et un instant n :

    u(i,j,n+1) = 2 u(i,j,n) - u(i,j,n-1)
                  + dt² v(i,j)² Laplacien(u(i,j,n))
                  + dt² s(i,j,n)

Le Laplacien discret est :

    Laplacien(u) =
        [u(i+1,j) - 2u(i,j) + u(i-1,j)] / dz²
      + [u(i,j+1) - 2u(i,j) + u(i,j-1)] / dx²

Le champ est initialisé à zéro.

## 5. Source sismique

La source est une impulsion de Ricker de fréquence centrale 25 Hz,
positionnée à :

    x_s = 60 m
    z_s = 6 m

Sa forme temporelle est :

    w(t) = [1 - 2 a²] exp(-a²)

avec :

    a = pi f_0 (t - t_0)
    t_0 = 1.5 / f_0

L'amplitude numérique est choisie pour rendre le champ observable et
stable dans notre expérience. Elle n'est pas une amplitude physique
calibrée sur un instrument ou un site réel.

## 6. Stabilité temporelle

Le pas de temps utilisé par le script est :

    dt = 0.0004 s

Le nombre CFL est défini par :

    C = v_max dt sqrt(1/dx² + 1/dz²)

Pour v_max = 2800 m/s et dx = dz = 2 m :

    C ≈ 0.792

Cette valeur respecte la condition C <= 1 du schéma explicite choisi.

Le programme calcule ce nombre à partir des paramètres réellement
utilisés et rejette un pas de temps qui dépasse la limite théorique.

## 7. Conditions aux limites

- Surface supérieure : champ fixé à zéro, interprété comme une limite
  à pression nulle dans ce modèle acoustique.
- Bords latéraux : zone absorbante simplifiée.
- Bord inférieur : zone absorbante simplifiée.

La zone absorbante applique une atténuation numérique dans une région
proche des bords latéraux et inférieur.

Elle réduit les réflexions artificielles, mais ne constitue pas une
PML (Perfectly Matched Layer). Des réflexions résiduelles restent possibles.

## 8. Paramètres de référence

| Paramètre | Valeur |
|---|---:|
| Largeur | 600 m |
| Profondeur | 300 m |
| Pas horizontal | 2 m |
| Pas vertical | 2 m |
| Nombre de points | 45 451 |
| Vitesse maximale | 2 800 m/s |
| Pas temporel | 0,0004 s |
| Durée demandée | 0,24 s |
| Fréquence source | 25 Hz |
| Position de la source | (60 m, 6 m) |

## 9. Résultats attendus

Le programme produit :

- une séquence de champs d'onde ;
- le maximum absolu du champ à chaque pas temporel ;
- le nombre CFL ;
- une animation GIF ;
- une figure statique d'un champ intermédiaire.

Fichiers attendus :

    results/figures/wave_propagation.gif
    results/figures/wavefield_t_0.120s.png

La valeur de temps utilisée dans le nom de la figure statique correspond
à l'instant effectivement enregistré le plus proche de 0,12 s.

## 10. Validation

Les tests vérifient :

1. le caractère fini des champs calculés ;
2. l'apparition d'un champ non nul après injection de la source ;
3. la conformité des dimensions des champs ;
4. l'enregistrement des instants initiaux et finaux ;
5. le rejet d'un pas temporel instable ;
6. le rejet de paramètres invalides de la source.

Ces tests constituent des contrôles numériques. Ils ne suffisent pas, à
eux seuls, à prouver l'exactitude géophysique du solveur.

Une validation scientifique plus poussée pourra comparer le temps
d'arrivée dans un milieu homogène à une estimation analytique, étudier
la convergence lorsque le maillage est raffiné et quantifier les
réflexions aux limites.

## 11. Reproductibilité

Depuis la racine du dépôt :

    python -m pytest -v
    python scripts/simulate_wave_propagation.py

Le même modèle, les mêmes paramètres et la même version du code doivent
permettre de reproduire les figures.

## 12. Limites

Cette étape ne traite pas encore :

- les ondes élastiques P et S ;
- un modèle de densité variable ;
- la topographie ;
- une géométrie complexe ;
- des données de terrain ;
- les sismogrammes issus de plusieurs géophones ;
- le bruit d'acquisition ;
- l'inversion du modèle ;
- un solveur 3D ;
- l'optimisation des noyaux numériques avec Intel oneMKL.

Ces fonctionnalités seront introduites seulement lorsque le socle
physique et numérique correspondant sera validé.
