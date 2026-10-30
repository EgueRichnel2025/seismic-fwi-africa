"""Conventions graphiques communes aux figures scientifiques du projet.

Les figures sont en niveaux de gris, sans code couleur, en police Tahoma
lorsqu'elle est disponible. Ce choix les rend lisibles à l'impression en
noir et blanc et cohérentes avec la documentation technique du projet.

Tahoma est présente sur Windows mais pas sur la plupart des systèmes
Linux. Plutôt que d'émettre un avertissement par élément de texte, la
police est sélectionnée parmi celles réellement installées.
"""

import matplotlib
from matplotlib import font_manager

# Par ordre de préférence. La dernière est toujours fournie avec
# matplotlib, donc la liste ne peut pas être vide à l'arrivée.
PREFERRED_FONTS = ("Tahoma", "Verdana", "DejaVu Sans")

# Rampe de gris utilisée pour distinguer les courbes sans couleur.
GREY_LEVELS = ("black", "0.45", "0.7")


def available_font() -> str:
    """Retourne la première police préférée réellement installée.

    Returns:
        Le nom de la police à utiliser.
    """

    installed = {font.name for font in font_manager.fontManager.ttflist}

    for candidate in PREFERRED_FONTS:
        if candidate in installed:
            return candidate

    return "DejaVu Sans"


def apply_project_style() -> str:
    """Applique les conventions graphiques du projet à matplotlib.

    Returns:
        Le nom de la police effectivement retenue.
    """

    font_name = available_font()

    matplotlib.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 12,
            "axes.grid": False,
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "image.cmap": "gray_r",
        }
    )

    return font_name
