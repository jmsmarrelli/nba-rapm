"""Plot styling shared by every figure in this repo.

One place to change the look. Everything downstream imports INK, GRID, SERIES
and calls apply() once at import time.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8880"
GRID = "#e7e6e1"

# Categorical slots, assigned in fixed order and never cycled.
SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Single-hue ramp for magnitude.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
# Warm/cool poles with a neutral middle, for signed quantities.
DIVERGING = ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a52a2a"]

NEUTRAL = "#c9c7c0"  # for the "everyone else" background layer in scatterplots


def apply():
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_3,
            "ytick.color": INK_3,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
        }
    )


def strip(ax, keep=("left", "bottom")):
    """Hairline frame: drop every spine not explicitly kept."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def caption(fig, text, y=-0.02):
    """Source / method note under a figure. Keeps the axis clean."""
    fig.text(0.0, y, text, ha="left", va="top", fontsize=7.6, color=INK_3, wrap=True)


def caption_under(fig, ax, text):
    """Caption pinned just below an axis, for figures with a fixed aspect ratio
    where the axis does not fill the figure."""
    fig.canvas.draw()
    box = ax.get_position()
    fig.text(box.x0, box.y0 - 0.045, text, ha="left", va="top",
             fontsize=7.6, color=INK_3)


def subtitle(ax, text):
    """A second line under the title, in the secondary ink."""
    ax.set_title(text, loc="left", fontsize=9.5, color=INK_2, fontweight="normal", pad=6)


def titled(ax, title, sub=None):
    """Bold title with a lighter line under it, both left-aligned to the axis.

    The subtitle rides in the axis's own title slot and the title is offset
    above it in points, so the pair keeps its spacing whatever the figure size
    or aspect ratio does to the axis box.
    """
    if sub:
        ax.set_title(sub, loc="left", fontsize=9.5, color=INK_2, fontweight="normal", pad=7)
        ax.annotate(
            title, xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(0, 21), textcoords="offset points",
            ha="left", va="bottom", fontsize=12.5, fontweight="semibold",
            color=INK, annotation_clip=False,
        )
    else:
        ax.set_title(title, loc="left")


apply()
