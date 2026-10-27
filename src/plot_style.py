"""One consistent, quiet look for every matplotlib figure in the project."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Reference palette (validated categorical order: blue, orange, aqua)
COLORS = {
    "history": "#2a78d6",   # series 1 - observed data
    "forecast": "#eb6834",  # series 2 - model forecast
    "third": "#1baf7a",     # series 3 - baseline / comparison
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}


def apply_style():
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 4),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.facecolor": COLORS["surface"],
            "axes.facecolor": COLORS["surface"],
            "savefig.facecolor": COLORS["surface"],
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink2"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(
                color=[COLORS["history"], COLORS["forecast"], COLORS["third"]]
            ),
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelcolor": COLORS["ink2"],
            "ytick.labelcolor": COLORS["ink2"],
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": COLORS["ink"],
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        }
    )


def savefig(fig, name: str):
    """Save a figure as PNG into reports/figures (for the report and PPT)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
