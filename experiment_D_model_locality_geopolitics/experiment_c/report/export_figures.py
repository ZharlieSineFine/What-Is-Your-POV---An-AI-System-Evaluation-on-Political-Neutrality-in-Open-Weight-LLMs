"""Export the three main-body figures for pasting into Word: SVG (sharp at any size) and 600-dpi PNG.

Usage (from the project root): python experiment_c/report/export_figures.py <stats dir> <output dir>
"""
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

STATS, DEST = Path(sys.argv[1]), Path(sys.argv[2])
DEST.mkdir(parents=True, exist_ok=True)
NAMES = {
    "fig1_rq1.png": "Figure_5.1_RQ1_model_differences",
    "fig2_rq2_rq3.png": "Figure_5.2_RQ2_RQ3_region_and_home_side",
    "fig3_rq4.png": "Figure_5.3_RQ4_regional_post_training",
}
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as editable text in the SVG
_savefig = matplotlib.figure.Figure.savefig


def savefig(self, fname, *args, **kwargs):
    name = NAMES.get(Path(fname).name)
    if name:
        _savefig(self, DEST / f"{name}.svg", bbox_inches="tight")
        _savefig(self, DEST / f"{name}.png", dpi=600, bbox_inches="tight")


matplotlib.figure.Figure.savefig = savefig
sys.argv = ["report_figures.py", str(STATS), str(DEST / "_unused")]
runpy.run_path(str(Path(__file__).with_name("report_figures.py")), run_name="__main__")
print("\n".join(sorted(p.name for p in DEST.iterdir() if p.is_file())))
