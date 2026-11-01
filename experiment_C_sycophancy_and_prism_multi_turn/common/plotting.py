"""
Publication-ready plotting configuration and styling for paper figures.
Standardizes palettes, fonts, DPI, and formatting across all experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Color Palette for Model Families & Political Perspectives
PALETTE = {
    # Perspectives
    "liberal": "#2b5c8f",       # Refined Academic Blue
    "conservative": "#b22222",  # Refined Academic Red
    "control": "#555555",       # Academic Neutral Gray
    "neutral": "#555555",
    
    # Model Families
    "western": "#1f77b4",       # Blue
    "chinese": "#d62728",       # Red
    "sea": "#2ca02c",           # Green
    
    # Specific Models
    "llama3.1": "#1f77b4",
    "gemma3": "#17becf",
    "qwen3": "#d62728",
    "glm4": "#ff7f0e",
    "llama_sealion": "#2ca02c",
    "qwen_sealion": "#8c564b",
    "typhoon2": "#9467bd",
}


def apply_publication_style():
    """Applies clean, high-DPI scientific styling to matplotlib."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#ebebeb",
    })
