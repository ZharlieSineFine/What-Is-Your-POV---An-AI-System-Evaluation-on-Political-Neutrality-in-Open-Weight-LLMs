"""Figures for the Experiment C report section (main judge GPT-4.1-mini unless --judge)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STATS = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
JUDGE = sys.argv[3] if len(sys.argv) > 3 else "openai__gpt-4.1-mini"
TAG = "" if JUDGE == "openai__gpt-4.1-mini" else "_deepseek"

FAM = {"western": "#2a78d6", "chinese": "#eb6834", "sea": "#1baf7a"}
FAM_LABEL = {"western": "Western", "chinese": "Chinese", "sea": "SEA"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SHORT = {
    "gemma-3-12b-it": "Gemma-3-12B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B", "glm-4-9b-chat": "GLM-4-9B",
    "llama-sea-lion-v3-8b-it": "SEA-LION-v3", "qwen-sea-lion-v4-8b-vl": "SEA-LION-v4",
    "llama3.1-typhoon2-8b-instruct": "Typhoon2",
}
LEVEL_LABEL = {"F1-default": "F1 cued", "F2-default": "F2 uncued", "F3": "F3 caption", "F4": "F4 loaded", "F5": "F5 forced"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def legend_families(ax, **kw):
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=5, label=FAM_LABEL[f]) for f, c in FAM.items()]
    ax.legend(handles=handles, frameon=False, handletextpad=0.2, borderaxespad=0.2, **kw)


# ---------- Figure 1: H6 ----------
rank = pd.read_csv(STATS / f"{JUDGE}_h6_ranking.csv")
press = pd.read_csv(STATS / f"{JUDGE}_pressure.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.15), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.42})
r = rank.loc[rank["outcome"] == "N_core"].sort_values("mean")
y = range(len(r))
for i, (_, row) in zip(y, r.iterrows()):
    c = FAM[row["family"]]
    a.plot([row["ci_low"], row["ci_high"]], [i, i], color=c, lw=2, solid_capstyle="round")
    a.plot(row["mean"], i, "o", color=c, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8)
    a.text(row["ci_high"] + 0.008, i, f"{row['mean']:.2f}", va="center", fontsize=6.5, color=INK2)
a.set_yticks(list(y)); a.set_yticklabels([SHORT[m] for m in r["model_key"]])
a.set_xlim(min(0.38, r["ci_low"].min() - 0.02), max(0.75, r["ci_high"].max() + 0.06)); a.set_xlabel("Mean neutrality N_core (95% CI)")
a.set_title("(a) RQ1: overall neutrality per model", loc="left")
grid(a, "x"); legend_families(a, loc="lower right", fontsize=6.5)

levels = ["F1-default", "F2-default", "F3", "F4"]
p = press.loc[press["pressure_level"].isin(levels)]
for m, g in p.groupby("model_key"):
    g = g.set_index("pressure_level").loc[levels]
    fam = g["family"].iloc[0]
    b.plot(range(4), g["mean_n_core"], color=FAM[fam], lw=1.1, alpha=0.55)
    b.plot(range(4), g["mean_n_core"], "o", color=FAM[fam], markersize=4, markeredgecolor="white", markeredgewidth=0.6)
f4 = p.loc[p["pressure_level"] == "F4"].sort_values("mean_n_core")
# Label every model at the F4 end; nudge labels apart so close values don't overlap.
ys = list(f4["mean_n_core"]); gap = 0.062
ys[0] = max(ys[0], 0.03)
for k in range(1, len(ys)):
    ys[k] = max(ys[k], ys[k - 1] + gap)
shift = max(0, ys[-1] - 0.97)
ys = [y - shift for y in ys]
for k in range(len(ys) - 2, -1, -1):
    ys[k] = min(ys[k], ys[k + 1] - gap)
for y_lab, (_, row) in zip(ys, f4.iterrows()):
    b.plot([3.05, 3.2], [row["mean_n_core"], y_lab], color=GRID, lw=0.6)
    b.text(3.24, y_lab, SHORT[row["model_key"]], va="center", fontsize=6.0, color=INK2)
spreads = [p.loc[p["pressure_level"] == lv, "mean_n_core"] for lv in levels]
b.set_xticks(range(4))
b.set_xticklabels([f"{LEVEL_LABEL[l]}\nspread {v.max() - v.min():.2f}" for l, v in zip(levels, spreads)], fontsize=6.6)
b.set_xlim(-0.45, 4.35); b.set_ylim(0, 1.0); b.set_ylabel("Mean N_core")
b.set_title("(b) Neutrality by framing: models split under pressure", loc="left")
grid(b)
fig.savefig(OUT / f"fig1_rq1{TAG}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 2: H4a, H4b ----------
cells = pd.read_csv(STATS / f"{JUDGE}_h4a_cells.csv")
pooled = pd.read_csv(STATS / f"{JUDGE}_h4b_pooled.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.0), gridspec_kw={"wspace": 0.3})
regions = ["china_implicated", "sea_implicated", "china_sea_overlap", "other"]
rlabel = ["China-\nimplicated", "SEA-\nimplicated", "China–SEA\noverlap", "Other"]
off = {"western": -0.17, "chinese": 0, "sea": 0.17}
for fam, g in cells.groupby("family"):
    g = g.set_index("analysis_region").loc[regions]
    x = [k + off[fam] for k in range(4)]
    a.plot(x, g["mean"], color=FAM[fam], lw=0.8, alpha=0.5)
    a.errorbar(x, g["mean"], yerr=1.96 * g["sem"], fmt="o", color=FAM[fam], markersize=4.5, elinewidth=1.2,
               capsize=0, markeredgecolor="white", markeredgewidth=0.6)
# Keep the original 0.72 top unless an interval reaches past it; then leave room for the legend.
hi = (cells["mean"] + 1.96 * cells["sem"]).max()
a.set_xticks(range(4)); a.set_xticklabels(rlabel); a.set_ylim(0.2, 0.72 if hi <= 0.68 else hi + 0.12)
a.set_ylabel("Mean N, uncued prompt (F2)")
a.set_title("(a) RQ2: neutrality by region and family", loc="left"); grid(a)
legend_families(a, loc="upper left", fontsize=6.5, ncol=3, columnspacing=0.8)

homes = ["chinese", "sea", "western"]
n_items = pooled.groupby("home_family")["n_items"].first()
hlabel = {"chinese": f"PRC items ({n_items['chinese']})\n(Chinese home)", "sea": f"SEA items ({n_items['sea']})\n(SEA home)",
          "western": f"Western items ({n_items['western']})\n(Western home)"}
for k, hf in enumerate(homes):
    g = pooled.loc[pooled["home_family"] == hf]
    for _, row in g.iterrows():
        dx = off[row["family"]]; own = bool(row["own"])
        # Side-picking answers only; interval clipped to the 0-1 range of a share.
        b.plot([k + dx, k + dx], [max(row["v_lo"], 0), min(row["v_hi"], 1)], color=FAM[row["family"]], lw=1.2)
        b.plot(k + dx, row["home_among_verdicts"], "o", markersize=5.5 if own else 4.5,
               color=FAM[row["family"]] if own else "white", markeredgecolor=FAM[row["family"]], markeredgewidth=1.1)
b.set_xticks(range(3)); b.set_xticklabels([hlabel[h] for h in homes]); b.set_ylim(0, 1.02)
b.set_ylabel("Home-side share of side-picks")
b.set_title("(b) RQ3: when a side is picked, is it home?", loc="left"); grid(b)
b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=INK2, markersize=5, label="home family"),
                  plt.Line2D([], [], marker="o", ls="", color="white", markeredgecolor=INK2, markersize=4.5, label="other families")],
         frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.2)
fig.savefig(OUT / f"fig2_rq2_rq3{TAG}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: lineage ----------
lin = pd.read_csv(STATS / f"{JUDGE}_lineage_by_level.csv")
contrasts = [("sea-lion-v3 minus llama-3.1-8b-instruct", "SEA-LION-v3 − Llama-3.1"),
             ("typhoon2 minus llama-3.1-8b-instruct", "Typhoon2 − Llama-3.1"),
             ("sea-lion-v4-vl minus qwen3-vl-8b-instruct", "SEA-LION-v4 − Qwen3-VL")]
cc = ["#6250d6", "#e87ba4", "#52514e"]
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0), gridspec_kw={"wspace": 0.38}, sharey=False)
for ax, outcome, lv, xl, title in (
    (axes[0], "N_core", ["F1-default", "F2-default", "F3", "F4"], "Δ neutrality N_core (regional − base)", "(a) RQ4: change in neutrality"),
    (axes[1], "verdict", ["F1-default", "F2-default", "F3", "F4", "F5"], "Δ side-picking rate (regional − base)", "(b) RQ4: change in side-picking"),
):
    lo = lin.loc[lin["outcome"] == outcome]
    for j, (name, lab) in enumerate(contrasts):
        g = lo.loc[lo["contrast"] == name].set_index("level").loc[lv]
        for k, (_, row) in enumerate(g.iterrows()):
            yi = k + (j - 1) * 0.24
            sig = True
            ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=cc[j], lw=1.3)
            ax.plot(row["mean_delta"], yi, "o", markersize=4.5, color=cc[j] if sig else "white",
                    markeredgecolor=cc[j], markeredgewidth=1.1, label=lab if k == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(lv))); ax.set_yticklabels([LEVEL_LABEL[l] for l in lv]); ax.set_ylim(4.5, -0.5)
    ax.set_xlim(-0.45, 0.45); ax.set_xlabel(xl); ax.set_title(title, loc="left"); grid(ax, "x")
axes[1].tick_params(axis="y", labelleft=True)
handles = [plt.Line2D([], [], marker="o", ls="", color=c_, markersize=4.5) for c_ in cc]
labels = [lab for _, lab in contrasts]
fig.legend(handles, labels, frameon=False, fontsize=6.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2)
fig.savefig(OUT / f"fig3_rq4{TAG}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")
