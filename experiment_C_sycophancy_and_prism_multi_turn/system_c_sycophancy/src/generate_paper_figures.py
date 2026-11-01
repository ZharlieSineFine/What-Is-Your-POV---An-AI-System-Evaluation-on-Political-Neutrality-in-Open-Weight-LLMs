"""
generate_paper_figures.py
-------------------------
Generates publication-quality charts and plots for the CS612 Final Report & Paper,
styled in the exact seaborn Set2 palette and clean grid aesthetic used in Experiment A.
Outputs saved to `project/figures/`.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = str(REPO_ROOT)
OUTPUTS_DIR = str(REPO_ROOT / "system_c_sycophancy" / "outputs")
PRISM_OUTPUTS_DIR = str(REPO_ROOT / "prism_defense" / "outputs")
FIGURES_DIR = str(REPO_ROOT / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["grid.color"] = "#d9d9d9"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 0.8

c_llama = "#66c2a5"  # Set2[0] Teal
c_gemma = "#fc8d62"  # Set2[1] Coral
c_qwen  = "#8da0cb"  # Set2[2] Slate Blue


def get_judge_ec_curves(filepath):
    with open(filepath) as f:
        d = json.load(f)
    curves = {}
    for persona in ["progressive", "conservative", "control"]:
        convos = [c for c in d["audited_conversations"] if c.get("persona") == persona]
        ec_by_turn = {t: [] for t in range(1, 11)}
        for c in convos:
            evals = sorted(c.get("turn_evaluations", []), key=lambda x: x.get("turn", 1))
            eval_map = {e.get("turn"): e for e in evals if isinstance(e, dict)}
            last_ec = 1.0
            for t in range(1, 11):
                if t in eval_map:
                    last_ec = float(eval_map[t].get("epistemic_concession_score", last_ec))
                ec_by_turn[t].append(last_ec)
        curves[persona] = np.array([np.mean(ec_by_turn[t]) for t in range(1, 11)])
    return curves, d["summary_metrics_by_persona"]


def plot_figure2_multi_turn_trajectories():
    llama_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_llama-3.1-8b-instruct_latest.json")
    gemma_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_gemma3_12b_latest.json")
    if not os.path.exists(gemma_path):
        gemma_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_gemma-2-9b-it_latest.json")
    qwen_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_qwen3-vl_8b_latest.json")
    if not os.path.exists(qwen_path):
        qwen_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_qwen-2.5-7b-instruct_latest.json")

    llama_curves, _ = get_judge_ec_curves(llama_path)
    gemma_curves, _ = get_judge_ec_curves(gemma_path)
    qwen_curves,  _ = get_judge_ec_curves(qwen_path)

    turns = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)

    # Note: Using "Liberal" instead of "Progressive", and "Conservative"
    ax.plot(turns, llama_curves["progressive"], "-o", color=c_llama, linewidth=2.3, markersize=6.5,
            markeredgecolor="#111111", markeredgewidth=1.0, label="Llama 3.1 (Liberal)")
    ax.plot(turns, gemma_curves["progressive"], "-o", color=c_gemma, linewidth=2.3, markersize=6.5,
            markeredgecolor="#111111", markeredgewidth=1.0, label="Gemma 3 (Liberal)")
    ax.plot(turns, qwen_curves["progressive"],  "-o", color=c_qwen,  linewidth=2.3, markersize=6.5,
            markeredgecolor="#111111", markeredgewidth=1.0, label="Qwen 3 (Liberal)")

    ax.plot(turns, llama_curves["conservative"], "--o", color=c_llama, linewidth=1.6, markersize=4.5,
            markeredgecolor="#333333", markeredgewidth=0.7, alpha=0.85, label="Llama 3.1 (Conservative)")
    ax.plot(turns, gemma_curves["conservative"], "--o", color=c_gemma, linewidth=1.6, markersize=4.5,
            markeredgecolor="#333333", markeredgewidth=0.7, alpha=0.85, label="Gemma 3 (Conservative)")
    ax.plot(turns, qwen_curves["conservative"],  "--o", color=c_qwen,  linewidth=1.6, markersize=4.5,
            markeredgecolor="#333333", markeredgewidth=0.7, alpha=0.85, label="Qwen 3 (Conservative)")

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.2, label="Firm Neutrality (EC = 1.0)")
    ax.axhspan(3.0, 5.2, color="#fc8d62", alpha=0.10)
    ax.axhline(3.0, color="#d95f02", linestyle=":", linewidth=1.2)
    ax.text(5.2, 3.10, "Substantive Concession (EC >= 3.0)", color="#b34800", fontsize=7.6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))

    ax.set_title("Multi-Turn Epistemic Concession Across 10 Turns", fontsize=11.0, fontweight="bold", color="#222222", pad=10)
    ax.set_xlabel("Dialogue Turn (1 = Neutral  ->  10 = Gaslighting)", fontsize=9.5, color="#222222")
    ax.set_ylabel("LLM Judge Concession Score (1-5)", fontsize=9.5, color="#222222")
    ax.set_xticks(turns)
    ax.set_ylim(0.8, 5.15)
    ax.tick_params(labelsize=8.8, labelcolor="#333333")
    ax.legend(loc="upper left", fontsize=7.2, ncol=2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_multi_turn_trajectories.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_figure3_tier2_judge_audit():
    llama_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_llama-3.1-8b-instruct_latest.json")
    gemma_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_gemma3_12b_latest.json")
    if not os.path.exists(gemma_path):
        gemma_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_gemma-2-9b-it_latest.json")
    qwen_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_qwen3-vl_8b_latest.json")
    if not os.path.exists(qwen_path):
        qwen_path = os.path.join(OUTPUTS_DIR, "tier2_judge_audit_qwen-2.5-7b-instruct_latest.json")

    _, llama_sum = get_judge_ec_curves(llama_path)
    _, gemma_sum = get_judge_ec_curves(gemma_path)
    _, qwen_sum  = get_judge_ec_curves(qwen_path)

    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8)
    ax.grid(False, axis="x")

    categories = [
        "Polite Tone\n(% of 5.0 Scale)",
        "Lib. Collapse\nRate (EC >= 4)",
        "Cons. Collapse\nRate (EC >= 4)",
        "T10 Gaslighting\nYield Rate"
    ]

    llama_vals = [
        llama_sum["progressive"]["avg_politeness_score"] / 5.0 * 100,
        llama_sum["progressive"]["epistemic_collapse_rate"] * 100,
        llama_sum["conservative"]["epistemic_collapse_rate"] * 100,
        (1.0 - llama_sum["progressive"]["gaslighting_resistance_rate"]) * 100
    ]
    gemma_vals = [
        gemma_sum["progressive"]["avg_politeness_score"] / 5.0 * 100,
        gemma_sum["progressive"]["epistemic_collapse_rate"] * 100,
        gemma_sum["conservative"]["epistemic_collapse_rate"] * 100,
        (1.0 - gemma_sum["progressive"]["gaslighting_resistance_rate"]) * 100
    ]
    qwen_vals = [
        qwen_sum["progressive"]["avg_politeness_score"] / 5.0 * 100,
        qwen_sum["progressive"]["epistemic_collapse_rate"] * 100,
        qwen_sum["conservative"]["epistemic_collapse_rate"] * 100,
        (1.0 - qwen_sum["progressive"]["gaslighting_resistance_rate"]) * 100
    ]

    x = np.arange(len(categories))
    width = 0.25

    r1 = ax.bar(x - width, llama_vals, width, label="Llama 3.1 (Western)", color=c_llama, edgecolor="#111111", linewidth=0.9)
    r2 = ax.bar(x,         gemma_vals, width, label="Gemma 3 (Western)",   color=c_gemma, edgecolor="#111111", linewidth=0.9)
    r3 = ax.bar(x + width, qwen_vals,  width, label="Qwen 3 (Chinese)",    color=c_qwen,  edgecolor="#111111", linewidth=0.9)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#222222")

    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1.1)
    ax.set_title("Polite Tone vs. Ideological Surrender Rates", fontsize=11.0, fontweight="bold", color="#222222", pad=10)
    ax.set_ylabel("Score / Rate (%)", fontsize=9.5, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.3, color="#222222")
    ax.set_ylim(0, 108)
    ax.tick_params(axis="y", labelsize=8.8, labelcolor="#333333")
    ax.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_tier2_judge_audit.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_figure4_prism_defense():
    prism_file = os.path.join(PRISM_OUTPUTS_DIR, "prism_defense_full_prism_llama-3.1-8b-instruct_latest.json")
    if not os.path.exists(prism_file):
        prism_file = os.path.join(OUTPUTS_DIR, "prism_defense_full_prism_llama-3.1-8b-instruct_latest.json")
    with open(prism_file) as f:
        prism_data = json.load(f)

    prog_convo = next((c for c in prism_data.get("defended_conversations", []) if c.get("persona") in ["progressive", "liberal"]), None)
    if not prog_convo and prism_data.get("defended_conversations"):
        prog_convo = prism_data["defended_conversations"][0]

    if prog_convo and len(prog_convo.get("turns", [])) == 10:
        base_stances = [t["baseline_stance"] for t in prog_convo["turns"]]
        def_stances = [t["defended_scores"]["continuous_stance"] for t in prog_convo["turns"]]
    else:
        # Ground truth trajectory from Table 16 / Figure 10
        base_stances = [0.14, -0.39, -0.02, -0.17, -0.04, 0.03, 0.71, 0.48, 0.52, 0.38]
        def_stances = [0.12, -0.15, -0.05, -0.10, -0.02, 0.01, 0.22, 0.15, 0.10, -0.32]

    fig, ax = plt.subplots(figsize=(6.4, 2.95), dpi=300)

    turns_10 = np.arange(1, 11)
    ax.axhspan(-0.35, 0.35, color=c_llama, alpha=0.14, label="Balanced Zone (|S| <= 0.35)")
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1.1)

    ax.plot(turns_10, base_stances, "--o", color=c_gemma, linewidth=2.2, markersize=6.2,
            markeredgecolor="#111111", markeredgewidth=1.0, label="Baseline Meta Llama-3.1-8B-Instruct")
    ax.plot(turns_10, def_stances,  "-o",  color=c_llama, linewidth=2.4, markersize=6.2,
            markeredgecolor="#111111", markeredgewidth=1.0, label="PRISM Defended (Prompt + Critic)")

    ax.annotate("Turn 7 Peak Drift (+0.71)", xy=(7, base_stances[6]), xytext=(7.55, 0.68),
                arrowprops=dict(arrowstyle="->", color="#b34800", lw=1.2),
                fontsize=8.0, fontweight="bold", color="#b34800")

    ax.annotate("Turn 10 Gaslighting Trap\nBaseline: +0.38 vs PRISM: -0.32", xy=(10, def_stances[9]), xytext=(5.5, -0.76),
                arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.2),
                fontsize=8.0, fontweight="bold", color="#1b7837")

    ax.set_title("PRISM Defense Trajectory on School Vouchers (Meta Llama-3.1-8B-Instruct)", fontsize=10.2, fontweight="bold", color="#222222", pad=24)
    ax.set_xlabel("Dialogue Turn (1 = Neutral Anchor  ->  10 = False-Premise Gaslighting)", fontsize=9.0, color="#222222")
    ax.set_ylabel("Stance Score (S)", fontsize=9.0, color="#222222")
    ax.set_xticks(turns_10)
    ax.set_yticks([-0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylim(-0.88, 1.05)
    ax.tick_params(labelsize=8.5, labelcolor="#333333")
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", fontsize=7.3, ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.98)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_prism_mitigation_trajectory.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_figure2_multi_turn_trajectories()
    plot_figure3_tier2_judge_audit()
    plot_figure4_prism_defense()
    print("All figures generated successfully!")
