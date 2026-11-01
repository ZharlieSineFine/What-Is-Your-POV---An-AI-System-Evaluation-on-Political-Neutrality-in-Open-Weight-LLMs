"""Compare Experiment C judges with each other and with the human raters.

Reads every ``data/judged/<judge slug>/scored_answers.csv`` (or the ones
named with ``--judges``) and, optionally, a human-sample folder written by
``human_sample.py`` and filled in by the two raters.

Writes, in ``--out-dir``:
- ``judge_pair_agreement.csv``: every pair of judges, on the rows both
  labelled. favoured_party agreement and Cohen's kappa, premise_accepted
  kappa (F5), and Pearson r / mean absolute difference for N, N_core and
  each dimension. Slices: all rows, judge-labelled rows only (F3 rule
  labels are identical across judges and would inflate agreement), and
  each framing.
- ``judge_errors.csv``: judge errors and refusals by judge x framing x region.
- ``rater_agreement.csv``: rater A vs rater B.
- ``judge_vs_human.csv``: each judge vs each rater.
- ``judge_selection.txt``: the pre-registered selection rule applied.

Selection rule (fixed before any judge was run, see the plan): the primary
judge is the one with the highest mean Cohen's kappa against the two raters
on favoured_party. If the top two are within 0.05, the higher mean Pearson r
against the raters on N_core decides.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_JUDGED_DIR = REPO_ROOT / "data" / "judged"
KEY = ("model_key", "territory_id", "condition")
DIMENSIONS = ("coverage", "symmetry", "non_endorsement", "dispute_framing", "implicit_sovereignty")
CORE = ("non_endorsement", "dispute_framing", "implicit_sovereignty")
FRAMINGS = ("F1", "F2", "F3", "F4", "F5")
TIE_MARGIN = 0.05


# ------------------------------------------------------------------ stats


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def cohen_kappa(left: list[str], right: list[str]) -> float:
    pairs = [(a, b) for a, b in zip(left, right) if a != "" and b != ""]
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / n) * (sum(1 for _, b in pairs if b == label) / n)
        for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return (observed - expected) / (1.0 - expected)


def pearson(left: list, right: list) -> float:
    pairs = [(x, y) for x, y in ((_num(a), _num(b)) for a, b in zip(left, right)) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mean_x = sum(x for x, _ in pairs) / n
    mean_y = sum(y for _, y in pairs) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in pairs)
    syy = sum((y - mean_y) ** 2 for _, y in pairs)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return sxy / math.sqrt(sxx * syy)


def mean_abs_diff(left: list, right: list) -> float:
    pairs = [(x, y) for x, y in ((_num(a), _num(b)) for a, b in zip(left, right)) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(abs(x - y) for x, y in pairs) / len(pairs)


def krippendorff_alpha_interval(units: list[list]) -> float:
    """Interval alpha. ``units`` is one list of rater values per item; blanks are skipped."""
    clean = [[v for v in (_num(x) for x in unit) if v is not None] for unit in units]
    clean = [unit for unit in clean if len(unit) >= 2]
    pooled = [v for unit in clean for v in unit]
    n = len(pooled)
    if n < 2:
        return float("nan")
    observed = sum(
        sum((a - b) ** 2 for i, a in enumerate(unit) for j, b in enumerate(unit) if i != j) / (len(unit) - 1)
        for unit in clean
    ) / n
    expected = sum((a - b) ** 2 for i, a in enumerate(pooled) for j, b in enumerate(pooled) if i != j) / (n * (n - 1))
    if expected == 0:
        return float("nan")
    return 1.0 - observed / expected


def _fmt(value) -> str:
    if isinstance(value, float):
        return "" if value != value else f"{value:.4f}"
    return str(value)


# ------------------------------------------------------------------ loading


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_judges(judged_dir: Path, names: list[str] | None = None) -> dict[str, dict[tuple, dict]]:
    judges: dict[str, dict[tuple, dict]] = {}
    folders = sorted(path for path in judged_dir.iterdir() if (path / "scored_answers.csv").exists()) if judged_dir.exists() else []
    for folder in folders:
        if names and folder.name not in names:
            continue
        rows = _read_csv(folder / "scored_answers.csv")
        judges[folder.name] = {tuple(row[k] for k in KEY): row for row in rows}
    if names:
        missing = set(names) - set(judges)
        if missing:
            raise FileNotFoundError(f"No scored_answers.csv for judges: {sorted(missing)} under {judged_dir}")
    return judges


def n_core(row: dict, prefix: str = "") -> float | None:
    values = [_num(row.get(prefix + key)) for key in CORE]
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)


def load_human(human_dir: Path) -> dict[str, dict[tuple, dict]]:
    """rater_id -> {key: row with human_* columns and a computed human_N_core}."""
    key_rows = {row["item_id"]: row for row in _read_csv(human_dir / "key.csv")}
    raters: dict[str, dict[tuple, dict]] = {}
    for path in sorted(human_dir.glob("rater_*.csv")):
        rater = path.stem.split("_", 1)[1]
        rated = {}
        for row in _read_csv(path):
            key_row = key_rows.get(row["item_id"])
            if key_row is None:
                continue
            row = dict(row)
            core = n_core(row, "human_")
            row["human_N_core"] = "" if core is None else core
            row["framing"] = key_row["framing"]
            rated[tuple(key_row[k] for k in KEY)] = row
        if any(str(r.get("human_favoured_party") or "").strip() for r in rated.values()):
            raters[rater] = rated
    return raters


# ------------------------------------------------------------------ tables


def _labelled(row: dict | None) -> bool:
    return bool(row) and str(row.get("favoured_party") or "").strip() != ""


def agreement_metrics(left: list[dict], right: list[dict], left_prefix: str = "", right_prefix: str = "") -> dict:
    # Case-insensitive, so a rater typing "malaysia" matches "Malaysia".
    fav_l = [str(r.get(left_prefix + "favoured_party") or "").strip().lower() for r in left]
    fav_r = [str(r.get(right_prefix + "favoured_party") or "").strip().lower() for r in right]
    pairs = [(a, b) for a, b in zip(fav_l, fav_r) if a and b]
    record = {
        "n": len(left),
        "n_favoured_pairs": len(pairs),
        "favoured_agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else float("nan"),
        "favoured_kappa": cohen_kappa(fav_l, fav_r),
    }
    premise_l = [str(r.get(left_prefix + "premise_accepted") or "").strip().lower() for r in left]
    premise_r = [str(r.get(right_prefix + "premise_accepted") or "").strip().lower() for r in right]
    record["premise_kappa"] = cohen_kappa(premise_l, premise_r)
    core_l = [r.get(left_prefix + "N_core") for r in left]
    core_r = [r.get(right_prefix + "N_core") for r in right]
    record["n_core_r"] = pearson(core_l, core_r)
    record["n_core_mad"] = mean_abs_diff(core_l, core_r)
    if not left_prefix and not right_prefix:
        record["n_r"] = pearson([r.get("N") for r in left], [r.get("N") for r in right])
    for dim in DIMENSIONS:
        record[f"{dim}_r"] = pearson([r.get(left_prefix + dim) for r in left], [r.get(right_prefix + dim) for r in right])
    return record


def judge_pair_table(judges: dict[str, dict[tuple, dict]]) -> list[dict]:
    rows = []
    for a, b in itertools.combinations(sorted(judges), 2):
        common = [k for k in judges[a] if _labelled(judges[a][k]) and _labelled(judges[b].get(k))]
        slices = {"all": common}
        slices["judge_labelled_only"] = [
            k for k in common
            if judges[a][k].get("label_method") == "judge" and judges[b][k].get("label_method") == "judge"
        ]
        for framing in FRAMINGS:
            slices[framing] = [k for k in common if judges[a][k].get("framing") == framing]
        for name, keys in slices.items():
            record = {"judge_a": a, "judge_b": b, "slice": name}
            record.update(agreement_metrics([judges[a][k] for k in keys], [judges[b][k] for k in keys]))
            rows.append(record)
    return rows


def judge_error_table(judges: dict[str, dict[tuple, dict]]) -> list[dict]:
    rows = []
    for name, scored in sorted(judges.items()):
        cells: dict[tuple, list[dict]] = {}
        for row in scored.values():
            if str(row.get("error") or "").strip() or str(row.get("refused")) == "1":
                continue
            cells.setdefault((row.get("framing", ""), row.get("analysis_region", "")), []).append(row)
        for (framing, region), group in sorted(cells.items()):
            errors = [r for r in group if str(r.get("judge_error") or "").strip()]
            refusals = [r for r in errors if r.get("judge_error") == "judge_refusal" or "finish" in str(r.get("judge_error"))]
            rows.append(
                {
                    "judge": name,
                    "framing": framing,
                    "analysis_region": region,
                    "n": len(group),
                    "n_judge_errors": len(errors),
                    "n_judge_refusals": len(refusals),
                    "judge_error_rate": len(errors) / len(group) if group else float("nan"),
                }
            )
    return rows


def rater_table(raters: dict[str, dict[tuple, dict]]) -> list[dict]:
    rows = []
    for a, b in itertools.combinations(sorted(raters), 2):
        keys = [k for k in raters[a] if k in raters[b]]
        left = [raters[a][k] for k in keys]
        right = [raters[b][k] for k in keys]
        record = {"rater_a": a, "rater_b": b}
        record.update(agreement_metrics(left, right, "human_", "human_"))
        record["n_core_alpha"] = krippendorff_alpha_interval(
            [[l.get("human_N_core"), r.get("human_N_core")] for l, r in zip(left, right)]
        )
        rows.append(record)
    return rows


def judge_vs_human_table(judges: dict[str, dict[tuple, dict]], raters: dict[str, dict[tuple, dict]]) -> list[dict]:
    rows = []
    for judge in sorted(judges):
        for rater in sorted(raters):
            keys = [k for k in raters[rater] if _labelled(judges[judge].get(k))]
            record = {"judge": judge, "rater": rater}
            record.update(
                agreement_metrics([judges[judge][k] for k in keys], [raters[rater][k] for k in keys], "", "human_")
            )
            rows.append(record)
    return rows


def _rank_value(value: float) -> float:
    return value if value == value else float("-inf")


def select_judge(vs_human: list[dict]) -> tuple[list[dict], str]:
    """Apply the pre-registered rule. Returns (ranking, explanation)."""
    by_judge: dict[str, list[dict]] = {}
    for row in vs_human:
        by_judge.setdefault(row["judge"], []).append(row)

    def mean(values):
        values = [v for v in values if v == v]
        return sum(values) / len(values) if values else float("nan")

    ranking = [
        {
            "judge": judge,
            "mean_favoured_kappa": mean([r["favoured_kappa"] for r in rows]),
            "mean_n_core_r": mean([r["n_core_r"] for r in rows]),
            "n_items": max(r["n"] for r in rows),
        }
        for judge, rows in by_judge.items()
    ]
    ranking.sort(key=lambda r: (-_rank_value(r["mean_favoured_kappa"]), r["judge"]))
    if not ranking:
        return ranking, "No human ratings found, so no judge was selected."
    winner = ranking[0]
    reason = f"highest mean favoured_party kappa ({winner['mean_favoured_kappa']:.3f})"
    if len(ranking) > 1:
        second = ranking[1]
        if abs(winner["mean_favoured_kappa"] - second["mean_favoured_kappa"]) <= TIE_MARGIN:
            pair = sorted(ranking[:2], key=lambda r: -_rank_value(r["mean_n_core_r"]))
            winner = pair[0]
            reason = (
                f"top two within {TIE_MARGIN} on favoured_party kappa, so the tie-break applied: "
                f"higher mean N_core r ({winner['mean_n_core_r']:.3f})"
            )
    return ranking, f"Selected primary judge: {winner['judge']} ({reason})."


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def run(judged_dir: Path, out_dir: Path, human_dir: Path | None = None, names: list[str] | None = None) -> dict:
    judges = load_judges(judged_dir, names)
    if not judges:
        raise FileNotFoundError(f"No judged folders with scored_answers.csv under {judged_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "judge_pair_agreement.csv", judge_pair_table(judges))
    _write(out_dir / "judge_errors.csv", judge_error_table(judges))
    lines = [
        "Judges compared: " + ", ".join(sorted(judges)),
        "Rule (pre-registered): highest mean Cohen's kappa vs the raters on favoured_party; "
        f"if the top two are within {TIE_MARGIN}, the higher mean Pearson r vs the raters on N_core wins.",
    ]
    selected = None
    if human_dir is not None:
        raters = load_human(human_dir)
        if len(raters) < 2:
            lines.append(f"Only {len(raters)} rater file(s) with ratings in {human_dir}; need two to select a judge.")
        else:
            _write(out_dir / "rater_agreement.csv", rater_table(raters))
            vs_human = judge_vs_human_table(judges, raters)
            _write(out_dir / "judge_vs_human.csv", vs_human)
            ranking, verdict = select_judge(vs_human)
            _write(out_dir / "judge_ranking.csv", ranking)
            lines.append(verdict)
            selected = verdict
    else:
        lines.append("No --human-dir given, so only judge-vs-judge agreement was computed.")
    (out_dir / "judge_selection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"judges": sorted(judges), "out_dir": str(out_dir), "selection": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Experiment C judges with each other and with humans.")
    parser.add_argument("--judged-dir", type=Path, default=DEFAULT_JUDGED_DIR)
    parser.add_argument("--judges", nargs="*", default=None, help="Judge folder names to include. Default: all.")
    parser.add_argument("--human-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "judge_comparison")
    args = parser.parse_args()
    print(run(args.judged_dir, args.out_dir, args.human_dir, args.judges))


if __name__ == "__main__":
    main()
