"""Draw the blind human-rating sample for Experiment C.

The sample comes straight from the generations, not from any judge's
output, so every judge is compared against the same human-coded items.

Writes, in ``--out-dir``:
- ``rater_A.csv`` and ``rater_B.csv``: one row per item, in a shuffled
  order, with the prompt, the answer and the gold parties, and empty
  ``human_*`` columns. No model name, family or judge score appears here.
- ``key.csv``: item_id -> model_key, territory_id, condition, region. The
  raters must not open this file. ``compare_judges.py`` uses it.

Items are stratified across framing x analysis region (round robin, seed
612). Generation errors and blanket refusals are left out. Existing rater
files that already contain ratings are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from experiment_c.judge import DEFAULT_PROMPTS, load_generations, load_territories, row_framing, should_call_judge

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TERRITORIES = REPO_ROOT / "data" / "territories.csv"
DEFAULT_N = 40
DEFAULT_SEED = 612
RATERS = ("A", "B")

HUMAN_COLUMNS = [
    "human_coverage",
    "human_symmetry",
    "human_non_endorsement",
    "human_dispute_framing",
    "human_implicit_sovereignty",
    "human_favoured_party",
    "human_premise_accepted",
    "human_notes",
]
RATER_FIELDS = [
    "item_id",
    "rater_id",
    "Territory",
    "framing",
    "target_party",
    "gold_parties_cleaned",
    "prompt",
    "response",
    *HUMAN_COLUMNS,
]
KEY_FIELDS = ["item_id", "model_key", "territory_id", "condition", "framing", "analysis_region"]


def eligible_items(generations: list[dict], territories: dict[str, dict]) -> list[dict]:
    items = []
    for row in generations:
        territory = territories.get(str(row.get("territory_id")))
        if territory is None or not should_call_judge(row):
            continue
        items.append(
            {
                "model_key": row["model_key"],
                "territory_id": row["territory_id"],
                "condition": row["condition"],
                "framing": row_framing(row),
                "target_party": row.get("target_party", ""),
                "analysis_region": territory["analysis_region"],
                "Territory": territory["Territory"],
                "gold_parties_cleaned": territory["gold_parties_cleaned"],
                "prompt": row.get("prompt", ""),
                "response": row.get("response", ""),
            }
        )
    return items


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin sample across framing x analysis_region."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = {}
    for row in sorted(rows, key=lambda r: (r["model_key"], r["territory_id"], r["condition"])):
        groups.setdefault((row["framing"], row["analysis_region"]), []).append(row)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    chosen: list[dict] = []
    cursors = {key: 0 for key in keys}
    while len(chosen) < n:
        moved = False
        for key in keys:
            index = cursors[key]
            if index < len(groups[key]) and len(chosen) < n:
                chosen.append(groups[key][index])
                cursors[key] = index + 1
                moved = True
        if not moved:
            break
    return chosen


def _has_ratings(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if any(str(row.get(column) or "").strip() for column in HUMAN_COLUMNS):
                return True
    return False


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_sample(
    generations_path: Path,
    territories_path: Path,
    out_dir: Path,
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    prompts_path: Path | None = None,
) -> dict:
    for rater in RATERS:
        if _has_ratings(out_dir / f"rater_{rater}.csv"):
            raise FileExistsError(
                f"{out_dir / f'rater_{rater}.csv'} already has ratings. "
                "Use a new --out-dir instead of overwriting human work."
            )
    items = eligible_items(load_generations(generations_path, prompts_path), load_territories(territories_path))
    sample = stratified_sample(items, n, seed)
    # Shuffle the presentation order, so neighbouring rows don't reveal the strata.
    random.Random(seed + 1).shuffle(sample)
    for index, item in enumerate(sample, start=1):
        item["item_id"] = f"H{index:03d}"
    for rater in RATERS:
        _write(
            out_dir / f"rater_{rater}.csv",
            RATER_FIELDS,
            [dict(item, rater_id=rater, **{column: "" for column in HUMAN_COLUMNS}) for item in sample],
        )
    _write(out_dir / "key.csv", KEY_FIELDS, sample)
    cells = {(item["framing"], item["analysis_region"]) for item in sample}
    return {"n_items": len(sample), "n_eligible": len(items), "n_cells": len(cells), "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the blind human-rating sample for Experiment C.")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--territories", type=Path, default=DEFAULT_TERRITORIES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(
        build_sample(
            args.generations, args.territories, args.out_dir, n=args.n, seed=args.seed, prompts_path=args.prompts
        )
    )


if __name__ == "__main__":
    main()
