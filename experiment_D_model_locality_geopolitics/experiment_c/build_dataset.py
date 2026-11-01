"""Build Experiment C v2 tables from BorderLines.

Writes:
- territories.csv: all 251 territories, with ``core_set`` and ``pilot`` flags
- prompts.csv: the F1-F5 prompt ladder for the 172-territory core set
- prompts_pilot.csv: the same rows for the 20 pilot territories only

Core set (decision D1): every China-implicated, SEA-implicated and overlap
territory, plus one territory per claimant cluster from Other. The Other
representative is the most populous territory in its cluster (ties broken
alphabetically), fixed before any answers are seen. The pilot is 5 core
territories per region, drawn with seed 612.

F1, F2 and F4 prompts contain the territory name only. F3 and F5 name the
gold parties, as their design requires.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from experiment_c.coding import (
    PILOT_PER_REGION,
    PILOT_SEED,
    REGION_OTHER,
    REGIONS,
    assert_roc_keep_present,
    analysis_region,
    clean_parties,
    cluster_id,
    prompt_rows,
    split_claimants,
    western_salient,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "borderlines" / "disputed_territories.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "data"

TERRITORY_FIELDS = [
    "territory_id",
    "Territory",
    "Claimants",
    "Controller",
    "analysis_region",
    "cluster_id",
    "gold_parties_cleaned",
    "western_salient",
    "Population",
    "core_set",
    "pilot",
]

PROMPT_FIELDS = [
    "query_id",
    "territory_id",
    "Territory",
    "condition",
    "framing",
    "instruction",
    "target_party",
    "option_order",
    "max_new_tokens",
    "prompt",
]


def load_borderlines(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    required = {"Territory", "Claimants", "Controller", "QueryID"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def _population(raw: str) -> int:
    try:
        return int(float(str(raw).replace(",", "").strip() or 0))
    except ValueError:
        return 0


def mark_core_set(territories: list[dict]) -> None:
    """Keep every non-Other territory, and one per cluster inside Other."""
    best: dict[str, dict] = {}
    for row in territories:
        if row["analysis_region"] != REGION_OTHER:
            row["core_set"] = 1
            continue
        row["core_set"] = 0
        current = best.get(row["cluster_id"])
        key = (-row["Population"], row["Territory"])
        if current is None or key < (-current["Population"], current["Territory"]):
            best[row["cluster_id"]] = row
    for row in best.values():
        row["core_set"] = 1


def mark_pilot(territories: list[dict], per_region: int = PILOT_PER_REGION, seed: int = PILOT_SEED) -> None:
    import random

    rng = random.Random(seed)
    for row in territories:
        row["pilot"] = 0
    for region in REGIONS:
        pool = sorted(
            (row for row in territories if row["core_set"] and row["analysis_region"] == region),
            key=lambda row: row["territory_id"],
        )
        for row in rng.sample(pool, min(per_region, len(pool))):
            row["pilot"] = 1


def build_tables(source_rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Return (all 251 territories, prompts for the core set)."""
    names = {row["Territory"] for row in source_rows}
    if len(names) != len(source_rows):
        raise ValueError("Territory names are not unique; refusing to build ids from them.")
    assert_roc_keep_present(names)

    territories: list[dict] = []
    seen_ids: set[str] = set()

    for row in source_rows:
        territory_id = row["QueryID"].strip()
        if not territory_id or territory_id in seen_ids:
            raise ValueError(f"Bad or duplicate QueryID: {territory_id!r}")
        seen_ids.add(territory_id)

        territory = row["Territory"].strip()
        claimants = split_claimants(row["Claimants"])
        if len(claimants) < 2:
            raise ValueError(f"{territory} has fewer than two claimants: {claimants}")

        region = analysis_region(claimants)
        parties = clean_parties(territory, claimants)
        territories.append(
            {
                "territory_id": territory_id,
                "Territory": territory,
                "Claimants": ";".join(claimants),
                "Controller": row["Controller"].strip(),
                "analysis_region": region,
                "cluster_id": cluster_id(claimants),
                "gold_parties_cleaned": ";".join(parties),
                "western_salient": int(western_salient(region, claimants)),
                "Population": _population(row.get("Population", "")),
            }
        )

    mark_core_set(territories)
    mark_pilot(territories)

    prompts: list[dict] = []
    for record in territories:
        if not record["core_set"]:
            continue
        parties = split_claimants(record["gold_parties_cleaned"])
        for item in prompt_rows(record["Territory"], parties):
            prompts.append(
                {
                    "query_id": f"{record['territory_id']}|{item['condition']}",
                    "territory_id": record["territory_id"],
                    "Territory": record["Territory"],
                    **item,
                }
            )
    return territories, prompts


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary(territories: list[dict], prompts: list[dict]) -> dict:
    core = [row for row in territories if row["core_set"]]
    regions = Counter(row["analysis_region"] for row in territories)
    core_regions = Counter(row["analysis_region"] for row in core)
    clusters = Counter(row["cluster_id"] for row in territories)
    core_clusters: dict[str, set] = {}
    for row in core:
        core_clusters.setdefault(row["analysis_region"], set()).add(row["cluster_id"])
    multi = sum(1 for count in clusters.values() if count >= 2)
    roc_kept = 0
    roc_dropped = 0
    for row in territories:
        raw = set(split_claimants(row["Claimants"]))
        gold = set(split_claimants(row["gold_parties_cleaned"]))
        if "Republic of China" in raw and "Republic of China" in gold:
            roc_kept += 1
        elif "Republic of China" in raw:
            roc_dropped += 1
    return {
        "n_territories": len(territories),
        "n_core_territories": len(core),
        "n_pilot_territories": sum(row["pilot"] for row in territories),
        "n_prompts": len(prompts),
        "n_prompts_by_framing": dict(sorted(Counter(row["framing"] for row in prompts).items())),
        "regions": dict(regions),
        "core_regions": dict(core_regions),
        "core_clusters_by_region": {key: len(value) for key, value in sorted(core_clusters.items())},
        "n_clusters": len(clusters),
        "n_multi_territory_clusters": multi,
        "n_western_salient": sum(row["western_salient"] for row in territories),
        "n_western_salient_core": sum(row["western_salient"] for row in core),
        "roc_kept_on_gold_list": roc_kept,
        "roc_dropped_from_gold_list": roc_dropped,
    }


def build(source: Path, out_dir: Path) -> dict:
    territories, prompts = build_tables(load_borderlines(source))
    write_csv(out_dir / "territories.csv", TERRITORY_FIELDS, territories)
    write_csv(out_dir / "prompts.csv", PROMPT_FIELDS, prompts)
    pilot_ids = {row["territory_id"] for row in territories if row["pilot"]}
    pilot = [row for row in prompts if row["territory_id"] in pilot_ids]
    write_csv(out_dir / "prompts_pilot.csv", PROMPT_FIELDS, pilot)
    payload = summary(territories, prompts)
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Experiment C v2 territories and prompts.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build(args.source, args.out_dir)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_dir / 'territories.csv'}")
    print(f"Wrote {args.out_dir / 'prompts.csv'}")
    print(f"Wrote {args.out_dir / 'prompts_pilot.csv'}")


if __name__ == "__main__":
    main()
