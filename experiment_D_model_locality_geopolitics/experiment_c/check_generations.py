"""Check copied generation files before judging.

For each of the seven models, reports how many answers in
``data/generations/<model>.jsonl`` were made with the current prompts file
(same prompt text, token cap and batch size 1), plus truncation, refusals,
GPU hours and the environment they were generated in. Exits with code 1 if
any model is incomplete or if the answers come from more than one
environment, so a gap is noticed before money is spent on judging.

    python -m experiment_c.check_generations --prompts experiment_c/data/prompts_pilot.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from experiment_c.generate_local import completed_records, current_keys, load_prompts
from experiment_c.models import MODELS

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GENERATIONS = REPO_ROOT / "data" / "generations"


def check(generations_dir: Path, prompts_path: Path) -> tuple[list[dict], bool, Counter]:
    prompts = load_prompts(prompts_path)
    rows = []
    environments: Counter = Counter()
    ok = True
    for spec in MODELS:
        path = generations_dir / f"{spec.model_key}.jsonl"
        if not path.exists():
            rows.append({"model": spec.model_key, "status": "MISSING FILE"})
            ok = False
            continue
        records = completed_records(path, spec.model_key)
        current = [records[key] for key in current_keys(records, prompts, 1, 2048)]
        for record in current:
            environments[(record.get("gpu"), record.get("torch_version"), record.get("transformers_version"))] += 1
        complete = len(current) == len(prompts)
        ok = ok and complete
        rows.append(
            {
                "model": spec.model_key,
                "status": "ok" if complete else "INCOMPLETE",
                "current": f"{len(current)}/{len(prompts)}",
                "truncated": sum(bool(r.get("truncated")) for r in current),
                "refused": sum(bool(r.get("refused")) for r in current),
                "gpu_hours": round(sum(r.get("gen_seconds") or 0 for r in current) / 3600, 2),
            }
        )
    if len(environments) > 1:
        ok = False
    return rows, ok, environments


def main() -> None:
    parser = argparse.ArgumentParser(description="Check copied generation files before judging.")
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()
    rows, ok, environments = check(args.generations, args.prompts)
    for row in rows:
        print("  ".join(f"{k}={v}" for k, v in row.items()))
    print("environments:", dict(environments))
    if len(environments) > 1:
        print("WARNING: answers come from more than one GPU/library environment; they are not directly comparable.")
    print("ALL COMPLETE" if ok else "NOT READY: fix the rows above before judging.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
