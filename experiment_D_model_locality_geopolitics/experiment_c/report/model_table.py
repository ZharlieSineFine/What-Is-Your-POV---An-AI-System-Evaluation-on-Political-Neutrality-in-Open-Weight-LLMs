"""Per-model table for the report: N on F2, N_core by framing, side-picking, refusal (no balance-instruction rows)."""
import importlib.util, sys
from pathlib import Path
import pandas as pd
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("an", ROOT / "experiment_c" / "analyze.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
for judge in ("openai__gpt-4.1-mini", "deepseek__deepseek-chat-v3-0324"):
    f = an.load_scored(an.long_path_safe(ROOT / "experiment_c/data/judged" / judge / "scored_answers.csv"))
    f = f.loc[f["instruction"] != "balance"].copy()
    if "--two-claimants" in sys.argv[3:]:
        f = an.two_claimant_rows(f)
    f = an.add_direction_columns(f)
    ok = an.completed_mask(f)
    rows = []
    for m, g in f.groupby("model_key"):
        f2 = g.loc[an.scored_mask(g) & (g["condition"] == "F2-default"), "N"]
        rows.append({"model_key": m, "family": g["family"].iloc[0], "N_F2": f2.mean(),
                     "refusal_rate": g.loc[ok.loc[g.index], "refused"].mean(), "n_refused": int(g["refused"].sum()),
                     "side_picking_all": g["verdict"].mean()})
    pd.DataFrame(rows).to_csv(OUT / f"{judge}_model_table.csv", index=False)
    print(judge); print(pd.DataFrame(rows).round(4).to_string(index=False))
