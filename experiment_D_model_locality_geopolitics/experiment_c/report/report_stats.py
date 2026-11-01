"""Experiment C report stats: H6, H4a, H4b, lineage, with balance-instruction rows removed."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("an", ROOT / "experiment_c" / "analyze.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

JUDGES = ["openai__gpt-4.1-mini", "deepseek__deepseek-chat-v3-0324"]
# --two-claimants drops the 17 territories left with one claimant once the dormant ROC claim is removed.
TWO_CLAIMANTS = "--two-claimants" in sys.argv[3:]
res = {}
for judge in JUDGES:
    path = an.long_path_safe(ROOT / "experiment_c" / "data" / "judged" / judge / "scored_answers.csv")
    f = an.load_scored(path)
    r = {"instruction_values": f["instruction"].value_counts().to_dict(), "rows_all": len(f)}
    f = f.loc[f["instruction"] != "balance"].copy()
    r["rows_no_balance"] = len(f)
    if TWO_CLAIMANTS:
        f = an.two_claimant_rows(f)
        r["rows_two_claimants"] = len(f)
    r["pressure_counts"] = f["pressure_level"].value_counts().to_dict()
    f = an.add_direction_columns(f)

    # H6
    tests, ranking = an.model_difference_tests(f)
    tests.to_csv(OUT / f"{judge}_h6_tests.csv", index=False)
    ranking.to_csv(OUT / f"{judge}_h6_ranking.csv", index=False)
    pt = an.pressure_table(f); pt.to_csv(OUT / f"{judge}_pressure.csv", index=False)

    # H4a on F2-default N
    d = f.loc[an.scored_mask(f) & (f["condition"] == "F2-default")].copy()
    fit = smf.ols(an.DEFAULT_SLICE_FORMULA, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["cluster_id"]})
    inter = [t for t in fit.params.index if ":" in t]
    w = fit.wald_test(", ".join(f"{t} = 0" for t in []) if False else np.eye(len(fit.params))[[list(fit.params.index).index(t) for t in inter]], scalar=True)
    r["h4a"] = {"n": len(d), "clusters": int(d["cluster_id"].nunique()), "wald_chi2": float(w.statistic), "df": len(inter), "p": float(w.pvalue),
                "terms": {t: [float(fit.params[t]), float(fit.pvalues[t])] for t in inter}}
    # same joint test on N_core across all open framings (robustness)
    dc = f.loc[an.scored_mask(f, "N_core") & f["pressure_level"].isin(["F1-default", "F2-default", "F3", "F4"])].copy()
    fc = smf.ols("N_core ~ C(analysis_region, Treatment(reference='other')) * C(family, Treatment(reference='western')) + C(pressure_level)", data=dc).fit(
        cov_type="cluster", cov_kwds={"groups": dc["cluster_id"]})
    ic = [t for t in fc.params.index if ":" in t]
    wc = fc.wald_test(np.eye(len(fc.params))[[list(fc.params.index).index(t) for t in ic]], scalar=True)
    r["h4a_ncore_all"] = {"n": len(dc), "wald_chi2": float(wc.statistic), "df": len(ic), "p": float(wc.pvalue)}
    cell = d.groupby(["family", "analysis_region"])["N"].agg(["mean", "count", "sem"]).reset_index()
    cell.to_csv(OUT / f"{judge}_h4a_cells.csv", index=False)

    # H4b
    detail, htests = an.home_side_tables(f)
    detail.to_csv(OUT / f"{judge}_h4b_detail.csv", index=False)
    htests.to_csv(OUT / f"{judge}_h4b_tests.csv", index=False)
    # pooled own vs other share per home family (F1-default, F2-default, F5, F3)
    from experiment_c.coding import HOME_PARTIES, split_claimants
    rows = []
    sub = f.loc[f["framing"].isin(["F1", "F2", "F5", "F3"])]
    for hf in ["western", "chinese", "sea"]:
        col = f"favours_home_{hf}"; it_all = sub.loc[sub[col].notna()]
        # Only disputes where a home claimant faces at least one claimant from outside the home group:
        # in e.g. SEA-vs-SEA disputes every pick counts as "home" for every model, so they say nothing about bias.
        mixed = it_all["gold_parties_cleaned"].map(lambda g: any(p not in HOME_PARTIES[hf] for p in split_claimants(g)))
        it = it_all.loc[mixed]
        for fam, g in it.groupby("family"):
            m, n, lo, hi = an.mean_ci(g[col])
            v = g.loc[g["verdict"] == 1, col]
            # Side-picking answers only; interval clustered by dispute (same territory recurs across models and F5 orders).
            vs = an.clustered_mean_stats(v, g.loc[v.index, "cluster_id"])
            rows.append({"home_family": hf, "family": fam, "own": fam == hf, "n": n, "home_share": m, "lo": lo, "hi": hi,
                         "n_items": int(it["territory_id"].nunique()), "n_items_all": int(it_all["territory_id"].nunique()),
                         "home_among_verdicts": float(v.mean()) if len(v) else np.nan,
                         "n_verdicts": int(len(v)), "v_lo": vs["ci_low"], "v_hi": vs["ci_high"],
                         "verdict_rate": float(g["verdict"].mean())})
    pd.DataFrame(rows).to_csv(OUT / f"{judge}_h4b_pooled.csv", index=False)
    # per-model home share on PRC items, F5 only (pilot signal)
    f5 = f.loc[(f["framing"] == "F5") & f["favours_home_chinese"].notna()].drop_duplicates(["model_key", "territory_id"])
    f5.groupby("model_key")["favours_home_chinese"].agg(["mean", "count"]).to_csv(OUT / f"{judge}_f5_prc_by_model.csv")

    # Lineage: primary (F2-default N) + per pressure level N_core and verdict
    lt = an.lineage_table(f); lt.to_csv(OUT / f"{judge}_lineage_primary.csv", index=False)
    lrows = []
    for c in an.LINEAGE_CONTRASTS:
        for level in ["F1-default", "F2-default", "F3", "F4", "F5"]:
            for outcome in ["N_core", "verdict"]:
                if outcome == "N_core" and level == "F5":
                    continue
                s = f.loc[f["pressure_level"] == level]
                s = s.loc[an.scored_mask(s, "N_core")] if outcome == "N_core" else s.loc[s["verdict"].notna()]
                per = s.groupby(["model_key", "territory_id", "cluster_id"])[outcome].mean().reset_index()
                a = per.loc[per["model_key"] == c.treatment]; b = per.loc[per["model_key"] == c.baseline]
                p = a.merge(b, on=["territory_id", "cluster_id"], suffixes=("_t", "_b"))
                st = an.clustered_mean_stats(p[f"{outcome}_t"] - p[f"{outcome}_b"], p["cluster_id"])
                st.update({"contrast": c.name, "level": level, "outcome": outcome})
                lrows.append(st)
    lf = pd.DataFrame(lrows)
    for outcome in ["N_core", "verdict"]:
        idx = lf.index[lf["outcome"] == outcome]
        lf.loc[idx, "p_holm"] = an.holm_adjust(list(lf.loc[idx, "p_two_sided"]))
    lf.to_csv(OUT / f"{judge}_lineage_by_level.csv", index=False)
    res[judge] = r

# Judge agreement on full run (no balance rows)
a = pd.read_csv(an.long_path_safe(ROOT / "experiment_c/data/judged" / JUDGES[0] / "scored_answers.csv"))
b = pd.read_csv(an.long_path_safe(ROOT / "experiment_c/data/judged" / JUDGES[1] / "scored_answers.csv"))
key = ["model_key", "territory_id", "condition"]
a = a.drop_duplicates(key, keep="last"); b = b.drop_duplicates(key, keep="last")
m = a.merge(b, on=key, suffixes=("_a", "_b"))
m = m.loc[m["instruction_a"].fillna("") != "balance"]
from experiment_c.compare_judges import cohen_kappa as _ck
cohen_kappa_score = lambda x, y: _ck(list(x), list(y))
fp = m.dropna(subset=["favoured_party_a", "favoured_party_b"])
jl = fp.loc[fp["label_method_a"].fillna("judge") != "rule"] if "label_method_a" in fp else fp
nc = m.dropna(subset=["N_core_a", "N_core_b"])
res["judge_agreement"] = {
    "n_pairs": len(m), "kappa_all": float(cohen_kappa_score(fp["favoured_party_a"], fp["favoured_party_b"])),
    "kappa_judge_labelled": float(cohen_kappa_score(jl["favoured_party_a"], jl["favoured_party_b"])), "n_judge_labelled": len(jl),
    "r_ncore": float(np.corrcoef(nc["N_core_a"], nc["N_core_b"])[0, 1]), "n_ncore": len(nc),
    "mean_ncore_a": float(nc["N_core_a"].mean()), "mean_ncore_b": float(nc["N_core_b"].mean()),
    "judge_error_a": int(a["judge_error"].notna().sum()), "judge_error_b": int(b["judge_error"].notna().sum()),
}
(OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
