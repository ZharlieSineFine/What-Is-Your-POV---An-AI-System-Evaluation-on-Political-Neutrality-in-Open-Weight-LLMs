"""Experiment C v2 analysis.

Which slice answers which hypothesis:

- H1 (region) and H4a (region x family) on N: F2-default, the uncued prompt.
  F1 is the cued ceiling reference and is reported, not tested.
- H2 (instruction) and H3 (instruction x family/region): F2-default versus
  F2-balance, paired by territory. F1 Delta-N is reported as a secondary
  table.
- H4b (directional locality): home-side share on F1, F2, F3 and F5 answers.
  F4 is left out because the prompt itself asks the model to favour one party.
- H5 (pressure): N_core and verdict rate across F1-default, F2-default, F3,
  F4 and F5.

Framing IDs here are the report numbering (F3 map caption, F4 loaded premise,
F5 forced choice). Stored answers use the generation IDs; load_scored
relabels them (coding.REPORT_FRAMING).

Confirmatory N model (F2 rows):

    N ~ instruction * region * family + (1|territory) + (1|cluster)

statsmodels MixedLM supports one random intercept, so the territory
intercept is fit there. The cluster intercept is handled by an OLS model
with cluster-robust standard errors on cluster_id. If pymer4 and R are
available, the two-intercept model is fit as well. The notes file records
which estimator ran.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_c.coding import (
    FAMILIES,
    FRAMINGS,
    HOME_PARTIES,
    INSTRUCTION_PAIRS,
    WORD_LIMIT,
    REGIONS,
    framing_of,
    report_condition,
    report_framing,
    split_claimants,
)
from experiment_c.models import LINEAGE_CONTRASTS

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_JUDGED_DIR = DATA_DIR / "judged"
DEFAULT_RESULTS_DIR = DATA_DIR / "results"

PRIMARY_SLICE = "F2-default"
PRESSURE_CONDITIONS = ("F1-default", "F2-default", "F3", "F4", "F5")
PREMISE_SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}
SCORE_COLUMNS = (
    "coverage",
    "symmetry",
    "non_endorsement",
    "dispute_framing",
    "implicit_sovereignty",
    "N",
    "N_core",
)

FULL_FORMULA = (
    "N ~ C(instruction, Treatment(reference='default'))"
    " * C(analysis_region, Treatment(reference='other'))"
    " * C(family, Treatment(reference='western'))"
)
DEFAULT_SLICE_FORMULA = (
    "N ~ C(analysis_region, Treatment(reference='other'))"
    " * C(family, Treatment(reference='western'))"
)
PRESSURE_FORMULA = (
    "N_core ~ C(pressure_level, Treatment(reference='F1-default'))"
    " * C(family, Treatment(reference='western'))"
)
VERDICT_FORMULA = (
    "verdict ~ C(pressure_level, Treatment(reference='F1-default'))"
    " * C(family, Treatment(reference='western'))"
)

# Main-effects ("additive") versions. Without interaction terms, each
# coefficient is the effect averaged over the other factors, not the effect in
# one reference group. These answer the overall questions (H5 framing effect,
# H1 region effect, H2 corroboration); the interaction models above answer
# whether effects differ between groups (H3, H4a, family differences in H5).
H2_MAIN_FORMULA = (
    "N ~ C(instruction, Treatment(reference='default'))"
    " + C(analysis_region, Treatment(reference='other'))"
    " + C(family, Treatment(reference='western'))"
)
DEFAULT_SLICE_MAIN_FORMULA = (
    "N ~ C(analysis_region, Treatment(reference='other'))"
    " + C(family, Treatment(reference='western'))"
)
PRESSURE_MAIN_FORMULA = (
    "N_core ~ C(pressure_level, Treatment(reference='F1-default'))"
    " + C(family, Treatment(reference='western'))"
)
VERDICT_MAIN_FORMULA = (
    "verdict ~ C(pressure_level, Treatment(reference='F1-default'))"
    " + C(family, Treatment(reference='western'))"
)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment. NaN inputs stay NaN and are not in the family."""
    indexed = [(p, i) for i, p in enumerate(pvalues) if p == p]
    indexed.sort(key=lambda item: (item[0], item[1]))
    adjusted = {i: float("nan") for i in range(len(pvalues))}
    running = 0.0
    family = len(indexed)
    for rank, (pvalue, index) in enumerate(indexed):
        value = min(1.0, (family - rank) * float(pvalue))
        running = max(running, value)
        adjusted[index] = running
    return [adjusted[i] for i in range(len(pvalues))]


def mean_ci(values, alpha: float = 0.05) -> tuple[float, int, float, float]:
    from scipy import stats

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return float("nan"), 0, float("nan"), float("nan")
    mean = float(array.mean())
    if n < 2:
        return mean, n, float("nan"), float("nan")
    se = float(array.std(ddof=1) / np.sqrt(n))
    critical = float(stats.t.ppf(1 - alpha / 2, n - 1))
    return mean, n, mean - critical * se, mean + critical * se


def clustered_mean_stats(diff, clusters) -> dict:
    """Average of paired differences, with a CI and test that respect clustering.

    The average is the plain mean, exactly as with a paired t-test. The
    uncertainty comes from regressing the differences on a constant with
    cluster-robust standard errors by dispute (``cluster_id``) and t-based
    inference with (clusters - 1) degrees of freedom. Pairs from the same
    dispute, or the same territory across models, are therefore not counted
    as independent evidence.
    """
    import statsmodels.api as sm

    values = np.asarray(diff, dtype=float)
    groups = np.asarray(list(clusters), dtype=object)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    n = int(values.size)
    n_clusters = int(len(set(groups.tolist())))
    out = {
        "n_pairs": n,
        "n_clusters": n_clusters,
        "mean_delta": float(values.mean()) if n else float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t": float("nan"),
        "p_two_sided": float("nan"),
        "p_one_sided_positive": float("nan"),
    }
    if n < 2 or n_clusters < 2:
        return out
    mean = out["mean_delta"]
    if float(values.std()) == 0.0:
        out.update(
            {
                "ci_low": mean,
                "ci_high": mean,
                "t": 0.0 if mean == 0 else float("inf"),
                "p_two_sided": 1.0 if mean == 0 else 0.0,
            }
        )
    else:
        codes = pd.factorize(pd.Series(groups))[0]
        fitted = sm.OLS(values, np.ones(n)).fit(cov_type="cluster", cov_kwds={"groups": codes}, use_t=True)
        low, high = fitted.conf_int()[0]
        out.update(
            {
                "ci_low": float(low),
                "ci_high": float(high),
                "t": float(fitted.tvalues[0]),
                "p_two_sided": float(fitted.pvalues[0]),
            }
        )
    p_two = out["p_two_sided"]
    out["p_one_sided_positive"] = p_two / 2.0 if mean > 0 else 1.0 - p_two / 2.0
    return out


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str).replace({"nan": ""})


def load_scored(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No scored file at {path}. Run experiment_c.judge after generation."
        )
    frame = pd.read_csv(path)
    required = {
        "model_key",
        "family",
        "lineage",
        "territory_id",
        "condition",
        "analysis_region",
        "cluster_id",
        "western_salient",
        "refused",
        "N",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates(["model_key", "territory_id", "condition"], keep="last")
    for column in SCORE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    frame["refused"] = pd.to_numeric(frame["refused"], errors="coerce").fillna(0).astype(int)
    frame["western_salient"] = (
        pd.to_numeric(frame["western_salient"], errors="coerce").fillna(0).astype(int)
    )
    for column in (
        "error",
        "judge_error",
        "framing",
        "instruction",
        "target_party",
        "favoured_party",
        "premise_accepted",
        "gold_parties_cleaned",
        "Controller",
    ):
        frame[column] = _text(frame, column)
    frame["condition"] = frame["condition"].astype(str)
    missing_framing = frame["framing"] == ""
    frame.loc[missing_framing, "framing"] = frame.loc[missing_framing, "condition"].map(framing_of)
    frame["framing"] = frame["framing"].map(report_framing)
    frame["condition"] = frame["condition"].map(report_condition)
    # One pressure level per prompt type. F5 orders and F4 targets pool.
    frame["pressure_level"] = frame["condition"].where(
        frame["framing"].isin(["F1", "F2"]), frame["framing"]
    )
    frame["premise_score"] = frame["premise_accepted"].str.lower().map(PREMISE_SCORE)
    return frame.reset_index(drop=True)


def _ordered(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    orders = {
        "family": FAMILIES,
        "analysis_region": REGIONS,
        "framing": FRAMINGS,
        "pressure_level": PRESSURE_CONDITIONS,
    }
    out = frame.copy()
    for column, levels in orders.items():
        if column in out.columns:
            out[column] = pd.Categorical(out[column], categories=list(levels), ordered=True)
    sort_cols = [column for column in columns if column in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    """Successful generations. Crashes are excluded from every rate."""
    return frame["error"] == ""


def two_claimant_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Territories with at least two gold parties.

    Dropping the dormant ROC claim leaves 17 territories with one claimant.
    Coverage, symmetry and dispute framing are not well defined there.
    """
    keep = frame["gold_parties_cleaned"].map(lambda raw: len(split_claimants(raw)) >= 2)
    return frame.loc[keep].copy()


def scored_mask(frame: pd.DataFrame, column: str = "N") -> pd.Series:
    return frame[column].notna() & (frame["refused"] == 0) & completed_mask(frame)


def labelled_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame["favoured_party"] != "") & (frame["refused"] == 0) & completed_mask(frame)


def mean_table(frame: pd.DataFrame, group_cols: list[str], column: str = "N") -> pd.DataFrame:
    rows = []
    subset = frame.loc[scored_mask(frame, column)]
    name = f"mean_{column.lower()}"
    if subset.empty:
        return pd.DataFrame(columns=group_cols + ["n_scored", name, "ci_low", "ci_high"])
    for key, group in subset.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        mean, n, low, high = mean_ci(group[column])
        record = dict(zip(group_cols, key))
        record.update({"n_scored": n, name: mean, "ci_low": low, "ci_high": high})
        rows.append(record)
    return _ordered(pd.DataFrame(rows), group_cols)


def mean_n_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return mean_table(frame, group_cols, "N")


def dimension_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Each dimension separately, so an average cannot hide which one moved."""
    subset = frame.loc[(frame["refused"] == 0) & completed_mask(frame)]
    columns = ["coverage", "symmetry", "non_endorsement", "dispute_framing", "implicit_sovereignty"]
    if subset.empty:
        return pd.DataFrame()
    table = (
        subset.groupby(["model_key", "family", "pressure_level"], observed=True)[columns]
        .mean()
        .reset_index()
    )
    return _ordered(table, ["family", "model_key", "pressure_level"])


def length_table(frame: pd.DataFrame, limit: int = WORD_LIMIT) -> pd.DataFrame:
    """Word-limit compliance on the long-answer framings (F1, F2, F4).

    Over-long answers are kept as written, so compliance is itself a result.
    ``within_10pct`` allows up to 10% over the limit.
    """
    if "response" not in frame.columns:
        return pd.DataFrame()
    subset = frame.loc[completed_mask(frame) & frame["framing"].isin(["F1", "F2", "F4"])].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["words"] = subset["response"].fillna("").astype(str).str.split().str.len()
    rows = []
    for (model_key, family, framing), group in subset.groupby(["model_key", "family", "framing"], observed=True):
        words = group["words"]
        rows.append(
            {
                "model_key": model_key,
                "family": family,
                "framing": framing,
                "n": int(words.size),
                "median_words": float(words.median()),
                "max_words": int(words.max()),
                "within_limit": float((words <= limit).mean()),
                "within_10pct": float((words <= limit * 1.1).mean()),
            }
        )
    return _ordered(pd.DataFrame(rows), ["family", "model_key", "framing"])


def truncation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of answers that ran out of token budget. Should be near 0 at 2048."""
    if "truncated" not in frame.columns:
        return pd.DataFrame()
    subset = frame.loc[completed_mask(frame)].copy()
    subset["truncated"] = pd.to_numeric(subset["truncated"], errors="coerce")
    subset = subset.loc[subset["truncated"].notna()]
    if subset.empty:
        return pd.DataFrame()
    table = (
        subset.groupby(["model_key", "family", "framing"], observed=True)["truncated"]
        .agg(n="size", n_truncated="sum", truncation_rate="mean")
        .reset_index()
    )
    return _ordered(table, ["family", "model_key", "framing"])


def refusal_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    subset = frame.loc[completed_mask(frame)]
    if subset.empty:
        return pd.DataFrame(columns=group_cols + ["n_completed", "n_refused", "refusal_rate"])
    for key, group in subset.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        n_completed = int(len(group))
        n_refused = int(group["refused"].sum())
        record = dict(zip(group_cols, key))
        record.update(
            {
                "n_completed": n_completed,
                "n_refused": n_refused,
                "refusal_rate": (n_refused / n_completed) if n_completed else float("nan"),
            }
        )
        rows.append(record)
    return _ordered(pd.DataFrame(rows), group_cols)


def delta_table(frame: pd.DataFrame, group_col: str, framing: str = "F2") -> pd.DataFrame:
    """Paired Delta-N = N(balance) - N(default) within one framing.

    Pairs are matched on model and territory. The CI and p-values are
    cluster-robust by dispute (clustered_mean_stats). Use group_col="all_models"
    for the pooled H2 test across every model.
    """
    default_cond, balance_cond = INSTRUCTION_PAIRS[framing]
    if group_col == "all_models":
        frame = frame.assign(all_models="all")
    columns = list(dict.fromkeys([group_col, "territory_id", "model_key", "cluster_id", "condition", "N"]))
    subset = frame.loc[scored_mask(frame), columns]
    rows = []
    if subset.empty:
        return pd.DataFrame()
    for group_value, group in subset.groupby(group_col, dropna=False, observed=True):
        default = group.loc[group["condition"] == default_cond, ["model_key", "territory_id", "cluster_id", "N"]]
        balance = group.loc[group["condition"] == balance_cond, ["model_key", "territory_id", "N"]]
        paired = default.merge(balance, on=["model_key", "territory_id"], suffixes=("_default", "_balance"))
        stats = clustered_mean_stats(paired["N_balance"].to_numpy() - paired["N_default"].to_numpy(), paired["cluster_id"])
        stats[group_col] = group_value
        stats["framing"] = framing
        rows.append(stats)
    columns = [
        group_col,
        "framing",
        "n_pairs",
        "n_clusters",
        "mean_delta",
        "ci_low",
        "ci_high",
        "t",
        "p_two_sided",
        "p_one_sided_positive",
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table[columns]
    if group_col == "model_key":
        pvals = table["p_one_sided_positive"].tolist()
        table["p_holm_secondary"] = holm_adjust([float(p) if p == p else float("nan") for p in pvals])
    if group_col in {"family", "analysis_region"}:
        return _ordered(table, [group_col])
    return table.sort_values(group_col).reset_index(drop=True)


def lineage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Primary-slice paired contrasts: regional model minus lineage baseline."""
    subset = frame.loc[scored_mask(frame) & (frame["condition"] == PRIMARY_SLICE)].copy()
    rows = []
    slices = ["all", *REGIONS]
    for contrast in LINEAGE_CONTRASTS:
        for region in slices:
            left = subset.loc[subset["model_key"] == contrast.treatment]
            right = subset.loc[subset["model_key"] == contrast.baseline]
            if region != "all":
                left = left.loc[left["analysis_region"] == region]
                right = right.loc[right["analysis_region"] == region]
            paired = left.merge(right, on="territory_id", suffixes=("_treatment", "_baseline"))
            diff = paired["N_treatment"].to_numpy() - paired["N_baseline"].to_numpy() if len(paired) else np.array([])
            clusters = paired["cluster_id_treatment"] if len(paired) else []
            stats = clustered_mean_stats(diff, clusters)
            stats.update(
                {
                    "contrast": contrast.name,
                    "lineage": contrast.lineage,
                    "treatment": contrast.treatment,
                    "baseline": contrast.baseline,
                    "slice": region,
                    "holm_family": "overall" if region == "all" else contrast.name,
                }
            )
            rows.append(stats)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = float("nan")
    for _, index in table.groupby("holm_family").groups.items():
        pvals = [float(p) if p == p else float("nan") for p in table.loc[index, "p_two_sided"]]
        table.loc[index, "p_holm"] = holm_adjust(pvals)
    order = [
        "contrast",
        "lineage",
        "treatment",
        "baseline",
        "slice",
        "n_pairs",
        "n_clusters",
        "mean_delta",
        "ci_low",
        "ci_high",
        "t",
        "p_two_sided",
        "p_holm",
        "holm_family",
    ]
    return table[order]


def western_salience_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Inside Other, on the primary slice: Western-salient items versus the rest of Other."""
    from scipy import stats

    subset = frame.loc[
        scored_mask(frame)
        & (frame["condition"] == PRIMARY_SLICE)
        & (frame["analysis_region"] == "other")
    ]
    rows = []
    if subset.empty:
        return pd.DataFrame()
    for family, group in subset.groupby("family", dropna=False, observed=True):
        salient = group.loc[group["western_salient"] == 1, "N"]
        other = group.loc[group["western_salient"] == 0, "N"]
        mean_s, n_s, low_s, high_s = mean_ci(salient)
        mean_o, n_o, low_o, high_o = mean_ci(other)
        if n_s >= 2 and n_o >= 2:
            t_stat, pvalue = stats.ttest_ind(salient, other, equal_var=False)
            t_stat, pvalue = float(t_stat), float(pvalue)
        else:
            t_stat, pvalue = float("nan"), float("nan")
        rows.append(
            {
                "family": family,
                "n_western_salient": n_s,
                "mean_n_western_salient": mean_s,
                "ci_low_western_salient": low_s,
                "ci_high_western_salient": high_s,
                "n_other_nonsalient": n_o,
                "mean_n_other_nonsalient": mean_o,
                "ci_low_other_nonsalient": low_o,
                "ci_high_other_nonsalient": high_o,
                "diff_salient_minus_nonsalient": mean_s - mean_o,
                "t_welch": t_stat,
                "p_welch": pvalue,
            }
        )
    return _ordered(pd.DataFrame(rows), ["family"])


# ---------------------------------------------------------------- direction


def resolve_f5_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Score F5 per territory, across the two option orders.

    A forced-choice pick only counts as a verdict if the model names the same
    party in both orders. If the pick changes with the order (for example it
    always takes the last-listed country), it reflects list position, not a
    stance, and both answers are recoded to favoured_party = "none".
    ``f5_order_status`` records consistent / order_dependent / declined /
    unpaired for every F5 row; ``favoured_party_raw`` keeps the original label.
    """
    out = frame.copy()
    out["favoured_party_raw"] = out["favoured_party"]
    out["f5_order_status"] = ""
    labelled = labelled_mask(out)
    f5 = out.loc[(out["framing"] == "F5") & labelled]
    for (_, _), group in f5.groupby(["model_key", "territory_id"]):
        if len(group) < 2:
            out.loc[group.index, "f5_order_status"] = "unpaired"
            continue
        picks = set(group["favoured_party"])
        if picks == {"none"}:
            status = "declined"
        elif len(picks) == 1:
            status = "consistent"
        elif "none" in picks:
            status = "declined"
        else:
            status = "order_dependent"
        out.loc[group.index, "f5_order_status"] = status
        if status in {"order_dependent", "declined"}:
            out.loc[group.index, "favoured_party"] = "none"
    return out


def f5_order_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Robustness table: how often each model's F5 pick depends on option order."""
    f5 = frame.loc[frame["f5_order_status"] != ""]
    if f5.empty:
        return pd.DataFrame()
    per_territory = f5.drop_duplicates(["model_key", "territory_id"])
    rows = []
    for (model_key, family), group in per_territory.groupby(["model_key", "family"], observed=True):
        counts = group["f5_order_status"].value_counts()
        picked = int(counts.get("consistent", 0) + counts.get("order_dependent", 0))
        rows.append(
            {
                "model_key": model_key,
                "family": family,
                "n_territories": int(len(group)),
                "consistent": int(counts.get("consistent", 0)),
                "order_dependent": int(counts.get("order_dependent", 0)),
                "declined": int(counts.get("declined", 0)),
                "unpaired": int(counts.get("unpaired", 0)),
                "order_dependent_share_of_picks": (counts.get("order_dependent", 0) / picked) if picked else float("nan"),
            }
        )
    return _ordered(pd.DataFrame(rows), ["family", "model_key"])


def add_direction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """verdict, controller agreement, and home-side flags per labelled answer.

    F5 is first resolved across its two option orders (resolve_f5_order), so
    an order-dependent pick never counts as a verdict.
    """
    out = resolve_f5_order(frame)
    labelled = labelled_mask(out)
    out["verdict"] = np.where(labelled, (out["favoured_party"] != "none").astype(float), np.nan)
    gold = out["gold_parties_cleaned"].map(lambda raw: set(split_claimants(raw)))
    controller_known = [c in g for c, g in zip(out["Controller"], gold)]
    out["favours_controller"] = np.where(
        labelled & pd.Series(controller_known, index=out.index),
        (out["favoured_party"] == out["Controller"]).astype(float),
        np.nan,
    )
    for family, home in HOME_PARTIES.items():
        has_home = gold.map(lambda parties: bool(parties & home))
        out[f"home_item_{family}"] = has_home.astype(int)
        out[f"favours_home_{family}"] = np.where(
            labelled & has_home,
            out["favoured_party"].isin(home).astype(float),
            np.nan,
        )
    return out


def verdict_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Share of answers that favour any party. Refusals are not in the denominator."""
    subset = frame.loc[frame["verdict"].notna()]
    rows = []
    for key, group in subset.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        mean, n, low, high = mean_ci(group["verdict"])
        record = dict(zip(group_cols, key))
        record.update({"n_labelled": n, "verdict_rate": mean, "ci_low": low, "ci_high": high})
        rows.append(record)
    return _ordered(pd.DataFrame(rows), group_cols) if rows else pd.DataFrame()


def controller_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Among answers that pick a side, how often the pick is the current controller."""
    subset = frame.loc[(frame["verdict"] == 1) & frame["favours_controller"].notna()]
    rows = []
    for (model_key, family, level), group in subset.groupby(
        ["model_key", "family", "pressure_level"], observed=True
    ):
        mean, n, low, high = mean_ci(group["favours_controller"])
        rows.append(
            {
                "model_key": model_key,
                "family": family,
                "pressure_level": level,
                "n_verdicts": n,
                "controller_agreement": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return _ordered(pd.DataFrame(rows), ["family", "model_key", "pressure_level"]) if rows else pd.DataFrame()


def home_side_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H4b. For each home family's items, every model's home-side share.

    ``home_share`` = answers favouring that family's home party / all
    labelled answers on those items ("none" counts in the denominator).
    ``home_rate_among_verdicts`` uses only answers that picked a side.
    F4 is excluded because its prompt asks for one party's case.
    """
    subset = frame.loc[frame["framing"].isin(["F1", "F2", "F3", "F5"])]
    detail = []
    tests = []
    import statsmodels.formula.api as smf

    for home_family in FAMILIES:
        column = f"favours_home_{home_family}"
        items = subset.loc[subset[column].notna()].copy()
        if items.empty:
            continue
        for (model_key, family, framing), group in items.groupby(
            ["model_key", "family", "framing"], observed=True
        ):
            share, n, low, high = mean_ci(group[column])
            verdicts = group.loc[group["verdict"] == 1, column]
            detail.append(
                {
                    "home_family": home_family,
                    "framing": framing,
                    "model_key": model_key,
                    "family": family,
                    "own_family": int(family == home_family),
                    "n_labelled": n,
                    "home_share": share,
                    "ci_low": low,
                    "ci_high": high,
                    "n_verdicts": int(len(verdicts)),
                    "home_rate_among_verdicts": float(verdicts.mean()) if len(verdicts) else float("nan"),
                }
            )
        items["own_family"] = (items["family"] == home_family).astype(int)
        items["y"] = items[column]
        record = {"home_family": home_family, "n": int(len(items))}
        if items["own_family"].nunique() == 2 and items["cluster_id"].nunique() >= 2 and items["y"].nunique() > 1:
            try:
                fitted = smf.ols("y ~ own_family + C(framing)", data=items).fit(
                    cov_type="cluster", cov_kwds={"groups": items["cluster_id"]}
                )
                coef = float(fitted.params["own_family"])
                p_two = float(fitted.pvalues["own_family"])
                record.update(
                    {
                        "own_minus_other_home_share": coef,
                        "se": float(fitted.bse["own_family"]),
                        "p_two_sided": p_two,
                        "p_one_sided_positive": _one_sided_positive(coef, p_two),
                    }
                )
            except Exception as exc:
                record["note"] = f"fit failed: {exc}"
        else:
            record["note"] = "not estimable (no variation, or one side missing)"
        tests.append(record)
    detail_frame = pd.DataFrame(detail)
    if not detail_frame.empty:
        detail_frame = _ordered(detail_frame, ["home_family", "framing", "family", "model_key"])
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty and "p_one_sided_positive" in test_frame.columns:
        test_frame["p_holm"] = holm_adjust(
            [float(p) if p == p else float("nan") for p in test_frame["p_one_sided_positive"]]
        )
    return detail_frame, test_frame


def premise_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F4. Mean acceptance, within-territory asymmetry, and home versus other targets."""
    subset = frame.loc[
        (frame["framing"] == "F4") & frame["premise_score"].notna() & (frame["refused"] == 0) & completed_mask(frame)
    ].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()
    spread = (
        subset.groupby(["model_key", "territory_id"])["premise_score"]
        .agg(lambda values: float(values.max() - values.min()) if len(values) >= 2 else np.nan)
        .rename("asymmetry")
        .reset_index()
    )
    summary_rows = []
    for (model_key, family), group in subset.groupby(["model_key", "family"], observed=True):
        acceptance, n, low, high = mean_ci(group["premise_score"])
        model_spread = spread.loc[spread["model_key"] == model_key, "asymmetry"]
        asym, n_terr, asym_low, asym_high = mean_ci(model_spread)
        summary_rows.append(
            {
                "model_key": model_key,
                "family": family,
                "n_answers": n,
                "mean_acceptance": acceptance,
                "ci_low": low,
                "ci_high": high,
                "n_territories": n_terr,
                "mean_asymmetry": asym,
                "asymmetry_ci_low": asym_low,
                "asymmetry_ci_high": asym_high,
            }
        )
    home_rows = []
    for (model_key, family), group in subset.groupby(["model_key", "family"], observed=True):
        home = HOME_PARTIES[family]
        group = group.assign(home_target=group["target_party"].isin(home))
        # Only territories that have both a home target and another target.
        mixed = group.groupby("territory_id")["home_target"].transform(lambda flags: flags.nunique() == 2)
        group = group.loc[mixed]
        if group.empty:
            continue
        per_territory = group.groupby(["territory_id", "home_target"])["premise_score"].mean().unstack()
        diff = (per_territory[True] - per_territory[False]).to_numpy()
        cluster_of = group.drop_duplicates("territory_id").set_index("territory_id")["cluster_id"]
        stats = clustered_mean_stats(diff, cluster_of.reindex(per_territory.index).to_numpy())
        stats.update({"model_key": model_key, "family": family})
        home_rows.append(stats)
    summary = _ordered(pd.DataFrame(summary_rows), ["family", "model_key"])
    home_table = pd.DataFrame(home_rows)
    if not home_table.empty:
        home_table = home_table.rename(columns={"mean_delta": "home_minus_other_acceptance"})
        home_table = _ordered(home_table, ["family", "model_key"])
    return summary, home_table


def pressure_table(frame: pd.DataFrame) -> pd.DataFrame:
    """H5 descriptive: N_core and verdict rate at each pressure level, per model."""
    subset = frame.loc[frame["pressure_level"].isin(PRESSURE_CONDITIONS)]
    rows = []
    for (model_key, family, level), group in subset.groupby(
        ["model_key", "family", "pressure_level"], observed=True
    ):
        core = group.loc[scored_mask(group, "N_core"), "N_core"]
        verdict = group["verdict"].dropna()
        rows.append(
            {
                "model_key": model_key,
                "family": family,
                "pressure_level": level,
                "n_core": int(core.size),
                "mean_n_core": float(core.mean()) if core.size else float("nan"),
                "n_labelled": int(verdict.size),
                "verdict_rate": float(verdict.mean()) if verdict.size else float("nan"),
            }
        )
    return _ordered(pd.DataFrame(rows), ["family", "model_key", "pressure_level"]) if rows else pd.DataFrame()


# ---------------------------------------------------------------- models


def _coef_frame(result) -> pd.DataFrame:
    conf = result.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    return pd.DataFrame(
        {
            "term": result.params.index.astype(str),
            "coef": result.params.to_numpy(),
            "se": result.bse.to_numpy(),
            "pvalue": result.pvalues.to_numpy(),
            "ci_low": conf["ci_low"].to_numpy(),
            "ci_high": conf["ci_high"].to_numpy(),
        }
    )


def _one_sided_positive(coef: float, p_two: float) -> float:
    if p_two != p_two:
        return float("nan")
    if coef > 0:
        return p_two / 2.0
    return 1.0 - p_two / 2.0


def _is_instruction_main(term: str) -> bool:
    return ":" not in term and "instruction" in term and "balance" in term


def instruction_main_term_from_frame(frame: pd.DataFrame) -> str | None:
    for term in frame["term"].astype(str):
        if _is_instruction_main(term):
            return term
    return None


def fit_models(frame: pd.DataFrame) -> dict:
    """Fit the confirmatory models. Never raises on a failed fit."""
    import statsmodels.formula.api as smf

    notes: list[str] = []
    outputs: dict[str, pd.DataFrame] = {}

    def _fit_ols(data: pd.DataFrame, formula: str, label: str) -> None:
        if len(data) < 8 or data["cluster_id"].nunique() < 2:
            notes.append(f"{label}: skipped OLS cluster-robust fit (too few rows or clusters).")
            return
        try:
            fitted = smf.ols(formula, data=data).fit(
                cov_type="cluster",
                cov_kwds={"groups": data["cluster_id"]},
            )
        except Exception as exc:
            notes.append(f"{label}: OLS failed ({exc}).")
            return
        outputs[label] = _coef_frame(fitted)
        notes.append(f"{label}: OLS with cluster-robust SEs on cluster_id, n={len(data)}.")

    def _fit_mixed(data: pd.DataFrame, formula: str, label: str) -> None:
        if len(data) < 8 or data["territory_id"].nunique() < 2:
            notes.append(f"{label}: skipped MixedLM (too few rows or territories).")
            return
        try:
            fitted = smf.mixedlm(formula, data=data, groups=data["territory_id"]).fit(
                method="lbfgs",
                reml=True,
            )
        except Exception as exc:
            notes.append(f"{label}: MixedLM failed ({exc}). Territory intercept not reported.")
            return
        outputs[label] = _coef_frame(fitted)
        notes.append(f"{label}: MixedLM random intercept grouped by territory_id.")

    notes.append(
        "statsmodels MixedLM supports one grouping factor. "
        "The territory intercept is the MixedLM. "
        "The cluster intercept is the OLS cluster-robust covariance on cluster_id. "
        "pymer4 is used for both intercepts together when R is available."
    )
    f2 = frame.loc[scored_mask(frame) & (frame["framing"] == "F2")].copy()
    _fit_ols(f2, FULL_FORMULA, "ols_f2_instruction_cluster")
    _fit_mixed(f2, FULL_FORMULA, "mixedlm_f2_instruction")
    _fit_ols(f2, H2_MAIN_FORMULA, "ols_f2_instruction_main_effects")
    primary = f2.loc[f2["condition"] == PRIMARY_SLICE].copy()
    notes.append(
        "H1 and H4a are read on the F2-default slice (uncued prompt, no instruction). "
        "F1 is the cued ceiling and is not used for H1 or H4."
    )
    _fit_ols(primary, DEFAULT_SLICE_FORMULA, "ols_primary_slice_cluster")
    _fit_mixed(primary, DEFAULT_SLICE_FORMULA, "mixedlm_primary_slice")
    _fit_ols(primary, DEFAULT_SLICE_MAIN_FORMULA, "ols_primary_slice_main_effects")
    notes.append(
        "H1: read the region rows of ols_primary_slice_main_effects (region effect averaged over "
        "families). H4a: read the region:family rows of ols_primary_slice_cluster."
    )
    _try_pymer4(f2, notes, outputs)

    pressure = frame.loc[frame["pressure_level"].isin(PRESSURE_CONDITIONS)].copy()
    pressure_core = pressure.loc[scored_mask(pressure, "N_core")]
    pressure_core = pressure_core.loc[pressure_core["pressure_level"] != "F5"]
    _fit_ols(pressure_core, PRESSURE_FORMULA, "ols_pressure_n_core")
    _fit_ols(pressure_core, PRESSURE_MAIN_FORMULA, "ols_pressure_n_core_main_effects")
    pressure_verdict = pressure.loc[pressure["verdict"].notna()]
    _fit_ols(pressure_verdict, VERDICT_FORMULA, "ols_pressure_verdict")
    _fit_ols(pressure_verdict, VERDICT_MAIN_FORMULA, "ols_pressure_verdict_main_effects")
    notes.append(
        "H5: read ols_pressure_n_core_main_effects and ols_pressure_verdict_main_effects. "
        "Their pressure_level coefficients are differences from F1-default (cued), averaged over "
        "model families. The *_n_core and *_verdict models with interactions show whether "
        "the drop differs by family (their plain pressure_level rows are the Western family only). "
        "N_core has no F5 rows, because F5 is labelled for a choice only. The verdict models "
        "include F5 and are linear probability models. All use cluster-robust SEs by dispute."
    )

    h2 = None
    main = outputs.get("ols_f2_instruction_main_effects")
    if main is not None:
        term = instruction_main_term_from_frame(main)
        if term is not None:
            hit = main.loc[main["term"] == term].iloc[0]
            h2 = {
                "term": term,
                "coef": float(hit["coef"]),
                "p_two_sided": float(hit["pvalue"]),
                "p_one_sided_positive": _one_sided_positive(float(hit["coef"]), float(hit["pvalue"])),
            }
    return {"notes": notes, "frames": outputs, "h2": h2}


def model_difference_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H6: do the seven models differ in neutrality, and do they differ more under pressure?

    H6a: joint Wald test that all model terms are zero, with framing as a
    covariate (N_core on F1-default, F2-default, F3, F4; verdict also on F5).
    H6b: joint Wald test of the model x framing interaction.
    Standard errors are cluster-robust on cluster_id. A joint test with more
    constraints than (clusters - 1) is not reliable, so it is flagged.

    The ranking table gives each model's mean over territories of its
    per-territory average across the framings, with a t-based 95% CI over
    territories.
    """
    import statsmodels.formula.api as smf

    pressure = frame.loc[frame["pressure_level"].isin(PRESSURE_CONDITIONS)].copy()
    datasets = {
        "N_core": pressure.loc[scored_mask(pressure, "N_core") & (pressure["pressure_level"] != "F5")],
        "verdict": pressure.loc[pressure["verdict"].notna()],
    }
    tests = []
    ranking_rows = []
    for outcome, data in datasets.items():
        data = data.copy()
        n_clusters = int(data["cluster_id"].nunique()) if not data.empty else 0
        for label, formula, prefix in (
            ("H6a model main effect", f"{outcome} ~ C(model_key) + C(pressure_level)", "C(model_key)"),
            ("H6b model x framing", f"{outcome} ~ C(model_key) * C(pressure_level)", "C(model_key):C(pressure_level)"),
        ):
            record = {"test": label, "outcome": outcome, "n": int(len(data)), "n_clusters": n_clusters}
            if len(data) < 8 or n_clusters < 3 or data["model_key"].nunique() < 2:
                record["note"] = "not estimable (too few rows, clusters or models)"
                tests.append(record)
                continue
            try:
                fitted = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["cluster_id"]})
                table = fitted.wald_test_terms(skip_single=False, scalar=True).table
                hit = table.loc[[term for term in table.index if term == prefix]]
                if hit.empty:
                    record["note"] = f"term {prefix} not found"
                else:
                    row = hit.iloc[0]
                    df_constraint = int(row["df_constraint"])
                    record.update(
                        {
                            "statistic": float(row["statistic"]),
                            "df": df_constraint,
                            "p_value": float(row["pvalue"]),
                            "reliable": df_constraint <= n_clusters - 1,
                        }
                    )
                    if df_constraint > n_clusters - 1:
                        record["note"] = (
                            f"{df_constraint} constraints but only {n_clusters} clusters: "
                            "the cluster-robust joint test is not reliable here"
                        )
            except Exception as exc:
                record["note"] = f"fit failed: {exc}"
            tests.append(record)

        if data.empty:
            continue
        per_territory = data.groupby(["model_key", "family", "territory_id"], observed=True)[outcome].mean().reset_index()
        for (model_key, family), group in per_territory.groupby(["model_key", "family"], observed=True):
            mean, n, low, high = mean_ci(group[outcome])
            ranking_rows.append(
                {"outcome": outcome, "model_key": model_key, "family": family, "n_territories": n, "mean": mean, "ci_low": low, "ci_high": high}
            )
    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["outcome", "mean"], ascending=[True, False]).reset_index(drop=True)
        ranking["rank"] = ranking.groupby("outcome").cumcount() + 1
    return pd.DataFrame(tests), ranking


def _try_pymer4(data: pd.DataFrame, notes: list[str], outputs: dict) -> None:
    try:
        from pymer4.models import Lmer
    except ImportError:
        notes.append(
            "pymer4 is not installed, so the joint (1|territory) + (1|cluster) "
            "model was not fit. Use the MixedLM and the cluster-robust OLS."
        )
        return
    formula = "N ~ instruction * analysis_region * family + (1|territory_id) + (1|cluster_id)"
    try:
        fitted = Lmer(formula, data=data).fit(summarize=False)
    except Exception as exc:
        notes.append(f"pymer4 failed ({exc}).")
        return
    coefs = fitted.coefs.reset_index().rename(columns={"index": "term"})
    outputs["pymer4_two_intercepts"] = coefs
    notes.append("pymer4 Lmer fit with random intercepts for territory_id and cluster_id.")


def dataset_notes(frame: pd.DataFrame) -> list[str]:
    lines = [
        f"rows={len(frame)}",
        f"with_N={int(scored_mask(frame).sum())}",
        f"with_N_core={int(scored_mask(frame, 'N_core').sum())}",
        f"labelled_favoured_party={int(labelled_mask(frame).sum())}",
        f"refused={int(((frame['refused'] == 1) & completed_mask(frame)).sum())}",
        f"generation_errors={int((frame['error'] != '').sum())}",
        f"judge_errors={int((frame['judge_error'] != '').sum())}",
        "Generation errors are excluded from every score and rate.",
        "Judge errors leave scores missing. They are not imputed as 0.",
        "Refusals are excluded from N and reported as a refusal rate.",
        "F5 answers are never flagged as refusals; declining to pick is favoured_party=none.",
    ]
    counts = frame.groupby("pressure_level", observed=True).size().to_dict()
    lines.append("rows by pressure_level: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines.append(
        "Effective sample size for regional claims is the number of clusters "
        "(china_implicated has 14, china_sea_overlap 6). Read the overlap cell descriptively."
    )
    return lines


def long_path_safe(path: Path) -> Path:
    """On Windows, allow paths past the 260-character limit.

    The project sits on a deep OneDrive path, and long judge folder names
    (e.g. deepseek__deepseek-chat-v3-0324) push some result files past the
    limit, which used to stop the analysis halfway with FileNotFoundError.
    """
    if os.name != "nt":
        return path
    prefix = "\\\\?\\"
    text = os.path.abspath(str(path))
    if text.startswith(prefix):
        return Path(text)
    return Path(prefix + text)


def run_analysis(scored_path: Path, out_dir: Path) -> dict[str, str]:
    frame = add_direction_columns(load_scored(scored_path))
    out_dir = long_path_safe(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    home_detail, home_tests = home_side_tables(frame)
    premise_summary, premise_home = premise_tables(frame)
    h6_tests, h6_ranking = model_difference_tests(frame)
    h2_overall = delta_table(frame, "all_models", "F2")
    tables = {
        "mean_n_by_family_region_condition.csv": mean_n_table(
            frame, ["family", "analysis_region", "condition"]
        ),
        "mean_n_by_model_condition.csv": mean_n_table(frame, ["model_key", "family", "lineage", "condition"]),
        "mean_n_by_model_region_condition.csv": mean_n_table(
            frame, ["model_key", "family", "analysis_region", "condition"]
        ),
        "dimensions_by_model_pressure.csv": dimension_table(frame),
        "refusal_rate_by_model_framing.csv": refusal_table(frame, ["model_key", "family", "framing"]),
        "truncation_by_model_framing.csv": truncation_table(frame),
        "length_compliance_by_model_framing.csv": length_table(frame),
        "refusal_rate_by_family_region_framing.csv": refusal_table(
            frame, ["family", "analysis_region", "framing"]
        ),
        "delta_n_f2_overall.csv": h2_overall,
        "delta_n_f2_by_model.csv": delta_table(frame, "model_key", "F2"),
        "delta_n_f2_by_family.csv": delta_table(frame, "family", "F2"),
        "delta_n_f2_by_region.csv": delta_table(frame, "analysis_region", "F2"),
        "delta_n_f1_by_model.csv": delta_table(frame, "model_key", "F1"),
        "lineage_contrasts_primary.csv": lineage_table(frame),
        "western_salience_primary.csv": western_salience_table(frame),
        "pressure_by_model.csv": pressure_table(frame),
        "model_difference_tests.csv": h6_tests,
        "model_ranking.csv": h6_ranking,
        "verdict_rate_by_model_pressure.csv": verdict_table(frame, ["model_key", "family", "pressure_level"]),
        "verdict_rate_by_family_region_pressure.csv": verdict_table(
            frame, ["family", "analysis_region", "pressure_level"]
        ),
        "controller_agreement_by_model.csv": controller_table(frame),
        "f5_order_sensitivity_by_model.csv": f5_order_table(frame),
        "home_side_by_model.csv": home_detail,
        "home_side_tests.csv": home_tests,
        "premise_by_model.csv": premise_summary,
        "premise_home_vs_other.csv": premise_home,
    }
    fit = fit_models(frame)
    for key, table in fit["frames"].items():
        tables[f"{key}_coefficients.csv"] = table

    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path, index=False)
        written[name] = str(path)

    note_lines = dataset_notes(frame) + fit["notes"]
    if not h2_overall.empty:
        row = h2_overall.iloc[0]
        note_lines.append(
            "H2 confirmatory test (delta_n_f2_overall.csv): mean paired Delta-N on F2 across all models "
            f"= {row['mean_delta']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
            f"{int(row['n_pairs'])} pairs in {int(row['n_clusters'])} disputes, "
            f"one-sided p = {row['p_one_sided_positive']:.4g} (cluster-robust by dispute)."
        )
    if fit["h2"] is not None:
        h2 = fit["h2"]
        note_lines.append(
            "H2 corroboration: main-effects regression on F2 (no interactions, cluster-robust), "
            f"instruction coefficient ({h2['term']}) = {h2['coef']:.4f}, "
            f"one-sided p = {h2['p_one_sided_positive']:.4g}."
        )
    note_lines.append(
        "Paired comparisons (Delta-N tables, lineage contrasts, premise home-vs-other) report the plain "
        "mean difference with cluster-robust CIs and p-values by dispute, so pairs from the same "
        "dispute are not counted as independent."
    )
    note_lines.append(
        "Lineage contrasts are paired territory differences on F2-default: "
        "SEA-LION-v3 and Typhoon2 minus Llama-3.1-8B-Instruct, and "
        "SEA-LION-v4-VL minus Qwen3-VL-8B-Instruct. "
        "Holm adjustment is within the three overall contrasts, and separately "
        "within the four regions of each contrast."
    )
    note_lines.append(
        "H4b home_side_tests: own-family minus other-family home share on each "
        "family's home items (F1, F2, F3 and F5 pooled, framing as a covariate), cluster-robust, "
        "Holm across the three home families."
    )
    notes_path = out_dir / "model_notes.txt"
    notes_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    written["model_notes.txt"] = str(notes_path)
    return written


def resolve_scored(scored: Path | None, judge: str | None) -> tuple[Path, str]:
    """Pick the scored file: --scored, else --judge, else the only judge folder."""
    if scored is not None:
        return scored, scored.parent.name
    folders = sorted(p.name for p in DEFAULT_JUDGED_DIR.glob("*") if (p / "scored_answers.csv").exists())
    if judge is None:
        if len(folders) != 1:
            raise SystemExit(
                "Choose which judge's scores to analyse with --judge. Available: "
                + (", ".join(folders) or "none (run experiment_c.judge first)")
            )
        judge = folders[0]
    path = DEFAULT_JUDGED_DIR / judge / "scored_answers.csv"
    if not path.exists():
        raise SystemExit(f"No scores for judge {judge!r}. Available: {', '.join(folders) or 'none'}")
    return path, judge


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment C scored answers.")
    parser.add_argument("--judge", default=None, help="Judge folder under data/judged/, e.g. openai__gpt-4o-mini")
    parser.add_argument("--scored", type=Path, default=None, help="Explicit scored_answers.csv (overrides --judge)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: data/results/<judge>")
    args = parser.parse_args()
    scored, judge = resolve_scored(args.scored, args.judge)
    out_dir = args.out_dir or DEFAULT_RESULTS_DIR / judge
    print(f"scores: {scored}")
    written = run_analysis(scored, out_dir)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
