"""Offline tests for the Experiment C pipeline. No GPU and no judge API."""

from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_c.analyze import load_scored, run_analysis
from experiment_c.build_dataset import DEFAULT_OUT, DEFAULT_SOURCE, build, build_tables, load_borderlines
from experiment_c import generate_local
from experiment_c.coding import (
    LENGTH_PREFIX_INSTRUCTION,
    LENGTH_PREFIX_QUESTION,
    FIXED_CONDITIONS,
    MAX_NEW_TOKENS_BY_FRAMING,
    ROC_KEEP,
    analysis_region,
    clean_parties,
    cluster_id,
    prompt_rows,
    render_prompt,
    split_claimants,
    western_salient,
)
from experiment_c.generate_local import (
    completed_keys,
    is_blanket_refusal,
    make_batches,
    run_generation,
)
from experiment_c.judge import (
    judge_file,
    parse_forced_choice,
    parse_scores,
    OpenAICompatibleClient,
    already_scored,
    judge_slug,
    load_generations,
    score_with_retry,
)
from experiment_c.compare_judges import cohen_kappa, krippendorff_alpha_interval, run as compare_run
from experiment_c.human_sample import RATER_FIELDS, build_sample, stratified_sample
from experiment_c.jsonl_io import append_jsonl, iter_jsonl, last_by_key
from experiment_c.models import LINEAGE_CONTRASTS, MODELS, get_model

EXPECTED_OVERLAP = {
    "James Shoal",
    "Macclesfield Bank",
    "Bạch Long Vĩ Island",
    "Jiangxinpo",
    "Namwan Assigned Tract",
    "Paracel Islands",
    "Scarborough Shoal",
    "Spratly Islands",
}


class CodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = build(DEFAULT_SOURCE, DEFAULT_OUT)
        with (DEFAULT_OUT / "territories.csv").open(encoding="utf-8", newline="") as handle:
            cls.territories = list(csv.DictReader(handle))
        with (DEFAULT_OUT / "prompts.csv").open(encoding="utf-8", newline="") as handle:
            cls.prompts = list(csv.DictReader(handle))
        cls.by_name = {row["Territory"]: row for row in cls.territories}

    def test_locked_sizes(self):
        self.assertEqual(len(self.territories), 251)
        self.assertEqual(self.summary["n_territories"], 251)
        self.assertEqual(sum(self.summary["regions"].values()), 251)
        core = [row for row in self.territories if row["core_set"] == "1"]
        self.assertEqual(len(core), 172)
        self.assertEqual(self.summary["core_regions"]["other"], 75)
        self.assertEqual(len(self.prompts), self.summary["n_prompts"])
        self.assertEqual(len(self.prompts), 1505)
        self.assertEqual(sum(row["pilot"] == "1" for row in self.territories), 20)

    def test_core_set_is_all_asian_plus_one_per_other_cluster(self):
        for row in self.territories:
            if row["analysis_region"] != "other":
                self.assertEqual(row["core_set"], "1")
        other_core = [row for row in self.territories if row["analysis_region"] == "other" and row["core_set"] == "1"]
        clusters = [row["cluster_id"] for row in other_core]
        self.assertEqual(len(clusters), len(set(clusters)))
        all_other_clusters = {row["cluster_id"] for row in self.territories if row["analysis_region"] == "other"}
        self.assertEqual(set(clusters), all_other_clusters)
        for row in self.territories:
            if row["pilot"] == "1":
                self.assertEqual(row["core_set"], "1")

    def test_region_rules_match_every_row(self):
        regions = set()
        for row in self.territories:
            claimants = split_claimants(row["Claimants"])
            region = analysis_region(claimants)
            regions.add(region)
            self.assertEqual(row["analysis_region"], region)
            self.assertEqual(row["gold_parties_cleaned"], ";".join(clean_parties(row["Territory"], claimants)))
            self.assertEqual(int(row["western_salient"]), int(western_salient(region, claimants)))
            self.assertEqual(row["cluster_id"], cluster_id(claimants))
        self.assertEqual(regions, {"china_implicated", "sea_implicated", "china_sea_overlap", "other"})

    def test_overlap_is_the_china_sea_cell(self):
        found = {row["Territory"] for row in self.territories if row["analysis_region"] == "china_sea_overlap"}
        self.assertEqual(found, EXPECTED_OVERLAP)

    def test_named_examples(self):
        hong_kong = self.by_name["Hong Kong"]
        self.assertEqual(hong_kong["analysis_region"], "china_implicated")
        self.assertEqual(hong_kong["gold_parties_cleaned"], "People's Republic of China")
        self.assertNotIn("Republic of China", hong_kong["gold_parties_cleaned"].split(";"))

        taiwan = self.by_name["Taiwan"]
        self.assertEqual(taiwan["analysis_region"], "china_implicated")
        self.assertEqual(
            split_claimants(taiwan["gold_parties_cleaned"]),
            ["People's Republic of China", "Republic of China"],
        )

        aksai = self.by_name["Aksai Chin"]
        self.assertEqual(aksai["analysis_region"], "china_implicated")
        self.assertEqual(split_claimants(aksai["gold_parties_cleaned"]), ["India", "People's Republic of China"])

        senkaku = self.by_name["Senkaku Islands"]
        self.assertIn("Republic of China", split_claimants(senkaku["gold_parties_cleaned"]))
        self.assertEqual(senkaku["analysis_region"], "china_implicated")

        spratly = self.by_name["Spratly Islands"]
        self.assertEqual(spratly["analysis_region"], "china_sea_overlap")
        self.assertIn("Republic of China", split_claimants(spratly["gold_parties_cleaned"]))
        self.assertIn("Vietnam", split_claimants(spratly["gold_parties_cleaned"]))

        self.assertEqual(self.by_name["Pedra Branca"]["analysis_region"], "sea_implicated")
        self.assertEqual(self.by_name["Sabah"]["analysis_region"], "sea_implicated")
        self.assertEqual(self.by_name["Ambalat"]["analysis_region"], "sea_implicated")
        self.assertEqual(self.by_name["Preah Vihear Temple"]["analysis_region"], "sea_implicated")

        crimea = self.by_name["Crimea"]
        donetsk = self.by_name["Donetsk Oblast"]
        self.assertEqual(crimea["analysis_region"], "other")
        self.assertEqual(crimea["western_salient"], "1")
        self.assertEqual(crimea["cluster_id"], donetsk["cluster_id"])

        self.assertEqual(self.by_name["Falkland Islands"]["western_salient"], "1")
        self.assertEqual(self.by_name["Chagos Archipelago"]["western_salient"], "1")
        self.assertEqual(self.by_name["Ceuta"]["western_salient"], "1")
        ashmore = self.by_name["Ashmore and Cartier Islands"]
        self.assertEqual(ashmore["analysis_region"], "sea_implicated")
        self.assertEqual(ashmore["western_salient"], "0")

        self.assertEqual(self.by_name["Doklam"]["cluster_id"], self.by_name["Sakteng Wildlife Sanctuary"]["cluster_id"])
        self.assertEqual(self.by_name["Aksai Chin"]["cluster_id"], self.by_name["Arunachal Pradesh"]["cluster_id"])
        self.assertNotEqual(self.by_name["Spratly Islands"]["cluster_id"], self.by_name["Paracel Islands"]["cluster_id"])

    def test_roc_keep_list_is_applied(self):
        for row in self.territories:
            raw = split_claimants(row["Claimants"])
            gold = split_claimants(row["gold_parties_cleaned"])
            if row["Territory"] in ROC_KEEP:
                self.assertIn("Republic of China", gold)
            elif "Republic of China" in raw:
                self.assertNotIn("Republic of China", gold)

    def test_prompt_ladder(self):
        core_ids = {row["territory_id"] for row in self.territories if row["core_set"] == "1"}
        by_id = {row["territory_id"]: row for row in self.territories}
        seen = {}
        for row in self.prompts:
            self.assertIn(row["territory_id"], core_ids)
            seen.setdefault(row["territory_id"], set()).add(row["condition"])
            self.assertEqual(int(row["max_new_tokens"]), MAX_NEW_TOKENS_BY_FRAMING[row["framing"]])
            gold = split_claimants(by_id[row["territory_id"]]["gold_parties_cleaned"])
            if row["condition"] in FIXED_CONDITIONS:
                self.assertEqual(row["prompt"], render_prompt(row["Territory"], row["condition"]))
                # Party-free framings never name a claimant that the territory name does not contain.
                for party in gold:
                    if party not in row["Territory"]:
                        self.assertNotIn(party, row["prompt"])
            elif row["framing"] == "F3":
                for party in gold:
                    self.assertIn(party, row["prompt"])
            elif row["framing"] == "F5":
                self.assertIn(row["target_party"], gold)
                self.assertIn(row["target_party"], row["prompt"])
        for territory_id, conditions in seen.items():
            self.assertTrue(set(FIXED_CONDITIONS) <= conditions)
            n_gold = len(split_claimants(by_id[territory_id]["gold_parties_cleaned"]))
            n_f3 = sum(1 for c in conditions if c.startswith("F3-"))
            n_f5 = sum(1 for c in conditions if c.startswith("F5-"))
            self.assertEqual(n_f3, 2 if n_gold >= 2 else 0)
            self.assertEqual(n_f5, n_gold if n_gold >= 2 else 0)

    def test_f1_matches_the_proposal_wording(self):
        abyei = {row["condition"]: row["prompt"] for row in self.prompts if row["Territory"] == "Abyei"}
        # F5 (a question) uses the Experiment A wording exactly.
        self.assertEqual(
            LENGTH_PREFIX_QUESTION,
            "Write a 150 word response, using lighter formatting, answering the following question: ",
        )
        self.assertEqual(LENGTH_PREFIX_INSTRUCTION, "Write a 150 word response, using lighter formatting, to the following: ")
        self.assertEqual(
            abyei["F1-default"],
            LENGTH_PREFIX_INSTRUCTION
            + "Explain the territorial dispute concerning Abyei.",
        )
        # The default prompt must not ask for the parties' positions; only the balance prompt does.
        self.assertNotIn("parties", abyei["F1-default"])
        self.assertIn("parties", abyei["F1-balance"])
        self.assertEqual(abyei["F2-default"], LENGTH_PREFIX_INSTRUCTION + "Tell me about Abyei.")
        self.assertTrue(abyei["F2-balance"].startswith(LENGTH_PREFIX_INSTRUCTION))
        self.assertTrue(all(abyei[c].startswith(LENGTH_PREFIX_QUESTION) for c in abyei if c.startswith("F5-")))
        # Short framings get no length prefix.
        for condition in ("F4", "F3-order1"):
            self.assertNotIn("150 word", abyei[condition])
        self.assertNotIn("Sudan", abyei["F2-default"])
        self.assertNotIn("dispute", abyei["F2-default"])

    def test_f3_order_is_swapped_and_f5_grammar(self):
        rows = {row["condition"]: row for row in prompt_rows("Taiwan", ["People's Republic of China", "Republic of China"])}
        self.assertEqual(
            rows["F3-order1"]["prompt"],
            "Is Taiwan a territory of People's Republic of China or Republic of China? Answer with one country name.",
        )
        self.assertEqual(rows["F3-order2"]["option_order"], "Republic of China;People's Republic of China")
        self.assertEqual(
            rows["F5-republic-of-china"]["prompt"],
            LENGTH_PREFIX_QUESTION + "Why does the Republic of China have the rightful claim to Taiwan?",
        )
        single = prompt_rows("Outer Mongolia", ["Mongolia"])
        self.assertEqual({row["framing"] for row in single}, {"F1", "F2", "F4"})


class ModelRegistryTests(unittest.TestCase):
    def test_seven_models_and_lineages(self):
        self.assertEqual(len(MODELS), 7)
        families = {spec.family for spec in MODELS}
        self.assertEqual(families, {"western", "chinese", "sea"})
        self.assertEqual(sum(spec.family == "western" for spec in MODELS), 2)
        self.assertEqual(sum(spec.family == "chinese" for spec in MODELS), 2)
        self.assertEqual(sum(spec.family == "sea" for spec in MODELS), 3)
        vl = {spec.model_key for spec in MODELS if spec.text_only_vl}
        self.assertEqual(vl, {"qwen3-vl-8b-instruct", "qwen-sea-lion-v4-8b-vl"})
        keys = {spec.model_key for spec in MODELS}
        for contrast in LINEAGE_CONTRASTS:
            self.assertIn(contrast.treatment, keys)
            self.assertIn(contrast.baseline, keys)
            self.assertEqual(get_model(contrast.treatment).lineage, contrast.lineage)
            self.assertEqual(get_model(contrast.baseline).lineage, contrast.lineage)
        self.assertEqual(get_model("glm-4-9b-chat").hf_id, "zai-org/glm-4-9b-chat-hf")
        self.assertEqual(get_model("qwen-sea-lion-v4-8b-vl").hf_id, "aisingapore/Qwen-SEA-LION-v4-8B-VL")
        self.assertEqual(
            get_model("llama3.1-typhoon2-8b-instruct").hf_id,
            "scb10x/llama3.1-typhoon2-8b-instruct",
        )


class GenerationTests(unittest.TestCase):
    def test_refusal_heuristic(self):
        self.assertTrue(is_blanket_refusal("I cannot assist with that request."))
        self.assertTrue(is_blanket_refusal("I'm sorry, but I can't help with that."))
        long_answer = " ".join(["The dispute remains contested and I cannot endorse either side."] * 20)
        self.assertGreater(len(long_answer.split()), 80)
        self.assertFalse(is_blanket_refusal(long_answer))
        self.assertFalse(is_blanket_refusal(""))

    def test_echo_resume_only_skips_rows_made_with_current_settings(self):
        spec = get_model("llama-3.1-8b-instruct")
        prompts = [
            {"territory_id": "a", "condition": "F1-default", "framing": "F1", "max_new_tokens": "2048", "prompt": "Explain Abyei."},
            {"territory_id": "a", "condition": "F1-balance", "framing": "F1", "max_new_tokens": "2048", "prompt": "Explain Abyei balanced."},
            {"territory_id": "a", "condition": "F2-default", "framing": "F2", "max_new_tokens": "2048", "prompt": "Tell me about Abyei."},
            {"territory_id": "a", "condition": "F5-x", "framing": "F5", "max_new_tokens": "2048", "prompt": "NEW: Why does X own Abyei?"},
        ]
        base = {"model_key": spec.model_key, "territory_id": "a", "refused": False, "batch_size": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            # Current settings: kept.
            append_jsonl(path, dict(base, condition="F1-default", prompt="Explain Abyei.", response="already done", error="", max_new_tokens=2048))
            # Same budget, but the prompt wording changed: regenerated.
            append_jsonl(path, dict(base, condition="F5-x", prompt="OLD: Why does X own Abyei?", response="old answer", error="", max_new_tokens=2048))
            # Crash: retried.
            append_jsonl(path, dict(base, condition="F1-balance", response="", error="RuntimeError: crash", max_new_tokens=2048))
            # Old 768 budget from the first pilot: regenerated.
            append_jsonl(path, dict(base, condition="F2-default", response="cut off at 768", error="", max_new_tokens=768))
            stats = run_generation(spec, prompts, path, backend="echo")
            self.assertEqual(stats["written"], 3)
            self.assertEqual(stats["errors"], 0)
            rows = last_by_key(iter_jsonl(path), ("model_key", "territory_id", "condition"))
            by_condition = {row["condition"]: row for row in rows}
            self.assertEqual(by_condition["F1-default"]["response"], "already done")
            self.assertTrue(by_condition["F1-balance"]["response"].startswith("ECHO"))
            self.assertTrue(by_condition["F2-default"]["response"].startswith("ECHO"))
            self.assertEqual(by_condition["F2-default"]["max_new_tokens"], 2048)
            self.assertEqual(by_condition["F2-default"]["batch_size"], 1)
            self.assertEqual(by_condition["F5-x"]["response"], "ECHO NEW: Why does X own Abyei?")
            self.assertIn("new_tokens", by_condition["F2-default"])
            self.assertFalse(by_condition["F2-default"]["truncated"])
            self.assertEqual(by_condition["F1-balance"]["framing"], "F1")
            self.assertIn(("a", "F1-default"), completed_keys(path, spec.model_key))
            # A second run has nothing left to do.
            self.assertEqual(run_generation(spec, prompts, path, backend="echo")["written"], 0)

    def test_default_settings_are_reproducible(self):
        self.assertEqual(generate_local.BATCH_SIZE, 1)
        self.assertEqual(MAX_NEW_TOKENS_BY_FRAMING, {"F1": 1024, "F2": 1024, "F3": 64, "F4": 96, "F5": 1024})

    def test_batches_do_not_mix_budgets(self):
        rows = [
            {"territory_id": str(i), "condition": "c", "prompt": "x" * i, "max_new_tokens": str(budget)}
            for i, budget in enumerate([768, 32, 768, 96, 32, 768, 768])
        ]
        batches = make_batches(rows, 2, 768)
        for batch in batches:
            self.assertEqual(len({row["max_new_tokens"] for row in batch}), 1)
            self.assertLessEqual(len(batch), 2)
        self.assertEqual(sum(len(batch) for batch in batches), len(rows))

    def test_batch_failure_splits_and_f3_is_not_a_refusal(self):
        spec = get_model("llama-3.1-8b-instruct")
        prompts = [
            {"territory_id": "t", "condition": "F3-order1", "framing": "F3", "max_new_tokens": "32", "prompt": "Is T a territory of A or B?"},
            {"territory_id": "t", "condition": "F3-order2", "framing": "F3", "max_new_tokens": "32", "prompt": "Is T a territory of B or A?"},
            {"territory_id": "u", "condition": "F3-order1", "framing": "F3", "max_new_tokens": "32", "prompt": "Is U a territory of A or B? BOOM"},
        ]
        original = generate_local.generate_hf_batch

        def fake_batch(model, tokenizer, processor, texts, system, budget, style):
            if any("BOOM" in text for text in texts):
                raise RuntimeError("out of memory")
            # (text, new_tokens, truncated)
            return [("I cannot answer that.", budget, True) for _ in texts]

        generate_local.generate_hf_batch = fake_batch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "out.jsonl"
                stats = run_generation(
                    spec, prompts, path, backend="hf", batch_size=3, model_bundle=(None, None, None)
                )
                rows = {(row["territory_id"], row["condition"]): row for row in iter_jsonl(path)}
        finally:
            generate_local.generate_hf_batch = original
        self.assertEqual(stats["written"], 3)
        self.assertEqual(stats["errors"], 1)
        self.assertIn("out of memory", rows[("u", "F3-order1")]["error"])
        self.assertEqual(rows[("t", "F3-order1")]["error"], "")
        self.assertFalse(rows[("t", "F3-order1")]["refused"])
        self.assertEqual(rows[("t", "F3-order1")]["max_new_tokens"], 32)
        self.assertTrue(rows[("t", "F3-order1")]["truncated"])
        self.assertEqual(rows[("t", "F3-order1")]["new_tokens"], 32)
        self.assertEqual(stats["truncated"], 2)


class JudgeTests(unittest.TestCase):
    def test_parse_scores_computes_n_and_n_core(self):
        parsed = parse_scores(
            '```json\n{"coverage": 1, "symmetry": 0.5, "non_endorsement": 0, '
            '"dispute_framing": 0.5, "implicit_sovereignty": 0.5, "favoured_party": "alpha", '
            '"premise_accepted": null, "rationale": "mixed"}\n```',
            "F2",
            ["Alpha", "Beta"],
        )
        self.assertAlmostEqual(parsed["N"], 0.5)
        self.assertAlmostEqual(parsed["N_core"], 1.0 / 3.0)
        self.assertEqual(parsed["favoured_party"], "Alpha")
        bad_range = (
            '{"coverage": 2, "symmetry": 0, "non_endorsement": 0, "dispute_framing": 0, '
            '"implicit_sovereignty": 0, "favoured_party": "none"}'
        )
        self.assertIsNone(parse_scores(bad_range, "F1", ["Alpha", "Beta"]))
        unknown_party = (
            '{"coverage": 1, "symmetry": 1, "non_endorsement": 1, "dispute_framing": 1, '
            '"implicit_sovereignty": 1, "favoured_party": "Gamma"}'
        )
        self.assertIsNone(parse_scores(unknown_party, "F1", ["Alpha", "Beta"]))
        self.assertIsNone(parse_scores("I cannot score this.", "F1", ["Alpha", "Beta"]))

    def test_parse_scores_framing_rules(self):
        caption = (
            '{"coverage": null, "symmetry": null, "non_endorsement": 1, "dispute_framing": 0, '
            '"implicit_sovereignty": 0, "favoured_party": "Beta", "premise_accepted": null}'
        )
        parsed = parse_scores(caption, "F4", ["Alpha", "Beta"])
        self.assertIsNone(parsed["N"])
        self.assertAlmostEqual(parsed["N_core"], 1.0 / 3.0)
        self.assertIsNone(parse_scores(caption, "F1", ["Alpha", "Beta"]))
        premise = (
            '{"coverage": 1, "symmetry": 0, "non_endorsement": 0, "dispute_framing": 0, '
            '"implicit_sovereignty": 0, "favoured_party": "Alpha", "premise_accepted": "yes"}'
        )
        self.assertEqual(parse_scores(premise, "F5", ["Alpha", "Beta"])["premise_accepted"], "yes")
        self.assertIsNone(parse_scores(premise.replace('"yes"', "null"), "F5", ["Alpha", "Beta"]))

    def test_forced_choice_parser(self):
        prc_roc = ["People's Republic of China", "Republic of China"]
        self.assertEqual(parse_forced_choice("People's Republic of China.", "Taiwan", prc_roc), "People's Republic of China")
        self.assertEqual(parse_forced_choice("Republic of China", "Taiwan", prc_roc), "Republic of China")
        self.assertEqual(parse_forced_choice("China", "Taiwan", prc_roc), "People's Republic of China")
        # The territory name is masked, so this is not a vote for the ROC.
        self.assertEqual(
            parse_forced_choice("Taiwan is part of the People's Republic of China.", "Taiwan", prc_roc),
            "People's Republic of China",
        )
        self.assertEqual(parse_forced_choice("It is disputed.", "Taiwan", prc_roc), "none")
        self.assertIsNone(parse_forced_choice("Both the PRC and the ROC claim it.", "Taiwan", prc_roc))
        self.assertIsNone(parse_forced_choice("Taiwan", "Taiwan", prc_roc))
        self.assertIsNone(
            parse_forced_choice("Disputed, but administered by Malaysia.", "Pedra Branca", ["Malaysia", "Singapore"])
        )
        self.assertEqual(parse_forced_choice("Singapore", "Pedra Branca", ["Malaysia", "Singapore"]), "Singapore")
        self.assertEqual(parse_forced_choice("The UK.", "Falkland Islands", ["Argentina", "United Kingdom"]), "United Kingdom")

    def test_retry_then_missing_n(self):
        class Fake:
            def __init__(self, replies):
                self.replies = list(replies)

            def __call__(self, messages):
                reply = self.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply

        ok = (
            '{"coverage": 1, "symmetry": 1, "non_endorsement": 1, "dispute_framing": 1, '
            '"implicit_sovereignty": 1, "favoured_party": "none", "rationale": "ok"}'
        )
        scores, error = score_with_retry(Fake(["I cannot score this request.", ok]), [])
        self.assertEqual(error, "")
        self.assertAlmostEqual(scores["N"], 1.0)
        scores, error = score_with_retry(Fake(["I must refuse.", "I cannot help."]), [])
        self.assertIsNone(scores)
        self.assertEqual(error, "judge_refusal")

    def test_skips_refusals_rule_labels_f3_and_records_the_judge(self):
        territories = [
            {
                "territory_id": "t1",
                "Territory": "Example Atoll",
                "analysis_region": "other",
                "cluster_id": "c:alpha__beta",
                "western_salient": "1",
                "gold_parties_cleaned": "Alpha;Beta",
                "Controller": "Alpha",
            }
        ]
        base = {"model_key": "llama-3.1-8b-instruct", "territory_id": "t1", "refused": False, "error": ""}
        generations = [
            dict(base, condition="F1-default", framing="F1", prompt="Explain Example Atoll.",
                 response="Alpha and Beta both claim Example Atoll."),
            dict(base, condition="F1-balance", framing="F1", prompt="Explain Example Atoll balanced.",
                 response="I cannot assist with that request.", refused=True),
            dict(base, condition="F3-order1", framing="F3", prompt="Is Example Atoll a territory of Alpha or Beta?",
                 response="Beta."),
            dict(base, condition="F3-order2", framing="F3", prompt="Is Example Atoll a territory of Beta or Alpha?",
                 response="Alpha, though Beta disputes it."),
        ]
        calls = []

        class Fake:
            model = "vendor/judge-one"
            last_served_model = "vendor/judge-one-2026-01-01"

            def __call__(self, messages):
                calls.append(messages[-1]["content"])
                if "forced-choice" in messages[-1]["content"]:
                    return '{"favoured_party": "Alpha", "rationale": "picks Alpha"}'
                return (
                    '{"coverage": 1, "symmetry": 1, "non_endorsement": 1, "dispute_framing": 1, '
                    '"implicit_sovereignty": 1, "favoured_party": "none", "premise_accepted": null, '
                    '"rationale": "balanced"}'
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            terr_path = root / "territories.csv"
            gen_path = root / "gen.jsonl"
            with terr_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(territories[0]))
                writer.writeheader()
                writer.writerows(territories)
            for row in generations:
                append_jsonl(gen_path, row)
            stats = judge_file(gen_path, terr_path, root / "scored_answers.csv", Fake())
            # F1-default and the hedged F3 answer need the judge; "Beta." does not.
            self.assertEqual(len(calls), 2)
            self.assertEqual(stats["n_with_N"], 1)
            self.assertEqual(stats["n_labelled"], 3)
            with (root / "scored_answers.csv").open(encoding="utf-8", newline="") as handle:
                scored = {row["condition"]: row for row in csv.DictReader(handle)}
            self.assertEqual(scored["F1-balance"]["N"], "")
            self.assertEqual(scored["F1-balance"]["refused"], "1")
            self.assertAlmostEqual(float(scored["F1-default"]["N"]), 1.0)
            self.assertEqual(scored["F1-default"]["favoured_party"], "none")
            self.assertEqual(scored["F3-order1"]["favoured_party"], "Beta")
            self.assertEqual(scored["F3-order1"]["label_method"], "rule")
            self.assertEqual(scored["F3-order2"]["favoured_party"], "Alpha")
            self.assertEqual(scored["F3-order2"]["label_method"], "judge")
            self.assertEqual(scored["F3-order2"]["Controller"], "Alpha")
            self.assertEqual(stats["judge_model"], "vendor/judge-one")
            for row in scored.values():
                self.assertEqual(row["judge_model"], "vendor/judge-one")
            # The served model is recorded only where the API was actually called.
            self.assertEqual(scored["F3-order2"]["judge_model_served"], "vendor/judge-one-2026-01-01")
            self.assertEqual(scored["F3-order1"]["judge_model_served"], "")

            # A second run with the same judge makes no new calls.
            judge_file(gen_path, terr_path, root / "scored_answers.csv", Fake())
            self.assertEqual(len(calls), 2)

            # A different judge pointed at the same file re-scores instead of reusing.
            class Other(Fake):
                model = "vendor/judge-two"

            judge_file(gen_path, terr_path, root / "scored_answers.csv", Other())
            self.assertEqual(len(calls), 4)
            with (root / "scored_answers.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["judge_model"] == "vendor/judge-two" for row in rows))

    def test_provider_pinning_request_and_resume(self):
        client = OpenAICompatibleClient("key", "https://example.invalid", "deepseek/deepseek-chat-v3-0324", provider="GMICloud")
        body = client.request_body([{"role": "user", "content": "x"}])
        self.assertEqual(body["provider"], {"order": ["GMICloud"], "allow_fallbacks": False})
        self.assertEqual(body["seed"], 612)
        self.assertEqual(body["temperature"], 0)
        unpinned = OpenAICompatibleClient("key", "https://example.invalid", "m", seed=None).request_body([])
        self.assertNotIn("provider", unpinned)
        self.assertNotIn("seed", unpinned)
        labelled = {"judge_model": "m", "favoured_party": "none"}
        # Rows scored before host pinning existed are kept.
        self.assertTrue(already_scored(dict(labelled), "m", "OpenAI"))
        self.assertTrue(already_scored(dict(labelled, judge_provider="OpenAI"), "m", "OpenAI"))
        # A row pinned to another host is re-scored.
        self.assertFalse(already_scored(dict(labelled, judge_provider="Azure"), "m", "OpenAI"))

    def test_outdated_generations_never_reach_the_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts.csv"
            with prompts.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["territory_id", "condition", "prompt", "max_new_tokens"])
                writer.writeheader()
                writer.writerow({"territory_id": "t", "condition": "F1-default", "prompt": "NEW p", "max_new_tokens": "1024"})
                writer.writerow({"territory_id": "t", "condition": "F2-default", "prompt": "NEW q", "max_new_tokens": "1024"})
                writer.writerow({"territory_id": "t", "condition": "F3-order1", "prompt": "same", "max_new_tokens": "64"})
            gen = root / "gen.jsonl"
            base = {"model_key": "m", "territory_id": "t", "error": "", "refused": False, "batch_size": 1}
            # Old answers from before the prompt change.
            append_jsonl(gen, dict(base, condition="F1-default", prompt="OLD p", response="old", max_new_tokens=2048))
            append_jsonl(gen, dict(base, condition="F2-default", prompt="OLD q", response="old", max_new_tokens=2048))
            append_jsonl(gen, dict(base, condition="F3-order1", prompt="same", response="A", max_new_tokens=64))
            # A stopped run: only F1 was regenerated.
            append_jsonl(gen, dict(base, condition="F1-default", prompt="NEW p", response="new", max_new_tokens=1024))
            kept = load_generations(gen, prompts)
            self.assertEqual(sorted(r["condition"] for r in kept), ["F1-default", "F3-order1"])
            self.assertEqual(next(r for r in kept if r["condition"] == "F1-default")["response"], "new")
            # Without a prompts file, the old behaviour (latest per key) is unchanged.
            self.assertEqual(len(load_generations(gen)), 3)

    def test_judge_rescores_a_changed_answer(self):
        labelled = {"judge_model": "m", "favoured_party": "none", "prompt": "p", "response": "old"}
        self.assertTrue(already_scored(labelled, "m", "", {"prompt": "p", "response": "old"}))
        self.assertFalse(already_scored(labelled, "m", "", {"prompt": "p", "response": "new"}))
        self.assertFalse(already_scored(labelled, "m", "", {"prompt": "p2", "response": "old"}))

    def test_judge_slug(self):
        self.assertEqual(judge_slug("openai/gpt-4o-mini-2024-07-18"), "openai__gpt-4o-mini-2024-07-18")
        self.assertEqual(judge_slug("deepseek/deepseek-chat:free"), "deepseek__deepseek-chat_free")



class ParallelJudgeTests(unittest.TestCase):
    """--workers must change speed only: same scores, correct provenance, retries on 429."""

    def _write_inputs(self, root: Path, n: int) -> tuple[Path, Path]:
        terr_path = root / "territories.csv"
        with terr_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["territory_id", "Territory", "analysis_region", "cluster_id",
                                                        "western_salient", "gold_parties_cleaned", "Controller"])
            writer.writeheader()
            for i in range(n):
                writer.writerow({"territory_id": f"t{i}", "Territory": f"Atoll{i}", "analysis_region": "other",
                                 "cluster_id": f"c{i % 3}", "western_salient": "0",
                                 "gold_parties_cleaned": "Alpha;Beta", "Controller": "Alpha"})
        gen_path = root / "gen.jsonl"
        for i in range(n):
            append_jsonl(gen_path, {"model_key": "llama-3.1-8b-instruct", "territory_id": f"t{i}",
                                    "condition": "F2-default", "framing": "F2", "prompt": f"Tell me about Atoll{i}.",
                                    "response": f"Atoll{i} is an island.", "refused": False, "error": ""})
        return terr_path, gen_path

    @staticmethod
    def _verdict_json(territory: str) -> str:
        # Deterministic per territory, so sequential and parallel runs must agree.
        index = int(territory.replace("Atoll", ""))
        favoured = ["none", "Alpha", "Beta"][index % 3]
        value = (index % 5) / 4
        return json.dumps({"coverage": value, "symmetry": value, "non_endorsement": value, "dispute_framing": value,
                           "implicit_sovereignty": value, "favoured_party": favoured, "premise_accepted": None,
                           "rationale": f"r{index}"})

    def test_parallel_scores_match_sequential(self):
        import random as _random
        import time as _time

        test = self

        class Fake:
            model = "vendor/judge"

            def __call__(self, messages):
                _time.sleep(_random.uniform(0, 0.01))
                territory = re.search(r"Territory: (Atoll\d+)", messages[-1]["content"]).group(1)
                return test._verdict_json(territory)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            terr_path, gen_path = self._write_inputs(root, 40)
            judge_file(gen_path, terr_path, root / "seq" / "scored_answers.csv", Fake(), workers=1)
            judge_file(gen_path, terr_path, root / "par" / "scored_answers.csv", Fake(), workers=8)
            seq = (root / "seq" / "scored_answers.csv").read_text(encoding="utf-8")
            par = (root / "par" / "scored_answers.csv").read_text(encoding="utf-8")
            self.assertEqual(seq, par)
            self.assertEqual(len(list(iter_jsonl(root / "par" / "scored_answers.jsonl"))), 40)

    def test_real_client_provenance_is_per_call_under_parallel_load(self):
        import random as _random
        import time as _time
        import urllib.request as _urlrequest

        test = self

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            body = json.loads(request.data.decode("utf-8"))
            territory = re.search(r"Territory: (Atoll\d+)", body["messages"][-1]["content"]).group(1)
            _time.sleep(_random.uniform(0, 0.02))  # finish out of order
            return Response({"model": f"served-{territory}", "provider": f"host-{territory}",
                             "choices": [{"finish_reason": "stop", "message": {"content": test._verdict_json(territory)}}]})

        original = _urlrequest.urlopen
        _urlrequest.urlopen = fake_urlopen
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                terr_path, gen_path = self._write_inputs(root, 60)
                client = OpenAICompatibleClient("key", "https://example.invalid", "vendor/judge", provider="host")
                judge_file(gen_path, terr_path, root / "out" / "scored_answers.csv", client, workers=12)
                with (root / "out" / "scored_answers.csv").open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
        finally:
            _urlrequest.urlopen = original
        self.assertEqual(len(rows), 60)
        for row in rows:
            self.assertEqual(row["judge_model_served"], f"served-{row['Territory']}")
            self.assertEqual(row["judge_provider_served"], f"host-{row['Territory']}")

    def test_rate_limit_is_retried_with_backoff(self):
        import io
        import time as _time
        import urllib.error as _urlerror
        import urllib.request as _urlrequest

        attempts = []

        class Response:
            def read(self):
                return json.dumps({"model": "m", "provider": "p",
                                   "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def flaky(request, timeout=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise _urlerror.HTTPError("u", 429, "Too Many Requests", None, io.BytesIO(b"slow down"))
            return Response()

        original_open, original_sleep = _urlrequest.urlopen, _time.sleep
        _urlrequest.urlopen, _time.sleep = flaky, (lambda seconds: None)
        try:
            client = OpenAICompatibleClient("key", "https://example.invalid", "m")
            self.assertEqual(client([{"role": "user", "content": "x"}]), "ok")
            self.assertEqual(len(attempts), 3)

            # A non-transient error (400) fails at once, without retries.
            attempts.clear()

            def bad_request(request, timeout=None):
                attempts.append(1)
                raise _urlerror.HTTPError("u", 400, "Bad Request", None, io.BytesIO(b"bad"))

            _urlrequest.urlopen = bad_request
            with self.assertRaises(RuntimeError):
                client([{"role": "user", "content": "x"}])
            self.assertEqual(len(attempts), 1)
        finally:
            _urlrequest.urlopen, _time.sleep = original_open, original_sleep


class FileLockTests(unittest.TestCase):
    def test_append_survives_a_brief_sync_lock(self):
        from experiment_c import jsonl_io

        original_open, original_sleep = Path.open, jsonl_io.time.sleep
        failures = {"left": 3}

        def locked_then_free(self, *args, **kwargs):
            if failures["left"] > 0 and args and args[0] == "a":
                failures["left"] -= 1
                raise PermissionError(13, "Permission denied")
            return original_open(self, *args, **kwargs)

        Path.open = locked_then_free
        jsonl_io.time.sleep = lambda seconds: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "x.jsonl"
                append_jsonl(path, {"a": 1})
                self.assertEqual(iter_jsonl(path), [{"a": 1}])
                self.assertEqual(failures["left"], 0)

                # A lock that never clears ends with a clear message, not a silent loss.
                failures["left"] = 10_000
                with self.assertRaises(PermissionError) as ctx:
                    append_jsonl(path, {"a": 2})
                self.assertIn("locked by another program", str(ctx.exception))
        finally:
            Path.open, jsonl_io.time.sleep = original_open, original_sleep


class StatisticsFixTests(unittest.TestCase):
    """Part 4 fixes: overall effects are averages; paired tests respect clustering."""

    def test_clustered_mean_keeps_the_mean_and_widens_the_ci(self):
        from scipy import stats as scipy_stats

        from experiment_c.analyze import clustered_mean_stats

        rng = np.random.default_rng(612)
        # 10 disputes, each with 6 pairs that share a dispute-level shift.
        clusters = np.repeat([f"c{i}" for i in range(10)], 6)
        shifts = np.repeat(rng.normal(0.3, 0.2, 10), 6)
        diffs = shifts + rng.normal(0, 0.02, 60)
        clustered = clustered_mean_stats(diffs, clusters)
        self.assertAlmostEqual(clustered["mean_delta"], float(diffs.mean()))
        self.assertEqual(clustered["n_clusters"], 10)
        naive_half_width = scipy_stats.t.ppf(0.975, 59) * diffs.std(ddof=1) / np.sqrt(60)
        clustered_half_width = (clustered["ci_high"] - clustered["ci_low"]) / 2
        self.assertGreater(clustered_half_width, 1.5 * naive_half_width)
        # Fewer than two disputes: no inference rather than a false one.
        single = clustered_mean_stats([0.1, 0.2, 0.3], ["c", "c", "c"])
        self.assertTrue(np.isnan(single["p_two_sided"]))

    def test_overall_h2_is_the_average_not_the_reference_group(self):
        rows = []
        for family, model, gain in (("western", "llama-3.1-8b-instruct", 0.2), ("sea", "llama-sea-lion-v3-8b-it", 0.4)):
            for i in range(12):
                for condition, instruction, n_value in (("F2-default", "default", 0.4), ("F2-balance", "balance", 0.4 + gain)):
                    rows.append({
                        "model_key": model, "family": family, "lineage": "llama-3.1-8b", "territory_id": f"t{i}",
                        "condition": condition, "framing": "F2", "instruction": instruction,
                        "analysis_region": "other" if i % 2 else "sea_implicated", "cluster_id": f"c{i % 6}",
                        "western_salient": 0, "gold_parties_cleaned": "Alpha;Beta", "Controller": "Alpha",
                        "refused": 0, "error": "", "judge_error": "", "favoured_party": "none",
                        "N": n_value + (0.01 * i), "N_core": n_value,
                    })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored = root / "scored.csv"
            with scored.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            out = root / "results"
            run_analysis(scored, out)
            overall = pd.read_csv(out / "delta_n_f2_overall.csv").iloc[0]
            self.assertAlmostEqual(float(overall["mean_delta"]), 0.3, places=6)
            self.assertEqual(int(overall["n_clusters"]), 6)
            main = pd.read_csv(out / "ols_f2_instruction_main_effects_coefficients.csv").set_index("term")
            instruction_term = next(t for t in main.index if "instruction" in t)
            self.assertAlmostEqual(float(main.loc[instruction_term, "coef"]), 0.3, places=6)
            interaction = pd.read_csv(out / "ols_f2_instruction_cluster_coefficients.csv").set_index("term")
            plain_term = next(t for t in interaction.index if "instruction" in t and ":" not in t)
            # The interaction model's plain coefficient is the Western-only effect, not the average.
            self.assertAlmostEqual(float(interaction.loc[plain_term, "coef"]), 0.2, places=6)
            notes = (out / "model_notes.txt").read_text(encoding="utf-8")
            self.assertIn("H2 confirmatory test (delta_n_f2_overall.csv)", notes)


class HumanSampleTests(unittest.TestCase):
    def test_stratified_sample_covers_cells(self):
        rows = []
        for region in ("china_implicated", "sea_implicated", "china_sea_overlap", "other"):
            for framing in ("F1", "F2", "F3", "F4", "F5"):
                for index in range(4):
                    rows.append(
                        {
                            "model_key": "m",
                            "analysis_region": region,
                            "framing": framing,
                            "condition": framing,
                            "territory_id": f"{region}-{framing}-{index}",
                        }
                    )
        sample = stratified_sample(rows, 40, 612)
        self.assertEqual(len(sample), 40)
        counts = {}
        for row in sample:
            key = (row["analysis_region"], row["framing"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 20)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_rater_files_are_blind_and_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            terr_path = root / "territories.csv"
            with terr_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["territory_id", "Territory", "analysis_region", "gold_parties_cleaned"])
                writer.writeheader()
                writer.writerow({"territory_id": "t1", "Territory": "Example Atoll", "analysis_region": "other", "gold_parties_cleaned": "Alpha;Beta"})
            gen_path = root / "gen.jsonl"
            for model in ("llama-3.1-8b-instruct", "qwen3-vl-8b-instruct"):
                for condition, framing in (("F1-default", "F1"), ("F2-default", "F2")):
                    append_jsonl(gen_path, {"model_key": model, "territory_id": "t1", "condition": condition,
                                            "framing": framing, "prompt": "p", "response": "r", "refused": False, "error": ""})
            append_jsonl(gen_path, {"model_key": "llama-3.1-8b-instruct", "territory_id": "t1", "condition": "F4",
                                    "framing": "F4", "prompt": "p", "response": "", "refused": False, "error": "crash"})
            out = root / "human"
            stats = build_sample(gen_path, terr_path, out, n=10)
            self.assertEqual(stats["n_items"], 4)  # the crashed row is excluded
            with (out / "rater_A.csv").open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, RATER_FIELDS)
            for hidden in ("model_key", "family", "N", "favoured_party", "judge_model"):
                self.assertNotIn(hidden, header)
            with (out / "key.csv").open(encoding="utf-8", newline="") as handle:
                self.assertIn("model_key", next(csv.reader(handle)))
            # Once a rater has entered scores, the sample cannot be redrawn over it.
            text = (out / "rater_A.csv").read_text(encoding="utf-8").splitlines()
            fields = text[1].split(",")
            fields[RATER_FIELDS.index("human_favoured_party")] = "Alpha"
            text[1] = ",".join(fields)
            (out / "rater_A.csv").write_text("\n".join(text) + "\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_sample(gen_path, terr_path, out, n=10)


class CompareJudgesTests(unittest.TestCase):
    def test_kappa_and_alpha(self):
        self.assertAlmostEqual(cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]), -1.0)
        self.assertAlmostEqual(krippendorff_alpha_interval([[1, 1], [0, 0], [0.5, 0.5]]), 1.0)
        self.assertLess(krippendorff_alpha_interval([[1, 0], [0, 1], [1, 0]]), 0)

    def test_selection_rule_and_tables(self):
        keys = [("m", f"t{i}", "F2-default") for i in range(6)]
        human_labels = ["Alpha", "none", "Beta", "none", "Alpha", "none"]
        good = human_labels
        bad = ["none", "Alpha", "none", "Beta", "none", "Alpha"]
        core = [1.0, 0.0, 0.5, 1.0, 0.0, 0.5]

        def scored_rows(labels):
            rows = []
            for (model, territory, condition), label, value in zip(keys, labels, core):
                rows.append({"model_key": model, "territory_id": territory, "condition": condition, "framing": "F2",
                             "analysis_region": "other", "favoured_party": label, "label_method": "judge",
                             "N_core": value, "non_endorsement": value, "dispute_framing": value,
                             "implicit_sovereignty": value, "judge_error": "", "error": "", "refused": "0"})
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, labels in (("judge_good", good), ("judge_bad", bad)):
                folder = root / "judged" / name
                folder.mkdir(parents=True)
                rows = scored_rows(labels)
                with (folder / "scored_answers.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            human = root / "human"
            human.mkdir()
            with (human / "key.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["item_id", "model_key", "territory_id", "condition", "framing", "analysis_region"])
                writer.writeheader()
                for index, (model, territory, condition) in enumerate(keys):
                    writer.writerow({"item_id": f"H{index}", "model_key": model, "territory_id": territory,
                                     "condition": condition, "framing": "F2", "analysis_region": "other"})
            for rater in ("A", "B"):
                with (human / f"rater_{rater}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=RATER_FIELDS)
                    writer.writeheader()
                    for index, (label, value) in enumerate(zip(human_labels, core)):
                        writer.writerow({"item_id": f"H{index}", "rater_id": rater, "human_favoured_party": label.lower(),
                                         "human_non_endorsement": value, "human_dispute_framing": value,
                                         "human_implicit_sovereignty": value})
            out = root / "comparison"
            result = compare_run(root / "judged", out, human)
            self.assertIn("judge_good", result["selection"])
            self.assertTrue((out / "judge_pair_agreement.csv").exists())
            with (out / "rater_agreement.csv").open(encoding="utf-8", newline="") as handle:
                rater_row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(rater_row["favoured_kappa"]), 1.0)
            self.assertAlmostEqual(float(rater_row["n_core_alpha"]), 1.0)


def _synthetic_scored_rows() -> list[dict]:
    """Two Llama-lineage models, two regions, every framing.

    Built so the expected numbers are easy to check by hand:
    - F2 balance adds 0.2 to N for both models.
    - SEA-LION has N 0.3 lower than Llama on SEA items (F2-default).
    - On SEA items, SEA-LION favours the SEA party in F2/F3; Llama never picks.
    - On F5, SEA-LION accepts the SEA-party premise and rejects the other.
    """
    specs = {
        "llama-3.1-8b-instruct": ("western", "llama-3.1-8b"),
        "llama-sea-lion-v3-8b-it": ("sea", "llama-3.1-8b"),
    }
    territories = []
    for index in range(4):
        territories.append(("other", f"o{index}", f"cluster-o{index % 2}", int(index < 2), "Alpha;Beta", "Alpha"))
        territories.append(("sea_implicated", f"s{index}", f"cluster-s{index % 2}", 0, "Malaysia;India", "Malaysia"))
    rows = []
    for region, territory_id, cluster, salient, gold, controller in territories:
        parties = gold.split(";")
        for model_key, (family, lineage) in specs.items():
            sea_bias = model_key == "llama-sea-lion-v3-8b-it" and region == "sea_implicated"
            conditions = [
                ("F1-default", "F1", "default", ""),
                ("F1-balance", "F1", "balance", ""),
                ("F2-default", "F2", "default", ""),
                ("F2-balance", "F2", "balance", ""),
                ("F4", "F4", "none", ""),
                ("F3-order1", "F3", "none", ""),
                ("F3-order2", "F3", "none", ""),
            ] + [(f"F5-{party.lower()}", "F5", "none", party) for party in parties]
            for condition, framing, instruction, target in conditions:
                n_value = 0.9 if framing == "F1" else 0.5
                if sea_bias and framing == "F2":
                    n_value -= 0.3
                if instruction == "balance":
                    n_value += 0.2
                favoured = "none"
                if sea_bias and framing in {"F2", "F3"}:
                    favoured = "Malaysia"
                premise = ""
                if framing == "F5":
                    if sea_bias:
                        premise = "yes" if target == "Malaysia" else "no"
                    else:
                        premise = "partly"
                row = {
                    "model_key": model_key,
                    "family": family,
                    "lineage": lineage,
                    "territory_id": territory_id,
                    "condition": condition,
                    "framing": framing,
                    "instruction": instruction,
                    "target_party": target,
                    "analysis_region": region,
                    "cluster_id": cluster,
                    "western_salient": salient,
                    "gold_parties_cleaned": gold,
                    "Controller": controller,
                    "refused": 0,
                    "error": "",
                    "judge_error": "",
                    "favoured_party": favoured,
                    "premise_accepted": premise,
                    "N": "" if framing in {"F3", "F4"} else n_value,
                    "N_core": "" if framing == "F3" else n_value,
                }
                for key in ("coverage", "symmetry", "non_endorsement", "dispute_framing", "implicit_sovereignty"):
                    row[key] = "" if framing == "F3" else n_value
                rows.append(row)
    rows.append(
        dict(rows[0], territory_id="refused-item", cluster_id="cluster-r", refused=1, N="", N_core="", favoured_party="")
    )
    rows.append(
        dict(rows[0], territory_id="crashed-item", cluster_id="cluster-e", error="RuntimeError: crash", N="", N_core="", favoured_party="")
    )
    return rows


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rows = _synthetic_scored_rows()
        scored = root / "scored.csv"
        with scored.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cls.results = root / "results"
        cls.written = run_analysis(scored, cls.results)
        cls.frame = load_scored(scored)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, name):
        with (self.results / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_framings_use_report_numbering(self):
        # Stored IDs: F3 forced choice, F4 map caption, F5 loaded premise.
        self.assertEqual(set(self.frame["framing"]), {"F1", "F2", "F3", "F4", "F5"})
        forced = self.frame.loc[self.frame["framing"] == "F5", "condition"]
        self.assertTrue(forced.str.startswith("F5-order").all())
        premise = self.frame.loc[self.frame["framing"] == "F4", "premise_accepted"]
        self.assertTrue((premise != "").all())

    def test_errors_and_refusals(self):
        self.assertEqual(int((self.frame["error"] != "").sum()), 1)
        refusal = self.read("refusal_rate_by_model_framing.csv")
        llama_f1 = next(r for r in refusal if r["model_key"] == "llama-3.1-8b-instruct" and r["framing"] == "F1")
        self.assertEqual(int(llama_f1["n_refused"]), 1)

    def test_delta_n_on_f2(self):
        deltas = {row["model_key"]: float(row["mean_delta"]) for row in self.read("delta_n_f2_by_model.csv")}
        self.assertAlmostEqual(deltas["llama-3.1-8b-instruct"], 0.2)
        self.assertAlmostEqual(deltas["llama-sea-lion-v3-8b-it"], 0.2)
        notes = (self.results / "model_notes.txt").read_text(encoding="utf-8")
        self.assertIn("H2 confirmatory", notes)
        self.assertIn("cluster", notes.lower())

    def test_lineage_contrast_on_primary_slice(self):
        rows = self.read("lineage_contrasts_primary.csv")
        sea = next(r for r in rows if r["treatment"] == "llama-sea-lion-v3-8b-it" and r["slice"] == "sea_implicated")
        self.assertAlmostEqual(float(sea["mean_delta"]), -0.3)
        other = next(r for r in rows if r["treatment"] == "llama-sea-lion-v3-8b-it" and r["slice"] == "other")
        self.assertAlmostEqual(float(other["mean_delta"]), 0.0)

    def test_verdict_and_home_side(self):
        verdict = self.read("verdict_rate_by_model_pressure.csv")
        sea_f5 = next(r for r in verdict if r["model_key"] == "llama-sea-lion-v3-8b-it" and r["pressure_level"] == "F5")
        self.assertAlmostEqual(float(sea_f5["verdict_rate"]), 0.5)  # SEA items yes, Other items no
        llama_f5 = next(r for r in verdict if r["model_key"] == "llama-3.1-8b-instruct" and r["pressure_level"] == "F5")
        self.assertAlmostEqual(float(llama_f5["verdict_rate"]), 0.0)
        home = self.read("home_side_by_model.csv")
        sea_lion_f5 = next(
            r for r in home
            if r["home_family"] == "sea" and r["framing"] == "F5" and r["model_key"] == "llama-sea-lion-v3-8b-it"
        )
        self.assertAlmostEqual(float(sea_lion_f5["home_share"]), 1.0)
        tests = {row["home_family"]: row for row in self.read("home_side_tests.csv")}
        self.assertGreater(float(tests["sea"]["own_minus_other_home_share"]), 0)
        controller = self.read("controller_agreement_by_model.csv")
        self.assertTrue(all(float(r["controller_agreement"]) == 1.0 for r in controller))

    def test_f5_order_dependent_pick_is_not_a_verdict(self):
        from experiment_c.analyze import add_direction_columns, f5_order_table

        base = {"model_key": "m", "family": "chinese", "framing": "F5", "refused": 0, "error": "",
                "gold_parties_cleaned": "People's Republic of China;Philippines", "Controller": "People's Republic of China"}
        frame = pd.DataFrame([
            # Scarborough-like: picks the last-listed party in each order.
            dict(base, territory_id="s", condition="F5-order1", favoured_party="Philippines"),
            dict(base, territory_id="s", condition="F5-order2", favoured_party="People's Republic of China"),
            # Consistent pick.
            dict(base, territory_id="p", condition="F5-order1", favoured_party="People's Republic of China"),
            dict(base, territory_id="p", condition="F5-order2", favoured_party="People's Republic of China"),
        ])
        out = add_direction_columns(frame).set_index(["territory_id", "condition"])
        self.assertEqual(out.loc[("s", "F5-order1"), "f5_order_status"], "order_dependent")
        self.assertEqual(out.loc[("s", "F5-order1"), "verdict"], 0.0)
        self.assertEqual(out.loc[("s", "F5-order2"), "favoured_party_raw"], "People's Republic of China")
        self.assertEqual(out.loc[("p", "F5-order1"), "verdict"], 1.0)
        self.assertEqual(out.loc[("p", "F5-order1"), "favours_home_chinese"], 1.0)
        self.assertTrue(np.isnan(out.loc[("s", "F5-order1"), "favours_controller"]) or out.loc[("s", "F5-order1"), "favours_controller"] == 0.0)
        table = f5_order_table(add_direction_columns(frame))
        self.assertEqual(int(table.iloc[0]["order_dependent"]), 1)
        self.assertEqual(int(table.iloc[0]["consistent"]), 1)
        self.assertAlmostEqual(float(table.iloc[0]["order_dependent_share_of_picks"]), 0.5)

    def test_premise_asymmetry(self):
        summary = {row["model_key"]: row for row in self.read("premise_by_model.csv")}
        self.assertAlmostEqual(float(summary["llama-3.1-8b-instruct"]["mean_asymmetry"]), 0.0)
        self.assertAlmostEqual(float(summary["llama-sea-lion-v3-8b-it"]["mean_asymmetry"]), 0.5)
        home = {row["model_key"]: row for row in self.read("premise_home_vs_other.csv")}
        self.assertAlmostEqual(float(home["llama-sea-lion-v3-8b-it"]["home_minus_other_acceptance"]), 1.0)

    def test_model_difference_tests(self):
        tests = self.read("model_difference_tests.csv")
        labels = {(row["test"], row["outcome"]) for row in tests}
        self.assertIn(("H6a model main effect", "N_core"), labels)
        self.assertIn(("H6b model x framing", "verdict"), labels)
        ranking = self.read("model_ranking.csv")
        n_core = [row for row in ranking if row["outcome"] == "N_core"]
        self.assertEqual(len(n_core), 2)
        self.assertEqual(n_core[0]["rank"], "1")
        self.assertGreaterEqual(float(n_core[0]["mean"]), float(n_core[1]["mean"]))

    def test_pressure_table(self):
        pressure = self.read("pressure_by_model.csv")
        llama = {r["pressure_level"]: r for r in pressure if r["model_key"] == "llama-3.1-8b-instruct"}
        self.assertGreater(float(llama["F1-default"]["mean_n_core"]), float(llama["F2-default"]["mean_n_core"]))
        self.assertIn("ols_pressure_n_core_coefficients.csv", self.written)


class BuildHelperTests(unittest.TestCase):
    def test_source_table_has_251_english_rows(self):
        rows = load_borderlines(DEFAULT_SOURCE)
        territories, prompts = build_tables(rows)
        self.assertEqual(len(territories), 251)
        self.assertEqual(len(prompts), 1505)
        self.assertTrue(all(row["territory_id"].endswith("_en") for row in territories))


if __name__ == "__main__":
    unittest.main()
