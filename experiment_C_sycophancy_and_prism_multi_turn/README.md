# CS612: Multi-Turn Adversarial Sycophancy & PRISM Defense

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker Certified](https://img.shields.io/badge/Docker-Verified%20Pass-brightgreen.svg)](#)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

This repository contains the evaluation suite, model weights, and mitigation pipelines for our CS612 final report:
* **Section 4:** *[Experiment C] Context Pollution (Sycophancy)*
* **Section 6:** *Proposed Mitigations to Improve Safety: The PRISM Framework*

---

## 1. Quickstart

### Option A: Run via Docker (Recommended, Zero Setup)
```bash
# 1. Clone repository
git clone https://github.com/siranishkd/cs612-political-neutrality-eval.git
cd cs612-political-neutrality-eval

# 2. Build and run (completes smoke test, simulation, defense & generates figures)
docker build -t cs612-eval-suite .
docker run --rm cs612-eval-suite
```

### Option B: Run Locally
```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run verification & deterministic reproduction
python3 verify_repo.py
./run_all.sh mock
```

---

## 2. Repository Layout

```text
cs612-political-neutrality-eval/
├── system_c_sycophancy/               # Section 4: Sycophancy Evaluation
│   ├── src/                           # multi_turn_simulator.py, run_tier2_judge.py, generate_paper_figures.py
│   ├── models/                        # Calibrated BERT stance model (weights tracked via Git LFS)
│   ├── data/                          # 20 controversial topics x 3 personas
│   └── outputs/                       # 1,800 turns simulation logs & Tier-2 judge audits
│
├── prism_defense/                     # Section 6: PRISM Defense Framework
│   ├── src/                           # prism_defense.py, evaluate_prism_mitigations.py
│   └── outputs/                       # Defended simulation transcripts & mitigation summaries
│
├── common/                            # Shared inference client, text utils & config
├── figures/                           # Publication-grade figures (Figures 2, 3, 4, 5, 6)
├── docs/                              # Technical reports, scoring methodology & rubrics
├── Dockerfile                         # Production Docker container
├── run_all.sh                         # Master test & reproduction runner
└── verify_repo.py                     # 5-stage automated integrity auditor
```

---

## 3. Overview of Experiments

### Section 4: Multi-Turn Adversarial Sycophancy (Experiment C)
* **Goal:** Evaluate how foundation models yield under 10 turns of escalating ideological pressure.
* **Models Evaluated (3):** Meta Llama-3.1-8B-Instruct, Google Gemma-3-12b-it, Alibaba Qwen3-VL-8B-Instruct.
* **Volume:** 20 topics $\times$ 3 personas (Liberal, Conservative, Control) $\times$ 10 turns = 600 turns/model (**1,800 turns total**).
* **Two-Tier Evaluation:**
  1. *Tier 1 (BERT Stance Scorer):* Calibrated neural classifier mapping model outputs to continuous stance ($S \in [-1.0, +1.0]$).
  2. *Tier 2 (LLM Judge Audit):* Decouples surface politeness (1–5) from substantive epistemic surrender (1–5).

### Section 6: PRISM Defense Framework
* **Goal:** Mitigate sycophantic drift without degrading conversational politeness.
* **Architecture:**
  1. *Tier 1 (Constitutional Prompting):* Enforces 5 Articles of Epistemic Neutrality.
  2. *Tier 2 (Reflection Critic):* Intercepts ideological surrender and triggers balanced rewrites.

---

## 4. Model Weights & Provenance Note

* **Trained Weights Included:** The calibrated BERT stance classifier (`model.safetensors`, 418 MB) is packaged directly in `system_c_sycophancy/models/final_calibrated_model/` and tracked via Git LFS. No retraining is required to run the evaluation.
* **SharePoint Archive:** For complete training provenance, the raw AllSides corpus, 1,280 PARETO goldenset pairs, and training scripts are archived on the team's SharePoint at `2_data/1_bert_3_classifier/`.

---

## 5. License
Distributed under the Apache-2.0 License.
