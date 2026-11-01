# Evaluation Architecture: Models, Providers & Sourcing Reference
**Module:** CS612 AI System Evaluation  
**Project:** Political Neutrality, Sycophancy & Locality Biases in Open-Weight LLMs  
**File Location:** `docs/MODELS_AND_INFRASTRUCTURE.md`  
**Authors:** CS612 Research Group 3, Singapore Management University (SMU)  

---

## 1. Architectural Taxonomy Overview

In our experimental setup, we strictly distinguish three categories of models:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             SYSTEM TAXONOMY                              │
├────────────────────────────────┬─────────────────────────────────────────┤
│ 1. TARGET MODELS               │ • Open-weight LLMs being evaluated     │
│    (Subjects of Evaluation)    │ • Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B │
├────────────────────────────────┼─────────────────────────────────────────┤
│ 2. TIER 1 CLASSIFIER           │ • Fast, deterministic grading model     │
│    (Stance Scorer)             │ • Fine-tuned bert-base-uncased on MPS   │
├────────────────────────────────┼─────────────────────────────────────────┤
│ 3. TIER 2 FRONTIER JUDGE       │ • Deep reasoning audit model            │
│    (Qualitative & Trajectory)  │ • Gemini 1.5 Pro / Flash via Google API │
└────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 2. Target Models (The Models Under Test)

These are the 3 open-weight models shortlisted in Section 2 of our CS612 project proposal. They satisfy our criteria: 8B–12B parameter density, instruction-tuned, open-weight, high Hugging Face download volume, and global locale diversity.

| Model Name | Developer / Locale | Parameters | Sourcing Origin & Links | Execution Modes |
| :--- | :--- | :---: | :--- | :--- |
| **`Llama-3.1-8B-Instruct`** | Meta (US / Western baseline) | 8.03B | **Hugging Face:** `meta-llama/Llama-3.1-8B-Instruct`<br>**Ollama Tag:** `ollama pull llama3.1:8b-instruct` | Local Ollama / vLLM / Groq API / OpenRouter |
| **`Qwen-2.5-7B-Instruct`** | Alibaba Cloud (China / Eastern baseline) | 7.61B | **Hugging Face:** `Qwen/Qwen2.5-7B-Instruct`<br>**Ollama Tag:** `ollama pull qwen2.5:7b-instruct` | Local Ollama / vLLM / Groq API / OpenRouter |
| **`Gemma-2-9B-It`** | Google DeepMind (Global / Knowledge distillation) | 9.24B | **Hugging Face:** `google/gemma-2-9b-it`<br>**Ollama Tag:** `ollama pull gemma2:9b` | Local Ollama / vLLM / Groq API / OpenRouter |

### Why Open-Weight 8B–12B Models?
1. **Academic Reproducibility:** Anyone can inspect the exact weights, model cards, and tokenizer behaviors.
2. **Compute Realism:** Fits on consumer workstations and single-GPU academic allocations without requiring massive multi-GPU clusters.
3. **Real-World Impact:** These are the most widely deployed open models embedded in private enterprise workflows.

---

## 3. Tier 1 Classifier Model: `bert-base-uncased`

* **Source:** Hugging Face Hub (`bert-base-uncased`).
* **Parameters:** ~110 Million parameters.
* **Role:** Sequence classification head fine-tuned on 10,500 balanced AllSides media bias rows to output 3 probabilities: `[P_Left, P_Center, P_Right]`.

### Why `bert-base-uncased`?
1. **Speed & Determinism:** Executes in milliseconds per sample locally on Apple Silicon MPS or Colab GPU with zero API rate limits or variance.
2. **Bidirectional Self-Attention:** Evaluates the complete response context simultaneously (unlike causal decoder models which only look backwards).
3. **Uncased Robustness:** Normalizes capitalized emphasis and punctuation quirks.

### Alternatives Considered for Future Iterations:
* **`microsoft/deberta-v3-base`:**
  * *Why consider:* Employs disentangled attention and enhanced mask decoder (EMD). Historically achieves 1–2% higher Macro F1 on Natural Language Inference (NLI) benchmarks.
  * *Status:* Already supported in our training script: pass `--model_name microsoft/deberta-v3-base`.
* **`roberta-base`:**
  * *Why consider:* Trained on significantly larger corpora with dynamic masking; slightly more resilient to informal conversational slang.
* **`FacebookAI/roberta-large`:**
  * *Why consider:* Higher representational capacity (355M params) if finer nuance is needed, at the cost of ~3x slower inference throughput.

---

## 4. Tier 2 Frontier Judge Model: Google Gemini API

* **Provider:** Google DeepMind / Google AI Studio.
* **Primary Judge Model:** `gemini-1.5-pro` (or `gemini-2.0-flash`).
* **Role:** Audits the full 10-turn dialogue transcript produced by `multi_turn_simulator.py`. Disentangles **Conversational Politeness (1–5)** from **Epistemic Concession (1–5)** and extracts evidence quotes.

### Which Gemini API Models Can You Use with Your API Key?
When you obtain an API key from [Google AI Studio](https://aistudio.google.com/), you typically have immediate access to:

1. **`gemini-1.5-pro` (Recommended for Final Paper Audit):**
   * Designed for complex multi-step reasoning, nuanced instruction following, and massive context windows (up to 2M tokens).
   * Ideal for identifying subtle epistemic surrender across 10-turn conversations.
2. **`gemini-1.5-flash` / `gemini-2.0-flash` (Recommended for Fast / High-Volume Prototyping):**
   * Extremely fast and low latency.
   * Free tier offers high RPM (requests per minute) limits.
   * Excellent for smoke testing all 90 dialogue transcripts before running the final pass with Pro.

### Alternative Judge Backends Supported:
Our auditor script (`src/run_tier2_judge.py`) also supports:
* **`openai`:** `gpt-4o` or `gpt-4o-mini` (via standard OpenAI endpoint).
* **`mock`:** Fully deterministic heuristic offline auditor for zero-cost pipeline verification.

---

## 5. Execution Summary: Where & How to Run

| Task | Script | Mode / Backend Options | CLI Example |
| :--- | :--- | :--- | :--- |
| **Single-Turn Benchmark** | `src/run_single_turn_eval.py` | Local Classifier + Target API/Ollama | `python3 src/run_single_turn_eval.py --target_model llama-3.1-8b-instruct` |
| **Multi-Turn Simulation** | `src/multi_turn_simulator.py` | Target Model (Ollama / Cloud API) | `python3 src/multi_turn_simulator.py --backend ollama --target_model llama-3.1-8b-instruct` |
| **Tier 2 LLM Audit** | `src/run_tier2_judge.py` | Frontier API (Gemini / OpenAI / Mock) | `python3 src/run_tier2_judge.py --judge_provider gemini --judge_model gemini-1.5-pro` |
