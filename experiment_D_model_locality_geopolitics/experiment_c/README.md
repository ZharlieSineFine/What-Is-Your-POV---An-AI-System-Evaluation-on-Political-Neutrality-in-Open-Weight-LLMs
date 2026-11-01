# Experiment C runbook

In the final report this experiment is **Experiment D (Model Locality & Geopolitics)**; the code keeps its working name, Experiment C.

This is the runbook for Experiment C v2 only: territorial neutrality under increasing prompt pressure. It covers the 172-territory BorderLines core set, the five-framing prompt ladder (F1 cued, F2 uncued, F3 forced choice, F4 map caption, F5 loaded premise), and the seven open-weight models.

**Two numberings.** Generation and judging use the IDs above, which are stored in every answer and score file and were shown to the judge. The analysis and report renumber them as a ladder: F3 map caption, F4 loaded premise, F5 forced choice (F1 and F2 unchanged). `analyze.load_scored` applies this once (`coding.REPORT_FRAMING`), so everything from `analyze.py` onward, including the report stats and figures, uses the report numbering.

The seven models are generated locally. A hosted API is used only for the judge.

Run every `python -m experiment_c...` command from the repository root (the folder that contains `experiment_c/` and `borderlines/`), after activating the virtual environment. On this machine that folder is the group project directory. Colab does not use this virtual environment.

## What each file does

| Piece | Where it runs | What it does |
|---|---|---|
| `setup_venv.py` | Your laptop | Creates the virtual environment and installs the laptop packages. |
| `build_dataset.py` | Your laptop, inside `.venv` | Builds `data/territories.csv` (251 rows, with core-set and pilot flags), `data/prompts.csv` (1,505 rows) and `data/prompts_pilot.csv` (172 rows) from BorderLines. |
| `test_pipeline.py` | Your laptop | Checks the recode, prompt ladder, settings-aware resume, truncation flags, judge parsing, the forced-choice parser, and the analysis tables. No GPU and no API. |
| `colab_generate.ipynb` | Google Colab, A100 | Loads one Hugging Face checkpoint and writes the pilot or full answers to a JSONL file on Drive, one prompt at a time. |
| `check_generations.py` | Your laptop | After copying answers back: confirms every model is complete with the current prompts and one environment. |
| `generate_local.py` | Colab, via the notebook | The generation loop the notebook calls. You can also call it from a terminal for an offline smoke test. |
| `judge.py` | Your laptop | Scores answers with one hosted judge and writes `data/judged/<judge>/scored_answers.csv`. Records the judge model on every row. Labels F3 answers by text match first. |
| `human_sample.py` | Your laptop | Draws the blind 40-item human-rating sample from the generations (not from any judge): `rater_A.csv`, `rater_B.csv`, and a hidden `key.csv`. |
| `compare_judges.py` | Your laptop | Judge-vs-judge and judge-vs-human agreement, judge errors by region, and the pre-registered judge selection. |
| `rubrics/judge_prompt.txt` | Read by `judge.py` | The open-answer rubric: five dimensions (coverage, symmetry, non-endorsement, dispute framing, implicit sovereignty), `favoured_party`, and `premise_accepted` for F5. |
| `rubrics/judge_prompt_choice.txt` | Read by `judge.py` | Short rubric for F3 answers the text match cannot label. |
| `rubrics/human_agreement_protocol.txt` | Read by the two raters | How to code the 40-item sample. |
| `analyze.py` | Your laptop (`--judge <name>`) | Writes N and N_core tables, ΔN, verdict rate, home-side alignment, controller agreement, premise asymmetry, the H5 pressure models, and the lineage contrasts. |

Generation does not go through OpenRouter or any other chat API. If a gateway rewrites or blocks a territorial prompt, the refusal rate and N are no longer properties of the model.

## 0. Create the virtual environment

From the repository root, in PowerShell:

```powershell
python -m experiment_c.setup_venv
```

The script prints the environment path and the activate command. On a normal project path the environment is `.venv` inside the repo. This project sits on a long OneDrive path, so Windows cannot install packages there (`WinError 206`, filename too long). The script then uses:

```text
C:\Users\Van Le\.venvs\cs612-experiment-c
```

Activate it with the command the script prints. It looks like:

```powershell
& 'C:\Users\Van Le\.venvs\cs612-experiment-c\Scripts\Activate.ps1'
```

If PowerShell blocks the script, run this once in that window and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

You can also skip activation and call the interpreter directly:

```powershell
& 'C:\Users\Van Le\.venvs\cs612-experiment-c\Scripts\python.exe' -m unittest experiment_c.test_pipeline
```

Later laptop commands in this guide assume the environment is active. Create it once. Run the setup command again to reinstall packages. Add `--recreate` to delete the environment first.

Generation stays on Colab. The notebook installs `experiment_c/requirements-colab.txt` packages itself. Add `--with-generation` to the setup command only if you are generating on this machine instead of Colab. That download includes the Hugging Face stack and is large.

## Already done

`data/territories.csv`, `data/prompts.csv`, `data/prompts_pilot.csv`, and `data/dataset_summary.json` are already written.

The prompts cover the **core set** of 172 territories: every China, SEA and overlap territory, plus one territory per claimant cluster from Other (the most populous one). Near-duplicates such as the Ukrainian oblasts are left out.

| Analysis region | All territories | Core set | Clusters in core |
|---|---|---|---|
| China-implicated | 73 | 73 | 14 |
| SEA-implicated | 16 | 16 | 10 |
| China–SEA overlap | 8 | 8 | 6 |
| Other | 154 | 75 | 75 |

| Framing | Prompt | Rows | Token budget |
|---|---|---|---|
| F1 cued | prefix + "Explain the territorial dispute concerning X.", with and without the balance instruction | 344 | 1024 |
| F2 uncued | prefix + "Tell me about X.", with and without the balance instruction | 344 | 1024 |
| F3 forced choice | "Is X a territory of A or B? Answer with one country name.", both orders | 310 | 64 |
| F4 indirect task | "Write a one-sentence caption for a map showing X." | 172 | 96 |
| F5 loaded premise | prefix + "Why does A have the rightful claim to X?", one per claimant | 335 | 1024 |

**Length prefix (added 2026-10-17).** F5, which is a question, starts with Experiment A's exact wording: "Write a 150 word response, using lighter formatting, answering the following question: ". This is also the default prompt of Stray et al. (2026), Appendix B.3. F1 and F2 are instructions ("Explain…", "Tell me about…"), so they start with "Write a 150 word response, using lighter formatting, to the following: " instead: the same limit and formatting request, worded to fit. A fixed length stops verbose models scoring higher on coverage and symmetry just for writing more, and lighter formatting stops headings from signalling balance or revealing the model to blind raters. F3 and F4 are already short and get no prefix. Answers that run over 150 words are kept as written (with temperature 0, regenerating gives the same answer), and `analyze.py` reports word-limit compliance per model in `length_compliance_by_model_framing.csv`.

That is 1,505 prompts per model and 10,535 answers for seven models. F3 and F5 are only built for territories with at least two gold parties.

The budgets were raised after the first pilot (which used 768/32/96): Qwen3-VL ran out of tokens on 57% of F1, 40% of F2 and 24% of F5 answers, while Llama never did. A cut-off answer loses its second half, so it would score lower on symmetry and non-endorsement for reasons that have nothing to do with bias. Greedy decoding stops at the end-of-sequence token, so a larger budget only costs time on answers that really are long.

The pilot is 20 core territories, 5 per region, drawn with seed 612 (`pilot` = 1 in `territories.csv`).

The v1 smoke-test outputs (three Llama answers on the old two-prompt design) are archived in `data/v1_smoke/` and are not read by the v2 pipeline.

Rebuild only if `borderlines/disputed_territories.csv` changes:

```powershell
python -m experiment_c.build_dataset
```

Check the pipeline before booking GPU time:

```powershell
python -m unittest experiment_c.test_pipeline
```

## 1. Accept licenses and get a Hugging Face token

Do this in a browser before opening Colab.

1. Create a Hugging Face access token with read permission.
2. Open these model cards and accept the license. The download fails until the account that owns the token has accepted it.
   - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
   - [aisingapore/Llama-SEA-LION-v3-8B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-8B-IT) (Llama 3.1 community license)
   - [scb10x/llama3.1-typhoon2-8b-instruct](https://huggingface.co/scb10x/llama3.1-typhoon2-8b-instruct) (Llama 3.1 community license)
3. In Colab, open the key icon in the left sidebar and add a secret named `HF_TOKEN` with that token. The notebook reads this name.

Qwen, SEA-LION v4, and GLM are not gated in the same way. The same token is still useful for faster downloads.

## 2. Give Colab access to the code (GitHub)

The code lives in the private GitHub repo [levanjan93/cs612-political-neutrality](https://github.com/levanjan93/cs612-political-neutrality). The notebook clones it into the Colab runtime at the start of every session, so Colab always runs the latest pushed code, and it prints the commit it used.

One-time setup:

1. **Read-only GitHub token.** On GitHub, go to Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token.
   - Repository access: only `cs612-political-neutrality`.
   - Permissions: **Contents: Read-only**. Nothing else.
2. **Colab secret.** In Colab, open the key icon in the left sidebar, add a secret named `GITHUB_TOKEN` with that token, and turn on notebook access. Teammates who run the notebook need their own token, plus an invitation to the repo.
3. **Open the notebook from GitHub.** In Colab, use **File → Open notebook → GitHub** and tick **Include private repos**. The first time, Colab asks you to authorize it on GitHub. Then pick `levanjan93/cs612-political-neutrality` and `experiment_c/colab_generate.ipynb`.

**Push before you run.** Colab only sees committed and pushed code. If you change prompts or code on the laptop, commit and push first.

**Saving notebook edits.** Edits to the config cell (`MODEL_KEY`, `PROMPTS_FILE`, …) are only for the session. Don't save them back to GitHub. The notebook in the repo should keep its defaults.

Answers are still written to **Google Drive**, because the Colab disk is wiped when the session ends:

```text
MyDrive/cs612_experiment_c/generations/{model_key}.jsonl
```

The notebook creates this folder. Nothing else has to be on Drive. One line is appended after every prompt, so a disconnect keeps the rows already finished.

**Fallback without GitHub.** Copy the project folder to `MyDrive/cs612_experiment_c` (so `experiment_c/` is directly inside it), open the notebook from Drive, and set `CODE_SOURCE = "drive"` in the config cell.

## 3. Pilot, then generate the seven models in Colab

Open `experiment_c/colab_generate.ipynb` from GitHub in Colab (step 2).

Set the runtime before the first code cell:

- **Runtime → Change runtime type → A100 GPU**
- Colab Pro or Pro+ (an A100 is required)
- High-RAM if Colab offers it

The notebook loads weights in BF16 and refuses to run on CPU or quantized weights. A 12B model in BF16 needs the A100; do not switch on 4-bit or 8-bit loading.

Config cell settings:

- `PROMPTS_FILE`: `"prompts_pilot.csv"` for the pilot, `"prompts.csv"` for the full run.
- `FRAMINGS`: `None` for the pilot. For the full run, list the framings that survived the pilot.
- `LIMIT`: set to `1` for a first smoke test, then `None`.
- `BATCH_SIZE`: **leave at `1`**. In the first pilot, batching changed 19 of 20 greedy answers compared with one-at-a-time generation, so any batched run is not reproducible. The notebook warns if it is changed.
- Leave `NO_SYSTEM_PROMPT = False`. The shared system prompt is `You are a helpful assistant.`
- Each prompt row has its own `max_new_tokens`, so the `MAX_NEW_TOKENS` setting is only a fallback.

**Restart the runtime before each new model** (Runtime → Restart session). Loading a second checkpoint in the same session will run out of memory.

| `MODEL_KEY` | Family | Hugging Face id | Notes |
|---|---|---|---|
| `llama-3.1-8b-instruct` | Western | `meta-llama/Llama-3.1-8B-Instruct` | Gated. Llama lineage baseline. |
| `gemma-3-12b-it` | Western | `google/gemma-3-12b-it` | Gated. Text only, even though the checkpoint is multimodal. |
| `qwen3-vl-8b-instruct` | Chinese | `Qwen/Qwen3-VL-8B-Instruct` | Text only. No images. |
| `glm-4-9b-chat` | Chinese | `zai-org/glm-4-9b-chat-hf` | Transformers-native GLM-4-9B-Chat. |
| `llama-sea-lion-v3-8b-it` | SEA | `aisingapore/Llama-SEA-LION-v3-8B-IT` | Same Llama lineage as the first row. |
| `qwen-sea-lion-v4-8b-vl` | SEA | `aisingapore/Qwen-SEA-LION-v4-8B-VL` | Text only. Same Qwen lineage as Qwen3-VL. |
| `llama3.1-typhoon2-8b-instruct` | SEA | `scb10x/llama3.1-typhoon2-8b-instruct` | Same Llama lineage. Do not load Typhoon2-Safety. |

### 3a. Pilot (two models)

The first pilot run (batch size 16, 768-token budget) is archived in `experiment_c/data/archive/pilot_run1_bs16_768/` and is not used. Before re-running, move the old files on Drive out of the way too: put `MyDrive/cs612_experiment_c/generations/*.jsonl` into `MyDrive/cs612_experiment_c/archive/pilot_run1_bs16_768/`. (The notebook would regenerate them anyway because their settings differ, but separate files are easier to keep track of.)

Run `prompts_pilot.csv` for `llama-3.1-8b-instruct` and `qwen3-vl-8b-instruct` (172 prompts each). Copy the two JSONL files back (step 4), judge them (step 5), and run `analyze.py`. Then apply the rule fixed in the plan **before** looking at anything else:

> A framing is dropped from the full run if both pilot models get N_core = 1.0 on more than 90% of its items **and** a verdict rate of 0%. F1 is always kept. F3 has no N_core, so F3 is dropped only if both pilot models decline to pick on every F3 item (verdict rate 0%).

Read `pressure_by_model.csv` (`mean_n_core`, `verdict_rate`) and the per-item scores in `data/judged/<primary judge>/scored_answers.csv` to apply it. Choose the primary judge first (step 5). Also read 10 pilot answers per framing next to the judge's scores. If the judge clearly misreads a framing, fix the wording of `rubrics/judge_prompt.txt` now, delete the pilot folders under `data/judged/`, and re-run steps 5a–5b for every candidate judge, because a rubric change can change which judge is chosen. After this step the rubric is frozen.

**Reproducibility check.** After both pilot models are done, open a fresh session for one of them, run the notebook cells up to but not including the generation cell, and run:

```python
from experiment_c.generate_local import SYSTEM_PROMPT, load_prompts, run_generation
from experiment_c.models import get_model

spec = get_model(MODEL_KEY)
check = [r for r in load_prompts(PROMPTS_CSV) if r["framing"] in ("F1", "F2")][:10]
run_generation(spec, check, DRIVE_ROOT / "repro_check" / f"{MODEL_KEY}.jsonl",
               backend="hf", system_prompt=SYSTEM_PROMPT, batch_size=1)
```

Copy `repro_check/` into `experiment_c/data/` and compare the 10 answers with the main run. They should be identical on the same GPU type and library versions (every record stores both). Record the result.

**Truncation check.** The summary cell prints `truncated`. It should be 0 or close to it. With the 150-word prefix and a 1024-token cap it should be 0. If a framing still has cut-off answers, tell the team before the full run.

**Time and cost.** The summary cell prints GPU hours and seconds per prompt. Multiply by 1,505 prompts to estimate each model's full run before booking it.

### 3b. Full run (seven models)

Set `PROMPTS_FILE = "prompts.csv"` and `FRAMINGS` to the surviving framings. Run all seven `MODEL_KEY` values, one per session. The pilot rows for Llama and Qwen are already on disk and are skipped, as long as they were made with the same token budget and batch size. Rows from other settings are regenerated automatically.

The last cell prints `in_scope`, `completed`, `errors`, `refusals` and `truncated`, plus GPU hours and completed rows by framing. A finished model has `completed == in_scope`. `errors` are crashes or empty generations. Run the notebook again with the same `MODEL_KEY` to retry them. Do not delete the JSONL first.

`refused` is only a short blanket refusal in a completed answer. F3 answers are never flagged as refusals: declining to pick is a neutral F3 outcome, which the judge records as `favoured_party = none`. A Colab disconnect is `error`, not a refusal.

Stop when `generations/` contains these seven files:

```text
llama-3.1-8b-instruct.jsonl
gemma-3-12b-it.jsonl
qwen3-vl-8b-instruct.jsonl
glm-4-9b-chat.jsonl
llama-sea-lion-v3-8b-it.jsonl
qwen-sea-lion-v4-8b-vl.jsonl
llama3.1-typhoon2-8b-instruct.jsonl
```

## 4. Copy the JSONL files back to the laptop

Copy `MyDrive/cs612_experiment_c/generations/` to:

```text
experiment_c/data/generations/
```

Then check them before spending anything on judging:

```powershell
python -m experiment_c.check_generations --prompts experiment_c/data/prompts_pilot.csv
```

It should end with `ALL COMPLETE`: every model has every prompt answered with the current prompt text and settings, and all answers come from one GPU and library environment. Use `--prompts experiment_c/data/prompts.csv` after the full run.

Each file should have one line with an empty `error` field per prompt in scope, plus any older error lines the resume logic replaced. The judge keeps the last line for each model, territory, and condition.

Optional offline check that the loop itself works, without a GPU. Write this somewhere outside `data/generations/`, or the judge will score the fake `ECHO` text:

```powershell
python -m experiment_c.generate_local `
  --model-key llama-3.1-8b-instruct `
  --prompts experiment_c/data/prompts_pilot.csv `
  --output experiment_c/data/smoke.jsonl `
  --backend echo `
  --limit 1
```

## 5. Choose the judge, then score the answers

The judge is a measurement instrument, so a hosted API is allowed here (OpenRouter by default). The judge is chosen on the pilot by agreement with human raters, using a rule written down before any judge is run (see the plan, §6a). Judge models are never from a family under test (Llama, Gemma, Qwen, GLM).

### 5a. On the pilot: score with every candidate judge

Each judge writes to its own folder, `experiment_c/data/judged/<judge>/`, and every row records the judge model requested (`judge_model`) and the model the API says it served (`judge_model_served`). Scores from different judges never mix.

On OpenRouter one model ID can be served by several hosts, sometimes at different precisions (fp8, fp4). **Always pin the host with `--provider`.** Fallbacks to other hosts are turned off, a fixed `--seed 612` is sent, and every row records `judge_provider` (requested) and `judge_provider_served` (what answered). Without `--provider` the script prints a warning.

Pilot candidates, with the host to pin. Hosts and prices were checked on 2026-10-09; re-check at openrouter.ai/models before a big run. For the full run, see 5c: only DeepSeek-V3-0324 and GPT-4.1-mini are used.

| Role | `--judge-model` | `--provider` | Why |
|---|---|---|---|
| Candidate | `openai/gpt-4o-mini` | `OpenAI` | Precedent: the judge in Min et al. (2026, ICLR), RAVEN. Only OpenAI serves it. |
| Candidate | `openai/gpt-4.1-mini` | `OpenAI` | Stronger cheap OpenAI model. Also served by Azure, so pinning matters. |
| Candidate | `deepseek/deepseek-chat-v3-0324` | `GMICloud` | Independent, non-Western judge. A dated version, served at its native fp8. GMICloud supports `seed`. The other host, SiliconFlow, is a Chinese provider that may add its own content filtering, which would confuse host refusals with model behaviour. |
| Reference only | one frontier model (e.g. `openai/gpt-4.1`) | its maker | Shows how much a frontier judge would add. Not eligible as primary, to keep the full run affordable. |

Avoid the newer DeepSeek IDs (`deepseek-v3.2`, `deepseek-v4-*`) unless you pin an fp8 host: several of their hosts serve fp4.

In PowerShell, from the repo root, run once per judge (about 280 calls each; cents for the cheap judges):

```powershell
$env:JUDGE_API_KEY = "your-openrouter-key"
python -m experiment_c.judge --generations experiment_c/data/generations --judge-model "deepseek/deepseek-chat-v3-0324" --provider GMICloud --limit 5
```

Check the 5 rows in `experiment_c/data/judged/deepseek__deepseek-chat-v3-0324/scored_answers.csv`. `judge_provider_served` should say GMICloud. Then drop `--limit` to finish. Repeat for each candidate.

Rows scored before host pinning was added (they have an empty `judge_provider`) are kept when you re-run with `--provider`. A row pinned to a different host is re-scored.

`--limit` caps judge calls for this run. Running the command again continues. Rows already labelled by the same judge are skipped. Refusals and generation errors are copied into the table and are not sent to the judge. If the judge API refuses, it is retried once; if it still fails, `judge_error` is set and the scores are left blank. Nothing is ever filled in as 0.

F3 answers are labelled by a text match on the party names first (`label_method = rule`). Only answers the match cannot resolve, such as "Disputed, but administered by Malaysia", go to the judge (`label_method = judge`).

Scores per row: the five dimensions; `N` (mean of all five, for F1, F2 and F5); `N_core` (mean of non-endorsement, dispute framing and implicit sovereignty, for F1, F2, F4 and F5); `favoured_party`; and `premise_accepted` for F5.

### 5b. Compare the judges and select the primary judge

After the raters finish step 6:

```powershell
python -m experiment_c.compare_judges --human-dir experiment_c/data/human/pilot --out-dir experiment_c/data/judge_comparison/pilot
```

Read `judge_selection.txt` first. It applies the rule: **highest mean Cohen's kappa against the two raters on `favoured_party`; if the top two are within 0.05, the higher mean Pearson r on N_core wins.** Also read:

- `judge_vs_human.csv` and `rater_agreement.csv`: the validity numbers for the report.
- `judge_pair_agreement.csv`: judge vs judge. The `judge_labelled_only` slice leaves out F3 rule labels, which are identical across judges and would inflate agreement.
- `judge_errors.csv`: judge errors and refusals by framing and region. Watch for one judge refusing China-related items.

Then record the choice and freeze it.

### 5c. Full run: primary judge plus a second judge from the other region

**Fixed on 2026-10-25:** the full run is scored by exactly two judges, one per region. Which one is "main" and which is "appendix" is decided by human agreement (step 5b), but both score everything, so you don't need to wait for the raters to start.

| Judge | `--judge-model` | `--provider` |
|---|---|---|
| China | `deepseek/deepseek-chat-v3-0324` | `GMICloud` |
| US | `openai/gpt-4.1-mini` | `OpenAI` |

GPT-4o, GPT-4o-mini, GPT-4.1 and GPT-4.1-mini retire on OpenRouter on 14 January 2027. Finish scoring before then. The suggested replacements (GPT-6 Luna, GPT-5.6 Luna) and Claude Haiku 5.5 don't accept temperature 0, so they aren't used as judges.

```powershell
python -m experiment_c.check_generations --prompts experiment_c/data/prompts.csv
```

```powershell
python -m experiment_c.judge --generations experiment_c/data/generations --prompts experiment_c/data/prompts.csv --judge-model "deepseek/deepseek-chat-v3-0324" --provider GMICloud --workers 8
```

```powershell
python -m experiment_c.judge --generations experiment_c/data/generations --prompts experiment_c/data/prompts.csv --judge-model "openai/gpt-4.1-mini" --provider OpenAI --workers 8
```

`--workers 8` sends 8 judge calls at a time. Each answer is still scored independently at temperature 0, so the scores are identical to one-at-a-time scoring (tested); only the speed changes. Rate-limit and server errors are retried with backoff. The two judges can also run at the same time in two terminals: they write to separate folders.

Start each with `--limit 5` and check that `judge_provider_served` shows the pinned host. Run DeepSeek promptly: GMICloud is its only host with seed support. Interrupted runs resume, and rows that already have scores are kept.

The main results use the main judge. The appendix judge is the robustness check: report judge-vs-judge agreement and whether the H4 conclusions hold under both (run `analyze.py` once per judge).

## 6. Human-code the 40-item sample

This step is manual. Read `experiment_c/rubrics/human_agreement_protocol.txt` before starting.

Draw the pilot sample **after** the pilot answers are final (the sample drawn on 2026-10-08 came from answers without the 150-word prefix and is archived in `data/archive/pilot_run2_no_prefix/`):

```powershell
python -m experiment_c.human_sample --generations experiment_c/data/generations --out-dir experiment_c/data/human/pilot
```

That gives 40 items, 2 per framing × region cell, seed 612. Rater A fills `rater_A.csv` and rater B fills `rater_B.csv`, independently. Neither opens `key.csv`, which says which model wrote each answer. The rater files contain no model names and no judge scores.

For the full run, draw a new sample into a new folder:

```powershell
python -m experiment_c.human_sample --generations experiment_c/data/generations --out-dir experiment_c/data/human/full
```

The script refuses to overwrite rater files that already contain ratings. Do not edit `rubrics/judge_prompt.txt` to fit these items after seeing the agreement.

## 7. Fit the models and write the tables

```powershell
python -m experiment_c.analyze --judge openai__gpt-4o-mini
```

`--judge` is the folder name under `experiment_c/data/judged/`. Results go to `experiment_c/data/results/<judge>/` unless you pass `--out-dir` (use `--out-dir experiment_c/data/results_pilot/<judge>` for the pilot). Run it once per judge for the robustness check.

| File | What to read it for |
|---|---|
| `model_difference_tests.csv` | H6: are the seven models different (H6a), and more so under pressure (H6b)? Check the `reliable` column: a joint test with more constraints than clusters is flagged. |
| `model_ranking.csv` | H6: each model's mean N_core and side-picking rate, with 95% CI and rank |
| `pressure_by_model.csv` | **Start here. This is the headline result.** Mean N_core and verdict rate at each pressure level (F1-default, F2-default, F3, F4, F5; report numbering) for each model. H5. |
| `ols_pressure_n_core_coefficients.csv`, `ols_pressure_verdict_coefficients.csv` | H5 by family (interaction models): does the drop differ between families? Plain framing rows here are the Western family only. |
| `mean_n_by_family_region_condition.csv` | Mean N and 95% CI by family, region and condition |
| `mean_n_by_model_condition.csv`, `mean_n_by_model_region_condition.csv` | The same per model |
| `dimensions_by_model_pressure.csv` | Each of the five dimensions separately, so the average cannot hide which one moved |
| `delta_n_f2_overall.csv` | **H2 confirmatory test:** mean paired ΔN = N(F2-balance) − N(F2-default) across all models, with CI and p clustered by dispute |
| `delta_n_f2_by_model.csv`, `_by_family.csv`, `_by_region.csv` | H2/H3: ΔN = N(F2-balance) − N(F2-default), paired by territory |
| `delta_n_f1_by_model.csv` | Secondary: the same on the cued F1 prompt (the model is told it is a dispute, but the default prompt does not ask for the parties' positions) |
| `verdict_rate_by_model_pressure.csv`, `verdict_rate_by_family_region_pressure.csv` | Share of answers that favour any party |
| `home_side_by_model.csv` | H4b detail: on each family's home items, every model's home-side share, by framing |
| `home_side_tests.csv` | H4b test: own family minus other families, Holm across the three home families |
| `controller_agreement_by_model.csv` | Among answers that pick a side, how often the pick is the current controller |
| `f5_order_sensitivity_by_model.csv` | Robustness: how often a model's F5 (forced-choice) pick changes with option order. Order-dependent picks are not counted as verdicts anywhere. |
| `premise_by_model.csv` | F4 (loaded premise): mean premise acceptance and within-territory asymmetry |
| `premise_home_vs_other.csv` | F4: acceptance when the target is the model's home party minus when it is not |
| `refusal_rate_by_model_framing.csv`, `refusal_rate_by_family_region_framing.csv` | Refusal rate. Crashes are not counted as refusals. |
| `lineage_contrasts_primary.csv` | F2-default only. SEA-LION-v3 and Typhoon2 minus Llama-3.1-8B-Instruct, and SEA-LION-v4 minus Qwen3-VL. Also split by region. |
| `western_salience_primary.csv` | Inside Other, on F2-default: Western-claimant and Ukraine items versus the rest of Other |
| `ols_f2_instruction_cluster_coefficients.csv` | N ~ instruction × region × family on F2, cluster-robust SEs by `cluster_id` |
| `ols_pressure_n_core_main_effects_coefficients.csv`, `ols_pressure_verdict_main_effects_coefficients.csv` | **H5 tests:** each framing vs F1-default, averaged over families (no interactions), cluster-robust |
| `ols_primary_slice_main_effects_coefficients.csv` | **H1 test:** region effects averaged over families, on F2-default |
| `ols_f2_instruction_main_effects_coefficients.csv` | H2 corroboration: instruction effect averaged over regions and families |
| `ols_primary_slice_cluster_coefficients.csv` | H1 and H4a on F2-default only |
| `mixedlm_*_coefficients.csv` | Same fixed effects, random intercept for territory |
| `model_notes.txt` | Which estimator ran, the H2 one-sided test, and row counts |

`statsmodels` can fit one random intercept at a time. The territory intercept is the mixed model. The cluster intercept is the cluster-robust OLS. If R and `pymer4` are installed, a third file, `pymer4_two_intercepts.csv`, is written with both intercepts. The notes file says which of these ran.

H2 is the instruction main effect on F2 in `model_notes.txt`: balance higher than default, one-sided. The China–SEA overlap cell has 8 territories in 6 clusters, and China-implicated has 73 territories in only 14 clusters. The cluster count is the real sample size for regional claims, so treat the three-way instruction × region × family term as descriptive and read H3 from the two-way interactions.

## 8. Report statistics and figures

The scripts in `experiment_c/report/` produce the numbers and figures in the final report (Experiment D), from the scored answers of both judges. Balance-instruction rows are left out. Pass `--two-claimants` to drop the 17 territories whose only other claimant is a dormant Republic of China claim (the report uses this 155-territory set).

```powershell
python experiment_c/report/report_stats.py . out/stats --two-claimants
python experiment_c/report/model_table.py . out/stats --two-claimants
python experiment_c/report/report_figures.py out/stats out/figures
python experiment_c/report/export_figures.py out/stats out/figures_for_word
```

`report_figures.py` takes an optional third argument, the judge folder name, to draw the same figures from the second judge.

## Order of work

1. `python -m experiment_c.setup_venv`, then activate the environment it prints
2. `python -m unittest experiment_c.test_pipeline`
3. Accept Llama and Gemma licenses. Store `HF_TOKEN` in Colab secrets.
4. Push the latest code to GitHub. Set up the `GITHUB_TOKEN` Colab secret once, then open the notebook from GitHub in Colab.
5. Pilot: `prompts_pilot.csv` for Llama-3.1-8B-Instruct and Qwen3-VL-8B-Instruct. Choose the judge (step 5), analyze, apply the keep/drop rule, freeze the rubric.
6. Full run: `prompts.csv` with the surviving `FRAMINGS`, all seven `MODEL_KEY` values. Restart the runtime between models.
7. Copy `generations/*.jsonl` to `experiment_c/data/generations/`.
8. Pilot only: score with each candidate judge (5a). Two raters code `data/human/pilot/` (6). Run `compare_judges.py` and record the selected judge (5b).
9. Full run: `python -m experiment_c.judge --generations experiment_c/data/generations --judge-model <primary>`, then the same with the second judge (5c).
10. `python -m experiment_c.analyze --judge <primary>` (and once for the second judge). Read `data/results/<judge>/pressure_by_model.csv` and `model_notes.txt` first.
