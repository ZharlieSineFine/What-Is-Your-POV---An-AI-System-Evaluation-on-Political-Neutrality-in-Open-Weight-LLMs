# What's Your POV? An AI System Evaluation on Political Neutrality in Open-Weight LLMs

Code and data for the CS612 AI System Evaluation group project (Group 3, Singapore Management University).

## Repository layout

| Folder | Report section | Author |
|---|---|---|
| `experiment_A_baseline_neutrality/` | 2. Experiment A: Baseline Neutrality | Yea Ram |
| `experiment_B_neutrality_robustness/` | 3. Experiment B: Neutrality Robustness | Zhao Changlin |
| `experiment_C_sycophancy_and_prism_multi_turn/` | 4. Experiment C: Context Pollution (Sycophancy), and the multi-turn PRISM results in Section 6 | Anish |
| `experiment_D_model_locality_geopolitics/` | 5. Experiment D: Model Locality & Geopolitics | Van |
| `prism_single_turn/` | The single-turn PRISM results in Section 6 | Zhao Changlin |

## Experiment A

- `pipeline/`: Yea Ram's five notebooks, in order: answer collection through the Hugging Face API (`syseval_exp1_1_scrapes`), judge calibration against PARETO's human ratings (`_2_goldenset`), liberal and conservative judge selection (`_3_liberal_judge`, `_4_conservative_judge`), and the metrics (`_5_analysis`). `PARETO_Prompts.xlsx` holds the 200 questions.
- `seven_model_runs/`: Yea Ram's pipeline run on a Colab GPU, one notebook per model, with each model's scored answers in `exp1_<model>.xlsx`.

## Experiment B

- `1_noise_floor/`: notebooks 1–7 answer the 200 original questions, one model each. Notebook 8 scores every answer five times to measure judge noise. Outputs are in `results/`.
- `2_prompt_rewrites/`: `rewrite_reviewer.ipynb` writes four rewrites per question, checks each one, and saves the 1,000 test prompts to `prompt_rewrites_review/versions.csv`.
- `3_answer_robustness/`: notebooks 1–7 answer the 1,000 prompts, one model each. Notebook 8 grades the 7,000 answers and computes Spread, Big Spread Rate, and Side Switch Rate. Outputs are in `results/`.
- `pilot_and_method_checks/`: the pilot run (`robustness_pilot.ipynb`), the comparison of NLI models behind the choice of DeBERTa-v2-xlarge-MNLI (`bidirectional_entailment_test.ipynb`), and the test of the GPT-4o-mini check in both directions (`gpt_two_way_check_test.ipynb`). Each reads the outputs of the one before it.
- `Experiment B Robustness - Figures.xlsx`: the data behind the report's Experiment B table and figures.

## Experiment C and multi-turn PRISM

Anish's repository [siranishkd/cs612-political-neutrality-eval](https://github.com/siranishkd/cs612-political-neutrality-eval) at commit `789df91`. Its README covers setup and the Docker run.

The fine-tuned classifier's weights (438 MB) are not copied here. Download [`model.safetensors`](https://github.com/siranishkd/cs612-political-neutrality-eval/blob/789df91ac419fd68f120c9b2d63b366fc22d7a90/system_c_sycophancy/models/final_calibrated_model/model.safetensors) from Anish's repository and save it to `experiment_C_sycophancy_and_prism_multi_turn/system_c_sycophancy/models/final_calibrated_model/` before running this folder's code or Docker image.

## Experiment D

Van's repository `levanjan93/cs612-political-neutrality` at commit `3e8ab29`. The code is in `experiment_c/`, its name before the report's lettering. `borderlines/` is a Git submodule pointing to the BorderLines dataset (Li, 2024). Clone this repository with `--recurse-submodules` to get it.

## PRISM single-turn

Ten notebooks that run full PRISM (Anish's system prompt plus his critic) on all seven models over Experiment B's 1,000 prompts, then compare the results with the baseline. `inputs/` holds the baseline scores and the copy of `prism_defense.py` the notebooks use. `results/` holds the answers, critic output, gradings, and analysis.

## Notes

- The notebooks ran on Google Colab and read and write files in Google Drive, so their paths point there. Each notebook keeps its saved outputs.
- Some code, file names, and saved outputs use working names from before the report's lettering: Experiment A appears as "Experiment 1" or `exp1`, and Experiment B as "Experiment D".
- Checkpoint files written during generation are not included. They hold the same rows as the final outputs.
