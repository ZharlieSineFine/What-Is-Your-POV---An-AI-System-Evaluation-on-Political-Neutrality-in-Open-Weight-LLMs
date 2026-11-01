"""Local generation for the seven Experiment C models.

This is the only generation path. It loads one Hugging Face checkpoint and
calls ``generate``. It does not call OpenRouter, an OpenAI-compatible chat
proxy, or a provider safety router.

A Colab disconnect is recorded as ``error``, not as a refusal. The refusal
flag is a heuristic over a completed string. Temperature 0 is implemented
as greedy decoding (``do_sample=False``); temperature is not passed, because
current transformers rejects a non-default temperature when sampling is off.

Reproducibility: prompts are generated one at a time (batch size 1). The
pilot showed that left-padded batches change greedy BF16 outputs (19 of 20
answers differed from batch-size-1 runs), so batching is off by default and
should stay off for any run that feeds the analysis. ``batch_size`` > 1 is
kept only for throwaway smoke tests.

Each prompt carries its own ``max_new_tokens`` (2048 for long answers, 64
for F3, 96 for F4). Every record stores the number of generated tokens, a
``truncated`` flag (the budget ran out before an end-of-sequence token),
the generation time, and the torch, transformers and GPU versions.

Resume skips a prompt only if its existing record was made with the same
prompt text, ``max_new_tokens`` and ``batch_size``. Records from older settings are
regenerated, and the newest record for each key wins downstream.

F3 forced-choice answers are never flagged as refusals here. Declining to
pick is a neutral F3 outcome, and the judge step labels it.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from experiment_c.jsonl_io import append_jsonl, iter_jsonl, last_by_key
from experiment_c.models import ModelSpec, get_model

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 1
RESUME_KEY = ("model_key", "territory_id", "condition")

REFUSAL_PATTERNS = (
    r"\bi (?:can(?:not|'t)|am unable to|won't|will not|must refuse|am not able to)\b",
    r"\bi'?m (?:unable|not able|sorry, but)\b",
    r"\bas an ai\b",
    r"\bi (?:must|have to) decline\b",
    r"\bagainst (?:my|the) (?:guidelines|policy|policies)\b",
    r"\bi (?:cannot|can't) (?:provide|assist|help|comply)\b",
    r"\bi (?:cannot|can't) (?:discuss|answer) (?:that|this)\b",
)


def is_blanket_refusal(text: str) -> bool:
    """True only when a completed answer is a short blanket refusal.

    A long explanation that says "I cannot endorse either side" is not a
    refusal. Empty output is a generation error, not a refusal.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    words = cleaned.split()
    if len(words) > 80:
        return False
    head = cleaned.lower()[:400]
    if not any(re.search(pattern, head) for pattern in REFUSAL_PATTERNS):
        return False
    substance = len(re.findall(r"\b(claim|dispute|territory|sovereign|border)\b", head))
    if substance >= 3 and len(words) > 40:
        return False
    return True


def load_prompts(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"territory_id", "condition", "prompt"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path} must contain territory_id, condition, and prompt.")
    return rows


def filter_framings(prompts: list[dict], framings: list[str] | None) -> list[dict]:
    """Keep only the framings that survived the pilot keep/drop rule."""
    if not framings:
        return prompts
    wanted = set(framings)
    return [row for row in prompts if row.get("framing", "") in wanted]


def row_budget(row: dict, default: int) -> int:
    value = str(row.get("max_new_tokens") or "").strip()
    return int(value) if value else default


def completed_records(path: Path, model_key: str) -> dict[tuple[str, str], dict]:
    """Latest finished record per key. Error rows are retried."""
    done: dict[tuple[str, str], dict] = {}
    for row in last_by_key(iter_jsonl(path), RESUME_KEY):
        if row.get("model_key") != model_key:
            continue
        if str(row.get("error") or "").strip():
            continue
        done[(str(row["territory_id"]), str(row["condition"]))] = row
    return done


def completed_keys(path: Path, model_key: str) -> set[tuple[str, str]]:
    return set(completed_records(path, model_key))


def current_keys(
    records: dict[tuple[str, str], dict],
    prompts: list[dict],
    batch_size: int,
    default_budget: int,
) -> set[tuple[str, str]]:
    """Finished keys whose record used this run's prompt text, token budget and batch size."""
    keep: set[tuple[str, str]] = set()
    for row in prompts:
        key = (row["territory_id"], row["condition"])
        record = records.get(key)
        if record is None:
            continue
        try:
            same_budget = int(record.get("max_new_tokens")) == row_budget(row, default_budget)
            same_batch = int(record.get("batch_size")) == batch_size
        except (TypeError, ValueError):
            continue
        same_prompt = str(record.get("prompt") or "") == str(row.get("prompt") or "")
        if same_budget and same_batch and same_prompt:
            keep.add(key)
    return keep


def pending_prompts(
    prompts: list[dict],
    done: set[tuple[str, str]],
    limit: int | None,
) -> list[dict]:
    rows = [
        row
        for row in prompts
        if (row["territory_id"], row["condition"]) not in done
    ]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _messages(user_text: str, system_prompt: str, content_style: str) -> list[dict]:
    def content(text: str):
        if content_style == "parts":
            return [{"type": "text", "text": text}]
        return text

    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": content(system_prompt)})
    messages.append({"role": "user", "content": content(user_text)})
    return messages


def _move_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


# One loaded checkpoint per Python session. Re-running the notebook cell (for
# example LIMIT = 1, then LIMIT = None) reuses it instead of loading a second
# copy, which used to leave too little GPU memory and push layers to the CPU.
_LOADED: dict[str, tuple] = {}


def _release_loaded_models() -> None:
    import gc

    _LOADED.clear()
    gc.collect()
    _free_cuda_cache()


def _assert_all_on_gpu(model, hf_id: str) -> None:
    off_gpu = sorted({str(param.device) for param in model.parameters() if param.device.type != "cuda"})
    if off_gpu:
        raise RuntimeError(
            f"{hf_id}: some weights are on {off_gpu}, not the GPU. That makes generation "
            "10-100x slower. Use Runtime > Restart session, then run the notebook again "
            "(finished rows are kept)."
        )


def load_hf_model(spec: ModelSpec):
    """Load one checkpoint in BF16, entirely on GPU 0. No quantization, no CPU offload.

    Raises if CUDA is missing, if the model does not fit (out-of-memory error
    instead of a silent offload), or if any weight ends up off the GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Experiment C generation expects a CUDA GPU (Colab Pro A100, BF16). "
            "Refusing to fall back to CPU or quantized weights."
        )
    if spec.hf_id in _LOADED:
        print(f"reusing {spec.hf_id}, already loaded in this session", flush=True)
        return _LOADED[spec.hf_id]
    # A different checkpoint (or leftovers from an interrupted cell) must not
    # hold GPU memory while this one loads.
    _release_loaded_models()

    dtype = torch.bfloat16
    common = dict(
        dtype=dtype,
        # Everything on GPU 0. With device_map="auto", accelerate silently offloads
        # layers to the CPU when it thinks GPU memory is short.
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
    )

    def _from_pretrained(cls, hf_id: str):
        try:
            return cls.from_pretrained(hf_id, **common)
        except TypeError:
            legacy = dict(common)
            legacy["torch_dtype"] = legacy.pop("dtype")
            return cls.from_pretrained(hf_id, **legacy)

    processor = None
    if spec.architecture == "image_text":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = _from_pretrained(AutoModelForImageTextToText, spec.hf_id)
        processor = AutoProcessor.from_pretrained(
            spec.hf_id,
            trust_remote_code=spec.trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            spec.hf_id,
            trust_remote_code=spec.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            trust_remote_code=spec.trust_remote_code,
        )
        model = _from_pretrained(AutoModelForCausalLM, spec.hf_id)

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    _assert_all_on_gpu(model, spec.hf_id)
    gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"loaded {spec.hf_id}: all weights on GPU, {gb:.1f} GB allocated", flush=True)
    _LOADED[spec.hf_id] = (model, tokenizer, processor)
    return _LOADED[spec.hf_id]


def render_chat(tokenizer, processor, user_text: str, system_prompt: str, content_style: str) -> str:
    """Chat-template text for one prompt.

    Tries the model's preferred content style first, then the other style,
    then the same two without a system turn for templates that reject one.
    """
    template_owner = processor if processor is not None and hasattr(processor, "apply_chat_template") else tokenizer
    alternate = "plain" if content_style == "parts" else "parts"
    attempts = [(content_style, system_prompt), (alternate, system_prompt)]
    if system_prompt.strip():
        attempts.extend([(content_style, ""), (alternate, "")])
    last_error: Exception | None = None
    for style, system in attempts:
        try:
            text = template_owner.apply_chat_template(
                _messages(user_text, system, style),
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(text, list):
                text = text[0]
            return str(text)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _eos_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    config = getattr(model, "generation_config", None)
    value = getattr(config, "eos_token_id", None) if config is not None else None
    if isinstance(value, int):
        ids.add(value)
    elif isinstance(value, (list, tuple)):
        ids.update(int(item) for item in value)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def generate_hf_batch(
    model,
    tokenizer,
    processor,
    user_texts: list[str],
    system_prompt: str,
    max_new_tokens: int,
    content_style: str,
) -> list[tuple[str, int, bool]]:
    """Greedy-decode prompts. Returns (text, new_tokens, truncated) per prompt.

    With batch size 1 there is no padding. With more than one prompt the
    batch is left-padded, which changes outputs; see the module docstring.
    """
    import torch

    texts = [render_chat(tokenizer, processor, text, system_prompt, content_style) for text in user_texts]
    tokenizer.padding_side = "left"
    # The chat template already contains BOS where the model needs one.
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)

    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        device = next(model.parameters()).device
    inputs = _move_to_device(inputs, device)
    input_len = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    results = []
    for index in range(len(texts)):
        seq = output[index, input_len:].tolist()
        # Drop padding that follows an early EOS inside a batch.
        while len(seq) > 1 and seq[-1] == pad_id and seq[-2] in eos_ids | {pad_id}:
            seq.pop()
        truncated = len(seq) >= max_new_tokens and (not seq or seq[-1] not in eos_ids)
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        results.append((text, len(seq), truncated))
    return results


def generate_hf(
    model,
    tokenizer,
    processor,
    user_text: str,
    system_prompt: str,
    max_new_tokens: int,
    content_style: str,
) -> str:
    return generate_hf_batch(
        model, tokenizer, processor, [user_text], system_prompt, max_new_tokens, content_style
    )[0][0]


def run_environment(backend: str) -> dict:
    """Library and hardware versions, stored on every record."""
    if backend != "hf":
        return {"backend": backend}
    info = {"backend": backend}
    try:
        import torch
        import transformers

        info["torch_version"] = torch.__version__
        info["transformers_version"] = transformers.__version__
        try:
            import accelerate

            info["accelerate_version"] = accelerate.__version__
        except ImportError:
            info["accelerate_version"] = ""
        info["cuda_version"] = torch.version.cuda or ""
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception as exc:  # recording versions must never stop a run
        info["environment_error"] = f"{type(exc).__name__}: {exc}"
    return info


def echo_response(prompt: str) -> str:
    """Offline stand-in so the loop can be smoke-tested without weights."""
    return f"ECHO {prompt}"


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_batches(queue: list[dict], batch_size: int, default_budget: int) -> list[list[dict]]:
    """Group by token budget, then by prompt length, so padding stays small."""
    by_budget: dict[int, list[dict]] = {}
    for row in queue:
        by_budget.setdefault(row_budget(row, default_budget), []).append(row)
    batches: list[list[dict]] = []
    for budget in sorted(by_budget):
        rows = sorted(by_budget[budget], key=lambda row: len(row["prompt"]))
        for start in range(0, len(rows), max(1, batch_size)):
            batches.append(rows[start : start + batch_size])
    return batches


def run_generation(
    spec: ModelSpec,
    prompts: list[dict],
    output_path: Path,
    *,
    backend: str = "hf",
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    framings: list[str] | None = None,
    model_bundle=None,
) -> dict:
    """``max_new_tokens`` is the fallback for prompt rows without their own budget."""
    if backend not in {"hf", "echo"}:
        raise ValueError(f"Unknown backend {backend!r}")

    if batch_size != 1:
        print(
            f"WARNING: batch_size={batch_size}. Batched outputs differ from batch-size-1 "
            "outputs, so do not use this run for analysis.",
            flush=True,
        )
    in_scope = filter_framings(prompts, framings)
    done = current_keys(completed_records(output_path, spec.model_key), in_scope, batch_size, max_new_tokens)
    queue = pending_prompts(in_scope, done, limit)
    content_style = "parts" if spec.architecture == "image_text" else "plain"
    environment = run_environment(backend)

    model = tokenizer = processor = None
    if backend == "hf":
        if model_bundle is None:
            model, tokenizer, processor = load_hf_model(spec)
        else:
            model, tokenizer, processor = model_bundle

    def generate_rows(rows: list[dict]) -> list[tuple[str, str, int, bool]]:
        """(response, error, new_tokens, truncated) per row. Splits the batch on failure."""
        budget = row_budget(rows[0], max_new_tokens)
        try:
            if backend == "echo":
                results = []
                for row in rows:
                    text = echo_response(row["prompt"])
                    results.append((text, len(text.split()), False))
            else:
                results = generate_hf_batch(
                    model,
                    tokenizer,
                    processor,
                    [row["prompt"] for row in rows],
                    system_prompt,
                    budget,
                    content_style,
                )
            return [(text, "", n_new, truncated) for text, n_new, truncated in results]
        except Exception as exc:  # generation failure is not a refusal
            if len(rows) > 1:
                _free_cuda_cache()
                half = len(rows) // 2
                return generate_rows(rows[:half]) + generate_rows(rows[half:])
            return [("", f"{type(exc).__name__}: {exc}", 0, False)]

    written = 0
    refused_n = 0
    error_n = 0
    truncated_n = 0
    total = len(queue)
    started = time.perf_counter()
    for batch in make_batches(queue, batch_size, max_new_tokens):
        batch_start = time.perf_counter()
        outcomes = generate_rows(batch)
        seconds_each = (time.perf_counter() - batch_start) / len(batch)
        for row, (response, error, new_tokens, truncated) in zip(batch, outcomes):
            refused = False
            if not error and not response.strip():
                error = "empty_response"
            elif not error and row.get("framing") != "F3":
                refused = is_blanket_refusal(response)

            record = {
                "model_key": spec.model_key,
                "family": spec.family,
                "lineage": spec.lineage,
                "hf_id": spec.hf_id,
                "territory_id": row["territory_id"],
                "condition": row["condition"],
                "framing": row.get("framing", ""),
                "instruction": row.get("instruction", ""),
                "target_party": row.get("target_party", ""),
                "option_order": row.get("option_order", ""),
                "prompt": row["prompt"],
                "response": response,
                "refused": refused,
                "error": error,
                "max_new_tokens": row_budget(row, max_new_tokens),
                "new_tokens": new_tokens,
                "truncated": truncated,
                "gen_seconds": round(seconds_each, 3),
                "batch_size": batch_size,
                "temperature": 0,
                "do_sample": False,
                **environment,
            }
            append_jsonl(output_path, record)
            written += 1
            refused_n += int(refused)
            error_n += int(bool(error))
            truncated_n += int(bool(truncated))
        elapsed = time.perf_counter() - started
        eta_min = elapsed / written * (total - written) / 60 if written else 0.0
        print(
            f"[{written}/{total}] last={batch[-1]['territory_id']} {batch[-1]['condition']} "
            f"{seconds_each:.1f}s/prompt refused={refused_n} errors={error_n} "
            f"truncated={truncated_n} eta={eta_min:.0f}min",
            flush=True,
        )

    return {
        "model_key": spec.model_key,
        "written": written,
        "skipped_existing": len(done),
        "refused": refused_n,
        "errors": error_n,
        "truncated": truncated_n,
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Experiment C answers locally.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "echo"], default="hf")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--framings",
        nargs="*",
        default=None,
        help="Only these framings (for example F1 F2 F3). Default: every row in the prompts file.",
    )
    args = parser.parse_args()

    spec = get_model(args.model_key)
    system_prompt = "" if args.no_system_prompt else args.system_prompt
    stats = run_generation(
        spec,
        load_prompts(args.prompts),
        args.output,
        backend=args.backend,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        batch_size=args.batch_size,
        framings=args.framings,
    )
    print(stats)


if __name__ == "__main__":
    main()
