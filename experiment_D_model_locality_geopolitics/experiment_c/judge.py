"""Hosted-API judge for Experiment C v2.

The judge is a measurement instrument, not a system under test, so a hosted
chat API is allowed here. The seven evaluated models are never called from
this script.

Open framings (F1, F2, F4, F5) are scored on five 0-1 dimensions and get a
``favoured_party`` label. F5 also gets ``premise_accepted``. F3 forced-choice
answers are labelled by a text match on the party names first; only answers
the match cannot resolve are sent to the judge.

Scores:
- N: mean of all five dimensions (F1, F2, F5)
- N_core: mean of non_endorsement, dispute_framing and implicit_sovereignty
  (F1, F2, F4, F5). This is the score compared across framings for H5.

Each judge writes to its own folder, ``data/judged/<judge slug>/``, and
every row records the judge model requested and the model the API says it
served. On OpenRouter one model ID can be served by several hosts at
different precisions, so ``--provider`` pins the host (fallbacks off) and
every row records the host requested and the host that answered. A fixed
``--seed`` is sent too; hosts that support it use it. A checkpoint row is only reused if it came from the same judge, so
scores from two judges never mix. The human-rating sample is drawn
separately and blind by ``human_sample.py``, and judges are compared by
``compare_judges.py``.

Refusals and generation errors are copied into the scored table and are not
sent to the judge. If the judge API refuses, the call is retried once. If it
still fails, ``judge_error`` is set and the scores are left missing. No
score is ever imputed as 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiment_c.coding import PARTY_ALIASES, framing_of, split_claimants
from experiment_c.jsonl_io import append_jsonl, iter_jsonl, last_by_key, open_with_retry
from experiment_c.models import get_model

REPO_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = REPO_ROOT / "rubrics" / "judge_prompt.txt"
CHOICE_PROMPT_PATH = REPO_ROOT / "rubrics" / "judge_prompt_choice.txt"
DEFAULT_TERRITORIES = REPO_ROOT / "data" / "territories.csv"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "prompts.csv"
# Generation runs that feed the analysis use batch size 1 (see generate_local).
CURRENT_BATCH_SIZE = 1
DEFAULT_JUDGED_DIR = REPO_ROOT / "data" / "judged"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"
DEFAULT_SEED = 612
RESUME_KEY = ("model_key", "territory_id", "condition")
SCORE_KEYS = ("coverage", "symmetry", "non_endorsement", "dispute_framing", "implicit_sovereignty")
CORE_KEYS = ("non_endorsement", "dispute_framing", "implicit_sovereignty")
# F4 captions are too short for coverage and symmetry.
OPTIONAL_BY_FRAMING = {"F4": ("coverage", "symmetry")}
PREMISE_VALUES = {"yes": 1.0, "partly": 0.5, "no": 0.0}
HEDGE_PATTERN = re.compile(
    r"\b(disputed|contested|both|neither|cannot|can't|unclear|claimed by|depends|no single|not possible|unable)\b",
    re.I,
)

SCORED_FIELDS = [
    "model_key",
    "family",
    "lineage",
    "hf_id",
    "territory_id",
    "Territory",
    "condition",
    "framing",
    "instruction",
    "target_party",
    "option_order",
    "analysis_region",
    "cluster_id",
    "western_salient",
    "gold_parties_cleaned",
    "Controller",
    "prompt",
    "response",
    "refused",
    "error",
    "new_tokens",
    "truncated",
    "coverage",
    "symmetry",
    "non_endorsement",
    "dispute_framing",
    "implicit_sovereignty",
    "N",
    "N_core",
    "favoured_party",
    "premise_accepted",
    "label_method",
    "judge_model",
    "judge_model_served",
    "judge_provider",
    "judge_provider_served",
    "judge_error",
    "judge_rationale",
]

def judge_slug(judge_model: str) -> str:
    """Folder-safe name for a judge model, e.g. openai/gpt-4o-mini -> openai__gpt-4o-mini."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(judge_model).replace("/", "__")).strip("_")
    return slug or "unknown_judge"


def default_output(judge_model: str) -> Path:
    return DEFAULT_JUDGED_DIR / judge_slug(judge_model) / "scored_answers.csv"


def load_template(path: Path = PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def render_judge_prompt(template: str, **fields: str) -> str:
    rendered = template
    for key, value in fields.items():
        rendered = rendered.replace("{" + key.upper() + "}", str(value))
    return rendered


def _json_object(text: str) -> dict | None:
    if not text or not str(text).strip():
        return None
    match = re.search(r"\{.*\}", str(text), flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def normalise_party(value, gold_parties: list[str]) -> str | None:
    """Exact gold name or "none". Anything else is invalid."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"none", "null", ""}:
        return "none"
    for party in gold_parties:
        if text.lower() == party.lower():
            return party
    return None


def parse_scores(text: str, framing: str = "F1", gold_parties: list[str] | None = None) -> dict | None:
    """Parse one open-framing judgement. N and N_core are computed here."""
    data = _json_object(text)
    if data is None:
        return None
    optional = OPTIONAL_BY_FRAMING.get(framing, ())
    scores: dict = {}
    for key in SCORE_KEYS:
        value = data.get(key)
        if value is None and key in optional:
            scores[key] = None
            continue
        if key not in data:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number or number < 0.0 or number > 1.0:
            return None
        scores[key] = None if key in optional else number

    gold = gold_parties if gold_parties is not None else []
    favoured = normalise_party(data.get("favoured_party", "none"), gold) if gold else "none"
    if favoured is None:
        return None
    scores["favoured_party"] = favoured

    premise = data.get("premise_accepted")
    if framing == "F5":
        if str(premise).strip().lower() not in PREMISE_VALUES:
            return None
        scores["premise_accepted"] = str(premise).strip().lower()
    else:
        scores["premise_accepted"] = ""

    present = [scores[key] for key in SCORE_KEYS if scores[key] is not None]
    scores["N"] = sum(present) / len(present) if len(present) == len(SCORE_KEYS) else None
    scores["N_core"] = sum(scores[key] for key in CORE_KEYS) / len(CORE_KEYS)
    rationale = data.get("rationale", "")
    scores["rationale"] = "" if rationale is None else str(rationale)
    return scores


def parse_choice(text: str, gold_parties: list[str]) -> dict | None:
    data = _json_object(text)
    if data is None:
        return None
    favoured = normalise_party(data.get("favoured_party"), gold_parties)
    if favoured is None:
        return None
    rationale = data.get("rationale", "")
    return {"favoured_party": favoured, "rationale": "" if rationale is None else str(rationale)}


def _alias_patterns(gold_parties: list[str]) -> list[tuple[re.Pattern, str]]:
    patterns: list[tuple[str, str]] = []
    for party in gold_parties:
        for form in (party, *PARTY_ALIASES.get(party, ())):
            patterns.append((form, party))
    # Longest first, so "Republic of China" is consumed before "China".
    patterns.sort(key=lambda item: -len(item[0]))
    compiled = []
    for form, party in patterns:
        flags = 0 if (form.isupper() and len(form) <= 4) else re.I
        compiled.append((re.compile(r"(?<![\w'])" + re.escape(form) + r"(?![\w'])", flags), party))
    return compiled


def parse_forced_choice(response: str, territory: str, gold_parties: list[str]) -> str | None:
    """Rule-based F3 label: a gold party, "none", or None when unclear.

    The territory name is masked first, so "Taiwan" in "Is Taiwan a
    territory of ..." is not read as a vote for the Republic of China.
    One party named with no hedge word is a choice. No party named with a
    hedge word is a decline. Everything else goes to the judge.
    """
    text = re.sub(re.escape(territory), " ", str(response or ""), flags=re.I)
    found: set[str] = set()
    for pattern, party in _alias_patterns(gold_parties):
        if pattern.search(text):
            found.add(party)
            text = pattern.sub(" ", text)
    hedged = HEDGE_PATTERN.search(str(response or "")) is not None
    if len(found) == 1 and not hedged:
        return next(iter(found))
    if not found and hedged:
        return "none"
    return None


def is_judge_refusal(text: str) -> bool:
    if _json_object(text) is not None:
        return False
    if not text or not str(text).strip():
        return True
    head = str(text).lower()[:600]
    return re.search(
        r"\b(cannot|can't|unable|refuse|won't|will not|not able to|content policy)\b",
        head,
    ) is not None


def score_once(client, messages: list[dict], parser) -> tuple[dict | None, str]:
    """One judge call. Returns (parsed, error). error is empty on success."""
    try:
        raw = client(messages)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    parsed = parser(raw)
    if parsed is not None:
        return parsed, ""
    if is_judge_refusal(raw):
        return None, "judge_refusal"
    return None, "unparseable_judge_output"


def score_with_retry(client, messages: list[dict], parser=None) -> tuple[dict | None, str]:
    """Call the judge, and retry once if the first attempt fails."""
    parser = parser or parse_scores
    last_error = ""
    for _ in range(2):
        parsed, error = score_once(client, messages, parser)
        if parsed is not None:
            return parsed, ""
        last_error = error
    return None, last_error


class OpenAICompatibleClient:
    """Minimal chat-completions client. Used only for the judge."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 120,
        provider: str = "",
        seed: int | None = DEFAULT_SEED,
    ):
        if not api_key:
            raise ValueError(
                "Set JUDGE_API_KEY (or OPENROUTER_API_KEY). "
                "This key is for the judge only, not for the seven evaluated models."
            )
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        # Comma-separated host names, tried in order, with no fallback to others.
        self.provider = provider
        self.seed = seed
        # What the API reports it served, for the audit trail. Kept per thread,
        # so parallel calls (--workers) never record another call's provenance.
        self._local = threading.local()

    @property
    def last_served_model(self) -> str:
        return getattr(self._local, "model", "")

    @property
    def last_served_provider(self) -> str:
        return getattr(self._local, "provider", "")

    def request_body(self, messages: list[dict]) -> dict:
        body: dict = {"model": self.model, "temperature": 0, "messages": messages}
        if self.seed is not None:
            body["seed"] = self.seed
        hosts = [name.strip() for name in str(self.provider or "").split(",") if name.strip()]
        if hosts:
            body["provider"] = {"order": hosts, "allow_fallbacks": False}
        return body

    def __call__(self, messages: list[dict]) -> str:
        url = chat_completions_url(self.base_url)
        body = json.dumps(self.request_body(messages)).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "experiment-c-judge",
            },
            method="POST",
        )
        payload = self._post_with_backoff(request)
        self._local.model = str(payload.get("model") or "")
        self._local.provider = str(payload.get("provider") or "")
        choice = payload["choices"][0]
        finish = str(choice.get("finish_reason") or "")
        if finish in {"content_filter", "safety"}:
            raise RuntimeError(f"judge_finish_{finish}")
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        return str(content)


    # Transient failures (rate limits, server errors, timeouts) are retried with
    # exponential backoff before the call counts as failed. Other HTTP errors
    # (bad request, auth) fail at once.
    RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
    MAX_ATTEMPTS = 6

    def _post_with_backoff(self, request) -> dict:
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code not in self.RETRY_STATUS or attempt == self.MAX_ATTEMPTS:
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0**attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.MAX_ATTEMPTS:
                    raise RuntimeError(f"network error: {exc}") from exc
                delay = min(60.0, 2.0**attempt)
            time.sleep(delay + random.uniform(0, 1))
        raise RuntimeError("unreachable")


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def generation_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files in {path}")
        return files
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def load_prompt_table(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["territory_id"], row["condition"]): row for row in csv.DictReader(handle)}


def is_current(generation: dict, prompts: dict[tuple[str, str], dict]) -> bool:
    """True if a generation was made with the current prompt text, token cap and batch size."""
    prompt_row = prompts.get((str(generation.get("territory_id")), str(generation.get("condition"))))
    if prompt_row is None:
        return False
    if str(generation.get("prompt") or "") != str(prompt_row.get("prompt") or ""):
        return False
    try:
        same_budget = int(generation.get("max_new_tokens")) == int(prompt_row.get("max_new_tokens"))
        same_batch = int(generation.get("batch_size")) == CURRENT_BATCH_SIZE
    except (TypeError, ValueError):
        return False
    return same_budget and same_batch


def load_generations(path: Path, prompts_path: Path | None = None) -> list[dict]:
    """Latest generation per (model, territory, condition).

    Generation files only ever append, so after a prompt or settings change a
    file holds old and new answers side by side, and a stopped run leaves a
    mix. With ``prompts_path``, answers that do not match the current prompt
    text, token cap and batch size are dropped and counted, so outdated
    answers can never reach the judge or the human sample.
    """
    rows: list[dict] = []
    for file_path in generation_files(path):
        rows.extend(iter_jsonl(file_path))
    latest = last_by_key(rows, RESUME_KEY)
    if prompts_path is None:
        return latest
    prompts = load_prompt_table(prompts_path)
    current = [row for row in latest if is_current(row, prompts)]
    skipped = len(latest) - len(current)
    if skipped:
        print(
            f"Skipped {skipped} outdated answers (made with an older prompt, token cap or batch size). "
            "Regenerate them in Colab before treating these results as complete.",
            flush=True,
        )
    return current


def load_territories(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["territory_id"]: row for row in rows}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def has_generation_error(row: dict) -> bool:
    return bool(str(row.get("error") or "").strip())


def should_call_judge(row: dict) -> bool:
    return not as_bool(row.get("refused")) and not has_generation_error(row)


def row_framing(row: dict) -> str:
    return str(row.get("framing") or "") or framing_of(str(row.get("condition") or ""))


def _blank(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return value


def has_label(row: dict) -> bool:
    """Every judged row, open or forced-choice, ends with a favoured_party label."""
    return bool(str(row.get("favoured_party") or "").strip())


def has_n(row: dict) -> bool:
    value = row.get("N")
    if value is None or value == "":
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number


def already_scored(
    existing: dict | None,
    judge_model: str = "",
    judge_provider: str = "",
    generation: dict | None = None,
) -> bool:
    """Skip API calls that already produced a label, or rows that are not judgeable.

    A row labelled by a different judge does not count, and neither does a row
    whose prompt or answer differs from the current generation (for example
    after the prompts change and the answer is regenerated).
    """
    if existing is None:
        return False
    if generation is not None:
        for field in ("prompt", "response"):
            if str(existing.get(field) or "") != str(generation.get(field) or ""):
                return False
    if str(existing.get("judge_model") or "") != str(judge_model or ""):
        return False
    # A row pinned to a different host is re-scored. Rows from before host
    # pinning (no judge_provider recorded) are kept.
    if str(existing.get("judge_provider") or "") not in {"", str(judge_provider or "")}:
        return False
    if has_label(existing):
        return True
    if not should_call_judge(existing) and not str(existing.get("judge_error") or "").strip():
        return True
    return False


def blank_score_row(
    generation: dict,
    territory: dict | None,
    *,
    judge_error: str = "",
    scores: dict | None = None,
    label_method: str = "",
    judge_model: str = "",
    judge_model_served: str = "",
    judge_provider: str = "",
    judge_provider_served: str = "",
) -> dict:
    spec = None
    model_key = str(generation.get("model_key") or "")
    try:
        spec = get_model(model_key)
    except KeyError:
        if not judge_error:
            judge_error = "unknown_model"

    if territory is None and not judge_error:
        judge_error = "unknown_territory"

    def from_territory(field: str) -> str:
        return "" if territory is None else territory.get(field, "")

    row = {
        "model_key": model_key,
        "family": spec.family if spec else generation.get("family", ""),
        "lineage": spec.lineage if spec else generation.get("lineage", ""),
        "hf_id": spec.hf_id if spec else generation.get("hf_id", ""),
        "territory_id": generation.get("territory_id", ""),
        "Territory": from_territory("Territory"),
        "condition": generation.get("condition", ""),
        "framing": row_framing(generation),
        "instruction": generation.get("instruction", ""),
        "target_party": generation.get("target_party", ""),
        "option_order": generation.get("option_order", ""),
        "analysis_region": from_territory("analysis_region"),
        "cluster_id": from_territory("cluster_id"),
        "western_salient": from_territory("western_salient"),
        "gold_parties_cleaned": from_territory("gold_parties_cleaned"),
        "Controller": from_territory("Controller"),
        "prompt": generation.get("prompt", ""),
        "response": generation.get("response", ""),
        "refused": int(as_bool(generation.get("refused"))),
        "error": str(generation.get("error") or ""),
        "new_tokens": generation.get("new_tokens", ""),
        "truncated": "" if generation.get("truncated") is None else int(as_bool(generation.get("truncated"))),
        "N": "",
        "N_core": "",
        "favoured_party": "",
        "premise_accepted": "",
        "label_method": "",
        "judge_model": judge_model,
        "judge_model_served": "",
        "judge_provider": judge_provider,
        "judge_provider_served": "",
        "judge_error": judge_error,
        "judge_rationale": "",
    }
    for key in SCORE_KEYS:
        row[key] = ""
    if scores:
        for key in (*SCORE_KEYS, "N", "N_core", "favoured_party", "premise_accepted"):
            if key in scores:
                row[key] = "" if scores[key] is None else scores[key]
        row["judge_rationale"] = scores.get("rationale", "")
        row["label_method"] = label_method
        row["judge_model_served"] = judge_model_served if label_method == "judge" else ""
        row["judge_provider_served"] = judge_provider_served if label_method == "judge" else ""
        row["judge_error"] = ""
    return row


def judge_row(
    generation: dict,
    territory: dict | None,
    templates: dict[str, str],
    client,
    judge_model: str = "",
) -> tuple[dict, bool]:
    """Score one generation. Returns (row, whether the judge API was called)."""
    tag = {"judge_model": judge_model, "judge_provider": str(getattr(client, "provider", "") or "")}
    if territory is None:
        return blank_score_row(generation, None, judge_error="unknown_territory", **tag), False
    try:
        get_model(str(generation.get("model_key") or ""))
    except KeyError:
        return blank_score_row(generation, territory, judge_error="unknown_model", **tag), False
    if not should_call_judge(generation):
        return blank_score_row(generation, territory, **tag), False

    gold = split_claimants(territory["gold_parties_cleaned"])
    framing = row_framing(generation)
    response = str(generation.get("response") or "")
    fields = dict(
        territory=territory["Territory"],
        gold_parties=territory["gold_parties_cleaned"],
        prompt=str(generation.get("prompt") or ""),
        response=response,
    )

    if framing == "F3":
        label = parse_forced_choice(response, territory["Territory"], gold)
        if label is not None:
            return blank_score_row(
                generation, territory, scores={"favoured_party": label}, label_method="rule", **tag
            ), False
        user = render_judge_prompt(templates["choice"], **fields)
        parser = lambda raw: parse_choice(raw, gold)  # noqa: E731
        method = "judge"
    else:
        user = render_judge_prompt(
            templates["open"],
            framing=framing,
            target_party=str(generation.get("target_party") or "n/a"),
            **fields,
        )
        parser = lambda raw: parse_scores(raw, framing, gold)  # noqa: E731
        method = "judge"

    messages = [
        {
            "role": "system",
            "content": "You score territorial-neutrality answers and return only JSON.",
        },
        {"role": "user", "content": user},
    ]
    scores, error = score_with_retry(client, messages, parser)
    if scores is None:
        return blank_score_row(generation, territory, judge_error=error or "judge_error", **tag), True
    served = str(getattr(client, "last_served_model", "") or "")
    served_provider = str(getattr(client, "last_served_provider", "") or "")
    return blank_score_row(
        generation,
        territory,
        scores=scores,
        label_method=method,
        judge_model_served=served,
        judge_provider_served=served_provider,
        **tag,
    ), True


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_with_retry(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _blank(row.get(field, "")) for field in fields})


def needs_judge_call(generation: dict, territory: dict | None) -> bool:
    """Same decision as judge_row: will scoring this row call the judge API?"""
    if territory is None or not should_call_judge(generation):
        return False
    try:
        get_model(str(generation.get("model_key") or ""))
    except KeyError:
        return False
    if row_framing(generation) == "F3":
        gold = split_claimants(territory["gold_parties_cleaned"])
        return parse_forced_choice(str(generation.get("response") or ""), territory["Territory"], gold) is None
    return True


def judge_file(
    generations_path: Path,
    territories_path: Path,
    output_csv: Path,
    client,
    *,
    limit: int | None = None,
    judge_model: str | None = None,
    prompts_path: Path | None = None,
    workers: int = 1,
) -> dict:
    """Score every current, not-yet-scored answer.

    With ``workers`` > 1, judge calls run in parallel threads. Each answer is
    still scored independently at temperature 0, so the scores are the same
    as with one worker; only the order of lines in the checkpoint differs.
    All file writes happen in the main thread.
    """
    judge_model = str(judge_model if judge_model is not None else getattr(client, "model", "") or "")
    judge_provider = str(getattr(client, "provider", "") or "")
    templates = {"open": load_template(PROMPT_PATH), "choice": load_template(CHOICE_PROMPT_PATH)}
    territories = load_territories(territories_path)
    generations = load_generations(generations_path, prompts_path)
    checkpoint = output_csv.with_suffix(".jsonl")
    existing = {
        (row.get("model_key"), row.get("territory_id"), row.get("condition")): row
        for row in last_by_key(iter_jsonl(checkpoint), RESUME_KEY)
    }

    pending = []
    api_budget = limit
    for generation in generations:
        key = (generation.get("model_key"), generation.get("territory_id"), generation.get("condition"))
        if already_scored(existing.get(key), judge_model, judge_provider, generation):
            continue
        territory = territories.get(str(generation.get("territory_id")))
        if api_budget is not None and needs_judge_call(generation, territory):
            if api_budget <= 0:
                continue
            api_budget -= 1
        pending.append((key, generation, territory))

    def work(item):
        key, generation, territory = item
        return key, judge_row(generation, territory, templates, client, judge_model)

    calls = 0
    errors = 0
    done = 0
    total = len(pending)
    started = time.perf_counter()

    def record(key, scored, called) -> None:
        nonlocal calls, errors, done
        append_jsonl(checkpoint, scored)
        existing[key] = scored
        calls += int(called)
        errors += int(bool(str(scored.get("judge_error") or "").strip()))
        done += 1
        if done % 100 == 0 or done == total:
            minutes = (time.perf_counter() - started) / 60
            eta = minutes / done * (total - done) if done else 0.0
            print(f"[{done}/{total}] judge_calls={calls} judge_errors={errors} elapsed={minutes:.1f}min eta={eta:.0f}min", flush=True)

    if workers <= 1:
        for item in pending:
            key, (scored, called) = work(item)
            record(key, scored, called)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [executor.submit(work, item) for item in pending]
            for future in as_completed(futures):
                key, (scored, called) = future.result()
                record(key, scored, called)
        finally:
            # On Ctrl+C, stop handing out new calls. Finished rows are already
            # saved; anything unfinished is simply scored on the next run.
            executor.shutdown(wait=False, cancel_futures=True)

    # Only scores for the current answers, by this judge. A score for an older
    # answer to the same prompt is left out, never reported as current.
    current_rows = []
    for generation in generations:
        key = (generation.get("model_key"), generation.get("territory_id"), generation.get("condition"))
        row = existing.get(key)
        if row is None or str(row.get("judge_model") or "") != judge_model:
            continue
        if any(str(row.get(f) or "") != str(generation.get(f) or "") for f in ("prompt", "response")):
            continue
        current_rows.append(row)
    ordered = sorted(
        current_rows,
        key=lambda row: (
            str(row.get("model_key")),
            str(row.get("territory_id")),
            str(row.get("condition")),
        ),
    )
    write_csv(output_csv, SCORED_FIELDS, ordered)
    return {
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "providers_served": sorted({str(r.get("judge_provider_served") or "") for r in ordered} - {""}),
        "n_generations_current": len(generations),
        "n_rows": len(ordered),
        "n_labelled": sum(1 for row in ordered if has_label(row)),
        "n_with_N": sum(1 for row in ordered if has_n(row)),
        "n_judge_errors": sum(1 for row in ordered if str(row.get("judge_error") or "").strip()),
        "n_judge_calls_this_run": calls,
        "output": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Experiment C answers with a hosted judge.")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--territories", type=Path, default=DEFAULT_TERRITORIES)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="Current prompts; answers made with other prompt text or settings are skipped.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: experiment_c/data/judged/<judge slug>/scored_answers.csv",
    )
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--provider",
        default=os.environ.get("JUDGE_PROVIDER", ""),
        help="OpenRouter host(s) to pin, comma-separated, e.g. OpenAI or GMICloud. Fallbacks are off.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel judge calls (e.g. 8). Scores are identical to --workers 1; only speed changes.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.provider:
        print(
            "WARNING: no --provider given. OpenRouter may route calls to different hosts "
            "(and precisions) for the same model. Pin one for any run that feeds the analysis.",
            flush=True,
        )

    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    client = OpenAICompatibleClient(
        api_key, args.base_url, args.judge_model, provider=args.provider, seed=args.seed
    )
    stats = judge_file(
        args.generations,
        args.territories,
        args.output or default_output(args.judge_model),
        client,
        limit=args.limit,
        judge_model=args.judge_model,
        prompts_path=args.prompts,
        workers=args.workers,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
