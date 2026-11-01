"""Locked Experiment C model registry (proposal section 2.2).

Seven instruction-tuned dense models, chosen for locale diversity and
A100 BF16 inference without quantization. Shared Llama and Qwen lineages
are intentional: they are the H4 within-lineage contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One evaluated checkpoint.

    ``architecture`` selects the transformers loader. ``image_text`` models
    (Gemma 3 and the Qwen3-VL pair) still receive text-only chat messages:
    no images and no image tensors. ``text_only_vl`` marks the two VL
    checkpoints the design runs without vision inputs.
    """

    model_key: str
    family: str
    lineage: str
    hf_id: str
    text_only_vl: bool = False
    architecture: str = "causal_lm"
    trust_remote_code: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.family not in {"western", "chinese", "sea"}:
            raise ValueError(f"Unknown family for {self.model_key}: {self.family}")
        if self.architecture not in {"causal_lm", "image_text"}:
            raise ValueError(f"Unknown architecture for {self.model_key}: {self.architecture}")


@dataclass(frozen=True)
class LineageContrast:
    """Planned default-slice contrast: regional post-training minus base instruct."""

    name: str
    lineage: str
    treatment: str
    baseline: str


# GLM: the THUDM org card now resolves as zai-org/glm-4-9b-chat. Zhipu's own
# note (2024-11-25) says to load glm-4-9b-chat-hf on transformers>=4.46 so
# Colab does not need custom modeling code. Same GLM-4-9B-Chat weights.
# Typhoon: scb10x/llama3.1-typhoon2-8b-instruct (full BF16; not an MLX quant).
# Qwen-SEA-LION-v4-8B-VL: aisingapore/Qwen-SEA-LION-v4-8B-VL, text only.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_key="llama-3.1-8b-instruct",
        family="western",
        lineage="llama-3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        notes="Gated Llama 3.1 license. Base of the Llama lineage contrast.",
    ),
    ModelSpec(
        model_key="gemma-3-12b-it",
        family="western",
        lineage="gemma-3-12b",
        hf_id="google/gemma-3-12b-it",
        architecture="image_text",
        notes=(
            "Gated Gemma license. Gemma 3 IT is an image-text checkpoint; "
            "Experiment C sends a text user turn only. Western architecture anchor."
        ),
    ),
    ModelSpec(
        model_key="qwen3-vl-8b-instruct",
        family="chinese",
        lineage="qwen3-vl-8b",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        text_only_vl=True,
        architecture="image_text",
        notes="Text-only. No images. Base of the Qwen lineage contrast.",
    ),
    ModelSpec(
        model_key="glm-4-9b-chat",
        family="chinese",
        lineage="glm-4-9b",
        hf_id="zai-org/glm-4-9b-chat-hf",
        notes=(
            "Legacy id THUDM/glm-4-9b-chat moved under zai-org and needs "
            "trust_remote_code. zai-org/glm-4-9b-chat-hf is the same 9B chat "
            "model in the transformers-native format. Chinese architecture anchor. "
            "Do not enable GLM web-search or tool adapters."
        ),
    ),
    ModelSpec(
        model_key="llama-sea-lion-v3-8b-it",
        family="sea",
        lineage="llama-3.1-8b",
        hf_id="aisingapore/Llama-SEA-LION-v3-8B-IT",
        notes="SEA post-training on the Llama-3.1-8B lineage. Leave any optional safety head off.",
    ),
    ModelSpec(
        model_key="qwen-sea-lion-v4-8b-vl",
        family="sea",
        lineage="qwen3-vl-8b",
        hf_id="aisingapore/Qwen-SEA-LION-v4-8B-VL",
        text_only_vl=True,
        architecture="image_text",
        notes="Text-only. SEA post-training on Qwen3-VL-8B-Instruct. No images.",
    ),
    ModelSpec(
        model_key="llama3.1-typhoon2-8b-instruct",
        family="sea",
        lineage="llama-3.1-8b",
        hf_id="scb10x/llama3.1-typhoon2-8b-instruct",
        notes=(
            "Thai/English Typhoon 2 post-training on Llama-3.1-8B. "
            "Do not load the separate Typhoon2-Safety classifier."
        ),
    ),
)

LINEAGE_CONTRASTS: tuple[LineageContrast, ...] = (
    LineageContrast(
        name="sea-lion-v3 minus llama-3.1-8b-instruct",
        lineage="llama-3.1-8b",
        treatment="llama-sea-lion-v3-8b-it",
        baseline="llama-3.1-8b-instruct",
    ),
    LineageContrast(
        name="typhoon2 minus llama-3.1-8b-instruct",
        lineage="llama-3.1-8b",
        treatment="llama3.1-typhoon2-8b-instruct",
        baseline="llama-3.1-8b-instruct",
    ),
    LineageContrast(
        name="sea-lion-v4-vl minus qwen3-vl-8b-instruct",
        lineage="qwen3-vl-8b",
        treatment="qwen-sea-lion-v4-8b-vl",
        baseline="qwen3-vl-8b-instruct",
    ),
)

MODELS_BY_KEY: dict[str, ModelSpec] = {spec.model_key: spec for spec in MODELS}

if len(MODELS_BY_KEY) != 7:
    raise RuntimeError("Experiment C requires exactly seven model keys.")


def get_model(model_key: str) -> ModelSpec:
    try:
        return MODELS_BY_KEY[model_key]
    except KeyError as exc:
        known = ", ".join(MODELS_BY_KEY)
        raise KeyError(f"Unknown model_key {model_key!r}. Known keys: {known}") from exc
