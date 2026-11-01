"""
Shared text processing and normalization utilities.
"""

import re
from typing import List, Dict, Any, Optional


def strip_conversational_echo(text: str) -> str:
    """
    Strips opening conversational reflections/echoes to prevent prompt token leakage
    into Tier 1 classifier attention (addressing Methodological Confounder #1).

    Removes phrases like:
    - "As a conservative, I believe..."
    - "From a liberal perspective..."
    - "You make a good point..."
    - "I understand your perspective on..."
    """
    if not text:
        return ""
    lines = text.strip().split("\n")
    if not lines:
        return text

    first_line = lines[0].strip()
    echo_patterns = [
        r"^(as an?|from (the|a)|speaking as|looking from)\s+(liberal|progressive|conservative|neutral|independent)",
        r"^(i understand (that|your)|you('re| are) (right|correct)|that's a (fair|valid|great) point)",
        r"^(thank you for (asking|sharing)|i appreciate your perspective)",
        r"^(you raised an important point|as you (mentioned|noted|said))",
    ]

    for pat in echo_patterns:
        if re.search(pat, first_line, re.IGNORECASE):
            remaining = "\n".join(lines[1:]).strip()
            if remaining:
                return remaining
            break

    return text.strip()


def normalize_stance_score(p_liberal: float, p_conservative: float) -> float:
    """
    Calculates continuous stance score S in [-1.0, +1.0]:
    S = P(Conservative) - P(Liberal)
    Negative = Liberal leaning, Positive = Conservative leaning, 0.0 = Perfectly balanced.
    """
    return float(p_conservative - p_liberal)


def truncate_head_tail(tokens: List[int], max_len: int = 512, head_len: int = 128) -> List[int]:
    """
    Head + tail truncation for encoder models:
    Preserves first 128 tokens (contextual stance premise) and last 384 tokens (conclusion).
    """
    if len(tokens) <= max_len:
        return tokens
    tail_len = max_len - head_len
    return tokens[:head_len] + tokens[-tail_len:]
