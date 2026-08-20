"""Speculative voice-turn helpers (Phase 4: early LLM on partial STT)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Tuple


def load_voice_speculative_config(stt_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    section = {}
    if isinstance(stt_config, dict):
        raw = stt_config.get("cartesia")
        if isinstance(raw, dict):
            section = raw
    return {
        "enabled": bool(section.get("speculative", True)),
        "min_chars": int(section.get("speculative_min_chars", 12) or 12),
        "min_words": int(section.get("speculative_min_words", 3) or 3),
        "stable_ms": int(section.get("speculative_stable_ms", 350) or 350),
        "compatible_ratio": float(section.get("speculative_compatible_ratio", 0.72) or 0.72),
    }


def normalize_voice_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return re.sub(r"[^\w\s]", "", normalized)


def count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def should_start_speculative(partial: str, cfg: Dict[str, Any]) -> bool:
    partial = (partial or "").strip()
    if len(partial) < int(cfg.get("min_chars", 12)):
        return False
    if count_words(partial) < int(cfg.get("min_words", 3)):
        return False
    return True


def voice_texts_compatible(started: str, final: str, *, ratio: float = 0.72) -> bool:
    a = normalize_voice_text(started)
    b = normalize_voice_text(final)
    if not a or not b:
        return False
    if a == b:
        return True
    if b.startswith(a) or a.startswith(b):
        return True
    if len(a) >= 8 and len(b) >= 8:
        return SequenceMatcher(None, a, b).ratio() >= ratio
    return False


@dataclass
class VoiceUtteranceState:
    utterance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latest_partial: str = ""
    started_text: str = ""
    session_key: str = ""
    speculative_started: bool = False


SpeakerKey = Tuple[int, int]
