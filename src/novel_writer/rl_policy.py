"""
RL-style runtime policy loader and helpers.

This module keeps code-level tuning parameters outside prompt text so we can
optimize behavior (scene counts, temperatures, cast fallback size, history cap)
using benchmark rewards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_POLICY_PATH = "data/rl_policy.json"
FIXED_PROSE_HISTORY_MAX_EPISODES = 999

DEFAULT_POLICY: dict[str, Any] = {
    "version": 1,
    "scene_target_bias": 0,
    "scene_target_min": 3,
    "scene_target_max": 10,
    "distiller_temperature": 0.30,
    "distiller_max_tokens": 4000,
    "prose_scene_temperature": 0.75,
    "prose_transition_temperature": 0.70,
    "prose_polish_temperature": 0.40,
    "prose_anchor_fix_temperature": 0.35,
    # Fixed by design for long-form continuity:
    # keep all available prior episodes in context (token budget should be
    # handled by summarization, not by forgetting).
    "prose_history_max_episodes": FIXED_PROSE_HISTORY_MAX_EPISODES,
    # Runtime fallback cast is now conditional in DirectorAI; keep this as a
    # legacy field but pin it to 1 to preserve "monologue possible" default.
    "director_fallback_cast_size": 1,
}


def _apply_fixed_policy_guards(policy: dict[str, Any]) -> dict[str, Any]:
    """Pin user-requested invariants so RL/search does not drift them."""
    out = dict(policy)
    out["prose_history_max_episodes"] = FIXED_PROSE_HISTORY_MAX_EPISODES
    # Director fallback is conditional at runtime. Keep stored field stable.
    out["director_fallback_cast_size"] = 1
    return out


def _policy_path(path: str | None = None) -> Path:
    raw = path or os.environ.get("RL_POLICY_PATH") or DEFAULT_POLICY_PATH
    return Path(raw)


def load_policy(path: str | None = None) -> dict[str, Any]:
    p = _policy_path(path)
    if not p.exists():
        return dict(DEFAULT_POLICY)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_POLICY)
    if not isinstance(payload, dict):
        return dict(DEFAULT_POLICY)
    merged = dict(DEFAULT_POLICY)
    merged.update(payload)
    return _apply_fixed_policy_guards(merged)


def save_policy(policy: dict[str, Any], path: str | None = None) -> str:
    p = _policy_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(DEFAULT_POLICY)
    if isinstance(policy, dict):
        merged.update(policy)
    merged = _apply_fixed_policy_guards(merged)
    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def episode_runtime_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Compact subset attached to episode_config for runtime consumers."""
    return {
        # Kept for logging/compatibility; DirectorAI computes conditional size.
        "director_fallback_cast_size": int(policy.get("director_fallback_cast_size", 1) or 1),
    }


def tuned_scene_target(base_target: int, policy: dict[str, Any]) -> int:
    bias = int(policy.get("scene_target_bias", 0) or 0)
    lo = int(policy.get("scene_target_min", 3) or 3)
    hi = int(policy.get("scene_target_max", 10) or 10)
    return max(lo, min(hi, int(base_target) + bias))
