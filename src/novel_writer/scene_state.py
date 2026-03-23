"""
Scene-state primitives shared across the novel-generation pipeline.

Provides:
  - DramaticFunctionLabel   : canonical set of narrative functions
  - CharacterAgenda         : per-character in-scene desire/agenda
  - ScenePhaseRecord        : immutable snapshot of a completed scene phase
  - SceneProgressionTracker : detects backslide, loops, and forces progression
  - BlockingState           : minimal scene-geometry tracking
  - detect_dramatic_loop()  : loop-detection helper used by distiller + polisher

None of these classes are episode-specific: they operate on abstract labels and
cast lists so they work identically across any episode configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Dramatic-Function Labels
# ─────────────────────────────────────────────────────────────────────────────

class DramaticFunctionLabel(str, Enum):
    """Canonical set of scene-level dramatic functions.

    Every DistilledScene carries exactly one dominant value.
    Adjacent scenes with the same value and overlapping cast are merge
    candidates unless a structural transition (exit/entry/access) separates them.
    """
    SETUP                = "setup"
    OBSERVATION          = "observation"
    WARNING              = "warning"
    TEMPTATION           = "temptation"
    OFFER                = "offer"
    PRESSURE             = "pressure"
    NEGOTIATION          = "negotiation"
    REFUSAL              = "refusal"
    CONDITION_SETTING    = "condition_setting"
    REVELATION           = "revelation"
    CONSEQUENCE          = "consequence"
    INTERRUPTION         = "interruption"
    TRANSITION           = "transition"
    COMMITMENT           = "commitment"
    RETREAT              = "retreat"
    REVERSAL             = "reversal"
    ETHICAL_FRAMING      = "ethical_framing"
    ORIENTATION          = "orientation"
    DISCOVERY            = "discovery"
    UNKNOWN              = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "DramaticFunctionLabel":
        """Normalise any distiller string to a label; falls back to UNKNOWN."""
        v = (value or "").strip().lower()
        for member in cls:
            if member.value == v:
                return member
        return cls.UNKNOWN

    def is_progression_function(self) -> bool:
        """True for functions that mark real narrative progress (should not be merged away)."""
        return self in {
            DramaticFunctionLabel.REVERSAL,
            DramaticFunctionLabel.COMMITMENT,
            DramaticFunctionLabel.CONSEQUENCE,
            DramaticFunctionLabel.TRANSITION,
            DramaticFunctionLabel.REVELATION,
            DramaticFunctionLabel.INTERRUPTION,
        }

    # Alias so old string comparisons still work transparently
    def __str__(self) -> str:
        return self.value


# Legacy string aliases kept for backward compatibility with existing prompt code
DRAMATIC_FUNCTION_PROGRESSION_SET = frozenset({
    "reversal", "commitment", "consequence", "transition", "revelation", "interruption",
})

DRAMATIC_FUNCTION_PRESSURE_SET = frozenset({
    "warning", "pressure", "temptation", "offer", "negotiation",
    "condition_setting", "ethical_framing",
})


# ─────────────────────────────────────────────────────────────────────────────
# Character Agenda
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CharacterAgenda:
    """What a character wants right now inside a scene.

    Extracted from character_profiles YAML fields or inferred from
    the episode's character invariants.  All fields are optional so callers
    can populate as much as they know.
    """
    character_id: str
    character_name: str

    # What they want for themselves this scene
    want_now: str = ""
    # What they want the protagonist to do right now
    want_from_protagonist: str = ""
    # What they are actively concealing
    hiding: str = ""
    # The cost or consequence they are implying (without yet revealing)
    implied_cost: str = ""
    # The line they must not cross in this scene (from invariants)
    hard_limit: str = ""
    # One-word function label for how they are behaving
    scene_function: str = ""

    def is_empty(self) -> bool:
        return not any([
            self.want_now, self.want_from_protagonist,
            self.hiding, self.implied_cost, self.hard_limit,
        ])

    def to_prompt_line(self) -> str:
        """Compact single-line representation for inclusion in LLM prompts."""
        parts: list[str] = []
        if self.want_now:
            parts.append(f"원하는 것: {self.want_now}")
        if self.want_from_protagonist:
            parts.append(f"주인공에게 원하는 것: {self.want_from_protagonist}")
        if self.hiding:
            parts.append(f"숨기는 것: {self.hiding}")
        if self.implied_cost:
            parts.append(f"암시하는 대가: {self.implied_cost}")
        if self.hard_limit:
            parts.append(f"넘지 못하는 선: {self.hard_limit}")
        if self.scene_function:
            parts.append(f"씬 기능: {self.scene_function}")
        return f"- {self.character_name}: " + " | ".join(parts) if parts else ""


def extract_character_agendas(
    characters_present: list[str],
    character_profiles: list[dict],
    protagonist_id: str = "",
) -> list[CharacterAgenda]:
    """Extract CharacterAgenda objects from character_profiles YAML data.

    Reads ``agenda``, ``scene_agenda``, ``invariants``, and ``current_goal``
    fields if present — no hardcoding.  Returns one agenda per in-scene
    non-protagonist character (protagonist is tracked separately via POV).
    """
    agendas: list[CharacterAgenda] = []
    name_to_profile: dict[str, dict] = {}
    for p in character_profiles:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).lower().strip()
        name = str(p.get("name", p.get("id", ""))).strip()
        if pid:
            name_to_profile[pid] = p
        if name:
            name_to_profile[name.lower()] = p

    for char_name in characters_present:
        key = char_name.lower().strip()
        profile = name_to_profile.get(key) or {}

        # Skip protagonist — their perspective is the POV lens, not a separate agenda
        char_profile_id = str(profile.get("id", "")).lower()
        if protagonist_id and (
            key == protagonist_id.lower() or
            char_profile_id == protagonist_id.lower()
        ):
            continue

        agenda_raw = profile.get("scene_agenda") or profile.get("agenda") or {}
        invariants = profile.get("character_invariants") or profile.get("invariants") or []
        if isinstance(invariants, str):
            invariants = [invariants]

        agenda = CharacterAgenda(
            character_id=str(profile.get("id", key)),
            character_name=char_name,
            want_now=str(agenda_raw.get("want_now", "") if isinstance(agenda_raw, dict) else ""),
            want_from_protagonist=str(agenda_raw.get("want_from_protagonist", "") if isinstance(agenda_raw, dict) else ""),
            hiding=str(agenda_raw.get("hiding", "") if isinstance(agenda_raw, dict) else ""),
            implied_cost=str(agenda_raw.get("implied_cost", "") if isinstance(agenda_raw, dict) else ""),
            hard_limit=str(invariants[0] if invariants else ""),
            scene_function=str(agenda_raw.get("scene_function", "") if isinstance(agenda_raw, dict) else ""),
        )
        # Fallback: try current_goal / goal if agenda fields absent
        if agenda.is_empty():
            goal = profile.get("current_goal") or profile.get("goal") or ""
            if goal:
                agenda.want_now = str(goal)
        agendas.append(agenda)
    return agendas


# ─────────────────────────────────────────────────────────────────────────────
# Scene Phase Record (immutable snapshot)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenePhaseRecord:
    """Immutable record of one completed scene phase.

    Stored in SceneProgressionTracker to detect backslide and dramatic loops.
    """
    scene_number: int
    dramatic_function: str          # DramaticFunctionLabel value
    location: str
    cast_key: frozenset             # frozenset of character ids (normalised)
    had_structural_transition: bool # True if entry/exit/access-change occurred
    # Summary tokens used for thematic overlap detection
    theme_tokens: frozenset = field(default_factory=frozenset)


# ─────────────────────────────────────────────────────────────────────────────
# Scene Progression Tracker
# ─────────────────────────────────────────────────────────────────────────────

class SceneProgressionTracker:
    """Tracks which scene phases have been completed and prevents backslide.

    Used by the distiller after scene list is built, and optionally by the
    polisher when scanning for structural problems in generated prose.
    """

    def __init__(self, policy: Optional[dict] = None) -> None:
        self._policy = policy or {}
        self._records: list[ScenePhaseRecord] = []
        # cast_location_function fingerprints of completed scenes
        self._completed: set[tuple] = set()

    def record(self, record: ScenePhaseRecord) -> None:
        self._records.append(record)
        fp = self._fingerprint(record)
        self._completed.add(fp)

    def would_backslide(self, record: ScenePhaseRecord) -> bool:
        """Return True if adding this record would be a backward repetition.

        A backslide is defined as: same (cast, location, dramatic_function)
        tuple appearing after it was already completed AND no structural
        transition (entry/exit/access) separates it from its previous instance.
        """
        fp = self._fingerprint(record)
        return fp in self._completed

    def detect_loop(
        self,
        window: int = 3,
        same_conflict_threshold: Optional[int] = None,
    ) -> Optional[str]:
        """Return a description string if a dramatic loop is detected, else None.

        A loop requires:
        - cast overlap >= 50% across the last `window` records
        - same dramatic_function in >= 2 of the last `window` records
        - no progression function (reversal/consequence/commitment/transition) in the window
        - no structural transition in the window
        """
        threshold = same_conflict_threshold or self._policy.get(
            "same_conflict_loop_turn_threshold", 3
        )
        if len(self._records) < threshold:
            return None

        recent = self._records[-window:]

        # Check for progression function — loop cannot exist if one appeared
        has_progression = any(
            DramaticFunctionLabel.from_string(r.dramatic_function).is_progression_function()
            for r in recent
        )
        if has_progression:
            return None

        # Check for structural transition
        has_transition = any(r.had_structural_transition for r in recent)
        if has_transition:
            return None

        # Cast overlap
        all_casts = [r.cast_key for r in recent]
        if not all_casts:
            return None
        union = frozenset().union(*all_casts)
        intersection = all_casts[0].intersection(*all_casts[1:]) if len(all_casts) > 1 else all_casts[0]
        overlap_ratio = len(intersection) / len(union) if union else 0.0

        # Repeated dramatic function
        functions = [r.dramatic_function for r in recent]
        most_common_fn = max(set(functions), key=functions.count)
        fn_count = functions.count(most_common_fn)

        if overlap_ratio >= 0.5 and fn_count >= max(2, threshold - 1):
            logger.info(
                "[SceneState] Dramatic loop detected: function='%s' repeated %d/%d times, "
                "cast overlap=%.0f%% — progression must be forced.",
                most_common_fn, fn_count, len(recent), overlap_ratio * 100,
            )
            return (
                f"loop: function='{most_common_fn}' ×{fn_count}, "
                f"cast_overlap={overlap_ratio:.0%}, no_transition"
            )
        return None

    @staticmethod
    def _fingerprint(record: ScenePhaseRecord) -> tuple:
        return (record.location, record.cast_key, record.dramatic_function)


# ─────────────────────────────────────────────────────────────────────────────
# Blocking State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlockingState:
    """Minimal scene-geometry tracking for a single scene.

    Prose generator uses this to inject one grounded spatial cue per scene
    without hardcoding room layouts.
    """
    location: str = ""
    character_positions: dict[str, str] = field(default_factory=dict)  # name -> position hint
    last_entrant: str = ""
    last_exit: str = ""
    # True if any blocking cue was already rendered in the current scene
    blocking_rendered: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Dramatic Loop Detection (prose-level, used by polisher)
# ─────────────────────────────────────────────────────────────────────────────

# Abstract gravitas nouns that flag tension-by-label (rather than by event)
_ABSTRACT_TENSION_NOUNS = frozenset({
    "위험", "선택", "부담", "책임", "운명", "압박", "긴장", "두려움", "불안",
    "위기", "딜레마", "갈등", "결단", "각오", "결심", "결의",
    "danger", "risk", "burden", "pressure", "tension", "choice", "decision",
    "fate", "destiny", "responsibility",
})

# Concrete consequence markers — their presence near abstract nouns is "OK"
_CONCRETE_CONSEQUENCE_MARKERS = frozenset({
    "거절", "승인", "중단", "문서", "접근", "차단", "알림", "경보", "퇴장",
    "잠금", "해제", "서명", "취소", "체포", "요청", "보고",
    "입장", "퇴실", "폭로", "드러났", "밝혀", "확인",
    "blocked", "revealed", "denied", "approved", "locked", "unlocked",
})

# Stock body/gesture cues (cap these per scene)
_STOCK_BODY_CUES = frozenset({
    "주먹", "어깨", "시선", "눈빛", "입술", "턱", "손목", "숨을", "심호흡",
    "침묵", "멈췄", "굳었", "떨렸", "좁혔", "내려다",
})

# Stock connective openers to diversify
STOCK_CONNECTIVES = frozenset({
    "그리고", "그러자", "다만", "하지만", "그래서", "그러나", "또한", "또",
})

# Legacy alias kept for backward compatibility
_STOCK_CONNECTIVES = STOCK_CONNECTIVES


def detect_abstract_tension_without_consequence(
    paragraphs: list[str],
    window: int = 2,
) -> list[int]:
    """Return paragraph indices where abstract tension appears without nearby consequence.

    Used by the polisher to flag paragraphs for structural repair.
    ``window`` controls how many following paragraphs to scan for consequence.
    """
    flagged: list[int] = []
    for i, para in enumerate(paragraphs):
        # Count abstract nouns
        para_lower = para.lower()
        abstract_count = sum(1 for n in _ABSTRACT_TENSION_NOUNS if n in para_lower)
        if abstract_count < 2:
            continue
        # Check window ahead for concrete consequence
        window_text = " ".join(paragraphs[i: i + window + 1]).lower()
        has_consequence = any(m in window_text for m in _CONCRETE_CONSEQUENCE_MARKERS)
        if not has_consequence:
            flagged.append(i)
    return flagged


def count_stock_body_cues(text: str) -> dict[str, int]:
    """Return per-cue usage counts for stock body/gesture cues."""
    text_lower = text.lower()
    return {cue: text_lower.count(cue) for cue in _STOCK_BODY_CUES if cue in text_lower}


def count_abstract_tension_nouns(text: str) -> dict[str, int]:
    """Return per-noun usage counts for abstract tension nouns."""
    text_lower = text.lower()
    return {noun: text_lower.count(noun) for noun in _ABSTRACT_TENSION_NOUNS if noun in text_lower}


def count_stock_connective_openers(paragraphs: list[str]) -> dict[str, int]:
    """Return per-connective count of paragraphs that START with that connective."""
    counts: dict[str, int] = {}
    for p in paragraphs:
        for conn in _STOCK_CONNECTIVES:
            if p.strip().startswith(conn):
                counts[conn] = counts.get(conn, 0) + 1
    return counts
