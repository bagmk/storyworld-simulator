"""
Scene constraints, scene state tracking, and voice profile helpers.

This module is standalone — no imports from other project modules (except logging).
All classes are content-agnostic and backward-compatible.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DramaticFunction(str, Enum):
    ORIENTATION     = "orientation"
    DISCOVERY       = "discovery"
    WARNING         = "warning"
    PRESSURE        = "pressure"
    ETHICAL_FRAMING = "ethical_framing"
    NEGOTIATION     = "negotiation"
    REFUSAL         = "refusal"
    CONDITION       = "condition_setting"
    CONSEQUENCE     = "consequence"
    TRANSITION      = "transition"
    REVERSAL        = "reversal"
    COMMITMENT      = "commitment"
    UNKNOWN         = "unknown"


class TurnFunction(str, Enum):
    WARNING     = "warning"
    PROPOSAL    = "proposal"
    PRESSURE    = "pressure"
    HESITATION  = "hesitation"
    REFUSAL     = "refusal"
    CONDITION   = "condition"
    CONSEQUENCE = "consequence"
    DECISION    = "decision"
    OBSERVATION = "observation"
    TRANSITION  = "transition"
    REVELATION  = "revelation"
    UNKNOWN     = "unknown"


@dataclass
class SceneConstraint:
    """Optional per-phase/scene constraints. All fields optional for backward compat."""
    phase_id: str = ""
    location: str = ""
    time_phase: str = ""
    allowed_characters: list[str] = field(default_factory=list)
    forbidden_characters: list[str] = field(default_factory=list)
    required_characters: list[str] = field(default_factory=list)
    forbidden_pairs: list[list[str]] = field(default_factory=list)  # pairs that cannot co-exist
    allowed_clues: list[str] = field(default_factory=list)
    blocked_clues: list[str] = field(default_factory=list)
    entry_conditions: list[str] = field(default_factory=list)    # free-text conditions
    exit_conditions: list[str] = field(default_factory=list)
    transition_conditions: list[str] = field(default_factory=list)
    allowed_interaction_types: list[str] = field(default_factory=list)
    forbidden_interaction_types: list[str] = field(default_factory=list)
    max_turns: int = 0   # 0 = no limit
    required_state_changes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneConstraint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_empty(self) -> bool:
        return not any([
            self.allowed_characters, self.forbidden_characters, self.required_characters,
            self.forbidden_pairs, self.allowed_clues, self.blocked_clues,
            self.entry_conditions, self.allowed_interaction_types,
            self.forbidden_interaction_types, self.max_turns,
        ])


@dataclass
class SceneState:
    """Tracks mutable state during simulation of one scene/phase."""
    phase_id: str = ""
    location: str = ""
    time_phase: str = ""
    active_cast: list[str] = field(default_factory=list)
    turn_count: int = 0
    completed_clues: list[str] = field(default_factory=list)
    pending_clues: list[str] = field(default_factory=list)
    concrete_changes: list[str] = field(default_factory=list)   # log of real changes this scene
    recent_turn_functions: list[TurnFunction] = field(default_factory=list)
    transition_occurred: bool = False

    def record_turn_function(self, fn: TurnFunction, window: int = 6) -> None:
        self.recent_turn_functions.append(fn)
        if len(self.recent_turn_functions) > window:
            self.recent_turn_functions = self.recent_turn_functions[-window:]

    def loop_detected(self, threshold: int = 3) -> bool:
        """True if the same turn function repeated >= threshold times in recent window."""
        if len(self.recent_turn_functions) < threshold:
            return False
        last = self.recent_turn_functions[-threshold:]
        return len(set(last)) == 1 and last[0] not in (
            TurnFunction.TRANSITION, TurnFunction.REVELATION, TurnFunction.DECISION
        )

    def turns_without_concrete_change(self) -> int:
        return self.turn_count - len(self.concrete_changes)

    def record_concrete_change(self, desc: str) -> None:
        self.concrete_changes.append(desc)


@dataclass
class VoiceProfile:
    """Structured character voice data derived from characters.yaml speech_profile."""
    character_id: str = ""
    tone: str = ""
    cadence: str = ""
    formality: str = ""
    lexicon: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    signature_tics: list[str] = field(default_factory=list)
    emotional_restraint: str = "medium"   # low / medium / high
    technical_density: str = "medium"
    directness: str = "medium"

    @classmethod
    def from_character_config(cls, char_id: str, char_dict: dict) -> "VoiceProfile":
        sp = char_dict.get("speech_profile", {}) or {}
        return cls(
            character_id=char_id,
            tone=str(sp.get("tone", "") or ""),
            cadence=str(sp.get("cadence", "") or ""),
            formality=str(sp.get("formality", "") or ""),
            lexicon=list(sp.get("lexicon", []) or []),
            avoid=list(sp.get("avoid", []) or []),
            signature_tics=list(sp.get("signature_tics", []) or []),
        )

    def to_prompt_block(self) -> str:
        """Compact prompt-injection string for LLM use."""
        parts = []
        if self.tone:
            parts.append(f"Tone: {self.tone}")
        if self.cadence:
            parts.append(f"Cadence: {self.cadence}")
        if self.formality:
            parts.append(f"Formality: {self.formality}")
        if self.lexicon:
            parts.append(f"Preferred words: {', '.join(self.lexicon[:6])}")
        if self.signature_tics:
            parts.append(f"Signature phrases: {', '.join(self.signature_tics[:4])}")
        if self.avoid:
            parts.append(f"Avoid: {', '.join(self.avoid[:4])}")
        return "\n".join(parts)

    def is_empty(self) -> bool:
        return not (self.tone or self.cadence or self.lexicon)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_cast_legality(
    candidates: list[str],
    constraint: "SceneConstraint | None",
    *,
    log_prefix: str = "",
) -> tuple[list[str], list[str]]:
    """
    Returns (allowed, blocked) lists.
    If constraint is None or empty, all candidates pass.
    """
    if constraint is None or constraint.is_empty():
        return list(candidates), []
    allowed, blocked = [], []
    for c in candidates:
        if constraint.forbidden_characters and c in constraint.forbidden_characters:
            if log_prefix:
                import logging
                logging.getLogger("scene_constraints").info(
                    "%s cast BLOCKED %s (forbidden)", log_prefix, c
                )
            blocked.append(c)
            continue
        if constraint.allowed_characters and c not in constraint.allowed_characters:
            if log_prefix:
                import logging
                logging.getLogger("scene_constraints").info(
                    "%s cast BLOCKED %s (not in allowed list)", log_prefix, c
                )
            blocked.append(c)
            continue
        allowed.append(c)
    # check forbidden pairs
    for pair in constraint.forbidden_pairs:
        if len(pair) >= 2 and pair[0] in allowed and pair[1] in allowed:
            # remove the second one
            if log_prefix:
                import logging
                logging.getLogger("scene_constraints").info(
                    "%s cast BLOCKED %s (forbidden pair with %s)", log_prefix, pair[1], pair[0]
                )
            allowed.remove(pair[1])
            blocked.append(pair[1])
    return allowed, blocked


def check_clue_legality(
    clue_id: str,
    constraint: "SceneConstraint | None",
    *,
    log_prefix: str = "",
) -> bool:
    """Returns True if clue is allowed to trigger in current scene/phase."""
    if constraint is None or constraint.is_empty():
        return True
    if constraint.blocked_clues and clue_id in constraint.blocked_clues:
        if log_prefix:
            import logging
            logging.getLogger("scene_constraints").info(
                "%s clue BLOCKED %s (blocked list)", log_prefix, clue_id
            )
        return False
    if constraint.allowed_clues and clue_id not in constraint.allowed_clues:
        if log_prefix:
            import logging
            logging.getLogger("scene_constraints").info(
                "%s clue BLOCKED %s (not in allowed list)", log_prefix, clue_id
            )
        return False
    return True


def load_scene_constraints_from_episode(episode_config: dict) -> list[SceneConstraint]:
    """Load optional scene_constraints list from episode config. Returns [] if absent."""
    raw = episode_config.get("scene_constraints") or []
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, dict):
            result.append(SceneConstraint.from_dict(item))
    return result


def classify_turn_function(content: str) -> TurnFunction:
    """
    Lightweight heuristic to classify a turn's dramatic function.
    Used for repetition detection.
    """
    c = content.lower()
    warning_kw    = ["경고", "위험", "조심", "주의", "위험해", "조심해", "위험하다", "위험합니다", "무너", "망할", "잃게"]
    pressure_kw   = ["협력해야", "선택이 아닌", "필수", "협조해야", "강제", "반드시", "어쩔 수 없", "불가피"]
    proposal_kw   = ["제안", "제공", "기회", "지원", "도울 수", "도와드릴", "협력하겠", "가능합니다"]
    refusal_kw    = ["거절", "받아들일 수 없", "동의하지 않", "아니요", "불가", "거부"]
    decision_kw   = ["결정했", "선택했", "동의합니다", "거절합니다", "결심", "알겠습니다", "하겠습니다"]
    condition_kw  = ["조건", "전제", "기준", "보장", "명확히", "분명히", "확인해야", "전제"]
    consequence_kw = ["결과적으로", "그 결과", "따라서", "때문에", "그러므로", "이로 인해"]
    revelation_kw  = ["사실은", "알아냈", "발견했", "밝혀졌", "알게 됐", "실제로는", "드러났"]
    transition_kw  = ["자리를 떠났", "이동했", "옮겼", "문을 열", "다음 방", "복도로", "나갔다", "들어왔다"]
    hesitation_kw  = ["망설", "주저", "고민", "갈등", "불안", "두려움", "걱정", "머뭇"]

    for kw in decision_kw:
        if kw in c:
            return TurnFunction.DECISION
    for kw in revelation_kw:
        if kw in c:
            return TurnFunction.REVELATION
    for kw in transition_kw:
        if kw in c:
            return TurnFunction.TRANSITION
    for kw in consequence_kw:
        if kw in c:
            return TurnFunction.CONSEQUENCE
    for kw in refusal_kw:
        if kw in c:
            return TurnFunction.REFUSAL
    for kw in condition_kw:
        if kw in c:
            return TurnFunction.CONDITION
    for kw in pressure_kw:
        if kw in c:
            return TurnFunction.PRESSURE
    for kw in warning_kw:
        if kw in c:
            return TurnFunction.WARNING
    for kw in proposal_kw:
        if kw in c:
            return TurnFunction.PROPOSAL
    for kw in hesitation_kw:
        if kw in c:
            return TurnFunction.HESITATION
    return TurnFunction.UNKNOWN


# ---------------------------------------------------------------------------
# New dataclasses: TransitionRule, EntryRuleResult, RelationshipProfile
# ---------------------------------------------------------------------------

@dataclass
class TransitionRule:
    """Defines valid conditions for a scene/phase transition."""
    from_phase: str = ""          # phase_id this rule applies to (empty = any)
    to_phase: str = ""            # target phase (empty = any exit)
    trigger_types: list[str] = field(default_factory=list)   # e.g. ["location_move", "director_event", "exit_action"]
    required_clues_completed: list[str] = field(default_factory=list)
    required_concrete_changes: int = 0   # min number of concrete changes before transition
    soft: bool = True             # if True: log warning but don't hard-block


@dataclass
class EntryRuleResult:
    """Result of evaluating whether a character may enter the current scene."""
    allowed: bool = True
    character_id: str = ""
    reason: str = ""              # human-readable explanation
    should_log: bool = True


@dataclass
class RelationshipProfile:
    """Compact structured relationship context for a character pair."""
    agent_id: str = ""
    other_id: str = ""
    relation_type: str = ""          # e.g. "mentor", "rival", "colleague", "stranger", "authority"
    familiarity_level: str = "low"   # low / medium / high
    hierarchy: str = "peer"          # agent_above / agent_below / peer
    trust_level: str = "neutral"     # low / neutral / high
    unresolved_tension: str = ""     # free text
    formality_expectation: str = "formal"  # casual / semi-formal / formal
    shared_history_density: str = "none"   # none / sparse / rich
    taboo_topics: list[str] = field(default_factory=list)
    typical_interaction_mode: str = ""     # free text
    protective_stance: bool = False
    adversarial_stance: bool = False

    @classmethod
    def from_character_pair(cls, agent_id: str, other_id: str,
                             agent_dict: dict, other_dict: dict,
                             relationship_matrix: dict) -> "RelationshipProfile":
        """
        Build a RelationshipProfile from character config dicts and relationship_matrix.
        Extracts what's available; falls back to defaults gracefully.
        """
        rel_value = float(relationship_matrix.get(other_id, 0.0))
        # Infer familiarity from relationship_matrix magnitude
        magnitude = abs(rel_value)
        if magnitude >= 0.6:
            familiarity = "high"
        elif magnitude >= 0.3:
            familiarity = "medium"
        else:
            familiarity = "low"

        # Trust from sign
        if rel_value >= 0.4:
            trust = "high"
        elif rel_value <= -0.3:
            trust = "low"
        else:
            trust = "neutral"

        # Try to extract relation_type from bio / invariants / goals text
        combined_text = " ".join([
            str(agent_dict.get("bio", "") or ""),
            " ".join(agent_dict.get("invariants", []) or []),
            str(other_dict.get("bio", "") or ""),
            " ".join(other_dict.get("invariants", []) or []),
        ]).lower()

        other_name_lower = str(other_dict.get("name", "") or "").lower()
        agent_name_lower = str(agent_dict.get("name", "") or "").lower()

        relation_type = "colleague"
        if any(w in combined_text for w in ["mentor", "지도", "교수", "professor", "advisor"]):
            relation_type = "mentor"
        elif any(w in combined_text for w in ["rival", "경쟁", "competitor"]):
            relation_type = "rival"
        elif any(w in combined_text for w in ["supervisor", "상관", "manager", "director"]):
            relation_type = "authority"
        elif familiarity == "low":
            relation_type = "acquaintance"

        # Hierarchy: check role fields
        agent_role = str(agent_dict.get("role", "") or "").lower()
        other_role = str(other_dict.get("role", "") or "").lower()
        if other_role in ("protagonist",) and agent_role not in ("protagonist",):
            hierarchy = "agent_above"
        else:
            hierarchy = "peer"

        formality = "formal"
        if familiarity == "high" and trust == "high":
            formality = "semi-formal"

        return cls(
            agent_id=agent_id,
            other_id=other_id,
            relation_type=relation_type,
            familiarity_level=familiarity,
            hierarchy=hierarchy,
            trust_level=trust,
            formality_expectation=formality,
            shared_history_density=familiarity,  # reuse as proxy
            adversarial_stance=(rel_value <= -0.4),
            protective_stance=(rel_value >= 0.6 and relation_type == "mentor"),
        )

    def to_prompt_block(self) -> str:
        """Compact relationship context for LLM injection."""
        lines = [f"Relationship to {self.other_id}: {self.relation_type}"]
        lines.append(f"  Familiarity: {self.familiarity_level} | Trust: {self.trust_level} | Hierarchy: {self.hierarchy}")
        lines.append(f"  Formality expected: {self.formality_expectation}")
        if self.shared_history_density not in ("none", "low"):
            lines.append(f"  Shared history: {self.shared_history_density} — skip re-explaining common context")
        if self.unresolved_tension:
            lines.append(f"  Unresolved tension: {self.unresolved_tension}")
        if self.typical_interaction_mode:
            lines.append(f"  Typical mode: {self.typical_interaction_mode}")
        if self.adversarial_stance:
            lines.append("  Stance: adversarial — caution, guarded phrasing")
        if self.protective_stance:
            lines.append("  Stance: protective — caring restraint, avoids direct pressure")
        # Behavioral guidance
        if self.familiarity_level == "high":
            lines.append("  → Speak with compressed references, skip pleasantries, assume shared assumptions")
        elif self.familiarity_level == "low":
            lines.append("  → More measured phrasing, less assumption, clearer signaling of intent")
        return "\n".join(lines)

    def is_meaningful(self) -> bool:
        return self.familiarity_level != "low" or self.relation_type not in ("colleague", "acquaintance", "stranger")


# ---------------------------------------------------------------------------
# New helper functions: evaluate_entry_rule, check_transition_legality
# ---------------------------------------------------------------------------

def evaluate_entry_rule(
    character_id: str,
    scene_state: "SceneState | None",
    constraint: "SceneConstraint | None",
    scene_turn: int,
    total_turns: int,
) -> EntryRuleResult:
    """
    Evaluate whether a character may enter based on current scene state and constraints.
    Returns EntryRuleResult. Soft by default — does not hard-block unless constraint says so.
    """
    if constraint is None or constraint.is_empty():
        return EntryRuleResult(allowed=True, character_id=character_id, reason="no constraint", should_log=False)

    if constraint.forbidden_characters and character_id in constraint.forbidden_characters:
        return EntryRuleResult(allowed=False, character_id=character_id,
                               reason=f"{character_id} is forbidden in this phase")

    if constraint.allowed_characters and character_id not in constraint.allowed_characters:
        # Allow if past 70% of scene — late entry is more forgivable
        progress = (scene_turn / max(1, total_turns))
        if progress < 0.7:
            return EntryRuleResult(allowed=False, character_id=character_id,
                                   reason=f"{character_id} not in allowed list at {progress:.0%} scene progress")

    return EntryRuleResult(allowed=True, character_id=character_id, reason="passed", should_log=False)


def check_transition_legality(
    scene_state: "SceneState | None",
    transition_rules: list["TransitionRule"],
    proposed_new_location: str = "",
    proposed_new_phase: str = "",
) -> tuple[bool, str]:
    """
    Returns (is_legal, reason).
    If no rules defined, always legal (backward compat).
    A transition is legal if:
    - No matching rule exists (permissive default), OR
    - A matching rule's conditions are satisfied
    """
    if not transition_rules or scene_state is None:
        return True, "no transition rules defined"

    relevant = [r for r in transition_rules
                if not r.from_phase or r.from_phase == scene_state.phase_id]

    if not relevant:
        return True, "no rule for current phase"

    for rule in relevant:
        # Check concrete change requirement
        if rule.required_concrete_changes > 0:
            n_changes = len(scene_state.concrete_changes)
            if n_changes < rule.required_concrete_changes:
                msg = (f"transition requires {rule.required_concrete_changes} concrete changes, "
                       f"only {n_changes} recorded")
                if rule.soft:
                    import logging
                    logging.getLogger("scene_constraints").warning("[TRANSITION] soft-blocked: %s", msg)
                    return True, f"soft-allowed: {msg}"
                return False, msg

        # Check required clues
        for clue_id in rule.required_clues_completed:
            if clue_id not in scene_state.completed_clues:
                msg = f"transition requires clue {clue_id} to be completed first"
                if rule.soft:
                    import logging
                    logging.getLogger("scene_constraints").warning("[TRANSITION] soft-blocked: %s", msg)
                    return True, f"soft-allowed: {msg}"
                return False, msg

    return True, "transition allowed"
