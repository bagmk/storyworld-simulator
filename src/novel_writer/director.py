"""
Director AI for the AI Story Simulation Engine.

Responsibilities:
  1. Invariant enforcement  – blocks/rewrites agent actions that violate character rules
  2. Knowledge leakage check – prevents agents from revealing facts they shouldn't know
  3. Storyline alignment     – keeps turns aligned with long-arc milestones
  4. Clue injection          – fires trigger events when required clues haven't surfaced
  5. Resolution validation   – verifies episode objectives are met before ending
  6. Pacing guidance         – tracks turn count vs. target length

The Director uses the premium LLM model for its evaluations.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Optional

from .models import Agent, WorldState, ClueManager
from .llm_client import LLMClient
from .reader_profile import build_reader_profile
from . import database as db

logger = logging.getLogger(__name__)


class DirectorAI:
    """
    Enforces story constraints while preserving agent autonomy.

    Parameters
    ----------
    episode_config : dict
        Loaded from episode YAML.
    world_facts : dict
        Loaded from world_facts YAML (contains hidden / discoverable / public).
    clue_manager : ClueManager
        Shared ClueManager instance.
    storyline : dict | None
        Optional long-arc storyline YAML structure.
    llm : LLMClient
        Used with use_premium=True for Director evaluations.
    debug_log : list
        Accumulated log of all Director interventions (written to debug file).
    """

    def __init__(
        self,
        episode_config: dict,
        world_facts: dict,
        clue_manager: ClueManager,
        llm: LLMClient,
        storyline: Optional[dict] = None,
        reader_feedback: Optional[dict] = None,
        guardian_briefing: Optional[str] = None,
    ) -> None:
        self.episode_config = episode_config
        self.world_facts = world_facts
        self.clue_manager = clue_manager
        self.storyline = storyline or {}
        self.reader_profile = build_reader_profile(reader_feedback)
        self.reader_feedback = self.reader_profile.as_dict()
        self.guardian_briefing = (guardian_briefing or "").strip()
        self.llm = llm
        self.debug_log: list[dict] = []

        # Flatten constraints for quick access
        self.character_invariants: dict[str, list[str]] = {}
        invariant_cfg = episode_config.get("character_invariants")
        if invariant_cfg is None:
            invariant_cfg = episode_config.get("characters", [])
        for char in invariant_cfg:
            cid = char.get("id", "")
            self.character_invariants[cid] = char.get("invariants", [])

        raw_must_resolve = episode_config.get("must_resolve")
        if raw_must_resolve is None:
            raw_must_resolve = episode_config.get("resolved", [])
        if not isinstance(raw_must_resolve, list):
            raw_must_resolve = []
        self.must_resolve: list[str] = [
            str(item).strip() for item in raw_must_resolve if str(item).strip()
        ]
        self.pacing: dict             = episode_config.get("pacing_guidelines", {})
        self.max_turns: int           = episode_config.get("max_turns", 80)
        self.min_turns_before_completion: int = int(
            episode_config.get(
                "min_turns_before_completion",
                max(6, int(self.max_turns * 0.25)),
            )
        )
        self.completion_check_interval: int = max(
            1, int(episode_config.get("completion_check_interval", 2))
        )
        self.completion_confidence_threshold: float = float(
            episode_config.get("completion_confidence_threshold", 0.7)
        )
        self._last_completion_check_turn: int = 0
        self._last_completion_result: Optional[tuple[bool, list[str]]] = None
        if self._reader_reports_stalled_progression():
            self.min_turns_before_completion = min(
                self.min_turns_before_completion,
                max(4, int(self.max_turns * 0.2)),
            )
            self.completion_check_interval = 1

        # Hidden facts as a set of strings for leak detection
        self._hidden_fact_texts: list[str] = [
            f.get("content", str(f)) if isinstance(f, dict) else str(f)
            for f in world_facts.get("hidden", [])
        ]
        self.storyline_context = self._build_storyline_context()

    # ------------------------------------------------------------------ #
    # 1. Invariant Check
    # ------------------------------------------------------------------ #

    def check_invariant(self, agent: Agent, proposed_action: str) -> tuple[bool, str]:
        """
        Verify the proposed action doesn't violate any character invariant.

        Returns
        -------
        (approved: bool, correction_context: str)
        If approved is False, correction_context is a hint to re-generate.
        """
        invariants = agent.invariants
        if not invariants:
            return True, ""

        inv_text = "\n".join(f"- {i}" for i in invariants)
        prompt = (
            f"You are a story continuity checker.\n\n"
            f"Character: {agent.name} ({agent.role})\n"
            f"Character invariants (rules that must NEVER be broken):\n{inv_text}\n\n"
            f"Proposed action/dialogue:\n\"\"\"\n{proposed_action}\n\"\"\"\n\n"
            f"Does this action violate ANY invariant?\n"
            f"Reply with JSON only, no markdown:\n"
            f"{{\"violation\": true/false, \"violated_invariant\": \"...\", \"reason\": \"...\"}}"
        )

        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_invariant_check",
        )
        parsed = self._parse_json(result)

        if parsed.get("violation"):
            msg = (
                f"Invariant violation detected for {agent.name}: "
                f"{parsed.get('violated_invariant')} — {parsed.get('reason')}"
            )
            self._log("invariant_violation", agent.id, msg, {"action": proposed_action[:200]})
            correction = (
                f"IMPORTANT: Your previous response violated a core character rule: "
                f"{parsed.get('violated_invariant')}. "
                f"You must stay true to: {parsed.get('reason')}. "
                f"Please respond again without violating this rule."
            )
            return False, correction

        return True, ""

    # ------------------------------------------------------------------ #
    # 2. Knowledge Leak Check
    # ------------------------------------------------------------------ #

    def check_knowledge_leak(
        self, agent: Agent, proposed_action: str, world: WorldState
    ) -> tuple[bool, str]:
        """
        Prevent agent from revealing hidden facts they couldn't know.

        Returns (approved, correction_context).
        """
        if not self._hidden_fact_texts:
            return True, ""

        # Build observable context — things agents CAN legitimately reference
        observable_text = self._build_observable_context(agent, world)

        # Improved keyword scan: use only meaningful keywords (len >= 3),
        # skip words that also appear in observable context
        combined = proposed_action.lower()
        observable_lower = observable_text.lower()

        potentially_leaking = False
        for fact in self._hidden_fact_texts:
            fact_words = [w for w in fact.lower().split() if len(w) >= 3]
            # Use first 5 meaningful words as probe keywords
            probe_words = fact_words[:5]
            for word in probe_words:
                # Skip if this word appears in observable context (not a leak)
                if word in observable_lower:
                    continue
                if word in combined:
                    potentially_leaking = True
                    break
            if potentially_leaking:
                break

        if not potentially_leaking:
            return True, ""

        # Full LLM check only when suspicious
        hidden_summary = "\n".join(f"- {f}" for f in self._hidden_fact_texts)
        known_clues_text = "\n".join(f"- {c}" for c in agent.memory.known_clues) or "(none)"

        prompt = (
            f"Check if this character is revealing information they shouldn't know.\n\n"
            f"Character: {agent.name}\n"
            f"Clues this character legitimately knows:\n{known_clues_text}\n\n"
            f"HIDDEN facts this character must NOT know or reveal:\n{hidden_summary}\n\n"
            f"OBSERVABLE context (things the character CAN see/reference — NOT leaks):\n"
            f"{self._truncate(observable_text, 600)}\n\n"
            f"Proposed action/dialogue:\n\"\"\"\n{proposed_action}\n\"\"\"\n\n"
            f"Does this action reveal any hidden fact that goes BEYOND what is observable?\n"
            f"If the character is merely reacting to something visible in the scene or "
            f"referencing recently spoken dialogue, that is NOT a leak.\n"
            f"Reply JSON only: {{\"leaks\": true/false, \"leaked_fact\": \"...\", \"explanation\": \"...\"}}"
        )

        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_knowledge_check",
        )
        parsed = self._parse_json(result)

        if parsed.get("leaks"):
            msg = (
                f"Knowledge leak by {agent.name}: revealed '{parsed.get('leaked_fact')}'"
            )
            self._log("knowledge_leak", agent.id, msg, {"action": proposed_action[:200]})
            correction = (
                f"IMPORTANT: Your response revealed information your character "
                f"({agent.name}) does not know. Remove any reference to "
                f"'{parsed.get('leaked_fact')}' and respond again based only on "
                f"what your character has actually observed or been told."
            )
            return False, correction

        return True, ""

    def _build_observable_context(self, agent: Agent, world: WorldState) -> str:
        """Build text of everything an agent can legitimately observe/reference."""
        parts = []
        # Current scene description
        if world.current_scene:
            parts.append(f"[Scene] {world.current_scene[-400:]}")
        # Visible context (last event, etc.)
        for key, val in world.visible_context.items():
            parts.append(f"[{key}] {str(val)[:200]}")
        # Recent interactions the agent witnessed
        recent = agent.memory.recent_interactions(8)
        if recent:
            recent_text = "\n".join(
                f"  {ix.get('speaker_name', '?')}: {str(ix.get('content', ''))[:150]}"
                for ix in recent
            )
            parts.append(f"[Recent dialogue]\n{recent_text}")
        # Episode summary (agents know where they are)
        ep_summary = self.episode_config.get("summary", "")
        if ep_summary:
            parts.append(f"[Episode setting] {ep_summary[:200]}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # 3. Storyline Alignment Check
    # ------------------------------------------------------------------ #

    def check_storyline_alignment(
        self,
        agent: Agent,
        proposed_action: str,
        world: WorldState,
        agents: list[Agent],
        recent_interactions: Optional[list[dict]] = None,
    ) -> tuple[bool, str]:
        """
        Keep the current episode aligned with long-arc storyline milestones.

        Returns (approved, correction_context).
        """
        current = self.storyline_context.get("current")
        if not current:
            return True, ""

        active_ids = {aid for aid in world.active_agents if isinstance(aid, str)}
        unplanned_entries = self._detect_unplanned_character_entries(
            proposed_action, active_ids, agents
        )
        if unplanned_entries:
            active_names = ", ".join(
                a.name for a in agents if a.id in active_ids
            ) or "(none)"
            msg = (
                f"Unplanned character entry by {agent.name}: "
                f"{', '.join(unplanned_entries)}"
            )
            self._log(
                "storyline_cast_drift",
                agent.id,
                msg,
                {
                    "active_cast": sorted(active_ids),
                    "unplanned_entries": unplanned_entries,
                    "action": proposed_action[:240],
                },
            )
            correction = (
                f"IMPORTANT: You introduced off-scene character(s): "
                f"{', '.join(unplanned_entries)}. "
                f"Only these characters are currently in-scene: {active_names}. "
                f"Do not introduce new entrants unless the Director explicitly injects them."
            )
            return False, correction

        first_meeting_drift = self._detect_first_meeting_drift_for_known_relation(
            agent=agent,
            proposed_action=proposed_action,
            active_ids=active_ids,
            agents=agents,
        )
        if first_meeting_drift:
            names = ", ".join(first_meeting_drift)
            msg = (
                f"First-meeting behavior drift by {agent.name} with known relation(s): {names}"
            )
            self._log(
                "relationship_drift",
                agent.id,
                msg,
                {
                    "known_relations": first_meeting_drift,
                    "action": proposed_action[:240],
                },
            )
            correction = (
                f"IMPORTANT: You treated {names} like a first-time meeting, but this conflicts "
                f"with your established relationship/background. "
                f"Regenerate this turn as an already-familiar interaction. "
                f"Do not use first-meeting rituals (formal self-intro, 'nice to meet you', "
                f"or unnecessary business-card exchange)."
            )
            return False, correction

        storyline_guidance = self.get_storyline_guidance() or ""
        active_names = ", ".join(
            a.name for a in agents if a.id in active_ids
        ) or "(none)"
        recent = recent_interactions or []
        recent_text = "\n".join(
            f"- {i.get('speaker_name', '?')}: {self._truncate(str(i.get('content', '')), 180)}"
            for i in recent[-4:]
        ) or "(none)"

        prompt = (
            f"You are a strict storyline continuity checker.\n\n"
            f"Storyline guidance:\n{storyline_guidance}\n\n"
            f"Episode summary:\n{self.episode_config.get('summary', '')}\n\n"
            f"Current location: {self.episode_config.get('location', world.location)}\n"
            f"Active in-scene cast: {active_names}\n\n"
            f"Recent interactions:\n{recent_text}\n\n"
            f"Candidate action by {agent.name}:\n\"\"\"\n{proposed_action}\n\"\"\"\n\n"
            f"Does this action stay aligned with the current milestone without derailing into unrelated "
            f"subplots or prematurely jumping to future milestone outcomes?\n"
            f"Reply JSON only:\n"
            f"{{\"off_track\": true/false, \"severity\": \"minor|major\", "
            f"\"reason\": \"...\", \"guidance\": \"...\"}}"
        )

        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_storyline_alignment",
            max_tokens=260,
        )
        parsed = self._parse_json(result)

        if parsed.get("off_track"):
            reason = parsed.get("reason", "action drifted from the episode's storyline direction")
            guidance = parsed.get(
                "guidance",
                "Keep the turn grounded in the current episode objective and active cast.",
            )
            self._log(
                "storyline_drift",
                agent.id,
                f"Storyline drift detected for {agent.name}: {reason}",
                {
                    "severity": parsed.get("severity", "unknown"),
                    "action": proposed_action[:240],
                },
            )
            correction = (
                f"IMPORTANT: Your previous turn drifted from the current storyline. "
                f"Reason: {reason}. "
                f"Please regenerate and follow this direction: {guidance}"
            )
            return False, correction

        return True, ""

    # ------------------------------------------------------------------ #
    # 4. Clue Injection
    # ------------------------------------------------------------------ #

    def should_inject_clue(
        self, turn: int, world: WorldState
    ) -> Optional[dict]:
        """
        Return a clue injection event if a required clue needs to be forced.
        Returns None if no injection needed.
        """
        turn_progress = turn / max(self.max_turns, 1)
        clue = self.clue_manager.needs_injection(turn_progress)
        if not clue:
            return None

        clue_id      = clue.get("id", "unknown")
        clue_content = clue.get("content", clue.get("description", ""))
        trigger_desc = clue.get("trigger", "environmental cue")
        inject_method = clue.get("inject_method", "environmental_cue")

        event_text = self._generate_injection_event(
            clue_content, trigger_desc, inject_method, world
        )

        injection = {
            "clue_id": clue_id,
            "clue_content": clue_content,
            "event_text": event_text,
            "inject_method": inject_method,
        }
        self._log("clue_injection", "director", f"Injecting clue: {clue_id}", injection)
        return injection

    def _generate_injection_event(
        self, clue_content: str, trigger: str,
        method: str, world: WorldState
    ) -> str:
        """Generate a natural in-world event that surfaces a clue."""
        builders = {
            "document_artifact": self._generate_document_artifact_event,
            "system_alert": self._generate_system_alert_event,
            "npc_offer": self._generate_npc_offer_event,
            "npc_question": self._generate_npc_question_event,
            "environmental_cue": self._generate_environmental_cue_event,
        }
        builder = builders.get(str(method or "").strip(), self._generate_generic_injection_event)
        return builder(clue_content, trigger, world)

    def _generate_document_artifact_event(
        self,
        clue_content: str,
        trigger: str,
        world: WorldState,
    ) -> str:
        prompt = (
            "You are staging a clue as a document, memo, notebook, badge, or handwritten artifact.\n\n"
            f"{self._scene_event_context(world)}"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n\n"
            f"{self._common_injection_guidance()}"
            "Write 1-3 Korean sentences of scene narration.\n"
            "Rules:\n"
            "- Sentence 1 must pin down where the protagonist is and exactly where the artifact appears.\n"
            "- Sentence 2 should show the word, symbol, or memo fragment that catches attention.\n"
            "- Keep movement linear: notice -> approach/take -> immediate consequence.\n"
            "- Do not turn it into a general explanation paragraph.\n"
            "- If this reads like a transition beat, make the spatial handoff explicit.\n"
        )
        return self._render_injection_event(prompt, purpose="director_clue_document_artifact")

    def _generate_system_alert_event(
        self,
        clue_content: str,
        trigger: str,
        world: WorldState,
    ) -> str:
        prompt = (
            "You are staging a clue as a warning sound, alert tone, device message, or access denial.\n\n"
            f"{self._scene_event_context(world)}"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n\n"
            f"{self._common_injection_guidance()}"
            "Write 1-3 Korean sentences of scene narration.\n"
            "Rules:\n"
            "- State where the sound or alert comes from and who is closest to it.\n"
            "- Keep the order concrete: source -> reaction -> next pressure.\n"
            "- If there is a beep or warning text, tie it to a door, screen, badge reader, or phone.\n"
            "- Avoid vague mood language without location.\n"
        )
        return self._render_injection_event(prompt, purpose="director_clue_system_alert")

    def _generate_npc_offer_event(
        self,
        clue_content: str,
        trigger: str,
        world: WorldState,
    ) -> str:
        prompt = (
            "You are staging a named entrance or offer beat by a new actor.\n\n"
            f"{self._scene_event_context(world)}"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n\n"
            f"{self._common_injection_guidance()}"
            "Write 1-3 Korean sentences of scene narration.\n"
            "Rules:\n"
            "- Sentence 1 must say from which doorway, corridor, row, or edge of the room the person appears.\n"
            "- Sentence 2 must say where the protagonist is when the newcomer stops or speaks.\n"
            "- Keep the entrance, stop, and pressure shift in a clear sequence.\n"
            "- Name any named actor clearly on first appearance; do not leave them as only a generic descriptor.\n"
            "- If this entrance resolves an earlier unnamed observer, make that link explicit once.\n"
        )
        return self._render_injection_event(prompt, purpose="director_clue_npc_offer")

    def _generate_npc_question_event(
        self,
        clue_content: str,
        trigger: str,
        world: WorldState,
    ) -> str:
        prompt = (
            "You are staging a clue through a pointed NPC question or short exchange.\n\n"
            f"{self._scene_event_context(world)}"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n\n"
            f"{self._common_injection_guidance()}"
            "Write 1-3 Korean sentences of scene narration.\n"
            "Rules:\n"
            "- Identify who steps closer, turns, or interrupts, and where both sides are standing.\n"
            "- Make the question feel like a pressure turn, not an essay.\n"
            "- Keep one concrete motion cue before or after the question.\n"
        )
        return self._render_injection_event(prompt, purpose="director_clue_npc_question")

    def _generate_environmental_cue_event(
        self,
        clue_content: str,
        trigger: str,
        world: WorldState,
    ) -> str:
        prompt = (
            "You are staging a clue as an environmental cue the protagonist notices.\n\n"
            f"{self._scene_event_context(world)}"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n\n"
            f"{self._common_injection_guidance()}"
            "Write 1-3 Korean sentences of scene narration.\n"
            "Rules:\n"
            "- Anchor the cue to one visible place in the room or corridor.\n"
            "- Name who notices it and from what position.\n"
            "- Let the cue change attention or movement immediately.\n"
        )
        return self._render_injection_event(prompt, purpose="director_clue_environmental_cue")

    def _generate_generic_injection_event(
        self,
        clue_content: str,
        trigger: str,
        world: WorldState,
    ) -> str:
        prompt = (
            "You are a story director. A required story clue hasn't surfaced naturally.\n\n"
            f"{self._scene_event_context(world)}"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n\n"
            f"{self._common_injection_guidance()}"
            "Write a brief (1-3 sentence) Korean in-world event or observation that naturally introduces this clue.\n"
            "Write it as scene narration, not as dialogue.\n"
        )
        return self._render_injection_event(prompt, purpose="director_clue_injection")

    def _scene_event_context(self, world: WorldState) -> str:
        scene_text = self._truncate(str(world.current_scene or ""), 700)
        location = str(world.location or self.episode_config.get("location", "")).strip()
        last_event = ""
        if isinstance(world.visible_context, dict):
            last_event = self._truncate(str(world.visible_context.get("last_event", "") or ""), 220)
        return (
            f"Current scene: {scene_text}\n"
            f"Location: {location}\n"
            f"Most recent visible event: {last_event or '(none)'}\n\n"
        )

    def _common_injection_guidance(self) -> str:
        guidance = (
            "Shared rules:\n"
            "- Keep the prose concrete and brief.\n"
            "- Make spatial continuity explicit: where the protagonist is, where the cue starts, and how attention moves.\n"
            "- Prefer one clear event over layered explanation.\n"
        )
        if self.reader_profile.prefers_technical_term_restraint():
            guidance += (
                "- Avoid checklist-like technical listing.\n"
                "- Keep at most one technical term in this event unless absolutely necessary.\n"
                "- Prioritize a concrete sensory/action cue over repeated metrics.\n"
                "- Do not add parenthetical explanation unless the clue would otherwise be unclear.\n"
                "- If a technical term or English keyword appears, follow it immediately with a visible human reaction or consequence.\n"
                "- If the term was already introduced once in-scene, do not explain it again; shift to the protagonist's judgment, emotion, or next choice.\n"
                "- Keep each sentence short and direct; split comma-heavy chains.\n"
            )
        if self.reader_profile.prefers_explicit_transition_cues():
            guidance += (
                "- Treat memo discovery, warning sound, and named arrival as separate beats; do not blur them together.\n"
                "- Name who moved and where they stopped before you describe what it meant.\n"
            )
        return guidance + "\n"

    def _render_injection_event(self, prompt: str, purpose: str) -> str:
        return self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose=purpose,
            use_premium=True,
        )

    def _feedback_mentions(self, *keywords: str) -> bool:
        return self.reader_profile.mentions(*keywords)

    def _reader_reports_stalled_progression(self) -> bool:
        return self.reader_profile.reports_stalled_progression()

    def _reader_wants_repeated_confrontation_merge(self) -> bool:
        return self.reader_profile.wants_repeated_confrontation_merge()

    def _feedback_style_constraints(self) -> dict:
        return self.reader_profile.style_constraints()

    def _feedback_flag_enabled(self, key: str, default: bool = False) -> bool:
        return self.reader_profile.flag_enabled(key, default=default)

    def _feedback_tension_phrase_cap(self, default: int = 2) -> int:
        return self.reader_profile.tension_phrase_cap(default=default)

    def _feedback_static_threat_signal_cap(self, default: int = 2) -> int:
        return self.reader_profile.static_threat_signal_cap(default=default)

    def _build_tension_curve(
        self,
        recent_interactions: list[dict],
        agents: Optional[list[Agent]] = None,
    ) -> dict[str, object]:
        """
        Summarize how pressure changes across the recent beat window.

        This keeps the turn allocator focused on progression rather than
        repeated surface wording.
        """
        recent = [
            row for row in (recent_interactions or [])
            if isinstance(row, dict) and str(row.get("speaker_id", "")).strip() != "director"
        ][-5:]
        curve: list[dict[str, object]] = []
        previous_fp = ""
        for row in recent:
            text = str(row.get("content", "") or "")
            fingerprint = self._content_fingerprint(text)
            pressure = 0
            if self._has_emotional_or_decisive_shift(text):
                pressure += 1
            if self._has_concrete_risk_marker(text):
                pressure += 1
            if self._has_inner_conflict_marker(text):
                pressure += 1
            if self._has_confrontation_resolution_shift(text):
                pressure += 1
            if self._technical_term_signature(text):
                pressure += 1
            if fingerprint and fingerprint == previous_fp:
                pressure += 1
            curve.append(
                {
                    "speaker_id": str(row.get("speaker_id", "")).strip(),
                    "speaker_name": str(row.get("speaker_name", "")).strip(),
                    "pressure": pressure,
                    "fingerprint": fingerprint,
                    "has_shift": bool(
                        self._has_emotional_or_decisive_shift(text)
                        or self._has_confrontation_resolution_shift(text)
                    ),
                }
            )
            previous_fp = fingerprint

        pressure_values = [int(item.get("pressure", 0) or 0) for item in curve]
        rising = any(
            pressure_values[idx] > pressure_values[idx - 1]
            for idx in range(1, len(pressure_values))
        )
        flat = bool(pressure_values) and max(pressure_values) > 0 and len(set(pressure_values)) <= 2 and not rising
        peak = bool(pressure_values) and max(pressure_values) >= 3 and (
            curve[-1].get("has_shift") if curve else False
        )
        decisive_shift = any(bool(item.get("has_shift")) for item in curve[-2:])
        emotional_mix = False
        if agents:
            emotional_mix = self._current_emotional_pressure_flags(
                [str(item.get("speaker_id", "")).strip() for item in recent],
                agents,
            ).get("emotional_conflict", False)

        return {
            "curve": curve,
            "rising": rising,
            "flat": flat,
            "peak": peak,
            "decisive_shift": decisive_shift,
            "emotional_mix": emotional_mix,
        }

    def _enhance_character_interaction(
        self,
        agent: Agent,
        emotion_family: str,
        intensity: float,
        progress_signal: dict[str, bool],
        tension_curve: dict[str, object],
        recent_speakers: list[str],
    ) -> str:
        """
        Shape the next-turn hint around contrast, not repeated paraphrase.
        """
        hints: list[str] = []
        curve_points = tension_curve.get("curve", []) if isinstance(tension_curve, dict) else []
        plateau = bool(tension_curve.get("flat")) if isinstance(tension_curve, dict) else False
        if agent.role == "protagonist":
            hints.append("surface hesitation, then choose")
        elif agent.id in recent_speakers[-2:]:
            hints.append("change leverage instead of restating the same concern")
        if emotion_family == "anxious":
            hints.append("ask for one concrete check or visible proof")
        elif emotion_family == "frustrated":
            hints.append("interrupt, counter, or refuse bluntly")
        elif emotion_family == "confident":
            hints.append("state terms or push the scene forward")
        elif emotion_family == "relieved":
            hints.append("soften the exchange and close the beat")
        if progress_signal.get("repeated_concern") or progress_signal.get("technical_stall"):
            hints.append("translate the repeated point into one concrete consequence or visible cue")
        if progress_signal.get("concrete_risk"):
            hints.append("name the cost as a limit, clause, deadline, or access rule")
        if plateau and curve_points:
            hints.append("break the plateau with a visible change of stance")
        if intensity >= 0.6:
            hints.append("let the emotion color the line")
        return "; ".join(hints) if hints else "react to the visible pressure"

    def _refine_dialogue_structure(
        self,
        progress_signal: dict[str, bool],
        recent_interactions: list[dict],
    ) -> str:
        """
        Turn repeated dialogue into a clearer pressure arc.
        """
        tension_curve = self._build_tension_curve(recent_interactions)
        lines: list[str] = []
        if progress_signal.get("repeated_concern"):
            lines.append(
                "Do not paraphrase the same concern again; let the next line change leverage, cost, choice, or visible action."
            )
        if progress_signal.get("technical_stall"):
            lines.append(
                "If terminology appears again, turn it into one concrete consequence and immediate human response instead of another explanation."
            )
        if progress_signal.get("pressure_peak"):
            lines.append(
                "Hold the exchange open until a concrete consequence, reveal, or exit cue lands."
            )
        if progress_signal.get("concrete_risk"):
            lines.append(
                "Name the cost as a limit, clause, deadline, or access rule rather than repeating an abstract warning."
            )
        if tension_curve.get("flat"):
            lines.append(
                "Let one speaker press, one counter, and one decide; avoid another same-pressure reset."
            )
        if tension_curve.get("rising"):
            lines.append(
                "Escalate step by step so each response sharpens the conflict instead of repeating it or recycling the same feeling."
            )
        return " ".join(lines).strip()

    # ------------------------------------------------------------------ #
    # 5. Resolution Validation
    # ------------------------------------------------------------------ #

    def validate_resolution(
        self,
        turn: int,
        world: Optional[WorldState] = None,
        recent_interactions: Optional[list[dict]] = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if episode objectives are met and whether the beat can end early.

        Returns (complete: bool, unresolved: list[str]).
        """
        # Reuse recent result if we are between configured check intervals.
        if (
            self._last_completion_result is not None
            and turn < self.max_turns
            and turn >= self.min_turns_before_completion
            and (turn - self._last_completion_check_turn) < self.completion_check_interval
        ):
            return self._last_completion_result

        unresolved: list[str] = []
        for clue in self.clue_manager.required_clues:
            cid = clue.get("id")
            if cid and not self.clue_manager.is_introduced(cid):
                unresolved.append(f"Clue not surfaced: {cid}")

        unresolved_threads = self._find_unresolved_threads(
            world=world,
            recent_interactions=recent_interactions,
        )
        for item in unresolved_threads:
            unresolved.append(f"Plot thread unresolved: {item}")

        beat_complete = False
        if (
            len(unresolved) == 0
            and turn >= self.min_turns_before_completion
        ):
            beat_complete = self._is_beat_complete_now(
                turn=turn,
                world=world,
                recent_interactions=recent_interactions,
            )

        complete = (len(unresolved) == 0 and beat_complete)

        self._last_completion_check_turn = turn
        self._last_completion_result = (complete, unresolved)
        if not complete:
            self._log("resolution_check", "director",
                      f"Episode not complete at turn {turn}: {unresolved}", {})
        return complete, unresolved

    # ------------------------------------------------------------------ #
    # 6. Storyline Guidance
    # ------------------------------------------------------------------ #

    def get_storyline_guidance(self) -> Optional[str]:
        """
        Return a concise storyline brief for this episode (if available).
        Includes story arc context for pacing and emotional trajectory.
        """
        current = self.storyline_context.get("current")
        if not current:
            return None

        prev = self.storyline_context.get("previous")
        nxt = self.storyline_context.get("next")
        nxt = nxt[0] if isinstance(nxt, list) and nxt else None

        lines = []
        title = self.storyline_context.get("title")
        if title:
            lines.append(f"Story: {title}")

        # Story Arc Information (NEW)
        arc_info = self.storyline_context.get("story_arc", {})
        if arc_info:
            arc_name = arc_info.get("name", "")
            arc_pos = arc_info.get("act_position", "")
            arc_desc = arc_info.get("description", "")
            emotional_traj = arc_info.get("emotional_trajectory", "")
            ep_in_arc = arc_info.get("episode_in_arc", 0)
            total_in_arc = arc_info.get("total_in_arc", 0)
            progress = arc_info.get("progress_percentage", 0)

            lines.append(f"\n📖 Story Arc: {arc_name} ({arc_pos})")
            lines.append(f"   Episode {ep_in_arc}/{total_in_arc} in this arc ({progress}% complete)")
            lines.append(f"   Arc Focus: {arc_desc}")
            lines.append(f"   Emotional Trajectory: {emotional_traj}")

            # Arc position guidance
            if arc_info.get("is_arc_opening"):
                lines.append("   ⚡ Arc Opening: Establish new dynamics, introduce key elements")
            elif arc_info.get("is_arc_climax"):
                lines.append("   🔥 Arc Climax: Peak tension, major revelations, critical decisions")

            key_reveals = arc_info.get("key_reveals", [])
            if key_reveals:
                reveals_text = ", ".join(key_reveals[:3])
                lines.append(f"   Expected Reveals: {reveals_text}")

        act_title = current.get("act_title", "")
        act_id = current.get("act_id", "")
        if act_title or act_id:
            lines.append(f"\nCurrent act: {act_id} - {act_title}".strip(" -"))

        lines.append(
            f"Current milestone ({current.get('id', '')}): "
            f"{self._truncate(current.get('description', ''), 220)}"
        )

        if prev:
            lines.append(
                f"Previous milestone: "
                f"{self._truncate(prev.get('description', ''), 180)}"
            )
        if nxt:
            lines.append(
                f"Next milestone: "
                f"{self._truncate(nxt.get('description', ''), 180)}"
            )

        lines.append(
            "\n⚠️ Guardrail: Stay within current milestone's scope. "
            "Do not jump to future outcomes or reveal information meant for later arcs."
        )
        if self.guardian_briefing:
            lines.append(
                "\n--- Guardian Story Briefing (이번 화 일관성 유의사항) ---\n"
                + self.guardian_briefing
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 7. Turn Pacing
    # ------------------------------------------------------------------ #

    def get_pacing_hint(self, turn: int, recent_interactions: list[dict]) -> Optional[str]:
        """
        Return a pacing nudge for the Director's injection prompt if the
        story is dragging or rushing.
        """
        target_turns = self.pacing.get("target_turns", self.max_turns)
        pacing_style = self.pacing.get("style", "normal")
        progress = turn / max(target_turns, 1)
        progress_signal = self._scene_progress_signal(recent_interactions)
        jargon_onboarded = self._recent_jargon_already_onboarded(recent_interactions)

        if progress_signal["stalled"] and self._reader_reports_stalled_progression():
            return (
                "Reader flagged stalled progression. If the beat already landed, close the scene now; "
                "otherwise force one concrete shift such as a decision, interruption, movement, reveal, or room-state cue."
            )
        if progress_signal["explanation_loop"]:
            return (
                "The scene is explaining instead of moving. Cut the next turn into short direct sentences, "
                "translate the point into plain consequence, and show it through reaction, choice, interruption, "
                "or physical movement instead of another interpretation pass. If a bridge opener keeps repeating, "
                "replace it with a concrete action or room cue."
            )
        if progress_signal["technical_stall"]:
            return (
                "Technical back-and-forth is looping. Treat latency/real-time-style terms as already explained "
                f"{'once ' if jargon_onboarded else ''}and move the next beat through Sumin's judgment, emotion, "
                "or choice. If a technical point remains, translate it into one plain consequence and force "
                "a decision, interruption, movement, or visible room reaction."
            )
        if progress_signal.get("anxious_pressure"):
            return (
                "The cast is emotionally keyed up. Let the next beat show hesitation, a question, a check-in, "
                "or a small bodily reaction instead of another neutral explanation."
            )
        if progress_signal.get("frustration_pressure"):
            return (
                "Frustration is rising. Use a blunt refusal, interruption, or sharper response to change the scene."
            )
        if progress_signal.get("confidence_pressure"):
            return (
                "A decisive posture is available. Let the next beat commit, set terms, or force a concrete choice."
            )
        if progress_signal.get("emotional_conflict"):
            return (
                "The cast's emotional state is split. Pick the next speaker who exposes that split with a reaction, "
                "refusal, or decision rather than another neutral summary."
            )
        if progress_signal["signal_stack"]:
            return (
                "The scene is stacking warning cues without cashing them out. Do not add another omen. "
                "Turn the sharpest existing cue into reaction, confrontation, movement, or a scene exit."
            )
        if progress_signal.get("pressure_peak"):
            return (
                "The scene is at a pressure peak. Keep it open and cash the pressure out through a concrete move, "
                "reveal, counteroffer, or room reaction instead of ending on atmosphere alone. "
                "Name the price explicitly: access, security, responsibility, contract, or deadline."
            )
        if progress_signal.get("scene_boundary_ready"):
            return (
                "A concrete scene boundary has already landed. Close the beat or move to the next turn "
                "instead of reopening the same pressure."
            )
        if progress_signal["repeated_concern"]:
            if self._reader_wants_repeated_confrontation_merge():
                return (
                    "The same high-pressure contact is being reopened. Keep one linear axis only: question/response, "
                    "then approach/terms, then intervention or scene close. Do not restart the exchange from zero. "
                    "Keep only two or three core conditions, then cash out the next beat through a new consequence, "
                    "choice, interruption, movement, or one concrete leverage detail such as affiliation, card, number, or room reaction."
                )
            return (
                "The exchange is revisiting the same leverage point. Do not restate it again. "
                "Keep only two or three core conditions, then cash out the next beat through a new consequence, "
                "choice, interruption, movement, or one concrete leverage detail such as affiliation, card, number, or room reaction."
            )
        if progress_signal["flat_tension"]:
            return (
                "The dialogue has stayed at one pressure level for too long. Add a rhythm change: "
                "brief human reaction, uneasy silence, dry aside, gaze shift, hand movement, or other small physical movement, "
                "then sharpen the next question or choice. Avoid recycling the same opening phrase; use a visible action, "
                "object, or room-state cue instead. Reserve clipped hard-stop lines for real reveals, choices, or scene exits."
            )
        if progress_signal["stalled"]:
            return (
                "The current exchange is reiterating itself. Force a concrete shift: "
                "decision, interruption, movement, or revelation instead of another paraphrase."
            )

        if progress < 0.3 and pacing_style in ("tense", "fast"):
            return "The story should be building tension quickly. Encourage conflict or revelation."
        if progress > 0.8 and not self.clue_manager.unintroduced_required():
            return "The story is nearing its end. Begin converging threads toward resolution."
        return None

    # ------------------------------------------------------------------ #
    # 8. Beat Completion
    # ------------------------------------------------------------------ #

    def _find_unresolved_threads(
        self,
        world: Optional[WorldState],
        recent_interactions: Optional[list[dict]],
    ) -> list[str]:
        if not self.must_resolve:
            return []

        recent = recent_interactions or []
        if not recent:
            return list(self.must_resolve)

        recent_text = "\n".join(
            f"- {i.get('speaker_name', '?')}: {self._truncate(str(i.get('content', '')), 220)}"
            for i in recent[-12:]
        ) or "(none)"
        world_scene = world.current_scene if world else self.episode_config.get("summary", "")

        prompt = (
            f"You are a strict episode-resolution checker.\n\n"
            f"Episode summary:\n{self.episode_config.get('summary', '')}\n\n"
            f"Current scene snapshot:\n{self._truncate(str(world_scene), 600)}\n\n"
            f"Required threads that must be resolved this episode:\n"
            + "\n".join(f"- {item}" for item in self.must_resolve)
            + "\n\n"
            f"Recent interactions:\n{recent_text}\n\n"
            f"Task: mark only clearly unresolved threads. Do not guess.\n"
            f"Important: use exact thread strings from the required-thread list.\n"
            f"Reply JSON only:\n"
            f"{{\"unresolved\": [\"thread text\", \"...\"], \"reason\": \"short reason\"}}"
        )
        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_thread_resolution_check",
            max_tokens=260,
        )
        parsed = self._parse_json(result)
        unresolved_raw = parsed.get("unresolved", [])
        if not isinstance(unresolved_raw, list):
            unresolved_raw = []

        unresolved: list[str] = []
        known = {self._normalize_key(item): item for item in self.must_resolve}
        for item in unresolved_raw:
            key = self._normalize_key(str(item))
            if key in known:
                unresolved.append(known[key])

        # Conservative fallback: if parser failed hard, keep unresolved to avoid false-ending.
        if not unresolved_raw and not parsed:
            return list(self.must_resolve)
        return self._dedupe_preserve_order(unresolved)

    def _is_beat_complete_now(
        self,
        turn: int,
        world: Optional[WorldState],
        recent_interactions: Optional[list[dict]],
    ) -> bool:
        recent = recent_interactions or []
        if not recent:
            return False

        current = self.storyline_context.get("current") or {}
        current_milestone = current.get("description", "")
        if not current_milestone:
            current_milestone = self.episode_config.get("summary", "")

        recent_text = "\n".join(
            f"- {i.get('speaker_name', '?')}: {self._truncate(str(i.get('content', '')), 220)}"
            for i in recent[-12:]
        ) or "(none)"
        scene_snapshot = world.current_scene if world else self.episode_config.get("summary", "")

        prompt = (
            f"You are deciding whether an episode beat is complete.\n\n"
            f"Current milestone objective:\n{current_milestone}\n\n"
            f"Episode summary:\n{self.episode_config.get('summary', '')}\n\n"
            f"Current scene snapshot:\n{self._truncate(str(scene_snapshot), 700)}\n\n"
            f"Recent interactions:\n{recent_text}\n\n"
            f"Has the beat reached a natural completion point for ending this episode now?\n"
            f"Do not require every possible detail; only require that core beat intent has clearly landed.\n"
            f"If the remaining material would mostly restate explanation, terminology, or mood that already landed, treat the beat as complete.\n"
            f"Reply JSON only:\n"
            f"{{\"complete\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
        )

        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_beat_completion_check",
            max_tokens=220,
        )
        parsed = self._parse_json(result)
        complete = bool(parsed.get("complete"))
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        if complete and confidence >= self.completion_confidence_threshold:
            self._log(
                "beat_complete",
                "director",
                f"Beat complete at turn {turn} (confidence={confidence:.2f})",
                {"reason": parsed.get("reason", ""), "turn": turn},
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # 9. Trial Failure Analysis
    # ------------------------------------------------------------------ #

    def analyze_failure(
        self,
        interactions: list[dict],
        clue_manager: ClueManager,
        world: WorldState,
        agents: list[Agent],
    ) -> dict[str, dict]:
        """
        Analyze why a trial failed and suggest per-agent steering updates.

        Returns dict[agent_id -> {"tactical_goals": [...], "steering_prompt": "...", "reasoning": "..."}]
        """
        undiscovered = clue_manager.unintroduced_required()
        undiscovered_text = "\n".join(
            f"- {c.get('id')}: {c.get('content', '')}" for c in undiscovered
        ) or "(all clues found)"

        unresolved_threads = self._find_unresolved_threads(
            world=world,
            recent_interactions=interactions[-12:],
        )
        threads_text = "\n".join(f"- {t}" for t in unresolved_threads) or "(all resolved)"

        agent_interaction_counts: dict[str, int] = {}
        for ix in interactions:
            sid = ix.get("speaker_id", "")
            agent_interaction_counts[sid] = agent_interaction_counts.get(sid, 0) + 1

        agent_map = {a.id: a.name for a in agents}
        agent_summary = "\n".join(
            f"- {agent_map.get(aid, aid)}: {count} interactions, "
            f"knows clues: {list(clue_manager.agent_knowledge.get(aid, set()))}"
            for aid, count in sorted(agent_interaction_counts.items(),
                                      key=lambda x: -x[1])
            if aid != "director"
        )

        recent_text = "\n".join(
            f"[Turn {i.get('turn')}] {i.get('speaker_name', '?')}: "
            f"{self._truncate(str(i.get('content', '')), 200)}"
            for i in interactions[-15:]
        )

        prompt = (
            f"You are a story simulation analyst. A trial of episode "
            f"'{self.episode_config.get('id', '')}' FAILED.\n\n"
            f"## Episode Summary\n{self.episode_config.get('summary', '')}\n\n"
            f"## Undiscovered Clues\n{undiscovered_text}\n\n"
            f"## Unresolved Plot Threads\n{threads_text}\n\n"
            f"## Agent Activity Summary\n{agent_summary}\n\n"
            f"## Final 15 Interactions\n{recent_text}\n\n"
            f"## Task\n"
            f"For each active agent, provide:\n"
            f"1. tactical_goals: 2-3 concrete short-term actions for the next trial "
            f"(specific, actionable directives, not vague instructions)\n"
            f"2. steering_prompt: A paragraph of guidance telling the agent what to "
            f"do differently to help discover the missing clues\n"
            f"3. reasoning: Why this agent is key to discovering specific clues\n\n"
            f"Reply JSON only:\n"
            f"{{\"agents\": {{\"agent_id\": {{\"tactical_goals\": [...], "
            f"\"steering_prompt\": \"...\", \"reasoning\": \"...\"}}, ...}}}}"
        )

        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_failure_analysis",
            use_premium=True,
            max_tokens=1200,
        )
        parsed = self._parse_json(result)
        agent_updates = parsed.get("agents", {})

        self._log("failure_analysis", "director",
                  f"Analyzed failure: {len(undiscovered)} undiscovered clues, "
                  f"{len(unresolved_threads)} unresolved threads",
                  {"undiscovered": [c.get("id") for c in undiscovered],
                   "unresolved": unresolved_threads})

        return agent_updates

    # ------------------------------------------------------------------ #
    # 10. Success Pattern Extraction
    # ------------------------------------------------------------------ #

    def extract_success_patterns(
        self,
        interactions: list[dict],
        clue_manager: ClueManager,
    ) -> list[dict]:
        """
        Extract successful interaction patterns from a winning trial.

        For each discovered clue, find the 2-3 interactions immediately
        preceding the discovery and package them as exemplar sequences.
        """
        patterns = []

        for clue_id, discovery_turn in clue_manager.introduced.items():
            discovering_agent = None
            for aid, clue_set in clue_manager.agent_knowledge.items():
                if clue_id in clue_set:
                    discovering_agent = aid
                    break

            preceding = [
                ix for ix in interactions
                if ix.get("turn", 0) >= discovery_turn - 2
                and ix.get("turn", 0) <= discovery_turn
            ]
            preceding = preceding[-3:]

            exemplar_text = "\n".join(
                f"[{ix.get('speaker_name', '?')}]: "
                f"{self._truncate(str(ix.get('content', '')), 300)}"
                for ix in preceding
            )

            patterns.append({
                "clue_id": clue_id,
                "discovery_turn": discovery_turn,
                "discovering_agent": discovering_agent,
                "exemplar_sequence": preceding,
                "exemplar_text": exemplar_text,
            })

        self._log("success_extraction", "director",
                  f"Extracted {len(patterns)} success patterns",
                  {"clue_ids": [p["clue_id"] for p in patterns]})

        return patterns

    # ------------------------------------------------------------------ #
    # 11. Dynamic Turn Allocation
    # ------------------------------------------------------------------ #

    def decide_next_speaker(
        self,
        turn: int,
        world: WorldState,
        agents: list[Agent],
        recent_interactions: Optional[list[dict]] = None,
        protagonist_id: Optional[str] = None,
    ) -> dict:
        """
        Decide who should speak next (or whether the current scene should end).

        Returns:
            {
              "speaker_id": "<agent_id>",
              "end_scene": bool,
              "reason": "<short explanation>",
            }
        """
        agent_map = {a.id: a for a in agents}
        active_ids = [aid for aid in world.active_agents if aid in agent_map]
        if not active_ids:
            fallback = protagonist_id if protagonist_id in agent_map else (
                agents[0].id if agents else ""
            )
            return {
                "speaker_id": fallback,
                "end_scene": False,
                "reason": "no active cast; fallback speaker",
            }

        if len(active_ids) == 1:
            return {
                "speaker_id": active_ids[0],
                "end_scene": False,
                "reason": "single active speaker",
            }

        recent = recent_interactions or []
        # Recent non-director speakers for anti-monologue balancing
        recent_speakers = [
            str(i.get("speaker_id", "")).strip()
            for i in recent[-6:]
            if str(i.get("speaker_id", "")).strip() != "director"
        ]
        progress_signal = self._scene_progress_signal(recent, agents=agents)
        preferred_speaker_id = self._choose_next_speaker(
            active_ids,
            agent_map,
            recent_speakers,
            progress_signal,
            protagonist_id=protagonist_id,
        )
        recent_text = "\n".join(
            (
                f"- T{i.get('turn', '?')} | {i.get('speaker_name', '?')}: "
                f"{self._truncate(str(i.get('content', '')), 180)}"
            )
            for i in recent[-8:]
        ) or "(none)"

        active_text = "\n".join(
            f"- {aid}: {agent_map[aid].name}" for aid in active_ids
        )
        speaker_signature_block = self._build_speaker_turn_signatures(
            agent_map,
            active_ids,
            recent_speakers,
            recent,
            progress_signal,
        )
        jargon_onboarded = self._recent_jargon_already_onboarded(recent)
        prefers_technical_restraint = self.reader_profile.prefers_technical_term_restraint()
        prefers_sentence_simplification = self.reader_profile.prefers_sentence_simplification()
        prefers_observable_emotion = self.reader_profile.prefers_observable_emotion_evidence()
        prefers_scene_compaction = self.reader_profile.prefers_stronger_scene_compaction()
        protagonist_focus_rule = ""
        if jargon_onboarded and protagonist_id in active_ids:
            protagonist_name = agent_map[protagonist_id].name
            protagonist_focus_rule = (
                f"10) Technical keywords already had a first explanation in recent turns. "
                f"Do not define them again. Route the next beat through {protagonist_name}'s "
                "judgment, emotion, question, or choice.\n"
            )
        jargon_reaction_rule = ""
        if progress_signal["technical_stall"] or prefers_technical_restraint:
            jargon_reaction_rule = (
                "11) If the next turn keeps a technical or English term, it must be followed by "
                "an immediate human reaction, emotion, or choice in the same turn. "
                "If the term already landed once, do not define it again.\n"
            )
        emotional_pressure_rule = ""
        if progress_signal.get("anxious_pressure"):
            emotional_pressure_rule += (
                "- The cast reads anxious or guarded. Prefer hesitation, probing, or a concrete check-in "
                "instead of another neutral explanation.\n"
            )
        if progress_signal.get("frustration_pressure"):
            emotional_pressure_rule += (
                "- Frustration is visible. Let the next turn sharpen the exchange with a blunt challenge, "
                "interruption, or refusal.\n"
            )
        if progress_signal.get("confidence_pressure"):
            emotional_pressure_rule += (
                "- A decisive posture is available. Use it to advance the scene with a choice, boundary, "
                "or direct offer.\n"
            )
        if progress_signal.get("emotional_conflict"):
            emotional_pressure_rule += (
                "- The cast's emotional state is split. Choose the speaker whose reaction would visibly "
                "change the room, not the one who merely explains it.\n"
            )
        brevity_rule = ""
        if progress_signal["explanation_loop"] or prefers_sentence_simplification:
            brevity_rule = (
                "12) Favor short direct sentences for the next turn. Split comma-heavy clause chains "
                "into 1-2 clear beats.\n"
            )
        show_dont_tell_rule = ""
        if progress_signal["explanation_loop"] or prefers_observable_emotion:
            show_dont_tell_rule = (
                "13) Do not spend the next turn defining or interpreting the situation again. "
                "Show pressure through visible reaction, interruption, gesture, movement, breath, or a blunt choice.\n"
            )
        repeated_concern_rule = ""
        if progress_signal["repeated_concern"]:
            repeated_concern_rule = (
                "14) Recent turns are revisiting the same concern. Do not paraphrase it again; "
                "either end the scene or force a new consequence, choice, interruption, or movement. "
                "Keep at most 2-3 core conditions and let any extra pressure arrive through a concrete detail like affiliation, card, number, or room reaction.\n"
            )
        confrontation_axis_rule = ""
        if self._reader_wants_repeated_confrontation_merge():
            confrontation_axis_rule = (
                "15) Keep high-pressure contact on one linear axis: question/response -> approach/terms -> intervention or scene close. "
                "If the same pair already made contact, do not restart from another fresh stare-down or another first question.\n"
            )
        pressure_peak_rule = ""
        if progress_signal.get("pressure_peak"):
            pressure_peak_rule = (
                "16) The scene is at a pressure peak. Keep it open until a concrete consequence, reveal, or exit cue lands; "
                "do not end it on atmosphere alone. If a proposal is on the table, make the cost concrete instead of abstract.\n"
            )
        scene_boundary_rule = ""
        if progress_signal.get("scene_boundary_ready"):
            scene_boundary_rule = (
                "17) A concrete scene boundary has already landed. Prefer ending the scene or moving to the next beat "
                "instead of reopening the same pressure.\n"
            )
        inner_conflict_rule = ""
        if progress_signal.get("inner_conflict"):
            inner_conflict_rule = (
                "18) The protagonist's next useful beat is the split itself: hesitation, self-correction, or a choice that resolves the split. "
                "Do not flatten it into neutral explanation.\n"
            )
        concrete_risk_rule = ""
        if progress_signal.get("concrete_risk"):
            concrete_risk_rule = (
                "19) A risk is in play. Name it through a specific limit, clause, deadline, access rule, or visible consequence instead of repeating an abstract warning.\n"
            )
        speaker_choice_rule = ""
        if preferred_speaker_id and preferred_speaker_id in active_ids:
            preferred_name = agent_map[preferred_speaker_id].name
            speaker_choice_rule = (
                f"20) If multiple speakers could work, prefer {preferred_name} because that turn best changes the room's pressure or emotion.\n"
            )

        prompt = (
            f"You are a story turn allocator.\n\n"
            f"Turn: {turn}\n"
            f"Episode summary: {self._truncate(str(self.episode_config.get('summary', '')), 500)}\n"
            f"Current location: {self.episode_config.get('location', world.location)}\n"
            f"Active cast (ONLY choose from these IDs):\n{active_text}\n\n"
            f"Recent interactions:\n{recent_text}\n\n"
            f"Character turn signatures:\n{speaker_signature_block}\n\n"
            f"Task:\n"
            f"1) Choose ONE next speaker from active cast IDs.\n"
            f"2) You may set end_scene=true only if the exchange naturally closed.\n"
            f"3) Do NOT force ping-pong dialogue. If one person is lecturing and others are listening, "
            f"it is valid to keep the lecturer as next speaker repeatedly.\n"
            f"4) If end_scene=true, still provide speaker_id for who should continue after scene closure.\n\n"
            f"5) Balance speaking opportunities: avoid choosing the same speaker 3+ turns in a row "
            f"when other active speakers are available.\n"
            f"{'6) Recent turns are stalling: pick a speaker who changes the situation, or end the scene if the beat already landed.\\n' if progress_signal['stalled'] else ''}"
            f"{'7) A natural exit cue is already present, so prefer end_scene=true unless another turn clearly adds pressure.\\n' if progress_signal['closure_ready'] else ''}"
            f"{'8) Recent turns are circling the same explanation or bridge opener without enough reaction or decision change; prefer a speaker who turns it into emotion, conflict, movement, or a scene close.\\n' if progress_signal['technical_stall'] else ''}"
            f"{'9) Recent turns keep landing on the same pressure note or opening phrase; either close the scene or insert a plain human reaction before another sharp line.\\n' if progress_signal['flat_tension'] else ''}"
            f"{'10) The scene is stacking warning cues; do not add another memo/alert/watcher beat. Cash out the sharpest cue through reaction, confrontation, movement, or a scene close.\\n' if progress_signal['signal_stack'] else ''}\n"
            f"{protagonist_focus_rule}"
            f"{jargon_reaction_rule}"
            f"{emotional_pressure_rule}"
            f"{brevity_rule}"
            f"{show_dont_tell_rule}"
            f"{repeated_concern_rule}"
            f"{confrontation_axis_rule}"
            f"{pressure_peak_rule}"
            f"{scene_boundary_rule}"
            f"{inner_conflict_rule}"
            f"{concrete_risk_rule}"
            f"{speaker_choice_rule}"
            f"Reply JSON only:\n"
            f"{{\"speaker_id\": \"agent_id\", \"end_scene\": true/false, \"reason\": \"...\"}}"
        )
        if prefers_scene_compaction:
            prompt += (
                "\nReader priority: if recent turns are paraphrasing the same point or mood without new information, "
                "prefer ending the scene over extending the exchange."
            )
        if prefers_technical_restraint:
            prompt += (
                "\nReader priority: do not spend another turn unpacking terminology unless the plot truly requires it; "
                "prefer visible reaction, choice, or interruption."
                "\nIf terminology still appears, the turn should translate it into plain consequence and immediate human response."
            )
        if prefers_observable_emotion:
            prompt += (
                "\nReader priority: cut low-value explanation first. If a point is already clear, use the next turn for action, reaction, silence, or a forced decision instead of restating it."
            )
        if prefers_scene_compaction:
            prompt += (
                "\nReader priority: if recent turns keep ending on similar sharp lines, end the scene or pivot to a calmer human reaction before escalating again."
            )
        if progress_signal["signal_stack"]:
            prompt += (
                "\nReader priority: do not add another warning-style cue. Turn the existing cue into an answer, confrontation, movement, or scene exit."
            )
        if prefers_sentence_simplification:
            prompt += (
                "\nReader priority: prefer turns that say one point at a time in short direct sentences."
            )
        if self._reader_reports_stalled_progression():
            prompt += (
                "\nReader priority: if the core beat has already landed, end the scene early instead of extending another diagnostic turn."
            )
        if progress_signal.get("inner_conflict"):
            prompt += (
                "\nReader priority: prefer the speaker who can expose the protagonist's hesitation or split motive instead of another neutral explanation."
            )
        if progress_signal.get("concrete_risk"):
            prompt += (
                "\nReader priority: prefer the speaker who can state the concrete cost, limit, or consequence in one line."
            )

        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_turn_allocation",
            use_premium=True,
            max_tokens=180,
        )
        parsed = self._parse_json(result)

        speaker_id = str(parsed.get("speaker_id", "")).strip()
        end_scene_raw = parsed.get("end_scene", False)
        end_scene = end_scene_raw if isinstance(end_scene_raw, bool) else (
            str(end_scene_raw).strip().lower() in {"true", "yes", "1", "y"}
        )
        reason = str(parsed.get("reason", "")).strip()

        # Validate chosen speaker conservatively
        if speaker_id not in active_ids:
            speaker_id = active_ids[0]
            reason = reason or "invalid speaker from allocator; fallback to first active"

        if (
            speaker_id != preferred_speaker_id
            and preferred_speaker_id in active_ids
            and (
                progress_signal["stalled"]
                or progress_signal["technical_stall"]
                or progress_signal["explanation_loop"]
                or progress_signal["repeated_concern"]
                or progress_signal["signal_stack"]
                or progress_signal.get("inner_conflict")
                or progress_signal.get("concrete_risk")
            )
        ):
            speaker_id = preferred_speaker_id
            reason = (reason + "; " if reason else "") + \
                "director override to use the speaker most likely to advance the scene"

        # Deterministic anti-monologue safety:
        # if the same speaker has taken the last 2 non-director turns,
        # rotate to another active speaker (when available).
        if len(active_ids) > 1 and len(recent_speakers) >= 2:
            if recent_speakers[-1] == speaker_id and recent_speakers[-2] == speaker_id:
                alternates = [aid for aid in active_ids if aid != speaker_id]
                if alternates:
                    speaker_id = alternates[0]
                    end_scene = False
                    reason = (reason + "; " if reason else "") + \
                        "anti-monologue rotation after consecutive same-speaker turns"

        if (
            not end_scene
            and protagonist_id in active_ids
            and speaker_id != protagonist_id
            and (jargon_onboarded or progress_signal["technical_stall"])
            and recent_speakers
            and recent_speakers[-1] != protagonist_id
        ):
            speaker_id = protagonist_id
            reason = (reason + "; " if reason else "") + \
                "protagonist reaction turn after technical onboarding"

        if (
            not end_scene
            and progress_signal.get("inner_conflict")
            and protagonist_id in active_ids
            and speaker_id != protagonist_id
            and recent_speakers
            and recent_speakers[-1] != protagonist_id
        ):
            speaker_id = protagonist_id
            reason = (reason + "; " if reason else "") + \
                "protagonist turn to surface the internal split"

        if (
            not end_scene
            and progress_signal["signal_stack"]
            and protagonist_id in active_ids
            and speaker_id != protagonist_id
            and recent_speakers
            and recent_speakers[-1] != protagonist_id
        ):
            speaker_id = protagonist_id
            reason = (reason + "; " if reason else "") + \
                "protagonist reaction turn to cash out stacked warning cues"

        if progress_signal["stalled"] and len(active_ids) > 1 and recent_speakers:
            if speaker_id == recent_speakers[-1]:
                alternates = [aid for aid in active_ids if aid != speaker_id]
                if alternates:
                    speaker_id = alternates[0]
                    end_scene = False
                    reason = (reason + "; " if reason else "") + \
                        "scene-stall rotation to force new pressure"

        if progress_signal.get("pressure_peak") and not progress_signal["closure_ready"]:
            end_scene = False
            reason = (reason + "; " if reason else "") + \
                "hold scene open at pressure peak until a concrete exit cue lands"
            if len(active_ids) > 1 and recent_speakers and speaker_id == recent_speakers[-1]:
                alternates = [aid for aid in active_ids if aid != speaker_id]
                if alternates:
                    speaker_id = alternates[0]
                    reason = (reason + "; " if reason else "") + \
                        "pressure-peak rotation to force a fresh reaction"
        end_scene, closure_reason = self._should_end_scene(
            progress_signal,
            recent,
            prefers_scene_compaction=prefers_scene_compaction,
            current_end_scene=end_scene,
        )
        if closure_reason:
            reason = (reason + "; " if reason else "") + closure_reason

        self._log(
            "turn_allocation",
            "director",
            f"Turn {turn}: speaker={speaker_id} end_scene={end_scene}",
            {
                "active_ids": active_ids,
                "reason": reason,
            },
        )
        return {
            "speaker_id": speaker_id,
            "end_scene": end_scene,
            "reason": reason,
        }

    # ------------------------------------------------------------------ #
    # 12. Episode Cast Selection
    # ------------------------------------------------------------------ #

    def select_active_agents(self, agents: list[Agent], world: WorldState) -> list[str]:
        """
        Choose which characters should take turns in this episode.
        Returns a validated list of agent IDs.

        Supports cross-episode cast continuity: if the current episode's
        location matches the previous episode's, reuse that cast as a baseline.
        """
        if not agents:
            return []

        candidate_ids = {a.id for a in agents}
        protagonist = next((a for a in agents if a.role == "protagonist"), None)

        # Optional manual cast in episode YAML:
        #   episode.characters: ["id1", "id2"] or [{id: "id1"}, ...]
        manual_cast = self._extract_episode_character_ids(self.episode_config.get("characters"))
        if manual_cast:
            selected = [cid for cid in manual_cast if cid in candidate_ids]
            selected = self._dedupe_preserve_order(selected)
            if selected:
                self._log(
                    "cast_selection",
                    "director",
                    f"Using episode-defined cast ({len(selected)} agents)",
                    {"active_agents": selected, "source": "episode.characters"},
                )
                return selected

        selected = self._select_cast_by_explicit_mentions(agents)
        if selected:
            self._log(
                "cast_selection",
                "director",
                f"Using strict explicit-mention cast ({len(selected)} agents)",
                {
                    "active_agents": selected,
                    "source": "strict_explicit_mentions",
                },
            )
            return selected

        # Cross-episode continuity: check if previous episode used same location
        episode_id = self.episode_config.get("id", "")
        current_location = self.episode_config.get("location", world.location)
        prev_state = db.load_previous_episode_final_state(str(episode_id))
        if prev_state:
            prev_location = prev_state.get("location", "")
            prev_active = prev_state.get("active_agents", [])
            if (prev_location and current_location
                    and self._normalize_key(prev_location) == self._normalize_key(current_location)
                    and prev_active):
                # Reuse previous episode's cast, filtered to current candidates
                carried = [aid for aid in prev_active if aid in candidate_ids]
                # Ensure protagonist is always included
                if protagonist and protagonist.id not in carried:
                    carried.insert(0, protagonist.id)
                carried = self._dedupe_preserve_order(carried)
                if carried:
                    self._log(
                        "cast_selection",
                        "director",
                        f"Reusing previous episode cast (same location: {prev_location})",
                        {
                            "active_agents": carried,
                            "source": "cross_episode_continuity",
                            "prev_location": prev_location,
                        },
                    )
                    return carried

        # Conservative no-guess fallback:
        # - default 1 (monologue/solo scene allowed)
        # - widen to 2 only for dialogue/continuity-sensitive episodes
        #   when explicit evidence is absent.
        fallback_size = self._conditional_fallback_cast_size(agents, world)

        fallback = [protagonist.id] if protagonist else [agents[0].id]
        if fallback_size > 1:
            candidate_pool = [a.id for a in agents if a.id not in fallback]
            fallback.extend(candidate_pool[: max(0, fallback_size - len(fallback))])
            fallback = self._dedupe_preserve_order(fallback)
        self._log(
            "cast_selection",
            "director",
            f"No explicit cast evidence found; using fallback cast size={len(fallback)}",
            {
                "active_agents": fallback,
                "source": "strict_fallback_minimal",
                "fallback_size_policy": fallback_size,
            },
        )
        return fallback

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _scene_progress_signal(
        self,
        recent_interactions: Optional[list[dict]],
        agents: Optional[list[Agent]] = None,
    ) -> dict[str, bool]:
        min_window = 3 if self._reader_wants_repeated_confrontation_merge() else 4
        recent = [
            i for i in (recent_interactions or [])
            if str(i.get("speaker_id", "")).strip() != "director"
        ][-5:]
        if len(recent) < min_window:
            return {
                "stalled": False,
                "closure_ready": False,
                "technical_stall": False,
                "flat_tension": False,
                "explanation_loop": False,
                "repeated_concern": False,
                "signal_stack": False,
                "emotional_shift": False,
                "decisive_shift": False,
                "pressure_peak": False,
                "anxious_pressure": False,
                "frustration_pressure": False,
                "confidence_pressure": False,
                "emotional_conflict": False,
                "inner_conflict": False,
                "concrete_risk": False,
            }

        tension_curve = self._build_tension_curve(recent, agents)
        recent_speakers = [
            str(i.get("speaker_id", "")).strip()
            for i in recent
            if str(i.get("speaker_id", "")).strip()
        ]
        repeated_speaker = len(recent_speakers) >= 3 and len(set(recent_speakers[-3:])) == 1
        repeated_pair = len(recent_speakers) >= (
            3 if self._reader_wants_repeated_confrontation_merge() else 4
        ) and len(set(recent_speakers[-(3 if self._reader_wants_repeated_confrontation_merge() else 4):])) <= 2
        mostly_dialogue = sum(
            1 for i in recent if str(i.get("action_type", "")).strip() == "dialogue"
        ) >= min(3, len(recent))
        fingerprints = [self._content_fingerprint(str(i.get("content", ""))) for i in recent]
        opener_fingerprints = [self._opening_phrase_fingerprint(str(i.get("content", ""))) for i in recent]
        repeated_openers = sum(
            1
            for idx in range(1, len(opener_fingerprints))
            if opener_fingerprints[idx] and opener_fingerprints[idx] == opener_fingerprints[idx - 1]
        )
        low_novelty = len({fp for fp in fingerprints if fp}) <= (
            1 if self._reader_wants_repeated_confrontation_merge() and len(recent) <= 3 else 2
        )
        technical_stall = self._has_repetitive_technical_exchange(recent)
        flat_tension = (
            self._has_flat_tension_plateau(recent)
            or repeated_openers >= 1
            or bool(tension_curve.get("flat"))
        )
        explanation_loop = self._has_explanatory_loop(recent) or repeated_openers >= 1
        repeated_concern = self._has_repeated_core_concern_exchange(recent)
        signal_stack = self._has_overloaded_threat_signal_stack(recent)
        closure_ready = self._has_scene_exit_cue(recent)
        emotional_shift = any(
            self._has_emotional_or_decisive_shift(str(i.get("content", "")))
            for i in recent[-2:]
        )
        consequence_shift = any(
            self._has_consequence_shift(str(i.get("content", "")))
            for i in recent[-2:]
        )
        motion_shift = any(
            re.search(
                r"(움직이|옮기|다가서|물러서|돌아서|떠나|나가|들어오|정리하|걸음|고개를 들|몸을 기울|손을 내밀|펜을 내려놓|종이를 건네|반걸음|한걸음)",
                str(i.get("content", "")),
            )
            for i in recent[-2:]
        )
        emotional_flags = self._current_emotional_pressure_flags(recent_speakers, agents)
        protagonist_ids = {
            agent.id
            for agent in (agents or [])
            if str(getattr(agent, "role", "")).strip() == "protagonist"
        }
        inner_conflict = self._has_inner_conflict_signal(recent, protagonist_ids)
        concrete_risk = self._has_concrete_risk_signal(recent)
        decisive_shift = any(
            re.search(
                r"(결정|선택|거절|수락|걸음을 옮|문으로 향|자료를 건넸|자리를 정리|돌아서|떠났|막아섰|고개를 끄덕|반걸음|물러섰|손을 내밀)",
                str(i.get("content", "")),
            )
            or self._has_confrontation_resolution_shift(str(i.get("content", "")))
            for i in recent[-2:]
        )
        pressure_peak = (
            (repeated_concern or signal_stack or technical_stall or flat_tension or explanation_loop)
            and (decisive_shift or emotional_shift)
        ) or bool(tension_curve.get("peak"))
        scene_boundary_ready = (
            (closure_ready and (decisive_shift or consequence_shift or motion_shift))
            or (emotional_shift and consequence_shift)
            or (pressure_peak and (consequence_shift or motion_shift))
            or (repeated_concern and closure_ready and len(recent) >= min_window)
            or (technical_stall and closure_ready and len(recent) >= min_window)
            or (flat_tension and repeated_speaker and len(recent) >= max(min_window, 4))
        )
        stalled = ((mostly_dialogue and (repeated_speaker or repeated_pair or low_novelty)) or (
            repeated_pair and low_novelty
        ) or technical_stall or flat_tension or explanation_loop or repeated_concern or signal_stack) and not decisive_shift
        if emotional_flags["emotional_conflict"]:
            stalled = False
        if inner_conflict:
            stalled = False
        return {
            "stalled": stalled,
            "closure_ready": closure_ready,
            "technical_stall": technical_stall,
            "flat_tension": flat_tension,
            "explanation_loop": explanation_loop,
            "repeated_concern": repeated_concern,
            "signal_stack": signal_stack,
            "emotional_shift": emotional_shift,
            "decisive_shift": decisive_shift,
            "pressure_peak": pressure_peak,
            "inner_conflict": inner_conflict,
            "concrete_risk": concrete_risk,
            "consequence_shift": consequence_shift,
            "motion_shift": motion_shift,
            "scene_boundary_ready": scene_boundary_ready,
            **emotional_flags,
        }

    def _choose_next_speaker(
        self,
        active_ids: list[str],
        agent_map: dict[str, Agent],
        recent_speakers: list[str],
        progress_signal: dict[str, bool],
        protagonist_id: Optional[str] = None,
    ) -> str:
        if not active_ids:
            return ""

        def score(aid: str) -> float:
            agent = agent_map[aid]
            emotion_family, intensity = self._dominant_emotion_family(agent.memory.emotional_state)
            value = 0.0
            if aid == protagonist_id:
                value += 1.0
            if recent_speakers and aid == recent_speakers[-1]:
                value -= 2.5
            if len(recent_speakers) >= 2 and recent_speakers[-1] == aid and recent_speakers[-2] == aid:
                value -= 4.0
            if progress_signal.get("inner_conflict"):
                if aid == protagonist_id:
                    value += 4.0
                elif emotion_family in {"anxious", "frustrated", "curious"}:
                    value += 2.0
            elif progress_signal.get("technical_stall"):
                if aid == protagonist_id:
                    value += 3.0
                elif emotion_family == "curious":
                    value += 2.0
            elif progress_signal.get("concrete_risk"):
                if emotion_family in {"confident", "frustrated"}:
                    value += 2.0
            elif progress_signal.get("pressure_peak"):
                if emotion_family in {"confident", "frustrated", "anxious"}:
                    value += 1.0
            elif progress_signal.get("closure_ready"):
                if emotion_family in {"confident", "relieved"}:
                    value += 1.5
            if progress_signal.get("stalled") and emotion_family in {"confident", "frustrated"}:
                value += 0.5
            if intensity >= 0.6:
                value += 0.5
            return value

        return max(active_ids, key=score)

    def _should_end_scene(
        self,
        progress_signal: dict[str, bool],
        recent: list[dict],
        *,
        prefers_scene_compaction: bool,
        current_end_scene: bool,
    ) -> tuple[bool, str]:
        high_pressure = (
            progress_signal.get("pressure_peak")
            or progress_signal.get("signal_stack")
            or progress_signal.get("concrete_risk")
            or progress_signal.get("inner_conflict")
            or progress_signal.get("emotional_conflict")
        )
        if high_pressure and not (
            progress_signal.get("decisive_shift")
            or progress_signal.get("consequence_shift")
            or progress_signal.get("motion_shift")
        ):
            if current_end_scene:
                return False, "pressure is still active; hold the scene open until it cashes out"
            return False, ""

        if progress_signal.get("scene_boundary_ready"):
            return True, "scene boundary landed on a concrete shift; move to the next beat"
        if progress_signal.get("closure_ready") and (
            progress_signal.get("repeated_concern")
            or progress_signal.get("technical_stall")
            or progress_signal.get("explanation_loop")
        ):
            return True, "scene closure to end a repeated beat once the exit cue is present"
        if progress_signal["stalled"] and progress_signal["closure_ready"]:
            return True, "scene closure to avoid drag after recent beat landed"
        if progress_signal["explanation_loop"] and len(recent) >= 4:
            return True, "scene closure to stop repeated explanation before it drags"
        if progress_signal["technical_stall"] and len(recent) >= 4:
            return True, "scene closure to stop repeated technical explanation without new emotional turn"
        if progress_signal["repeated_concern"] and len(recent) >= (
            3 if self._reader_wants_repeated_confrontation_merge() else 4
        ):
            return True, "scene closure to stop revisiting the same concern without new consequence"
        if progress_signal["signal_stack"] and (progress_signal["closure_ready"] or len(recent) >= 5):
            return True, "scene closure to avoid piling more warning cues without payoff"
        if progress_signal["flat_tension"] and (progress_signal["closure_ready"] or len(recent) >= 6):
            return True, "scene closure to keep repeated tension beats from flattening out"
        if progress_signal["stalled"] and len(recent) >= 4 and self._reader_reports_stalled_progression():
            return True, "reader-priority scene closure for stalled progression"
        if (
            progress_signal["stalled"]
            and len(recent) >= 5
            and prefers_scene_compaction
        ):
            return True, "reader-priority scene closure to cut repeated exchange"
        return current_end_scene, ""

    @staticmethod
    def _emotion_family(label: str) -> str:
        low = str(label or "").lower()
        if re.search(r"(불안|초조|긴장|걱정|두렵|fear|anx|unease|stress|nerv)", low):
            return "anxious"
        if re.search(r"(분노|짜증|격앙|불만|irrit|anger|frustr)", low):
            return "frustrated"
        if re.search(r"(확신|결심|의지|안정|calm|confid|resol)", low):
            return "confident"
        if re.search(r"(호기심|궁금|의문|surpris|shock|curious)", low):
            return "curious"
        if re.search(r"(안도|진정|안심|relief)", low):
            return "relieved"
        return "neutral"

    @classmethod
    def _dominant_emotion_family(cls, emotional_state: dict) -> tuple[str, float]:
        best_family = "neutral"
        best_value = 0.0
        if not isinstance(emotional_state, dict):
            return best_family, best_value
        for emotion, raw_value in emotional_state.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            family = cls._emotion_family(str(emotion))
            if value > best_value:
                best_family = family
                best_value = value
        return best_family, best_value

    def _current_emotional_pressure_flags(
        self,
        recent_speakers: list[str],
        agents: Optional[list[Agent]] = None,
    ) -> dict[str, bool]:
        flags = {
            "anxious_pressure": False,
            "frustration_pressure": False,
            "confidence_pressure": False,
            "emotional_conflict": False,
        }
        if not agents:
            return flags

        speaker_ids = {sid for sid in recent_speakers if sid}
        relevant_agents = [
            agent for agent in agents
            if not speaker_ids or agent.id in speaker_ids
        ]
        strong_families: set[str] = set()
        for agent in relevant_agents:
            family, intensity = self._dominant_emotion_family(agent.memory.emotional_state)
            if intensity < 0.35:
                continue
            if family == "anxious":
                flags["anxious_pressure"] = True
            elif family == "frustrated":
                flags["frustration_pressure"] = True
            elif family == "confident":
                flags["confidence_pressure"] = True
            if family != "neutral":
                strong_families.add(family)
        flags["emotional_conflict"] = len(strong_families) >= 2
        return flags

    @staticmethod
    def _has_inner_conflict_marker(text: str) -> bool:
        low = str(text or "").lower()
        return bool(re.search(
            r"(하지만|그러나|그런데|그치만|원하지만|원하긴 하지만|해야 하지만|하면서도|망설|주저|흔들|양가|둘 중|어느 쪽|선택을 못|결정하지 못|미루|꺼려)",
            low,
        ))

    @staticmethod
    def _has_concrete_risk_marker(text: str) -> bool:
        low = str(text or "").lower()
        return bool(re.search(
            r"(대가|조건|한계|지원|보안|출입|배지|명함|계약|조항|책임|허가|시설|deadline|limit|access|security|support|clause)",
            low,
        ))

    def _has_inner_conflict_signal(self, recent: list[dict], protagonist_ids: set[str]) -> bool:
        if not protagonist_ids:
            return False
        for row in recent[-4:]:
            speaker_id = str(row.get("speaker_id", "")).strip()
            if speaker_id not in protagonist_ids:
                continue
            if self._has_inner_conflict_marker(str(row.get("content", ""))):
                return True
        return False

    def _has_concrete_risk_signal(self, recent: list[dict]) -> bool:
        return any(
            self._has_concrete_risk_marker(str(row.get("content", "")))
            for row in recent[-4:]
        )

    def _speaker_response_hint(
        self,
        emotion_family: str,
        intensity: float,
        progress_signal: dict[str, bool],
    ) -> str:
        hint = {
            "anxious": "hesitate, verify, or ask for a concrete check",
            "frustrated": "interrupt, challenge, or refuse bluntly",
            "confident": "state terms, decide, or push the scene forward",
            "curious": "probe for a missing detail or next step",
            "relieved": "soften the exchange and close the beat",
        }.get(emotion_family, "react to the visible pressure")
        if progress_signal.get("technical_stall"):
            hint += "; translate any jargon into consequence instead of repeating it"
        if progress_signal.get("repeated_concern") and emotion_family in {"confident", "frustrated"}:
            hint += "; turn the repeated concern into a consequence, not a recap"
        if progress_signal.get("inner_conflict"):
            hint += "; expose a split motive or hesitation before the next choice"
        if progress_signal.get("concrete_risk"):
            hint += "; name the cost as a specific limit, clause, deadline, or access restriction"
        if intensity >= 0.6:
            hint += "; the emotion is strong enough to color the line"
        return hint

    def _build_speaker_turn_signatures(
        self,
        agent_map: dict[str, Agent],
        active_ids: list[str],
        recent_speakers: list[str],
        recent_interactions: list[dict],
        progress_signal: dict[str, bool],
    ) -> str:
        lines: list[str] = []
        tail_recent = recent_speakers[-2:] if recent_speakers else []
        tension_curve = self._build_tension_curve(recent_interactions)
        for aid in active_ids:
            agent = agent_map[aid]
            emotion_family, emotion_level = self._dominant_emotion_family(agent.memory.emotional_state)
            speech = agent.speech_profile if isinstance(agent.speech_profile, dict) else {}
            voice_bits: list[str] = []
            for key in ("tone", "cadence", "formality"):
                value = str(speech.get(key, "")).strip()
                if value:
                    voice_bits.append(f"{key}={value}")
            lexicon = speech.get("lexicon", [])
            if isinstance(lexicon, list):
                lexicon_bits = [str(item).strip() for item in lexicon[:4] if str(item).strip()]
                if lexicon_bits:
                    voice_bits.append(f"lexicon={', '.join(lexicon_bits)}")
            tics = speech.get("signature_tics", [])
            tic_bits: list[str] = []
            if isinstance(tics, list):
                tic_bits = [str(item).strip() for item in tics[:2] if str(item).strip()]
            if tic_bits:
                voice_bits.append(f"tics={', '.join(tic_bits)}")
            if not voice_bits:
                voice_bits.append("voice=default")
            hint = self._speaker_response_hint(emotion_family, emotion_level, progress_signal)
            focus_bits: list[str] = []
            if progress_signal.get("inner_conflict") and agent.role == "protagonist":
                focus_bits.append("protagonist: surface hesitation, then choose")
            if progress_signal.get("concrete_risk"):
                focus_bits.append("risk: make the cost concrete, not abstract")
            if focus_bits:
                voice_bits.append(f"focus={'; '.join(focus_bits)}")
            if len(tail_recent) == 2 and tail_recent[-1] == aid and tail_recent[-2] == aid:
                hint += "; avoid another identical follow-up"
            interaction_hint = self._enhance_character_interaction(
                agent=agent,
                emotion_family=emotion_family,
                intensity=emotion_level,
                progress_signal=progress_signal,
                tension_curve=tension_curve,
                recent_speakers=recent_speakers,
            )
            if interaction_hint:
                voice_bits.append(f"interaction={interaction_hint}")
            lines.append(
                f"- {agent.name} ({aid}): emotional posture={emotion_family} ({emotion_level:.2f}); "
                f"{' | '.join(voice_bits)}; likely next move={hint}"
            )
        return "\n".join(lines) if lines else "(none)"

    def _has_explanatory_loop(self, recent_interactions: list[dict]) -> bool:
        recent = recent_interactions[-4:]
        explain_hits = 0
        technical_hits = 0
        for row in recent:
            text = str(row.get("content", ""))
            if re.search(r"(즉|다시 말해|정리하면|요약하면|핵심은|설명하자면|왜냐하면|쉽게 말해|의미는|뜻이었다|해석하면)", text):
                explain_hits += 1
            if (text.count(",") + text.count(";")) >= 2:
                explain_hits += 1
            if len(re.findall(r"(그리고|그러자|그러나|하지만|다만|또한|한편|그래서)", text)) >= 2:
                explain_hits += 1
            if self._technical_term_signature(text):
                technical_hits += 1
        shift_hits = sum(
            1 for row in recent
            if self._has_emotional_or_decisive_shift(str(row.get("content", "")))
        )
        return explain_hits >= 3 and shift_hits == 0 and (technical_hits >= 1 or explain_hits >= 4)

    def _has_repetitive_technical_exchange(self, recent_interactions: list[dict]) -> bool:
        signatures = [
            self._technical_term_signature(str(i.get("content", "")))
            for i in recent_interactions
        ]
        jargon_rich = [sig for sig in signatures if len(sig) >= 2]
        if len(jargon_rich) < 2:
            return False
        overlap_found = any(
            len(jargon_rich[idx] & jargon_rich[idx - 1]) >= 1
            for idx in range(1, len(jargon_rich))
        )
        if not overlap_found:
            return False
        emotion_or_action_shift = any(
            self._has_emotional_or_decisive_shift(str(i.get("content", "")))
            for i in recent_interactions[-3:]
        )
        return not emotion_or_action_shift

    def _recent_jargon_already_onboarded(self, recent_interactions: list[dict]) -> bool:
        recent = recent_interactions[-4:]
        for row in recent:
            text = str(row.get("content", ""))
            if not self._technical_term_signature(text):
                continue
            if re.search(
                r"(즉|쉽게 말해|쉽게 말하면|다시 말해|정리하면|핵심은|의미는|뜻이었다|말이었다|셈이었다)",
                text,
            ):
                return True
        return False

    def _has_repeated_core_concern_exchange(self, recent_interactions: list[dict]) -> bool:
        if self._has_negotiation_condition_loop(recent_interactions):
            return True
        concern_signatures = [
            self._progress_concern_signature(str(i.get("content", "")))
            for i in recent_interactions[-4:]
        ]
        concern_signatures = [sig for sig in concern_signatures if sig]
        if len(concern_signatures) >= 2:
            overlap_pairs = sum(
                1
                for idx in range(1, len(concern_signatures))
                if concern_signatures[idx] & concern_signatures[idx - 1]
            )
            if overlap_pairs >= 2 and not any(
                self._has_emotional_or_decisive_shift(str(i.get("content", "")))
                for i in recent_interactions[-2:]
            ):
                return True
        return self._has_repeated_confrontation_exchange(recent_interactions)

    def _has_negotiation_condition_loop(self, recent_interactions: list[dict]) -> bool:
        window = 3 if self._reader_wants_repeated_confrontation_merge() else 4
        recent = recent_interactions[-4:]
        if len(recent) < window:
            return False
        dialogue_rows = [
            row for row in recent
            if str(row.get("action_type", "")).strip() == "dialogue"
        ]
        if len(dialogue_rows) < min(3, window):
            return False

        concern_signatures: list[set[str]] = []
        negotiation_hits = 0
        concrete_detail_hits = 0
        for row in dialogue_rows:
            text = str(row.get("content", ""))
            sig = self._progress_concern_signature(text)
            if sig:
                concern_signatures.append(sig)
            low = text.lower()
            if re.search(r"(조건|제안|요구|통제권|권한|책임|외부 지원|지원|실시간|real-time|latency|협상|대가)", low):
                negotiation_hits += 1
            if re.search(r"(명함|소속|직함|직책|슬라이드|수치|퍼센트|박수|웅성|배지|모니터|복도 끝|문 앞)", low):
                concrete_detail_hits += 1

        required_hits = 2 if window == 3 else 3
        if len(concern_signatures) < required_hits or negotiation_hits < required_hits:
            return False
        overlap_pairs = sum(
            1
            for idx in range(1, len(concern_signatures))
            if concern_signatures[idx] & concern_signatures[idx - 1]
        )
        if overlap_pairs < (1 if window == 3 else 2) or concrete_detail_hits >= 1:
            return False
        return not any(
            self._has_consequence_shift(str(row.get("content", "")))
            or self._has_emotional_or_decisive_shift(str(row.get("content", "")))
            for row in recent[-2:]
        )

    def _has_repeated_confrontation_exchange(self, recent_interactions: list[dict]) -> bool:
        window = 3 if self._reader_wants_repeated_confrontation_merge() else 4
        recent = recent_interactions[-4:]
        if len(recent) < window:
            return False
        if sum(1 for row in recent if str(row.get("action_type", "")).strip() == "dialogue") < min(3, window):
            return False
        recent_speakers = [
            str(row.get("speaker_id", "")).strip()
            for row in recent
            if str(row.get("speaker_id", "")).strip()
        ]
        if len(set(recent_speakers)) > 2:
            return False
        signatures = [
            self._confrontation_signature(str(row.get("content", "")))
            for row in recent
        ]
        signatures = [sig for sig in signatures if sig]
        if len(signatures) < 2:
            return False
        overlap_pairs = sum(
            1 for idx in range(1, len(signatures))
            if self._confrontation_signatures_overlap(signatures[idx - 1], signatures[idx])
        )
        if overlap_pairs < (1 if window == 3 else 2):
            return False
        return not any(
            self._has_confrontation_resolution_shift(str(row.get("content", "")))
            for row in recent[-2:]
        )

    def _has_overloaded_threat_signal_stack(self, recent_interactions: list[dict]) -> bool:
        if not self.reader_profile.prefers_threat_signal_stack_compression():
            return False
        recent = recent_interactions[-4:]
        if len(recent) < 3:
            return False
        signal_rows = 0
        distinct_signals: set[str] = set()
        consequence_hits = 0
        for row in recent:
            signals = self._threat_signal_signature(str(row.get("content", "")))
            if signals:
                signal_rows += 1
                distinct_signals.update(signals)
            if self._has_consequence_shift(str(row.get("content", ""))):
                consequence_hits += 1
        cap = self._feedback_static_threat_signal_cap(default=1)
        return (
            signal_rows >= max(2, cap + 1)
            and len(distinct_signals) >= 2
            and consequence_hits == 0
        )

    def _has_flat_tension_plateau(self, recent_interactions: list[dict]) -> bool:
        if len(recent_interactions) < 4:
            return False
        recent = recent_interactions[-4:]
        mostly_dialogue = sum(
            1 for i in recent if str(i.get("action_type", "")).strip() == "dialogue"
        ) >= 3
        if not mostly_dialogue:
            return False

        pressure_hits = 0
        relief_hits = 0
        decisive_hits = 0
        tail_fingerprints: list[str] = []
        pressure_fingerprints: list[str] = []
        for row in recent:
            text = str(row.get("content", ""))
            if re.search(r"(긴장|압박|침묵|정적|날카|차갑|굳었|버텼|몰아붙|목소리를 낮추|숨을 죽였)", text):
                pressure_hits += 1
                pressure_fp = self._content_fingerprint(text)
                if pressure_fp:
                    pressure_fingerprints.append(pressure_fp)
            if re.search(r"(웃|미소|숨을 고르|한숨|물컵|잔|메모|의자|어깨를 풀|고개를 끄덕)", text):
                relief_hits += 1
            if re.search(r"(결정|선택|드러났|밝혀졌|확인됐|거절|수락|합의|결론)", text):
                decisive_hits += 1
            fingerprint = self._content_tail_fingerprint(text)
            if fingerprint:
                tail_fingerprints.append(fingerprint)
        tension_cap = self._feedback_tension_phrase_cap(default=2)
        repeated_tail = bool(tail_fingerprints) and (
            len(set(tail_fingerprints)) <= max(1, len(tail_fingerprints) - 2)
        )
        repeated_pressure = len(pressure_fingerprints) >= 2 and len(set(pressure_fingerprints)) <= max(
            1,
            len(pressure_fingerprints) - 1,
        )
        return (
            pressure_hits >= max(2, tension_cap + 1)
            and relief_hits == 0
            and decisive_hits == 0
            and (repeated_tail or repeated_pressure)
        )

    @staticmethod
    def _content_tail_fingerprint(text: str) -> str:
        cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", str(text or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = [tok for tok in cleaned.split() if len(tok) >= 2]
        if not tokens:
            return ""
        return " ".join(tokens[-2:])

    @staticmethod
    def _technical_term_signature(text: str) -> set[str]:
        raw = str(text or "")
        if not raw.strip():
            return set()
        tokens = set(re.findall(r"\b[A-Z]{2,8}(?:-\d+)?\b", raw))
        low = raw.lower()
        english_tokens = {
            token
            for token in re.findall(r"\b[a-z]{4,}(?:-\d+)?\b", low)
            if token not in {"there", "where", "which", "while", "about", "after", "before", "their"}
        }
        tokens.update(english_tokens)
        for term in (
            "latency", "coherence", "drift", "protocol", "fail-safe",
            "보정", "지연", "드리프트", "결맞음", "위상", "알고리즘", "프로토콜",
            "양자", "회로", "파라미터", "오차", "보상",
        ):
            if term in low:
                tokens.add(term)
        return tokens

    @staticmethod
    def _progress_concern_signature(text: str) -> set[str]:
        low = str(text or "").lower()
        if not low.strip():
            return set()
        tokens: set[str] = set()
        if re.search(r"(외부 지원|지원 구조|지원|후원|자원|예산|resource|support|funding)", low):
            tokens.add("support")
        if re.search(r"(실시간|real-time|latency|지연|보정|control loop|제어 루프|compensation)", low):
            tokens.add("realtime")
        if re.search(r"(통제|통제권|권한|주도권|authority|control)", low):
            tokens.add("control")
        if re.search(r"(책임|책임질|감시 한도|oversight|accountability|liability)", low):
            tokens.add("responsibility")
        if re.search(r"(위험|리스크|불안|압박|긴장|대가|후폭풍|부담)", low):
            tokens.add("stakes")
        return tokens

    @staticmethod
    def _confrontation_signature(text: str) -> str:
        low = str(text or "").lower()
        if not low.strip():
            return ""
        if not re.search(r"(다가섰|접근|말을 걸|멈춰 섰|질문|물었|되물었|대답|응답|조건|요구|거절|수락|압박)", low):
            return ""
        parts: list[str] = []
        if re.search(r"(복도|hallway|corridor|문가|문 앞|doorway)", low):
            parts.append("hallway")
        if re.search(r"(밀러|miller|모레노|moreno|수트|보안요원|security)", low):
            parts.append("named")
        concern = DirectorAI._progress_concern_signature(text)
        if concern:
            parts.append("leverage")
        if re.search(r"(질문|물었|되물었|대답|응답)", low):
            parts.append("qa")
        if re.search(r"(다가섰|접근|말을 걸|멈춰 섰)", low):
            parts.append("approach")
        if re.search(r"(조건|요구|거절|수락)", low):
            parts.append("terms")
        return "|".join(parts)

    @staticmethod
    def _confrontation_signatures_overlap(left: str, right: str) -> bool:
        left_tokens = {token for token in str(left or "").split("|") if token}
        right_tokens = {token for token in str(right or "").split("|") if token}
        if not left_tokens or not right_tokens:
            return False
        shared = left_tokens & right_tokens
        if len(shared) >= 2:
            return True
        if "qa" in shared and (left_tokens | right_tokens) & {"hallway", "named", "leverage", "approach", "terms"}:
            return True
        return False

    @staticmethod
    def _threat_signal_signature(text: str) -> set[str]:
        low = str(text or "").lower()
        if not low.strip():
            return set()
        signals: set[str] = set()
        if re.search(r"(메모|메모장|문서|서류|봉투|memo|document)", low):
            signals.add("document")
        if re.search(r"(경고음|경보|알람|비프|모니터|alert|alarm|warning|monitor)", low):
            signals.add("alert")
        if re.search(r"(보안요원|경호원|감시|watch|stare|gaze|시선)", low):
            signals.add("watcher")
        return signals

    @staticmethod
    def _has_consequence_shift(text: str) -> bool:
        return bool(re.search(
            r"(결정|선택|다가섰|다가갔|걸음을 옮|문으로 향|자료를 건넸|질문|반박|거절|수락|말을 걸|자리를 정리|고개를 끄덕)",
            str(text or ""),
        ))

    @staticmethod
    def _has_confrontation_resolution_shift(text: str) -> bool:
        return bool(re.search(
            r"(결정|선택|걸음을 옮|문으로 향|자료를 건넸|반박|거절|수락|자리를 정리|고개를 끄덕|돌아서|떠났|끝냈)",
            str(text or ""),
        ))

    @staticmethod
    def _has_emotional_or_decisive_shift(text: str) -> bool:
        return bool(re.search(
            r"(숨을 골랐|시선을|표정|떨|움찔|웃|멈칫|고개를|침묵|결정|선택|거절|수락|질문|반박|망설|불안|안도|초조|긴장)",
            str(text or ""),
        ))

    @staticmethod
    def _content_fingerprint(text: str) -> str:
        cleaned = re.sub(r"\[[^\]]*\]", " ", str(text or "").lower())
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        stop = {"그리고", "하지만", "그러나", "그는", "그녀는", "정말", "아주", "that", "with"}
        tokens = [t for t in cleaned.split() if len(t) >= 2 and t not in stop]
        return " ".join(tokens[:8])

    @staticmethod
    def _opening_phrase_fingerprint(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not cleaned:
            return ""
        cleaned = re.split(r"[.!?…]", cleaned, maxsplit=1)[0]
        cleaned = re.sub(r"^[\"“”'‘’\(\)\[\]\s]+", "", cleaned)
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        return " ".join(tokens[:4])

    @staticmethod
    def _has_scene_exit_cue(recent_interactions: list[dict]) -> bool:
        if not recent_interactions:
            return False
        tail = " ".join(str(i.get("content", "")) for i in recent_interactions[-2:])
        return bool(re.search(
            r"(고개를 끄덕|침묵이 흘렀|말을 멈췄|자리에서 일어|대화를 마무리|회의는 끝|문으로 향했|돌아서|숨을 골랐)",
            tail,
        ))

    def _conditional_fallback_cast_size(self, agents: list[Agent], world: WorldState) -> int:
        if not isinstance(self.episode_config, dict):
            return 1

        # Explicit opt-in from episode config wins if provided.
        try:
            explicit = self.episode_config.get("fallback_cast_size")
            if explicit is not None:
                return max(1, min(2, int(explicit)))
        except (TypeError, ValueError):
            pass

        # Default: keep monologue possible.
        fallback_size = 1

        episode_id = str(self.episode_config.get("id", ""))
        ep_num_match = re.match(r"^ep(\d+)_", episode_id)
        ep_num = int(ep_num_match.group(1)) if ep_num_match else 0
        max_turns = int(self.episode_config.get("max_turns", 0) or 0)

        # Heuristic 1: non-opening episodes are often continuity-sensitive.
        continuity_sensitive = ep_num >= 2

        # Heuristic 2: dialogue/meeting/briefing episodes should avoid too many
        # accidental solo scenes when cast extraction fails.
        text_blob = " ".join(
            str(self.episode_config.get(k, "") or "")
            for k in ("summary", "scene", "location")
        ).lower()
        dialogue_keywords = (
            "meeting", "briefing", "interview", "conversation", "call", "contact",
            "회의", "브리핑", "면담", "대화", "통화", "접촉",
        )
        dialogue_centric = any(k in text_blob for k in dialogue_keywords)

        # Heuristic 3: if episode config lists multiple likely participants,
        # prefer two-person fallback for continuity.
        cfg_chars = self.episode_config.get("characters")
        has_multi_character_hint = isinstance(cfg_chars, list) and len(cfg_chars) >= 2

        if continuity_sensitive and (dialogue_centric or has_multi_character_hint or max_turns >= 14):
            fallback_size = 2

        # Never exceed 2 in the new fixed policy; solo remains possible.
        return max(1, min(2, fallback_size))

    def _safe_llm_call(
        self,
        messages: list[dict],
        purpose: str = "director",
        use_premium: bool = False,
        max_tokens: int = 400,
    ) -> str:
        try:
            return self.llm.chat(
                messages=messages,
                use_premium=use_premium,
                purpose=purpose,
                temperature=0.3,    # Low temp for deterministic checks
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Director LLM call failed (%s): %s", purpose, exc)
            return "{}"

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Safely parse JSON from LLM response."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try extracting first {...} block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _extract_episode_character_ids(raw_characters) -> list[str]:
        if not isinstance(raw_characters, list):
            return []

        ids: list[str] = []
        for entry in raw_characters:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("id"), str):
                ids.append(entry["id"])
        return ids

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    def _select_cast_by_explicit_mentions(self, agents: list[Agent]) -> list[str]:
        """
        Select cast strictly from explicit textual evidence in episode content.
        """
        episode_text = self._episode_text_for_cast_selection()
        if not episode_text.strip():
            return []

        candidate_text = "\n".join(
            f"- id={a.id} | name={a.name} | role={a.role} | aliases={', '.join(self._agent_name_variants(a.name))}"
            for a in agents
        )
        prompt = (
            f"You are a strict cast linker.\n\n"
            f"Episode text:\n\"\"\"\n{episode_text}\n\"\"\"\n\n"
            f"Candidate characters:\n{candidate_text}\n\n"
            f"Task:\n"
            f"1) Return ONLY character IDs that are explicitly mentioned in the episode text.\n"
            f"2) DO NOT infer by role, world knowledge, or likely involvement.\n"
            f"3) For each selected ID, include an exact evidence substring copied from the episode text.\n\n"
            f"Reply JSON only:\n"
            f"{{\"mentions\": [{{\"agent_id\": \"...\", \"evidence\": \"exact substring\"}}], "
            f"\"reason\": \"short reason\"}}"
        )
        result = self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_cast_strict_mentions",
            use_premium=True,
            max_tokens=500,
        )
        parsed = self._parse_json(result)
        mentions = parsed.get("mentions", [])
        if not isinstance(mentions, list):
            mentions = []

        candidate_ids = {a.id for a in agents}
        selected: list[str] = []
        evidence_log: list[dict] = []
        for item in mentions:
            if not isinstance(item, dict):
                continue
            agent_id = item.get("agent_id")
            evidence = item.get("evidence", "")
            if not isinstance(agent_id, str) or agent_id not in candidate_ids:
                continue
            if not isinstance(evidence, str) or not evidence.strip():
                continue
            # Hard validation: the evidence string must appear in episode text.
            if evidence.strip() not in episode_text:
                continue
            selected.append(agent_id)
            evidence_log.append({"agent_id": agent_id, "evidence": evidence.strip()})

        selected = self._dedupe_preserve_order(selected)
        if evidence_log:
            self._log(
                "cast_evidence",
                "director",
                f"Validated cast evidence for {len(evidence_log)} mentions",
                {"mentions": evidence_log},
            )
        return selected

    def _episode_text_for_cast_selection(self) -> str:
        summary = str(self.episode_config.get("summary", "")).strip()
        location = str(self.episode_config.get("location", "")).strip()
        clues = self.episode_config.get("introduced_clues", [])
        clue_lines = []
        if isinstance(clues, list):
            clue_lines = [
                str(c.get("content", "")).strip()
                for c in clues
                if isinstance(c, dict) and str(c.get("content", "")).strip()
            ]

        current = self.storyline_context.get("current") or {}
        milestone_text = str(current.get("description", "")).strip()

        parts = []
        if location:
            parts.append(f"[Location] {location}")
        if summary:
            parts.append(f"[Summary] {summary}")
        if milestone_text:
            parts.append(f"[Milestone] {milestone_text}")
        if clue_lines:
            parts.append("[Clues]\n" + "\n".join(f"- {line}" for line in clue_lines))
        return "\n\n".join(parts)

    def _build_storyline_context(self) -> dict:
        """
        Build an index around the current episode within storyline milestones.
        Includes story arc information for pacing and emotional trajectory guidance.
        """
        acts = self.storyline.get("acts", [])
        if not isinstance(acts, list) or not acts:
            return {}

        milestones = self._flatten_storyline_milestones(acts)
        if not milestones:
            return {}

        current_idx = self._find_storyline_milestone_index(milestones)

        # Extract story arcs information
        story_arcs = self.storyline.get("story_arcs", {})
        arc_info = self._determine_current_arc(story_arcs, current_idx, len(milestones))

        ctx = {
            "title": self.storyline.get("title", ""),
            "all": milestones,
            "story_arc": arc_info,  # New: current arc context
        }
        if current_idx is None:
            return ctx

        prev_item = milestones[current_idx - 1] if current_idx > 0 else None
        next_items = milestones[current_idx + 1: current_idx + 3]
        ctx.update(
            {
                "current_index": current_idx,
                "current": milestones[current_idx],
                "previous": prev_item,
                "next": next_items,
            }
        )
        return ctx

    def _determine_current_arc(
        self,
        story_arcs: dict,
        current_idx: Optional[int],
        total_milestones: int
    ) -> dict:
        """
        Determine which story arc the current episode belongs to.

        Story arcs define the 6-act structure:
        - Setup (3 episodes)
        - Discovery (11 episodes)
        - Technical (16 episodes)
        - Crisis (14 episodes)
        - Climax (3 episodes)
        - Resolution (2 episodes)
        """
        if not story_arcs or current_idx is None:
            return {}

        # Calculate cumulative episode counts
        arc_order = ["setup", "discovery", "technical", "crisis", "climax", "resolution"]
        cumulative = 0
        episode_position = current_idx + 1  # 1-indexed for human readability

        for arc_name in arc_order:
            arc_data = story_arcs.get(arc_name, {})
            arc_episodes = arc_data.get("episodes", 0)

            if episode_position <= cumulative + arc_episodes:
                # Found the current arc
                position_in_arc = episode_position - cumulative
                progress_pct = (position_in_arc / arc_episodes) * 100 if arc_episodes > 0 else 0

                return {
                    "name": arc_name.upper(),
                    "act_position": arc_data.get("arc_position", ""),
                    "description": arc_data.get("description", ""),
                    "emotional_trajectory": arc_data.get("emotional_trajectory", ""),
                    "key_reveals": arc_data.get("key_reveals", []),
                    "episode_in_arc": position_in_arc,
                    "total_in_arc": arc_episodes,
                    "progress_percentage": round(progress_pct, 1),
                    "is_arc_opening": position_in_arc <= 2,
                    "is_arc_climax": position_in_arc >= arc_episodes - 1,
                }

            cumulative += arc_episodes

        return {}

    def _find_storyline_milestone_index(self, milestones: list[dict]) -> Optional[int]:
        episode_id = str(self.episode_config.get("id", "")).strip()
        source_id = str(self.episode_config.get("storyline_source_id", "")).strip()

        for idx, milestone in enumerate(milestones):
            mid = str(milestone.get("id", "")).strip()
            if not mid:
                continue
            if source_id and self._normalize_key(mid) == self._normalize_key(source_id):
                return idx
            if episode_id and self._normalize_key(mid) == self._normalize_key(episode_id):
                return idx

        if episode_id:
            ep_slug = self._episode_slug(episode_id)
            for idx, milestone in enumerate(milestones):
                mid = str(milestone.get("id", "")).strip()
                if self._episode_slug(mid) == ep_slug:
                    return idx
        return None

    @staticmethod
    def _flatten_storyline_milestones(acts: list[dict]) -> list[dict]:
        flattened: list[dict] = []
        for act in acts:
            if not isinstance(act, dict):
                continue
            act_id = str(act.get("id", "")).strip()
            act_title = str(act.get("title", "")).strip()
            milestones = act.get("milestones", [])
            if not isinstance(milestones, list):
                continue
            for milestone in milestones:
                if not isinstance(milestone, dict):
                    continue
                flattened.append(
                    {
                        "id": str(milestone.get("id", "")).strip(),
                        "description": str(milestone.get("description", "")).strip(),
                        "act_id": act_id,
                        "act_title": act_title,
                    }
                )
        return flattened

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")

    @staticmethod
    def _episode_slug(value: str) -> str:
        key = DirectorAI._normalize_key(value)
        return re.sub(r"^ep\d+_", "", key)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if not text:
            return ""
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def _detect_unplanned_character_entries(
        self,
        proposed_action: str,
        active_ids: set[str],
        agents: list[Agent],
    ) -> list[str]:
        """
        Detect obvious off-scene character entry attempts in a generated turn.
        """
        action_text = self._extract_structured_field(proposed_action, "ACTION")
        text = (action_text or proposed_action or "").lower()
        if not text:
            return []

        entry_cues = [
            "enters", "walks in", "arrives", "shows up", "joins", "steps in",
            "comes in", "appears", "pulls up", "leans in", "sits down",
            "들어오", "등장", "나타나", "합류", "다가오", "말을 건",
        ]
        cue_pattern = "|".join(re.escape(cue) for cue in entry_cues)
        if not cue_pattern:
            return []

        offenders: list[str] = []
        for other in agents:
            if other.id in active_ids:
                continue

            for variant in self._agent_name_variants(other.name):
                escaped = re.escape(variant.lower())
                pattern = rf"(?:{escaped}).{{0,40}}(?:{cue_pattern})|(?:{cue_pattern}).{{0,40}}(?:{escaped})"
                if re.search(pattern, text, re.DOTALL):
                    offenders.append(other.name)
                    break
        return self._dedupe_preserve_order(offenders)

    def _detect_first_meeting_drift_for_known_relation(
        self,
        agent: Agent,
        proposed_action: str,
        active_ids: set[str],
        agents: list[Agent],
    ) -> list[str]:
        """
        Detect first-meeting rituals against characters the speaker already
        appears to know from background text.
        """
        text = (proposed_action or "").lower()
        if not text:
            return []

        first_meeting_cues = [
            "nice to meet you",
            "first time meeting",
            "let me introduce myself",
            "my name is",
            "business card",
            "명함",
            "처음 뵙",
            "처음 만나",
            "자기소개",
            "소개드리",
            "성함이",
        ]
        if not any(cue in text for cue in first_meeting_cues):
            return []

        offenders: list[str] = []
        for other in agents:
            if other.id == agent.id or other.id not in active_ids:
                continue
            if not self._has_prior_relationship_signal(agent, other):
                continue

            variants = self._agent_name_variants(other.name)
            if any(v.lower() in text for v in variants if v):
                offenders.append(other.name)

        return self._dedupe_preserve_order(offenders)

    def _has_prior_relationship_signal(self, agent: Agent, other: Agent) -> bool:
        """
        Heuristic: treat as known relation if one appears in the other's
        profile text or there is explicit initial relationship wiring.
        """
        if other.id in agent.memory.relationship_matrix:
            return True
        if agent.id in other.memory.relationship_matrix:
            return True

        haystack = " ".join(
            [
                str(agent.bio or ""),
                " ".join(agent.invariants or []),
                " ".join(agent.goals or []),
                str(other.bio or ""),
                " ".join(other.invariants or []),
                " ".join(other.goals or []),
            ]
        ).lower()
        for variant in self._agent_name_variants(other.name):
            if variant and variant.lower() in haystack:
                return True
        for variant in self._agent_name_variants(agent.name):
            if variant and variant.lower() in haystack:
                return True
        return False

    @staticmethod
    def _extract_structured_field(text: str, field: str) -> str:
        pattern = rf"^{field}:\s*(.+?)(?=\n[A-Z]+:|$)"
        match = re.search(pattern, text or "", re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _agent_name_variants(name: str) -> list[str]:
        variants = [name.strip()]
        tokens = [t.strip("()") for t in re.split(r"\s+", name) if t.strip()]
        if len(tokens) >= 2:
            variants.extend([tokens[0], tokens[-1]])
        elif tokens:
            variants.append(tokens[0])
        return [v for v in DirectorAI._dedupe_preserve_order(variants) if v]

    def _log(self, event_type: str, agent_id: str, message: str, details: dict) -> None:
        entry = {
            "event_type": event_type,
            "agent_id": agent_id,
            "message": message,
            "details": details,
        }
        self.debug_log.append(entry)
        if event_type == "resolution_check":
            logger.debug("[Director] %s | %s | %s", event_type, agent_id, message)
        else:
            logger.info("[Director] %s | %s | %s", event_type, agent_id, message)
