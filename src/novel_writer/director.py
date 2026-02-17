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
    ) -> None:
        self.episode_config = episode_config
        self.world_facts = world_facts
        self.clue_manager = clue_manager
        self.storyline = storyline or {}
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
        prompt = (
            f"You are a story director. A required story clue hasn't surfaced naturally.\n\n"
            f"Current scene: {world.current_scene}\n"
            f"Location: {world.location}\n"
            f"Clue to surface: {clue_content}\n"
            f"Suggested trigger: {trigger}\n"
            f"Method: {method}\n\n"
            f"Write a brief (1-3 sentence) in-world event or observation that naturally "
            f"introduces this clue without being too on-the-nose. "
            f"Write it as a scene narration, not as dialogue."
        )
        return self._safe_llm_call(
            [{"role": "user", "content": prompt}],
            purpose="director_clue_injection",
            use_premium=True,
        )

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
        recent_text = "\n".join(
            (
                f"- T{i.get('turn', '?')} | {i.get('speaker_name', '?')} "
                f"[{i.get('action_type', 'dialogue')}] "
                f"(mode={i.get('metadata', {}).get('turn_mode', 'n/a')}): "
                f"{self._truncate(str(i.get('content', '')), 180)}"
            )
            for i in recent[-8:]
        ) or "(none)"

        active_text = "\n".join(
            f"- {aid}: {agent_map[aid].name}" for aid in active_ids
        )

        prompt = (
            f"You are a story turn allocator.\n\n"
            f"Turn: {turn}\n"
            f"Episode summary: {self._truncate(str(self.episode_config.get('summary', '')), 500)}\n"
            f"Current location: {self.episode_config.get('location', world.location)}\n"
            f"Active cast (ONLY choose from these IDs):\n{active_text}\n\n"
            f"Recent interactions:\n{recent_text}\n\n"
            f"Task:\n"
            f"1) Choose ONE next speaker from active cast IDs.\n"
            f"2) You may set end_scene=true only if the exchange naturally closed.\n"
            f"3) Do NOT force ping-pong dialogue. If one person is lecturing and others are listening, "
            f"it is valid to keep the lecturer as next speaker repeatedly.\n"
            f"4) If end_scene=true, still provide speaker_id for who should continue after scene closure.\n\n"
            f"5) Balance speaking opportunities: avoid choosing the same speaker 3+ turns in a row "
            f"when other active speakers are available.\n\n"
            f"Reply JSON only:\n"
            f"{{\"speaker_id\": \"agent_id\", \"end_scene\": true/false, \"reason\": \"...\"}}"
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
        # when explicit evidence is absent, keep the cast minimal.
        fallback = [protagonist.id] if protagonist else [agents[0].id]
        self._log(
            "cast_selection",
            "director",
            "No explicit cast evidence found; using minimal fallback cast",
            {
                "active_agents": fallback,
                "source": "strict_fallback_minimal",
            },
        )
        return fallback

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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
