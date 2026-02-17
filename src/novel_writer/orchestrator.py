"""
Simulation Orchestrator for the AI Story Simulation Engine.

Manages the turn-based episode loop:
  1. Determine active agent
  2. Build context (world state + filtered memory)
  3. Generate agent action via LLM
  4. Director evaluation (invariant / knowledge / clue checks)
  5. Apply to world state
  6. Update all agent memories
  7. Persist to database
  8. Check completion criteria
"""

from __future__ import annotations
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Optional

from .models import Agent, WorldState, ClueManager, Interaction, Memory, SteeringContext
from .director import DirectorAI
from .llm_client import LLMClient
from . import database as db

logger = logging.getLogger(__name__)

MAX_REGENERATION_ATTEMPTS = 3


class SimulationOrchestrator:
    """
    Runs a full episode simulation.

    Parameters
    ----------
    agents       : list of Agent objects (loaded from characters YAML)
    director     : DirectorAI instance
    world        : WorldState instance
    llm          : LLMClient instance
    episode_id   : unique episode identifier
    episode_config : parsed episode YAML
    """

    def __init__(
        self,
        agents: list[Agent],
        director: DirectorAI,
        world: WorldState,
        llm: LLMClient,
        episode_id: str,
        episode_config: dict,
        steering_contexts: Optional[dict[str, SteeringContext]] = None,
    ) -> None:
        self.agents       = agents
        self.agent_map    = {a.id: a for a in agents}
        self._agent_reference_index = self._build_agent_reference_index()
        self.director     = director
        self.world        = world
        self.llm          = llm
        self.episode_id   = episode_id
        self.episode_config = episode_config
        self.steering_contexts = steering_contexts or {}

        self.turn         = 0
        self.max_turns    = episode_config.get("max_turns", 60)
        self.interactions: list[Interaction] = []
        self._agent_cycle_index = 0
        self._agent_agendas: dict[str, str] = {}   # agent_id -> last AGENDA text
        # If an agent chooses to only observe/listen, temporarily deprioritise
        # them so another speaker can carry the scene naturally.
        self._agent_skip_until_turn: dict[str, int] = {}
        self._loop_guard_window: int = 6           # check last N turns for repetition
        self._loop_guard_threshold: int = 3        # fire after K similar turns
        self._loop_guard_fired: bool = False       # prevent double-fire per window
        self._loop_guard_cooldown_turn: int = 0    # turn when guard last fired

        # Determine protagonist (used for perspective filtering in novel gen)
        self.protagonist_id: Optional[str] = None
        for a in agents:
            if a.role == "protagonist":
                self.protagonist_id = a.id
                break

    # ------------------------------------------------------------------ #
    # Public: Run Episode
    # ------------------------------------------------------------------ #

    def run_episode(self) -> list[dict]:
        """
        Execute the full episode. Returns list of all interactions as dicts.
        """
        logger.info("Starting episode %s with %d agents", self.episode_id, len(self.agents))

        selected_cast = self.director.select_active_agents(self.agents, self.world)
        if selected_cast:
            self.world.active_agents = selected_cast
            self._agent_cycle_index = 0
            selected_names = [
                self.agent_map[aid].name for aid in selected_cast if aid in self.agent_map
            ]
            logger.info("Active cast for episode: %s", ", ".join(selected_names))

        db.upsert_episode(
            self.episode_id,
            self.episode_config,
            status="running",
            start_time=datetime.utcnow().isoformat(),
        )

        # Persist initial agent records
        for agent in self.agents:
            db.upsert_agent(agent)

        # Initial world state snapshot
        db.save_world_state(self.episode_id, 0, self.world.to_dict())
        # Persist carried-over emotional state at turn 0 so cross-episode
        # continuity is visible directly in DB plots.
        for agent in self.agents:
            for emotion, intensity in agent.memory.emotional_state.items():
                if intensity > 0:
                    db.save_emotion(
                        agent.id,
                        self.episode_id,
                        0,
                        emotion,
                        intensity,
                        None,
                    )

        while self.turn < self.max_turns:
            self.turn += 1
            self.world.turn = self.turn
            logger.info("─── Turn %d / %d ───", self.turn, self.max_turns)

            # Reset loop guard cooldown after 3 turns
            if (self._loop_guard_fired
                    and self.turn - self._loop_guard_cooldown_turn >= 3):
                self._loop_guard_fired = False

            # Check for director clue injection
            injection = self.director.should_inject_clue(self.turn, self.world)
            if injection:
                self._apply_injection(injection)

            # Run agent turn
            self._run_turn()

            # Loop guard: detect repetition and force scene transition
            if self._detect_repetition_loop():
                self._force_scene_transition()

            # Persist world state every 5 turns
            if self.turn % 5 == 0:
                db.save_world_state(self.episode_id, self.turn, self.world.to_dict())

            # Check completion
            if self._check_completion():
                logger.info("Episode %s complete at turn %d", self.episode_id, self.turn)
                break

        db.update_episode_status(
            self.episode_id, "complete",
            end_time=datetime.utcnow().isoformat()
        )
        logger.info(
            "Episode finished. %d interactions logged. Budget: %s",
            len(self.interactions),
            self.llm.budget_summary(),
        )
        return [i.to_dict() for i in self.interactions]

    # ------------------------------------------------------------------ #
    # Single Turn
    # ------------------------------------------------------------------ #

    def _run_turn(self) -> None:
        agent = self._next_agent()
        context = self._build_context(agent)

        proposed_action, approved = self._generate_and_validate(agent, context)

        if not approved:
            logger.warning("Turn %d: Agent %s action not approved after retries",
                           self.turn, agent.id)
            return

        # Apply to world
        self.world.update({"content": proposed_action})

        # Parse structured response
        text, emotions, relationship_deltas, clue_references, turn_mode, exit_scene = \
            self._parse_agent_response(proposed_action, agent)

        action_type = "dialogue"
        if turn_mode == "monologue":
            action_type = "inner_thought"
        elif turn_mode in ("observe", "action"):
            action_type = "action"

        # Create interaction record
        interaction = Interaction(
            id=str(uuid.uuid4()),
            episode_id=self.episode_id,
            turn=self.turn,
            speaker_id=agent.id,
            speaker_name=agent.name,
            content=text,
            action_type=action_type,
            timestamp=datetime.utcnow(),
            metadata={
                "emotions": emotions,
                "relationship_deltas": relationship_deltas,
                "agenda": self._agent_agendas.get(agent.id, ""),
                "turn_mode": turn_mode,
                "exit_scene": exit_scene,
            },
        )
        self.interactions.append(interaction)
        db.save_interaction(interaction)
        logger.info("[Turn %d] %s: %s",
                    self.turn, agent.name, self._preview_text(text))

        # Update emotions (EMA smoothing + decay unmentioned)
        mentioned_emotions = set(emotions.keys())
        for emotion, intensity in emotions.items():
            agent.memory.record_emotion(emotion, intensity, self.turn)
        agent.memory.decay_unmentioned_emotions(mentioned_emotions, self.turn)
        # Persist all current emotional state
        for emotion, intensity in agent.memory.emotional_state.items():
            if intensity > 0:
                db.save_emotion(agent.id, self.episode_id, self.turn,
                                emotion, intensity, interaction.id)

        # Update relationships
        for other_id, delta in relationship_deltas.items():
            agent.memory.update_relationship(other_id, delta)
            new_val = agent.memory.get_relationship(other_id)
            db.save_relationship(agent.id, other_id, new_val,
                                 self.episode_id, self.turn)

        # Handle clue discoveries
        for clue_ref in clue_references:
            self._handle_clue_discovery(agent, clue_ref, interaction.id)

        # Observation/listening turns should not force back-and-forth dialogue.
        # Skip this agent's next speaking opportunity once.
        if turn_mode == "observe" and len(self.world.active_agents) > 1:
            self._agent_skip_until_turn[agent.id] = self.turn + 2
        else:
            self._agent_skip_until_turn.pop(agent.id, None)

        # Optional scene exit lets conversations end naturally and enables solo POV turns.
        if exit_scene and len(self.world.active_agents) > 1:
            self.world.active_agents = [aid for aid in self.world.active_agents if aid != agent.id]
            logger.info(
                "Turn %d: %s exits the current scene. Active cast now: %s",
                self.turn,
                agent.name,
                ", ".join(
                    self.agent_map[aid].name for aid in self.world.active_agents
                    if aid in self.agent_map
                ) or "(none)",
            )

        # Update memories of ALL agents (with perspective filters)
        self._propagate_memory(interaction, agent)

        # Persona evolution check (every 10 turns or on key events)
        if self.turn % 10 == 0 or emotions.get("shock", 0) > 0.7:
            self._evolve_persona(agent, interaction)

    # ------------------------------------------------------------------ #
    # Context Building
    # ------------------------------------------------------------------ #

    def _build_context(self, agent: Agent) -> dict:
        """Build the full context payload for an agent's LLM call."""
        world_ctx = self.world.get_context_for_agent(agent.id)

        # Recent interactions (last 8)
        recent = [i.to_dict() for i in self.interactions[-8:]]

        # Known clues
        known_clue_ids = list(agent.memory.known_clues)

        # Relationships summary
        rel_summary = {
            other_name: round(val, 2)
            for other_id, val in agent.memory.relationship_matrix.items()
            if (other_name := self.agent_map.get(other_id, type("X", (), {"name": other_id})()).name)
        }

        # Goals
        goals_text = "\n".join(f"- {g}" for g in agent.goals)

        return {
            "agent":       agent,
            "world":       world_ctx,
            "recent":      recent,
            "known_clues": known_clue_ids,
            "relations":   rel_summary,
            "goals":       goals_text,
            "pacing_hint": self.director.get_pacing_hint(self.turn, recent),
            "storyline_hint": self.director.get_storyline_guidance(),
            "steering": self.steering_contexts.get(agent.id),
        }

    def _build_episode_context(self) -> str:
        """Build episode-level context string for agent prompts (tone, pacing, setting)."""
        ep = self.episode_config
        ep_id = ep.get("id", "")
        pacing = ep.get("pacing", "normal")
        location = ep.get("location", "")
        summary = ep.get("summary", "")

        # Extract episode number from id (e.g., "ep01_academic_presentation" → 1)
        ep_num = 0
        ep_id_str = str(ep_id)
        for part in ep_id_str.split("_"):
            digits = "".join(c for c in part if c.isdigit())
            if digits:
                ep_num = int(digits)
                break

        total_episodes = 49  # Total planned episodes

        # Map pacing to tone guidance
        pacing_guidance = {
            "slow": "Take your time. Focus on observation, introspection, and subtle detail. "
                    "Avoid rushing into dramatic action or conflict.",
            "normal": "Balance dialogue and observation naturally. "
                      "Let tension build through subtext, not explosive action.",
            "tense": "Heightened alertness. Short, sharp exchanges. "
                     "Every word carries weight. Internal conflict is palpable.",
            "fast": "Events unfold rapidly. Decisions are forced. "
                    "There is no time for reflection — only reaction.",
        }
        tone = pacing_guidance.get(pacing, pacing_guidance["normal"])

        # Position in story arc
        if ep_num <= 4:
            arc_phase = "Early introduction — establishing characters, setting, and initial mysteries."
        elif ep_num <= 15:
            arc_phase = "Building tension — alliances form, secrets deepen, stakes become personal."
        elif ep_num <= 24:
            arc_phase = "Midpoint complexity — loyalties tested, betrayals surface, no easy answers."
        elif ep_num <= 34:
            arc_phase = "Escalation — weapons emerge, power plays intensify, lines are crossed."
        elif ep_num <= 38:
            arc_phase = "Approaching climax — critical decisions, irreversible consequences."
        else:
            arc_phase = "Endgame — final confrontations, revelations, and lasting consequences."

        # First line of summary (truncated)
        summary_line = summary.strip().split("\n")[0][:200] if summary else ""

        lines = [
            f"Episode {ep_num} of {total_episodes} ({arc_phase})",
            f"Setting: {location}" if location else "",
            f"Scene: {summary_line}" if summary_line else "",
            f"Pacing: {pacing} — {tone}",
            "",
            "IMPORTANT: Stay grounded in the current setting. "
            "Do not introduce spy-thriller elements, encrypted devices, or dramatic "
            "action unless they are explicitly part of the scene. "
            "Your character is a real person in a real situation.",
        ]
        return "\n".join(line for line in lines if line or line == "")

    def _build_agent_prompt(self, agent: Agent, context: dict) -> tuple[str, list[dict]]:
        """Build system prompt and messages for agent LLM call."""
        emotions_text = json.dumps(agent.memory.emotional_state, indent=2)

        active_ids = set(self.world.active_agents)
        others = [a for a in self.agents if a.id in active_ids and a.id != agent.id]
        cast_text = "\n".join(
            f"- {a.name} (relationship: {context['relations'].get(a.name, 0.0):+.2f})"
            for a in others
        ) or "- (no other characters currently present)"
        active_cast_names = ", ".join(
            a.name for a in self.agents if a.id in active_ids
        ) or "(none)"

        # Episode context for tone/pacing control
        ep_ctx = self._build_episode_context()

        system = (
            f"You are {agent.name}, a character in a story.\n\n"
            f"## Story Context\n{ep_ctx}\n\n"
            f"## Your Background\n{agent.bio}\n\n"
            f"## Your Core Rules (NEVER violate these)\n"
            + "\n".join(f"- {inv}" for inv in agent.invariants) +
            f"\n\n## Your Current Goals\n{context['goals']}\n\n"
            f"## Other Characters Present\n{cast_text}\n\n"
            f"## Your Current Emotional State\n{emotions_text}\n\n"
            f"Stay in character at all times. Write your next action or dialogue.\n"
            f"Be specific and grounded in the current scene.\n"
            f"Match the tone and pacing described in Story Context.\n"
            f"Write all content in Korean.\n"
            f"Keep output concise and concrete: avoid repeating nearly identical actions.\n"
            f"Do not introduce new in-scene characters or event or place.\n"
            f"If you already know another present character from your background/history, "
            f"do NOT behave like a first meeting (no self-introduction ritual, no "
            f"'nice to meet you' framing, no unnecessary business-card exchange).\n"
        )
        if context.get("pacing_hint"):
            system += f"\n## Story Pacing\n{context['pacing_hint']}\n"
        if context.get("storyline_hint"):
            system += (
                f"\n## Storyline Guardrail\n{context['storyline_hint']}\n"
                f"Keep this turn aligned with the current milestone.\n"
            )

        steering = context.get("steering")
        if steering and isinstance(steering, SteeringContext):
            if steering.tactical_goals:
                tg_text = "\n".join(f"- {g}" for g in steering.tactical_goals)
                system += (
                    f"\n## Tactical Objectives (Attempt {steering.attempt_number})\n"
                    f"These are your PRIORITY actions for this scene:\n{tg_text}\n"
                )
            if steering.steering_prompt:
                system += (
                    f"\n## Director Guidance\n{steering.steering_prompt}\n"
                )
            if steering.exemplar_actions:
                examples_text = "\n---\n".join(steering.exemplar_actions[:3])
                system += (
                    f"\n## Successful Action Examples\n"
                    f"Here are examples of effective actions from similar scenes:\n"
                    f"{examples_text}\n"
                    f"Use these as inspiration for the type and quality of "
                    f"response expected.\n"
                )

        world = context["world"]
        recent_text = "\n".join(
            f"[{i['speaker_name']}]: {i['content']}"
            for i in context["recent"]
        )

        user_msg = (
            f"## Current Scene\n{world['scene']}\n"
            f"Location: {world['location']} | Time: {world['time']}\n\n"
            f"Active cast in this scene: {active_cast_names}\n\n"
            f"## Recent Events\n{recent_text or '(start of scene)'}\n\n"
        )

        # Inject previous turn's AGENDA if available
        prev_agenda = self._agent_agendas.get(agent.id, "")
        if prev_agenda:
            user_msg += (
                f"## Your Previous Intention\n"
                f"Last turn you planned: {prev_agenda}\n"
                f"Continue from this intention or adapt based on what happened.\n\n"
            )

        user_msg += (
            f"## Your Turn\n"
            f"What do you say or do next? Respond as {agent.name}.\n"
            f"You may choose to stay silent, simply observe, or leave the scene if natural.\n\n"
            f"Format your response as:\n"
            f"TURN_MODE: [dialogue | observe | monologue | action]\n"
            f"ACTION: [brief description of physical action, if any]\n"
            f"DIALOGUE: [what you say, in quotes, or (silent) if not speaking]\n"
            f"INNER: [one sentence of internal thought]\n"
            f"EMOTION: [JSON dict of emotions like {{\"tension\": 0.7, \"curiosity\": 0.4}}]\n"
            f"RELATIONSHIPS: [JSON dict of relationship changes like {{\"other_agent_id\": 0.05}}]\n"
            f"CLUES: [comma-separated clue IDs discovered this turn, or (none)]\n"
            f"EXIT_SCENE: [yes/no — use yes only if this character naturally leaves]\n"
            f"Do not invent named events/meetings/sessions not present in Current Scene or Recent Events.\n"
            f"AGENDA: [1-2 sentence plan for what you intend to do or explore next turn]\n"
            f"Keep ACTION + DIALOGUE + INNER together under ~90 Korean words total.\n"
        )

        return system, [{"role": "user", "content": user_msg}]

    # ------------------------------------------------------------------ #
    # Generation + Validation
    # ------------------------------------------------------------------ #

    def _generate_and_validate(
        self, agent: Agent, context: dict
    ) -> tuple[str, bool]:
        """
        Generate agent response and run Director validation.
        Retries up to MAX_REGENERATION_ATTEMPTS times on failure.
        """
        system, messages = self._build_agent_prompt(agent, context)
        correction_prefix = ""

        for attempt in range(MAX_REGENERATION_ATTEMPTS):
            if correction_prefix:
                messages = messages + [
                    {"role": "assistant", "content": "(previous response rejected)"},
                    {"role": "user", "content": correction_prefix},
                ]

            response = self.llm.chat(
                messages=messages,
                system=system,
                purpose="agent_turn",
                max_tokens=4000,
            )

            if not response or not response.strip():
                correction_prefix = (
                    "Your previous response was empty. Respond again with actual content "
                    "using the required format fields ACTION, DIALOGUE, INNER, EMOTION, "
                    "RELATIONSHIPS, and CLUES."
                )
                logger.warning("Turn %d: Empty LLM response for %s; regenerating.",
                               self.turn, agent.id)
                continue

            # Invariant check
            ok, correction = self.director.check_invariant(agent, response)
            if not ok:
                correction_prefix = correction
                continue

            # Knowledge leak check
            ok, correction = self.director.check_knowledge_leak(
                agent, response, self.world
            )
            if not ok:
                correction_prefix = correction
                continue

            # Storyline/cast continuity check
            ok, correction = self.director.check_storyline_alignment(
                agent=agent,
                proposed_action=response,
                world=self.world,
                agents=self.agents,
                recent_interactions=context.get("recent", []),
            )
            if not ok:
                correction_prefix = correction
                continue

            return response, True

        return "", False

    # ------------------------------------------------------------------ #
    # Response Parsing
    # ------------------------------------------------------------------ #

    def _parse_agent_response(
        self, raw: str, agent: Agent
    ) -> tuple[str, dict, dict, list[str], str, bool]:
        """
        Parse structured agent response into components.

        Returns (dialogue_text, emotions_dict, relationship_deltas, clue_refs,
                 turn_mode, exit_scene)
        Also extracts and stores AGENDA for next-turn injection.
        """
        turn_mode_raw = (self._extract_field(raw, "TURN_MODE") or "dialogue").strip().lower()
        turn_mode = {
            "dialogue": "dialogue",
            "observe": "observe",
            "monologue": "monologue",
            "action": "action",
            "silent": "observe",
            "listening": "observe",
            "listen": "observe",
        }.get(turn_mode_raw, "dialogue")

        dialogue = self._extract_field(raw, "DIALOGUE") or ""
        inner    = self._extract_field(raw, "INNER") or ""
        action   = self._extract_field(raw, "ACTION") or ""
        exit_scene = self._parse_bool_field(raw, "EXIT_SCENE")

        # Fallback: extract first quoted span if DIALOGUE field is missing.
        if not dialogue:
            quoted = re.findall(r'"([^"]{2,})"', raw)
            if quoted:
                dialogue = f"\"{quoted[0]}\""
            else:
                # Remove structured field labels and keep a short natural fragment.
                cleaned = re.sub(r'(?m)^[A-Z_]+:\s*', '', raw).strip()
                dialogue = cleaned[:220]

        # Combine into readable text
        parts = []
        if action and action.lower() != "(none)":
            parts.append(f"*{action}*")
        if dialogue and dialogue.lower() not in ("(silent)", "(none)"):
            parts.append(dialogue)
        if inner:
            parts.append(f"[{inner}]")
        text = "  ".join(parts) or dialogue[:220] or raw[:220]

        # Keep observation-only turns concise and narration-like.
        if turn_mode in ("observe", "monologue") and dialogue.lower() in ("(silent)", "(none)"):
            text = "  ".join(
                p for p in [f"*{action}*" if action else "", f"[{inner}]" if inner else ""]
                if p
            ) or text

        # Parse JSON fields
        emotions    = self._parse_json_field(raw, "EMOTION")
        rel_deltas  = self._parse_json_field(raw, "RELATIONSHIPS")
        rel_deltas  = self._normalize_relationship_deltas(rel_deltas, source_agent_id=agent.id)

        # Clue references
        clue_text = self._extract_field(raw, "CLUES") or "(none)"
        clues = (
            [] if "(none)" in clue_text.lower()
            else [c.strip() for c in clue_text.split(",") if c.strip()]
        )

        # AGENDA — store for next-turn injection
        agenda = self._extract_field(raw, "AGENDA") or ""
        if agenda and agenda.lower() not in ("(none)", "(없음)"):
            self._agent_agendas[agent.id] = agenda

        return text, emotions, rel_deltas, clues, turn_mode, exit_scene

    @staticmethod
    def _normalize_ref(value: str) -> str:
        # Keep Hangul tokens too so Korean aliases can be matched if provided.
        return re.sub(r"[^0-9a-z가-힣]+", "_", (value or "").lower()).strip("_")

    def _build_agent_reference_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for agent in self.agents:
            candidates = {
                agent.id,
                agent.id.replace("_", " "),
                agent.name,
                agent.name.replace("(", " ").replace(")", " "),
            }
            if agent.id.startswith("agent_"):
                candidates.add(agent.id[len("agent_"):])
            if agent.name.lower().startswith("agent "):
                candidates.add(agent.name[6:])
            for alias in getattr(agent, "aliases", []) or []:
                candidates.add(str(alias))

            for candidate in candidates:
                norm = self._normalize_ref(str(candidate))
                if norm:
                    index.setdefault(norm, agent.id)
        return index

    def _resolve_agent_reference(self, raw_key: str) -> Optional[str]:
        if not isinstance(raw_key, str):
            return None
        key = raw_key.strip()
        if not key:
            return None
        if key in self.agent_map:
            return key

        norm = self._normalize_ref(key)
        if not norm:
            return None

        direct = self._agent_reference_index.get(norm)
        if direct:
            return direct

        # Fallback fuzzy containment for noisy strings (e.g., "Agent Christian Miller").
        candidates = {
            aid for ref, aid in self._agent_reference_index.items()
            if norm in ref or ref in norm
        }
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _normalize_relationship_deltas(
        self,
        rel_deltas: dict,
        source_agent_id: str,
    ) -> dict[str, float]:
        if not isinstance(rel_deltas, dict):
            return {}

        cleaned: dict[str, float] = {}
        for raw_key, raw_delta in rel_deltas.items():
            target_id = self._resolve_agent_reference(str(raw_key))
            if not target_id:
                logger.debug(
                    "Turn %d: ignored unknown relationship target '%s' from %s",
                    self.turn, raw_key, source_agent_id,
                )
                continue
            if target_id == source_agent_id:
                continue

            try:
                delta = float(raw_delta)
            except (TypeError, ValueError):
                continue

            delta = max(-1.0, min(1.0, delta))
            cleaned[target_id] = round(cleaned.get(target_id, 0.0) + delta, 4)

        return cleaned

    @staticmethod
    def _extract_field(text: str, field: str) -> Optional[str]:
        """Extract a labeled field from structured response."""
        pattern = rf"^{field}:\s*(.+?)(?=\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _parse_json_field(text: str, field: str) -> dict:
        """Extract and parse a JSON dict field from structured response."""
        raw = SimulationOrchestrator._extract_field(text, field) or "{}"
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            result = json.loads(raw)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    @staticmethod
    def _parse_bool_field(text: str, field: str) -> bool:
        raw = (SimulationOrchestrator._extract_field(text, field) or "").strip().lower()
        return raw in {"yes", "true", "1", "y", "네", "예"}

    # ------------------------------------------------------------------ #
    # Memory Propagation
    # ------------------------------------------------------------------ #

    def _propagate_memory(self, interaction: Interaction, speaker: Agent) -> None:
        """
        Update memory for all agents who witnessed this interaction.
        Agents only remember what they could perceive.
        """
        interaction_dict = interaction.to_dict()
        for agent in self.agents:
            if agent.id == speaker.id:
                agent.memory.store_interaction(interaction_dict)
            elif agent.id in self.world.active_agents:
                # Other agents in the scene witness this
                agent.memory.store_interaction({
                    **interaction_dict,
                    "_perspective": agent.id,
                })

    # ------------------------------------------------------------------ #
    # Persona Evolution
    # ------------------------------------------------------------------ #

    def _evolve_persona(self, agent: Agent, interaction: Interaction) -> None:
        """Compute persona drift based on recent events."""
        recent_events = agent.memory.event_log[-5:]
        if not recent_events:
            return

        events_text = json.dumps(recent_events, indent=2)
        prompt = (
            f"Based on these recent events experienced by {agent.name}:\n{events_text}\n\n"
            f"Current persona summary: {json.dumps(agent.persona)}\n\n"
            f"Describe any small personality shifts (max 2). "
            f"Reply JSON: {{\"changes\": {{\"trait\": \"new_description\"}}, "
            f"\"significant\": true/false}}"
        )

        result = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="persona_evolution",
            max_tokens=200,
            temperature=0.4,
        )
        parsed = DirectorAI._parse_json_result(result)  # reuse static parser
        changes = parsed.get("changes", {})
        if changes:
            agent.persona.update(changes)
            agent.memory.track_persona_delta(changes, interaction.content[:50], self.turn)
            db.save_persona_delta(agent.id, self.episode_id, self.turn,
                                  changes, interaction.content[:50])

    # ------------------------------------------------------------------ #
    # Clue Discovery
    # ------------------------------------------------------------------ #

    def _handle_clue_discovery(
        self, agent: Agent, clue_ref: str, interaction_id: str
    ) -> None:
        """Record that an agent discovered a clue."""
        # Match against discoverable clues from world_facts
        clue_entry = self._find_clue(clue_ref)
        if not clue_entry:
            return

        clue_id = clue_entry.get("id", clue_ref)
        agent.memory.add_clue(clue_id)
        self.director.clue_manager.track_discovery(clue_id, self.turn, agent.id)

        db.upsert_clue(clue_id, self.episode_id,
                       clue_entry.get("content", ""), self.turn)
        db.save_agent_knowledge(agent.id, clue_id, self.episode_id, self.turn)

        logger.info("Clue '%s' discovered by %s at turn %d", clue_id, agent.name, self.turn)

    def _find_clue(self, clue_ref: str) -> Optional[dict]:
        """
        Find a clue definition with priority-based matching:
          1. Exact ID match (highest priority)
          2. Match against clue aliases/trigger fields
          3. Conservative fuzzy: clue_ref words are ALL present in content
        Loose substring matching (old behavior) is removed to prevent false positives.
        """
        if not clue_ref or not clue_ref.strip():
            return None

        clue_ref_clean = clue_ref.strip()
        discoverable = self.director.world_facts.get("discoverable", [])

        # Priority 1: Exact ID match
        for c in discoverable:
            if isinstance(c, dict) and c.get("id") == clue_ref_clean:
                return c

        # Also check episode required clues (they may not be in discoverable)
        for c in self.director.clue_manager.required_clues:
            if isinstance(c, dict) and c.get("id") == clue_ref_clean:
                return c

        # Priority 2: Match against trigger or alias fields
        for c in discoverable:
            if not isinstance(c, dict):
                continue
            trigger = c.get("trigger", "")
            aliases = c.get("aliases", [])
            if isinstance(aliases, list):
                if clue_ref_clean in aliases:
                    return c
            if trigger and clue_ref_clean.lower() == trigger.lower():
                return c

        # Priority 3: Conservative fuzzy — ALL words in clue_ref must appear in content
        ref_words = set(clue_ref_clean.lower().split())
        if len(ref_words) >= 2:  # Only attempt fuzzy if ref has multiple words
            for c in discoverable:
                if not isinstance(c, dict):
                    continue
                content = c.get("content", "").lower()
                if all(word in content for word in ref_words):
                    return c

        return None

    @staticmethod
    def _preview_text(text: str, max_len: int = 180) -> str:
        flat = re.sub(r"\s+", " ", (text or "")).strip()
        if len(flat) <= max_len:
            return flat
        return flat[:max_len - 3] + "..."

    # ------------------------------------------------------------------ #
    # Loop Guard: Repetition Detection
    # ------------------------------------------------------------------ #

    def _detect_repetition_loop(self) -> bool:
        """Check if the last N agent turns are repetitive (similar action verbs/topics)."""
        if self._loop_guard_fired:
            return False

        # Gather recent non-director interactions
        recent = [
            i for i in self.interactions[-self._loop_guard_window * 2:]
            if i.action_type != "director_event"
        ][-self._loop_guard_window:]

        if len(recent) < self._loop_guard_window:
            return False

        # Extract ACTION fields for comparison
        actions = []
        for ix in recent:
            raw = ix.content
            act = self._extract_field(raw, "ACTION") or ""
            if not act:
                # Use first 60 chars of content as fallback
                act = raw[:60]
            actions.append(act.lower().strip())

        # Check similarity: count how many pairs share >60% word overlap
        from collections import Counter
        similar_count = 0
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                words_a = set(actions[i].split())
                words_b = set(actions[j].split())
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                if overlap > 0.6:
                    similar_count += 1

        # If enough similar pairs, we have a loop
        return similar_count >= self._loop_guard_threshold

    def _force_scene_transition(self) -> None:
        """Director injects a scene-advancing event to break the repetition loop."""
        self._loop_guard_fired = True
        logger.warning("[LoopGuard] Repetition detected at turn %d — forcing scene transition", self.turn)

        ep_summary = self.episode_config.get("summary", "")
        prompt = (
            f"The story has been repeating similar actions for several turns and is stuck.\n\n"
            f"Current scene: {self.world.current_scene[-300:]}\n"
            f"Location: {self.world.location}\n"
            f"Episode summary: {ep_summary[:300]}\n\n"
            f"Write a brief (2-3 sentence) scene transition event that:\n"
            f"1. Naturally interrupts the current repetitive pattern\n"
            f"2. Introduces a new stimulus (a person arriving, a sound, a message, a change)\n"
            f"3. Gives characters a new reason to act or react\n"
            f"Write as scene narration in Korean, not as dialogue."
        )
        event_text = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="loop_guard_transition",
            use_premium=True,
            temperature=0.7,
            max_tokens=200,
        )

        event_interaction = Interaction(
            id=str(uuid.uuid4()),
            episode_id=self.episode_id,
            turn=self.turn,
            speaker_id="director",
            speaker_name="[Scene]",
            content=event_text,
            action_type="director_event",
            timestamp=datetime.utcnow(),
            metadata={"trigger": "loop_guard"},
        )
        self.interactions.append(event_interaction)
        db.save_interaction(event_interaction)
        self.world.visible_context["last_event"] = event_text
        self.world.current_scene += f"  {event_text}"

        # Reset guard so it can fire again if needed after some turns
        # (will reset _loop_guard_fired after 3 more turns)
        self._loop_guard_cooldown_turn = self.turn

        self.director._log("loop_guard", "director",
                           f"Forced scene transition at turn {self.turn}",
                           {"event_text": event_text[:200]})

    # ------------------------------------------------------------------ #
    # Injection Event
    # ------------------------------------------------------------------ #

    def _apply_injection(self, injection: dict) -> None:
        """Insert a director-generated event into the interaction stream."""
        clue_id = injection["clue_id"]

        event_interaction = Interaction(
            id=str(uuid.uuid4()),
            episode_id=self.episode_id,
            turn=self.turn,
            speaker_id="director",
            speaker_name="[Scene]",
            content=injection["event_text"],
            action_type="director_event",
            timestamp=datetime.utcnow(),
            metadata={"clue_id": clue_id, "inject_method": injection["inject_method"]},
        )
        self.interactions.append(event_interaction)
        db.save_interaction(event_interaction)

        # Add to world visible context so agents can reference it
        self.world.visible_context["last_event"] = injection["event_text"]
        self.world.current_scene += f"  {injection['event_text']}"

        # ── FIX: Record clue discovery in clue_manager + DB + agent memory ──
        clue_content = injection.get("clue_content", "")

        # Track in clue_manager (marks as introduced)
        self.director.clue_manager.track_discovery(clue_id, self.turn, "director")

        # Persist clue to DB
        db.upsert_clue(clue_id, self.episode_id, clue_content, self.turn)

        # All active agents in the scene witness the injected event
        for aid in self.world.active_agents:
            agent = self.agent_map.get(aid)
            if agent:
                agent.memory.add_clue(clue_id)
                self.director.clue_manager.track_discovery(clue_id, self.turn, aid)
                db.save_agent_knowledge(aid, clue_id, self.episode_id, self.turn)

        logger.info("[Director] Injected clue event: %s (recorded for %d active agents)",
                     clue_id, len(self.world.active_agents))

    # ------------------------------------------------------------------ #
    # Agent Scheduling
    # ------------------------------------------------------------------ #

    def _next_agent(self) -> Agent:
        """Select next speaker, prioritising Director turn allocation."""
        active = [a for a in self.agents if a.id in self.world.active_agents]
        if not active:
            protagonist = next((a for a in self.agents if a.role == "protagonist"), None)
            active = [protagonist] if protagonist else [self.agents[0]]

        active_ids = [a.id for a in active]
        recent = [i.to_dict() for i in self.interactions[-8:]]

        # Director dynamically decides who should speak next (or whether scene ends).
        decision = self.director.decide_next_speaker(
            turn=self.turn,
            world=self.world,
            agents=self.agents,
            recent_interactions=recent,
            protagonist_id=self.protagonist_id,
        )

        if decision.get("end_scene") and len(active_ids) > 1:
            keep_id = str(decision.get("speaker_id", "")).strip()
            if keep_id not in active_ids:
                if self.protagonist_id in active_ids:
                    keep_id = str(self.protagonist_id)
                else:
                    keep_id = active_ids[0]

            self.world.active_agents = [keep_id]
            self._agent_cycle_index = 0
            self.director._log(
                "scene_end",
                "director",
                f"Scene ended at turn {self.turn}; active cast reduced",
                {"keep_id": keep_id, "reason": decision.get("reason", "")},
            )
            logger.info(
                "Turn %d: Director ended current exchange. Active cast now: %s",
                self.turn,
                ", ".join(
                    self.agent_map[aid].name for aid in self.world.active_agents
                    if aid in self.agent_map
                ) or "(none)",
            )

            active = [a for a in self.agents if a.id in self.world.active_agents]
            active_ids = [a.id for a in active]

        chosen_id = str(decision.get("speaker_id", "")).strip()
        if chosen_id in active_ids and chosen_id in self.agent_map:
            return self.agent_map[chosen_id]

        # Respect temporary observe/listen cooldowns while preserving fairness.
        attempts = 0
        while attempts < len(active):
            agent = active[self._agent_cycle_index % len(active)]
            self._agent_cycle_index += 1
            attempts += 1

            skip_until = self._agent_skip_until_turn.get(agent.id, 0)
            if len(active) > 1 and self.turn <= skip_until:
                continue
            return agent

        # If everyone is on cooldown, fall back to normal round-robin pick.
        agent = active[self._agent_cycle_index % len(active)]
        self._agent_cycle_index += 1
        return agent

    # ------------------------------------------------------------------ #
    # Completion Check
    # ------------------------------------------------------------------ #

    def _check_completion(self) -> bool:
        """Return True if episode objectives are all satisfied."""
        complete, unresolved = self.director.validate_resolution(
            self.turn,
            world=self.world,
            recent_interactions=[i.to_dict() for i in self.interactions[-12:]],
        )
        if not complete and self.turn < self.max_turns:
            return False
        if self.turn >= self.max_turns:
            logger.info("Max turns reached. Forcing episode end.")
            return True
        return complete


# ── Static helper imported in director.py to avoid circular import ────────────
# We attach it to DirectorAI post-hoc for backward compat
import json as _json
import re as _re

def _parse_json_result(text: str) -> dict:
    text = _re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except _json.JSONDecodeError:
                pass
    return {}

DirectorAI._parse_json_result = staticmethod(_parse_json_result)
