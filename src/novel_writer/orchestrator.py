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
from collections import deque
from datetime import datetime
from typing import Optional

from .models import Agent, WorldState, ClueManager, Interaction, Memory, SteeringContext
from .director import DirectorAI
from .llm_client import LLMClient
from . import database as db
from .review_feedback import build_feedback_prompt_block, count_feedback_term_occurrences

logger = logging.getLogger(__name__)

MAX_REGENERATION_ATTEMPTS = 3
TURN_LOCAL_REPEAT_JACCARD = 0.68
TURN_LOCAL_REPEAT_WINDOW = 3


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
        reader_feedback: Optional[dict] = None,
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
        self.reader_feedback = reader_feedback or {}

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
        text, emotions, relationship_deltas, clue_references, turn_mode, exit_scene, action_text, dialogue_text = \
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
                "action": action_text,
                "dialogue": dialogue_text,
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
        speech_profile = agent.speech_profile if isinstance(agent.speech_profile, dict) else {}
        visual_profile = agent.visual_profile if isinstance(agent.visual_profile, dict) else {}
        speech_guide = ""
        if speech_profile:
            cadence = str(speech_profile.get("cadence", "")).strip()
            tone = str(speech_profile.get("tone", "")).strip()
            lexical = speech_profile.get("lexicon", []) or []
            avoid = speech_profile.get("avoid", []) or []
            tics = speech_profile.get("signature_tics", []) or []
            formality = str(speech_profile.get("formality", "")).strip()
            parts = []
            if tone:
                parts.append(f"- Tone: {tone}")
            if cadence:
                parts.append(f"- Cadence: {cadence}")
            if formality:
                parts.append(f"- Formality: {formality}")
            if lexical:
                parts.append(f"- Preferred lexicon: {', '.join(str(x) for x in lexical[:8])}")
            if avoid:
                parts.append(f"- Avoid patterns: {', '.join(str(x) for x in avoid[:8])}")
            if tics:
                parts.append(f"- Signature tics: {', '.join(str(x) for x in tics[:6])}")
            if parts:
                speech_guide = "\n## Character Voice\n" + "\n".join(parts) + "\n"

        visual_guide = ""
        if visual_profile:
            vp_parts = []
            archetype = str(visual_profile.get("archetype", "")).strip()
            wardrobe = str(visual_profile.get("wardrobe", "")).strip()
            silhouette = str(visual_profile.get("silhouette", "")).strip()
            body_language = str(visual_profile.get("body_language", "")).strip()
            vibe = str(visual_profile.get("vibe", "")).strip()
            if archetype:
                vp_parts.append(f"- Archetype: {archetype}")
            if wardrobe:
                vp_parts.append(f"- Wardrobe: {wardrobe}")
            if silhouette:
                vp_parts.append(f"- Silhouette: {silhouette}")
            if body_language:
                vp_parts.append(f"- Body language: {body_language}")
            if vibe:
                vp_parts.append(f"- Vibe: {vibe}")
            if vp_parts:
                visual_guide = "\n## Character Presence\n" + "\n".join(vp_parts) + "\n"

        cast_visual_lines = []
        for other in others:
            vp = other.visual_profile if isinstance(other.visual_profile, dict) else {}
            if not vp:
                continue
            archetype = str(vp.get("archetype", "")).strip()
            wardrobe = str(vp.get("wardrobe", "")).strip()
            body_language = str(vp.get("body_language", "")).strip()
            parts = [p for p in [archetype, wardrobe, body_language] if p]
            if parts:
                cast_visual_lines.append(f"- {other.name}: {' | '.join(parts)}")
        cast_visual_guide = ""
        if cast_visual_lines:
            cast_visual_guide = "\n## In-Scene Visual Cues\n" + "\n".join(cast_visual_lines) + "\n"

        system = (
            f"You are {agent.name}, a character in a story.\n\n"
            f"## Story Context\n{ep_ctx}\n\n"
            f"## Your Background\n{agent.bio}\n\n"
            f"## Your Core Rules (NEVER violate these)\n"
            + "\n".join(f"- {inv}" for inv in agent.invariants) +
            f"\n\n## Your Current Goals\n{context['goals']}\n\n"
            f"## Other Characters Present\n{cast_text}\n\n"
            f"## Your Current Emotional State\n{emotions_text}\n\n"
            f"{speech_guide}"
            f"{visual_guide}"
            f"{cast_visual_guide}"
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
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=4)
        if review_guidance:
            system += (
                "\n## Reader-Focused Style Guardrail\n"
                "Minimize repetitive phrase loops and over-dense jargon.\n"
                "Keep emotional beats precise instead of repeatedly restating the same tension.\n"
                "When multiple characters are present, keep addressee/speaker reference explicit.\n"
                "If technical terms are used, only the first mention can carry a short plain-language hint.\n"
                f"{review_guidance}\n"
            )
        repeat_terms = self._feedback_repeat_terms()
        if repeat_terms:
            system += (
                "\n## Reader Repetition Watch Terms\n"
                f"- {', '.join(repeat_terms[:6])}\n"
                "- Use at most one of these terms per turn unless plot-critical.\n"
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

        action_dialogue_inner_cap, action_cap, dialogue_cap, inner_cap = self._reader_turn_word_caps()

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
            f"Keep ACTION + DIALOGUE + INNER together under ~{action_dialogue_inner_cap} Korean words total.\n"
            f"Target caps per field: ACTION <= {action_cap} words, DIALOGUE <= {dialogue_cap} words, INNER <= {inner_cap} words.\n"
            f"If 2+ characters are in-scene, include at least one explicit addressee or name cue in DIALOGUE.\n"
            f"Avoid repeating the same technical metric unless you add new actionable meaning.\n"
        )
        if self._feedback_mentions("긴 문장", "문장이 길", "긴 문단", "문단이 길", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴"):
            user_msg += (
                "Prefer 1-2 short sentences in DIALOGUE/INNER each; avoid long explanatory chains.\n"
            )
        if self._feedback_mentions("25단어", "25 단어", "25word", "긴 문장 자동 분할", "문장 자동 분할기"):
            sentence_cap = self._feedback_sentence_word_cap(default=25)
            user_msg += (
                f"Keep each DIALOGUE/INNER sentence under about {sentence_cap} words; split longer lines into 2 short beats.\n"
            )
        if self._feedback_mentions("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"):
            user_msg += (
                "If a technical term appears this turn, mention it once and move to action/reaction.\n"
            )
        if self._feedback_mentions("처음 등장", "첫 등장", "첫 언급", "괄호", "정의", "풀어쓰기", "비유", "약어"):
            user_msg += (
                "If you introduce a new technical term/acronym, add one short plain-language gloss once.\n"
                "Use a concise parenthetical cue or one brief analogy, then continue with action.\n"
            )
        if self._feedback_mentions("동의어", "통일", "의미 중복", "혼선"):
            user_msg += (
                "For the same concept, keep one stable term only (e.g., pick one and reuse consistently).\n"
                "Do not alternate near-synonyms across adjacent turns.\n"
            )
        if repeat_terms:
            repeat_cap = self._feedback_term_repeat_cap(default=2)
            user_msg += (
                f"Reader-flagged repetition words this turn: {', '.join(repeat_terms[:6])}. "
                f"Keep each such term under about {repeat_cap} use(s) per turn.\n"
            )
        jargon_term_cap = self._feedback_jargon_term_cap(default=2)
        if self._feedback_mentions("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym"):
            user_msg += (
                f"Limit distinct technical terms to about {jargon_term_cap} per turn. "
                "If more are needed, keep only core terms and compress the rest.\n"
            )
        paragraph_sentence_cap = self._feedback_paragraph_sentence_cap()
        if paragraph_sentence_cap is not None:
            user_msg += (
                f"Keep each field compact: ACTION/DIALOGUE/INNER should usually stay within about {paragraph_sentence_cap} sentence(s).\n"
            )
        if self._feedback_mentions("목록", "나열", "줄바꿈", "쪼개", "분할"):
            user_msg += (
                "If information feels list-like, split it into short beat-sized lines instead of one dense sentence.\n"
            )
        if self._feedback_mentions("반복", "중복", "늘어지", "제스처", "표정", "손동작", "관찰", "시선", "행동 묘사"):
            user_msg += (
                "Do not stack similar gesture/observation beats in one turn.\n"
                "Choose one strongest gesture cue and advance the scene.\n"
            )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "이름이 반복", "인물", "역할", "구분", "호칭", "이름", "speaker"):
            user_msg += (
                "Speaker clarity priority: in each spoken DIALOGUE line, include a clear addressee/name cue.\n"
                "Use stable naming (avoid switching titles for the same person in adjacent turns).\n"
                "Do not restate character introductions for people already in-scene.\n"
                "Add one short action/tone cue in ACTION to disambiguate who is speaking.\n"
            )
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여"):
            user_msg += (
                "Emotion rendering priority: do not over-explain feelings in INNER.\n"
                "Show one concrete action/expression cue in ACTION and keep INNER short.\n"
            )
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            user_msg += (
                "Dialogue impact priority: avoid lecture-style DIALOGUE.\n"
                "Compress factual delivery to 1 short sentence, then add one emotional/action beat.\n"
            )
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            user_msg += (
                "If dialogue keeps running, insert one concise action/reaction beat to keep pacing from flattening.\n"
                "Avoid back-to-back long explanatory dialogue turns.\n"
            )
        if self._feedback_mentions("장면 전환", "전환", "복도", "발표장", "흐름", "단문"):
            user_msg += (
                "On scene/focus shift turns, start with one concrete short transition sentence before details.\n"
            )
        if self._feedback_mentions("감정의 고저", "감정 고저", "감정의 파고", "긴장 완화", "유머", "친근한 묘사"):
            user_msg += (
                "Emotion-wave priority: maintain tension but add one brief humanizing beat when natural.\n"
                "Avoid keeping emotional intensity at one flat level for multiple turns.\n"
            )
        if self._feedback_mentions("가능성", "계산", "추론", "판단", "반복", "중복", "늘어지"):
            user_msg += (
                "Avoid repeating the same analytic words (e.g., possibility/calculation) across turns.\n"
                "If intent is unchanged, imply it through action or one short callback.\n"
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

            if self._is_locally_repetitive_turn(agent, response):
                correction_prefix = (
                    "Your previous response repeated your recent turn pattern too closely. "
                    "Keep the same scene facts, but change action/wording and advance the interaction."
                )
                logger.info(
                    "Turn %d: local repetition detected for %s; regenerating.",
                    self.turn, agent.id,
                )
                continue

            guardrail_correction = self._reader_guardrail_correction(agent, response)
            if guardrail_correction:
                correction_prefix = guardrail_correction
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
    ) -> tuple[str, dict, dict, list[str], str, bool, str, str]:
        """
        Parse structured agent response into components.

        Returns (dialogue_text, emotions_dict, relationship_deltas, clue_refs,
                 turn_mode, exit_scene, parsed_action_text, parsed_dialogue_text)
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
        action_raw = self._extract_field(raw, "ACTION") or ""
        action = action_raw.strip()
        if action.lower() in ("(none)", "none", "없음", "(없음)"):
            action = ""
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

        return text, emotions, rel_deltas, clues, turn_mode, exit_scene, action, dialogue

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

    def _feedback_mentions(self, *keywords: str) -> bool:
        if not self.reader_feedback:
            return False
        corpus = []
        for key in ("what_felt_boring_or_hard", "style_tips"):
            vals = self.reader_feedback.get(key, []) or []
            corpus.extend(str(v) for v in vals if isinstance(v, str))
        corpus.append(str(self.reader_feedback.get("reader_comment", "") or ""))
        all_text = " ".join(corpus).lower()
        lowered = [str(k).lower() for k in keywords if k]
        if any(k in all_text for k in lowered):
            return True

        # Rhythm monotony complaints are often phrased without explicit "긴 문장" wording.
        if any(k in lowered for k in ("긴 문장", "문장이 길", "긴 문단", "문단이 길", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴")):
            if any(token in all_text for token in ("비슷한 리듬", "같은 리듬", "단조", "단조롭", "단조롭게", "리듬이 반복", "속도감이 단조", "속도감이 떨어", "템포가 느려", "템포가 떨어")):
                return True

        # Reader often describes jargon overload as "체크리스트/나열/목록" without saying "기술 용어".
        if any(k in lowered for k in ("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym")):
            if any(token in all_text for token in ("체크리스트", "나열", "리스트", "목록", "목록처럼", "긴 목록", "기술 항목", "건조", "단조롭")):
                return True

        # Early reader confusion can be expressed without explicit "화자/대사 구분" terms.
        if any(k in lowered for k in ("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker")):
            if any(token in all_text for token in ("초반", "따라가기 힘들", "맥락", "인물 설명 없이", "누군지")):
                return True

        # Expository dialogue pain can appear as "정보 전달 위주/감정의 고저 부족".
        if any(k in lowered for k in ("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트")):
            if any(token in all_text for token in ("정보 전달 위주", "설명 위주", "감정의 고저", "감정 고저", "대화가 대부분", "건조", "긴 회의", "회의·대화", "대화 장면", "대사가 계속")):
                return True

        # Emotional wave notes may appear as "긴장 완화/유머/친근한 묘사/파고".
        if any(k in lowered for k in ("감정선", "감정의 고저", "감정 고저", "감정의 파고", "긴장 완화", "유머", "친근한 묘사")):
            if any(token in all_text for token in ("감정의 파고", "감정 파고", "감정의 고저", "감정 고저", "긴장 완화", "작은 유머", "유머", "친근한 묘사")):
                return True
        return False

    def _feedback_repeat_terms(self) -> list[str]:
        terms = self.reader_feedback.get("repetition_watch_terms", []) or []
        generic_terms = {
            "중요한", "간단한", "자연스러운", "효과적인", "신선한", "분명한",
            "설명", "감정", "정보", "표현", "분위기", "임팩트", "감각", "비유", "감각비유", "감각 비유",
        }
        out: list[str] = []
        seen: set[str] = set()
        for raw in terms:
            term = str(raw or "").strip()
            term = re.sub(r"^[^0-9a-zA-Z가-힣]+|[^0-9a-zA-Z가-힣]+$", "", term)
            term = re.sub(r"[^0-9a-zA-Z가-힣\s]+", " ", term)
            term = re.sub(r"\s+", " ", term).strip()
            term = re.sub(r"\s+(반복|중복|과다|과잉|묘사|표현)$", "", term, flags=re.IGNORECASE).strip()
            if len(term) > 24:
                continue
            if len(term) < 2 and not re.fullmatch(r"[가-힣]", term):
                continue
            if len(term.split()) > 2:
                continue
            if re.search(r"[가-힣A-Za-z0-9]+(?:이나|거나)\s+[가-힣A-Za-z0-9]+", term):
                continue
            if re.search(r"\s(?:또는|및|와|과)\s", f" {term} "):
                continue
            if not re.search(r"[0-9a-zA-Z가-힣]", term):
                continue
            compact = re.sub(r"\s+", "", term)
            variants = [term]
            if compact != term and len(compact) >= 2:
                variants.append(compact)
            for variant in variants:
                key = variant.lower()
                if len(key) < 2 and not re.fullmatch(r"[가-힣]", key):
                    continue
                if key in seen or key in generic_terms:
                    continue
                seen.add(key)
                out.append(variant)
        return out[:8]

    def _feedback_jargon_terms(self) -> list[str]:
        terms = self.reader_feedback.get("jargon_watch_terms", []) or []
        generic_terms = {"기술", "기술 용어", "전문 용어", "용어", "약어", "약자", "설명", "표현"}
        out: list[str] = []
        seen: set[str] = set()
        for raw in terms:
            term = str(raw or "").strip()
            term = term.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
            term = re.sub(r"[^0-9a-zA-Z가-힣\s_\-\.]+", " ", term)
            term = re.sub(r"\s+", " ", term).strip()
            if len(term) < 2 or len(term) > 32:
                continue
            key = term.lower()
            if key in seen or key in generic_terms:
                continue
            if not re.search(r"[A-Za-z가-힣]", term):
                continue
            seen.add(key)
            out.append(term)
            if len(out) >= 8:
                break
        return out

    def _feedback_style_constraints(self) -> dict:
        raw = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        return raw if isinstance(raw, dict) else {}

    def _feedback_term_repeat_cap(self, default: int = 2) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_term_repeats_per_scene")
        if raw is None:
            raw = constraints.get("max_term_repeats_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(5, cap))

    def _feedback_sentence_word_cap(self, default: int = 25) -> int:
        constraints = self._feedback_style_constraints()
        raw_hi = constraints.get("sentence_chars_max")
        try:
            hi = int(raw_hi)
        except (TypeError, ValueError):
            return default
        return max(10, min(default, int(round(hi / 3.2))))

    def _feedback_paragraph_sentence_cap(self) -> Optional[int]:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_sentences_per_paragraph")
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            return None
        return max(1, min(8, cap))

    def _feedback_jargon_term_cap(self, default: int = 2) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_jargon_terms_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(8, cap))

    def _reader_turn_word_caps(self) -> tuple[int, int, int, int]:
        """
        Dynamic compactness caps for ACTION/DIALOGUE/INNER based on reader pain points.
        Returns (total_cap, action_cap, dialogue_cap, inner_cap).
        """
        total_cap = 90
        if self._feedback_mentions("긴 문장", "문장이 길", "긴 문단", "문단이 길", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴"):
            total_cap = min(total_cap, 70)
        if self._feedback_mentions("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"):
            total_cap = min(total_cap, 65)
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "speaker"):
            total_cap = min(total_cap, 58)
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            total_cap = min(total_cap, 56)
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여"):
            total_cap = min(total_cap, 60)
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            total_cap = min(total_cap, 60)

        action_cap = max(12, min(26, total_cap // 3))
        dialogue_cap = max(24, min(52, total_cap - action_cap - 12))
        inner_cap = max(12, min(24, total_cap - action_cap - dialogue_cap))
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "speaker"):
            dialogue_cap = min(dialogue_cap, 36)
            inner_cap = min(inner_cap, 18)
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            dialogue_cap = min(dialogue_cap, 34)
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여", "긴 회의", "회의·대화", "템포가 느려"):
            inner_cap = min(inner_cap, 14)
        return total_cap, action_cap, dialogue_cap, inner_cap

    @staticmethod
    def _count_feedback_term_occurrences(text: str, term: str) -> int:
        return count_feedback_term_occurrences(text, term)

    @staticmethod
    def _dialogue_voice_is_monotone(text: str) -> bool:
        quoted = re.findall(r"[\"“]([^\"”\n]{3,120})[\"”]", text or "")
        if len(quoted) < 3:
            return False
        endings = []
        for q in quoted[:8]:
            q = re.sub(r"\s+", " ", q).strip()
            if not q:
                continue
            endings.append(q[-2:] if len(q) >= 2 else q)
        if len(endings) < 3:
            return False
        return len(set(endings)) <= 1

    def _reader_guardrail_correction(self, agent: Agent, raw_response: str) -> str:
        """
        Lightweight deterministic checks tied to reader feedback.
        Returns correction text when a regeneration is needed.
        """
        if not raw_response or not self.reader_feedback:
            return ""

        dialogue = (self._extract_field(raw_response, "DIALOGUE") or "").strip()
        inner = (self._extract_field(raw_response, "INNER") or "").strip()
        action = (self._extract_field(raw_response, "ACTION") or "").strip()
        merged = " ".join(x for x in [action, dialogue, inner] if x).strip() or raw_response
        total_cap, action_cap, dialogue_cap, inner_cap = self._reader_turn_word_caps()
        count_words = lambda s: len(re.findall(r"[0-9A-Za-z가-힣]+", str(s or "")))
        if count_words(merged) > total_cap:
            return (
                "Your previous response is too long for this scene pacing. "
                "Compress ACTION/DIALOGUE/INNER while keeping the same intent."
            )
        if count_words(action) > action_cap or count_words(dialogue) > dialogue_cap or count_words(inner) > inner_cap:
            return (
                "Your previous field lengths are unbalanced. "
                "Rewrite with shorter ACTION/DIALOGUE/INNER and keep one clear focus."
            )

        if self._feedback_mentions("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"):
            metric_tokens = self._extract_metric_tokens(merged)
            acronym_hits = len(re.findall(r"\b[A-Z]{2,8}\b", merged))
            recent_overlap = self._recent_metric_overlap_count(metric_tokens, window=5)
            if len(metric_tokens) >= 3 or acronym_hits >= 3 or recent_overlap >= 2:
                return (
                    "Your previous response repeats or over-packs technical metrics/jargon. "
                    "Keep at most one core technical term, avoid acronym stacking, and focus on action/reaction."
                )
            jargon_terms = self._feedback_jargon_terms()
            if jargon_terms:
                repeated_jargon_terms = [
                    t for t in jargon_terms
                    if self._count_feedback_term_occurrences(merged, t) >= 2
                ]
                if repeated_jargon_terms:
                    return (
                        "Your previous response repeats reader-flagged technical terms too much. "
                        "Keep one core term with a short plain cue, then move to action."
                    )
            repeat_terms = self._feedback_repeat_terms()
            if repeat_terms:
                repeat_cap = self._feedback_term_repeat_cap(default=2)
                repeated = [
                    t for t in repeat_terms
                    if self._count_feedback_term_occurrences(merged, t) > repeat_cap
                ]
                reused = [
                    t for t in repeat_terms
                    if self._count_feedback_term_occurrences(merged, t) >= 1
                ]
                if repeated or len(reused) > repeat_cap:
                    return (
                        "Your previous response reuses reader-flagged repeated terms too heavily. "
                        "Keep only one such term and vary wording via concrete action."
                    )
        if self._feedback_mentions("반복", "중복", "늘어지", "제스처", "표정", "손동작", "관찰", "시선", "행동 묘사"):
            if self._has_dense_repetitive_imagery(merged):
                return (
                    "Your previous response repeats gesture/observation cues too densely. "
                    "Keep one strongest physical cue and move the interaction forward."
                )

        if self._feedback_mentions("긴 문장", "문장이 길", "긴 문단", "문단이 길", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴"):
            dialogue_sent = len([s for s in re.split(r'(?<=[.!?…])\s+|(?<=[다요죠]\.)\s+', dialogue) if s.strip()])
            inner_sent = len([s for s in re.split(r'(?<=[.!?…])\s+|(?<=[다요죠]\.)\s+', inner) if s.strip()])
            if len(dialogue) > 100 or len(inner) > 80 or dialogue_sent > 2 or inner_sent > 2:
                return (
                    "Your previous response is too dense. Rewrite with shorter DIALOGUE/INNER "
                    "using 1-2 short sentences each."
                )
        if self._feedback_mentions("25단어", "25 단어", "25word", "긴 문장 자동 분할", "문장 자동 분할기"):
            sentence_cap = self._feedback_sentence_word_cap(default=25)
            if self._has_overlong_sentence(dialogue, max_words=sentence_cap) or self._has_overlong_sentence(inner, max_words=sentence_cap):
                return (
                    f"Your previous response includes a sentence over about {sentence_cap} words. "
                    "Split long lines into 2 shorter sentences while preserving intent."
                )
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여"):
            if len(inner) > 70 and not action:
                return (
                    "Your previous response over-explains inner feelings without observable action. "
                    "Add one concrete ACTION cue and shorten INNER."
                )
            if self._recent_cognitive_overlap_count(inner or merged, window=6) >= 2:
                return (
                    "Your previous response reuses similar analytical/cognitive wording from recent turns. "
                    "Cut inner-analysis phrasing and advance via one concrete action cue."
                )
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            if self._is_expository_dialogue(dialogue):
                return (
                    "Your previous DIALOGUE is too expository. "
                    "Compress facts into one short line and add one emotional/action cue."
                )
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            if count_words(dialogue) > 30 and count_words(action) < 4:
                return (
                    "Your previous DIALOGUE runs too long without action beats. "
                    "Shorten dialogue and add one concrete action/reaction cue for pacing."
                )
        if self._feedback_mentions("가능성", "계산", "추론", "판단", "반복", "중복", "늘어지", "심리", "내면"):
            if self._has_repetitive_cognitive_terms(inner or merged):
                return (
                    "Your previous response repeats analytical inner wording. "
                    "Use one short inner beat and move the intent into concrete action/dialogue."
                )

        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "이름이 반복", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker"):
            low = dialogue.lower()
            if dialogue and low not in ("(silent)", "(none)"):
                active_names = [
                    self.agent_map[aid].name
                    for aid in self.world.active_agents
                    if aid in self.agent_map and aid != agent.id
                ]
                has_name_cue = any(name and name in dialogue for name in active_names)
                has_action_name_cue = any(name and name in action for name in active_names)
                has_address_cue = bool(re.search(r"(씨|님|선배|교수|자네|너|당신|에게|한테)", dialogue))
                if len(active_names) >= 1 and not (has_name_cue or has_action_name_cue or has_address_cue):
                    return (
                        "Your previous response is ambiguous about who is being addressed. "
                        "Rewrite DIALOGUE/ACTION with one explicit name/addressee cue."
                    )
                if len(dialogue) > 85:
                    return (
                        "Your previous DIALOGUE is too long and risks speaker ambiguity. "
                        "Split into shorter lines and keep one clear addressee cue."
                    )
                if self._feedback_mentions("말투", "어투", "톤", "고유한 말투") and self._dialogue_voice_is_monotone(dialogue):
                    return (
                        "Your previous DIALOGUE sounds too uniform in tone and sentence endings. "
                        "Keep intent same but vary diction and cadence to keep speaker distinction."
                    )

        return ""

    @staticmethod
    def _has_overlong_sentence(text: str, max_words: int = 25) -> bool:
        if not text:
            return False
        sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?…])\s+|(?<=[다요죠]\.)\s+|\n+', text)
            if s.strip()
        ]
        if not sentences:
            sentences = [text.strip()]
        for sent in sentences:
            words = re.findall(r"[0-9A-Za-z가-힣]+", sent)
            if len(words) > max_words:
                return True
        return False

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

        # Compare normalized action signatures first (high precision), then full text fallback.
        actions = []
        metric_mentions: list[set[str]] = []
        for ix in recent:
            act = str((ix.metadata or {}).get("action", "")).strip()
            if not act:
                # content is already parsed prose; use compact fallback from it.
                act = re.sub(r"\[[^\]]+\]", "", ix.content or "")
                act = re.sub(r"[*\"'“”‘’]", "", act).strip()
            actions.append(self._normalize_loop_text(act))
            metric_mentions.append(self._extract_metric_tokens(act))

        # Check similarity: count how many pairs share >60% word overlap
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

        metric_counts: dict[str, int] = {}
        for row in metric_mentions:
            for token in row:
                metric_counts[token] = metric_counts.get(token, 0) + 1
        repeated_metric_loop = any(c >= 3 for c in metric_counts.values())

        # If enough similar pairs, we have a loop
        return similar_count >= self._loop_guard_threshold or repeated_metric_loop

    def _is_locally_repetitive_turn(self, agent: Agent, raw_response: str) -> bool:
        """
        Reject near-duplicate turns from the same agent before they are persisted.
        This catches short-range repetition earlier than global loop-guard injection.
        """
        recent_same_agent = [
            i for i in reversed(self.interactions[-8:])
            if i.speaker_id == agent.id and i.action_type != "director_event"
        ][:TURN_LOCAL_REPEAT_WINDOW]
        if not recent_same_agent:
            return False

        action = self._extract_field(raw_response, "ACTION") or ""
        dialogue = self._extract_field(raw_response, "DIALOGUE") or ""
        inner = self._extract_field(raw_response, "INNER") or ""
        candidate = " ".join(x for x in [action, dialogue, inner] if x).strip() or raw_response
        cand_norm = self._normalize_loop_text(candidate)
        if not cand_norm:
            return False
        cand_words = set(cand_norm.split())
        if not cand_words:
            return False
        cand_metric_tokens = self._extract_metric_tokens(candidate)
        cand_dialogue_norm = self._normalize_loop_text(dialogue)
        cand_dialogue_words = set(cand_dialogue_norm.split()) if cand_dialogue_norm else set()

        recent_scores = deque(maxlen=TURN_LOCAL_REPEAT_WINDOW)
        dialogue_scores = deque(maxlen=TURN_LOCAL_REPEAT_WINDOW)
        metric_overlap_hits = 0
        metric_overlap_threshold = 1 if self._feedback_mentions(
            "기술", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"
        ) else 2
        for ix in recent_same_agent:
            prev_action = str((ix.metadata or {}).get("action", "")).strip()
            prev_text = " ".join(x for x in [prev_action, ix.content or ""] if x).strip()
            prev_norm = self._normalize_loop_text(prev_text)
            if not prev_norm:
                continue
            prev_words = set(prev_norm.split())
            if not prev_words:
                continue
            overlap = len(cand_words & prev_words) / max(len(cand_words | prev_words), 1)
            recent_scores.append(overlap)
            prev_metric_tokens = self._extract_metric_tokens(prev_text)
            if cand_metric_tokens and prev_metric_tokens and cand_metric_tokens & prev_metric_tokens:
                metric_overlap_hits += 1

            prev_dialogue = ""
            if isinstance(ix.metadata, dict):
                prev_dialogue = str(ix.metadata.get("dialogue", "") or "")
            if not prev_dialogue:
                prev_dialogue = str(ix.content or "")
            prev_dialogue_norm = self._normalize_loop_text(prev_dialogue)
            prev_dialogue_words = set(prev_dialogue_norm.split()) if prev_dialogue_norm else set()
            if cand_dialogue_words and prev_dialogue_words:
                d_overlap = len(cand_dialogue_words & prev_dialogue_words) / max(
                    len(cand_dialogue_words | prev_dialogue_words), 1
                )
                dialogue_scores.append(d_overlap)

        return (
            any(score >= TURN_LOCAL_REPEAT_JACCARD for score in recent_scores)
            or (
                self._feedback_mentions(
                    "반복", "중복", "늘어지", "같은 정보", "같은 문구", "기술", "용어", "약자", "acronym"
                )
                and metric_overlap_hits >= 1
                and any(score >= 0.58 for score in dialogue_scores)
            )
            or metric_overlap_hits >= metric_overlap_threshold
        )

    @staticmethod
    def _normalize_loop_text(text: str) -> str:
        """Normalize text for repetition-loop comparison."""
        if not text:
            return ""
        cleaned = str(text).lower()
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        stop = {
            "그리고", "하지만", "그냥", "정말", "매우", "조금", "다시", "그", "이", "저",
            "to", "the", "a", "an", "and", "or", "is", "are",
        }
        toks = [t for t in cleaned.split() if len(t) > 1 and t not in stop]
        return " ".join(toks[:16])

    @staticmethod
    def _extract_metric_tokens(text: str) -> set[str]:
        """Extract repeated technical metric tokens for loop detection."""
        raw = str(text or "").lower()
        aliases = {
            "t₂": "t2",
            "t2": "t2",
            "latency": "latency",
            "rsa-2048": "rsa-2048",
            "coherence": "coherence",
            "코히런스": "coherence",
            "drift": "drift",
            "드리프트": "phase-drift",
            "위상 드리프트": "phase-drift",
            "실시간": "realtime",
            "보상": "compensation",
            "보상 회로": "compensation-circuit",
            "nsa": "nsa",
            "darpa": "darpa",
            "qpu": "qpu",
            "phase-guard": "phase-guard",
            "greyshore": "greyshore",
        }
        out: set[str] = set()
        for key, val in aliases.items():
            if key in raw:
                out.add(val)
        # Catch repeated all-caps style clue markers (e.g., COHERENCE, DRIFT).
        for token in re.findall(r"\b[A-Z]{3,14}\b", str(text or "")):
            out.add(token.lower())
        for m in re.findall(r"\d+(?:\.\d+)?(?:ms|초|배|%|x)?", raw):
            if len(m) >= 2:
                out.add(m)
        return out

    def _recent_metric_overlap_count(self, candidate_tokens: set[str], window: int = 5) -> int:
        """
        Count how many recent turns reuse the candidate's metric/jargon tokens.
        Helps block cross-speaker jargon echo loops before they propagate.
        """
        if not candidate_tokens:
            return 0
        recent = [
            i for i in self.interactions[-max(1, window):]
            if i.action_type != "director_event"
        ]
        hits = 0
        for ix in recent:
            action = str((ix.metadata or {}).get("action", "")).strip()
            dialogue = str((ix.metadata or {}).get("dialogue", "")).strip()
            prev_text = " ".join(x for x in [action, dialogue, ix.content or ""] if x).strip()
            prev_tokens = self._extract_metric_tokens(prev_text)
            if prev_tokens and candidate_tokens & prev_tokens:
                hits += 1
        return hits

    def _recent_cognitive_overlap_count(self, text: str, window: int = 6) -> int:
        """
        Count overlap of cognitive-analysis cues with recent turns.
        Used to prevent multi-turn "analysis loop" wording.
        """
        candidate_tokens = self._extract_cognitive_tokens(text)
        if not candidate_tokens:
            return 0
        recent = [
            i for i in self.interactions[-max(1, window):]
            if i.action_type != "director_event"
        ]
        hits = 0
        for ix in recent:
            action = str((ix.metadata or {}).get("action", "")).strip()
            dialogue = str((ix.metadata or {}).get("dialogue", "")).strip()
            prev_text = " ".join(x for x in [action, dialogue, ix.content or ""] if x).strip()
            prev_tokens = self._extract_cognitive_tokens(prev_text)
            if prev_tokens and candidate_tokens & prev_tokens:
                hits += 1
        return hits

    @staticmethod
    def _extract_cognitive_tokens(text: str) -> set[str]:
        """
        Extract short cognitive-analysis cues that often get repeated in slow loops.
        """
        raw = str(text or "").lower()
        cue_map = {
            "가능성": "possibility",
            "계산": "calculation",
            "추론": "inference",
            "판단": "judgement",
            "가정": "assumption",
            "시나리오": "scenario",
            "확률": "probability",
            "결론": "conclusion",
            "고민": "deliberation",
        }
        out: set[str] = set()
        for key, val in cue_map.items():
            if key in raw:
                out.add(val)
        return out

    @staticmethod
    def _has_dense_repetitive_imagery(text: str) -> bool:
        """
        Detect dense repetition of gesture/observation cues in one turn.
        Keeps sensitivity modest to avoid suppressing natural physical detail.
        """
        raw = str(text or "").lower()
        if not raw:
            return False
        cues = [
            "제스처", "표정", "손동작", "손끝", "시선",
            "어깨", "숨", "고개", "정적", "침묵", "미간", "관찰",
        ]
        counts = [raw.count(c) for c in cues]
        max_count = max(counts) if counts else 0
        repeated_types = sum(1 for c in counts if c >= 2)
        return max_count >= 3 or repeated_types >= 2

    @staticmethod
    def _has_repetitive_cognitive_terms(text: str) -> bool:
        """
        Detect repetitive inner-analysis wording that slows pacing.
        """
        raw = str(text or "").lower()
        if not raw:
            return False
        cues = [
            "가능성", "계산", "판단", "추론", "결론", "시나리오", "확률", "가정",
        ]
        repeated_types = 0
        for cue in cues:
            if raw.count(cue) >= 2:
                repeated_types += 1
        return repeated_types >= 2

    @staticmethod
    def _is_expository_dialogue(dialogue: str) -> bool:
        """
        Detect dialogue lines that over-index on explanation delivery.
        """
        text = str(dialogue or "").strip()
        if not text:
            return False
        low = text.lower()
        explain_hits = len(re.findall(r"(왜냐하면|즉|다시 말해|정리하면|요약하면|핵심은|결론은|설명하자면)", low))
        technical_hits = len(re.findall(r"\b[A-Z]{2,8}\b", text)) + len(re.findall(r"\d+(?:\.\d+)?", text))
        if len(text) >= 110 and (explain_hits >= 1 or technical_hits >= 2):
            return True
        return explain_hits >= 2

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
