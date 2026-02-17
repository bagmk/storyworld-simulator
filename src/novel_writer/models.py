"""
Core data models for the AI Story Simulation Engine.
All structures are content-agnostic — story details come from YAML configs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    """Per-agent memory store."""
    interaction_history: list[dict] = field(default_factory=list)
    emotional_state: dict = field(default_factory=dict)         # {emotion: intensity}
    relationship_matrix: dict[str, float] = field(default_factory=dict)  # agent_id -> -1..1
    event_log: list[dict] = field(default_factory=list)
    persona_deltas: list[dict] = field(default_factory=list)
    known_clues: set = field(default_factory=set)

    def store_interaction(self, interaction: dict) -> None:
        self.interaction_history.append(interaction)

    def store_event(self, event: dict) -> None:
        self.event_log.append(event)

    def add_clue(self, clue_id: str) -> None:
        self.known_clues.add(clue_id)

    def get_relationship(self, agent_id: str) -> float:
        return self.relationship_matrix.get(agent_id, 0.0)

    def update_relationship(self, agent_id: str, delta: float) -> None:
        current = self.relationship_matrix.get(agent_id, 0.0)
        self.relationship_matrix[agent_id] = max(-1.0, min(1.0, current + delta))

    # Emotion smoothing constants
    _EMOTION_ALPHA: float = 0.85       # EMA weight for previous state
    _EMOTION_MAX_DELTA: float = 0.15   # Max change per turn
    _EMOTION_DECAY: float = 0.05       # Decay rate for unmentioned emotions

    def record_emotion(self, emotion: str, intensity: float, turn: int) -> None:
        """Record emotion with EMA smoothing and per-turn delta clamp."""
        raw = max(0.0, min(1.0, intensity))  # clamp input to [0, 1]
        prev = self.emotional_state.get(emotion, 0.0)

        # EMA: blend previous and new
        smoothed = self._EMOTION_ALPHA * prev + (1.0 - self._EMOTION_ALPHA) * raw

        # Clamp max delta per turn
        delta = smoothed - prev
        if abs(delta) > self._EMOTION_MAX_DELTA:
            smoothed = prev + self._EMOTION_MAX_DELTA * (1.0 if delta > 0 else -1.0)

        smoothed = max(0.0, min(1.0, smoothed))
        self.emotional_state[emotion] = round(smoothed, 4)
        self.event_log.append({"type": "emotion", "emotion": emotion,
                                "intensity": round(smoothed, 4),
                                "raw_input": round(raw, 4), "turn": turn})

    def decay_unmentioned_emotions(self, mentioned: set[str], turn: int) -> None:
        """Decay emotions not mentioned this turn instead of dropping to 0."""
        for emotion in list(self.emotional_state):
            if emotion not in mentioned:
                prev = self.emotional_state[emotion]
                decayed = prev * (1.0 - self._EMOTION_DECAY)
                # Drop to 0 if negligible
                if decayed < 0.01:
                    decayed = 0.0
                self.emotional_state[emotion] = round(decayed, 4)

    def track_persona_delta(self, changes: dict, trigger: str, turn: int) -> None:
        self.persona_deltas.append({"turn": turn, "trigger": trigger, "changes": changes})

    def recent_interactions(self, n: int = 10) -> list[dict]:
        return self.interaction_history[-n:]

    def to_dict(self) -> dict:
        d = {
            "interaction_history": self.interaction_history,
            "emotional_state": self.emotional_state,
            "relationship_matrix": self.relationship_matrix,
            "event_log": self.event_log,
            "persona_deltas": self.persona_deltas,
            "known_clues": list(self.known_clues),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        m = cls(
            interaction_history=d.get("interaction_history", []),
            emotional_state=d.get("emotional_state", {}),
            relationship_matrix=d.get("relationship_matrix", {}),
            event_log=d.get("event_log", []),
            persona_deltas=d.get("persona_deltas", []),
            known_clues=set(d.get("known_clues", [])),
        )
        return m


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Represents a story character/agent."""
    id: str
    name: str
    role: str                       # protagonist / supporting / background
    bio: str
    invariants: list[str]
    goals: list[str]
    aliases: list[str] = field(default_factory=list)
    memory: Memory = field(default_factory=Memory)
    persona: dict = field(default_factory=dict)  # dynamic personality state

    @classmethod
    def from_config(cls, cfg: dict) -> "Agent":
        """Build an Agent from a character YAML block."""
        a = cls(
            id=cfg["id"],
            name=cfg["name"],
            role=cfg.get("role", "supporting"),
            bio=cfg.get("bio", ""),
            aliases=cfg.get("aliases", []),
            invariants=cfg.get("invariants", []),
            goals=cfg.get("goals", []),
        )
        # Seed relationship matrix from config
        for other_id, val in cfg.get("initial_relationships", {}).items():
            a.memory.relationship_matrix[other_id] = float(val)
        # Build base persona
        a.persona = {
            "name": a.name,
            "aliases": a.aliases,
            "bio": a.bio,
            "invariants": a.invariants,
            "goals": a.goals,
            "role": a.role,
        }
        return a

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "bio": self.bio,
            "aliases": self.aliases,
            "invariants": self.invariants,
            "goals": self.goals,
            "persona": self.persona,
            "memory": self.memory.to_dict(),
        }


# ---------------------------------------------------------------------------
# World State
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """Global story world at a given turn."""
    current_scene: str = ""
    active_agents: list[str] = field(default_factory=list)
    location: str = ""
    time: datetime = field(default_factory=datetime.utcnow)
    hidden_facts: list[str] = field(default_factory=list)
    visible_context: dict = field(default_factory=dict)
    turn: int = 0

    def update(self, action: dict) -> None:
        if "location" in action:
            self.location = action["location"]
        if "scene_update" in action:
            self.current_scene += " " + action["scene_update"]

    def get_context_for_agent(self, agent_id: str) -> dict:
        """Return world context without hidden facts."""
        return {
            "scene": self.current_scene,
            "location": self.location,
            "time": self.time.isoformat(),
            "active_agents": self.active_agents,
            "visible_context": self.visible_context,
            "turn": self.turn,
        }

    def to_dict(self) -> dict:
        return {
            "current_scene": self.current_scene,
            "active_agents": self.active_agents,
            "location": self.location,
            "time": self.time.isoformat(),
            "hidden_facts": self.hidden_facts,
            "visible_context": self.visible_context,
            "turn": self.turn,
        }


# ---------------------------------------------------------------------------
# Clue Manager
# ---------------------------------------------------------------------------

@dataclass
class ClueManager:
    """Tracks clue lifecycle across an episode."""
    required_clues: list[dict] = field(default_factory=list)
    introduced: dict[str, int] = field(default_factory=dict)    # clue_id -> turn
    resolved: dict[str, int] = field(default_factory=dict)      # clue_id -> turn
    agent_knowledge: dict[str, set] = field(default_factory=dict)  # agent_id -> {clue_ids}
    injection_triggers: list[dict] = field(default_factory=list)

    def track_discovery(self, clue_id: str, turn: int, agent_id: str) -> None:
        if clue_id not in self.introduced:
            self.introduced[clue_id] = turn
        if agent_id not in self.agent_knowledge:
            self.agent_knowledge[agent_id] = set()
        self.agent_knowledge[agent_id].add(clue_id)

    def mark_resolved(self, clue_id: str, turn: int) -> None:
        self.resolved[clue_id] = turn

    def is_introduced(self, clue_id: str) -> bool:
        return clue_id in self.introduced

    def agent_knows(self, agent_id: str, clue_id: str) -> bool:
        return clue_id in self.agent_knowledge.get(agent_id, set())

    def unintroduced_required(self) -> list[dict]:
        """Return required clues not yet introduced."""
        return [c for c in self.required_clues
                if c.get("id") not in self.introduced]

    def needs_injection(self, turn_progress: float) -> Optional[dict]:
        """
        Return a clue that needs director injection if it hasn't surfaced
        by a certain % of the episode (defined per-clue or default 0.6).
        """
        for clue in self.required_clues:
            clue_id = clue.get("id")
            threshold = clue.get("inject_threshold", 0.6)
            if clue_id not in self.introduced and turn_progress >= threshold:
                return clue
        return None

    def to_dict(self) -> dict:
        return {
            "required_clues": self.required_clues,
            "introduced": self.introduced,
            "resolved": self.resolved,
            "agent_knowledge": {k: list(v) for k, v in self.agent_knowledge.items()},
        }


# ---------------------------------------------------------------------------
# Interaction record
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    """A single turn's action/dialogue."""
    id: str
    episode_id: str
    turn: int
    speaker_id: str
    speaker_name: str
    content: str
    action_type: str = "dialogue"   # dialogue / action / inner_thought / director_event
    target_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "turn": self.turn,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "content": self.content,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Steering Context (Trial-and-Learn)
# ---------------------------------------------------------------------------

@dataclass
class SteeringContext:
    """Per-agent dynamic steering data that evolves across trial attempts."""
    agent_id: str
    tactical_goals: list[str] = field(default_factory=list)
    steering_prompt: str = ""
    exemplar_actions: list[str] = field(default_factory=list)
    attempt_number: int = 1

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "tactical_goals": self.tactical_goals,
            "steering_prompt": self.steering_prompt,
            "exemplar_actions": self.exemplar_actions,
            "attempt_number": self.attempt_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringContext":
        return cls(
            agent_id=d["agent_id"],
            tactical_goals=d.get("tactical_goals", []),
            steering_prompt=d.get("steering_prompt", ""),
            exemplar_actions=d.get("exemplar_actions", []),
            attempt_number=d.get("attempt_number", 1),
        )
