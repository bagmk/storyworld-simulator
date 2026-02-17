"""
YAML Configuration Loader for the AI Story Simulation Engine.

Parses and validates:
  - characters.yaml     → list of Agent objects
  - episodes/ep01.yaml  → episode beat config
  - world_facts.yaml    → hidden / discoverable / public facts

All story content lives in these files — the engine is fully content-agnostic.
"""

from __future__ import annotations
import logging
from pathlib import Path
import re
from typing import Any

import yaml

from .models import Agent, WorldState, ClueManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Characters Loader
# ---------------------------------------------------------------------------

def load_characters(path: str) -> list[Agent]:
    """
    Load character definitions from a YAML file.

    Expected format:
        characters:
          - id: "char_001"
            name: "Alice Vance"
            role: "protagonist"
            bio: "..."
            invariants:
              - "Always protects her sister above all else"
            goals:
              - "Find the missing research files"
            initial_relationships:
              char_002: 0.6
    """
    data = _load_yaml(path)
    characters_raw = data.get("characters", [])
    if not characters_raw:
        raise ValueError(f"No 'characters' key found in {path}")

    agents = []
    for cfg in characters_raw:
        _require_fields(cfg, ["id", "name"], context=f"character in {path}")
        agent = Agent.from_config(cfg)
        agents.append(agent)
        logger.info("Loaded character: %s (%s) — %s", agent.name, agent.id, agent.role)

    logger.info("Loaded %d characters from %s", len(agents), path)
    return agents


# ---------------------------------------------------------------------------
# Episode Loader
# ---------------------------------------------------------------------------

def load_episode(path: str) -> dict:
    """
    Load an episode beat config from YAML.

    Expected top-level keys:
        episode:
          id, date, summary, introduced_clues, resolved,
          recommended_length, pacing, max_turns, characters
    """
    data = _load_yaml(path)
    ep = data.get("episode", data)  # support both nested and flat

    _require_fields(ep, ["id"], context=f"episode in {path}")
    ep["source_file"] = str(Path(path))
    source_id = _extract_episode_source_id(path)
    if source_id and "storyline_source_id" not in ep:
        ep["storyline_source_id"] = source_id

    # Normalise introduced_clues to list of dicts with id/content
    raw_clues = ep.get("introduced_clues", [])
    normalised_clues = []
    for i, c in enumerate(raw_clues):
        if isinstance(c, str):
            normalised_clues.append({
                "id": f"clue_{i+1:03d}",
                "content": c,
                "inject_threshold": 0.6,
                "trigger": "natural discovery",
                "inject_method": "environmental_cue",
            })
        elif isinstance(c, dict):
            c.setdefault("id", f"clue_{i+1:03d}")
            c.setdefault("inject_threshold", 0.6)
            c.setdefault("inject_method", "environmental_cue")
            normalised_clues.append(c)
    ep["introduced_clues"] = normalised_clues

    # Compute max_turns from recommended_length if not set
    if "max_turns" not in ep:
        words    = ep.get("recommended_length", 3500)
        pacing   = ep.get("pacing", "normal")
        words_per_turn = {"slow": 60, "normal": 50, "tense": 40, "fast": 30}.get(pacing, 50)
        ep["max_turns"] = max(20, words // words_per_turn)

    # Pacing guidelines sub-dict
    ep.setdefault("pacing_guidelines", {
        "style": ep.get("pacing", "normal"),
        "target_turns": ep["max_turns"],
    })

    logger.info(
        "Loaded episode: %s | max_turns=%d | clues=%d",
        ep["id"], ep["max_turns"], len(normalised_clues)
    )
    return ep


# ---------------------------------------------------------------------------
# Storyline Loader
# ---------------------------------------------------------------------------

def load_storyline(path: str) -> dict:
    """
    Load long-arc storyline YAML.

    Expected structure:
        title: "..."
        acts:
          - id: "act1"
            title: "..."
            milestones:
              - id: "ep0_xxx"
                description: "..."
    """
    data = _load_yaml(path)
    data.setdefault("title", "Untitled Storyline")
    acts_raw = data.get("acts", [])
    acts = [a for a in acts_raw if isinstance(a, dict)]
    data["acts"] = acts

    milestone_count = 0
    for act in acts:
        milestones = act.get("milestones", [])
        if isinstance(milestones, list):
            milestone_count += len([m for m in milestones if isinstance(m, dict)])

    logger.info(
        "Loaded storyline: %s | acts=%d | milestones=%d",
        data.get("title", "Untitled Storyline"),
        len(acts),
        milestone_count,
    )
    return data


# ---------------------------------------------------------------------------
# World Facts Loader
# ---------------------------------------------------------------------------

def load_world_facts(path: str) -> dict:
    """
    Load world facts YAML.

    Expected structure:
        world_facts:
          hidden:
            - "Secret organization controls the lab"
          discoverable:
            - id: "clue_001"
              content: "Financial records show discrepancy"
              trigger: "agent searches documents"
          public:
            - "The year is 2043"
    """
    data = _load_yaml(path)
    wf = data.get("world_facts", data)

    wf.setdefault("hidden", [])
    wf.setdefault("discoverable", [])
    wf.setdefault("public", [])

    logger.info(
        "Loaded world facts: %d hidden, %d discoverable, %d public",
        len(wf["hidden"]), len(wf["discoverable"]), len(wf["public"])
    )
    return wf


# ---------------------------------------------------------------------------
# WorldState Builder
# ---------------------------------------------------------------------------

def build_world_state(episode_config: dict, world_facts: dict,
                      agents: list[Agent]) -> WorldState:
    """
    Construct the initial WorldState from configs.
    """
    ep_date_str = episode_config.get("date", "")

    from datetime import datetime
    try:
        ep_time = datetime.fromisoformat(ep_date_str) if ep_date_str else datetime.utcnow()
    except ValueError:
        ep_time = datetime.utcnow()

    world = WorldState(
        current_scene=episode_config.get("summary", ""),
        active_agents=[a.id for a in agents],
        location=episode_config.get("location", "Unknown Location"),
        time=ep_time,
        hidden_facts=[
            f.get("content", str(f)) if isinstance(f, dict) else str(f)
            for f in world_facts.get("hidden", [])
        ],
        visible_context={
            "public_facts": world_facts.get("public", []),
        },
    )
    logger.info("World state initialised. Scene: %s...", world.current_scene[:80])
    return world


# ---------------------------------------------------------------------------
# ClueManager Builder
# ---------------------------------------------------------------------------

def build_clue_manager(episode_config: dict, world_facts: dict) -> ClueManager:
    """
    Build ClueManager from episode clue list + discoverable world facts.
    """
    # Merge episode introduced_clues with world_facts.discoverable
    required = list(episode_config.get("introduced_clues", []))

    # Cross-reference with discoverable world facts
    discoverable_ids = {
        c["id"] for c in world_facts.get("discoverable", [])
        if isinstance(c, dict) and "id" in c
    }
    for clue in required:
        cid = clue.get("id", "")
        if cid not in discoverable_ids:
            # Add to discoverable if not already there
            world_facts.setdefault("discoverable", []).append({
                "id": cid,
                "content": clue.get("content", ""),
                "trigger": clue.get("trigger", ""),
            })

    manager = ClueManager(required_clues=required)
    logger.info("ClueManager ready with %d required clues", len(required))
    return manager


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict at top level in {path}, got {type(data)}")
    return data


def _extract_episode_source_id(path: str) -> str:
    """
    Extract storyline source id from episode YAML comments.
    Example: '# source: ep3_double_agent'
    """
    p = Path(path)
    if not p.exists():
        return ""

    source_pattern = re.compile(r"^\s*#\s*source:\s*([A-Za-z0-9_\-]+)\s*$")
    try:
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                match = source_pattern.match(line.rstrip("\n"))
                if match:
                    return match.group(1).strip()
                if idx >= 30:
                    break
    except OSError:
        return ""
    return ""


def _require_fields(data: dict, fields: list[str], context: str = "") -> None:
    for field in fields:
        if field not in data:
            raise ValueError(
                f"Missing required field '{field}' in {context or 'config'}"
            )
