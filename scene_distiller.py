"""
Scene Distiller for the AI Story Simulation Engine.

Takes raw turn-by-turn simulation interactions (30-60+ turns) and distills
them into 6-12 essential narrative scenes. This eliminates repetition,
merges redundant turns, and maps each scene back to the original YAML beats.

Pipeline:
  1. Load interactions from DB
  2. Filter to protagonist's perspective
  3. Detect scene boundaries (location change, time skip, cast change, topic shift)
  4. Merge consecutive turns that belong to the same dramatic beat
  5. For each scene: extract key dialogue, actions, discoveries, emotional shifts
  6. Cross-reference with original YAML beats to ensure beat fidelity
  7. Output: list of DistilledScene objects ready for prose generation
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from llm_client import LLMClient
import database as db

logger = logging.getLogger(__name__)


@dataclass
class DistilledScene:
    """A compressed narrative scene distilled from multiple simulation turns."""
    scene_number: int
    title: str                          # Short scene title
    turn_range: tuple[int, int]         # (start_turn, end_turn)
    location: str
    characters_present: list[str]
    key_dialogue: list[dict]            # [{speaker, line}] — essential lines only
    key_actions: list[str]              # Important physical actions
    discoveries: list[str]             # Clues or revelations in this scene
    emotional_arc: str                  # Brief emotional trajectory
    beat_references: list[str]          # YAML clue IDs this scene covers
    narrative_summary: str              # 2-3 sentence summary of what happens
    pacing: str                         # "opening" / "building" / "climax" / "resolution"
    raw_turn_count: int                 # How many raw turns were compressed

    def to_dict(self) -> dict:
        return {
            "scene_number": self.scene_number,
            "title": self.title,
            "turn_range": list(self.turn_range),
            "location": self.location,
            "characters_present": self.characters_present,
            "key_dialogue": self.key_dialogue,
            "key_actions": self.key_actions,
            "discoveries": self.discoveries,
            "emotional_arc": self.emotional_arc,
            "beat_references": self.beat_references,
            "narrative_summary": self.narrative_summary,
            "pacing": self.pacing,
            "raw_turn_count": self.raw_turn_count,
        }


class SceneDistiller:
    """
    Distills raw simulation interactions into essential narrative scenes.

    Parameters
    ----------
    llm : LLMClient
        Used for intelligent scene boundary detection and summarization.
    episode_config : dict
        Original YAML episode configuration (for beat cross-referencing).
    """

    def __init__(
        self,
        llm: LLMClient,
        episode_config: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.episode_config = episode_config or {}

    # ------------------------------------------------------------------ #
    # Public: Distill Episode
    # ------------------------------------------------------------------ #

    def distill(
        self,
        episode_id: str,
        protagonist_id: str,
        target_scenes: int = 8,
    ) -> list[DistilledScene]:
        """
        Distill an episode's interactions into essential narrative scenes.

        Parameters
        ----------
        episode_id      : episode to process (loads from DB)
        protagonist_id  : whose perspective to use
        target_scenes   : approximate number of output scenes (6-12)

        Returns
        -------
        list[DistilledScene] : ordered list of distilled scenes
        """
        # 1. Load and filter interactions
        raw = db.load_episode_interactions(episode_id)
        if not raw:
            raise ValueError(f"No interactions found for episode {episode_id}")

        pov = self._filter_perspective(raw, protagonist_id)
        logger.info("Loaded %d raw interactions, %d after POV filter", len(raw), len(pov))

        # 2. Extract YAML beat info for cross-referencing
        beats = self._extract_beats()

        # 3. Use LLM to intelligently segment and distill
        scenes = self._llm_distill(pov, beats, protagonist_id, target_scenes)

        logger.info("Distilled %d turns into %d scenes", len(pov), len(scenes))
        return scenes

    # ------------------------------------------------------------------ #
    # Perspective Filter
    # ------------------------------------------------------------------ #

    def _filter_perspective(
        self, interactions: list[dict], protagonist_id: str
    ) -> list[dict]:
        """Keep only interactions the protagonist witnessed."""
        filtered = []
        for ix in interactions:
            if ix["speaker_id"] == protagonist_id:
                filtered.append({**ix, "_is_self": True})
                continue
            if ix.get("action_type") == "director_event":
                filtered.append({**ix, "_is_scene": True})
                continue
            content = ix.get("content", "")
            # Skip other characters' inner thoughts (wrapped in [])
            if content.startswith("[") and content.endswith("]"):
                continue
            filtered.append(ix)
        return filtered

    # ------------------------------------------------------------------ #
    # Beat Extraction from YAML
    # ------------------------------------------------------------------ #

    def _extract_beats(self) -> list[dict]:
        """Extract clue/beat definitions from episode config."""
        clues = self.episode_config.get("introduced_clues", [])
        beats = []
        for c in clues:
            if isinstance(c, dict):
                beats.append({
                    "id": c.get("id", ""),
                    "content": c.get("content", ""),
                    "method": c.get("inject_method", ""),
                })
        return beats

    # ------------------------------------------------------------------ #
    # LLM-Powered Distillation
    # ------------------------------------------------------------------ #

    def _llm_distill(
        self,
        interactions: list[dict],
        beats: list[dict],
        protagonist_id: str,
        target_scenes: int,
    ) -> list[DistilledScene]:
        """Use LLM to segment interactions into distilled scenes."""
        # Format interactions compactly
        turns_text = self._format_turns_compact(interactions)

        # Format beats for reference
        beats_text = "\n".join(
            f"- [{b['id']}]: {b['content']}" for b in beats
        ) or "(no beats defined)"

        ep_summary = self.episode_config.get("summary", "")
        ep_location = self.episode_config.get("location", "")
        ep_pacing = self.episode_config.get("pacing", "normal")

        prompt = (
            f"You are a narrative editor distilling a raw simulation log into "
            f"essential story scenes.\n\n"
            f"## Episode Info\n"
            f"Location: {ep_location}\n"
            f"Pacing: {ep_pacing}\n"
            f"Summary: {ep_summary}\n\n"
            f"## Required Story Beats (clues that should appear)\n{beats_text}\n\n"
            f"## Raw Simulation Log ({len(interactions)} turns)\n{turns_text}\n\n"
            f"## Task\n"
            f"Distill these {len(interactions)} turns into exactly {target_scenes} "
            f"narrative scenes. For each scene:\n\n"
            f"1. **Merge** consecutive turns that describe the same dramatic moment\n"
            f"2. **Eliminate** repetitive content (if projector refocuses 5 times, "
            f"keep it once)\n"
            f"3. **Keep** only the most impactful dialogue lines (2-4 per scene)\n"
            f"4. **Identify** which YAML beats/clues each scene covers\n"
            f"5. **Assign** pacing: opening / building / climax / resolution\n"
            f"6. Do NOT invent content not present in the log. Only compress and select.\n\n"
            f"Reply with a JSON array of {target_scenes} scene objects:\n"
            f"```json\n"
            f"[\n"
            f"  {{\n"
            f"    \"title\": \"short scene title\",\n"
            f"    \"turn_start\": 1,\n"
            f"    \"turn_end\": 8,\n"
            f"    \"location\": \"specific location\",\n"
            f"    \"characters\": [\"Name1\", \"Name2\"],\n"
            f"    \"key_dialogue\": [{{\"speaker\": \"Name\", \"line\": \"actual quote\"}}],\n"
            f"    \"key_actions\": [\"action description\"],\n"
            f"    \"discoveries\": [\"what was discovered\"],\n"
            f"    \"emotional_arc\": \"brief emotional trajectory\",\n"
            f"    \"beat_refs\": [\"clue_id_1\"],\n"
            f"    \"summary\": \"2-3 sentence narrative summary\",\n"
            f"    \"pacing\": \"building\"\n"
            f"  }}\n"
            f"]\n"
            f"```\n"
            f"Return ONLY the JSON array, no other text."
        )

        result = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="scene_distillation",
            use_premium=True,
            temperature=0.3,
            max_tokens=4000,
        )

        scenes_data = self._parse_json_array(result)

        if not scenes_data:
            logger.warning("LLM distillation returned no scenes; falling back to chunking")
            return self._fallback_chunk(interactions, beats, target_scenes)

        # Convert to DistilledScene objects
        scenes: list[DistilledScene] = []
        for i, sd in enumerate(scenes_data):
            scene = DistilledScene(
                scene_number=i + 1,
                title=sd.get("title", f"Scene {i + 1}"),
                turn_range=(sd.get("turn_start", 0), sd.get("turn_end", 0)),
                location=sd.get("location", ep_location),
                characters_present=sd.get("characters", []),
                key_dialogue=sd.get("key_dialogue", []),
                key_actions=sd.get("key_actions", []),
                discoveries=sd.get("discoveries", []),
                emotional_arc=sd.get("emotional_arc", ""),
                beat_references=sd.get("beat_refs", []),
                narrative_summary=sd.get("summary", ""),
                pacing=sd.get("pacing", "building"),
                raw_turn_count=sd.get("turn_end", 0) - sd.get("turn_start", 0) + 1,
            )
            scenes.append(scene)

        # Validate beat coverage
        covered_beats = set()
        for s in scenes:
            covered_beats.update(s.beat_references)
        required = {b["id"] for b in beats}
        missing = required - covered_beats
        if missing:
            logger.warning("Beats not covered in distilled scenes: %s", missing)

        return scenes

    # ------------------------------------------------------------------ #
    # Fallback: Simple Chunking
    # ------------------------------------------------------------------ #

    def _fallback_chunk(
        self,
        interactions: list[dict],
        beats: list[dict],
        target_scenes: int,
    ) -> list[DistilledScene]:
        """Simple equal-size chunking if LLM distillation fails."""
        chunk_size = max(1, len(interactions) // target_scenes)
        scenes = []
        for i in range(0, len(interactions), chunk_size):
            chunk = interactions[i:i + chunk_size]
            if not chunk:
                continue

            start_turn = chunk[0].get("turn", 0)
            end_turn = chunk[-1].get("turn", 0)
            chars = list({ix.get("speaker_name", "?") for ix in chunk if ix.get("speaker_id") != "director"})
            dialogue = [
                {"speaker": ix["speaker_name"], "line": ix["content"][:150]}
                for ix in chunk
                if ix.get("speaker_id") != "director" and not ix.get("content", "").startswith("[")
            ][:3]

            scene_num = len(scenes) + 1
            scenes.append(DistilledScene(
                scene_number=scene_num,
                title=f"Scene {scene_num}",
                turn_range=(start_turn, end_turn),
                location=self.episode_config.get("location", ""),
                characters_present=chars,
                key_dialogue=dialogue,
                key_actions=[],
                discoveries=[],
                emotional_arc="",
                beat_references=[],
                narrative_summary="",
                pacing="building",
                raw_turn_count=len(chunk),
            ))

        return scenes

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _format_turns_compact(self, interactions: list[dict]) -> str:
        """Format interactions compactly for LLM context."""
        lines = []
        for ix in interactions:
            turn = ix.get("turn", "?")
            speaker = ix.get("speaker_name", "?")
            content = ix.get("content", "")
            # Truncate very long content
            if len(content) > 250:
                content = content[:247] + "..."
            action_type = ix.get("action_type", "dialogue")
            tag = "[SCENE]" if action_type == "director_event" else ""
            lines.append(f"T{turn} {tag}{speaker}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """Parse a JSON array from LLM response."""
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # Try finding array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        return []
