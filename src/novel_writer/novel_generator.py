"""
Novel Generator for the AI Story Simulation Engine.

Converts raw simulation interaction logs into polished first-person
narrative prose from the protagonist's point of view.

Pipeline:
  1. Load all interactions from the database for an episode
  2. Filter to protagonist's perspective (what they saw / heard)
  3. Analyze emotional arc and identify key moments
  4. Generate narrative prose section by section
  5. Add internal monologue
  6. Polish for readability and pacing
  7. Write final chapter as Markdown
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from . import database as db

logger = logging.getLogger(__name__)


class NovelGenerator:
    """
    Translates a simulation episode into novel-quality prose.

    Parameters
    ----------
    llm : LLMClient
        Used with premium model for narrative generation.
    output_dir : str
        Directory where chapter .md files are saved.
    """

    def __init__(self, llm: LLMClient, output_dir: str = "output") -> None:
        self.llm        = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main Entry Point
    # ------------------------------------------------------------------ #

    def generate_chapter(
        self,
        episode_id: str,
        protagonist_id: str,
        style: str = "first_person",
        target_words: int = 3500,
    ) -> str:
        """
        Generate a full novel chapter for the episode.

        Parameters
        ----------
        episode_id      : episode to process
        protagonist_id  : agent whose perspective to use
        style           : "first_person" or "third_person_close"
        target_words    : approximate target word count

        Returns
        -------
        str : path to the generated .md file
        """
        logger.info("Generating chapter for episode %s, protagonist %s",
                    episode_id, protagonist_id)

        # 1. Load all interactions
        interactions = db.load_episode_interactions(episode_id)
        if not interactions:
            raise ValueError(f"No interactions found for episode {episode_id}")

        # 2. Filter by protagonist perspective
        pov_interactions = self._filter_by_perspective(interactions, protagonist_id)

        # 3. Analyse narrative structure
        arc           = self._analyze_emotional_arc(pov_interactions, protagonist_id, episode_id)
        key_moments   = self._identify_key_moments(pov_interactions)
        chapter_title = self._generate_title(pov_interactions, arc)

        # 4. Split into scenes and generate prose per scene
        scenes = self._split_into_scenes(pov_interactions)
        prose_sections: list[str] = []

        for i, scene in enumerate(scenes):
            is_climax = (i == len(scenes) // 2)  # middle section is climax
            section = self._generate_prose_section(
                scene, protagonist_id, arc, key_moments,
                style=style, is_climax=is_climax
            )
            prose_sections.append(section)
            logger.info("Generated scene %d/%d (%d words)",
                        i + 1, len(scenes), len(section.split()))

        # 5. Combine and add internal monologue bridges
        combined = self._combine_sections(prose_sections, protagonist_id, arc)

        # 6. Polish
        final = self._polish_narrative(combined, target_words)

        # 7. Write to file
        out_path = self.output_dir / f"{episode_id}_chapter.md"
        self._write_chapter(out_path, chapter_title, final, episode_id, arc)

        logger.info("Chapter written to %s (%d words)", out_path, len(final.split()))
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Step 2: Perspective Filter
    # ------------------------------------------------------------------ #

    def _filter_by_perspective(
        self, interactions: list[dict], protagonist_id: str
    ) -> list[dict]:
        """
        Keep only interactions the protagonist could have witnessed:
        - Their own actions
        - Scene events (director_event, action_type != 'director_event' is fine too)
        - Other agents' dialogue (they were in the same scene)
        Filter out: inner thoughts of OTHER agents, background actions protagonist wasn't present for.
        """
        filtered = []
        for ix in interactions:
            # Always include protagonist's own
            if ix["speaker_id"] == protagonist_id:
                filtered.append({**ix, "_is_self": True})
                continue
            # Include director events (scene narration)
            if ix["action_type"] == "director_event":
                filtered.append({**ix, "_is_scene": True})
                continue
            # Include other characters' dialogue (protagonist heard it)
            # Exclude inner thoughts of other characters
            content = ix.get("content", "")
            if content.startswith("[") and content.endswith("]"):
                # Likely inner thought marker — skip
                continue
            filtered.append(ix)

        return filtered

    # ------------------------------------------------------------------ #
    # Step 3: Arc + Key Moment Analysis
    # ------------------------------------------------------------------ #

    def _analyze_emotional_arc(
        self,
        interactions: list[dict],
        protagonist_id: str,
        episode_id: str,
    ) -> dict:
        """Return a structured emotional arc for the episode."""
        emotions_data = db.load_agent_emotions(protagonist_id, episode_id)

        # Summarise interactions
        synopsis = "\n".join(
            f"Turn {ix['turn']}: [{ix['speaker_name']}] {ix['content'][:100]}"
            for ix in interactions[:30]
        )

        prompt = (
            f"Analyze the emotional arc of this story episode from the protagonist's perspective.\n\n"
            f"Episode synopsis (first 30 turns):\n{synopsis}\n\n"
            f"Emotional data: {json.dumps(emotions_data[:20], indent=2)}\n\n"
            f"Reply with JSON:\n"
            f"{{\n"
            f"  \"opening_mood\": \"...\",\n"
            f"  \"turning_point\": \"...\",\n"
            f"  \"climax_emotion\": \"...\",\n"
            f"  \"resolution_mood\": \"...\",\n"
            f"  \"dominant_theme\": \"...\",\n"
            f"  \"protagonist_change\": \"...\"\n"
            f"}}"
        )

        result = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="arc_analysis",
            use_premium=True,
            temperature=0.3,
            max_tokens=4000,
        )
        arc = self._parse_json(result)
        logger.info("Arc: %s", arc)
        return arc

    def _identify_key_moments(self, interactions: list[dict]) -> list[dict]:
        """Identify pivotal interactions for emphasis in narrative."""
        # Heuristics: director events, high emotional content, turning points
        key = []
        for ix in interactions:
            content = ix.get("content", "")
            meta    = ix.get("metadata", {})
            emotions = meta.get("emotions", {})

            # High emotional intensity
            max_intensity = max(emotions.values(), default=0.0)
            if max_intensity > 0.7:
                key.append({**ix, "_reason": "high_emotion"})
                continue

            # Director-injected event
            if ix.get("action_type") == "director_event":
                key.append({**ix, "_reason": "director_event"})
                continue

            # Contains discovery language
            discovery_words = ["discover", "notice", "realize", "reveal", "found",
                               "secret", "suddenly", "understand"]
            if any(w in content.lower() for w in discovery_words):
                key.append({**ix, "_reason": "discovery"})

        return key[:10]  # Cap at 10 key moments

    # ------------------------------------------------------------------ #
    # Step 4: Prose Generation
    # ------------------------------------------------------------------ #

    def _split_into_scenes(self, interactions: list[dict]) -> list[list[dict]]:
        """Split interactions into scene chunks for prose generation."""
        chunk_size = max(8, len(interactions) // 4)
        scenes = []
        for i in range(0, len(interactions), chunk_size):
            chunk = interactions[i:i + chunk_size]
            if chunk:
                scenes.append(chunk)
        return scenes

    def _generate_prose_section(
        self,
        scene_interactions: list[dict],
        protagonist_id: str,
        arc: dict,
        key_moments: list[dict],
        style: str = "first_person",
        is_climax: bool = False,
    ) -> str:
        """Generate polished prose for one scene chunk."""
        # Format interaction log
        log_text = "\n".join(
            f"Turn {ix['turn']} | {ix['speaker_name']}: {ix['content'][:200]}"
            for ix in scene_interactions
        )

        # Key moment turns in this scene
        key_turns = {k["turn"] for k in key_moments}
        scene_has_key = any(ix["turn"] in key_turns for ix in scene_interactions)

        pov = "first person" if style == "first_person" else "third person close"
        intensity = "high tension and significance" if is_climax else "natural pacing"

        system = (
            f"You are a literary fiction author writing a {pov} narrative chapter. "
            f"Your prose is immersive, character-driven, and emotionally resonant. "
            f"Avoid clichés. Use specific sensory details. Show don't tell."
        )

        prompt = (
            f"## Story Arc Context\n"
            f"Theme: {arc.get('dominant_theme', 'unknown')}\n"
            f"Protagonist's emotional journey: {arc.get('protagonist_change', '')}\n"
            f"Opening mood: {arc.get('opening_mood', '')}\n\n"
            f"## Scene Log to Transform into Prose\n{log_text}\n\n"
            f"## Instructions\n"
            f"Write this as flowing {pov} narrative prose. Pacing: {intensity}.\n"
            f"{'This is a PIVOTAL scene — give it emotional weight.' if scene_has_key else ''}\n"
            f"- Include vivid scene-setting and atmosphere\n"
            f"- Convey character emotions through action and dialogue subtext\n"
            f"- Keep dialogue natural and character-specific\n"
            f"- Protagonist observes and reflects on what unfolds\n"
            f"- Do NOT include simulation metadata, turn numbers, or speaker labels\n"
            f"- Write 3-5 paragraphs, approximately 300-500 words\n"
            f"Write only the narrative prose, no headings or labels:"
        )

        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            purpose="narrative_gen",
            use_premium=True,
            temperature=0.8,
            max_tokens=800,
        )

    # ------------------------------------------------------------------ #
    # Step 5: Combine + Internal Monologue
    # ------------------------------------------------------------------ #

    def _combine_sections(
        self,
        sections: list[str],
        protagonist_id: str,
        arc: dict,
    ) -> str:
        """Combine sections with internal monologue bridges."""
        if len(sections) <= 1:
            return sections[0] if sections else ""

        combined_parts = [sections[0]]

        for i in range(1, len(sections)):
            # Add transition / internal monologue bridge
            bridge = self._generate_bridge(sections[i - 1], sections[i], arc)
            combined_parts.append(bridge)
            combined_parts.append(sections[i])

        return "\n\n".join(combined_parts)

    def _generate_bridge(
        self, prev_section: str, next_section: str, arc: dict
    ) -> str:
        """Generate a short internal monologue bridging two scene sections."""
        prompt = (
            f"Write a 2-3 sentence internal monologue or brief scene transition "
            f"that bridges these two story sections.\n\n"
            f"End of previous section: ...{prev_section[-200:]}\n\n"
            f"Start of next section: {next_section[:200]}...\n\n"
            f"Story theme: {arc.get('dominant_theme', '')}\n"
            f"Write only the bridge text, in first person, introspective:"
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="bridge_gen",
            use_premium=True,
            temperature=0.7,
            max_tokens=120,
        )

    # ------------------------------------------------------------------ #
    # Step 6: Polish
    # ------------------------------------------------------------------ #

    def _polish_narrative(self, text: str, target_words: int) -> str:
        """Final pass: consistency, flow, and word count adjustment."""
        current_words = len(text.split())
        instruction = ""
        if current_words < target_words * 0.7:
            instruction = f"Expand the narrative with more sensory detail and reflection to reach ~{target_words} words."
        elif current_words > target_words * 1.4:
            instruction = f"Tighten the prose and trim repetition to reach ~{target_words} words."
        else:
            instruction = "Review for consistent voice and smooth flow. Make minor improvements only."

        prompt = (
            f"{instruction}\n\n"
            f"Also ensure:\n"
            f"- Consistent first-person voice throughout\n"
            f"- No simulation artifacts (turn numbers, metadata, etc.)\n"
            f"- Natural paragraph breaks\n"
            f"- No abrupt tone shifts\n\n"
            f"Text to polish:\n\n{text}"
        )

        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="narrative_polish",
            use_premium=True,
            temperature=0.5,
            max_tokens=4000,
        )

    def _generate_title(self, interactions: list[dict], arc: dict) -> str:
        """Generate a chapter title from arc and content."""
        synopsis = " ".join(ix["content"][:80] for ix in interactions[:5])
        prompt = (
            f"Suggest a literary chapter title (3-7 words) for a story chapter with:\n"
            f"Theme: {arc.get('dominant_theme', '')}\n"
            f"Opening: {synopsis[:200]}\n"
            f"Reply with only the title, no punctuation or quotes."
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="title_gen",
            use_premium=False,
            temperature=0.9,
            max_tokens=30,
        ).strip()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _write_chapter(
        self,
        path: Path,
        title: str,
        content: str,
        episode_id: str,
        arc: dict,
    ) -> None:
        """Write the chapter Markdown file."""
        header = (
            f"# {title}\n\n"
            f"*Episode: {episode_id}*\n\n"
            f"---\n\n"
        )
        footer = (
            f"\n\n---\n\n"
            f"*Narrative arc: {arc.get('protagonist_change', '')}*\n"
        )
        path.write_text(header + content + footer, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {}
